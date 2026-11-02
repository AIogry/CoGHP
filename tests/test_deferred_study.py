"""Deferred-launch safety: dependencies, failed runs, and duplicate prevention."""

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts.launch_study_after import launch_once


class DeferredStudyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.after = Path(self.temp.name) / 'parent'
        self.root = Path(self.temp.name) / 'new'
        self.after.mkdir()
        self.root.mkdir()
        self.source = self.root / 'source'
        self.source.mkdir()
        self.job_dir = self.after / 'runs' / 'old_method'
        self.job_dir.mkdir(parents=True)
        self.job = dict(method='old_method', run_dir=str(self.job_dir))
        self.write(self.after / 'campaign.json', dict(study={}, source={'code_signature': 'old'}))
        self.write(self.root / 'campaign.json', dict(study={}, source_dir=str(self.source), source={'code_signature': 'new'}))
        self.runner = SimpleNamespace(
            plan=Mock(return_value=[self.job]), validate_completed=Mock(), verify_snapshot=Mock(),
            check_gpu_occupancy=Mock(), write_json=self.write,
        )

    @staticmethod
    def write(path, obj):
        path.write_text(json.dumps(obj))

    def state(self, **updates):
        record = dict(status='completed', returncode=0, source_signature='old')
        record.update(updates)
        self.write(self.job_dir / 'run.json', record)

    def invoke(self, execute=True):
        return launch_once(self.runner, self.after, self.root, ['1'], 2, execute)

    @patch('scripts.launch_study_after.subprocess.Popen')
    def test_missing_or_running_dependency_does_not_launch(self, popen):
        self.assertEqual(self.invoke()['status'], 'waiting')
        self.state(status='running', returncode=None)
        self.assertEqual(self.invoke()['status'], 'waiting')
        popen.assert_not_called()
        self.runner.check_gpu_occupancy.assert_not_called()

    @patch('scripts.launch_study_after.subprocess.Popen')
    def test_failed_dependency_requires_attention(self, popen):
        self.state(status='failed', returncode=1)
        with self.assertRaisesRegex(ValueError, 'needs attention'):
            self.invoke()
        popen.assert_not_called()

    @patch('scripts.launch_study_after.subprocess.Popen')
    def test_completed_flag_without_valid_metrics_is_not_enough(self, popen):
        self.state()
        self.runner.validate_completed.side_effect = ValueError('missing exact evaluation step')
        with self.assertRaisesRegex(ValueError, 'missing exact'):
            self.invoke()
        popen.assert_not_called()

    @patch('scripts.launch_study_after.subprocess.Popen')
    def test_ready_check_is_read_only(self, popen):
        self.state()
        self.assertEqual(self.invoke(False)['status'], 'ready')
        self.assertFalse((self.root / 'launcher.log').exists())
        popen.assert_not_called()

    @patch('scripts.launch_study_after.subprocess.Popen')
    def test_successful_launch_is_not_repeated(self, popen):
        self.state()
        popen.return_value.pid = 12345
        first = self.invoke()
        self.assertEqual(first['status'], 'started')
        self.assertEqual(first['pid'], 12345)
        command = popen.call_args.args[0]
        self.assertEqual(command[command.index('--gpus')+1], '1')
        self.assertEqual(command[command.index('--workers-per-gpu')+1], '2')
        self.assertTrue(popen.call_args.kwargs['start_new_session'])
        self.assertEqual(self.invoke()['status'], 'already_launched')
        self.assertEqual(popen.call_count, 1)

    @patch('scripts.launch_study_after.subprocess.Popen')
    def test_busy_gpu_waits_without_creating_launch_intent(self, popen):
        self.state()
        self.runner.check_gpu_occupancy.side_effect = ValueError('Selected GPUs already have compute processes: example')
        self.assertEqual(self.invoke()['status'], 'waiting_gpu')
        self.assertFalse((self.root / 'launcher.log').exists())
        popen.assert_not_called()

    @patch('scripts.launch_study_after.subprocess.Popen')
    def test_partial_launch_intent_is_not_retried(self, popen):
        (self.root / 'launcher.log').write_text('startup was interrupted')
        with self.assertRaisesRegex(ValueError, 'inspect before retrying'):
            self.invoke()
        popen.assert_not_called()


if __name__ == '__main__':
    unittest.main()
