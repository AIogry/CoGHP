"""Regression checks for experiment identity, scheduling, provenance and metrics."""

import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('study_runner', ROOT / 'scripts/run_study.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class StudyRunnerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.study = runner.load_study(runner.DEFAULT_STUDY)

    def test_global_position_matrix_seed_priority_and_isolated_axes(self):
        study = runner.load_study(ROOT / 'experiments/navigation_global_position.json')
        jobs = runner.plan(study, self.root)
        self.assertEqual(len(jobs), 12)
        self.assertEqual([j['seed'] for j in jobs], [0]*4+[1]*4+[2]*4)
        self.assertEqual([j['method'] for j in jobs[:4]], ['baseline','g_only','p_only','g_plus_p'])
        common = [{k:v for k,v in j['agent'].items()
                   if k not in ('command_content','high_weight_source')} for j in jobs]
        self.assertTrue(all(c == common[0] for c in common))
        self.assertEqual(study['metric']['steps'], [800000,900000,1000000])

    def test_diagnostics_are_cpu_only_and_fail_closed(self):
        study = runner.load_study(ROOT / 'experiments/navigation_global_position.json')
        job = runner.plan(study, self.root, methods='p_only')[0]
        Path(job['run_dir']).mkdir(parents=True)
        observed = []
        def probe(cmd, **kwargs):
            observed.append((cmd, kwargs['env']))
            return subprocess.CompletedProcess(cmd, 1)
        with mock.patch.object(runner.subprocess, 'run', side_effect=probe):
            result = runner.run_diagnostics(job, study, ROOT, self.root)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(len(observed), 1)
        self.assertEqual(observed[0][1]['JAX_PLATFORMS'], 'cpu')
        self.assertEqual(observed[0][1]['CUDA_VISIBLE_DEVICES'], '')

    def test_diagnostic_checkpoints_must_be_saved(self):
        self.study['diagnostics'] = {'steps':[123]}
        path = self.root / 'study.json'
        path.write_text(json.dumps(self.study))
        with self.assertRaisesRegex(ValueError, 'saved checkpoints'):
            runner.load_study(path)

    def test_matrix_and_matched_environment_settings(self):
        jobs = runner.plan(self.study, self.root)
        self.assertEqual(len(jobs), 24)
        self.assertEqual(sum(job['method'] == 'coghp' for job in jobs), 18)
        for env in self.study['environments']:
            selected = [job for job in jobs if job['environment'] == env]
            self.assertEqual([job['seed'] for job in selected], [0, 1, 2, 0])
            base, hrm = selected[0]['agent'], selected[-1]['agent']
            self.assertFalse(hrm['action_hrm_bptt'])
            self.assertEqual(hrm['planner_core_type'], 'vanilla_coghp')
            for key, value in base.items():
                self.assertEqual(hrm[key], value, (env, key))
        point = next(j for j in jobs if j['environment'] == 'pointmaze-giant-navigate-v0')
        self.assertEqual(point['agent']['lr'], 1e-5)
        counts = {j['environment']: j['agent']['expected_action_core_params']
                  for j in jobs if j['method'] != 'coghp'}
        self.assertEqual(counts['pointmaze-giant-navigate-v0'], 58368)
        self.assertEqual(counts['antmaze-giant-navigate-v0'], 204288)

    def test_extension_adds_30_disjoint_runs_and_completes_54(self):
        extension = runner.load_study(ROOT / 'experiments/navigation_hrm_action_multiseed_extension.json')
        old = runner.plan(self.study, self.root/'old')
        new = runner.plan(extension, self.root/'new')
        self.assertEqual(len(new), 30)
        identity = lambda j: (j['method'], j['environment'], j['seed'])
        self.assertFalse(set(map(identity, old)) & set(map(identity, new)))
        self.assertEqual(len(set(map(identity, old+new))), 54)
        for env in extension['environments']:
            for method in extension['methods']:
                selected = [j for j in old+new if j['environment']==env and j['method']==method]
                self.assertEqual(sorted(j['seed'] for j in selected), [0,1,2])
        for job in new:
            if job['environment'].startswith('humanoidmaze'):
                self.assertEqual(job['agent']['discount'], 0.995)
                self.assertEqual(job['agent']['subgoal_steps'], 100)
                self.assertEqual(job['agent']['feature_dim'], 128)
            else:
                self.assertEqual(job['method'], 'hrm_h2l2_truncated')
                self.assertIn(job['seed'], [1,2])

    def test_invalid_environment_seed_scope_is_rejected(self):
        study = copy.deepcopy(self.study)
        study['methods']['coghp']['seeds_by_environment']={'unknown-environment':[1]}
        path=self.root/'study.json';path.write_text(json.dumps(study))
        with self.assertRaisesRegex(ValueError, 'scope'):
            runner.load_study(path)

    def test_joint_summary_rejects_duplicate_runs(self):
        jobs=runner.plan(self.study,self.root)
        with self.assertRaisesRegex(ValueError, 'Duplicate run identity'):
            runner.summarize(jobs,self.study,[(jobs,self.study)])

    def test_joint_summary_counts_all_three_seeds(self):
        extension=runner.load_study(ROOT/'experiments/navigation_hrm_action_multiseed_extension.json')
        env='pointmaze-medium-navigate-v0'
        old=runner.plan(self.study,self.root/'old',methods='hrm_h2l2_truncated',environments=env)
        new=runner.plan(extension,self.root/'new',methods='hrm_h2l2_truncated',environments=env)
        for job in old+new:
            p=Path(job['run_dir']);p.mkdir(parents=True)
            (p/'run.json').write_text('{"status":"completed"}')
        def metric(job,study):
            value=(job['seed']+1)/10
            return {'values':{str(step):value for step in study['metric']['steps']},'mean':value}
        output=io.StringIO()
        with mock.patch.object(runner,'validate_completed',side_effect=metric),contextlib.redirect_stdout(output):
            self.assertEqual(runner.summarize(new,extension,[(old,self.study)]),0)
        self.assertIn('n=3 | 20.00 | SD=10.00',output.getvalue())

    def test_joint_provenance_allows_launcher_changes_but_not_training_or_data_changes(self):
        a=dict(study=self.study,datasets={'data.npz':{'sha256':'datahash'}},
               source={'files':{'impls/main.py':'runtimehash','impls/agents/coghp.py':'modelhash','scripts/run_study.py':'old'}})
        b=copy.deepcopy(a);b['source']['files']['scripts/run_study.py']='new'
        runner.validate_comparable_campaigns(a,b)
        b['source']['files']['impls/agents/coghp.py']='changed'
        with self.assertRaisesRegex(ValueError,'source'):
            runner.validate_comparable_campaigns(a,b)
        b=copy.deepcopy(a);b['datasets']['data.npz']['sha256']='changed'
        with self.assertRaisesRegex(ValueError,'dataset'):
            runner.validate_comparable_campaigns(a,b)

    def test_bad_filters_and_attempt_identity(self):
        with self.assertRaises(ValueError):
            runner.plan(self.study, self.root, methods='unknown')
        first = runner.plan(self.study, self.root, attempt=0)
        retry = runner.plan(self.study, self.root, attempt=1)
        self.assertEqual(len({j['run_dir'] for j in first + retry}), 48)

    def test_protocol_rejects_unobservable_metric_steps(self):
        self.study['metric']['steps'] = [80000, 90000, 100000]
        path = self.root / 'study.json'
        path.write_text(json.dumps(self.study))
        with self.assertRaisesRegex(ValueError, 'schedule'):
            runner.load_study(path)

    def test_command_pins_core_and_ignores_legacy_shell_overrides(self):
        job = runner.plan(self.study, self.root, methods='hrm_h2l2_truncated')[0]
        provenance = dict(git_commit='sha', code_signature='hash')
        with mock.patch.dict(os.environ, ACTION_HRM_BPTT='True', JAX_PLATFORMS='cpu', XLA_FLAGS='stale'):
            cmd = runner.command(job, self.study, ROOT, provenance, 'disabled')
            env = runner.child_environment(ROOT, self.root, '1', 0)
        self.assertIn('--agent.action_hrm_bptt=False', cmd)
        self.assertIn('--require_gpu=True', cmd)
        self.assertIn('--save_steps=800000,900000,1000000', cmd)
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '1')
        self.assertEqual(env['XLA_FLAGS'], '--xla_gpu_deterministic_ops')
        self.assertNotIn('JAX_PLATFORMS', env)

    def test_metric_uses_exact_steps_not_tail_or_best(self):
        path = self.root / 'eval.csv'
        path.write_text('step,evaluation/overall_success\n700000,1\n800000,0.2\n900000,0.4\n1000000,0.6\n1100000,1\n')
        self.assertAlmostEqual(runner.read_metric(path, self.study['metric'])['mean'], 0.4)

    def test_metric_rejects_missing_duplicate_and_nonfinite(self):
        path = self.root / 'eval.csv'
        for rows in ('800000,0.2\n1000000,0.6\n',
                     '800000,0.2\n800000,0.3\n900000,0.4\n1000000,0.6\n',
                     '800000,nan\n900000,0.4\n1000000,0.6\n'):
            with self.subTest(rows=rows):
                path.write_text('step,evaluation/overall_success\n' + rows)
                with self.assertRaises(ValueError):
                    runner.read_metric(path, self.study['metric'])

    def test_one_sequential_worker_per_gpu_with_unequal_durations(self):
        active = set()
        seen = []
        lock = threading.Lock()
        overlap = []
        def run(job, gpu):
            with lock:
                if gpu in active:
                    overlap.append(gpu)
                active.add(gpu)
            time.sleep(0.01 if job % 2 else 0.03)
            with lock:
                active.remove(gpu)
                seen.append(job)
            return 0
        result = runner.dispatch(list(range(12)), ['0', '1'], run, threading.Event())
        self.assertEqual(result, 0)
        self.assertEqual(sorted(seen), list(range(12)))
        self.assertFalse(overlap)

    def test_failure_stops_queue_without_cancelling_inflight(self):
        stop = threading.Event()
        seen = []
        def fail(job, gpu):
            seen.append(job)
            return 1
        self.assertEqual(runner.dispatch(list(range(10)), ['0'], fail, stop), 1)
        self.assertEqual(seen, [0])
        self.assertFalse(stop.is_set())

    def test_freeze_captures_dirty_and_untracked_source_and_rejects_tampering(self):
        repo = self.root / 'repo'
        repo.mkdir()
        subprocess.run(['git', 'init', '-q', str(repo)], check=True)
        (repo / 'code.py').write_text('original\n')
        subprocess.run(['git', '-C', str(repo), 'add', 'code.py'], check=True)
        subprocess.run(['git', '-C', str(repo), '-c', 'user.name=Test', '-c', 'user.email=test@example.org',
                        'commit', '-qm', 'fixture'], check=True)
        (repo / 'code.py').write_text('modified\n')
        (repo / 'new.py').write_text('new\n')
        campaign = self.root / 'campaign'
        with mock.patch.object(runner, 'ROOT', repo):
            source, provenance = runner.freeze(campaign, self.study, {})
            self.assertTrue(provenance['git_status'])
            self.assertEqual((source / 'code.py').read_text(), 'modified\n')
            self.assertEqual((source / 'new.py').read_text(), 'new\n')
            (repo / 'code.py').write_text('later edit\n')
            self.assertEqual(runner.freeze(campaign, self.study, {})[1], provenance)
            (source / 'code.py').chmod(0o644)
            (source / 'code.py').write_text('tampered\n')
            with self.assertRaisesRegex(ValueError, 'hashes'):
                runner.freeze(campaign, self.study, {})

    def test_external_source_snapshot_is_reused_and_never_overwritten(self):
        repo=self.root/'repo';repo.mkdir()
        (repo/'code.py').write_text('original\n')
        provenance=dict(git_commit='sha',files={'code.py':runner.digest(repo/'code.py')})
        provenance['code_signature']=runner.signature(provenance['files'])
        campaign=self.root/'results';source=self.root/'Navigation'/'snapshot'
        with mock.patch.object(runner,'ROOT',repo),mock.patch.object(runner,'source_inventory',return_value=provenance):
            frozen,_=runner.freeze(campaign,self.study,{},source)
            self.assertEqual(frozen,source)
            self.assertTrue((campaign/'source').is_symlink())
            (repo/'code.py').write_text('development edit\n')
            self.assertEqual(runner.freeze(campaign,self.study,{})[0],source)
            self.assertEqual((source/'code.py').read_text(),'original\n')
            with self.assertRaisesRegex(ValueError,'differs'):
                runner.freeze(campaign,self.study,{},self.root/'other')

    def test_prepare_does_not_access_gpu_or_dispatch_training(self):
        campaign=self.root/'campaign';source=self.root/'Navigation'/'snapshot'
        with mock.patch.object(runner,'dataset_inventory',return_value={}), \
             mock.patch.object(runner,'freeze',return_value=(source,dict(git_commit='sha',code_signature='hash'))) as freeze, \
             mock.patch.object(runner,'check_gpu_occupancy',side_effect=AssertionError('GPU accessed')), \
             mock.patch.object(runner,'dispatch',side_effect=AssertionError('Training started')), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.main(['--run-root',str(campaign),'--source-root',str(source),'--prepare']),0)
            self.assertEqual(freeze.call_args.args[3],source)

    def test_dry_run_writes_nothing_and_never_queries_gpu(self):
        dataset = self.root / 'data'
        dataset.mkdir()
        for env in self.study['environments']:
            for suffix in ('.npz', '-val.npz'):
                (dataset / (env + suffix)).write_bytes(b'fixture')
        campaign = self.root / 'campaign'
        with mock.patch.object(runner, 'check_gpus', side_effect=AssertionError('GPU access')), contextlib.redirect_stdout(io.StringIO()):
            result = runner.main(['--run-root', str(campaign), '--dataset-root', str(dataset), '--dry-run'])
        self.assertEqual(result, 0)
        self.assertFalse(campaign.exists())

    def test_failed_attempt_cannot_be_overwritten(self):
        job = runner.plan(self.study, self.root)[0]
        run_dir = Path(job['run_dir'])
        run_dir.mkdir(parents=True)
        (run_dir / 'run.json').write_text('{"status":"failed"}')
        with mock.patch.object(runner, 'dataset_inventory', return_value={}), self.assertRaisesRegex(ValueError, 'Existing failed'):
            runner.main(['--run-root', str(self.root), '--dry-run'])
        self.assertEqual((run_dir / 'run.json').read_text(), '{"status":"failed"}')

    def test_missing_data_fails_instead_of_skipping_tasks(self):
        with self.assertRaises(FileNotFoundError):
            runner.dataset_inventory(self.study, self.root)

    def test_process_lifecycle_validates_artifacts_and_single_seed_summary(self):
        job = runner.plan(self.study, self.root, methods='hrm_h2l2_truncated')[0]
        flags = {**self.study['protocol'], 'agent': dict(job['agent'], agent_name=job['agent_name']),
                 'env_name': job['environment'], 'seed': job['seed'],
                 'save_steps': ['800000', '900000', '1000000'],
                 'experiment_git_commit': 'sha', 'experiment_code_signature': 'hash'}
        program = '''
import json,os,pathlib,sys
assert os.environ['CUDA_VISIBLE_DEVICES']=='1'
p=pathlib.Path(sys.argv[1]); p.mkdir()
(p/'flags.json').write_text(sys.argv[2])
(p/'eval.csv').write_text('step,evaluation/overall_success\\n800000,0.2\\n900000,0.4\\n1000000,0.6\\n')
for step in (800000,900000,1000000): (p/f'params_{step}.pkl').write_bytes(b'fixture')
'''
        cmd = [os.sys.executable, '-c', program, str(Path(job['run_dir'])/'artifacts'), json.dumps(flags)]
        with mock.patch.object(runner, 'command', return_value=cmd), contextlib.redirect_stdout(io.StringIO()):
            status = runner.run_one(job, '1', self.study, ROOT,
                                    dict(git_commit='sha', code_signature='hash'),
                                    self.root, 'disabled', threading.Event())
        self.assertEqual(status, 0)
        state = json.loads((Path(job['run_dir'])/'run.json').read_text())
        self.assertEqual(state['status'], 'completed')
        self.assertAlmostEqual(state['metric']['mean'], 0.4)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(runner.summarize([job], self.study), 0)
        self.assertIn('N/A (single seed)', output.getvalue())
        self.assertIn('20.00 | 40.00 | 60.00 | 40.00', output.getvalue())
        with self.assertRaises(FileExistsError):
            runner.run_one(job, '1', self.study, ROOT, {}, self.root, 'disabled', threading.Event())
        artifacts = Path(job['run_dir'])/'artifacts'
        (artifacts/'params_900000.pkl').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing checkpoint'):
            runner.validate_completed(job, self.study)

    def test_nonzero_training_exit_is_failed_not_completed(self):
        job = runner.plan(self.study, self.root)[0]
        with mock.patch.object(runner, 'command', return_value=[os.sys.executable, '-c', 'raise SystemExit(7)']), contextlib.redirect_stdout(io.StringIO()):
            status = runner.run_one(job, '0', self.study, ROOT,
                                    dict(git_commit='sha', code_signature='hash'),
                                    self.root, 'disabled', threading.Event())
        self.assertEqual(status, 1)
        state = json.loads((Path(job['run_dir'])/'run.json').read_text())
        self.assertEqual((state['status'], state['returncode']), ('failed', 7))

    def test_gpu_preflight_checks_only_selected_devices(self):
        replies = ['0, GPU-a, A\n1, GPU-b, B\n', 'GPU-b, 234\n']
        with mock.patch.object(subprocess, 'check_output', side_effect=replies):
            self.assertIn('0', runner.check_gpu_occupancy(['0']))
        with mock.patch.object(subprocess, 'check_output', side_effect=replies):
            with self.assertRaisesRegex(ValueError, 'compute processes'):
                runner.check_gpu_occupancy(['1'])

    def test_legacy_hrm_launcher_matches_all_six_environment_recipes(self):
        capture = self.root / 'arguments.json'
        fake = self.root / 'python'
        fake.write_text(f'#!{os.sys.executable}\nimport json,os,sys\n'
                        'open(os.environ["CAPTURE"],"w").write(json.dumps(sys.argv[1:]))\n')
        fake.chmod(0o755)
        env = dict(os.environ, PYTHON=str(fake), CAPTURE=str(capture),
                   HRM_ACTION_SKIP_DATASET_CHECK='1', LOG_DIR=str(self.root / 'logs'),
                   ACTION_HRM_BPTT='False')
        extension = runner.load_study(ROOT / 'experiments/navigation_hrm_action_multiseed_extension.json')
        for name, expected in extension['environments'].items():
            task = name.removesuffix('-navigate-v0').replace('-', '_')
            result = subprocess.run(['bash', str(ROOT / 'scripts/train_hrm_action_coghp.sh'), task],
                                    env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            args = json.loads(capture.read_text())
            self.assertIn(f'--env_name={name}', args)
            self.assertIn('--agent.action_hrm_bptt=False', args)
            for key in ('num_subgoals', 'subgoal_steps', 'feature_dim'):
                self.assertIn(f'--agent.{key}={expected[key]}', args)
            if 'pointmaze-giant' in name:
                self.assertIn('--agent.lr=1e-5', args)


if __name__ == '__main__':
    unittest.main()
