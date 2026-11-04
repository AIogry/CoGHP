"""Actual first-exit events and passive evaluation capture contracts."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'impls'))

from utils.evaluation import evaluate
from utils.junction_behavior import aggregate_episodes, save_behavior_rollouts, summarize_episode


def topology():
    cells = [[1, 0], [1, 1], [1, 2], [1, 3], [2, 1]]
    adjacency = {}
    for row, col in cells:
        adjacency[f'{row},{col}'] = [cell for cell in cells
                                    if abs(cell[0] - row) + abs(cell[1] - col) == 1]
    return dict(cell_width=4.0, core_half_width=1.6, origin_xy=[0.0, 0.0],
                free_cells=cells, junction_cells=[[1, 1]], adjacency=adjacency)


def observations(cells):
    return np.asarray([[4.0 * col, 4.0 * row, float(i)]
                       for i, (row, col) in enumerate(cells)], dtype=np.float32)


class JunctionBehaviorTests(unittest.TestCase):
    def test_first_exit_return_and_border_jitter(self):
        path = observations([(1, 0), (1, 1), (1, 1), (1, 1), (1, 2), (1, 2), (1, 1), (2, 1)])
        path[2, :2] = [6.0, 4.0]  # Boundary between J and east, outside both stable cores.
        summary = summarize_episode(path, path[-1], topology(), success=True, terminated=True)
        first, second = summary['events']
        self.assertEqual(first['incoming'], [1, 0])
        self.assertEqual(first['first_exit'], [1, 2])
        self.assertEqual(first['first_exit_step'], 4)
        self.assertEqual(first['first_return_step'], 6)
        self.assertEqual(first['first_exit_latency_steps'], 3)
        self.assertEqual(first['return_excursion_steps'], 2)
        self.assertEqual(first['outcome'], 'returned')
        self.assertEqual(second['outcome'], 'no_return_before_success')
        self.assertFalse(second['first_visit_at_junction'])
        self.assertEqual(summary['union_return_excursion_steps'], 2)
        self.assertEqual(summary['first_visit_returns'], 1)

    def test_episode_start_and_endpoint_censoring_are_retained(self):
        path = observations([(1, 1), (1, 1)])
        summary = summarize_episode(path, observations([(1, 3)])[0], topology(), truncated=True)
        event = summary['events'][0]
        self.assertEqual(event['incoming_status'], 'episode_start')
        self.assertEqual(event['exit_status'], 'episode_end')
        self.assertEqual(event['outcome'], 'no_exit_observed')
        self.assertIsNone(event['first_exit'])
        self.assertEqual(summary['no_exit_observed_visits'], 1)
        self.assertEqual(summary['known_exit_visits'], 0)

        path = observations([(1, 0), (1, 1), (1, 2)])
        summary = summarize_episode(path, path[-1], topology(), truncated=True)
        self.assertEqual(summary['events'][0]['outcome'], 'no_return_observed_before_truncation')
        self.assertEqual(summary['events'][0]['observed_steps_after_exit'], 0)

    def test_skipped_or_unknown_core_does_not_invent_an_exit(self):
        for middle in ((1, 3), (0, 1)):
            path = observations([(1, 0), (1, 1), middle, (1, 2), (1, 1)])
            summary = summarize_episode(path, path[-1], topology(), truncated=True)
            event = summary['events'][0]
            self.assertEqual(event['exit_status'], 'nonadjacent_core')
            self.assertIsNone(event['first_exit'])
            self.assertIsNone(event['return_excursion_steps'])
            self.assertEqual(event['first_return_step'], 4)
        self.assertEqual(summary['unknown_topology_frames'], 1)

    def test_grouping_preserves_first_visits_and_complete_episode_identity(self):
        path = observations([(1, 0), (1, 1), (1, 2), (1, 1), (1, 0), (1, 1), (1, 2)])
        episode = summarize_episode(path, path[-1], topology(), truncated=True, episode_index=7)
        groups = aggregate_episodes([episode])
        west = next(group for group in groups if group['incoming'] == [1, 0])
        self.assertEqual(west['visits'], 2)
        self.assertEqual(west['distinct_episodes'], 1)
        self.assertEqual(west['episode_indices'], [7])
        self.assertEqual(west['return_fraction_given_known_exit'], 0.5)
        first = aggregate_episodes([episode], first_visits_only=True)
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0]['visits'], 1)
        self.assertEqual(first[0]['return_fraction_given_known_exit'], 1.0)

    def test_overlapping_excursions_use_union_steps(self):
        geometry = topology()
        geometry['junction_cells'].append([1, 2])
        # J1 -> J2 -> east -> J2 -> J1; excursions [2,5) and [3,4) overlap.
        path = observations([(1, 0), (1, 1), (1, 2), (1, 3), (1, 2), (1, 1), (1, 0)])
        summary = summarize_episode(path, path[-1], geometry, success=True)
        intervals = [event['return_excursion_steps'] for event in summary['events']
                     if event['return_excursion_steps'] is not None]
        self.assertEqual(sum(intervals), 4)
        self.assertEqual(summary['union_return_excursion_steps'], 3)

    def test_save_is_portable_numeric_and_refuses_overwrite(self):
        path = observations([(1, 0), (1, 1), (1, 2), (1, 1), (2, 1)])
        trajectory = dict(
            observation=path[:-1], next_observation=path[1:], action=np.zeros((4, 2), np.float32),
            reward=np.zeros(4), done=np.array([False, False, False, True]),
            _metadata=dict(task_id=1, episode_index=3, evaluation_seed=123, episode_seed=45,
                           actor_seed=46, noise_seed=47, goal=path[-1].copy(),
                           success=True, terminated=True, truncated=False),
        )
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / 'step_1000000' / 'task_1'
            metrics = save_behavior_rollouts(prefix, [trajectory], topology(), task_id=1, checkpoint_step=1000000)
            self.assertEqual(metrics['returned_visits'], 1)
            with np.load(prefix.with_suffix('.npz'), allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['observation'], path[:-1])
                np.testing.assert_array_equal(saved['next_observation'], path[1:])
                np.testing.assert_array_equal(saved['goal'], path[-1:])
                np.testing.assert_array_equal(saved['episode_offsets'], [0, 4])
                np.testing.assert_array_equal(saved['episode_indices'], [3])
            report = json.loads(prefix.with_suffix('.json').read_text())
            self.assertEqual(report['episodes'][0]['metadata']['actor_seed'], 46)
            self.assertEqual(report['trajectory_file'], 'task_1.npz')
            self.assertTrue(report['junction_coverage'][0]['visited'])
            with self.assertRaises(FileExistsError):
                save_behavior_rollouts(prefix, [trajectory], topology(), task_id=1, checkpoint_step=1000000)

    def test_metadata_capture_does_not_change_actions_rng_or_trajectories(self):
        class FakeAgent:
            def __init__(self):
                self.keys = []

            def sample_actions(self, observations, goals, temperature, seed):
                key = np.asarray(seed).copy()
                self.keys.append(key)
                return (key % 7).astype(np.float32) * 0.01

        class FakeEnv:
            def reset(self, seed, options):
                self.steps = 0
                self.state = np.array([0.0, 4.0, float(seed % 13)], np.float32)
                self.goal = np.array([8.0, 4.0, float(seed % 17)], np.float32)
                return self.state.copy(), dict(goal=self.goal.copy())

            def step(self, action):
                self.steps += 1
                self.state = self.state.copy()
                self.state[:2] += action
                return self.state.copy(), 0.0, self.steps == 3, False, dict(success=float(self.steps == 3))

        results = []
        agents = []
        for capture in (False, True):
            agent = FakeAgent()
            with patch('utils.evaluation.trange', range):
                result = evaluate(agent, FakeEnv(), task_id=1, config={'discrete': False},
                                  num_eval_episodes=2, seed=91, record_trajectory_metadata=capture)
            agents.append(agent)
            results.append(result)
        self.assertEqual(results[0][0], results[1][0])
        np.testing.assert_array_equal(agents[0].keys, agents[1].keys)
        self.assertEqual(len(agents[0].keys), 6)
        for old, new in zip(results[0][1], results[1][1]):
            self.assertNotIn('_metadata', old)
            for key in ('observation', 'next_observation', 'action', 'reward', 'done'):
                np.testing.assert_array_equal(old[key], new[key])
            metadata = new['_metadata']
            self.assertEqual(metadata['goal'].shape, (3,))
            self.assertTrue(metadata['success'])
            self.assertTrue(metadata['terminated'])
            self.assertFalse(metadata['truncated'])


if __name__ == '__main__':
    unittest.main()
