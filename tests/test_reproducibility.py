import pathlib
import sys
import unittest

IMPLS_DIR = pathlib.Path(__file__).resolve().parents[1] / 'impls'
if str(IMPLS_DIR) not in sys.path:
    sys.path.insert(0, str(IMPLS_DIR))

import jax
import ml_collections
import numpy as np

from utils.datasets import Dataset, MultiHGCDataset
from utils.evaluation import derive_seed, evaluate


def make_dataset(seed):
    size = 12
    terminals = np.zeros(size, dtype=np.float32)
    terminals[[5, 11]] = 1
    data = Dataset.create(
        seed=seed,
        observations=np.arange(size * 3, dtype=np.float32).reshape(size, 3),
        actions=np.arange(size * 2, dtype=np.float32).reshape(size, 2),
        terminals=terminals,
        valids=np.ones(size, dtype=np.float32),
    )
    config = ml_collections.ConfigDict(
        dict(
            frame_stack=None,
            value_p_curgoal=0.2,
            value_p_trajgoal=0.5,
            value_p_randomgoal=0.3,
            value_geom_sample=True,
            actor_p_curgoal=0.0,
            actor_p_trajgoal=1.0,
            actor_p_randomgoal=0.0,
            actor_geom_sample=False,
            discount=0.99,
            gc_negative=True,
            p_aug=0.0,
            subgoal_steps=2,
            num_subgoals=2,
        )
    )
    return MultiHGCDataset(data, config, rng=np.random.default_rng(seed))


def assert_tree_equal(test_case, left, right):
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    test_case.assertEqual(len(left_leaves), len(right_leaves))
    for left_leaf, right_leaf in zip(left_leaves, right_leaves):
        np.testing.assert_array_equal(left_leaf, right_leaf)


class DatasetReproducibilityTest(unittest.TestCase):
    def test_same_seed_produces_identical_batch_sequence(self):
        left = make_dataset(123)
        right = make_dataset(123)
        for _ in range(4):
            assert_tree_equal(self, left.sample(8), right.sample(8))

    def test_validation_sampling_does_not_perturb_training_stream(self):
        train_with_validation = make_dataset(10)
        train_only = make_dataset(10)
        validation = make_dataset(20)

        first_with_validation = train_with_validation.sample(8)
        for _ in range(5):
            validation.sample(8)
        second_with_validation = train_with_validation.sample(8)

        first_train_only = train_only.sample(8)
        second_train_only = train_only.sample(8)
        assert_tree_equal(self, first_with_validation, first_train_only)
        assert_tree_equal(self, second_with_validation, second_train_only)

    def test_explicit_validation_rng_reuses_identical_batch(self):
        dataset = make_dataset(30)
        first = dataset.sample(8, evaluation=True, rng=np.random.default_rng(40))
        dataset.sample(8)
        second = dataset.sample(8, evaluation=True, rng=np.random.default_rng(40))
        assert_tree_equal(self, first, second)


class FakeAgent:
    def sample_actions(self, observations, goals=None, seed=None, temperature=0):
        del observations, goals, temperature
        return jax.random.uniform(seed, shape=(1,), minval=-1.0, maxval=1.0)


class FakeEnv:
    def __init__(self, first_episode_length=2):
        self.first_episode_length = first_episode_length
        self.reset_count = 0

    def reset(self, seed=None, options=None):
        del options
        self.reset_count += 1
        self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.episode_length = self.first_episode_length if self.reset_count == 1 else 3
        observation = self.rng.normal(size=2)
        return observation, {'goal': np.zeros(2, dtype=np.float32)}

    def step(self, action):
        self.step_count += 1
        observation = self.rng.normal(size=2)
        terminated = self.step_count >= self.episode_length
        info = {'success': float(action[0] > 0)}
        return observation, 0.0, terminated, False, info


class EvaluationReproducibilityTest(unittest.TestCase):
    def test_seed_derivation_is_stable_and_stream_specific(self):
        self.assertEqual(derive_seed(7, 1, 2), derive_seed(7, 1, 2))
        self.assertNotEqual(derive_seed(7, 1, 2), derive_seed(7, 1, 3))

    def test_repeated_evaluation_is_identical(self):
        kwargs = dict(
            agent=FakeAgent(),
            task_id=1,
            config={'discrete': False},
            num_eval_episodes=3,
            num_video_episodes=0,
            seed=12345,
        )
        stats_a, trajs_a, _ = evaluate(env=FakeEnv(), **kwargs)
        stats_b, trajs_b, _ = evaluate(env=FakeEnv(), **kwargs)
        self.assertEqual(stats_a, stats_b)
        assert_tree_equal(self, trajs_a, trajs_b)

    def test_later_episode_action_stream_does_not_depend_on_prior_length(self):
        kwargs = dict(
            agent=FakeAgent(),
            task_id=1,
            config={'discrete': False},
            num_eval_episodes=2,
            num_video_episodes=0,
            seed=54321,
        )
        _, short_first_trajs, _ = evaluate(env=FakeEnv(first_episode_length=1), **kwargs)
        _, long_first_trajs, _ = evaluate(env=FakeEnv(first_episode_length=5), **kwargs)
        np.testing.assert_array_equal(
            short_first_trajs[1]['action'][0],
            long_first_trajs[1]['action'][0],
        )


if __name__ == '__main__':
    unittest.main()
