"""Fast synthetic tests for FM-CoGHP interfaces and gradient boundaries."""

import pathlib
import sys
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np


IMPLS_DIR = pathlib.Path(__file__).resolve().parent
if str(IMPLS_DIR) not in sys.path:
    sys.path.insert(0, str(IMPLS_DIR))

from agents.fm_coghp import FMCoGHPAgent, get_config  # noqa: E402
from utils.datasets import Dataset, MultiHGCDataset  # noqa: E402
from utils.flax_utils import restore_agent, save_agent  # noqa: E402


def small_config(num_subgoals=2):
    config = get_config()
    config.encoder = None
    config.frame_stack = None
    config.actor_hidden_dims = (16,)
    config.value_hidden_dims = (16,)
    config.enc_hidden_dims = (16,)
    config.feature_dim = 8
    config.mixer_hidden = 8
    config.num_mixer_blocks = 1
    config.num_subgoals = num_subgoals
    config.subgoal_steps = 2
    config.flow_num_blocks = 1
    config.flow_token_hidden = 8
    config.flow_channel_hidden = 16
    config.flow_time_embed_dim = 8
    config.flow_steps = 2
    config.flow_num_candidates = 2
    config.flow_diagnostic_candidates = 2
    config.crl_latent_dim = 8
    config.crl_hidden_dims = (16,)
    return config


def synthetic_batch(batch_size=4, num_subgoals=2):
    observations = jnp.arange(batch_size * 6, dtype=jnp.float32).reshape(batch_size, 6) / 20.0
    offsets = jnp.arange(num_subgoals, 0, -1, dtype=jnp.float32)
    targets = observations[:, None, :] + offsets[None, :, None] * 0.05
    indices = jnp.arange(batch_size, dtype=jnp.int32) * 10
    target_idxs = indices[:, None] + jnp.arange(num_subgoals, 0, -1, dtype=jnp.int32)[None, :] * 2
    return {
        'observations': observations,
        'next_observations': observations + 0.01,
        'actions': jnp.zeros((batch_size, 2), dtype=jnp.float32),
        'value_goals': observations[::-1],
        'high_actor_goals': observations[::-1],
        'high_actor_targets': targets,
        'rewards': -jnp.ones((batch_size,), dtype=jnp.float32),
        'masks': jnp.ones((batch_size,), dtype=jnp.float32),
        'indices': indices,
        'trajectory_final_idxs': jnp.full((batch_size,), 100, dtype=jnp.int32),
        'high_actor_target_idxs': target_idxs,
        'high_actor_target_final_idxs': jnp.full(
            (batch_size, num_subgoals), 100, dtype=jnp.int32
        ),
    }


class FMCoGHPAgentTest(unittest.TestCase):
    def test_shapes_update_selectors_and_latent_radius(self):
        config = small_config(num_subgoals=2)
        batch = synthetic_batch(num_subgoals=2)
        agent = FMCoGHPAgent.create(
            0,
            batch['observations'][:1],
            batch['actions'][:1],
            config,
        )
        loss, info = agent.total_loss(batch, grad_params=None)
        self.assertTrue(np.isfinite(float(loss)))
        self.assertIn('flow/nearest_clean_mse', info)
        self.assertIn('crl/top1_accuracy', info)

        updated_agent, update_info = agent.update(batch)
        self.assertEqual(int(updated_agent.network.step), 2)
        self.assertTrue(np.isfinite(float(update_info['total_loss'])))

        chains, _ = updated_agent.sample_flow_chains(
            batch['observations'],
            batch['high_actor_goals'],
            jax.random.PRNGKey(1),
            num_candidates=2,
            num_steps=2,
        )
        self.assertEqual(chains.shape, (4, 2, 2, 8))
        np.testing.assert_allclose(
            np.asarray(jnp.linalg.norm(chains, axis=-1)),
            np.sqrt(8.0),
            rtol=1e-5,
            atol=1e-5,
        )
        for selector in ('none', 'coghp_value', 'local_crl'):
            selected_chain, scores, selected_idx = updated_agent.select_flow_chain(
                batch['observations'], chains, selector
            )
            self.assertEqual(selected_chain.shape, (4, 2, 8))
            self.assertEqual(scores.shape, (4, 2))
            self.assertEqual(selected_idx.shape, (4,))

        diagnostics = updated_agent.validation_rollout_info(batch, jax.random.PRNGKey(4))
        self.assertIn('fm_selector/value_oracle_agreement', diagnostics)
        self.assertIn('fm_selector/crl_action_mse', diagnostics)
        self.assertGreater(float(diagnostics['flow/sample_pairwise_distance']), 0.0)

        action = updated_agent.sample_actions(
            batch['observations'][0],
            batch['high_actor_goals'][0],
            jax.random.PRNGKey(2),
            temperature=0.0,
        )
        self.assertEqual(action.shape, (2,))

        with tempfile.TemporaryDirectory() as checkpoint_dir:
            save_agent(updated_agent, checkpoint_dir, 2)
            restored_agent = restore_agent(agent, checkpoint_dir, 2)
            self.assertEqual(int(restored_agent.network.step), 2)
            restored_action = restored_agent.sample_actions(
                batch['observations'][0],
                batch['high_actor_goals'][0],
                jax.random.PRNGKey(2),
                temperature=0.0,
            )
            np.testing.assert_allclose(np.asarray(restored_action), np.asarray(action))

    def test_flow_loss_does_not_update_non_flow_modules(self):
        config = small_config(num_subgoals=1)
        batch = synthetic_batch(num_subgoals=1)
        agent = FMCoGHPAgent.create(
            0,
            batch['observations'][:1],
            batch['actions'][:1],
            config,
        )

        def loss_fn(params):
            return agent.flow_loss(batch, params, jax.random.PRNGKey(3))[0]

        grads = jax.grad(loss_fn)(agent.network.params)
        for module_name in ('modules_actor_mixer', 'modules_value', 'modules_target_value', 'modules_crl_critic'):
            leaves = jax.tree_util.tree_leaves(grads[module_name])
            self.assertTrue(all(np.allclose(np.asarray(leaf), 0.0) for leaf in leaves))
        flow_norm = sum(
            float(jnp.linalg.norm(leaf))
            for leaf in jax.tree_util.tree_leaves(grads['modules_flow_planner'])
        )
        self.assertGreater(flow_norm, 0.0)


class MultiHGCDatasetMetadataTest(unittest.TestCase):
    def test_far_to_near_order_and_metadata(self):
        observations = np.arange(60, dtype=np.float32).reshape(10, 6)
        dataset = Dataset.create(
            observations=observations,
            actions=np.zeros((10, 2), dtype=np.float32),
            terminals=np.array([0, 0, 0, 1, 1, 0, 0, 0, 1, 1], dtype=np.float32),
            valids=np.array([1, 1, 1, 1, 0, 1, 1, 1, 1, 0], dtype=np.float32),
        )
        config = small_config(num_subgoals=2)
        wrapped = MultiHGCDataset(dataset, config)
        batch = wrapped.sample(2, idxs=np.array([0, 5]))
        self.assertTrue(np.all(batch['high_actor_target_offsets'][:, 0] >= batch['high_actor_target_offsets'][:, 1]))
        self.assertEqual(batch['high_actor_target_idxs'].shape, (2, 2))
        self.assertEqual(batch['high_actor_target_final_idxs'].shape, (2, 2))
        np.testing.assert_array_equal(batch['indices'], np.array([0, 5], dtype=np.int32))

        config.actor_p_trajgoal = 0.0
        config.actor_p_randomgoal = 1.0
        random_goal_wrapped = MultiHGCDataset(dataset, config)
        random_goal_batch = random_goal_wrapped.sample(2, idxs=np.array([0, 5]))
        self.assertTrue(np.all(random_goal_batch['high_actor_target_offsets'] >= 0))
        self.assertTrue(
            np.all(
                random_goal_batch['high_actor_target_offsets'][:, 0]
                >= random_goal_batch['high_actor_target_offsets'][:, 1]
            )
        )
        self.assertTrue(
            np.all(
                random_goal_batch['high_actor_target_idxs']
                <= random_goal_batch['trajectory_final_idxs'][:, None]
            )
        )


if __name__ == '__main__':
    unittest.main()
