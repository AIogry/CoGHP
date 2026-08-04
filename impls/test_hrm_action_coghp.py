"""Synthetic tests for the clean full-sequence HRM-Action-CoGHP."""

import pathlib
import sys
import unittest

import flax
import jax
import jax.numpy as jnp
import numpy as np


IMPLS_DIR = pathlib.Path(__file__).resolve().parent
if str(IMPLS_DIR) not in sys.path:
    sys.path.insert(0, str(IMPLS_DIR))

from agents.hrm_action_coghp import (  # noqa: E402
    HRMActionCoGHPAgent,
    get_config,
)
from utils.coghp_network import HierarchicalPolicyNetwork  # noqa: E402
from utils.hrm_action_coghp_network import (  # noqa: E402
    HRMActionCoGHPPolicyNetwork,
)
from utils.networks import GCActor  # noqa: E402


STATE_DIM = 8
ACTION_DIM = 2
BATCH_SIZE = 2
NUM_SUBGOALS = 2


def actor_heads():
    high_actor = GCActor(
        hidden_dims=(8,),
        action_dim=STATE_DIM,
        state_dependent_std=False,
        const_std=True,
        gc_encoder=None,
    )
    low_actor = GCActor(
        hidden_dims=(8,),
        action_dim=ACTION_DIM,
        state_dependent_std=False,
        const_std=True,
        gc_encoder=None,
    )
    return high_actor, low_actor


def make_network(is_mlp_mixer=True, bptt=True, chain_source='teacher'):
    high_actor, low_actor = actor_heads()
    return HRMActionCoGHPPolicyNetwork(
        state_dim=STATE_DIM,
        high_actor_head=high_actor,
        low_actor_head=low_actor,
        num_mixer_blocks=1,
        mixer_token_hidden=8,
        mixer_channel_hidden=8,
        gc_encoder=None,
        enc_hidden=(8,),
        num_subgoals=NUM_SUBGOALS,
        causal_mixer=True,
        action_hrm_num_layers=1,
        action_hrm_intermediate_size=16,
        action_hrm_head_dim=2,
        action_hrm_h_cycles=2,
        action_hrm_l_cycles=2,
        action_hrm_bptt=bptt,
        action_hrm_is_causal=True,
        action_hrm_is_mlp_mixer=is_mlp_mixer,
        action_hrm_mlp_mixer_intermediate_size=12,
        train_action_chain_source=chain_source,
    )


def synthetic_inputs():
    observations = jnp.arange(BATCH_SIZE * 5, dtype=jnp.float32).reshape(
        BATCH_SIZE, 5
    ) / 10.0
    goals = jnp.flip(observations, axis=0)
    teacher_subgoals = jnp.arange(
        BATCH_SIZE * NUM_SUBGOALS * STATE_DIM,
        dtype=jnp.float32,
    ).reshape(BATCH_SIZE, NUM_SUBGOALS, STATE_DIM)
    teacher_subgoals = teacher_subgoals / teacher_subgoals.size
    actions = jnp.linspace(
        -0.5,
        0.5,
        BATCH_SIZE * ACTION_DIM,
        dtype=jnp.float32,
    ).reshape(BATCH_SIZE, ACTION_DIM)
    return observations, goals, teacher_subgoals, actions


def initialize(network):
    observations, goals, _, _ = synthetic_inputs()
    return network.init(
        jax.random.PRNGKey(0),
        observations,
        goals,
        jax.random.PRNGKey(1),
    )


def subtree_norm(tree, pattern):
    flat = flax.traverse_util.flatten_dict(tree)
    return sum(
        float(jnp.linalg.norm(value))
        for path, value in flat.items()
        if pattern in '/'.join(path)
    )


class HRMActionPolicyNetworkTest(unittest.TestCase):
    def test_reads_complete_sequence_and_returns_one_action_distribution(self):
        network = make_network()
        variables = initialize(network)
        observations, goals, teacher_subgoals, _ = synthetic_inputs()

        outputs, mutable = network.apply(
            variables,
            observations,
            goals,
            jax.random.PRNGKey(2),
            subgoal_reps=teacher_subgoals,
            return_diagnostics=True,
            mutable=['intermediates'],
        )
        high_dists, action_dist, actions, diagnostics = outputs
        self.assertEqual(len(high_dists), NUM_SUBGOALS)
        self.assertEqual(action_dist.mode().shape, (BATCH_SIZE, ACTION_DIM))
        self.assertEqual(actions.shape, (BATCH_SIZE, ACTION_DIM))

        token_sequence = mutable['intermediates'][
            'action_token_sequence'
        ][0]
        self.assertEqual(
            token_sequence.shape,
            (BATCH_SIZE, NUM_SUBGOALS + 3, STATE_DIM),
        )
        np.testing.assert_allclose(
            token_sequence[:, 2 : 2 + NUM_SUBGOALS, :],
            teacher_subgoals,
        )
        self.assertEqual(int(diagnostics['action_hrm/outer_cycles']), 1)
        self.assertEqual(int(diagnostics['action_hrm/high_level_calls']), 2)
        self.assertEqual(int(diagnostics['action_hrm/low_level_calls']), 4)
        self.assertEqual(
            int(diagnostics['action_hrm/token_sequence_length']),
            NUM_SUBGOALS + 3,
        )

        self.assertIn('constants', variables)
        constant_shapes = {
            '/'.join(path): value.shape
            for path, value in flax.traverse_util.flatten_dict(
                variables['constants']
            ).items()
        }
        self.assertEqual(
            constant_shapes['action_hrm/zH_init'],
            (STATE_DIM,),
        )
        self.assertEqual(
            constant_shapes['action_hrm/zL_init'],
            (STATE_DIM,),
        )

    def test_mixer_and_attention_paths_are_jittable(self):
        observations, goals, teacher_subgoals, _ = synthetic_inputs()
        for is_mlp_mixer in (True, False):
            with self.subTest(is_mlp_mixer=is_mlp_mixer):
                network = make_network(is_mlp_mixer=is_mlp_mixer)
                variables = initialize(network)

                @jax.jit
                def run(obs, goal, subgoals):
                    high_dists, action_dist, actions = network.apply(
                        variables,
                        obs,
                        goal,
                        jax.random.PRNGKey(3),
                        subgoal_reps=subgoals,
                    )
                    return (
                        jnp.stack(
                            [dist.mode() for dist in high_dists],
                            axis=1,
                        ),
                        action_dist.mode(),
                        actions,
                    )

                high_modes, action_mode, actions = run(
                    observations,
                    goals,
                    teacher_subgoals,
                )
                self.assertEqual(
                    high_modes.shape,
                    (BATCH_SIZE, NUM_SUBGOALS, STATE_DIM),
                )
                self.assertEqual(action_mode.shape, (BATCH_SIZE, ACTION_DIM))
                self.assertEqual(actions.shape, (BATCH_SIZE, ACTION_DIM))

    def test_action_loss_trains_both_official_hrm_levels(self):
        network = make_network()
        variables = initialize(network)
        observations, goals, teacher_subgoals, actions = synthetic_inputs()

        def action_loss(params):
            _, action_dist, _ = network.apply(
                {'params': params, 'constants': variables['constants']},
                observations,
                goals,
                jax.random.PRNGKey(4),
                subgoal_reps=teacher_subgoals,
            )
            return -jnp.mean(action_dist.log_prob(actions))

        grads = jax.grad(action_loss)(variables['params'])
        self.assertGreater(subtree_norm(grads, 'action_hrm/H_level'), 0.0)
        self.assertGreater(subtree_norm(grads, 'action_hrm/L_level'), 0.0)
        self.assertGreater(subtree_norm(grads, 'low_actor_head'), 0.0)
        self.assertGreater(subtree_norm(grads, 'action_query_token'), 0.0)
        self.assertEqual(subtree_norm(grads, 'planner_mixer'), 0.0)
        self.assertEqual(subtree_norm(grads, 'high_actor_head'), 0.0)

    def test_legacy_segment_and_residual_parameters_are_absent(self):
        variables = initialize(make_network())
        parameter_paths = [
            '/'.join(path)
            for path in flax.traverse_util.flatten_dict(
                variables['params']
            )
        ]
        forbidden = (
            'segment',
            'readout',
            'residual',
            'action_context_mixer',
            'action_high_cell',
            'action_low_cell',
        )
        for name in forbidden:
            self.assertFalse(
                any(name in path for path in parameter_paths),
                name,
            )


class SplitActionMixerBaselineTest(unittest.TestCase):
    def test_split_baseline_has_independent_mixer_and_action_query(self):
        high_actor, low_actor = actor_heads()
        network = HierarchicalPolicyNetwork(
            num_tokens=1,
            state_dim=STATE_DIM,
            num_action_dims=ACTION_DIM,
            joint_embed_dim=STATE_DIM,
            num_mixer_blocks=1,
            mixer_token_hidden=8,
            mixer_channel_hidden=8,
            gc_encoder=None,
            high_actor_head=high_actor,
            low_actor_head=low_actor,
            enc_hidden=(8,),
            num_subgoals=NUM_SUBGOALS,
            causal_mixer=True,
            action_use_full_subgoal_chain=True,
            separate_action_mixer=True,
        )
        observations, goals, teacher_subgoals, actions = synthetic_inputs()
        params = network.init(
            jax.random.PRNGKey(5),
            observations,
            goals,
            jax.random.PRNGKey(6),
        )['params']

        self.assertIn('subgoal_mixer_blocks_0', params)
        self.assertIn('action_mixer_blocks_0', params)
        self.assertIn('action_query_token', params)

        def action_loss(current_params):
            _, action_dist, _ = network.apply(
                {'params': current_params},
                observations,
                goals,
                jax.random.PRNGKey(7),
                subgoal_reps=teacher_subgoals,
            )
            return -jnp.mean(action_dist.log_prob(actions))

        grads = jax.grad(action_loss)(params)
        self.assertEqual(subtree_norm(grads, 'subgoal_mixer_blocks'), 0.0)
        self.assertGreater(subtree_norm(grads, 'action_mixer_blocks'), 0.0)
        self.assertEqual(subtree_norm(grads, 'prev_tokens'), 0.0)
        self.assertGreater(subtree_norm(grads, 'action_query_token'), 0.0)


class HRMActionAgentTest(unittest.TestCase):
    def make_agent(self):
        config = get_config()
        config.encoder = None
        config.frame_stack = None
        config.action_chunk = 1
        config.feature_dim = STATE_DIM
        config.mixer_hidden = 8
        config.num_mixer_blocks = 1
        config.enc_hidden_dims = (16,)
        config.actor_hidden_dims = (16,)
        config.value_hidden_dims = (16,)
        config.num_subgoals = NUM_SUBGOALS
        config.action_hrm_num_layers = 1
        config.action_hrm_intermediate_size = 16
        config.action_hrm_head_dim = 2
        config.action_hrm_h_cycles = 2
        config.action_hrm_l_cycles = 2
        config.action_hrm_mlp_mixer_intermediate_size = 12
        observations = jnp.ones((BATCH_SIZE, 5), dtype=jnp.float32)
        actions = jnp.zeros((BATCH_SIZE, ACTION_DIM), dtype=jnp.float32)
        return HRMActionCoGHPAgent.create(0, observations, actions, config)

    @staticmethod
    def make_batch():
        observations, goals, _, actions = synthetic_inputs()
        return {
            'observations': observations,
            'next_observations': observations + 0.01,
            'actions': actions,
            'rewards': jnp.zeros((BATCH_SIZE,), dtype=jnp.float32),
            'masks': jnp.ones((BATCH_SIZE,), dtype=jnp.float32),
            'value_goals': goals,
            'high_actor_goals': goals,
            'high_actor_targets': jnp.stack(
                [observations + 0.1, observations + 0.2],
                axis=1,
            ),
        }

    def test_train_state_preserves_constants_without_optimizing_them(self):
        agent = self.make_agent()
        self.assertIn('constants', agent.network.model_state)
        old_constants = agent.network.model_state['constants']
        new_agent, info = agent.update(self.make_batch())
        self.assertIn('low_actor/actor_loss', info)
        self.assertIn('action_hrm/z_high_final_norm', info)
        for old, new in zip(
            jax.tree_util.tree_leaves(old_constants),
            jax.tree_util.tree_leaves(
                new_agent.network.model_state['constants']
            ),
        ):
            np.testing.assert_array_equal(old, new)

    def test_validation_metrics_exist_and_are_finite(self):
        agent = self.make_agent()
        batch = self.make_batch()
        info = agent.validation_rollout_info(
            batch,
            rng=jax.random.PRNGKey(8),
        )
        expected = {
            'validation_action/teacher/mse',
            'validation_action/predicted/mse',
            'validation_action/gap/mse',
            'validation_action/teacher/bc_log_prob',
            'validation_action/predicted/bc_log_prob',
        }
        self.assertEqual(set(info), expected)
        for value in info.values():
            self.assertTrue(np.isfinite(np.asarray(value)).all())

    def test_config_contains_only_clean_hrm_controls(self):
        config = get_config()
        self.assertEqual(config.agent_name, 'hrm_action_coghp')
        self.assertEqual(config.action_hrm_num_layers, 1)
        self.assertEqual(config.action_hrm_h_cycles, 2)
        self.assertEqual(config.action_hrm_l_cycles, 2)
        self.assertTrue(config.action_hrm_bptt)
        self.assertTrue(config.action_hrm_is_mlp_mixer)
        self.assertEqual(config.train_action_chain_source, 'teacher')
        for legacy_name in (
            'action_hrm_segments',
            'action_hrm_max_segments',
            'action_hrm_residual_alpha',
            'action_hrm_state_init_mode',
            'action_hrm_trainable_init',
            'action_inference_segment',
        ):
            self.assertNotIn(legacy_name, config)


if __name__ == '__main__':
    unittest.main()
