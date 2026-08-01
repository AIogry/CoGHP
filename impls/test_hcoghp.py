"""Synthetic tests for segment-based autoregressive HCoGHP."""

import inspect
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

from agents.hcoghp import get_config  # noqa: E402
from utils.hcoghp_network import (  # noqa: E402
    HCoGHPPolicyNetwork,
    SubgoalReadout,
    apply_predicted_token_gradient_boundary,
    initialize_segment_states,
)
from utils.networks import GCActor  # noqa: E402


STATE_DIM = 8
ACTION_DIM = 3
BATCH_SIZE = 4


def make_network(
    num_subgoals,
    trainable_init=False,
    full_chain=True,
    state_init_mode='carry',
    h_cycles=2,
    l_cycles=2,
    residual_alpha=0.05,
    readout_mode='residual',
    stage_embedding_in_context=False,
):
    return HCoGHPPolicyNetwork(
        num_tokens=1,
        state_dim=STATE_DIM,
        num_action_dims=ACTION_DIM,
        joint_embed_dim=STATE_DIM,
        num_mixer_blocks=2,
        mixer_token_hidden=8,
        mixer_channel_hidden=12,
        gc_encoder=None,
        layer_norm=True,
        high_actor_head=GCActor(
            hidden_dims=(16,),
            action_dim=STATE_DIM,
            state_dependent_std=False,
            const_std=True,
            gc_encoder=None,
        ),
        low_actor_head=GCActor(
            hidden_dims=(16,),
            action_dim=ACTION_DIM,
            state_dependent_std=False,
            const_std=True,
            gc_encoder=None,
        ),
        enc_hidden=(16,),
        num_subgoals=num_subgoals,
        causal_mixer=True,
        action_use_full_subgoal_chain=full_chain,
        hrm_h_cycles=h_cycles,
        hrm_l_cycles=l_cycles,
        hrm_init_std=0.1,
        hrm_trainable_init=trainable_init,
        hrm_low_num_blocks=1,
        hrm_high_num_blocks=1,
        subgoal_readout_hidden_dim=16,
        hrm_readout_mode=readout_mode,
        hrm_readout_residual_alpha=residual_alpha,
        stage_embedding_in_context=stage_embedding_in_context,
        segment_state_init_mode=state_init_mode,
        segment_carry_gradient_mode='detached',
    )


def synthetic_inputs(num_subgoals):
    observations = jnp.arange(
        BATCH_SIZE * 6,
        dtype=jnp.float32,
    ).reshape(BATCH_SIZE, 6) / 20.0
    goals = jnp.flip(observations, axis=0)
    subgoals = jnp.arange(
        BATCH_SIZE * num_subgoals * STATE_DIM,
        dtype=jnp.float32,
    ).reshape(BATCH_SIZE, num_subgoals, STATE_DIM)
    subgoals = subgoals / max(subgoals.size, 1)
    return observations, goals, subgoals


def initialize(network, num_subgoals):
    observations, goals, _ = synthetic_inputs(num_subgoals)
    variables = network.init(
        jax.random.PRNGKey(0),
        observations,
        goals,
        jax.random.PRNGKey(1),
    )
    return variables['params']


def tree_norm(tree):
    return sum(
        float(jnp.linalg.norm(leaf))
        for leaf in jax.tree_util.tree_leaves(tree)
    )


def assert_finite_tree(test_case, tree):
    test_case.assertTrue(
        all(
            np.isfinite(np.asarray(leaf)).all()
            for leaf in jax.tree_util.tree_leaves(tree)
        )
    )


def run_with_state_trace(network, params, subgoal_reps):
    observations, goals, _ = synthetic_inputs(network.num_subgoals)
    outputs, mutable = network.apply(
        {'params': params},
        observations,
        goals,
        jax.random.PRNGKey(2),
        subgoal_reps=subgoal_reps,
        return_diagnostics=True,
        mutable=['intermediates'],
    )
    trace = mutable['intermediates']
    return outputs, {
        'low_before': trace['segment_z_low_before'],
        'high_before': trace['segment_z_high_before'],
        'low_after': trace['segment_z_low_after'],
        'high_after': trace['segment_z_high_after'],
    }


class HCoGHPPolicyNetworkTest(unittest.TestCase):
    def test_reset_initializes_every_segment_from_model_state(self):
        num_subgoals = 2
        network = make_network(num_subgoals, state_init_mode='reset')
        params = initialize(network, num_subgoals)
        _, _, subgoals = synthetic_inputs(num_subgoals)
        _, trace = run_with_state_trace(network, params, subgoals)

        expected_low = jnp.tile(params['z_l_init'], (BATCH_SIZE, 1))
        expected_high = jnp.tile(params['z_h_init'], (BATCH_SIZE, 1))
        self.assertEqual(len(trace['low_before']), num_subgoals)
        for segment_idx in range(num_subgoals):
            np.testing.assert_array_equal(
                trace['low_before'][segment_idx],
                expected_low,
            )
            np.testing.assert_array_equal(
                trace['high_before'][segment_idx],
                expected_high,
            )

        self.assertFalse(
            np.allclose(trace['low_before'][1], trace['low_after'][0])
        )
        self.assertFalse(
            np.allclose(trace['high_before'][1], trace['high_after'][0])
        )

    def test_carry_uses_previous_values_and_detaches_boundary(self):
        num_subgoals = 2
        network = make_network(num_subgoals, state_init_mode='carry')
        params = initialize(network, num_subgoals)
        _, _, subgoals = synthetic_inputs(num_subgoals)
        _, trace = run_with_state_trace(network, params, subgoals)

        np.testing.assert_array_equal(trace['low_before'][1], trace['low_after'][0])
        np.testing.assert_array_equal(
            trace['high_before'][1],
            trace['high_after'][0],
        )

        initial_low = jnp.zeros((BATCH_SIZE, STATE_DIM))
        initial_high = jnp.ones((BATCH_SIZE, STATE_DIM))
        previous_low = jnp.arange(BATCH_SIZE * STATE_DIM, dtype=jnp.float32).reshape(
            BATCH_SIZE,
            STATE_DIM,
        )
        previous_high = previous_low + 0.5

        def next_segment_loss(low, high):
            next_low, next_high = initialize_segment_states(
                initial_low,
                initial_high,
                low,
                high,
                mode='carry',
                carry_gradient_mode='detached',
            )
            return jnp.sum(next_low**2) + jnp.sum(next_high**2)

        low_grad, high_grad = jax.grad(
            next_segment_loss,
            argnums=(0, 1),
        )(previous_low, previous_high)
        np.testing.assert_array_equal(low_grad, 0.0)
        np.testing.assert_array_equal(high_grad, 0.0)

    def test_readout_has_only_three_inputs(self):
        signature = inspect.signature(SubgoalReadout.__call__)
        self.assertEqual(
            tuple(signature.parameters),
            ('self', 'context', 'z_low', 'stage_embedding'),
        )

        network = make_network(num_subgoals=1)
        params = initialize(network, num_subgoals=1)
        readout_kernel = params['subgoal_readout']['hidden_dense']['kernel']
        self.assertEqual(readout_kernel.shape[0], 3 * STATE_DIM)

    def test_stage_embedding_is_excluded_from_context_by_default(self):
        token_sequence = jnp.arange(
            BATCH_SIZE * 4 * STATE_DIM,
            dtype=jnp.float32,
        ).reshape(BATCH_SIZE, 4, STATE_DIM)
        stage_a = jnp.zeros((BATCH_SIZE, STATE_DIM))
        stage_b = jnp.ones((BATCH_SIZE, STATE_DIM)) * 100.0

        def context_for(network, params, stage_emb):
            return network.apply(
                {'params': params},
                token_sequence,
                2,
                stage_emb,
                method=network._context_from_query,
            )

        default_network = make_network(
            num_subgoals=1,
            stage_embedding_in_context=False,
        )
        default_params = initialize(default_network, num_subgoals=1)
        np.testing.assert_array_equal(
            context_for(default_network, default_params, stage_a),
            context_for(default_network, default_params, stage_b),
        )

        legacy_network = make_network(
            num_subgoals=1,
            stage_embedding_in_context=True,
        )
        legacy_params = initialize(legacy_network, num_subgoals=1)
        self.assertFalse(
            np.allclose(
                context_for(legacy_network, legacy_params, stage_a),
                context_for(legacy_network, legacy_params, stage_b),
            )
        )

    def test_residual_readout_uses_configured_alpha(self):
        network = make_network(num_subgoals=1, residual_alpha=0.05)
        params = initialize(network, num_subgoals=1)
        observations, goals, subgoals = synthetic_inputs(num_subgoals=1)
        _, _, _, diagnostics = network.apply(
            {'params': params},
            observations,
            goals,
            jax.random.PRNGKey(12),
            subgoal_reps=subgoals,
            return_diagnostics=True,
        )
        self.assertAlmostEqual(float(diagnostics['stage_0/residual_alpha']), 0.05)
        self.assertEqual(float(diagnostics['stage_0/readout_is_residual']), 1.0)
        self.assertEqual(
            float(diagnostics['stage_0/stage_embedding_in_context']),
            0.0,
        )
        self.assertIn('stage_0/hrm_delta_norm', diagnostics)
        self.assertIn('stage_0/residual_update_norm', diagnostics)
        self.assertIn('stage_0/residual_update_ratio', diagnostics)
        self.assertGreater(float(diagnostics['stage_0/hrm_delta_norm']), 0.0)
        self.assertGreater(
            float(diagnostics['stage_0/residual_update_norm']),
            0.0,
        )

    def test_terminal_loss_reaches_high_cell_through_final_low_state(self):
        network = make_network(
            num_subgoals=1,
            h_cycles=2,
            l_cycles=2,
        )
        params = initialize(network, num_subgoals=1)
        observations, goals, subgoals = synthetic_inputs(num_subgoals=1)

        def terminal_loss(current_params):
            high_dists, _, _ = network.apply(
                {'params': current_params},
                observations,
                goals,
                jax.random.PRNGKey(3),
                subgoal_reps=subgoals,
            )
            return -jnp.mean(high_dists[0].log_prob(subgoals[:, 0, :]))

        grads = jax.grad(terminal_loss)(params)
        self.assertGreater(tree_norm(grads['high_cell']), 0.0)
        self.assertGreater(tree_norm(grads['low_cell']), 0.0)

    def test_terminal_loss_full_bptt_trains_entire_segment(self):
        network = make_network(
            num_subgoals=1,
            trainable_init=True,
            h_cycles=2,
            l_cycles=2,
        )
        params = initialize(network, num_subgoals=1)
        observations, goals, subgoals = synthetic_inputs(num_subgoals=1)

        def terminal_loss(current_params):
            high_dists, _, _ = network.apply(
                {'params': current_params},
                observations,
                goals,
                jax.random.PRNGKey(4),
                subgoal_reps=subgoals,
            )
            return -jnp.mean(high_dists[0].log_prob(subgoals[:, 0, :]))

        grads = jax.grad(terminal_loss)(params)
        for module_name in (
            'context_mixer_blocks_0',
            'context_mixer_blocks_1',
            'low_cell',
            'high_cell',
            'subgoal_readout',
            'high_actor_head',
        ):
            assert_finite_tree(self, grads[module_name])
            self.assertGreater(tree_norm(grads[module_name]), 0.0)

        # With one segment and trainable initial states, these gradients prove
        # that the terminal loss reaches the beginning of the recurrent graph.
        self.assertGreater(float(jnp.linalg.norm(grads['z_l_init'])), 0.0)
        self.assertGreater(float(jnp.linalg.norm(grads['z_h_init'])), 0.0)

    def test_one_terminal_distribution_per_segment(self):
        num_subgoals = 3
        h_cycles = 3
        l_cycles = 2
        network = make_network(
            num_subgoals,
            h_cycles=h_cycles,
            l_cycles=l_cycles,
        )
        params = initialize(network, num_subgoals)
        observations, goals, subgoals = synthetic_inputs(num_subgoals)
        high_dists, _, _, diagnostics = network.apply(
            {'params': params},
            observations,
            goals,
            jax.random.PRNGKey(5),
            subgoal_reps=subgoals,
            return_diagnostics=True,
        )

        self.assertEqual(len(high_dists), num_subgoals)
        for distribution in high_dists:
            self.assertEqual(distribution.mode().shape, (BATCH_SIZE, STATE_DIM))
        for segment_idx in range(num_subgoals):
            high_keys = [
                key
                for key in diagnostics
                if key.startswith(f'stage_{segment_idx}/high_cycle_')
                and key.endswith('/relative_change')
                and '/low_cycle_' not in key
            ]
            low_keys = [
                key
                for key in diagnostics
                if key.startswith(f'stage_{segment_idx}/high_cycle_')
                and '/low_cycle_' in key
            ]
            self.assertEqual(len(high_keys), h_cycles)
            self.assertEqual(len(low_keys), h_cycles * l_cycles)
        self.assertFalse(any('cycle_loss' in key for key in diagnostics))
        self.assertFalse(any('cycle_prediction' in key for key in diagnostics))

    def test_predicted_prefix_boundary_is_independent(self):
        predicted = jnp.arange(
            BATCH_SIZE * STATE_DIM,
            dtype=jnp.float32,
        ).reshape(BATCH_SIZE, STATE_DIM)

        detached_grad = jax.grad(
            lambda value: jnp.sum(
                apply_predicted_token_gradient_boundary(value, True) ** 2
            )
        )(predicted)
        open_grad = jax.grad(
            lambda value: jnp.sum(
                apply_predicted_token_gradient_boundary(value, False) ** 2
            )
        )(predicted)
        np.testing.assert_array_equal(detached_grad, 0.0)
        self.assertGreater(float(jnp.linalg.norm(open_grad)), 0.0)

    def test_free_running_prefix_is_detached_from_action_loss(self):
        network = make_network(num_subgoals=2)
        params = initialize(network, num_subgoals=2)
        observations, goals, _ = synthetic_inputs(num_subgoals=2)

        def action_loss(current_params):
            _, low_dist, _ = network.apply(
                {'params': current_params},
                observations,
                goals,
                jax.random.PRNGKey(6),
                subgoal_reps=None,
                scheduled_sampling_stop_gradient=False,
            )
            return jnp.sum(low_dist.mode())

        grads = jax.grad(action_loss)(params)
        self.assertEqual(tree_norm(grads['high_actor_head']), 0.0)
        self.assertEqual(tree_norm(grads['high_cell']), 0.0)
        self.assertEqual(tree_norm(grads['low_cell']), 0.0)
        self.assertEqual(tree_norm(grads['subgoal_readout']), 0.0)
        self.assertGreater(tree_norm(grads['low_actor_head']), 0.0)

    def test_scheduled_sampling_stop_gradient_flag_is_honored(self):
        network = make_network(num_subgoals=1)
        params = initialize(network, num_subgoals=1)
        observations, goals, subgoals = synthetic_inputs(num_subgoals=1)

        def action_grad(stop_gradient):
            def action_loss(current_params):
                _, low_dist, _ = network.apply(
                    {'params': current_params},
                    observations,
                    goals,
                    jax.random.PRNGKey(7),
                    subgoal_reps=subgoals,
                    scheduled_sampling_prob=1.0,
                    scheduled_sampling_use_mode=True,
                    scheduled_sampling_stop_gradient=stop_gradient,
                )
                return jnp.sum(low_dist.mode())

            return jax.grad(action_loss)(params)

        detached_grads = action_grad(True)
        open_grads = action_grad(False)
        self.assertEqual(tree_norm(detached_grads['high_actor_head']), 0.0)
        self.assertGreater(tree_norm(open_grads['high_actor_head']), 0.0)

    def test_reset_carry_teacher_free_shapes_and_action_interface(self):
        num_subgoals = 2
        observations, goals, subgoals = synthetic_inputs(num_subgoals)
        expected_shapes = None

        for state_init_mode in ('reset', 'carry'):
            network = make_network(
                num_subgoals,
                state_init_mode=state_init_mode,
            )
            params = initialize(network, num_subgoals)
            for teacher_targets in (subgoals, None):
                high_dists, low_dist, actions, diagnostics = network.apply(
                    {'params': params},
                    observations,
                    goals,
                    jax.random.PRNGKey(8),
                    subgoal_reps=teacher_targets,
                    return_diagnostics=True,
                )
                shapes = (
                    tuple(dist.mode().shape for dist in high_dists),
                    low_dist.mode().shape,
                    actions.shape,
                )
                if expected_shapes is None:
                    expected_shapes = shapes
                self.assertEqual(shapes, expected_shapes)
                self.assertEqual(len(high_dists), num_subgoals)
                self.assertEqual(low_dist.mode().shape, (BATCH_SIZE, ACTION_DIM))
                self.assertEqual(actions.shape, (BATCH_SIZE, ACTION_DIM))
                for segment_idx in range(num_subgoals):
                    for metric in (
                        'predicted_subgoal_norm',
                        'final_z_low_norm',
                        'final_z_high_norm',
                    ):
                        self.assertIn(
                            f'stage_{segment_idx}/{metric}',
                            diagnostics,
                        )
                assert_finite_tree(self, diagnostics)

    def test_cells_and_readout_share_parameters_across_cycles_and_segments(self):
        network = make_network(num_subgoals=2, h_cycles=3, l_cycles=2)
        params = initialize(network, num_subgoals=2)
        flat_paths = {
            '/'.join(path)
            for path in flax.traverse_util.flatten_dict(params).keys()
        }
        for module_name in ('low_cell', 'high_cell', 'subgoal_readout'):
            self.assertTrue(
                any(path.startswith(f'{module_name}/') for path in flat_paths)
            )
        forbidden_names = (
            'low_cell_1',
            'high_cell_1',
            'subgoal_readout_1',
            'cycle_readout',
            'cycle_actor',
        )
        self.assertFalse(
            any(
                forbidden in path
                for path in flat_paths
                for forbidden in forbidden_names
            )
        )

    def test_forward_is_jittable_for_reset_and_carry(self):
        num_subgoals = 2
        observations, goals, subgoals = synthetic_inputs(num_subgoals)
        for state_init_mode in ('reset', 'carry'):
            with self.subTest(state_init_mode=state_init_mode):
                network = make_network(
                    num_subgoals,
                    state_init_mode=state_init_mode,
                )
                params = initialize(network, num_subgoals)

                @jax.jit
                def run(current_params, obs, goal, teacher_subgoals, key):
                    high_dists, low_dist, actions = network.apply(
                        {'params': current_params},
                        obs,
                        goal,
                        key,
                        subgoal_reps=teacher_subgoals,
                    )
                    return (
                        jnp.stack([dist.mode() for dist in high_dists], axis=1),
                        low_dist.mode(),
                        actions,
                    )

                high_modes, low_mode, actions = run(
                    params,
                    observations,
                    goals,
                    subgoals,
                    jax.random.PRNGKey(9),
                )
                self.assertEqual(
                    high_modes.shape,
                    (BATCH_SIZE, num_subgoals, STATE_DIM),
                )
                self.assertEqual(low_mode.shape, (BATCH_SIZE, ACTION_DIM))
                self.assertEqual(actions.shape, (BATCH_SIZE, ACTION_DIM))

    def test_temperature_zero_matches_deterministic_free_running(self):
        network = make_network(num_subgoals=2)
        params = initialize(network, num_subgoals=2)
        observations, goals, _ = synthetic_inputs(num_subgoals=2)

        def second_subgoal(key, temperature):
            high_dists, _, _ = network.apply(
                {'params': params},
                observations,
                goals,
                key,
                subgoal_reps=None,
                temperature=temperature,
            )
            return high_dists[1].mode()

        zero_a = second_subgoal(jax.random.PRNGKey(13), 0.0)
        zero_b = second_subgoal(jax.random.PRNGKey(14), 0.0)
        noisy_a = second_subgoal(jax.random.PRNGKey(13), 1.0)
        noisy_b = second_subgoal(jax.random.PRNGKey(14), 1.0)
        np.testing.assert_array_equal(zero_a, zero_b)
        self.assertFalse(np.allclose(noisy_a, noisy_b))

    def test_config_and_invalid_modes(self):
        config = get_config()
        self.assertEqual(config.hrm_h_cycles, 2)
        self.assertEqual(config.hrm_l_cycles, 2)
        self.assertEqual(config.segment_state_init_mode, 'carry')
        self.assertEqual(config.segment_carry_gradient_mode, 'detached')
        self.assertTrue(config.hrm_trainable_init)
        self.assertEqual(config.hrm_readout_mode, 'residual')
        self.assertEqual(config.hrm_readout_residual_alpha, 0.05)
        self.assertFalse(config.stage_embedding_in_context)
        self.assertNotIn('hrm_gradient_mode', config)

        observations, goals, _ = synthetic_inputs(num_subgoals=1)
        invalid_network = make_network(
            num_subgoals=1,
            state_init_mode='unknown',
        )
        with self.assertRaisesRegex(ValueError, 'segment_state_init_mode'):
            invalid_network.init(
                jax.random.PRNGKey(10),
                observations,
                goals,
                jax.random.PRNGKey(11),
            )


if __name__ == '__main__':
    unittest.main()
