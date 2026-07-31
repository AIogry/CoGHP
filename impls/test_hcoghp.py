"""Synthetic tests for autoregressive-subgoal HCoGHP."""

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

from utils.hcoghp_network import (  # noqa: E402
    HCoGHPPolicyNetwork,
    apply_stage_gradient_boundary,
)
from utils.networks import GCActor  # noqa: E402


STATE_DIM = 8
ACTION_DIM = 3
BATCH_SIZE = 4


def make_network(
    num_subgoals,
    trainable_init=False,
    full_chain=True,
    gradient_mode='stage_truncated',
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
        hrm_l_cycles=2,
        hrm_init_std=0.1,
        hrm_trainable_init=trainable_init,
        hrm_low_num_blocks=1,
        hrm_high_num_blocks=1,
        subgoal_readout_hidden_dim=16,
        hrm_gradient_mode=gradient_mode,
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


class HCoGHPPolicyNetworkTest(unittest.TestCase):
    def test_current_stage_awr_path_trains_all_planner_modules(self):
        network = make_network(num_subgoals=1)
        params = initialize(network, num_subgoals=1)
        observations, goals, subgoals = synthetic_inputs(num_subgoals=1)

        def stage_loss(current_params):
            high_dists, _, _ = network.apply(
                {'params': current_params},
                observations,
                goals,
                jax.random.PRNGKey(10),
                subgoal_reps=subgoals,
            )
            return -jnp.mean(high_dists[0].log_prob(subgoals[:, 0, :]))

        grads = jax.grad(stage_loss)(params)
        for module_name in (
            'context_mixer_blocks_0',
            'low_cell',
            'high_cell',
            'subgoal_readout',
            'high_actor_head',
        ):
            assert_finite_tree(self, grads[module_name])
            self.assertGreater(tree_norm(grads[module_name]), 0.0)

    def test_stage_boundary_truncates_low_high_and_token_gradients(self):
        z_low = jnp.arange(BATCH_SIZE * STATE_DIM, dtype=jnp.float32).reshape(
            BATCH_SIZE,
            STATE_DIM,
        ) / 10.0
        z_high = z_low + 0.25
        token = z_low - 0.5

        def next_stage_loss(low, high, selected, mode):
            next_low, next_high, next_token = apply_stage_gradient_boundary(
                low,
                high,
                selected,
                mode,
            )
            return (
                jnp.sum(next_low**2)
                + jnp.sum(next_high**2)
                + jnp.sum(next_token**2)
            )

        truncated_grads = jax.grad(
            lambda low, high, selected: next_stage_loss(
                low,
                high,
                selected,
                'stage_truncated',
            ),
            argnums=(0, 1, 2),
        )(z_low, z_high, token)
        self.assertTrue(
            all(np.allclose(np.asarray(grad), 0.0) for grad in truncated_grads)
        )

        full_grads = jax.grad(
            lambda low, high, selected: next_stage_loss(
                low,
                high,
                selected,
                'full_bptt',
            ),
            argnums=(0, 1, 2),
        )(z_low, z_high, token)
        self.assertTrue(
            all(float(jnp.linalg.norm(grad)) > 0.0 for grad in full_grads)
        )

    def test_shapes_teacher_forcing_free_running_and_diagnostics(self):
        for num_subgoals in (0, 1, 2):
            with self.subTest(num_subgoals=num_subgoals):
                network = make_network(num_subgoals)
                params = initialize(network, num_subgoals)
                observations, goals, subgoals = synthetic_inputs(num_subgoals)

                for teacher_targets in (subgoals, None):
                    outputs = network.apply(
                        {'params': params},
                        observations,
                        goals,
                        jax.random.PRNGKey(2),
                        subgoal_reps=teacher_targets,
                        return_diagnostics=True,
                    )
                    high_dists, low_dist, actions, diagnostics = outputs
                    self.assertEqual(len(high_dists), num_subgoals)
                    for high_dist in high_dists:
                        self.assertEqual(
                            high_dist.mode().shape,
                            (BATCH_SIZE, STATE_DIM),
                        )
                    self.assertEqual(
                        low_dist.mode().shape,
                        (BATCH_SIZE, ACTION_DIM),
                    )
                    self.assertEqual(actions.shape, (BATCH_SIZE, ACTION_DIM))
                    self.assertEqual(
                        float(diagnostics['hrm/num_stages']),
                        num_subgoals,
                    )
                    self.assertIn('action/context_norm', diagnostics)
                    self.assertIn('action/action_mode_norm', diagnostics)
                    if num_subgoals > 0:
                        self.assertIn(
                            'stage_0/subgoal_readout_norm',
                            diagnostics,
                        )
                    assert_finite_tree(self, diagnostics)

    def test_low_and_high_states_carry_between_subgoal_stages(self):
        num_subgoals = 2
        network = make_network(num_subgoals)
        params = initialize(network, num_subgoals)
        observations, goals, subgoals = synthetic_inputs(num_subgoals)
        _, _, _, diagnostics = network.apply(
            {'params': params},
            observations,
            goals,
            jax.random.PRNGKey(3),
            subgoal_reps=subgoals,
            return_diagnostics=True,
        )

        expected_low_init_norm = jnp.mean(
            jnp.linalg.norm(
                jnp.tile(params['z_l_init'], (BATCH_SIZE, 1)),
                axis=-1,
            )
        )
        expected_high_init_norm = jnp.mean(
            jnp.linalg.norm(
                jnp.tile(params['z_h_init'], (BATCH_SIZE, 1)),
                axis=-1,
            )
        )
        np.testing.assert_allclose(
            np.asarray(diagnostics['stage_0/z_low_before_norm']),
            np.asarray(expected_low_init_norm),
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            np.asarray(diagnostics['stage_0/z_high_before_norm']),
            np.asarray(expected_high_init_norm),
            rtol=0.0,
            atol=0.0,
        )
        for stage in range(1, num_subgoals):
            np.testing.assert_allclose(
                np.asarray(diagnostics[f'stage_{stage}/z_low_before_norm']),
                np.asarray(
                    diagnostics[f'stage_{stage - 1}/z_low_after_norm']
                ),
                rtol=0.0,
                atol=0.0,
            )
            self.assertFalse(
                np.allclose(
                    np.asarray(diagnostics[f'stage_{stage}/z_low_before_norm']),
                    np.asarray(expected_low_init_norm),
                )
            )
            np.testing.assert_allclose(
                np.asarray(diagnostics[f'stage_{stage}/z_high_before_norm']),
                np.asarray(
                    diagnostics[f'stage_{stage - 1}/z_high_after_norm']
                ),
                rtol=0.0,
                atol=0.0,
            )

    def test_action_stage_does_not_run_hrm(self):
        num_subgoals = 2
        network = make_network(num_subgoals)
        params = initialize(network, num_subgoals)
        observations, goals, subgoals = synthetic_inputs(num_subgoals)
        _, _, _, diagnostics = network.apply(
            {'params': params},
            observations,
            goals,
            jax.random.PRNGKey(4),
            subgoal_reps=subgoals,
            return_diagnostics=True,
        )

        self.assertFalse(
            any(key.startswith(f'stage_{num_subgoals}/') for key in diagnostics)
        )
        self.assertTrue(any(key.startswith('action/') for key in diagnostics))
        self.assertEqual(float(diagnostics['hrm/num_stages']), num_subgoals)

        # Under pure teacher forcing the action token chain is independent of
        # HRM outputs, so action-only gradients cannot reach HRM/readout params.
        def action_objective(current_params):
            _, low_dist, _ = network.apply(
                {'params': current_params},
                observations,
                goals,
                jax.random.PRNGKey(5),
                subgoal_reps=subgoals,
                scheduled_sampling_prob=0.0,
            )
            return jnp.sum(low_dist.mode())

        grads = jax.grad(action_objective)(params)
        for module_name in ('low_cell', 'high_cell', 'subgoal_readout'):
            self.assertEqual(tree_norm(grads[module_name]), 0.0)
        self.assertGreater(tree_norm(grads['context_mixer_blocks_0']), 0.0)
        self.assertGreater(tree_norm(grads['low_actor_head']), 0.0)

    def test_action_loss_cannot_cross_detached_subgoal_prefix(self):
        num_subgoals = 2
        observations, goals, subgoals = synthetic_inputs(num_subgoals)

        def action_input_gradient(mode):
            network = make_network(num_subgoals, gradient_mode=mode)
            params = initialize(network, num_subgoals)

            def action_loss(explicit_subgoals):
                _, low_dist, _ = network.apply(
                    {'params': params},
                    observations,
                    goals,
                    jax.random.PRNGKey(11),
                    subgoal_reps=explicit_subgoals,
                    scheduled_sampling_prob=0.0,
                )
                return jnp.sum(low_dist.mode())

            return jax.grad(action_loss)(subgoals)

        truncated_grad = action_input_gradient('stage_truncated')
        np.testing.assert_array_equal(np.asarray(truncated_grad), 0.0)
        full_bptt_grad = action_input_gradient('full_bptt')
        self.assertGreater(float(jnp.linalg.norm(full_bptt_grad)), 0.0)

    def test_gradient_modes_have_identical_forward_values(self):
        num_subgoals = 2
        truncated_network = make_network(
            num_subgoals,
            gradient_mode='stage_truncated',
        )
        full_network = make_network(num_subgoals, gradient_mode='full_bptt')
        params = initialize(truncated_network, num_subgoals)
        observations, goals, subgoals = synthetic_inputs(num_subgoals)

        def run(network):
            high_dists, low_dist, actions, diagnostics = network.apply(
                {'params': params},
                observations,
                goals,
                jax.random.PRNGKey(12),
                subgoal_reps=subgoals,
                scheduled_sampling_prob=1.0,
                return_diagnostics=True,
            )
            return (
                jnp.stack([dist.mode() for dist in high_dists], axis=1),
                low_dist.mode(),
                actions,
                diagnostics,
            )

        truncated_high, truncated_low, truncated_actions, truncated_info = run(
            truncated_network
        )
        full_high, full_low, full_actions, full_info = run(full_network)
        np.testing.assert_array_equal(truncated_high, full_high)
        np.testing.assert_array_equal(truncated_low, full_low)
        np.testing.assert_array_equal(truncated_actions, full_actions)

        for stage in range(num_subgoals):
            for metric in (
                'z_low_before_norm',
                'z_low_after_norm',
                'z_high_before_norm',
                'z_high_after_norm',
            ):
                np.testing.assert_array_equal(
                    truncated_info[f'stage_{stage}/{metric}'],
                    full_info[f'stage_{stage}/{metric}'],
                )
            self.assertEqual(
                float(truncated_info[f'stage_{stage}/carry_truncated']),
                1.0,
            )
            self.assertEqual(
                float(full_info[f'stage_{stage}/carry_truncated']),
                0.0,
            )
        self.assertEqual(
            float(truncated_info['hrm/gradient_mode_is_stage_truncated']),
            1.0,
        )
        self.assertEqual(
            float(full_info['hrm/gradient_mode_is_stage_truncated']),
            0.0,
        )

    def test_stage_embedding_shape_identity_and_parameter_sharing(self):
        num_subgoals = 2
        network = make_network(num_subgoals)
        params = initialize(network, num_subgoals)
        embedding = params['stage_embedding']['embedding']
        self.assertEqual(embedding.shape, (num_subgoals + 1, STATE_DIM))
        self.assertFalse(np.allclose(np.asarray(embedding[0]), np.asarray(embedding[1])))
        self.assertFalse(np.allclose(np.asarray(embedding[1]), np.asarray(embedding[2])))

        flat_paths = {
            '/'.join(path)
            for path in flax.traverse_util.flatten_dict(params).keys()
        }
        for module_name in (
            'context_mixer_blocks_0',
            'context_mixer_blocks_1',
            'stage_embedding',
            'low_cell',
            'high_cell',
            'subgoal_readout',
        ):
            self.assertTrue(
                any(path.startswith(f'{module_name}/') for path in flat_paths)
            )
        forbidden_names = (
            'low_cell_1',
            'high_cell_1',
            'subgoal_readout_1',
            'action_low_cell',
            'action_high_cell',
            'action_stage_embedding',
        )
        self.assertFalse(
            any(
                forbidden in path
                for path in flat_paths
                for forbidden in forbidden_names
            )
        )

    def test_subgoal_distribution_uses_readout(self):
        num_subgoals = 2
        network = make_network(num_subgoals)
        params = initialize(network, num_subgoals)
        observations, goals, subgoals = synthetic_inputs(num_subgoals)

        def subgoal_objective(current_params):
            high_dists, _, _ = network.apply(
                {'params': current_params},
                observations,
                goals,
                jax.random.PRNGKey(6),
                subgoal_reps=subgoals,
            )
            return sum(jnp.sum(dist.mode()) for dist in high_dists)

        grads = jax.grad(subgoal_objective)(params)
        assert_finite_tree(self, grads['subgoal_readout'])
        self.assertGreater(tree_norm(grads['subgoal_readout']), 0.0)
        self.assertGreater(tree_norm(grads['high_actor_head']), 0.0)

    def test_scheduled_sampling_endpoints(self):
        num_subgoals = 2
        network = make_network(num_subgoals)
        params = initialize(network, num_subgoals)
        observations, goals, subgoals = synthetic_inputs(num_subgoals)

        for probability, expected_ratio in ((0.0, 0.0), (1.0, 1.0)):
            with self.subTest(probability=probability):
                high_dists, low_dist, actions, diagnostics = network.apply(
                    {'params': params},
                    observations,
                    goals,
                    jax.random.PRNGKey(7),
                    subgoal_reps=subgoals,
                    scheduled_sampling_prob=probability,
                    scheduled_sampling_use_mode=True,
                    scheduled_sampling_stop_gradient=True,
                    return_sampling_diagnostics=True,
                )
                self.assertEqual(len(high_dists), num_subgoals)
                self.assertEqual(
                    low_dist.mode().shape,
                    (BATCH_SIZE, ACTION_DIM),
                )
                self.assertEqual(actions.shape, (BATCH_SIZE, ACTION_DIM))
                assert_finite_tree(self, diagnostics)
                self.assertAlmostEqual(
                    float(diagnostics['scheduled_sampling_ratio']),
                    expected_ratio,
                )
                for stage in range(num_subgoals):
                    self.assertAlmostEqual(
                        float(
                            diagnostics[
                                f'scheduled_sampling_ratio_{stage}'
                            ]
                        ),
                        expected_ratio,
                    )

                def downstream_loss(explicit_subgoals):
                    downstream_high, downstream_low, _ = network.apply(
                        {'params': params},
                        observations,
                        goals,
                        jax.random.PRNGKey(7),
                        subgoal_reps=explicit_subgoals,
                        scheduled_sampling_prob=probability,
                        scheduled_sampling_use_mode=True,
                        scheduled_sampling_stop_gradient=False,
                    )
                    return (
                        jnp.sum(downstream_high[-1].mode())
                        + jnp.sum(downstream_low.mode())
                    )

                selected_token_input_grad = jax.grad(downstream_loss)(subgoals)
                np.testing.assert_array_equal(
                    np.asarray(selected_token_input_grad),
                    0.0,
                )

    def test_teacher_forced_awr_style_losses_are_finite(self):
        num_subgoals = 2
        network = make_network(num_subgoals)
        params = initialize(network, num_subgoals)
        observations, goals, subgoals = synthetic_inputs(num_subgoals)
        target_actions = jnp.zeros((BATCH_SIZE, ACTION_DIM), dtype=jnp.float32)

        def awr_style_loss(current_params):
            high_dists, low_dist, _ = network.apply(
                {'params': current_params},
                observations,
                goals,
                jax.random.PRNGKey(13),
                subgoal_reps=subgoals,
                scheduled_sampling_prob=0.0,
            )
            high_loss = -sum(
                jnp.mean(dist.log_prob(subgoals[:, idx, :]))
                for idx, dist in enumerate(high_dists)
            )
            low_loss = -jnp.mean(low_dist.log_prob(target_actions))
            return high_loss + low_loss

        loss, grads = jax.value_and_grad(awr_style_loss)(params)
        self.assertTrue(np.isfinite(float(loss)))
        assert_finite_tree(self, grads)
        self.assertGreater(tree_norm(grads), 0.0)

    def test_fixed_initial_states_have_zero_gradient(self):
        num_subgoals = 2
        network = make_network(num_subgoals, trainable_init=False)
        params = initialize(network, num_subgoals)
        observations, goals, subgoals = synthetic_inputs(num_subgoals)

        def objective(current_params):
            high_dists, low_dist, _ = network.apply(
                {'params': current_params},
                observations,
                goals,
                jax.random.PRNGKey(8),
                subgoal_reps=subgoals,
            )
            return (
                sum(jnp.sum(dist.mode()) for dist in high_dists)
                + jnp.sum(low_dist.mode())
            )

        grads = jax.grad(objective)(params)
        np.testing.assert_array_equal(np.asarray(grads['z_l_init']), 0.0)
        np.testing.assert_array_equal(np.asarray(grads['z_h_init']), 0.0)
        other_grads = {
            key: value
            for key, value in grads.items()
            if key not in ('z_l_init', 'z_h_init')
        }
        assert_finite_tree(self, other_grads)
        self.assertGreater(tree_norm(other_grads), 0.0)

    def test_forward_is_jittable(self):
        num_subgoals = 2
        observations, goals, subgoals = synthetic_inputs(num_subgoals)

        for gradient_mode in ('stage_truncated', 'full_bptt'):
            with self.subTest(gradient_mode=gradient_mode):
                network = make_network(
                    num_subgoals,
                    gradient_mode=gradient_mode,
                )
                params = initialize(network, num_subgoals)

                @jax.jit
                def run(current_params, obs, goal, teacher_subgoals, key):
                    high_dists, low_dist, actions, diagnostics = network.apply(
                        {'params': current_params},
                        obs,
                        goal,
                        key,
                        subgoal_reps=teacher_subgoals,
                        return_diagnostics=True,
                    )
                    high_modes = jnp.stack(
                        [dist.mode() for dist in high_dists],
                        axis=1,
                    )
                    return (
                        high_modes,
                        low_dist.mode(),
                        actions,
                        diagnostics['action/context_norm'],
                    )

                high_modes, low_mode, actions, action_context_norm = run(
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
                self.assertTrue(np.isfinite(np.asarray(action_context_norm)))

    def test_unknown_gradient_mode_is_rejected(self):
        network = make_network(num_subgoals=1, gradient_mode='unknown')
        observations, goals, _ = synthetic_inputs(num_subgoals=1)
        with self.assertRaisesRegex(ValueError, 'Unknown hrm_gradient_mode'):
            network.init(
                jax.random.PRNGKey(14),
                observations,
                goals,
                jax.random.PRNGKey(15),
            )


if __name__ == '__main__':
    unittest.main()
