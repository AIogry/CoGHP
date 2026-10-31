"""Contracts for final-readout HRM planning with CoGHP peripherals.

The pilot's detached deep supervision remains covered separately. These
tests check the intended v0 objective, gradient path, and experiment controls.
"""

import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "impls"))

import flax
import jax
import jax.numpy as jnp
import numpy as np

from agents.navigation import NavigationAgent, get_config
from networks.backbones import BlockStack, SequenceBackbone
from utils.flax_utils import (
    build_checkpoint_metadata,
    restore_agent,
    save_agent,
    validate_checkpoint_metadata,
)
from utils.networks import MLP


def small_config(**overrides):
    config = get_config("joint_planner").to_dict()
    # Shrink only peripheral hidden layers and the latent state for CPU tests;
    # keep the authorized 32-wide Mixer cores and full BPTT.
    config.update(
        actor_hidden_dims=(16,), value_hidden_dims=(16,), enc_hidden_dims=(16,),
        feature_dim=8, subgoal_steps=2,
        global_hidden_dims=(16,), global_dim=8, global_goal_tolerance=0.1,
    )
    config.update(overrides)
    return config


def batch():
    rng = np.random.default_rng(105)

    def normal(shape):
        return jnp.asarray(rng.normal(size=shape), dtype=jnp.float32)

    return dict(
        observations=normal((3, 5)), next_observations=normal((3, 5)),
        actions=normal((3, 2)), value_goals=normal((3, 5)),
        high_actor_goals=normal((3, 5)), high_actor_targets=normal((3, 3, 5)),
        rewards=jnp.array([-1., 0., -1.]), masks=jnp.array([1., 0., 1.]),
    )


def assert_trees_equal(test, a, b):
    test.assertEqual(jax.tree_util.tree_structure(a), jax.tree_util.tree_structure(b))
    for left, right in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
        np.testing.assert_array_equal(left, right)


class JointPlannerV0Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.batch = batch()
        cls.config = small_config()
        cls.agent = NavigationAgent.create(
            3, cls.batch["observations"][:1], cls.batch["actions"][:1], cls.config
        )

    def plans(self, commands=None, params=None, agent=None):
        agent = self.agent if agent is None else agent
        return agent.policy(
            self.batch["observations"], self.batch["high_actor_goals"],
            jax.random.PRNGKey(9), commands, params,
            return_latents=True, return_planner_rounds=True,
        )

    def test_profile_keeps_only_planner_hrm_and_32_wide_mixers(self):
        config = get_config("joint_planner")
        self.assertEqual(config.joint_plan_objective, "waypoint_final")
        self.assertEqual(config.planner_refinement_steps, 1)
        self.assertTrue(config.planner_bptt)
        self.assertEqual((config.planner_h_cycles, config.planner_l_cycles), (2, 2))
        self.assertEqual(config.planner_mixer_intermediate_size, 32)
        self.assertEqual(config.mixer_hidden, 32)
        self.assertEqual(config.network_backbone, "mlp")
        self.assertEqual(config.action_core_type, "vanilla_coghp")
        self.assertEqual(config.action_block_type, "coghp_gelu")
        for key in ("enc_hidden_dims", "value_hidden_dims", "actor_hidden_dims"):
            self.assertEqual(tuple(config[key]), (512, 512, 512))

        modules = self.agent.network.model_def.modules
        self.assertIsInstance(modules["goal_rep"].layers[0], MLP)
        self.assertTrue(modules["goal_rep"].layers[0].layer_norm)
        for name in ("value", "target_value"):
            self.assertIsNone(modules[name].backbone)
            self.assertTrue(modules[name].layer_norm)
        actor = modules["actor_mixer"]
        self.assertIsNone(actor.feature_backbone)
        self.assertIsNone(actor.high_actor_head.backbone)
        self.assertIsNone(actor.low_actor_head.backbone)

        actor_params = self.agent.network.params["modules_actor_mixer"]
        self.assertIn("planner_core", actor_params)
        self.assertNotIn("action_hrm", actor_params)
        self.assertNotIn("shared_actor_core", actor_params)
        self.assertTrue(any(name.startswith("action_mixer_blocks_") for name in actor_params))
        # Gate projection parameters must be confined to the planner core.
        gated = [path for path in flax.traverse_util.flatten_dict(self.agent.network.params)
                 if any("gate" in name for name in path)]
        self.assertTrue(gated)
        self.assertTrue(all("planner_core" in path for path in gated))
        for head in ("high_actor_head", "low_actor_head"):
            head_paths = [path for path in flax.traverse_util.flatten_dict(actor_params)
                          if head in path]
            self.assertTrue(head_paths)
            self.assertFalse(any("LayerNorm" in name for path in head_paths for name in path))

    def test_only_final_plan_is_read_and_it_has_no_teacher_prefix(self):
        commands = self.agent.command_labels(self.batch)
        teacher = self.plans(commands)
        changed = self.plans(commands + 10)
        predicted = self.plans()
        self.assertEqual(len(teacher[0]), 1)
        self.assertEqual(len(teacher[0][0]), 3)
        for a, b, c in zip(teacher[0][0], changed[0][0], predicted[0][0]):
            self.assertEqual(a.mode().shape, (3, 8))
            np.testing.assert_array_equal(a.mode(), b.mode())
            np.testing.assert_array_equal(a.mode(), c.mode())
        np.testing.assert_array_equal(teacher[3], changed[3])
        np.testing.assert_array_equal(teacher[3], predicted[3])
        # Teacher targets still condition the CoGHP executor during training.
        self.assertGreater(float(jnp.linalg.norm(teacher[1].mode() - changed[1].mode())), 1e-6)

    def test_final_objective_matches_existing_per_waypoint_awr(self):
        dists = self.plans()[0][0]
        obs_expand = jnp.broadcast_to(
            self.batch["observations"][:, None], self.batch["high_actor_targets"].shape
        )
        actual, info = self.agent.multi_high_actor_loss(
            self.batch, dists, obs_expand, self.agent.network.params
        )
        original, _ = self.agent._detail_high_actor_loss(
            self.batch, dists, obs_expand, self.agent.network.params
        )
        np.testing.assert_allclose(actual, original, rtol=1e-6, atol=1e-6)

        labels = self.agent.command_labels(self.batch)
        advantages = self.agent.high_advantages(self.batch)
        weights = jnp.minimum(jnp.exp(3.0 * advantages), 100.0)
        logps = jnp.stack([dist.log_prob(labels[:, j]) for j, dist in enumerate(dists)], axis=1)
        expected = -jnp.mean(jnp.sum(weights * logps * jnp.array([.64, .8, 1.]), axis=1)) / (3 * 2)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
        for j in range(3):
            self.assertIn(f"weight_{j}/mean", info)
            np.testing.assert_allclose(info[f"weight_{j}/mean"], weights[:, j].mean(), rtol=1e-6)

        # Check the training entry point actually chooses this objective.
        total, high_info, low_info, _ = self.agent.actor_loss(
            self.batch, self.agent.network.params, jax.random.PRNGKey(9)
        )
        np.testing.assert_allclose(high_info["joint_plan_loss"], actual, rtol=1e-6)
        np.testing.assert_allclose(total, actual + low_info["actor_loss"], rtol=1e-6)

    def test_h_l_sweep_preserves_initial_weights_at_fixed_plan_size(self):
        for h, ell in ((1, 1), (4, 2)):
            with self.subTest(h=h, ell=ell):
                other = NavigationAgent.create(
                    3, self.batch["observations"][:1], self.batch["actions"][:1],
                    small_config(planner_h_cycles=h, planner_l_cycles=ell),
                )
                assert_trees_equal(self, self.agent.network.params, other.network.params)
                assert_trees_equal(self, self.agent.network.model_state, other.network.model_state)
                self.assertEqual(len(self.plans(agent=other)[0]), 1)

    def test_v0_rejects_detached_or_multi_readout_settings(self):
        for overrides in (
            dict(planner_refinement_steps=2),
            dict(planner_bptt=False),
            dict(joint_plan_objective="unknown"),
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                NavigationAgent.create(
                    3, self.batch["observations"][:1], self.batch["actions"][:1],
                    small_config(**overrides),
                )

    def test_update_checkpoint_and_exact_resume(self):
        updated, info = self.agent.update(self.batch)
        self.assertIn("high_actor/joint_plan_loss", info)
        for leaf in jax.tree_util.tree_leaves((updated.network.params, info)):
            self.assertTrue(np.isfinite(np.asarray(leaf)).all())
        metadata = build_checkpoint_metadata(
            self.config, environment="synthetic", seed=3, git_commit="fixture"
        )
        with tempfile.TemporaryDirectory() as directory:
            save_agent(updated, directory, 1, checkpoint_metadata=metadata)
            restored = restore_agent(
                self.agent, directory, 1, expected_checkpoint_metadata=metadata,
                require_checkpoint_metadata=True,
            )
        continued, continued_info = updated.update(self.batch)
        resumed, resumed_info = restored.update(self.batch)
        assert_trees_equal(self, (continued, continued_info), (resumed, resumed_info))
        actions = restored.sample_actions(
            self.batch["observations"][0], self.batch["high_actor_goals"][0],
            jax.random.PRNGKey(4), 0.0,
        )
        self.assertEqual(actions.shape, (2,))
        self.assertTrue(np.isfinite(np.asarray(actions)).all())
        for overrides in (
            dict(joint_plan_objective="endpoint_deep"),
            dict(planner_h_cycles=4), dict(planner_l_cycles=1),
            dict(num_subgoals=1), dict(planner_mixer_intermediate_size=64),
            dict(high_discount=1.0),
        ):
            changed = build_checkpoint_metadata(
                small_config(**overrides), environment="synthetic", seed=3, git_commit="fixture",
            )
            with self.subTest(overrides=overrides), self.assertRaisesRegex(ValueError, "signature mismatch"):
                validate_checkpoint_metadata(metadata, changed, require_metadata=True)


class FinalReadoutBPTTTests(unittest.TestCase):
    def test_final_output_backpropagates_through_all_hrm_updates(self):
        common = dict(
            seq_len=6, num_layers=1, hidden_size=8, intermediate_size=16,
            head_dim=4, is_causal=False, is_mlp_mixer=True,
            block_type="swiglu_rmsnorm", mlp_mixer_intermediate_size=32,
        )
        core = SequenceBackbone(**common, h_cycles=3, l_cycles=2, bptt=True)
        inputs = jax.random.normal(jax.random.PRNGKey(91), (2, 6, 8))
        variables = core.init(jax.random.PRNGKey(92), None, inputs)
        carry = {
            "z_H": jax.random.normal(jax.random.PRNGKey(93), inputs.shape),
            "z_L": jax.random.normal(jax.random.PRNGKey(94), inputs.shape),
        }
        target = jax.random.normal(jax.random.PRNGKey(95), inputs.shape)
        block = BlockStack(**common)

        def explicit_unroll(params, initial, x):
            high, low = initial["z_H"], initial["z_L"]
            for _ in range(3):
                for _ in range(2):
                    low = block.apply({"params": params["L_level"]}, low + high + x)
                high = block.apply({"params": params["H_level"]}, high + low)
            return high

        def actual(params, initial, x):
            return core.apply({**variables, "params": params}, initial, x)[1]

        np.testing.assert_allclose(
            actual(variables["params"], carry, inputs),
            explicit_unroll(variables["params"], carry, inputs), rtol=1e-6, atol=1e-6,
        )

        def gradients(forward):
            return jax.jit(jax.grad(
                lambda params, initial, x: jnp.mean((forward(params, initial, x) - target) ** 2),
                argnums=(0, 1, 2),
            ))(variables["params"], carry, inputs)

        full = gradients(actual)
        expected = gradients(explicit_unroll)
        for a, b in zip(jax.tree_util.tree_leaves(full), jax.tree_util.tree_leaves(expected)):
            np.testing.assert_allclose(a, b, rtol=3e-5, atol=3e-6)
            self.assertTrue(np.isfinite(np.asarray(a)).all())
        for name in ("z_H", "z_L"):
            self.assertGreater(float(jnp.linalg.norm(full[1][name])), 1e-7)

        # This counterexample ensures the check detects the old truncation:
        # final H/L parameters can have gradients even when earlier state does not.
        truncated = core.clone(bptt=False)
        detached = jax.grad(lambda initial: jnp.mean((
            truncated.apply(variables, initial, inputs)[1] - target
        ) ** 2))(carry)
        for leaf in jax.tree_util.tree_leaves(detached):
            np.testing.assert_array_equal(leaf, jnp.zeros_like(leaf))


if __name__ == "__main__":
    unittest.main()
