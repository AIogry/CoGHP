"""Gradient contracts for joint encoders and detached executor commands.

The planner still receives its own high loss through full BPTT. The executor
uses the same predicted means, but cannot train the planner through those
means. All three losses retain their intended direct goal-encoder routes.
"""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "impls"))

import flax
import jax
import jax.numpy as jnp
import numpy as np

from agents.navigation import NavigationAgent, get_config
from utils.flax_utils import build_checkpoint_metadata, validate_checkpoint_metadata


def small_config(profile="joint_planner_v2", **overrides):
    config = get_config(profile).to_dict()
    config.update(
        actor_hidden_dims=(16,), value_hidden_dims=(16,), enc_hidden_dims=(16,),
        feature_dim=8, subgoal_steps=2,
        global_hidden_dims=(16,), global_dim=8, global_goal_tolerance=0.1,
        planner_h_cycles=2, planner_l_cycles=1, num_subgoals=3,
    )
    config.update(overrides)
    return config


def make_batch():
    rng = np.random.default_rng(219)

    def normal(shape):
        return jnp.asarray(rng.normal(size=shape), dtype=jnp.float32)

    return dict(
        observations=normal((3, 5)), next_observations=normal((3, 5)),
        actions=normal((3, 2)), value_goals=normal((3, 5)),
        high_actor_goals=normal((3, 5)), high_actor_targets=normal((3, 3, 5)),
        rewards=jnp.array([-1., 0., -1.]), masks=jnp.array([1., 0., 1.]),
    )


def magnitude(tree):
    return sum(float(jnp.linalg.norm(x)) for x in jax.tree_util.tree_leaves(tree))


def assert_trees_equal(test, left, right):
    test.assertEqual(jax.tree_util.tree_structure(left), jax.tree_util.tree_structure(right))
    for a, b in zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right)):
        np.testing.assert_array_equal(a, b)


def actor_group(gradients, group):
    flat = flax.traverse_util.flatten_dict(gradients["modules_actor_mixer"])
    return [grad for path, grad in flat.items()
            if any(part.startswith(group) for part in path)]


class JointPlannerV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.batch = make_batch()
        cls.config = small_config()
        cls.agent = cls.make_agent(cls.config)
        cls.attached_agent = cls.make_agent(small_config(executor_command_stop_grad=False))

    @classmethod
    def make_agent(cls, config):
        return NavigationAgent.create(
            23, cls.batch["observations"][:1], cls.batch["actions"][:1], config,
        )

    def policy(self, agent, *, temperature=1., seed=7, commands=None):
        return agent.policy(
            self.batch["observations"], self.batch["high_actor_goals"],
            jax.random.PRNGKey(seed), commands, agent.network.params,
            temperature=temperature, return_latents=True,
        )

    def test_profile_changes_only_encoder_and_executor_gradient_contract(self):
        before = get_config("joint_planner_v1").to_dict()
        after = get_config("joint_planner_v2").to_dict()
        self.assertEqual(after["train_action_chain_source"], "predicted")
        self.assertEqual(after["planner_command_mode"], "mean")
        self.assertTrue(after["actor_goal_rep_grad"])
        self.assertTrue(after["executor_command_stop_grad"])
        self.assertEqual(after["high_alpha"], 3.)
        self.assertEqual(after["joint_plan_objective"], "waypoint_final")
        self.assertTrue(after["planner_bptt"])
        self.assertEqual(after["planner_refinement_steps"], 1)
        # This experiment must not silently change the objective, dataset,
        # sampling stride, architecture, optimizer, or old defaults.
        allowed = {"actor_goal_rep_grad", "executor_command_stop_grad", "navigation_protocol"}
        changed = {key for key in before.keys() | after.keys()
                   if before.get(key) != after.get(key)}
        self.assertTrue(changed <= allowed, changed)
        self.assertFalse(before.get("executor_command_stop_grad", False))
        for profile in ("joint_planner", "joint_planner_pilot", "legacy_navigation"):
            self.assertFalse(get_config(profile).get("executor_command_stop_grad", False))

    def test_detachment_preserves_parameters_predictions_and_likelihoods(self):
        assert_trees_equal(self, self.agent.network.params, self.attached_agent.network.params)
        assert_trees_equal(self, self.agent.network.model_state, self.attached_agent.network.model_state)
        commands = self.agent.command_labels(self.batch)
        reference = self.policy(self.agent)
        for agent, temperature, seed, labels in (
            (self.attached_agent, 1., 7, None),
            (self.agent, 1., 91, commands + 10.),
            (self.agent, 0., 38, None),
        ):
            with self.subTest(detached=agent.config["executor_command_stop_grad"],
                              temperature=temperature, seed=seed):
                actual = self.policy(agent, temperature=temperature, seed=seed, commands=labels)
                np.testing.assert_array_equal(reference[3], actual[3])
                np.testing.assert_array_equal(reference[1].mode(), actual[1].mode())
                for expected_dist, actual_dist in zip(reference[0], actual[0]):
                    np.testing.assert_array_equal(expected_dist.mode(), actual_dist.mode())
        for dist in (*reference[0], reference[1]):
            np.testing.assert_allclose(dist.scale_diag, 1.)
            self.assertTrue(np.isfinite(np.asarray(dist.log_prob(dist.mode()))).all())
        evaluation = self.policy(self.agent, temperature=0.)
        np.testing.assert_array_equal(evaluation[2], evaluation[1].mode())

    def test_scalar_losses_have_the_intended_gradient_routes(self):
        def gradients(agent, component):
            # Differentiate each scalar independently, as the optimizer does.
            # Stacking losses and calling jacrev is not this regression test.
            def scalar(params):
                if component == "value":
                    return agent.value_loss(self.batch, params)[0]
                _, high_info, low_info, _ = agent.actor_loss(
                    self.batch, params, jax.random.PRNGKey(7),
                )
                return (high_info["joint_plan_loss"] if component == "high"
                        else low_info["actor_loss"])

            return jax.jit(jax.grad(scalar))(agent.network.params)

        high = gradients(self.agent, "high")
        low = gradients(self.agent, "low")
        value = gradients(self.agent, "value")
        attached_low = gradients(self.attached_agent, "low")
        for name, grad in (("high", high), ("low", low), ("value", value)):
            self.assertGreater(magnitude(grad["modules_goal_rep"]), 1e-8, name)
            for leaf in jax.tree_util.tree_leaves(grad):
                self.assertTrue(np.isfinite(np.asarray(leaf)).all(), name)

        for group in ("H_level", "L_level", "planner_query_tokens", "high_actor_head"):
            with self.subTest(planner_group=group):
                high_group = actor_group(high, group)
                low_group = actor_group(low, group)
                control_group = actor_group(attached_low, group)
                self.assertTrue(high_group, f"Missing parameter group: {group}")
                self.assertGreater(magnitude(high_group), 1e-8, group)
                # The same initial network really has a nonzero executor-to-
                # planner signal when its command is left attached.
                self.assertGreater(magnitude(control_group), 1e-8, group)
                for leaf in low_group:
                    np.testing.assert_array_equal(leaf, jnp.zeros_like(leaf), err_msg=group)

        for group in ("low_actor_head", "action_mixer_blocks_", "feature_embed"):
            self.assertGreater(magnitude(actor_group(low, group)), 1e-8, group)

    def test_real_jitted_update_is_finite_and_changes_encoder(self):
        updated, info = self.agent.update(self.batch)
        for leaf in jax.tree_util.tree_leaves((updated.network.params, info)):
            self.assertTrue(np.isfinite(np.asarray(leaf)).all())
        self.assertIn("high_actor/joint_plan_loss", info)
        delta = jax.tree_util.tree_map(
            lambda new, old: new - old,
            updated.network.params["modules_goal_rep"],
            self.agent.network.params["modules_goal_rep"],
        )
        self.assertGreater(magnitude(delta), 1e-8)

    def test_checkpoint_signature_records_detachment_without_changing_old_defaults(self):
        def metadata(config):
            return build_checkpoint_metadata(
                config, environment="synthetic", seed=23, git_commit="fixture",
            )

        detached = metadata(self.config)
        attached = metadata(small_config(executor_command_stop_grad=False))
        for loaded, expected in ((detached, attached), (attached, detached)):
            with self.assertRaisesRegex(ValueError, "signature mismatch"):
                validate_checkpoint_metadata(loaded, expected, require_metadata=True)

        for profile in ("joint_planner", "joint_planner_v1"):
            explicit = small_config(profile, executor_command_stop_grad=False)
            historical = dict(explicit)
            historical.pop("executor_command_stop_grad")
            with self.subTest(profile=profile):
                self.assertEqual(metadata(explicit)["signature"], metadata(historical)["signature"])
                validate_checkpoint_metadata(
                    metadata(historical), metadata(explicit), require_metadata=True,
                )

    def test_rejects_non_boolean_detachment_flag(self):
        for invalid in ("false", "true", 0, 1, None):
            with self.subTest(value=invalid), self.assertRaisesRegex(
                ValueError, "executor_command_stop_grad must be a bool",
            ):
                self.make_agent(small_config(executor_command_stop_grad=invalid))


if __name__ == "__main__":
    unittest.main()
