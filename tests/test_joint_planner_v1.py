"""Contracts for predicted-command training and a value-only goal encoder.

These tests distinguish the deterministic command sent to the executor from
the Gaussian likelihood used by AWR, and check the actual gradient routes.
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


def small_config(profile="joint_planner_v1", **overrides):
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


class JointPlannerV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.batch = make_batch()
        cls.config = small_config()
        cls.agent = NavigationAgent.create(
            23, cls.batch["observations"][:1], cls.batch["actions"][:1], cls.config
        )

    def policy(self, agent=None, commands=None, temperature=1., seed=7, params=None,
               chain_source=None):
        agent = self.agent if agent is None else agent
        return agent.policy(
            self.batch["observations"], self.batch["high_actor_goals"],
            jax.random.PRNGKey(seed), commands, params,
            temperature=temperature, chain_source=chain_source, return_latents=True,
        )

    def test_profile_changes_command_and_encoder_training_contract(self):
        config = get_config("joint_planner_v1")
        self.assertEqual(config.train_action_chain_source, "predicted")
        self.assertEqual(config.planner_command_mode, "mean")
        self.assertFalse(config.actor_goal_rep_grad)
        self.assertEqual(config.joint_plan_objective, "waypoint_final")
        self.assertTrue(config.planner_bptt)
        self.assertEqual(config.planner_refinement_steps, 1)
        self.assertEqual(config.network_backbone, "mlp")
        self.assertEqual(config.action_core_type, "vanilla_coghp")
        self.assertEqual(config.mixer_hidden, 32)
        self.assertEqual(config.planner_mixer_intermediate_size, 32)

    def test_command_means_are_rng_and_temperature_independent(self):
        train = self.policy(temperature=1., seed=7)
        another_rng = self.policy(temperature=1., seed=81)
        evaluation = self.policy(temperature=0., seed=91)
        means = jnp.stack([dist.mode() for dist in train[0]], axis=1)
        np.testing.assert_array_equal(train[3], means)
        for other in (another_rng, evaluation):
            np.testing.assert_array_equal(train[3], other[3])
            np.testing.assert_array_equal(train[1].mode(), other[1].mode())
        # Deterministic commands must not turn AWR's training likelihood into
        # a zero-variance Gaussian. Its primitive action samples may vary.
        for dist in train[0]:
            np.testing.assert_allclose(dist.scale_diag, 1.)
            self.assertTrue(np.isfinite(np.asarray(dist.log_prob(means[:, 0]))).all())
        np.testing.assert_allclose(train[1].scale_diag, 1.)
        self.assertGreater(magnitude(train[2] - another_rng[2]), 1e-6)
        np.testing.assert_array_equal(evaluation[2], evaluation[1].mode())

    def test_teacher_commands_cannot_change_predicted_policy(self):
        commands = self.agent.command_labels(self.batch)
        supplied = self.policy(commands=commands)
        perturbed = self.policy(commands=commands + 10.)
        absent = self.policy()
        for other in (perturbed, absent):
            np.testing.assert_array_equal(supplied[3], other[3])
            np.testing.assert_array_equal(supplied[1].mode(), other[1].mode())
            np.testing.assert_array_equal(supplied[2], other[2])
        # An explicit teacher-chain override remains available for diagnostic
        # probes, but is not the training default of this profile.
        teacher = self.policy(commands=commands, chain_source="teacher")
        changed_teacher = self.policy(commands=commands + 10., chain_source="teacher")
        self.assertGreater(magnitude(teacher[1].mode() - changed_teacher[1].mode()), 1e-6)

    def test_value_only_encoder_and_end_to_end_planner_gradient_routes(self):
        def losses(params):
            _, high_info, low_info, _ = self.agent.actor_loss(
                self.batch, params, jax.random.PRNGKey(7)
            )
            value_loss = self.agent.value_loss(self.batch, params)[0]
            return jnp.stack([
                high_info["joint_plan_loss"], low_info["actor_loss"], value_loss,
            ])

        gradients = jax.jit(jax.jacrev(losses))(self.agent.network.params)
        goal_grads = gradients["modules_goal_rep"]
        for index, name in ((0, "high"), (1, "low")):
            for grad in jax.tree_util.tree_leaves(goal_grads):
                np.testing.assert_array_equal(grad[index], jnp.zeros_like(grad[index]),
                                              err_msg=f"{name} loss entered goal encoder")
        self.assertGreater(magnitude(jax.tree_util.tree_map(lambda x: x[2], goal_grads)), 1e-8)

        actor_grads = gradients["modules_actor_mixer"]
        flat = flax.traverse_util.flatten_dict(actor_grads)
        for index, name in ((0, "high"), (1, "low")):
            for group in ("H_level", "L_level", "high_actor_head"):
                group_norm = sum(float(jnp.linalg.norm(g[index]))
                                 for path, g in flat.items() if group in path)
                self.assertGreater(group_norm, 1e-8, (name, group))
        for group in ("low_actor_head", "action_mixer_blocks_"):
            group_norm = sum(float(jnp.linalg.norm(g[1])) for path, g in flat.items()
                             if any(part.startswith(group) for part in path))
            self.assertGreater(group_norm, 1e-8, group)

    def test_actor_loss_uses_predicted_commands_and_unit_likelihoods(self):
        commands = self.agent.command_labels(self.batch, self.agent.network.params)
        high, low, _, _ = self.policy(commands=commands, params=self.agent.network.params)
        expected_high, _ = self.agent.multi_high_actor_loss(
            self.batch, high, None, self.agent.network.params
        )
        expected_low, _ = self.agent.low_actor_loss(self.batch, low)
        actual, high_info, low_info, _ = self.agent.actor_loss(
            self.batch, self.agent.network.params, jax.random.PRNGKey(7)
        )
        np.testing.assert_allclose(high_info["joint_plan_loss"], expected_high, rtol=1e-6)
        np.testing.assert_allclose(low_info["actor_loss"], expected_low, rtol=1e-6)
        np.testing.assert_allclose(actual, expected_high + expected_low, rtol=1e-6)

    def test_jitted_update_is_finite_and_learns_value_encoder(self):
        updated, info = self.agent.update(self.batch)
        for leaf in jax.tree_util.tree_leaves((updated.network.params, info)):
            self.assertTrue(np.isfinite(np.asarray(leaf)).all())
        self.assertIn("high_actor/joint_plan_loss", info)
        change = jax.tree_util.tree_map(
            lambda x, y: x - y,
            updated.network.params["modules_goal_rep"],
            self.agent.network.params["modules_goal_rep"],
        )
        self.assertGreater(magnitude(change), 1e-8)

    def test_v0_keeps_stochastic_commands_and_actor_encoder_gradients(self):
        config = small_config("joint_planner")
        self.assertEqual(config.get("planner_command_mode", "sample"), "sample")
        self.assertTrue(config.get("actor_goal_rep_grad", True))
        self.assertEqual(config["train_action_chain_source"], "teacher")
        legacy = NavigationAgent.create(
            23, self.batch["observations"][:1], self.batch["actions"][:1], config
        )
        first = self.policy(agent=legacy, seed=7)
        second = self.policy(agent=legacy, seed=81)
        self.assertGreater(magnitude(first[3] - second[3]), 1e-6)

        def actor_loss(params):
            return legacy.actor_loss(self.batch, params, jax.random.PRNGKey(7))[0]

        gradients = jax.jit(jax.grad(actor_loss))(legacy.network.params)
        self.assertGreater(magnitude(gradients["modules_goal_rep"]), 1e-8)

    def test_checkpoint_signatures_reject_changed_training_semantics(self):
        def metadata(config):
            return build_checkpoint_metadata(
                config, environment="synthetic", seed=23, git_commit="fixture"
            )

        original = metadata(self.config)
        for overrides in (
            dict(actor_goal_rep_grad=True),
            dict(planner_command_mode="sample"),
            dict(train_action_chain_source="teacher"),
            dict(high_alpha=0.5), dict(high_alpha=2.0),
        ):
            if all(self.config.get(key) == value for key, value in overrides.items()):
                continue
            with self.subTest(overrides=overrides), self.assertRaisesRegex(ValueError, "signature mismatch"):
                validate_checkpoint_metadata(original, metadata(small_config(**overrides)),
                                             require_metadata=True)
        # New defaults should not retroactively change archived v0 signatures.
        legacy = small_config("joint_planner")
        historical = dict(legacy)
        historical.pop("actor_goal_rep_grad", None)
        historical.pop("planner_command_mode", None)
        self.assertEqual(metadata(legacy)["signature"], metadata(historical)["signature"])

    def test_missing_new_fields_preserve_legacy_parameters_and_policy(self):
        explicit = small_config("joint_planner")
        historical = dict(explicit)
        historical.pop("actor_goal_rep_grad", None)
        historical.pop("planner_command_mode", None)
        agents = [NavigationAgent.create(
            23, self.batch["observations"][:1], self.batch["actions"][:1], config
        ) for config in (explicit, historical)]
        for left, right in zip(
            jax.tree_util.tree_leaves(agents[0].network.params),
            jax.tree_util.tree_leaves(agents[1].network.params),
        ):
            np.testing.assert_array_equal(left, right)
        first, second = [self.policy(agent=agent) for agent in agents]
        np.testing.assert_array_equal(first[3], second[3])
        np.testing.assert_array_equal(first[1].mode(), second[1].mode())
        np.testing.assert_array_equal(first[2], second[2])


if __name__ == "__main__":
    unittest.main()
