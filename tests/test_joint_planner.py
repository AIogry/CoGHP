"""Small CPU checks for whole-plan HRM refinement and deep supervision."""

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
from utils.flax_utils import (
    build_checkpoint_metadata,
    restore_agent,
    save_agent,
    validate_checkpoint_metadata,
)


def small_config(**overrides):
    # Preserve the October 27 prototype as a compatibility profile: deep
    # supervision, detached refinement carry, and one endpoint weight.
    config = get_config("joint_planner_pilot").to_dict()
    config.update(
        actor_hidden_dims=(16,), value_hidden_dims=(16,), enc_hidden_dims=(16,),
        feature_dim=8, planner_mixer_intermediate_size=16,
        hrm_mixer_intermediate_size=16, subgoal_steps=2,
        global_hidden_dims=(16,), global_dim=8, global_goal_tolerance=0.1,
    )
    config.update(overrides)
    return config


def batch():
    rng = np.random.default_rng(5)
    def normal(shape):
        return jnp.asarray(rng.normal(size=shape), dtype=jnp.float32)
    return dict(
        observations=normal((3, 5)), next_observations=normal((3, 5)),
        actions=normal((3, 2)), value_goals=normal((3, 5)),
        high_actor_goals=normal((3, 5)), high_actor_targets=normal((3, 3, 5)),
        rewards=jnp.array([-1., 0., -1.]), masks=jnp.array([1., 0., 1.]),
        global_goals=normal((3, 2)), global_observations=normal((3, 5)),
        global_rewards=jnp.array([-1., 0., -1.]),
        global_masks=jnp.array([1., 0., 1.]), global_valid=jnp.ones(3),
        global_offsets=jnp.array([2, 1, 2]),
    )


class JointPlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.batch = batch()
        cls.config = small_config()
        cls.agent = NavigationAgent.create(
            0, cls.batch["observations"][:1], cls.batch["actions"][:1], cls.config
        )

    def plans(self, commands=None, params=None, temperature=1.0):
        return self.agent.policy(
            self.batch["observations"], self.batch["high_actor_goals"],
            jax.random.PRNGKey(9), commands, params, temperature=temperature,
            return_latents=True, return_planner_rounds=True,
        )

    def test_every_round_predicts_whole_plan_without_teacher_prefix(self):
        commands = self.agent.command_labels(self.batch)
        a = self.plans(commands)
        b = self.plans(commands + 10)
        c = self.plans()
        self.assertEqual(len(a[0]), 3)
        for ra, rb, rc in zip(a[0], b[0], c[0]):
            self.assertEqual(len(ra), 3)
            for da, db, dc in zip(ra, rb, rc):
                self.assertEqual(da.mode().shape, (3, 8))
                np.testing.assert_array_equal(da.mode(), db.mode())
                np.testing.assert_array_equal(da.mode(), dc.mode())
        np.testing.assert_array_equal(a[3], b[3])
        np.testing.assert_array_equal(a[3], c[3])
        self.assertGreater(float(jnp.linalg.norm(a[1].mode() - b[1].mode())), 1e-6)

    def test_carry_changes_plan_and_each_round_trains_both_hrm_levels(self):
        rounds = self.plans()[0]
        first = jnp.stack([d.mode() for d in rounds[0]])
        last = jnp.stack([d.mode() for d in rounds[-1]])
        self.assertGreater(float(jnp.linalg.norm(first - last)), 1e-6)
        targets = self.agent.command_labels(self.batch)

        def round_losses(params):
            refinements = self.plans(params=params)[0]
            return jnp.stack([
                -jnp.mean(jnp.stack([
                    d.log_prob(targets[:, j]) for j, d in enumerate(ds)
                ])) for ds in refinements
            ])

        gradients = jax.jit(jax.jacrev(round_losses))(self.agent.network.params)
        flat = flax.traverse_util.flatten_dict(gradients)
        for r in range(3):
            for level in ("H_level", "L_level"):
                magnitude = sum(float(jnp.linalg.norm(v[r])) for p, v in flat.items()
                                if "planner_core" in p and level in p)
                self.assertGreater(magnitude, 1e-8, (r, level))

    def test_deep_supervision_averages_rounds_and_shares_endpoint_weight(self):
        rounds = self.plans()[0]
        loss, info = self.agent.joint_high_actor_loss(self.batch, rounds)
        individual = [self.agent.joint_high_actor_loss(self.batch, [r])[0] for r in rounds]
        np.testing.assert_allclose(loss, jnp.mean(jnp.stack(individual)), rtol=1e-6)
        repeated, _ = self.agent.joint_high_actor_loss(self.batch, [rounds[0]] * 3)
        np.testing.assert_allclose(repeated, individual[0], rtol=1e-6)
        altered = dict(self.batch)
        altered["high_actor_targets"] = self.batch["high_actor_targets"].at[:, 1:].add(10)
        new_loss, new_info = self.agent.joint_high_actor_loss(altered, rounds)
        np.testing.assert_array_equal(info["weight_mean"], new_info["weight_mean"])
        self.assertGreater(abs(float(new_loss - loss)), 1e-6)

    def test_update_checkpoint_restore_and_continuation(self):
        updated, info = self.agent.update(self.batch)
        for leaf in jax.tree_util.tree_leaves((updated.network.params, info)):
            self.assertTrue(np.isfinite(np.asarray(leaf)).all())
        metadata = build_checkpoint_metadata(
            self.config, environment="synthetic", seed=0, git_commit="fixture"
        )
        with tempfile.TemporaryDirectory() as directory:
            save_agent(updated, directory, 1, checkpoint_metadata=metadata)
            restored = restore_agent(
                self.agent, directory, 1, expected_checkpoint_metadata=metadata,
                require_checkpoint_metadata=True,
            )
        continued, ci = updated.update(self.batch)
        resumed, ri = restored.update(self.batch)
        for a, b in zip(jax.tree_util.tree_leaves((continued, ci)),
                        jax.tree_util.tree_leaves((resumed, ri))):
            np.testing.assert_array_equal(a, b)
        actions = restored.sample_actions(
            self.batch["observations"][0], self.batch["high_actor_goals"][0],
            jax.random.PRNGKey(4), 0.0,
        )
        self.assertEqual(actions.shape, (2,))
        self.assertTrue(np.isfinite(np.asarray(actions)).all())

    def test_global_value_can_supply_the_same_joint_objective(self):
        agent = NavigationAgent.create(
            0, self.batch["observations"][:1], self.batch["actions"][:1],
            small_config(high_weight_source="global_ota"),
        )
        updated, info = agent.update(self.batch)
        self.assertIn("global/loss", info)
        self.assertIn("high_actor/round_3/loss", info)
        for leaf in jax.tree_util.tree_leaves((updated.network.params, info)):
            self.assertTrue(np.isfinite(np.asarray(leaf)).all())

    def test_checkpoint_cannot_silently_change_generation_or_round_budget(self):
        def metadata(config):
            return build_checkpoint_metadata(
                config, environment="synthetic", seed=0, git_commit="fixture"
            )
        joint = metadata(self.config)
        historical_joint = {key: value for key, value in self.config.items()
                            if key != "joint_plan_objective"}
        self.assertEqual(joint, metadata(historical_joint))
        self.assertEqual(joint, metadata({**historical_joint, "joint_plan_objective": "endpoint_deep"}))
        for config in (small_config(planner_refinement_steps=1),
                       small_config(planner_generation="autoregressive", planner_refinement_steps=1)):
            with self.assertRaisesRegex(ValueError, "signature mismatch"):
                validate_checkpoint_metadata(joint, metadata(config), require_metadata=True)
        native = get_config().to_dict()
        historical = {k: v for k, v in native.items()
                      if k not in ("planner_generation", "planner_refinement_steps")}
        self.assertEqual(metadata(native), metadata(historical))

    def test_one_round_control_preserves_parameters_and_first_round_prediction(self):
        single = NavigationAgent.create(
            0, self.batch["observations"][:1], self.batch["actions"][:1],
            small_config(planner_refinement_steps=1),
        )
        for a, b in zip(jax.tree_util.tree_leaves(self.agent.network.params),
                        jax.tree_util.tree_leaves(single.network.params)):
            np.testing.assert_array_equal(a, b)
        one = single.policy(
            self.batch["observations"], self.batch["high_actor_goals"],
            jax.random.PRNGKey(9), return_planner_rounds=True,
        )[0]
        three = self.plans()[0]
        self.assertEqual(len(one), 1)
        for da, db in zip(one[0], three[0]):
            np.testing.assert_array_equal(da.mode(), db.mode())


if __name__ == "__main__":
    unittest.main()
