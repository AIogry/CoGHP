"""Mechanism tests for the four navigation conditions (CPU, small networks)."""

import copy
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "impls"))
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax

from agents.navigation import NavigationAgent as HRMActionCoGHPAgent
from agents.navigation import (
    NavigationAgent as NavigationCoGHPAgent,
    get_legacy_navigation_config as get_config,
    global_ivl_loss,
    stable_awr,
)
from utils.datasets import Dataset, MultiHGCDataset
from utils.flax_utils import build_checkpoint_metadata, restore_agent, save_agent
from utils.encoders import PositionCommand
from utils.datasets import GlobalFragmentSampler, NavigationDataset


class NativeNavigationTests(unittest.TestCase):
    def test_native_import_does_not_load_baseline_modules(self):
        import subprocess

        code = "import agents.navigation; import sys; assert 'agents.coghp' not in sys.modules; assert 'utils.coghp_network' not in sys.modules"
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_native_method_has_no_baseline_inheritance_or_gelu_execution(self):
        import agents.navigation as navigation
        from agents.coghp import CoGHPAgent
        from networks.backbones import SwiGLUBackbone

        self.assertFalse(issubclass(navigation.NavigationAgent, CoGHPAgent))
        config = navigation.get_config().to_dict()
        small = config_for(True, True)
        for key in (
            "actor_hidden_dims",
            "value_hidden_dims",
            "enc_hidden_dims",
            "feature_dim",
            "position_hidden_dims",
            "global_hidden_dims",
            "global_dim",
            "num_subgoals",
            "subgoal_steps",
            "action_hrm_bptt",
            "hrm_mixer_intermediate_size",
            "global_abstraction_steps",
            "global_goal_tolerance",
            "position_scale",
            "command_content",
            "high_weight_source",
        ):
            config[key] = small[key]
        config.update(
            agent_name="navigation",
            network_backbone="swiglu_rmsnorm",
            planner_core_type="matched_feedforward",
            planner_mixer_intermediate_size=16,
        )
        b = batch()
        with (
            mock.patch(
                "agents.navigation.MLP", side_effect=AssertionError("legacy MLP")
            ),
            mock.patch(
                "networks.policies.MLP", side_effect=AssertionError("legacy MLP")
            ),
            mock.patch("utils.networks.MLP", side_effect=AssertionError("legacy MLP")),
            mock.patch("utils.encoders.MLP", side_effect=AssertionError("legacy MLP")),
            mock.patch("flax.linen.gelu", side_effect=AssertionError("GELU")),
        ):
            agent = navigation.NavigationAgent.create(
                0, b["observations"][:1], b["actions"][:1], config
            )
            updated, info = agent.update(b)
            actions = updated.sample_actions(
                b["observations"][0],
                b["high_actor_goals"][0],
                jax.random.PRNGKey(1),
                0.0,
            )
        self.assertIsInstance(
            agent.network.model_def.modules["value"].backbone, SwiGLUBackbone
        )
        self.assertEqual(actions.shape, (2,))
        for leaf in jax.tree_util.tree_leaves((updated.network.params, info, actions)):
            self.assertTrue(np.isfinite(np.asarray(leaf)).all())
        metadata = build_checkpoint_metadata(
            config, environment="synthetic", seed=0, git_commit="fixture"
        )
        with tempfile.TemporaryDirectory() as directory:
            save_agent(updated, directory, 1, checkpoint_metadata=metadata)
            restored = restore_agent(
                agent,
                directory,
                1,
                expected_checkpoint_metadata=metadata,
                require_checkpoint_metadata=True,
            )
        assert_tree_close(self, restored.network.params, updated.network.params, atol=0)
        assert_tree_close(
            self, restored.network.opt_state, updated.network.opt_state, atol=0
        )
        np.testing.assert_array_equal(restored.rng, updated.rng)

    def test_native_checkpoint_signature_identifies_representation_backbone(self):
        from agents.navigation import get_config as native_config

        a = native_config().to_dict()
        b = dict(a, network_backbone="mlp")
        ma = build_checkpoint_metadata(
            a, environment="synthetic", seed=0, git_commit="fixture"
        )
        mb = build_checkpoint_metadata(
            b, environment="synthetic", seed=0, git_commit="fixture"
        )
        self.assertNotEqual(ma["signature"], mb["signature"])


def config_for(position=False, glob=False):
    c = get_config().to_dict()
    c.update(
        actor_hidden_dims=(16,),
        value_hidden_dims=(16,),
        enc_hidden_dims=(16,),
        feature_dim=8,
        position_hidden_dims=(8,),
        global_hidden_dims=(8,),
        global_dim=8,
        num_subgoals=2,
        subgoal_steps=2,
        action_hrm_bptt=False,
        action_hrm_mlp_mixer_intermediate_size=16,
        hrm_mixer_intermediate_size=16,
        action_block_type="swiglu_rmsnorm",
        hrm_block_type="swiglu_rmsnorm",
        global_abstraction_steps=3,
        global_goal_tolerance=0.1,
        position_scale=(2.0, 3.0),
        command_content="position" if position else "full_state",
        high_weight_source="global_ota" if glob else "legacy_value",
    )
    return c


def batch():
    rng = np.random.default_rng(19)
    b = dict(
        observations=rng.normal(size=(3, 5)).astype(np.float32),
        next_observations=rng.normal(size=(3, 5)).astype(np.float32),
        actions=rng.uniform(-1, 1, (3, 2)).astype(np.float32),
        value_goals=rng.normal(size=(3, 5)).astype(np.float32),
        high_actor_goals=rng.normal(size=(3, 5)).astype(np.float32),
        high_actor_targets=rng.normal(size=(3, 2, 5)).astype(np.float32),
        rewards=np.array([-1, 0, -1], np.float32),
        masks=np.array([1, 0, 1], np.float32),
    )
    b.update(
        global_goals=b["value_goals"][:, :2],
        global_observations=b["next_observations"],
        global_rewards=b["rewards"],
        global_masks=b["masks"],
        global_valid=np.array([1, 1, 0], np.float32),
        global_offsets=np.array([3, 0, 1], np.int32),
    )
    return b


def assert_tree_close(test, a, b, atol=1e-6):
    la, ta = jax.tree_util.tree_flatten(a)
    lb, tb = jax.tree_util.tree_flatten(b)
    test.assertEqual(ta, tb)
    for x, y in zip(la, lb):
        np.testing.assert_allclose(x, y, atol=atol, rtol=1e-6)


def norm(tree):
    return float(jnp.sqrt(sum(jnp.sum(x**2) for x in jax.tree_util.tree_leaves(tree))))


class NavigationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b = batch()
        cls.agents = {}
        for p, g in [(False, False), (False, True), (True, False), (True, True)]:
            cls.agents[p, g] = NavigationCoGHPAgent.create(
                0, cls.b["observations"][:1], cls.b["actions"][:1], config_for(p, g)
            )

    def test_baseline_exact_losses_actions_and_update(self):
        a = self.agents[False, False]
        old = HRMActionCoGHPAgent.create(
            0, self.b["observations"][:1], self.b["actions"][:1], config_for()
        )
        assert_tree_close(self, a.network.params, old.network.params, atol=0)
        key = jax.random.PRNGKey(12)
        assert_tree_close(
            self,
            a.value_loss(self.b, a.network.params),
            old.value_loss(self.b, old.network.params),
            atol=0,
        )
        assert_tree_close(
            self,
            a.actor_loss(self.b, a.network.params, key),
            old.actor_loss(self.b, old.network.params, key),
            atol=0,
        )
        np.testing.assert_array_equal(
            a.sample_actions(
                self.b["observations"][0], self.b["high_actor_goals"][0], key, 0
            ),
            old.sample_actions(
                self.b["observations"][0], self.b["high_actor_goals"][0], key, 0
            ),
        )
        aa, ai = a.update(self.b)
        oo, oi = old.update(self.b)
        assert_tree_close(self, aa.network.params, oo.network.params)
        assert_tree_close(self, ai, oi)

    def test_shared_initialization_and_independent_modules(self):
        base = self.agents[False, False]
        for agent in self.agents.values():
            for key in base.network.params:
                assert_tree_close(
                    self, agent.network.params[key], base.network.params[key], atol=0
                )
        assert_tree_close(
            self,
            self.agents[False, True].network.params["modules_global_value"],
            self.agents[True, True].network.params["modules_global_value"],
            atol=0,
        )
        assert_tree_close(
            self,
            self.agents[True, False].network.params["modules_position_encoder"],
            self.agents[True, True].network.params["modules_position_encoder"],
            atol=0,
        )
        self.assertNotIn(
            "modules_global_value", self.agents[True, False].network.params
        )

    def test_position_future_body_invariance_and_causality(self):
        a = self.agents[True, True]
        changed = copy.deepcopy(self.b)
        changed["high_actor_goals"][:, 2:] += 200
        changed["high_actor_targets"][:, :, 2:] -= 200
        key = jax.random.PRNGKey(4)

        def outputs(b):
            cmds = a.command_labels(b)
            hs, low, _ = a.policy(
                b["observations"], b["high_actor_goals"], key, cmds, temperature=0
            )
            return [x.mode() for x in hs], low.mode()

        assert_tree_close(self, outputs(self.b), outputs(changed), atol=0)
        changed = copy.deepcopy(self.b)
        changed["high_actor_targets"][:, 1, :2] += 4
        first, second = outputs(self.b), outputs(changed)
        # Neither far nor near high prediction can read the current/future teacher label.
        assert_tree_close(self, first[0], second[0], atol=0)
        changed["high_actor_targets"][:, 0, :2] += 4
        np.testing.assert_array_equal(outputs(changed)[0][0], first[0][0])
        self.assertGreater(float(np.max(np.abs(outputs(changed)[1] - first[1]))), 1e-7)

    def test_each_loss_gradient_and_isolated_parameter_updates(self):
        a = self.agents[True, True]
        requirements = {
            "high": {"modules_actor_mixer"},
            "low": {"modules_actor_mixer", "modules_position_encoder"},
            "position": {"modules_position_encoder", "modules_position_decoder"},
            "global_value": {"modules_global_value"},
            "value": {"modules_value", "modules_goal_rep"},
        }
        for loss, allowed in requirements.items():
            with self.subTest(loss=loss):
                grad_fn = jax.jit(
                    jax.grad(lambda p: a.loss_components(self.b, p, a.rng)[0][loss])
                )
                grads = grad_fn(a.network.params)
                updates, _ = a.network.tx.update(
                    grads, a.network.opt_state, a.network.params
                )
                changed = optax.apply_updates(a.network.params, updates)
                for key in grads:
                    if key in allowed:
                        self.assertGreater(norm(grads[key]), 1e-9, (loss, key))
                        self.assertGreater(
                            norm(
                                jax.tree_util.tree_map(
                                    lambda x, y: x - y,
                                    changed[key],
                                    a.network.params[key],
                                )
                            ),
                            0,
                        )
                    else:
                        self.assertEqual(norm(grads[key]), 0.0, (loss, key))
                        assert_tree_close(
                            self, changed[key], a.network.params[key], atol=0
                        )
                core = grads["modules_actor_mixer"]
                if loss == "high":
                    self.assertEqual(norm(core["action_hrm"]), 0)
                    self.assertEqual(norm(core["low_actor_head"]), 0)
                if loss == "low":
                    self.assertEqual(norm(core["high_actor_head"]), 0)
                    for name in core:
                        if name.startswith("planner_mixer"):
                            self.assertEqual(norm(core[name]), 0)
                del grad_fn

    def test_global_loss_manual_reference_and_stable_weights(self):
        vals = jnp.array([[2.0, 3.0, 100.0], [4.0, 1.0, 100.0]])
        cur = jnp.array([[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]])
        nxt = jnp.array([[2.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        r, m, valid = (
            jnp.array([-1.0, 0.0, -1.0]),
            jnp.array([1.0, 0.0, 1.0]),
            jnp.array([1.0, 1.0, 0.0]),
        )
        adv = np.asarray(r + 0.99 * m * jnp.min(nxt, axis=0) - cur.mean(axis=0))
        target = np.asarray(r[None] + 0.99 * m[None] * nxt)
        expected = (
            np.where(adv >= 0, 0.7, 0.3)[None]
            * (target - np.asarray(vals)) ** 2
            * np.asarray(valid)[None]
        ).sum() / 2
        self.assertAlmostEqual(
            float(global_ivl_loss(vals, cur, nxt, r, m, valid, 0.99, 0.7)),
            expected,
            places=5,
        )
        w = stable_awr(jnp.array([-1e10, 0.0, 1e10]), 3.0)
        self.assertTrue(np.isfinite(w).all())
        np.testing.assert_allclose(w, [0, 1, 100], rtol=1e-6)

    def test_high_awr_exact_chain_coefficients(self):
        a = self.agents[True, True]
        commands = a.command_labels(self.b)
        hs, _, _ = a.policy(
            self.b["observations"], self.b["high_actor_goals"], a.rng, commands
        )
        loss, _ = a.multi_high_actor_loss(self.b, hs, None, a.network.params)
        weights = stable_awr(a.high_advantages(self.b), a.config["high_alpha"])
        expected = (
            sum(
                -jnp.mean(weights[:, j] * hs[j].log_prob(commands[:, j]))
                / 2
                * 0.8 ** (1 - j)
                for j in range(2)
            )
            / 2
        )
        np.testing.assert_allclose(loss, expected, rtol=1e-6)

    def test_checkpoint_optimizer_constants_and_resume(self):
        a = self.agents[True, True]
        a, _ = a.update(self.b)
        metadata = build_checkpoint_metadata(
            a.config, environment="test", seed=0, git_commit="test"
        )
        with tempfile.TemporaryDirectory() as directory:
            save_agent(a, directory, 1, metadata)
            restored = restore_agent(
                self.agents[True, True], directory, 1, metadata, True
            )
            assert_tree_close(
                self,
                flax.serialization.to_state_dict(a),
                flax.serialization.to_state_dict(restored),
                atol=0,
            )
            x, _ = a.update(self.b)
            y, _ = restored.update(self.b)
            assert_tree_close(self, x.network.params, y.network.params, atol=0)
            wrong = copy.deepcopy(metadata)
            wrong["signature"]["position_scale"] = [10.0, 10.0]
            with self.assertRaises(ValueError):
                restore_agent(self.agents[True, True], directory, 1, wrong, True)

    def test_complete_global_ema_and_diagnostics(self):
        a = self.agents[True, True]
        b, _ = a.update(self.b)
        c, _ = b.update(self.b)
        expected = jax.tree_util.tree_map(
            lambda p, t: 0.005 * p + 0.995 * t,
            b.network.params["modules_global_value"],
            b.network.params["modules_target_global_value"],
        )
        assert_tree_close(
            self, c.network.params["modules_target_global_value"], expected
        )
        diagnostics = c.validation_rollout_info(self.b)
        self.assertTrue(all(np.isfinite(x).all() for x in diagnostics.values()))
        self.assertIn("validation_position/teacher/cycle_mse", diagnostics)

    def test_zero_input_normalization(self):
        module = PositionCommand(8, (1.0, 1.0), (8,))
        s, q = jnp.zeros((1, 5)), jnp.zeros((1, 2))
        variables = module.init(jax.random.PRNGKey(1), s, q)
        self.assertTrue(np.isfinite(module.apply(variables, s, q)).all())
        grad = jax.grad(
            lambda p: jnp.sum(
                module.apply({"params": p, "constants": variables["constants"]}, s, q)
            )
        )(variables["params"])
        self.assertTrue(
            all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(grad))
        )


class FragmentTests(unittest.TestCase):
    def setUp(self):
        # First trajectory passes x=1 and then leaves; second is distant.
        xy = np.array(
            [
                [0, 0],
                [1, 0],
                [2, 0],
                [3, 0],
                [4, 0],
                [5, 0],
                [100, 0],
                [101, 0],
                [102, 0],
                [103, 0],
            ],
            np.float32,
        )
        self.data = Dataset.create(
            observations=np.pad(xy, ((0, 0), (0, 3))),
            actions=np.zeros((10, 2), np.float32),
            terminals=np.array([0, 0, 0, 0, 1, 1, 0, 0, 1, 1], np.float32),
            valids=np.array([1, 1, 1, 1, 1, 0, 1, 1, 1, 0], np.float32),
        )
        self.config = config_for(True, True)
        self.sampler = GlobalFragmentSampler(self.data, self.config)

    def test_first_hit_incomplete_boundaries_and_no_crossing(self):
        idxs = np.array([0, 0, 0, 4, 4, 4, 6])
        goals = np.array(
            [[0, 0], [1, 0], [30, 0], [5, 0], [7, 0], [100, 0], [103, 0]], np.float32
        )
        b = dict(
            indices=idxs,
            value_goals=np.pad(goals, ((0, 0), (0, 3))),
            rewards=np.ones(7),
            masks=np.ones(7),
        )
        out = self.sampler.add_fields(b)
        np.testing.assert_array_equal(out["global_rewards"], [0, 0, -1, 0, -1, -1, 0])
        np.testing.assert_array_equal(out["global_masks"], [0, 0, 1, 0, 1, 1, 0])
        np.testing.assert_array_equal(out["global_valid"], [1, 1, 1, 1, 0, 0, 1])
        np.testing.assert_array_equal(out["global_offsets"], [0, 1, 3, 1, 1, 1, 3])
        np.testing.assert_array_equal(
            out["global_observations"][:, 0], [0, 1, 3, 5, 5, 5, 103]
        )
        self.assertIs(out["rewards"], b["rewards"])
        self.assertIs(out["masks"], b["masks"])

    def test_no_legacy_sampler_or_rng_change(self):
        plain = MultiHGCDataset(self.data, self.config, rng=np.random.default_rng(17))
        wrapped = NavigationDataset(
            MultiHGCDataset(self.data, self.config, rng=np.random.default_rng(17)),
            self.config,
        )
        for _ in range(4):
            a, b = plain.sample(8), wrapped.sample(8)
            for key in a:
                np.testing.assert_array_equal(a[key], b[key])
            self.assertEqual(
                plain.rng.bit_generator.state, wrapped.rng.bit_generator.state
            )


if __name__ == "__main__":
    unittest.main()
