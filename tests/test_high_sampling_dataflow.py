"""Independent contracts for high-only sampling, using small real agent graphs.

These tests compare loss values and parameter gradients at identical parameters;
training trajectories may subsequently diverge through the shared encoder.
"""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'impls'))

import flax
import jax
import jax.numpy as jnp
import numpy as np
from agents.navigation import NavigationAgent, get_config


def small_config(probability):
    config = get_config('joint_planner').to_dict()
    config.update(actor_hidden_dims=(16,), value_hidden_dims=(16,), enc_hidden_dims=(16,),
                  feature_dim=8, subgoal_steps=2, planner_h_cycles=2, planner_l_cycles=1,
                  num_subgoals=3, high_sampling_probability=probability)
    return config


def batch_fixture():
    rng = np.random.default_rng(3901)
    normal = lambda shape: jnp.asarray(rng.normal(size=shape), dtype=jnp.float32)
    return dict(observations=normal((3, 5)), next_observations=normal((3, 5)),
                actions=normal((3, 2)), value_goals=normal((3, 5)),
                high_actor_goals=normal((3, 5)), high_actor_targets=normal((3, 3, 5)),
                rewards=jnp.array([-1., 0., -1.]), masks=jnp.array([1., 0., 1.]))


def with_high(batch, changed=False):
    result = dict(batch)
    if changed:
        result.update(high_only_observations=batch['observations'][::-1] + .4,
                      high_only_goals=batch['high_actor_goals'][::-1] - .3,
                      high_only_targets=batch['high_actor_targets'][::-1] + .7)
    else:
        result.update(high_only_observations=batch['observations'],
                      high_only_goals=batch['high_actor_goals'],
                      high_only_targets=batch['high_actor_targets'])
    return result


def assert_tree(test, actual, expected, exact=True):
    test.assertEqual(jax.tree_util.tree_structure(actual), jax.tree_util.tree_structure(expected))
    for actual_leaf, expected_leaf in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected)):
        if exact:
            np.testing.assert_array_equal(actual_leaf, expected_leaf)
        else:
            np.testing.assert_allclose(actual_leaf, expected_leaf, rtol=3e-6, atol=3e-6)


class HighOnlySamplingDataflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.batch = batch_fixture()
        cls.agent = NavigationAgent.create(29, cls.batch['observations'][:1],
                                          cls.batch['actions'][:1], small_config(.5))
        cls.baseline = NavigationAgent.create(29, cls.batch['observations'][:1],
                                             cls.batch['actions'][:1], small_config(0.))
        cls.key = jax.random.PRNGKey(900)

    def components(self, batch, params=None, agent=None):
        agent = self.agent if agent is None else agent
        return agent.loss_components(batch, agent.network.params if params is None else params, self.key)

    def gradient(self, batch, loss):
        return jax.grad(lambda p: self.components(batch, p)[0][loss])(self.agent.network.params)

    def test_sampling_controls_do_not_change_parameters_or_no_fields_behavior(self):
        assert_tree(self, self.agent.network.params, self.baseline.network.params)
        assert_tree(self, self.agent.network.model_state, self.baseline.network.model_state)
        baseline = self.components(self.batch, agent=self.baseline)
        actual = self.components(self.batch)
        assert_tree(self, actual, baseline)
        old_updated, old_info = self.baseline.update(self.batch)
        new_updated, new_info = self.agent.update(self.batch)
        assert_tree(self, new_updated.network.params, old_updated.network.params)
        assert_tree(self, new_updated.network.opt_state, old_updated.network.opt_state)
        assert_tree(self, new_updated.rng, old_updated.rng)
        assert_tree(self, new_info, old_info)

    def test_identical_high_payload_preserves_all_loss_values_and_total_gradient(self):
        original, original_info = self.components(self.batch)
        actual, actual_info = self.components(with_high(self.batch))
        assert_tree(self, actual, original, exact=False)
        # Runtime diagnostics must still refer to the original executor pass.
        for key in original_info:
            if key.startswith(('action_core/', 'action_hrm/')):
                np.testing.assert_array_equal(actual_info[key], original_info[key])
        original_grad = jax.grad(lambda p: self.agent.total_loss(self.batch, p, self.key)[0])(self.agent.network.params)
        new_grad = jax.grad(lambda p: self.agent.total_loss(with_high(self.batch), p, self.key)[0])(self.agent.network.params)
        assert_tree(self, new_grad, original_grad, exact=False)

    def test_changed_high_payload_leaves_low_value_losses_and_gradients_exact(self):
        original, original_info = self.components(self.batch)
        changed_batch = with_high(self.batch, changed=True)
        changed, changed_info = self.components(changed_batch)
        for loss in ('low', 'value'):
            np.testing.assert_array_equal(changed[loss], original[loss])
            assert_tree(self, self.gradient(changed_batch, loss), self.gradient(self.batch, loss))
        self.assertGreater(abs(float(changed['high'] - original['high'])), 1e-5)
        for key in original_info:
            if key.startswith(('action_core/', 'action_hrm/')):
                np.testing.assert_array_equal(changed_info[key], original_info[key])
        original_high = self.gradient(self.batch, 'high')
        changed_high = self.gradient(changed_batch, 'high')
        difference = sum(float(jnp.linalg.norm(a-b)) for a,b in zip(
            jax.tree_util.tree_leaves(original_high), jax.tree_util.tree_leaves(changed_high)))
        self.assertGreater(difference, 1e-5)

    def test_high_loss_is_exactly_the_unchanged_v0_formula_on_donor_data(self):
        altered = with_high(self.batch, changed=True)
        donor = dict(self.batch, observations=altered['high_only_observations'],
                     high_actor_goals=altered['high_only_goals'],
                     high_actor_targets=altered['high_only_targets'])
        expected = self.components(donor)[0]['high']
        actual = self.components(altered)[0]['high']
        np.testing.assert_array_equal(actual, expected)
        assert_tree(self, self.gradient(altered, 'high'), self.gradient(donor, 'high'))


if __name__ == '__main__':
    unittest.main()
