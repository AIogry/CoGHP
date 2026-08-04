"""Synthetic tests for the faithful Flax HRM-Mini port."""

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

from utils.hrm import HRM, HRMCore, TransformerBlock  # noqa: E402


def tree_norm(tree):
    return sum(
        float(jnp.linalg.norm(leaf))
        for leaf in jax.tree_util.tree_leaves(tree)
    )


class OfficialHRMTest(unittest.TestCase):
    def test_mlp_mixer_uses_official_swiglu_parameterization(self):
        seq_len = 4
        hidden_size = 8
        mixer_intermediate = 6
        block = TransformerBlock(
            seq_len=seq_len,
            hidden_size=hidden_size,
            intermediate_size=16,
            head_dim=4,
            is_causal=False,
            is_mlp_mixer=True,
            mlp_mixer_intermediate_size=mixer_intermediate,
        )
        inputs = jnp.arange(
            2 * seq_len * hidden_size,
            dtype=jnp.float32,
        ).reshape(2, seq_len, hidden_size)
        variables = block.init(jax.random.PRNGKey(0), inputs)
        outputs = block.apply(variables, inputs)

        self.assertEqual(outputs.shape, inputs.shape)
        self.assertTrue(bool(jnp.all(jnp.isfinite(outputs))))

        # A bias-free SwiGLU has 3 * input * intermediate parameters.
        expected = 3 * mixer_intermediate * (seq_len + hidden_size)
        actual = sum(
            int(leaf.size)
            for leaf in jax.tree_util.tree_leaves(variables['params'])
        )
        self.assertEqual(actual, expected)
        flat_params = flax.traverse_util.flatten_dict(variables['params'])
        self.assertFalse(any('bias' in path for path in flat_params))

    def test_core_supports_official_mixer_and_attention_backbones(self):
        inputs = jnp.ones((2, 4, 8), dtype=jnp.float32)
        for is_mlp_mixer in (True, False):
            with self.subTest(is_mlp_mixer=is_mlp_mixer):
                core = HRMCore(
                    seq_len=4,
                    num_layers=2,
                    hidden_size=8,
                    intermediate_size=16,
                    head_dim=4,
                    is_causal=False,
                    h_cycles=2,
                    l_cycles=2,
                    bptt=True,
                    is_mlp_mixer=is_mlp_mixer,
                    mlp_mixer_intermediate_size=12,
                )
                variables = core.init(jax.random.PRNGKey(1), None, inputs)

                self.assertEqual(set(variables), {'params', 'constants'})
                self.assertEqual(variables['constants']['zH_init'].shape, (8,))
                self.assertEqual(variables['constants']['zL_init'].shape, (8,))
                self.assertNotIn('zH_init', variables['params'])
                self.assertNotIn('zL_init', variables['params'])

                run = jax.jit(lambda x: core.apply(variables, None, x))
                carry, outputs = run(inputs)
                self.assertEqual(outputs.shape, inputs.shape)
                self.assertEqual(carry['z_H'].shape, inputs.shape)
                self.assertEqual(carry['z_L'].shape, inputs.shape)
                self.assertTrue(bool(jnp.all(jnp.isfinite(outputs))))

    def test_bptt_flag_matches_official_preliminary_iteration_cut(self):
        model_kwargs = dict(
            seq_len=3,
            num_layers=1,
            hidden_size=6,
            intermediate_size=12,
            head_dim=2,
            is_causal=False,
            h_cycles=2,
            l_cycles=2,
            is_mlp_mixer=True,
            mlp_mixer_intermediate_size=7,
        )
        full_core = HRMCore(**model_kwargs, bptt=True)
        truncated_core = HRMCore(**model_kwargs, bptt=False)
        inputs = jnp.linspace(-1.0, 1.0, 18, dtype=jnp.float32).reshape(1, 3, 6)
        variables = full_core.init(jax.random.PRNGKey(2), None, inputs)
        initial_carry = {
            'z_H': jnp.full_like(inputs, 0.25),
            'z_L': jnp.full_like(inputs, -0.15),
        }
        weights = jnp.linspace(0.1, 1.0, inputs.size).reshape(inputs.shape)

        def loss(core, carry):
            _, output = core.apply(variables, carry, inputs)
            return jnp.sum(output * weights)

        full_grad = jax.grad(lambda carry: loss(full_core, carry))(initial_carry)
        truncated_grad = jax.grad(
            lambda carry: loss(truncated_core, carry)
        )(initial_carry)

        self.assertGreater(tree_norm(full_grad), 1e-8)
        self.assertEqual(tree_norm(truncated_grad), 0.0)

    def test_token_wrapper_returns_high_state_logits_and_detached_carry(self):
        model = HRM(
            vocab_size=11,
            seq_len=4,
            num_layers=1,
            hidden_size=8,
            intermediate_size=16,
            head_dim=4,
            is_causal=False,
            h_cycles=2,
            l_cycles=2,
            bptt=False,
            is_mlp_mixer=False,
        )
        input_ids = jnp.asarray([[0, 1, 2, 3], [3, 2, 1, 0]])
        variables = model.init(jax.random.PRNGKey(3), None, input_ids)
        carry, logits = jax.jit(
            lambda ids: model.apply(variables, None, ids)
        )(input_ids)

        self.assertEqual(logits.shape, (2, 4, 11))
        self.assertEqual(carry['z_H'].shape, (2, 4, 8))
        self.assertEqual(carry['z_L'].shape, (2, 4, 8))
        self.assertTrue(np.isfinite(np.asarray(logits)).all())

        def apply_with_params(params):
            return model.apply(
                {'params': params, 'constants': variables['constants']},
                None,
                input_ids,
            )

        logits_grad = jax.grad(
            lambda params: jnp.sum(apply_with_params(params)[1] ** 2)
        )(variables['params'])
        carry_grad = jax.grad(
            lambda params: sum(
                jnp.sum(value ** 2)
                for value in apply_with_params(params)[0].values()
            )
        )(variables['params'])
        self.assertGreater(tree_norm(logits_grad), 1e-8)
        self.assertEqual(tree_norm(carry_grad), 0.0)


if __name__ == '__main__':
    unittest.main()
