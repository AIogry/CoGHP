"""Synthetic tests for the Flax HRM core and configurable Mixer blocks."""

import pathlib
import sys
import unittest

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

IMPLS_DIR = pathlib.Path(__file__).resolve().parents[1] / "impls"
if str(IMPLS_DIR) not in sys.path:
    sys.path.insert(0, str(IMPLS_DIR))

from utils.coghp_network import MixerBlock
from networks.backbones import (
    BACKBONE_TYPES as ACTION_CORE_TYPES,
    GELUMixerBlock as CoGHPMixerBlock,
    GELULayerNormMixerBlock,
    SequenceBackbone as HRMCore,
    BlockStack as Transformer,
    SequenceBlock as TransformerBlock,
)


def tree_norm(tree):
    return sum(float(jnp.linalg.norm(leaf)) for leaf in jax.tree_util.tree_leaves(tree))


def parameter_count(tree):
    return sum(int(leaf.size) for leaf in jax.tree_util.tree_leaves(tree))


def flattened_shapes(tree):
    return {
        path: value.shape
        for path, value in flax.traverse_util.flatten_dict(tree).items()
    }


def make_configurable_core(
    block_type,
    *,
    h_cycles=2,
    l_cycles=2,
    token_hidden=7,
    channel_hidden=9,
    mixer_intermediate_size=11,
    layernorm_affine=True,
    core_type="hrm",
    bptt=True,
):
    return HRMCore(
        seq_len=5,
        num_layers=1,
        hidden_size=128,
        intermediate_size=32,
        head_dim=32,
        is_causal=True,
        h_cycles=h_cycles,
        l_cycles=l_cycles,
        core_type=core_type,
        bptt=bptt,
        is_mlp_mixer=True,
        mlp_mixer_intermediate_size=mixer_intermediate_size,
        block_type=block_type,
        token_hidden=token_hidden,
        channel_hidden=channel_hidden,
        layernorm_affine=layernorm_affine,
    )


class OfficialHRMTest(unittest.TestCase):
    def test_matched_feedforward_mi256_exact_204k_contract(self):
        inputs = jnp.linspace(
            -1.0,
            1.0,
            2 * 5 * 128,
            dtype=jnp.float32,
        ).reshape(2, 5, 128)
        core = make_configurable_core(
            "swiglu_rmsnorm",
            core_type="matched_feedforward",
            mixer_intermediate_size=256,
        )
        variables = core.init(jax.random.PRNGKey(204), None, inputs)
        self.assertEqual(set(variables), {"params"})
        self.assertEqual(core.apply(variables, None, inputs)[0], {})

        low_count = parameter_count(variables["params"]["L_level"])
        high_count = parameter_count(variables["params"]["H_level"])
        self.assertEqual(low_count, 102144)
        self.assertEqual(high_count, 102144)
        self.assertEqual(parameter_count(variables["params"]), 204288)

        eager = core.apply(variables, None, inputs)[1]
        compiled = jax.jit(lambda value: core.apply(variables, None, value)[1])(inputs)
        self.assertEqual(eager.shape, inputs.shape)
        np.testing.assert_allclose(compiled, eager, rtol=1e-5, atol=1e-5)

        grads = jax.grad(
            lambda params: jnp.mean(
                core.apply({"params": params}, None, inputs)[1] ** 2
            )
        )(variables["params"])
        self.assertGreater(tree_norm(grads["L_level"]), 0.0)
        self.assertGreater(tree_norm(grads["H_level"]), 0.0)
        self.assertTrue(
            all(
                bool(jnp.all(jnp.isfinite(leaf)))
                for leaf in jax.tree_util.tree_leaves(grads)
            )
        )

    def test_configurable_action_core_parameter_targets(self):
        inputs = jnp.ones((1, 5, 128), dtype=jnp.float32)
        baseline = MixerBlock(
            num_tokens=5,
            embed_dim=128,
            hidden_dim_tokens=32,
            hidden_dim_channels=32,
            causal=True,
        )
        baseline_params = baseline.init(
            jax.random.PRNGKey(9),
            inputs,
        )["params"]
        baseline_count = parameter_count(baseline_params)
        self.assertEqual(baseline_count, 8734)
        cases = (
            (
                "coghp_gelu",
                dict(token_hidden=32, channel_hidden=15),
                8730,
            ),
            (
                "gelu_layernorm",
                dict(
                    token_hidden=32,
                    channel_hidden=14,
                    layernorm_affine=True,
                ),
                8698,
            ),
            (
                "swiglu_rmsnorm",
                dict(mixer_intermediate_size=11),
                8778,
            ),
        )
        for block_type, overrides, expected in cases:
            reference_params = None
            for core_type in sorted(ACTION_CORE_TYPES):
                with self.subTest(
                    core_type=core_type,
                    block_type=block_type,
                ):
                    core = make_configurable_core(
                        block_type,
                        core_type=core_type,
                        **overrides,
                    )
                    variables = core.init(
                        jax.random.PRNGKey(10),
                        None,
                        inputs,
                    )
                    actual = parameter_count(variables["params"])
                    self.assertEqual(actual, expected)
                    self.assertEqual(
                        set(variables),
                        {"params", "constants"} if core_type == "hrm" else {"params"},
                    )
                    if block_type == "gelu_layernorm":
                        self.assertLessEqual(
                            abs(actual - baseline_count) / baseline_count,
                            0.01,
                        )
                    if reference_params is None:
                        reference_params = variables["params"]
                    else:
                        self.assertEqual(
                            flattened_shapes(reference_params),
                            flattened_shapes(variables["params"]),
                        )
                        for left, right in zip(
                            jax.tree_util.tree_leaves(reference_params),
                            jax.tree_util.tree_leaves(variables["params"]),
                        ):
                            np.testing.assert_array_equal(left, right)

    def test_cycles_reuse_identical_parameter_tree(self):
        inputs = jnp.ones((1, 5, 128), dtype=jnp.float32)
        settings = {
            "coghp_gelu": dict(token_hidden=32, channel_hidden=15),
            "gelu_layernorm": dict(
                token_hidden=32,
                channel_hidden=14,
            ),
            "swiglu_rmsnorm": dict(mixer_intermediate_size=11),
        }
        for block_type, overrides in settings.items():
            with self.subTest(block_type=block_type):
                h1l1 = make_configurable_core(
                    block_type,
                    h_cycles=1,
                    l_cycles=1,
                    **overrides,
                )
                h2l2 = make_configurable_core(
                    block_type,
                    h_cycles=2,
                    l_cycles=2,
                    **overrides,
                )
                key = jax.random.PRNGKey(11)
                params_h1l1 = h1l1.init(key, None, inputs)["params"]
                params_h2l2 = h2l2.init(key, None, inputs)["params"]
                self.assertEqual(
                    flattened_shapes(params_h1l1),
                    flattened_shapes(params_h2l2),
                )
                for left, right in zip(
                    jax.tree_util.tree_leaves(params_h1l1),
                    jax.tree_util.tree_leaves(params_h2l2),
                ):
                    np.testing.assert_array_equal(left, right)

    def test_coghp_gelu_is_an_exact_independent_mirror(self):
        for input_dtype in (jnp.float32, jnp.bfloat16):
            inputs = jnp.linspace(
                -1.0,
                1.0,
                2 * 5 * 128,
                dtype=input_dtype,
            ).reshape(2, 5, 128)
            for causal in (True, False):
                with self.subTest(causal=causal, input_dtype=input_dtype):
                    local_block = CoGHPMixerBlock(
                        seq_len=5,
                        hidden_size=128,
                        token_hidden=32,
                        channel_hidden=15,
                        is_causal=causal,
                    )
                    original_block = MixerBlock(
                        num_tokens=5,
                        embed_dim=128,
                        hidden_dim_tokens=32,
                        hidden_dim_channels=15,
                        causal=causal,
                    )
                    key = jax.random.PRNGKey(12)
                    local_variables = local_block.init(key, inputs)
                    original_variables = original_block.init(key, inputs)
                    self.assertEqual(
                        flattened_shapes(local_variables["params"]),
                        flattened_shapes(original_variables["params"]),
                    )
                    for local, original in zip(
                        jax.tree_util.tree_leaves(local_variables["params"]),
                        jax.tree_util.tree_leaves(original_variables["params"]),
                    ):
                        np.testing.assert_array_equal(local, original)
                    local_output = local_block.apply(local_variables, inputs)
                    original_output = original_block.apply(
                        original_variables,
                        inputs,
                    )
                    self.assertEqual(local_output.dtype, original_output.dtype)
                    np.testing.assert_allclose(
                        local_output,
                        original_output,
                        rtol=0.0,
                        atol=0.0,
                    )

    def test_gelu_layernorm_affine_parameter_shapes(self):
        inputs = jnp.ones((1, 5, 128), dtype=jnp.float32)
        affine = make_configurable_core(
            "gelu_layernorm",
            token_hidden=32,
            channel_hidden=14,
            layernorm_affine=True,
        ).init(jax.random.PRNGKey(13), None, inputs)["params"]
        no_affine = make_configurable_core(
            "gelu_layernorm",
            token_hidden=32,
            channel_hidden=14,
            layernorm_affine=False,
        ).init(jax.random.PRNGKey(13), None, inputs)["params"]
        flat_affine = flax.traverse_util.flatten_dict(affine)
        norm_shapes = {
            "/".join(path): value.shape
            for path, value in flat_affine.items()
            if "norm" in "/".join(path)
        }
        self.assertEqual(len(norm_shapes), 8)
        self.assertEqual(
            set(norm_shapes.values()),
            {(5,), (128,)},
        )
        self.assertFalse(
            any(
                "norm" in "/".join(path)
                for path in flax.traverse_util.flatten_dict(no_affine)
            )
        )
        self.assertEqual(parameter_count(affine), 8698)
        self.assertEqual(parameter_count(no_affine), 8166)

    def test_gelu_layernorm_residual_and_norm_order_matches_reference(self):
        inputs = jnp.linspace(
            -0.75,
            0.75,
            2 * 5 * 8,
            dtype=jnp.float32,
        ).reshape(2, 5, 8)
        block = GELULayerNormMixerBlock(
            seq_len=5,
            hidden_size=8,
            token_hidden=4,
            channel_hidden=6,
            norm_eps=1e-6,
            layernorm_affine=True,
        )
        variables = block.init(jax.random.PRNGKey(17), inputs)
        params = variables["params"]

        token_input = jnp.swapaxes(inputs, -1, -2)
        token_update = nn.gelu(
            token_input @ params["token_dense1"]["kernel"]
            + params["token_dense1"]["bias"]
        )
        token_update = (
            token_update @ params["token_dense2"]["kernel"]
            + params["token_dense2"]["bias"]
        )
        token_output = nn.LayerNorm(
            epsilon=1e-6,
            use_bias=True,
            use_scale=True,
        ).apply(
            {"params": params["token_norm"]},
            token_input + token_update,
        )
        channel_input = jnp.swapaxes(token_output, -1, -2)
        channel_update = nn.gelu(
            channel_input @ params["channel_dense1"]["kernel"]
            + params["channel_dense1"]["bias"]
        )
        channel_update = (
            channel_update @ params["channel_dense2"]["kernel"]
            + params["channel_dense2"]["bias"]
        )
        reference = nn.LayerNorm(
            epsilon=1e-6,
            use_bias=True,
            use_scale=True,
        ).apply(
            {"params": params["channel_norm"]},
            channel_input + channel_update,
        )
        np.testing.assert_allclose(
            block.apply(variables, inputs),
            reference,
            rtol=0.0,
            atol=0.0,
        )

    def test_configurable_blocks_forward_gradient_and_jit(self):
        inputs = jnp.linspace(
            -1.0,
            1.0,
            2 * 5 * 128,
            dtype=jnp.float32,
        ).reshape(2, 5, 128)
        weights = jnp.linspace(
            0.1,
            1.0,
            inputs.size,
            dtype=jnp.float32,
        ).reshape(inputs.shape)
        for block_type in (
            "swiglu_rmsnorm",
            "coghp_gelu",
            "gelu_layernorm",
        ):
            for core_type in sorted(ACTION_CORE_TYPES):
                with self.subTest(
                    core_type=core_type,
                    block_type=block_type,
                ):
                    core = make_configurable_core(
                        block_type,
                        core_type=core_type,
                    )
                    variables = core.init(
                        jax.random.PRNGKey(14),
                        None,
                        inputs,
                    )
                    eager_carry, eager_output = core.apply(
                        variables,
                        None,
                        inputs,
                    )
                    compiled_carry, compiled_output = jax.jit(
                        lambda value: core.apply(variables, None, value)
                    )(inputs)
                    self.assertEqual(eager_output.shape, inputs.shape)
                    self.assertTrue(bool(jnp.all(jnp.isfinite(eager_output))))
                    # XLA may reassociate float32 reductions under JIT.
                    np.testing.assert_allclose(
                        compiled_output,
                        eager_output,
                        rtol=1e-5,
                        atol=1e-5,
                    )
                    for compiled, eager in zip(
                        jax.tree_util.tree_leaves(compiled_carry),
                        jax.tree_util.tree_leaves(eager_carry),
                    ):
                        np.testing.assert_allclose(
                            compiled,
                            eager,
                            rtol=1e-5,
                            atol=1e-5,
                        )

                    model_state = {
                        name: collection
                        for name, collection in variables.items()
                        if name != "params"
                    }

                    def loss(params):
                        _, output = core.apply(
                            {"params": params, **model_state},
                            None,
                            inputs,
                        )
                        return jnp.sum(output * weights)

                    grads = jax.grad(loss)(variables["params"])
                    self.assertGreater(tree_norm(grads["H_level"]), 1e-8)
                    self.assertGreater(tree_norm(grads["L_level"]), 1e-8)
                    self.assertTrue(
                        all(
                            bool(jnp.all(jnp.isfinite(leaf)))
                            for leaf in jax.tree_util.tree_leaves(grads)
                        )
                    )
                    input_grad = jax.grad(
                        lambda value: jnp.sum(
                            core.apply(variables, None, value)[1] * weights
                        )
                    )(inputs)
                    self.assertGreater(float(jnp.linalg.norm(input_grad)), 1e-8)
                    self.assertTrue(bool(jnp.all(jnp.isfinite(input_grad))))

    def test_real_block_call_counts_under_jit(self):
        inputs = jnp.ones((1, 5, 128), dtype=jnp.float32)
        block_classes = (
            TransformerBlock,
            CoGHPMixerBlock,
            GELULayerNormMixerBlock,
        )
        core_cases = (
            ("matched_feedforward", 2, 2, 1, 1),
            ("single_state_iterative", 2, 2, 2, 4),
            ("hrm", 1, 1, 1, 1),
            ("hrm", 2, 2, 2, 4),
        )
        for block_type in (
            "swiglu_rmsnorm",
            "coghp_gelu",
            "gelu_layernorm",
        ):
            for (
                core_type,
                h_cycles,
                l_cycles,
                expected_h_calls,
                expected_l_calls,
            ) in core_cases:
                with self.subTest(
                    core_type=core_type,
                    block_type=block_type,
                    h_cycles=h_cycles,
                    l_cycles=l_cycles,
                ):
                    core = make_configurable_core(
                        block_type,
                        h_cycles=h_cycles,
                        l_cycles=l_cycles,
                        core_type=core_type,
                    )
                    variables = core.init(
                        jax.random.PRNGKey(15),
                        None,
                        inputs,
                    )

                    def capture(module, method_name):
                        return (
                            isinstance(module, block_classes)
                            and method_name == "__call__"
                        )

                    (_, _), captured = jax.jit(
                        lambda value: core.apply(
                            variables,
                            None,
                            value,
                            capture_intermediates=capture,
                            mutable=["intermediates"],
                        )
                    )(inputs)
                    flat = flax.traverse_util.flatten_dict(captured["intermediates"])
                    h_calls = sum(
                        len(values)
                        for path, values in flat.items()
                        if path[0] == "H_level" and path[-1] == "__call__"
                    )
                    l_calls = sum(
                        len(values)
                        for path, values in flat.items()
                        if path[0] == "L_level" and path[-1] == "__call__"
                    )
                    self.assertEqual(h_calls, expected_h_calls)
                    self.assertEqual(l_calls, expected_l_calls)

    def test_single_h1l1_exactly_degenerates_to_matched_feedforward(self):
        inputs = jnp.linspace(
            -0.5,
            0.5,
            2 * 5 * 128,
            dtype=jnp.float32,
        ).reshape(2, 5, 128)
        for block_type in (
            "swiglu_rmsnorm",
            "coghp_gelu",
            "gelu_layernorm",
        ):
            with self.subTest(block_type=block_type):
                matched = make_configurable_core(
                    block_type,
                    core_type="matched_feedforward",
                )
                single = make_configurable_core(
                    block_type,
                    core_type="single_state_iterative",
                    h_cycles=1,
                    l_cycles=1,
                )
                key = jax.random.PRNGKey(18)
                matched_variables = matched.init(key, None, inputs)
                single_variables = single.init(key, None, inputs)
                self.assertEqual(
                    flattened_shapes(matched_variables["params"]),
                    flattened_shapes(single_variables["params"]),
                )
                for left, right in zip(
                    jax.tree_util.tree_leaves(matched_variables["params"]),
                    jax.tree_util.tree_leaves(single_variables["params"]),
                ):
                    np.testing.assert_array_equal(left, right)
                matched_output = matched.apply(
                    matched_variables,
                    None,
                    inputs,
                )[1]
                single_output = single.apply(
                    single_variables,
                    None,
                    inputs,
                )[1]
                np.testing.assert_array_equal(single_output, matched_output)

    def test_single_h2l2_schedule_and_input_injection_match_reference(self):
        inputs = jnp.linspace(
            -0.8,
            0.8,
            2 * 5 * 128,
            dtype=jnp.float32,
        ).reshape(2, 5, 128)
        for block_type in (
            "swiglu_rmsnorm",
            "coghp_gelu",
            "gelu_layernorm",
        ):
            with self.subTest(block_type=block_type):
                core = make_configurable_core(
                    block_type,
                    core_type="single_state_iterative",
                    h_cycles=2,
                    l_cycles=2,
                )
                variables = core.init(jax.random.PRNGKey(19), None, inputs)
                level_kwargs = dict(
                    seq_len=5,
                    num_layers=1,
                    hidden_size=128,
                    intermediate_size=32,
                    head_dim=32,
                    is_causal=True,
                    is_mlp_mixer=True,
                    mlp_mixer_intermediate_size=11,
                    block_type=block_type,
                    token_hidden=7,
                    channel_hidden=9,
                    layernorm_affine=True,
                )
                low = Transformer(**level_kwargs)
                high = Transformer(**level_kwargs)
                low_variables = {"params": variables["params"]["L_level"]}
                high_variables = {"params": variables["params"]["H_level"]}

                # u0=0; L,L,H,L,L,H, with x injected before every L only.
                reference = jnp.zeros_like(inputs)
                reference = low.apply(low_variables, reference + inputs)
                reference = low.apply(low_variables, reference + inputs)
                reference = high.apply(high_variables, reference)
                reference = low.apply(low_variables, reference + inputs)
                reference = low.apply(low_variables, reference + inputs)
                reference = high.apply(high_variables, reference)
                output = core.apply(variables, None, inputs)[1]
                np.testing.assert_array_equal(output, reference)

    def test_explicit_swiglu_matches_legacy_configuration(self):
        inputs = jnp.linspace(
            -0.5,
            0.5,
            5 * 128,
            dtype=jnp.float32,
        ).reshape(1, 5, 128)
        common = dict(
            seq_len=5,
            num_layers=1,
            hidden_size=128,
            intermediate_size=32,
            head_dim=32,
            is_causal=True,
            h_cycles=2,
            l_cycles=2,
            bptt=True,
            is_mlp_mixer=True,
            mlp_mixer_intermediate_size=11,
        )
        legacy = HRMCore(**common)
        explicit = HRMCore(**common, block_type="swiglu_rmsnorm")
        key = jax.random.PRNGKey(16)
        legacy_variables = legacy.init(key, None, inputs)
        explicit_variables = explicit.init(key, None, inputs)
        self.assertEqual(
            flattened_shapes(legacy_variables),
            flattened_shapes(explicit_variables),
        )
        for left, right in zip(
            jax.tree_util.tree_leaves(legacy_variables),
            jax.tree_util.tree_leaves(explicit_variables),
        ):
            np.testing.assert_array_equal(left, right)
        legacy_output = legacy.apply(legacy_variables, None, inputs)
        explicit_output = explicit.apply(explicit_variables, None, inputs)
        for left, right in zip(
            jax.tree_util.tree_leaves(legacy_output),
            jax.tree_util.tree_leaves(explicit_output),
        ):
            np.testing.assert_array_equal(left, right)

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
            int(leaf.size) for leaf in jax.tree_util.tree_leaves(variables["params"])
        )
        self.assertEqual(actual, expected)
        flat_params = flax.traverse_util.flatten_dict(variables["params"])
        self.assertFalse(any("bias" in path for path in flat_params))

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

                self.assertEqual(set(variables), {"params", "constants"})
                self.assertEqual(variables["constants"]["zH_init"].shape, (8,))
                self.assertEqual(variables["constants"]["zL_init"].shape, (8,))
                self.assertNotIn("zH_init", variables["params"])
                self.assertNotIn("zL_init", variables["params"])

                run = jax.jit(lambda x: core.apply(variables, None, x))
                carry, outputs = run(inputs)
                self.assertEqual(outputs.shape, inputs.shape)
                self.assertEqual(carry["z_H"].shape, inputs.shape)
                self.assertEqual(carry["z_L"].shape, inputs.shape)
                self.assertTrue(bool(jnp.all(jnp.isfinite(outputs))))

    def test_bptt_flag_matches_official_preliminary_iteration_cut(self):
        common_kwargs = dict(
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
            token_hidden=5,
            channel_hidden=7,
        )
        inputs = jnp.linspace(-1.0, 1.0, 18, dtype=jnp.float32).reshape(1, 3, 6)
        initial_carry = {
            "z_H": jnp.full_like(inputs, 0.25),
            "z_L": jnp.full_like(inputs, -0.15),
        }
        weights = jnp.linspace(0.1, 1.0, inputs.size).reshape(inputs.shape)
        for block_type in (
            "swiglu_rmsnorm",
            "coghp_gelu",
            "gelu_layernorm",
        ):
            with self.subTest(block_type=block_type):
                model_kwargs = dict(
                    common_kwargs,
                    block_type=block_type,
                )
                full_core = HRMCore(**model_kwargs, bptt=True)
                truncated_core = HRMCore(**model_kwargs, bptt=False)
                variables = full_core.init(
                    jax.random.PRNGKey(2),
                    None,
                    inputs,
                )

                def loss(core, carry):
                    _, output = core.apply(variables, carry, inputs)
                    return jnp.sum(output * weights)

                full_grad = jax.grad(lambda carry: loss(full_core, carry))(
                    initial_carry
                )
                truncated_grad = jax.grad(lambda carry: loss(truncated_core, carry))(
                    initial_carry
                )

                self.assertGreater(tree_norm(full_grad), 1e-8)
                self.assertEqual(tree_norm(truncated_grad), 0.0)

    def test_single_state_bptt_uses_the_same_final_pair_cut(self):
        inputs = jnp.linspace(-1.0, 1.0, 18, dtype=jnp.float32).reshape(1, 3, 6)
        initial_carry = {"u": jnp.full_like(inputs, 0.2)}
        weights = jnp.linspace(0.1, 1.0, inputs.size).reshape(inputs.shape)
        common = dict(
            seq_len=3,
            num_layers=1,
            hidden_size=6,
            intermediate_size=12,
            head_dim=2,
            is_causal=False,
            h_cycles=2,
            l_cycles=2,
            core_type="single_state_iterative",
            is_mlp_mixer=True,
            mlp_mixer_intermediate_size=7,
            token_hidden=5,
            channel_hidden=7,
        )
        for block_type in (
            "swiglu_rmsnorm",
            "coghp_gelu",
            "gelu_layernorm",
        ):
            with self.subTest(block_type=block_type):
                full = HRMCore(**common, block_type=block_type, bptt=True)
                truncated = HRMCore(
                    **common,
                    block_type=block_type,
                    bptt=False,
                )
                variables = full.init(
                    jax.random.PRNGKey(20),
                    None,
                    inputs,
                )

                def loss(core, carry):
                    output = core.apply(variables, carry, inputs)[1]
                    return jnp.sum(output * weights)

                full_grad = jax.grad(lambda carry: loss(full, carry))(initial_carry)
                truncated_grad = jax.grad(lambda carry: loss(truncated, carry))(
                    initial_carry
                )
                self.assertGreater(tree_norm(full_grad), 1e-8)
                self.assertEqual(tree_norm(truncated_grad), 0.0)


if __name__ == "__main__":
    unittest.main()
