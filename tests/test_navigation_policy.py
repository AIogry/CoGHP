"""Synthetic tests for shared-block CoGHP action cores."""

import pathlib
import sys
import tempfile
import unittest

import flax
import jax
import jax.numpy as jnp
import numpy as np

IMPLS_DIR = pathlib.Path(__file__).resolve().parents[1] / "impls"
if str(IMPLS_DIR) not in sys.path:
    sys.path.insert(0, str(IMPLS_DIR))

from agents.navigation import (
    NavigationAgent as HRMActionCoGHPAgent,
    get_legacy_action_config as get_config,
)
from networks.policies import (
    NavigationPolicy as CoGHPPolicyNetwork,
    resolve_action_block_type,
)
from utils.coghp_network import HierarchicalPolicyNetwork
from utils.flax_utils import build_checkpoint_metadata, restore_agent, save_agent
from utils.networks import GCActor

STATE_DIM = 8
ACTION_DIM = 2
BATCH_SIZE = 2
NUM_SUBGOALS = 2


def actor_heads():
    high_actor = GCActor(
        hidden_dims=(8,),
        action_dim=STATE_DIM,
        state_dependent_std=False,
        const_std=True,
        gc_encoder=None,
    )
    low_actor = GCActor(
        hidden_dims=(8,),
        action_dim=ACTION_DIM,
        state_dependent_std=False,
        const_std=True,
        gc_encoder=None,
    )
    return high_actor, low_actor


def parameter_count(tree):
    return sum(int(leaf.size) for leaf in jax.tree_util.tree_leaves(tree))


def make_network(
    is_mlp_mixer=True,
    bptt=True,
    chain_source="teacher",
    block_type="legacy",
    token_hidden=5,
    channel_hidden=6,
    mixer_intermediate_size=-1,
    layernorm_affine=True,
    planner_core_type="vanilla_coghp",
    planner_block_type="swiglu_rmsnorm",
    planner_num_layers=1,
    planner_mixer_intermediate_size=12,
    planner_h_cycles=2,
    planner_l_cycles=2,
    planner_bptt=True,
    core_type="hrm",
    action_block_type="auto",
    h_cycles=2,
    l_cycles=2,
    share_actor_core=False,
):
    high_actor, low_actor = actor_heads()
    return CoGHPPolicyNetwork(
        state_dim=STATE_DIM,
        high_actor_head=high_actor,
        low_actor_head=low_actor,
        num_mixer_blocks=1,
        mixer_token_hidden=8,
        mixer_channel_hidden=8,
        gc_encoder=None,
        enc_hidden=(8,),
        num_subgoals=NUM_SUBGOALS,
        causal_mixer=True,
        planner_core_type=planner_core_type,
        planner_block_type=planner_block_type,
        planner_num_layers=planner_num_layers,
        planner_mixer_intermediate_size=(planner_mixer_intermediate_size),
        planner_h_cycles=planner_h_cycles,
        planner_l_cycles=planner_l_cycles,
        planner_bptt=planner_bptt,
        planner_norm_eps=1e-6,
        planner_forward_dtype="float32",
        share_actor_core=share_actor_core,
        action_hrm_num_layers=1,
        action_hrm_intermediate_size=16,
        action_hrm_head_dim=2,
        action_hrm_h_cycles=h_cycles,
        action_hrm_l_cycles=l_cycles,
        action_hrm_bptt=bptt,
        action_hrm_is_causal=True,
        action_hrm_is_mlp_mixer=is_mlp_mixer,
        action_hrm_mlp_mixer_intermediate_size=12,
        action_core_type=core_type,
        action_block_type=action_block_type,
        hrm_block_type=block_type,
        hrm_token_hidden=token_hidden,
        hrm_channel_hidden=channel_hidden,
        hrm_mixer_intermediate_size=mixer_intermediate_size,
        hrm_layernorm_affine=layernorm_affine,
        train_action_chain_source=chain_source,
    )


def synthetic_inputs():
    observations = (
        jnp.arange(BATCH_SIZE * 5, dtype=jnp.float32).reshape(BATCH_SIZE, 5) / 10.0
    )
    goals = jnp.flip(observations, axis=0)
    teacher_subgoals = jnp.arange(
        BATCH_SIZE * NUM_SUBGOALS * STATE_DIM,
        dtype=jnp.float32,
    ).reshape(BATCH_SIZE, NUM_SUBGOALS, STATE_DIM)
    teacher_subgoals = teacher_subgoals / teacher_subgoals.size
    actions = jnp.linspace(
        -0.5,
        0.5,
        BATCH_SIZE * ACTION_DIM,
        dtype=jnp.float32,
    ).reshape(BATCH_SIZE, ACTION_DIM)
    return observations, goals, teacher_subgoals, actions


def initialize(network):
    observations, goals, _, _ = synthetic_inputs()
    return network.init(
        jax.random.PRNGKey(0),
        observations,
        goals,
        jax.random.PRNGKey(1),
    )


def subtree_norm(tree, pattern):
    flat = flax.traverse_util.flatten_dict(tree)
    return sum(
        float(jnp.linalg.norm(value))
        for path, value in flat.items()
        if pattern in "/".join(path)
    )


class HRMActionPolicyNetworkTest(unittest.TestCase):
    def test_action_block_selector_preserves_all_legacy_fallbacks(self):
        self.assertEqual(
            resolve_action_block_type("coghp_gelu", "swiglu_rmsnorm", True),
            "coghp_gelu",
        )
        self.assertEqual(
            resolve_action_block_type("auto", "gelu_layernorm", True),
            "gelu_layernorm",
        )
        self.assertEqual(
            resolve_action_block_type("auto", "legacy", True),
            "swiglu_rmsnorm",
        )
        self.assertEqual(
            resolve_action_block_type("auto", "legacy", False),
            "attention",
        )
        self.assertEqual(
            resolve_action_block_type("attention", "swiglu_rmsnorm", True),
            "attention",
        )

    def test_reads_complete_sequence_and_returns_one_action_distribution(self):
        network = make_network()
        variables = initialize(network)
        observations, goals, teacher_subgoals, _ = synthetic_inputs()

        outputs, mutable = network.apply(
            variables,
            observations,
            goals,
            jax.random.PRNGKey(2),
            subgoal_reps=teacher_subgoals,
            return_diagnostics=True,
            mutable=["intermediates"],
        )
        high_dists, action_dist, actions, diagnostics = outputs
        self.assertEqual(len(high_dists), NUM_SUBGOALS)
        self.assertEqual(action_dist.mode().shape, (BATCH_SIZE, ACTION_DIM))
        self.assertEqual(actions.shape, (BATCH_SIZE, ACTION_DIM))

        token_sequence = mutable["intermediates"]["action_token_sequence"][0]
        self.assertEqual(
            token_sequence.shape,
            (BATCH_SIZE, NUM_SUBGOALS + 3, STATE_DIM),
        )
        np.testing.assert_allclose(
            token_sequence[:, 2 : 2 + NUM_SUBGOALS, :],
            teacher_subgoals,
        )
        z_high = mutable["intermediates"]["action_z_high"][0]
        action_representation = mutable["intermediates"]["action_representation"][0]
        np.testing.assert_array_equal(
            action_representation,
            z_high[:, -1, :].astype(jnp.float32),
        )
        self.assertEqual(int(diagnostics["action_hrm/outer_cycles"]), 1)
        self.assertEqual(int(diagnostics["action_hrm/high_level_calls"]), 2)
        self.assertEqual(int(diagnostics["action_hrm/low_level_calls"]), 4)
        self.assertEqual(
            int(diagnostics["action_hrm/token_sequence_length"]),
            NUM_SUBGOALS + 3,
        )

        self.assertIn("constants", variables)
        constant_shapes = {
            "/".join(path): value.shape
            for path, value in flax.traverse_util.flatten_dict(
                variables["constants"]
            ).items()
        }
        self.assertEqual(
            constant_shapes["action_hrm/zH_init"],
            (STATE_DIM,),
        )
        self.assertEqual(
            constant_shapes["action_hrm/zL_init"],
            (STATE_DIM,),
        )

    def test_vanilla_coghp_executor_uses_original_mixer_stack(self):
        network = make_network(core_type="vanilla_coghp")
        variables = initialize(network)
        self.assertIn("action_mixer_blocks_0", variables["params"])
        self.assertNotIn("action_hrm", variables["params"])
        self.assertNotIn("constants", variables)

        # One original CoGHP MixerBlock with N=5, D=8, token/channel width 8.
        expected_params = (
            (5 * 8 + 8)  # token_dense1
            + (8 * 5 + 5)  # token_dense2
            + (8 * 8 + 8)  # channel_dense1
            + (8 * 8 + 8)  # channel_dense2
            + (5 * 5)  # tm_weights
        )
        self.assertEqual(
            parameter_count(variables["params"]["action_mixer_blocks_0"]),
            expected_params,
        )

        observations, goals, teacher_subgoals, actions = synthetic_inputs()
        model_state = {
            name: collection
            for name, collection in variables.items()
            if name != "params"
        }
        high_dists, _, _, diagnostics = network.apply(
            {"params": variables["params"], **model_state},
            observations,
            goals,
            jax.random.PRNGKey(19),
            subgoal_reps=teacher_subgoals,
            return_diagnostics=True,
        )
        self.assertEqual(len(high_dists), NUM_SUBGOALS)
        self.assertEqual(
            int(diagnostics["action_core/total_block_calls"]),
            1,
        )
        self.assertEqual(
            int(diagnostics["action_core/high_block_calls"]),
            0,
        )
        self.assertEqual(
            int(diagnostics["action_core/low_block_calls"]),
            0,
        )

        def action_loss(params):
            _, dist, _ = network.apply(
                {"params": params, **model_state},
                observations,
                goals,
                jax.random.PRNGKey(19),
                subgoal_reps=teacher_subgoals,
            )
            return -jnp.mean(dist.log_prob(actions))

        grads = jax.grad(action_loss)(variables["params"])
        self.assertGreater(
            subtree_norm(grads, "action_mixer_blocks_0"),
            0.0,
        )

    def test_every_core_reads_final_query_without_fake_hrm_states(self):
        observations, goals, teacher_subgoals, _ = synthetic_inputs()
        expected_calls = {
            "matched_feedforward": (1, 1),
            "single_state_iterative": (4, 2),
            "hrm": (4, 2),
        }
        for core_type, (low_calls, high_calls) in expected_calls.items():
            with self.subTest(core_type=core_type):
                network = make_network(
                    core_type=core_type,
                    action_block_type="swiglu_rmsnorm",
                )
                variables = initialize(network)
                outputs, mutable = network.apply(
                    variables,
                    observations,
                    goals,
                    jax.random.PRNGKey(21),
                    subgoal_reps=teacher_subgoals,
                    return_diagnostics=True,
                    mutable=["intermediates"],
                )
                diagnostics = outputs[-1]
                core_output = mutable["intermediates"]["action_core_output"][0]
                representation = mutable["intermediates"]["action_representation"][0]
                np.testing.assert_array_equal(
                    representation,
                    core_output[:, -1, :].astype(jnp.float32),
                )
                self.assertEqual(
                    int(diagnostics["action_core/low_block_calls"]),
                    low_calls,
                )
                self.assertEqual(
                    int(diagnostics["action_core/high_block_calls"]),
                    high_calls,
                )
                self.assertEqual(
                    int(diagnostics["action_core/total_block_calls"]),
                    low_calls + high_calls,
                )
                if core_type == "hrm":
                    self.assertIn("constants", variables)
                    self.assertIn("action_z_high", mutable["intermediates"])
                    self.assertIn("action_z_low", mutable["intermediates"])
                else:
                    self.assertNotIn("constants", variables)
                    self.assertNotIn("action_z_high", mutable["intermediates"])
                    self.assertNotIn("action_z_low", mutable["intermediates"])
                if core_type == "single_state_iterative":
                    self.assertIn(
                        "action_single_state",
                        mutable["intermediates"],
                    )

    def test_all_action_core_and_block_combinations_are_jittable(self):
        observations, goals, teacher_subgoals, _ = synthetic_inputs()
        cases = [
            (core_type, block_type, True)
            for core_type in (
                "matched_feedforward",
                "single_state_iterative",
                "hrm",
            )
            for block_type in (
                "swiglu_rmsnorm",
                "coghp_gelu",
                "gelu_layernorm",
            )
        ]
        cases.append(("hrm", "attention", False))
        for core_type, block_type, is_mlp_mixer in cases:
            with self.subTest(
                core_type=core_type,
                block_type=block_type,
                is_mlp_mixer=is_mlp_mixer,
            ):
                network = make_network(
                    is_mlp_mixer=is_mlp_mixer,
                    core_type=core_type,
                    action_block_type=(
                        block_type if block_type != "attention" else "auto"
                    ),
                    block_type="legacy",
                )
                variables = initialize(network)

                @jax.jit
                def run(obs, goal, subgoals):
                    high_dists, action_dist, actions = network.apply(
                        variables,
                        obs,
                        goal,
                        jax.random.PRNGKey(3),
                        subgoal_reps=subgoals,
                    )
                    return (
                        jnp.stack(
                            [dist.mode() for dist in high_dists],
                            axis=1,
                        ),
                        action_dist.mode(),
                        actions,
                    )

                high_modes, action_mode, actions = run(
                    observations,
                    goals,
                    teacher_subgoals,
                )
                self.assertEqual(
                    high_modes.shape,
                    (BATCH_SIZE, NUM_SUBGOALS, STATE_DIM),
                )
                self.assertEqual(action_mode.shape, (BATCH_SIZE, ACTION_DIM))
                self.assertEqual(actions.shape, (BATCH_SIZE, ACTION_DIM))

    def test_action_loss_trains_both_levels_for_every_core_and_block(self):
        observations, goals, teacher_subgoals, actions = synthetic_inputs()
        for core_type in (
            "matched_feedforward",
            "single_state_iterative",
            "hrm",
        ):
            for block_type in (
                "swiglu_rmsnorm",
                "coghp_gelu",
                "gelu_layernorm",
            ):
                with self.subTest(
                    core_type=core_type,
                    block_type=block_type,
                ):
                    network = make_network(
                        core_type=core_type,
                        action_block_type=block_type,
                    )
                    variables = initialize(network)
                    model_state = {
                        name: collection
                        for name, collection in variables.items()
                        if name != "params"
                    }

                    def action_loss(params):
                        _, action_dist, _ = network.apply(
                            {"params": params, **model_state},
                            observations,
                            goals,
                            jax.random.PRNGKey(4),
                            subgoal_reps=teacher_subgoals,
                        )
                        return -jnp.mean(action_dist.log_prob(actions))

                    grads = jax.grad(action_loss)(variables["params"])
                    self.assertGreater(
                        subtree_norm(grads, "action_hrm/H_level"),
                        0.0,
                    )
                    self.assertGreater(
                        subtree_norm(grads, "action_hrm/L_level"),
                        0.0,
                    )
                    self.assertGreater(
                        subtree_norm(grads, "low_actor_head"),
                        0.0,
                    )
                    self.assertGreater(
                        subtree_norm(grads, "action_query_token"),
                        0.0,
                    )
                    self.assertEqual(
                        subtree_norm(grads, "planner_mixer"),
                        0.0,
                    )
                    self.assertEqual(
                        subtree_norm(grads, "high_actor_head"),
                        0.0,
                    )

    def test_swiglu_planner_matched_feedforward_integration(self):
        # Default path must still preserve the legacy CoGHP planner.
        legacy_network = make_network()
        legacy_variables = initialize(legacy_network)

        self.assertIn(
            "planner_mixer_0",
            legacy_variables["params"],
        )
        self.assertNotIn(
            "planner_core",
            legacy_variables["params"],
        )

        # New planner path.
        planner_width = 12
        network = make_network(
            planner_core_type="matched_feedforward",
            planner_block_type="swiglu_rmsnorm",
            planner_num_layers=1,
            planner_mixer_intermediate_size=planner_width,
        )
        variables = initialize(network)

        self.assertIn("planner_core", variables["params"])
        self.assertNotIn(
            "planner_mixer_0",
            variables["params"],
        )

        # For the synthetic test:
        # sequence length = NUM_SUBGOALS + 3
        # hidden size = STATE_DIM
        # one SwiGLU block = 3 * M * (N + D)
        # matched FF owns independent B_L and B_H.
        sequence_length = NUM_SUBGOALS + 3
        expected_params = 2 * 3 * planner_width * (sequence_length + STATE_DIM)

        self.assertEqual(
            parameter_count(variables["params"]["planner_core"]),
            expected_params,
        )

        # The matched-feedforward planner must not create zL/zH state.
        constant_paths = {
            path
            for path in flax.traverse_util.flatten_dict(variables.get("constants", {}))
        }
        self.assertFalse(
            any(path and path[0] == "planner_core" for path in constant_paths)
        )

        observations, goals, teacher_subgoals, _ = synthetic_inputs()
        model_state = {
            name: collection
            for name, collection in variables.items()
            if name != "params"
        }

        def planner_loss(params):
            high_dists, _, _ = network.apply(
                {
                    "params": params,
                    **model_state,
                },
                observations,
                goals,
                jax.random.PRNGKey(31),
                subgoal_reps=teacher_subgoals,
            )
            return sum(jnp.mean(dist.mode()) for dist in high_dists)

        grads = jax.grad(planner_loss)(variables["params"])

        self.assertGreater(
            subtree_norm(grads, "planner_core"),
            1e-8,
        )

    def test_official_hrm_planner_uses_independent_recurrent_core(self):
        planner_width = 12
        network = make_network(
            planner_core_type="hrm",
            planner_block_type="swiglu_rmsnorm",
            planner_num_layers=1,
            planner_mixer_intermediate_size=planner_width,
            planner_h_cycles=2,
            planner_l_cycles=2,
            planner_bptt=False,
        )
        variables = initialize(network)
        self.assertIn("planner_core", variables["params"])
        self.assertNotIn("planner_mixer_0", variables["params"])

        sequence_length = NUM_SUBGOALS + 3
        expected_params = 2 * 3 * planner_width * (sequence_length + STATE_DIM)
        self.assertEqual(
            parameter_count(variables["params"]["planner_core"]),
            expected_params,
        )

        constant_paths = {
            path
            for path in flax.traverse_util.flatten_dict(variables.get("constants", {}))
        }
        self.assertTrue(
            any(path[:2] == ("planner_core", "zH_init") for path in constant_paths)
        )
        self.assertTrue(
            any(path[:2] == ("planner_core", "zL_init") for path in constant_paths)
        )

        observations, goals, teacher_subgoals, _ = synthetic_inputs()
        model_state = {
            name: collection
            for name, collection in variables.items()
            if name != "params"
        }

        def apply_model(params):
            return network.apply(
                {"params": params, **model_state},
                observations,
                goals,
                jax.random.PRNGKey(37),
                subgoal_reps=teacher_subgoals,
            )

        high_dists, _, _ = jax.jit(apply_model)(variables["params"])
        self.assertEqual(len(high_dists), NUM_SUBGOALS)

        def planner_loss(params):
            high_dists, _, _ = apply_model(params)
            return sum(jnp.mean(dist.mode()) for dist in high_dists)

        grads = jax.grad(planner_loss)(variables["params"])
        self.assertGreater(subtree_norm(grads, "planner_core"), 1e-8)

    def test_shared_official_hrm_core_has_two_readouts_and_one_parameter_tree(self):
        network = make_network(
            planner_core_type="hrm",
            planner_block_type="swiglu_rmsnorm",
            planner_h_cycles=2,
            planner_l_cycles=2,
            planner_bptt=False,
            core_type="hrm",
            action_block_type="swiglu_rmsnorm",
            h_cycles=2,
            l_cycles=2,
            bptt=False,
            share_actor_core=True,
        )
        variables = initialize(network)
        actor_params = variables["params"]
        self.assertIn("shared_actor_core", actor_params)
        self.assertNotIn("planner_core", actor_params)
        self.assertNotIn("action_hrm", actor_params)
        self.assertEqual(
            parameter_count(actor_params["shared_actor_core"]),
            2 * 3 * 12 * (NUM_SUBGOALS + 3 + STATE_DIM),
        )

        constant_paths = {
            path
            for path in flax.traverse_util.flatten_dict(variables.get("constants", {}))
        }
        self.assertTrue(
            any(path[:2] == ("shared_actor_core", "zH_init") for path in constant_paths)
        )
        self.assertTrue(
            any(path[:2] == ("shared_actor_core", "zL_init") for path in constant_paths)
        )

        observations, goals, teacher_subgoals, _ = synthetic_inputs()
        model_state = {
            name: collection
            for name, collection in variables.items()
            if name != "params"
        }

        def apply_model(params):
            return network.apply(
                {"params": params, **model_state},
                observations,
                goals,
                jax.random.PRNGKey(41),
                subgoal_reps=teacher_subgoals,
                return_diagnostics=True,
            )

        outputs = jax.jit(apply_model)(variables["params"])
        self.assertEqual(len(outputs[0]), NUM_SUBGOALS)
        self.assertEqual(
            int(outputs[-1]["action_core/total_block_calls"]),
            6,
        )

        def actor_loss(params):
            high_dists, action_dist, _, _ = apply_model(params)
            return sum(jnp.mean(dist.mode()) for dist in high_dists) + jnp.mean(
                action_dist.mode()
            )

        grads = jax.grad(actor_loss)(variables["params"])
        self.assertGreater(subtree_norm(grads, "shared_actor_core"), 1e-8)
        self.assertGreater(subtree_norm(grads, "high_actor_head"), 1e-8)
        self.assertGreater(subtree_norm(grads, "low_actor_head"), 1e-8)

    def test_block_selector_reaches_only_the_action_hrm_core(self):
        expected_markers = {
            "swiglu_rmsnorm": "gate_up_proj",
            "coghp_gelu": "tm_weights",
            "gelu_layernorm": "token_norm/scale",
        }
        reference = flax.traverse_util.flatten_dict(
            initialize(make_network(block_type="swiglu_rmsnorm"))["params"]
        )
        reference_non_core = {
            path: value for path, value in reference.items() if "action_hrm" not in path
        }
        for block_type, marker in expected_markers.items():
            with self.subTest(block_type=block_type):
                flat_params = flax.traverse_util.flatten_dict(
                    initialize(make_network(block_type=block_type))["params"]
                )
                paths = {"/".join(path) for path in flat_params}
                matching_paths = [path for path in paths if marker in path]
                self.assertTrue(matching_paths)
                self.assertTrue(any("action_hrm/" in path for path in matching_paths))
                current_non_core = {
                    path: value
                    for path, value in flat_params.items()
                    if "action_hrm" not in path
                }
                self.assertEqual(
                    set(current_non_core),
                    set(reference_non_core),
                )
                for path in reference_non_core:
                    np.testing.assert_array_equal(
                        current_non_core[path],
                        reference_non_core[path],
                    )

    def test_core_selector_changes_schedule_not_any_parameter(self):
        reference = initialize(
            make_network(
                core_type="hrm",
                action_block_type="swiglu_rmsnorm",
            )
        )["params"]
        for core_type in (
            "matched_feedforward",
            "single_state_iterative",
            "hrm",
        ):
            with self.subTest(core_type=core_type):
                current = initialize(
                    make_network(
                        core_type=core_type,
                        action_block_type="swiglu_rmsnorm",
                    )
                )["params"]
                self.assertEqual(
                    {
                        path: value.shape
                        for path, value in flax.traverse_util.flatten_dict(
                            current
                        ).items()
                    },
                    {
                        path: value.shape
                        for path, value in flax.traverse_util.flatten_dict(
                            reference
                        ).items()
                    },
                )
                for left, right in zip(
                    jax.tree_util.tree_leaves(current),
                    jax.tree_util.tree_leaves(reference),
                ):
                    np.testing.assert_array_equal(left, right)

    def test_legacy_segment_and_residual_parameters_are_absent(self):
        variables = initialize(make_network())
        parameter_paths = [
            "/".join(path)
            for path in flax.traverse_util.flatten_dict(variables["params"])
        ]
        forbidden = (
            "segment",
            "readout",
            "residual",
            "action_context_mixer",
            "action_high_cell",
            "action_low_cell",
        )
        for name in forbidden:
            self.assertFalse(
                any(name in path for path in parameter_paths),
                name,
            )


class SplitActionMixerBaselineTest(unittest.TestCase):
    def test_split_baseline_has_independent_mixer_and_action_query(self):
        high_actor, low_actor = actor_heads()
        network = HierarchicalPolicyNetwork(
            num_tokens=1,
            state_dim=STATE_DIM,
            num_action_dims=ACTION_DIM,
            joint_embed_dim=STATE_DIM,
            num_mixer_blocks=1,
            mixer_token_hidden=8,
            mixer_channel_hidden=8,
            gc_encoder=None,
            high_actor_head=high_actor,
            low_actor_head=low_actor,
            enc_hidden=(8,),
            num_subgoals=NUM_SUBGOALS,
            causal_mixer=True,
            action_use_full_subgoal_chain=True,
            separate_action_mixer=True,
        )
        observations, goals, teacher_subgoals, actions = synthetic_inputs()
        params = network.init(
            jax.random.PRNGKey(5),
            observations,
            goals,
            jax.random.PRNGKey(6),
        )["params"]

        self.assertIn("subgoal_mixer_blocks_0", params)
        self.assertIn("action_mixer_blocks_0", params)
        self.assertIn("action_query_token", params)

        def action_loss(current_params):
            _, action_dist, _ = network.apply(
                {"params": current_params},
                observations,
                goals,
                jax.random.PRNGKey(7),
                subgoal_reps=teacher_subgoals,
            )
            return -jnp.mean(action_dist.log_prob(actions))

        grads = jax.grad(action_loss)(params)
        self.assertEqual(subtree_norm(grads, "subgoal_mixer_blocks"), 0.0)
        self.assertGreater(subtree_norm(grads, "action_mixer_blocks"), 0.0)
        self.assertEqual(subtree_norm(grads, "prev_tokens"), 0.0)
        self.assertGreater(subtree_norm(grads, "action_query_token"), 0.0)


class HRMActionAgentTest(unittest.TestCase):
    @staticmethod
    def checkpoint_metadata(config, *, seed=0):
        return build_checkpoint_metadata(
            config,
            environment="antmaze-giant-navigate-v0",
            seed=seed,
            git_commit="test-commit",
            code_signature="test-signature",
            resume=False,
            restore_path=None,
        )

    def make_agent(
        self,
        *,
        block_type="swiglu_rmsnorm",
        token_hidden=5,
        channel_hidden=6,
        mixer_intermediate_size=-1,
        layernorm_affine=True,
        core_type="hrm",
        use_legacy_selector=False,
        drop_new_selectors=False,
        legacy_is_mlp_mixer=True,
        expected_action_core_params=-1,
    ):
        config = get_config()
        config.encoder = None
        config.frame_stack = None
        config.action_chunk = 1
        config.feature_dim = STATE_DIM
        config.mixer_hidden = 8
        config.num_mixer_blocks = 1
        config.enc_hidden_dims = (16,)
        config.actor_hidden_dims = (16,)
        config.value_hidden_dims = (16,)
        config.num_subgoals = NUM_SUBGOALS
        config.action_hrm_num_layers = 1
        config.action_hrm_intermediate_size = 16
        config.action_hrm_head_dim = 2
        config.action_hrm_h_cycles = 2
        config.action_hrm_l_cycles = 2
        config.action_hrm_mlp_mixer_intermediate_size = 12
        config.action_hrm_is_mlp_mixer = legacy_is_mlp_mixer
        config.action_core_type = core_type
        config.action_block_type = "auto" if use_legacy_selector else block_type
        config.hrm_block_type = block_type if use_legacy_selector else "legacy"
        config.hrm_token_hidden = token_hidden
        config.hrm_channel_hidden = channel_hidden
        config.hrm_mixer_intermediate_size = mixer_intermediate_size
        config.hrm_layernorm_affine = layernorm_affine
        config.expected_action_core_params = expected_action_core_params
        if drop_new_selectors:
            for field in (
                "action_core_type",
                "action_block_type",
                "hrm_block_type",
                "hrm_token_hidden",
                "hrm_channel_hidden",
                "hrm_mixer_intermediate_size",
                "hrm_layernorm_affine",
            ):
                del config[field]
        observations = jnp.ones((BATCH_SIZE, 5), dtype=jnp.float32)
        actions = jnp.zeros((BATCH_SIZE, ACTION_DIM), dtype=jnp.float32)
        return HRMActionCoGHPAgent.create(0, observations, actions, config)

    def test_old_attention_boolean_is_not_shadowed_by_new_defaults(self):
        legacy_agent = self.make_agent(
            block_type="attention",
            use_legacy_selector=True,
            drop_new_selectors=True,
            legacy_is_mlp_mixer=False,
        )
        explicit_agent = self.make_agent(block_type="attention")
        legacy_flat = flax.traverse_util.flatten_dict(legacy_agent.network.params)
        explicit_flat = flax.traverse_util.flatten_dict(explicit_agent.network.params)
        self.assertEqual(set(legacy_flat), set(explicit_flat))
        self.assertTrue(any("attn/qkv_proj" in "/".join(path) for path in legacy_flat))
        for path in legacy_flat:
            np.testing.assert_array_equal(
                legacy_flat[path],
                explicit_flat[path],
            )
        for old, new in zip(
            jax.tree_util.tree_leaves(legacy_agent.network.model_state),
            jax.tree_util.tree_leaves(explicit_agent.network.model_state),
        ):
            np.testing.assert_array_equal(old, new)

    def test_vanilla_executor_parameter_contract_reaches_agent(self):
        agent = self.make_agent(
            core_type="vanilla_coghp",
            expected_action_core_params=262,
        )
        actor_params = agent.network.params["modules_actor_mixer"]
        self.assertIn("action_mixer_blocks_0", actor_params)
        self.assertNotIn("action_hrm", actor_params)

    def test_old_config_and_checkpoint_restore_into_default_hrm(self):
        legacy_agent = self.make_agent(
            block_type="swiglu_rmsnorm",
            use_legacy_selector=True,
            drop_new_selectors=True,
        )
        current_agent = self.make_agent(block_type="swiglu_rmsnorm")
        self.assertEqual(
            {
                path: value.shape
                for path, value in flax.traverse_util.flatten_dict(
                    legacy_agent.network.params
                ).items()
            },
            {
                path: value.shape
                for path, value in flax.traverse_util.flatten_dict(
                    current_agent.network.params
                ).items()
            },
        )
        for old, new in zip(
            jax.tree_util.tree_leaves(legacy_agent.network.params),
            jax.tree_util.tree_leaves(current_agent.network.params),
        ):
            np.testing.assert_array_equal(old, new)
        for old, new in zip(
            jax.tree_util.tree_leaves(legacy_agent.network.model_state),
            jax.tree_util.tree_leaves(current_agent.network.model_state),
        ):
            np.testing.assert_array_equal(old, new)

        with tempfile.TemporaryDirectory() as checkpoint_dir:
            save_agent(legacy_agent, checkpoint_dir, 0)
            restored = restore_agent(current_agent, checkpoint_dir, 0)
        for old, new in zip(
            jax.tree_util.tree_leaves(legacy_agent.network.params),
            jax.tree_util.tree_leaves(restored.network.params),
        ):
            np.testing.assert_array_equal(old, new)
        for old, new in zip(
            jax.tree_util.tree_leaves(legacy_agent.network.model_state),
            jax.tree_util.tree_leaves(restored.network.model_state),
        ):
            np.testing.assert_array_equal(old, new)
        for old, new in zip(
            jax.tree_util.tree_leaves(legacy_agent.network.opt_state),
            jax.tree_util.tree_leaves(restored.network.opt_state),
        ):
            np.testing.assert_array_equal(old, new)
        observations, goals, _, _ = synthetic_inputs()
        old_action = legacy_agent.sample_actions(
            observations[0],
            goals=goals[0],
            seed=jax.random.PRNGKey(22),
            temperature=0.0,
        )
        restored_action = restored.sample_actions(
            observations[0],
            goals=goals[0],
            seed=jax.random.PRNGKey(22),
            temperature=0.0,
        )
        np.testing.assert_array_equal(old_action, restored_action)

    def test_signed_checkpoint_rejects_mismatched_action_core(self):
        agent = self.make_agent(
            block_type="swiglu_rmsnorm",
            core_type="matched_feedforward",
        )
        metadata = self.checkpoint_metadata(agent.config)
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            save_agent(
                agent,
                checkpoint_dir,
                0,
                checkpoint_metadata=metadata,
            )
            restored = restore_agent(
                agent,
                checkpoint_dir,
                0,
                expected_checkpoint_metadata=metadata,
                require_checkpoint_metadata=True,
            )
            for old, new in zip(
                jax.tree_util.tree_leaves(agent.network.params),
                jax.tree_util.tree_leaves(restored.network.params),
            ):
                np.testing.assert_array_equal(old, new)

            mismatched_config = dict(agent.config)
            mismatched_config["action_core_type"] = "single_state_iterative"
            mismatched_metadata = self.checkpoint_metadata(mismatched_config)
            with self.assertRaisesRegex(
                ValueError,
                "configuration signature mismatch",
            ):
                restore_agent(
                    agent,
                    checkpoint_dir,
                    0,
                    expected_checkpoint_metadata=mismatched_metadata,
                    require_checkpoint_metadata=True,
                )

    def test_new_config_reaches_core_through_agent_and_train_state(self):
        cases = (
            ("swiglu_rmsnorm", 7, True),
            ("coghp_gelu", -1, True),
            ("gelu_layernorm", -1, False),
        )
        for block_type, mixer_size, affine in cases:
            reference_shapes = None
            for core_type in (
                "matched_feedforward",
                "single_state_iterative",
                "hrm",
            ):
                with self.subTest(
                    core_type=core_type,
                    block_type=block_type,
                ):
                    agent = self.make_agent(
                        block_type=block_type,
                        mixer_intermediate_size=mixer_size,
                        layernorm_affine=affine,
                        core_type=core_type,
                    )
                    flat = {
                        "/".join(path): value
                        for path, value in flax.traverse_util.flatten_dict(
                            agent.network.params
                        ).items()
                        if "action_hrm" in path
                    }
                    self.assertTrue(flat)
                    current_shapes = {path: value.shape for path, value in flat.items()}
                    if reference_shapes is None:
                        reference_shapes = current_shapes
                    else:
                        self.assertEqual(current_shapes, reference_shapes)
                    if core_type == "hrm":
                        self.assertIn("constants", agent.network.model_state)
                    else:
                        self.assertNotIn(
                            "constants",
                            agent.network.model_state,
                        )

                    if block_type == "swiglu_rmsnorm":
                        gate_shapes = {
                            value.shape
                            for path, value in flat.items()
                            if "gate_up_proj/linear/kernel" in path
                        }
                        self.assertIn((5, 14), gate_shapes)
                        self.assertIn((STATE_DIM, 14), gate_shapes)
                    elif block_type == "coghp_gelu":
                        self.assertTrue(
                            any(path.endswith("/tm_weights") for path in flat)
                        )
                        channel_shapes = {
                            value.shape
                            for path, value in flat.items()
                            if "channel_dense1/kernel" in path
                        }
                        self.assertEqual(channel_shapes, {(STATE_DIM, 6)})
                    else:
                        self.assertFalse(any("token_norm/" in path for path in flat))
                        self.assertFalse(any("channel_norm/" in path for path in flat))

    @staticmethod
    def make_batch():
        observations, goals, _, actions = synthetic_inputs()
        return {
            "observations": observations,
            "next_observations": observations + 0.01,
            "actions": actions,
            "rewards": jnp.zeros((BATCH_SIZE,), dtype=jnp.float32),
            "masks": jnp.ones((BATCH_SIZE,), dtype=jnp.float32),
            "value_goals": goals,
            "high_actor_goals": goals,
            "high_actor_targets": jnp.stack(
                [observations + 0.1, observations + 0.2],
                axis=1,
            ),
        }

    def test_train_state_preserves_constants_without_optimizing_them(self):
        agent = self.make_agent()
        self.assertIn("constants", agent.network.model_state)
        old_constants = agent.network.model_state["constants"]
        new_agent, info = agent.update(self.make_batch())
        self.assertIn("low_actor/actor_loss", info)
        self.assertIn("action_hrm/z_high_final_norm", info)
        for old, new in zip(
            jax.tree_util.tree_leaves(old_constants),
            jax.tree_util.tree_leaves(new_agent.network.model_state["constants"]),
        ):
            np.testing.assert_array_equal(old, new)

    def test_validation_metrics_exist_and_are_finite(self):
        agent = self.make_agent()
        batch = self.make_batch()
        info = agent.validation_rollout_info(
            batch,
            rng=jax.random.PRNGKey(8),
        )
        expected = {
            "validation_action/teacher/mse",
            "validation_action/predicted/mse",
            "validation_action/gap/mse",
            "validation_action/teacher/bc_log_prob",
            "validation_action/predicted/bc_log_prob",
        }
        self.assertEqual(set(info), expected)
        for value in info.values():
            self.assertTrue(np.isfinite(np.asarray(value)).all())

    def test_config_contains_only_clean_hrm_controls(self):
        config = get_config()
        self.assertEqual(config.agent_name, "hrm_action_coghp")
        self.assertEqual(config.planner_core_type, "vanilla_coghp")
        self.assertEqual(config.planner_h_cycles, 2)
        self.assertEqual(config.planner_l_cycles, 2)
        self.assertTrue(config.planner_bptt)
        self.assertEqual(config.action_hrm_num_layers, 1)
        self.assertEqual(config.action_hrm_h_cycles, 2)
        self.assertEqual(config.action_hrm_l_cycles, 2)
        self.assertTrue(config.action_hrm_bptt)
        self.assertTrue(config.action_hrm_is_mlp_mixer)
        self.assertEqual(config.action_core_type, "hrm")
        self.assertEqual(config.action_block_type, "auto")
        self.assertEqual(config.hrm_block_type, "legacy")
        self.assertEqual(config.hrm_token_hidden, 32)
        self.assertEqual(config.hrm_channel_hidden, 32)
        self.assertEqual(config.hrm_mixer_intermediate_size, -1)
        self.assertTrue(config.hrm_layernorm_affine)
        self.assertEqual(config.train_action_chain_source, "teacher")
        for legacy_name in (
            "action_hrm_segments",
            "action_hrm_max_segments",
            "action_hrm_residual_alpha",
            "action_hrm_state_init_mode",
            "action_hrm_trainable_init",
            "action_inference_segment",
        ):
            self.assertNotIn(legacy_name, config)


if __name__ == "__main__":
    unittest.main()
