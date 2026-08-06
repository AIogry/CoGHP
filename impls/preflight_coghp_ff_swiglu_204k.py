"""Fail-fast preflight for the CoGHP FF SwiGLU-RMSNorm 204k action core."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import flax
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np

from agents.hrm_action_coghp import HRMActionCoGHPAgent, get_config
from utils.datasets import Dataset, GCDataset, HGCDataset, MultiHGCDataset
from utils.env_utils import make_env_and_datasets
from utils.evaluation import derive_seed
from utils.flax_utils import (
    build_checkpoint_metadata,
    restore_agent,
    save_agent,
)


EXPECTED_ACTION_CORE_PARAMS = 204288
EXPECTED_LEVEL_PARAMS = 102144


def make_experiment_config(env_name: str):
    config = get_config()
    config.action_core_type = 'matched_feedforward'
    config.action_block_type = 'swiglu_rmsnorm'
    config.hrm_block_type = 'swiglu_rmsnorm'
    config.hrm_mixer_intermediate_size = 256
    config.action_hrm_mlp_mixer_intermediate_size = 256
    config.action_hrm_num_layers = 1
    # Retain the noon truncated-HRM flags for provenance.  The matched core
    # never reads these fields; checkpoint metadata records them as N/A.
    config.action_hrm_h_cycles = 2
    config.action_hrm_l_cycles = 2
    config.action_hrm_bptt = False
    config.expected_action_core_params = EXPECTED_ACTION_CORE_PARAMS
    config.feature_dim = 128
    config.num_subgoals = 2
    config.batch_size = 256
    config.train_action_chain_source = 'teacher'
    config.action_use_full_subgoal_chain = True
    if env_name == 'antmaze-giant-navigate-v0':
        config.subgoal_steps = 50
    elif env_name == 'humanoidmaze-giant-navigate-v0':
        config.subgoal_steps = 100
        config.discount = 0.995
    else:
        raise ValueError(f'Unsupported environment: {env_name}')
    return config


def create_agent_and_batch(
    env_name: str,
    seed: int,
    *,
    config=None,
    sample_batch_size: int = 2,
):
    if config is None:
        config = make_experiment_config(env_name)
    environment_seed = derive_seed(seed, 3)
    train_dataset_seed = derive_seed(seed, 1)
    env, raw_train_dataset, _ = make_env_and_datasets(
        env_name,
        frame_stack=config['frame_stack'],
        seed=environment_seed,
    )
    dataset_class = {
        'GCDataset': GCDataset,
        'HGCDataset': HGCDataset,
        'MultiHGCDataset': MultiHGCDataset,
    }[config['dataset_class']]
    train_dataset = dataset_class(
        Dataset.create(seed=train_dataset_seed, **raw_train_dataset),
        config,
        rng=np.random.default_rng(train_dataset_seed),
    )
    example_batch = train_dataset.dataset.get_subset(
        np.array([0], dtype=np.int64)
    )
    agent = HRMActionCoGHPAgent.create(
        seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )
    batch = train_dataset.sample(sample_batch_size)
    batch = jax.tree_util.tree_map(jnp.asarray, batch)
    return env, train_dataset, agent, batch


def count_params(tree) -> int:
    return sum(int(leaf.size) for leaf in jax.tree_util.tree_leaves(tree))


def action_core_flat_params(agent):
    return {
        path: value
        for path, value in flax.traverse_util.flatten_dict(
            agent.network.params
        ).items()
        if path[:2] == ('modules_actor_mixer', 'action_hrm')
    }


def action_branch_param_count(agent) -> int:
    branch_names = {'action_query_token', 'action_hrm', 'low_actor_head'}
    return sum(
        int(value.size)
        for path, value in flax.traverse_util.flatten_dict(
            agent.network.params
        ).items()
        if len(path) >= 2
        and path[0] == 'modules_actor_mixer'
        and path[1] in branch_names
    )


def teacher_subgoals(agent, batch):
    observations = batch['observations']
    observation_sequence = jnp.repeat(
        observations[:, None, :],
        agent.config['num_subgoals'],
        axis=1,
    )
    return agent.network.select('goal_rep')(
        jnp.concatenate(
            [observation_sequence, batch['high_actor_targets']],
            axis=-1,
        )
    )


def measure_action_latency_ms(agent, observation, goal, repeats=100):
    key = jax.random.PRNGKey(991)

    @jax.jit
    def infer(obs, target):
        return agent.sample_actions(
            obs,
            goals=target,
            seed=key,
            temperature=0.0,
        )

    for _ in range(10):
        infer(observation, goal).block_until_ready()
    start = time.perf_counter()
    for _ in range(repeats):
        infer(observation, goal).block_until_ready()
    return (time.perf_counter() - start) * 1000.0 / repeats


def run_preflight(args):
    config = make_experiment_config(args.env_name)
    _, _, agent, batch = create_agent_and_batch(
        args.env_name,
        args.seed,
        config=config,
    )
    flat_core = action_core_flat_params(agent)
    low_params = sum(
        int(value.size)
        for path, value in flat_core.items()
        if path[2] == 'L_level'
    )
    high_params = sum(
        int(value.size)
        for path, value in flat_core.items()
        if path[2] == 'H_level'
    )
    core_params = low_params + high_params
    assert low_params == EXPECTED_LEVEL_PARAMS, low_params
    assert high_params == EXPECTED_LEVEL_PARAMS, high_params
    assert core_params == EXPECTED_ACTION_CORE_PARAMS, core_params

    all_variable_paths = [
        '/'.join(path)
        for collection in (
            flax.traverse_util.flatten_dict(agent.network.params),
            flax.traverse_util.flatten_dict(agent.network.model_state),
        )
        for path in collection
    ]
    has_zl = any(
        marker in path for path in all_variable_paths for marker in ('zL', 'z_L')
    )
    has_zh = any(
        marker in path for path in all_variable_paths for marker in ('zH', 'z_H')
    )
    assert not has_zl
    assert not has_zh
    assert not agent.network.model_state

    subgoals = teacher_subgoals(agent, batch)

    def actor_forward(observations, goals, teacher_chain):
        _, action_dist, actions, diagnostics = agent.network.select(
            'actor_mixer'
        )(
            observations,
            goals,
            jax.random.PRNGKey(992),
            subgoal_reps=teacher_chain,
            temperature=0.0,
            return_diagnostics=True,
        )
        return action_dist.mode(), actions, diagnostics[
            'action_core/total_block_calls'
        ]

    eager_mode, eager_actions, eager_calls = actor_forward(
        batch['observations'],
        batch['high_actor_goals'],
        subgoals,
    )
    assert eager_mode.shape == batch['actions'].shape
    assert eager_actions.shape == batch['actions'].shape
    assert int(eager_calls) == 2
    compiled_mode, compiled_actions, compiled_calls = jax.jit(actor_forward)(
        batch['observations'],
        batch['high_actor_goals'],
        subgoals,
    )
    # GPU XLA may reassociate the Mixer reductions between eager and JIT.
    # The tolerance is still far below the action scale, while avoiding a
    # false failure from expected float32 reduction-order noise.
    np.testing.assert_allclose(
        compiled_mode,
        eager_mode,
        rtol=2e-3,
        atol=5e-5,
    )
    np.testing.assert_allclose(
        compiled_actions,
        eager_actions,
        rtol=2e-3,
        atol=5e-5,
    )
    assert int(compiled_calls) == 2

    loss_key = jax.random.PRNGKey(993)

    def loss_fn(params):
        return agent.total_loss(batch, params, rng=loss_key)[0]

    grads = jax.grad(loss_fn)(agent.network.params)
    core_grads = {
        path: value
        for path, value in flax.traverse_util.flatten_dict(grads).items()
        if path[:2] == ('modules_actor_mixer', 'action_hrm')
    }
    assert core_grads
    assert all(bool(jnp.all(jnp.isfinite(value))) for value in core_grads.values())
    assert sum(float(jnp.linalg.norm(value)) for value in core_grads.values()) > 0

    updated_agent, update_info = agent.update(batch)
    assert int(updated_agent.network.step) == int(agent.network.step) + 1
    assert all(
        bool(jnp.all(jnp.isfinite(value)))
        for value in jax.tree_util.tree_leaves(update_info)
    )

    metadata = build_checkpoint_metadata(
        config,
        environment=args.env_name,
        seed=args.seed,
        git_commit=args.git_commit,
        code_signature=args.code_signature,
        resume=False,
        restore_path=None,
    )
    assert metadata['signature']['h_cycles'] == 'N/A'
    assert metadata['signature']['l_cycles'] == 'N/A'
    assert metadata['signature']['gradient_mode'] == 'N/A'
    with tempfile.TemporaryDirectory() as checkpoint_dir:
        save_agent(
            updated_agent,
            checkpoint_dir,
            1,
            checkpoint_metadata=metadata,
        )
        restored_agent = restore_agent(
            agent,
            checkpoint_dir,
            1,
            expected_checkpoint_metadata=metadata,
            require_checkpoint_metadata=True,
        )
        for expected, actual in zip(
            jax.tree_util.tree_leaves(updated_agent.network.params),
            jax.tree_util.tree_leaves(restored_agent.network.params),
        ):
            np.testing.assert_array_equal(expected, actual)
        mismatched = json.loads(json.dumps(metadata))
        mismatched['signature']['action_core_type'] = 'hrm'
        try:
            restore_agent(
                agent,
                checkpoint_dir,
                1,
                expected_checkpoint_metadata=mismatched,
                require_checkpoint_metadata=True,
            )
        except ValueError as error:
            assert 'signature mismatch' in str(error)
        else:
            raise AssertionError('Mismatched checkpoint signature was accepted.')

    latency_ms = measure_action_latency_ms(
        agent,
        batch['observations'][0],
        batch['high_actor_goals'][0],
    )
    tree_summary = [
        {
            'path': '/'.join(path[2:]),
            'shape': list(value.shape),
            'params': int(value.size),
        }
        for path, value in sorted(flat_core.items())
    ]
    report = {
        'status': 'passed',
        'environment': args.env_name,
        'seed': args.seed,
        'device': str(jax.devices()[0]),
        'action_core_type': 'matched_feedforward',
        'action_block_type': 'swiglu_rmsnorm',
        'mixer_intermediate_size': 256,
        'num_layers': 1,
        'feature_dim': 128,
        'num_subgoals': 2,
        'sequence_length': 5,
        'B_L_params': low_params,
        'B_H_params': high_params,
        'action_core_trainable_params': core_params,
        'action_branch_total_params': action_branch_param_count(agent),
        'whole_agent_params': count_params(agent.network.params),
        'total_block_calls': int(eager_calls),
        'has_zL': has_zl,
        'has_zH': has_zh,
        'has_model_state': bool(agent.network.model_state),
        'effective_h_cycles': 'N/A',
        'effective_l_cycles': 'N/A',
        'effective_gradient_mode': 'N/A',
        'random_init_action_inference_latency_ms': latency_ms,
        'checkpoint_metadata': metadata,
        'parameter_tree': tree_summary,
        'tests': {
            'forward': 'passed',
            'jax_jit': 'passed',
            'finite_gradient': 'passed',
            'one_step_update': 'passed',
            'checkpoint_save_restore': 'passed',
            'checkpoint_mismatch_rejected': 'passed',
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding='utf-8',
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--git-commit', required=True)
    parser.add_argument('--code-signature', required=True)
    parser.add_argument('--output', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    run_preflight(parse_args())
