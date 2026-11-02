#!/usr/bin/env python3
"""Paired local full-state command interventions on frozen planner checkpoints.

This diagnostic does not train, modify the campaign, or replace its official
evaluation. Every model starts at the same saved validation simulator states.
Data waypoints stay fixed in physical state space and are re-encoded from the
current observation on every step. The native planner replans on every step.
"""

import argparse
from functools import partial
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys

import numpy as np


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=32)
    parser.add_argument('--budget', type=int, default=100)
    parser.add_argument('--checkpoint', type=int, default=1000000)
    parser.add_argument('--seed', type=int, default=20261031)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to overwrite a previous diagnostic.')
    if args.episodes <= 0 or args.budget < 50:
        raise ValueError('Need positive episodes and a budget of at least 50.')
    os.environ.update(JAX_PLATFORMS='cpu', MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', EGL_PLATFORM='surfaceless')
    os.environ.pop('MUJOCO_EGL_DEVICE_ID', None)
    manifest = json.loads((args.snapshot / 'campaign.json').read_text())
    capture = json.loads((args.snapshot / 'captured_at.json').read_text())
    if Path(capture['campaign']).resolve() != args.campaign.resolve():
        raise ValueError('Snapshot/campaign mismatch.')
    source = Path(manifest['source_dir'])
    for relative, expected in manifest['source']['files'].items():
        if sha256(source / relative) != expected:
            raise ValueError(f'Frozen source mismatch: {relative}')
    sys.path[:0] = [str(source / 'impls'), str(source)]
    import jax
    import jax.numpy as jnp
    import mujoco
    import ogbench
    from ogbench.utils import load_dataset
    from agents.navigation import NavigationAgent
    from utils.flax_utils import restore_agent
    assert jax.default_backend() == 'cpu'
    assert Path(inspect.getfile(NavigationAgent)).is_relative_to(source)
    runs = []
    for directory in sorted((args.snapshot / 'runs').iterdir()):
        state = json.loads((directory / 'run.json').read_text())
        if state['status'] == 'completed':
            flags = json.loads((directory / 'flags.json').read_text())
            runs.append((directory.name, directory, state, flags))
    reference = runs[0][3]
    dataset_path = Path(manifest['datasets'][reference['env_name'] + '-val.npz']['path'])
    expected_hash = manifest['datasets'][reference['env_name'] + '-val.npz']['sha256']
    if sha256(dataset_path) != expected_hash:
        raise ValueError('Validation dataset hash mismatch.')
    data = load_dataset(dataset_path, compact_dataset=True, add_info=True)
    if not {'qpos', 'qvel'} <= data.keys():
        raise ValueError('Simulator state missing from validation data.')
    env = ogbench.make_env_and_datasets(reference['env_name'], env_only=True)
    base = env.unwrapped
    tolerance = float(base._goal_tol)
    delta = int(reference['agent']['subgoal_steps'])
    ends = np.flatnonzero(data['valids'] == 0)
    indices = np.flatnonzero(data['valids'] > 0)
    terminal = ends[np.searchsorted(ends, indices)]
    indices = indices[indices + 4 * delta <= terminal]
    np.random.default_rng(args.seed).shuffle(indices)
    chosen = []
    for t in indices:
        if np.linalg.norm(data['observations'][t + delta, :2] - data['observations'][t, :2]) > 2 * tolerance:
            chosen.append(int(t))
        if len(chosen) == args.episodes:
            break
    if len(chosen) != args.episodes:
        raise ValueError('Not enough nontrivial validation segments.')
    offsets = np.array([3, 2, 1]) * delta
    np.savez_compressed(args.output.with_suffix('.queries.npz'),
                        indices=chosen, target_indices=np.array(chosen)[:, None] + offsets,
                        goal_indices=np.array(chosen) + 4 * delta)
    result = dict(
        protocol='full-state-paired-local-command-interventions-v1', complete=False,
        campaign=str(args.campaign), source_signature=manifest['source']['code_signature'],
        checkpoint_step=args.checkpoint, dataset=str(dataset_path), dataset_sha256=expected_hash,
        seed=args.seed, indices=chosen, episodes_per_condition=args.episodes,
        budget=args.budget, tolerance=tolerance, nominal_waypoint_steps=delta,
        final_goal_offset=4 * delta, backend='cpu',
        caveats=[
            'Success means visiting the data nearest waypoint in XY; it does not measure full-state matching.',
            'These local data-start tasks are not the five official evaluation tasks or their success rates.',
            'Alternate valid routes may miss the reference waypoint: teacher commands are not a universal upper bound.',
            'Teacher targets are fixed raw states, re-encoded against the current observation every step.',
            'Mixed predicted/teacher chains can be inconsistent; a replacement effect is not a clean isolated causal effect.',
            'Only the 1M checkpoint is used; 32 paired starts are diagnostic cases, not independent training seeds.',
        ], runs={},
    )
    for name, directory, state, flags in runs:
        config = {k: tuple(v) if isinstance(v, list) else v for k, v in flags['agent'].items()}
        m = config['num_subgoals']
        assert config['command_content'] == 'full_state' and m in (1, 3)
        assert config['subgoal_steps'] == delta and flags['env_name'] == reference['env_name']
        metadata = json.loads((directory / 'checkpoint_metadata.json').read_text())
        artifacts = Path(state['run_dir']) / 'artifacts'
        agent = NavigationAgent.create(flags['seed'], data['observations'][:1], data['actions'][:1], config)
        agent = restore_agent(agent, str(artifacts), args.checkpoint, metadata, True)

        @partial(jax.jit, static_argnames=('mode',))
        def action(agent, observation, goal, raw_targets, mode):
            obs, goals = observation[None], goal[None]
            key = jax.random.PRNGKey(0)
            if mode == 'predicted':
                return agent.sample_actions(observation, goal, key, 0.0)
            states = raw_targets if m == 3 else raw_targets[-1:]
            repeated = jnp.broadcast_to(observation, states.shape)
            labels = agent.network.select('goal_rep')(jnp.concatenate([repeated, states], axis=-1))[None]
            if mode == 'teacher':
                commands = labels
            else:
                _, _, _, predicted = agent.policy(obs, goals, key, temperature=0.0, return_latents=True)
                commands = (predicted.at[:, -1].set(labels[:, -1]) if mode == 'corrected_near'
                            else predicted.at[:, :-1].set(labels[:, :-1]))
            dist = agent.policy(obs, goals, key, commands=commands, temperature=0.0, chain_source='teacher')[1]
            return jnp.clip(dist.mode()[0], -1, 1)

        record = dict(checkpoint_sha256=sha256(artifacts / f'params_{args.checkpoint}.pkl'),
                      summary={}, episodes=[])
        result['runs'][name] = record
        modes = ('predicted', 'teacher') if m == 1 else ('predicted', 'teacher', 'corrected_near', 'corrected_far')
        for mode in modes:
            for t in chosen:
                raw_targets = data['observations'][t + offsets]
                goal = data['observations'][t + 4 * delta]
                target = raw_targets[-1, :2]
                env.reset(seed=args.seed + t)
                base.set_state(data['qpos'][t].copy(), data['qvel'][t].copy())
                mujoco.mj_forward(base.model, base.data)
                obs = base.get_ob()
                error = float(np.max(np.abs(obs - data['observations'][t])))
                if not np.allclose(obs, data['observations'][t], atol=2e-4, rtol=2e-4):
                    raise ValueError(f'Simulator state restoration failed: {t}: {error}')
                base.set_goal(goal_xy=target)
                initial_distance = float(np.linalg.norm(obs[:2] - target))
                minimum_distance = initial_distance
                path = [obs[:2].tolist()]
                saturated = []
                success = False
                for step in range(args.budget):
                    a = np.asarray(action(agent, obs.astype(np.float32), goal, raw_targets, mode))
                    if not np.isfinite(a).all():
                        raise ValueError(f'Nonfinite action: {name}: {t}: {step}')
                    saturated.append(float(np.mean(np.abs(a) >= .999)))
                    obs, _, terminated, truncated, _ = env.step(a)
                    path.append(obs[:2].tolist())
                    distance = float(np.linalg.norm(obs[:2] - target))
                    minimum_distance = min(minimum_distance, distance)
                    success = distance <= tolerance
                    if success or terminated or truncated:
                        break
                record['episodes'].append(dict(
                    index=t, mode=mode, success=bool(success), success_at_50=bool(success and step < 50),
                    steps=step + 1, terminated=bool(terminated), truncated=bool(truncated),
                    restore_observation_max_error=error, initial_distance=initial_distance,
                    final_distance=distance, minimum_distance=minimum_distance,
                    action_saturation_fraction=float(np.mean(saturated)),
                    reference_waypoints=raw_targets[:, :2].tolist(), final_goal_xy=goal[:2].tolist(), path_xy=path,
                ))
            selected = [row for row in record['episodes'] if row['mode'] == mode]
            record['summary'][mode] = dict(
                count=len(selected), success_count=sum(r['success'] for r in selected),
                success_rate=float(np.mean([r['success'] for r in selected])),
                success_at_50=float(np.mean([r['success_at_50'] for r in selected])),
                mean_final_distance=float(np.mean([r['final_distance'] for r in selected])),
                mean_minimum_distance=float(np.mean([r['minimum_distance'] for r in selected])),
                mean_steps=float(np.mean([r['steps'] for r in selected])),
                action_saturation_fraction=float(np.mean([r['action_saturation_fraction'] for r in selected])),
            )
            args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
            print(json.dumps(dict(method=name, mode=mode, **record['summary'][mode])), flush=True)
        jax.clear_caches()
    result['complete'] = True
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    env.close()
    print('Saved local rollout probes:', args.output, flush=True)


if __name__ == '__main__':
    main()
