#!/usr/bin/env python3
"""CPU-only, read-only probes of a frozen v0 campaign on paired offline queries.

Geometry is used only to label diagnostic progress. No simulator rollouts,
parameter updates, online candidate selection, or changes to training are made.
"""

import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np


def bfs(grid, goal):
    distances = np.full(grid.shape, -1, dtype=np.int32)
    if not (0 <= goal[0] < grid.shape[0] and 0 <= goal[1] < grid.shape[1]) or grid[goal] != 0:
        return distances
    distances[goal] = 0
    pending = deque([goal])
    while pending:
        row, col = pending.popleft()
        for a, b in ((row-1, col), (row+1, col), (row, col-1), (row, col+1)):
            if 0 <= a < grid.shape[0] and 0 <= b < grid.shape[1] and grid[a, b] == 0 and distances[a, b] < 0:
                distances[a, b] = distances[row, col] + 1
                pending.append((a, b))
    return distances


def effective_rank(array):
    centered = array - array.mean(axis=0)
    eigenvalues = np.linalg.eigvalsh(centered.T @ centered / len(array))
    eigenvalues = np.maximum(eigenvalues, 0)
    if eigenvalues.sum() <= 1e-12:
        return 0.0
    probabilities = eigenvalues / eigenvalues.sum()
    positive = probabilities > 1e-12
    return float(np.exp(-np.sum(probabilities[positive] * np.log(probabilities[positive]))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True,
                        help='Analysis snapshot containing stable run metadata and flags.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=2048)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=20261030)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to overwrite an existing probe output.')
    if args.samples <= 0 or args.samples % args.batch_size:
        raise ValueError('samples must be positive and divisible by batch-size.')
    os.environ.update(JAX_PLATFORMS='cpu', MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', EGL_PLATFORM='surfaceless')
    os.environ.pop('MUJOCO_EGL_DEVICE_ID', None)
    manifest = json.loads((args.snapshot / 'campaign.json').read_text())
    capture = json.loads((args.snapshot / 'captured_at.json').read_text())
    if Path(capture['campaign']).resolve() != args.campaign.resolve():
        raise ValueError('Snapshot belongs to a different campaign.')
    source = Path(manifest['source_dir'])
    sys.path[:0] = [str(source / 'impls'), str(source)]
    import jax
    import jax.numpy as jnp
    import ogbench
    from ogbench.utils import load_dataset
    from agents.navigation import NavigationAgent
    from utils.datasets import Dataset, MultiHGCDataset
    from utils.flax_utils import restore_agent
    import inspect
    assert jax.default_backend() == 'cpu'
    assert Path(inspect.getfile(NavigationAgent)).is_relative_to(source)

    runs = []
    for directory in sorted((args.snapshot / 'runs').iterdir()):
        state = json.loads((directory / 'run.json').read_text())
        if state['status'] == 'completed':
            flags = json.loads((directory / 'flags.json').read_text())
            runs.append((directory.name, directory, state, flags))
    reference = next(row for row in runs if row[3]['agent']['num_subgoals'] == 3)
    flags = reference[3]
    config = {k: tuple(v) if isinstance(v, list) else v for k, v in flags['agent'].items()}
    dataset_path = Path(manifest['datasets'][flags['env_name'] + '-val.npz']['path'])
    data = Dataset.create(**load_dataset(dataset_path, compact_dataset=True))
    query = MultiHGCDataset(data, config, rng=np.random.default_rng(args.seed)).sample(args.samples, evaluation=True)
    # Sample the M=3 queries once. M=1 sees precisely the same nearest state.
    np.savez_compressed(args.output.with_suffix('.queries.npz'),
                        indices=query['indices'], goal_indices=query['high_actor_goal_idxs'],
                        target_indices=query['high_actor_target_idxs'],
                        target_offsets=query['high_actor_target_offsets'])
    env = ogbench.make_env_and_datasets(flags['env_name'], env_only=True)
    base = env.unwrapped
    cache = {}
    progress = np.full((args.samples, 3), np.nan)
    before_distances = np.full(args.samples, np.nan)
    for i in range(args.samples):
        goal = base.xy_to_ij(query['high_actor_goals'][i, :2])
        if goal not in cache:
            cache[goal] = bfs(base.maze_map, goal)
        grid = cache[goal]

        def distance(xy):
            row, col = base.xy_to_ij(xy)
            return grid[row, col] if 0 <= row < grid.shape[0] and 0 <= col < grid.shape[1] else -1

        before = distance(query['observations'][i, :2])
        if before >= 0:
            before_distances[i] = before
        for j in range(3):
            after = distance(query['high_actor_targets'][i, j, :2])
            if before >= 0 and after >= 0:
                progress[i, j] = before - after
    env.close()
    output = dict(protocol='paired-offline-queries-v1', source_signature=manifest['source']['code_signature'],
                  samples=args.samples, query_seed=args.seed, backend='cpu',
                  caveats=[
                      'Geometric cell progress is a diagnostic proxy, not dynamics-optimal distance.',
                      'Action metrics are one-step agreement/sensitivity, not local rollout success.',
                      'One-step action metrics use the Gaussian mean before environment action clipping.',
                      'Latent representations differ across checkpoints; MSE is not physical waypoint error.',
                      'The same M=3 raw-state query pool is used for all models; M=1 selects its nearest target.',
                  ],
                  repeated_far_near_target_fraction=float(np.mean(query['high_actor_target_idxs'][:, 0] == query['high_actor_target_idxs'][:, -1])),
                  runs={})
    for name, directory, state, flags in runs:
        c = {k: tuple(v) if isinstance(v, list) else v for k, v in flags['agent'].items()}
        m = c['num_subgoals']
        if m not in (1, 3):
            raise ValueError('This paired probe supports the current M=1/3 experiment only.')
        model = NavigationAgent.create(flags['seed'], query['observations'][:1], query['actions'][:1], c)
        artifacts = Path(state['run_dir']) / 'artifacts'
        metadata = json.loads((directory / 'checkpoint_metadata.json').read_text())
        output['runs'][name] = {}

        @jax.jit
        def forward(agent, obs, goals, target_states):
            expanded = jnp.broadcast_to(obs[:, None], target_states.shape)
            labels_all = agent.network.select('goal_rep')(jnp.concatenate([expanded, target_states], axis=-1))
            labels = labels_all if m == 3 else labels_all[:, -1:]
            key = jax.random.PRNGKey(19)
            _, predicted_dist, _, predicted = agent.policy(obs, goals, key, temperature=0.0, return_latents=True)

            def action(commands):
                return agent.policy(obs, goals, key, commands=commands, temperature=0.0,
                                    chain_source='teacher')[1].mode()

            teacher_action = action(labels)
            corrected_near = action(predicted.at[:, -1].set(labels[:, -1]))
            corrected_far = action(predicted.at[:, :-1].set(labels[:, :-1])) if m == 3 else predicted_dist.mode()
            # Match input norm when testing corrective-direction sensitivity.
            # This is a directional Jacobian probe, not closed-loop benefit.
            near_direction = jnp.zeros_like(predicted).at[:, -1].set(labels[:, -1] - predicted[:, -1])
            near_direction /= jnp.maximum(jnp.linalg.norm(near_direction, axis=(1, 2), keepdims=True), 1e-12)
            near_jvp = jax.jvp(action, (predicted,), (near_direction,))[1]
            if m == 3:
                far_direction = jnp.zeros_like(predicted).at[:, :-1].set(labels[:, :-1] - predicted[:, :-1])
                far_direction /= jnp.maximum(jnp.linalg.norm(far_direction, axis=(1, 2), keepdims=True), 1e-12)
                far_jvp = jax.jvp(action, (predicted,), (far_direction,))[1]
            else:
                far_jvp = jnp.zeros_like(near_jvp)
            value = agent.network.select('value')
            initial_v = jnp.mean(value(obs, goals), axis=0)
            future_v = jnp.stack([jnp.mean(value(target_states[:, j], goals), axis=0) for j in range(3)], axis=1)
            return labels, predicted, teacher_action, predicted_dist.mode(), corrected_near, corrected_far, future_v - initial_v[:, None], near_jvp, far_jvp

        for step in manifest['study']['metric']['steps']:
            agent = restore_agent(model, str(artifacts), step, metadata, True)
            pieces = []
            for start in range(0, args.samples, args.batch_size):
                sl = slice(start, start + args.batch_size)
                values = forward(agent, query['observations'][sl], query['high_actor_goals'][sl], query['high_actor_targets'][sl])
                pieces.append(tuple(np.asarray(x) for x in values))
            labels, predicted, teacher_a, predicted_a, corrected_near_a, corrected_far_a, adv_all, near_jvp, far_jvp = [
                np.concatenate([piece[j] for piece in pieces]) for j in range(9)]
            adv = adv_all if m == 3 else adv_all[:, -1:]
            selected_progress = progress if m == 3 else progress[:, -1:]
            weight = np.exp(np.minimum(c['high_alpha'] * adv, np.log(c['awr_max_weight'])))
            errors = np.mean((predicted - labels) ** 2, axis=-1)
            per_position = []
            for j in range(m):
                mask = np.isfinite(selected_progress[:, j]) & (selected_progress[:, j] != 0)
                distant = mask & (before_distances >= 10)
                z = labels[:, j].astype(np.float64)
                y = predicted[:, j].astype(np.float64)
                w = weight[:, j].astype(np.float64)
                cosine = np.sum(z*y, axis=-1) / np.maximum(np.linalg.norm(z, axis=-1)*np.linalg.norm(y, axis=-1), 1e-12)
                high = w >= c['awr_max_weight'] * (1-1e-5)
                per_position.append(dict(
                    index_far_to_near=j, nominal_offset=(m-j)*c['subgoal_steps'],
                    mse=float(errors[:, j].mean()), weighted_mse=float(np.sum(w*errors[:, j])/w.sum()),
                    error_p90=float(np.quantile(errors[:, j], .9)), cosine_mean=float(cosine.mean()),
                    target_norm_mean=float(np.linalg.norm(z, axis=-1).mean()),
                    predicted_norm_mean=float(np.linalg.norm(y, axis=-1).mean()),
                    constant_target_mean_mse=float(np.mean((z-z.mean(axis=0))**2)),
                    target_effective_rank=effective_rank(z), predicted_effective_rank=effective_rank(y),
                    weight_mean=float(w.mean()), cap_fraction=float(high.mean()),
                    ess_fraction=float(w.sum()**2 / (len(w)*np.sum(w*w))),
                    geometric_non_tie_pairs=int(mask.sum()),
                    geometric_sign_agreement=float(np.mean(np.sign(adv[mask,j]) == np.sign(selected_progress[mask,j]))) if mask.any() else None,
                    distant_geometric_sign_agreement=float(np.mean(np.sign(adv[distant,j]) == np.sign(selected_progress[distant,j]))) if distant.any() else None,
                    geometrically_backward_capped_fraction=float(np.mean(selected_progress[mask & high,j] < 0)) if (mask & high).any() else None,
                ))
            action_mse = lambda a: float(np.mean((a-query['actions'])**2))
            with (artifacts / f'params_{step}.pkl').open('rb') as stream:
                checkpoint_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
            result=dict(checkpoint_sha256=checkpoint_sha256, waypoints=per_position, actions=dict(
                teacher_data_mse=action_mse(teacher_a), predicted_data_mse=action_mse(predicted_a),
                corrected_near_data_mse=action_mse(corrected_near_a), corrected_far_data_mse=action_mse(corrected_far_a),
                predicted_teacher_action_distance=float(np.mean((predicted_a-teacher_a)**2)),
                near_replacement_action_distance=float(np.mean((predicted_a-corrected_near_a)**2)),
                far_replacement_action_distance=float(np.mean((predicted_a-corrected_far_a)**2)),
                near_unit_direction_action_jvp_mse=float(np.mean(near_jvp**2)),
                far_unit_direction_action_jvp_mse=float(np.mean(far_jvp**2)),
            ))
            output['runs'][name][str(step)] = result
            print(json.dumps(dict(method=name, step=step, near_mse=per_position[-1]['mse'], actions=result['actions'])), flush=True)
            args.output.write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    print('Saved checkpoint probes:', args.output, flush=True)


if __name__ == '__main__':
    main()
