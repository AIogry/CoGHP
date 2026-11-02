#!/usr/bin/env python3
"""Record a paired subset of official task trajectories for diagnosis only."""

import argparse
from collections import deque
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys

import numpy as np


def bfs(grid, goal):
    distances = np.full(grid.shape, -1, dtype=np.int32)
    distances[goal] = 0
    pending = deque([goal])
    while pending:
        i, j = pending.popleft()
        for a, b in ((i-1,j), (i+1,j), (i,j-1), (i,j+1)):
            if 0 <= a < grid.shape[0] and 0 <= b < grid.shape[1] and grid[a,b] == 0 and distances[a,b] < 0:
                distances[a,b] = distances[i,j] + 1
                pending.append((a,b))
    return distances


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=int, default=1000000)
    parser.add_argument('--episodes', type=int, default=8)
    parser.add_argument('--tasks', type=int, nargs='+', default=[1,5])
    parser.add_argument('--methods', nargs='+', default=[
        'hrm_h1l1_m1', 'hrm_h2l1_m1', 'hrm_h4l1_m1',
        'hrm_h4l1_m3', 'hrm_h4l2_m1', 'hrm_h4l2_m3'])
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to overwrite an existing path diagnostic.')
    os.environ.update(JAX_PLATFORMS='cpu', MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', EGL_PLATFORM='surfaceless')
    os.environ.pop('MUJOCO_EGL_DEVICE_ID', None)
    manifest = json.loads((args.snapshot / 'campaign.json').read_text())
    source = Path(manifest['source_dir'])
    sys.path[:0] = [str(source / 'impls'), str(source)]
    import jax
    import ogbench
    from agents.navigation import NavigationAgent
    from utils.flax_utils import restore_agent
    from utils.evaluation import derive_seed, supply_rng
    assert jax.default_backend() == 'cpu'
    assert Path(inspect.getfile(NavigationAgent)).is_relative_to(source)
    result = dict(protocol='official-evaluation-fixed-prefix-path-diagnostic-v1',
                  checkpoint=args.checkpoint, source_signature=manifest['source']['code_signature'],
                  episode_indices=list(range(args.episodes)), tasks=args.tasks, methods=args.methods,
                  backend='cpu', complete=False, runs={},
                  caveats=[
                      'A post-hoc diagnostic subset: do not replace the preregistered 800k/900k/1M score.',
                      'The same first evaluation episode seeds and native termination rule are used.',
                      'The original campaign also requested CPU evaluation. Exact fresh-process trajectory reproduction is not assumed without verification.',
                      'Maze geometry is recorded for visualization/distance diagnostics only; it is not an actor input.',
                      'Cell distance ignores within-cell offsets, orientation, and dynamics.',
                  ])
    for name in args.methods:
        directory = args.snapshot / 'runs' / name
        state = json.loads((directory / 'run.json').read_text())
        flags = json.loads((directory / 'flags.json').read_text())
        result.setdefault('original_eval_on_cpu', {})[name] = flags['eval_on_cpu']
        repro = json.loads((directory / 'reproducibility.json').read_text())
        config = {k: tuple(v) if isinstance(v, list) else v for k,v in flags['agent'].items()}
        metadata = json.loads((directory / 'checkpoint_metadata.json').read_text())
        env = ogbench.make_env_and_datasets(flags['env_name'], env_only=True)
        base = env.unwrapped
        obs, _ = env.reset(seed=0)
        agent = NavigationAgent.create(flags['seed'], obs[None].astype(np.float32), np.zeros((1,*env.action_space.shape), dtype=np.float32), config)
        artifacts = Path(state['run_dir']) / 'artifacts'
        agent = restore_agent(agent, str(artifacts), args.checkpoint, metadata, True)
        with (artifacts / f'params_{args.checkpoint}.pkl').open('rb') as stream:
            checkpoint_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
        result['runs'][name] = dict(checkpoint_sha256=checkpoint_hash, tasks={})
        if 'maze' not in result:
            rows, cols = base.maze_map.shape
            result['maze'] = dict(grid=base.maze_map.tolist(),
                                 cell_xy=[[list(base.ij_to_xy((i,j))) for j in range(cols)] for i in range(rows)],
                                 tasks=[{k: (v.tolist() if isinstance(v,np.ndarray) else v) for k,v in t.items()} for t in base.task_infos])
        for task_id in args.tasks:
            records = []
            task_seed = derive_seed(repro['eval_seed'], task_id)
            for i in range(args.episodes):
                actor = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(derive_seed(task_seed,i,1)))
                obs, info = env.reset(seed=derive_seed(task_seed,i,0), options=dict(task_id=task_id, render_goal=False))
                goal = info['goal']
                distances = bfs(base.maze_map, base.xy_to_ij(goal[:2]))
                path = [obs[:2].tolist()]
                cell_path = [base.xy_to_ij(obs[:2])]
                distances_to_goal = [int(distances[cell_path[-1]])]
                done = False
                while not done:
                    action = np.clip(np.asarray(actor(observations=obs, goals=goal, temperature=0.0)), -1, 1)
                    obs, _, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    path.append(obs[:2].tolist())
                    cell = base.xy_to_ij(obs[:2])
                    cell_path.append(cell)
                    distances_to_goal.append(int(distances[cell]))
                tail = np.array(path[-201:])
                records.append(dict(
                    episode=i, episode_seed=derive_seed(task_seed,i,0), success=bool(info['success']),
                    steps=len(path)-1, goal_xy=goal[:2].tolist(), path_xy=path,
                    initial_distance_cells=distances_to_goal[0], best_distance_cells=min(distances_to_goal),
                    final_distance_cells=distances_to_goal[-1], unique_cells=len(set(cell_path)),
                    last_200_steps_span_xy=np.ptp(tail,axis=0).tolist(),
                    last_200_steps_path_length=float(np.linalg.norm(np.diff(tail,axis=0),axis=1).sum()),
                    final_distance_xy=float(np.linalg.norm(obs[:2]-goal[:2])),
                ))
            summary=dict(success_count=sum(r['success'] for r in records), count=len(records),
                         mean_best_distance_cells=float(np.mean([r['best_distance_cells'] for r in records])),
                         mean_final_distance_cells=float(np.mean([r['final_distance_cells'] for r in records])))
            result['runs'][name]['tasks'][str(task_id)] = dict(summary=summary, episodes=records)
            args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
            print(json.dumps(dict(method=name, task=task_id, **summary)), flush=True)
        env.close()
        jax.clear_caches()
    result['complete'] = True
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('Saved task paths:', args.output, flush=True)


if __name__ == '__main__':
    main()
