#!/usr/bin/env python3
"""Resume a completed smoke run in a NEW directory and compare full checkpoints.

Uses the original immutable source, command, GPU protocol and sampler state.
Never modifies the original run. Intended for short preflight runs, not 1M jobs.
"""
import argparse
import json
import os
from pathlib import Path
import pickle
import subprocess

import numpy as np

from run_study import check_gpu_occupancy, child_environment, write_json


def compare_tree(a, b, prefix=''):
    if isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b):
            raise AssertionError(f'Checkpoint keys differ at {prefix}')
        return sum(compare_tree(a[k], b[k], f'{prefix}/{k}') for k in a)
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            raise AssertionError(f'Checkpoint sequence differs at {prefix}')
        return sum(compare_tree(x, y, f'{prefix}/{i}') for i, (x, y) in enumerate(zip(a, b)))
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=prefix)
    return 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--from-step', type=int, default=1000)
    p.add_argument('--to-step', type=int, default=2000)
    p.add_argument('--gpu', default='1')
    args = p.parse_args()
    original = json.loads((args.run / 'run.json').read_text())
    if original['status'] != 'completed':
        raise ValueError('Reference run must be completed.')
    if args.output.exists():
        raise ValueError('Resume verification output must be a new directory.')
    artifacts = args.run / 'artifacts'
    if not (artifacts / f'params_{args.to_step}.pkl').exists():
        raise ValueError('Reference checkpoint is absent.')
    check_gpu_occupancy([args.gpu])
    args.output.mkdir(parents=True)
    replace = dict(run_dir=str(args.output / 'artifacts'), train_steps=str(args.to_step),
                   restore_path=str(artifacts), restore_epoch=str(args.from_step))
    cmd = [part for part in original['command']
           if not any(part.startswith(f'--{key}=') for key in replace)]
    cmd += [f'--{key}={value}' for key, value in replace.items()]
    source = Path(original['source_dir'])
    env = child_environment(source, Path(original['runtime_environment']['OGBENCH_DATASET_DIR']),
                            args.gpu, original['seed'])
    # Only local, trusted checkpoints are read; comparison requires no GPU backend.
    os.environ['JAX_PLATFORMS'] = 'cpu'
    write_json(args.output / 'command.json', dict(command=cmd, source=str(source),
                                                source_signature=original['source_signature']))
    with (args.output / 'stdout.log').open('x') as log:
        subprocess.run(cmd, cwd=source / 'impls', env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    with (artifacts / f'params_{args.to_step}.pkl').open('rb') as f:
        expected = pickle.load(f)['agent']
    with (args.output / 'artifacts' / f'params_{args.to_step}.pkl').open('rb') as f:
        actual = pickle.load(f)['agent']
    leaves = compare_tree(expected, actual)
    sampler_name = f'training_state_{args.to_step}.json'
    assert json.loads((artifacts / sampler_name).read_text()) == json.loads(
        (args.output / 'artifacts' / sampler_name).read_text())
    result = dict(passed=True, from_step=args.from_step, to_step=args.to_step,
                  exact_agent_leaves=leaves, sampler_state_exact=True,
                  reference=str(artifacts), resumed=str(args.output / 'artifacts'))
    write_json(args.output / 'verification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
