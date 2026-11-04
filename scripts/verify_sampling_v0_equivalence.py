#!/usr/bin/env python3
"""Compare archived v0 and current source in isolated CPU Python processes.

Uses actual archived configuration/checkpoint and one unchanged held-out sampler
batch. No training artifacts are overwritten; one diagnostic update is discarded.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

DEFAULT_CAMPAIGN = Path('/data/qijunrong/06-RL/offline-rl/exp/joint_planner_v0_antmaze_giant_width32_seed0_20261029')


def worker(args):
    os.environ.update(JAX_PLATFORMS='cpu', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(args.source / 'impls'), str(args.source)]
    import jax
    import numpy as np
    from agents.navigation import NavigationAgent
    from ogbench.utils import load_dataset
    from utils.datasets import Dataset, MultiHGCDataset
    from utils.flax_utils import build_checkpoint_metadata, restore_agent, validate_checkpoint_metadata

    artifacts = args.campaign / 'runs/hrm_h4l2_m3/antmaze-giant-navigate-v0/seed_000/attempt_000/artifacts'
    flags = json.loads((artifacts / 'flags.json').read_text())
    metadata = json.loads((artifacts / 'checkpoint_metadata.json').read_text())
    config = {k: tuple(v) if isinstance(v, list) else v for k, v in flags['agent'].items()}
    signature = metadata['signature']
    rebuilt = build_checkpoint_metadata(config, environment=signature['environment'], seed=signature['seed'],
                                        git_commit=signature['git_commit'], code_signature=signature.get('code_signature'))
    validate_checkpoint_metadata(metadata, rebuilt, require_metadata=True)
    manifest = json.loads((args.campaign / 'campaign.json').read_text())
    dataset_path = manifest['datasets']['antmaze-giant-navigate-v0-val.npz']['path']
    sampler = MultiHGCDataset(Dataset.create(**load_dataset(dataset_path, compact_dataset=True)), config,
                              rng=np.random.default_rng(2026110407))
    batch = sampler.sample(args.batch_size, evaluation=True)
    agent = NavigationAgent.create(flags['seed'], batch['observations'][:1], batch['actions'][:1], config)
    payload = {}

    def record_tree(prefix, tree):
        for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
            payload[prefix + '/' + jax.tree_util.keystr(path)] = np.asarray(leaf)

    record_tree('initial_params', agent.network.params)
    record_tree('initial_constants', agent.network.model_state)
    record_tree('batch', batch)
    agent = restore_agent(agent, str(artifacts), 1000000, rebuilt, True)
    record_tree('checkpoint_params', agent.network.params)
    losses, _ = agent.loss_components(batch, agent.network.params, jax.random.PRNGKey(814))
    record_tree('losses', losses)
    updated, info = agent.update(batch)
    jax.block_until_ready(updated.network.params)
    record_tree('updated_params', updated.network.params)
    record_tree('updated_optimizer', updated.network.opt_state)
    record_tree('updated_rng', updated.rng)
    record_tree('updated_step', updated.network.step)
    record_tree('update_info', info)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez(args.output / 'arrays.npz', **payload)
    (args.output / 'metadata.json').write_text(json.dumps(dict(source=str(args.source),
        checkpoint_metadata_compatible=True, leaf_count=len(payload), batch_size=args.batch_size,
        backend=jax.default_backend(), config=config), indent=2) + '\n')
    print(json.dumps(dict(source=str(args.source), arrays=len(payload), done=True)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument('--current-source', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--source', type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    for name, source in [('frozen', args.campaign / 'source'), ('current', args.current_source)]:
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--source', str(source),
                   '--campaign', str(args.campaign), '--output', str(args.output / name),
                   '--batch-size', str(args.batch_size)]
        with (args.output / (name + '.log')).open('w') as stream:
            subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT,
                           env={**os.environ, 'JAX_PLATFORMS': 'cpu', 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'})
    import numpy as np
    summary = {'separate_processes': True, 'all_exact': True, 'nonexact': [], 'category_leaf_counts': {}}
    with np.load(args.output / 'frozen/arrays.npz') as old, np.load(args.output / 'current/arrays.npz') as new:
        if set(old.files) != set(new.files):
            raise AssertionError({'missing_current': sorted(set(old.files)-set(new.files)),
                                  'extra_current': sorted(set(new.files)-set(old.files))})
        for key in old.files:
            category = key.split('/')[0]
            summary['category_leaf_counts'][category] = summary['category_leaf_counts'].get(category, 0) + 1
            if not np.array_equal(old[key], new[key]):
                summary['all_exact'] = False
                summary['nonexact'].append({'key':key, 'max_abs_error':float(np.max(np.abs(old[key]-new[key])))})
    (args.output / 'result.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    if not summary['all_exact']:
        raise SystemExit('Frozen v0/current equivalence is not exact; inspect result.json.')


if __name__ == '__main__':
    main()
