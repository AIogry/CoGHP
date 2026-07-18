#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import flax
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
IMPLS_DIR = os.path.join(PROJECT_ROOT, 'impls')
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if IMPLS_DIR not in sys.path:
    sys.path.insert(0, IMPLS_DIR)

from agents import agents  # noqa: E402
from utils.datasets import Dataset, GCDataset, HGCDataset, MultiHGCDataset  # noqa: E402
from utils.env_utils import make_env_and_datasets  # noqa: E402
from utils.flax_utils import restore_agent  # noqa: E402


DATASET_CLASSES = {
    'GCDataset': GCDataset,
    'HGCDataset': HGCDataset,
    'MultiHGCDataset': MultiHGCDataset,
}


def flatten_metrics(metrics):
    return {key: float(np.asarray(value)) for key, value in metrics.items()}


def mean_metrics(rows):
    keys = sorted({key for row in rows for key in row})
    return {key: float(np.mean([row[key] for row in rows if key in row])) for key in keys}


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_flags(run_dir):
    with open(os.path.join(run_dir, 'flags.json'), 'r') as f:
        return json.load(f)


def diagnose_run(run_dir, restore_epoch, num_batches, batch_size, output_dir, seed):
    flags = load_flags(run_dir)
    config = ConfigDict(flags['agent'])
    config.enable_hcoghp_diagnostics = True
    config.enable_free_running_validation = True

    env_name = flags['env_name']
    env, train_dataset, val_dataset = make_env_and_datasets(env_name, frame_stack=config['frame_stack'])
    del train_dataset
    dataset_class = DATASET_CLASSES[config['dataset_class']]
    val_dataset = dataset_class(Dataset.create(**val_dataset), config)

    rng = jax.random.PRNGKey(seed)
    np.random.seed(seed)

    example_batch = val_dataset.sample(1)
    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )
    agent = restore_agent(agent, run_dir, restore_epoch)
    agent = agent.replace(
        config=agent.config.copy(
            {
                'enable_hcoghp_diagnostics': True,
                'enable_free_running_validation': True,
            }
        )
    )

    rows = []
    for batch_idx in range(num_batches):
        batch = val_dataset.sample(batch_size)
        rng, batch_rng = jax.random.split(rng)
        info = flatten_metrics(agent.validation_rollout_info(batch, rng=batch_rng))
        info['batch'] = batch_idx
        rows.append(info)
        print(
            f"[{batch_idx + 1}/{num_batches}] "
            f"teacher_high={info.get('validation_teacher/high_actor/mse', float('nan')):.6f} "
            f"free_high={info.get('validation_free/high_actor/mse', float('nan')):.6f} "
            f"gap_high={info.get('validation_gap/high_actor_mse', float('nan')):.6f}",
            flush=True,
        )

    summary = mean_metrics(rows)
    summary['batch'] = 'mean'

    run_name = os.path.basename(os.path.normpath(run_dir))
    group_name = os.path.basename(os.path.dirname(os.path.normpath(run_dir)))
    run_output_dir = os.path.join(output_dir, group_name, run_name)
    write_csv(os.path.join(run_output_dir, 'diagnostics_batches.csv'), rows)
    write_csv(os.path.join(run_output_dir, 'diagnostics_summary.csv'), [summary])
    print(f"Wrote diagnostics to {run_output_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', action='append', required=True)
    parser.add_argument('--restore_epoch', type=int, default=1000000)
    parser.add_argument('--num_batches', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--output_dir',
        default=os.path.join(
            os.environ.get('DATA_ROOT', '/data/qijunrong/06-RL/offline-rl'),
            'exp',
            'CoGHP',
            'diagnostics',
        ),
    )
    args = parser.parse_args()

    for run_dir in args.run_dir:
        diagnose_run(
            run_dir=os.path.abspath(run_dir),
            restore_epoch=args.restore_epoch,
            num_batches=args.num_batches,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            seed=args.seed,
        )


if __name__ == '__main__':
    main()
