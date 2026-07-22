#!/usr/bin/env python3
"""Offline candidate and selector diagnostics for FM-CoGHP checkpoints."""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import jax
import numpy as np
from ml_collections import ConfigDict


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
IMPLS_DIR = os.path.join(PROJECT_ROOT, 'impls')
for path in (PROJECT_ROOT, IMPLS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from agents import agents  # noqa: E402
from utils.datasets import Dataset, GCDataset, HGCDataset, MultiHGCDataset  # noqa: E402
from utils.env_utils import make_env_and_datasets  # noqa: E402
from utils.flax_utils import restore_agent  # noqa: E402


DATASET_CLASSES = {
    'GCDataset': GCDataset,
    'HGCDataset': HGCDataset,
    'MultiHGCDataset': MultiHGCDataset,
}


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with open(path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_agent(run_dir, restore_epoch, seed, diagnostic_candidates):
    with open(os.path.join(run_dir, 'flags.json'), 'r') as file:
        flags = json.load(file)
    config = ConfigDict(flags['agent'])
    config.enable_fm_diagnostics = True
    config.flow_diagnostic_candidates = diagnostic_candidates
    env, train_dataset, val_dataset = make_env_and_datasets(
        flags['env_name'],
        frame_stack=config['frame_stack'],
    )
    del env, train_dataset
    dataset_class = DATASET_CLASSES[config['dataset_class']]
    val_dataset = dataset_class(Dataset.create(**val_dataset), config)
    example_batch = val_dataset.sample(1)
    agent = agents[config['agent_name']].create(
        seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )
    agent = restore_agent(agent, run_dir, restore_epoch)
    agent = agent.replace(
        config=agent.config.copy(
            {
                'enable_fm_diagnostics': True,
                'flow_diagnostic_candidates': diagnostic_candidates,
            }
        )
    )
    return agent, val_dataset, flags['env_name']


def diagnose_run(run_dir, restore_epoch, num_batches, batch_size, num_candidates, output_dir, seed):
    agent, val_dataset, env_name = load_agent(run_dir, restore_epoch, seed, num_candidates)
    rng = jax.random.PRNGKey(seed)
    np.random.seed(seed)
    candidate_rows = []
    slot_rows = []
    selector_accumulators = defaultdict(lambda: defaultdict(list))

    for batch_idx in range(num_batches):
        batch = val_dataset.sample(batch_size)
        rng, flow_rng, action_rng = jax.random.split(rng, 3)
        target_chain = agent.flow_target_reps(batch)
        candidates, _ = agent.sample_flow_chains(
            batch['observations'],
            batch['high_actor_goals'],
            flow_rng,
            num_candidates=num_candidates,
        )
        nearest_mse = np.asarray(
            jax.device_get(
                np.mean(
                    np.square(
                        np.asarray(candidates[:, :, -1, :])
                        - np.asarray(target_chain[:, None, -1, :])
                    ),
                    axis=-1,
                )
            )
        )
        full_chain_mse = np.mean(
            np.square(np.asarray(candidates) - np.asarray(target_chain[:, None, :, :])),
            axis=(-1, -2),
        )
        value_scores = np.asarray(
            agent.score_flow_candidates(batch['observations'], candidates, 'coghp_value')
        )
        crl_scores = np.asarray(
            agent.score_flow_candidates(batch['observations'], candidates, 'local_crl')
        )
        oracle_idx = np.argmin(nearest_mse, axis=1)
        selected = {
            'first': np.zeros(batch_size, dtype=np.int32),
            'value': np.argmax(value_scores, axis=1),
            'crl': np.argmax(crl_scores, axis=1),
            'oracle': oracle_idx,
        }

        for sample_idx in range(batch_size):
            for candidate_idx in range(num_candidates):
                candidate_rows.append(
                    {
                        'batch_id': batch_idx,
                        'sample_id': sample_idx,
                        'candidate_id': candidate_idx,
                        'nearest_mse': float(nearest_mse[sample_idx, candidate_idx]),
                        'full_chain_mse': float(full_chain_mse[sample_idx, candidate_idx]),
                        'value_score': float(value_scores[sample_idx, candidate_idx]),
                        'crl_score': float(crl_scores[sample_idx, candidate_idx]),
                        'selected_by_value': int(selected['value'][sample_idx] == candidate_idx),
                        'selected_by_crl': int(selected['crl'][sample_idx] == candidate_idx),
                        'selected_by_oracle': int(oracle_idx[sample_idx] == candidate_idx),
                        'nearest_latent_norm': float(
                            np.linalg.norm(np.asarray(candidates[sample_idx, candidate_idx, -1]))
                        ),
                        'full_chain_latent_norm': float(
                            np.mean(np.linalg.norm(np.asarray(candidates[sample_idx, candidate_idx]), axis=-1))
                        ),
                    }
                )
                for slot_idx in range(candidates.shape[2]):
                    slot_rows.append(
                        {
                            'batch_id': batch_idx,
                            'sample_id': sample_idx,
                            'candidate_id': candidate_idx,
                            'slot': slot_idx,
                            'slot_mse': float(
                                np.mean(
                                    np.square(
                                        np.asarray(candidates[sample_idx, candidate_idx, slot_idx])
                                        - np.asarray(target_chain[sample_idx, slot_idx])
                                    )
                                )
                            ),
                            'slot_norm': float(
                                np.linalg.norm(np.asarray(candidates[sample_idx, candidate_idx, slot_idx]))
                            ),
                        }
                    )

        action_keys = jax.random.split(action_rng, len(selected))
        for selector_idx, (selector, indices) in enumerate(selected.items()):
            batch_indices = np.arange(batch_size)
            selected_chain = candidates[batch_indices, indices]
            selector_accumulators[selector]['nearest_mse'].extend(
                nearest_mse[batch_indices, indices].tolist()
            )
            selector_accumulators[selector]['full_chain_mse'].extend(
                full_chain_mse[batch_indices, indices].tolist()
            )
            selector_accumulators[selector]['oracle_agreement'].extend(
                (indices == oracle_idx).astype(np.float32).tolist()
            )
            ranks = 1 + np.sum(
                nearest_mse < nearest_mse[batch_indices, indices][:, None],
                axis=1,
            )
            selector_accumulators[selector]['oracle_rank'].extend(ranks.tolist())
            action_mse = float(
                agent._action_mse_for_chain(batch, selected_chain, action_keys[selector_idx])
            )
            selector_accumulators[selector]['action_mse'].append(action_mse)

        print(f'[{batch_idx + 1}/{num_batches}] collected {batch_size * num_candidates} candidates', flush=True)

    summary_rows = []
    for selector, metrics in selector_accumulators.items():
        summary_rows.append(
            {
                'selector': selector,
                'nearest_mse_mean': float(np.mean(metrics['nearest_mse'])),
                'nearest_mse_std': float(np.std(metrics['nearest_mse'])),
                'full_chain_mse_mean': float(np.mean(metrics['full_chain_mse'])),
                'action_mse_mean': float(np.mean(metrics['action_mse'])),
                'oracle_agreement': float(np.mean(metrics['oracle_agreement'])),
                'average_oracle_rank': float(np.mean(metrics['oracle_rank'])),
            }
        )

    run_name = os.path.basename(os.path.normpath(run_dir))
    run_output_dir = os.path.join(output_dir, env_name, run_name)
    write_csv(os.path.join(run_output_dir, 'candidate_metrics.csv'), candidate_rows)
    write_csv(os.path.join(run_output_dir, 'selector_summary.csv'), summary_rows)
    write_csv(os.path.join(run_output_dir, 'slot_metrics.csv'), slot_rows)
    print(f'Wrote FM-CoGHP diagnostics to {run_output_dir}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', action='append', required=True)
    parser.add_argument('--restore_epoch', type=int, default=1000000)
    parser.add_argument('--num_batches', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_candidates', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--output_dir',
        default=os.path.join(
            os.environ.get('DATA_ROOT', '/data/qijunrong/06-RL/offline-rl'),
            'exp',
            'FM_CoGHP',
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
            num_candidates=args.num_candidates,
            output_dir=args.output_dir,
            seed=args.seed,
        )


if __name__ == '__main__':
    main()
