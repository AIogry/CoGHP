"""Create post-training CSV summaries for the 204k feed-forward baseline."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import flax
import jax
import ml_collections
import numpy as np

from preflight_coghp_ff_swiglu_204k import (
    action_branch_param_count,
    action_core_flat_params,
    count_params,
    create_agent_and_batch,
    measure_action_latency_ms,
)
from utils.flax_utils import build_checkpoint_metadata, restore_agent


SUMMARY_FIELDS = (
    'experiment_name',
    'environment',
    'seed',
    'final_success_rate',
    'peak_success_rate',
    'last_5_evaluation_average',
    'evaluation_curve_auc_normalized',
    'per_task_success_rate_json',
    'final_train_action_mse',
    'final_validation_action_mse',
    'final_validation_teacher_action_mse',
    'final_validation_predicted_action_mse',
    'final_train_low_actor_loss',
    'final_train_low_actor_log_prob',
    'final_train_low_actor_adv',
    'final_train_low_actor_std',
    'final_validation_low_actor_loss',
    'final_validation_low_actor_log_prob',
    'final_validation_low_actor_adv',
    'final_validation_low_actor_std',
    'action_core_params',
    'action_branch_total_params',
    'whole_agent_params',
    'mean_update_time_seconds',
    'total_wall_time_seconds',
    'peak_gpu_memory_mib',
    'action_inference_latency_ms',
    'inference_device',
    'checkpoint_epoch',
    'git_commit',
    'code_signature',
)


def read_csv(path: Path):
    with path.open(newline='', encoding='utf-8') as file:
        return list(csv.DictReader(file))


def number(row, key):
    value = row.get(key, '')
    return float(value) if value not in ('', None) else float('nan')


def normalized_auc(eval_rows, train_steps):
    points = [(0.0, 0.0)]
    points.extend(
        (number(row, 'step'), number(row, 'evaluation/overall_success'))
        for row in eval_rows
    )
    points = [(x, y) for x, y in points if np.isfinite(x) and np.isfinite(y)]
    if len(points) < 2 or train_steps <= 0:
        return float('nan')
    area = sum(
        (right_x - left_x) * (left_y + right_y) / 2.0
        for (left_x, left_y), (right_x, right_y) in zip(
            points[:-1], points[1:]
        )
    )
    return area / train_steps


def peak_gpu_memory(path: Path):
    if not path.exists():
        return float('nan')
    values = []
    for line in path.read_text(encoding='utf-8').splitlines():
        fields = [field.strip() for field in line.split(',')]
        if len(fields) >= 2:
            try:
                values.append(float(fields[1]))
            except ValueError:
                pass
    return max(values) if values else float('nan')


def write_summary(path: Path, row, *, append=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = 'a' if append and path.exists() else 'w'
    with path.open(mode, newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        if mode == 'w':
            writer.writeheader()
        writer.writerow({key: row.get(key, '') for key in SUMMARY_FIELDS})


def summarize(args):
    run_dir = Path(args.run_dir).resolve()
    flags_data = json.loads((run_dir / 'flags.json').read_text(encoding='utf-8'))
    config = ml_collections.ConfigDict(flags_data['agent'])
    env_name = flags_data['env_name']
    seed = int(flags_data['seed'])
    _, _, agent, batch = create_agent_and_batch(
        env_name,
        seed,
        config=config,
    )
    checkpoint_files = list(run_dir.glob('params_*.pkl'))
    if not checkpoint_files:
        raise FileNotFoundError(f'No checkpoints found in {run_dir}')
    checkpoint_epoch = max(
        int(re.search(r'params_(\d+)\.pkl$', path.name).group(1))
        for path in checkpoint_files
    )
    expected_metadata = build_checkpoint_metadata(
        config,
        environment=env_name,
        seed=seed,
        git_commit=flags_data['experiment_git_commit'],
        code_signature=flags_data['experiment_code_signature'],
        resume=False,
        restore_path=None,
    )
    agent = restore_agent(
        agent,
        str(run_dir),
        checkpoint_epoch,
        expected_checkpoint_metadata=expected_metadata,
        require_checkpoint_metadata=True,
    )

    eval_rows = read_csv(run_dir / 'eval.csv')
    train_rows = read_csv(run_dir / 'train.csv')
    if not eval_rows or not train_rows:
        raise ValueError(f'Incomplete CSV logs in {run_dir}')
    successes = [
        number(row, 'evaluation/overall_success') for row in eval_rows
    ]
    final_eval = eval_rows[-1]
    final_train = train_rows[-1]
    per_task = {
        key.removeprefix('evaluation/').removesuffix('_success'): float(value)
        for key, value in final_eval.items()
        if key.startswith('evaluation/task')
        and key.endswith('_success')
        and value not in ('', None)
    }

    flat_core = action_core_flat_params(agent)
    core_params = sum(int(value.size) for value in flat_core.values())
    if core_params != 204288:
        raise ValueError(f'Unexpected action core size: {core_params}')
    latency_ms = measure_action_latency_ms(
        agent,
        batch['observations'][0],
        batch['high_actor_goals'][0],
    )
    update_times = [
        number(row, 'time/epoch_time')
        for row in train_rows
        if np.isfinite(number(row, 'time/epoch_time'))
    ]
    row = {
        'experiment_name': flags_data['run_group'],
        'environment': env_name,
        'seed': seed,
        'final_success_rate': successes[-1],
        'peak_success_rate': max(successes),
        'last_5_evaluation_average': float(np.mean(successes[-5:])),
        'evaluation_curve_auc_normalized': normalized_auc(
            eval_rows,
            int(flags_data['train_steps']),
        ),
        'per_task_success_rate_json': json.dumps(per_task, sort_keys=True),
        'final_train_action_mse': number(
            final_train,
            'training/low_actor/mse',
        ),
        'final_validation_action_mse': number(
            final_train,
            'validation/low_actor/mse',
        ),
        'final_validation_teacher_action_mse': number(
            final_train,
            'validation_action/teacher/mse',
        ),
        'final_validation_predicted_action_mse': number(
            final_train,
            'validation_action/predicted/mse',
        ),
        'final_train_low_actor_loss': number(
            final_train,
            'training/low_actor/actor_loss',
        ),
        'final_train_low_actor_log_prob': number(
            final_train,
            'training/low_actor/bc_log_prob',
        ),
        'final_train_low_actor_adv': number(
            final_train,
            'training/low_actor/adv',
        ),
        'final_train_low_actor_std': number(
            final_train,
            'training/low_actor/std',
        ),
        'final_validation_low_actor_loss': number(
            final_train,
            'validation/low_actor/actor_loss',
        ),
        'final_validation_low_actor_log_prob': number(
            final_train,
            'validation/low_actor/bc_log_prob',
        ),
        'final_validation_low_actor_adv': number(
            final_train,
            'validation/low_actor/adv',
        ),
        'final_validation_low_actor_std': number(
            final_train,
            'validation/low_actor/std',
        ),
        'action_core_params': core_params,
        'action_branch_total_params': action_branch_param_count(agent),
        'whole_agent_params': count_params(agent.network.params),
        'mean_update_time_seconds': float(np.mean(update_times)),
        'total_wall_time_seconds': number(final_train, 'time/total_time'),
        'peak_gpu_memory_mib': peak_gpu_memory(Path(args.gpu_monitor)),
        'action_inference_latency_ms': latency_ms,
        'inference_device': str(jax.devices()[0]),
        'checkpoint_epoch': checkpoint_epoch,
        'git_commit': flags_data['experiment_git_commit'],
        'code_signature': flags_data['experiment_code_signature'],
    }
    write_summary(run_dir / 'summary.csv', row)
    write_summary(Path(args.summary_csv), row, append=True)
    (run_dir / 'summary.json').write_text(
        json.dumps(row, indent=2, sort_keys=True),
        encoding='utf-8',
    )
    print(json.dumps(row, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--gpu-monitor', required=True)
    parser.add_argument('--summary-csv', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    summarize(parse_args())
