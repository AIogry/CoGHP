#!/usr/bin/env python3
"""Compare two same-seed runs while ignoring wall-clock timing columns."""

import argparse
import csv
import glob
import os
import pickle
import sys

import jax
import numpy as np


def read_csv(path):
    with open(path, newline='') as file:
        return list(csv.DictReader(file))


def same_number(left, right, atol):
    left_value = float(left)
    right_value = float(right)
    if np.isnan(left_value) and np.isnan(right_value):
        return True
    if atol == 0:
        return left_value == right_value
    return bool(np.isclose(left_value, right_value, rtol=0, atol=atol))


def compare_csv(left_path, right_path, atol):
    if not os.path.exists(left_path) and not os.path.exists(right_path):
        return []
    if not os.path.exists(left_path) or not os.path.exists(right_path):
        return [f'CSV exists in only one run: {os.path.basename(left_path)}']

    left_rows = read_csv(left_path)
    right_rows = read_csv(right_path)
    if len(left_rows) != len(right_rows):
        return [
            f'{os.path.basename(left_path)} row count differs: '
            f'{len(left_rows)} != {len(right_rows)}'
        ]

    errors = []
    for row_index, (left_row, right_row) in enumerate(zip(left_rows, right_rows)):
        keys = sorted(
            key
            for key in set(left_row) | set(right_row)
            if not key.startswith('time/')
        )
        for key in keys:
            if key not in left_row or key not in right_row:
                errors.append(f'row {row_index}: missing key {key!r}')
            elif not same_number(left_row[key], right_row[key], atol):
                errors.append(
                    f'row {row_index}, {key}: {left_row[key]} != {right_row[key]}'
                )
            if len(errors) >= 20:
                return errors
    return errors


def checkpoint_epochs(run_dir):
    epochs = set()
    for path in glob.glob(os.path.join(run_dir, 'params_*.pkl')):
        name = os.path.basename(path)
        epochs.add(int(name.removeprefix('params_').removesuffix('.pkl')))
    return epochs


def compare_checkpoint(left_dir, right_dir, epoch, atol):
    left_path = os.path.join(left_dir, f'params_{epoch}.pkl')
    right_path = os.path.join(right_dir, f'params_{epoch}.pkl')
    with open(left_path, 'rb') as file:
        left = pickle.load(file)
    with open(right_path, 'rb') as file:
        right = pickle.load(file)

    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    if left_tree != right_tree:
        return ['Checkpoint parameter tree structures differ.']

    errors = []
    for leaf_index, (left_leaf, right_leaf) in enumerate(
        zip(left_leaves, right_leaves)
    ):
        left_array = np.asarray(left_leaf)
        right_array = np.asarray(right_leaf)
        if left_array.shape != right_array.shape:
            errors.append(
                f'leaf {leaf_index}: shape {left_array.shape} != {right_array.shape}'
            )
            continue
        if atol == 0:
            equal = np.array_equal(left_array, right_array, equal_nan=True)
        else:
            equal = np.allclose(
                left_array,
                right_array,
                rtol=0,
                atol=atol,
                equal_nan=True,
            )
        if not equal:
            max_diff = float(np.nanmax(np.abs(left_array - right_array)))
            errors.append(f'leaf {leaf_index}: max absolute difference {max_diff}')
        if len(errors) >= 20:
            break
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_a')
    parser.add_argument('run_b')
    parser.add_argument(
        '--epoch',
        type=int,
        default=None,
        help='Checkpoint epoch. Default: latest common checkpoint.',
    )
    parser.add_argument(
        '--atol',
        type=float,
        default=0.0,
        help='Absolute tolerance. Default 0 requires exact equality.',
    )
    args = parser.parse_args()

    run_a = os.path.abspath(args.run_a)
    run_b = os.path.abspath(args.run_b)
    errors = []
    for filename in ('train.csv', 'eval.csv'):
        errors.extend(
            compare_csv(
                os.path.join(run_a, filename),
                os.path.join(run_b, filename),
                args.atol,
            )
        )

    common_epochs = checkpoint_epochs(run_a) & checkpoint_epochs(run_b)
    epoch = args.epoch
    if epoch is None:
        if not common_epochs:
            errors.append('No common checkpoints found.')
        else:
            epoch = max(common_epochs)
    if epoch is not None:
        if epoch not in common_epochs:
            errors.append(f'Checkpoint {epoch} does not exist in both runs.')
        else:
            errors.extend(compare_checkpoint(run_a, run_b, epoch, args.atol))

    if errors:
        print('REPRODUCIBILITY CHECK FAILED')
        for error in errors:
            print(f'- {error}')
        return 1

    mode = 'exact' if args.atol == 0 else f'atol={args.atol}'
    print(f'REPRODUCIBILITY CHECK PASSED ({mode}, checkpoint={epoch})')
    print('Wall-clock columns under time/ were intentionally ignored.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
