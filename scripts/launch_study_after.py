#!/usr/bin/env python3
"""Launch a prepared study once all dependency runs have completed successfully.

Each invocation returns promptly. A scheduler may call --execute repeatedly:
waiting does not launch anything, and recorded launches are never duplicated.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


def read(path):
    return json.loads(path.read_text())


def dependency_status(runner, root, manifest):
    waiting = []
    for job in runner.plan(manifest['study'], root, None, None, 0):
        path = Path(job['run_dir']) / 'run.json'
        if not path.exists():
            waiting.append(dict(method=job['method'], status='pending'))
            continue
        state = read(path)
        status = state.get('status')
        if status in ('failed', 'cancelled') or state.get('diagnostics', {}).get('status') == 'failed':
            raise ValueError(f"Dependency needs attention: {job['method']} ({status}).")
        if status != 'completed':
            waiting.append(dict(method=job['method'], status=status))
            continue
        if state.get('returncode') != 0 or state.get('source_signature') != manifest['source']['code_signature']:
            raise ValueError(f"Invalid completed dependency: {job['method']}.")
        runner.validate_completed(job, manifest['study'])
    return waiting


def launch_once(runner, after, root, gpus, workers_per_gpu, execute=False):
    manifest = read(root / 'campaign.json')
    source = Path(manifest['source_dir'])
    # Any recorded attempt or launch intent requires inspection, never a retry.
    if (root / 'launcher_process.json').exists() or any((root / 'runs').glob('*/**/run.json')):
        return dict(status='already_launched', run_root=str(root))
    if (root / 'launcher.log').exists():
        raise ValueError('Launcher log already exists without a recorded attempt; inspect before retrying.')
    parent = read(after / 'campaign.json')
    pending = dependency_status(runner, after, parent)
    if pending:
        return dict(status='waiting', dependency=str(after), unfinished=pending)
    if not execute:
        return dict(status='ready', run_root=str(root))
    runner.verify_snapshot(source, manifest['source'])
    try:
        runner.check_gpu_occupancy(gpus)
    except ValueError as error:
        if str(error).startswith('Selected GPUs already have compute processes:'):
            return dict(status='waiting_gpu', run_root=str(root), detail=str(error))
        raise
    command = [sys.executable, '-u', str(source / 'scripts/run_study.py'),
               '--run-root', str(root), '--gpus', ','.join(gpus),
               '--workers-per-gpu', str(workers_per_gpu), '--wandb-mode', 'offline', '--execute']
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    for name in ('JAX_PLATFORMS', 'JAX_PLATFORM_NAME'):
        env.pop(name, None)
    record = dict(started_at=datetime.now(timezone.utc).isoformat(), command=command,
                  source_signature=manifest['source']['code_signature'],
                  dependency=str(after), dependency_source_signature=parent['source']['code_signature'],
                  physical_gpus=gpus, workers_per_gpu=workers_per_gpu)
    # An exclusive log is also a durable launch-intent marker if recording the
    # PID fails after Popen. In that case later invocations fail closed.
    with (root / 'launcher.log').open('x') as log:
        process = subprocess.Popen(command, cwd=source, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    record['pid'] = process.pid
    runner.write_json(root / 'launcher_process.json', record)
    return dict(status='started', **record, run_root=str(root))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True, help='Already prepared campaign.')
    parser.add_argument('--gpus', default='1')
    parser.add_argument('--workers-per-gpu', type=int, default=2)
    parser.add_argument('--execute', action='store_true', help='Otherwise inspect readiness only.')
    args = parser.parse_args()
    after, root = args.after.resolve(), args.run_root.resolve()
    if after == root or args.workers_per_gpu < 1:
        raise ValueError('Dependency and new campaign must differ; worker count must be positive.')
    gpus = args.gpus.split(',')
    if len(set(gpus)) != len(gpus) or any(not gpu.isdigit() for gpu in gpus):
        raise ValueError('GPU ids must be unique nonnegative integers.')
    source = Path(read(root / 'campaign.json')['source_dir'])
    sys.dont_write_bytecode = True  # Importing the frozen runner must not change its snapshot.
    spec = importlib.util.spec_from_file_location('prepared_study_runner', source / 'scripts/run_study.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    if args.execute:
        with (root / '.deferred-launch.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            result = launch_once(runner, after, root, gpus, args.workers_per_gpu, True)
    else:
        result = launch_once(runner, after, root, gpus, args.workers_per_gpu)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(json.dumps(dict(status='needs_attention', error=str(error))), file=sys.stderr)
        sys.exit(2)
