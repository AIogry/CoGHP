#!/usr/bin/env python3
"""Plan, execute, or summarize a declarative local study (standard library only).

Execution copies and hashes the actual source, including uncommitted changes.
Training uses that snapshot; Git HEAD alone is never claimed to identify dirty
code. Each GPU has one sequential worker. Existing attempts are never replaced.
"""

import argparse
import csv
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import queue
import re
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def safe_name(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', value):
        raise ValueError(f'Unsafe identifier: {value!r}')
    return value


def load_study(path):
    study = json.loads(path.read_text())
    safe_name(study['study_id'])
    if not study['methods'] or not study['environments']:
        raise ValueError('Study must contain methods and environments.')
    for name, method in study['methods'].items():
        safe_name(name)
        if method['agent'] not in ('coghp', 'navigation', 'hrm_action_coghp', 'navigation_coghp'):
            raise ValueError(f'Unsupported agent: {method["agent"]}')
        environments = method.get('environments', list(study['environments']))
        if (not environments or len(environments) != len(set(environments))
                or set(environments) - set(study['environments'])):
            raise ValueError(f'Invalid environment scope for {name}')
        overrides = method.get('seeds_by_environment', {})
        if set(overrides) - set(environments):
            raise ValueError(f'Seed override outside environment scope for {name}')
        for seeds in [method['seeds'], *overrides.values()]:
            if not seeds or len(seeds) != len(set(seeds)) or any(type(s) is not int or s < 0 for s in seeds):
                raise ValueError(f'Invalid seeds for {name}')
    for name in study['environments']:
        safe_name(name)
    protocol = study['protocol']
    allowed = {'train_steps', 'log_interval', 'eval_interval', 'save_interval', 'save_steps',
               'eval_episodes', 'eval_temperature', 'video_episodes', 'eval_on_cpu',
               'fixed_validation_batch', 'require_deterministic_gpu'}
    if set(protocol) - allowed:
        raise ValueError(f'Unknown protocol fields: {set(protocol) - allowed}')
    for field in ('train_steps', 'log_interval', 'eval_interval', 'save_interval', 'eval_episodes'):
        if type(protocol[field]) is not int or protocol[field] <= 0:
            raise ValueError(f'{field} must be a positive integer')
    steps = study['metric']['steps']
    if not steps or len(steps) != len(set(steps)):
        raise ValueError('Metric steps must be nonempty and unique.')
    for step in steps:
        if type(step) is not int or step <= 0 or step > protocol['train_steps'] or step % protocol['eval_interval']:
            raise ValueError(f'Metric step {step} is outside the evaluation schedule.')
    for step in protocol.get('save_steps', []):
        if type(step) is not int or not 0 < step <= protocol['train_steps']:
            raise ValueError(f'Invalid checkpoint step: {step}')
    if study.get('diagnostics'):
        diagnostic_steps = study['diagnostics']['steps']
        saved = set(protocol.get('save_steps', [])) | set(
            range(protocol['save_interval'], protocol['train_steps'] + 1, protocol['save_interval']))
        if not diagnostic_steps or set(diagnostic_steps) - saved:
            raise ValueError('Diagnostic steps must have saved checkpoints.')
        for key in ('value_samples', 'local_episodes', 'local_budget'):
            value = study['diagnostics'].get(key, 1)
            if type(value) is not int or value <= 0:
                raise ValueError(f'Invalid diagnostic {key}')
    return study


def select(values, requested, label):
    if requested is None:
        return list(values)
    chosen = requested.split(',')
    if not chosen or len(chosen) != len(set(chosen)) or set(chosen) - set(values):
        raise ValueError(f'Invalid {label}: {requested}; available: {",".join(values)}')
    return chosen


def plan(study, campaign, methods=None, environments=None, attempt=0):
    jobs = []
    for environment in select(study['environments'], environments, 'environments'):
        for method in select(study['methods'], methods, 'methods'):
            spec = study['methods'][method]
            if environment not in spec.get('environments', study['environments']):
                continue
            agent = {**study['common_agent'], **study['environments'][environment], **spec['overrides']}
            if (agent.get('action_core_type') == 'hrm'
                    and agent.get('action_block_type') == 'swiglu_rmsnorm'):
                agent['expected_action_core_params'] = (
                    6 * agent['action_hrm_num_layers'] * agent['hrm_mixer_intermediate_size']
                    * (agent['num_subgoals'] + 3 + agent['feature_dim'])
                )
            for seed in spec.get('seeds_by_environment', {}).get(environment, spec['seeds']):
                jobs.append(dict(method=method, environment=environment, seed=seed,
                                 agent_name=spec['agent'], agent=agent, attempt=attempt,
                                 run_dir=str(campaign / 'runs' / method / environment
                                             / f'seed_{seed:03d}' / f'attempt_{attempt:03d}')))
    if study.get('seed_major_order', False):
        jobs.sort(key=lambda job: job['seed'])
    return jobs


def flag_value(value):
    if isinstance(value, list):
        return ','.join(str(item) for item in value)
    return str(value)


def command(job, study, source, provenance, wandb_mode):
    agent_file = f'agents/{job["agent_name"]}.py'
    # Old frozen snapshots retain their entry files; current code uses profiles.
    if not (source / 'impls' / agent_file).is_file():
        agent_file = {
            'hrm_action_coghp': 'agents/navigation.py:legacy_action',
            'navigation_coghp': 'agents/navigation.py:legacy_navigation',
        }.get(job['agent_name'], agent_file)
    arguments = {
        **study['protocol'], 'run_dir': str(Path(job['run_dir']) / 'artifacts'),
        'run_group': job['method'], 'wandb_project': 'CoGHPNavigation', 'wandb_mode': wandb_mode,
        'env_name': job['environment'], 'seed': job['seed'], 'require_gpu': True,
        'checkpoint_signature_required': True,
        'experiment_git_commit': provenance['git_commit'],
        'experiment_code_signature': provenance['code_signature'],
        'agent': agent_file,
    }
    return [sys.executable, '-u', str(source / 'impls/main.py'),
            *[f'--{key}={flag_value(value)}' for key, value in arguments.items()],
            *[f'--agent.{key}={flag_value(value)}' for key, value in job['agent'].items()]]


def child_environment(source, dataset_root, gpu, seed):
    env = os.environ.copy()
    # Use one known numerical setup for every run, regardless of old shell scripts.
    env.update(CUDA_VISIBLE_DEVICES=gpu, CUDA_DEVICE_ORDER='PCI_BUS_ID',
               PYTHONPATH=f'{source}:{source / "impls"}', PYTHONHASHSEED=str(seed),
               PYTHONDONTWRITEBYTECODE='1', OGBENCH_DATASET_DIR=str(dataset_root),
               XLA_FLAGS='--xla_gpu_deterministic_ops', XLA_PYTHON_CLIENT_PREALLOCATE='false',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl')
    for name in ('JAX_PLATFORMS', 'JAX_PLATFORM_NAME'):
        env.pop(name, None)
    return env


def dataset_inventory(study, root, hash_files=False):
    result = {}
    for environment in study['environments']:
        for suffix in ('.npz', '-val.npz'):
            path = root / (environment + suffix)
            stat = path.stat()  # A missing dataset is an error, never a skipped run/download.
            if not path.is_file() or stat.st_size == 0:
                raise ValueError(f'Empty or invalid dataset: {path}')
            record = dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
            if hash_files:
                record['sha256'] = digest(path)
            result[path.name] = record
    return result


def source_inventory(root):
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args])
    names = git('ls-files', '-z', '--cached', '--others', '--exclude-standard').decode().split('\0')
    files = {}
    for name in sorted(set(filter(None, names))):
        path = root / name
        if not path.exists():  # Record deleted tracked files in git_status, not in the snapshot.
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'Source snapshot requires regular files: {path}')
        files[name] = digest(path)
    return dict(git_commit=git('rev-parse', 'HEAD').decode().strip(),
                git_branch=git('rev-parse', '--abbrev-ref', 'HEAD').decode().strip(),
                git_status=git('status', '--porcelain', '--untracked-files=all').decode(),
                original_worktree=str(root), files=files, code_signature=signature(files))


def verify_snapshot(source, provenance):
    actual = {str(p.relative_to(source)): digest(p) for p in source.rglob('*') if p.is_file()}
    if actual != provenance['files']:
        raise ValueError('Frozen source snapshot differs from its recorded file hashes.')


def freeze(campaign, study, datasets, source_root=None):
    manifest_path = campaign / 'campaign.json'
    if campaign.exists():
        manifest = json.loads(manifest_path.read_text())
        source = Path(manifest.get('source_dir', campaign / 'source'))
        if source_root is not None and source_root.resolve() != source.resolve():
            raise ValueError('Requested source directory differs from the frozen campaign.')
        if manifest['study'] != study or manifest['datasets'] != datasets:
            raise ValueError('Existing campaign has different study/data. Use a new run root.')
        verify_snapshot(source, manifest['source'])
        return source, manifest['source']
    source = source_root.resolve() if source_root is not None else campaign / 'source'
    if source.exists():
        raise ValueError(f'Source destination already exists: {source}')
    if source.is_relative_to(ROOT) or ROOT.is_relative_to(source):
        raise ValueError('Frozen source must be separate from the development tree.')
    provenance = source_inventory(ROOT)
    campaign.mkdir(parents=True, exist_ok=False)
    source.mkdir(parents=True, exist_ok=False)
    for name in provenance['files']:
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
        target.chmod(0o444 | (target.stat().st_mode & 0o111))
    verify_snapshot(source, provenance)
    if source != campaign / 'source':
        (campaign / 'source').symlink_to(source, target_is_directory=True)
    # Persist the complete study even when this invocation selects only some jobs.
    write_json(manifest_path, dict(created_at=now(), study=study, datasets=datasets,
                                   source=provenance, source_dir=str(source)))
    return source, provenance


def check_gpu_occupancy(gpus, allowed_processes=None):
    output = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid,name', '--format=csv,noheader'], text=True)
    devices = {row[0].strip(): (row[1].strip(), row[2].strip()) for row in csv.reader(output.splitlines())}
    if set(gpus) - set(devices):
        raise ValueError(f'Requested GPU IDs absent from nvidia-smi: {gpus}')
    processes = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name', '--format=csv,noheader'], text=True)
    selected = {devices[gpu][0] for gpu in gpus}
    allowed_processes = allowed_processes or {}
    occupied = []
    for line in processes.splitlines():
        row = [part.strip() for part in next(csv.reader([line]))]
        if row[0] not in selected:
            continue
        # Exact PID AND executable matching, never a blanket occupancy bypass.
        if len(row) >= 3 and allowed_processes.get(row[1]) == row[2]:
            continue
        occupied.append(line)
    if occupied:
        raise ValueError('Selected GPUs already have compute processes: ' + '; '.join(occupied))
    return devices


def check_gpus(gpus, dataset_root, source, allowed_processes=None):
    devices = check_gpu_occupancy(gpus, allowed_processes)
    probe = ('import jax, ogbench, pathlib; '
             'assert jax.default_backend() == "gpu", jax.devices(); '
             'assert pathlib.Path(ogbench.__file__).resolve().is_relative_to(pathlib.Path.cwd().parent); '
             'print(jax.devices())')
    for gpu in gpus:
        result = subprocess.run([sys.executable, '-c', probe], cwd=source / 'impls',
                                env=child_environment(source, dataset_root, gpu, 0),
                                capture_output=True, text=True, timeout=120)
        if result.returncode:
            raise ValueError(f'JAX GPU {gpu} preflight failed:\n{result.stderr}\n{result.stdout}')
        print(f'GPU {gpu}: {devices[gpu][1]} {result.stdout.strip()}', flush=True)


def read_metric(path, metric):
    wanted = metric['steps']
    values = {}
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        if not {'step', metric['column']} <= set(reader.fieldnames or []):
            raise ValueError(f'Missing metric/step column: {path}')
        for row in reader:
            step = int(row['step'])
            if step not in wanted:
                continue
            if step in values:
                raise ValueError(f'Duplicate evaluation step {step}: {path}')
            value = float(row[metric['column']])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'Invalid success rate at {step}: {value}')
            values[step] = value
    if set(values) != set(wanted):
        raise ValueError(f'Missing evaluation steps {sorted(set(wanted) - set(values))}: {path}')
    return dict(values={str(step): values[step] for step in wanted}, mean=statistics.mean(values.values()))


def validate_completed(job, study):
    artifacts = Path(job['run_dir']) / 'artifacts'
    flags = json.loads((artifacts / 'flags.json').read_text())
    if flags['env_name'] != job['environment'] or flags['seed'] != job['seed']:
        raise ValueError(f'Run identity mismatch: {artifacts}')
    if flags['agent']['agent_name'] != job['agent_name']:
        raise ValueError(f'Agent implementation mismatch: {artifacts}')
    if flags.get('eval_tasks') is not None or flags.get('eval_gaussian') is not None:
        raise ValueError(f'Expected all evaluation tasks and no added action noise: {artifacts}')
    metadata = json.loads((artifacts.parent / 'run.json').read_text())
    if (flags.get('experiment_code_signature') != metadata['source_signature']
            or flags.get('experiment_git_commit') != metadata['git_commit']):
        raise ValueError(f'Checkpoint provenance mismatch: {artifacts}')
    for key, value in job['agent'].items():
        if flags['agent'].get(key) != value:
            raise ValueError(f'Agent config mismatch for {key}: {artifacts}')
    for key, value in study['protocol'].items():
        actual = flags[key]
        if key == 'save_steps':
            actual = [int(step) for step in actual]
        if actual != value:
            raise ValueError(f'Protocol mismatch for {key}: {artifacts}')
    checkpoint_steps = set(study['protocol'].get('save_steps', []))
    checkpoint_steps.update(range(study['protocol']['save_interval'],
                                  study['protocol']['train_steps'] + 1,
                                  study['protocol']['save_interval']))
    for step in checkpoint_steps:
        if not (artifacts / f'params_{step}.pkl').is_file():
            raise ValueError(f'Missing checkpoint at {step}: {artifacts}')
    return read_metric(artifacts / 'eval.csv', study['metric'])


def run_diagnostics(job, study, source, dataset_root):
    """CPU-only, held-out probes AFTER training; never used to select actions."""
    spec = study.get('diagnostics')
    if not spec or job['agent_name'] not in ('navigation', 'navigation_coghp'):
        return dict(status='not_requested')
    directory = Path(job['run_dir']) / 'diagnostics'
    directory.mkdir(exist_ok=False)
    artifacts = Path(job['run_dir']) / 'artifacts'
    env = child_environment(source, dataset_root, '', job['seed'])
    env.update(JAX_PLATFORMS='cpu', OPENBLAS_NUM_THREADS='1')
    records = []
    for step in spec['steps']:
        probes = [('values', 'diagnose_navigation_values.py',
                   [f'--samples={spec.get("value_samples", 2048)}'])]
        if job['agent']['command_content'] == 'position':
            probes.append(('commands', 'diagnose_navigation_commands.py',
                           [f'--episodes={spec.get("local_episodes", 24)}',
                            f'--budget={spec.get("local_budget", 100)}']))
        for name, script, extra in probes:
            output = directory / f'{name}_{step}.json'
            cmd = [sys.executable, '-u', str(source / 'scripts' / script),
                   f'--artifacts={artifacts}', f'--step={step}',
                   f'--dataset-root={dataset_root}', f'--output={output}', *extra]
            with (directory / f'{name}_{step}.log').open('x') as log:
                result = subprocess.run(cmd, cwd=source, env=env, stdout=log,
                                        stderr=subprocess.STDOUT, timeout=1800)
            record = dict(probe=name, step=step, command=cmd,
                          returncode=result.returncode, output=str(output))
            records.append(record)
            if result.returncode != 0:
                return dict(status='failed', records=records)
    return dict(status='completed', records=records)


def run_one(job, gpu, study, source, provenance, dataset_root, wandb_mode, stop):
    run_dir = Path(job['run_dir'])
    run_dir.mkdir(parents=True, exist_ok=False)
    cmd = command(job, study, source, provenance, wandb_mode)
    env = child_environment(source, dataset_root, gpu, job['seed'])
    metadata = dict(job, status='running', started_at=now(), physical_gpu=gpu, command=cmd,
                    source_signature=provenance['code_signature'], source_dir=str(source),
                    git_commit=provenance['git_commit'], python=sys.executable,
                    runtime_environment={k: env[k] for k in ('CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER',
                        'PYTHONPATH', 'PYTHONHASHSEED', 'OGBENCH_DATASET_DIR', 'XLA_FLAGS',
                        'XLA_PYTHON_CLIENT_PREALLOCATE', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')})
    write_json(run_dir / 'run.json', metadata)
    print(f'START GPU={gpu} {job["method"]} {job["environment"]} seed={job["seed"]}', flush=True)
    try:
        with (run_dir / 'stdout.log').open('x') as log:
            process = subprocess.Popen(cmd, cwd=source / 'impls', env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            metadata['pid'] = process.pid
            write_json(run_dir / 'run.json', metadata)
            while process.poll() is None:
                if stop.wait(0.25):
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    break
            metadata['returncode'] = process.returncode
        if stop.is_set():
            metadata['status'] = 'cancelled'
        elif process.returncode != 0:
            metadata['status'] = 'failed'
        else:
            metadata['metric'] = validate_completed(job, study)
            metadata['status'] = 'completed'
            # Keep valid training results even if an independent probe fails.
            metadata['training_finished_at'] = now()
            write_json(run_dir / 'run.json', metadata)
            try:
                metadata['diagnostics'] = run_diagnostics(job, study, source, dataset_root)
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                metadata['diagnostics'] = dict(status='failed', error=str(error))
    except Exception as error:
        metadata.update(status='failed', error=str(error))
    metadata['finished_at'] = now()
    write_json(run_dir / 'run.json', metadata)
    print(f'{metadata["status"].upper()} {run_dir}', flush=True)
    return int(metadata['status'] != 'completed'
               or metadata.get('diagnostics', {}).get('status') == 'failed')


def dispatch(jobs, gpus, runner, stop):
    pending = queue.Queue()
    failed = threading.Event()
    for job in jobs:
        pending.put(job)

    def worker(gpu):
        while not stop.is_set() and not failed.is_set():
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return 0
            try:
                if runner(job, gpu):
                    failed.set()  # Let in-flight runs finish, but do not dispatch more jobs.
                    return 1
            except Exception:
                failed.set()
                raise
            finally:
                pending.task_done()
        return 1

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(worker, gpu) for gpu in gpus]
        return int(any([future.result() for future in futures]))


def validate_comparable_campaigns(current, reference):
    """Reject pooling results whose training semantics or shared data differ."""
    for field in ('protocol', 'metric', 'common_agent'):
        if current['study'][field] != reference['study'][field]:
            raise ValueError(f'Cannot combine campaigns: different {field}')
    a, b = current['study']['environments'], reference['study']['environments']
    for environment in a.keys() & b.keys():
        if a[environment] != b[environment]:
            raise ValueError(f'Cannot combine campaigns: different recipe for {environment}')
    a, b = current['study']['methods'], reference['study']['methods']
    for method in a.keys() & b.keys():
        if any(a[method][key] != b[method][key] for key in ('agent', 'overrides')):
            raise ValueError(f'Cannot combine campaigns: different method {method}')
    a, b = current['datasets'], reference['datasets']
    for name in a.keys() & b.keys():
        if not a[name].get('sha256') or a[name]['sha256'] != b[name].get('sha256'):
            raise ValueError(f'Cannot combine campaigns: different/unhashed dataset {name}')
    def training_sources(manifest):
        return {name: value for name, value in manifest['source']['files'].items()
                if name == 'impls/main.py' or name.startswith(('impls/agents/', 'impls/utils/', 'ogbench/'))}
    if not training_sources(current) or training_sources(current) != training_sources(reference):
        raise ValueError('Cannot combine campaigns: different training/model/environment source')


def summarize(jobs, study, references=(), output=None):
    def emit(*args):
        print(*args, file=output)
    entries = [(job, study) for job in jobs]
    for reference_jobs, reference_study in references:
        entries.extend((job, reference_study) for job in reference_jobs)
    expected_counts = {}
    identities = set()
    for job, _ in entries:
        identity = (job['method'], job['environment'], job['seed'])
        if identity in identities:
            raise ValueError(f'Duplicate run identity across campaigns: {identity}')
        identities.add(identity)
        group = identity[:2]
        expected_counts[group] = expected_counts.get(group, 0) + 1
    grouped = {}
    invalid = False
    emit('method | environment | seed | ' + ' | '.join(map(str, study['metric']['steps'])) + ' | mean (%)')
    for job, run_study in sorted(entries, key=lambda item: (item[0]['environment'], item[0]['method'], item[0]['seed'])):
        path = Path(job['run_dir'])
        try:
            state = json.loads((path / 'run.json').read_text())
            if state['status'] != 'completed':
                raise ValueError(f'status={state["status"]}')
            metric = validate_completed(job, run_study)
            values = [100 * value for value in metric['values'].values()]
            emit(f'{job["method"]} | {job["environment"]} | {job["seed"]} | '
                  + ' | '.join(f'{v:.2f}' for v in values) + f' | {100 * metric["mean"]:.2f}')
            grouped.setdefault((job['method'], job['environment']), []).append(metric['mean'])
        except (OSError, ValueError, KeyError) as error:
            invalid = True
            emit(f'{job["method"]} | {job["environment"]} | {job["seed"]} | INCOMPLETE: {error}')
    emit('\nAcross-seed summaries (sample standard deviation; no SD for a single seed):')
    for (method, environment), values in grouped.items():
        expected = expected_counts[method, environment]
        if len(values) != expected:
            emit(f'{method} | {environment} | INCOMPLETE seeds={len(values)}/{expected}')
            continue
        sd = f'{100 * statistics.stdev(values):.2f}' if len(values) > 1 else 'N/A (single seed)'
        emit(f'{method} | {environment} | n={len(values)} | {100 * statistics.mean(values):.2f} | SD={sd}')
    return int(invalid)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, help='External study JSON; optional when the campaign already contains study.json.')
    parser.add_argument('--run-root', type=Path, required=True, help='An external campaign directory.')
    parser.add_argument('--source-root', type=Path, help='Separate frozen code directory, e.g. /home/eai/Research/Navigation/<campaign>.')
    parser.add_argument('--summarize-with', type=Path, action='append', default=[],
                        help='Pool with another campaign after checking protocol, source and data identity.')
    parser.add_argument('--dataset-root', type=Path, default=Path('/data/qijunrong/06-RL/offline-rl/data/raw_ogbench'))
    parser.add_argument('--gpus', default='0,1', help='Comma-separated physical GPU indices; one worker per GPU.')
    parser.add_argument('--allow-gpu-process', action='append', default=[], metavar='PID=EXECUTABLE',
                        help='Explicitly permit a verified background GPU process; exact PID and executable match required.')
    parser.add_argument('--methods', help='Comma-separated method IDs from the study.')
    parser.add_argument('--environments', help='Comma-separated complete environment IDs.')
    parser.add_argument('--attempt', type=int, default=0, help='Explicit retry identity. Never overwrites earlier attempts.')
    parser.add_argument('--wandb-mode', choices=('offline', 'online', 'disabled'), default='offline')
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--dry-run', action='store_true', help='Read-only plan; does not require a visible GPU.')
    modes.add_argument('--execute', action='store_true')
    modes.add_argument('--prepare', action='store_true', help='Hash data and freeze code without querying GPUs or starting training.')
    modes.add_argument('--summarize', action='store_true', help='Read only exact-step results; print to the terminal.')
    args = parser.parse_args(argv)
    allowed_processes = {}
    for item in args.allow_gpu_process:
        pid, sep, executable = item.partition('=')
        if not sep or not pid.isdigit() or int(pid) <= 0 or not executable.startswith('/') or pid in allowed_processes:
            raise ValueError('GPU process exception must be a unique positive PID=absolute/executable/path.')
        allowed_processes[pid] = executable
    campaign = args.run_root.resolve()
    dataset_root = args.dataset_root.resolve()
    if args.attempt < 0:
        raise ValueError('--attempt must be non-negative')
    manifest_path = campaign / 'campaign.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if args.study is None and manifest is None:
        parser.error('--study is required for a new campaign; keep study JSON files outside the source tree.')
    study = manifest['study'] if args.study is None else load_study(args.study)
    if manifest_path.exists():
        recorded = json.loads(manifest_path.read_text())['study']
        if study != recorded:
            raise ValueError('Study differs from campaign.json; use the frozen study or a new run root.')
    jobs = plan(study, campaign, args.methods, args.environments, args.attempt)
    if not jobs:
        raise ValueError('Selected method/environment combination has no declared runs.')
    if args.summarize_with and not args.summarize:
        raise ValueError('--summarize-with requires --summarize')
    if args.summarize:
        references = []
        for path in args.summarize_with:
            other = json.loads((path / 'campaign.json').read_text())
            if manifest is None:
                raise ValueError('Joint summary requires a prepared current campaign.')
            validate_comparable_campaigns(manifest, other)
            reference_jobs = [job for job in plan(other['study'], path.resolve())
                              if (args.methods is None or job['method'] in args.methods.split(','))
                              and (args.environments is None or job['environment'] in args.environments.split(','))]
            references.append((reference_jobs, other['study']))
        return summarize(jobs, study, references)
    gpus = args.gpus.split(',')
    if not gpus or len(gpus) != len(set(gpus)) or any(not re.fullmatch(r'0|[1-9][0-9]*', g) for g in gpus):
        raise ValueError('--gpus must contain unique physical GPU indices, e.g. 0,1')
    if campaign.is_relative_to(ROOT):
        raise ValueError('Place --run-root outside the development/snapshot source directory.')
    source = (Path(manifest.get('source_dir', campaign / 'source')) if manifest is not None
              else (args.source_root.resolve() if args.source_root else campaign / 'source'))
    if args.source_root is not None and args.source_root.resolve() != source.resolve():
        raise ValueError('Requested source directory differs from the frozen campaign.')
    datasets = dataset_inventory(study, dataset_root)
    pending = []
    for job in jobs:
        run_dir = Path(job['run_dir'])
        if run_dir.exists():
            state = json.loads((run_dir / 'run.json').read_text())
            if state['status'] != 'completed':
                raise ValueError(f'Existing {state["status"]} attempt: {run_dir}. '
                                 'Select explicit --attempt N and filters to retry; no automatic overwrite/resume.')
            if state.get('diagnostics', {}).get('status') == 'failed':
                raise ValueError(f'Existing diagnostic failure requires investigation before dispatch: {run_dir}')
            validate_completed(job, study)
        else:
            pending.append(job)
    print(f'Study={study["study_id"]} total={len(jobs)} completed={len(jobs)-len(pending)} pending={len(pending)}')
    print(f'GPUs={gpus}; one worker/GPU; Python={sys.executable}; W&B={args.wandb_mode}')
    print(f'Protocol={json.dumps(study["protocol"])}\nMetric={study["metric"]}')
    print(f'Datasets={len(datasets)} files; output={campaign}')
    preview = (manifest['source'] if manifest is not None
               else dict(git_commit='<recorded-at-execution>', code_signature='<source-snapshot-sha256>'))
    for job in pending:
        print(shlex.join(command(job, study, source, preview, args.wandb_mode)))
    if args.dry_run:
        print('DRY RUN: no files written, no training started; GPU availability and dataset SHA256 not checked.')
        return 0
    if not pending:
        return 0
    if not args.prepare:
        print('Checking GPU occupancy...', flush=True)
        check_gpu_occupancy(gpus, allowed_processes)
    print('Hashing datasets and freezing actual source...', flush=True)
    datasets = dataset_inventory(study, dataset_root, hash_files=True)
    source, provenance = freeze(campaign, study, datasets, args.source_root)
    print(f'Frozen source={source} Git={provenance["git_commit"]} signature={provenance["code_signature"]}', flush=True)
    if args.prepare:
        print('PREPARED: code and datasets recorded; no training started.', flush=True)
        return 0
    with (campaign / '.launcher.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('Another launcher owns this campaign.') from error
        print('Checking JAX on the assigned GPUs using frozen source...', flush=True)
        check_gpus(gpus, dataset_root, source, allowed_processes)
        write_json(campaign / 'launcher_runtime.json', dict(
            started_at=now(), launcher_pid=os.getpid(), physical_gpus=gpus,
            allowed_gpu_processes=allowed_processes))
        stop = threading.Event()
        previous = {sig: signal.signal(sig, lambda *_: stop.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            def run_and_report(job, gpu):
                check_gpu_occupancy([gpu], allowed_processes)
                result = run_one(job, gpu, study, source, provenance,
                                 dataset_root, args.wandb_mode, stop)
                # Atomically publish completed exact-checkpoint scores plus the
                # explicit incomplete queue after every run, including failures.
                with report_lock:
                    report = io.StringIO()
                    summarize(jobs, study, output=report)
                    temporary = campaign / 'results.txt.tmp'
                    temporary.write_text(report.getvalue())
                    temporary.replace(campaign / 'results.txt')
                return result
            report_lock = threading.Lock()
            return dispatch(pending, gpus, run_and_report, stop)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(2)
