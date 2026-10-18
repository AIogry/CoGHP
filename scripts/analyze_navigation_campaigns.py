#!/usr/bin/env python3
"""Audit and aggregate the three October navigation campaigns, without training.

Raw campaign directories are read-only inputs. Scores are fractions; plots use
percent and differences use percentage points. Checkpoint existence/metadata are
checked, but checkpoint payloads and dataset contents are not rehashed here.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT.parent / 'docs' / 'CoGHP' / 'experiments'
NAMES = [
    '20261009_coghp_vs_hrm_h2l2_point_ant_pilot',
    '20261010_coghp_vs_hrm_h2l2_humanoid_and_seed_completion',
    '20261013_global_position_four_tasks_seed0',
]
STEPS = [800000, 900000, 1000000]
TASKS = [f'evaluation/task{i}_success' for i in range(1, 6)]
ORDER = [f'{family}maze-{size}-navigate-v0' for family in ('point', 'ant', 'humanoid')
         for size in ('medium', 'large', 'giant')]
GP_ORDER = [ORDER[i] for i in (7, 8, 2, 5)]
METHODS = ['baseline', 'g_only', 'p_only', 'g_plus_p']


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def csv_rows(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def mean(values):
    return statistics.mean(values)


def source_scope(files):
    return {k: v for k, v in files.items() if k == 'impls/main.py'
            or k.startswith(('impls/agents/', 'impls/utils/', 'ogbench/'))}


def short(environment):
    return environment.replace('-navigate-v0', '').replace('maze-', 'Maze-').capitalize().replace('maze', 'Maze')


def analyze(base, out):
    spec = importlib.util.spec_from_file_location('study_runner', ROOT / 'scripts/run_study.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    manifests = [read_json(base / name / 'campaign.json') for name in NAMES]
    runner.validate_comparable_campaigns(manifests[0], manifests[1])
    audits, per_run, evaluations, train_metrics, values, local, configurations = [], [], [], [], [], [], []
    complete_configs, parameter_counts, source_changes = {}, [], []
    run_objects = {}
    software_records = set()
    for index, (name, manifest) in enumerate(zip(NAMES, manifests)):
        root = base / name
        source = root / 'source'
        mismatches = []
        for rel, expected in manifest['source']['files'].items():
            path = source / rel
            if not path.is_file() or digest(path) != expected:
                mismatches.append(rel)
        if mismatches:
            raise ValueError(f'Frozen source mismatch: {name}: {mismatches}')
        expected_signature = hashlib.sha256(json.dumps(manifest['source']['files'], sort_keys=True).encode()).hexdigest()
        if expected_signature != manifest['source']['code_signature']:
            raise ValueError(f'Invalid manifest source signature: {name}')
        study = manifest['study']
        jobs = runner.plan(study, root)
        actual = set((root / 'runs').glob('**/run.json'))
        planned = {Path(job['run_dir']) / 'run.json' for job in jobs}
        if actual != planned:
            raise ValueError(f'Unexpected/missing attempts: {name}: {actual ^ planned}')
        relocated = 0
        for job in jobs:
            run = Path(job['run_dir'])
            state = read_json(run / 'run.json')
            flags = read_json(run / 'artifacts/flags.json')
            repro = read_json(run / 'artifacts/reproducibility.json')
            software_records.add(json.dumps({k:repro[k] for k in ('python_version','numpy_version','jax_version','package_versions','platform','xla_flags')},sort_keys=True))
            if repro['seed'] != job['seed'] or not repro['fixed_validation_batch'] or not repro['require_deterministic_gpu']:
                raise ValueError(f'Reproducibility identity/protocol mismatch: {run}')
            checkpoint = read_json(run / 'artifacts/checkpoint_metadata.json')
            if state['status'] != 'completed' or state['returncode'] != 0:
                raise ValueError(f'Incomplete run: {run}')
            if state['source_signature'] != manifest['source']['code_signature']:
                raise ValueError(f'Source identity mismatch: {run}')
            metric = runner.validate_completed(job, study)
            if any(not math.isclose(metric['values'][str(s)], state['metric']['values'][str(s)], abs_tol=1e-12) for s in STEPS):
                raise ValueError(f'Stored summary does not match CSV: {run}')
            for key, expected in [('code_signature', state['source_signature']), ('git_commit', state['git_commit']),
                                  ('agent_name', job['agent_name']), ('environment', job['environment']), ('seed', job['seed'])]:
                if checkpoint['signature'][key] != expected:
                    raise ValueError(f'Checkpoint signature {key} mismatch: {run}')
            if checkpoint['resume'] or flags['restore_path'] is not None:
                raise ValueError(f'Unexpected resume: {run}')
            relocated += state['run_dir'] != str(run)
            env, method, seed = job['environment'], job['method'], job['seed']
            identity = (name, method, env, seed)
            key = '|'.join(map(str, identity))
            complete_configs[key] = dict(flags=flags, reproducibility=repro, checkpoint_metadata=checkpoint,
                                         actual_run_dir=str(run), recorded_run_dir=state['run_dir'])
            configuration = dict(campaign=name, method=method, environment=env, seed=seed,
                                 **flags['agent'])
            configurations.append({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                                   for k, v in configuration.items()})
            eval_rows = csv_rows(run / 'artifacts/eval.csv')
            if sorted(int(r['step']) for r in eval_rows) != list(range(100000, 1000001, 100000)):
                raise ValueError(f'Incomplete/full duplicate evaluation history: {run}')
            selected = [r for r in eval_rows if int(r['step']) in STEPS]
            for row in eval_rows:
                scores = [float(row[t]) for t in TASKS]
                if any(not math.isfinite(v) or not 0 <= v <= 1 or not math.isclose(v * 50, round(v * 50), abs_tol=1e-7) for v in scores):
                    raise ValueError(f'Invalid task success/count: {run}')
                if not math.isclose(mean(scores), float(row['evaluation/overall_success']), abs_tol=1e-12):
                    raise ValueError(f'Overall score does not equal mean of five tasks: {run}')
                evaluations.append(dict(campaign=name, method=method, environment=env, seed=seed,
                                        step=int(row['step']), overall_success=float(row['evaluation/overall_success']),
                                        **{f'task{i}_success': scores[i-1] for i in range(1,6)}))
            wall = (datetime.fromisoformat(state.get('training_finished_at', state['finished_at']))
                    - datetime.fromisoformat(state['started_at'])).total_seconds()
            record = dict(campaign=name, method=method, environment=env, seed=seed, agent_name=job['agent_name'],
                          score_800k=metric['values']['800000'], score_900k=metric['values']['900000'],
                          score_1m=metric['values']['1000000'], score_mean=metric['mean'],
                          **{f'task{i}_mean': mean(float(r[TASKS[i-1]]) for r in selected) for i in range(1,6)},
                          elapsed_hours=wall/3600, git_commit=state['git_commit'], source_signature=state['source_signature'],
                          source_was_dirty=bool(manifest['source']['git_status']), run_dir=str(run),
                          eval_csv_sha256=digest(run/'artifacts/eval.csv'), flags_sha256=digest(run/'artifacts/flags.json'),
                          train_csv_sha256=digest(run/'artifacts/train.csv'),run_json_sha256=digest(run/'run.json'),
                          reproducibility_sha256=digest(run/'artifacts/reproducibility.json'),
                          checkpoint_metadata_sha256=digest(run/'artifacts/checkpoint_metadata.json'),
                          status=state['status'], physical_gpu=state['physical_gpu'])
            per_run.append(record)
            run_objects[identity] = dict(run=run, flags=flags, repro=repro, evaluations=eval_rows)
            rows = csv_rows(run/'artifacts/train.csv')
            if sorted(int(r['step']) for r in rows) != list(range(5000,1000001,5000)):
                raise ValueError(f'Incomplete/duplicate training log: {run}')
            for row in rows:
                for column, value in row.items():
                    if not math.isfinite(float(value)):
                        raise ValueError(f'Nonfinite training metric {column}: {run}')
            tail = [r for r in rows if 800000 <= int(r['step']) <= 1000000]
            for column in rows[0]:
                if column != 'step':
                    series = [float(r[column]) for r in tail]
                    train_metrics.append(dict(campaign=name,method=method,environment=env,seed=seed,
                                              metric=column,window='800k_to_1m_inclusive_41_logs',mean=mean(series),
                                              minimum=min(series),maximum=max(series),at_1m=float(rows[-1][column])))
            if (run/'artifacts/model_parameter_counts.json').exists():
                counts = read_json(run/'artifacts/model_parameter_counts.json')
                parameter_counts.append(dict(campaign=name,method=method,environment=env,seed=seed,
                                             total_stored_parameters=counts['total_stored_parameters'], **counts['by_group']))
            if index == 2:
                if state.get('diagnostics',{}).get('status') != 'completed':
                    raise ValueError(f'Diagnostics incomplete: {run}')
                expected_probes = 6 if method in ('p_only','g_plus_p') else 3
                if len(state['diagnostics']['records']) != expected_probes:
                    raise ValueError(f'Diagnostic record count mismatch: {run}')
                for step in STEPS:
                    q = run/'diagnostics'/f'values_{step}.json'
                    v = read_json(q)
                    if v['checkpoint_step']!=step or v['artifacts']!=str(run/'artifacts') or v['samples']!=2048:
                        raise ValueError(f'Value probe identity mismatch: {q}')
                    values.append(dict(campaign=name,method=method,environment=env,seed=seed,step=step,
                                       all_count=v['all_pairs']['count'],all_sign_agreement=v['all_pairs']['sign_agreement'],
                                       distant_count=v['distant_pairs']['count'],distant_sign_agreement=v['distant_pairs']['sign_agreement'],
                                       cap_fraction=mean(v['weight_cap_fraction']),weight_mean=mean(v['weight_mean']),
                                       wall_cell_fraction=v.get('decoded_waypoint_wall_cell_fraction'),
                                       **{f'value_offset_{offset}_mean': item['mean'] for offset,item in v['behavior_goal_offset_value_slices'].items()},
                                       source_json=str(q),source_sha256=digest(q)))
                    if method in ('p_only','g_plus_p'):
                        q = run/'diagnostics'/f'commands_{step}.json'
                        v = read_json(q)
                        if v['checkpoint_step']!=step or v['artifacts']!=str(run/'artifacts') or v['budget']!=100 or len(v['indices'])!=24:
                            raise ValueError(f'Local probe identity mismatch: {q}')
                        if any(r['restore_observation_max_error'] > 2e-4 for r in v['episodes']):
                            raise ValueError(f'Local state restoration mismatch: {run}')
                        for mode in ('teacher','predicted'):
                            episodes = [r for r in v['episodes'] if r['mode']==mode]
                            if len(episodes)!=24 or not math.isclose(mean(r['success'] for r in episodes),v['summary'][mode]['success_rate'],abs_tol=1e-12):
                                raise ValueError(f'Local summary does not match episode records: {q}')
                            local.append(dict(campaign=name,method=method,environment=env,seed=seed,step=step,mode=mode,
                                              successes=sum(r['success'] for r in episodes),episodes=len(episodes),
                                              success_rate=v['summary'][mode]['success_rate'],
                                              initial_distance=mean(r['initial_distance'] for r in episodes),
                                              final_distance=mean(r['final_distance'] for r in episodes),
                                              mean_steps=v['summary'][mode]['mean_steps'],
                                              max_restore_error=max(r['restore_observation_max_error'] for r in episodes),
                                              indices=json.dumps(v['indices']),source_json=str(q),source_sha256=digest(q)))
        timestamps = [read_json(p) for p in actual]
        audits.append(dict(campaign=name,runs=len(jobs),completed=len(jobs),frozen_files_verified=len(manifest['source']['files']),
                           source_signature=manifest['source']['code_signature'],git_commit=manifest['source']['git_commit'],
                           source_was_dirty=bool(manifest['source']['git_status']),
                           relocated_run_metadata=relocated,manifest_sha256=digest(root/'campaign.json'),
                           started_at_shanghai=min(datetime.fromisoformat(r['started_at']) for r in timestamps).astimezone(timezone(timedelta(hours=8))).isoformat(),
                           finished_at_shanghai=max(datetime.fromisoformat(r['finished_at']) for r in timestamps).astimezone(timezone(timedelta(hours=8))).isoformat()))
    # Validate legacy pooling and preserve the independent GP baseline reruns.
    old = [r for r in per_run if r['campaign'] in NAMES[:2]]
    gp = [r for r in per_run if r['campaign']==NAMES[2]]
    groups = defaultdict(list)
    for r in old:
        groups[r['method'],r['environment']].append(r)
    old_summary, paired = [], []
    for (method,env), rows in sorted(groups.items()):
        if sorted(r['seed'] for r in rows) != [0,1,2]:
            raise ValueError(f'Incorrect pooled seed set: {method} {env}')
        old_summary.append(dict(method=method,environment=env,n=3,score_mean=mean(r['score_mean'] for r in rows),
                                seed_sample_sd=statistics.stdev(r['score_mean'] for r in rows),
                                **{f'seed_{r["seed"]}_score':r['score_mean'] for r in rows},
                                **{f'task{i}_mean':mean(r[f'task{i}_mean'] for r in rows) for i in range(1,6)},
                                elapsed_hours_mean=mean(r['elapsed_hours'] for r in rows)))
    for env in ORDER:
        by_method={m:{r['seed']:r for r in groups[m,env]} for m in ('coghp','hrm_h2l2_truncated')}
        delta=[by_method['hrm_h2l2_truncated'][s]['score_mean']-by_method['coghp'][s]['score_mean'] for s in range(3)]
        paired.append(dict(environment=env,mean_difference=mean(delta),paired_seed_sample_sd=statistics.stdev(delta),
                           seeds_improved=sum(d>0 for d in delta),**{f'seed_{s}_difference':d for s,d in enumerate(delta)}))
    factors, bridge = [], []
    for env in GP_ORDER:
        r={x['method']:x for x in gp if x['environment']==env}
        b,g,p,gp_score=[r[m]['score_mean'] for m in METHODS]
        factors.append(dict(environment=env,baseline=b,g_only=g,p_only=p,g_plus_p=gp_score,
                            g_at_full_state=g-b,p_at_legacy=p-b,g_at_position=gp_score-p,
                            p_at_global=gp_score-g,interaction=gp_score-g-p+b))
        historic=next(x for x in old if x['environment']==env and x['method']=='hrm_h2l2_truncated' and x['seed']==0)
        current=r['baseline']
        a=run_objects[(historic['campaign'],'hrm_h2l2_truncated',env,0)]
        c=run_objects[(NAMES[2],'baseline',env,0)]
        differences={k:[a['flags']['agent'].get(k),c['flags']['agent'].get(k)] for k in a['flags']['agent'].keys()|c['flags']['agent'].keys()
                     if a['flags']['agent'].get(k)!=c['flags']['agent'].get(k)}
        bridge.append(dict(environment=env,historical_seed0=historic['score_mean'],new_baseline_seed0=current['score_mean'],
                           difference=current['score_mean']-historic['score_mean'],
                           full_eval_history_identical=a['evaluations']==c['evaluations'],
                           reproducibility_equal_except_gpu={k:v for k,v in a['repro'].items() if k!='cuda_visible_devices'}==
                                                            {k:v for k,v in c['repro'].items() if k!='cuda_visible_devices'},
                           configuration_differences=differences))
    a,b=manifests[1]['source']['files'],manifests[2]['source']['files']
    for k in sorted(a.keys()|b.keys()):
        if a.get(k)!=b.get(k):
            source_changes.append(dict(file=k,status='added' if k not in a else 'removed' if k not in b else 'changed',
                                       old_sha256=a.get(k),new_sha256=b.get(k)))
    data_checks=[]
    for i in range(3):
        for j in range(i+1,3):
            common=manifests[i]['datasets'].keys()&manifests[j]['datasets'].keys()
            mismatched=[k for k in common if manifests[i]['datasets'][k]['sha256']!=manifests[j]['datasets'][k]['sha256']]
            if mismatched:raise ValueError(f'Dataset hashes differ: {mismatched}')
            data_checks.append(dict(a=NAMES[i],b=NAMES[j],shared_dataset_files=len(common),recorded_sha256_equal=True))
    prior=csv_rows(EXPERIMENTS/'navigation_results_20261012.csv')
    # The earlier consolidated CSV is checked separately from primary raw data.
    current_index={(r['campaign'],r['method'],r['environment'],r['seed']):r for r in old}
    if len(prior)!=len(current_index):
        raise ValueError('Earlier consolidated CSV has the wrong number of runs')
    for row in prior:
        item=current_index[(row['campaign_id'],row['method'],row['environment'],int(row['seed']))]
        for previous,current in [('mean_success_rate','score_mean'),('success_rate_800000','score_800k'),
                                 ('success_rate_900000','score_900k'),('success_rate_1000000','score_1m')]:
            if not math.isclose(float(row[previous]),item[current],abs_tol=1e-12):
                raise ValueError(f'Earlier consolidation score mismatch: {item["run_dir"]}')
        if row['eval_csv_sha256']!=item['eval_csv_sha256'] or row['flags_json_sha256']!=item['flags_sha256']:
            raise ValueError(f'Earlier consolidation hash mismatch: {item["run_dir"]}')
    prior_check=dict(rows=len(prior),scores_and_recorded_csv_config_hashes_match=True)
    global_pairs=[]
    local_pairs=[]
    for env in GP_ORDER:
        a=run_objects[(NAMES[2],'g_only',env,0)]['run']
        b=run_objects[(NAMES[2],'g_plus_p',env,0)]['run']
        ar,br=csv_rows(a/'artifacts/train.csv'),csv_rows(b/'artifacts/train.csv')
        columns=[k for k in ar[0] if '/global/' in k]
        identical=all(x[k]==y[k] for x,y in zip(ar,br) for k in columns)
        if not identical:raise ValueError(f'Global training metrics unexpectedly differ: {env}')
        global_pairs.append(dict(environment=env,global_metric_columns=len(columns),training_rows=len(ar),exactly_equal=True))
        a=run_objects[(NAMES[2],'p_only',env,0)]['run']
        for step in STEPS:
            x,y=read_json(a/'diagnostics'/f'commands_{step}.json'),read_json(b/'diagnostics'/f'commands_{step}.json')
            same=x['indices']==y['indices']
            if not same:raise ValueError(f'Local starts are not paired: {env} {step}')
            local_pairs.append(dict(environment=env,step=step,identical_start_indices=True,episodes=len(x['indices'])))
    audit=dict(created_date='2026-10-14',campaigns=audits,total_runs=len(per_run),
               legacy_runs=len(old),global_position_runs=len(gp),formal_eval_rows=len(evaluations),
               checkpoint_files_checked=3*len(per_run),value_probes=len(values),local_probe_summaries=len(local),
               old_campaign_training_source_identical=source_scope(manifests[0]['source']['files'])==source_scope(manifests[1]['source']['files']),
               old_training_source_files=len(source_scope(manifests[0]['source']['files'])),
               software_record_variants=len(software_records),
               shared_dataset_manifests=data_checks,baseline_bridge=bridge,earlier_consolidation=prior_check,
               independent_global_metrics=global_pairs,local_paired_starts=local_pairs,
               boundary='Dataset/checkpoint payloads were not rehashed or deserialized. Dataset identity uses stored SHA256; checkpoint checks cover nonempty files and semantic metadata.',
               metric=dict(per_run='mean at exact 800k/900k/1m',legacy_seeds=[0,1,2],sample_sd_ddof=1,
                           gp_seeds=[0],unit='fraction',difference_unit='fraction; multiply by 100 for percentage points'))
    if len(software_records)!=1:
        raise ValueError('Software/runtime records differ across runs')
    for r in per_run:
        for s in STEPS:
            if not (Path(r['run_dir'])/'artifacts'/f'params_{s}.pkl').stat().st_size:
                raise ValueError(f'Empty checkpoint: {r["run_dir"]}')
    out.mkdir(parents=True,exist_ok=True)
    outputs={'runs.csv':per_run,'evaluations.csv':evaluations,'legacy_summary.csv':old_summary,'legacy_paired_differences.csv':paired,
             'factorial_effects.csv':factors,'train_metrics.csv':train_metrics,'value_diagnostics.csv':values,
             'local_execution_diagnostics.csv':local,'resolved_agent_configs.csv':configurations,'stored_parameter_counts.csv':parameter_counts,
             'source_changes.csv':source_changes}
    for name,rows in outputs.items():write_csv(out/name,rows)
    write_json(out/'audit.json',audit)
    write_json(out/'resolved_configs.json',complete_configs)
    write_json(out/'campaign_manifests.json',dict(zip(NAMES,manifests)))
    print(json.dumps(dict(audit=audits,legacy=old_summary,factorial=factors,bridge=bridge),ensure_ascii=False,indent=2))
    return dict(runs=per_run,evaluations=evaluations,legacy=old_summary,factors=factors,values=values,local=local)


def plots(data,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160})
    colors={'coghp':'#64748b','hrm_h2l2_truncated':'#2563eb','baseline':'#64748b','g_only':'#0891b2','p_only':'#d97706','g_plus_p':'#7c3aed'}
    labels={'coghp':'Original CoGHP','hrm_h2l2_truncated':'CoGHP + HRM action','baseline':'HRM baseline','g_only':'G only','p_only':'P only','g_plus_p':'G + P'}
    fig,ax=plt.subplots(figsize=(11.5,4.4),layout='constrained')
    x=np.arange(9)
    for i,m in enumerate(('coghp','hrm_h2l2_truncated')):
        rows=[next(r for r in data['legacy'] if r['method']==m and r['environment']==e) for e in ORDER]
        ax.bar(x+(i-.5)*.35,[100*r['score_mean'] for r in rows],.35,yerr=[100*r['seed_sample_sd'] for r in rows],
               color=colors[m],label=labels[m],capsize=3,alpha=.92)
    ax.set_xticks(x,[short(e) for e in ORDER],rotation=28,ha='right');ax.set_ylim(0,105)
    ax.set_ylabel('Success (%)');ax.set_title('October 9 + 10: exact 800k / 900k / 1m mean, 3 seeds; error bars = sample SD')
    ax.legend(loc='upper right');ax.grid(axis='y',alpha=.15)
    fig.savefig(out/'legacy_comparison.png');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11.5,7),layout='constrained')
    for ax,e in zip(axes.flat,GP_ORDER):
        for m in METHODS:
            rows=sorted([r for r in data['evaluations'] if r['campaign']==NAMES[2] and r['method']==m and r['environment']==e],key=lambda r:r['step'])
            ax.plot([r['step']/1000 for r in rows],[100*r['overall_success'] for r in rows],marker='o',markersize=3,color=colors[m],label=labels[m])
        ax.axvspan(800,1000,color='#e2e8f0',alpha=.35);ax.set_ylim(0,100);ax.set_title(short(e));ax.set_xlabel('Training updates (thousands)');ax.set_ylabel('Success (%)');ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=9,loc='upper left');fig.suptitle('October 13: seed 0 learning curves; shaded window defines the primary score')
    fig.savefig(out/'global_position_learning_curves.png');plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(13,3.7),layout='constrained')
    for ax,e in zip(axes,GP_ORDER):
        rows=[next(r for r in data['runs'] if r['campaign']==NAMES[2] and r['method']==m and r['environment']==e) for m in METHODS]
        scores=[100*r['score_mean'] for r in rows]
        bars=ax.bar(range(4),scores,color=[colors[m] for m in METHODS])
        for bar,s in zip(bars,scores):ax.text(bar.get_x()+bar.get_width()/2,s+1,f'{s:.1f}',ha='center',fontsize=9)
        ax.set_xticks(range(4),['Base','G','P','G+P']);ax.set_ylim(0,100);ax.set_title(short(e));ax.grid(axis='y',alpha=.15)
    axes[0].set_ylabel('Success (%)');fig.suptitle('October 13 factorial comparison: 800k / 900k / 1m mean, seed 0 (no seed SD)')
    fig.savefig(out/'global_position_comparison.png');plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(12,9),layout='constrained')
    for ax,e in zip(axes.flat,ORDER):
        for m in ('coghp','hrm_h2l2_truncated'):
            series=[]
            for seed in range(3):
                rows=sorted([r for r in data['evaluations'] if r['campaign'] in NAMES[:2] and r['method']==m and r['environment']==e and r['seed']==seed],key=lambda r:r['step'])
                series.append([100*r['overall_success'] for r in rows])
            matrix=np.asarray(series);x=np.arange(100,1001,100)
            ax.plot(x,matrix.mean(0),color=colors[m],label=labels[m]);ax.fill_between(x,matrix.mean(0)-matrix.std(0,ddof=1),matrix.mean(0)+matrix.std(0,ddof=1),color=colors[m],alpha=.13)
        ax.set_ylim(0,100);ax.set_title(short(e));ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=8);fig.supxlabel('Training updates (thousands)');fig.supylabel('Success (%)');fig.suptitle('Legacy learning curves: mean over 3 seeds, shaded sample SD')
    fig.savefig(out/'legacy_learning_curves.png');plt.close(fig)


def report(data, out):
    """Write a Chinese research report with tables generated from audited data."""
    audit = read_json(out/'audit.json')
    config = read_json(out/'resolved_configs.json')
    train = csv_rows(out/'train_metrics.csv')
    counts = csv_rows(out/'stored_parameter_counts.csv')
    def table(headers, rows):
        return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join('---' for _ in headers)+' |']
                         +['| '+' | '.join(map(str,row))+' |' for row in rows])
    def pct(x): return f'{100*float(x):.2f}'
    def pp(x): return f'{100*float(x):+.2f}'
    def highlight(value,score,ranks):
        if score==ranks[0]:return f'**★ {value}**'
        if len(ranks)>1 and score==ranks[1]:return f'***☆ {value}***'
        return value
    def label(e):
        return e.replace('-navigate-v0','').replace('pointmaze','PointMaze').replace('antmaze','AntMaze').replace('humanoidmaze','HumanoidMaze').replace('-medium',' Medium').replace('-large',' Large').replace('-giant',' Giant')
    def metric(e,m,k):
        return float(next(r['mean'] for r in train if r['campaign']==NAMES[2] and r['environment']==e and r['method']==m and r['metric']==k))
    def ref(name,label_text=None): return f'[{label_text or name}]({out/name})'
    legacy_table=table(['环境','CoGHP 均值 ± SD','HRM action 均值 ± SD','HRM 减 CoGHP','提升种子数'],[
        [label(e),highlight(f'{pct(a["score_mean"])} ± {pct(a["seed_sample_sd"])}',a['score_mean'],sorted({a['score_mean'],b['score_mean']},reverse=True)),
         highlight(f'{pct(b["score_mean"])} ± {pct(b["seed_sample_sd"])}',b['score_mean'],sorted({a['score_mean'],b['score_mean']},reverse=True)),pp(b['score_mean']-a['score_mean']),
         f'{sum(next(r for r in data["runs"] if r["campaign"] in NAMES[:2] and r["method"]=="hrm_h2l2_truncated" and r["environment"]==e and r["seed"]==s)["score_mean"] > next(r for r in data["runs"] if r["campaign"] in NAMES[:2] and r["method"]=="coghp" and r["environment"]==e and r["seed"]==s)["score_mean"] for s in range(3))}/3']
        for e in ORDER
        for a in [next(r for r in data['legacy'] if r['environment']==e and r['method']=='coghp')]
        for b in [next(r for r in data['legacy'] if r['environment']==e and r['method']=='hrm_h2l2_truncated')]])
    gp_table=table(['环境','HRM Baseline','G only','P only','G 加 P'],[
        [label(e)]+[highlight(pct(score),score,sorted({r['score_mean'] for r in data['runs'] if r['campaign']==NAMES[2] and r['environment']==e},reverse=True))
                    for m in METHODS for score in [next(r for r in data['runs'] if r['campaign']==NAMES[2] and r['environment']==e and r['method']==m)['score_mean']]] for e in GP_ORDER])
    factor_table=table(['环境','G 在原 command 下','P 在 legacy Value 下','G 在 P 下','P 在 G 下','G 与 P 交互'],[
        [label(r['environment'])]+[pp(r[k]) for k in ['g_at_full_state','p_at_legacy','g_at_position','p_at_global','interaction']] for r in data['factors']])
    checkpoint_table=table(['环境','条件','800k','900k','1M','三点均值'],[
        [label(e),m,pct(r['score_800k']),pct(r['score_900k']),pct(r['score_1m']),pct(r['score_mean'])]
        for e in GP_ORDER for m in METHODS
        for r in [next(x for x in data['runs'] if x['campaign']==NAMES[2] and x['environment']==e and x['method']==m)]])
    task_table=table(['环境','条件','Task 1','Task 2','Task 3','Task 4','Task 5'],[
        [label(e),m]+[pct(r[f'task{i}_mean']) for i in range(1,6)] for e in GP_ORDER for m in METHODS
        for r in [next(x for x in data['runs'] if x['campaign']==NAMES[2] and x['environment']==e and x['method']==m)]])
    recipes=[]
    for e in ORDER:
        c=next(v['flags']['agent'] for k,v in config.items() if k.startswith(NAMES[0]+'|') or k.startswith(NAMES[1]+'|') if v['flags']['env_name']==e)
        recipes.append([label(e),c['num_subgoals'],c['subgoal_steps'],c['feature_dim'],f'{c["lr"]:.0e}',c['discount']])
    recipe_table=table(['环境','K','subgoal steps','latent dim','学习率','legacy discount'],recipes)
    value_rows=[]
    for e in GP_ORDER:
        a=[r for r in data['values'] if r['environment']==e and r['method']=='baseline']
        b=[r for r in data['values'] if r['environment']==e and r['method']=='g_only']
        value_rows.append([label(e),a[0]['distant_count'],pct(mean(r['distant_sign_agreement'] for r in a)),
                           pct(mean(r['distant_sign_agreement'] for r in b)),pct(mean(r['cap_fraction'] for r in a)),pct(mean(r['cap_fraction'] for r in b))])
    value_table=table(['环境','远距离有效 pair 数','Baseline 符号一致率','G 符号一致率','Baseline cap 比例','G cap 比例'],value_rows)
    position_table=table(['环境','条件','归一化重建 MSE','teacher 位置误差','predicted 位置误差','predicted cycle MSE','落墙比例'],[
        [label(e),m,f'{metric(e,m,"validation/position/reconstruction_mse"):.6f}',
         f'{metric(e,m,"validation_position/teacher/position_error"):.3f}',f'{metric(e,m,"validation_position/predicted/position_error"):.3f}',
         f'{metric(e,m,"validation_position/predicted/cycle_mse"):.5f}',
         pct(mean(r['wall_cell_fraction'] for r in data['values'] if r['environment']==e and r['method']==m))]
        for e in GP_ORDER for m in ('p_only','g_plus_p')])
    local_table=table(['环境','条件','teacher 达到参考 waypoint','predicted 达到参考 waypoint'],[
        [label(e),m]+[f'{next(r for r in data["local"] if r["environment"]==e and r["method"]==m and r["mode"]==mode and r["step"]==1000000)["successes"]}/24' for mode in ('teacher','predicted')]
        for e in GP_ORDER for m in ('p_only','g_plus_p')])
    mse_table=table(['环境','Baseline teacher MSE','Baseline predicted MSE','G teacher MSE','G predicted MSE'],[
        [label(e)]+[f'{metric(e,m,"validation_action/"+mode+"/mse"):.5f}' for m in ('baseline','g_only') for mode in ('teacher','predicted')] for e in GP_ORDER])
    drift_table=table(['环境','条件','每 5000 步 latent drift MSE','低层与重建 encoder 梯度范数均值比'],[
        [label(e),m,f'{metric(e,m,"position_drift/mse_since_previous_log"):.5f}',
         f'{metric(e,m,"gradient/low/position_encoder")/metric(e,m,"gradient/position/position_encoder"):.1f}']
        for e in GP_ORDER for m in ('p_only','g_plus_p')])
    param_table=table(['环境','Baseline','G only','P only','G 加 P'],[
        [label(e)]+[f'{int(next(r for r in counts if r["environment"]==e and r["method"]==m)["total_stored_parameters"]):,}' for m in METHODS] for e in GP_ORDER])
    bridge_table=table(['环境','原始 CoGHP seed 0','历史 HRM seed 0','新 Baseline seed 0','新旧 10 次评估一致'],[
        [label(e),pct(next(r for r in data['runs'] if r['campaign'] in NAMES[:2] and r['method']=='coghp' and r['environment']==e and r['seed']==0)['score_mean']),
         pct(b['historical_seed0']),pct(b['new_baseline_seed0']),'是']
        for e in GP_ORDER for b in [next(r for r in audit['baseline_bridge'] if r['environment']==e)]])
    from navigation_literature import section, design_section
    literature_section=section(data,out)
    evolution_section=design_section()
    text=f'''# CoGHP 导航实验与文献基线配置结果和诊断分析

整理日期：2026 年 10 月 14 日。三组实验共完成 70 个训练 run：原始 CoGHP 与 HRM-action 的三种子对照共 54 个，Global value 和 position command 的四条件 seed 0 实验共 16 个。

现有结果支持三个主要判断。HRM-action 的收益依赖环境：AntMaze Giant 的三个种子均提升，PointMaze Giant 的三个种子均下降。独立 Global value 在 Humanoid Large 和 Giant 上产生很大的 seed 0 提升，但没有带来普遍收益。Position command 的收益也依赖环境，且与 Global value 存在明显交互；准确重建数据中的位置指令，并不足以保证 planner 生成的指令有效或闭环导航成功。

本地实验表使用统一主指标，三种子结果与单种子结果分别报告；新一轮 Baseline 是 **CoGHP 加 HRM action**。新增文献参照覆盖六种 OGBench 基线、OTA、LAVL 和 CoGHP，另行标明来源与统计口径。原始全 vanilla CoGHP 的本地结果与论文结果分别显示，文献与本地种子不合并。

## 实验范围与版本对应

| 实验组 | 实际训练范围 | 完成 run | 上海时间开始 | 上海时间结束 |
| --- | --- | --- | --- | --- |
| 10 月 9 日 Point 和 Ant pilot | Point 与 Ant 各 Medium Large Giant；CoGHP seeds 0 1 2，HRM seed 0 | 24 | 10 月 9 日 22:39 | 10 月 10 日 12:56 |
| 10 月 10 日 Humanoid 与种子补全 | Humanoid 三个规模、两方法、三种子共 18；补 Point 与 Ant 的 HRM seeds 1 2 共 12 | 30 | 10 月 10 日 14:09 | 10 月 11 日 21:56 |
| 10 月 13 日 Global 与 position | Humanoid Large Giant、Point Giant、Ant Giant；四条件；仅 seed 0 | 16 | 10 月 13 日 00:46 | 10 月 14 日 01:32 |

前两组可以合并成九环境、两方法、每组合三个种子。它们的训练、模型、环境源文件 140 个逐项一致，公共数据文件的已记录 SHA256 一致，协议和公共环境配置一致；差异来自启动、文档、测试和扩展实验配置。两组满足当前统计口径下的合并条件。

| 实验组 | 启动时 Git 基准 | 实际代码状态 | 冻结代码 SHA256 |
| --- | --- | --- | --- |
| 10 月 9 日 | `4f5242ff9f76` | 有未提交修改，采用冻结内容快照 | `9931309998ed4798c2c5c5ecafa9482d49d040e4cce6f73ab4565f012f04036e` |
| 10 月 10 日 | `4f5242ff9f76` | 有未提交修改，采用冻结内容快照 | `c034a8974e40ec439deb10a2c69f8142c2bc31ca95b0875a9c6fcf4ce35179a0` |
| 10 月 13 日 | `17e819fc2267` | 启动时 clean，采用冻结内容快照 | `56e69af79d1d2d89f769f9df88b23da355b46f0d1b03ac58cbb28695477e5c20` |

前两组的完整快照 hash 不同，但上述 140 个训练相关文件相同。不能只用旧 Git 基准识别当时运行的代码。第三组引入了新 agent、sampler 和诊断实现，其 Baseline 在代码中委托历史 HRM 路径。

### 新旧 HRM Baseline 的实测衔接

{bridge_table}

一致性覆盖每个环境的全部 10 次评估、每次五个 task 与 overall，而非仅最后三点均值。随机流和软件记录也一致，允许物理 GPU 编号不同。这验证了本轮 Baseline 的衔接；重复 seed 0 run 不作为第四个独立种子，也不混入三种子 SD。

10 月 9 日目录曾从 `CoGHPNavigation_20261009` 迁移，24 个 `run.json` 内保留旧绝对路径。表格的实际路径以当前提供的目录为准，旧原始元数据保留。

## 统一评估和训练配置

每个 run 训练 1,000,000 updates，batch 256；每 5,000 步记录 train 和固定 validation batch；每 100,000 步评估全部五个 task，每 task 50 episodes，温度 0，无额外动作噪声，CPU 评估。保留 800k、900k、1M checkpoint。每个评估点的 overall 是五个 task success 的算术均值。

单个种子的主分数为 `S = (success_800k + success_900k + success_1m) / 3`。前两组报告三个 S 的均值和样本标准差，`ddof=1`。第三组只有 seed 0，不计算跨种子 SD。800k、900k、1M 是同一训练轨迹上的相关检查点，不能当成三个独立训练种子；单 task 的 50 episodes 也不能替代跨训练种子的重复。

分数以百分数显示，差值使用百分点。末次 1M、最佳 checkpoint 或最后五次均值均未替代主指标。每个 run 都有独立初始化、训练采样、validation、环境和评估随机流，启用了固定 validation batch 与确定性 GPU 设置。

公共算法设置：expectile 0.7，target EMA tau 0.005，高层链折扣 0.8，高低层 inverse temperature 均为 3，AWR 权重上限 100。Actor、legacy Value 和旧 goal encoder 的隐藏层为 512 512 512；state-based 输入，未启用视觉 encoder 或 frame stacking。原始 planner 为一个 Mixer block，token/channel hidden 为 32，command latent 的维数随环境调整。

{recipe_table}

九个任务使用对应 `*-navigate-v0` 的 train 和 validation NPZ。第三组继承重叠环境相同配方，没有为 G 或 P 单独调参。运行软件记录一致：Python 3.11.15，NumPy 2.4.6，JAX/JAXlib 0.10.2，Flax 0.12.7，Optax 0.2.8，MuJoCo 3.10.0，Gymnasium 1.3.0。

## 模型与训练目标

{evolution_section}

### 原始 CoGHP 和 HRM action

原始 CoGHP 使用 full-state command encoder：将当前状态 s 与未来完整状态 w 拼接，经 MLP 得到长度归一化 latent。planner 自回归预测从较远到较近的 K 个 subgoal latent；动作分支使用当前状态、最终 goal 和完整 subgoal 链。训练时使用数据 subgoal 的 teacher forcing，评估时使用 planner 自己生成的链。

本地原始配置 `separate_action_mixer=false`：planner 与动作预测复用同一组原始 GELU 残差 token/channel Mixer 参数。`share_mixer_weights=false` 控制深度方向的权重共享，不意味着 planner 和 executor 已分开。

HRM-action 保留原始类型的 vanilla Mixer planner，为动作分支引入独立 HRM core 与 query token；当前状态特征和 goal representation 仍有共享。HRM 设置为 H=2、L=2、一层、SwiGLU/RMSNorm、intermediate 256、float32、causal，且 `action_hrm_bptt=false`。一次动作前向有四次 L 更新和两次 H 更新，共六次内部 block 调用；历史状态在最终 L/H 对之前截断梯度，最终一对可微。每个环境步重新初始化 transient carry，没有跨环境步的循环记忆、adaptive halting 或搜索。

因此前两组检验的是 **整体 HRM-action 改造**，同时涉及 core 独立化、block 家族、参数量和计算量。它们没有参数匹配的 feedforward 对照，不能把结果全部归因于递归本身。保留 vanilla planner 类型也不表示其训练后的参数与原始 CoGHP 相同，动作分支和共享特征的梯度会影响联合学习。

旧高层训练权重由 `mean_i V_i(w_j,g) - mean_i V_i(s,g)` 产生，取 `min(exp(3 A_j),100)`；高层 loss 按链位置乘 `0.8^(K-j-1)/subgoal_steps`，并对 K 平均。低层 AWR 使用下一步相对于最近完整未来 subgoal 的 legacy Value 进度。高低层都保留各自数据监督目标。

### G 的实际含义

G 增加独立双头 bilinear Global potential：`Phi_i(s,q)=u_i(s)^T c_i(q)/sqrt(128)`，当前状态输入仍是完整状态，goal 只取绝对 XY。state 与 goal encoder 隐藏层均为 512 512 512，带 LayerNorm；整个 twin target 从 online 拷贝并做 EMA，不与 legacy Value 共用参数。

Global 使用最长 25 步的同轨迹片段。offset 0 至 25 内第一次达到 XY goal 容差时 reward=0、mask=0；完整未成功片段 reward=-1、mask=1；未成功的不完整片段不进入 loss，成功的不完整片段保留。它复用 legacy value goal 的当前/未来/随机混合 0.2/0.5/0.3，不消耗额外 sampler RNG。容差为 Point 1.0，Ant/Humanoid 0.5。

目标为 `y_i = r + 0.99 m target_Phi_i(endpoint,q)`，使用 legacy 双头 expectile 的符号约定。这里只按片段应用一次 0.99，未采用 `0.99^duration`。所以 G 同时改变 goal 语义、函数结构、奖励时间尺度和高层 AWR 权重分布；不能解释为只替换 Value 输入。

G 只把高层权重来源换成 `mean_i Phi_i(w_j,q_g)-mean_i Phi_i(s,q_g)`。旧 Value 继续训练，并继续供应低层权重。Global 不在推理时做搜索、候选筛选或重排；正式 rollout 仍由已学的 planner 和 executor 生成动作。

### P 的实际含义

P 使用 `r=(q-XY(s))/sigma`，`z=phi(s,r)`，sigma 是所有 TRAIN observations 的 XY population std，带 1e-6 下限。输入仍包含当前完整状态 s；command 不再含未来状态的身体分量。encoder/decoder 隐藏层均为 128 128，latent dim 仍为 Point 32、其他 128，latent 长度归一化到 sqrt(dim)。原始二维位移本身未归一化成单位方向。

decoder 重建相对位置 r，辅助 MSE 系数为 1，覆盖最终 goal 和全部 K 个 subgoal。高层 targets、teacher prefix 和最终 goal command 对 position encoder 均 stop-gradient；低层 action loss 与重建 loss 可更新 position encoder，decoder 仅受重建 loss。共享当前状态 feature embedding 仍同时接收高低层梯度。旧 goal encoder 保留用于 legacy critic，但不再承担原来的 Actor command 耦合。

因此 P 是输入语义、representation 学习、辅助 loss 和梯度协议的组合改动。PointMaze observation 已经只有 XY，P 在这个环境不涉及删除未来姿态信息，主要检验新的 command 学习和梯度协议。Ant/Humanoid 的低层 AWR 权重仍使用完整未来状态，P 没有把整个训练过程都改为纯位置条件。

四条件对应：Baseline 为 full-state command 加 legacy Value；G only 为 full-state command 加 Global 高层权重；P only 为 position command 加 legacy Value；G 加 P 同时启用两项。总 loss 是旧 Value、高层和低层 loss，加启用条件下的 Global loss 与 position reconstruction loss。

### 存储参数与可比性

{param_table}

这些是运行时记录的存储参数数目，包含 target 副本和保留的 legacy 模块，不是全部活跃可训练参数或前向 FLOPs。G 额外增加约 238 万至 245 万 online Global 参数及等量 target；P 增加约 4.3 万至 7.7 万 encoder/decoder 参数。第三组的 Actor core 结构在四条件间固定，但完整模型容量和训练工作量不同。逐模块计数见 {ref('stored_parameter_counts.csv')}。

## 原始 CoGHP 与 HRM action 的三种子结果

以下均为百分数，差值为百分点，SD 反映训练种子差异。
**★** 标记本表同环境两方法中最高均值，***☆*** 标记次高；仅表示均值排序。

{legacy_table}

![三种子对照结果]({out/'legacy_comparison.png'})

AntMaze Giant 是整体 HRM-action 最一致的正向结果：71.91 提升到 79.96，增加 8.04 个百分点，三个种子分别增加 11.07、7.20、5.87；五个 task 的种子均值都提升。Ant Large 也在三个种子上均提升 1.20、4.13、2.40。Ant Medium 已接近饱和，0.80 的均值差较小。

PointMaze Giant 是主要反例：56.18 降到 41.91，下降 14.27 个百分点，三个配对种子都下降，但幅度为 37.20、4.13、1.47，平均差被 seed 0 大幅拉低。HRM 的 SD 15.94、原始 CoGHP 的 SD 12.85 都很大，不能只用 seed 0 的极端差代表典型收益。分 task 看，HRM 的 Task 2 和 Task 5 均值相对原始方法分别下降约 36.00、26.67 个百分点，Task 3 反而提升；这是任务覆盖重分配。

Humanoid 三个规模的平均收益只有 0.89 至 2.44 个百分点；Large 只有一个种子提升。Giant 虽三个配对种子均提升，最终仍只有 28.62。当前结果支持 HRM-action 可以改善部分 executor 行为，但没有凭借这种改造解除 Humanoid 的困难。Humanoid Large 的 Task 2 在 HRM 三种子平均仍为 0，原始 CoGHP 也只有 1.33。

Point Giant 的 HRM 固定 validation batch 上，teacher action MSE 约 0.28 至 0.33，free-running predicted command 下约 0.62 至 0.72；Ant Giant 对应约 0.07 至 0.08，两种输入间差距很小。这提示 Point 上存在较明显的训练 command 与预测 command 使用差异。不过与数据动作的 MSE 只衡量匹配行为动作，不是有效控制的唯一指标，不能直接认定它就是成功率下降的全部原因。

![历史三种子学习曲线]({out/'legacy_learning_curves.png'})

## G 和 P 的单种子四条件结果

**★** 标记同环境四条件最高均值，***☆*** 标记次高；排序仅限本表 seed 0 条件。

{gp_table}

![四条件主指标]({out/'global_position_comparison.png'})

Humanoid Large：G only 相对 Baseline 增加 23.73 个百分点；P only 减少 20.67；G 加 P 比 Baseline 低 6.80，比 G only 低 30.53。G 能部分弥补 P 下的退化，但两者组合并未超过 Baseline。

Humanoid Giant：G only 增加 43.33 个百分点，G 加 P 增加 46.80；P only 仍下降 10.80。G 加 P 比 G only 高 3.47，但这个差距远小于 G 的整体收益，且只有一个种子。不能把 G 加 P 相对 Baseline 的大幅提升主要记在 P 名下。

Point Giant：P only 提升 17.87 个百分点，G only 与 Baseline 接近，G 加 P 反而减少 18.53。P only 的 46.53 仍低于历史原始 CoGHP 的同 seed 0 65.87，不能把“修复部分 HRM seed 0 退化”写成“超过原始 CoGHP”。

Ant Giant：G only 与 Baseline 接近，P only 和 G 加 P 分别下降 27.07、30.00 个百分点。这个场景此前已从 HRM-action 获益，现有 P 协议破坏了其中一部分收益。

### G 与 P 的交互

交互差值定义为 `I = S(G+P) - S(G) - S(P) + S(Baseline)`，使用同一 seed 0 的主指标。这是描述性的四格差值，不是有跨种子重复的因果显著性检验。

{factor_table}

Point Giant 的交互为 -35.87 个百分点，是最强的负交互；Humanoid Giant 为 +14.27，说明 G 在 P 下的作用更大。Humanoid Large 为 -9.87，Ant Giant 为 -1.60。故 G 和 P 不能按独立增益简单叠加，也不能给 P 一个跨环境统一的正负评价。

### 学习过程而非单一终点

![单种子四条件学习曲线]({out/'global_position_learning_curves.png'})

G only 在 Humanoid Large/Giant 的 100k 已达到 44.4/27.6，Baseline 是 8.8/3.2；优势在整个训练过程较早出现。Ant Giant 的 G only 100k 是 55.6，Baseline 是 16.8，但最终三点均值几乎追平：这里更明显的是早期学习加速，而非最终成功率优势。

Point Giant 很不稳定：P only 在 500k 为 1.2、700k 为 51.2；G 加 P 在 700k 为 0、800k 为 0.4，900k/1M 才到 14.4/15.6。这支持既存在 command 接口和训练困难，也可能存在有限 1M 预算下的收敛速度差异；当前记录不能断言其渐近性能永久较差。仍按预先指定的三点主指标评价，不追加训练或挑最优终点来改变排序。

### 分 task 的成功率

以下是各 task 的 800k、900k、1M 平均百分数。

{task_table}

Humanoid Giant 的 G only 和 G 加 P 对五个 task 都明显提高，收益不来自单个 task。Large 的 G only 也对五个 task 都提高，并将长期为零的 Task 2 提至 25.33，提示高层训练信号可能影响了此前未覆盖的路线。

Point Giant 的 G only 与 Baseline overall 接近，却将 Task 1 提到 100、Task 2 从 73.33 降到 2.67；G 加 P 只有 Task 2 达到 50，其余几乎为零。overall 会掩盖这种路线偏好或覆盖坍缩。Ant P only 的 Task 5 从 88 降到 14，但 Task 3 从 68 到 66.67，相比之下受影响小得多；不能把退化简单描述成所有低层动作都坏掉。

{literature_section}

## 诊断证据与原因分析

### Global 的几何进度和高层权重

下表将三个 checkpoint 上的探针数值作描述性平均。pair 来自固定 held-out 状态和数据 subgoal；远距离指当前状态到 goal 的 maze BFS 距离至少十格。cap 比例先对 K 个 slot 平均，再对 checkpoint 平均。

{value_table}

Humanoid 的 Global 几何进度符号一致率提升，同时高层 AWR cap 比例从约 73 至 78 降到约 52 至 58。它与导航成功率上升共同支持“更贴近位置目标的高层训练信号，且权重分布发生改变”这一解释。但无法拆分位置 goal、时间抽象、独立 bilinear 模型、容量和权重标度各自贡献。尤其 Humanoid Large 的远距离有效样本只有 104 个 pair，不应把它的一致率视为精确的泛化测量。

Point 的 Global 一致率达到 97.38，overall 却不升；Ant 的一致率从 80.84 降到 75.88，overall 仅小幅变化。这表明几何符号正确率不是任务成功率的充分条件。它不测量优势幅度的校准、planner 生成 command 的可实现性、身体动态或完整路线。BFS 也只反映迷宫几何，不等于机器人动力学下的最优时间距离；多个 pair 共享状态和轨迹，不能作为独立样本进行显著性推断。

四个环境中 G only 与 G 加 P 的全部 200 次日志、14 个 training/validation Global 指标逐项完全一致；对应 value 探针的排序结果也一致。Global 的训练是独立的，P 没有把 Global value 学坏。因此 Point 上 G 加 P 明显退化更可能发生在高层权重与 command representation、planner、executor 的耦合处，而非 Global loss 或独立数据随机流发生变化。

Global 的行为轨迹 goal offset 切片保留在 {ref('value_diagnostics.csv')}：offset 0/5/10/25/50/100 是行为轨迹的步数，不是最短路径距离。它可以检查短程平台和远程值标度，但不能据此直接宣称得到了最优距离函数。

### 重建成功与预测 command 有效性

下表除落墙比例外，使用 800k 至 1M 的 41 次固定 validation 日志均值。位置误差为环境 XY 单位；落墙比例使用三个 checkpoint 的 2048 状态解码探针均值。Teacher 指 encoder 产生的数据 command，predicted 指自由运行 planner 的输出。

{position_table}

P 的数据 command 重建 MSE 已很小，teacher 的平均 XY 误差约 0.18 至 0.31；predicted 的误差约 1.5 至 7.5，明显较大。Point G 加 P 的重建约 0.000325，却仍有 7.46 的 predicted 位置误差和 13.65% 的解码落墙比例。低重建误差证明 encoder/decoder 能表达数据中的位置，未证明 planner 的预测 latent 位于有效 command 流形或沿合法路线。

P 的所有高层到 position encoder/decoder 梯度记录都为零，低层到 encoder 与 reconstruction 梯度非零，符合设计。800k 至 1M 窗口内，“低层梯度范数均值 / reconstruction 梯度范数均值”在 Point 约 27 至 34，在 Ant/Humanoid 约 331 至 854。它提示 position encoder 主要受到动作目标塑形，重建良好仍可能留下对 planner 困难的 latent 几何或随训练移动的 labels。这些范数是日志中的诊断 batch 上的量，不能当成全部参数真实更新比例，也不能单靠比例证明发生 representation drift；latent drift 原始列已保留在 train_metrics 中以便复查。

{drift_table}

固定 batch 的 latent 在相邻 5000 步日志之间确实仍有非零变化，晚期均值约 0.0076 至 0.0174。这为高层 label 持续移动提供直接证据，但差异没有随正式成功率单调变化：Humanoid Giant 的 G 加 P 与 Point 的 P only 都有较小漂移，导航表现却很不相同。漂移可以作为耦合机制的线索，尚不能单独解释各环境的排序。

但 predicted 位置误差是相对数据 waypoint 的误差，合法替代路线也会产生较大误差；decoder 落墙只审查解码位置，正式策略没有经 decoder 修复或筛选。因此这些诊断支持接口问题的可能性，并不逐项等价于真实执行失败。

### 局部执行的 teacher 与 predicted 对照

以下采用 1M checkpoint，每条件 24 个相同 held-out 起点，恢复 qpos/qvel 并校验 observation；100 步预算，参考目标是数据最近 waypoint。P only 与 G 加 P 的起点 indices 相同。

{local_table}

Point 的 teacher 均为 24/24，predicted 均为 17/24；Ant 的 teacher 为 22/24、23/24，predicted 为 16/24、14/24。数据 waypoint 已知时 executor 通常可以完成这些局部目标，但自由运行 planner 到 executor 的链存在缺口。Point 的两个 P 条件在这个 probe 上相同，正式整体成功率却差 36.40 个百分点，说明局部达到单个参考 waypoint 无法捕捉完整五任务导航的差异。

Humanoid Large 的 teacher 也只有 8/24 或 12/24，predicted 为 4/24 或 5/24，提示还包含给定数据 command 的有限预算执行难度。Humanoid Giant 的 G 加 P predicted 为 9/24，teacher 为 7/24，进一步说明 teacher 不是普适上界。预测路线可以选择不同 waypoint；参考 waypoint 的 100 步预算也不是官方任务时间预算。该 probe 没有 full-state Baseline 对照，不能由它直接证明 P 相对原接口的全部 executor 损失。

### 动作拟合不等于闭环能力

{mse_table}

这是相同环境的固定 validation batch 在 800k 至 1M 的均值。Humanoid Giant 的 G only 将 overall 从 35.33 提至 78.67，而 predicted action MSE 从 0.06451 到 0.06452，几乎不变。Point 的预测 command action MSE 显著高于 teacher，支持接口偏移的关注方向，但 P only 的 predicted MSE 并未优于 Baseline，正式成功率仍更高。动作 MSE 衡量对离线行为动作的匹配，不涵盖路线选择、多模态有效动作和 rollout 状态分布；不同环境动作维数和动力学也使 MSE 不适合直接跨环境排名。

## 可以支持的解释与尚未分开的因素

| 现象 | 已有直接证据 | 目前较合理的解释 | 尚未分开的因素 |
| --- | --- | --- | --- |
| HRM 在 Ant Giant 改善 | 三种子均提升，五 task 均值提升 | 整体 executor 改造适合部分 Ant 控制问题 | 递归、core 独立化、block 家族、容量和计算量 |
| HRM 在 Point Giant 下降 | 三种子均下降，seed 0 尤其差，predicted 与 teacher action MSE 差大 | 预测 command 的使用和路线覆盖存在敏感性 | K=3、小学习率、优化速度、不同初始化和原始参数共享方式 |
| G 在 Humanoid 大幅提升 | 两规模 seed 0 提升、各 task 提升、早期已分开、几何一致率上升 | legacy 高层训练信号是当前困难的重要候选瓶颈 | 位置语义、25 步抽象、容量、AWR 标度及其他种子 |
| P 在 Ant/Humanoid 多数退化 | 重建已好，predicted command 误差和局部执行缺口仍在 | command 表达与生成、控制之间的匹配不足 | 删除未来身体信息、梯度隔离、辅助 loss、encoder label 移动 |
| Point 的 G 加 P 强负交互 | Global 指标完全相同，任务覆盖近乎坍缩，落墙比例较高 | 相同高层权重在新 latent 接口中产生不同学习行为 | 权重到 latent 回归的几何、自由运行链和训练时长 |

当前最有信息量的后续问题是“哪一个环节限制了哪一个环境”：Global 权重是否校准到可执行进度；planner 是否能生成 executor 擅长使用的 command；P 的信息约束与梯度协议分别改变了什么；Point 的任务覆盖为何重排。这些是由现有证据提出的研究问题，还不是经验证的优化处方。

对单种子 G/P 大幅差异，下一次解释必须保留相同种子和配置对应关系；不能把本轮单种子赢家与历史三种子均值并列成等精度排行榜。已有三种子结果也只包含三个重复，报告配对差值和 SD，不使用小样本显著性语言。

运行 wall time 同样已保存在逐 run 表中。前两组 HRM 的每 run 平均时长相对 CoGHP 约为 1.05 至 1.95 倍；它包含编译、评估和 checkpoint I/O，跨 campaign 的执行条件也可能影响时间，不是等 FLOPs、推理延迟或纯训练吞吐的基准。

## 逐 checkpoint 成绩

{checkpoint_table}

## 审计与复查入口

本次从原始 `campaign.json`、`run.json`、实际 `flags.json`、`reproducibility.json`、checkpoint semantic metadata、`eval.csv`、`train.csv` 与诊断 JSON 生成结果。检查全部 70 个 run 完成且 returncode=0；全部 700 次评估有唯一正确步数、成功率范围与 50 episodes 的计数粒度；overall 与五 task 平均一致；全部 14,000 条训练日志指标有限；210 个所需 checkpoint 文件存在且非空；三组冻结代码共 530 个已记录文件项按 SHA256 重验。

48 个 value probe 与 24 个局部执行 probe JSON 均齐全，后者形成 48 条 teacher/predicted 汇总记录。前两组旧汇总 CSV 的 54 行成绩和所记录原始 CSV/config hashes 与本次重算一致。新旧 Baseline 衔接、Global 独立性与局部起点配对见 {ref('audit.json')}。

原始实验目录保持不变。数据集一致性使用 campaign 中记录的 SHA256，本次不重算大型 NPZ；checkpoint 只检查存在、非空和单独 semantic metadata，未反序列化验证全部 payload。冻结代码、原始 CSV/config 与诊断 JSON 的 hash 已记录，成绩来源可逐 run 追溯。

| 文件 | 内容 |
| --- | --- |
| {ref('runs.csv')} | 70 个 run 的三点评分、五 task 均值、实际路径、版本、耗时和原始文件 hash |
| {ref('evaluations.csv')} | 700 次完整正式评估，保留各 task 和 overall |
| {ref('legacy_summary.csv')} | 九环境两方法三种子的均值、样本 SD 和各 seed 分数 |
| {ref('legacy_paired_differences.csv')} | 同 seed 的 HRM 减 CoGHP 差值 |
| {ref('factorial_effects.csv')} | 单种子四条件的条件效应与交互差值 |
| {ref('train_metrics.csv')} | 各 run 在 800k 至 1M 的训练和固定 validation 指标窗口统计 |
| {ref('value_diagnostics.csv')} | 48 个 value probe，含 cap、几何排序、行为 goal 切片和落墙率 |
| {ref('local_execution_diagnostics.csv')} | 24 个局部 probe 的 48 条 teacher/predicted 汇总 |
| {ref('resolved_agent_configs.csv')} 与 {ref('resolved_configs.json')} | 全部实际模型配置及软件、随机流、checkpoint semantic metadata |
| {ref('stored_parameter_counts.csv')} | 第三组的全部存储参数按模块计数 |
| {ref('source_changes.csv')} | 第三组相对第二组冻结源文件变化 |
| {ref('campaign_manifests.json')} 与 {ref('audit.json')} | 三组完整配置、源与数据来源和一致性检查 |

重算命令：在仓库运行 `python3 scripts/analyze_navigation_campaigns.py`。脚本仅读取这三组实验目录，写入本报告所在目录，不启动训练或修改原始实验。
'''
    (out/'report.md').write_text(text)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=Path('/data/qijunrong/06-RL/offline-rl/exp/Navigation'))
    parser.add_argument('--output',type=Path,default=EXPERIMENTS/'analysis_20261014')
    args=parser.parse_args()
    data=analyze(args.base,args.output)
    plots(data,args.output)
    report(data,args.output)
