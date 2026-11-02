#!/usr/bin/env python3
"""Snapshot and analyze a planner campaign without changing training outputs."""

import argparse
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
import shutil
import statistics

import numpy as np


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def rows(path):
    if not path.exists():
        return []
    raw = path.read_bytes()
    # An active CsvLogger may be in the middle of writing its last row.
    raw = raw[:raw.rfind(b'\n') + 1]
    reader = csv.DictReader(io.StringIO(raw.decode()))
    result = []
    seen = set()
    for row in reader:
        if any(value is None for value in row.values()):
            raise ValueError(f'Incomplete internal CSV row: {path}')
        parsed = {key: float(value) for key, value in row.items()}
        if not all(math.isfinite(value) for value in parsed.values()):
            raise ValueError(f'Nonfinite metric: {path}')
        step = int(parsed['step'])
        if step in seen:
            raise ValueError(f'Duplicate step: {path}: {step}')
        seen.add(step)
        parsed['step'] = step
        result.append(parsed)
    return result


def snapshot(campaign, output):
    output.mkdir(parents=True, exist_ok=False)
    target = output / 'snapshot'
    target.mkdir()
    for name in ('campaign.json', 'study.json', 'launcher_runtime.json', 'launcher_process.json'):
        path = campaign / name
        if path.exists():
            shutil.copy2(path, target / name)
    manifest = json.loads((target / 'campaign.json').read_text())
    source = Path(manifest['source_dir'])
    source_signature = hashlib.sha256(json.dumps(manifest['source']['files'], sort_keys=True).encode()).hexdigest()
    if source_signature != manifest['source']['code_signature']:
        raise ValueError('Manifest source signature is inconsistent.')
    for relative, expected in manifest['source']['files'].items():
        if digest(source / relative) != expected:
            raise ValueError(f'Frozen source mismatch: {relative}')
    inventory = {}
    for method in manifest['study']['methods']:
        found = list((campaign / 'runs' / method).glob('*/seed_000/attempt_000/run.json'))
        if len(found) > 1:
            raise ValueError('Expected one environment/seed/attempt for this initial study.')
        if not found:
            continue
        root = found[0].parent
        destination = target / 'runs' / method
        destination.mkdir(parents=True)
        for path in [root / 'run.json', *[(root / 'artifacts' / name) for name in (
            'train.csv', 'eval.csv', 'flags.json', 'reproducibility.json',
            'checkpoint_metadata.json', 'model_parameter_counts.json')]]:
            if path.exists():
                copy = destination / path.name
                copy.write_bytes(path.read_bytes())
                inventory[str(copy.relative_to(target))] = dict(original=str(path), sha256=digest(copy))
    dump(target / 'snapshot_inventory.json', inventory)
    dump(target / 'captured_at.json', dict(
        time=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        campaign=str(campaign), source_signature=manifest['source']['code_signature'],
        source_hashes_verified=True,
        note='Logs captured once per file; active runs may advance after this snapshot. Incomplete trailing CSV rows are excluded.',
    ))


def analyze(output):
    root = output / 'snapshot'
    manifest = json.loads((root / 'campaign.json').read_text())
    study = manifest['study']
    score_steps = study['metric']['steps']
    summary = dict(captured=json.loads((root / 'captured_at.json').read_text()),
                   metric_steps=score_steps, seeds=[0], runs={}, effects={})
    for name, method in study['methods'].items():
        directory = root / 'runs' / name
        spec = method['overrides']
        record = dict(h=spec['planner_h_cycles'], l=spec['planner_l_cycles'], M=spec['num_subgoals'], status='pending')
        summary['runs'][name] = record
        if not directory.exists():
            continue
        state = json.loads((directory / 'run.json').read_text())
        config = json.loads((directory / 'flags.json').read_text())
        agent = config['agent']
        metadata = json.loads((directory / 'checkpoint_metadata.json').read_text())['signature']
        if state['source_signature'] != manifest['source']['code_signature'] or metadata['code_signature'] != state['source_signature']:
            raise ValueError(f'Source identity mismatch: {name}')
        expected = {**study['common_agent'], **study['environments'][state['environment']], **spec}
        for key, value in expected.items():
            if agent[key] != value:
                raise ValueError(f'Configuration mismatch: {name}: {key}')
        record.update(status=state['status'], train=rows(directory / 'train.csv'), evaluation=rows(directory / 'eval.csv'))
        record['last_train_step'] = record['train'][-1]['step'] if record['train'] else 0
        record['last_eval_step'] = record['evaluation'][-1]['step'] if record['evaluation'] else 0
        for row in record['evaluation']:
            tasks = [row[f'evaluation/task{i}_success'] for i in range(1, 6)]
            if any(not 0 <= value <= 1 for value in tasks) or not math.isclose(statistics.mean(tasks), row['evaluation/overall_success'], abs_tol=1e-10):
                raise ValueError(f'Invalid task/overall evaluation: {name}')
        m, d = agent['num_subgoals'], agent['feature_dim']
        for row in record['train']:
            mass = sum(agent['high_discount']**(m-j-1) * row[f'validation/high_actor/weight_{j}/mean'] for j in range(m))
            row['derived/normalized_weighted_latent_mse'] = 2 * (
                m * agent['subgoal_steps'] * row['validation/high_actor/joint_plan_loss']
                - .5 * d * math.log(2*math.pi) * mass) / (d * mass)
        if state['status'] == 'completed':
            if state.get('returncode') != 0:
                raise ValueError(f'Completed run has nonzero return code: {name}')
            chosen = {row['step']: row for row in record['evaluation'] if row['step'] in score_steps}
            if set(chosen) != set(score_steps):
                raise ValueError(f'Missing exact score checkpoint: {name}')
            record['score'] = statistics.mean(chosen[step]['evaluation/overall_success'] for step in score_steps)
            if not math.isclose(record['score'], state['metric']['mean'], abs_tol=1e-12):
                raise ValueError(f'Saved score differs from CSV: {name}')
            record['per_task_score'] = [statistics.mean(chosen[s][f'evaluation/task{i}_success'] for s in score_steps) for i in range(1, 6)]
            record['final_score'] = chosen[score_steps[-1]]['evaluation/overall_success']
            tail = [row for row in record['train'] if score_steps[0] <= row['step'] <= score_steps[-1]]
            record['diagnostics_window'] = [score_steps[0], score_steps[-1]]
            record['diagnostics_n_logs'] = len(tail)
            record['diagnostics'] = {key: statistics.mean(row[key] for row in tail) for key in tail[0] if key.startswith(('validation/', 'validation_action/', 'derived/'))}
            grad_norm = np.array([row['training/grad/norm'] for row in record['train']])
            record['logged_total_gradient_norm'] = dict(median=float(np.median(grad_norm)),
                p95=float(np.quantile(grad_norm,.95)), maximum=float(grad_norm.max()),
                late_median=float(np.median([row['training/grad/norm'] for row in tail])))
            record['planner_block_calls'] = record['h'] * (record['l'] + 1)
            record['total_wall_hours'] = (datetime.fromisoformat(state.get('training_finished_at', state['finished_at'])) - datetime.fromisoformat(state['started_at'])).total_seconds()/3600
            counts = json.loads((directory / 'model_parameter_counts.json').read_text())
            record['parameters_excluding_target'] = counts['total_stored_parameters'] - counts['by_group']['modules_target_value']
    lookup = {(r['h'], r['l'], r['M']): r for r in summary['runs'].values() if 'score' in r}
    for h in (1, 2, 4):
        for ell in (1, 2):
            if (h, ell, 1) in lookup and (h, ell, 3) in lookup:
                one, three = lookup[h, ell, 1], lookup[h, ell, 3]
                summary['effects'][f'M3_minus_M1_h{h}l{ell}'] = dict(
                    score_pp=100*(three['score']-one['score']),
                    per_task_pp=[100*(b-a) for a,b in zip(one['per_task_score'], three['per_task_score'])])
        for m in (1, 3):
            if (h, 1, m) in lookup and (h, 2, m) in lookup:
                summary['effects'][f'l2_minus_l1_h{h}M{m}'] = dict(score_pp=100*(lookup[h, 2, m]['score']-lookup[h, 1, m]['score']))
    for ell in (1, 2):
        for m in (1, 3):
            for low, high in ((1,2), (2,4), (1,4)):
                if (low,ell,m) in lookup and (high,ell,m) in lookup:
                    summary['effects'][f'h{high}_minus_h{low}_l{ell}M{m}'] = dict(score_pp=100*(lookup[high,ell,m]['score']-lookup[low,ell,m]['score']))
    summary['status_counts'] = {status: sum(row['status'] == status for row in summary['runs'].values()) for status in ('completed', 'running', 'pending', 'failed')}
    dump(output / 'analysis.json', summary)
    return summary


def render(output, data):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/coghp-analysis-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    colors = {(1,1): '#235789', (1,2): '#d97924', (2,1): '#2b8c75', (2,2): '#8759a0', (4,1): '#bd4967', (4,2): '#687d28'}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharex=True, sharey=True, constrained_layout=True)
    for ax, m in zip(axes, (1, 3)):
        for name, record in data['runs'].items():
            if record['M'] != m or not record.get('evaluation'):
                continue
            x = [row['step']/1000 for row in record['evaluation']]
            y = [100*row['evaluation/overall_success'] for row in record['evaluation']]
            ax.plot(x, y, marker='o', markersize=4, linewidth=1.9,
                    linestyle='-' if record['status']=='completed' else '--',
                    color=colors[record['h'],record['l']], label=f"h={record['h']}, l={record['l']}" + (' (running)' if record['status']!='completed' else ''))
        ax.axvspan(800,1000,alpha=.07,color='black')
        ax.set(title=f'M = {m}', xlabel='Training updates (thousands)', xlim=(0,1020), ylim=(0,100))
        ax.grid(alpha=.18); ax.legend(fontsize=8, loc='upper left')
    axes[0].set_ylabel('Success (%)')
    fig.suptitle('AntMaze-Giant: paired learning curves | seed 0 only')
    fig.savefig(output/'learning_curves.png', dpi=180); plt.close(fig)

    completed = [(name,r) for name,r in data['runs'].items() if 'score' in r]
    fig, axes = plt.subplots(1, 2, figsize=(14, max(4.1, .42*len(completed))), constrained_layout=True,
                             gridspec_kw={'width_ratios':[1,1.25]})
    paired = sorted({(r['h'],r['l']) for _,r in completed})
    for i,(h,l) in enumerate(paired):
        for m,offset,color in ((1,-.18,'#235789'),(3,.18,'#d97924')):
            row = next((r for _,r in completed if (r['h'],r['l'],r['M'])==(h,l,m)),None)
            if row is None: continue
            score=row['score']*100
            axes[0].bar(i+offset,score,.31,color=color,label=f'M={m}' if i==0 else None)
            axes[0].text(i+offset,score+2,f'{score:.2f}',ha='center',fontsize=9)
            dots=[v['evaluation/overall_success']*100 for v in row['evaluation'] if v['step'] in data['metric_steps']]
            axes[0].scatter([i+offset]*len(dots),dots,s=20,color='black',alpha=.65,zorder=3)
    axes[0].set(xticks=range(len(paired)),xticklabels=[f'h={h}\nl={l}' for h,l in paired],ylim=(0,100),ylabel='Success (%)',title='Mean of 800k / 900k / 1M')
    axes[0].legend(); axes[0].grid(axis='y',alpha=.15)
    matrix=np.array([r['per_task_score'] for _,r in completed])*100
    im=axes[1].imshow(matrix,vmin=0,vmax=100,cmap='YlGnBu',aspect='auto')
    axes[1].set(xticks=range(5),xticklabels=[f'Task {i}' for i in range(1,6)],yticks=range(len(completed)),yticklabels=[name.replace('hrm_','') for name,_ in completed],title='Task-level means at the same checkpoints')
    for i in range(len(completed)):
        for j in range(5):
            axes[1].text(j,i,f'{matrix[i,j]:.1f}',ha='center',va='center',color='white' if matrix[i,j]>55 else '#222',fontsize=9)
    fig.colorbar(im,ax=axes[1],label='Success (%)',shrink=.8)
    fig.suptitle('Completed runs only | dots are correlated checkpoints, not independent seeds')
    fig.savefig(output/'completed_comparison.png',dpi=180); plt.close(fig)

    fig, axes = plt.subplots(2,2,figsize=(11.5,7),constrained_layout=True)
    for name,r in completed:
        x=[v['step']/1000 for v in r['train']]
        common=dict(label=name.replace('hrm_',''),color=colors[r['h'],r['l']],linestyle='-' if r['M']==1 else '--',linewidth=1.2,alpha=.8)
        keys=['validation/high_actor/mse','derived/normalized_weighted_latent_mse',f"validation/high_actor/weight_{r['M']-1}/cap_fraction",'validation_action/gap/mse']
        for ax,key in zip(axes.flat,keys):
            ax.plot(x,[v[key] for v in r['train']],**common)
    for ax,title in zip(axes.flat,['Unweighted latent MSE (all waypoints)','Normalized AWR-weighted latent MSE','Nearest waypoint: weight cap fraction','Predicted minus teacher action-data MSE']):
        ax.set(title=title,xlabel='Training updates (thousands)'); ax.grid(alpha=.18)
    axes[1,0].set_ylim(0,1);axes[1,1].axhline(0,color='black',linewidth=.7)
    axes[0,0].legend(fontsize=7,ncol=2)
    fig.suptitle('Diagnostics | fixed validation inputs; learned latent labels can drift')
    fig.savefig(output/'training_diagnostics.png',dpi=180);plt.close(fig)

    def table(headers, values):
        return '<table><thead><tr>'+''.join(f'<th>{html.escape(str(h))}</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in row)+'</tr>' for row in values)+'</tbody></table>'
    status_table=table(['配置','状态',*data['metric_steps'],'主分数'],[
        [name,r['status'],*[next((f"{100*v['evaluation/overall_success']:.2f}%" for v in r.get('evaluation',[]) if v['step']==step),'—') for step in data['metric_steps']],f"{100*r['score']:.2f}%" if 'score' in r else '未形成完整主指标'] for name,r in data['runs'].items()])
    effects_table=table(['对照差值','百分点'],[[name,f"{r['score_pp']:+.2f}"] for name,r in data['effects'].items()])
    probes=''
    probe_path=output/'checkpoint_probes_extended.json'
    if not probe_path.exists():
        probe_path=output/'checkpoint_probes.json'
    if probe_path.exists():
        p=json.loads(probe_path.read_text())
        values=[]
        for name,checkpoints in p['runs'].items():
            if len(checkpoints)!=len(data['metric_steps']):continue
            mean=lambda function:statistics.mean(function(v) for v in checkpoints.values())
            values.append([name,f"{mean(lambda v:v['waypoints'][-1]['mse']):.4f}",
                           f"{mean(lambda v:v['waypoints'][-1]['weighted_mse']):.4f}",
                           f"{mean(lambda v:v['actions']['teacher_data_mse']):.4f}",
                           f"{mean(lambda v:v['actions']['predicted_data_mse']):.4f}",
                           f"{mean(lambda v:v['actions']['near_replacement_action_distance']):.5f}",
                           f"{mean(lambda v:v['actions']['far_replacement_action_distance']):.5f}"])
        probes='<h2>配对 checkpoint 探针</h2><p>同一组 2048 个离线查询，CPU 前向；每项为 800k、900k、1M 的均值。动作差异是敏感性指标，不是局部执行成功率。</p>'+table(['配置','近端 MSE','近端加权 MSE','真值链动作 MSE','预测链动作 MSE','替换近端的动作变化','替换远端的动作变化'],values)
        fig, ax = plt.subplots(1,2,figsize=(14,4.8),constrained_layout=True)
        for i,(name,checkpoints) in enumerate(p['runs'].items()):
            near=statistics.mean(row['waypoints'][-1]['mse'] for row in checkpoints.values())
            ax[0].bar(i,near,color='#235789' if name.endswith('m1') else '#d97924')
            ax[0].text(i,near+.003,f'{near:.3f}',ha='center',fontsize=8,rotation=45)
        names=list(p['runs'])
        ax[0].set(xticks=range(len(names)),xticklabels=[n.replace('hrm_','') for n in names],ylabel='Nearest waypoint latent MSE',title='Paired raw-state targets; model-specific latent spaces')
        ax[0].set_ylim(0, max(statistics.mean(v['waypoints'][-1]['mse'] for v in ck.values()) for ck in p['runs'].values())*1.25)
        ax[0].tick_params(axis='x',rotation=55)
        m3=[(name,ck) for name,ck in p['runs'].items() if name.endswith('m3')]
        for i,(name,ck) in enumerate(m3):
            for offset,part,color in ((-.18,'near','#235789'),(.18,'far','#d97924')):
                key=f'{part}_unit_direction_action_jvp_mse'
                if key not in next(iter(ck.values()))['actions']:continue
                response=math.sqrt(statistics.mean(row['actions'][key] for row in ck.values()))
                ax[1].bar(i+offset,response,.31,color=color,label=part if i==0 else None)
                ax[1].text(i+offset,response+.001,f'{response:.3f}',ha='center',fontsize=8,rotation=45)
        ax[1].set(xticks=range(len(m3)),xticklabels=[name.replace('hrm_','') for name,_ in m3],ylabel='RMS action derivative / unit latent direction',title='Executor sensitivity along equal-norm correction directions')
        ax[1].set_ylim(0, ax[1].get_ylim()[1]*1.2)
        ax[1].tick_params(axis='x',rotation=45)
        ax[1].legend()
        for a in ax:a.grid(axis='y',alpha=.15)
        fig.suptitle('Checkpoint probes | 2048 paired queries, averaged over 800k / 900k / 1M')
        fig.savefig(output/'checkpoint_diagnostics.png',dpi=180);plt.close(fig)
        probes += '<img src="checkpoint_diagnostics.png" alt="paired checkpoint diagnostics"><p>方向导数沿预测目标指向数据目标的方向计算，每个样本的方向范数归一到 1；不是任意方向的全 Jacobian 范数，也不等于闭环控制收益。</p>'
    interpretation_path=output/'interpretation.json'
    interpretation=''
    if interpretation_path.exists():
        findings=json.loads(interpretation_path.read_text())
        interpretation='<h2>研究解读</h2>'+''.join('<p><strong>'+html.escape(item['finding'])+'</strong> '+html.escape(item['evidence'])+'</p><p>'+html.escape(item['boundary'])+'</p>' for item in findings)
    extra_path=output/'diagnostic_sections.html'
    extra=extra_path.read_text() if extra_path.exists() else ''
    phase='完整实验分析' if data['status_counts']['completed']==len(data['runs']) else '进行中实验的快照分析'
    body=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>AntMaze-Giant v0 {phase}</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1180px;margin:35px auto;padding:0 24px;color:#233044;line-height:1.7}}img{{width:100%;margin:16px 0}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:9px;border-bottom:1px solid #dde3eb;text-align:left}}th{{background:#f0f4f9}}.note{{background:#edf3f8;padding:16px}}code{{overflow-wrap:anywhere}}</style>
<h1>AntMaze-Giant v0：{phase}</h1>
<p>快照时间：{data['captured']['time']}。{html.escape(str(data['status_counts']))}</p>
<p class="note">仅 seed 0。主分数严格取 800k、900k、1M 的成功率均值；未完成组不以最后一次成绩替代主指标。灰色区间为主指标窗口。固定验证输入不等于固定 latent 标签：编码器也在学习。所有分析与训练隔离。</p>
{status_table}<img src="learning_curves.png" alt="learning curves"><img src="completed_comparison.png" alt="completed comparison">
<h2>已完成条件之间的配对差值</h2>{effects_table}
<p>上述差值是单种子的描述性结果，不是跨种子的统计显著性。三个 checkpoint 也不是三个独立 seed。改变 M 同时改变计划覆盖范围、目标数量、部分参数量和损失分配。h、l 改变循环计算量，不增加参数量；本轮比较固定训练更新次数，没有配平计算时间。</p>
<img src="training_diagnostics.png" alt="training diagnostics">
<p>归一化加权 MSE 从固定单位方差 Gaussian NLL 扣除常数项后计算，并除以总有效权重。它用于拆分拟合误差和权重尺度，不改变训练目标。</p>
{probes}{extra}{interpretation}<h2>复现与边界</h2><p>原始日志快照、逐文件 SHA256、配置和 checkpoint metadata 位于 snapshot/；冻结代码已逐文件校验。结构化结果在 analysis.json；checkpoint 探针在 checkpoint_probes_extended.json（若运行）。地图仅用于几何进展诊断，不进入模型输入。分析脚本副本及其 SHA256 位于 analysis_code/ 和 analysis_provenance.json。</p>
<p>原始实验：<code>{html.escape(data['captured']['campaign'])}</code></p></html>'''
    (output/'report.html').write_text(body)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--refresh-report',action='store_true',help='Re-render the captured snapshot, without reading live logs again.')
    args=parser.parse_args()
    if not args.refresh_report:
        snapshot(args.campaign,args.output)
    summary=analyze(args.output)
    render(args.output,summary)
    print(json.dumps({'output':str(args.output),'status_counts':summary['status_counts'],'effects':summary['effects']},indent=2))


if __name__=='__main__':main()
