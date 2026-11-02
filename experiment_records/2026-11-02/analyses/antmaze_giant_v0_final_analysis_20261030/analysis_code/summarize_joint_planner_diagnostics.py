#!/usr/bin/env python3
"""Combine completed log, checkpoint, and rollout diagnostics into report assets."""

import argparse
import html
import json
import math
import os
from pathlib import Path
import statistics

import numpy as np


def table(headers, rows):
    tr = lambda row: '<tr>' + ''.join('<td>'+html.escape(str(x))+'</td>' for x in row) + '</tr>'
    return '<table><thead>'+tr(headers)+'</thead><tbody>'+''.join(tr(row) for row in rows)+'</tbody></table>'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output
    analysis = json.loads((root/'analysis.json').read_text())
    probes = json.loads((root/'checkpoint_probes_extended.json').read_text())
    rollouts = json.loads((root/'local_rollout_probes.json').read_text())
    assert analysis['status_counts']['completed'] == 12 and rollouts['complete']
    assert set(probes['runs']) == set(analysis['runs']) == set(rollouts['runs'])
    steps = {str(step) for step in analysis['metric_steps']}
    diagnostic = dict(checkpoints={}, local_interventions={},
                      observations='Descriptive seed-0 comparisons; no between-seed significance tests.')
    for name, ck in probes['runs'].items():
        assert set(ck) == steps
        mean = lambda f: statistics.mean(f(row) for row in ck.values())
        m = analysis['runs'][name]['M']
        near = mean(lambda row: row['actions']['near_unit_direction_action_jvp_mse'])
        far = mean(lambda row: row['actions']['far_unit_direction_action_jvp_mse'])
        mass = np.array([.8**(m-j-1)*mean(lambda row:row['waypoints'][j]['weight_mean']) for j in range(m)])
        diagnostic['checkpoints'][name] = dict(
            near_mse=mean(lambda row: row['waypoints'][-1]['mse']),
            near_weighted_mse=mean(lambda row: row['waypoints'][-1]['weighted_mse']),
            near_error_to_constant_variance=mean(lambda row:row['waypoints'][-1]['mse']/row['waypoints'][-1]['constant_target_mean_mse']),
            target_rank=mean(lambda row: row['waypoints'][-1]['target_effective_rank']),
            predicted_rank=mean(lambda row: row['waypoints'][-1]['predicted_effective_rank']),
            near_to_far_unit_response_rms_ratio=math.sqrt(near/far) if far else None,
            near_cap_fraction=mean(lambda row: row['waypoints'][-1]['cap_fraction']),
            far_cap_fraction=mean(lambda row: row['waypoints'][0]['cap_fraction']),
            coefficient_mass_share= (mass/mass.sum()).tolist(),
            near_geometric_sign_agreement=mean(lambda row: row['waypoints'][-1]['geometric_sign_agreement']),
            far_geometric_sign_agreement=mean(lambda row: row['waypoints'][0]['geometric_sign_agreement']),
            near_distant_geometric_sign_agreement=mean(lambda row: row['waypoints'][-1]['distant_geometric_sign_agreement']),
            backward_capped_fraction=mean(lambda row: row['waypoints'][-1]['geometrically_backward_capped_fraction']),
        )
        record = rollouts['runs'][name]
        assert record['checkpoint_sha256'] == ck[str(rollouts['checkpoint_step'])]['checkpoint_sha256']
        modes = ['predicted','teacher'] if m==1 else ['predicted','teacher','corrected_near','corrected_far']
        assert set(record['summary']) == set(modes)
        by = {(r['mode'],r['index']):r for r in record['episodes']}
        assert len(by) == len(record['episodes']) == len(modes)*len(rollouts['indices'])
        changes = {}
        for mode in modes[1:]:
            pairs = [(by['predicted',i]['success'], by[mode,i]['success']) for i in rollouts['indices']]
            changes[mode] = dict(rescued=sum(not a and b for a,b in pairs), lost=sum(a and not b for a,b in pairs))
        diagnostic['local_interventions'][name] = dict(summary=record['summary'], paired_changes=changes)

    os.environ.setdefault('MPLCONFIGDIR','/tmp/coghp-analysis-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    names = list(analysis['runs'])
    lookup = {(r['h'],r['l'],r['M']):r for r in analysis['runs'].values()}
    fig, axes = plt.subplots(1,2,figsize=(10.5,4.2),sharey=True,constrained_layout=True)
    for ax,ell in zip(axes,(1,2)):
        for m,color in ((1,'#235789'),(3,'#c96d27')):
            scores = [100*lookup[h,ell,m]['score'] for h in (1,2,4)]
            ax.plot((1,2,4),scores,marker='o',linewidth=2.4,color=color,label=f'M={m}')
            for h,s in zip((1,2,4),scores): ax.annotate(f'{s:.1f}',(h,s),xytext=(0,8),textcoords='offset points',ha='center',color=color)
        ax.set(title=f'l={ell}',xlabel='H cycles (h)',xticks=[1,2,4],ylim=(30,72))
        ax.grid(alpha=.2);ax.legend(loc='upper right')
    axes[0].set_ylabel('Official success: mean of 800k / 900k / 1M (%)')
    fig.suptitle('Depth and chain length interact | seed 0, fixed training-update budget')
    fig.savefig(root/'interaction.png',dpi=180);plt.close(fig)

    modes = ['predicted','teacher','corrected_near','corrected_far']
    fig,axes=plt.subplots(1,2,figsize=(12,6.5),constrained_layout=True)
    for ax,key,title in zip(axes,('success_at_50','success_rate'),('Within 50 environment steps','Within 100 environment steps')):
        matrix = np.full((len(names),4),np.nan)
        for i,name in enumerate(names):
            for j,mode in enumerate(modes):
                if mode in rollouts['runs'][name]['summary']:
                    matrix[i,j]=rollouts['runs'][name]['summary'][mode][key]*32
        cmap=plt.get_cmap('YlGnBu').copy();cmap.set_bad('#f0f0f0')
        im=ax.imshow(matrix,vmin=0,vmax=32,cmap=cmap,aspect='auto')
        ax.set(xticks=range(4),xticklabels=['Native\nprediction','Data\ncommands','Correct\nnear','Correct\nfar'],
               yticks=range(len(names)),yticklabels=[n.replace('hrm_','') for n in names],title=title)
        for i in range(len(names)):
            for j in range(4):
                x=matrix[i,j]
                ax.text(j,i,'—' if np.isnan(x) else f'{x:.0f}/32',ha='center',va='center',color='white' if x>20 else '#222',fontsize=9)
    fig.colorbar(im,ax=axes,label='Paired starts reaching the reference XY waypoint',shrink=.8)
    fig.suptitle('Local command interventions | 1M checkpoint only; not official task scores')
    fig.savefig(root/'local_interventions.png',dpi=180);plt.close(fig)

    sections = '<h2>循环深度与 chain 长度的交互</h2><img src="interaction.png" alt="h and M interaction">'
    sections += '<p>每个 h、l 条件都重新训练了整个 agent。本轮不是给同一个 checkpoint 增加推理次数，因此不能据此声称 test-time scaling。h、l 改变 HRM block 调用次数 h(l+1)。结构核验更正：planner 与 executor 的序列长度均为 M+3；M=1、3 均对应 4、6 个 token。planner 有 M+1 个 query，其中最后一个不直接读出，但参与联合计算。此前误写的 planner 3、5 个 token 已更正；实验数据不受影响。</p>'
    sections += '<h2>局部执行：同一验证集模拟器状态下替换 command</h2><p>12 个 1M checkpoint × 同一组 32 个起点。M=1 测 2 种输入，M=3 测 4 种输入，共 1,152 条轨迹。全部恢复状态与数据 observation 的最大偏差为 0。nearest waypoint 来自 t+50，最终目标来自 t+200；目标 XY 容差 0.5，最多执行 100 步。数据 command 保留完整 state，每步针对当前 observation 重新编码。</p>'
    sections += '<img src="local_interventions.png" alt="paired local command intervention heatmaps">'
    sections += '<p>这测的是是否经过数据轨迹指定的近端 waypoint。直接预测的路线可以选择不同的有效路径，也会每步重新规划；数据 command 则一直指向固定参考 state。因此差值表示该参考目标下的可利用执行空间，不是正式任务收益或纯粹的 planner 错误率。50 步与 100 步结果差异也提示：给定数据目标，不代表 executor 一定在名义 Δ=50 内到达。</p>'
    rows=[]
    for name,x in diagnostic['local_interventions'].items():
        for mode,change in x['paired_changes'].items():
            rows.append([name,mode,change['rescued'],change['lost'],f"{100*(change['rescued']-change['lost'])/32:+.1f}"])
    sections += table(['配置','替换方式','从失败到成功','从成功到失败','净变化/百分点'],rows)
    sections += '<h2>AWR 权重及学习信号</h2>'
    sections += table(['配置','近端 cap','远端 cap','近端/远端动作响应','近端系数份额','几何符号一致/近端','远距离样本一致/近端'],[
        [name,f"{100*x['near_cap_fraction']:.1f}%",f"{100*x['far_cap_fraction']:.1f}%",
         '—' if x['near_to_far_unit_response_rms_ratio'] is None else f"{x['near_to_far_unit_response_rms_ratio']:.2f}×",
         f"{100*x['coefficient_mass_share'][-1]:.1f}%",f"{100*x['near_geometric_sign_agreement']:.1f}%",f"{100*x['near_distant_geometric_sign_agreement']:.1f}%"] for name,x in diagnostic['checkpoints'].items()])
    sections += '<p>近端系数份额是 ρʲE[wⱼ] 的归一化份额，不是实际梯度份额。M=3 的 loss 对近端单项额外除以 3，且近端只占总系数约 38%；但不同条件的 encoder、误差和梯度也不同，所以不能推断其梯度恰为 M=1 的三分之一。超过 cap 的样本失去相互间的权重区分，仍保留与低权重样本的区分。几何符号一致率只核对迷宫格点距离的进退，不评价动态可达性、价值校准或候选生成目标的排序能力。</p>'
    sections += f"<p>这批 2048 个查询中，有 {100*probes['repeated_far_near_target_fraction']:.1f}% 的最远与最近 waypoint 索引相同，来自到目标/轨迹末端的截断。M=3 不代表每个样本都提供三个不同的状态；这一数字是诊断查询池的统计，不是全训练集比例。</p>"
    sections += '<h2>记录到的总梯度与参数量</h2>'
    sections += table(['配置','记录梯度中位数','95%分位','最大值','末期中位数','非 target 参数','每次 planner 的 block 调用'],[
        [name,*[f"{r['logged_total_gradient_norm'][k]:.1f}" for k in ('median','p95','maximum','late_median')],r['parameters_excluding_target'],r['planner_block_calls']] for name,r in analysis['runs'].items()])
    sections += '<p>日志每 5,000 更新记录一次梯度统计，具体 grad/norm 为各参数张量 L2 范数之和，不是拼接后的全局 L2 范数。不能排除未记录步的尖峰，也没有分解到 H/L 或不同循环。所有已记录指标有限，h=4 的记录梯度没有呈现持续爆炸；这不证明内部推理收敛。M 增加只多出 790 个参数（约 0.021%），不能把 chain 效果解释成明显增加参数容量。双进程共享 GPU 的耗时受资源竞争影响，不作为严格计算效率基准。</p>'

    task_path=root/'official_task_paths.json'
    if task_path.exists():
        task=json.loads(task_path.read_text())
        assert task['complete']
        diagnostic['official_task_path_subset']={}
        grid=np.array(task['maze']['grid']);xy=np.array(task['maze']['cell_xy'])
        dx=xy[0,1,0]-xy[0,0,0];dy=xy[1,0,1]-xy[0,0,1]
        extent=(xy[0,0,0]-dx/2,xy[0,-1,0]+dx/2,xy[0,0,1]-dy/2,xy[-1,0,1]+dy/2)
        for task_id in task['tasks']:
            fig,axes=plt.subplots(2,3,figsize=(13,8.2),sharex=True,sharey=True,constrained_layout=True)
            for ax,(name,record) in zip(axes.flat,task['runs'].items()):
                assert record['checkpoint_sha256']==probes['runs'][name][str(task['checkpoint'])]['checkpoint_sha256']
                group=record['tasks'][str(task_id)]
                assert len(group['episodes'])==8
                ax.imshow(grid,extent=extent,origin='lower',cmap=ListedColormap(['#f6f7f9','#aeb7c2']),vmin=0,vmax=1)
                for episode in group['episodes']:
                    path=np.array(episode['path_xy'])
                    color='#238569' if episode['success'] else '#c74e4a'
                    ax.plot(path[:,0],path[:,1],color=color,alpha=.65,linewidth=.9)
                    ax.scatter(path[0,0],path[0,1],marker='o',s=12,c='#235789',zorder=3)
                    ax.scatter(path[-1,0],path[-1,1],marker='x',s=22,c=color,zorder=3)
                g=group['episodes'][0]['goal_xy'];ax.scatter(*g,marker='*',s=95,c='#edbc3b',edgecolors='#664400',zorder=4)
                ax.set(title=f"{name.replace('hrm_','')} | {group['summary']['success_count']}/8",aspect='equal')
            fig.suptitle(f'Task {task_id}: first 8 official evaluation seeds, CPU replay at 1M\nGreen: success; red: timeout/failure; blue: start; gold: goal')
            fig.savefig(root/f'task{task_id}_paths.png',dpi=180);plt.close(fig)
        sections += '<h2>正式任务固定起点的轨迹诊断</h2><p>补看 6 个配置在任务 1、5 上的前 8 个正式评估起点，共 96 条轨迹。沿用正式终止规则和种子，用 CPU 重算动作；原实验也设置了 eval_on_cpu=1。这里是新进程执行的诊断子集，不替换 50 次/任务的正式结果，也不把 8 次结果当作模型排名。</p>'
        sections += table(['配置','任务','成功/8','平均最小剩余格点距离','平均最终剩余格点距离'],[
            [name,task_id,f"{v['summary']['success_count']}/8",f"{v['summary']['mean_best_distance_cells']:.1f}",f"{v['summary']['mean_final_distance_cells']:.1f}"]
            for name,x in task['runs'].items() for task_id,v in x['tasks'].items()])
        for name,x in task['runs'].items():
            diagnostic['official_task_path_subset'][name]={t:v['summary'] for t,v in x['tasks'].items()}
        for task_id in task['tasks']:
            sections += f'<img src="task{task_id}_paths.png" alt="Task {task_id} paths">'
    reproduction_path=root/'official_task1_reproduction.json'
    if reproduction_path.exists():
        reproduction=json.loads(reproduction_path.read_text())
        assert reproduction['complete']
        group=reproduction['runs']['hrm_h2l1_m1']['tasks']['1']
        observed=group['summary']['success_count']/group['summary']['count']
        logged=next(row['evaluation/task1_success'] for row in analysis['runs']['hrm_h2l1_m1']['evaluation'] if row['step']==1000000)
        assert math.isclose(observed,logged,abs_tol=1e-12)
        diagnostic['reproduction_check']=dict(method='hrm_h2l1_m1',task=1,checkpoint=1000000,
            success_count=group['summary']['success_count'],episodes=group['summary']['count'],replayed_success=observed,logged_success=logged,match=True)
        sections += '<p><strong>复核：</strong>h2l1M1 在任务 1 的前 8 个起点只成功 1 次，不能据此改变它的正式排名。因此额外重跑了该 checkpoint 的全部 50 个正式起点，得到 28/50=56%，与原 eval.csv 完全一致。这也说明 8 条轨迹只能用于看失败形态，不能用于稳定估计成功率。</p>'
    (root/'diagnostic_summary.json').write_text(json.dumps(diagnostic,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (root/'diagnostic_sections.html').write_text(sections)
    print('Saved diagnostic report assets:',root)


if __name__=='__main__':
    main()
