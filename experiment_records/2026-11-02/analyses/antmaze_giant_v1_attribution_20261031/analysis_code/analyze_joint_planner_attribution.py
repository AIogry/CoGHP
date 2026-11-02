#!/usr/bin/env python3
"""Combine paired v0/v1 diagnostics without claiming an unobserved factorial decomposition."""

import argparse
import hashlib
import html
import json
import math
import os
from pathlib import Path
import pickle
import shutil
import statistics

os.environ.setdefault('JAX_PLATFORMS','cpu')
import numpy as np
from analyze_joint_planner_v0 import rows, digest


def fingerprint(tree):
    h=hashlib.sha256()
    def visit(x,path=''):
        if isinstance(x,dict):
            for key in sorted(x):visit(x[key],path+'/'+key)
        else:
            a=np.asarray(x)
            h.update(path.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    visit(tree)
    return h.hexdigest()


def table(headers,values):
    row=lambda items:'<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in items)+'</tr>'
    return '<table><thead>'+row(headers)+'</thead><tbody>'+''.join(row(x) for x in values)+'</tbody></table>'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.output
    context=json.loads((root/'context.json').read_text())
    oldroot=Path(context['v0_analysis'])
    old=json.loads((oldroot/'analysis.json').read_text())
    manifest=json.loads((root/'v1/snapshot/campaign.json').read_text())
    capture=json.loads((root/'v1/snapshot/captured_at.json').read_text())
    result=dict(captured_at=capture['time'],runs={},paired_probes={},gradient_probes={},parameter_fingerprints={},
                identifiability='Beta comparisons are conditional on the v1 bundle. Command and encoder changes are confounded in training outcomes.')
    for name in ['hrm_h2l1_m1','hrm_h4l2_m3']:
        r=old['runs'][name]
        result['runs'][name]=dict(version='v0',h=r['h'],l=r['l'],M=r['M'],beta=3.,status=r['status'],
            score=r['score'],evaluation=r['evaluation'],per_task_score=r['per_task_score'])
    for name,method in manifest['study']['methods'].items():
        d=root/'v1/snapshot/runs'/name;c=method['overrides'];state=json.loads((d/'run.json').read_text())
        e=rows(d/'eval.csv');tr=rows(d/'train.csv')
        selected=[x for x in e if x['step'] in (800000,900000,1000000)]
        score=statistics.mean(x['evaluation/overall_success'] for x in selected) if len(selected)==3 and state['status']=='completed' else None
        if score is not None:assert math.isclose(score,state['metric']['mean'],abs_tol=1e-12)
        result['runs'][name]=dict(version='v1',h=c['planner_h_cycles'],l=c['planner_l_cycles'],M=c['num_subgoals'],beta=c['high_alpha'],
            status=state['status'],score=score,evaluation=e,last_train_step=tr[-1]['step'],
            per_task_score=[statistics.mean(x[f'evaluation/task{i}_success'] for x in selected) for i in range(1,6)] if score is not None else None)
        if score is not None:
            result['parameter_fingerprints'][name]={}
            for step in (800000,900000,1000000):
                path=Path(state['run_dir'])/'artifacts'/f'params_{step}.pkl'
                with path.open('rb') as stream:checkpoint=pickle.load(stream)
                params=checkpoint['agent']['network']['params']
                result['parameter_fingerprints'][name][str(step)]={k:fingerprint(params[k]) for k in ('modules_goal_rep','modules_value','modules_target_value')}
    names=list(result['parameter_fingerprints'])
    for step in ('800000','900000','1000000'):
        for group in ('modules_goal_rep','modules_value','modules_target_value'):
            assert len({result['parameter_fingerprints'][name][step][group] for name in names})==1
    result['value_and_goal_encoder_identical_across_v1_completed']=True

    old_roll=json.loads((oldroot/'local_rollout_probes.json').read_text())
    new_roll=json.loads((root/'v1_local_rollouts.json').read_text())
    assert new_roll['complete'] and old_roll['indices']==new_roll['indices']
    original_jvp=json.loads((oldroot/'checkpoint_probes_extended.json').read_text())
    for version in ('v0','v1'):
        p=json.loads((root/f'{version}_checkpoint_probes.json').read_text())
        oldq=np.load(oldroot/'checkpoint_probes_extended.queries.npz');q=np.load(root/f'{version}_checkpoint_probes.queries.npz')
        for key in oldq.files:assert np.array_equal(oldq[key],q[key]),key
        for name,ck in p['runs'].items():
            assert set(ck)=={'800000','900000','1000000'}
            mean=lambda fn:statistics.mean(fn(row) for row in ck.values())
            r=result['runs'][name]
            record=dict(near_mse=mean(lambda x:x['waypoints'][-1]['mse']),
                near_target_rank=mean(lambda x:x['waypoints'][-1]['target_effective_rank']),
                near_prediction_rank=mean(lambda x:x['waypoints'][-1]['predicted_effective_rank']),
                near_constant_prediction_mse=mean(lambda x:x['waypoints'][-1]['constant_target_mean_mse']),
                teacher_action_mse=mean(lambda x:x['actions']['teacher_data_mse']),
                predicted_action_mse=mean(lambda x:x['actions']['predicted_data_mse']),
                prediction_teacher_action_distance=mean(lambda x:x['actions']['predicted_teacher_action_distance']),
                near_unit_response_rms=math.sqrt(mean(lambda x:x['actions']['near_unit_direction_action_fd_mse'])),
                far_unit_response_rms=math.sqrt(mean(lambda x:x['actions']['far_unit_direction_action_fd_mse'])),
                near_geometric_sign=mean(lambda x:x['waypoints'][-1]['geometric_sign_agreement']),
                near_distant_geometric_sign=mean(lambda x:x['waypoints'][-1]['distant_geometric_sign_agreement']),
                near_cap_fraction=mean(lambda x:x['waypoints'][-1]['cap_fraction']),
                near_ess_fraction=mean(lambda x:x['waypoints'][-1]['ess_fraction']))
            roll=(old_roll if version=='v0' else new_roll)['runs'][name]
            assert roll['checkpoint_sha256']==ck['1000000']['checkpoint_sha256']
            record['local_success_counts']={mode:v['success_count'] for mode,v in roll['summary'].items()}
            if version=='v0':
                for step,x in ck.items():
                    for part in ('near','far'):
                        a=x['actions'][f'{part}_unit_direction_action_fd_mse']
                        b=original_jvp['runs'][name][step]['actions'][f'{part}_unit_direction_action_jvp_mse']
                        assert math.isclose(a,b,rel_tol=.01,abs_tol=1e-9),(name,step,part,a,b)
            result['paired_probes'][name]=record

    g=json.loads((root/'gradient_probes/mechanism_probes.json').read_text());assert g['complete']
    for name,x in g['runs'].items():
        high=statistics.mean(v['l2_norm_by_group']['high']['planner'] for v in x['gradients'])
        low=statistics.mean(v['l2_norm_by_group']['low']['planner'] for v in x['gradients'])
        assert all(v['l2_norm_by_group'][obj]['goal_rep']==0 for v in x['gradients'] for obj in ('high','low'))
        result['gradient_probes'][name]=dict(high_norm=high,low_norm=low,ratio_of_mean_norms=low/high,
            high_low_cosine_mean=statistics.mean(v['cosine']['planner:high:low'] for v in x['gradients']))
    (root/'attribution_summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

    os.environ.setdefault('MPLCONFIGDIR','/tmp/coghp-analysis-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors={.5:'#235789',1.:'#28876f',2.:'#d67b29'}
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),sharey=True,constrained_layout=True)
    for ax,m in zip(axes,(1,3)):
        for name,r in result['runs'].items():
            if r['M']!=m:continue
            label='v0, beta=3' if r['version']=='v0' else f"v1, beta={r['beta']:g}"+(' (incomplete)' if r['score'] is None else '')
            ax.plot([v['step']/1000 for v in r['evaluation']],[v['evaluation/overall_success']*100 for v in r['evaluation']],
                    color='#777777' if r['version']=='v0' else colors[r['beta']],linestyle='--' if r['version']=='v0' else '-',marker='o',markersize=3,label=label)
        ax.axvspan(800,1000,color='black',alpha=.05)
        ax.set(title='H2/L1/M1' if m==1 else 'H4/L2/M3',xlabel='Training updates (thousands)',ylim=(0,80));ax.grid(alpha=.18);ax.legend(fontsize=9)
    axes[0].set_ylabel('Official success (%)')
    fig.suptitle('Observed outcome of the v1 bundle | paired seed 0; partial snapshot')
    fig.savefig(root/'success_comparison.png',dpi=180);plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(14,5),constrained_layout=True)
    names=list(result['paired_probes']);x=np.arange(len(names));labels=[n.replace('hrm_','').replace('beta','b') for n in names]
    for i,name in enumerate(names):
        p=result['paired_probes'][name];color='#777777' if result['runs'][name]['version']=='v0' else colors[result['runs'][name]['beta']]
        axes[0].bar(i,p['near_target_rank'],color=color);axes[0].text(i,p['near_target_rank']+.3,f"{p['near_target_rank']:.1f}",ha='center',fontsize=8)
        for shift,mode,c in [(-.18,'predicted','#235789'),(.18,'teacher','#d67b29')]:
            count=p['local_success_counts'][mode];axes[1].bar(i+shift,count,.32,color=c,label=mode if i==0 else None)
        for shift,part,c in [(-.18,'near','#235789'),(.18,'far','#d67b29')]:
            axes[2].bar(i+shift,p[f'{part}_unit_response_rms'],.32,color=c,label=part if i==0 else None)
    for ax in axes:
        ax.set(xticks=x,xticklabels=labels);ax.tick_params(axis='x',rotation=65,labelsize=8);ax.grid(axis='y',alpha=.15)
    axes[0].set(title='Goal-representation effective rank',ylabel='Near-command target rank',ylim=(0,22))
    axes[1].set(title='Local reference-waypoint reach',ylabel='Successful starts out of 32',ylim=(0,35));axes[1].legend(fontsize=8)
    axes[2].set(title='Unit correction direction response',ylabel='RMS action derivative (finite difference)');axes[2].legend(fontsize=8)
    fig.savefig(root/'mechanism_comparison.png',dpi=180);plt.close(fig)

    score_rows=[]
    for name,r in result['runs'].items():
        common=next(v['evaluation/overall_success'] for v in r['evaluation'] if v['step']==600000)
        score_rows.append([name,r['status'],f"{100*common:.2f}%",'尚未形成' if r['score'] is None else f"{100*r['score']:.2f}%"])
    metrics=table(['配置','近端 MSE','目标有效秩','预测/数据 command 动作差','近端响应','远端响应','几何进展符号一致'],[
        [n,f"{p['near_mse']:.4f}",f"{p['near_target_rank']:.2f}",f"{p['prediction_teacher_action_distance']:.5f}",f"{p['near_unit_response_rms']:.5f}",f"{p['far_unit_response_rms']:.5f}",f"{100*p['near_geometric_sign']:.1f}%"] for n,p in result['paired_probes'].items()])
    local=table(['配置','原生预测','数据 command','只换近端','只换远端'],[[n,*[p['local_success_counts'].get(k,'—') for k in ('predicted','teacher','corrected_near','corrected_far')]] for n,p in result['paired_probes'].items()])
    gradtable=table(['配置','high→planner L2','low→planner L2','均值范数之比 low/high','两梯度平均 cosine'],[
        [n,f"{p['high_norm']:.2f}",f"{p['low_norm']:.2f}",f"{p['ratio_of_mean_norms']:.2f}",f"{p['high_low_cosine_mean']:.3f}"] for n,p in result['gradient_probes'].items()])
    body=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>CoGHP v1 条件影响分析</title>
<style>body{{font:16px/1.8 system-ui,sans-serif;max-width:1180px;margin:36px auto;padding:0 24px;color:#243249}}table{{width:100%;border-collapse:collapse;font-size:14px}}td{{padding:9px;border-bottom:1px solid #dce3eb}}thead{{background:#edf2f7}}img{{width:100%}}.note{{padding:16px;background:#edf3f8}}h2{{margin-top:34px}}code{{overflow-wrap:anywhere}}</style>
<h1>CoGHP v1：哪些影响已能分离，哪些仍未识别</h1>
<p>快照：{capture['time']}。5 组完成，1 组仍在训练。所有补充诊断使用 CPU；未修改训练、checkpoint 或补充实验排程。</p>
<p class="note">可直接比较：同一 h/ℓ/M 下 β 的效果。可定位机制：goal_rep 的变化、command 输入敏感性、low loss 到 planner 的梯度。不可分配：最终成功率下降中 command 与 value-only encoder 各占多少。缺失独立训练对照，不能假定效果可线性相加。</p>
{table(['配置','快照状态','600k 对齐成绩','800k/900k/1M 主分数'],score_rows)}
<img src="success_comparison.png" alt="v0 v1 learning curves">
<h2>1. β 的影响可以在 v1 内部直接比较</h2>
<p>M1 的主分数随 β=0.5→1→2 为 1.87→5.87→6.53%；M3 已完成两组为 11.60→18.80%。M3 的 β=2 尚未形成主指标；600k 时，β=0.5/1/2 为 8.0/17.6/13.2%，不支持当前就断言越大越好。</p>
<p>更强的控制证据：五个已完成模型在 800k、900k、1M 的 goal_rep、value、target_value 参数指纹分别完全相同。它们采用同一个 critic 和 encoder，β 之间的差异发生在 policy 学习通路。该结果符合 value-only 梯度隔离、相同初始化与采样流、无全局梯度裁剪的实现。具体模型间的 β 对照仍只有 seed 0。</p>
<h2>2. value-only 改变了表征，但低秩不等于失去控制信息</h2>
<p>同样 2048 个原始状态查询上，近端目标表征有效秩从 v0 的 18.74/17.85，降到 v1 各组共同的 5.16。value 与几何进展的符号一致率仍约 89.7%，与旧版约 89–90% 接近。因此没有证据表明 critic 的粗略进展判断整体崩溃；表征压缩和控制适用性需要分开分析。</p>
<p>在 value-only 分支中，encoder 更新不依赖 actor command 或 β，因而可把该分支的表征变化定位到梯度来源变化。但这不说明低有效秩一定丢失了必要信息：有效秩受坐标表示影响，近端 latent MSE 也不能跨编码器直接解释为物理误差。最终成功率贡献仍需 data-command/value-only 的训练对照。</p>
{metrics}<img src="mechanism_comparison.png" alt="representation and execution diagnostics">
<h2>3. command gap 的缩小可能伴随目标控制能力下降</h2>
<p>v0 M1 的预测/数据 command 动作差约 0.01409；v1 β=0.5 只有 0.00189，但其正式成功率也由 63.60% 降为 1.87%。单看 gap 缩小会得出错误的乐观结论；近端纠正方向响应也从 0.02851 降到 0.01267，说明输入改变对动作的影响变弱。</p>
<p>M3 变化更明显：v0 近端/远端单位纠正响应约 0.03257/0.00337；v1 β=0.5 为 0.01273/0.04106，β=1 为 0.03623/0.02337。远端槽位在新版中对动作更有影响，但它们是否仍承担期望的 waypoint 语义，不能仅凭敏感性判断。</p>
<p>下表为 1M checkpoint、相同 32 个验证数据起点、最多 100 步内经过参考近端 XY waypoint 的次数；输入 command 保留完整 state 编码：</p>
{local}
<p>旧版数据 command 是训练分布；新版数据 command 已是替换输入，不能称为新版的 oracle 性能上界。该诊断衡量参考目标的局部控制能力，不是五项正式任务成功率；原生 planner 可以走另一条有效路线。M3 中给数据 command 反而变差，支持“planner 与 executor 对生成 command 的共同适应改变了接口语义”的假设，尚不能把该现象完全归给单独的 command 开关。</p>
<h2>4. 预测 command 同时打开了 low loss 到 planner 的梯度</h2>
<p>这轮条件（2）包含两项机制：executor 看到预测输入，以及 low loss 经预测均值回到 planner。旧版第二条路径为零。当前 1M 模型的四个固定 64 样本批次中：</p>
{gradtable}
<p>M3 β=0.5 的 low→planner 平均梯度范数约为 high→planner 的 4.86 倍；β=1 降至 1.91 倍。β 不仅改变样本排序，还改变两个目标的相对训练压力。两梯度大多近正交，不能说已经发现强烈的梯度抵消。范数是当前批次的原始梯度，不是 Adam 实际更新量，也不是训练历史或成功率的因果分解。</p>
<p>一个优先检验的机制假设是：部分 plan 槽位被用来承载重建数据动作的内部特征，偏离了可替换、可执行的目标编码。有限的 high loss 仍约束它拟合数据 latent，但这种约束可能不足以维持原有接口语义。</p>
<h2>5. 补齐哪些训练条件才可分离总体效果</h2>
{table(['固定同一 β、h/ℓ/M','encoder 联合更新','encoder 仅 value'],[['数据 command','已有 v0，β=3','缺少对照 A'],['预测 command','缺少对照 B','已有 v1；β=3 补充实验已排程']])}
<p>在 β=3 下补 A/B，每种 M 各两组，共四个训练运行，才构成完整 2×2 对照。既有 v1 的 β=3 能去除跨轮 β 差异，但仍只能评价两个改动的组合。若进一步区分条件（2）的两个机制，可在预测 command/value-only 下单独停止 low loss 经 command 回到 planner；这不能用纯推理开关代替重新训练。</p>
<p>这些是建议的后续对照，本次没有启动。也没有把旧 encoder 直接替换进新 policy 来声称因果效果：它们的 latent 坐标和配套网络可能不兼容。对纯训练梯度开关做推理期切换，不能撤销其历史训练影响。</p>
<h2>诊断复现与限制</h2>
<p>数据：attribution_summary.json、v0/v1_checkpoint_probes.json、v1_local_rollouts.json、gradient_probes/mechanism_probes.json。v1 原始日志快照在 v1/snapshot/，旧版对照来自上一份冻结分析。所有 checkpoint 带签名和 SHA256。</p>
<p>敏感性采用中心有限差分 ε=0.02。value-only 模式对外部数据 command 有 stop_gradient，直接对这个接口求 JVP 会产生人为的零；有限差分比较真实前向值，不会受到这个梯度门的干扰。旧版有限差分与之前的解析 JVP 结果已在 1% 相对误差内核对。方向是预测目标到数据目标的单位纠正方向，不是完整 Jacobian 范数。动作探针使用裁剪前 Gaussian 均值；局部环境 rollout 使用实际裁剪动作。</p>
</html>'''
    (root/'report.html').write_text(body)
    copies=root/'analysis_code';copies.mkdir(exist_ok=True)
    files=['analyze_joint_planner_attribution.py','probe_joint_planner_attribution.py','audit_joint_planner_attribution_gradients.py','probe_joint_planner_rollouts.py','analyze_joint_planner_v0.py']
    for name in files:shutil.copy2(Path(__file__).parent/name,copies/name)
    provenance=dict(snapshot_time=capture['time'],diagnostic_agents_updated=False,training_schedule_changed=False,
        completed_v1_models=5,paired_offline_queries=2048,probed_checkpoints=21,local_rollouts=448,
        gradient_batches=20,code_sha256={name:digest(copies/name) for name in files})
    (root/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('Saved attribution report:',root/'report.html')


if __name__=='__main__':main()
