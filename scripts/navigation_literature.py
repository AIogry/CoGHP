"""Published navigation references and report additions for the local audit.

Literature is curated from primary paper tables. It is deliberately kept out of
the local-run audit and seed pooling. This module never downloads or trains.
"""

import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT.parent / 'docs' / 'CoGHP' / 'experiments'
SOURCE = EXPERIMENTS/'navigation_literature_20261014.json'
LOCAL = ['CoGHP local', 'HRM local', 'HRM baseline seed0', 'G only seed0', 'P only seed0', 'G+P seed0']


def rows_for_comparison(data):
    manifest = json.loads(SOURCE.read_text())
    verification=manifest['verification']
    for key in ('results','sources'):
        if hashlib.sha256(json.dumps(manifest[key],sort_keys=True).encode()).hexdigest()!=verification[f'verified_{key}_sha256']:
            raise ValueError(f'Literature {key} differ from the primary-table verification record')
    rows = []
    for method in manifest['method_order']:
        spec = manifest['results'][method]
        source = manifest['sources'][spec['source']]
        if len(spec['mean'])!=9 or len(spec['sd'])!=9:
            raise ValueError(f'Invalid literature environment alignment: {method}')
        for i,env in enumerate(manifest['environment_order']):
            if spec['mean'][i] is None:
                if spec['sd'][i] is not None:raise ValueError(f'SD without mean: {method} {env}')
                continue
            if not math.isfinite(spec['mean'][i]) or not 0<=spec['mean'][i]<=100 or not math.isfinite(spec['sd'][i]) or spec['sd'][i]<0:
                raise ValueError(f'Invalid published score: {method} {env}')
            rows.append(dict(method=method,environment=env,mean_percent=spec['mean'][i],
                             sd_percent=spec['sd'][i],seeds=source['seeds'],origin='published',
                             source_url=source['url'],source_table=source['table'],
                             source_version=source['version']))
    if len(rows)!=verification['published_records']:
        raise ValueError('Published record count differs from primary verification')
    for method, original in [('CoGHP local','coghp'),('HRM local','hrm_h2l2_truncated')]:
        for r in data['legacy']:
            if r['method']==original:
                rows.append(dict(method=method,environment=r['environment'],mean_percent=100*r['score_mean'],
                                 sd_percent=100*r['seed_sample_sd'],seeds=3,origin='local',
                                 source_url=str(EXPERIMENTS/'analysis_20261014/legacy_summary.csv'),
                                 source_table='54 audited legacy runs; exact three-checkpoint average',source_version='2026-10-09 + 2026-10-10'))
    for method, original in zip(LOCAL[2:],['baseline','g_only','p_only','g_plus_p']):
        for r in data['runs']:
            if r['campaign'].startswith('20261013') and r['method']==original:
                rows.append(dict(method=method,environment=r['environment'],mean_percent=100*r['score_mean'],
                                 sd_percent=None,seeds=1,origin='local',source_url=r['run_dir'],
                                 source_table='exact three-checkpoint average; seed 0',source_version='2026-10-13'))
    return manifest, rows


def render_plots(manifest,rows,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np
    methods=manifest['method_order']+LOCAL
    envs=manifest['environment_order']
    lookup={(r['method'],r['environment']):r for r in rows}
    labels=[m+' [8]' for m in manifest['method_order']]+[
        'CoGHP local [3]','HRM action local [3]','HRM baseline [1]','HRM + G [1]','HRM + P [1]','HRM + G + P [1]']
    matrix=np.full((len(methods),len(envs)),np.nan)
    for i,m in enumerate(methods):
        for j,e in enumerate(envs):
            if (m,e) in lookup:matrix[i,j]=lookup[m,e]['mean_percent']
    plt.rcParams.update({'font.size':11,'figure.dpi':170})
    fig,ax=plt.subplots(figsize=(15.5,10),layout='constrained')
    cmap=plt.get_cmap('YlGnBu').copy();cmap.set_bad('#e8edf2')
    im=ax.imshow(np.ma.masked_invalid(matrix),vmin=0,vmax=100,cmap=cmap,aspect='auto')
    ax.set_xticks(range(9),[f'{fam}\n{size}' for fam in ('Point','Ant','Humanoid') for size in ('Medium','Large','Giant')])
    ax.xaxis.tick_top();ax.set_yticks(range(len(methods)),labels)
    for i,m in enumerate(methods):
        for j,e in enumerate(envs):
            r=lookup.get((m,e))
            if r is None:
                text='—';color='#6b7280'
            else:
                digits=0 if r['origin']=='published' else 1
                text=f'{r["mean_percent"]:.{digits}f}'
                if r['sd_percent'] is not None:text+=f' ± {r["sd_percent"]:.{digits}f}'
                color='white' if r['mean_percent']>=65 else '#172033'
            ax.text(j,i,text,ha='center',va='center',fontsize=10,color=color)
    for y in [5.5,8.5,10.5]:ax.axhline(y,color='white',lw=3)
    for x in [2.5,5.5]:ax.axvline(x,color='white',lw=3)
    ax.set_xticks(np.arange(-.5,9,1),minor=True);ax.set_yticks(np.arange(-.5,len(methods),1),minor=True)
    ax.grid(which='minor',color='white',linewidth=.7);ax.tick_params(which='minor',bottom=False,left=False)
    for spine in ax.spines.values():spine.set_visible(False)
    fig.colorbar(im,ax=ax,shrink=.7,pad=.02,label='Success (%)')
    fig.suptitle('Published OGBench references and local CoGHP / HRM / G / P results',fontsize=16)
    fig.supxlabel('[n] = training seeds; ± = reported / sample SD; — = unreported or unrun.\nRows are references with different recipes, not a controlled cross-paper ranking.',fontsize=11)
    for suffix in ['png','svg']:fig.savefig(out/f'literature_and_local_overview.{suffix}')
    plt.close(fig)
    selected=[envs[i] for i in [7,8,2,5]]
    colors={m:'#94a3b8' for m in methods}
    colors.update({'OTA':'#0f766e','LAVL':'#ea580c','CoGHP paper':'#4338ca',
                   'CoGHP local':'#2563eb','HRM local':'#be185d','HRM baseline seed0':'#475569',
                   'G only seed0':'#0891b2','P only seed0':'#d97706','G+P seed0':'#7c3aed'})
    fig,axes=plt.subplots(1,4,figsize=(19.5,8),sharey=True)
    fig.subplots_adjust(left=.165,right=.985,bottom=.15,top=.9,wspace=.14)
    for ax,e in zip(axes,selected):
        for i,m in enumerate(methods):
            r=lookup.get((m,e))
            if r is None:continue
            marker='o' if r['origin']=='published' else 's' if r['seeds']==3 else '^'
            face='white' if r['origin']=='published' else colors[m]
            ax.errorbar(r['mean_percent'],i,xerr=r['sd_percent'],fmt=marker,color=colors[m],
                        markerfacecolor=face,markeredgewidth=1.8,markersize=7,capsize=3,elinewidth=1.2)
        for y in [5.5,8.5,10.5]:ax.axhline(y,color='#cbd5e1',lw=1)
        ax.set_xlim(-3,107);ax.set_xticks([0,25,50,75,100]);ax.grid(axis='x',alpha=.18)
        ax.set_title(e.replace('-navigate-v0','').replace('maze-','Maze-'))
        ax.set_xlabel('Success (%)');ax.spines[['top','right']].set_visible(False)
    axes[0].set_yticks(range(len(methods)),labels);axes[0].invert_yaxis()
    fig.legend(handles=[Line2D([0],[0],marker='o',color='#475569',markerfacecolor='white',ls='',label='Published: 8 seeds'),
                        Line2D([0],[0],marker='s',color='#2563eb',ls='',label='Local: 3 seeds'),
                        Line2D([0],[0],marker='^',color='#7c3aed',ls='',label='Local: seed 0 (no SD)')],
               loc='lower center',bbox_to_anchor=(.55,.02),ncol=3,frameon=False)
    fig.suptitle('Four current tasks: published references beside local results; error bars = SD',fontsize=16)
    for suffix in ['png','svg']:fig.savefig(out/f'literature_and_local_four_tasks.{suffix}')
    plt.close(fig)


def section(data,out):
    manifest,rows=rows_for_comparison(data)
    out.mkdir(exist_ok=True,parents=True)
    with (out/'literature_and_local_results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (out/'literature_sources.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    (out/'literature_audit.json').write_text(json.dumps(dict(
        source=str(SOURCE),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        **manifest['verification']),ensure_ascii=False,indent=2)+'\n')
    render_plots(manifest,rows,out)
    lookup={(r['method'],r['environment']):r for r in rows}
    def table(headers,items):
        return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join('---' for _ in headers)+' |']+
                         ['| '+' | '.join(map(str,r))+' |' for r in items])
    blocks=[]
    for family in ('point','ant','humanoid'):
        envs=[f'{family}maze-{size}-navigate-v0' for size in ('medium','large','giant')]
        ranks={e:sorted({lookup[m,e]['mean_percent'] for m in manifest['method_order'] if (m,e) in lookup},reverse=True) for e in envs}
        entries=[]
        for method in manifest['method_order']:
            cells=[]
            for e in envs:
                r=lookup.get((method,e))
                if r is None:cells.append('—');continue
                value=f'{r["mean_percent"]:.0f} ± {r["sd_percent"]:.0f}'
                if r['mean_percent']==ranks[e][0]:value=f'**★ {value}**'
                elif r['mean_percent']==ranks[e][1]:value=f'***☆ {value}***'
                cells.append(value)
            entries.append([method]+cells)
        blocks.append('### '+family.capitalize()+'Maze 文献结果\n\n'+table(['方法 8 seeds','Medium','Large','Giant'],entries))
    references='[OGBench Table 2](https://arxiv.org/html/2410.20092v2)、[OTA Table 6](https://arxiv.org/html/2505.12737v2)、[LAVL Table 1](https://arxiv.org/html/2605.25740v1)、[CoGHP Table 1](https://arxiv.org/html/2602.03389v1)'
    comparison=table(['环境','OTA 文献','LAVL 文献','本地 HRM 三种子','G only seed 0','P only seed 0','G 加 P seed 0'],[
        [e.replace('-navigate-v0','')]+[f'{lookup[m,e]["mean_percent"]:.2f}' for m in ['OTA','LAVL','HRM local','G only seed0','P only seed0','G+P seed0']]
        for e in [manifest['environment_order'][i] for i in [7,8,2,5]]])
    return f'''## OGBench 文献基线与本地结果的差距

本节覆盖 GCBC、GCIVL、GCIQL、QRL、CRL、HIQL 六种 OGBench 原始基线，以及 OTA、LAVL 和已发表的 CoGHP 数字。所有数据限定为 state-based `*-navigate-v0` 的 Medium、Large、Giant，与本地任务名称对应；stitch、explore、oraclerep、visual 和 100m 结果不混入。

OGBench、OTA 和 LAVL 明确使用 800k、900k、1M 平均、五个目标、每目标 50 rollouts，主表为八种子结果。CoGHP 表 1 报告八种子 SD 和五目标均值，并说明遵循 OGBench，但正文未逐项展开 checkpoint 聚合和每目标 rollout 数；该行保留为论文参照。公开结果的 SD 原样保留，不假定论文的 ddof 等于本地 ddof=1。LAVL 附录 B.2 的 best-evaluation 结果不用于本节。来源和版本见 {references} 及 [来源记录]({out/'literature_sources.json'})。

### 九环境合图

![文献与本地结果总览]({out/'literature_and_local_overview.png'})

同一图中，上方是论文八种子成绩，中间为本地三种子均值，下方为本轮 seed 0。方括号是独立训练种子数，灰色空格表示未报告或未运行。CoGHP paper 与 CoGHP local 分成两行，不能用论文 79 替换本地 Point Giant 的 56.18；HRM 三种子与本轮 HRM seed 0 也分别显示，后者是 G/P 的配对基准。

![四个本轮任务的文献和本地对照]({out/'literature_and_local_four_tasks.png'})

空心圆是文献，实心方块是本地三种子，三角是本地单种子；误差线为各自 SD，单种子不画跨种子误差线。两幅图显示相同任务下的差距，但没有控制代码版本、调参预算、batch size 和网络结构，因此不能当成完全受控的跨论文胜负检验。图的 SVG 版本与 PNG 同目录，可用于放大或排版。

### 基线方法在解决什么问题

| 方法 | 主要学习机制 | 与本项目的关系 |
| --- | --- | --- |
| GCBC | 以未来 goal 条件化的行为克隆 | 检查简单模仿能达到的水平 |
| GCIVL | expectile TD 学 V，再抽取策略 | legacy Value 的基础路线 |
| GCIQL | expectile V 加 Q 的离线策略抽取 | 单层 value-based 对照 |
| QRL | 带约束的 quasimetric 距离学习 | 强调时间距离的结构归纳偏置 |
| CRL | 对比学习状态与未来 goal 的匹配 | 通过不同目标学习可达性信息 |
| HIQL | 一个 Value 驱动高层 subgoal 和低层 action | 两层策略抽取的参照 |
| OTA | 用 option 时间抽象改进高层 Value | G 的备份设计借鉴点 |
| LAVL | 非对称 state/goal latent 的负距离 Value，加连续性约束 | Value 表示几何的独立研究方向 |

六种基线的机制对应 [OGBench 官方实现](https://github.com/seohongpark/ogbench)。OTA 改变高层训练的时间尺度；LAVL 的 LAN 使用 `V(s,g)=-||phi_S(s)-phi_G(g)||`，其 state/goal encoder 不必相同。它与我们 bilinear Global 的结构不同，也不等同于 P 的 position-command encoder。来源：[OTA 论文](https://arxiv.org/html/2505.12737v2)、[LAVL 官方实现](https://github.com/oh-lab/LAVL)。

### 文献成绩及醒目标记

下列三表按每个环境、已列出的文献方法的**显示均值**标记：**★ 最优**，***☆ 次优***；并列均值保留并列，次优取第二个不同均值。标记不表示统计显著性，也不同于原论文使用的“最佳值附近全部加粗”规则。本地结果的醒目标记在各自三种子表和四条件表内单独计算。来源为 {references}。

{(chr(10)*2).join(blocks)}

### 差距如何解读

{comparison}

Humanoid Giant 的 G 加 P 82.13 与 LAVL 的文献均值 83 数值接近，仍比 OTA 的 92 低 9.87 个百分点；G only 为 78.67。这里的“接近”只描述单种子与八种子显示均值的距离，尚不构成方法等效性。Humanoid Large 的 G only 59.33，距离 LAVL 的 74 和 OTA 的 83 分别为 14.67、23.67 个百分点，改善还没有补齐该任务的差距。

Point Giant 是最明显的不足：LAVL、CoGHP paper、OTA 参照分别为 91、79、72；本地原始 CoGHP 为 56.18 ± 12.85，本地 HRM 为 41.91 ± 15.94。本轮 P only 为 46.53，G 加 P 为 10.13。当前存在本地原始方法与论文成绩的差距，也存在 HRM 改造后的进一步退化，这两层问题应分别分析。

Ant Giant 的本地 HRM 三种子均值为 79.96，数值处于 CoGHP paper 78、OTA 77 与 LAVL 85 之间；本轮 G/P 未带来新的最终分数优势。Medium 任务接近饱和，少量百分点差距的解释价值低于 Giant 的大幅差距。

文献是外部参照，未在本机重新训练 OTA/LAVL。OGBench 的原始实现按任务调策略抽取参数；OTA 按任务选择抽象长度和高层温度；LAVL 官方配置使用 batch 1024、maze expectile 0.9、discount 0.999，而本轮为 batch 256、expectile 0.7、legacy discount 0.99/0.995、Global 抽象 discount 0.99。这些差异可从各论文附录核对，意味着图中差距同时包含方法和配方差异。[OTA 参数表](https://arxiv.org/html/2505.12737v2)、[LAVL 参数表](https://arxiv.org/html/2605.25740v1)

论文 CoGHP 的导航超参数是按规模合写的汇总列，本地使用的是已记录的逐环境配方。例如本地 Point Giant 的 K=3、latent dim=32、lr=1e-5；论文表 3 的导航统一列是 K={{1,2,2}}、embedding dim={{32,32,128}}、lr=3e-4。两者并非一份已验证逐任务一致的参数清单，所以本地数字标作复现结果，不声称精确重现论文主表。[CoGHP 参数表](https://arxiv.org/html/2602.03389v1)

数值源表为 [literature_and_local_results.csv]({out/'literature_and_local_results.csv'})，78 个文献 mean/SD 对照项已逐项核对原 PDF 表格，记录见 [literature_audit.json]({out/'literature_audit.json'})。公开结果仅保留原表精度，缺失值保持缺失；复制在多篇论文中的同一基线数字不作为额外独立实验。

'''


def design_section():
    return f'''### 从 HRM action 到 Global value 和 position command

这次设计沿着两个问题展开。原来的 HRM-action 强化的是“拿到 command 以后如何产生动作”，高层 command 的评价方式和 full-state 语义仍沿用旧 CoGHP。因此，executor 改造后如果仍无法稳定导航，需要分别检查“哪个 command 值得学习”和“command 表达了什么”。这里把已有 HRM 结果作为设计背景，并不把当前诊断事后当成当初已知事实。

| 设计步骤 | 从 HRM baseline 改动的位置 | 设计目的 | 本轮如何区分作用 |
| --- | --- | --- | --- |
| 保留 HRM-action 主干 | vanilla Mixer planner、HRM H2L2 executor、远到近链 | 在已验证执行框架上研究评价与接口 | 新旧 Baseline 的全部评估逐项核对 |
| 增加独立 Global | 高层权重从旧 V 切到 Phi_G | 评价位置目标下的较长程进度 | G only 加原 full-state command |
| 改为 position latent | teacher command 从 phi_old(s,w) 切到 phi_P(s,q) | 去除未来身体状态作为命令输入，保留当前状态条件 | P only 加旧 Value |
| 加位置 decoder | encoder 输出附加 reconstruction 约束 | 给 latent 明确的位置监督，检查能否恢复命令 | 重建、位置误差、cycle 和落墙诊断 |
| 分开表示与梯度 | Global、position、legacy critic 分工 | 让各轴独立运行，防止高层拟合通过改变标签获得低 loss | G 加 P 及逐 loss 梯度检查 |

**Global 的形成过程。** 我们保留旧 Value 服务低层短程 AWR，新增 `Phi_G(s,q)` 专门服务高层。当前 state 仍含身体动态，goal 只取 XY；独立 state/goal encoder 通过 bilinear head 输出 twin value。其训练采用 OTA 式最多 25 步的备份，遇到位置 goal 即终止，否则在片段端点 bootstrap，一次应用抽象 discount。这样把高层学习权重的语义改为位置进度，同时避免直接替换低层已有监督。这个版本是位置目标与边界 mask 的适配，不是官方 OTA 完整复现。[实施需求]({ROOT/'docs/10-12/prompt1.md'})、[代码]({ROOT/'impls/agents/navigation.py'})

**Position 的形成过程。** 原 latent `phi_old(s,w_j)` 包含数据中未来 robot state 的信息。我们将未来状态 w_j 投影成 q_j=XY(w_j)，用完整当前状态 s 与标准化相对位移共同编码，得到 `z_j=phi_P(s,(q_j-XY(s))/sigma)`。planner 仍直接预测高维 z 链；executor 保留现有 token 维数并读取完整链。没有改成直接二维坐标输出，是为了保留原接口和当前状态条件化的表示。decoder 给 z 提供位置重建监督，也让错误预测可以被检查；高维 latent 本身不会增加未来信息或自动保证可达。[设计讨论]({ROOT/'docs/10-12/plan2.md'})、[position 模块]({ROOT/'impls/utils/encoders.py'})

**为什么不共用 Global goal encoder 和 position encoder。** Global 的 `c_G(q)` 为粗尺度 value 查询服务，而 `phi_P(s,q)` 为当前局部动作命令服务；两者的状态条件和损失来源不同。独立模块让 P only 不依赖 G，也使 G only 与 G 加 P 的 Global 学习可直接核对。低层动作损失与重建损失训练 position encoder；高层 targets、prefix 和最终 goal command 都 stop-gradient，防止遗漏最终 goal 的梯度入口。共享的当前 state feature 仍接收高低层梯度，这一共享关系保留在报告的模型说明中。[梯度设计]({ROOT/'docs/10-12/prompt1.md'})

两个配置轴最终形成 Baseline、G only、P only、G 加 P。它们固定同环境的训练预算、subgoal 数与间隔、HRM 计算结构和正式评估协议；G 检验高层权重改造，P 检验 command 学习协议，两者组合检验相互作用。四格可以分析条件效应，但 P 仍同时包含输入、辅助 loss 和梯度改动，不能据它单独隔离“删除未来姿态”的因果贡献。

'''
