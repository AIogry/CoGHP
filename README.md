<div align="center">

# Chain-of-Goals Hierarchical Policy for Long-Horizon Offline Goal-Conditioned RL

## [Paper](https://arxiv.org/abs/2602.03389) / [Project Page](https://wlsdn9350.github.io/projects/coghp/)

<img src="assets/coghp.png" alt="CoGHP main figure" width="75%">

</div>

# Overview
**CoGHP** is a unified hierarchical policy for long-horizon offline goal-conditioned reinforcement learning. It generates a **chain-of-goals**, a sequence of latent subgoals followed by the primitive action, using an autoregressive policy. This formulation enables the agent to decompose long-horizon tasks into intermediate steps while maintaining awareness of the final goal.

# Installation
CoGHP implementations require Python 3.9+ and additional dependencies, including `jax >= 0.4.26`.
To install these dependencies, run:

```shell
cd impls
pip install -r requirements.txt
```

# Navigation research framework

The research method and the original CoGHP baseline have independent agent and
policy implementations. `agents/navigation.py` does not inherit from or import
the CoGHP agent or its Mixer policy. The algorithm registry loads the baseline
only when it is selected.

| Location | Responsibility |
| --- | --- |
| `impls/agents/navigation.py` | Navigation objectives, global/detail values, command learning and parameter updates |
| `impls/agents/coghp.py` | Original CoGHP comparison baseline |
| `impls/networks/backbones.py` | Backbones, including their local SwiGLU/RMSNorm implementation and H/L state updates |
| `impls/networks/policies.py` | Subgoal-chain generation, planner/executor assembly and action readout |
| `impls/utils/networks.py` | Higher-level actor/value definitions; accepts native backbones |
| `impls/utils/encoders.py` | Observation, goal and position-command encoders/decoders |
| `impls/utils/datasets.py` | Goal relabeling, trajectory sampling and global fragment fields |
| `impls/utils/coghp_network.py` | Original CoGHP policy, used only as a baseline |

There is no separate computation package. Numerical layers belong to their
backbone. The old HRM token-vocabulary wrapper and external model configuration
classes are retired; H/L recurrence is a state-update option in the framework.

From `impls/`, select the method explicitly:

```bash
python main.py --agent=agents/navigation.py --env_name=antmaze-large-navigate-v0
python main.py --agent=agents/coghp.py --env_name=antmaze-large-navigate-v0
```

The new navigation defaults use independent SwiGLU/RMSNorm backbones for feature
and command encoding, actor heads, values, the feed-forward planner and the HRM
executor. They are a new development configuration, not the configuration used
by the completed October experiments. Global value and position commands remain
explicit method options (`high_weight_source=global_ota`, `command_content=position`).

Historical configurations remain available through `agents/navigation.py:legacy_action`
and `agents/navigation.py:legacy_navigation`. Those profiles retain their previous
GELU/MLP paths and serialized parameter names, and are not the new-method defaults.
The study runner also accepts the historical method IDs and resolves them to
these profiles when the old implementation files are absent. Existing frozen
experiment snapshots continue to use their own original entry files.

Historical study JSON files and generated reports live outside this source tree
in `../docs/CoGHP/experiments/`. The current planner v0 specification is versioned
with the implementation in `configs/joint_planner_v0.json`.
A new campaign requires an explicit `--study` path;
an existing campaign reads its frozen specification from `campaign.json`.
Campaign-specific regression checks use that external directory, configurable
through `COGHP_STUDY_DIR`.

## Joint HRM planner v0

`agents/navigation.py:joint_planner` selects the first controlled planner design.
Only the high actor uses HRM: it updates the full plan's H/L states, then reads
out **all sparse waypoints together at the end**. The planner never receives a
ground-truth waypoint prefix. Its objective is `joint_plan_objective=waypoint_final`.

| Component | v0 setting |
| --- | --- |
| State/goal encoders and value | CoGHP-style GELU MLPs, hidden-layer LayerNorm; encoder outputs retain length normalization |
| High/low actor readout heads | CoGHP-style GELU MLPs, no LayerNorm |
| Peripheral MLP dimensions | `(512,512,512)` |
| Latent/token dimension | 32 by default; 128 in the AntMaze-Giant campaign |
| Planner H/L blocks | SwiGLU Mixer with RMSNorm, intermediate dimension **32** |
| Executor | Independent CoGHP-style GELU Mixer, token/channel intermediate dimensions **32** |
| External refinement segments \(K\) | 1 |
| Internal H cycles \(h\), L updates before each H cycle \(\ell\) | 2, 2 by default |
| Sparse targets \(M\) | 3 by default |

One planning call performs `h * l` L updates and `h` H updates, then applies the
same high-actor head to every waypoint position. H and L blocks have separate
parameters, reused across their respective updates. The final loss backpropagates
through **all H/L updates within that call** (`planner_bptt=True`). External
`planner_refinement_steps` is fixed to 1, so the generic detached returned carry
does not truncate any v0 training path. Recurrent states reset on each planning
call; full BPTT does not extend across environment steps.

Planner SwiGLU token mixing is noncausal. The executor retains CoGHP's original
partial triangular mask on `tm_weights`; its preceding token MLP still mixes
all positions. This is not a strict causal attention mask. Planner and executor
are separate cores (`share_actor_core=False`), unlike default original CoGHP's
shared core. No SwiGLU/RMSNorm peripheral networks or HRM executor are used in v0.

Research note (2026-10-28): prefer a **shared readout head for intermediate
recurrent states and the final state**, with `y_r = D_phi(h_r)` using the same
parameters `phi` across readout depths/rounds. Revisit this idea with the user
when designing intermediate supervision, early exits or variable inference
budgets. The pilot shared `high_actor_head` across refinement rounds. V0 uses
only the final readout, with the same head shared across waypoint positions;
optional intermediate H/L readouts remain a future design choice.

The v0 objective assigns each data waypoint its own detached AWR weight:

```text
A_j = mean_twin V(s_target_j, g) - mean_twin V(s_current, g)
w_j = stop_gradient(exp(min(high_alpha * A_j, log(awr_max_weight))))
c_j = high_discount ** (M - j)       # j=1 is farthest, j=M is nearest
L_high = -mean_batch [sum_j c_j * w_j * log p_final(target_j)] / (M * subgoal_steps)
```

Only the final plan is supervised, with no average or scaling by `h` or `l`.
The data targets and weights retain stopped gradients. The fixed unit Gaussian
variance makes this objective equivalent in planner gradient to weighted squared
latent-target error. With `high_discount=0.8`, the far-to-near weights for M=3 are
`[0.64, 0.8, 1.0]`. These are existing detail-value progress scores, not a new
trajectory critic: no accumulated rewards or explicit segment-feasibility scores
are introduced. Formal v0 experiments fix `high_weight_source=detail_value`.

Targets are full-state **encoded commands**, preserving the existing goal
encoder and executor interface; they are not raw-state predictions or a
position/velocity-only reduction. Slots remain far-to-near with nominal offsets
`[75, 50, 25]` under profile defaults; formal PointMaze-Large runs use
`[150, 100, 50]` for M=3. Sampling clips targets at the relabeled goal or
episode boundary, so repeated endpoint targets retain the existing padding
semantics. The executor consumes the entire final plan. It still trains on data
commands by default and replans at every environment step during evaluation.
Thus planner-prefix teacher forcing is removed, while the executor's
data-command/predicted-command distribution difference remains. Interval-based
execution, online critic feedback and adaptive stopping are later experiments.

Run a CPU pipeline check from the repository root (existing data only):

```bash
bash scripts/run_joint_planner_smoke.sh
# Arguments are run directory, h, l, M; external K is always 1:
bash scripts/run_joint_planner_smoke.sh "$PWD/impls/exp/joint_v0_h1l1_m1" 1 1 1
# A cheaper check shrinks only peripheral MLPs, and is not a formal model run:
SMOKE_SMALL=1 TRAIN_STEPS=16 bash scripts/run_joint_planner_smoke.sh
```

The default check uses full `(512,512,512)` peripheral networks, latent and Mixer
dimensions 32, batch 16 and 128 updates on pointmaze-medium (`subgoal_steps=25`).
It includes fixed-batch validation, one environment evaluation episode and
checkpoint saving. `SMOKE_SMALL=1` changes peripheral MLPs to `(32,32)` while
keeping the latent/Mixer dimensions at 32. Both modes are pipeline checks, not
performance comparisons. The launcher prints the selected dimensions and saves
the complete resolved configuration in `flags.json`. Set `PYTHON`,
`OGBENCH_DATASET_DIR` and `TRAIN_STEPS` as needed. Each run directory must be new.

### First formal h/l/M experiment

`configs/joint_planner_v0.json` specifies 12 configurations with
`h in {1,2,4}`, `l in {1,2}` and `M in {1,3}`, external K=1, and seed 0.
It uses PointMaze-Large, `subgoal_steps=50`, latent/Mixer dimensions 32,
peripheral MLPs `(512,512,512)`, batch 256, learning rate 3e-4 and 1M updates.
Evaluation covers every task, with 50 episodes per task every 100k updates and
temperature 0. The primary score averages updates 800k, 900k and 1M.

The study runner loads the native agent entry point, so this JSON explicitly
overrides its development defaults to the v0 settings. It does not rely on the
native default configuration being changed. Preview it without starting training:

```bash
PYTHON=/home/eai/Tools/miniforge3/envs/brain_nav/bin/python
RUN_ROOT=/data/qijunrong/06-RL/offline-rl/exp/joint_planner_v0_width32_seed0
"$PYTHON" scripts/run_study.py --study configs/joint_planner_v0.json --run-root "$RUN_ROOT" --gpus 0,1 --dry-run
# Use --execute in place of --dry-run when starting the campaign.
# Add --methods hrm_h2l2_m3 to start with the reference configuration alone.
```

The runner requires existing datasets and a campaign path outside this source
tree, freezes the actual source/configuration, and records dataset hashes.
Seed 0 is exploratory; add seeds 1/2 in a new study/campaign after reviewing this
first pass, without modifying an existing campaign's frozen specification.
Increasing h/l changes compute, while changing M also changes horizon, token
count and token-mixer parameter count. Report update count and runtime alongside
success. These are within-family HRM controls, not a capacity-matched comparison
with original CoGHP.

### Completed AntMaze-Giant campaign

`configs/joint_planner_v0_antmaze_giant.json` records the October 29 campaign:
the same 12 h/l/M combinations, seed 0, `subgoal_steps=50`, latent dimension
**128**, and Mixer intermediate dimensions **32**. Only the planner uses HRM;
the objective remains final-only, per-waypoint AWR with full within-call BPTT.
Training used physical GPU 1 with two concurrent processes, and evaluations
ran on CPU. The primary score remains the mean at 800k, 900k and 1M updates.

Preview the corresponding launch settings with a new campaign directory:

```bash
PYTHON=/home/eai/Tools/miniforge3/envs/brain_nav/bin/python
RUN_ROOT=/data/qijunrong/06-RL/offline-rl/exp/joint_planner_v0_antmaze_giant_new
"$PYTHON" scripts/run_study.py --study configs/joint_planner_v0_antmaze_giant.json \
  --run-root "$RUN_ROOT" --gpus 1 --workers-per-gpu 2 --wandb-mode offline --dry-run
```

The runner checks GPU occupancy and tracks its own live children so that only
the requested concurrent runs share the selected GPU. It freezes source and
configuration before training and keeps existing attempts intact.

The analysis entry points are `scripts/analyze_joint_planner_v0.py` (log
snapshots and reports), `scripts/probe_joint_planner_checkpoints.py` (paired
offline queries), `scripts/probe_joint_planner_rollouts.py` (local command
interventions), `scripts/probe_joint_planner_task_paths.py` (task trajectories),
`scripts/summarize_joint_planner_diagnostics.py` (combined diagnostic figures),
and `scripts/audit_joint_planner_mechanisms.py` (shared-head readouts and gradient
routing). They read frozen experiment sources and saved checkpoints; they do
not update model parameters. Run each with `--help` for its inputs and output
directory contract. Generated analysis outputs under `impls/exp/`, datasets,
and checkpoints are separate from the versioned implementation.

### Earlier endpoint/deep-supervision pilot

`agents/navigation.py:joint_planner_pilot` preserves the October 27 prototype:
three H2/L2 segments, detached cross-segment states, truncated within-segment
gradients, a readout/loss at every segment, and one farthest-endpoint AWR weight
shared by all waypoints and segments. It also retains the native SwiGLU/RMSNorm
peripheral networks and HRM executor. Prior smoke results describe that pilot,
not v0. Do not resume pilot checkpoints using the v0 objective or architecture.
Autoregressive/default and historical profiles retain their original behavior.

# Reproducing the main results
An agent can be trained by executing `main.py` script.

Commends:
```shell
# pointmaze-medium-navigate-v0
python main.py --env_name=pointmaze-medium-navigate-v0 --eval_episodes=50 --agent=agents/coghp.py
# pointmaze-large-navigate-v0
python main.py --env_name=pointmaze-large-navigate-v0 --eval_episodes=50 --agent=agents/coghp.py --agent.num_subgoals=1 --agent.subgoal_steps=50
# pointmaze-giant-navigate-v0
python main.py --env_name=pointmaze-giant-navigate-v0 --eval_episodes=50 --agent=agents/coghp.py --agent.num_subgoals=3 --agent.subgoal_steps=50 --agent.feature_dim=32 --agent.lr=1e-5

# antmaze-medium-navigate-v0
python main.py --env_name=antmaze-medium-navigate-v0 --eval_episodes=50 --agent=agents/coghp.py
# antmaze-large-navigate-v0    
python main.py --env_name=antmaze-large-navigate-v0 --eval_episodes=50 --agent=agents/coghp.py --agent.num_subgoals=1 --agent.subgoal_steps=50 --agent.feature_dim=128
# antmaze-giant-navigate-v0
python main.py --env_name=antmaze-giant-navigate-v0 --eval_episodes=50 --agent=agents/coghp.py --agent.num_subgoals=2 --agent.subgoal_steps=50 --agent.feature_dim=128

# cube-single-noisy-v0
python main.py --env_name=cube-single-noisy-v0 --eval_episodes=50 --agent=agents/coghp.py --agent.subgoal_steps=10 --agent.num_subgoals=1 --agent.gc_enc=None --agent.feature_dim=32
# cube-double-noisy-v0
python main.py --env_name=cube-double-noisy-v0 --eval_episodes=50 --agent=agents/coghp.py --agent.subgoal_steps=10 --agent.num_subgoals=1 --agent.gc_enc=None --agent.feature_dim=128 --agent.lr=3e-5
# cube-triple-noisy-v0
python main.py --env_name=cube-triple-noisy-v0 --eval_episodes=50 --agent=agents/coghp.py --agent.subgoal_steps=10 --agent.num_subgoals=1 --agent.gc_enc=None --agent.feature_dim=256 --agent.lr=3e-5

# scene-play-v0
python main.py --env_name=scene-play-v0 --eval_episodes=50 --agent=agents/coghp.py --agent.subgoal_steps=10 --agent.num_subgoals=1 --agent.gc_enc=None --agent.feature_dim=256

# visual-antmaze-medium-navigate-v0
python main.py --env_name=visual-antmaze-medium-navigate-v0 --train_steps=500000 --eval_episodes=50 --agent=agents/coghp.py --agent.encoder=impala_small --agent.high_alpha=3.0 --agent.low_actor_rep_grad=True --agent.low_alpha=3.0
# visual-cube-single-noisy-v0
python main.py --env_name=visual-cube-single-noisy-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/coghp.py --agent.encoder=impala_small --agent.high_alpha=3.0 --agent.low_actor_rep_grad=True --agent.low_alpha=3.0 --agent.p_aug=0.5 --agent.feature_dim=32 --agent.subgoal_steps=10
```

# Navigation CoGHP / HRM Action study

The completed October 9–10 navigation campaigns are associated with one
consolidated baseline, tagged **`exp/navigation-20261012-baseline`** on branch
**`research/navigation`**. [Experiment registry](../docs/CoGHP/experiments/registry.json)
records both campaign locations, original snapshot hashes, protocol and software
versions. [Per-run results](../docs/CoGHP/experiments/navigation_results_20261012.csv) contains
54 runs (nine environments, two methods, three seeds), the three exact evaluation
scores, their mean, and source CSV/config hashes. Success rates are fractions in
`[0, 1]`; multiply by 100 for percentages.

This is a retrospective consolidated version: the original runs used uncommitted
working-tree snapshots based on `4f5242f`. Both snapshots and this baseline have
identical training/model/environment source (140 files); the two original full
snapshots differ in launch/management/test/documentation files, recorded in the
registry. Their original metadata is preserved. The fixed tag associates both
campaigns with this baseline without claiming they ran from a pre-existing Git
commit. Resolve its commit with `git rev-parse exp/navigation-20261012-baseline^{commit}`.
Future development continues on the branch; the experiment tag remains fixed.

`../docs/CoGHP/experiments/navigation_hrm_action.json` declares six navigation environments,
original CoGHP seeds 0/1/2, and vanilla-planner HRM H2L2 **truncated** seed 0
(24 runs). Scientific settings live in this file, not shell environment variables.
The protocol is 1M updates, evaluation every 100k, all tasks, 50 episodes/task,
temperature 0. The primary score averages steps **800000, 900000, 1000000**;
those three checkpoints are retained. HRM has one seed and no across-seed SD.

```bash
cd /home/eai/Research/CoGHP
PYTHON=/home/eai/Tools/miniforge3/envs/brain_nav/bin/python
RUN_ROOT=/data/qijunrong/06-RL/offline-rl/exp/CoGHPNavigation_20261009
"$PYTHON" scripts/run_study.py --study ../docs/CoGHP/experiments/navigation_hrm_action.json --run-root "$RUN_ROOT" --gpus 0,1 --dry-run
# Run manually after reviewing the plan and reserving both GPUs:
"$PYTHON" scripts/run_study.py --study ../docs/CoGHP/experiments/navigation_hrm_action.json --run-root "$RUN_ROOT" --gpus 0,1 --execute
"$PYTHON" scripts/run_study.py --run-root "$RUN_ROOT" --summarize
```

The launcher uses one sequential worker per GPU, checks GPU occupancy/JAX,
hashes all datasets, and copies the actual tracked/untracked source into
`<run-root>/source/`. The snapshot records the observed Git commit, dirty state,
and SHA256 hashes, so uncommitted source is not misrepresented as clean HEAD.
Training imports only the copied project. Keep the snapshot unchanged; the same
campaign always reuses it. This is a content snapshot, not a detached Git worktree.
W&B defaults to offline (`--wandb-mode online` is optional); local CSVs are primary.

Artifacts live at `runs/<method>/<environment>/seed_NNN/attempt_NNN/`:
`run.json` records command/GPU/status, `stdout.log` captures output, and
`artifacts/` contains `flags.json`, `reproducibility.json`, `train.csv`, `eval.csv`,
checkpoint metadata and `params_<step>.pkl`. Summaries print to the terminal and
reject missing/duplicate metric steps instead of substituting a tail average.
Validated completed attempts are skipped; partial/failed attempts are preserved
and require an explicit `--attempt N` with `--methods`/`--environments` filters.
Retries are fresh training runs, not optimizer-state resume. A failure stops new
dispatch while in-flight jobs finish; Ctrl-C/SIGTERM stops child process groups.
For later launches use the copied `source/scripts/run_study.py` if development
code has changed. The old grid scripts remain legacy entry points and do not
define this study.

## Humanoid and three-seed completion

`../docs/CoGHP/experiments/navigation_hrm_action_multiseed_extension.json` adds exactly 30 runs:
HumanoidMaze medium/large/giant with both methods at seeds 0/1/2 (18), and HRM
PointMaze/AntMaze medium/large/giant seeds 1/2 (12). The previous 24 runs are reused,
giving 54 distinct identities. Humanoid uses `subgoal_steps=100`, `feature_dim=128`,
`discount=0.995`, and K=1/1/2. Other numerical settings and evaluation remain fixed.

Results use `/data/qijunrong/06-RL/offline-rl/exp/Navigation/<date>_<description>/`.
Frozen runtime code uses `/home/eai/Research/Navigation/<date>_<description>/`;
continue development in `/home/eai/Research/CoGHP`, never in the frozen directory.
`--source-root` separates code from artifacts, and `--prepare` hashes datasets and
copies source without accessing GPUs or starting training. `campaign.json` records
the source location; `source/` in the result directory is a compatibility symlink.
On subsequent invocations the stored study and source are used automatically.

```bash
PYTHON=/home/eai/Tools/miniforge3/envs/brain_nav/bin/python
NAME=20261010_coghp_vs_hrm_h2l2_humanoid_and_seed_completion
RUN_ROOT=/data/qijunrong/06-RL/offline-rl/exp/Navigation/$NAME
SOURCE_ROOT=/home/eai/Research/Navigation/$NAME

# One-time preparation from the development tree (no training):
"$PYTHON" scripts/run_study.py \
  --study ../docs/CoGHP/experiments/navigation_hrm_action_multiseed_extension.json \
  --run-root "$RUN_ROOT" --source-root "$SOURCE_ROOT" --prepare

# Review and launch from the prepared code:
"$PYTHON" "$SOURCE_ROOT/scripts/run_study.py" --run-root "$RUN_ROOT" --gpus 0,1 --dry-run
"$PYTHON" "$SOURCE_ROOT/scripts/run_study.py" --run-root "$RUN_ROOT" --gpus 0,1 --execute

# Joint 54-run summary; no files are rewritten:
"$PYTHON" "$SOURCE_ROOT/scripts/run_study.py" --run-root "$RUN_ROOT" --summarize \
  --summarize-with /data/qijunrong/06-RL/offline-rl/exp/Navigation/20261009_coghp_vs_hrm_h2l2_point_ant_pilot
```

Joint summaries check training/model/environment source hashes, shared dataset
hashes, agent settings and evaluation protocol. Duplicate method/environment/seed
identities are rejected. Missing runs remain incomplete, not a reduced-seed mean.
Different seeds within a method can be declared per environment; methods can also
restrict their environment scope. These declarations prevent baseline duplication.

## Consolidated October 14 navigation results

[Configuration, results and diagnostic analysis](../docs/CoGHP/experiments/analysis_20261014/report.md)
integrates the three completed navigation campaigns: 54 three-seed original
CoGHP / HRM-action runs and 16 seed-0 Global / Position runs. The latter use
**HRM-action CoGHP as their baseline**, not original CoGHP. The new baseline
reproduces all ten historical seed-0 evaluation records exactly in each of the
four overlapping environments; these reruns are not additional independent seeds.

[Per-run results](../docs/CoGHP/experiments/analysis_20261014/runs.csv),
[full evaluation histories](../docs/CoGHP/experiments/analysis_20261014/evaluations.csv),
[resolved configurations](../docs/CoGHP/experiments/analysis_20261014/resolved_configs.json),
and [audit checks](../docs/CoGHP/experiments/analysis_20261014/audit.json) preserve provenance.
The primary score is the exact 800k/900k/1M mean. Three-seed sample SD and
single-seed factorial results are reported separately.

The same report includes primary-paper navigation results for the six OGBench
baselines, OTA, LAVL, and published CoGHP, beside local results in combined
figures. [Literature source records](../docs/CoGHP/experiments/navigation_literature_20261014.json)
preserve table versions, seed counts, PDF hashes, and the verified numerical
matrix. Published and local results are not pooled.

Recompute the derived tables, plots and report without training or modifying
the raw campaign directories:

```bash
python3 scripts/analyze_navigation_campaigns.py
```

# Acknowledgments
This code is based on [OGBench](https://github.com/seohongpark/ogbench) repository.

# Citation
```shell
@article{choi2026chain,
  title={Chain-of-Goals Hierarchical Policy for Long-Horizon Offline Goal-Conditioned RL},
  author={Choi, Jinwoo and Lee, Sang-Hyun and Seo, Seung-Woo},
  journal={arXiv preprint arXiv:2602.03389},
  year={2026}
}
```
