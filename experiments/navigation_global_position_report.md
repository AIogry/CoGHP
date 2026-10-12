# Global Potential / Position-Latent Navigation implementation and experiment report

This report documents the implementation requested in `docs/10-12/prompt1.md`.
Status at implementation freeze: correctness, four GPU smoke runs, paired local
execution and exact process-resume checks passed. Formal launch is the next
authorized action. **No final 1M-update result exists yet.** The campaign's
`campaign.json`, per-run `run.json` and `results.txt` are the live authorities;
the source snapshot of this report records pre-launch evidence.

## Research and scope

The question is whether an independently trained, temporally abstracted global
value and a position-conditioned latent command improve navigation separately
and together. All four conditions retain the verified vanilla CoGHP Mixer
planner and HRM H2L2 stage-truncated executor. The baseline is the HRM-action
agent, not the original all-vanilla CoGHP comparison from the earlier campaign.

| Condition | High weight | Command |
|---|---|---|
| baseline | legacy Value | original full-state-derived latent |
| g_only | Global OTA-style value | original full-state-derived latent |
| p_only | legacy Value | position latent |
| g_plus_p | Global OTA-style value | position latent |

No inference-time search, reranking, dynamics model, local Q, new core family,
or persistent recurrent memory has been added. HRM cycles remain fixed.

## Provenance and configuration

- Starting branch / commit: `research/navigation`, `b8a520dd2354c24c73089249642deee4de475410`.
- Starting worktree: clean. Development branch: `research/global-position-latent`.
- Historical flags checked: AntMaze-Giant HRM H2L2 seed 0, campaign
  `20261009_coghp_vs_hrm_h2l2_point_ant_pilot`.
- Formal configuration: `experiments/navigation_global_position.json`.
- Environment: `antmaze-giant-navigate-v0`; fixed existing train/validation NPZ files.
- K=2, subgoal interval=50, feature dimension=128, batch=256, Adam lr=3e-4.
- Actor/value/legacy encoder hidden layers=(512,512,512); one original planner
  Mixer block; HRM one layer, intermediate=256, H=2, L=2, BPTT=False,
  SwiGLU/RMSNorm, float32, causal; separate planner/executor cores.
- Legacy discount=.99, expectile=.7, tau=.005, actor chain discount=.8,
  high/low inverse temperature=3. Action training remains teacher-forced.
- 1M updates; formal evaluation every 100k, all 5 tasks, 50 episodes/task,
  temperature=0, independent deterministic evaluation seed, CPU evaluation.
- Main score: average success at exactly 800k/900k/1M within each seed, then
  across-seed mean and sample SD (ddof=1). No final-only or last-5 score.
- Queue is seed-major: all four seed-0 conditions, then seed 1, then seed 2.
- Formal experiment ID: `20261013_global_potential_position_latent_antmaze_giant`.
- Formal source: `/home/eai/Research/Navigation/20261013_global_potential_position_latent_antmaze_giant/source`.
- Formal results: `/data/qijunrong/06-RL/offline-rl/exp/Navigation/20261013_global_potential_position_latent_antmaze_giant`.
- Exact implementation commit, full source hashes, dataset SHA256 and resolved
  config are recorded by the launcher in `campaign.json`; no dirty snapshot is
  to be passed off as a clean Git commit.

## Implementation map

- `impls/agents/navigation_coghp.py`: four modes, losses, target update,
  gradient/representation diagnostics. The baseline delegates to the old path.
- `impls/utils/navigation_commands.py`: position encoder/decoder, independent
  twin bilinear Global with complete target encoders and saved coordinate constants.
- `impls/utils/navigation_dataset.py`: non-mutating legacy sampler wrapper and
  position-success abstract fragments.
- `impls/utils/hrm_action_coghp_network.py`: optional external goal token,
  planner-only command detach, and latent diagnostic output. Default path unchanged.
- `impls/main.py`: train-only coordinate statistics, resolved flags, diagnostics,
  sampler-state checkpoint/resume for the new agent.
- `impls/utils/flax_utils.py`: additional semantic checkpoint signature fields.
- `scripts/run_study.py`: new agent support, seed-major queue, corrected runtime
  environment metadata key (it previously overwrote the task name).
- `scripts/verify_navigation_baseline.py`: real historical checkpoint compatibility.
- `scripts/verify_navigation_resume.py`: real GPU process restart, original
  frozen source, new output directory, exact full-state/sampler comparison.
- `scripts/diagnose_navigation_commands.py`: paired restored-state local execution.
- `scripts/diagnose_navigation_values.py`: geometry-only evaluation of value ordering.
- `impls/test_navigation_coghp.py`: mathematical, gradient, invariance, boundary,
  initialization, checkpoint, and baseline tests.
- `model_parameter_counts.json` in each formal run records stored parameter
  counts by group, explicitly including target copies and inactive modules.
- Each completed training run executes held-out CPU value/command probes at
  800k/900k/1M, then refreshes `results.txt`. Failures stop dispatch of new jobs;
  a probe failure is recorded separately from an already valid training result.
  Occupancy is checked again before each queued GPU job.

## Mathematical definitions and gradients

Position commands: r=(q-pos(s))/sigma; z=phi(s,r). Encoder and decoder hidden
layers=(128,128). The learned 128-D latent uses stable length normalization to
sqrt(128); raw 2-D displacement is never length-normalized. The scale is the
population standard deviation of all TRAIN observations' XY, floored at 1e-6.
Global absolute goals use (q-train_mean)/sigma. Statistics are stored in resolved
flags, a statistics file, the checkpoint signature, and model constants.

Position reconstruction is mean squared error averaged over batch, K+1 command
slots (final goal plus all subgoals), and XY dimensions. Coefficient=1.0. No
extra action labels or relabeled random-goal actions are introduced.

\[
\mathcal L=\mathcal L_{V,legacy}+\mathcal L_L+\mathcal L_H
  +\mathbf1_P\mathcal L_{pos}+\mathbf1_G\mathcal L_G,
\qquad
\mathcal L_H=-\mathbb E\left[\frac1K\sum_{j=1}^K
\frac{.8^{K-j}}{50}\operatorname{sg}(\min(e^{3A_j},100))
\log\pi_H(\operatorname{sg}(z_j)\mid s,\operatorname{sg}(z_g),
\operatorname{sg}(z_{<j}))\right].
\]

The shared current-state feature still receives the high loss. In G conditions,
\(A_j=\frac12\sum_i[\Phi_i(w_j,q_g)-\Phi_i(s,q_g)]\); the low loss uses its
unchanged legacy weights. Here K=2 and the displayed 50 is the fixed subgoal
interval, not the Global abstraction length (25).

Planner targets, teacher prefixes, and final goal tokens all stop the position
encoder gradient. Executor tokens remain live. Position encoder receives low
AWR and reconstruction gradients; decoder receives reconstruction only. The
existing current-state `feature_embed` is intentionally shared between actors
and receives both high and low losses. In position mode the old goal encoder
belongs to the legacy critic; its original actor-gradient coupling is removed.
This is part of the P protocol, not a claim of pure input deletion.

Global head i: Phi_i(s,q)=u_i(s)^T c_i(q)/sqrt(128), separate MLPs with
(512,512,512) hidden layers and GELU/LayerNorm. No final unit normalization.
Global loss updates only online Global parameters. Its entire twin target is
initialized as an online copy. EMA matches existing IVL's convention: target
at the end of an update uses the PRE-update online parameters and tau=.005.

High progress is mean_i Phi_i(w_j,q_g)-mean_i Phi_i(s,q_g), or the identical
legacy Value difference in P-only/baseline. Weights are detached and capped at
100; new paths cap log-weight before exp to avoid overflow. High loss uses
gamma_H^(K-j)/subgoal_steps and averages across K exactly as legacy code.
The progress score is not claimed to be a complete option-return advantage.

Low AWR retains V_old(s',w_near)-V_old(s,w_near), including the complete future
state in its training weights. The Global endpoint query also uses full w_j.
Only the position Actor's forward command inputs exclude future body state.

## Global data protocol

Abstraction length=25; gamma_G=.99. Environment `_goal_tol` is read at runtime
(AntMaze=.5). Global reuses the already sampled legacy `value_goals` XY:
current/future/random mixture=.2/.5/.3 with legacy future-goal sampling. This
does not consume extra RNG or modify legacy samples/rewards/masks/goals.

For each start, inspect offsets 0..25 within its actual trajectory. First
position success gives reward=0, mask=0 and that first-hit endpoint. Otherwise
a complete 25-step fragment gives reward=-1, mask=1 at its endpoint. An
unsuccessful incomplete fragment is excluded using a separate `global_valid`;
an early successful incomplete fragment remains usable. Exclusion is logged.
Dataset endpoints/timeouts are not labeled as permanently unreachable goals.

Compact OGBench retains the final observation at valids=0 and also marks the
preceding action terminal. Global uses the actual final-observation boundary;
it never joins episodes. Legacy sampling remains unchanged. Since source NPZ
does not distinguish physical termination from timeout, both are treated as
fragment boundaries; no additional absorbing-failure semantics are invented.

Targets y_i=r+gamma_G*m*target_Phi_i(endpoint,q). The shared expectile sign uses
r+gamma_G*m*min_i target_Phi_i(endpoint,q)-mean_i target_Phi_i(start,q).
Head-specific residuals and the sum of twin losses follow legacy IVL. Valid
examples alone enter the loss denominator (safe zero if none are valid).

OTA source checked: https://github.com/ota-v/ota-v/blob/main/agents/ota.py and
https://github.com/ota-v/ota-v/blob/main/utils/datasets.py. The first-hit XY
success and incomplete-fragment mask are our explicit adaptation; this is not
advertised as an unchanged official OTA reproduction.

## Validation evidence

- Final combined validation: **58 tests passed in 65.330 seconds**.
- 11 new tests passed, including per-loss/per-parameter gradients AND actual
  isolated Adam parameter changes, final-goal gradient separation, actor
  future-body invariance, autoregressive causality, finite zero-input gradients,
  hand-computed twin expectile loss, exact chain coefficients, complete Global
  EMA, independent modules and identical common initializations, checkpoint
  optimizer/constants/next-update equality, compact boundaries/first hits, and
  unchanged legacy samples and RNG continuation.
- 26 study-runner tests passed (23 existing plus the four-condition seed order,
  diagnostic CPU isolation/failure behavior, and saved-checkpoint validation).
- Real historical AntMaze-Giant 1M HRM checkpoint: old vs new baseline Value
  loss, Actor loss (including diagnostics), and deterministic action arrays
  were exactly equal, maximum absolute difference 0.
- Existing HRM-core regression suite: 21 tests passed.
- Real GPU smoke: four conditions x seed 0 x 2000 updates, logs every 500,
  checkpoints 1000/2000, 2 episodes/task at 2000. Smoke success is NOT a formal
  method comparison. All four completed; all recorded numerical metrics finite.
- G+P GPU process resume from 1000 to 2000: all 681 serialized agent leaves
  (parameters, target copies, optimizer, constants, RNG/update count) exactly
  equal to uninterrupted training; both sampler RNG states exactly equal.
- Paired local diagnostics: 24 held-out starts for each P condition; recovered
  observations exactly equal to stored observations (max absolute error=0).
- All four post-training diagnostic hooks were exercised on smoke checkpoints.

Evidence directory: `/home/eai/Research/Navigation/preflight_global_position_20261012`.
Primary records: `final_tests.log`, `real_checkpoint_compatibility.json`,
`resume_g_plus_p/verification.json`, `smoke_summary.json`, and
`diagnostic_hook_verification.json`. The smoke campaign has its own dirty-code
snapshot and hash `e03f81d2acf3007f1219d805587a4416bb25d512b2a3737392c2f9fc2ebf55a0`;
its starting Git HEAD alone does not identify that implementation. Later changes
add diagnostic probes, launcher reporting and parameter inventory; the tested
learning equations and gradients are unchanged.

### Observed smoke values (2000 updates; NOT final performance)

| Check | P-only | G+P |
|---|---:|---:|
| Validation normalized position reconstruction MSE | .018131 | .018335 |
| High loss gradient norm to position encoder | 0 | 0 |
| Low loss gradient norm to position encoder | 17.6269 | 9.2750 |
| Reconstruction gradient norm to position encoder | 1.0898 | 1.3901 |
| Latent MSE drift since 1500 updates | .21806 | .24898 |
| Teacher / Predicted local success (24 starts) | 1/24 / 1/24 | 3/24 / 3/24 |
| Decoded predicted waypoint wall-cell fraction (512 starts) | .32520 | .35449 |

All four 2-episode/task smoke navigation scores were zero. These very short runs
only validate operation; they provide no final ranking of methods. G-only and
G+P Global losses/values/weights were identical at every logged step, an
additional real-data check of isolation. Final validation high-weight cap rates
were .203/.129 for G groups and .203/.172 for P-only; a separate 512-example
probe gave baseline .236/.160 versus G .260/.188. Thus clipping is substantial
but not an unexplained G-only saturation; beta/cap were left unchanged.

The 512-start geometry probe had 223 non-tied pairs at >=10-cell distance:
Global progress sign agreed with geometric progress on .6996 of pairs versus
legacy baseline .4709. This is a diagnostic observation from an immature value,
with dependent pairs and a geometric proxy, not evidence of statistically
established planning ability. On held-out behavior goals at offsets 0/5/10/25,
the Global mean values were -.008/.035/.052/.002, versus -.303/-1.203 at
50/100. The near-goal plateau is consistent with the chosen first-hit,
25-step abstract reward; its effect on useful high-level ordering must be
assessed after training.

## Local execution and research diagnostics

AntMaze train/validation and Humanoid validation NPZ were checked and contain
qpos/qvel. The diagnostic loader requests them explicitly. Each rollout reset
restores both arrays, runs MuJoCo forward, and requires observation agreement
(atol=rtol=2e-4). Failure produces no fabricated rollout metric.

Teacher mode holds data's absolute near/far waypoints and re-encodes each step.
Predicted mode uses the native free-running planner each step, without decoding
and re-encoding as a repair. Both use identical saved starts, a common reference
near waypoint and 100-step budget. This comparison is a LOCAL diagnostic, not
official task success or a strict upper bound: a different valid predicted
route can miss the reference waypoint. No geometric/oracle correction is used.

Logs separate teacher reconstruction, predicted decoded position error,
teacher vs predicted decode/re-encode cycle error, latent prediction MSE,
teacher/predicted action MSE, conditional latent spread, and encoder gradient
norms from high/low/reconstruction losses. Fixed-batch latent drift is measured
between log points; probe arrays are saved at evaluation/checkpoint steps.

Value ordering diagnostics compare signed advantage against cell-BFS progress
on held-out state/goal pairs, excluding geometric ties and invalid cells, with
a separate >=10-cell distance bin. BFS is diagnostic-only and is not the
robot's dynamics-optimal temporal distance. Position waypoint wall-cell rate
is also diagnostic, not an inference-time filter.
The additional fixed-state value slices query on-trajectory goals at behavior
offsets 0/5/10/25/50/100. They expose near-goal plateaus without treating those
behavior offsets as optimal temporal distances.

The diagnostic scripts initially passed the dataset ID directly to Gym,
which rejected `antmaze-giant-navigate-v0`. This was an environment factory bug
in the standalone probes, fixed by using OGBench's `env_only=True` factory.
The failed logs were retained and successful probes rerun. It did not affect
the training environment, dataset, checkpoint or official evaluation.

## Resource audit and launch status

Host: 32 logical CPUs, about 60 GiB RAM with about 50 GiB available at audit.
Two RTX 4090 24 GiB devices. Sandbox nvidia-smi failed, but non-invasive host
inspection and a JAX calculation succeeded. GPU 0 has a ToDesk process; it was
left alone. GPU 1 was empty and is the only assigned training GPU.

Smoke tmux: `nav-gp-smoke-20261012`; launcher PID at start: 396871.
Smoke logs and immutable source are in the evidence directory's `smoke/`.
Formal launch gates have passed. Assigned GPU: 1 only, one process at a time.
Formal tmux session: `nav-global-position-20261013`.
Launcher log: `/home/eai/Research/Navigation/20261013_global_potential_position_latent_antmaze_giant/launcher.log`.
Run logs and checkpoints use
`<results>/runs/<baseline|g_only|p_only|g_plus_p>/antmaze-giant-navigate-v0/seed_<000|001|002>/attempt_000/`:
`stdout.log`, `run.json`, `artifacts/train.csv`, `artifacts/eval.csv`,
`artifacts/params_{800000,900000,1000000}.pkl`, sampler sidecars and diagnostic
JSON/logs. Resumes must use a new artifact directory and a matching signature.

The queue stops after a training/probe error, never silently changes methods or
restarts from an uncertain checkpoint. No automatic hyperparameter tuning or
additional environment runs are authorized by this campaign. Results become
formal only when all three designated checkpoint scores exist for that seed;
across-seed SD is shown only after all declared seeds have completed.

## Interpretation limits and next steps

Position reconstruction prevents a trivial constant code from reconstructing
different commands, but does not guarantee navigability or good off-manifold
predictions. The auxiliary loss and gradient decoupling are part of P. AWR
clipping, temporal reward scale, additional Global parameters, and label drift
remain alternative explanations. No learned adaptive HRM halting or online
queryable planning is claimed. Do not expand the algorithm in response to
poor early success. Source/config are frozen before the formal four-condition
queue, seed 0 first and seeds 1/2 if preceding runs finish without errors.
