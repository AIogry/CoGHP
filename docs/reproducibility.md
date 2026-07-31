# Reproducible CoGHP experiments

The training entry point derives independent random streams from `--seed`:

- training dataset sampling;
- validation dataset sampling;
- environment initialization;
- evaluation tasks and episodes;
- JAX model initialization and updates.

Every evaluation checkpoint uses the same task/episode seeds. Within an
episode, environment randomness, policy sampling, and optional Gaussian action
noise use separate streams. Validation uses a fixed batch by default and never
advances the training dataset generator.

## Required launch environment

Use the repository training scripts. With their default `DETERMINISTIC=1`,
they set:

```bash
PYTHONHASHSEED=${SEED}
XLA_FLAGS="... --xla_gpu_deterministic_ops"
```

For direct launches, set the same variables before Python imports JAX:

```bash
SEED=0
PYTHONHASHSEED=${SEED} \
XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_deterministic_ops" \
python impls/main.py --seed=${SEED} ...
```

The following must also remain identical:

- Git commit and uncommitted source changes;
- dataset files;
- Python/JAX/XLA/CUDA/cuDNN/MuJoCo versions;
- model configuration and batch size;
- GPU model and number of processes per GPU.

Each run writes `reproducibility.json` with resolved RNG seeds, software
versions, devices, and relevant environment variables.

## Verification

Run two experiments from scratch with the same configuration and seed, then:

```bash
python scripts/verify_reproducibility.py \
  /path/to/run_a \
  /path/to/run_b
```

The verifier requires exact equality of every non-time CSV metric and every
checkpoint tensor. Wall-clock metrics under `time/` are ignored. To diagnose
small platform-level differences instead of requiring bitwise equality, pass
an explicit tolerance such as `--atol=1e-7`.

## Scope

The guarantee targets fresh runs on the same machine. Exact interrupted-run
resume is not yet guaranteed because NumPy dataset-generator state is not part
of the model checkpoint, and the current restore path does not resume the
outer training-loop index.
