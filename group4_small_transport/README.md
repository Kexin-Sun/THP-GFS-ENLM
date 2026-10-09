# Group 4: small hydraulic and thermal transport

Five complete four-field GFS runs use seed `12345` and the following matrix:

| Case | nu | K | Theta |
| --- | ---: | ---: | ---: |
| Baseline | 0.3 | 0.1 | 0.1 |
| Small thermal transport | 0.3 | 0.1 | 1e-6 |
| Small hydraulic transport | 0.3 | 1e-6 | 0.1 |
| Both transports small | 0.3 | 1e-6 | 1e-6 |
| Both small, nearly incompressible | 0.4999999 | 1e-6 | 1e-6 |

This is a physical-parameter study. It is separate from the storage-degeneracy
and residual-metric experiments in Group 5. The storage coefficients remain
`a0=c0=0.2`, `b0=0.1`; the residual metric remains `raw`.

## Before running commands

Commands below assume that your terminal is at the `THP-GFS-ENLM` repository
root. In a Linux/Bash terminal, enter your actual project path and activate
your existing JAX GPU environment first:

```bash
cd /your/actual/path/THP-GFS-ENLM
python -c "import jax; print(jax.__version__); print(jax.devices())"
```

Use a Python environment with NumPy, Matplotlib, SciPy, JAX and a GPU-compatible
JAXLIB installation for production training. Preserve the environment used for
the original runs when possible. `requirements.txt` at the root lists the
training dependencies; `requirements_analysis.txt` is enough for offline
analysis. The recorded environment versions remain in each frozen run's metadata.
See the root [README](../README.md#software-and-hardware-environment) for
the recorded package versions, GPU hardware and historical evidence coverage.
The full supplied environment inventory is
[`environment/pip_freeze_20261009.txt`](../environment/pip_freeze_20261009.txt).
Offline commands read existing records and do not train a model.

## Reproduce existing results

From the repository root:

```bash
python group4_small_transport/summarize_group4_small_transport.py --batch-tag v1
python export_component_tables.py
```

Inputs default to this group's `results/`; reports default to
`archive_summary/`. The summarizer exports final errors, fixed-budget
checkpoints, time-to-accuracy data, solver diagnostics, and publication tables.
The component exporter additionally writes `table7_component_errors.csv`,
`.md`, and `.tex`, including the relative-L2 and absolute spatial-H1 errors
for `u1,u2,xi,p,T`. All component values come directly from the FS50 histories.
The exporter also regenerates the Group 3 component table.

The original Group 4 study uses tables rather than a separate plotting
program; no extra plotting script is required. These are single-seed cases,
so the individual errors are not multi-seed estimates. The summarizer's
mean/std exports preserve its original format but have only one run per case.

## Run new experiments

Preview the plan without training, then launch on physical GPU 0.
Use a fresh batch tag for every new batch:

```bash
THP_DRY_RUN=1 bash group4_small_transport/run_group4_small_transport.sh
GPU_ID=0 BATCH_TAG=new_group4 bash group4_small_transport/run_group4_small_transport.sh
```

The launcher calls the canonical root `thp-four-field-core.py`. It retains
50 sweeps, 20 inner updates per block, fixed ridge `1e-7`, two refinement
steps, pointwise assembly, the original collocation grids, and disabled
early stopping. `CASE_MATRIX` is a semicolon-separated list of `nu,K,Theta`
triplets; `SEEDS` defaults to `12345`.

New runs go to `runs/` and their reports to `reports/<batch-tag>/`.
The launcher rejects reuse of a run directory and training inside frozen
`results/`. Reduced-budget `THP_FAST_DEBUG=1` is a code-path check and skips
production reports.

### Run from this group's directory

You may also start from `group4_small_transport/`; the launcher locates the shared
root core relative to its own path:

```bash
GPU_ID=0 BATCH_TAG=new_group4 bash run_group4_small_transport.sh
```

`GPU_ID` selects the physical GPU. Set it explicitly; do not rely on different
launchers' historical defaults. `PYTHON_BIN` can select another interpreter.
`BATCH_TAG` is an output label; if omitted, a timestamp is generated.
`RESULT_ROOT` can override the new-training destination, for example when
running from the repository root:

```bash
GPU_ID=0 BATCH_TAG=new_group4_custom \
RESULT_ROOT="$(pwd)/group4_small_transport/runs" \
bash group4_small_transport/run_group4_small_transport.sh
```

New training defaults to `group4_small_transport/runs/`; its analysis defaults to
`group4_small_transport/reports/<batch-tag>/`. Frozen inputs remain in `group4_small_transport/results/`
and offline exports in `group4_small_transport/archive_summary/`. Do not set `RESULT_ROOT` to
any group's frozen `results/` tree. Existing scientific run directories are
refused rather than overwritten. On one GPU, run experimental groups serially.
