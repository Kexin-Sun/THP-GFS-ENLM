# Group 1: optimizer and Gram-assembly comparisons

Seven required methods reproduce Tables 1--3 under the same four-field GFS
protocol. The extra historical `full_eigh0` run is retained as a supplement.
There is no local core copy: `thp_group1_gfs_comparison.py` dynamically loads
the unique root `../thp-four-field-core.py` before importing JAX.

| Method key | Assembly | Solver | Damping | Paper role |
| --- | --- | --- | ---: | --- |
| adam | None | Adam | 0 | Tables 1--2 and Figures 2--3 |
| adam_lbfgs | None | Adam followed by L-BFGS | 0 | Tables 1--2 and Figures 2--3 |
| engd_fjls | Full Jacobian | Least squares | 0 | Tables 1--3 and Figures 2--3 |
| js_lstsq0 | Streamed | Least squares | 0 | Table 3 assembly/solver control |
| js_eigh0 | Streamed | EIGH with refinement | 0 | Tables 1--3 and Figures 2--3 |
| js_eigh1e7 | Streamed | EIGH with refinement | 1e-7 | Tables 1--3 and Figures 2--3 |
| full_eigh1e7 | Full Jacobian | EIGH with refinement | 1e-7 | Table 3 damped assembly control |

`plot_group1_paper_figures.py` is retained and intentionally plots the five
representative methods in the manuscript. The analyzer plots all seven.
Positive damping is labeled ENLM; zero damping is labeled ENGD.
Original metadata terminology is retained without modification.

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

## Reproduce existing tables and figures

From the repository root:

```bash
python group1_optimizer/analyze_group1_gfs.py
python group1_optimizer/plot_group1_paper_figures.py
```

Both read `results/` by default. The archived prefix is
`group1_gfs_distinct_seed12345_20260824_022014_`.
Analysis writes `archive_summary/`; paper figures write
`archive_summary/paper_figures/`. Explicit `--root`, `--prefix`, and `--output`
are supported. Report output inside any frozen result directory is refused.

To include the optional historical eighth method in the full comparison:

```bash
python group1_optimizer/analyze_group1_gfs.py --include-optional \
    --output group1_optimizer/archive_summary/with_optional
```

The analyzer exports full-precision CSV tables and explicit four-row Table 3
CSV/Markdown/LaTeX files. The original `full_eigh1e7` result is already complete;
it does not need to be trained to recover the fourth Table 3 comparison.

## Preserved protocol

- Seed 12345; distinct manufactured solution, raw p/T residuals.
- `E=1`, `nu=0.3`, `alpha=beta=0.1`, `a0=c0=0.2`, `b0=0.1`, `K=Theta=0.1`.
- Networks `[3,32,32,3]` and `[3,32,32,2]`; FP64.
- PDE grid 16x16x16; initial grid 22x22; boundary grid 24x17 per side;
  evaluation grid 40x40x40. Residual weights are one.
- 100 complete GFS sweeps, p/T first and u/xi second; no early stopping.
- ENGD/ENLM: 20 updates per block; line-search grid 1 through 2^-30;
  two EIGH refinement steps and relative cutoff zero.
- Adam: 1000 updates per block, learning rate 1e-3.
- Adam+L-BFGS: switch after FS30; maxiter=200, maxfun=10000 per block,
  ftol=1e-15, gtol=1e-10.
- Full and streamed point blocks are 512; rank-safe ridge is disabled.

The full path forms each complete physical-block Jacobian from point batches
before forming J.T@J. The streamed path accumulates its Gram matrix without
materializing that complete Jacobian. The damped assembly comparison uses
the same EIGH solver and damping in both paths, including warmup.

TTA requires three consecutive completed sweeps meeting the threshold and
reports the first sweep in the confirmed streak. Effective time excludes JIT
warmup and evaluation. Spatial H1 columns are absolute errors.

## Run new experiments

Preview the plan without training, then launch on physical GPU 0.
Use a fresh batch tag for every new batch:

```bash
THP_DRY_RUN=1 bash group1_optimizer/run_group1_gfs.sh
GPU_ID=0 BATCH_TAG=new_group1 bash group1_optimizer/run_group1_gfs.sh
```

Default new runs go to `runs/`, and their reports to `reports/<batch-tag>/`.
`THP_GROUP1_METHODS` selects a comma-separated method subset. The historical
`THP_GROUP1_RESULT_ROOT`, `THP_GROUP1_BATCH_TAG`, and positional GPU argument
are also supported. Shared-core paths are resolved automatically.
`THP_FAST_DEBUG=1` is a reduced-budget execution check and skips production
reports. Existing runs and frozen trees cannot be overwritten.

### Run from this group's directory

You may also start from `group1_optimizer/`; the launcher locates the shared
root core relative to its own path:

```bash
GPU_ID=0 BATCH_TAG=new_group1 bash run_group1_gfs.sh
```

`GPU_ID` selects the physical GPU. Set it explicitly; do not rely on different
launchers' historical defaults. `PYTHON_BIN` can select another interpreter.
`BATCH_TAG` is an output label; if omitted, a timestamp is generated.
`RESULT_ROOT` can override the new-training destination, for example when
running from the repository root:

```bash
GPU_ID=0 BATCH_TAG=new_group1_custom \
RESULT_ROOT="$(pwd)/group1_optimizer/runs" \
bash group1_optimizer/run_group1_gfs.sh
```

New training defaults to `group1_optimizer/runs/`; its analysis defaults to
`group1_optimizer/reports/<batch-tag>/`. Frozen inputs remain in `group1_optimizer/results/`
and offline exports in `group1_optimizer/archive_summary/`. Do not set `RESULT_ROOT` to
any group's frozen `results/` tree. Existing scientific run directories are
refused rather than overwritten. On one GPU, run experimental groups serially.

## JSON summary conventions

`archive_summary/group1_summary.json` is strict JSON. Unavailable or undefined
numeric report values are `null`, never `NaN` or `Infinity`. In `js_speedup`,
a ratio is undefined when its denominator is zero. Streamed residual evaluation
and J.T@J/J.T@r products are fused into the Gram kernel, so their separate
recorded stage times are zero. This does not mean that the work has zero cost
or that the speedup is infinite. Compare total assembly or complete effective
training time for these implementations. All finite values remain unchanged.
