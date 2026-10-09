# Group 2: architecture and formulation comparison

The archive contains four methods at seeds `12345,23456,34567,45678,56789`:

| Method key | Formulation and networks | Parameters |
| --- | --- | ---: |
| `coupled_single` | Monolithic four-field model, one `[3,32,32,5]` network | 1349 |
| `coupled_dual` | Monolithic four-field model, separate u/xi and p/T networks | 2533 |
| `gfs_three_field` | Three-field GFS, separate u and p/T networks | 2500 |
| `gfs_four_field` | Four-field GFS, separate u/xi and p/T networks | 2533 |

The shared physics is `E=1`, `nu=0.3`, `alpha=beta=0.1`,
`a0=c0=0.2`, `b0=0.1`, and `K=Theta=0.1`. All runs use the
`distinct` manufactured solution and the original fixed collocation grids,
unit residual-block weights, FP64 calculations, and raw p/T residual metric.

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

## Reproduce the archived figures and Table 4

From the repository root:

```bash
python group2_formulation/plot_group2_comparison.py \
  --batch-tag multiseed_v1 --seeds 12345,23456,34567,45678,56789
```

The default input is this directory's `results/`; the default output is
`archive_summary/`. The script can also accept an explicit input directory
as its first positional argument and an independent `--output-dir`.
It refuses an output directory inside the input results tree.

`group2_publication_tables.md` and the publication accuracy/efficiency CSV
files provide the Table 4 values. The other outputs retain per-seed values,
time-to-accuracy statistics, timing breakdowns, outer-convergence measures,
and six comparison figures in PNG and PDF formats.

The equivalent-update axis is `20 * outer iteration`, reaching 1000 at
iteration 50. A coupled round has 20 full-model updates; a GFS sweep has
20 p/T updates and 20 mechanics updates. Thus the equivalent-update axis
does not claim identical total block-update counts or identical work per update.

The convergence figures use a geometric mean: the center is
`10**mean(log10(error))`, and the band is plus/minus one sample standard
deviation in log10 space. The final-value tables use arithmetic means and
sample standard deviations over five seeds. These are different statistics
by design. Timing values exclude JIT warmup and evaluation as recorded by
the original experiment metadata. Peak GPU memory is reported in GiB.

## Run new experiments

Preview the plan without training, then launch on physical GPU 0.
Use a fresh batch tag for every new batch:

```bash
THP_DRY_RUN=1 bash group2_formulation/run_group2_architecture_comparison.sh
GPU_ID=0 BATCH_TAG=new_group2 bash group2_formulation/run_group2_architecture_comparison.sh
```

The launcher defaults to all five archived seeds, 50 rounds/sweeps,
fixed ridge `1e-7`, two refinement steps, 31 positive line-search levels,
and streamed pointwise Gram assembly. It resolves the coupled and three-field
programs here and the four-field core at the repository root. The point-block
sizes are 512 for the split networks and 128 for the coupled model.

New experiments go to `runs/`; their reports go to `reports/<batch-tag>/`.
The frozen `results/` tree is input data. A fresh batch tag is required when
a requested run directory already exists. `THP_FAST_DEBUG=1` is only a short
code-path check and does not reproduce the manuscript experiment.

### Run from this group's directory

You may also start from `group2_formulation/`; the launcher locates the shared
root core relative to its own path:

```bash
GPU_ID=0 BATCH_TAG=new_group2 bash run_group2_architecture_comparison.sh
```

`GPU_ID` selects the physical GPU. Set it explicitly; do not rely on different
launchers' historical defaults. `PYTHON_BIN` can select another interpreter.
`BATCH_TAG` is an output label; if omitted, a timestamp is generated.
`RESULT_ROOT` can override the new-training destination, for example when
running from the repository root:

```bash
GPU_ID=0 BATCH_TAG=new_group2_custom \
RESULT_ROOT="$(pwd)/group2_formulation/runs" \
bash group2_formulation/run_group2_architecture_comparison.sh
```

New training defaults to `group2_formulation/runs/`; its analysis defaults to
`group2_formulation/reports/<batch-tag>/`. Frozen inputs remain in `group2_formulation/results/`
and offline exports in `group2_formulation/archive_summary/`. Do not set `RESULT_ROOT` to
any group's frozen `results/` tree. Existing scientific run directories are
refused rather than overwritten. On one GPU, run experimental groups serially.
