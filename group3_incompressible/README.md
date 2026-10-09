# Group 3: near-incompressible limit

Eight complete runs compare three-field GFS and four-field GFS at
`nu=0.49,0.4999,0.49999,0.4999999`, all with seed `12345`.
The other physical parameters and the fixed grids follow Group 2,
including `K=Theta=0.1`. Every archived run completes the common FS50 budget.

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
python group3_incompressible/plot_group3_incompressible.py --batch-tag v1
python export_component_tables.py
```

The plotting script reads this group's `results/` and writes
`archive_summary/`. It exports the final-error, fixed-budget, time-to-accuracy,
Gram-diagnostic, and FS100-decision tables, plus the two comparison figures
as PNG/PDF. Its `group3_publication_main_table.csv` combines the relative-L2
errors and solver diagnostics; it does not contain the spatial-H1 errors.

`table5_component_errors.csv`, `.md`, and `.tex` include both relative-L2
and absolute spatial-H1 component errors copied directly from the archived
FS50 histories. An independent xi error is absent for the three-field model.
The export command also regenerates the Group 4 component table.

Gram condition numbers and effective rank fractions are medians over
FS41–FS50. Acceptance rates use FS1–FS50. Time to accuracy requires three
consecutive completed sweeps and is measured using the original effective
training times. `NR` means the threshold was not reached under that rule.
The condition numbers depend on the numerical-rank convention retained in
the original diagnostic script; do not interpret them as unregularized
full-rank matrix condition numbers without that convention.

The stored FS100-decision table is an analysis aid. No FS100 experiment is
included or required here. The shared comparison budget remains FS50.
The archive contains one seed per case, so no multi-seed uncertainty band
is claimed.

## Run new experiments

Preview the plan without training, then launch on physical GPU 0.
Use a fresh batch tag for every new batch:

```bash
THP_DRY_RUN=1 bash group3_incompressible/run_group3_incompressible.sh
GPU_ID=0 BATCH_TAG=new_group3 bash group3_incompressible/run_group3_incompressible.sh
```

`NUS` and `SEEDS` can override the default matrix. The launcher calls
`../group2_formulation/thp-three-field.py` and the root `thp-four-field-core.py`.
Keep these shared dependencies when moving this directory. It retains
raw residuals, fixed ridge `1e-7`, two refinement steps, 20 inner updates
per split block, 50 sweeps, and disabled early stopping.

New training goes to `runs/` and new reports to `reports/<batch-tag>/`.
Existing run directories are never merged. `THP_FAST_DEBUG=1` checks code
paths with a reduced budget and skips the FS50 report builder.

### Run from this group's directory

You may also start from `group3_incompressible/`; the launcher locates the shared
root core relative to its own path:

```bash
GPU_ID=0 BATCH_TAG=new_group3 bash run_group3_incompressible.sh
```

`GPU_ID` selects the physical GPU. Set it explicitly; do not rely on different
launchers' historical defaults. `PYTHON_BIN` can select another interpreter.
`BATCH_TAG` is an output label; if omitted, a timestamp is generated.
`RESULT_ROOT` can override the new-training destination, for example when
running from the repository root:

```bash
GPU_ID=0 BATCH_TAG=new_group3_custom \
RESULT_ROOT="$(pwd)/group3_incompressible/runs" \
bash group3_incompressible/run_group3_incompressible.sh
```

New training defaults to `group3_incompressible/runs/`; its analysis defaults to
`group3_incompressible/reports/<batch-tag>/`. Frozen inputs remain in `group3_incompressible/results/`
and offline exports in `group3_incompressible/archive_summary/`. Do not set `RESULT_ROOT` to
any group's frozen `results/` tree. Existing scientific run directories are
refused rather than overwritten. On one GPU, run experimental groups serially.
