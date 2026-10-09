# Group 5: storage degeneracy and modal residual preconditioning

This archive aligns the final Group 5 experiments with Tables 8--11 of the
supplied manuscript. It retains 22 complete historical runs. Twelve runs are
used by the paper; the other ten are supplementary or exploratory controls.
All main paper values are read at FS50, even when the original run continued
to FS100. No historical record is renamed, edited, or retrained.

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

## Case identities and physical storage

| Paper case | Historical case | a0 | b0 | c0 | K | Theta | Physical storage |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| S1 | S1 | 0.2 | 0.1 | 0.2 | 0.1 | 0.1 | Positive definite |
| S2 | S2 | 0.1 | 0.1 | 0.2 | 0.1 | 0.1 | Positive definite |
| S3 | S3 | 0.2 | 0 | 0.2 | 0.1 | 0.1 | Positive definite |
| S4 | S4 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | Rank one |
| S5 | S5 | 0.2 | 0.1 | 0.1 | 0.1 | 0.1 | Positive definite |
| S6 | S6 | 0 | 0 | 0 | 0.1 | 0.1 | Zero |
| S7 | S7 | 0 | 0 | 0 | 1e-6 | 1e-6 | Zero |
| S8 | S10 | 0 | 0 | 0 | 1e-6 | 2e-6 | Zero |
| S9 | S11 | 0 | 0 | 0 | 2e-6 | 1e-6 | Zero |
| S10 | S12 | 0 | 0 | 0 | 1e-6 | 1e-8 | Zero |
| S11 | S13 | 0 | 0 | 0 | 1e-8 | 1e-6 | Zero |

The physical storage matrix is `[[c0,-b0],[-b0,a0]]`.
S2 and S5 have determinant 0.01 and are not singular. This corrects the
descriptions in the old README without changing any experimental coefficient.
S4 has rank one; S6--S11 have zero storage. Storage degeneracy alone in S6
is distinct from joint storage/weak-transport degeneracy in S7--S11.

## Exact source selection for paper tables

- Table 9: S1--S6 raw, FS50.
- Table 10: the original S7 raw FS50 run and the FS50 checkpoint of
  `both_storage_unweighted`. The separate S7 raw FS100 run is supplemental.
- Table 11: the unweighted strict-modal FS50 checkpoints of historical
  S10--S13, displayed as paper S8--S11.
- S7 modal-error discussion: stored `q_parallel_rel_l2`, `q_perp_rel_l2`,
  `loss_pt`, and `loss_pt_coupled_pde` at FS50.

The full mapping, actual scales/weights, original budgets, and source directory
names are in `experiment_matrix.json`. Every key is unique; frozen results
are validated against its physical parameters, metric, scales, weights, and
ridge settings before reporting. Main tables use seed 12345.

The paper modal preconditioner is `modal_both_scaled`, with parallel scale
`sigma_parallel=0.034666666666666665` and perpendicular scale
`d_perp=(beta^2*K+alpha^2*Theta)/(alpha^2+beta^2)`.
For alpha=beta=0.1, d_perp is 1e-6 in S7, 1.5e-6 in S8/S9, and
5.05e-7 in S10/S11. The exact cross-diffusion term is retained. All external
loss weights are one, and the base Gram ridge remains fixed at 1e-7.

The weighted S7 run changes the p/T Dirichlet and initial weights to 16.
Generalized-physical runs use a different residual metric; historical S8/S9
physical controls also use mechanics rank-safe safety=3. These are not paper
modal configurations and never enter Tables 9--11.

## Offline reproduction

From the repository root:

```bash
python group5_storage_degeneracy/summarize_group5_storage_degeneracy.py
```

The script defaults to `results/` and `archive_summary/`. It reads stored
histories and never imports the core or JAX. The missing source summarizer
in the original TAR has been replaced with a complete readable Python script;
the old bytecode is not needed.

| Output | Contents |
| --- | --- |
| `table8_parameter_configurations.*` | Paper configurations and correct storage ranks |
| `table9_component_errors.*` | S1--S6 relative L2 and absolute spatial H1 |
| `table10_component_errors.*` | S7 raw/modal component errors at FS50 |
| `table10_s7_modal_diagnostics.*` | Modal solution errors and residual-objective values |
| `table11_component_errors.*` | Paper S8--S11 component errors at FS50 |
| `group5_all_runs_FS50.csv` | All 22 historical runs at FS50, including exploratory controls |
| `group5_FS100_supplement.csv` | Actual FS100 checkpoints only |
| `group5_solver_diagnostics_FS50.csv` | FS1--FS50 acceptance and FS41--FS50 median diagnostics |
| `group5_time_to_accuracy.csv` | Original TTA records, mapped to paper labels and actual budgets |
| `group5_run_map.csv` | Paper/historical labels, source directories, roles, and completed budgets |

CSV preserves full precision; Markdown/LaTeX files round the manuscript tables
to three significant figures. The manuscript y_parallel/y_perp variables are
the stored q_parallel/q_perp variables. Raw and modal `loss_pt` values use
different residual metrics and are not directly comparable. FS100 TTA records
are historical-budget supplements, not FS50-only claims.

## Run new experiments

Preview the plan without training, then launch on physical GPU 0.
Use a fresh batch tag for every new batch:

```bash
THP_DRY_RUN=1 bash group5_storage_degeneracy/run_group5_final_matrix.sh
GPU_ID=0 BATCH_TAG=new_group5 bash group5_storage_degeneracy/run_group5_final_matrix.sh
```

`ALL` selects twelve runs per seed: S1--S6 raw, S7 raw/modal, and S8--S11
modal. The default new training budget is FS50, aligned with the manuscript.
Use `BUDGET=historical` to retain each selected configuration's original
FS50/FS100 training budget. Main tables still evaluate FS50.

```bash
THP_DRY_RUN=1 BUDGET=historical RUN_ONLY=S7_MODAL,S8_MODAL,S9_MODAL \
    bash group5_storage_degeneracy/run_group5_final_matrix.sh
THP_DRY_RUN=1 RUN_ONLY=S7_RAW_FS100,S7_MODAL_WEIGHTED16,S10_PHYSICAL \
    bash group5_storage_degeneracy/run_group5_final_matrix.sh
```

`RUN_ONLY` uses paper-numbered keys. All 22 keys are listed in the JSON matrix.
The root core retains the historical internal identifiers for compatibility;
the launcher also supplies the paper identifier. New metadata reports the
paper case and retains `legacy_case_id` explicitly. Numerical kernels are
unchanged. The comparable-transport guard accepts FS50 or FS100. Raw S7
paper runs now include the existing modal error evaluation in their reports.

New runs go to `runs/`; new reports go to `reports/<batch-tag>/`. All planned
directories are checked before training starts. `RESUME=1` only skips complete
runs. It never merges or restarts an incomplete run. `THP_FAST_DEBUG=1` is an
execution check and skips production analysis. `DRY_RUN=1` remains an alias
for `THP_DRY_RUN=1`. Report output inside frozen/input results is refused.

## Manuscript rounding discrepancy

The supplied manuscript prints Table 9, S4, pressure spatial H1 as 1.17e-4. The frozen history is 0.00011788844874673654, which rounds to 1.18e-4 at three significant figures. The exports retain the history value; this is a documented manuscript discrepancy.

### Run from this group's directory

You may also start from `group5_storage_degeneracy/`; the launcher locates the shared
root core relative to its own path:

```bash
GPU_ID=0 BATCH_TAG=new_group5 bash run_group5_final_matrix.sh
```

`GPU_ID` selects the physical GPU. Set it explicitly; do not rely on different
launchers' historical defaults. `PYTHON_BIN` can select another interpreter.
`BATCH_TAG` is an output label; if omitted, a timestamp is generated.
`RESULT_ROOT` can override the new-training destination, for example when
running from the repository root:

```bash
GPU_ID=0 BATCH_TAG=new_group5_custom \
RESULT_ROOT="$(pwd)/group5_storage_degeneracy/runs" \
bash group5_storage_degeneracy/run_group5_final_matrix.sh
```

New training defaults to `group5_storage_degeneracy/runs/`; its analysis defaults to
`group5_storage_degeneracy/reports/<batch-tag>/`. Frozen inputs remain in `group5_storage_degeneracy/results/`
and offline exports in `group5_storage_degeneracy/archive_summary/`. Do not set `RESULT_ROOT` to
any group's frozen `results/` tree. Existing scientific run directories are
refused rather than overwritten. On one GPU, run experimental groups serially.
