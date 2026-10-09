# THP-GFS-ENLM

Frozen experiments and reproducible analysis for four-field generalized
fixed-stress energy-natural Levenberg-Marquardt PINNs for thermo-poroelasticity.
The archive contains all five experimental groups and one canonical
`thp-four-field-core.py` at the repository root, based on the Group 5 version.
Existing results do not require training again.

| Directory | Experiment | Complete historical runs | Runs used by the paper |
| --- | --- | ---: | ---: |
| `group1_optimizer/` | Optimizers, damping, and Gram assembly | 8 | 7 |
| `group2_formulation/` | Coupled/split architectures over five seeds | 20 | 20 |
| `group3_incompressible/` | Three-field/four-field near-incompressible comparison | 8 | 8 |
| `group4_small_transport/` | Five small-transport configurations | 5 | 5 |
| `group5_storage_degeneracy/` | Storage degeneracy and modal residual preconditioning | 22 | 12 |

There are 63 complete historical runs and 1004 original per-run files.
Seven required Group 1 methods include `full_eigh1e7`; the extra `full_eigh0`
run is an optional diagnostic. Group 5 retains all completed supplementary
and exploratory runs while exporting only the selected cases into the main
paper tables. Original filenames, metadata, checkpoints, and logs are unchanged.

## Reproduce every existing figure and table

From the repository root:

```bash
python -m pip install -r requirements_analysis.txt
bash regenerate_reports.sh
```

This workflow reads `results/` and writes each group's `archive_summary/`.
It does not import JAX or a training program. SHA-256 verification runs before
and after report generation. The package already includes regenerated reports.
The command works from another working directory when given the script's path.

| Paper material | Export |
| --- | --- |
| Tables 1--3, Figures 2--3 | `group1_optimizer/archive_summary/`, with five representative figures under `paper_figures/` |
| Table 4, Figure 4 | `group2_formulation/archive_summary/` |
| Table 5, Figure 5 | `group3_incompressible/archive_summary/` |
| Table 7 | `group4_small_transport/archive_summary/table7_component_errors.*` |
| Table 8 | `group5_storage_degeneracy/archive_summary/table8_parameter_configurations.*` |
| Tables 9--11 | `group5_storage_degeneracy/archive_summary/table9_component_errors.*`, `table10_component_errors.*`, and `table11_component_errors.*` |
| S7 modal-error and residual discussion | `group5_storage_degeneracy/archive_summary/table10_s7_modal_diagnostics.*` |

CSV exports retain full numeric precision. Markdown/LaTeX exports round for
presentation. L2 component errors are relative; spatial H1 errors are absolute.
The common maximum relative L2 metric excludes xi. Three-field experiments
have no independent xi component.

Group 2 final statistics use arithmetic means and one sample standard deviation
over five seeds. Its convergence curves use geometric centers and one sample
standard deviation in log10 error space. Groups 3--5 use seed 12345 for paper
tables and are not presented as multi-seed uncertainty studies.

## One shared four-field implementation

All four-field training paths use the root `thp-four-field-core.py`.
Group 1 and the coupled baseline load the hyphenated file using
`importlib.util`; there is no import-compatible duplicate core.
Group 3 reuses `group2_formulation/thp-three-field.py` for its three-field
baseline. Every launcher resolves these dependencies relative to its location.

The Group 5 physics, residuals, exact solution, derivatives, neural networks,
Gram assembly, solver, and line search remain unchanged. Integration changes
paths, output protection, paper-label bookkeeping, and configuration validation.
The comparable-transport Group 5 guard now accepts the paper FS50 budget as
well as the historical FS100 budget. Raw S7 paper runs report modal solution
errors using the existing evaluator; this is an evaluation diagnostic.

The environment-variable spellings containing `ENGD` are historical interfaces.
Positive fixed damping is described as ENLM in the paper and documentation;
zero damping remains ENGD. Group 5 generalized residual metrics and rank-safe
ridge are historical exploratory options, excluded from the default paper matrix.

## Group 5 alignment with the manuscript

| Historical identifier in frozen files | Paper identifier | Transport pair (K, Theta) |
| --- | --- | --- |
| S10 | S8 | (1e-6, 2e-6) |
| S11 | S9 | (2e-6, 1e-6) |
| S12 | S10 | (1e-6, 1e-8) |
| S13 | S11 | (1e-8, 1e-6) |

S1--S7 keep their identifiers. `experiment_matrix.json` records the exact
paper-to-source mapping and all historical configurations. Table 10 uses the
original S7 raw FS50 run and the FS50 checkpoint of the unweighted strict-modal
run. Weighted and generalized-physical runs do not enter the main paper tables.
All Group 5 main tables use FS50, even when the original trajectory continued
to FS100. FS100 exports are supplementary and retain their actual budgets.

S2 and S5 have positive-definite storage matrices. Only S4 is rank one;
S6--S11 have zero physical storage. The old singular-storage descriptions of
S2 and S5 have been corrected without changing their parameters.

Some manually rounded entries in the supplied manuscript's Tables 5 and 9 differ
from direct history values. The archive preserves the source values and records
examples in `VALIDATION.md`; no record is changed to force a printed match.

The supplied manuscript prints Table 9, S4, pressure spatial H1 as 1.17e-4. The frozen history is 0.00011788844874673654, which rounds to 1.18e-4 at three significant figures. The exports retain the history value; this is a documented manuscript discrepancy.

## Software and hardware environment

The following versions were supplied from the author's existing `jax`
environment on 2026-10-09 using `python --version` and `python -m pip freeze`.
They document the current environment snapshot rather than a verified lockfile
for every historical run.

| Component | Recorded version |
| --- | --- |
| Python | 3.10.10 |
| JAX | 0.4.30 |
| JAXLIB | 0.4.30 |
| jax-cuda12-plugin | 0.4.30 |
| jax-cuda12-pjrt | 0.4.30 |
| NumPy | 2.2.6 |
| SciPy | 1.13.1 |
| Matplotlib | 3.9.4 |
| ml_dtypes | 0.5.3 |
| nvidia-cuda-runtime-cu12 (installed package) | 12.9.79 |
| nvidia-cudnn-cu12 (installed package) | 9.13.0.50 |

The full supplied package list is preserved in
`environment/pip_freeze_20261009.txt`. It includes packages beyond this project's direct dependencies
and is an environment inventory, not the project's minimal dependency
list or a tested clean-environment installation recipe. In particular,
TensorFlow, PyTorch-related packages and Jupyter are not required by this
project's training or offline-report entry points.

The frozen metadata independently confirms Python 3.10.10, JAX 0.4.30,
the GPU backend and FP64 (`jax_enable_x64=true`) for 45 runs: the 10 Group 2
GFS runs and all runs in Groups 3--5. Group 1's eight runs and Group 2's ten
coupled runs do not include a runtime-version section. Their exact historical
software versions cannot be confirmed from the frozen metadata alone.
All 63 runs record an NVIDIA GeForce RTX 3090 GPU.

Installed CUDA package versions do not establish the system CUDA toolkit,
NVIDIA driver, or the libraries actually loaded by JAX. Both CUDA 11 and
CUDA 12 packages occur in the supplied inventory; this alone does not establish
which packages are used by this project. The original driver, system toolkit
and actual loaded-library versions are not confirmed by this snapshot.

`requirements_analysis.txt` lists NumPy and Matplotlib for offline reports;
this workflow does not require JAX or a GPU. `requirements.txt` lists the
project's direct training dependencies. Both remain unpinned installation
guidance. No clean installation or new production training was performed to
validate the supplied environment snapshot. Matching recorded package versions
alone does not guarantee identical floating-point results or training times.

## Frozen results and optional new training

- `results/`: original input data, never a training/report destination.
- `archive_summary/`: reproducible figures and tables from frozen data.
- `runs/`: default destination for new experiments.
- `reports/<batch-tag>/`: reports for new experiments.

Validate original files without training:

```bash
python verify_frozen_results.py
```

Inspect the optional training matrices without importing JAX:

```bash
THP_DRY_RUN=1 bash group1_optimizer/run_group1_gfs.sh
THP_DRY_RUN=1 bash group2_formulation/run_group2_architecture_comparison.sh
THP_DRY_RUN=1 bash group3_incompressible/run_group3_incompressible.sh
THP_DRY_RUN=1 bash group4_small_transport/run_group4_small_transport.sh
THP_DRY_RUN=1 bash group5_storage_degeneracy/run_group5_final_matrix.sh
```

Training requires `requirements.txt` and JAX/JAXLIB matched to the local
CPU/GPU environment. See the software and hardware section above for the
supplied versions and the coverage of historical runtime metadata. Unpinned
dependencies are installation guidance, not an exact historical environment lock.

```bash
GPU_ID=0 BATCH_TAG=new_group1 bash group1_optimizer/run_group1_gfs.sh
GPU_ID=0 BATCH_TAG=new_group5 bash group5_storage_degeneracy/run_group5_final_matrix.sh
```

Use a fresh `BATCH_TAG`. `RESULT_ROOT` and `PYTHON_BIN` can be overridden.
Launchers reject all frozen result trees and existing scientific run directories.
The training modules enforce the same protection. Group 5 `RESUME=1` only
skips complete runs; it never restarts or overwrites an incomplete directory.
Fresh training can differ across hardware or dependency versions.

## Archive contents and verification

Necessary comments, docstrings, and documentation are English. Historical
logs and metadata are byte-identical, including any original terminology.
Python caches, interrupted runs, duplicate cores, obsolete aggregate reports,
and external launcher logs/PID files are excluded. Original per-run files
are retained in full.

`FROZEN_RESULTS_SHA256.json`, `FROZEN_RUNS.json`, and
`ARCHIVE_PROVENANCE.json` document integrity and source lineage.
See `VALIDATION.md` for completed checks and their scope.

## Upload to GitHub

See [GITHUB_UPLOAD.md](GITHUB_UPLOAD.md) for initial commit, authentication,
repository creation, push and independent-clone verification commands.
The included `.gitignore` excludes new runs/reports and caches while keeping
all frozen results and archive summaries. `.gitattributes` disables newline
conversion inside frozen result trees to preserve their SHA-256 hashes.
