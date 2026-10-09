# Archive validation

All checks described below were completed without rerunning the formal experiments.

## Frozen data and reports

- 63 complete runs: Group 1 = 8, Group 2 = 20, Group 3 = 8, Group 4 = 5, Group 5 = 22.
- All 1004 archived per-run files were compared directly with members of the supplied ZIP/TAR attachments using SHA-256. Every file is byte-identical.
- The complete `regenerate_reports.sh` workflow succeeded. Integrity verification passed before and after regeneration.
- Group 1 exports seven required methods, including full-Jacobian EIGH with damping 1e-7. Its paper figures use the five representative methods. The undamped full-Jacobian run remains supplementary.
- Group 2 reports all five seeds and distinguishes arithmetic statistics from convergence curves in log space. Groups 3 and 4 export the component-error tables.
- Group 5 validates the physical parameters, residual metric, modal scales, weights, damping and budgets before exporting Tables 8--11. Main-table records use FS50 and exclude weighted/generalized exploratory variants.
- Of 120 component-error cells audited against the displayed Group 5 Tables 9--11, 119 agree at printed precision. Tables 10 and 11 agree in every audited cell. The remaining cell is documented below.
- Regenerated Group 1 paper figures and Group 2/3 figures were visually inspected, including legends, axes, clipping and panel layout.

## Implementation and output protection

- Every Python source parses and every shell launcher passes `bash -n`.
- All five launchers pass dry-run checks from a different working directory, resolve the shared core and reject frozen output trees or existing run directories.
- All six analysis/plot entry points reject frozen result trees as report destinations.
- The original Group 5 core's 113 functions/classes were compared by AST after removing source locations, inert strings and logger-only statements. Numerical functions/classes are unchanged. Only `_comparison_run_identity`, `save_metadata` and `pt_residual_metric_metadata` change report bookkeeping.
- Group 1's 47 functions/classes match its previously corrected seven-method driver. Three-field (106) and coupled (33) functions/classes match their original implementations under the same comparison.
- All 12 default Group 5 FS50 configurations passed CPU configuration import and metadata serialization with zero optimizer updates. This includes the comparable-transport FS50 guard and raw S7 modal diagnostic metadata.
- Two CPU debug sweeps of Group 1 `full_eigh1e7` passed through the unique root core. Two CPU debug sweeps of Group 5 S7 raw passed including modal evaluation and serialization. These smoke outputs are outside this archive and do not replace any historical result.

These checks establish preserved data, report reproducibility and basic integration. They do not establish bitwise reproduction of complete training on another GPU or with different JAX versions. Historical runtime metadata is retained; installation requirements are not an exact environment lock.

## Differences from manually printed manuscript values

The supplied manuscript prints Table 9, S4, pressure spatial H1 as 1.17e-4. The frozen history is 0.00011788844874673654, which rounds to 1.18e-4 at three significant figures. The exports retain the history value; this is a documented manuscript discrepancy.

Some Table 5 entries also differ from direct history values at the manuscript's displayed precision. Examples:

| Case and field | Manuscript relative L2 | Direct history (approximately) |
| --- | ---: | ---: |
| nu=0.49, four-field pressure | 1.40e-5 | 1.360e-5 |
| nu=0.49, four-field u2 | 5.10e-5 | 5.124e-5 |
| nu=0.4999, four-field pressure | 1.30e-5 | 1.259e-5 |
| nu=0.4999999, three-field temperature | 1.30e-5 | 1.330e-5 |

The full-precision CSV exports and all original histories remain authoritative. No value is altered to force agreement with manually rounded manuscript text.

## Documentation and strict JSON update (2026-10-09)

All five group READMEs now contain explicit environment setup, dry-run and
production commands, instructions for both repository-root and group-directory
execution, output locations and batch-tag protection. The Group 1 analyzer
converts non-finite report values to JSON null and rejects non-standard numeric
tokens during serialization. Three undefined per-stage speedup entries formerly
written as NaN are null; their zero timing denominators and all finite metrics
are unchanged. No formal experiment was rerun for this update.

## Environment documentation update (2026-10-09)

The author's supplied current Python/pip inventory has been archived verbatim
as package lines, with Python 3.10.10 recorded in the root README. Direct
dependency and CUDA-package versions were checked against the supplied text.
All 45 frozen runtime sections independently confirm Python 3.10.10 and JAX
0.4.30 with GPU/FP64. All 63 runs record the RTX 3090. Documentation explicitly
distinguishes current inventory from exact historical software evidence; no
clean installation or formal experiment was run. All 1004 frozen-file hashes
still match.

## Final pre-upload audit (2026-10-09)

All five dry-run launchers passed from a different working directory. All
25 launcher/frozen-tree combinations were rejected, as were existing run
directories and frozen report destinations for all six analysis entry points.
The full report regeneration succeeded and verified 1004 hashes before and
after. All derived summary JSON files parsed as strict JSON.

Output paths in the four-field core and three-field program are now resolved
before directory protection, including custom run names containing `../`.
The actual module path/mkdir statements were extracted and exercised without
JAX imports: all five normalized frozen paths were rejected, a fresh directory
was accepted, and an existing directory and its sentinel were preserved.
Training numerical function/class bodies remain unchanged. No full GPU
experiment was rerun; the earlier CPU smoke/configuration tests remain the
training-integration evidence, while this final path change was checked
independently of JAX.

A temporary Git repository staged all frozen files, committed, and was cloned
with `core.autocrlf=true`. All 1004 staged blobs and cloned files were
byte-identical. New runs/reports were correctly ignored. The largest source
file is 7,077,993 bytes, below GitHub's ordinary Git file-size limit.
