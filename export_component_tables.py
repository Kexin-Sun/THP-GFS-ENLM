#!/usr/bin/env python3
"""Export the FS50 component errors for Tables 5 and 7 without training.

The archived histories already contain relative L2 and absolute spatial H1
errors. Read these values directly; do not reconstruct them from model files
or normalize H1 errors. The manifest fixes the set of original experiments.
"""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FIELDS = ("u1", "u2", "xi", "p", "T")


def export_group(group, table, run_names):
    """Write independent report files while leaving frozen runs untouched."""
    directory = ROOT / {3: "group3_incompressible", 4: "group4_small_transport"}[group]
    output = directory / "archive_summary"
    output.mkdir(exist_ok=True)
    runs = []
    for name in run_names:
        run = directory / "results" / name
        metadata = json.loads((run / "metadata.json").read_text())
        history = json.loads((run / "training_history.json").read_text())
        checkpoints = [row for row in history if row.get("fs_iteration") == 50]
        if len(checkpoints) != 1:
            raise ValueError(f"Expected exactly one FS50 row in {run}")
        physics = metadata["physical_parameters"]
        form = "3F-GFS" if name.endswith("gfs_three_field") else "4F-GFS"
        runs.append((physics, form, metadata["seed"], name, checkpoints[0]))

    if group == 3:
        runs.sort(key=lambda item: (item[0]["nu"], item[1]))
    else:
        # Preserve the five-case order used in the manuscript.
        order = [(0.3, 0.1, 0.1), (0.3, 0.1, 1e-6),
                 (0.3, 1e-6, 0.1), (0.3, 1e-6, 1e-6),
                 (0.4999999, 1e-6, 1e-6)]
        runs.sort(key=lambda item: order.index(tuple(item[0][k] for k in ("nu", "K", "Theta"))))

    columns = ["run_name", "seed", "nu", "K", "Theta", "formulation",
               "fs_iteration", "field", "relative_L2", "spatial_H1"]
    with (output / f"table{table}_component_errors.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for physics, form, seed, name, row in runs:
            for field in FIELDS:
                if field == "xi" and form == "3F-GFS":
                    continue
                writer.writerow(dict(run_name=name, seed=seed, nu=physics["nu"],
                                     K=physics["K"], Theta=physics["Theta"],
                                     formulation=form, fs_iteration=50, field=field,
                                     relative_L2=row[f"{field}_rel_l2"],
                                     spatial_H1=row[f"{field}_h1"]))

    headings = ["nu", "K", "Theta", "Formulation", "Metric", *FIELDS]
    markdown = [f"# Table {table}: FS50 component errors", "",
                "Relative L2 and absolute spatial H1 errors are copied directly "
                "from the archived FS50 histories. Each case uses seed 12345. "
                "A dash means that xi is absent from the three-field formulation.", "",
                "| " + " | ".join(headings) + " |",
                "| " + " | ".join(["---"] * len(headings)) + " |"]
    latex = [r"\begin{tabular}{lllllrrrrr}", r"\hline",
             " & ".join(headings) + r" \\", r"\hline"]
    for physics, form, seed, name, row in runs:
        for label, suffix in [("Relative L2", "rel_l2"), ("Spatial H1", "h1")]:
            values = [f"{physics[k]:.7g}" for k in ("nu", "K", "Theta")]
            values += [form, label]
            values += ["--" if field == "xi" and form == "3F-GFS"
                       else f"{row[f'{field}_{suffix}']:.3e}" for field in FIELDS]
            markdown.append("| " + " | ".join(values) + " |")
            latex.append(" & ".join(values) + r" \\")
    markdown.append("")
    latex.extend([r"\hline", r"\end{tabular}", ""])
    (output / f"table{table}_component_errors.md").write_text("\n".join(markdown))
    (output / f"table{table}_component_errors.tex").write_text("\n".join(latex))
    print(f"Exported Table {table}: {len(runs)} frozen runs -> {output}")


def main():
    manifest = json.loads((ROOT / "FROZEN_RUNS.json").read_text())
    for group, table in [(3, 5), (4, 7)]:
        export_group(group, table, manifest[str(group)])


if __name__ == "__main__":
    main()
