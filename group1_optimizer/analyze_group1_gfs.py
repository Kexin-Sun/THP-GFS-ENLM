"""Aggregate and plot the seven required first-group 4F-GFS comparison runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXPECTED_METHODS = (
    "adam",
    "adam_lbfgs",
    "engd_fjls",
    "js_lstsq0",
    "js_eigh0",
    "js_eigh1e7",
    "full_eigh1e7",
)

OPTIONAL_METHODS = ("full_eigh0",)
PAPER_METHODS = ("adam", "adam_lbfgs", "engd_fjls", "js_eigh0", "js_eigh1e7")


def json_safe(value):
    """Represent unavailable/non-finite report values as standard JSON null."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value

METHOD_ORDER = {method: index for index, method in enumerate(EXPECTED_METHODS + OPTIONAL_METHODS)}
METHOD_LABELS = {
    "adam": "Adam",
    "adam_lbfgs": "Adam+L-BFGS",
    "engd_fjls": "ENGD (FJ+LS, $\\gamma=0$)",
    "js_lstsq0": "ENGD (S+LS, $\\gamma=0$)",
    "js_eigh0": "ENGD (S+EIGH, $\\gamma=0$)",
    "js_eigh1e7": "ENLM (S+EIGH, $\\gamma=10^{-7}$)",
    "full_eigh1e7": "ENLM (FJ+EIGH, $\\gamma=10^{-7}$)",
    "full_eigh0": "ENGD (FJ+EIGH, $\\gamma=0$)",
}
COLORS = {
    "adam": "#1f77b4",
    "adam_lbfgs": "#2ca02c",
    "engd_fjls": "#8c564b",
    "js_lstsq0": "#7f7f7f",
    "js_eigh0": "#ff7f0e",
    "js_eigh1e7": "#17becf",
    "full_eigh1e7": "#9467bd",
    "full_eigh0": "#e377c2",
}
MARKERS = {
    "adam": "o",
    "adam_lbfgs": "s",
    "engd_fjls": "D",
    "js_lstsq0": "^",
    "js_eigh0": "v",
    "js_eigh1e7": "P",
    "full_eigh1e7": "X",
    "full_eigh0": "*",
}

ACCURACY_COLUMNS = (
    "u1_rel_l2",
    "u2_rel_l2",
    "xi_rel_l2",
    "p_rel_l2",
    "T_rel_l2",
    "E_u_max",
    "E_pT_max",
    "E_common_max",
    "E_4F_max",
    "u1_h1",
    "u2_h1",
    "xi_h1",
    "p_h1",
    "T_h1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--prefix", default="group1_gfs_distinct_seed12345_20260824_022014_")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "archive_summary")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Permit a method subset or smoke-test histories.")
    parser.add_argument("--include-optional", action="store_true",
                        help="Also analyze the historical full_eigh0 diagnostic.")
    return parser.parse_args()


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def discover_runs(
    root: Path, prefix: str, allow_incomplete: bool, include_optional: bool = False
) -> Dict[str, Dict[str, object]]:
    runs: Dict[str, Dict[str, object]] = {}
    for directory in sorted(root.glob(f"{prefix}*")):
        metadata_path = directory / "metadata.json"
        history_path = directory / "training_history.json"
        if not directory.is_dir() or not metadata_path.exists():
            continue
        metadata = read_json(metadata_path)
        if metadata.get("protocol") != "THP-Group1-4F-GFS-optimizer-JS-v3":
            continue
        method = str(metadata["method_key"])
        if method not in EXPECTED_METHODS:
            if not include_optional or method not in OPTIONAL_METHODS:
                continue
        if directory.name != f"{prefix}{method}":
            continue
        if method in runs:
            raise RuntimeError(
                f"Duplicate method {method!r} under prefix {prefix!r}."
            )
        if not history_path.exists():
            raise FileNotFoundError(history_path)
        history = read_json(history_path)
        fs = [int(row["fs_iteration"]) for row in history]
        if not allow_incomplete and fs != list(range(101)):
            raise RuntimeError(f"{directory}: expected complete FS0--FS100 history.")
        if not history:
            raise RuntimeError(f"Empty history in {directory}")
        runs[method] = {
            "directory": directory,
            "metadata": metadata,
            "history": history,
            "profile": read_json(directory / "optimizer_profile.json"),
            "tta": read_json(directory / "time_to_accuracy.json"),
        }

    missing = [method for method in EXPECTED_METHODS if method not in runs]
    if missing and not allow_incomplete:
        raise RuntimeError(
            "Incomplete Group-1 batch; missing methods: " + ", ".join(missing)
        )
    if not runs:
        raise RuntimeError(
            f"No v3 Group-1 runs found in {root} with prefix {prefix!r}."
        )
    return dict(sorted(runs.items(), key=lambda item: METHOD_ORDER[item[0]]))


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def assert_common_protocol(runs: Dict[str, Dict[str, object]]) -> None:
    signatures: Dict[str, Tuple[object, ...]] = {}
    for method, run in runs.items():
        metadata = run["metadata"]
        signatures[method] = (
            metadata["seed"],
            metadata["exact_case"],
            json.dumps(metadata["physical_parameters"], sort_keys=True),
            json.dumps(metadata["network_sizes"], sort_keys=True),
            json.dumps(metadata["sampling"], sort_keys=True),
            json.dumps(metadata["loss_weighting"], sort_keys=True),
            metadata["budget"]["fs_outer_iterations"],
        )
    reference_method = next(iter(signatures))
    reference = signatures[reference_method]
    mismatched = [method for method, value in signatures.items() if value != reference]
    if mismatched:
        raise RuntimeError(
            "Protocol mismatch relative to "
            f"{reference_method}: {', '.join(mismatched)}"
        )


def history_at_fs(run: Dict[str, object], fs_iteration: int):
    for row in run["history"]:
        if int(row["fs_iteration"]) == fs_iteration:
            return row
    return None


def build_accuracy_table(
    runs: Dict[str, Dict[str, object]], fs_iteration: int
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for method, run in runs.items():
        source = history_at_fs(run, fs_iteration)
        if source is None:
            continue
        row: Dict[str, object] = {
            "method": METHOD_LABELS[method],
            "method_key": method,
            "fs_iteration": fs_iteration,
            "effective_training_seconds": source["effective_training_seconds"],
            "context_seconds": source["context_seconds_cumulative"],
            "optimizer_seconds": source["optimizer_seconds_cumulative"],
            "convergence_seconds": source["convergence_seconds_cumulative"],
        }
        row.update({key: source[key] for key in ACCURACY_COLUMNS})
        rows.append(row)
    return rows


def build_fs50_to_fs100_table(
    runs: Dict[str, Dict[str, object]]
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    metrics = (
        "u1_rel_l2",
        "u2_rel_l2",
        "xi_rel_l2",
        "p_rel_l2",
        "T_rel_l2",
        "E_common_max",
        "E_4F_max",
    )
    for method, run in runs.items():
        fs50 = history_at_fs(run, 50)
        fs100 = history_at_fs(run, 100)
        if fs50 is None or fs100 is None:
            continue
        row: Dict[str, object] = {
            "method": METHOD_LABELS[method],
            "method_key": method,
            "additional_effective_seconds": (
                float(fs100["effective_training_seconds"])
                - float(fs50["effective_training_seconds"])
            ),
            "additional_time_fraction_of_fs100": (
                (
                    float(fs100["effective_training_seconds"])
                    - float(fs50["effective_training_seconds"])
                )
                / max(float(fs100["effective_training_seconds"]), 1.0e-300)
            ),
        }
        for metric in metrics:
            value50 = float(fs50[metric])
            value100 = float(fs100[metric])
            row[f"{metric}_fs50"] = value50
            row[f"{metric}_fs100"] = value100
            row[f"{metric}_improvement_factor"] = value50 / max(
                value100, 1.0e-300
            )
            row[f"{metric}_relative_reduction"] = 1.0 - value100 / max(
                value50, 1.0e-300
            )
        rows.append(row)
    return rows


def sum_profile(profile: Iterable[Dict[str, object]], key: str) -> float:
    total = 0.0
    for row in profile:
        value = row.get(key, 0.0)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            total += float(value)
    return total


def build_stage_table(
    runs: Dict[str, Dict[str, object]]
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    keys = (
        "context_seconds",
        "residual_seconds",
        "jacobian_seconds",
        "gram_matmul_seconds",
        "gradient_matmul_seconds",
        "symmetrize_seconds",
        "assembly_seconds",
        "solve_seconds",
        "line_search_seconds",
        "wall_seconds",
    )
    for method, run in runs.items():
        metadata = run["metadata"]
        row: Dict[str, object] = {
            "method": METHOD_LABELS[method],
            "method_key": method,
            "backend": metadata["optimizer"]["jacobian_backend"],
            "solver": metadata["optimizer"]["linear_solver"],
            "ridge": metadata["optimizer"]["gram_ridge"],
            "effective_training_seconds": metadata["timing"][
                "effective_training_seconds"
            ],
            "optimizer_seconds_ledger": metadata["timing"]["optimizer_seconds"],
            "convergence_seconds": metadata["timing"]["convergence_seconds"],
            "peak_rss_bytes": metadata["memory"].get(
                "process_peak_rss_bytes", math.nan
            ),
            "device_peak_bytes": metadata["memory"].get(
                "device_peak_bytes_in_use", math.nan
            ),
        }
        row.update({key: sum_profile(run["profile"], key) for key in keys})
        rows.append(row)
    return rows


def build_js_speedup_table(
    stage_rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    by_method = {row["method_key"]: row for row in stage_rows}
    if "engd_fjls" not in by_method or "js_lstsq0" not in by_method:
        return []
    full = by_method["engd_fjls"]
    streamed = by_method["js_lstsq0"]
    stages = (
        ("GFS frozen-context construction", "context_seconds"),
        ("residual evaluation (full-J only)", "residual_seconds"),
        ("full Jacobian / fused streamed Gram kernel", "jacobian_seconds"),
        ("explicit J.T@J (full-J only)", "gram_matmul_seconds"),
        ("explicit J.T@r (full-J only)", "gradient_matmul_seconds"),
        ("total Gram/gradient assembly", "assembly_seconds"),
        ("same undamped lstsq solve", "solve_seconds"),
        ("same 31-level line search", "line_search_seconds"),
        ("all ENGD update wall time", "wall_seconds"),
        ("complete effective training", "effective_training_seconds"),
    )
    rows: List[Dict[str, object]] = []
    for stage, key in stages:
        full_seconds = float(full.get(key, 0.0))
        streamed_seconds = float(streamed.get(key, 0.0))
        rows.append(
            {
                "comparison": "ENGD full-J+lstsq vs JS-ENGD streamed+lstsq",
                "stage": stage,
                "field": key,
                "full_seconds": full_seconds,
                "streamed_seconds": streamed_seconds,
                "speedup_full_over_streamed": (
                    full_seconds / streamed_seconds
                    if streamed_seconds > 0.0
                    else math.nan
                ),
                "seconds_saved": full_seconds - streamed_seconds,
                "streamed_fraction_of_full": (
                    streamed_seconds / full_seconds
                    if full_seconds > 0.0
                    else math.nan
                ),
            }
        )
    return rows


def build_accuracy_control_table(
    runs: Dict[str, Dict[str, object]]
) -> List[Dict[str, object]]:
    comparisons = (
        (
            "streaming_only",
            "engd_fjls",
            "js_lstsq0",
            "same lstsq and ridge=0; only Gram backend changes",
        ),
        (
            "dense_solver_only",
            "js_lstsq0",
            "js_eigh0",
            "same streamed Gram and ridge=0; only dense solver changes",
        ),
        (
            "streaming_with_fixed_damping",
            "full_eigh1e7",
            "js_eigh1e7",
            "same EIGH and ridge=1e-7; only Gram backend changes",
        ),
        (
            "damping_only",
            "js_eigh0",
            "js_eigh1e7",
            "same streamed Gram and EIGH; only ridge changes",
        ),
    )
    metrics = (
        "u1_rel_l2",
        "u2_rel_l2",
        "xi_rel_l2",
        "p_rel_l2",
        "T_rel_l2",
        "E_common_max",
        "E_4F_max",
    )
    rows: List[Dict[str, object]] = []
    for name, baseline_method, candidate_method, definition in comparisons:
        if baseline_method not in runs or candidate_method not in runs:
            continue
        baseline = runs[baseline_method]["history"][-1]
        candidate = runs[candidate_method]["history"][-1]
        for metric in metrics:
            baseline_value = float(baseline[metric])
            candidate_value = float(candidate[metric])
            rows.append(
                {
                    "control": name,
                    "definition": definition,
                    "baseline": METHOD_LABELS[baseline_method],
                    "candidate": METHOD_LABELS[candidate_method],
                    "metric": metric,
                    "baseline_value": baseline_value,
                    "candidate_value": candidate_value,
                    "candidate_over_baseline": candidate_value
                    / max(baseline_value, 1.0e-300),
                    "absolute_difference": candidate_value - baseline_value,
                }
            )
    return rows


TABLE3_CONTROLS = (
    ("Streaming", "FJ+LS, 0 -> S+LS, 0", "engd_fjls", "js_lstsq0"),
    ("Solver", "S+LS, 0 -> S+EIGH, 0", "js_lstsq0", "js_eigh0"),
    ("Damping", "S+EIGH, 0 -> S+EIGH, 1e-7", "js_eigh0", "js_eigh1e7"),
    ("Streamed, damped", "FJ+EIGH, 1e-7 -> S+EIGH, 1e-7", "full_eigh1e7", "js_eigh1e7"),
)


def build_table3(runs):
    """Read completed FS100 checkpoints for the four manuscript controls."""
    rows = []
    for control, configuration, baseline_key, candidate_key in TABLE3_CONTROLS:
        if baseline_key not in runs or candidate_key not in runs:
            continue
        baseline = history_at_fs(runs[baseline_key], 100)
        candidate = history_at_fs(runs[candidate_key], 100)
        if baseline is None or candidate is None:
            continue
        rows.append({
            "control": control, "configuration": configuration,
            "baseline_method_key": baseline_key, "candidate_method_key": candidate_key,
            "fs_iteration": 100,
            "baseline_time_seconds": baseline["effective_training_seconds"],
            "candidate_time_seconds": candidate["effective_training_seconds"],
            "baseline_E_common_max": baseline["E_common_max"],
            "candidate_E_common_max": candidate["E_common_max"],
        })
    return rows


def write_table3_text(output, rows):
    """Export display-rounded Markdown and LaTeX from unrounded CSV values."""
    def scientific(value):
        mantissa, exponent = f"{float(value):.2e}".split("e")
        return f"${mantissa} \\times 10^{{{int(exponent)}}}$"
    md = ["# Table 3: Controlled comparisons", "",
          "| Control | Configuration | Time (s) | E_common_max |",
          "|---|---|---:|---:|"]
    tex = [r"\begin{tabular}{llcc}", r"\hline",
           r"Control & Configuration & Time (s) & $E_{\mathrm{common}}^{\max}$ \\" , r"\hline"]
    for row in rows:
        t = f"{row['baseline_time_seconds']:.1f} -> {row['candidate_time_seconds']:.1f}"
        e = f"{scientific(row['baseline_E_common_max'])} -> {scientific(row['candidate_E_common_max'])}"
        md.append(f"| {row['control']} | {row['configuration']} | {t} | {e} |")
        configuration = row['configuration'].replace('1e-7', '$10^{-7}$').replace('->', r'$\to$')
        latex_time = t.replace('->', r'$\to$')
        latex_error = e.replace('->', r'$\to$')
        tex.append(f"{row['control']} & {configuration} & {latex_time} & {latex_error} " + r"\\")
    tex += [r"\hline", r"\end{tabular}"]
    (output / "table3_controlled_comparisons.md").write_text("\n".join(md)+"\n")
    (output / "table3_controlled_comparisons.tex").write_text("\n".join(tex)+"\n")


def combined_rows(
    runs: Dict[str, Dict[str, object]], key: str
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for method, run in runs.items():
        for source in run[key]:
            row = dict(source)
            row["display_method"] = METHOD_LABELS[method]
            rows.append(row)
    return rows


def plot_component_figure(
    runs: Dict[str, Dict[str, object]],
    fields: Sequence[Tuple[str, str]],
    x_key: str,
    x_label: str,
    output_stem: Path,
) -> None:
    fig, axes = plt.subplots(
        1, len(fields), figsize=(5.2 * len(fields), 4.2), sharey=False
    )
    if len(fields) == 1:
        axes = [axes]

    for axis_index, (field, title) in enumerate(fields):
        axis = axes[axis_index]
        for method, run in runs.items():
            history = run["history"]
            x = np.asarray([float(row[x_key]) for row in history])
            y = np.asarray([float(row[field]) for row in history])
            axis.plot(
                x,
                y,
                label=METHOD_LABELS[method],
                color=COLORS[method],
                marker=MARKERS[method],
                markevery=max(1, len(x) // 10),
                markersize=4,
                linewidth=1.7,
            )
        axis.set_yscale("log")
        axis.set_title(title)
        axis.set_xlabel(x_label)
        axis.grid(True, which="both", alpha=0.28)
        if axis_index == 0:
            axis.set_ylabel("Relative $L^2$ error")

        if x_key == "fs_iteration" and "adam_lbfgs" in runs:
            axis.axvline(
                30,
                color=COLORS["adam_lbfgs"],
                linestyle=":",
                linewidth=1.2,
                alpha=0.8,
            )
        elif x_key == "effective_training_seconds" and "adam_lbfgs" in runs:
            switch = history_at_fs(runs["adam_lbfgs"], 30)
            if switch is not None:
                axis.axvline(
                    float(switch["effective_training_seconds"]),
                    color=COLORS["adam_lbfgs"],
                    linestyle=":",
                    linewidth=1.2,
                    alpha=0.8,
                )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(3, len(handles)),
        bbox_to_anchor=(0.5, 1.05),
        frameon=True,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(output_stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def make_plots(runs: Dict[str, Dict[str, object]], output: Path) -> None:
    mechanics = (
        ("u1_rel_l2", r"$u_1$"),
        ("u2_rel_l2", r"$u_2$"),
        ("xi_rel_l2", r"$\xi$"),
    )
    flow_thermal = (
        ("p_rel_l2", r"$p$"),
        ("T_rel_l2", r"$T$"),
    )
    plot_component_figure(
        runs,
        mechanics,
        "effective_training_seconds",
        "Effective training time (s)",
        output / "mechanics_errors_vs_effective_time",
    )
    plot_component_figure(
        runs,
        flow_thermal,
        "effective_training_seconds",
        "Effective training time (s)",
        output / "flow_thermal_errors_vs_effective_time",
    )
    plot_component_figure(
        runs,
        mechanics,
        "fs_iteration",
        "GFS outer iteration",
        output / "mechanics_errors_vs_fs_iteration",
    )
    plot_component_figure(
        runs,
        flow_thermal,
        "fs_iteration",
        "GFS outer iteration",
        output / "flow_thermal_errors_vs_fs_iteration",
    )
    plot_component_figure(
        runs,
        (("E_common_max", r"$E_{\mathrm{common}}^{\max}$"),),
        "effective_training_seconds",
        "Effective training time (s)",
        output / "common_max_error_vs_effective_time",
    )


def main() -> None:
    args = parse_args()
    args.root = args.root.resolve()
    args.output = args.output.resolve()
    protected = [args.root, *Path(__file__).resolve().parents[1].glob("group*/results")]
    if any(args.output == path.resolve() or path.resolve() in args.output.parents
           for path in protected):
        raise ValueError("Choose a report output directory outside all frozen results.")
    args.output.mkdir(parents=True, exist_ok=True)
    runs = discover_runs(args.root, args.prefix, args.allow_incomplete, args.include_optional)
    assert_common_protocol(runs)

    fs50_rows = build_accuracy_table(runs, 50)
    fs100_rows = build_accuracy_table(runs, 100)
    improvement_rows = build_fs50_to_fs100_table(runs)
    stage_rows = build_stage_table(runs)
    speedup_rows = build_js_speedup_table(stage_rows)
    control_rows = build_accuracy_control_table(runs)
    tta_rows = combined_rows(runs, "tta")

    fixed_time_rows: List[Dict[str, object]] = []
    for method, run in runs.items():
        path = run["directory"] / "fixed_time_accuracy.csv"
        if not path.exists():
            continue
        with path.open("r", newline="", encoding="utf-8") as file:
            for source in csv.DictReader(file):
                row = dict(source)
                row["display_method"] = METHOD_LABELS[method]
                fixed_time_rows.append(row)

    write_csv(args.output / "table_accuracy_fs50.csv", fs50_rows)
    write_csv(args.output / "table_accuracy_fs100.csv", fs100_rows)
    write_csv(args.output / "table_fs50_to_fs100_tradeoff.csv", improvement_rows)
    write_csv(args.output / "table_optimizer_stage_times.csv", stage_rows)
    write_csv(args.output / "table_js_speedup_by_stage.csv", speedup_rows)
    write_csv(args.output / "table_controlled_accuracy_comparisons.csv", control_rows)
    write_csv(args.output / "table_time_to_accuracy.csv", tta_rows)
    write_csv(args.output / "table_fixed_time_accuracy.csv", fixed_time_rows)
    table3_rows = build_table3(runs)
    write_csv(args.output / "table3_controlled_comparisons.csv", table3_rows)
    write_table3_text(args.output, table3_rows)
    paper_accuracy = [row for row in fs100_rows if row["method_key"] in PAPER_METHODS]
    write_csv(args.output / "table1_paper_accuracy_fs100.csv", paper_accuracy)
    paper_tta = [row for method in PAPER_METHODS if method in runs
                 for row in combined_rows({method: runs[method]}, "tta")
                 if row["metric"] == "E_common_max" and float(row["threshold"]) in (1e-3,6e-4,4e-4,2e-4,1e-4)]
    write_csv(args.output / "table2_paper_time_to_accuracy.csv", paper_tta)
    make_plots(runs, args.output)

    summary = {
        "protocol": "THP-Group1-4F-GFS-optimizer-JS-v3",
        "json_value_convention": (
            "null denotes an unavailable or undefined report value. "
            "A stage speedup is undefined when the denominator is zero. "
            "Streamed residual and matrix products are fused into the Gram "
            "kernel; their separate recorded times are zero and do not imply "
            "that this work has no computational cost."
        ),
        "batch_prefix": args.prefix,
        "methods_found": list(runs),
        "main_claim_controls": {
            "optimizer_accuracy": (
                "Adam, Adam+L-BFGS, ENGD and ENLM variants share the same 4F-GFS "
                "outer iteration."
            ),
            "streaming": (
                "engd_fjls vs js_lstsq0: same undamped lstsq; only full-J "
                "versus streamed Gram changes."
            ),
            "dense_solver": (
                "js_lstsq0 vs js_eigh0: same streamed Gram and zero ridge."
            ),
            "streaming_damped": (
                "full_eigh1e7 vs js_eigh1e7: same EIGH and ridge=1e-7."
            ),
            "damping": (
                "js_eigh0 vs js_eigh1e7: only ridge changes."
            ),
        },
        "table3_controls": table3_rows,
        "fs50_accuracy": fs50_rows,
        "fs100_accuracy": fs100_rows,
        "fs50_to_fs100": improvement_rows,
        "stage_times": stage_rows,
        "js_speedup": speedup_rows,
        "controlled_accuracy": control_rows,
    }
    with (args.output / "group1_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(json_safe(summary), file, indent=2, ensure_ascii=False, allow_nan=False)

    print("Group-1 aggregation complete:", args.output)
    print("Methods:", ", ".join(runs))


if __name__ == "__main__":
    main()
