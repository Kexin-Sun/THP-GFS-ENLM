"""Aggregate THP Group-3 near-incompressible 3F/4F experiments.

The script consumes the machine-readable files already emitted by the current
``thp-three-field.py`` and ``thp-four-field-core.py`` programs:

* metadata.json
* training_history.csv
* fixed_budget_accuracy.csv
* time_to_accuracy.csv
* engd_profile.csv

It produces final-error, fixed-budget, TTA, Gram-spectrum/effective-rank and
line-search acceptance tables, together with compact cross-nu figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "matplotlib-thp-group3"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHOD_ORDER = ("gfs_three_field", "gfs_four_field")
METHOD_STYLE = {
    "gfs_three_field": {
        "label": "3F-GFS-JS-EIGH-1e-7",
        "short": "3F",
        "color": "#0072B2",
        "marker": "^",
        "linestyle": "--",
    },
    "gfs_four_field": {
        "label": "4F-GFS-JS-EIGH-1e-7",
        "short": "4F",
        "color": "#009E73",
        "marker": "D",
        "linestyle": "-",
    },
}
DEFAULT_NUS = (0.49, 0.4999, 0.49999, 0.4999999)
CHECKPOINTS = (20, 30, 40, 50)
TTA_THRESHOLDS = (1.0e-2, 1.0e-3, 8.0e-4, 6.0e-4, 4.0e-4, 2.0e-4)
PROFILE_FINAL_WINDOW_START = 41
SEED_PATTERN = re.compile(r"(?:^|_)seed(-?\d+)(?:_|$)", re.IGNORECASE)


@dataclass
class Run:
    method: str
    seed: int
    nu: float
    directory: Path
    metadata: Mapping[str, object]
    history: List[Mapping[str, str]]
    fixed_budget: List[Mapping[str, str]]
    tta: List[Mapping[str, str]]
    profile: List[Mapping[str, str]]


def read_csv(path: Path) -> List[Mapping[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def as_float(value: object, default: float = math.nan) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if text == "":
        return default
    try:
        return float(text)
    except (TypeError, ValueError):
        return default


def as_int(value: object, default: int = -1) -> int:
    numeric = as_float(value)
    return int(numeric) if math.isfinite(numeric) else default


def close_float(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1.0e-11, abs_tol=1.0e-13)


def parse_csv_numbers(value: str, converter=float) -> List[object]:
    return [converter(item.strip()) for item in value.split(",") if item.strip()]


def classify_run(directory: Path) -> Optional[str]:
    name = directory.name.lower()
    if "gfs_three_field" in name:
        return "gfs_three_field"
    if "gfs_four_field" in name:
        return "gfs_four_field"
    return None


def nested_mapping(
    mapping: Mapping[str, object], name: str
) -> Mapping[str, object]:
    value = mapping.get(name, {})
    return value if isinstance(value, Mapping) else {}


def network_parameter_count(sizes: object) -> int:
    if not isinstance(sizes, Sequence) or isinstance(sizes, (str, bytes)):
        return 0
    widths = [int(value) for value in sizes]
    return sum((left + 1) * right for left, right in zip(widths, widths[1:]))


def load_run(directory: Path, method: str) -> Run:
    required = (
        "metadata.json",
        "training_history.csv",
        "fixed_budget_accuracy.csv",
        "time_to_accuracy.csv",
        "engd_profile.csv",
    )
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Incomplete run {directory}: missing {', '.join(missing)}"
        )
    with (directory / "metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    physical = nested_mapping(metadata, "physical_parameters")
    nu = as_float(physical.get("nu"))
    seed = as_int(metadata.get("seed"))
    if seed < 0:
        match = SEED_PATTERN.search(directory.name)
        if match is None:
            raise ValueError(f"Cannot determine seed for {directory}")
        seed = int(match.group(1))
    if not math.isfinite(nu):
        raise ValueError(f"Cannot determine nu for {directory}")
    return Run(
        method=method,
        seed=seed,
        nu=nu,
        directory=directory,
        metadata=metadata,
        history=read_csv(directory / "training_history.csv"),
        fixed_budget=read_csv(directory / "fixed_budget_accuracy.csv"),
        tta=read_csv(directory / "time_to_accuracy.csv"),
        profile=read_csv(directory / "engd_profile.csv"),
    )


def discover_runs(
    root: Path,
    batch_tag: str,
    requested_seeds: Sequence[int],
    requested_nus: Sequence[float],
) -> List[Run]:
    token = f"group3_{batch_tag}_nu"
    candidates: Dict[Tuple[str, int, float], List[Run]] = {}
    for history_path in root.rglob("training_history.csv"):
        directory = history_path.parent
        if token not in directory.name:
            continue
        method = classify_run(directory)
        if method is None:
            continue
        run = load_run(directory, method)
        if run.seed not in requested_seeds:
            continue
        matched_nu = next(
            (nu for nu in requested_nus if close_float(run.nu, nu)), None
        )
        if matched_nu is None:
            continue
        candidates.setdefault((method, run.seed, matched_nu), []).append(run)

    selected: List[Run] = []
    missing: List[str] = []
    for nu in requested_nus:
        for seed in requested_seeds:
            for method in METHOD_ORDER:
                key = (method, seed, nu)
                found = candidates.get(key, [])
                if not found:
                    missing.append(f"{method}:nu={nu}:seed={seed}")
                    continue
                found.sort(key=lambda item: item.directory.stat().st_mtime)
                if len(found) > 1:
                    print(
                        "warning: multiple matching runs; using latest: "
                        f"{found[-1].directory}"
                    )
                selected.append(found[-1])
    if missing:
        raise FileNotFoundError(
            "Missing complete Group-3 runs: " + ", ".join(missing)
        )
    return selected


def require_value(
    actual: float, expected: float, label: str, run: Run
) -> None:
    if not close_float(actual, expected):
        raise ValueError(
            f"{run.directory.name}: expected {label}={expected}, got {actual}"
        )


def validate_run(run: Run) -> None:
    metadata = run.metadata
    physical = nested_mapping(metadata, "physical_parameters")
    profiling = nested_mapping(metadata, "profiling")
    engd_u = nested_mapping(metadata, "engd_u")
    engd_pt = nested_mapping(metadata, "engd_pt")
    split_name = "gfs" if run.method == "gfs_three_field" else "four_field_split"
    split = nested_mapping(metadata, split_name)
    accuracy = nested_mapping(metadata, "accuracy_reporting")

    if metadata.get("exact_case") != "distinct":
        raise ValueError(f"{run.directory.name}: exact_case must be distinct")
    for name, expected in (
        ("a0", 0.2),
        ("b0", 0.1),
        ("c0", 0.2),
        ("K", 0.1),
        ("Theta", 0.1),
    ):
        require_value(as_float(physical.get(name)), expected, name, run)
    require_value(as_float(profiling.get("gram_ridge")), 1.0e-7, "gram_ridge", run)

    if profiling.get("solver") != "eigh-refined":
        raise ValueError(f"{run.directory.name}: solver must be eigh-refined")
    if as_int(profiling.get("line_search_positive_levels")) != 31:
        raise ValueError(f"{run.directory.name}: line search must use 31 levels")
    if as_int(profiling.get("point_block_u")) != 512:
        raise ValueError(f"{run.directory.name}: u block must be 512")
    if as_int(profiling.get("point_block_pt")) != 512:
        raise ValueError(f"{run.directory.name}: pT block must be 512")
    if as_int(engd_u.get("steps_per_gfs_iteration")) != 20:
        raise ValueError(f"{run.directory.name}: mechanics updates must be 20")
    if as_int(engd_pt.get("steps_per_gfs_iteration")) != 20:
        raise ValueError(f"{run.directory.name}: pT updates must be 20")
    if as_int(split.get("iterations_requested")) != 50:
        raise ValueError(f"{run.directory.name}: requested FS budget must be 50")
    if as_int(split.get("iterations_completed")) != 50:
        raise ValueError(f"{run.directory.name}: run did not complete FS50")
    if parse_bool(split.get("adaptive_stop_enabled", False)):
        raise ValueError(f"{run.directory.name}: adaptive stopping was enabled")

    checkpoints = accuracy.get("fixed_budget_checkpoints")
    if not isinstance(checkpoints, Sequence) or [int(x) for x in checkpoints] != list(CHECKPOINTS):
        raise ValueError(
            f"{run.directory.name}: checkpoints must be {list(CHECKPOINTS)}"
        )
    if not run.history or as_int(run.history[-1].get("fs_iteration")) != 50:
        raise ValueError(f"{run.directory.name}: history does not end at FS50")


def fixed_row(run: Run, checkpoint: int) -> Mapping[str, str]:
    rows = [
        row
        for row in run.fixed_budget
        if as_int(row.get("requested_fs_iteration")) == checkpoint
    ]
    if len(rows) != 1 or not parse_bool(rows[0].get("checkpoint_reached")):
        raise ValueError(
            f"{run.directory.name}: missing reached FS{checkpoint} record"
        )
    return rows[0]


def final_summary_row(run: Run, batch_tag: str) -> Dict[str, object]:
    row = fixed_row(run, 50)
    physical = nested_mapping(run.metadata, "physical_parameters")
    values: Dict[str, object] = {
        "batch_tag": batch_tag,
        "run_name": run.directory.name,
        "method_key": run.method,
        "method": METHOD_STYLE[run.method]["label"],
        "seed": run.seed,
        "nu": run.nu,
        "one_minus_2nu": 1.0 - 2.0 * run.nu,
        "lambda": as_float(physical.get("lambda")),
        "fs_iteration": 50,
    }
    for name in (
        "pT_engd_updates_completed",
        "u_engd_updates_completed",
        "total_engd_updates_completed",
        "effective_training_seconds",
        "u1_rel_l2",
        "u2_rel_l2",
        "p_rel_l2",
        "T_rel_l2",
        "u_mean_rel_l2",
        "pT_mean_rel_l2",
        "E_common_max",
    ):
        values[name] = as_float(row.get(name))
    values["xi_rel_l2"] = (
        as_float(row.get("xi_rel_l2"))
        if run.method == "gfs_four_field"
        else math.nan
    )
    return values


def fixed_budget_rows(run: Run, batch_tag: str) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    for checkpoint in CHECKPOINTS:
        source = fixed_row(run, checkpoint)
        output.append(
            {
                "batch_tag": batch_tag,
                "run_name": run.directory.name,
                "method_key": run.method,
                "method": METHOD_STYLE[run.method]["label"],
                "seed": run.seed,
                "nu": run.nu,
                "one_minus_2nu": 1.0 - 2.0 * run.nu,
                "fs_iteration": checkpoint,
                "effective_training_seconds": as_float(
                    source.get("effective_training_seconds")
                ),
                "u1_rel_l2": as_float(source.get("u1_rel_l2")),
                "u2_rel_l2": as_float(source.get("u2_rel_l2")),
                "xi_rel_l2": (
                    as_float(source.get("xi_rel_l2"))
                    if run.method == "gfs_four_field"
                    else math.nan
                ),
                "p_rel_l2": as_float(source.get("p_rel_l2")),
                "T_rel_l2": as_float(source.get("T_rel_l2")),
                "E_common_max": as_float(source.get("E_common_max")),
                "total_engd_updates_completed": as_float(
                    source.get("total_engd_updates_completed")
                ),
            }
        )
    return output


def tta_rows(run: Run, batch_tag: str) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    for threshold in TTA_THRESHOLDS:
        matches = [
            row
            for row in run.tta
            if row.get("metric_name") == "E_common_max"
            and close_float(as_float(row.get("threshold")), threshold)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"{run.directory.name}: expected one E_common TTA row at {threshold}"
            )
        source = matches[0]
        reached = parse_bool(source.get("reached"))
        output.append(
            {
                "batch_tag": batch_tag,
                "run_name": run.directory.name,
                "method_key": run.method,
                "method": METHOD_STYLE[run.method]["label"],
                "seed": run.seed,
                "nu": run.nu,
                "one_minus_2nu": 1.0 - 2.0 * run.nu,
                "metric_name": "E_common_max",
                "threshold": threshold,
                "patience": as_int(source.get("patience")),
                "reached": reached,
                "reached_numeric": 1.0 if reached else 0.0,
                "first_fs_iteration": as_float(source.get("first_fs_iteration")),
                "confirmation_fs_iteration": as_float(
                    source.get("confirmation_fs_iteration")
                ),
                "time_to_accuracy_seconds": as_float(
                    source.get("time_to_accuracy_seconds")
                ),
                "effective_training_lower_bound_seconds": as_float(
                    source.get("effective_training_lower_bound_seconds")
                ),
                "final_metric_value": as_float(source.get("final_metric_value")),
            }
        )
    return output


def finite_values(rows: Sequence[Mapping[str, str]], name: str) -> np.ndarray:
    values = np.asarray([as_float(row.get(name)) for row in rows], dtype=float)
    return values[np.isfinite(values)]


def percentile(values: np.ndarray, quantile: float) -> float:
    return float(np.percentile(values, quantile)) if values.size else math.nan


def profile_diagnostic_rows(run: Run, batch_tag: str) -> List[Dict[str, object]]:
    network = nested_mapping(run.metadata, "network")
    u_sizes = network.get("uxi_sizes", network.get("u_sizes"))
    pt_sizes = network.get("pt_sizes")
    parameter_counts = {
        "u": network_parameter_count(u_sizes),
        "pT": network_parameter_count(pt_sizes),
    }
    output: List[Dict[str, object]] = []
    for subproblem in ("u", "pT"):
        all_rows = [
            row for row in run.profile if row.get("subproblem") == subproblem
        ]
        final_rows = [
            row
            for row in all_rows
            if as_int(row.get("fs_iteration")) >= PROFILE_FINAL_WINDOW_START
        ]
        if not all_rows or not final_rows:
            raise ValueError(
                f"{run.directory.name}: missing {subproblem} ENGD profile rows"
            )
        condition = finite_values(final_rows, "effective_condition")
        rank = finite_values(final_rows, "effective_rank")
        lambda_max = finite_values(final_rows, "lambda_max")
        lambda_min = finite_values(final_rows, "lambda_min")
        lambda_min_kept = finite_values(final_rows, "lambda_min_kept")
        solve_residual = finite_values(final_rows, "solve_relative_residual")
        parameter_count = parameter_counts[subproblem]
        accepted_all = np.asarray(
            [parse_bool(row.get("accepted")) for row in all_rows], dtype=float
        )
        accepted_final = np.asarray(
            [parse_bool(row.get("accepted")) for row in final_rows], dtype=float
        )
        rank_fraction = rank / parameter_count if parameter_count else np.asarray([])
        output.append(
            {
                "batch_tag": batch_tag,
                "run_name": run.directory.name,
                "method_key": run.method,
                "method": METHOD_STYLE[run.method]["label"],
                "seed": run.seed,
                "nu": run.nu,
                "one_minus_2nu": 1.0 - 2.0 * run.nu,
                "subproblem": subproblem,
                "spectrum_window": "FS41-FS50",
                "parameter_count": parameter_count,
                "updates_all": int(len(all_rows)),
                "updates_FS41_FS50": int(len(final_rows)),
                "acceptance_rate_all": float(np.mean(accepted_all)),
                "acceptance_rate_FS41_FS50": float(np.mean(accepted_final)),
                "effective_rank_median": percentile(rank, 50),
                "effective_rank_q25": percentile(rank, 25),
                "effective_rank_q75": percentile(rank, 75),
                "effective_rank_fraction_median": percentile(rank_fraction, 50),
                "effective_condition_median": percentile(condition, 50),
                "effective_condition_q25": percentile(condition, 25),
                "effective_condition_q75": percentile(condition, 75),
                "lambda_max_median": percentile(lambda_max, 50),
                "lambda_min_median": percentile(lambda_min, 50),
                "lambda_min_kept_median": percentile(lambda_min_kept, 50),
                "solve_relative_residual_median": percentile(solve_residual, 50),
                "negative_eigenvalue_updates": sum(
                    as_int(row.get("negative_count"), 0) > 0 for row in final_rows
                ),
            }
        )
    return output


def group_rows(
    rows: Sequence[Mapping[str, object]], keys: Sequence[str]
) -> Dict[Tuple[object, ...], List[Mapping[str, object]]]:
    grouped: Dict[Tuple[object, ...], List[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)
    return grouped


def aggregate_numeric_rows(
    rows: Sequence[Mapping[str, object]],
    keys: Sequence[str],
    numeric_names: Sequence[str],
) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    for key, subset in sorted(group_rows(rows, keys).items(), key=lambda item: item[0]):
        record: Dict[str, object] = dict(zip(keys, key))
        record["n_seeds"] = len(subset)
        for name in numeric_names:
            values = np.asarray([as_float(row.get(name)) for row in subset], dtype=float)
            values = values[np.isfinite(values)]
            record[f"{name}_mean"] = float(np.mean(values)) if values.size else math.nan
            record[f"{name}_std"] = (
                float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            ) if values.size else math.nan
        output.append(record)
    return output


FINAL_NUMERIC = (
    "lambda",
    "pT_engd_updates_completed",
    "u_engd_updates_completed",
    "total_engd_updates_completed",
    "effective_training_seconds",
    "u1_rel_l2",
    "u2_rel_l2",
    "xi_rel_l2",
    "p_rel_l2",
    "T_rel_l2",
    "u_mean_rel_l2",
    "pT_mean_rel_l2",
    "E_common_max",
)

DIAGNOSTIC_NUMERIC = (
    "parameter_count",
    "updates_all",
    "updates_FS41_FS50",
    "acceptance_rate_all",
    "acceptance_rate_FS41_FS50",
    "effective_rank_median",
    "effective_rank_fraction_median",
    "effective_condition_median",
    "lambda_max_median",
    "lambda_min_median",
    "lambda_min_kept_median",
    "solve_relative_residual_median",
    "negative_eigenvalue_updates",
)


def lookup_aggregate(
    rows: Sequence[Mapping[str, object]], method: str, nu: float
) -> Mapping[str, object]:
    matches = [
        row
        for row in rows
        if row.get("method_key") == method and close_float(as_float(row.get("nu")), nu)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one aggregate row for {method}, nu={nu}")
    return matches[0]


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.labelsize": 11,
            "axes.titlesize": 11.5,
            "legend.fontsize": 9,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "savefig.bbox": "tight",
        }
    )


def make_error_figure(
    final_aggregate: Sequence[Mapping[str, object]],
    nus: Sequence[float],
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for method in METHOD_ORDER:
        rows = [lookup_aggregate(final_aggregate, method, nu) for nu in nus]
        x = np.asarray([1.0 - 2.0 * nu for nu in nus], dtype=float)
        mean = np.asarray([as_float(row.get("E_common_max_mean")) for row in rows])
        std = np.asarray([as_float(row.get("E_common_max_std"), 0.0) for row in rows])
        lower_error = np.minimum(std, mean * (1.0 - 1.0e-12))
        asymmetric_error = np.vstack((lower_error, std))
        style = METHOD_STYLE[method]
        ax.errorbar(
            x,
            mean,
            yerr=asymmetric_error,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.0,
            markersize=7,
            capsize=3,
            label=style["label"],
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel(r"Incompressibility parameter $1-2\nu$ (smaller $\rightarrow$)")
    ax.set_ylabel(r"Final $E_{\mathrm{common}}^{\max}$ at FS50")
    ax.set_title("Near-incompressible accuracy at a fixed update budget")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, output_dir, "group3_Ecommon_vs_incompressibility")


def diagnostic_lookup(
    rows: Sequence[Mapping[str, object]], method: str, nu: float, subproblem: str
) -> Mapping[str, object]:
    matches = [
        row
        for row in rows
        if row.get("method_key") == method
        and row.get("subproblem") == subproblem
        and close_float(as_float(row.get("nu")), nu)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one diagnostic row for {method}, nu={nu}, {subproblem}"
        )
    return matches[0]


def make_optimizer_diagnostic_figure(
    diagnostic_aggregate: Sequence[Mapping[str, object]],
    nus: Sequence[float],
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.0), sharex=True)
    panels = (
        ("u", "effective_condition_median_mean", "Regularized condition", True),
        ("u", "effective_rank_fraction_median_mean", "Effective-rank fraction", False),
        ("u", "acceptance_rate_all_mean", "Accepted-update fraction", False),
        ("pT", "effective_condition_median_mean", "Regularized condition", True),
        ("pT", "effective_rank_fraction_median_mean", "Effective-rank fraction", False),
        ("pT", "acceptance_rate_all_mean", "Accepted-update fraction", False),
    )
    for panel_index, (ax, (subproblem, metric, ylabel, log_y)) in enumerate(
        zip(axes.flat, panels)
    ):
        for method in METHOD_ORDER:
            values = [
                diagnostic_lookup(diagnostic_aggregate, method, nu, subproblem)
                for nu in nus
            ]
            x = np.asarray([1.0 - 2.0 * nu for nu in nus], dtype=float)
            y = np.asarray([as_float(row.get(metric)) for row in values])
            style = METHOD_STYLE[method]
            ax.plot(
                x,
                y,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=1.8,
                markersize=6,
                label=style["short"],
            )
        ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        ax.set_title("Mechanics block" if subproblem == "u" else "Flow-heat block")
        ax.set_ylabel(ylabel)
        if panel_index >= 3:
            ax.set_xlabel(r"$1-2\nu$")
        if "fraction" in ylabel.lower():
            ax.set_ylim(-0.03, 1.03)
    # Axes share x limits, so invert exactly once.  Repeated inversion would
    # toggle the shared axis back to the wrong direction.
    axes[0, 0].invert_xaxis()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(
        "JS-EIGH diagnostics (spectrum: FS41-FS50; acceptance: FS1-FS50)",
        y=0.99,
    )
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    save_figure(fig, output_dir, "group3_optimizer_diagnostics")


def build_fs100_decisions(
    fixed_rows_all: Sequence[Mapping[str, object]],
    nus: Sequence[float],
) -> List[Dict[str, object]]:
    decisions: List[Dict[str, object]] = []
    per_nu_flags: Dict[float, bool] = {}
    temporary: List[Dict[str, object]] = []
    for nu in nus:
        for method in METHOD_ORDER:
            subset = [
                row
                for row in fixed_rows_all
                if row.get("method_key") == method
                and close_float(as_float(row.get("nu")), nu)
            ]
            by_seed = group_rows(subset, ("seed",))
            ratios: List[float] = []
            for rows in by_seed.values():
                row40 = next(row for row in rows if as_int(row.get("fs_iteration")) == 40)
                row50 = next(row for row in rows if as_int(row.get("fs_iteration")) == 50)
                e40 = as_float(row40.get("E_common_max"))
                e50 = as_float(row50.get("E_common_max"))
                if e40 > 0.0 and math.isfinite(e50):
                    ratios.append(e50 / e40)
            mean_ratio = float(np.mean(ratios)) if ratios else math.nan
            still_descending = bool(math.isfinite(mean_ratio) and mean_ratio <= 0.90)
            temporary.append(
                {
                    "nu": nu,
                    "one_minus_2nu": 1.0 - 2.0 * nu,
                    "method_key": method,
                    "method": METHOD_STYLE[method]["label"],
                    "n_seeds": len(ratios),
                    "Ecommon_FS50_over_FS40_mean": mean_ratio,
                    "relative_reduction_FS40_to_FS50": (
                        1.0 - mean_ratio if math.isfinite(mean_ratio) else math.nan
                    ),
                    "method_still_descending_at_least_10pct": still_descending,
                }
            )
            per_nu_flags[nu] = per_nu_flags.get(nu, False) or still_descending
    most_extreme_nu = max(nus)
    for row in temporary:
        nu = as_float(row["nu"])
        row["is_most_extreme_nu"] = close_float(nu, most_extreme_nu)
        row["recommend_pair_FS100"] = (
            close_float(nu, most_extreme_nu) and per_nu_flags.get(nu, False)
        )
        decisions.append(row)
    return decisions


def metric_text(mean: float, std: float, n: int, scientific: bool = True) -> str:
    if not math.isfinite(mean):
        return "--"
    formatter = (lambda value: f"{value:.3e}") if scientific else (lambda value: f"{value:.3f}")
    if n <= 1:
        return formatter(mean)
    return f"{formatter(mean)} +/- {formatter(std)}"


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def build_publication_main_rows(
    final_aggregate: Sequence[Mapping[str, object]],
    diagnostic_aggregate: Sequence[Mapping[str, object]],
    tta_rows_all: Sequence[Mapping[str, object]],
    nus: Sequence[float],
) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    tta_aggregate = aggregate_numeric_rows(
        [row for row in tta_rows_all if close_float(as_float(row.get("threshold")), 1.0e-3)],
        ("nu", "one_minus_2nu", "method_key", "method", "threshold"),
        ("reached_numeric", "time_to_accuracy_seconds"),
    )
    for nu in nus:
        for method in METHOD_ORDER:
            final = lookup_aggregate(final_aggregate, method, nu)
            u_diag = diagnostic_lookup(diagnostic_aggregate, method, nu, "u")
            pt_diag = diagnostic_lookup(diagnostic_aggregate, method, nu, "pT")
            tta_match = [
                row
                for row in tta_aggregate
                if row.get("method_key") == method
                and close_float(as_float(row.get("nu")), nu)
            ]
            all_reached = bool(
                tta_match
                and close_float(
                    as_float(tta_match[0].get("reached_numeric_mean")), 1.0
                )
            )
            tta_mean = (
                as_float(tta_match[0].get("time_to_accuracy_seconds_mean"))
                if all_reached
                else math.nan
            )
            output.append(
                {
                    "nu": nu,
                    "one_minus_2nu": 1.0 - 2.0 * nu,
                    "method": METHOD_STYLE[method]["short"],
                    "n_seeds": as_int(final.get("n_seeds")),
                    "u1_rel_l2_mean": as_float(final.get("u1_rel_l2_mean")),
                    "u2_rel_l2_mean": as_float(final.get("u2_rel_l2_mean")),
                    "p_rel_l2_mean": as_float(final.get("p_rel_l2_mean")),
                    "T_rel_l2_mean": as_float(final.get("T_rel_l2_mean")),
                    "xi_rel_l2_mean": as_float(final.get("xi_rel_l2_mean")),
                    "E_common_max_mean": as_float(final.get("E_common_max_mean")),
                    "TTA_Ecommon_1e-3_seconds_mean": tta_mean,
                    "effective_training_seconds_mean": as_float(
                        final.get("effective_training_seconds_mean")
                    ),
                    "u_condition_median_mean": as_float(
                        u_diag.get("effective_condition_median_mean")
                    ),
                    "u_rank_fraction_mean": as_float(
                        u_diag.get("effective_rank_fraction_median_mean")
                    ),
                    "u_acceptance_rate_mean": as_float(
                        u_diag.get("acceptance_rate_all_mean")
                    ),
                    "pT_condition_median_mean": as_float(
                        pt_diag.get("effective_condition_median_mean")
                    ),
                    "pT_rank_fraction_mean": as_float(
                        pt_diag.get("effective_rank_fraction_median_mean")
                    ),
                    "pT_acceptance_rate_mean": as_float(
                        pt_diag.get("acceptance_rate_all_mean")
                    ),
                }
            )
    return output


def write_publication_markdown(
    path: Path,
    main_rows: Sequence[Mapping[str, object]],
    decisions: Sequence[Mapping[str, object]],
) -> None:
    table_rows: List[List[object]] = []
    for row in main_rows:
        table_rows.append(
            [
                f"{as_float(row['nu']):.7f}".rstrip("0").rstrip("."),
                row["method"],
                f"{as_float(row['u1_rel_l2_mean']):.3e}",
                f"{as_float(row['u2_rel_l2_mean']):.3e}",
                f"{as_float(row['p_rel_l2_mean']):.3e}",
                f"{as_float(row['T_rel_l2_mean']):.3e}",
                (
                    f"{as_float(row['xi_rel_l2_mean']):.3e}"
                    if math.isfinite(as_float(row["xi_rel_l2_mean"]))
                    else "--"
                ),
                f"{as_float(row['E_common_max_mean']):.3e}",
                (
                    f"{as_float(row['TTA_Ecommon_1e-3_seconds_mean']):.2f}"
                    if math.isfinite(as_float(row["TTA_Ecommon_1e-3_seconds_mean"]))
                    else "NR"
                ),
                f"{as_float(row['effective_training_seconds_mean']):.2f}",
                f"{as_float(row['u_condition_median_mean']):.3e}",
                f"{as_float(row['u_rank_fraction_mean']):.3f}",
                f"{as_float(row['u_acceptance_rate_mean']):.3f}",
                f"{as_float(row['pT_condition_median_mean']):.3e}",
                f"{as_float(row['pT_rank_fraction_mean']):.3f}",
                f"{as_float(row['pT_acceptance_rate_mean']):.3f}",
            ]
        )
    decision_rows = [
        [
            f"{as_float(row['nu']):.7f}".rstrip("0").rstrip("."),
            METHOD_STYLE[str(row["method_key"])]["short"],
            f"{as_float(row['Ecommon_FS50_over_FS40_mean']):.4f}",
            f"{as_float(row['relative_reduction_FS40_to_FS50']):.2%}",
            str(bool(row["recommend_pair_FS100"])),
        ]
        for row in decisions
    ]
    text = "\n".join(
        (
            "# THP Group 3: near-incompressible 3F/4F comparison",
            "",
            "FS50 is the mandatory common-budget result. Gram condition and rank are medians over FS41-FS50; acceptance rates use FS1-FS50. TTA uses three consecutive completed sweeps.",
            "",
            markdown_table(
                (
                    "nu", "Form", "u1", "u2", "p", "T", "xi",
                    "Ecommon", "TTA@1e-3(s)", "Time(s)",
                    "cond-u", "rank/P-u", "accept-u",
                    "cond-pT", "rank/P-pT", "accept-pT",
                ),
                table_rows,
            ),
            "",
            "## FS100 decision",
            "",
            "The pair is extended only at the most extreme nu when either formulation decreases by at least 10% from FS40 to FS50.",
            "",
            markdown_table(
                ("nu", "Form", "E50/E40", "Reduction", "Extend pair to FS100"),
                decision_rows,
            ),
            "",
        )
    )
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root", type=Path, nargs="?", default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--batch-tag", required=True)
    parser.add_argument("--seeds", default="12345")
    parser.add_argument(
        "--nus", default=",".join(str(value) for value in DEFAULT_NUS)
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_root = args.result_root.resolve()
    seeds = [int(value) for value in parse_csv_numbers(args.seeds, int)]
    nus = [float(value) for value in parse_csv_numbers(args.nus, float)]
    if len(set(seeds)) != len(seeds):
        raise ValueError("--seeds contains duplicates")
    if len(set(nus)) != len(nus):
        raise ValueError("--nus contains duplicates")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else Path(__file__).resolve().parent / "archive_summary"
    )
    # Reports must not modify any frozen or training run directory.
    if output_dir == result_root or result_root in output_dir.parents:
        raise ValueError("Choose --output-dir outside the input result directory.")
    protected = Path(__file__).resolve().parents[1].glob("group*/results")
    if any(output_dir == path.resolve() or path.resolve() in output_dir.parents
           for path in protected):
        raise ValueError("Choose a report output directory outside all frozen results.")
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(result_root, args.batch_tag, seeds, nus)
    for run in runs:
        validate_run(run)

    final_rows = [final_summary_row(run, args.batch_tag) for run in runs]
    fixed_rows_all = [row for run in runs for row in fixed_budget_rows(run, args.batch_tag)]
    tta_rows_all = [row for run in runs for row in tta_rows(run, args.batch_tag)]
    diagnostic_rows = [
        row for run in runs for row in profile_diagnostic_rows(run, args.batch_tag)
    ]

    final_aggregate = aggregate_numeric_rows(
        final_rows,
        ("nu", "one_minus_2nu", "method_key", "method"),
        FINAL_NUMERIC,
    )
    fixed_aggregate = aggregate_numeric_rows(
        fixed_rows_all,
        ("nu", "one_minus_2nu", "method_key", "method", "fs_iteration"),
        (
            "effective_training_seconds",
            "u1_rel_l2",
            "u2_rel_l2",
            "xi_rel_l2",
            "p_rel_l2",
            "T_rel_l2",
            "E_common_max",
            "total_engd_updates_completed",
        ),
    )
    tta_aggregate = aggregate_numeric_rows(
        tta_rows_all,
        ("nu", "one_minus_2nu", "method_key", "method", "threshold"),
        (
            "reached_numeric",
            "time_to_accuracy_seconds",
            "effective_training_lower_bound_seconds",
            "final_metric_value",
        ),
    )
    diagnostic_aggregate = aggregate_numeric_rows(
        diagnostic_rows,
        ("nu", "one_minus_2nu", "method_key", "method", "subproblem"),
        DIAGNOSTIC_NUMERIC,
    )
    decisions = build_fs100_decisions(fixed_rows_all, nus)
    publication_main = build_publication_main_rows(
        final_aggregate, diagnostic_aggregate, tta_rows_all, nus
    )

    write_csv(output_dir / "group3_final_summary.csv", final_rows)
    write_csv(output_dir / "group3_final_summary_mean_std.csv", final_aggregate)
    write_csv(output_dir / "group3_fixed_budget.csv", fixed_rows_all)
    write_csv(output_dir / "group3_fixed_budget_mean_std.csv", fixed_aggregate)
    write_csv(output_dir / "group3_tta_Ecommon.csv", tta_rows_all)
    write_csv(output_dir / "group3_tta_Ecommon_mean_std.csv", tta_aggregate)
    write_csv(output_dir / "group3_gram_diagnostics.csv", diagnostic_rows)
    write_csv(
        output_dir / "group3_gram_diagnostics_mean_std.csv", diagnostic_aggregate
    )
    write_csv(output_dir / "group3_FS100_decision.csv", decisions)
    write_csv(output_dir / "group3_publication_main_table.csv", publication_main)
    write_publication_markdown(
        output_dir / "group3_publication_tables.md", publication_main, decisions
    )

    configure_style()
    make_error_figure(final_aggregate, nus, output_dir)
    make_optimizer_diagnostic_figure(diagnostic_aggregate, nus, output_dir)

    manifest = {
        "batch_tag": args.batch_tag,
        "result_root": str(result_root),
        "output_dir": str(output_dir),
        "seeds": seeds,
        "nus": nus,
        "methods": list(METHOD_ORDER),
        "fs_budget": 50,
        "checkpoints": list(CHECKPOINTS),
        "profile_spectrum_window": "FS41-FS50",
        "acceptance_window": "FS1-FS50",
        "tta_thresholds": list(TTA_THRESHOLDS),
        "tta_patience": 3,
        "primary_metric": "max(u1_rel_l2,u2_rel_l2,p_rel_l2,T_rel_l2)",
        "figures": [
            "group3_Ecommon_vs_incompressibility",
            "group3_optimizer_diagnostics",
        ],
    }
    with (output_dir / "group3_report_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    print(f"Loaded and validated {len(runs)} runs")
    print(f"Group-3 reports written to: {output_dir}")


if __name__ == "__main__":
    main()
