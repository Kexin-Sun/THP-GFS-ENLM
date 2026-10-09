"""Aggregate and plot THP Group-2 architecture-comparison experiments.

Examples
--------
Single-seed or latest completed run for every seed::

    python plot_group2_comparison.py results

One named multi-seed batch::

    python plot_group2_comparison.py results \
        --batch-tag multiseed_v1 \
        --seeds 12345,23456,34567,45678,56789

The script pairs the four methods by seed. Positive-valued convergence curves
use the geometric mean across seeds and a shaded +/- one sample-standard-
deviation band in log10 space. This is the appropriate aggregation for the
logarithmic error axes and guarantees positive band limits. Final-error,
runtime, timing, and time-to-accuracy figures retain arithmetic mean +/- one
sample standard deviation error bars. With one seed, the same code remains
valid and the standard deviation is reported as zero.
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
    str(Path(tempfile.gettempdir()) / "matplotlib-thp-group2"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHOD_ORDER = (
    "coupled_single",
    "coupled_dual",
    "gfs_three_field",
    "gfs_four_field",
)

METHOD_STYLE = {
    "coupled_single": {
        "label": "Coupled-Single",
        "color": "#D55E00",
        "linestyle": "--",
        "marker": "o",
    },
    "coupled_dual": {
        "label": "Coupled-Dual",
        "color": "#CC79A7",
        "linestyle": ":",
        "marker": "s",
    },
    "gfs_three_field": {
        "label": "3F-GFS",
        "color": "#0072B2",
        "linestyle": "-.",
        "marker": "^",
    },
    "gfs_four_field": {
        "label": "4F-GFS",
        "color": "#009E73",
        "linestyle": "-",
        "marker": "D",
    },
}

TTA_THRESHOLDS = (1.0e-2, 1.0e-3, 8.0e-4, 6.0e-4, 4.0e-4, 2.0e-4)
TTA_PATIENCE = 3
SEED_PATTERN = re.compile(r"(?:^|_)seed(-?\d+)(?:_|$)", re.IGNORECASE)


@dataclass
class Run:
    key: str
    seed: int
    directory: Path
    metadata: Mapping[str, object]
    rows: List[Mapping[str, str]]
    values: Dict[str, np.ndarray]


def read_csv(path: Path) -> List[Mapping[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def to_float(row: Mapping[str, str], name: str) -> float:
    value = row.get(name, "")
    if value is None or value.strip() == "":
        return math.nan
    return float(value)


def column(rows: Sequence[Mapping[str, str]], name: str) -> np.ndarray:
    return np.asarray([to_float(row, name) for row in rows], dtype=float)


def classify_run(directory: Path) -> Optional[str]:
    name = directory.name.lower()
    if "coupled_single" in name:
        return "coupled_single"
    if "coupled_dual" in name:
        return "coupled_dual"
    if "gfs_three_field" in name:
        return "gfs_three_field"
    if "gfs_four_field" in name:
        return "gfs_four_field"
    return None


def extract_seed(directory: Path, metadata: Mapping[str, object]) -> int:
    match = SEED_PATTERN.search(directory.name)
    if match is not None:
        return int(match.group(1))
    for name in ("seed", "random_seed"):
        value = metadata.get(name)
        if value is not None:
            return int(value)
    raise ValueError(f"Cannot determine seed from {directory}")


def load_run(directory: Path, key: str) -> Run:
    with (directory / "metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    rows = read_csv(directory / "training_history.csv")
    if not rows:
        raise ValueError(f"Empty training history: {directory}")

    coupled = key.startswith("coupled_")
    if coupled:
        equivalent_iteration = column(rows, "equivalent_iteration")
        effective_time = column(rows, "effective_training_seconds")
        common = column(rows, "E_common_max")
        four_field = column(rows, "E_4F_max")
        outer_change_common = column(rows, "prediction_change")
        u_outer_change = np.full_like(common, np.nan)
        pt_outer_change = np.full_like(common, np.nan)
    else:
        engd_pt = metadata.get("engd_pt", {})
        steps = float(
            engd_pt.get("steps_per_gfs_iteration", 20)
            if isinstance(engd_pt, Mapping) else 20
        )
        equivalent_iteration = column(rows, "fs_iteration") * steps
        effective_time = column(rows, "training_time")
        common = column(rows, "common_rel_l2_max")
        u_outer_change = column(rows, "u_outer_change")
        pt_outer_change = column(rows, "pt_outer_change")
        outer_change_common = np.maximum(u_outer_change, pt_outer_change)
        if key == "gfs_four_field":
            four_field = column(rows, "four_field_component_rel_l2_max")
        else:
            four_field = np.full_like(common, np.nan)

    xi = (
        column(rows, "xi_rel_l2")
        if "xi_rel_l2" in rows[0]
        else np.full_like(common, np.nan)
    )
    values = {
        "equivalent_iteration": equivalent_iteration,
        "effective_time": effective_time,
        "u1_rel_l2": column(rows, "u1_rel_l2"),
        "u2_rel_l2": column(rows, "u2_rel_l2"),
        "xi_rel_l2": xi,
        "p_rel_l2": column(rows, "p_rel_l2"),
        "T_rel_l2": column(rows, "T_rel_l2"),
        "u_mean_rel_l2": column(rows, "u_mean_rel_l2"),
        "pT_mean_rel_l2": column(rows, "pT_mean_rel_l2"),
        "E_common_max": common,
        "E_4F_max": four_field,
        "outer_change_common": outer_change_common,
        "u_outer_change": u_outer_change,
        "pt_outer_change": pt_outer_change,
    }
    return Run(
        key=key,
        seed=extract_seed(directory, metadata),
        directory=directory,
        metadata=metadata,
        rows=rows,
        values=values,
    )


def parse_seed_csv(value: Optional[str]) -> Optional[List[int]]:
    if value is None:
        return None
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("--seeds did not contain any valid integer seed")
    if len(set(seeds)) != len(seeds):
        raise ValueError("--seeds contains duplicate values")
    return seeds


def discover_runs(
    root: Path,
    batch_tag: Optional[str],
    requested_seeds: Optional[Sequence[int]],
) -> Dict[str, List[Run]]:
    candidates: Dict[str, Dict[int, List[Run]]] = {
        key: {} for key in METHOD_ORDER
    }
    batch_token = f"group2_{batch_tag}_seed" if batch_tag else None

    for history in root.rglob("training_history.csv"):
        directory = history.parent
        if not (directory / "metadata.json").is_file():
            continue
        if batch_token is not None and batch_token not in directory.name:
            continue
        key = classify_run(directory)
        if key is None:
            continue
        run = load_run(directory, key)
        if requested_seeds is not None and run.seed not in requested_seeds:
            continue
        candidates[key].setdefault(run.seed, []).append(run)

    available_sets = [set(candidates[key]) for key in METHOD_ORDER]
    if requested_seeds is not None:
        selected_seeds = list(requested_seeds)
        missing = [
            f"{key}:seed{seed}"
            for key in METHOD_ORDER
            for seed in selected_seeds
            if seed not in candidates[key]
        ]
        if missing:
            raise FileNotFoundError(
                "Missing requested Group-2 runs: " + ", ".join(missing)
            )
    else:
        common_seeds = set.intersection(*available_sets) if available_sets else set()
        if not common_seeds:
            raise FileNotFoundError(
                "No seed has a complete set of all four Group-2 methods"
            )
        selected_seeds = sorted(common_seeds)
        all_seeds = set.union(*available_sets)
        incomplete = sorted(all_seeds - common_seeds)
        if incomplete:
            print(
                "warning: ignoring seeds without all four methods: "
                + ", ".join(str(seed) for seed in incomplete)
            )

    selected: Dict[str, List[Run]] = {key: [] for key in METHOD_ORDER}
    for key in METHOD_ORDER:
        for seed in selected_seeds:
            method_seed_runs = candidates[key][seed]
            method_seed_runs.sort(key=lambda run: run.directory.stat().st_mtime)
            if len(method_seed_runs) > 1:
                print(
                    f"warning: found {len(method_seed_runs)} runs for {key}, "
                    f"seed={seed}; using latest "
                    f"{method_seed_runs[-1].directory.name}"
                )
            selected[key].append(method_seed_runs[-1])

    print(
        "paired seeds: "
        + ", ".join(str(seed) for seed in selected_seeds)
        + f" (n={len(selected_seeds)})"
    )
    return selected


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.labelsize": 11,
            "axes.titlesize": 11.5,
            "legend.fontsize": 9.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.7,
            "savefig.bbox": "tight",
        }
    )


def clean_curve(
    run: Run, x_name: str, y_name: str
) -> Tuple[np.ndarray, np.ndarray]:
    x = run.values[x_name]
    y = run.values[y_name]
    valid = np.isfinite(x) & np.isfinite(y) & (y > 0.0)
    x = x[valid]
    y = y[valid]
    if x.size == 0:
        return x, y
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    unique_x, unique_indices = np.unique(x, return_index=True)
    return unique_x, y[unique_indices]


def aligned_log_curve_matrix(
    runs: Sequence[Run], x_name: str, y_name: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Align positive curves and return their log10 values on one grid."""
    curves = [clean_curve(run, x_name, y_name) for run in runs]
    curves = [(x, y) for x, y in curves if x.size]
    if not curves:
        return np.asarray([]), np.empty((0, 0), dtype=float)

    lower = max(float(x[0]) for x, _ in curves)
    upper = min(float(x[-1]) for x, _ in curves)
    if upper < lower:
        return np.asarray([]), np.empty((0, 0), dtype=float)

    if x_name == "equivalent_iteration":
        grid = curves[0][0]
        grid = grid[(grid >= lower) & (grid <= upper)]
    else:
        grid = np.linspace(lower, upper, 160)
    if grid.size == 0:
        return np.asarray([]), np.empty((0, 0), dtype=float)

    log_samples = []
    for x, y in curves:
        # Interpolation in log space preserves positivity and respects the
        # multiplicative scale of errors spanning several orders of magnitude.
        log_samples.append(np.interp(grid, x, np.log10(y)))
    return grid, np.vstack(log_samples)


def aggregate_curve(
    runs: Sequence[Run], x_name: str, y_name: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return arithmetic mean/SD on an aligned grid for tabular reporting."""
    grid, log_matrix = aligned_log_curve_matrix(runs, x_name, y_name)
    if grid.size == 0:
        return np.asarray([]), np.asarray([]), np.asarray([])

    matrix = 10.0 ** log_matrix
    mean = np.mean(matrix, axis=0)
    std = (
        np.std(matrix, axis=0, ddof=1)
        if matrix.shape[0] > 1 else np.zeros_like(mean)
    )
    return grid, mean, std


def aggregate_log_curve_band(
    runs: Sequence[Run], x_name: str, y_name: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return geometric center and a +/-1-SD band in log10 space."""
    grid, log_matrix = aligned_log_curve_matrix(runs, x_name, y_name)
    if grid.size == 0:
        empty = np.asarray([])
        return empty, empty, empty, empty, empty

    log_mean = np.mean(log_matrix, axis=0)
    log_std = (
        np.std(log_matrix, axis=0, ddof=1)
        if log_matrix.shape[0] > 1 else np.zeros_like(log_mean)
    )
    center = 10.0 ** log_mean
    lower = 10.0 ** (log_mean - log_std)
    upper = 10.0 ** (log_mean + log_std)
    return grid, center, lower, upper, log_std


def plot_curve_band(
    axis: plt.Axes,
    runs: Sequence[Run],
    x_name: str,
    y_name: str,
) -> None:
    x, center, lower, upper, _ = aggregate_log_curve_band(
        runs, x_name, y_name
    )
    if x.size == 0:
        return
    style = METHOD_STYLE[runs[0].key]
    marker_every = max(1, x.size // 10)
    axis.plot(
        x,
        center,
        label=style["label"],
        color=style["color"],
        linestyle=style["linestyle"],
        linewidth=2.0,
        marker=style["marker"],
        markersize=4.2,
        markevery=marker_every,
    )
    if len(runs) > 1:
        axis.fill_between(
            x, lower, upper, color=style["color"], alpha=0.16, linewidth=0.0
        )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=320, facecolor="white")
    fig.savefig(output_dir / f"{stem}.pdf", facecolor="white")
    plt.close(fig)


def add_shared_legend(fig: plt.Figure, axes: Iterable[plt.Axes]) -> None:
    handles: List[object] = []
    labels: List[str] = []
    for axis in axes:
        current_handles, current_labels = axis.get_legend_handles_labels()
        for handle, label in zip(current_handles, current_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
    )


def make_component_figure(
    runs: Mapping[str, Sequence[Run]], output_dir: Path
) -> None:
    panels = (
        ("u1_rel_l2", r"(a) $u_1$"),
        ("u2_rel_l2", r"(b) $u_2$"),
        ("p_rel_l2", r"(c) $p$"),
        ("T_rel_l2", r"(d) $T$"),
        ("xi_rel_l2", r"(e) $\xi$ (four-field only)"),
        ("E_common_max", r"(f) $E_{\mathrm{common}}$"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.4), sharex=True)
    flat_axes = list(axes.flat)
    for axis, (metric, title) in zip(flat_axes, panels):
        for key in METHOD_ORDER:
            plot_curve_band(
                axis, runs[key], "equivalent_iteration", metric
            )
        axis.set_yscale("log")
        axis.set_title(title)
        axis.set_xlabel("Equivalent ENLM updates")
        axis.set_ylabel(r"Relative $L^2$ error")
        axis.set_xlim(left=0.0)
    add_shared_legend(fig, flat_axes)
    fig.text(
        0.5, 0.008,
        "Lines: geometric mean across seeds; bands: +/- 1 sample SD in "
        "log10(error).",
        ha="center", fontsize=9.2, color="#555555",
    )
    fig.tight_layout(rect=(0.0, 0.025, 1.0, 0.94))
    save_figure(fig, output_dir, "group2_component_errors_vs_iterations")


def make_time_figure(
    runs: Mapping[str, Sequence[Run]], output_dir: Path
) -> None:
    panels = (
        ("u_mean_rel_l2", r"(a) Mean displacement error $E_u$"),
        ("pT_mean_rel_l2", r"(b) Mean pressure-temperature error $E_{pT}$"),
        ("E_common_max", r"(c) Common maximum error $E_{\mathrm{common}}$"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.9))
    for axis, (metric, title) in zip(axes, panels):
        for key in METHOD_ORDER:
            plot_curve_band(axis, runs[key], "effective_time", metric)
        axis.set_yscale("log")
        axis.set_title(title)
        axis.set_xlabel("Effective training time (s)")
        axis.set_ylabel(r"Relative $L^2$ error")
        axis.set_xlim(left=0.0)
    add_shared_legend(fig, axes)
    fig.text(
        0.5, 0.008,
        "Time grids are aligned by log-error interpolation; lines show the "
        "geometric mean and bands show +/- 1 sample SD in log10(error).",
        ha="center", fontsize=9.2, color="#555555",
    )
    fig.tight_layout(rect=(0.0, 0.035, 1.0, 0.90))
    save_figure(fig, output_dir, "group2_main_errors_vs_time")


def make_outer_convergence_figure(
    runs: Mapping[str, Sequence[Run]], output_dir: Path
) -> None:
    """Plot the convergence diagnostic saved at every outer iteration."""
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.9), sharex=True)

    for key in METHOD_ORDER:
        plot_curve_band(
            axes[0], runs[key], "equivalent_iteration", "outer_change_common"
        )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Equivalent ENLM updates")
    axes[0].set_ylabel("Relative prediction change")
    axes[0].set_title("(a) Common outer-convergence diagnostic")
    axes[0].set_xlim(left=0.0)
    axes[0].legend(frameon=False, ncol=2)

    for key in ("gfs_three_field", "gfs_four_field"):
        style = METHOD_STYLE[key]
        for metric, component, linestyle, marker in (
            ("u_outer_change", "mechanics", "-", "o"),
            ("pt_outer_change", "pressure-temperature", "--", "s"),
        ):
            x, center, lower, upper, _ = aggregate_log_curve_band(
                runs[key], "equivalent_iteration", metric
            )
            if x.size == 0:
                continue
            axes[1].plot(
                x, center,
                label=f"{style['label']}: {component}",
                color=style["color"], linestyle=linestyle,
                linewidth=2.0, marker=marker, markersize=4.0,
                markevery=max(1, x.size // 10),
            )
            if len(runs[key]) > 1:
                axes[1].fill_between(
                    x, lower, upper,
                    color=style["color"], alpha=0.13, linewidth=0.0,
                )
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Equivalent ENLM updates")
    axes[1].set_ylabel("Relative prediction change")
    axes[1].set_title("(b) GFS subproblem outer changes")
    axes[1].set_xlim(left=0.0)
    axes[1].legend(frameon=False, fontsize=8.7)

    fig.text(
        0.5, 0.008,
        "Coupled: full-network prediction change; GFS common diagnostic: "
        "max(mechanics, pressure-temperature). Lines show the geometric "
        "mean; bands show +/- 1 sample SD in log10(change).",
        ha="center", fontsize=8.8, color="#555555",
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    save_figure(fig, output_dir, "group2_outer_convergence_vs_iterations")


def final_value(run: Run, metric: str) -> float:
    values = run.values[metric]
    valid = values[np.isfinite(values)]
    return float(valid[-1]) if valid.size else math.nan


def sample_mean_std(values: Sequence[float]) -> Tuple[float, float, int]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return math.nan, math.nan, 0
    return (
        float(np.mean(array)),
        float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        int(array.size),
    )


def final_mean_std(
    runs: Sequence[Run], metric: str
) -> Tuple[float, float, int]:
    return sample_mean_std([final_value(run, metric) for run in runs])


def make_final_summary_figure(
    runs: Mapping[str, Sequence[Run]], output_dir: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.9))
    x = np.arange(len(METHOD_ORDER), dtype=float)
    width = 0.24
    metric_specs = (
        ("u_mean_rel_l2", r"$E_u$", -width),
        ("pT_mean_rel_l2", r"$E_{pT}$", 0.0),
        ("E_common_max", r"$E_{common}$", width),
    )
    metric_colors = ("#56B4E9", "#E69F00", "#4D4D4D")
    for (metric, label, offset), color in zip(metric_specs, metric_colors):
        means = []
        lower_errors = []
        upper_errors = []
        for key in METHOD_ORDER:
            mean, std, _ = final_mean_std(runs[key], metric)
            means.append(mean)
            lower_errors.append(min(std, mean * 0.95))
            upper_errors.append(std)
        axes[0].bar(
            x + offset,
            means,
            width=width,
            label=label,
            color=color,
            yerr=np.asarray([lower_errors, upper_errors]),
            capsize=3,
            error_kw={"elinewidth": 1.0, "capthick": 1.0},
        )
    axes[0].set_yscale("log")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(
        [METHOD_STYLE[key]["label"] for key in METHOD_ORDER], rotation=12
    )
    axes[0].set_ylabel(r"Final relative $L^2$ error")
    axes[0].set_title("(a) Final accuracy at 1000 equivalent iterations")
    axes[0].legend(frameon=False, ncol=3)

    runtime_stats = [
        final_mean_std(runs[key], "effective_time") for key in METHOD_ORDER
    ]
    runtime_means = [item[0] for item in runtime_stats]
    runtime_stds = [item[1] for item in runtime_stats]
    colors = [METHOD_STYLE[key]["color"] for key in METHOD_ORDER]
    bars = axes[1].bar(
        x, runtime_means, yerr=runtime_stds, color=colors, width=0.68,
        capsize=4, error_kw={"elinewidth": 1.2, "capthick": 1.2},
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        [METHOD_STYLE[key]["label"] for key in METHOD_ORDER], rotation=12
    )
    axes[1].set_ylabel("Effective training time (s)")
    axes[1].set_title("(b) Full-budget runtime")
    for bar, mean, std in zip(bars, runtime_means, runtime_stds):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            mean + std,
            f"{mean:.1f} +/- {std:.1f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    upper = max(mean + std for mean, std in zip(runtime_means, runtime_stds))
    axes[1].set_ylim(0.0, upper * 1.17)
    fig.text(
        0.5, 0.005,
        "Bars: mean across paired seeds; error bars: +/- 1 sample SD.",
        ha="center", fontsize=9.2, color="#555555",
    )
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 1.0))
    save_figure(fig, output_dir, "group2_final_accuracy_and_runtime")


def timing_value(run: Run, name: str) -> float:
    timing = run.metadata.get("timing", {})
    value = timing.get(name, 0.0) if isinstance(timing, Mapping) else 0.0
    return float(value or 0.0)


def memory_gib(run: Run, name: str) -> float:
    memory = run.metadata.get("memory", {})
    value = memory.get(name) if isinstance(memory, Mapping) else None
    if value is None:
        return math.nan
    return float(value) / float(1024 ** 3)


def completed_outer_iterations(run: Run) -> float:
    if "round" in run.rows[0]:
        return to_float(run.rows[-1], "round")
    return to_float(run.rows[-1], "fs_iteration")


def make_timing_breakdown_figure(
    runs: Mapping[str, Sequence[Run]], output_dir: Path
) -> None:
    components = (
        ("assembly_seconds", "Gram assembly", "#0072B2"),
        ("solve_seconds", "Eigensolve", "#E69F00"),
        ("line_search_seconds", "Line search", "#009E73"),
        ("convergence_check_seconds", "Convergence check", "#CC79A7"),
        ("context_seconds", "Context", "#999999"),
    )
    x = np.arange(len(METHOD_ORDER))
    bottom = np.zeros(len(METHOD_ORDER), dtype=float)
    fig, axis = plt.subplots(figsize=(8.7, 5.1))
    for name, label, color in components:
        means = np.asarray(
            [
                sample_mean_std(
                    [timing_value(run, name) for run in runs[key]]
                )[0]
                for key in METHOD_ORDER
            ]
        )
        axis.bar(x, means, bottom=bottom, label=label, color=color, width=0.68)
        bottom += means

    runtime_stds = np.asarray(
        [final_mean_std(runs[key], "effective_time")[1] for key in METHOD_ORDER]
    )
    axis.errorbar(
        x, bottom, yerr=runtime_stds, fmt="none", ecolor="black",
        capsize=4, elinewidth=1.2, capthick=1.2,
        label="Total runtime SD",
    )
    axis.set_xticks(x)
    axis.set_xticklabels(
        [METHOD_STYLE[key]["label"] for key in METHOD_ORDER], rotation=10
    )
    axis.set_ylabel("Effective training time (s)")
    axis.set_title("Mean optimizer-time decomposition")
    axis.legend(frameon=False, ncol=3)
    for index, (total, std) in enumerate(zip(bottom, runtime_stds)):
        axis.text(
            index, total + std, f"{total:.1f}", ha="center", va="bottom"
        )
    axis.set_ylim(0.0, max(bottom + runtime_stds) * 1.16)
    fig.tight_layout()
    save_figure(fig, output_dir, "group2_timing_breakdown")


def first_patience_hit(
    metric_values: np.ndarray, threshold: float, patience: int
) -> Optional[int]:
    for index in range(len(metric_values) - patience + 1):
        window = metric_values[index:index + patience]
        if np.all(np.isfinite(window)) and np.all(window <= threshold):
            return index
    return None


def build_tta_rows(
    runs: Mapping[str, Sequence[Run]]
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for key in METHOD_ORDER:
        for run in runs[key]:
            values = run.values["E_common_max"]
            for threshold in TTA_THRESHOLDS:
                hit = first_patience_hit(values, threshold, TTA_PATIENCE)
                rows.append(
                    {
                        "method": METHOD_STYLE[key]["label"],
                        "seed": run.seed,
                        "result_directory": run.directory.name,
                        "threshold": threshold,
                        "patience": TTA_PATIENCE,
                        "reached": hit is not None,
                        "first_equivalent_iteration": (
                            float(run.values["equivalent_iteration"][hit])
                            if hit is not None else ""
                        ),
                        "time_to_accuracy_seconds": (
                            float(run.values["effective_time"][hit])
                            if hit is not None else ""
                        ),
                        "effective_time_lower_bound_seconds": (
                            "" if hit is not None
                            else float(run.values["effective_time"][-1])
                        ),
                        "final_E_common": final_value(run, "E_common_max"),
                    }
                )
    return rows


def aggregate_tta_rows(
    tta_rows: Sequence[Mapping[str, object]]
) -> List[Dict[str, object]]:
    aggregate: List[Dict[str, object]] = []
    for key in METHOD_ORDER:
        label = METHOD_STYLE[key]["label"]
        for threshold in TTA_THRESHOLDS:
            subset = [
                row for row in tta_rows
                if row["method"] == label
                and float(row["threshold"]) == threshold
            ]
            reached = [row for row in subset if bool(row["reached"])]
            times = [float(row["time_to_accuracy_seconds"]) for row in reached]
            iterations = [
                float(row["first_equivalent_iteration"]) for row in reached
            ]
            lower_bounds = [
                float(row["effective_time_lower_bound_seconds"])
                for row in subset if not bool(row["reached"])
            ]
            time_mean, time_std, _ = sample_mean_std(times)
            iter_mean, iter_std, _ = sample_mean_std(iterations)
            lower_mean, _, _ = sample_mean_std(lower_bounds)
            aggregate.append(
                {
                    "method": label,
                    "threshold": threshold,
                    "patience": TTA_PATIENCE,
                    "n_seeds": len(subset),
                    "n_reached": len(reached),
                    "reach_rate": len(reached) / len(subset),
                    "time_to_accuracy_mean_seconds": time_mean,
                    "time_to_accuracy_std_seconds": time_std,
                    "first_equivalent_iteration_mean": iter_mean,
                    "first_equivalent_iteration_std": iter_std,
                    "non_reached_time_lower_bound_mean_seconds": lower_mean,
                }
            )
    return aggregate


def make_tta_figure(
    tta_aggregate: Sequence[Mapping[str, object]],
    output_dir: Path,
) -> None:
    x = np.arange(len(TTA_THRESHOLDS), dtype=float)
    labels = [
        r"$10^{-2}$", r"$10^{-3}$", r"$8\!\times\!10^{-4}$",
        r"$6\!\times\!10^{-4}$", r"$4\!\times\!10^{-4}$",
        r"$2\!\times\!10^{-4}$",
    ]
    fig, axis = plt.subplots(figsize=(9.2, 5.4))
    for key in METHOD_ORDER:
        style = METHOD_STYLE[key]
        method_rows = [
            row for row in tta_aggregate if row["method"] == style["label"]
        ]
        means = np.asarray(
            [float(row["time_to_accuracy_mean_seconds"]) for row in method_rows],
            dtype=float,
        )
        stds = np.asarray(
            [float(row["time_to_accuracy_std_seconds"]) for row in method_rows],
            dtype=float,
        )
        n_reached = np.asarray([int(row["n_reached"]) for row in method_rows])
        valid = np.isfinite(means) & (n_reached > 0)
        axis.errorbar(
            x[valid], means[valid], yerr=stds[valid],
            label=style["label"], color=style["color"],
            linestyle=style["linestyle"], linewidth=2.0,
            marker=style["marker"], markersize=6, capsize=3,
        )
        for index, row in enumerate(method_rows):
            reached_count = int(row["n_reached"])
            total_count = int(row["n_seeds"])
            if reached_count == 0:
                lower = float(row["non_reached_time_lower_bound_mean_seconds"])
                axis.scatter(
                    [x[index]], [lower], facecolors="none",
                    edgecolors=style["color"], marker="X", s=70,
                    linewidths=1.5,
                )
                axis.annotate(
                    "0/" + str(total_count), (x[index], lower), xytext=(4, 4),
                    textcoords="offset points", color=style["color"],
                    fontsize=8.5,
                )
            elif reached_count < total_count:
                axis.scatter(
                    [x[index]], [means[index]], facecolors="white",
                    edgecolors=style["color"], marker=style["marker"],
                    s=54, linewidths=1.5, zorder=4,
                )
                axis.annotate(
                    f"{reached_count}/{total_count}",
                    (x[index], means[index]), xytext=(4, 4),
                    textcoords="offset points", color=style["color"],
                    fontsize=8.5,
                )
    axis.set_yscale("log")
    axis.set_xticks(x)
    axis.set_xticklabels(labels)
    axis.set_xlabel(r"Target threshold for $E_{\mathrm{common}}$")
    axis.set_ylabel("Time to accuracy (s)")
    axis.set_title("Patience-confirmed time to accuracy (mean +/- 1 SD)")
    axis.legend(frameon=False, ncol=2)
    fig.text(
        0.5, 0.01,
        "Open markers are annotated reached/total; 0/n uses the mean runtime lower bound.",
        ha="center", fontsize=8.8, color="#555555",
    )
    fig.tight_layout(rect=(0.0, 0.035, 1.0, 1.0))
    save_figure(fig, output_dir, "group2_tta_Ecommon")


def parameter_count(run: Run) -> int:
    direct = run.metadata.get("parameter_count")
    if direct is not None:
        return int(direct)
    network = run.metadata.get("network", {})
    if not isinstance(network, Mapping):
        return 0

    def mlp_count(sizes: Sequence[int]) -> int:
        return sum(
            int(m) * int(n) + int(n)
            for m, n in zip(sizes[:-1], sizes[1:])
        )

    if run.key == "gfs_three_field":
        return mlp_count(network["u_sizes"]) + mlp_count(network["pt_sizes"])
    if run.key.startswith("coupled_"):
        sizes = network.get("sizes")
        if sizes is not None:
            return mlp_count(sizes)
    return mlp_count(network["uxi_sizes"]) + mlp_count(network["pt_sizes"])


SUMMARY_METRICS = (
    "completed_outer_iterations",
    "completed_equivalent_iterations",
    "effective_training_seconds",
    "u1_rel_l2",
    "u2_rel_l2",
    "xi_rel_l2",
    "p_rel_l2",
    "T_rel_l2",
    "u_mean_rel_l2",
    "pT_mean_rel_l2",
    "E_common_max",
    "E_4F_max",
    "assembly_seconds",
    "solve_seconds",
    "line_search_seconds",
    "device_peak_memory_gib",
    "process_peak_rss_gib",
    "final_outer_change_common",
    "final_u_outer_change",
    "final_pt_outer_change",
)


def build_summary_rows(
    runs: Mapping[str, Sequence[Run]]
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for key in METHOD_ORDER:
        for run in runs[key]:
            rows.append(
                {
                    "method": METHOD_STYLE[key]["label"],
                    "seed": run.seed,
                    "result_directory": run.directory.name,
                    "parameter_count": parameter_count(run),
                    "completed_outer_iterations": completed_outer_iterations(
                        run
                    ),
                    "completed_equivalent_iterations": final_value(
                        run, "equivalent_iteration"
                    ),
                    "effective_training_seconds": final_value(
                        run, "effective_time"
                    ),
                    "u1_rel_l2": final_value(run, "u1_rel_l2"),
                    "u2_rel_l2": final_value(run, "u2_rel_l2"),
                    "xi_rel_l2": final_value(run, "xi_rel_l2"),
                    "p_rel_l2": final_value(run, "p_rel_l2"),
                    "T_rel_l2": final_value(run, "T_rel_l2"),
                    "u_mean_rel_l2": final_value(run, "u_mean_rel_l2"),
                    "pT_mean_rel_l2": final_value(run, "pT_mean_rel_l2"),
                    "E_common_max": final_value(run, "E_common_max"),
                    "E_4F_max": final_value(run, "E_4F_max"),
                    "assembly_seconds": timing_value(run, "assembly_seconds"),
                    "solve_seconds": timing_value(run, "solve_seconds"),
                    "line_search_seconds": timing_value(
                        run, "line_search_seconds"
                    ),
                    "device_peak_memory_gib": memory_gib(
                        run, "device_peak_bytes_in_use"
                    ),
                    "process_peak_rss_gib": memory_gib(
                        run, "process_peak_rss_bytes"
                    ),
                    "final_outer_change_common": final_value(
                        run, "outer_change_common"
                    ),
                    "final_u_outer_change": final_value(
                        run, "u_outer_change"
                    ),
                    "final_pt_outer_change": final_value(
                        run, "pt_outer_change"
                    ),
                }
            )
    return rows


def aggregate_summary_rows(
    summary_rows: Sequence[Mapping[str, object]]
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for key in METHOD_ORDER:
        label = METHOD_STYLE[key]["label"]
        subset = [row for row in summary_rows if row["method"] == label]
        aggregate: Dict[str, object] = {
            "method": label,
            "n_seeds": len(subset),
            "seeds": ",".join(str(row["seed"]) for row in subset),
            "parameter_count": int(subset[0]["parameter_count"]),
        }
        for metric in SUMMARY_METRICS:
            mean, std, count = sample_mean_std(
                [float(row[metric]) for row in subset]
            )
            aggregate[f"{metric}_mean"] = mean
            aggregate[f"{metric}_std"] = std
            aggregate[f"{metric}_n"] = count
        rows.append(aggregate)
    return rows


def build_convergence_rows(
    runs: Mapping[str, Sequence[Run]]
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for key in METHOD_ORDER:
        for run in runs[key]:
            for index, equivalent_iteration in enumerate(
                run.values["equivalent_iteration"]
            ):
                rows.append(
                    {
                        "method": METHOD_STYLE[key]["label"],
                        "seed": run.seed,
                        "equivalent_iteration": float(equivalent_iteration),
                        "outer_change_common": float(
                            run.values["outer_change_common"][index]
                        ),
                        "u_outer_change": float(
                            run.values["u_outer_change"][index]
                        ),
                        "pt_outer_change": float(
                            run.values["pt_outer_change"][index]
                        ),
                    }
                )
    return rows


def aggregate_convergence_rows(
    runs: Mapping[str, Sequence[Run]]
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for key in METHOD_ORDER:
        aggregates = {
            metric: aggregate_curve(
                runs[key], "equivalent_iteration", metric
            )
            for metric in (
                "outer_change_common", "u_outer_change", "pt_outer_change"
            )
        }
        grid = aggregates["outer_change_common"][0]
        for index, equivalent_iteration in enumerate(grid):
            row: Dict[str, object] = {
                "method": METHOD_STYLE[key]["label"],
                "n_seeds": len(runs[key]),
                "equivalent_iteration": float(equivalent_iteration),
            }
            for metric, (_, mean, std) in aggregates.items():
                row[f"{metric}_mean"] = (
                    float(mean[index]) if index < mean.size else math.nan
                )
                row[f"{metric}_std"] = (
                    float(std[index]) if index < std.size else math.nan
                )
            rows.append(row)
    return rows


def formatted_stat(
    rows: Sequence[Mapping[str, object]], metric: str, style: str
) -> str:
    mean, std, count = sample_mean_std([float(row[metric]) for row in rows])
    if count == 0:
        return "--"
    if style == "scientific":
        return f"{mean:.3e} ± {std:.2e}"
    if style == "seconds":
        return f"{mean:.2f} ± {std:.2f}"
    if style == "memory":
        return f"{mean:.3f} ± {std:.3f}"
    if style == "iterations":
        return f"{mean:.1f} ± {std:.1f}"
    raise ValueError(f"Unknown table format: {style}")


def build_publication_tables(
    summary_rows: Sequence[Mapping[str, object]],
    tta_aggregate: Sequence[Mapping[str, object]],
) -> Tuple[
    List[Dict[str, object]],
    List[Dict[str, object]],
    List[Dict[str, object]],
]:
    accuracy_rows: List[Dict[str, object]] = []
    efficiency_rows: List[Dict[str, object]] = []
    tta_table_rows: List[Dict[str, object]] = []
    threshold_columns = (
        "TTA_1e-2_s", "TTA_1e-3_s", "TTA_8e-4_s",
        "TTA_6e-4_s", "TTA_4e-4_s", "TTA_2e-4_s",
    )

    for key in METHOD_ORDER:
        label = METHOD_STYLE[key]["label"]
        subset = [row for row in summary_rows if row["method"] == label]
        accuracy_rows.append(
            {
                "Method": label,
                "Seeds": len(subset),
                "Parameters": int(subset[0]["parameter_count"]),
                "E_u1": formatted_stat(subset, "u1_rel_l2", "scientific"),
                "E_u2": formatted_stat(subset, "u2_rel_l2", "scientific"),
                "E_xi": formatted_stat(subset, "xi_rel_l2", "scientific"),
                "E_p": formatted_stat(subset, "p_rel_l2", "scientific"),
                "E_T": formatted_stat(subset, "T_rel_l2", "scientific"),
                "E_common": formatted_stat(
                    subset, "E_common_max", "scientific"
                ),
                "E_4F": formatted_stat(subset, "E_4F_max", "scientific"),
            }
        )
        efficiency_rows.append(
            {
                "Method": label,
                "Seeds": len(subset),
                "Effective_time_s": formatted_stat(
                    subset, "effective_training_seconds", "seconds"
                ),
                "Peak_GPU_memory_GiB": formatted_stat(
                    subset, "device_peak_memory_gib", "memory"
                ),
                "Peak_RSS_GiB": formatted_stat(
                    subset, "process_peak_rss_gib", "memory"
                ),
                "Outer_iterations": formatted_stat(
                    subset, "completed_outer_iterations", "iterations"
                ),
                "Final_outer_change": formatted_stat(
                    subset, "final_outer_change_common", "scientific"
                ),
            }
        )

        method_tta = [
            row for row in tta_aggregate if row["method"] == label
        ]
        tta_row: Dict[str, object] = {"Method": label}
        for column_name, row in zip(threshold_columns, method_tta):
            reached = int(row["n_reached"])
            total = int(row["n_seeds"])
            if reached == 0:
                value = f"NR (0/{total})"
            else:
                mean = float(row["time_to_accuracy_mean_seconds"])
                std = float(row["time_to_accuracy_std_seconds"])
                value = f"{mean:.2f} ± {std:.2f} ({reached}/{total})"
            tta_row[column_name] = value
        tta_table_rows.append(tta_row)

    return accuracy_rows, efficiency_rows, tta_table_rows


def write_markdown_tables(
    path: Path,
    tables: Sequence[Tuple[str, Sequence[Mapping[str, object]]]],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# THP Group-2 publication tables\n\n")
        handle.write(
            "Values are paired-seed mean ± one sample standard deviation. "
            "TTA cells also report reached/total seeds.\n\n"
        )
        for title, rows in tables:
            handle.write(f"## {title}\n\n")
            if not rows:
                handle.write("No data.\n\n")
                continue
            columns = list(rows[0].keys())
            handle.write("| " + " | ".join(columns) + " |\n")
            handle.write("| " + " | ".join("---" for _ in columns) + " |\n")
            for row in rows:
                handle.write(
                    "| " + " | ".join(str(row[column]) for column in columns)
                    + " |\n"
                )
            handle.write("\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results_root", type=Path, nargs="?", default=Path(__file__).resolve().parent / "results",
        help="Directory containing Group-2 run directories.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Defaults to archive_summary beside this script.",
    )
    parser.add_argument(
        "--batch-tag", default=None,
        help="Only include directories named group2_BATCH_TAG_seed...",
    )
    parser.add_argument(
        "--seeds", default=None,
        help="Comma-separated paired seeds. Missing method/seed pairs are errors.",
    )
    args = parser.parse_args()

    results_root = args.results_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else Path(__file__).resolve().parent / "archive_summary"
    )
    # Reports must not modify any frozen or training run directory.
    if output_dir == results_root or results_root in output_dir.parents:
        raise ValueError("Choose --output-dir outside the input result directory.")
    protected = Path(__file__).resolve().parents[1].glob("group*/results")
    if any(output_dir == path.resolve() or path.resolve() in output_dir.parents
           for path in protected):
        raise ValueError("Choose a report output directory outside all frozen results.")
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_style()
    requested_seeds = parse_seed_csv(args.seeds)
    runs = discover_runs(results_root, args.batch_tag, requested_seeds)
    summary_rows = build_summary_rows(runs)
    summary_aggregate = aggregate_summary_rows(summary_rows)
    tta_rows = build_tta_rows(runs)
    tta_aggregate = aggregate_tta_rows(tta_rows)
    convergence_rows = build_convergence_rows(runs)
    convergence_aggregate = aggregate_convergence_rows(runs)
    accuracy_table, efficiency_table, tta_table = build_publication_tables(
        summary_rows, tta_aggregate
    )

    make_component_figure(runs, output_dir)
    make_time_figure(runs, output_dir)
    make_outer_convergence_figure(runs, output_dir)
    make_final_summary_figure(runs, output_dir)
    make_timing_breakdown_figure(runs, output_dir)
    make_tta_figure(tta_aggregate, output_dir)
    write_csv(output_dir / "group2_final_summary.csv", summary_rows)
    write_csv(
        output_dir / "group2_final_summary_mean_std.csv", summary_aggregate
    )
    write_csv(output_dir / "group2_tta_Ecommon.csv", tta_rows)
    write_csv(
        output_dir / "group2_tta_Ecommon_mean_std.csv", tta_aggregate
    )
    write_csv(
        output_dir / "group2_outer_convergence_per_seed.csv",
        convergence_rows,
    )
    write_csv(
        output_dir / "group2_outer_convergence_mean_std.csv",
        convergence_aggregate,
    )
    write_csv(
        output_dir / "group2_publication_accuracy_table.csv", accuracy_table
    )
    write_csv(
        output_dir / "group2_publication_efficiency_table.csv",
        efficiency_table,
    )
    write_csv(
        output_dir / "group2_publication_tta_table.csv", tta_table
    )
    write_markdown_tables(
        output_dir / "group2_publication_tables.md",
        (
            ("Final accuracy", accuracy_table),
            ("Efficiency and resources", efficiency_table),
            ("Time to accuracy", tta_table),
        ),
    )

    manifest = {
        "results_root": str(results_root),
        "batch_tag": args.batch_tag,
        "paired_seeds": [run.seed for run in runs[METHOD_ORDER[0]]],
        "selected_runs": {
            key: [str(run.directory) for run in method_runs]
            for key, method_runs in runs.items()
        },
        "curve_aggregation": (
            "geometric mean with +/- one sample standard deviation in "
            "log10(error); time curves aligned by log-error interpolation "
            "over the common effective-time interval"
        ),
        "curve_band_limits": (
            "10**(mean(log10(values)) +/- sample_std(log10(values)))"
        ),
        "bar_aggregation": "arithmetic mean +/- one sample standard deviation",
        "equivalent_iteration": "coupled round*20; GFS outer iteration*20",
        "time_metric": "effective training seconds; JIT/evaluation excluded",
        "tta_metric": "E_common_max",
        "tta_thresholds": list(TTA_THRESHOLDS),
        "tta_patience": TTA_PATIENCE,
        "tta_partial_reach_rule": (
            "mean/std are conditional on reached seeds; n_reached/n_seeds "
            "is always reported"
        ),
        "memory_metric": (
            "device_peak_bytes_in_use and process_peak_rss_bytes converted "
            "to GiB"
        ),
        "outer_convergence_metric": (
            "coupled prediction_change; GFS max(u_outer_change, "
            "pt_outer_change), with both GFS components also reported"
        ),
    }
    with (output_dir / "group2_plot_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    print(f"Group-2 comparison figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
