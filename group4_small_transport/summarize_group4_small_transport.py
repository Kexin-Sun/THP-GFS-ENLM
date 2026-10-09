"""Validate and tabulate the five four-field THP Group-4 archive runs.

The default cases are the four combinations of K,Theta in {0.1,1e-6} at
nu=0.3, followed by the single joint-limit case
nu=0.4999999, K=Theta=1e-6. No figures are created.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


METHOD_KEY = "gfs_four_field"
METHOD_LABEL = "4F-GFS-JS-EIGH-1e-7"
DEFAULT_CASES = (
    (0.3, 0.1, 0.1),
    (0.3, 0.1, 1.0e-6),
    (0.3, 1.0e-6, 0.1),
    (0.3, 1.0e-6, 1.0e-6),
    (0.4999999, 1.0e-6, 1.0e-6),
)
CHECKPOINTS = (20, 30, 40, 50)
TTA_THRESHOLDS = (1.0e-2, 1.0e-3, 8.0e-4, 6.0e-4, 4.0e-4, 2.0e-4)
PROFILE_WINDOW_START = 41
SEED_PATTERN = re.compile(r"(?:^|_)seed(-?\d+)(?:_|$)", re.IGNORECASE)


@dataclass(frozen=True)
class Run:
    seed: int
    nu: float
    kappa: float
    theta: float
    case_index: int
    directory: Path
    metadata: Mapping[str, object]
    history: Sequence[Mapping[str, str]]
    fixed_budget: Sequence[Mapping[str, str]]
    tta: Sequence[Mapping[str, str]]
    profile: Sequence[Mapping[str, str]]


def read_csv(path: Path) -> List[Mapping[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def nested(mapping: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = mapping.get(key, {})
    return value if isinstance(value, Mapping) else {}


def as_float(value: object, default: float = math.nan) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        return float(text)
    except (TypeError, ValueError):
        return default


def as_int(value: object, default: int = -1) -> int:
    numeric = as_float(value)
    return int(numeric) if math.isfinite(numeric) else default


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1.0e-10, abs_tol=1.0e-13)


def parse_seeds(text: str) -> List[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one seed is required")
    return values


def parse_cases(text: str) -> List[Tuple[float, float, float]]:
    cases: List[Tuple[float, float, float]] = []
    for entry in text.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        parts = [part.strip() for part in entry.split(",")]
        if len(parts) != 3:
            raise ValueError(
                f"Invalid case '{entry}'; expected semicolon-separated nu,K,Theta"
            )
        case = tuple(float(part) for part in parts)
        if not 0.0 < case[0] < 0.5 or case[1] <= 0.0 or case[2] <= 0.0:
            raise ValueError(f"Invalid physical case {case}")
        if any(all(close(a, b) for a, b in zip(case, old)) for old in cases):
            raise ValueError(f"Duplicate physical case {case}")
        cases.append(case)
    if not cases:
        raise ValueError("At least one physical case is required")
    return cases


def parameter_count(sizes: object) -> int:
    if not isinstance(sizes, Sequence) or isinstance(sizes, (str, bytes)):
        return 0
    widths = [int(value) for value in sizes]
    return sum((left + 1) * right for left, right in zip(widths, widths[1:]))


def find_case_index(
    nu: float,
    kappa: float,
    theta: float,
    cases: Sequence[Tuple[float, float, float]],
) -> Optional[int]:
    actual = (nu, kappa, theta)
    for index, expected in enumerate(cases):
        if all(close(left, right) for left, right in zip(actual, expected)):
            return index
    return None


def load_run(directory: Path, cases: Sequence[Tuple[float, float, float]]) -> Run:
    required = (
        "metadata.json",
        "fixed_budget_accuracy.csv",
        "time_to_accuracy.csv",
        "engd_profile.csv",
    )
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Incomplete run {directory}: missing {', '.join(missing)}"
        )
    if not (directory / "training_history.csv").is_file() and not (
        directory / "training_history.npz"
    ).is_file():
        raise FileNotFoundError(
            f"Incomplete run {directory}: missing training history"
        )
    with (directory / "metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    physical = nested(metadata, "physical_parameters")
    seed = as_int(metadata.get("seed"))
    if seed < 0:
        match = SEED_PATTERN.search(directory.name)
        if match is None:
            raise ValueError(f"Cannot determine seed for {directory}")
        seed = int(match.group(1))
    nu = as_float(physical.get("nu"))
    kappa = as_float(physical.get("K"))
    theta = as_float(physical.get("Theta"))
    case_index = find_case_index(nu, kappa, theta, cases)
    if case_index is None:
        raise ValueError(f"Run does not match a requested case: {directory}")
    return Run(
        seed=seed,
        nu=nu,
        kappa=kappa,
        theta=theta,
        case_index=case_index,
        directory=directory,
        metadata=metadata,
        history=(
            read_csv(directory / "training_history.csv")
            if (directory / "training_history.csv").is_file()
            else []
        ),
        fixed_budget=read_csv(directory / "fixed_budget_accuracy.csv"),
        tta=read_csv(directory / "time_to_accuracy.csv"),
        profile=read_csv(directory / "engd_profile.csv"),
    )


def discover_runs(
    root: Path,
    batch_tag: str,
    seeds: Sequence[int],
    cases: Sequence[Tuple[float, float, float]],
) -> List[Run]:
    token = f"group4_{batch_tag}_nu"
    candidates: Dict[Tuple[int, int], List[Run]] = defaultdict(list)
    incomplete: List[str] = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        if token not in directory.name or "gfs_four_field" not in directory.name:
            continue
        try:
            run = load_run(directory, cases)
        except FileNotFoundError as error:
            incomplete.append(str(error))
            continue
        if run.seed in seeds:
            candidates[(run.seed, run.case_index)].append(run)

    selected: List[Run] = []
    missing: List[str] = []
    for case_index, case in enumerate(cases):
        for seed in seeds:
            found = candidates.get((seed, case_index), [])
            if not found:
                missing.append(
                    f"4F:nu={case[0]}:K={case[1]}:Theta={case[2]}:seed={seed}"
                )
                continue
            found.sort(key=lambda item: item.directory.stat().st_mtime)
            if len(found) > 1:
                print(
                    "warning: multiple complete matches; using latest: "
                    f"{found[-1].directory}"
                )
            selected.append(found[-1])
    if missing:
        details = "Missing complete Group-4 runs: " + ", ".join(missing)
        if incomplete:
            details += "\nIncomplete candidate directories:\n" + "\n".join(incomplete)
        raise FileNotFoundError(details)
    return selected


def require_close(run: Run, actual: object, expected: float, label: str) -> None:
    value = as_float(actual)
    if not close(value, expected):
        raise ValueError(
            f"{run.directory.name}: expected {label}={expected}, got {actual}"
        )


def validate_run(run: Run) -> None:
    metadata = run.metadata
    physical = nested(metadata, "physical_parameters")
    profiling = nested(metadata, "profiling")
    engd_u = nested(metadata, "engd_u")
    engd_pt = nested(metadata, "engd_pt")
    split = nested(metadata, "four_field_split")
    accuracy = nested(metadata, "accuracy_reporting")
    sampling = nested(metadata, "sampling")
    grid_shape = nested(sampling, "grid_shape")
    weights = nested(nested(metadata, "loss_weighting"), "weights")

    if metadata.get("method") != "THP-four-field-iterative-decoupling-ENGD":
        raise ValueError(f"{run.directory.name}: not the four-field production method")
    if metadata.get("exact_case") != "distinct":
        raise ValueError(f"{run.directory.name}: exact_case must be distinct")
    for name, expected in (
        ("E", 1.0),
        ("alpha", 0.1),
        ("beta", 0.1),
        ("a0", 0.2),
        ("b0", 0.1),
        ("c0", 0.2),
        ("t_final", 1.0),
        ("nu", run.nu),
        ("K", run.kappa),
        ("Theta", run.theta),
    ):
        require_close(run, physical.get(name), expected, name)
    require_close(run, profiling.get("gram_ridge"), 1.0e-7, "gram_ridge")
    require_close(run, profiling.get("relative_cutoff"), 0.0, "relative_cutoff")

    if profiling.get("solver") != "eigh-refined":
        raise ValueError(f"{run.directory.name}: solver must be eigh-refined")
    if profiling.get("jacobian_backend") != "pointwise":
        raise ValueError(f"{run.directory.name}: backend must be pointwise")
    for key, expected in (
        ("refinement_steps", 2),
        ("line_search_positive_levels", 31),
        ("point_block_u", 512),
        ("point_block_pt", 512),
    ):
        if as_int(profiling.get(key)) != expected:
            raise ValueError(f"{run.directory.name}: expected {key}={expected}")
    if not as_bool(profiling.get("fuse_gram_blocks")):
        raise ValueError(f"{run.directory.name}: fused Gram blocks must be enabled")
    if not as_bool(profiling.get("cache_line_search")):
        raise ValueError(f"{run.directory.name}: cached line search must be enabled")
    if as_int(engd_u.get("steps_per_gfs_iteration")) != 20:
        raise ValueError(f"{run.directory.name}: mechanics updates must be 20")
    if as_int(engd_pt.get("steps_per_gfs_iteration")) != 20:
        raise ValueError(f"{run.directory.name}: pT updates must be 20")
    if as_int(split.get("iterations_requested")) != 50:
        raise ValueError(f"{run.directory.name}: requested budget must be FS50")
    if as_int(split.get("iterations_completed")) != 50:
        raise ValueError(f"{run.directory.name}: run did not complete FS50")
    if as_bool(split.get("adaptive_stop_enabled")):
        raise ValueError(f"{run.directory.name}: early stopping was enabled")
    residual_mode = nested(split, "pt_pde_residual_metric").get("mode")
    # Older archived runs predate the modal-residual option; absence therefore
    # denotes the original raw equations. Current production runs record raw.
    if residual_mode not in (None, "raw"):
        raise ValueError(f"{run.directory.name}: pT residual mode must be raw")

    checkpoints = accuracy.get("fixed_budget_checkpoints")
    if not isinstance(checkpoints, Sequence) or [int(x) for x in checkpoints] != list(CHECKPOINTS):
        raise ValueError(
            f"{run.directory.name}: checkpoints must equal {list(CHECKPOINTS)}"
        )
    if run.history and as_int(run.history[-1].get("fs_iteration")) != 50:
        raise ValueError(f"{run.directory.name}: history must end at FS50")

    expected_grids = {
        "pde": [16, 16, 16],
        "ic": [22, 22],
        "boundary_per_side": [24, 17],
        "evaluation": [40, 40, 40],
    }
    for key, expected in expected_grids.items():
        actual = grid_shape.get(key)
        if not isinstance(actual, Sequence) or [int(x) for x in actual] != expected:
            raise ValueError(f"{run.directory.name}: unexpected sampling grid {key}")

    for block_name, expected_keys in (
        ("u", ("pde", "dirichlet", "neumann", "ic")),
        ("pt", ("pde", "dirichlet", "ic")),
    ):
        block = nested(weights, block_name)
        for key in expected_keys:
            require_close(run, block.get(key), 1.0, f"weight.{block_name}.{key}")


def fixed_row(run: Run, checkpoint: int) -> Mapping[str, str]:
    rows = [
        row
        for row in run.fixed_budget
        if as_int(row.get("requested_fs_iteration")) == checkpoint
    ]
    if len(rows) != 1 or not as_bool(rows[0].get("checkpoint_reached")):
        raise ValueError(
            f"{run.directory.name}: missing reached FS{checkpoint} record"
        )
    return rows[0]


def tta_record(run: Run, threshold: float) -> Mapping[str, str]:
    rows = [
        row
        for row in run.tta
        if row.get("metric_name") == "E_common_max"
        and close(as_float(row.get("threshold")), threshold)
    ]
    if len(rows) != 1:
        raise ValueError(
            f"{run.directory.name}: expected one E_common TTA row at {threshold}"
        )
    return rows[0]


def finite_median(values: Iterable[float]) -> float:
    retained = [value for value in values if math.isfinite(value)]
    return float(median(retained)) if retained else math.nan


def profile_diagnostics(run: Run, subproblem: str) -> Dict[str, object]:
    rows = [row for row in run.profile if row.get("subproblem") == subproblem]
    final_rows = [
        row
        for row in rows
        if PROFILE_WINDOW_START <= as_int(row.get("fs_iteration")) <= 50
    ]
    if not rows or not final_rows:
        raise ValueError(f"{run.directory.name}: incomplete {subproblem} profile")

    network = nested(run.metadata, "network")
    sizes_key = "pt_sizes" if subproblem == "pT" else "u_sizes"
    count = parameter_count(network.get(sizes_key))
    if count <= 0:
        raise ValueError(
            f"{run.directory.name}: cannot determine {subproblem} parameter count"
        )
    rank_median = finite_median(
        as_float(row.get("effective_rank")) for row in final_rows
    )
    accepted_all = sum(as_bool(row.get("accepted")) for row in rows)
    accepted_final = sum(as_bool(row.get("accepted")) for row in final_rows)
    return {
        "subproblem": subproblem,
        "parameter_count": count,
        "updates_attempted": len(rows),
        "updates_accepted": accepted_all,
        "acceptance_rate": accepted_all / len(rows),
        "final_window_updates_attempted": len(final_rows),
        "final_window_acceptance_rate": accepted_final / len(final_rows),
        "effective_rank_median_FS41_50": rank_median,
        "effective_rank_ratio_median_FS41_50": rank_median / count,
        "lambda_max_median_FS41_50": finite_median(
            as_float(row.get("lambda_max")) for row in final_rows
        ),
        "lambda_min_kept_median_FS41_50": finite_median(
            as_float(row.get("lambda_min_kept")) for row in final_rows
        ),
        "regularized_condition_median_FS41_50": finite_median(
            as_float(row.get("effective_condition")) for row in final_rows
        ),
        "solve_relative_residual_median_FS41_50": finite_median(
            as_float(row.get("solve_relative_residual")) for row in final_rows
        ),
    }


def value_tag(value: float) -> str:
    return f"{value:.10g}".replace(".", "p").replace("-", "m").replace("+", "p")


def transport_case(kappa: float, theta: float) -> str:
    k_small = kappa < 1.0e-3
    theta_small = theta < 1.0e-3
    if k_small and theta_small:
        return "both_small"
    if k_small:
        return "K_small"
    if theta_small:
        return "Theta_small"
    return "baseline"


def identity(run: Run, batch_tag: str) -> Dict[str, object]:
    return {
        "batch_tag": batch_tag,
        "run_name": run.directory.name,
        "case_index": run.case_index + 1,
        "case_id": (
            f"nu{value_tag(run.nu)}_K{value_tag(run.kappa)}_"
            f"Theta{value_tag(run.theta)}"
        ),
        "regime": "joint_limit" if run.nu > 0.49 else "small_transport",
        "method_key": METHOD_KEY,
        "method": METHOD_LABEL,
        "seed": run.seed,
        "nu": run.nu,
        "one_minus_2nu": 1.0 - 2.0 * run.nu,
        "K": run.kappa,
        "Theta": run.theta,
        "transport_case": transport_case(run.kappa, run.theta),
    }


def final_row(run: Run, batch_tag: str) -> Dict[str, object]:
    source = fixed_row(run, 50)
    row = identity(run, batch_tag)
    for key in (
        "pT_engd_updates_completed",
        "u_engd_updates_completed",
        "total_engd_updates_completed",
        "effective_training_seconds",
        "loss_pt",
        "loss_pt_coupled_pde",
        "loss_u",
        "u_rel_l2",
        "u1_rel_l2",
        "u2_rel_l2",
        "xi_rel_l2",
        "p_rel_l2",
        "T_rel_l2",
        "u_mean_rel_l2",
        "pT_mean_rel_l2",
        "E_common_max",
        "E_4F_max",
    ):
        row[key] = as_float(source.get(key))
    tta = tta_record(run, 1.0e-3)
    reached = as_bool(tta.get("reached"))
    row["TTA_Ecommon_1e-3_reached"] = reached
    row["TTA_Ecommon_1e-3_seconds"] = (
        as_float(tta.get("time_to_accuracy_seconds")) if reached else math.nan
    )
    row["TTA_Ecommon_1e-3_first_FS"] = (
        as_int(tta.get("first_fs_iteration")) if reached else -1
    )
    diagnostics = {
        item["subproblem"]: item
        for item in (
            profile_diagnostics(run, "pT"),
            profile_diagnostics(run, "u"),
        )
    }
    for subproblem, prefix in (("pT", "pT"), ("u", "mechanics")):
        item = diagnostics[subproblem]
        for key in (
            "updates_attempted",
            "updates_accepted",
            "acceptance_rate",
            "effective_rank_ratio_median_FS41_50",
            "regularized_condition_median_FS41_50",
            "solve_relative_residual_median_FS41_50",
        ):
            row[f"{prefix}_{key}"] = item[key]
    return row


def fixed_budget_rows(run: Run, batch_tag: str) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    for checkpoint in CHECKPOINTS:
        source = fixed_row(run, checkpoint)
        row = identity(run, batch_tag)
        row["fs_iteration"] = checkpoint
        for key in (
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
            "E_4F_max",
        ):
            row[key] = as_float(source.get(key))
        output.append(row)
    return output


def all_tta_rows(run: Run, batch_tag: str) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    for threshold in TTA_THRESHOLDS:
        source = tta_record(run, threshold)
        reached = as_bool(source.get("reached"))
        row = identity(run, batch_tag)
        row.update(
            {
                "metric": "E_common_max",
                "threshold": threshold,
                "patience": as_int(source.get("patience")),
                "reached": reached,
                "first_fs_iteration": (
                    as_int(source.get("first_fs_iteration")) if reached else -1
                ),
                "confirmation_fs_iteration": (
                    as_int(source.get("confirmation_fs_iteration")) if reached else -1
                ),
                "time_to_accuracy_seconds": (
                    as_float(source.get("time_to_accuracy_seconds"))
                    if reached
                    else math.nan
                ),
                "effective_training_lower_bound_seconds": as_float(
                    source.get("effective_training_lower_bound_seconds")
                ),
                "final_metric_value": as_float(source.get("final_metric_value")),
            }
        )
        output.append(row)
    return output


def diagnostic_rows(run: Run, batch_tag: str) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    for subproblem in ("pT", "u"):
        row = identity(run, batch_tag)
        row.update(profile_diagnostics(run, subproblem))
        output.append(row)
    return output


def mean_std_rows(final_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[int, List[Mapping[str, object]]] = defaultdict(list)
    for row in final_rows:
        groups[as_int(row["case_index"])].append(row)
    numeric_keys = (
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
        "TTA_Ecommon_1e-3_seconds",
        "mechanics_acceptance_rate",
        "pT_acceptance_rate",
        "mechanics_effective_rank_ratio_median_FS41_50",
        "pT_effective_rank_ratio_median_FS41_50",
        "mechanics_regularized_condition_median_FS41_50",
        "pT_regularized_condition_median_FS41_50",
    )
    output: List[Dict[str, object]] = []
    for case_index in sorted(groups):
        members = groups[case_index]
        first = members[0]
        item = {
            key: first[key]
            for key in (
                "case_index",
                "case_id",
                "regime",
                "method_key",
                "method",
                "nu",
                "one_minus_2nu",
                "K",
                "Theta",
                "transport_case",
            )
        }
        item["n_seeds"] = len({member["seed"] for member in members})
        for key in numeric_keys:
            values = [as_float(member.get(key)) for member in members]
            values = [value for value in values if math.isfinite(value)]
            mean_value = sum(values) / len(values) if values else math.nan
            item[f"{key}_mean"] = mean_value
            item[f"{key}_std"] = (
                math.sqrt(
                    sum((value - mean_value) ** 2 for value in values)
                    / (len(values) - 1)
                )
                if len(values) > 1
                else (0.0 if values else math.nan)
            )
        output.append(item)
    return output


def robustness_rows(final_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    by_seed_case = {
        (as_int(row["seed"]), as_int(row["case_index"])): row for row in final_rows
    }
    output: List[Dict[str, object]] = []
    metrics = (
        "u1_rel_l2",
        "u2_rel_l2",
        "xi_rel_l2",
        "p_rel_l2",
        "T_rel_l2",
        "E_common_max",
        "effective_training_seconds",
    )
    for row in final_rows:
        case_index = as_int(row["case_index"])
        reference_index = 4 if case_index == 5 else 1
        reference = by_seed_case.get((as_int(row["seed"]), reference_index))
        if reference is None:
            raise ValueError(
                f"Missing reference case {reference_index} for seed {row['seed']}"
            )
        item = {
            key: row[key]
            for key in (
                "batch_tag",
                "run_name",
                "case_index",
                "case_id",
                "regime",
                "method",
                "seed",
                "nu",
                "K",
                "Theta",
                "transport_case",
            )
        }
        item["reference_case_index"] = reference_index
        item["reference_definition"] = (
            "same_transport_nu0p3"
            if case_index == 5
            else "nu0p3_K0p1_Theta0p1"
        )
        for metric in metrics:
            numerator = as_float(row.get(metric))
            denominator = as_float(reference.get(metric))
            item[f"{metric}_ratio_to_reference"] = (
                numerator / denominator
                if math.isfinite(numerator) and denominator > 0.0
                else math.nan
            )
        output.append(item)
    return output


def continuation_rows(runs: Sequence[Run], batch_tag: str) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    for run in runs:
        e40 = as_float(fixed_row(run, 40).get("E_common_max"))
        e50 = as_float(fixed_row(run, 50).get("E_common_max"))
        ratio = e50 / e40 if e40 > 0.0 else math.nan
        row = identity(run, batch_tag)
        row.update(
            {
                "E_common_FS40": e40,
                "E_common_FS50": e50,
                "FS50_over_FS40": ratio,
                "extend_to_FS100": math.isfinite(ratio) and ratio <= 0.90,
            }
        )
        output.append(row)
    return output


def fmt(value: object, kind: str = "scientific") -> str:
    numeric = as_float(value)
    if not math.isfinite(numeric):
        return "—"
    if kind == "seconds":
        return f"{numeric:.2f}"
    if kind == "percent":
        return f"{100.0 * numeric:.1f}%"
    return f"{numeric:.3e}"


def fmt_parameter(value: object) -> str:
    numeric = as_float(value)
    if not math.isfinite(numeric):
        return "—"
    return f"{numeric:.10f}".rstrip("0").rstrip(".")


def markdown_report(
    path: Path,
    final_rows: Sequence[Mapping[str, object]],
    diagnostics: Sequence[Mapping[str, object]],
    continuation: Sequence[Mapping[str, object]],
) -> None:
    lines = [
        "# THP Group 4: five-case four-field robustness table",
        "",
        "All runs use 4F-GFS-JS-EIGH-1e-7, FS50, 20 updates per "
        "subproblem per sweep, and the raw p-T residual.",
        "",
        "## Main results",
        "",
        "| Case | nu | K | Theta | u1 | u2 | p | T | xi | Ecommon max | TTA@1e-3/s | Time/s |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in final_rows:
        tta = (
            fmt(row["TTA_Ecommon_1e-3_seconds"], "seconds")
            if bool(row["TTA_Ecommon_1e-3_reached"])
            else "NR"
        )
        lines.append(
            "| "
            + " | ".join(
                (
                    str(as_int(row["case_index"])),
                    fmt_parameter(row["nu"]),
                    fmt(row["K"]),
                    fmt(row["Theta"]),
                    fmt(row["u1_rel_l2"]),
                    fmt(row["u2_rel_l2"]),
                    fmt(row["p_rel_l2"]),
                    fmt(row["T_rel_l2"]),
                    fmt(row["xi_rel_l2"]),
                    fmt(row["E_common_max"]),
                    tta,
                    fmt(row["effective_training_seconds"], "seconds"),
                )
            )
            + " |"
        )

    diagnostic_by_case: Dict[int, Dict[str, Mapping[str, object]]] = defaultdict(dict)
    for row in diagnostics:
        diagnostic_by_case[as_int(row["case_index"])][str(row["subproblem"])] = row
    lines.extend(
        [
            "",
            "## Optimizer diagnostics",
            "",
            "All spectral statistics are medians over FS41--FS50.",
            "",
            "| Case | Mechanics cond. | pT cond. | Mechanics rank | pT rank | Mechanics accept | pT accept |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in final_rows:
        case_index = as_int(row["case_index"])
        u = diagnostic_by_case[case_index]["u"]
        pt = diagnostic_by_case[case_index]["pT"]
        lines.append(
            "| "
            + " | ".join(
                (
                    str(case_index),
                    fmt(u["regularized_condition_median_FS41_50"]),
                    fmt(pt["regularized_condition_median_FS41_50"]),
                    fmt(u["effective_rank_ratio_median_FS41_50"], "percent"),
                    fmt(pt["effective_rank_ratio_median_FS41_50"], "percent"),
                    fmt(u["acceptance_rate"], "percent"),
                    fmt(pt["acceptance_rate"], "percent"),
                )
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## FS100 supplement decision",
            "",
            "Trigger: Ecommon(FS50)/Ecommon(FS40) <= 0.90.",
            "",
            "| Case | FS40 | FS50 | Ratio | Extend to FS100? |",
            "|---:|---:|---:|---:|:---:|",
        ]
    )
    for row in continuation:
        lines.append(
            "| "
            + " | ".join(
                (
                    str(as_int(row["case_index"])),
                    fmt(row["E_common_FS40"]),
                    fmt(row["E_common_FS50"]),
                    fmt(row["FS50_over_FS40"]),
                    "yes" if bool(row["extend_to_FS100"]) else "no",
                )
            )
            + " |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    default_case_text = ";".join(
        ",".join(f"{value:.10g}" for value in case) for case in DEFAULT_CASES
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root", type=Path, nargs="?", default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--batch-tag", required=True)
    parser.add_argument("--seeds", default="12345")
    parser.add_argument("--cases", default=default_case_text)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    cases = parse_cases(args.cases)
    result_root = args.result_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else Path(__file__).resolve().parent / "archive_summary"
    )

    runs = discover_runs(result_root, args.batch_tag, seeds, cases)
    for run in runs:
        validate_run(run)
    runs = sorted(runs, key=lambda run: (run.case_index, run.seed))

    final_rows = [final_row(run, args.batch_tag) for run in runs]
    fixed_rows = [
        row for run in runs for row in fixed_budget_rows(run, args.batch_tag)
    ]
    tta_rows = [row for run in runs for row in all_tta_rows(run, args.batch_tag)]
    diagnostics = [
        row for run in runs for row in diagnostic_rows(run, args.batch_tag)
    ]
    robustness = robustness_rows(final_rows)
    continuation = continuation_rows(runs, args.batch_tag)
    mean_std = mean_std_rows(final_rows)

    # Reports must not modify any frozen or training run directory.
    if output_dir == result_root or result_root in output_dir.parents:
        raise ValueError("Choose --output-dir outside the input result directory.")
    protected = Path(__file__).resolve().parents[1].glob("group*/results")
    if any(output_dir == path.resolve() or path.resolve() in output_dir.parents
           for path in protected):
        raise ValueError("Choose a report output directory outside all frozen results.")
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "group4_final_summary.csv": final_rows,
        "group4_final_summary_mean_std.csv": mean_std,
        "group4_fixed_budget.csv": fixed_rows,
        "group4_tta_Ecommon.csv": tta_rows,
        "group4_gram_diagnostics.csv": diagnostics,
        "group4_robustness_ratios.csv": robustness,
        "group4_FS100_decision.csv": continuation,
        "group4_publication_main_table.csv": final_rows,
    }
    for filename, rows in outputs.items():
        write_csv(output_dir / filename, rows)
    markdown_report(
        output_dir / "group4_publication_tables.md",
        final_rows,
        diagnostics,
        continuation,
    )

    manifest = {
        "group": 4,
        "description": "Five-case four-field robustness archive",
        "batch_tag": args.batch_tag,
        "result_root": str(result_root),
        "n_runs": len(runs),
        "expected_runs": len(seeds) * len(cases),
        "seeds": seeds,
        "cases": [
            {"case_index": index + 1, "nu": case[0], "K": case[1], "Theta": case[2]}
            for index, case in enumerate(cases)
        ],
        "method": METHOD_LABEL,
        "fixed_budget": 50,
        "checkpoints": list(CHECKPOINTS),
        "tables_only": True,
        "files": list(outputs) + ["group4_publication_tables.md"],
    }
    (output_dir / "group4_report_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Validated {len(runs)} complete four-field Group-4 runs.")
    print(f"Wrote table-only report to {output_dir}")


if __name__ == "__main__":
    main()
