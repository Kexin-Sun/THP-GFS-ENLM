"""Group 1: optimizer and Gram-assembly comparisons under identical 4F-GFS.

The physics, exact solution, fixed sampling, networks, and error evaluation
come from the root thp-four-field-core.py. Each of 100 sweeps first updates p,T with
u,xi frozen, then updates u,xi with p,T frozen. Errors and TTA are recorded
only after both subproblems finish. Seven methods reproduce Tables 1--3:
adam, adam_lbfgs, engd_fjls, js_lstsq0, js_eigh0, js_eigh1e7, full_eigh1e7.
The last method isolates full versus streamed assembly at fixed EIGH solver
and damping 1e-7. full_eigh0 is retained as an optional historical diagnostic."""

from __future__ import annotations

import csv
import json
import math
import os
import resource
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import importlib.util
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("THP_RESULT_ROOT", str(SCRIPT_DIR / "runs"))
CORE_PATH = Path(os.environ.get(
    "THP_FOUR_FIELD_CORE", str(SCRIPT_DIR.parent / "thp-four-field-core.py")
)).resolve()
if not CORE_PATH.is_file():
    raise FileNotFoundError(f"Missing canonical root core: {CORE_PATH}")
_spec = importlib.util.spec_from_file_location("thp_four_field_core", CORE_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load canonical core: {CORE_PATH}")
core = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = core
_spec.loader.exec_module(core)

import jax
import jax.numpy as jnp
import numpy as np


Array = jax.Array


# =============================================================================
# 1. Fixed protocol and configuration guards
# =============================================================================

METHOD = os.environ.get("THP_GROUP1_METHOD", "js_eigh1e7").strip().lower()

METHOD_PRESETS: Dict[str, Dict[str, object]] = {
    "adam": {
        "label": "4F-GFS-Adam",
        "family": "first-order",
        "backend": "none",
        "solver": "adam",
        "ridge": 0.0,
    },
    "adam_lbfgs": {
        "label": "4F-GFS-Adam+L-BFGS",
        "family": "first/second-order",
        "backend": "none",
        "solver": "adam+lbfgs",
        "ridge": 0.0,
    },
    "engd_fjls": {
        "label": "4F-GFS-ENGD-FJLS-0",
        "family": "ENGD",
        "backend": "full-blockwise-local-jacrev",
        "solver": "lstsq",
        "ridge": 0.0,
    },
    "js_lstsq0": {
        "label": "4F-GFS-JS-ENGD-LSTSQ-0",
        "family": "ENGD",
        "backend": "pointwise",
        "solver": "lstsq",
        "ridge": 0.0,
    },
    "js_eigh0": {
        "label": "4F-GFS-JS-ENGD-EIGH-0",
        "family": "ENGD",
        "backend": "pointwise",
        "solver": "eigh-refined",
        "ridge": 0.0,
    },
    "full_eigh1e7": {
        "label": "4F-GFS-ENLM-FULLJ-EIGH-1e-7",
        "family": "ENLM",
        "backend": "full-blockwise-local-jacrev",
        "solver": "eigh-refined",
        "ridge": 1.0e-7,
    },
    "full_eigh0": {
        "label": "4F-GFS-ENGD-FULLJ-EIGH-0",
        "family": "ENGD",
        "backend": "full-blockwise-local-jacrev",
        "solver": "eigh-refined",
        "ridge": 0.0,
    },
    "js_eigh1e7": {
        "label": "4F-GFS-ENLM-S-EIGH-1e-7",
        "family": "ENLM",
        "backend": "pointwise",
        "solver": "eigh-refined",
        "ridge": 1.0e-7,
    },
}

if METHOD not in METHOD_PRESETS:
    raise ValueError(
        "THP_GROUP1_METHOD must be one of "
        f"{sorted(METHOD_PRESETS)}, but received {METHOD!r}."
    )

PRESET = METHOD_PRESETS[METHOD]
METHOD_LABEL = str(PRESET["label"])
METHOD_FAMILY = str(PRESET["family"])
JACOBIAN_BACKEND = str(PRESET["backend"])
LINEAR_SOLVER = str(PRESET["solver"])
GRAM_RIDGE = float(PRESET["ridge"])

ALLOW_NONSTANDARD = (
    os.environ.get("THP_ALLOW_NONSTANDARD_GROUP1", "0") == "1"
)
EXPECTED_PHYSICS = {
    "nu": 0.3,
    "a0": 0.2,
    "b0": 0.1,
    "c0": 0.2,
    "kappa": 0.1,
    "theta": 0.1,
}
ACTUAL_PHYSICS = {
    "nu": core.NU,
    "a0": core.A0,
    "b0": core.B0,
    "c0": core.C0,
    "kappa": core.KAPPA,
    "theta": core.THETA,
}

if not ALLOW_NONSTANDARD:
    for name, expected in EXPECTED_PHYSICS.items():
        actual = ACTUAL_PHYSICS[name]
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-14):
            raise ValueError(
                f"Group-1 requires {name}={expected}, but received {actual}. "
                "Set THP_ALLOW_NONSTANDARD_GROUP1=1 only for a labeled "
                "additional experiment."
            )
    if core.PT_RESIDUAL_MODE != "raw":
        raise ValueError(
            "Group-1 requires THP_PT_RESIDUAL_MODE=raw. Modal residual "
            "preconditioning belongs to the singular-storage experiment."
        )

FS_ITERATIONS = int(os.environ.get("THP_GROUP1_FS_ITERATIONS", "100"))
ENGD_STEPS_PER_BLOCK = int(
    os.environ.get("THP_GROUP1_ENGD_STEPS_PER_BLOCK", "20")
)
ADAM_STEPS_PER_BLOCK = int(
    os.environ.get("THP_GROUP1_ADAM_STEPS_PER_BLOCK", "1000")
)
ADAM_LEARNING_RATE = float(
    os.environ.get("THP_GROUP1_ADAM_LR", "1e-3")
)
ADAM_LBFGS_SWITCH_FS = int(
    os.environ.get("THP_GROUP1_ADAM_LBFGS_SWITCH_FS", "30")
)
LBFGS_MAXITER_PER_BLOCK = int(
    os.environ.get("THP_GROUP1_LBFGS_MAXITER_PER_BLOCK", "200")
)
LBFGS_MAXFUN_PER_BLOCK = int(
    os.environ.get("THP_GROUP1_LBFGS_MAXFUN_PER_BLOCK", "10000")
)
LBFGS_FTOL = float(os.environ.get("THP_GROUP1_LBFGS_FTOL", "1e-15"))
LBFGS_GTOL = float(os.environ.get("THP_GROUP1_LBFGS_GTOL", "1e-10"))
LINE_SEARCH_LEVELS = int(os.environ.get("THP_LINE_SEARCH_LEVELS", "31"))
POINT_BLOCK_PT = int(os.environ.get("THP_POINT_JAC_BLOCK_PT", "512"))
POINT_BLOCK_U = int(os.environ.get("THP_POINT_JAC_BLOCK_U", "512"))
FULL_JAC_POINT_BLOCK_PT = int(
    os.environ.get("THP_FULL_JAC_POINT_BLOCK_PT", "512")
)
FULL_JAC_POINT_BLOCK_U = int(
    os.environ.get("THP_FULL_JAC_POINT_BLOCK_U", "512")
)
EIGH_RELATIVE_CUTOFF = float(
    os.environ.get("THP_EIGH_RELATIVE_CUTOFF", "0.0")
)
EIGH_REFINEMENT_STEPS = int(
    os.environ.get("THP_EIGH_REFINEMENT_STEPS", "2")
)
LSTSQ_RCOND = core._optional_float_from_env("THP_LSTSQ_RCOND")

TTA_PATIENCE = int(os.environ.get("THP_TTA_PATIENCE", "3"))
TTA_THRESHOLDS = tuple(
    float(value.strip())
    for value in os.environ.get(
        "THP_TIME_TO_ACCURACY_THRESHOLDS",
        "1e-2,1e-3,8e-4,6e-4,4e-4,2e-4,1e-4",
    ).split(",")
    if value.strip()
)
FIXED_FS_CHECKPOINTS = tuple(
    sorted(
        {
            int(value.strip())
            for value in os.environ.get(
                "THP_GROUP1_FIXED_FS_CHECKPOINTS", "0,50,100"
            ).split(",")
            if value.strip()
        }
    )
)
FIXED_TIME_BUDGETS = tuple(
    sorted(
        {
            float(value.strip())
            for value in os.environ.get(
                "THP_FIXED_TIME_BUDGETS_SECONDS",
                "10,30,60,120,300,600,1200,1800,3600",
            ).split(",")
            if value.strip()
        }
    )
)

if FS_ITERATIONS != 100 and not core.FAST_DEBUG:
    raise ValueError(
        "The frozen first-group protocol requires exactly 100 GFS outer "
        "iterations. THP_FAST_DEBUG=1 is the only automatic exception."
    )
if min(ENGD_STEPS_PER_BLOCK, ADAM_STEPS_PER_BLOCK) <= 0:
    raise ValueError("Inner optimizer budgets must be positive.")
if ADAM_LEARNING_RATE <= 0.0:
    raise ValueError("THP_GROUP1_ADAM_LR must be positive.")
if not 0 < ADAM_LBFGS_SWITCH_FS < FS_ITERATIONS:
    raise ValueError("Adam/L-BFGS switch must lie strictly inside the FS budget.")
if min(LBFGS_MAXITER_PER_BLOCK, LBFGS_MAXFUN_PER_BLOCK) <= 0:
    raise ValueError("L-BFGS budgets must be positive.")
if LINE_SEARCH_LEVELS != 31:
    raise ValueError("The frozen protocol requires the 31-level line search.")
if min(
    POINT_BLOCK_PT,
    POINT_BLOCK_U,
    FULL_JAC_POINT_BLOCK_PT,
    FULL_JAC_POINT_BLOCK_U,
) <= 0:
    raise ValueError("Jacobian point-block sizes must be positive.")
if TTA_PATIENCE <= 0:
    raise ValueError("THP_TTA_PATIENCE must be positive.")
if not TTA_THRESHOLDS or any(value <= 0.0 for value in TTA_THRESHOLDS):
    raise ValueError("TTA thresholds must be non-empty and positive.")

if core.FAST_DEBUG:
    FS_ITERATIONS = min(FS_ITERATIONS, 2)
    ENGD_STEPS_PER_BLOCK = min(ENGD_STEPS_PER_BLOCK, 1)
    ADAM_STEPS_PER_BLOCK = min(ADAM_STEPS_PER_BLOCK, 2)
    ADAM_LBFGS_SWITCH_FS = 1
    LBFGS_MAXITER_PER_BLOCK = min(LBFGS_MAXITER_PER_BLOCK, 2)
    LBFGS_MAXFUN_PER_BLOCK = min(LBFGS_MAXFUN_PER_BLOCK, 10)

# Historical variable name includes both undamped ENGD and damped ENLM.
IS_ENGD = METHOD_FAMILY in {"ENGD", "ENLM"}
IS_FULL_JACOBIAN = JACOBIAN_BACKEND == "full-blockwise-local-jacrev"
IS_POINTWISE = JACOBIAN_BACKEND == "pointwise"


# =============================================================================
# 2. Shared losses, cached Jacobians, and error metrics
# =============================================================================


@jax.jit
def pt_loss_cached(flat_pt: Array, pde_frozen: Array) -> Array:
    return core.half_squared_norm(
        core.residual_pt_cached_flat(flat_pt, pde_frozen)
    )


@jax.jit
def u_loss_cached(
    flat_u: Array,
    pde_frozen: Array,
    neu_frozen: Array,
) -> Array:
    return core.half_squared_norm(
        core.residual_u_cached_flat(flat_u, pde_frozen, neu_frozen)
    )


PT_LOSS_AND_GRAD = jax.jit(jax.value_and_grad(pt_loss_cached, argnums=0))
U_LOSS_AND_GRAD = jax.jit(jax.value_and_grad(u_loss_cached, argnums=0))

# Full and streamed assembly use identical frozen GFS residuals.
# Apply jacrev to each local residual, then vmap over points.
# Concatenate point batches before forming each full-block Gram matrix.
# Weights sqrt(weight*measure/N) are already included in local residuals.




@jax.jit
def pt_pde_block_residual(flat_pt: Array, pde_frozen: Array) -> Array:
    return jax.vmap(core.pt_pde_local_flat, in_axes=(None, 0, 0))(
        flat_pt, core.PDE_POINTS, pde_frozen
    ).reshape(-1)


@jax.jit
def pt_dir_block_residual(flat_pt: Array) -> Array:
    return jax.vmap(core.pt_dir_local_flat, in_axes=(None, 0, 0))(
        flat_pt, core.PT_DIR_POINTS, core.PT_DIR_TARGET
    ).reshape(-1)


@jax.jit
def pt_ic_block_residual(flat_pt: Array) -> Array:
    return jax.vmap(core.pt_ic_local_flat, in_axes=(None, 0, 0))(
        flat_pt, core.IC_POINTS, core.PT_IC_TARGET
    ).reshape(-1)


@jax.jit
def u_pde_block_residual(flat_u: Array, pde_frozen: Array) -> Array:
    return jax.vmap(core.u_pde_local_flat, in_axes=(None, 0, 0))(
        flat_u, core.PDE_POINTS, pde_frozen
    ).reshape(-1)


@jax.jit
def u_dir_block_residual(flat_u: Array) -> Array:
    return jax.vmap(core.u_dir_local_flat, in_axes=(None, 0, 0))(
        flat_u, core.U_DIR_POINTS, core.U_DIR_TARGET
    ).reshape(-1)


@jax.jit
def u_neu_block_residual(flat_u: Array, neu_frozen: Array) -> Array:
    return jax.vmap(core.u_neu_local_flat, in_axes=(None, 0, 0))(
        flat_u, core.U_NEU_POINTS, neu_frozen
    ).reshape(-1)


@jax.jit
def u_ic_block_residual(flat_u: Array) -> Array:
    return jax.vmap(core.u_ic_local_flat, in_axes=(None, 0, 0))(
        flat_u, core.IC_POINTS, core.U_IC_TARGET
    ).reshape(-1)


PT_PDE_LOCAL_JACOBIAN_BATCH = jax.jit(
    jax.vmap(
        jax.jacrev(core.pt_pde_local_flat, argnums=0),
        in_axes=(None, 0, 0),
    )
)
PT_DIR_LOCAL_JACOBIAN_BATCH = jax.jit(
    jax.vmap(
        jax.jacrev(core.pt_dir_local_flat, argnums=0),
        in_axes=(None, 0, 0),
    )
)
PT_IC_LOCAL_JACOBIAN_BATCH = jax.jit(
    jax.vmap(
        jax.jacrev(core.pt_ic_local_flat, argnums=0),
        in_axes=(None, 0, 0),
    )
)
U_PDE_LOCAL_JACOBIAN_BATCH = jax.jit(
    jax.vmap(
        jax.jacrev(core.u_pde_local_flat, argnums=0),
        in_axes=(None, 0, 0),
    )
)
U_DIR_LOCAL_JACOBIAN_BATCH = jax.jit(
    jax.vmap(
        jax.jacrev(core.u_dir_local_flat, argnums=0),
        in_axes=(None, 0, 0),
    )
)
U_NEU_LOCAL_JACOBIAN_BATCH = jax.jit(
    jax.vmap(
        jax.jacrev(core.u_neu_local_flat, argnums=0),
        in_axes=(None, 0, 0),
    )
)
U_IC_LOCAL_JACOBIAN_BATCH = jax.jit(
    jax.vmap(
        jax.jacrev(core.u_ic_local_flat, argnums=0),
        in_axes=(None, 0, 0),
    )
)


def materialize_complete_jacobian(
    local_jacobian_batch: Callable[[Array, Array, Array], Array],
    flat_params: Array,
    points: Array,
    auxiliary: Array,
    point_block_size: int,
) -> Array:
    """Construct point batches and concatenate the complete physical-block Jacobian.

    Batching limits automatic-differentiation workspace. All rows are
    materialized in their original sampling order before J.T@J; this is
    the full-Jacobian reference rather than streamed Gram accumulation."""

    number_of_points = int(points.shape[0])
    parameter_count = int(flat_params.size)
    jacobian_chunks: List[Array] = []
    for start in range(0, number_of_points, point_block_size):
        stop = min(start + point_block_size, number_of_points)
        local_jacobian = local_jacobian_batch(
            flat_params,
            points[start:stop],
            auxiliary[start:stop],
        )
        local_jacobian.block_until_ready()
        jacobian_chunks.append(
            local_jacobian.reshape((-1, parameter_count))
        )

    if not jacobian_chunks:
        return jnp.zeros((0, parameter_count), dtype=flat_params.dtype)
    if len(jacobian_chunks) == 1:
        return jacobian_chunks[0]
    complete_jacobian = jnp.concatenate(jacobian_chunks, axis=0)
    complete_jacobian.block_until_ready()
    return complete_jacobian


def jacobian_pt_pde_block(flat_pt: Array, pde_frozen: Array) -> Array:
    return materialize_complete_jacobian(
        PT_PDE_LOCAL_JACOBIAN_BATCH,
        flat_pt,
        core.PDE_POINTS,
        pde_frozen,
        FULL_JAC_POINT_BLOCK_PT,
    )


def jacobian_pt_dir_block(flat_pt: Array) -> Array:
    return materialize_complete_jacobian(
        PT_DIR_LOCAL_JACOBIAN_BATCH,
        flat_pt,
        core.PT_DIR_POINTS,
        core.PT_DIR_TARGET,
        FULL_JAC_POINT_BLOCK_PT,
    )


def jacobian_pt_ic_block(flat_pt: Array) -> Array:
    return materialize_complete_jacobian(
        PT_IC_LOCAL_JACOBIAN_BATCH,
        flat_pt,
        core.IC_POINTS,
        core.PT_IC_TARGET,
        FULL_JAC_POINT_BLOCK_PT,
    )


def jacobian_u_pde_block(flat_u: Array, pde_frozen: Array) -> Array:
    return materialize_complete_jacobian(
        U_PDE_LOCAL_JACOBIAN_BATCH,
        flat_u,
        core.PDE_POINTS,
        pde_frozen,
        FULL_JAC_POINT_BLOCK_U,
    )


def jacobian_u_dir_block(flat_u: Array) -> Array:
    return materialize_complete_jacobian(
        U_DIR_LOCAL_JACOBIAN_BATCH,
        flat_u,
        core.U_DIR_POINTS,
        core.U_DIR_TARGET,
        FULL_JAC_POINT_BLOCK_U,
    )


def jacobian_u_neu_block(flat_u: Array, neu_frozen: Array) -> Array:
    return materialize_complete_jacobian(
        U_NEU_LOCAL_JACOBIAN_BATCH,
        flat_u,
        core.U_NEU_POINTS,
        neu_frozen,
        FULL_JAC_POINT_BLOCK_U,
    )


def jacobian_u_ic_block(flat_u: Array) -> Array:
    return materialize_complete_jacobian(
        U_IC_LOCAL_JACOBIAN_BATCH,
        flat_u,
        core.IC_POINTS,
        core.U_IC_TARGET,
        FULL_JAC_POINT_BLOCK_U,
    )


def group_error_metrics(errors: Dict[str, float]) -> Dict[str, float]:
    u_components = np.asarray(
        [errors["u1_rel_l2"], errors["u2_rel_l2"]], dtype=float
    )
    pt_components = np.asarray(
        [errors["p_rel_l2"], errors["T_rel_l2"]], dtype=float
    )
    common = np.concatenate((u_components, pt_components))
    four_field = np.concatenate(
        (common, np.asarray([errors["xi_rel_l2"]], dtype=float))
    )
    return {
        "E_u_max": float(np.max(u_components)),
        "E_u_mean": float(np.mean(u_components)),
        "E_u_rms": float(np.sqrt(np.mean(u_components**2))),
        "E_pT_max": float(np.max(pt_components)),
        "E_pT_mean": float(np.mean(pt_components)),
        "E_pT_rms": float(np.sqrt(np.mean(pt_components**2))),
        "E_common_max": float(np.max(common)),
        "E_common_mean": float(np.mean(common)),
        "E_common_rms": float(np.sqrt(np.mean(common**2))),
        "E_4F_max": float(np.max(four_field)),
        "E_4F_mean": float(np.mean(four_field)),
        "E_4F_rms": float(np.sqrt(np.mean(four_field**2))),
    }


@jax.jit
def coupled_pt_pde_loss(flat_u: Array, flat_pt: Array) -> Array:
    residual = core.V_COUPLED_FLOW_HEAT_RES(
        core.UNRAVEL_PT(flat_pt), core.UNRAVEL_U(flat_u), core.PDE_POINTS
    )
    return (
        0.5
        * core.WEIGHTS_PT["pde"]
        * core.MEASURE_PDE
        * jnp.mean(jnp.sum(residual**2, axis=1))
    )


def process_memory_snapshot() -> Dict[str, object]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss_bytes = int(usage.ru_maxrss * 1024)
    snapshot: Dict[str, object] = {"process_peak_rss_bytes": rss_bytes}
    try:
        device = jax.devices()[0]
        snapshot["device_platform"] = device.platform
        snapshot["device_kind"] = device.device_kind
        stats = device.memory_stats() or {}
        for source, target in (
            ("bytes_in_use", "device_bytes_in_use"),
            ("peak_bytes_in_use", "device_peak_bytes_in_use"),
            ("bytes_limit", "device_bytes_limit"),
        ):
            if source in stats:
                snapshot[target] = int(stats[source])
    except Exception as error:
        snapshot["device_memory_stats_error"] = repr(error)
    return snapshot


# =============================================================================
# 3. Timing, history, and frozen subproblem contexts
# =============================================================================


@dataclass
class TimingLedger:
    context_seconds: float = 0.0
    optimizer_seconds: float = 0.0
    convergence_seconds: float = 0.0
    evaluation_excluded_seconds: float = 0.0
    jit_warmup_excluded_seconds: float = 0.0

    @property
    def effective_training_seconds(self) -> float:
        return (
            self.context_seconds
            + self.optimizer_seconds
            + self.convergence_seconds
        )


LEDGER = TimingLedger()
HISTORY_ROWS: List[Dict[str, object]] = []
PROFILE_ROWS: List[Dict[str, object]] = []


@dataclass
class AdamState:
    first_moment: Array
    second_moment: Array
    step: Array


@dataclass
class PTProblem:
    pde_frozen: Array
    context: Optional[core.PointwisePTContext]
    context_seconds: float


@dataclass
class UProblem:
    pde_frozen: Array
    neu_frozen: Array
    context: Optional[core.PointwiseUContext]
    context_seconds: float


def build_pt_problem(flat_u_old: Array) -> PTProblem:
    start = time.perf_counter()
    if IS_POINTWISE:
        context = core.build_pointwise_pt_context(flat_u_old, POINT_BLOCK_PT)
        pde_frozen = context.pde_frozen.reshape(-1, 2)[
            : core.PDE_POINTS.shape[0]
        ]
    else:
        context = None
        pde_frozen = core.BUILD_PT_FROZEN_PDE_TERMS(flat_u_old)
        pde_frozen.block_until_ready()
    duration = time.perf_counter() - start
    return PTProblem(pde_frozen, context, duration)


def build_u_problem(flat_pt: Array) -> UProblem:
    start = time.perf_counter()
    if IS_POINTWISE:
        context = core.build_pointwise_u_context(flat_pt, POINT_BLOCK_U)
        pde_frozen = context.pde_frozen.reshape(-1, 3)[
            : core.PDE_POINTS.shape[0]
        ]
        neu_frozen = context.neu_frozen.reshape(-1, 2)[
            : core.U_NEU_POINTS.shape[0]
        ]
    else:
        context = None
        pde_frozen, neu_frozen = core.BUILD_U_FROZEN_TERMS(flat_pt)
        pde_frozen.block_until_ready()
        neu_frozen.block_until_ready()
    duration = time.perf_counter() - start
    return UProblem(pde_frozen, neu_frozen, context, duration)


def append_context_profile(
    fs_iteration: int,
    subproblem: str,
    duration: float,
) -> None:
    PROFILE_ROWS.append(
        {
            "method": METHOD_LABEL,
            "method_key": METHOD,
            "fs_iteration": fs_iteration,
            "subproblem": subproblem,
            "stage": "context",
            "inner_iteration": 0,
            "updates_in_row": 0,
            "context_seconds": duration,
            "wall_seconds": duration,
            "assembly_seconds": 0.0,
            "solve_seconds": 0.0,
            "line_search_seconds": 0.0,
            "optimizer_seconds": 0.0,
        }
    )


# =============================================================================
# 4. Adam: retain separate momentum states for both networks
# =============================================================================


@jax.jit
def adam_pt_block(
    flat_pt: Array,
    first_moment: Array,
    second_moment: Array,
    step: Array,
    pde_frozen: Array,
    steps: Array,
) -> Tuple[Array, Array, Array, Array, Array]:
    beta1 = jnp.asarray(0.9, dtype=flat_pt.dtype)
    beta2 = jnp.asarray(0.999, dtype=flat_pt.dtype)
    epsilon = jnp.asarray(1.0e-8, dtype=flat_pt.dtype)
    learning_rate = jnp.asarray(ADAM_LEARNING_RATE, dtype=flat_pt.dtype)

    def body(_, state):
        parameters, first, second, count, _ = state
        loss, gradient = PT_LOSS_AND_GRAD(parameters, pde_frozen)
        count = count + 1
        first = beta1 * first + (1.0 - beta1) * gradient
        second = beta2 * second + (1.0 - beta2) * gradient**2
        first_hat = first / (1.0 - beta1**count)
        second_hat = second / (1.0 - beta2**count)
        parameters = parameters - learning_rate * first_hat / (
            jnp.sqrt(second_hat) + epsilon
        )
        return parameters, first, second, count, loss

    initial_loss = pt_loss_cached(flat_pt, pde_frozen)
    return jax.lax.fori_loop(
        0,
        steps,
        body,
        (flat_pt, first_moment, second_moment, step, initial_loss),
    )


@jax.jit
def adam_u_block(
    flat_u: Array,
    first_moment: Array,
    second_moment: Array,
    step: Array,
    pde_frozen: Array,
    neu_frozen: Array,
    steps: Array,
) -> Tuple[Array, Array, Array, Array, Array]:
    beta1 = jnp.asarray(0.9, dtype=flat_u.dtype)
    beta2 = jnp.asarray(0.999, dtype=flat_u.dtype)
    epsilon = jnp.asarray(1.0e-8, dtype=flat_u.dtype)
    learning_rate = jnp.asarray(ADAM_LEARNING_RATE, dtype=flat_u.dtype)

    def body(_, state):
        parameters, first, second, count, _ = state
        loss, gradient = U_LOSS_AND_GRAD(
            parameters, pde_frozen, neu_frozen
        )
        count = count + 1
        first = beta1 * first + (1.0 - beta1) * gradient
        second = beta2 * second + (1.0 - beta2) * gradient**2
        first_hat = first / (1.0 - beta1**count)
        second_hat = second / (1.0 - beta2**count)
        parameters = parameters - learning_rate * first_hat / (
            jnp.sqrt(second_hat) + epsilon
        )
        return parameters, first, second, count, loss

    initial_loss = u_loss_cached(flat_u, pde_frozen, neu_frozen)
    return jax.lax.fori_loop(
        0,
        steps,
        body,
        (flat_u, first_moment, second_moment, step, initial_loss),
    )


def run_adam_pt(
    flat_pt: Array,
    state: AdamState,
    problem: PTProblem,
    fs_iteration: int,
) -> Tuple[Array, AdamState, float]:
    loss_before = float(jax.device_get(pt_loss_cached(flat_pt, problem.pde_frozen)))
    start = time.perf_counter()
    updated, first, second, step, _ = adam_pt_block(
        flat_pt,
        state.first_moment,
        state.second_moment,
        state.step,
        problem.pde_frozen,
        jnp.asarray(ADAM_STEPS_PER_BLOCK, dtype=jnp.int32),
    )
    updated.block_until_ready()
    duration = time.perf_counter() - start
    loss_after = float(jax.device_get(pt_loss_cached(updated, problem.pde_frozen)))
    PROFILE_ROWS.append(
        {
            "method": METHOD_LABEL,
            "method_key": METHOD,
            "fs_iteration": fs_iteration,
            "subproblem": "pT",
            "stage": "adam",
            "inner_iteration": ADAM_STEPS_PER_BLOCK,
            "updates_in_row": ADAM_STEPS_PER_BLOCK,
            "loss_before": loss_before,
            "loss_after": loss_after,
            "context_seconds": 0.0,
            "wall_seconds": duration,
            "assembly_seconds": 0.0,
            "solve_seconds": 0.0,
            "line_search_seconds": 0.0,
            "optimizer_seconds": duration,
        }
    )
    return updated, AdamState(first, second, step), loss_after


def run_adam_u(
    flat_u: Array,
    state: AdamState,
    problem: UProblem,
    fs_iteration: int,
) -> Tuple[Array, AdamState, float]:
    loss_before = float(
        jax.device_get(
            u_loss_cached(flat_u, problem.pde_frozen, problem.neu_frozen)
        )
    )
    start = time.perf_counter()
    updated, first, second, step, _ = adam_u_block(
        flat_u,
        state.first_moment,
        state.second_moment,
        state.step,
        problem.pde_frozen,
        problem.neu_frozen,
        jnp.asarray(ADAM_STEPS_PER_BLOCK, dtype=jnp.int32),
    )
    updated.block_until_ready()
    duration = time.perf_counter() - start
    loss_after = float(
        jax.device_get(
            u_loss_cached(updated, problem.pde_frozen, problem.neu_frozen)
        )
    )
    PROFILE_ROWS.append(
        {
            "method": METHOD_LABEL,
            "method_key": METHOD,
            "fs_iteration": fs_iteration,
            "subproblem": "uXi",
            "stage": "adam",
            "inner_iteration": ADAM_STEPS_PER_BLOCK,
            "updates_in_row": ADAM_STEPS_PER_BLOCK,
            "loss_before": loss_before,
            "loss_after": loss_after,
            "context_seconds": 0.0,
            "wall_seconds": duration,
            "assembly_seconds": 0.0,
            "solve_seconds": 0.0,
            "line_search_seconds": 0.0,
            "optimizer_seconds": duration,
        }
    )
    return updated, AdamState(first, second, step), loss_after


# =============================================================================
# 5. L-BFGS: reset history for each GFS subproblem
# =============================================================================


def run_lbfgs_block(
    flat_params: Array,
    subproblem: str,
    problem: object,
    fs_iteration: int,
) -> Tuple[Array, float, int]:
    try:
        from scipy.optimize import minimize
    except ImportError as exc:
        raise RuntimeError(
            "Adam+L-BFGS requires scipy in the experiment environment."
        ) from exc

    if subproblem == "pT":
        assert isinstance(problem, PTProblem)

        def value_gradient(flat_numpy: np.ndarray):
            value, gradient = PT_LOSS_AND_GRAD(
                jnp.asarray(flat_numpy), problem.pde_frozen
            )
            value.block_until_ready()
            gradient.block_until_ready()
            return float(jax.device_get(value)), np.asarray(jax.device_get(gradient))

        initial_loss = float(
            jax.device_get(pt_loss_cached(flat_params, problem.pde_frozen))
        )
    elif subproblem == "uXi":
        assert isinstance(problem, UProblem)

        def value_gradient(flat_numpy: np.ndarray):
            value, gradient = U_LOSS_AND_GRAD(
                jnp.asarray(flat_numpy),
                problem.pde_frozen,
                problem.neu_frozen,
            )
            value.block_until_ready()
            gradient.block_until_ready()
            return float(jax.device_get(value)), np.asarray(jax.device_get(gradient))

        initial_loss = float(
            jax.device_get(
                u_loss_cached(
                    flat_params, problem.pde_frozen, problem.neu_frozen
                )
            )
        )
    else:
        raise ValueError(f"Unknown L-BFGS subproblem {subproblem!r}.")

    start = time.perf_counter()
    result = minimize(
        value_gradient,
        np.asarray(jax.device_get(flat_params)),
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": LBFGS_MAXITER_PER_BLOCK,
            "maxfun": LBFGS_MAXFUN_PER_BLOCK,
            "ftol": LBFGS_FTOL,
            "gtol": LBFGS_GTOL,
            "maxls": 50,
        },
    )
    duration = time.perf_counter() - start
    updated = jnp.asarray(result.x, dtype=flat_params.dtype)
    updated.block_until_ready()
    PROFILE_ROWS.append(
        {
            "method": METHOD_LABEL,
            "method_key": METHOD,
            "fs_iteration": fs_iteration,
            "subproblem": subproblem,
            "stage": "lbfgs",
            "inner_iteration": int(result.nit),
            "updates_in_row": int(result.nit),
            "function_evaluations": int(result.nfev),
            "loss_before": initial_loss,
            "loss_after": float(result.fun),
            "context_seconds": 0.0,
            "wall_seconds": duration,
            "assembly_seconds": 0.0,
            "solve_seconds": 0.0,
            "line_search_seconds": 0.0,
            "optimizer_seconds": duration,
            "optimizer_success": bool(result.success),
            "optimizer_status": int(result.status),
            "optimizer_message": str(result.message),
        }
    )
    core.LOGGER.info(
        "FS %03d %s L-BFGS: success=%s nit=%d nfev=%d "
        "loss %.4e -> %.4e (%s)",
        fs_iteration,
        subproblem,
        result.success,
        result.nit,
        result.nfev,
        initial_loss,
        result.fun,
        result.message,
    )
    return updated, float(result.fun), int(result.nit)


# =============================================================================
# 6. Energy-natural / Levenberg-Marquardt block optimizer
# =============================================================================


ENGD_CONFIG = core.ENGDConfig(
    line_search_levels=LINE_SEARCH_LEVELS,
    gram_ridge=GRAM_RIDGE,
    rank_safe_ridge_safety=0.0,
    solver=LINEAR_SOLVER if IS_ENGD else "eigh-refined",
    compare_lstsq_reference=False,
    relative_cutoff=EIGH_RELATIVE_CUTOFF,
    refinement_steps=EIGH_REFINEMENT_STEPS,
    lstsq_rcond=LSTSQ_RCOND,
    jacobian_backend=JACOBIAN_BACKEND if IS_ENGD else "pointwise",
    gradient_tolerance=1.0e-14,
    require_loss_decrease=True,
)


def full_blockwise_engd_step(
    flat_params: Array,
    blocks: Sequence[
        Tuple[
            str,
            Callable[[Array], Array],
            Callable[[Array], Array],
        ]
    ],
    candidate_losses_fn: Callable[[Array], Array],
) -> Tuple[Array, core.ENGDStepStats]:
    """Perform an energy-natural update with a complete Jacobian per physical block.

    Accumulate G=sum(J_b.T@J_b), g=sum(J_b.T@r_b), then solve using the
    selected dense solver and damping. Each complete J_b is materialized
    before its Gram product. Full and streamed paths minimize the same loss."""

    residual_seconds = 0.0
    jacobian_seconds = 0.0
    gram_matmul_seconds = 0.0
    gradient_matmul_seconds = 0.0
    loss_before = 0.0
    gram_matrix_raw: Optional[Array] = None
    gradient_flat: Optional[Array] = None
    assembly_start = time.perf_counter()

    for _, residual_function, jacobian_function in blocks:
        start = time.perf_counter()
        residual = residual_function(flat_params)
        residual.block_until_ready()
        residual_seconds += time.perf_counter() - start
        loss_before += float(jax.device_get(core.half_squared_norm(residual)))

        start = time.perf_counter()
        jacobian = jacobian_function(flat_params)
        jacobian.block_until_ready()
        jacobian_seconds += time.perf_counter() - start

        start = time.perf_counter()
        block_gram = jacobian.T @ jacobian
        block_gram.block_until_ready()
        gram_matrix_raw = (
            block_gram
            if gram_matrix_raw is None
            else gram_matrix_raw + block_gram
        )
        gram_matrix_raw.block_until_ready()
        gram_matmul_seconds += time.perf_counter() - start

        start = time.perf_counter()
        block_gradient = jacobian.T @ residual
        block_gradient.block_until_ready()
        gradient_flat = (
            block_gradient
            if gradient_flat is None
            else gradient_flat + block_gradient
        )
        gradient_flat.block_until_ready()
        gradient_matmul_seconds += time.perf_counter() - start

        # Release each physical-block Jacobian after its full Gram product.
        
        del jacobian, residual, block_gram, block_gradient

    if gram_matrix_raw is None or gradient_flat is None:
        raise ValueError("At least one residual/Jacobian block is required.")

    start = time.perf_counter()
    gram_matrix, symmetry_device, gradient_norm_device = (
        core._prepare_gram_and_diagnostics(
            gram_matrix_raw,
            gradient_flat,
        )
    )
    gram_matrix.block_until_ready()
    symmetry_error, gradient_norm = jax.device_get(
        (symmetry_device, gradient_norm_device)
    )
    symmetrize_seconds = time.perf_counter() - start
    assembly_seconds = time.perf_counter() - assembly_start

    solve_start = time.perf_counter()
    lstsq_rank = -1
    if LINEAR_SOLVER == "lstsq":
        solve_result, lstsq_rank = core._lstsq_result(
            gram_matrix, gradient_flat, damping=GRAM_RIDGE, rcond=LSTSQ_RCOND,
        )
    elif LINEAR_SOLVER == "eigh-refined":
        solve_result = core.solve_eigh_refined(
            gram_matrix, gradient_flat,
            damping=GRAM_RIDGE,
            rank_safe_ridge_safety=0.0,
            relative_cutoff=EIGH_RELATIVE_CUTOFF,
            refinement_steps=EIGH_REFINEMENT_STEPS,
        )
    else:
        raise ValueError(f"Unsupported full-Jacobian solver: {LINEAR_SOLVER}")
    solve_result.direction.block_until_ready()
    solve_seconds = time.perf_counter() - solve_start

    line_search_start = time.perf_counter()
    updated, loss_after, step_size, accepted = (
        core.vectorized_logarithmic_line_search(
            flat_params,
            solve_result.direction,
            candidate_losses_fn,
            loss_before,
            ENGD_CONFIG,
        )
    )
    updated.block_until_ready()
    line_search_seconds = time.perf_counter() - line_search_start

    nan = float("nan")
    stats = core.ENGDStepStats(
        accepted=accepted,
        loss_before=loss_before,
        loss_after=loss_after,
        step_size=step_size,
        solver_used=LINEAR_SOLVER,
        gradient_norm=float(gradient_norm),
        natural_gradient_norm=float(
            jax.device_get(jnp.linalg.norm(solve_result.direction))
        ),
        effective_rank=solve_result.effective_rank,
        spectral_cutoff=solve_result.spectral_cutoff,
        lambda_max=solve_result.max_eigenvalue,
        lambda_min=solve_result.min_eigenvalue,
        lambda_min_kept=solve_result.min_kept_eigenvalue,
        effective_condition=solve_result.effective_condition,
        negative_count=solve_result.negative_count,
        max_negative_abs=solve_result.max_negative_abs,
        negative_above_cutoff=solve_result.negative_above_cutoff,
        solve_relative_residual=solve_result.relative_residual,
        predicted_reduction=solve_result.predicted_reduction,
        refinement_steps=EIGH_REFINEMENT_STEPS if LINEAR_SOLVER == "eigh-refined" else 0,
        applied_gram_ridge=solve_result.applied_gram_ridge,
        rank_safe_fp_floor=solve_result.rank_safe_fp_floor,
        ridge_to_rank_safe_floor=solve_result.ridge_to_rank_safe_floor,
        lstsq_rank=lstsq_rank,
        lstsq_effective_condition=solve_result.effective_condition if LINEAR_SOLVER == "lstsq" else nan,
        lstsq_solve_relative_residual=solve_result.relative_residual if LINEAR_SOLVER == "lstsq" else nan,
        direction_relative_l2_error=nan,
        direction_cosine=nan,
        direction_relative_energy_error=nan,
        residual_seconds=residual_seconds,
        jacobian_seconds=jacobian_seconds,
        gram_matmul_seconds=gram_matmul_seconds,
        gradient_matmul_seconds=gradient_matmul_seconds,
        symmetrize_seconds=symmetrize_seconds,
        gram_symmetry_relative_error=float(symmetry_error),
        assembly_seconds=assembly_seconds,
        solve_seconds=solve_seconds,
        reference_seconds=0.0,
        line_search_seconds=line_search_seconds,
    )
    return updated, stats


def run_engd_pt(
    flat_pt: Array,
    problem: PTProblem,
    fs_iteration: int,
) -> Tuple[Array, float, int]:
    candidate_fn = lambda candidates: core.candidate_losses_pt_cached(
        candidates, problem.pde_frozen
    )
    gram_fn: Optional[Callable[[Array], Tuple[Array, Array, Array]]] = None
    if IS_POINTWISE:
        assert problem.context is not None
        gram_fn = lambda z: core.pointwise_pt_gram_gradient(z, problem.context)

    accepted_updates = 0
    last_loss = float(jax.device_get(pt_loss_cached(flat_pt, problem.pde_frozen)))
    for inner in range(1, ENGD_STEPS_PER_BLOCK + 1):
        wall_start = time.perf_counter()
        if IS_FULL_JACOBIAN:
            blocks = (
                (
                    "pde",
                    lambda z: pt_pde_block_residual(z, problem.pde_frozen),
                    lambda z: jacobian_pt_pde_block(z, problem.pde_frozen),
                ),
                ("dirichlet", pt_dir_block_residual, jacobian_pt_dir_block),
                ("initial", pt_ic_block_residual, jacobian_pt_ic_block),
            )
            flat_pt, stats = full_blockwise_engd_step(
                flat_pt, blocks, candidate_fn
            )
        else:
            residual_fn = lambda z: core.residual_pt_cached_flat(
                z, problem.pde_frozen
            )
            jacobian_fn = lambda z: core.JACOBIAN_PT_FLAT(z, core.FLAT_U)
            flat_pt, stats = core.explicit_engd_step(
                flat_pt,
                residual_fn,
                jacobian_fn,
                candidate_fn,
                ENGD_CONFIG,
                gram_gradient_fn=gram_fn,
            )
        wall_seconds = time.perf_counter() - wall_start
        accepted_updates += int(stats.accepted)
        last_loss = stats.loss_after
        row: Dict[str, object] = {
            "method": METHOD_LABEL,
            "method_key": METHOD,
            "fs_iteration": fs_iteration,
            "subproblem": "pT",
            "stage": "engd",
            "inner_iteration": inner,
            "updates_in_row": 1,
            "accepted_updates": int(stats.accepted),
            "context_seconds": 0.0,
            "wall_seconds": wall_seconds,
            "optimizer_seconds": wall_seconds,
        }
        row.update(asdict(stats))
        PROFILE_ROWS.append(row)
    return flat_pt, last_loss, accepted_updates


def run_engd_u(
    flat_u: Array,
    problem: UProblem,
    fs_iteration: int,
) -> Tuple[Array, float, int]:
    candidate_fn = lambda candidates: core.candidate_losses_u_cached(
        candidates, problem.pde_frozen, problem.neu_frozen
    )
    gram_fn: Optional[Callable[[Array], Tuple[Array, Array, Array]]] = None
    if IS_POINTWISE:
        assert problem.context is not None
        gram_fn = lambda z: core.pointwise_u_gram_gradient(z, problem.context)

    accepted_updates = 0
    last_loss = float(
        jax.device_get(
            u_loss_cached(flat_u, problem.pde_frozen, problem.neu_frozen)
        )
    )
    for inner in range(1, ENGD_STEPS_PER_BLOCK + 1):
        wall_start = time.perf_counter()
        if IS_FULL_JACOBIAN:
            blocks = (
                (
                    "pde",
                    lambda z: u_pde_block_residual(z, problem.pde_frozen),
                    lambda z: jacobian_u_pde_block(z, problem.pde_frozen),
                ),
                ("dirichlet", u_dir_block_residual, jacobian_u_dir_block),
                (
                    "neumann",
                    lambda z: u_neu_block_residual(z, problem.neu_frozen),
                    lambda z: jacobian_u_neu_block(z, problem.neu_frozen),
                ),
                ("initial", u_ic_block_residual, jacobian_u_ic_block),
            )
            flat_u, stats = full_blockwise_engd_step(
                flat_u, blocks, candidate_fn
            )
        else:
            residual_fn = lambda z: core.residual_u_cached_flat(
                z, problem.pde_frozen, problem.neu_frozen
            )
            jacobian_fn = lambda z: core.JACOBIAN_U_FLAT(z, core.FLAT_PT)
            flat_u, stats = core.explicit_engd_step(
                flat_u,
                residual_fn,
                jacobian_fn,
                candidate_fn,
                ENGD_CONFIG,
                gram_gradient_fn=gram_fn,
            )
        wall_seconds = time.perf_counter() - wall_start
        accepted_updates += int(stats.accepted)
        last_loss = stats.loss_after
        row: Dict[str, object] = {
            "method": METHOD_LABEL,
            "method_key": METHOD,
            "fs_iteration": fs_iteration,
            "subproblem": "uXi",
            "stage": "engd",
            "inner_iteration": inner,
            "updates_in_row": 1,
            "accepted_updates": int(stats.accepted),
            "context_seconds": 0.0,
            "wall_seconds": wall_seconds,
            "optimizer_seconds": wall_seconds,
        }
        row.update(asdict(stats))
        PROFILE_ROWS.append(row)
    return flat_u, last_loss, accepted_updates


# =============================================================================
# 7. Error evaluation, TTA, and fixed budgets
# =============================================================================


def record_snapshot(
    flat_u: Array,
    flat_pt: Array,
    fs_iteration: int,
    stage: str,
    loss_pt_split: float,
    loss_u_split: float,
    coupled_pt_loss_value: float,
    u_outer_change: float,
    pt_outer_change: float,
    pt_updates_completed: int,
    u_updates_completed: int,
) -> None:
    start = time.perf_counter()
    errors = core.evaluate_errors(flat_u, flat_pt)
    metrics = group_error_metrics(errors)
    duration = time.perf_counter() - start
    LEDGER.evaluation_excluded_seconds += duration

    row: Dict[str, object] = {
        "method": METHOD_LABEL,
        "method_key": METHOD,
        "method_family": METHOD_FAMILY,
        "jacobian_backend": JACOBIAN_BACKEND,
        "linear_solver": LINEAR_SOLVER,
        "gram_ridge": GRAM_RIDGE,
        "fs_iteration": fs_iteration,
        "stage": stage,
        "pT_optimizer_updates_completed": pt_updates_completed,
        "uXi_optimizer_updates_completed": u_updates_completed,
        "total_block_optimizer_updates_completed": (
            pt_updates_completed + u_updates_completed
        ),
        "effective_training_seconds": LEDGER.effective_training_seconds,
        "context_seconds_cumulative": LEDGER.context_seconds,
        "optimizer_seconds_cumulative": LEDGER.optimizer_seconds,
        "convergence_seconds_cumulative": LEDGER.convergence_seconds,
        "evaluation_excluded_seconds_this_checkpoint": duration,
        "loss_pt_split": loss_pt_split,
        "loss_u_split": loss_u_split,
        "loss_pt_coupled_pde": coupled_pt_loss_value,
        "u_outer_change": u_outer_change,
        "pt_outer_change": pt_outer_change,
    }
    row.update({name: float(value) for name, value in errors.items()})
    row.update(metrics)
    HISTORY_ROWS.append(row)

    core.LOGGER.info(
        "FS %03d stage=%s effective=%.3f s | "
        "u1=%.4e u2=%.4e xi=%.4e p=%.4e T=%.4e | "
        "E_common=%.4e E_4F=%.4e | loss pT=%.4e uXi=%.4e "
        "coupled-pT=%.4e | outer du=%.3e dpT=%.3e",
        fs_iteration,
        stage,
        LEDGER.effective_training_seconds,
        errors["u1_rel_l2"],
        errors["u2_rel_l2"],
        errors["xi_rel_l2"],
        errors["p_rel_l2"],
        errors["T_rel_l2"],
        metrics["E_common_max"],
        metrics["E_4F_max"],
        loss_pt_split,
        loss_u_split,
        coupled_pt_loss_value,
        u_outer_change,
        pt_outer_change,
    )


def build_tta_rows() -> List[Dict[str, object]]:
    metric_names = (
        "u1_rel_l2",
        "u2_rel_l2",
        "xi_rel_l2",
        "p_rel_l2",
        "T_rel_l2",
        "E_u_max",
        "E_pT_max",
        "E_common_max",
        "E_4F_max",
    )
    rows: List[Dict[str, object]] = []
    for metric in metric_names:
        values = np.asarray([float(row[metric]) for row in HISTORY_ROWS])
        for threshold in TTA_THRESHOLDS:
            hit: Optional[int] = None
            for index in range(len(values) - TTA_PATIENCE + 1):
                if np.all(values[index : index + TTA_PATIENCE] <= threshold):
                    hit = index
                    break
            if hit is None:
                rows.append(
                    {
                        "method": METHOD_LABEL,
                        "method_key": METHOD,
                        "metric": metric,
                        "threshold": threshold,
                        "patience": TTA_PATIENCE,
                        "reached": False,
                        "first_fs_iteration": -1,
                        "confirmation_fs_iteration": -1,
                        "time_to_accuracy_seconds": math.nan,
                        "value_at_accuracy": math.nan,
                        "final_value": float(values[-1]),
                    }
                )
            else:
                source = HISTORY_ROWS[hit]
                confirmation = HISTORY_ROWS[hit + TTA_PATIENCE - 1]
                rows.append(
                    {
                        "method": METHOD_LABEL,
                        "method_key": METHOD,
                        "metric": metric,
                        "threshold": threshold,
                        "patience": TTA_PATIENCE,
                        "reached": True,
                        "first_fs_iteration": int(source["fs_iteration"]),
                        "confirmation_fs_iteration": int(
                            confirmation["fs_iteration"]
                        ),
                        "time_to_accuracy_seconds": float(
                            source["effective_training_seconds"]
                        ),
                        "value_at_accuracy": float(source[metric]),
                        "final_value": float(values[-1]),
                    }
                )
    return rows


def build_fixed_fs_rows() -> List[Dict[str, object]]:
    by_fs = {int(row["fs_iteration"]): row for row in HISTORY_ROWS}
    return [dict(by_fs[fs]) for fs in FIXED_FS_CHECKPOINTS if fs in by_fs]


def build_fixed_time_rows() -> List[Dict[str, object]]:
    times = np.asarray(
        [float(row["effective_training_seconds"]) for row in HISTORY_ROWS]
    )
    rows: List[Dict[str, object]] = []
    for budget in FIXED_TIME_BUDGETS:
        candidates = np.nonzero(times <= budget + 1.0e-12)[0]
        index = int(candidates[-1]) if candidates.size else 0
        source = dict(HISTORY_ROWS[index])
        source["requested_time_budget_seconds"] = budget
        source["budget_fully_used"] = bool(times[-1] >= budget)
        source["selection_rule"] = "last completed GFS sweep at or before budget"
        rows.append(source)
    return rows


# =============================================================================
# 8. Compilation warmup, training, and checkpoints
# =============================================================================


def warmup() -> None:
    start = time.perf_counter()
    flat_u = jnp.array(core.FLAT_U)
    flat_pt = jnp.array(core.FLAT_PT)

    pt_problem = build_pt_problem(flat_u)
    u_problem = build_u_problem(flat_pt)

    if METHOD in {"adam", "adam_lbfgs"}:
        zero_pt = jnp.zeros_like(flat_pt)
        zero_u = jnp.zeros_like(flat_u)
        warm_pt = adam_pt_block(
            flat_pt,
            zero_pt,
            zero_pt,
            jnp.asarray(0, dtype=jnp.int64),
            pt_problem.pde_frozen,
            jnp.asarray(1, dtype=jnp.int32),
        )[0]
        warm_u = adam_u_block(
            flat_u,
            zero_u,
            zero_u,
            jnp.asarray(0, dtype=jnp.int64),
            u_problem.pde_frozen,
            u_problem.neu_frozen,
            jnp.asarray(1, dtype=jnp.int32),
        )[0]
        warm_pt.block_until_ready()
        warm_u.block_until_ready()
    else:
        candidate_pt_fn = lambda candidates: core.candidate_losses_pt_cached(
            candidates, pt_problem.pde_frozen
        )
        if IS_FULL_JACOBIAN:
            pt_blocks = (
                (
                    "pde",
                    lambda z: pt_pde_block_residual(
                        z, pt_problem.pde_frozen
                    ),
                    lambda z: jacobian_pt_pde_block(
                        z, pt_problem.pde_frozen
                    ),
                ),
                ("dirichlet", pt_dir_block_residual, jacobian_pt_dir_block),
                ("initial", pt_ic_block_residual, jacobian_pt_ic_block),
            )
            warm_pt, _ = full_blockwise_engd_step(
                flat_pt, pt_blocks, candidate_pt_fn
            )
        else:
            assert pt_problem.context is not None
            gram_pt_fn = lambda z: core.pointwise_pt_gram_gradient(
                z, pt_problem.context
            )
            warm_pt, _ = core.explicit_engd_step(
                flat_pt,
                lambda z: core.residual_pt_cached_flat(
                    z, pt_problem.pde_frozen
                ),
                lambda z: core.JACOBIAN_PT_FLAT(z, core.FLAT_U),
                candidate_pt_fn,
                ENGD_CONFIG,
                gram_gradient_fn=gram_pt_fn,
            )
        warm_pt.block_until_ready()

        candidate_u_fn = lambda candidates: core.candidate_losses_u_cached(
            candidates, u_problem.pde_frozen, u_problem.neu_frozen
        )
        if IS_FULL_JACOBIAN:
            u_blocks = (
                (
                    "pde",
                    lambda z: u_pde_block_residual(
                        z, u_problem.pde_frozen
                    ),
                    lambda z: jacobian_u_pde_block(
                        z, u_problem.pde_frozen
                    ),
                ),
                ("dirichlet", u_dir_block_residual, jacobian_u_dir_block),
                (
                    "neumann",
                    lambda z: u_neu_block_residual(
                        z, u_problem.neu_frozen
                    ),
                    lambda z: jacobian_u_neu_block(
                        z, u_problem.neu_frozen
                    ),
                ),
                ("initial", u_ic_block_residual, jacobian_u_ic_block),
            )
            warm_u, _ = full_blockwise_engd_step(
                flat_u, u_blocks, candidate_u_fn
            )
        else:
            assert u_problem.context is not None
            gram_u_fn = lambda z: core.pointwise_u_gram_gradient(
                z, u_problem.context
            )
            warm_u, _ = core.explicit_engd_step(
                flat_u,
                lambda z: core.residual_u_cached_flat(
                    z, u_problem.pde_frozen, u_problem.neu_frozen
                ),
                lambda z: core.JACOBIAN_U_FLAT(z, core.FLAT_PT),
                candidate_u_fn,
                ENGD_CONFIG,
                gram_gradient_fn=gram_u_fn,
            )
        warm_u.block_until_ready()

    # Exclude compilation of convergence diagnostics from effective training time.
    coupled_pt_pde_loss(flat_u, flat_pt).block_until_ready()
    core.relative_prediction_change(flat_u, flat_u, core.UNRAVEL_U)
    core.relative_prediction_change(flat_pt, flat_pt, core.UNRAVEL_PT)
    LEDGER.jit_warmup_excluded_seconds = time.perf_counter() - start


def save_model_checkpoint(
    flat_u: Array,
    flat_pt: Array,
    fs_iteration: int,
) -> None:
    output = core.RESULT_DIR / f"model_fs{fs_iteration:04d}.npz"
    np.savez(
        output,
        flat_uxi=np.asarray(jax.device_get(flat_u)),
        flat_u=np.asarray(jax.device_get(flat_u)),
        flat_pt=np.asarray(jax.device_get(flat_pt)),
        net_uxi_sizes=np.asarray(core.NET_U_SIZES),
        net_pt_sizes=np.asarray(core.NET_PT_SIZES),
        fs_iteration=np.asarray(fs_iteration),
        method=np.asarray(METHOD_LABEL),
    )


def initial_losses(flat_u: Array, flat_pt: Array) -> Tuple[float, float, float]:
    pde_frozen = core.BUILD_PT_FROZEN_PDE_TERMS(flat_u)
    pde_u, neu_u = core.BUILD_U_FROZEN_TERMS(flat_pt)
    loss_pt = float(jax.device_get(pt_loss_cached(flat_pt, pde_frozen)))
    loss_u = float(jax.device_get(u_loss_cached(flat_u, pde_u, neu_u)))
    coupled = float(jax.device_get(coupled_pt_pde_loss(flat_u, flat_pt)))
    return loss_pt, loss_u, coupled


def train() -> Tuple[Array, Array, float]:
    flat_u = jnp.array(core.FLAT_U)
    flat_pt = jnp.array(core.FLAT_PT)
    adam_state_u = AdamState(
        jnp.zeros_like(flat_u),
        jnp.zeros_like(flat_u),
        jnp.asarray(0, dtype=jnp.int64),
    )
    adam_state_pt = AdamState(
        jnp.zeros_like(flat_pt),
        jnp.zeros_like(flat_pt),
        jnp.asarray(0, dtype=jnp.int64),
    )

    loss_pt, loss_u, coupled = initial_losses(flat_u, flat_pt)
    record_snapshot(
        flat_u,
        flat_pt,
        fs_iteration=0,
        stage="initial",
        loss_pt_split=loss_pt,
        loss_u_split=loss_u,
        coupled_pt_loss_value=coupled,
        u_outer_change=0.0,
        pt_outer_change=0.0,
        pt_updates_completed=0,
        u_updates_completed=0,
    )

    pt_updates_completed = 0
    u_updates_completed = 0
    solver_wall_start = time.perf_counter()

    for fs_iteration in range(1, FS_ITERATIONS + 1):
        flat_u_old = jnp.array(flat_u)
        flat_pt_old = jnp.array(flat_pt)

        pt_problem = build_pt_problem(flat_u_old)
        LEDGER.context_seconds += pt_problem.context_seconds
        append_context_profile(
            fs_iteration, "pT", pt_problem.context_seconds
        )

        optimizer_start = time.perf_counter()
        if METHOD == "adam" or (
            METHOD == "adam_lbfgs"
            and fs_iteration <= ADAM_LBFGS_SWITCH_FS
        ):
            flat_pt, adam_state_pt, loss_pt = run_adam_pt(
                flat_pt, adam_state_pt, pt_problem, fs_iteration
            )
            pt_updates_completed += ADAM_STEPS_PER_BLOCK
            pt_stage = "adam"
        elif METHOD == "adam_lbfgs":
            flat_pt, loss_pt, actual_updates = run_lbfgs_block(
                flat_pt, "pT", pt_problem, fs_iteration
            )
            pt_updates_completed += actual_updates
            pt_stage = "lbfgs"
        else:
            flat_pt, loss_pt, accepted = run_engd_pt(
                flat_pt, pt_problem, fs_iteration
            )
            pt_updates_completed += ENGD_STEPS_PER_BLOCK
            pt_stage = f"engd({accepted}/{ENGD_STEPS_PER_BLOCK} accepted)"
        LEDGER.optimizer_seconds += time.perf_counter() - optimizer_start

        u_problem = build_u_problem(flat_pt)
        LEDGER.context_seconds += u_problem.context_seconds
        append_context_profile(
            fs_iteration, "uXi", u_problem.context_seconds
        )

        optimizer_start = time.perf_counter()
        if METHOD == "adam" or (
            METHOD == "adam_lbfgs"
            and fs_iteration <= ADAM_LBFGS_SWITCH_FS
        ):
            flat_u, adam_state_u, loss_u = run_adam_u(
                flat_u, adam_state_u, u_problem, fs_iteration
            )
            u_updates_completed += ADAM_STEPS_PER_BLOCK
            u_stage = "adam"
        elif METHOD == "adam_lbfgs":
            flat_u, loss_u, actual_updates = run_lbfgs_block(
                flat_u, "uXi", u_problem, fs_iteration
            )
            u_updates_completed += actual_updates
            u_stage = "lbfgs"
        else:
            flat_u, loss_u, accepted = run_engd_u(
                flat_u, u_problem, fs_iteration
            )
            u_updates_completed += ENGD_STEPS_PER_BLOCK
            u_stage = f"engd({accepted}/{ENGD_STEPS_PER_BLOCK} accepted)"
        LEDGER.optimizer_seconds += time.perf_counter() - optimizer_start

        convergence_start = time.perf_counter()
        coupled = float(jax.device_get(coupled_pt_pde_loss(flat_u, flat_pt)))
        u_outer_change = core.relative_prediction_change(
            flat_u, flat_u_old, core.UNRAVEL_U
        )
        pt_outer_change = core.relative_prediction_change(
            flat_pt, flat_pt_old, core.UNRAVEL_PT
        )
        LEDGER.convergence_seconds += time.perf_counter() - convergence_start

        record_snapshot(
            flat_u,
            flat_pt,
            fs_iteration=fs_iteration,
            stage=f"pT:{pt_stage}|uXi:{u_stage}",
            loss_pt_split=loss_pt,
            loss_u_split=loss_u,
            coupled_pt_loss_value=coupled,
            u_outer_change=u_outer_change,
            pt_outer_change=pt_outer_change,
            pt_updates_completed=pt_updates_completed,
            u_updates_completed=u_updates_completed,
        )

        if fs_iteration in {50, FS_ITERATIONS}:
            save_model_checkpoint(flat_u, flat_pt, fs_iteration)

    solver_wall_seconds = time.perf_counter() - solver_wall_start
    return flat_u, flat_pt, solver_wall_seconds


# =============================================================================
# 9. Output files
# =============================================================================


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
        for row in rows:
            writer.writerow(row)


def profile_totals() -> Dict[str, float]:
    keys = (
        "context_seconds",
        "wall_seconds",
        "optimizer_seconds",
        "residual_seconds",
        "jacobian_seconds",
        "gram_matmul_seconds",
        "gradient_matmul_seconds",
        "symmetrize_seconds",
        "assembly_seconds",
        "solve_seconds",
        "reference_seconds",
        "line_search_seconds",
    )
    return {
        key: float(
            sum(
                float(row.get(key, 0.0) or 0.0)
                for row in PROFILE_ROWS
                if math.isfinite(float(row.get(key, 0.0) or 0.0))
            )
        )
        for key in keys
    }


def save_outputs(
    flat_u: Array,
    flat_pt: Array,
    solver_wall_seconds: float,
) -> None:
    tta_rows = build_tta_rows()
    fixed_fs_rows = build_fixed_fs_rows()
    fixed_time_rows = build_fixed_time_rows()
    final_row = dict(HISTORY_ROWS[-1])

    write_csv(core.RESULT_DIR / "training_history.csv", HISTORY_ROWS)
    write_csv(core.RESULT_DIR / "optimizer_profile.csv", PROFILE_ROWS)
    write_csv(core.RESULT_DIR / "time_to_accuracy.csv", tta_rows)
    write_csv(core.RESULT_DIR / "fixed_fs_accuracy.csv", fixed_fs_rows)
    write_csv(core.RESULT_DIR / "fixed_time_accuracy.csv", fixed_time_rows)

    with (core.RESULT_DIR / "training_history.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(HISTORY_ROWS, file, indent=2, ensure_ascii=False)
    with (core.RESULT_DIR / "optimizer_profile.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(PROFILE_ROWS, file, indent=2, ensure_ascii=False)
    with (core.RESULT_DIR / "time_to_accuracy.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(tta_rows, file, indent=2, ensure_ascii=False)

    numeric_history = {
        key: np.asarray([row[key] for row in HISTORY_ROWS])
        for key in HISTORY_ROWS[0]
        if all(
            isinstance(row.get(key), (bool, int, float, np.number))
            for row in HISTORY_ROWS
        )
    }
    np.savez(core.RESULT_DIR / "training_history.npz", **numeric_history)
    np.savez(
        core.RESULT_DIR / "final_model.npz",
        flat_uxi=np.asarray(jax.device_get(flat_u)),
        flat_u=np.asarray(jax.device_get(flat_u)),
        flat_pt=np.asarray(jax.device_get(flat_pt)),
        net_uxi_sizes=np.asarray(core.NET_U_SIZES),
        net_pt_sizes=np.asarray(core.NET_PT_SIZES),
        fs_iteration=np.asarray(FS_ITERATIONS),
        method=np.asarray(METHOD_LABEL),
    )

    fs50 = next(
        (row for row in HISTORY_ROWS if int(row["fs_iteration"]) == 50),
        None,
    )
    fs50_to_fs100: Dict[str, object] = {}
    if fs50 is not None and int(final_row["fs_iteration"]) == 100:
        fs50_to_fs100 = {
            "additional_effective_seconds": float(
                final_row["effective_training_seconds"]
                - fs50["effective_training_seconds"]
            ),
            "metrics": {
                metric: {
                    "fs50": float(fs50[metric]),
                    "fs100": float(final_row[metric]),
                    "improvement_factor_fs50_over_fs100": float(
                        fs50[metric] / max(float(final_row[metric]), 1.0e-300)
                    ),
                    "relative_reduction": float(
                        1.0 - final_row[metric] / max(float(fs50[metric]), 1.0e-300)
                    ),
                }
                for metric in (
                    "u1_rel_l2",
                    "u2_rel_l2",
                    "xi_rel_l2",
                    "p_rel_l2",
                    "T_rel_l2",
                    "E_common_max",
                    "E_4F_max",
                )
            },
        }

    metadata = {
        "protocol": "THP-Group1-4F-GFS-optimizer-JS-v3",
        "archive_revision": "seven-method-table3-20261008",
        "fast_debug": core.FAST_DEBUG,
        "method": METHOD_LABEL,
        "method_key": METHOD,
        "method_family": METHOD_FAMILY,
        "definition": (
            "Every method uses the same four-field GFS outer iteration; "
            "only the optimizer/Gram backend differs."
        ),
        "seed": core.SEED,
        "exact_case": core.EXACT_CASE,
        "physical_parameters": ACTUAL_PHYSICS,
        "network_sizes": {
            "u_xi": list(core.NET_U_SIZES),
            "p_T": list(core.NET_PT_SIZES),
        },
        "parameter_counts": {
            "u_xi": int(core.FLAT_U.size),
            "p_T": int(core.FLAT_PT.size),
        },
        "sampling": {
            "PDE": int(core.PDE_POINTS.shape[0]),
            "U_Dirichlet": int(core.U_DIR_POINTS.shape[0]),
            "U_Neumann": int(core.U_NEU_POINTS.shape[0]),
            "PT_Dirichlet": int(core.PT_DIR_POINTS.shape[0]),
            "initial": int(core.IC_POINTS.shape[0]),
            "evaluation": int(core.EVAL_POINTS.shape[0]),
        },
        "loss_weighting": {
            "u": {key: float(value) for key, value in core.WEIGHTS_U.items()},
            "pT": {key: float(value) for key, value in core.WEIGHTS_PT.items()},
            "pT_residual_mode": core.PT_RESIDUAL_MODE,
        },
        "budget": {
            "fs_outer_iterations": FS_ITERATIONS,
            "fixed_early_stop_disabled": True,
            "save_fs50_and_fs100": True,
            "engd_steps_per_subproblem_per_fs": ENGD_STEPS_PER_BLOCK,
            "adam_steps_per_subproblem_per_fs": ADAM_STEPS_PER_BLOCK,
            "adam_lbfgs_switch_after_fs": ADAM_LBFGS_SWITCH_FS,
            "lbfgs_maxiter_per_subproblem_per_fs": LBFGS_MAXITER_PER_BLOCK,
        },
        "optimizer": {
            "jacobian_backend": JACOBIAN_BACKEND,
            "linear_solver": LINEAR_SOLVER,
            "gram_ridge": GRAM_RIDGE,
            "rank_safe_ridge_safety": 0.0,
            "line_search_levels": LINE_SEARCH_LEVELS if IS_ENGD else 0,
            "line_search_grid": (
                "{1,1/2,...,2^-30}" if IS_ENGD else None
            ),
            "point_block_pt": POINT_BLOCK_PT if IS_POINTWISE else None,
            "point_block_u": POINT_BLOCK_U if IS_POINTWISE else None,
            "full_jacobian_point_block_pt": (
                FULL_JAC_POINT_BLOCK_PT if IS_FULL_JACOBIAN else None
            ),
            "full_jacobian_point_block_u": (
                FULL_JAC_POINT_BLOCK_U if IS_FULL_JACOBIAN else None
            ),
            "full_jacobian_autodiff_order": (
                "vmap(jacrev(single-point local residual)); concatenate all "
                "point batches before J.T@J"
                if IS_FULL_JACOBIAN
                else None
            ),
            "eigh_relative_cutoff": (
                EIGH_RELATIVE_CUTOFF
                if LINEAR_SOLVER == "eigh-refined"
                else None
            ),
            "eigh_refinement_steps": (
                EIGH_REFINEMENT_STEPS
                if LINEAR_SOLVER == "eigh-refined"
                else None
            ),
            "adam_learning_rate": (
                ADAM_LEARNING_RATE
                if METHOD in {"adam", "adam_lbfgs"}
                else None
            ),
            "lbfgs_ftol": LBFGS_FTOL if METHOD == "adam_lbfgs" else None,
            "lbfgs_gtol": LBFGS_GTOL if METHOD == "adam_lbfgs" else None,
        },
        "evaluation": {
            "primary_metric": (
                "E_common_max=max(u1_rel_l2,u2_rel_l2,p_rel_l2,T_rel_l2)"
            ),
            "four_field_metric": (
                "E_4F_max=max(u1_rel_l2,u2_rel_l2,xi_rel_l2,p_rel_l2,T_rel_l2)"
            ),
            "TTA_thresholds": list(TTA_THRESHOLDS),
            "TTA_patience_completed_sweeps": TTA_PATIENCE,
            "TTA_evaluated_after_full_GFS_sweep_only": True,
            "fixed_time_selection": (
                "last completed GFS sweep at or before the time budget"
            ),
            "jit_warmup_excluded": True,
            "test_error_evaluation_excluded": True,
        },
        "timing": {
            "effective_training_seconds": LEDGER.effective_training_seconds,
            "context_seconds": LEDGER.context_seconds,
            "optimizer_seconds": LEDGER.optimizer_seconds,
            "convergence_seconds": LEDGER.convergence_seconds,
            "evaluation_excluded_seconds": LEDGER.evaluation_excluded_seconds,
            "jit_warmup_excluded_seconds": LEDGER.jit_warmup_excluded_seconds,
            "solver_wall_seconds_after_warmup": solver_wall_seconds,
            "profile_totals": profile_totals(),
        },
        "final_accuracy": final_row,
        "fs50_to_fs100": fs50_to_fs100,
        "memory": process_memory_snapshot(),
        "environment": {
            key: value
            for key, value in os.environ.items()
            if key.startswith("THP_") or key == "CUDA_VISIBLE_DEVICES"
        },
    }
    with (core.RESULT_DIR / "metadata.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)
    with (core.RESULT_DIR / "final_errors.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(final_row, file, indent=2, ensure_ascii=False)
    with (core.RESULT_DIR / "fs50_to_fs100.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(fs50_to_fs100, file, indent=2, ensure_ascii=False)


# =============================================================================
# 10. Entry point
# =============================================================================


def main() -> None:
    core.LOGGER.info("============================================================")
    core.LOGGER.info("Group-1 method: %s", METHOD_LABEL)
    core.LOGGER.info(
        "shared formulation: 4F-GFS, N_uXi=%s, N_pT=%s",
        core.NET_U_SIZES,
        core.NET_PT_SIZES,
    )
    core.LOGGER.info("physical parameters=%s", ACTUAL_PHYSICS)
    core.LOGGER.info("exact_case=%s seed=%d", core.EXACT_CASE, core.SEED)
    core.LOGGER.info("WEIGHTS_U=%s", core.WEIGHTS_U)
    core.LOGGER.info("WEIGHTS_PT=%s", core.WEIGHTS_PT)
    core.LOGGER.info(
        "budgets: FS=%d, ENGD/subproblem/FS=%d, "
        "Adam/subproblem/FS=%d, Adam->L-BFGS switch=%d, "
        "L-BFGS maxiter/subproblem/FS=%d",
        FS_ITERATIONS,
        ENGD_STEPS_PER_BLOCK,
        ADAM_STEPS_PER_BLOCK,
        ADAM_LBFGS_SWITCH_FS,
        LBFGS_MAXITER_PER_BLOCK,
    )
    core.LOGGER.info(
        "backend=%s solver=%s ridge=%.3e line-search-levels=%d",
        JACOBIAN_BACKEND,
        LINEAR_SOLVER,
        GRAM_RIDGE,
        LINE_SEARCH_LEVELS if IS_ENGD else 0,
    )
    if IS_FULL_JACOBIAN:
        core.LOGGER.info(
            "full-J construction=vmap(jacrev(local residual)) then "
            "concatenate; point blocks: pT=%d uXi=%d",
            FULL_JAC_POINT_BLOCK_PT,
            FULL_JAC_POINT_BLOCK_U,
        )
    core.LOGGER.info(
        "TTA thresholds=%s patience=%d full-sweep-only=True",
        TTA_THRESHOLDS,
        TTA_PATIENCE,
    )
    core.LOGGER.info("============================================================")

    device = jax.devices()[0]
    if not core.FAST_DEBUG and device.platform != "gpu":
        raise RuntimeError(
            "Production Group-1 timing requires a visible JAX GPU. "
            "Use THP_FAST_DEBUG=1 only for a CPU syntax/smoke run."
        )

    warmup()
    core.LOGGER.info(
        "JIT warmup excluded: %.3f s",
        LEDGER.jit_warmup_excluded_seconds,
    )
    flat_u, flat_pt, solver_wall_seconds = train()
    save_outputs(flat_u, flat_pt, solver_wall_seconds)
    core.LOGGER.info(
        "finished %s: effective=%.3f s [context=%.3f optimizer=%.3f "
        "convergence=%.3f], evaluation(excluded)=%.3f s, "
        "JIT(excluded)=%.3f s, solver-wall(after warmup)=%.3f s, results=%s",
        METHOD_LABEL,
        LEDGER.effective_training_seconds,
        LEDGER.context_seconds,
        LEDGER.optimizer_seconds,
        LEDGER.convergence_seconds,
        LEDGER.evaluation_excluded_seconds,
        LEDGER.jit_warmup_excluded_seconds,
        solver_wall_seconds,
        core.RESULT_DIR,
    )


if __name__ == "__main__":
    main()
