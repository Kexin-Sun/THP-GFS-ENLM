"""
Monolithic four-field THP PINN with streamed energy-natural training.
Positive fixed damping gives the ENLM configuration used in this study.

This is the coupled baseline for the Group-2 architecture/formulation study.
It imports the validated physics, data, ENGD linear solver, and reporting
constants from ``thp-four-field-core.py`` and changes only the training map:

* ``THP_COUPLED_ARCH=dual`` uses
  N_uxi: [3,32,32,3] and N_pt: [3,32,32,2], but optimizes their concatenated
  parameter vector against the complete five-equation residual in one ENGD
  step.
* ``THP_COUPLED_ARCH=single`` uses one N_all: [3,32,32,5] network with output
  order (u1,u2,xi,p,T), again optimized against the same complete residual.

The pointwise-Jacrev backend accumulates the exact J^T J and J^T r without
materializing the global residual Jacobian.  JIT warm-up and evaluation-grid
accuracy are excluded from effective training time, matching the GFS scripts.

Examples
--------
THP_COUPLED_ARCH=dual python thp-coupled.py
THP_COUPLED_ARCH=single python thp-coupled.py
THP_FAST_DEBUG=1 THP_SAVE_PLOTS=0 THP_COUPLED_ARCH=single \
    python thp-coupled.py
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Resolve run identity before importing the core: the core creates RESULT_DIR
# and the logger during import.
ARCHITECTURE = os.environ.get("THP_COUPLED_ARCH", "dual").strip().lower()
if ARCHITECTURE not in {"single", "dual"}:
    raise ValueError("THP_COUPLED_ARCH must be 'single' or 'dual'.")

_seed_for_name = int(os.environ.get("THP_SEED", "12345"))
os.environ.setdefault(
    "THP_RUN_NAME",
    (
        f"group2_coupled_{ARCHITECTURE}_seed{_seed_for_name}_"
        f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_pid{os.getpid()}"
    ),
)

SCRIPT_DIR = Path(__file__).resolve().parent
CORE_PATH = Path(
    os.environ.get(
        "THP_FOUR_FIELD_CORE",
        str(SCRIPT_DIR.parent / "thp-four-field-core.py"),
    )
).resolve()
if not CORE_PATH.is_file():
    raise FileNotFoundError(
        f"Cannot find {CORE_PATH}. Keep thp-coupled.py in group2_formulation/ and "
        "thp-four-field-core.py at the repository root, or set "
        "THP_FOUR_FIELD_CORE."
    )

_spec = importlib.util.spec_from_file_location("thp_four_field_core", CORE_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot import four-field core from {CORE_PATH}.")
core = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = core
_spec.loader.exec_module(core)

# Reuse the exact runtime imported by the core.  In particular, JAX FP64 and
# XLA allocator settings have already been established before JAX import.
jax = core.jax
jnp = core.jnp
np = core.np
plt = core.plt
random = core.random
jit = core.jit
vmap = core.vmap
jacfwd = core.jacfwd
jacrev = core.jacrev
hessian = core.hessian
lax = core.lax
ravel_pytree = core.ravel_pytree
Array = core.Array
Params = core.Params

LOGGER = core.LOGGER
RESULT_DIR = core.RESULT_DIR


# -----------------------------------------------------------------------------
# Experiment budget and monolithic network initialization
# -----------------------------------------------------------------------------

COUPLED_ROUNDS = int(
    os.environ.get("THP_COUPLED_ROUNDS", str(core.FS_ITERATIONS))
)
COUPLED_ENGD_STEPS = int(
    os.environ.get("THP_COUPLED_ENGD_STEPS", str(core.ENGD_STEPS_PT))
)
POINT_BLOCK_COUPLED = int(
    os.environ.get("THP_POINT_JAC_BLOCK_COUPLED", "128")
)
ENABLE_COUPLED_EARLY_STOP = (
    os.environ.get("THP_ENABLE_COUPLED_EARLY_STOP", "0") == "1"
)

if COUPLED_ROUNDS <= 0:
    raise ValueError("THP_COUPLED_ROUNDS must be positive.")
if COUPLED_ENGD_STEPS <= 0:
    raise ValueError("THP_COUPLED_ENGD_STEPS must be positive.")
if POINT_BLOCK_COUPLED <= 0:
    raise ValueError("THP_POINT_JAC_BLOCK_COUPLED must be positive.")

if core.FAST_DEBUG:
    COUPLED_ROUNDS = min(COUPLED_ROUNDS, 2)
    COUPLED_ENGD_STEPS = min(COUPLED_ENGD_STEPS, 2)

NET_SINGLE_SIZES = [3, 32, 32, 5]
NET_UXI_SIZES = list(core.NET_U_SIZES)
NET_PT_SIZES = list(core.NET_PT_SIZES)

if ARCHITECTURE == "dual":
    INITIAL_FLAT = jnp.concatenate((core.FLAT_U, core.FLAT_PT))
    DUAL_UXI_PARAMETER_COUNT = int(core.FLAT_U.size)
    DUAL_PT_PARAMETER_COUNT = int(core.FLAT_PT.size)
    SINGLE_UNRAVEL = None
else:
    # Match the first network key used by the dual/GFS initialization.  Exact
    # parameter matching is impossible because the output layer has a different
    # shape, but the seed and Xavier rule are identical.
    _, single_key, _ = random.split(random.PRNGKey(core.SEED), 3)
    single_params = core.init_mlp(NET_SINGLE_SIZES, single_key)
    INITIAL_FLAT, SINGLE_UNRAVEL = ravel_pytree(single_params)
    DUAL_UXI_PARAMETER_COUNT = 0
    DUAL_PT_PARAMETER_COUNT = 0

PARAMETER_COUNT = int(INITIAL_FLAT.size)


def split_dual_flat(flat: Array) -> Tuple[Array, Array]:
    """Return the u/xi and p/T subvectors of a dual-network parameter vector."""
    cut = DUAL_UXI_PARAMETER_COUNT
    return flat[:cut], flat[cut:]


def fields_value(flat: Array, xyt: Array) -> Array:
    """Predict fields in the fixed order (u1,u2,xi,p,T)."""
    if ARCHITECTURE == "dual":
        flat_uxi, flat_pt = split_dual_flat(flat)
        uxi = core.predict(core.UNRAVEL_U(flat_uxi), xyt)
        pt = core.predict(core.UNRAVEL_PT(flat_pt), xyt)
        return jnp.concatenate((uxi, pt))
    return core.predict(SINGLE_UNRAVEL(flat), xyt)


FIELDS_JAC = jacfwd(fields_value, argnums=1)
FIELDS_HESS = hessian(fields_value, argnums=1)
V_FIELDS = jit(vmap(fields_value, in_axes=(None, 0)))
V_FIELDS_JAC = jit(vmap(FIELDS_JAC, in_axes=(None, 0)))


# -----------------------------------------------------------------------------
# Complete four-field residual, boundary conditions, and exact Gram backend
# -----------------------------------------------------------------------------

# Fixed source data prevents analytic high-order AD from re-entering every
# pointwise Jacobian trace.  Columns are (-f1,-f2,-g,-Hs).
PDE_DATA = jnp.concatenate(
    (
        -core.BODY_FORCE_PDE,
        -core.MASS_SOURCE_PDE[:, None],
        -core.HEAT_SOURCE_PDE[:, None],
    ),
    axis=1,
)


def pde_local_flat(flat: Array, xyt: Array, source_data: Array) -> Array:
    values = fields_value(flat, xyt)
    jacobian = FIELDS_JAC(flat, xyt)
    hess = FIELDS_HESS(flat, xyt)

    momentum = core.four_field_mechanics_operator(
        hess[:2], jacobian[2, :2]
    ) + source_data[:2]
    constraint = (
        jacobian[0, 0]
        + jacobian[1, 1]
        + core.INV_LAMBDA * values[2]
        - core.ALPHA * core.INV_LAMBDA * values[3]
        - core.BETA * core.INV_LAMBDA * values[4]
    )

    p_t, temperature_t = jacobian[3, 2], jacobian[4, 2]
    lap_p = hess[3, 0, 0] + hess[3, 1, 1]
    lap_temperature = hess[4, 0, 0] + hess[4, 1, 1]
    xi_t = jacobian[2, 2]
    raw_flow_heat = jnp.array(
        [
            core.C_ALPHA * p_t
            + core.C_ALPHA_BETA * temperature_t
            - core.KAPPA * lap_p
            - core.ALPHA * core.INV_LAMBDA * xi_t
            + source_data[2],
            core.C_ALPHA_BETA * p_t
            + core.C_BETA * temperature_t
            - core.THETA * lap_temperature
            - core.BETA * core.INV_LAMBDA * xi_t
            + source_data[3],
        ]
    )
    flow_heat = core.transform_flow_heat_training_residual(raw_flow_heat)

    mechanics_scaled = core.SCALE_U_PDE * jnp.concatenate(
        (momentum, jnp.array([constraint]))
    )
    flow_heat_scaled = core.SCALE_PT_PDE * flow_heat
    return jnp.concatenate((mechanics_scaled, flow_heat_scaled))


def raw_flow_heat_local(flat: Array, xyt: Array, source_data: Array) -> Array:
    jacobian = FIELDS_JAC(flat, xyt)
    hess = FIELDS_HESS(flat, xyt)
    p_t, temperature_t = jacobian[3, 2], jacobian[4, 2]
    lap_p = hess[3, 0, 0] + hess[3, 1, 1]
    lap_temperature = hess[4, 0, 0] + hess[4, 1, 1]
    xi_t = jacobian[2, 2]
    return jnp.array(
        [
            core.C_ALPHA * p_t
            + core.C_ALPHA_BETA * temperature_t
            - core.KAPPA * lap_p
            - core.ALPHA * core.INV_LAMBDA * xi_t
            + source_data[2],
            core.C_ALPHA_BETA * p_t
            + core.C_BETA * temperature_t
            - core.THETA * lap_temperature
            - core.BETA * core.INV_LAMBDA * xi_t
            + source_data[3],
        ]
    )


def traction_from_flat(flat: Array, xyt: Array) -> Array:
    values = fields_value(flat, xyt)
    jacobian = FIELDS_JAC(flat, xyt)
    du1_dx, du1_dy = jacobian[0, 0], jacobian[0, 1]
    du2_dx, du2_dy = jacobian[1, 0], jacobian[1, 1]
    xi_value = values[2]
    sigma = jnp.array(
        [
            [2.0 * core.MU * du1_dx - xi_value,
             core.MU * (du1_dy + du2_dx)],
            [core.MU * (du1_dy + du2_dx),
             2.0 * core.MU * du2_dy - xi_value],
        ]
    )
    return sigma @ core.outward_normal(xyt)


def u_dir_local_flat(flat: Array, xyt: Array, target: Array) -> Array:
    return core.SCALE_U_DIR * (fields_value(flat, xyt)[:2] - target)


def u_neu_local_flat(flat: Array, xyt: Array, target: Array) -> Array:
    return core.SCALE_U_NEU * (traction_from_flat(flat, xyt) - target)


def u_ic_local_flat(flat: Array, xyt: Array, target: Array) -> Array:
    return core.SCALE_U_IC * (fields_value(flat, xyt)[:3] - target)


def pt_dir_local_flat(flat: Array, xyt: Array, target: Array) -> Array:
    return core.SCALE_PT_DIR * (fields_value(flat, xyt)[3:] - target)


def pt_ic_local_flat(flat: Array, xyt: Array, target: Array) -> Array:
    return core.SCALE_PT_IC * (fields_value(flat, xyt)[3:] - target)


V_PDE_LOCAL = jit(vmap(pde_local_flat, in_axes=(None, 0, 0)))
V_RAW_FLOW_HEAT_LOCAL = jit(
    vmap(raw_flow_heat_local, in_axes=(None, 0, 0))
)
V_U_DIR_LOCAL = jit(vmap(u_dir_local_flat, in_axes=(None, 0, 0)))
V_U_NEU_LOCAL = jit(vmap(u_neu_local_flat, in_axes=(None, 0, 0)))
V_U_IC_LOCAL = jit(vmap(u_ic_local_flat, in_axes=(None, 0, 0)))
V_PT_DIR_LOCAL = jit(vmap(pt_dir_local_flat, in_axes=(None, 0, 0)))
V_PT_IC_LOCAL = jit(vmap(pt_ic_local_flat, in_axes=(None, 0, 0)))


@jit
def residual_coupled_flat(flat: Array) -> Array:
    return jnp.concatenate(
        (
            V_PDE_LOCAL(flat, core.PDE_POINTS, PDE_DATA).reshape(-1),
            V_U_DIR_LOCAL(
                flat, core.U_DIR_POINTS, core.U_DIR_TARGET
            ).reshape(-1),
            V_U_NEU_LOCAL(
                flat, core.U_NEU_POINTS, core.U_NEU_EXACT_TRACTION
            ).reshape(-1),
            V_U_IC_LOCAL(flat, core.IC_POINTS, core.U_IC_TARGET).reshape(-1),
            V_PT_DIR_LOCAL(
                flat, core.PT_DIR_POINTS, core.PT_DIR_TARGET
            ).reshape(-1),
            V_PT_IC_LOCAL(flat, core.IC_POINTS, core.PT_IC_TARGET).reshape(-1),
        )
    )


JACOBIAN_COUPLED_FLAT = jit(jacfwd(residual_coupled_flat))


@jit
def candidate_losses_coupled(candidates: Array) -> Array:
    return vmap(
        lambda candidate: core.half_squared_norm(
            residual_coupled_flat(candidate)
        )
    )(candidates)


PDE_LOCAL_JAC_AUX = jacrev(
    core._with_aux(pde_local_flat), argnums=0, has_aux=True
)
U_DIR_LOCAL_JAC_AUX = jacrev(
    core._with_aux(u_dir_local_flat), argnums=0, has_aux=True
)
U_NEU_LOCAL_JAC_AUX = jacrev(
    core._with_aux(u_neu_local_flat), argnums=0, has_aux=True
)
U_IC_LOCAL_JAC_AUX = jacrev(
    core._with_aux(u_ic_local_flat), argnums=0, has_aux=True
)
PT_DIR_LOCAL_JAC_AUX = jacrev(
    core._with_aux(pt_dir_local_flat), argnums=0, has_aux=True
)
PT_IC_LOCAL_JAC_AUX = jacrev(
    core._with_aux(pt_ic_local_flat), argnums=0, has_aux=True
)


@dataclass(frozen=True)
class PointwiseCoupledContext:
    pde_points: Array
    pde_data: Array
    pde_mask: Array
    u_dir_points: Array
    u_dir_targets: Array
    u_dir_mask: Array
    u_neu_points: Array
    u_neu_targets: Array
    u_neu_mask: Array
    u_ic_points: Array
    u_ic_targets: Array
    u_ic_mask: Array
    pt_dir_points: Array
    pt_dir_targets: Array
    pt_dir_mask: Array
    pt_ic_points: Array
    pt_ic_targets: Array
    pt_ic_mask: Array


def build_pointwise_coupled_context(block_size: int) -> PointwiseCoupledContext:
    blocks = (
        core._blockify(core.PDE_POINTS, PDE_DATA, block_size),
        core._blockify(
            core.U_DIR_POINTS, core.U_DIR_TARGET, block_size
        ),
        core._blockify(
            core.U_NEU_POINTS, core.U_NEU_EXACT_TRACTION, block_size
        ),
        core._blockify(core.IC_POINTS, core.U_IC_TARGET, block_size),
        core._blockify(
            core.PT_DIR_POINTS, core.PT_DIR_TARGET, block_size
        ),
        core._blockify(core.IC_POINTS, core.PT_IC_TARGET, block_size),
    )
    for triple in blocks:
        for value in triple:
            value.block_until_ready()
    flattened = tuple(value for triple in blocks for value in triple)
    return PointwiseCoupledContext(*flattened)


def _scan_local_blocks(
    flat: Array,
    point_blocks: Array,
    data_blocks: Array,
    mask_blocks: Array,
    jacobian_aux,
) -> Tuple[Array, Array, Array]:
    initial = (
        jnp.array(0.0, dtype=flat.dtype),
        jnp.zeros((flat.size, flat.size), dtype=flat.dtype),
        jnp.zeros((flat.size,), dtype=flat.dtype),
    )

    def body(carry, values):
        points, data, mask = values
        jacobian, residual = vmap(
            jacobian_aux, in_axes=(None, 0, 0)
        )(flat, points, data)
        block = core._block_contribution(jacobian, residual, mask)
        return tuple(a + b for a, b in zip(carry, block)), None

    result, _ = lax.scan(
        body, initial, (point_blocks, data_blocks, mask_blocks)
    )
    return result


@jit
def _pointwise_coupled_kernel(
    flat: Array,
    pde_points: Array,
    pde_data: Array,
    pde_mask: Array,
    u_dir_points: Array,
    u_dir_targets: Array,
    u_dir_mask: Array,
    u_neu_points: Array,
    u_neu_targets: Array,
    u_neu_mask: Array,
    u_ic_points: Array,
    u_ic_targets: Array,
    u_ic_mask: Array,
    pt_dir_points: Array,
    pt_dir_targets: Array,
    pt_dir_mask: Array,
    pt_ic_points: Array,
    pt_ic_targets: Array,
    pt_ic_mask: Array,
) -> Tuple[Array, Array, Array]:
    contributions = (
        _scan_local_blocks(
            flat,
            pde_points,
            pde_data,
            pde_mask,
            PDE_LOCAL_JAC_AUX,
        ),
        _scan_local_blocks(
            flat,
            u_dir_points,
            u_dir_targets,
            u_dir_mask,
            U_DIR_LOCAL_JAC_AUX,
        ),
        _scan_local_blocks(
            flat,
            u_neu_points,
            u_neu_targets,
            u_neu_mask,
            U_NEU_LOCAL_JAC_AUX,
        ),
        _scan_local_blocks(
            flat,
            u_ic_points,
            u_ic_targets,
            u_ic_mask,
            U_IC_LOCAL_JAC_AUX,
        ),
        _scan_local_blocks(
            flat,
            pt_dir_points,
            pt_dir_targets,
            pt_dir_mask,
            PT_DIR_LOCAL_JAC_AUX,
        ),
        _scan_local_blocks(
            flat,
            pt_ic_points,
            pt_ic_targets,
            pt_ic_mask,
            PT_IC_LOCAL_JAC_AUX,
        ),
    )
    return tuple(sum(values) for values in zip(*contributions))


def pointwise_coupled_gram_gradient(
    flat: Array,
    context: PointwiseCoupledContext,
) -> Tuple[Array, Array, Array]:
    return _pointwise_coupled_kernel(
        flat,
        context.pde_points,
        context.pde_data,
        context.pde_mask,
        context.u_dir_points,
        context.u_dir_targets,
        context.u_dir_mask,
        context.u_neu_points,
        context.u_neu_targets,
        context.u_neu_mask,
        context.u_ic_points,
        context.u_ic_targets,
        context.u_ic_mask,
        context.pt_dir_points,
        context.pt_dir_targets,
        context.pt_dir_mask,
        context.pt_ic_points,
        context.pt_ic_targets,
        context.pt_ic_mask,
    )


ENGD_CONFIG = core.ENGDConfig(
    line_search_levels=core.LINE_SEARCH_LEVELS,
    gram_ridge=core.GRAM_RIDGE,
    solver=core.ENGD_SOLVER,
    compare_lstsq_reference=core.COMPARE_LSTSQ_REFERENCE,
    relative_cutoff=core.RELATIVE_CUTOFF,
    refinement_steps=core.REFINEMENT_STEPS,
    lstsq_rcond=core.LSTSQ_RCOND,
    jacobian_backend=core.JACOBIAN_BACKEND,
    gradient_tolerance=1.0e-14,
    require_loss_decrease=True,
)


# -----------------------------------------------------------------------------
# Diagnostics and common accuracy metrics
# -----------------------------------------------------------------------------


def coupled_loss_breakdown(flat: Array) -> Dict[str, float]:
    pde = V_PDE_LOCAL(flat, core.PDE_POINTS, PDE_DATA)
    u_dir = V_U_DIR_LOCAL(flat, core.U_DIR_POINTS, core.U_DIR_TARGET)
    u_neu = V_U_NEU_LOCAL(
        flat, core.U_NEU_POINTS, core.U_NEU_EXACT_TRACTION
    )
    u_ic = V_U_IC_LOCAL(flat, core.IC_POINTS, core.U_IC_TARGET)
    pt_dir = V_PT_DIR_LOCAL(flat, core.PT_DIR_POINTS, core.PT_DIR_TARGET)
    pt_ic = V_PT_IC_LOCAL(flat, core.IC_POINTS, core.PT_IC_TARGET)
    raw_pt = V_RAW_FLOW_HEAT_LOCAL(flat, core.PDE_POINTS, PDE_DATA)

    values = {
        "u_pde": float(core.half_squared_norm(pde[:, :3].reshape(-1))),
        "u_dirichlet": float(core.half_squared_norm(u_dir.reshape(-1))),
        "u_neumann": float(core.half_squared_norm(u_neu.reshape(-1))),
        "u_ic": float(core.half_squared_norm(u_ic.reshape(-1))),
        "pt_pde": float(core.half_squared_norm(pde[:, 3:].reshape(-1))),
        "pt_dirichlet": float(core.half_squared_norm(pt_dir.reshape(-1))),
        "pt_ic": float(core.half_squared_norm(pt_ic.reshape(-1))),
        "pt_coupled_raw_pde": core.residual_block_loss(
            raw_pt, core.WEIGHTS_PT["pde"], core.MEASURE_PDE
        ),
    }
    values["loss_u"] = sum(
        values[name]
        for name in ("u_pde", "u_dirichlet", "u_neumann", "u_ic")
    )
    values["loss_pt"] = sum(
        values[name]
        for name in ("pt_pde", "pt_dirichlet", "pt_ic")
    )
    values["loss_total"] = values["loss_u"] + values["loss_pt"]
    return values


def evaluate_errors(flat: Array) -> Dict[str, float]:
    prediction = V_FIELDS(flat, core.EVAL_POINTS)
    jac_prediction = V_FIELDS_JAC(flat, core.EVAL_POINTS)
    u_reference = core.V_U_EXACT(core.EVAL_POINTS)
    xi_reference = core.V_XI_EXACT(core.EVAL_POINTS)
    pt_reference = core.V_PT_EXACT(core.EVAL_POINTS)

    u_l2, u_rel = core.absolute_relative_l2_vector(
        prediction[:, :2], u_reference
    )
    u1_l2, u1_rel = core.absolute_relative_l2_scalar(
        prediction[:, 0], u_reference[:, 0]
    )
    u2_l2, u2_rel = core.absolute_relative_l2_scalar(
        prediction[:, 1], u_reference[:, 1]
    )
    xi_l2, xi_rel = core.absolute_relative_l2_scalar(
        prediction[:, 2], xi_reference
    )
    p_l2, p_rel = core.absolute_relative_l2_scalar(
        prediction[:, 3], pt_reference[:, 0]
    )
    temperature_l2, temperature_rel = core.absolute_relative_l2_scalar(
        prediction[:, 4], pt_reference[:, 1]
    )

    u_jac_reference = core.V_U_EXACT_JAC(core.EVAL_POINTS)
    xi_jac_reference = vmap(core.XI_EXACT_GRAD)(core.EVAL_POINTS)
    p_jac_reference = vmap(core.P_EXACT_GRAD)(core.EVAL_POINTS)
    t_jac_reference = vmap(core.T_EXACT_GRAD)(core.EVAL_POINTS)

    def spatial_gradient_error_sq(
        predicted: Array, reference: Array
    ) -> float:
        return core.MEASURE_PDE * float(
            jnp.mean(jnp.sum((predicted - reference) ** 2, axis=-1))
        )

    u_grad_sq = core.MEASURE_PDE * float(
        jnp.mean(
            jnp.sum(
                (jac_prediction[:, :2, :2] - u_jac_reference[:, :, :2])
                ** 2,
                axis=(1, 2),
            )
        )
    )
    u1_grad_sq = spatial_gradient_error_sq(
        jac_prediction[:, 0, :2], u_jac_reference[:, 0, :2]
    )
    u2_grad_sq = spatial_gradient_error_sq(
        jac_prediction[:, 1, :2], u_jac_reference[:, 1, :2]
    )
    xi_grad_sq = spatial_gradient_error_sq(
        jac_prediction[:, 2, :2], xi_jac_reference[:, :2]
    )
    p_grad_sq = spatial_gradient_error_sq(
        jac_prediction[:, 3, :2], p_jac_reference[:, :2]
    )
    t_grad_sq = spatial_gradient_error_sq(
        jac_prediction[:, 4, :2], t_jac_reference[:, :2]
    )

    return {
        "u_l2": u_l2,
        "u_rel_l2": u_rel,
        "u_h1": math.sqrt(u_l2 * u_l2 + u_grad_sq),
        "u1_l2": u1_l2,
        "u1_rel_l2": u1_rel,
        "u1_h1": math.sqrt(u1_l2 * u1_l2 + u1_grad_sq),
        "u2_l2": u2_l2,
        "u2_rel_l2": u2_rel,
        "u2_h1": math.sqrt(u2_l2 * u2_l2 + u2_grad_sq),
        "xi_l2": xi_l2,
        "xi_rel_l2": xi_rel,
        "xi_h1": math.sqrt(xi_l2 * xi_l2 + xi_grad_sq),
        "p_l2": p_l2,
        "p_rel_l2": p_rel,
        "p_h1": math.sqrt(p_l2 * p_l2 + p_grad_sq),
        "T_l2": temperature_l2,
        "T_rel_l2": temperature_rel,
        "T_h1": math.sqrt(
            temperature_l2 * temperature_l2 + t_grad_sq
        ),
    }


def prediction_change(flat_new: Array, flat_old: Array) -> float:
    new = V_FIELDS(flat_new, core.EVAL_POINTS)
    old = V_FIELDS(flat_old, core.EVAL_POINTS)
    numerator = float(jnp.linalg.norm(new - old))
    denominator = float(jnp.linalg.norm(old))
    return numerator / max(denominator, 1.0e-30)


@dataclass
class Timing:
    jit_warmup_seconds_excluded: float = 0.0
    context_seconds: float = 0.0
    assembly_seconds: float = 0.0
    solve_seconds: float = 0.0
    line_search_seconds: float = 0.0
    convergence_check_seconds: float = 0.0
    evaluation_seconds_excluded: float = 0.0
    reference_validation_seconds_excluded: float = 0.0
    total_solver_wall_seconds: float = 0.0

    @property
    def optimizer_seconds(self) -> float:
        return (
            self.context_seconds
            + self.assembly_seconds
            + self.solve_seconds
            + self.line_search_seconds
        )

    @property
    def effective_training_seconds(self) -> float:
        return self.optimizer_seconds + self.convergence_check_seconds


HISTORY_ROWS: List[Dict[str, object]] = []
PROFILE_ROWS: List[Dict[str, object]] = []


def warmup_training_kernels(
    flat: Array,
    context: Optional[PointwiseCoupledContext],
) -> float:
    start = time.perf_counter()
    gram_fn = None
    if core.JACOBIAN_BACKEND == "pointwise":
        gram_fn = lambda z: pointwise_coupled_gram_gradient(z, context)
    warm, _ = core.explicit_engd_step(
        flat,
        residual_coupled_flat,
        JACOBIAN_COUPLED_FLAT,
        candidate_losses_coupled,
        ENGD_CONFIG,
        gram_gradient_fn=gram_fn,
    )
    warm.block_until_ready()
    coupled_loss_breakdown(flat)
    prediction_change(flat, flat)
    return time.perf_counter() - start


def verify_manufactured_solution() -> None:
    if not core.VERIFY_MANUFACTURED:
        return
    residual = core.V_MANUFACTURED_FOUR_FIELD_RESIDUAL(core.PDE_POINTS)
    residual.block_until_ready()
    maximum = float(jnp.max(jnp.abs(residual)))
    LOGGER.info("manufactured four-field strong residual max|r|=%.3e", maximum)
    if not math.isfinite(maximum) or maximum > 1.0e-7:
        raise RuntimeError(
            f"Manufactured four-field residual check failed: {maximum:.3e}."
        )


def train() -> Tuple[Array, Timing, str]:
    HISTORY_ROWS.clear()
    PROFILE_ROWS.clear()
    flat = jnp.array(INITIAL_FLAT)
    timing = Timing()

    LOGGER.info("============================================================")
    LOGGER.info(
        "Four-field THP monolithic coupled JS-ENGD: architecture=%s",
        ARCHITECTURE,
    )
    device = jax.devices()[0]
    LOGGER.info(
        "runtime: jax=%s backend=%s device=%s x64=%s",
        jax.__version__, jax.default_backend(), device.device_kind,
        jax.config.jax_enable_x64,
    )
    if device.platform != "gpu":
        LOGGER.warning(
            "JAX is running on %s; this run is unsuitable for paper timing.",
            device.platform,
        )
    LOGGER.info(
        "physics: E=%g nu=%g lambda=%.6e mu=%.6e alpha=%g beta=%g "
        "a0=%g b0=%g c0=%g K=%g Theta=%g exact_case=%s",
        core.E, core.NU, core.LAMBDA, core.MU, core.ALPHA, core.BETA,
        core.A0, core.B0, core.C0, core.KAPPA, core.THETA, core.EXACT_CASE,
    )
    LOGGER.info(
        "network: single=%s dual_uXi=%s dual_pT=%s parameters=%d "
        "[uXi=%d,pT=%d]",
        NET_SINGLE_SIZES if ARCHITECTURE == "single" else None,
        NET_UXI_SIZES if ARCHITECTURE == "dual" else None,
        NET_PT_SIZES if ARCHITECTURE == "dual" else None,
        PARAMETER_COUNT,
        DUAL_UXI_PARAMETER_COUNT,
        DUAL_PT_PARAMETER_COUNT,
    )
    LOGGER.info(
        "budget: rounds=%d steps/round=%d equivalent_iterations=%d",
        COUPLED_ROUNDS,
        COUPLED_ENGD_STEPS,
        COUPLED_ROUNDS * COUPLED_ENGD_STEPS,
    )
    LOGGER.info(
        "points: PDE=%d IC=%d uDir=%d uNeu=%d pTDir=%d eval=%d",
        core.PDE_POINTS.shape[0], core.IC_POINTS.shape[0],
        core.U_DIR_POINTS.shape[0], core.U_NEU_POINTS.shape[0],
        core.PT_DIR_POINTS.shape[0], core.EVAL_POINTS.shape[0],
    )
    LOGGER.info(
        "ENGD: solver=%s ridge=%.3e relative_cutoff=%.3e refinement=%d "
        "line_search=%d backend=%s block=%d",
        core.ENGD_SOLVER, core.GRAM_RIDGE, core.RELATIVE_CUTOFF,
        core.REFINEMENT_STEPS, core.LINE_SEARCH_LEVELS,
        core.JACOBIAN_BACKEND, POINT_BLOCK_COUPLED,
    )
    LOGGER.info(
        "weights: u=%s pT=%s residual_mode=%s",
        core.WEIGHTS_U, core.WEIGHTS_PT, core.PT_RESIDUAL_MODE,
    )

    verify_manufactured_solution()

    context: Optional[PointwiseCoupledContext] = None
    if core.JACOBIAN_BACKEND == "pointwise":
        context_start = time.perf_counter()
        context = build_pointwise_coupled_context(POINT_BLOCK_COUPLED)
        timing.context_seconds = time.perf_counter() - context_start

    timing.jit_warmup_seconds_excluded = warmup_training_kernels(flat, context)
    LOGGER.info(
        "JIT warmup excluded from training time: %.3f s",
        timing.jit_warmup_seconds_excluded,
    )
    LOGGER.info("============================================================")

    solver_wall_start = time.perf_counter()
    stop_reason = "maximum_rounds"
    cumulative_updates = 0
    patience_counter = 0

    for round_index in range(1, COUPLED_ROUNDS + 1):
        flat_old = jnp.array(flat)
        accepted_in_round = 0
        LOGGER.info(
            "--- coupled round %d/%d (equivalent iterations %d-%d) ---",
            round_index,
            COUPLED_ROUNDS,
            (round_index - 1) * COUPLED_ENGD_STEPS + 1,
            round_index * COUPLED_ENGD_STEPS,
        )

        gram_fn = None
        if core.JACOBIAN_BACKEND == "pointwise":
            gram_fn = lambda z: pointwise_coupled_gram_gradient(z, context)

        for inner_index in range(1, COUPLED_ENGD_STEPS + 1):
            flat, stats = core.explicit_engd_step(
                flat,
                residual_coupled_flat,
                JACOBIAN_COUPLED_FLAT,
                candidate_losses_coupled,
                ENGD_CONFIG,
                gram_gradient_fn=gram_fn,
            )
            flat.block_until_ready()
            cumulative_updates += 1
            accepted_in_round += int(stats.accepted)
            timing.assembly_seconds += stats.assembly_seconds
            timing.solve_seconds += stats.solve_seconds
            timing.line_search_seconds += stats.line_search_seconds
            timing.reference_validation_seconds_excluded += stats.reference_seconds

            profile = {
                "subproblem": "coupled",
                "round": round_index,
                "inner_iteration": inner_index,
                "equivalent_iteration": (
                    (round_index - 1) * COUPLED_ENGD_STEPS + inner_index
                ),
            }
            profile.update(asdict(stats))
            PROFILE_ROWS.append(profile)
            LOGGER.info(
                "coupled ENGD %02d: accepted=%s loss %.4e -> %.4e "
                "eta=%.3e |g|=%.3e |d|=%.3e rank=%d cond=%.3e "
                "times[A=%.3f S=%.3f LS=%.3f]",
                inner_index, stats.accepted, stats.loss_before,
                stats.loss_after, stats.step_size, stats.gradient_norm,
                stats.natural_gradient_norm, stats.effective_rank,
                stats.effective_condition, stats.assembly_seconds,
                stats.solve_seconds, stats.line_search_seconds,
            )
            if stats.step_size == 0.0:
                LOGGER.info(
                    "Coupled ENGD inner loop stopped: no decreasing line-search step."
                )
                break

        convergence_start = time.perf_counter()
        losses = coupled_loss_breakdown(flat)
        relative_change = prediction_change(flat, flat_old)
        timing.convergence_check_seconds += (
            time.perf_counter() - convergence_start
        )

        evaluation_start = time.perf_counter()
        errors = evaluate_errors(flat)
        timing.evaluation_seconds_excluded += time.perf_counter() - evaluation_start

        u_mean = 0.5 * (errors["u1_rel_l2"] + errors["u2_rel_l2"])
        pt_mean = 0.5 * (errors["p_rel_l2"] + errors["T_rel_l2"])
        common_max = max(
            errors["u1_rel_l2"], errors["u2_rel_l2"],
            errors["p_rel_l2"], errors["T_rel_l2"],
        )
        four_field_max = max(common_max, errors["xi_rel_l2"])

        stop_condition = (
            round_index >= core.FS_MIN_ITERATIONS
            and relative_change <= core.FS_EARLY_STOP_OUTER_TOL
            and losses["pt_coupled_raw_pde"]
            <= core.FS_EARLY_STOP_COUPLED_PT_TOL
            and losses["loss_u"] <= core.FS_EARLY_STOP_U_LOSS_TOL
        )
        if ENABLE_COUPLED_EARLY_STOP and stop_condition:
            patience_counter += 1
        else:
            patience_counter = 0

        row: Dict[str, object] = {
            "round": round_index,
            "equivalent_iteration": round_index * COUPLED_ENGD_STEPS,
            "engd_updates_attempted": cumulative_updates,
            "accepted_updates_in_round": accepted_in_round,
            "effective_training_seconds": timing.effective_training_seconds,
            "prediction_change": relative_change,
            "stop_condition": stop_condition,
            "stop_counter": patience_counter,
            "u_mean_rel_l2": u_mean,
            "pT_mean_rel_l2": pt_mean,
            "E_common_max": common_max,
            "E_4F_max": four_field_max,
        }
        row.update(losses)
        row.update(errors)
        HISTORY_ROWS.append(row)

        LOGGER.info(
            "round %02d losses: total=%.4e u=%.4e pT=%.4e raw-pT-PDE=%.4e",
            round_index, losses["loss_total"], losses["loss_u"],
            losses["loss_pt"], losses["pt_coupled_raw_pde"],
        )
        LOGGER.info(
            "round %02d errors: u1=%.4e u2=%.4e xi=%.4e p=%.4e T=%.4e "
            "u_mean=%.4e pT_mean=%.4e E_common=%.4e E_4F=%.4e",
            round_index, errors["u1_rel_l2"], errors["u2_rel_l2"],
            errors["xi_rel_l2"], errors["p_rel_l2"], errors["T_rel_l2"],
            u_mean, pt_mean, common_max, four_field_max,
        )
        LOGGER.info(
            "effective training time=%.2f s [context=%.2f assembly=%.2f "
            "solve=%.2f line-search=%.2f convergence=%.2f]; "
            "evaluation excluded=%.2f s",
            timing.effective_training_seconds, timing.context_seconds,
            timing.assembly_seconds, timing.solve_seconds,
            timing.line_search_seconds, timing.convergence_check_seconds,
            timing.evaluation_seconds_excluded,
        )

        if (
            ENABLE_COUPLED_EARLY_STOP
            and patience_counter >= core.FS_EARLY_STOP_PATIENCE
        ):
            stop_reason = "adaptive_convergence"
            break
        if accepted_in_round == 0:
            stop_reason = "line_search_stagnation"
            LOGGER.info(
                "Coupled training stopped: the parameter vector did not change "
                "during the complete round."
            )
            break

    timing.total_solver_wall_seconds = time.perf_counter() - solver_wall_start
    return flat, timing, stop_reason


# -----------------------------------------------------------------------------
# Machine-readable reports and figures
# -----------------------------------------------------------------------------


def method_name() -> str:
    return f"THP-four-field-coupled-{ARCHITECTURE}-network-JS-ENGD"


def save_rows(path_stem: str, rows: List[Dict[str, object]]) -> None:
    with open(RESULT_DIR / f"{path_stem}.json", "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)
    if rows:
        with open(
            RESULT_DIR / f"{path_stem}.csv", "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def save_checkpoint(flat: Array) -> None:
    if ARCHITECTURE == "dual":
        flat_uxi, flat_pt = split_dual_flat(flat)
        np.savez(
            RESULT_DIR / "final_model.npz",
            architecture=ARCHITECTURE,
            flat_all=np.asarray(flat),
            flat_uxi=np.asarray(flat_uxi),
            flat_pt=np.asarray(flat_pt),
            net_uxi_sizes=np.asarray(NET_UXI_SIZES),
            net_pt_sizes=np.asarray(NET_PT_SIZES),
        )
    else:
        np.savez(
            RESULT_DIR / "final_model.npz",
            architecture=ARCHITECTURE,
            flat_all=np.asarray(flat),
            net_all_sizes=np.asarray(NET_SINGLE_SIZES),
        )


def fixed_budget_rows() -> List[Dict[str, object]]:
    by_round = {int(row["round"]): row for row in HISTORY_ROWS}
    rows: List[Dict[str, object]] = []
    for checkpoint in core.FIXED_BUDGET_CHECKPOINTS:
        source = by_round.get(int(checkpoint))
        row: Dict[str, object] = {
            "run_name": RESULT_DIR.name,
            "method": method_name(),
            "architecture": ARCHITECTURE,
            "seed": core.SEED,
            "requested_round": int(checkpoint),
            "requested_equivalent_iteration": int(checkpoint)
            * COUPLED_ENGD_STEPS,
            "checkpoint_reached": source is not None,
            "completed_rounds": len(HISTORY_ROWS),
        }
        if source is not None:
            row.update(source)
        rows.append(row)
    return rows


def time_to_accuracy_rows() -> List[Dict[str, object]]:
    if not HISTORY_ROWS:
        return []
    metrics = (
        "u1_rel_l2", "u2_rel_l2", "xi_rel_l2", "p_rel_l2", "T_rel_l2",
        "u_mean_rel_l2", "pT_mean_rel_l2", "E_common_max", "E_4F_max",
    )
    rows: List[Dict[str, object]] = []
    for metric in metrics:
        values = np.asarray([float(row[metric]) for row in HISTORY_ROWS])
        for threshold in core.TIME_TO_ACCURACY_THRESHOLDS:
            hit = None
            for index in range(len(values) - core.TTA_PATIENCE + 1):
                if np.all(values[index:index + core.TTA_PATIENCE] <= threshold):
                    hit = index
                    break
            record = HISTORY_ROWS[hit] if hit is not None else HISTORY_ROWS[-1]
            rows.append(
                {
                    "run_name": RESULT_DIR.name,
                    "method": method_name(),
                    "architecture": ARCHITECTURE,
                    "seed": core.SEED,
                    "metric_name": metric,
                    "threshold": float(threshold),
                    "patience": core.TTA_PATIENCE,
                    "reached": hit is not None,
                    "first_round": int(record["round"]) if hit is not None else None,
                    "first_equivalent_iteration": (
                        int(record["equivalent_iteration"])
                        if hit is not None else None
                    ),
                    "time_to_accuracy_seconds": (
                        float(record["effective_training_seconds"])
                        if hit is not None else None
                    ),
                    "final_metric_value": float(values[-1]),
                    "effective_training_lower_bound_seconds": (
                        None if hit is not None else
                        float(HISTORY_ROWS[-1]["effective_training_seconds"])
                    ),
                }
            )
    return rows


def save_metadata(flat: Array, timing: Timing, stop_reason: str) -> None:
    final_errors = evaluate_errors(flat)
    metadata = {
        "method": method_name(),
        "formulation": "four-field-monolithic-coupled",
        "architecture": ARCHITECTURE,
        "field_order": ["u1", "u2", "xi", "p", "T"],
        "network_sizes": (
            {"all": NET_SINGLE_SIZES}
            if ARCHITECTURE == "single"
            else {"uXi": NET_UXI_SIZES, "pT": NET_PT_SIZES}
        ),
        "parameter_count": PARAMETER_COUNT,
        "parameter_counts_dual": {
            "uXi": DUAL_UXI_PARAMETER_COUNT,
            "pT": DUAL_PT_PARAMETER_COUNT,
        },
        "seed": core.SEED,
        "exact_case": core.EXACT_CASE,
        "physics": {
            "E": core.E, "nu": core.NU, "lambda": core.LAMBDA,
            "mu": core.MU, "alpha": core.ALPHA, "beta": core.BETA,
            "a0": core.A0, "b0": core.B0, "c0": core.C0,
            "K": core.KAPPA, "Theta": core.THETA,
            "T_final": core.T_FINAL,
        },
        "budget": {
            "rounds_requested": COUPLED_ROUNDS,
            "rounds_completed": len(HISTORY_ROWS),
            "engd_steps_per_round": COUPLED_ENGD_STEPS,
            "equivalent_iteration_definition": "round*engd_steps_per_round",
            "termination_reason": stop_reason,
        },
        "collocation": {
            "N_xy_pde": core.N_XY_PDE, "N_t_pde": core.N_T_PDE,
            "N_xy_ic": core.N_XY_IC, "N_side": core.N_SIDE,
            "N_t_boundary": core.N_T_BOUNDARY,
            "N_xy_eval": core.N_XY_EVAL, "N_t_eval": core.N_T_EVAL,
        },
        "optimizer": {
            **asdict(ENGD_CONFIG),
            "point_block_coupled": POINT_BLOCK_COUPLED,
        },
        "weights": {"u": core.WEIGHTS_U, "pT": core.WEIGHTS_PT},
        "pt_residual_mode": core.PT_RESIDUAL_MODE,
        "timing": {
            **asdict(timing),
            "optimizer_seconds": timing.optimizer_seconds,
            "effective_training_seconds": timing.effective_training_seconds,
        },
        "memory": core.memory_snapshot(),
        "final_errors": final_errors,
    }
    with open(RESULT_DIR / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)


def plot_history() -> None:
    if not HISTORY_ROWS:
        return
    x = np.asarray([row["equivalent_iteration"] for row in HISTORY_ROWS])
    fig, axis = plt.subplots(figsize=(9, 6))
    for name, label, marker in (
        ("u1_rel_l2", r"$u_1$", "o"),
        ("u2_rel_l2", r"$u_2$", "v"),
        ("xi_rel_l2", r"$\xi$", "D"),
        ("p_rel_l2", r"$p$", "s"),
        ("T_rel_l2", r"$T$", "^"),
    ):
        axis.semilogy(x, [row[name] for row in HISTORY_ROWS], marker=marker,
                      label=label)
    axis.set_xlabel("Equivalent ENGD iteration")
    axis.set_ylabel(r"Relative $L^2$ error")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    core.save_figure(fig, "history_relative_l2")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 6))
    seconds = [row["effective_training_seconds"] for row in HISTORY_ROWS]
    axis.semilogy(seconds, [row["E_common_max"] for row in HISTORY_ROWS],
                   marker="o", label=r"$E_{common}$")
    axis.semilogy(seconds, [row["E_4F_max"] for row in HISTORY_ROWS],
                   marker="D", label=r"$E_{4F}$")
    axis.set_xlabel("Effective training time (s)")
    axis.set_ylabel("Maximum component relative error")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    core.save_figure(fig, "history_accuracy_vs_time")
    plt.close(fig)


def plot_final_fields(flat: Array) -> None:
    n_plot = 120
    axis_values = jnp.linspace(0.0, 1.0, n_plot)
    xx, yy = jnp.meshgrid(axis_values, axis_values, indexing="ij")
    points = jnp.stack(
        (xx.reshape(-1), yy.reshape(-1),
         jnp.full((n_plot * n_plot,), core.T_FINAL)), axis=1
    )
    prediction = np.asarray(V_FIELDS(flat, points))
    exact = np.concatenate(
        (
            np.asarray(core.V_UXI_EXACT(points)),
            np.asarray(core.V_PT_EXACT(points)),
        ), axis=1
    )
    names = [r"$u_1$", r"$u_2$", r"$\xi$", r"$p$", r"$T$"]
    fig, axes = plt.subplots(3, 5, figsize=(23, 12))
    for column, name in enumerate(names):
        fields = (
            prediction[:, column], exact[:, column],
            np.abs(prediction[:, column] - exact[:, column]),
        )
        for row_index, (field, title) in enumerate(
            zip(fields, ("prediction", "exact", "absolute error"))
        ):
            image = axes[row_index, column].imshow(
                field.reshape(n_plot, n_plot), origin="lower",
                extent=(0.0, 1.0, 0.0, 1.0), aspect="auto"
            )
            axes[row_index, column].set_title(f"{name}: {title}")
            axes[row_index, column].set_xlabel("x")
            axes[row_index, column].set_ylabel("y")
            fig.colorbar(image, ax=axes[row_index, column])
    fig.tight_layout()
    core.save_figure(fig, "final_fields")
    plt.close(fig)


def main() -> None:
    final_flat, timing, stop_reason = train()
    save_checkpoint(final_flat)
    save_rows("training_history", HISTORY_ROWS)
    np.savez(
        RESULT_DIR / "training_history.npz",
        **{
            key: np.asarray([row[key] for row in HISTORY_ROWS])
            for key in HISTORY_ROWS[0]
        } if HISTORY_ROWS else {},
    )
    save_rows("engd_profile", PROFILE_ROWS)
    save_rows("fixed_budget_accuracy", fixed_budget_rows())
    save_rows("time_to_accuracy", time_to_accuracy_rows())
    save_metadata(final_flat, timing, stop_reason)
    if not core.FAST_DEBUG and core.SAVE_PLOTS:
        try:
            plot_history()
            plot_final_fields(final_flat)
        except Exception:
            LOGGER.exception(
                "Plotting failed, but model, reports, and metadata were saved."
            )
    LOGGER.info("All outputs saved to %s", RESULT_DIR)


if __name__ == "__main__":
    main()
