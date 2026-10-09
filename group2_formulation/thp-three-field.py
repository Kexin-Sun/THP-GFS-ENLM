"""
THP-GFS-ENGD
============

Three-field linear thermo-poroelasticity PINN based on generalized fixed-stress
(GFS) splitting and Energy Natural Gradient Descent (ENGD).  This version keeps
the validated three-field/compressible physics and GFS iteration unchanged, but
uses the frozen production ENGD backend from our optimizer validation work:

* exact streamed pointwise Gram/gradient assembly (no global Jacobian storage);
* symmetric EIGH pseudoinverse with an automatic FP64 spectral cutoff;
* two fixed-shape iterative-refinement corrections (no guarded reduced SVD);
* one compiled/vectorized 31-level logarithmic grid search that reuses all
  quantities frozen within the current GFS subproblem;
* explicit JIT warmup, optimizer/convergence/evaluation timing, and memory stats.

This file is designed as an optimizer-only counterpart of the existing
THP-GFS-MF-GN/LM implementation:

* The physical model, exact solution, source terms, initial/boundary conditions,
  network architecture, collocation sets, weighted residual vectors, diagnostics,
  error metrics, plotting, and GFS outer iteration are retained.
* The matrix-free Gauss-Newton/Levenberg-Marquardt inner solver is replaced by
  explicit ENGD:

      G(theta) = J(theta)^T J(theta),
      g(theta) = grad L(theta) = J(theta)^T r(theta),
      G(theta) d(theta) ~= g(theta),
      theta_new = theta - eta d(theta),

  where r is the physically weighted residual vector. The default solve uses
  ``eigh-refined``: for G_mu = G + mu I it keeps eigenvalues above

      tau = max(eps_fp64 * P * max(||lambda(G_mu)||_inf, 1),
                relative_cutoff * max(||lambda(G_mu)||_inf, 1)),

  applies the corresponding spectral pseudoinverse, and performs two residual
  corrections with the same truncated inverse. The production default is
  ``mu=1e-7`` and ``relative_cutoff=0``; the automatic FP64 cutoff remains
  active in addition to the fixed damping.

  The Jacobian/Gram backend can be selected between:

  * ``jacfwd``: the validated global explicit-Jacobian baseline;
  * ``pointwise``: exact pointwise ``jacrev`` with static point blocks and
    streaming accumulation of ``J.T @ J`` and ``J.T @ r`` without storing the
    complete residual-by-parameter Jacobian.

The paper protocol uses a fixed 50-sweep GFS budget and 20 ENGD updates for
each subproblem in every completed sweep. Adaptive stopping remains available
as an opt-in diagnostic, but is disabled by default so equal-budget comparisons
are not silently shortened. The exact Gram microbenchmark remains available as
an opt-in diagnostic, and ``jax.numpy.linalg.lstsq`` remains available as an
optional full-Gram direction reference.

Model (u, p, T):

    -div(2 mu eps(u) + lambda div(u) I)
        + alpha grad(p) + beta grad(T) = f,

    d_t(c0 p - b0 T + alpha div(u)) - K Delta p = g,

    d_t(a0 T - b0 p + beta div(u)) - Theta Delta T = H_s.

Generalized fixed-stress split for q=(p,T)^T:

    C = [[c0, -b0], [-b0, a0]],
    S_FS = gamma [alpha,beta]^T [alpha,beta],
    gamma = 1 / (2 (lambda + 2 mu / d)).

At GFS iteration k+1:

    (C + S_FS) q_t^{k+1} - diag(K,Theta) Delta q^{k+1}
      = s - [alpha,beta]^T d_t div(u^k) + S_FS q_t^k,

followed by the mechanics solve using q^{k+1}.

The default direct run executes the production pointwise solver with:

* manufactured solution: ``distinct`` (p is not equal to T);
* GFS sweeps: 50, without adaptive early termination;
* ENGD updates: 20 per subproblem and sweep;
* point blocks: 512 for both subproblems;
* solver: ``eigh-refined`` with ``mu=1e-7`` and two refinements;
* 31 positive logarithmic line-search levels;
* high-density sampling: PDE=4096, IC=484, u-Dir=816, u-Neu=816,
  pT-Dir=1632, evaluation=64000.

    python thp-three-field.py

The exact Gram microbenchmark is now opt-in:

    THP_JACOBIAN_MICROBENCH=1 \
    python thp-three-field.py

The validated global baseline is available with ``THP_JACOBIAN_BACKEND=jacfwd``.
The full LSTSQ reference is disabled by default; set
``THP_EIGH_COMPARE_LSTSQ=1`` only for direction-validation runs.
"""

# =============================================================================
# 0. Imports and runtime configuration
# =============================================================================

from __future__ import annotations

import csv
import json
import logging
import math
import os
import resource
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# XLA allocator variables must be resolved before importing JAX.  A dedicated
# training GPU benefits from preallocation because it avoids allocator churn
# and fragmentation during the repeated Gram/eigensolver/line-search kernels.
# Existing XLA_* variables always take precedence, so cluster launch scripts
# remain authoritative.
_thp_xla_preallocate = os.environ.get("THP_XLA_PREALLOCATE", "true").lower()
if _thp_xla_preallocate not in {"true", "false"}:
    raise ValueError("THP_XLA_PREALLOCATE must be 'true' or 'false'.")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", _thp_xla_preallocate)
if "THP_XLA_MEM_FRACTION" in os.environ:
    _thp_xla_mem_fraction = float(os.environ["THP_XLA_MEM_FRACTION"])
    if not 0.0 < _thp_xla_mem_fraction <= 1.0:
        raise ValueError("THP_XLA_MEM_FRACTION must lie in (0, 1].")
    os.environ.setdefault(
        "XLA_PYTHON_CLIENT_MEM_FRACTION",
        str(_thp_xla_mem_fraction),
    )

import jax
import jax.numpy as jnp
from jax import grad, hessian, jacfwd, jacrev, jit, lax, random, vmap

# Flatten the network pytree and obtain its inverse mapping.

from jax.flatten_util import ravel_pytree

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Enable FP64 before creating arrays: spectral solves are sensitive to roundoff.

jax.config.update("jax_enable_x64", True)

Array = jax.Array
# Each layer is a (weights, biases) pair.
Params = List[Tuple[Array, Array]]


ResidualFunction = Callable[[Array], Array]


JacobianFunction = Callable[[Array], Array]


# =============================================================================
# 1. Logging and reproducibility
# =============================================================================
# A fixed seed reproduces initialization in the same numerical environment.

SEED = int(os.environ.get("THP_SEED", "12345"))

MASTER_KEY = random.PRNGKey(SEED)


SCRIPT_DIR = Path(__file__).resolve().parent


TIME_STAMP = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

# Default result root is the script directory; THP_RESULT_ROOT overrides it.

RESULT_ROOT = Path(os.environ.get("THP_RESULT_ROOT", str(SCRIPT_DIR / "runs"))).resolve()


# The default run name includes seed, timestamp, and PID.
RUN_NAME = os.environ.get(
    "THP_RUN_NAME",
    f"thp_gfs_engd_seed{SEED}_{TIME_STAMP}_pid{os.getpid()}",
)


RESULT_DIR = (RESULT_ROOT / RUN_NAME).resolve()



# Frozen scientific data are never a training destination.
for _frozen in SCRIPT_DIR.parent.glob("group*/results"):
    if RESULT_DIR == _frozen.resolve() or _frozen.resolve() in RESULT_DIR.parents:
        raise ValueError("Training output must be outside all frozen results directories.")

# Scientific run directories are never merged or overwritten.
RESULT_DIR.mkdir(parents=True, exist_ok=False)


def build_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("THP-GFS-ENGD")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


LOGGER = build_logger(RESULT_DIR / "training.log")


# =============================================================================
# 2. Physical parameters and generalized fixed-stress coefficient
# =============================================================================

E = float(os.environ.get("THP_E", "1.0"))
NU = float(os.environ.get("THP_NU", "0.3"))
ALPHA = float(os.environ.get("THP_ALPHA", "0.1"))
BETA = float(os.environ.get("THP_BETA", "0.1"))
C0 = float(os.environ.get("THP_C0", "0.2"))
A0 = float(os.environ.get("THP_A0", "0.2"))
B0 = float(os.environ.get("THP_B0", "0.1"))
KAPPA = float(os.environ.get("THP_KAPPA", "0.1"))
THETA = float(os.environ.get("THP_THETA", "0.1"))

SPACE_DIMENSION = 2

if E <= 0.0:
    raise ValueError("THP_E must be positive.")
if not 0.0 < NU < 0.5:
    raise ValueError("THP_NU must lie in (0, 0.5).")
if min(ALPHA, BETA, C0, A0, B0, KAPPA, THETA) < 0.0:
    raise ValueError("THP physical coefficients must be non-negative.")
if KAPPA <= 0.0 or THETA <= 0.0:
    raise ValueError("THP_KAPPA and THP_THETA must be positive.")

LAMBDA = E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
MU = E / (2.0 * (1.0 + NU))

# drained bulk-type coefficient:
#
# K_dr = lambda + 2 mu / d.
#
# For two spatial dimensions, K_dr=lambda+mu.
K_DRAINED = LAMBDA + 2.0 * MU / SPACE_DIMENSION

# GFS stabilization coefficient gamma_FS=1/(2*K_dr).
#
# gamma_FS = 1 / (2 K_dr).
GAMMA_FS = 1.0 / (2.0 * K_DRAINED)

# S_FS=gamma_FS * a a.T is a symmetric positive semidefinite 2x2 matrix.
#
# S_FS = gamma_FS [alpha, beta]^T [alpha, beta]
#

S11 = GAMMA_FS * ALPHA * ALPHA
S12 = GAMMA_FS * ALPHA * BETA
S21 = S12
S22 = GAMMA_FS * BETA * BETA

# Coupling vector multiplying the time derivative of displacement divergence.
COUPLING_VECTOR = jnp.array([ALPHA, BETA])

# Physical pressure-temperature storage matrix C0=[[c0,-b0],[-b0,a0]].
#
# C = [[c0, -b0],
#      [-b0, a0]].
STORAGE_MATRIX = jnp.array([[C0, -B0], [-B0, A0]])

# Keep the stabilization matrix for diagnostics; kernels use its scalar entries.


FS_MATRIX = jnp.array([[S11, S12], [S21, S22]])

# Hydraulic and thermal diffusivities.
DIFFUSIVITY = jnp.array([KAPPA, THETA])

T_FINAL = float(os.environ.get("THP_T_FINAL", "1"))

# Fixed production budget used for cross-formulation comparisons.
FS_ITERATIONS = int(os.environ.get("THP_FS_ITERATIONS", "50"))
ENGD_STEPS_PT = int(os.environ.get("THP_ENGD_STEPS_PT", "20"))
ENGD_STEPS_U = int(os.environ.get("THP_ENGD_STEPS_U", "20"))

# Optional convergence-based stopping; controlled experiments disable it.
ENABLE_FS_EARLY_STOP = (
    os.environ.get("THP_ENABLE_FS_EARLY_STOP", "0") == "1"
)


FS_MIN_ITERATIONS = int(os.environ.get("THP_FS_MIN_ITERATIONS", "20"))


FS_EARLY_STOP_PATIENCE = int(
    os.environ.get("THP_FS_EARLY_STOP_PATIENCE", "2")
)


FS_EARLY_STOP_OUTER_TOL = float(
    os.environ.get("THP_FS_EARLY_STOP_OUTER_TOL", "1e-5")
)


FS_EARLY_STOP_COUPLED_PT_TOL = float(
    os.environ.get("THP_FS_EARLY_STOP_COUPLED_PT_TOL", "1e-8")
)


FS_EARLY_STOP_U_LOSS_TOL = float(
    os.environ.get("THP_FS_EARLY_STOP_U_LOSS_TOL", "1e-8")
)

TIME_TO_ACCURACY_THRESHOLDS = tuple(
    float(value.strip())
    for value in os.environ.get(
        "THP_TIME_TO_ACCURACY_THRESHOLDS",
        "1e-2,1e-3,8e-4,6e-4,4e-4,2e-4,1e-4",
    ).split(",")
    if value.strip()
)
TTA_PATIENCE = int(os.environ.get("THP_TTA_PATIENCE", "3"))
FIXED_BUDGET_CHECKPOINTS = tuple(
    sorted(
        {
            int(value.strip())
            for value in os.environ.get(
                "THP_FIXED_BUDGET_CHECKPOINTS", "20,30,40,50"
            ).split(",")
            if value.strip()
        }
    )
)

# ---------------------------------------------------------------------------
# Validate configuration before expensive JAX compilation.
# ---------------------------------------------------------------------------
if FS_ITERATIONS <= 0:
    raise ValueError("THP_FS_ITERATIONS must be positive.")
if ENGD_STEPS_PT <= 0 or ENGD_STEPS_U <= 0:
    raise ValueError("ENGD step counts must be positive.")
if FS_MIN_ITERATIONS < 1:
    raise ValueError("THP_FS_MIN_ITERATIONS must be at least one.")
if FS_EARLY_STOP_PATIENCE < 1:
    raise ValueError("THP_FS_EARLY_STOP_PATIENCE must be at least one.")
for name, value in (
    ("THP_FS_EARLY_STOP_OUTER_TOL", FS_EARLY_STOP_OUTER_TOL),
    ("THP_FS_EARLY_STOP_COUPLED_PT_TOL", FS_EARLY_STOP_COUPLED_PT_TOL),
    ("THP_FS_EARLY_STOP_U_LOSS_TOL", FS_EARLY_STOP_U_LOSS_TOL),
):
    if value <= 0.0:
        raise ValueError(f"{name} must be positive.")
if not TIME_TO_ACCURACY_THRESHOLDS:
    raise ValueError("THP_TIME_TO_ACCURACY_THRESHOLDS must not be empty.")
if TTA_PATIENCE <= 0:
    raise ValueError("THP_TTA_PATIENCE must be positive.")
if any(
    not math.isfinite(value) or value <= 0.0
    for value in TIME_TO_ACCURACY_THRESHOLDS
):
    raise ValueError(
        "THP_TIME_TO_ACCURACY_THRESHOLDS must contain positive finite values."
    )
if not FIXED_BUDGET_CHECKPOINTS:
    raise ValueError("THP_FIXED_BUDGET_CHECKPOINTS must not be empty.")
if any(value <= 0 for value in FIXED_BUDGET_CHECKPOINTS):
    raise ValueError(
        "THP_FIXED_BUDGET_CHECKPOINTS must contain positive integers."
    )

# Smoke tests reduce sampling and iteration budgets.
FAST_DEBUG = os.environ.get("THP_FAST_DEBUG", "0") == "1"


SAVE_PLOTS = os.environ.get("THP_SAVE_PLOTS", "1") == "1"

# FAST_DEBUG caps outer sweeps at two; an explicit smaller budget is respected.

if FAST_DEBUG:
    FS_ITERATIONS = min(FS_ITERATIONS, 2)
    ENGD_STEPS_PT = min(ENGD_STEPS_PT, 2)
    ENGD_STEPS_U = min(ENGD_STEPS_U, 2)


# =============================================================================
# 3. Fixed uniform collocation sets
# =============================================================================


def tensor_product_space_time(
    n_xy: int,          
    n_t: int,           
    t_final: float,     
    include_t0: bool,   
) -> Array:
    """Uniform interior tensor grid on (0,1)^2 x time."""
    """Build an interior tensor-product grid of shape (n_xy*n_xy*n_t, 3).

    Spatial endpoints are excluded. Time nodes cover [0,t_final] when
    include_t0=True (evaluation), and (0,t_final] otherwise (PDE).
    Each row stores [x,y,t]; t varies fastest, followed by y and x."""
    # Exclude spatial boundary nodes.
    
    x = jnp.linspace(0.0, 1.0, n_xy + 2)[1:-1]
    y = jnp.linspace(0.0, 1.0, n_xy + 2)[1:-1]

    
    if include_t0:
        
        #
        
        #     0, T_final/39, ..., T_final.
        t = jnp.linspace(0.0, t_final, n_t)
    else:
        # PDE points exclude t=0 because the initial condition is separate.
        # Exclude t=0 for PDE points; initial conditions are enforced separately.

        
        
        #
        
        #     T_final/n_t, 2*T_final/n_t, ..., T_final.
        t = jnp.linspace(0.0, t_final, n_t + 1)[1:]
    x_grid, y_grid, t_grid = jnp.meshgrid(x, y, t, indexing="ij")
    # Flatten coordinates with t varying fastest.
    
    return jnp.stack(
        [
            x_grid.ravel(),  
            y_grid.ravel(),  
            t_grid.ravel(),  
        ],
        axis=1,              
    )


def initial_grid(n_xy: int) -> Array:
    """Uniform interior spatial points on t=0."""
    """Build (n_xy*n_xy, 3) interior spatial points at t=0.

    Spatial boundaries are excluded to avoid duplicating boundary constraints."""
    
    x = jnp.linspace(0.0, 1.0, n_xy + 2)[1:-1]

    
    y = jnp.linspace(0.0, 1.0, n_xy + 2)[1:-1]
    x_grid, y_grid = jnp.meshgrid(x, y, indexing="ij")
    
    t_grid = jnp.zeros_like(x_grid)
    
    return jnp.stack(
        [x_grid.ravel(), y_grid.ravel(), t_grid.ravel()], axis=1
    )

def boundary_side_grid(
    side: str,                    
    n_s: int,                     
    n_t: int,                     
    t_final: float,               
    include_corners: bool = True, 
) -> Array:
    """Uniform points on one spatial side crossed with time."""
    """Build (n_s*n_t, 3) points on one space-time boundary.

    left/right mean x=0/1; bottom/top mean y=0/1. Time includes both
    endpoints. include_corners controls spatial endpoints; left/right include
    corners and bottom/top exclude them, assigning mechanics corners to the
    displacement Dirichlet boundary without duplicate corner constraints."""
    if include_corners:
        
        
        s = jnp.linspace(0.0, 1.0, n_s)
    else:
        
        
        s = jnp.linspace(0.0, 1.0, n_s + 2)[1:-1]

    
    t = jnp.linspace(0.0, t_final, n_t)
    s_grid, t_grid = jnp.meshgrid(s, t, indexing="ij")

    
    if side == "left":
        
        x_grid = jnp.zeros_like(s_grid)
        y_grid = s_grid
    elif side == "right":
        
        x_grid = jnp.ones_like(s_grid)
        y_grid = s_grid
    elif side == "bottom":
        
        x_grid = s_grid
        y_grid = jnp.zeros_like(s_grid)
    elif side == "top":
        
        x_grid = s_grid
        y_grid = jnp.ones_like(s_grid)
    else:
        
        raise ValueError(f"Unknown boundary side: {side}")

    return jnp.stack(
        [x_grid.ravel(), y_grid.ravel(), t_grid.ravel()], axis=1
    )


def concatenate_sides(*sides: Array) -> Array:
    """Concatenate any number of (N_i, 3) boundary arrays along the point axis."""

    
    return jnp.concatenate(sides, axis=0)


N_XY_PDE = int(os.environ.get("THP_N_XY_PDE", "16"))
N_T_PDE = int(os.environ.get("THP_N_T_PDE", "16"))
N_XY_IC = int(os.environ.get("THP_N_XY_IC", "22"))
N_SIDE = int(os.environ.get("THP_N_SIDE", "24"))
N_T_BOUNDARY = int(os.environ.get("THP_N_T_BOUNDARY", "17"))
N_XY_EVAL = int(os.environ.get("THP_N_XY_EVAL", "40"))
N_T_EVAL = int(os.environ.get("THP_N_T_EVAL", "40"))

for name, value in (
    ("THP_N_XY_PDE", N_XY_PDE),
    ("THP_N_T_PDE", N_T_PDE),
    ("THP_N_XY_IC", N_XY_IC),
    ("THP_N_SIDE", N_SIDE),
    ("THP_N_T_BOUNDARY", N_T_BOUNDARY),
    ("THP_N_XY_EVAL", N_XY_EVAL),
    ("THP_N_T_EVAL", N_T_EVAL),
):
    if value <= 0:
        raise ValueError(f"{name} must be positive.")

# Small point sets reduce smoke-test compilation and assembly costs.

if FAST_DEBUG:
    N_XY_PDE = 5
    N_T_PDE = 3
    N_XY_IC = 6
    N_SIDE = 8
    N_T_BOUNDARY = 4
    N_XY_EVAL = 10
    N_T_EVAL = 5

PDE_POINTS = tensor_product_space_time(
    N_XY_PDE, N_T_PDE, T_FINAL, include_t0=False
)
IC_POINTS = initial_grid(N_XY_IC)

# Left/right include corners; bottom/top exclude them.

LEFT_POINTS = boundary_side_grid(
    "left", N_SIDE, N_T_BOUNDARY, T_FINAL, include_corners=True
)


RIGHT_POINTS = boundary_side_grid(
    "right", N_SIDE, N_T_BOUNDARY, T_FINAL, include_corners=True
)


BOTTOM_POINTS = boundary_side_grid(
    "bottom", N_SIDE, N_T_BOUNDARY, T_FINAL, include_corners=False
)


TOP_POINTS = boundary_side_grid(
    "top", N_SIDE, N_T_BOUNDARY, T_FINAL, include_corners=False
)

# u: Dirichlet on x=0,1; total-traction Neumann on y=0,1.
U_DIR_POINTS = concatenate_sides(LEFT_POINTS, RIGHT_POINTS)
U_NEU_POINTS = concatenate_sides(BOTTOM_POINTS, TOP_POINTS)

# p and T: Dirichlet on the whole boundary.
PT_DIR_POINTS = concatenate_sides(
    LEFT_POINTS, RIGHT_POINTS, BOTTOM_POINTS, TOP_POINTS
)

# Evaluation uses interior spatial nodes and includes t=0 and t=T_final.


EVAL_POINTS = tensor_product_space_time(
    N_XY_EVAL, N_T_EVAL, T_FINAL, include_t0=True
)

# Domain/boundary measures for integral-like residual weighting.

MEASURE_PDE = T_FINAL           # Space-time volume on the unit square.
MEASURE_IC = 1.0                # Initial-condition area.
MEASURE_U_DIR = 2.0 * T_FINAL   # Two Dirichlet sides times the time interval.
MEASURE_U_NEU = 2.0 * T_FINAL   # Two Neumann sides times the time interval.
MEASURE_PT_DIR = 4.0 * T_FINAL  # All four pressure-temperature Dirichlet sides.


# =============================================================================
# 4. Neural networks
# =============================================================================


def xavier_layer_params(m: int, n: int, key: Array) -> Tuple[Array, Array]:
    limit = jnp.sqrt(6.0 / (m + n))
    weights = random.uniform(key, (n, m), minval=-limit, maxval=limit)
    # Initialize biases to zero with the weight dtype.
    
    biases = jnp.zeros((n,), dtype=weights.dtype)
    return weights, biases


def init_mlp(sizes: Sequence[int], key: Array) -> Params:
    keys = random.split(key, len(sizes) - 1)
    return [
        xavier_layer_params(m, n, layer_key)
        for m, n, layer_key in zip(sizes[:-1], sizes[1:], keys)
    ]


@jit
def predict(params: Params, xyt: Array) -> Array:
    value = xyt
    for weights, biases in params[:-1]:
        value = jnp.tanh(weights @ value + biases)
    weights, biases = params[-1]
    return weights @ value + biases


V_PREDICT = jit(vmap(predict, in_axes=(None, 0)))

NET_U_SIZES = [3, 32, 32, 2]
NET_PT_SIZES = [3, 32, 32, 2]

MASTER_KEY, KEY_U, KEY_PT = random.split(MASTER_KEY, 3)
PARAMS_U = init_mlp(NET_U_SIZES, KEY_U)
PARAMS_PT = init_mlp(NET_PT_SIZES, KEY_PT)

FLAT_U, UNRAVEL_U = ravel_pytree(PARAMS_U)
FLAT_PT, UNRAVEL_PT = ravel_pytree(PARAMS_PT)


# =============================================================================
# 5. Exact fields, sources, and boundary data
# =============================================================================

# paper:
#   p = T = exp(-t) sin(pi*x) sin(pi*y)
#
# distinct:
#   p = exp(-t) sin(pi*x) sin(pi*y)
#   T = exp(-2t) sin(2*pi*x) sin(pi*y)
EXACT_CASE = os.environ.get(
    "THP_EXACT_CASE", "distinct"
).strip().lower()

if EXACT_CASE not in {"paper", "distinct"}:
    raise ValueError(
        "THP_EXACT_CASE must be 'paper' or 'distinct', "
        f"but received {EXACT_CASE!r}."
    )

LOGGER.info("exact_case=%s", EXACT_CASE)


@jit
def u_exact(xyt: Array) -> Array:
    x, y, t = xyt

    common = (
        jnp.sin(jnp.pi * x)
        * jnp.sin(jnp.pi * y)
        / (MU + LAMBDA)
    )

    u1 = (
        jnp.sin(2.0 * jnp.pi * y)
        * (jnp.cos(2.0 * jnp.pi * x) - 1.0)
        + common
    )

    u2 = (
        jnp.sin(2.0 * jnp.pi * x)
        * (1.0 - jnp.cos(2.0 * jnp.pi * y))
        + common
    )

    return jnp.exp(-t) * jnp.array([u1, u2])


@jit
def p_exact(xyt: Array) -> Array:
    x, y, t = xyt

    return (
        jnp.exp(-t)
        * jnp.sin(jnp.pi * x)
        * jnp.sin(jnp.pi * y)
    )


if EXACT_CASE == "paper":

    @jit
    def temperature_exact(xyt: Array) -> Array:
        """Original paper benchmark: T = p."""
        return p_exact(xyt)

else:

    @jit
    def temperature_exact(xyt: Array) -> Array:
        """Non-symmetric manufactured temperature field."""
        x, y, t = xyt

        return (
            jnp.exp(-2.0 * t)
            * jnp.sin(2.0 * jnp.pi * x)
            * jnp.sin(jnp.pi * y)
        )


@jit
def pt_exact(xyt: Array) -> Array:
    """Return the exact pressure-temperature vector [p, T]."""
    return jnp.array(
        [
            p_exact(xyt),
            temperature_exact(xyt),
        ]
    )


# Exact derivatives generated by automatic differentiation.
U_EXACT_JAC = jacfwd(u_exact)
U_EXACT_HESS = hessian(u_exact)

P_EXACT_GRAD = grad(p_exact)
P_EXACT_HESS = hessian(p_exact)

T_EXACT_GRAD = grad(temperature_exact)
T_EXACT_HESS = hessian(temperature_exact)


@jit
def elastic_operator_from_hessian(hess_u: Array) -> Array:
    """Return -div(2 mu eps(u) + lambda div(u) I)."""
    u1_xx = hess_u[0, 0, 0]
    u1_yy = hess_u[0, 1, 1]
    u1_xy = hess_u[0, 0, 1]

    u2_xx = hess_u[1, 0, 0]
    u2_yy = hess_u[1, 1, 1]
    u2_xy = hess_u[1, 0, 1]

    
    # -[(lambda+2mu)u1_xx + mu*u1_yy + (lambda+mu)u2_xy]
    component_1 = -(
        (LAMBDA + 2.0 * MU) * u1_xx
        + MU * u1_yy
        + (LAMBDA + MU) * u2_xy
    )
    
    # -[mu*u2_xx + (lambda+2mu)u2_yy + (lambda+mu)u1_xy]
    component_2 = -(
        MU * u2_xx
        + (LAMBDA + 2.0 * MU) * u2_yy
        + (LAMBDA + MU) * u1_xy
    )
    return jnp.array([component_1, component_2])


@jit
def body_force_exact(xyt: Array) -> Array:
    '''f = -div[2mu*epsilon(u)+lambda*div(u)I] + alpha*grad(p) + beta*grad(T)'''
    return (
        elastic_operator_from_hessian(U_EXACT_HESS(xyt))
        + ALPHA * P_EXACT_GRAD(xyt)[:2]
        + BETA * T_EXACT_GRAD(xyt)[:2]
    )


@jit
def mass_source_exact(xyt: Array) -> Array:
    '''s_p = c0*p_t - b0*T_t + alpha*(div u)_t - kappa*Delta(p)'''
    grad_p = P_EXACT_GRAD(xyt)
    grad_t = T_EXACT_GRAD(xyt)
    hess_p = P_EXACT_HESS(xyt)
    hess_u = U_EXACT_HESS(xyt)

    div_u_t = hess_u[0, 0, 2] + hess_u[1, 1, 2]
    lap_p = hess_p[0, 0] + hess_p[1, 1]

    return (
        C0 * grad_p[2]
        - B0 * grad_t[2]
        + ALPHA * div_u_t
        - KAPPA * lap_p
    )


@jit
def heat_source_exact(xyt: Array) -> Array:
    '''s_T = a0*T_t - b0*p_t + beta*(div u)_t - Theta*Delta(T)'''
    grad_p = P_EXACT_GRAD(xyt)
    grad_t = T_EXACT_GRAD(xyt)
    hess_t = T_EXACT_HESS(xyt)
    hess_u = U_EXACT_HESS(xyt)

    div_u_t = hess_u[0, 0, 2] + hess_u[1, 1, 2]
    lap_t = hess_t[0, 0] + hess_t[1, 1]

    return (
        A0 * grad_t[2]
        - B0 * grad_p[2]
        + BETA * div_u_t
        - THETA * lap_t
    )


@jit
def outward_normal(xyt: Array, tolerance: float = 1.0e-10) -> Array:
    """Return the outward normal on the unit-square spatial boundary.

    left=(-1,0), right=(1,0), bottom=(0,-1), top=(0,1).
    The input is assumed to lie on the boundary."""
    x, y = xyt[0], xyt[1]
    return jnp.where(
        jnp.abs(y) < tolerance,
        jnp.array([0.0, -1.0]),
        jnp.where(
            jnp.abs(y - 1.0) < tolerance,
            jnp.array([0.0, 1.0]),
            jnp.where(
                jnp.abs(x - 1.0) < tolerance,
                jnp.array([1.0, 0.0]),
                jnp.array([-1.0, 0.0]),
            ),
        ),
    )


@jit
def exact_total_traction(xyt: Array) -> Array:
    """Evaluate exact total traction sigma_total @ n.

    sigma_total = 2*mu*eps(u) + lambda*div(u)*I - (alpha*p+beta*T)*I."""
    jac_u = U_EXACT_JAC(xyt)
    p_value = p_exact(xyt)
    t_value = temperature_exact(xyt)
    coupling = ALPHA * p_value + BETA * t_value

    du1_dx, du1_dy = jac_u[0, 0], jac_u[0, 1]
    du2_dx, du2_dy = jac_u[1, 0], jac_u[1, 1]

    sigma11 = (
        LAMBDA * (du1_dx + du2_dy)
        + 2.0 * MU * du1_dx
        - coupling
    )
    sigma22 = (
        LAMBDA * (du1_dx + du2_dy)
        + 2.0 * MU * du2_dy
        - coupling
    )
    sigma12 = MU * (du1_dy + du2_dx)

    sigma_total = jnp.array(
        [[sigma11, sigma12], [sigma12, sigma22]]
    )
    return sigma_total @ outward_normal(xyt)


# =============================================================================
# 6. Differential operators and PDE residuals
# =============================================================================


def u_value(params: Params, xyt: Array) -> Array:
    return predict(params, xyt)


def pt_value(params: Params, xyt: Array) -> Array:
    return predict(params, xyt)


U_JAC = jacfwd(u_value, argnums=1)     
U_HESS = hessian(u_value, argnums=1)   
PT_JAC = jacfwd(pt_value, argnums=1)   # Spatial-time Jacobian has shape (2,3).
PT_HESS = hessian(pt_value, argnums=1)  # Spatial-time Hessian has shape (2,3,3).


@jit
def mechanics_pde_residual_one(
    params_u: Params,
    params_pt: Params,
    xyt: Array,
) -> Array:
    hess_u = U_HESS(params_u, xyt)
    jac_pt = PT_JAC(params_pt, xyt)
    grad_p = jac_pt[0, :2]
    grad_t = jac_pt[1, :2]
    return (
        elastic_operator_from_hessian(hess_u)
        + ALPHA * grad_p
        + BETA * grad_t
        - body_force_exact(xyt)
    )


@jit
def total_traction_pred_one(
    params_u: Params,
    params_pt: Params,
    xyt: Array,
) -> Array:
    jac_u = U_JAC(params_u, xyt)
    p_value, t_value = pt_value(params_pt, xyt)
    coupling = ALPHA * p_value + BETA * t_value

    du1_dx, du1_dy = jac_u[0, 0], jac_u[0, 1]
    du2_dx, du2_dy = jac_u[1, 0], jac_u[1, 1]

    sigma11 = (
        LAMBDA * (du1_dx + du2_dy)
        + 2.0 * MU * du1_dx
        - coupling
    )
    sigma22 = (
        LAMBDA * (du1_dx + du2_dy)
        + 2.0 * MU * du2_dy
        - coupling
    )
    sigma12 = MU * (du1_dy + du2_dx)

    sigma_total = jnp.array(
        [[sigma11, sigma12], [sigma12, sigma22]]
    )
    # Predicted total traction.
    return sigma_total @ outward_normal(xyt)


@jit
def flow_heat_pde_residual_one(
    params_pt: Params,
    params_u_old: Params,
    params_pt_old: Params,
    xyt: Array,
) -> Array:
    """Evaluate the three-field GFS pressure-temperature residual at one point.

    With q=(p,T), a=(alpha,beta), and S=S_FS, the split equation is
    (C+S) q_t - D Laplacian(q) + a (div u_old)_t - S q_old_t - source = 0.
    Current p,T are optimized; previous u,p,T remain frozen."""
    """GFS p/T subproblem residual at one point."""
    jac_q = PT_JAC(params_pt, xyt)
    hess_q = PT_HESS(params_pt, xyt)

    p_t, t_t = jac_q[0, 2], jac_q[1, 2]
    lap_p = hess_q[0, 0, 0] + hess_q[0, 1, 1]
    lap_t = hess_q[1, 0, 0] + hess_q[1, 1, 1]

    hess_u_old = U_HESS(params_u_old, xyt)
    div_u_t_old = (
        hess_u_old[0, 0, 2] + hess_u_old[1, 1, 2]
    )

    jac_q_old = PT_JAC(params_pt_old, xyt)
    p_t_old, t_t_old = jac_q_old[0, 2], jac_q_old[1, 2]

    residual_p = (
        (C0 + S11) * p_t
        + (-B0 + S12) * t_t
        - KAPPA * lap_p
        + ALPHA * div_u_t_old
        - S11 * p_t_old
        - S12 * t_t_old
        - mass_source_exact(xyt)
    )

    residual_t = (
        (-B0 + S21) * p_t
        + (A0 + S22) * t_t
        - THETA * lap_t
        + BETA * div_u_t_old
        - S21 * p_t_old
        - S22 * t_t_old
        - heat_source_exact(xyt)
    )

    return jnp.array([residual_p, residual_t])


@jit
def coupled_flow_heat_pde_residual_one(
    params_pt: Params,
    params_u: Params,
    xyt: Array,
) -> Array:
    """Evaluate the original coupled flow-heat PDE at the current GFS state.

    The residual is C q_t - D Laplacian(q) + a (div u)_t - source.
    It contains neither stabilization S nor frozen previous parameters."""
    """Residual of the original coupled flow/heat PDE at the current state."""
    jac_q = PT_JAC(params_pt, xyt)
    hess_q = PT_HESS(params_pt, xyt)

    q_t = jac_q[:, 2]
    lap_q = jnp.array(
        [
            hess_q[0, 0, 0] + hess_q[0, 1, 1],
            hess_q[1, 0, 0] + hess_q[1, 1, 1],
        ]
    )

    hess_u = U_HESS(params_u, xyt)
    div_u_t = hess_u[0, 0, 2] + hess_u[1, 1, 2]

    sources = jnp.array(
        [mass_source_exact(xyt), heat_source_exact(xyt)]
    )

    return (
        STORAGE_MATRIX @ q_t
        - DIFFUSIVITY * lap_q
        + COUPLING_VECTOR * div_u_t
        - sources
    )


V_MECH_RES = jit(
    vmap(mechanics_pde_residual_one, in_axes=(None, None, 0))
)
V_FLOW_HEAT_RES = jit(
    vmap(
        flow_heat_pde_residual_one,
        in_axes=(None, None, None, 0),
    )
)
V_COUPLED_FLOW_HEAT_RES = jit(
    vmap(
        coupled_flow_heat_pde_residual_one,
        in_axes=(None, None, 0),
    )
)
V_TOTAL_TRACTION_PRED = jit(
    vmap(total_traction_pred_one, in_axes=(None, None, 0))
)
V_EXACT_TRACTION = jit(vmap(exact_total_traction))
V_U_EXACT = jit(vmap(u_exact))
V_PT_EXACT = jit(vmap(pt_exact))
V_U_EXACT_JAC = jit(vmap(U_EXACT_JAC))

# Manufactured data and geometry never change during training.  Materialize
# them once and reuse them in every GFS context and line-search candidate.
# This removes repeated high-order AD of the analytic solution from the hot
# path without changing any residual value.
V_BODY_FORCE_EXACT = jit(vmap(body_force_exact))
V_MASS_SOURCE_EXACT = jit(vmap(mass_source_exact))
V_HEAT_SOURCE_EXACT = jit(vmap(heat_source_exact))
V_OUTWARD_NORMAL = jit(vmap(outward_normal))

BODY_FORCE_PDE = V_BODY_FORCE_EXACT(PDE_POINTS)
MASS_SOURCE_PDE = V_MASS_SOURCE_EXACT(PDE_POINTS)
HEAT_SOURCE_PDE = V_HEAT_SOURCE_EXACT(PDE_POINTS)
U_NEU_NORMALS = V_OUTWARD_NORMAL(U_NEU_POINTS)
U_NEU_EXACT_TRACTION = V_EXACT_TRACTION(U_NEU_POINTS)

PT_DIR_TARGET = V_PT_EXACT(PT_DIR_POINTS)
PT_IC_TARGET = V_PT_EXACT(IC_POINTS)
U_DIR_TARGET = V_U_EXACT(U_DIR_POINTS)
U_IC_TARGET = V_U_EXACT(IC_POINTS)


# =============================================================================
# 7. Weighted residual-vector construction
# =============================================================================

WEIGHTS_U: Dict[str, float] = {
    "pde": float(os.environ.get("THP_WEIGHT_U_PDE", "1.0")),
    "dirichlet": float(
        os.environ.get("THP_WEIGHT_U_DIRICHLET", "1.0")
    ),
    "neumann": float(os.environ.get("THP_WEIGHT_U_NEUMANN", "1.0")),
    "ic": float(os.environ.get("THP_WEIGHT_U_IC", "1.0")),
}
WEIGHTS_PT: Dict[str, float] = {
    "pde": float(os.environ.get("THP_WEIGHT_PT_PDE", "1.0")),
    "dirichlet": float(
        os.environ.get("THP_WEIGHT_PT_DIRICHLET", "1.0")
    ),
    "ic": float(os.environ.get("THP_WEIGHT_PT_IC", "1.0")),
}

for group_name, weights in (("u", WEIGHTS_U), ("pT", WEIGHTS_PT)):
    for block_name, value in weights.items():
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"Residual weight {group_name}.{block_name} must be "
                "finite and non-negative."
            )


def block_scale(weight: float, measure: float, n_points: int) -> float:
    """Return sqrt(weight*measure/n_points) for integral-weighted residuals.

    n_points counts collocation points, not scalar residual components."""
    """Scale so 0.5||r_scaled||^2 approximates 0.5 w integral(r^2)."""
    if n_points <= 0:
        raise ValueError("A residual block must contain at least one point.")
    return math.sqrt(weight * measure / float(n_points)) 

# Mechanics block scaling.
SCALE_U_PDE = block_scale(
    WEIGHTS_U["pde"], MEASURE_PDE, int(PDE_POINTS.shape[0])
)
SCALE_U_DIR = block_scale(
    WEIGHTS_U["dirichlet"], MEASURE_U_DIR, int(U_DIR_POINTS.shape[0])
)
SCALE_U_NEU = block_scale(
    WEIGHTS_U["neumann"], MEASURE_U_NEU, int(U_NEU_POINTS.shape[0])
)
SCALE_U_IC = block_scale(
    WEIGHTS_U["ic"], MEASURE_IC, int(IC_POINTS.shape[0])
)

# Pressure-temperature block scaling.
SCALE_PT_PDE = block_scale(
    WEIGHTS_PT["pde"], MEASURE_PDE, int(PDE_POINTS.shape[0])
)
SCALE_PT_DIR = block_scale(
    WEIGHTS_PT["dirichlet"],
    MEASURE_PT_DIR,
    int(PT_DIR_POINTS.shape[0]),
)
SCALE_PT_IC = block_scale(
    WEIGHTS_PT["ic"], MEASURE_IC, int(IC_POINTS.shape[0])
)

# Assemble the complete weighted subproblem residual.
@jit
def residual_u_flat(flat_u: Array, flat_pt: Array) -> Array:
    params_u = UNRAVEL_U(flat_u)
    params_pt = UNRAVEL_PT(flat_pt)

    r_pde = V_MECH_RES(params_u, params_pt, PDE_POINTS).reshape(-1) 
    r_dir = (
        V_PREDICT(params_u, U_DIR_POINTS)
        - U_DIR_TARGET
    ).reshape(-1)                                                   
    r_neu = (
        V_TOTAL_TRACTION_PRED(
            params_u, params_pt, U_NEU_POINTS
        )
        - U_NEU_EXACT_TRACTION
    ).reshape(-1)                                                   
    r_ic = (
        V_PREDICT(params_u, IC_POINTS)
        - U_IC_TARGET
    ).reshape(-1)                                                   

    return jnp.concatenate(
        [
            SCALE_U_PDE * r_pde,
            SCALE_U_DIR * r_dir,
            SCALE_U_NEU * r_neu,
            SCALE_U_IC * r_ic,
        ]
    )                                                               # (N_XY_PDE*N_T_PDE*2 + N_SIDE*N_T_BOUNDARY*2 + N_SIDE*N_T_BOUNDARY*2 + N_XY_IC*2,)


@jit
def residual_pt_flat(
    flat_pt: Array,
    flat_u_old: Array,
    flat_pt_old: Array,
) -> Array:
    params_pt = UNRAVEL_PT(flat_pt)
    params_u_old = UNRAVEL_U(flat_u_old)
    params_pt_old = UNRAVEL_PT(flat_pt_old)

    r_pde = V_FLOW_HEAT_RES(
        params_pt,
        params_u_old,
        params_pt_old,
        PDE_POINTS,
    ).reshape(-1)
    r_dir = (
        V_PREDICT(params_pt, PT_DIR_POINTS)
        - PT_DIR_TARGET
    ).reshape(-1)
    r_ic = (
        V_PREDICT(params_pt, IC_POINTS)
        - PT_IC_TARGET
    ).reshape(-1)

    return jnp.concatenate(
        [
            SCALE_PT_PDE * r_pde,
            SCALE_PT_DIR * r_dir,
            SCALE_PT_IC * r_ic,
        ]
    )


# Compile the explicit Jacobians once per fixed array shape.  These are the
# parameter Jacobians of the same weighted residual vectors used by MF-GN/LM.
# Explicit parameter Jacobians have shape (scalar residual count, parameter count).
JACOBIAN_U_FLAT = jit(jacfwd(residual_u_flat, argnums=0))
JACOBIAN_PT_FLAT = jit(jacfwd(residual_pt_flat, argnums=0))


# =============================================================================
# 7b. Exact pointwise-Jacrev streaming Gram backend
# =============================================================================

# The local residual at every collocation point has only two outputs.  Reverse
# mode is therefore attractive locally even though the full residual vector is
# much longer than the parameter vector.  The functions below differentiate
# each two-component local residual with respect to the 1250 network parameters,
# vectorize over a static point block, and accumulate J_i^T J_i and J_i^T r_i.
# The full M x P Jacobian is never materialized by this backend.


def build_pt_frozen_pde_terms(
    flat_u_old: Array,
    flat_pt_old: Array,
) -> Array:
    """Terms in the GFS p/T PDE that are frozen during one inner solve."""
    params_u_old = UNRAVEL_U(flat_u_old)
    params_pt_old = UNRAVEL_PT(flat_pt_old)

    hess_u_old = vmap(U_HESS, in_axes=(None, 0))(
        params_u_old, PDE_POINTS
    ) 
    jac_pt_old = vmap(PT_JAC, in_axes=(None, 0))(
        params_pt_old, PDE_POINTS
    ) 
    div_u_t_old = (
        hess_u_old[:, 0, 0, 2] + hess_u_old[:, 1, 1, 2]
    ) 
    p_t_old = jac_pt_old[:, 0, 2]
    t_t_old = jac_pt_old[:, 1, 2]

    frozen_p = (
        ALPHA * div_u_t_old
        - S11 * p_t_old
        - S12 * t_t_old
        - MASS_SOURCE_PDE
    )
    frozen_t = (
        BETA * div_u_t_old
        - S21 * p_t_old
        - S22 * t_t_old
        - HEAT_SOURCE_PDE
    )
    return jnp.stack([frozen_p, frozen_t], axis=1)


BUILD_PT_FROZEN_PDE_TERMS = jit(build_pt_frozen_pde_terms)


def build_u_frozen_terms(flat_pt: Array) -> Tuple[Array, Array]:
    """Current p/T contributions frozen during one mechanics inner solve."""
    params_pt = UNRAVEL_PT(flat_pt)

    jac_pt_pde = vmap(PT_JAC, in_axes=(None, 0))(
        params_pt, PDE_POINTS
    ) 
    pde_frozen = (
        ALPHA * jac_pt_pde[:, 0, :2]
        + BETA * jac_pt_pde[:, 1, :2]
        - BODY_FORCE_PDE
    ) 

    pt_neu = V_PREDICT(params_pt, U_NEU_POINTS)
    coupling_neu = ALPHA * pt_neu[:, 0] + BETA * pt_neu[:, 1]
    # Local mechanics residual is elastic_traction(u) + frozen_neu.
    frozen_neu = (
        -coupling_neu[:, None] * U_NEU_NORMALS
        - U_NEU_EXACT_TRACTION
    ) 
    return pde_frozen, frozen_neu


BUILD_U_FROZEN_TERMS = jit(build_u_frozen_terms)


def pt_pde_local_flat(
    flat_pt: Array,
    xyt: Array,
    frozen_term: Array,
) -> Array:
    '''Compute the local GFS p/T PDE residual at one point.'''
    params_pt = UNRAVEL_PT(flat_pt)
    jac_q = PT_JAC(params_pt, xyt)
    hess_q = PT_HESS(params_pt, xyt)

    p_t, t_t = jac_q[0, 2], jac_q[1, 2]
    lap_p = hess_q[0, 0, 0] + hess_q[0, 1, 1]
    lap_t = hess_q[1, 0, 0] + hess_q[1, 1, 1]

    return SCALE_PT_PDE * jnp.array(
        [
            (C0 + S11) * p_t
            + (-B0 + S12) * t_t
            - KAPPA * lap_p
            + frozen_term[0],
            (-B0 + S21) * p_t
            + (A0 + S22) * t_t
            - THETA * lap_t
            + frozen_term[1],
        ]
    )


def pt_dir_local_flat(flat_pt: Array, xyt: Array, target: Array) -> Array:
    return SCALE_PT_DIR * (predict(UNRAVEL_PT(flat_pt), xyt) - target)


def pt_ic_local_flat(flat_pt: Array, xyt: Array, target: Array) -> Array:
    return SCALE_PT_IC * (predict(UNRAVEL_PT(flat_pt), xyt) - target)


def elastic_traction_u_only_flat(flat_u: Array, xyt: Array) -> Array:
    '''Compute the local elastic traction at one point, ignoring p/T contributions.'''
    params_u = UNRAVEL_U(flat_u)
    jac_u = U_JAC(params_u, xyt)
    du1_dx, du1_dy = jac_u[0, 0], jac_u[0, 1]
    du2_dx, du2_dy = jac_u[1, 0], jac_u[1, 1]

    sigma11 = LAMBDA * (du1_dx + du2_dy) + 2.0 * MU * du1_dx
    sigma22 = LAMBDA * (du1_dx + du2_dy) + 2.0 * MU * du2_dy
    sigma12 = MU * (du1_dy + du2_dx)
    sigma = jnp.array([[sigma11, sigma12], [sigma12, sigma22]])
    return sigma @ outward_normal(xyt)


def u_pde_local_flat(
    flat_u: Array,
    xyt: Array,
    frozen_term: Array,
) -> Array:
    params_u = UNRAVEL_U(flat_u)
    return SCALE_U_PDE * (
        elastic_operator_from_hessian(U_HESS(params_u, xyt))
        + frozen_term
    )


def u_dir_local_flat(flat_u: Array, xyt: Array, target: Array) -> Array:
    return SCALE_U_DIR * (predict(UNRAVEL_U(flat_u), xyt) - target)


def u_neu_local_flat(
    flat_u: Array,
    xyt: Array,
    frozen_term: Array,
) -> Array:
    return SCALE_U_NEU * (
        elastic_traction_u_only_flat(flat_u, xyt) + frozen_term
    )


def u_ic_local_flat(flat_u: Array, xyt: Array, target: Array) -> Array:
    return SCALE_U_IC * (predict(UNRAVEL_U(flat_u), xyt) - target)


@jit
def residual_pt_cached_flat(
    flat_pt: Array,
    pde_frozen: Array,
) -> Array:
    """Exact weighted p/T residual using one outer iteration's cached data."""

    r_pde = vmap(pt_pde_local_flat, in_axes=(None, 0, 0))(
        flat_pt, PDE_POINTS, pde_frozen
    ).reshape(-1)
    r_dir = vmap(pt_dir_local_flat, in_axes=(None, 0, 0))(
        flat_pt, PT_DIR_POINTS, PT_DIR_TARGET
    ).reshape(-1)
    r_ic = vmap(pt_ic_local_flat, in_axes=(None, 0, 0))(
        flat_pt, IC_POINTS, PT_IC_TARGET
    ).reshape(-1)
    return jnp.concatenate((r_pde, r_dir, r_ic))


@jit
def residual_u_cached_flat(
    flat_u: Array,
    pde_frozen: Array,
    neu_frozen: Array,
) -> Array:
    """Exact weighted mechanics residual using cached p/T contributions."""

    r_pde = vmap(u_pde_local_flat, in_axes=(None, 0, 0))(
        flat_u, PDE_POINTS, pde_frozen
    ).reshape(-1)
    r_dir = vmap(u_dir_local_flat, in_axes=(None, 0, 0))(
        flat_u, U_DIR_POINTS, U_DIR_TARGET
    ).reshape(-1)
    r_neu = vmap(u_neu_local_flat, in_axes=(None, 0, 0))(
        flat_u, U_NEU_POINTS, neu_frozen
    ).reshape(-1)
    r_ic = vmap(u_ic_local_flat, in_axes=(None, 0, 0))(
        flat_u, IC_POINTS, U_IC_TARGET
    ).reshape(-1)
    return jnp.concatenate((r_pde, r_dir, r_neu, r_ic))


# jacrev(..., has_aux=True) returns the local Jacobian and the already-computed
# local residual, avoiding a second primal evaluation inside the point block.
# has_aux returns residual values and their local parameter Jacobians together.
def _with_aux(function: Callable[..., Array]) -> Callable[..., Tuple[Array, Array]]:
    def wrapped(*args):
        value = function(*args)
        return value, value # The auxiliary output is returned unchanged and is not differentiated.
    return wrapped

# Reverse-mode parameter Jacobians for the seven local residual types.
PT_PDE_LOCAL_JAC_AUX = jacrev(_with_aux(pt_pde_local_flat), argnums=0, has_aux=True)
PT_DIR_LOCAL_JAC_AUX = jacrev(_with_aux(pt_dir_local_flat), argnums=0, has_aux=True)
PT_IC_LOCAL_JAC_AUX = jacrev(_with_aux(pt_ic_local_flat), argnums=0, has_aux=True)
U_PDE_LOCAL_JAC_AUX = jacrev(_with_aux(u_pde_local_flat), argnums=0, has_aux=True)
U_DIR_LOCAL_JAC_AUX = jacrev(_with_aux(u_dir_local_flat), argnums=0, has_aux=True)
U_NEU_LOCAL_JAC_AUX = jacrev(_with_aux(u_neu_local_flat), argnums=0, has_aux=True)
U_IC_LOCAL_JAC_AUX = jacrev(_with_aux(u_ic_local_flat), argnums=0, has_aux=True)


def _block_contribution(
    jacobian_block: Array,
    residual_block: Array,
    mask_block: Array,
) -> Tuple[Array, Array, Array]:
    """Accumulate loss, Gram matrix, and gradient from one masked point block.

    The local Jacobian has shape (B, components, P). Flattening the first
    two axes gives J; the returned quantities are 0.5*r.T@r, J.T@J, J.T@r."""
    mask_residual = mask_block[:, None]       # Broadcast the mask over residual components.
    mask_jacobian = mask_block[:, None, None] # Broadcast the mask over components and parameters.
    residual_block = residual_block * mask_residual
    jacobian_block = jacobian_block * mask_jacobian

    jacobian_2d = jacobian_block.reshape(-1, jacobian_block.shape[-1]) # Flatten points and residual components; retain the parameter axis.
    residual_1d = residual_block.reshape(-1)                           
    loss = 0.5 * jnp.vdot(residual_1d, residual_1d).real               # Masked block loss.
    gram = jacobian_2d.T @ jacobian_2d                                 # Block Gram matrix (P,P).
    gradient = jacobian_2d.T @ residual_1d                             # Block parameter gradient (P,).
    return loss, gram, gradient


def _scan_pt_pde(
    flat_pt: Array,
    point_blocks: Array,
    frozen_blocks: Array,
    mask_blocks: Array,
) -> Tuple[Array, Array, Array]:
    """Scan all pressure-temperature PDE blocks and sum loss, Gram, and gradient."""
    parameter_count = flat_pt.shape[0]
    initial = (
        jnp.array(0.0, dtype=flat_pt.dtype),
        jnp.zeros((parameter_count, parameter_count), dtype=flat_pt.dtype),
        jnp.zeros((parameter_count,), dtype=flat_pt.dtype),
    )

    def body(carry, values):
        """Accumulate one point block; retain only the running totals."""
        points, frozen, mask = values # points.shape = (B,3), frozen.shape = (B,2), mask.shape = (B,)
        jacobian, residual = vmap(
            PT_PDE_LOCAL_JAC_AUX,
            in_axes=(None, 0, 0),
        )(flat_pt, points, frozen)    # Vectorize the local reverse-mode Jacobian over points.
        block = _block_contribution(jacobian, residual, mask)
        return tuple(first + second for first, second in zip(carry, block)), None # Keep only running totals; no per-block scan outputs.

    result, _ = lax.scan(
        body, initial, (point_blocks, frozen_blocks, mask_blocks)
    )
    return result


def _scan_pt_dir(flat_pt, point_blocks, target_blocks, mask_blocks):
    parameter_count = flat_pt.shape[0]
    initial = (
        jnp.array(0.0, dtype=flat_pt.dtype),
        jnp.zeros((parameter_count, parameter_count), dtype=flat_pt.dtype),
        jnp.zeros((parameter_count,), dtype=flat_pt.dtype),
    )

    def body(carry, values):
        points, targets, mask = values
        jacobian, residual = vmap(
            PT_DIR_LOCAL_JAC_AUX,
            in_axes=(None, 0, 0),
        )(flat_pt, points, targets)
        block = _block_contribution(jacobian, residual, mask)
        return tuple(first + second for first, second in zip(carry, block)), None

    result, _ = lax.scan(
        body, initial, (point_blocks, target_blocks, mask_blocks)
    )
    return result


def _scan_pt_ic(flat_pt, point_blocks, target_blocks, mask_blocks):
    parameter_count = flat_pt.shape[0]
    initial = (
        jnp.array(0.0, dtype=flat_pt.dtype),
        jnp.zeros((parameter_count, parameter_count), dtype=flat_pt.dtype),
        jnp.zeros((parameter_count,), dtype=flat_pt.dtype),
    )

    def body(carry, values):
        points, targets, mask = values
        jacobian, residual = vmap(
            PT_IC_LOCAL_JAC_AUX,
            in_axes=(None, 0, 0),
        )(flat_pt, points, targets)
        block = _block_contribution(jacobian, residual, mask)
        return tuple(first + second for first, second in zip(carry, block)), None

    result, _ = lax.scan(
        body, initial, (point_blocks, target_blocks, mask_blocks)
    )
    return result


def _scan_u_pde(flat_u, point_blocks, frozen_blocks, mask_blocks):
    parameter_count = flat_u.shape[0]
    initial = (
        jnp.array(0.0, dtype=flat_u.dtype),
        jnp.zeros((parameter_count, parameter_count), dtype=flat_u.dtype),
        jnp.zeros((parameter_count,), dtype=flat_u.dtype),
    )

    def body(carry, values):
        points, frozen, mask = values
        jacobian, residual = vmap(
            U_PDE_LOCAL_JAC_AUX,
            in_axes=(None, 0, 0),
        )(flat_u, points, frozen)
        block = _block_contribution(jacobian, residual, mask)
        return tuple(first + second for first, second in zip(carry, block)), None

    result, _ = lax.scan(
        body, initial, (point_blocks, frozen_blocks, mask_blocks)
    )
    return result


def _scan_u_dir(flat_u, point_blocks, target_blocks, mask_blocks):
    parameter_count = flat_u.shape[0]
    initial = (
        jnp.array(0.0, dtype=flat_u.dtype),
        jnp.zeros((parameter_count, parameter_count), dtype=flat_u.dtype),
        jnp.zeros((parameter_count,), dtype=flat_u.dtype),
    )

    def body(carry, values):
        points, targets, mask = values
        jacobian, residual = vmap(
            U_DIR_LOCAL_JAC_AUX,
            in_axes=(None, 0, 0),
        )(flat_u, points, targets)
        block = _block_contribution(jacobian, residual, mask)
        return tuple(first + second for first, second in zip(carry, block)), None

    result, _ = lax.scan(
        body, initial, (point_blocks, target_blocks, mask_blocks)
    )
    return result


def _scan_u_neu(flat_u, point_blocks, frozen_blocks, mask_blocks):
    parameter_count = flat_u.shape[0]
    initial = (
        jnp.array(0.0, dtype=flat_u.dtype),
        jnp.zeros((parameter_count, parameter_count), dtype=flat_u.dtype),
        jnp.zeros((parameter_count,), dtype=flat_u.dtype),
    )

    def body(carry, values):
        points, frozen, mask = values
        jacobian, residual = vmap(
            U_NEU_LOCAL_JAC_AUX,
            in_axes=(None, 0, 0),
        )(flat_u, points, frozen)
        block = _block_contribution(jacobian, residual, mask)
        return tuple(first + second for first, second in zip(carry, block)), None

    result, _ = lax.scan(
        body, initial, (point_blocks, frozen_blocks, mask_blocks)
    )
    return result


def _scan_u_ic(flat_u, point_blocks, target_blocks, mask_blocks):
    parameter_count = flat_u.shape[0]
    initial = (
        jnp.array(0.0, dtype=flat_u.dtype),
        jnp.zeros((parameter_count, parameter_count), dtype=flat_u.dtype),
        jnp.zeros((parameter_count,), dtype=flat_u.dtype),
    )

    def body(carry, values):
        points, targets, mask = values
        jacobian, residual = vmap(
            U_IC_LOCAL_JAC_AUX,
            in_axes=(None, 0, 0),
        )(flat_u, points, targets)
        block = _block_contribution(jacobian, residual, mask)
        return tuple(first + second for first, second in zip(carry, block)), None

    result, _ = lax.scan(
        body, initial, (point_blocks, target_blocks, mask_blocks)
    )
    return result


@jit
def _pointwise_pt_gram_gradient_kernel(
    flat_pt: Array,
    pde_points: Array,
    pde_frozen: Array,
    pde_mask: Array,
    dir_points: Array,
    dir_targets: Array,
    dir_mask: Array,
    ic_points: Array,
    ic_targets: Array,
    ic_mask: Array,
) -> Tuple[Array, Array, Array]:
    """Fuse all residual blocks into one top-level device dispatch."""
    """Fuse all p/T residual blocks into one device dispatch."""

    pde = _scan_pt_pde(flat_pt, pde_points, pde_frozen, pde_mask)
    boundary = _scan_pt_dir(flat_pt, dir_points, dir_targets, dir_mask)
    initial = _scan_pt_ic(flat_pt, ic_points, ic_targets, ic_mask)
    return tuple(
        pde[index] + boundary[index] + initial[index]
        for index in range(3)
    )


@jit
def _pointwise_u_gram_gradient_kernel(
    flat_u: Array,
    pde_points: Array,
    pde_frozen: Array,
    pde_mask: Array,
    dir_points: Array,
    dir_targets: Array,
    dir_mask: Array,
    neu_points: Array,
    neu_frozen: Array,
    neu_mask: Array,
    ic_points: Array,
    ic_targets: Array,
    ic_mask: Array,
) -> Tuple[Array, Array, Array]:
    """Fuse all mechanics residual blocks into one device dispatch."""

    pde = _scan_u_pde(flat_u, pde_points, pde_frozen, pde_mask)
    boundary = _scan_u_dir(flat_u, dir_points, dir_targets, dir_mask)
    neumann = _scan_u_neu(flat_u, neu_points, neu_frozen, neu_mask)
    initial = _scan_u_ic(flat_u, ic_points, ic_targets, ic_mask)
    return tuple(
        pde[index] + boundary[index] + neumann[index] + initial[index]
        for index in range(3)
    )


# Debug/compatibility path: dispatch residual blocks separately. These kernels
# compile only when THP_FUSE_GRAM_BLOCKS=0 is selected.
# Lazy JIT wrappers support separate scans when fusion is disabled.
SCAN_PT_PDE = jit(_scan_pt_pde)
SCAN_PT_DIR = jit(_scan_pt_dir)
SCAN_PT_IC = jit(_scan_pt_ic)
SCAN_U_PDE = jit(_scan_u_pde)
SCAN_U_DIR = jit(_scan_u_dir)
SCAN_U_NEU = jit(_scan_u_neu)
SCAN_U_IC = jit(_scan_u_ic)


def _blockify(
    points: Array,
    data: Array,
    block_size: int,
) -> Tuple[Array, Array, Array]:
    """Pad aligned point/data arrays into fixed-size blocks and return a mask.

    Padded rows have zero mask and contribute nothing to loss, Gram, or gradient."""
    """Pad to a static point-block shape and return a validity mask."""
    point_count = int(points.shape[0])
    padded_count = ((point_count + block_size - 1) // block_size) * block_size # Round the point count up to a block-size multiple.
    pad_count = padded_count - point_count                                     

    points_padded = jnp.pad(points, ((0, pad_count), (0, 0)))                  
    data_padded = jnp.pad(data, ((0, pad_count), (0, 0)))                      
    mask = jnp.concatenate(
        [
            jnp.ones((point_count,), dtype=points.dtype),
            jnp.zeros((pad_count,), dtype=points.dtype),
        ]
    )                                                                          # Masked padded rows do not affect loss, Gram, or gradient.
    block_count = padded_count // block_size                                   
    return (
        points_padded.reshape(block_count, block_size, points.shape[1]),       # Point blocks (K,B,3).
        data_padded.reshape(block_count, block_size, data.shape[1]),           # Data blocks (K,B,components).
        mask.reshape(block_count, block_size),                                 # Mask blocks (K,B).
    )


@dataclass(frozen=True)
class PointwisePTContext:
    """Coordinates, frozen terms, targets, and masks for pressure-temperature blocks."""
    pde_points: Array
    pde_frozen: Array
    pde_mask: Array
    dir_points: Array
    dir_targets: Array
    dir_mask: Array
    ic_points: Array
    ic_targets: Array
    ic_mask: Array


@dataclass(frozen=True)
class PointwiseUContext:
    """Coordinates, frozen terms, targets, and masks for mechanics blocks."""
    pde_points: Array
    pde_frozen: Array
    pde_mask: Array
    dir_points: Array
    dir_targets: Array
    dir_mask: Array
    neu_points: Array
    neu_frozen: Array
    neu_mask: Array
    ic_points: Array
    ic_targets: Array
    ic_mask: Array


def build_pointwise_pt_context(
    flat_u_old: Array,
    flat_pt_old: Array,
    block_size: int,
) -> PointwisePTContext:
    """Build point blocks from frozen previous displacement and pressure-temperature states."""
    frozen = BUILD_PT_FROZEN_PDE_TERMS(flat_u_old, flat_pt_old)     
    frozen.block_until_ready()                                      # Synchronize frozen terms so context construction is timed correctly.
    pde = _blockify(PDE_POINTS, frozen, block_size)                 
    boundary = _blockify(PT_DIR_POINTS, PT_DIR_TARGET, block_size)  
    initial = _blockify(IC_POINTS, PT_IC_TARGET, block_size)        
    # ``jnp.pad`` is asynchronous on GPU.  Synchronizing the finished context
    # here keeps context construction in ``context_seconds`` instead of
    # accidentally charging its tail to the first Gram assembly.
    for value in (*pde, *boundary, *initial):
        value.block_until_ready()
    return PointwisePTContext(*pde, *boundary, *initial)            # Field order follows each (points,data,mask) block triple.


def build_pointwise_u_context(
    flat_pt: Array,
    block_size: int,
) -> PointwiseUContext:
    """Build mechanics blocks using the newly updated pressure-temperature state."""
    frozen_pde, frozen_neu = BUILD_U_FROZEN_TERMS(flat_pt)
    frozen_pde.block_until_ready()
    frozen_neu.block_until_ready()
    pde = _blockify(PDE_POINTS, frozen_pde, block_size)
    boundary = _blockify(U_DIR_POINTS, U_DIR_TARGET, block_size)
    neumann = _blockify(U_NEU_POINTS, frozen_neu, block_size)
    initial = _blockify(IC_POINTS, U_IC_TARGET, block_size)
    # See ``build_pointwise_pt_context``: close the asynchronous context work
    # before its timing interval ends.
    for value in (*pde, *boundary, *neumann, *initial):
        value.block_until_ready()
    return PointwiseUContext(*pde, *boundary, *neumann, *initial)


def pointwise_pt_gram_gradient(
    flat_pt: Array,
    context: PointwisePTContext,
) -> Tuple[Array, Array, Array]:
    """Return total loss, Gram, and gradient for the pressure-temperature context."""
    if not FUSE_GRAM_BLOCKS: # Without fusion, evaluate the residual scans separately and sum them.
        pde = SCAN_PT_PDE(
            flat_pt, context.pde_points, context.pde_frozen, context.pde_mask
        )
        boundary = SCAN_PT_DIR(
            flat_pt, context.dir_points, context.dir_targets, context.dir_mask
        )
        initial = SCAN_PT_IC(
            flat_pt, context.ic_points, context.ic_targets, context.ic_mask
        )
        return tuple(sum(values) for values in zip(pde, boundary, initial)) 

    return _pointwise_pt_gram_gradient_kernel(
        flat_pt,
        context.pde_points,
        context.pde_frozen,
        context.pde_mask,
        context.dir_points,
        context.dir_targets,
        context.dir_mask,
        context.ic_points,
        context.ic_targets,
        context.ic_mask,
    ) # The fused path uses one top-level device dispatch.


def pointwise_u_gram_gradient(
    flat_u: Array,
    context: PointwiseUContext,
) -> Tuple[Array, Array, Array]:
    """Return total loss, Gram, and gradient for the mechanics context."""
    if not FUSE_GRAM_BLOCKS:
        pde = SCAN_U_PDE(
            flat_u, context.pde_points, context.pde_frozen, context.pde_mask
        )
        boundary = SCAN_U_DIR(
            flat_u, context.dir_points, context.dir_targets, context.dir_mask
        )
        neumann = SCAN_U_NEU(
            flat_u, context.neu_points, context.neu_frozen, context.neu_mask
        )
        initial = SCAN_U_IC(
            flat_u, context.ic_points, context.ic_targets, context.ic_mask
        )
        return tuple(
            sum(values) for values in zip(pde, boundary, neumann, initial)
        )

    return _pointwise_u_gram_gradient_kernel(
        flat_u,
        context.pde_points,
        context.pde_frozen,
        context.pde_mask,
        context.dir_points,
        context.dir_targets,
        context.dir_mask,
        context.neu_points,
        context.neu_frozen,
        context.neu_mask,
        context.ic_points,
        context.ic_targets,
        context.ic_mask,
    )


# =============================================================================
# 8. Explicit ENGD: EIGH pseudoinverse + iterative refinement
# =============================================================================


def _optional_float_from_env(name: str) -> Optional[float]:
    """Read an optional floating-point environment value; unset/empty returns None."""
    value = os.environ.get(name)
    if value is None or value.strip().lower() in {"", "none", "null"}:
        return None
    return float(value)


ENGD_SOLVER = os.environ.get(
    "THP_ENGD_SOLVER", "eigh-refined"
).strip().lower()                                                   
if ENGD_SOLVER not in {"eigh-refined", "eigh", "lstsq"}:            
    raise ValueError(
        "THP_ENGD_SOLVER must be one of 'eigh-refined', 'eigh', or "
        f"'lstsq', but received {ENGD_SOLVER!r}."
    )

# Optional validation only. Production runs leave this disabled.
COMPARE_LSTSQ_REFERENCE = (
    os.environ.get("THP_EIGH_COMPARE_LSTSQ", "0") == "1"
)                                                                  

RELATIVE_CUTOFF = float(
    os.environ.get("THP_EIGH_RELATIVE_CUTOFF", "0.0")
)                                                                  # Zero requested cutoff still retains the automatic floating-point cutoff.
REFINEMENT_STEPS = int(
    os.environ.get("THP_EIGH_REFINEMENT_STEPS", "2")
)                                                                  
LSTSQ_RCOND = _optional_float_from_env("THP_LSTSQ_RCOND")          # None delegates rcond selection to the least-squares solver.
GRAM_RIDGE = float(os.environ.get("THP_GRAM_RIDGE", "1e-7"))        
LINE_SEARCH_LEVELS = int(
    os.environ.get("THP_LINE_SEARCH_LEVELS", "31")
)                                                                  # Search the 31 steps {1,1/2,...,2^-30}.

JACOBIAN_BACKEND = os.environ.get(
    "THP_JACOBIAN_BACKEND", "pointwise"
).strip().lower()                                                  # Select streamed point blocks or an explicitly materialized jacfwd Jacobian.
if JACOBIAN_BACKEND not in {"jacfwd", "pointwise"}:
    raise ValueError(
        "THP_JACOBIAN_BACKEND must be 'jacfwd' or 'pointwise', "
        f"but received {JACOBIAN_BACKEND!r}."
    )

POINT_BLOCK_PT = int(os.environ.get("THP_POINT_JAC_BLOCK_PT", "512"))      
POINT_BLOCK_U = int(os.environ.get("THP_POINT_JAC_BLOCK_U", "512"))        
FUSE_GRAM_BLOCKS = os.environ.get("THP_FUSE_GRAM_BLOCKS", "1") == "1"      
CACHE_LINE_SEARCH = os.environ.get("THP_CACHE_LINE_SEARCH", "1") == "1"    
JACOBIAN_MICROBENCH = os.environ.get("THP_JACOBIAN_MICROBENCH", "0") == "1"
MICROBENCH_WARMUP = int(os.environ.get("THP_MICROBENCH_WARMUP", "2"))      
MICROBENCH_REPEATS = int(os.environ.get("THP_MICROBENCH_REPEATS", "10"))   
MICROBENCH_BLOCK_SIZES = tuple(
    int(value.strip())
    for value in os.environ.get(
        "THP_MICROBENCH_BLOCK_SIZES", "64,128,256,512,1024"
    ).split(",")
    if value.strip()
)                                                                          

if RELATIVE_CUTOFF < 0.0:
    raise ValueError("THP_EIGH_RELATIVE_CUTOFF must be non-negative.")
if REFINEMENT_STEPS < 0:
    raise ValueError("THP_EIGH_REFINEMENT_STEPS must be non-negative.")
if GRAM_RIDGE < 0.0:
    raise ValueError("THP_GRAM_RIDGE must be non-negative.")
if LINE_SEARCH_LEVELS <= 0:
    raise ValueError("THP_LINE_SEARCH_LEVELS must be positive.")
for name, value in (
    ("THP_POINT_JAC_BLOCK_PT", POINT_BLOCK_PT),
    ("THP_POINT_JAC_BLOCK_U", POINT_BLOCK_U),
):
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
if MICROBENCH_WARMUP < 0 or MICROBENCH_REPEATS <= 0:
    raise ValueError(
        "Microbenchmark warmup must be nonnegative and repeats positive."
    )
if (
    not MICROBENCH_BLOCK_SIZES
    or any(value <= 0 for value in MICROBENCH_BLOCK_SIZES)
):
    raise ValueError(
        "THP_MICROBENCH_BLOCK_SIZES must contain positive integers."
    )


@dataclass(frozen=True)
class ENGDConfig:
    """Configuration of one exact ENGD update."""
    """Immutable energy-natural direction configuration."""
    line_search_levels: int = 31           
    gram_ridge: float = 0.0                
    solver: str = "eigh-refined"           
    compare_lstsq_reference: bool = False  # Optional same-Gram least-squares direction check.
    relative_cutoff: float = 0.0           # Requested relative spectral cutoff.
    refinement_steps: int = 2              # Spectral iterative refinement count.
    lstsq_rcond: Optional[float] = None    
    jacobian_backend: str = "pointwise"    
    gradient_tolerance: float = 1.0e-14    # Skip updates when the parameter gradient is sufficiently small.
    require_loss_decrease: bool = True     # Reject the update unless an actual candidate loss decreases.


@dataclass(frozen=True)
class DenseSolveResult:
    """Dense direction solve with linear-residual and spectral diagnostics."""
    direction: Array            
    relative_residual: float    # Relative linear-system residual.
    min_eigenvalue: float       
    max_eigenvalue: float       
    min_kept_eigenvalue: float  # Smallest retained shifted eigenvalue.
    effective_rank: int         # Number of retained spectral modes.
    spectral_cutoff: float      
    effective_condition: float  # Largest divided by smallest retained shifted eigenvalue.
    predicted_reduction: float  # Predicted decrease in the local quadratic model.
    negative_count: int         
    max_negative_abs: float     
    negative_above_cutoff: int  # Negative eigenvalues whose magnitude exceeds the cutoff.
# A Gram matrix is positive semidefinite; negative eigenvalues reflect numerical error.

@dataclass
class ENGDStepStats:
    """Record update acceptance, spectral diagnostics, direction checks, and timings."""
    accepted: bool               
    loss_before: float           
    loss_after: float            
    step_size: float             
    solver_used: str             
    gradient_norm: float         
    natural_gradient_norm: float 

    # Spectral and linear-system diagnostics.
    effective_rank: int
    spectral_cutoff: float
    lambda_max: float
    lambda_min: float
    lambda_min_kept: float
    effective_condition: float
    negative_count: int
    max_negative_abs: float
    negative_above_cutoff: int
    solve_relative_residual: float
    predicted_reduction: float
    refinement_steps: int

    # Optional same-Gram LSTSQ validation. NaN/-1 when disabled.
    lstsq_rank: int
    lstsq_effective_condition: float
    lstsq_solve_relative_residual: float
    direction_relative_l2_error: float
    direction_cosine: float
    direction_relative_energy_error: float

    # Timings synchronize the device before the interval is closed.
    residual_seconds: float
    jacobian_seconds: float
    gram_matmul_seconds: float
    gradient_matmul_seconds: float
    symmetrize_seconds: float
    gram_symmetry_relative_error: float # Relative Frobenius asymmetry of the assembled Gram matrix.
    assembly_seconds: float
    solve_seconds: float
    reference_seconds: float
    line_search_seconds: float


PROFILE_ROWS: List[Dict[str, object]] = [] 


@jit
def half_squared_norm(values: Array) -> Array:
    """Return the scalar 0.5*||values||_2^2."""
    return 0.5 * jnp.vdot(values, values).real


@jit
def candidate_losses_pt(
    candidates: Array,
    flat_u_old: Array,
    flat_pt_old: Array,
) -> Array:
    """Evaluate all p/T grid-search candidates in one compiled vmap."""
    """Evaluate uncached pressure-temperature residual losses at all candidates."""
    return vmap(
        lambda candidate: half_squared_norm(
            residual_pt_flat(candidate, flat_u_old, flat_pt_old)
        )
    )(candidates) 


@jit
def candidate_losses_u(
    candidates: Array,
    flat_pt: Array,
) -> Array:
    """Evaluate all mechanics grid-search candidates in one compiled vmap."""
    """Evaluate uncached mechanics residual losses at all candidates."""
    return vmap(
        lambda candidate: half_squared_norm(
            residual_u_flat(candidate, flat_pt)
        )
    )(candidates) 


@jit
def candidate_losses_pt_cached(
    candidates: Array,
    pde_frozen: Array,
) -> Array:
    """Vectorized p/T line-search losses with frozen GFS terms reused."""
    """Evaluate candidate losses using cached pressure-temperature frozen terms."""
    return vmap(
        lambda candidate: half_squared_norm(
            residual_pt_cached_flat(candidate, pde_frozen) # Reuse frozen network derivatives for every candidate step.
        )
    )(candidates)


@jit
def candidate_losses_u_cached(
    candidates: Array,
    pde_frozen: Array,
    neu_frozen: Array,
) -> Array:
    """Vectorized mechanics line-search losses with frozen p/T reused."""
    """Evaluate candidate losses using cached mechanics and traction frozen terms."""
    return vmap(
        lambda candidate: half_squared_norm(
            residual_u_cached_flat(candidate, pde_frozen, neu_frozen)
        )
    )(candidates)


@jit
def _select_best_line_search_candidate(
    candidates: Array,
    losses: Array,
    steps: Array,
) -> Tuple[Array, Array, Array]:
    """Keep finite-loss filtering and argmin on device in one kernel."""

    safe_losses = jnp.where(jnp.isfinite(losses), losses, jnp.inf)
    best_index = jnp.argmin(safe_losses)
    return (
        candidates[best_index],
        safe_losses[best_index],
        steps[best_index],
    )


def vectorized_logarithmic_line_search(
    flat_params: Array,
    natural_gradient: Array,
    candidate_losses_fn: Callable[[Array], Array],
    loss_before: float,
    config: ENGDConfig,
) -> Tuple[Array, float, float, bool]:
    """
    Select eta from {1, 1/2, ..., 2^-(m-1)} with one compiled evaluation.

    Only the original positive candidates are evaluated. If their minimum does
    not strictly improve loss_before, the update is rejected and theta is kept.
    This preserves the original serial 31-level acceptance rule exactly.
    """

    positive_steps = 0.5 ** jnp.arange(
        config.line_search_levels, dtype=flat_params.dtype
    )
    steps = positive_steps
    candidates = (
        flat_params[None, :]
        - steps[:, None] * natural_gradient[None, :]
    )
    losses = candidate_losses_fn(candidates)
    updated, best_loss_device, best_step_device = (
        _select_best_line_search_candidate(candidates, losses, steps)
    )
    best_loss, best_step = jax.device_get(
        (best_loss_device, best_step_device)
    )
    best_loss = float(best_loss)
    best_step = float(best_step)

    accepted = (
        best_step > 0.0
        and math.isfinite(best_loss)
        and (
            (not config.require_loss_decrease)
            or best_loss < loss_before
        )
    )
    if not accepted:
        return flat_params, loss_before, 0.0, False # No candidate decreases the loss: retain parameters and report rejection.

    updated.block_until_ready()
    return updated, best_loss, best_step, True


@jit
def _prepare_gram_and_diagnostics(
    gram_matrix_raw: Array,
    gradient_flat: Array,
) -> Tuple[Array, Array, Array]:
    """Symmetrize G and compute scalar diagnostics without host round-trips."""

    antisymmetric_part = gram_matrix_raw - gram_matrix_raw.T
    gram_norm = jnp.linalg.norm(gram_matrix_raw)
    symmetry_error = (
        jnp.linalg.norm(antisymmetric_part)
        / jnp.maximum(gram_norm, jnp.array(1.0e-30, gram_norm.dtype))
    )                                                                 # Relative Frobenius asymmetry before symmetrization.
    gram_matrix = 0.5 * (gram_matrix_raw + gram_matrix_raw.T)         # Enforce symmetry.
    gradient_norm = jnp.linalg.norm(gradient_flat)
    return gram_matrix, symmetry_error, gradient_norm                 # Return diagnostics together to minimize host-device transfers.


def _eigh_refined_kernel(
    gram: Array,
    gradient_flat: Array,
    damping: float,
    relative_cutoff: float,
    refinement_steps: int,
) -> Tuple[Array, ...]:
    """Fixed-shape EIGH pseudoinverse + residual refinement."""
    gram = 0.5 * (gram + gram.T)                            
    eigenvalues, eigenvectors = jnp.linalg.eigh(gram)       
    damping_array = jnp.asarray(damping, dtype=gram.dtype)  
    shifted = eigenvalues + damping_array                   # Shift the spectrum by damping.

    spectral_scale = jnp.maximum(
        jnp.max(jnp.abs(shifted)),
        jnp.array(1.0, dtype=shifted.dtype),
    )                                                       
    automatic_cutoff = (
        jnp.finfo(shifted.dtype).eps
        * shifted.shape[0]
        * spectral_scale
    )                                                       # Automatic cutoff: machine epsilon times parameter count times spectral scale.
    requested_cutoff = (
        jnp.asarray(relative_cutoff, dtype=shifted.dtype)
        * spectral_scale
    )                                                       
    threshold = jnp.maximum(automatic_cutoff, requested_cutoff) # Apply the larger of the automatic and requested cutoffs.
    retained = shifted > threshold                          # Retain only shifted eigenvalues strictly above the cutoff.
    safe_shifted = jnp.where(retained, shifted, 1.0)        # Avoid division by zero in discarded modes.
    inverse = jnp.where(retained, 1.0 / safe_shifted, 0.0)  # Truncated inverse in the eigenbasis.

    def apply_inverse(vector: Array) -> Array:
        """Apply the retained spectral inverse: Q @ diag(inverse) @ Q.T @ vector."""
        coefficients = eigenvectors.T @ vector
        return eigenvectors @ (inverse * coefficients)

    direction = apply_inverse(gradient_flat) 
    operator = gram + damping_array * jnp.eye(
        gram.shape[0], dtype=gram.dtype
    ) # Form the shifted operator for residuals and iterative refinement.

    def refine(_: int, current: Array) -> Array:
        """Correct the direction with the retained inverse applied to g - A @ d."""
        residual = gradient_flat - operator @ current
        return current + apply_inverse(residual)

    direction = lax.fori_loop(
        0, refinement_steps, refine, direction
    ) # Refinement corrects roundoff within retained modes; it does not increase rank.
    linear_residual = gradient_flat - operator @ direction 
    denominator = jnp.maximum(
        jnp.linalg.norm(gradient_flat),
        jnp.array(1.0e-30, dtype=gradient_flat.dtype),
    )
    relative_residual = jnp.linalg.norm(linear_residual) / denominator # Relative residual of the shifted linear system.

    rank = jnp.sum(retained) 
    min_kept = jnp.min(
        jnp.where(retained, shifted, jnp.inf)
    )
    shifted_max = jnp.max(shifted)
    effective_condition = jnp.where(
        rank > 0,
        shifted_max / min_kept,
        jnp.inf,
    )
    predicted_reduction = (
        jnp.dot(gradient_flat, direction)
        - 0.5 * jnp.dot(direction, gram @ direction)
        - 0.5 * damping_array * jnp.dot(direction, direction)
    ) # Acceptance uses the actual line-search loss rather than predicted decrease.
    negative_mask = eigenvalues < 0.0
    negative_count = jnp.sum(negative_mask)
    max_negative_abs = jnp.max(
        jnp.where(negative_mask, -eigenvalues, 0.0)
    )
    negative_above_cutoff = jnp.sum(
        eigenvalues < -threshold
    )

    return (
        direction,
        relative_residual,
        jnp.min(eigenvalues),
        jnp.max(eigenvalues),
        min_kept,
        rank,
        threshold,
        effective_condition,
        predicted_reduction,
        negative_count,
        max_negative_abs,
        negative_above_cutoff,
        jnp.all(jnp.isfinite(direction)),
    )


_EIGH_REFINED_KERNEL = jax.jit(
    _eigh_refined_kernel,
    static_argnames=("refinement_steps",),
)


def solve_eigh_refined(
    gram: Array,
    gradient_flat: Array,
    damping: float,
    relative_cutoff: float,
    refinement_steps: int,
) -> DenseSolveResult:
    if damping < 0.0:
        raise ValueError("ENGD damping/ridge must be non-negative.")
    if relative_cutoff < 0.0:
        raise ValueError("Relative EIGH cutoff must be non-negative.")
    if refinement_steps < 0:
        raise ValueError("Refinement steps must be non-negative.")

    outputs = _EIGH_REFINED_KERNEL(
        gram,
        gradient_flat,
        damping,
        relative_cutoff,
        refinement_steps=refinement_steps,
    )
    (
        direction,
        relative_residual,
        min_eigenvalue,
        max_eigenvalue,
        min_kept_eigenvalue,
        effective_rank,
        spectral_cutoff,
        effective_condition,
        predicted_reduction,
        negative_count,
        max_negative_abs,
        negative_above_cutoff,
        direction_is_finite,
    ) = outputs
    direction.block_until_ready()

    host_diagnostics = jax.device_get(
        (
            relative_residual,
            min_eigenvalue,
            max_eigenvalue,
            min_kept_eigenvalue,
            effective_rank,
            spectral_cutoff,
            effective_condition,
            predicted_reduction,
            negative_count,
            max_negative_abs,
            negative_above_cutoff,
            direction_is_finite,
        )
    )
    (
        relative_residual,
        min_eigenvalue,
        max_eigenvalue,
        min_kept_eigenvalue,
        effective_rank,
        spectral_cutoff,
        effective_condition,
        predicted_reduction,
        negative_count,
        max_negative_abs,
        negative_above_cutoff,
        direction_is_finite,
    ) = host_diagnostics

    if not bool(direction_is_finite):
        raise FloatingPointError(
            "EIGH-refined ENGD direction contains NaN or Inf."
        )

    return DenseSolveResult(
        direction=direction,
        relative_residual=float(relative_residual),
        min_eigenvalue=float(min_eigenvalue),
        max_eigenvalue=float(max_eigenvalue),
        min_kept_eigenvalue=float(min_kept_eigenvalue),
        effective_rank=int(effective_rank),
        spectral_cutoff=float(spectral_cutoff),
        effective_condition=float(effective_condition),
        predicted_reduction=float(predicted_reduction),
        negative_count=int(negative_count),
        max_negative_abs=float(max_negative_abs),
        negative_above_cutoff=int(negative_above_cutoff),
    )


def _lstsq_result(
    gram: Array,
    gradient_flat: Array,
    damping: float,
    rcond: Optional[float],
) -> Tuple[DenseSolveResult, int]:
    """Optional same-Gram reference solve used only for validation."""

    gram = 0.5 * (gram + gram.T)
    operator = gram + damping * jnp.eye(
        gram.shape[0], dtype=gram.dtype
    )
    direction, _, rank, singular_values = jnp.linalg.lstsq(
        operator, gradient_flat, rcond=rcond
    )
    direction.block_until_ready()
    rank_host = int(jax.device_get(rank))
    singular_host = np.asarray(jax.device_get(singular_values))

    denominator = max(
        float(jax.device_get(jnp.linalg.norm(gradient_flat))),
        1.0e-30,
    )
    relative_residual = float(
        jax.device_get(
            jnp.linalg.norm(operator @ direction - gradient_flat)
        )
    ) / denominator
    if rank_host > 0 and singular_host.size:
        largest = float(singular_host[0])
        smallest = float(singular_host[rank_host - 1])
        condition = (
            largest / smallest if smallest > 0.0 else math.inf
        )
    else:
        condition = math.inf

    # LSTSQ is not the production solver, so EIGH-only diagnostics are NaN.
    result = DenseSolveResult(
        direction=direction,
        relative_residual=relative_residual,
        min_eigenvalue=math.nan,
        max_eigenvalue=math.nan,
        min_kept_eigenvalue=math.nan,
        effective_rank=rank_host,
        spectral_cutoff=math.nan,
        effective_condition=condition,
        predicted_reduction=float(
            jax.device_get(
                jnp.dot(gradient_flat, direction)
                - 0.5 * jnp.dot(direction, gram @ direction)
                - 0.5 * damping * jnp.dot(direction, direction)
            )
        ),
        negative_count=-1,
        max_negative_abs=math.nan,
        negative_above_cutoff=-1,
    )
    return result, rank_host


def _safe_cosine(first: Array, second: Array) -> float:
    """Compute direction cosine with protection against a zero denominator."""
    first_norm = float(jax.device_get(jnp.linalg.norm(first)))
    second_norm = float(jax.device_get(jnp.linalg.norm(second)))
    denominator = first_norm * second_norm
    if denominator <= 1.0e-30:
        return (
            1.0
            if first_norm <= 1.0e-30 and second_norm <= 1.0e-30
            else 0.0
        )
    return (
        float(jax.device_get(jnp.vdot(first, second).real))
        / denominator
    )


def _direction_comparison(
    candidate: Array,
    reference: Array,
    gram_matrix: Array,
) -> Tuple[float, float, float]:
    difference = candidate - reference                           
    reference_norm = float(
        jax.device_get(jnp.linalg.norm(reference))
    )                                                      
    relative_l2 = (
        float(jax.device_get(jnp.linalg.norm(difference)))       
        / max(reference_norm, 1.0e-30)
    )                                                            # Relative parameter-space direction difference.
    cosine = _safe_cosine(candidate, reference)                  

    difference_energy_sq = jnp.maximum(
        jnp.vdot(difference, gram_matrix @ difference).real, 0.0
    )                                                            # Nonnegative Gram seminorm of the direction difference.
    reference_energy_sq = jnp.maximum(
        jnp.vdot(reference, gram_matrix @ reference).real, 0.0
    )
    relative_energy = (
        float(jax.device_get(jnp.sqrt(difference_energy_sq)))
        / max(
            float(jax.device_get(jnp.sqrt(reference_energy_sq))),
            1.0e-30,
        )
    )
    return relative_l2, cosine, relative_energy


def explicit_engd_step(
    flat_params: Array,
    residual_fn: ResidualFunction,
    jacobian_fn: JacobianFunction,
    candidate_losses_fn: Callable[[Array], Array],
    config: ENGDConfig,
    gram_gradient_fn: Optional[
        Callable[[Array], Tuple[Array, Array, Array]]
    ] = None,
) -> Tuple[Array, ENGDStepStats]:
    """Perform one exact ENGD update with a selectable Gram backend."""

    total_assembly_start = time.perf_counter()                     # Assembly timing includes residuals, Jacobians, products, and symmetrization.

    if config.jacobian_backend == "pointwise":                     
        if gram_gradient_fn is None:
            raise ValueError(
                "pointwise backend requires gram_gradient_fn."
            )
        pointwise_start = time.perf_counter()
        loss_device, gram_matrix_raw, gradient_flat = (
            gram_gradient_fn(flat_params)
        )
        loss_device.block_until_ready()
        gram_matrix_raw.block_until_ready()
        gradient_flat.block_until_ready()
        jacobian_seconds = time.perf_counter() - pointwise_start   
        loss_before = float(jax.device_get(loss_device))

        # Zero component times mean the streamed kernel fuses these operations.
        residual_seconds = 0.0
        gram_matmul_seconds = 0.0
        gradient_matmul_seconds = 0.0

    elif config.jacobian_backend == "jacfwd":                      
        residual_start = time.perf_counter()
        residual = residual_fn(flat_params)
        residual.block_until_ready()
        residual_seconds = time.perf_counter() - residual_start

        jacobian_start = time.perf_counter()
        jacobian = jacobian_fn(flat_params)
        jacobian.block_until_ready()
        jacobian_seconds = time.perf_counter() - jacobian_start

        gram_start = time.perf_counter()
        gram_matrix_raw = jacobian.T @ jacobian
        gram_matrix_raw.block_until_ready()
        gram_matmul_seconds = time.perf_counter() - gram_start

        gradient_start = time.perf_counter()
        gradient_flat = jacobian.T @ residual
        gradient_flat.block_until_ready()
        gradient_matmul_seconds = (
            time.perf_counter() - gradient_start
        )
        loss_before = float(
            jax.device_get(half_squared_norm(residual))
        )
    else:
        raise ValueError(
            f"Unsupported Jacobian backend "
            f"{config.jacobian_backend!r}."
        )

    if not math.isfinite(loss_before):
        raise FloatingPointError(
            "ENGD loss contains NaN or Inf before the update."
        )

    symmetrize_start = time.perf_counter()
    (
        gram_matrix,
        gram_symmetry_error_device,
        gradient_norm_device,
    ) = _prepare_gram_and_diagnostics(
        gram_matrix_raw,
        gradient_flat,
    ) 
    gram_matrix.block_until_ready()
    gram_symmetry_relative_error, gradient_norm = jax.device_get(
        (gram_symmetry_error_device, gradient_norm_device)
    )
    gram_symmetry_relative_error = float(gram_symmetry_relative_error)
    gradient_norm = float(gradient_norm)
    symmetrize_seconds = time.perf_counter() - symmetrize_start
    assembly_seconds = time.perf_counter() - total_assembly_start 

    if not math.isfinite(gradient_norm):
        raise FloatingPointError(
            "ENGD gradient contains NaN or Inf."
        )

    nan = float("nan")
    if gradient_norm <= config.gradient_tolerance:
        return flat_params, ENGDStepStats(
            accepted=False,
            loss_before=loss_before,
            loss_after=loss_before,
            step_size=0.0,
            solver_used=config.solver,
            gradient_norm=gradient_norm,
            natural_gradient_norm=0.0,
            effective_rank=0,
            spectral_cutoff=nan,
            lambda_max=nan,
            lambda_min=nan,
            lambda_min_kept=nan,
            effective_condition=math.inf,
            negative_count=0,
            max_negative_abs=0.0,
            negative_above_cutoff=0,
            solve_relative_residual=0.0,
            predicted_reduction=0.0,
            refinement_steps=(
                config.refinement_steps
                if config.solver == "eigh-refined"
                else 0
            ),
            lstsq_rank=-1,
            lstsq_effective_condition=nan,
            lstsq_solve_relative_residual=nan,
            direction_relative_l2_error=nan,
            direction_cosine=nan,
            direction_relative_energy_error=nan,
            residual_seconds=residual_seconds,
            jacobian_seconds=jacobian_seconds,
            gram_matmul_seconds=gram_matmul_seconds,
            gradient_matmul_seconds=gradient_matmul_seconds,
            symmetrize_seconds=symmetrize_seconds,
            gram_symmetry_relative_error=(
                gram_symmetry_relative_error
            ),
            assembly_seconds=assembly_seconds,
            solve_seconds=0.0,
            reference_seconds=0.0,
            line_search_seconds=0.0,
        )

    solve_start = time.perf_counter()
    if config.solver in {"eigh-refined", "eigh"}:
        solve_result = solve_eigh_refined(
            gram_matrix,
            gradient_flat,
            damping=config.gram_ridge,
            relative_cutoff=config.relative_cutoff,
            refinement_steps=(
                config.refinement_steps
                if config.solver == "eigh-refined"
                else 0
            ),
        )
    elif config.solver == "lstsq":
        solve_result, _ = _lstsq_result(
            gram_matrix,
            gradient_flat,
            damping=config.gram_ridge,
            rcond=config.lstsq_rcond,
        )
    else:
        raise ValueError(f"Unsupported ENGD solver: {config.solver}")
    solve_result.direction.block_until_ready()
    solve_seconds = time.perf_counter() - solve_start

    reference_seconds = 0.0
    lstsq_rank = -1
    lstsq_effective_condition = nan
    lstsq_solve_relative_residual = nan
    direction_relative_l2_error = nan
    direction_cosine = nan
    direction_relative_energy_error = nan

    if (
        config.compare_lstsq_reference
        and config.solver != "lstsq"
    ):
        reference_start = time.perf_counter()
        reference_result, lstsq_rank = _lstsq_result(
            gram_matrix,
            gradient_flat,
            damping=config.gram_ridge,
            rcond=config.lstsq_rcond,
        )
        (
            direction_relative_l2_error,
            direction_cosine,
            direction_relative_energy_error,
        ) = _direction_comparison(
            solve_result.direction,
            reference_result.direction,
            gram_matrix,
        )
        lstsq_effective_condition = (
            reference_result.effective_condition
        )
        lstsq_solve_relative_residual = (
            reference_result.relative_residual
        )
        reference_seconds = (
            time.perf_counter() - reference_start
        )

    line_search_start = time.perf_counter()
    (
        updated_params,
        loss_after,
        step_size,
        accepted,
    ) = vectorized_logarithmic_line_search(
        flat_params,
        solve_result.direction,
        candidate_losses_fn,
        loss_before,
        config,
    )
    updated_params.block_until_ready()
    line_search_seconds = (
        time.perf_counter() - line_search_start
    )

    stats = ENGDStepStats(
        accepted=accepted,
        loss_before=loss_before,
        loss_after=loss_after,
        step_size=step_size,
        solver_used=config.solver,
        gradient_norm=gradient_norm,
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
        negative_above_cutoff=(
            solve_result.negative_above_cutoff
        ),
        solve_relative_residual=solve_result.relative_residual,
        predicted_reduction=solve_result.predicted_reduction,
        refinement_steps=(
            config.refinement_steps
            if config.solver == "eigh-refined"
            else 0
        ),
        lstsq_rank=lstsq_rank,
        lstsq_effective_condition=lstsq_effective_condition,
        lstsq_solve_relative_residual=(
            lstsq_solve_relative_residual
        ),
        direction_relative_l2_error=(
            direction_relative_l2_error
        ),
        direction_cosine=direction_cosine,
        direction_relative_energy_error=(
            direction_relative_energy_error
        ),
        residual_seconds=residual_seconds,
        jacobian_seconds=jacobian_seconds,
        gram_matmul_seconds=gram_matmul_seconds,
        gradient_matmul_seconds=gradient_matmul_seconds,
        symmetrize_seconds=symmetrize_seconds,
        gram_symmetry_relative_error=(
            gram_symmetry_relative_error
        ),
        assembly_seconds=assembly_seconds,
        solve_seconds=solve_seconds,
        reference_seconds=reference_seconds,
        line_search_seconds=line_search_seconds,
    )
    return updated_params, stats


ENGD_CONFIG_PT = ENGDConfig(
    line_search_levels=LINE_SEARCH_LEVELS,
    gram_ridge=GRAM_RIDGE,
    solver=ENGD_SOLVER,
    compare_lstsq_reference=COMPARE_LSTSQ_REFERENCE,
    relative_cutoff=RELATIVE_CUTOFF,
    refinement_steps=REFINEMENT_STEPS,
    lstsq_rcond=LSTSQ_RCOND,
    jacobian_backend=JACOBIAN_BACKEND,
    gradient_tolerance=1.0e-14,
    require_loss_decrease=True,
)
ENGD_CONFIG_U = ENGDConfig(
    line_search_levels=LINE_SEARCH_LEVELS,
    gram_ridge=GRAM_RIDGE,
    solver=ENGD_SOLVER,
    compare_lstsq_reference=COMPARE_LSTSQ_REFERENCE,
    relative_cutoff=RELATIVE_CUTOFF,
    refinement_steps=REFINEMENT_STEPS,
    lstsq_rcond=LSTSQ_RCOND,
    jacobian_backend=JACOBIAN_BACKEND,
    gradient_tolerance=1.0e-14,
    require_loss_decrease=True,
)

def append_profile_row(
    subproblem: str,
    fs_iteration: int,
    inner_iteration: int,
    stats: ENGDStepStats,
) -> None:
    """Record subproblem, outer/inner iteration indices, and complete step statistics."""
    row: Dict[str, object] = {
        "subproblem": subproblem,
        "fs_iteration": fs_iteration,
        "inner_iteration": inner_iteration,
    }
    row.update(asdict(stats))
    PROFILE_ROWS.append(row)


# =============================================================================
# 9. Loss breakdown and coupled-equation diagnostics
# =============================================================================

# Individual residual-block losses.
def residual_block_loss(
    values: Array,
    weight: float,
    measure: float,
) -> float:
    n_points = int(values.shape[0])
    if n_points <= 0:
        raise ValueError("A residual block must contain at least one point.")
    return float(
        0.5
        * weight
        * measure
        * jnp.sum(values * values)
        / float(n_points)
    )

# Mechanics loss decomposition.
def mechanics_loss_breakdown(
    flat_u: Array,
    flat_pt: Array,
) -> Dict[str, float]:
    params_u = UNRAVEL_U(flat_u)
    params_pt = UNRAVEL_PT(flat_pt)

    r_pde = V_MECH_RES(params_u, params_pt, PDE_POINTS)
    r_dir = (
        V_PREDICT(params_u, U_DIR_POINTS)
        - U_DIR_TARGET
    )
    r_neu = (
        V_TOTAL_TRACTION_PRED(
            params_u, params_pt, U_NEU_POINTS
        )
        - U_NEU_EXACT_TRACTION
    )
    r_ic = (
        V_PREDICT(params_u, IC_POINTS)
        - U_IC_TARGET
    )

    return {
        "pde": residual_block_loss(
            r_pde, WEIGHTS_U["pde"], MEASURE_PDE
        ),
        "dirichlet": residual_block_loss(
            r_dir, WEIGHTS_U["dirichlet"], MEASURE_U_DIR
        ),
        "neumann": residual_block_loss(
            r_neu, WEIGHTS_U["neumann"], MEASURE_U_NEU
        ),
        "ic": residual_block_loss(
            r_ic, WEIGHTS_U["ic"], MEASURE_IC
        ),
    }

# Split flow-heat loss decomposition.
def flow_heat_loss_breakdown(
    flat_pt: Array,
    flat_u_old: Array,
    flat_pt_old: Array,
) -> Dict[str, float]:
    params_pt = UNRAVEL_PT(flat_pt)
    params_u_old = UNRAVEL_U(flat_u_old)
    params_pt_old = UNRAVEL_PT(flat_pt_old)

    r_pde = V_FLOW_HEAT_RES(
        params_pt,
        params_u_old,
        params_pt_old,
        PDE_POINTS,
    )
    r_dir = (
        V_PREDICT(params_pt, PT_DIR_POINTS)
        - PT_DIR_TARGET
    )
    r_ic = (
        V_PREDICT(params_pt, IC_POINTS)
        - PT_IC_TARGET
    )

    return {
        "pde": residual_block_loss(
            r_pde, WEIGHTS_PT["pde"], MEASURE_PDE
        ),
        "dirichlet": residual_block_loss(
            r_dir, WEIGHTS_PT["dirichlet"], MEASURE_PT_DIR
        ),
        "ic": residual_block_loss(
            r_ic, WEIGHTS_PT["ic"], MEASURE_IC
        ),
    }

# Original coupled flow-heat diagnostics.
def coupled_flow_heat_pde_loss(
    flat_u: Array,
    flat_pt: Array,
) -> float:
    params_u = UNRAVEL_U(flat_u)
    params_pt = UNRAVEL_PT(flat_pt)
    residual = V_COUPLED_FLOW_HEAT_RES(
        params_pt, params_u, PDE_POINTS
    )
    return residual_block_loss(
        residual, WEIGHTS_PT["pde"], MEASURE_PDE
    )


# =============================================================================
# 10. Error metrics
# =============================================================================

# Scalar absolute and relative L2 errors.
def absolute_relative_l2_scalar(
    prediction: Array,
    exact: Array,
) -> Tuple[float, float]:
    error_l2 = math.sqrt(
        MEASURE_PDE * float(jnp.mean((prediction - exact) ** 2))
    )
    exact_l2 = math.sqrt(
        MEASURE_PDE * float(jnp.mean(exact**2))
    )
    return error_l2, error_l2 / max(exact_l2, 1.0e-30)

# Vector absolute and relative L2 errors.
def absolute_relative_l2_vector(
    prediction: Array,
    exact: Array,
) -> Tuple[float, float]:
    error_l2 = math.sqrt(
        MEASURE_PDE
        * float(jnp.mean(jnp.sum((prediction - exact) ** 2, axis=1)))
    )
    exact_l2 = math.sqrt(
        MEASURE_PDE
        * float(jnp.mean(jnp.sum(exact**2, axis=1)))
    )
    return error_l2, error_l2 / max(exact_l2, 1.0e-30)

# Evaluate prediction errors, including L2 and spatial H1 norms.
def evaluate_errors(flat_u: Array, flat_pt: Array) -> Dict[str, float]:
    params_u = UNRAVEL_U(flat_u)
    params_pt = UNRAVEL_PT(flat_pt)

    u_prediction = V_PREDICT(params_u, EVAL_POINTS)
    pt_prediction = V_PREDICT(params_pt, EVAL_POINTS)
    u_reference = V_U_EXACT(EVAL_POINTS)
    pt_reference = V_PT_EXACT(EVAL_POINTS)

    u_l2, u_rel_l2 = absolute_relative_l2_vector(
        u_prediction, u_reference
    )
    u1_l2, u1_rel_l2 = absolute_relative_l2_scalar(
        u_prediction[:, 0], u_reference[:, 0]
    )
    u2_l2, u2_rel_l2 = absolute_relative_l2_scalar(
        u_prediction[:, 1], u_reference[:, 1]
    )
    p_l2, p_rel_l2 = absolute_relative_l2_scalar(
        pt_prediction[:, 0], pt_reference[:, 0]
    )
    t_l2, t_rel_l2 = absolute_relative_l2_scalar(
        pt_prediction[:, 1], pt_reference[:, 1]
    )

    u_jac_prediction = vmap(
        lambda point: U_JAC(params_u, point)
    )(EVAL_POINTS)
    u_jac_reference = V_U_EXACT_JAC(EVAL_POINTS)
    pt_jac_prediction = vmap(
        lambda point: PT_JAC(params_pt, point)
    )(EVAL_POINTS)

    p_jac_reference = vmap(P_EXACT_GRAD)(EVAL_POINTS)
    t_jac_reference = vmap(T_EXACT_GRAD)(EVAL_POINTS)

    # H1 uses spatial derivatives x,y; time derivatives are not part of H1(Omega).
    u_grad_error_sq = MEASURE_PDE * float(
        jnp.mean(
            jnp.sum(
                (
                    u_jac_prediction[:, :, :2]
                    - u_jac_reference[:, :, :2]
                )
                ** 2,
                axis=(1, 2),
            )
        )
    )
    u1_grad_error_sq = MEASURE_PDE * float(
        jnp.mean(
            jnp.sum(
                (
                    u_jac_prediction[:, 0, :2]
                    - u_jac_reference[:, 0, :2]
                )
                ** 2,
                axis=1,
            )
        )
    )
    u2_grad_error_sq = MEASURE_PDE * float(
        jnp.mean(
            jnp.sum(
                (
                    u_jac_prediction[:, 1, :2]
                    - u_jac_reference[:, 1, :2]
                )
                ** 2,
                axis=1,
            )
        )
    )
    p_grad_error_sq = MEASURE_PDE * float(
        jnp.mean(
            jnp.sum(
                (
                    pt_jac_prediction[:, 0, :2]
                    - p_jac_reference[:, :2]
                )
                ** 2,
                axis=1,
            )
        )
    )
    t_grad_error_sq = MEASURE_PDE * float(
        jnp.mean(
            jnp.sum(
                (
                    pt_jac_prediction[:, 1, :2]
                    - t_jac_reference[:, :2]
                )
                ** 2,
                axis=1,
            )
        )
    )

    return {
        "u_l2": u_l2,
        "u_rel_l2": u_rel_l2,
        "u_h1": math.sqrt(u_l2 * u_l2 + u_grad_error_sq),
        "u1_l2": u1_l2,
        "u1_rel_l2": u1_rel_l2,
        "u1_h1": math.sqrt(u1_l2 * u1_l2 + u1_grad_error_sq),
        "u2_l2": u2_l2,
        "u2_rel_l2": u2_rel_l2,
        "u2_h1": math.sqrt(u2_l2 * u2_l2 + u2_grad_error_sq),
        "p_l2": p_l2,
        "p_rel_l2": p_rel_l2,
        "p_h1": math.sqrt(p_l2 * p_l2 + p_grad_error_sq),
        "T_l2": t_l2,
        "T_rel_l2": t_rel_l2,
        "T_h1": math.sqrt(t_l2 * t_l2 + t_grad_error_sq),
    }

# Relative prediction change between consecutive GFS sweeps.
def relative_prediction_change(
    flat_new: Array,
    flat_old: Array,
    unravel: Callable[[Array], Params],
) -> float:
    prediction_new = V_PREDICT(unravel(flat_new), EVAL_POINTS)
    prediction_old = V_PREDICT(unravel(flat_old), EVAL_POINTS)
    numerator = float(jnp.linalg.norm(prediction_new - prediction_old))
    denominator = float(jnp.linalg.norm(prediction_old))
    return numerator / max(denominator, 1.0e-30)


# =============================================================================
# 11. History and training loop
# =============================================================================

# Outer-iteration history.
class History:
    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self.fs_iteration: List[int] = []
        self.loss_pt: List[float] = []
        self.loss_pt_coupled_pde: List[float] = []
        self.loss_u: List[float] = []
        self.u_rel_l2: List[float] = []
        self.u1_rel_l2: List[float] = []
        self.u2_rel_l2: List[float] = []
        self.p_rel_l2: List[float] = []
        self.T_rel_l2: List[float] = []
        self.u_mean_rel_l2: List[float] = []
        self.pT_mean_rel_l2: List[float] = []
        self.common_rel_l2_max: List[float] = []
        self.u_h1: List[float] = []
        self.u1_h1: List[float] = []
        self.u2_h1: List[float] = []
        self.p_h1: List[float] = []
        self.T_h1: List[float] = []
        self.u_outer_change: List[float] = []
        self.pt_outer_change: List[float] = []
        self.stop_condition: List[bool] = []
        self.stop_counter: List[int] = []
        self.training_time: List[float] = []


HISTORY = History()


@dataclass
class TrainingTiming:
    """Auditable timing ledger for one complete GFS-ENGD training run."""

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
        # The convergence check determines when training terminates and is
        # therefore part of the effective solver cost.
        return self.optimizer_seconds + self.convergence_check_seconds

# Compile the production paths before timed training.
def warmup_training_kernels(flat_u: Array, flat_pt: Array) -> float:
    """
    Compile the exact production optimizer and convergence-check paths.

    Returned updates are discarded, so warmup does not change initialization.
    Test-grid error evaluation is intentionally not warmed because it is
    excluded from effective training time.
    """

    start = time.perf_counter()
    flat_u_old = jnp.array(flat_u)
    flat_pt_old = jnp.array(flat_pt)

    residual_pt_fn = lambda z: residual_pt_flat(
        z, flat_u_old, flat_pt_old
    )
    jacobian_pt_fn = lambda z: JACOBIAN_PT_FLAT(
        z, flat_u_old, flat_pt_old
    )
    candidate_pt_fn = lambda candidates: candidate_losses_pt(
        candidates, flat_u_old, flat_pt_old
    )
    gram_pt_fn = None
    if JACOBIAN_BACKEND == "pointwise":
        context_pt = build_pointwise_pt_context(
            flat_u_old, flat_pt_old, POINT_BLOCK_PT
        )
        pde_frozen_pt = context_pt.pde_frozen.reshape(-1, 2)[
            : PDE_POINTS.shape[0]
        ]
        if CACHE_LINE_SEARCH:
            candidate_pt_fn = lambda candidates: candidate_losses_pt_cached(
                candidates, pde_frozen_pt
            )
        gram_pt_fn = lambda z: pointwise_pt_gram_gradient(
            z, context_pt
        )

    warm_pt, _ = explicit_engd_step(
        flat_pt,
        residual_pt_fn,
        jacobian_pt_fn,
        candidate_pt_fn,
        ENGD_CONFIG_PT,
        gram_gradient_fn=gram_pt_fn,
    )
    warm_pt.block_until_ready()

    # Compile mechanics using the original p/T state: shapes and code paths are
    # identical to those used after the first p/T inner solve.
    residual_u_fn = lambda z: residual_u_flat(z, flat_pt)
    jacobian_u_fn = lambda z: JACOBIAN_U_FLAT(z, flat_pt)
    candidate_u_fn = lambda candidates: candidate_losses_u(
        candidates, flat_pt
    )
    gram_u_fn = None
    if JACOBIAN_BACKEND == "pointwise":
        context_u = build_pointwise_u_context(
            flat_pt, POINT_BLOCK_U
        )
        pde_frozen_u = context_u.pde_frozen.reshape(-1, 2)[
            : PDE_POINTS.shape[0]
        ]
        neu_frozen_u = context_u.neu_frozen.reshape(-1, 2)[
            : U_NEU_POINTS.shape[0]
        ]
        if CACHE_LINE_SEARCH:
            candidate_u_fn = lambda candidates: candidate_losses_u_cached(
                candidates, pde_frozen_u, neu_frozen_u
            )
        gram_u_fn = lambda z: pointwise_u_gram_gradient(
            z, context_u
        )

    warm_u, _ = explicit_engd_step(
        flat_u,
        residual_u_fn,
        jacobian_u_fn,
        candidate_u_fn,
        ENGD_CONFIG_U,
        gram_gradient_fn=gram_u_fn,
    )
    warm_u.block_until_ready()

    # Warm only diagnostics that affect the adaptive stopping decision.
    flow_heat_loss_breakdown(flat_pt, flat_u_old, flat_pt_old)
    mechanics_loss_breakdown(flat_u, flat_pt)
    coupled_flow_heat_pde_loss(flat_u, flat_pt)
    relative_prediction_change(flat_u, flat_u_old, UNRAVEL_U)
    relative_prediction_change(flat_pt, flat_pt_old, UNRAVEL_PT)

    return time.perf_counter() - start


def train() -> Tuple[Array, Array, TrainingTiming]:
    HISTORY.clear()
    PROFILE_ROWS.clear()

    flat_u = FLAT_U
    flat_pt = FLAT_PT
    flat_u_old = jnp.array(flat_u)
    flat_pt_old = jnp.array(flat_pt)
    timing = TrainingTiming()

    LOGGER.info("============================================================")
    LOGGER.info(
        "Three-field compressible THP: generalized FS + exact streamed ENGD"
    )
    device = jax.devices()[0]
    LOGGER.info(
        "runtime: jax=%s backend=%s device=%s x64=%s preallocate=%s "
        "memory_fraction=%s",
        jax.__version__,
        jax.default_backend(),
        device.device_kind,
        jax.config.jax_enable_x64,
        os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "<jax-default>"),
        os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", "<jax-default>"),
    )
    if device.platform != "gpu":
        LOGGER.warning(
            "JAX is running on %s, not GPU; this run is unsuitable for "
            "paper timing or GPU-utilization measurements.",
            device.platform,
        )
    LOGGER.info(
        "E=%s, nu=%s, lambda=%.6e, mu=%.6e",
        E,
        NU,
        LAMBDA,
        MU,
    )
    LOGGER.info(
        "alpha=%s, beta=%s, a0=%s, b0=%s, c0=%s, K=%s, Theta=%s",
        ALPHA,
        BETA,
        A0,
        B0,
        C0,
        KAPPA,
        THETA,
    )
    LOGGER.info(
        "K_drained=%.6e, gamma_FS=%.6e",
        K_DRAINED,
        GAMMA_FS,
    )
    LOGGER.info(
        "S_FS=[[%.6e, %.6e], [%.6e, %.6e]]",
        S11,
        S12,
        S21,
        S22,
    )
    LOGGER.info(
        "fixed uniform points: PDE=%d, IC=%d, u-Dir=%d, "
        "u-Neu=%d, pT-Dir=%d, eval=%d",
        PDE_POINTS.shape[0],
        IC_POINTS.shape[0],
        U_DIR_POINTS.shape[0],
        U_NEU_POINTS.shape[0],
        PT_DIR_POINTS.shape[0],
        EVAL_POINTS.shape[0],
    )
    LOGGER.info("exact_case=%s seed=%d", EXACT_CASE, SEED)
    LOGGER.info("WEIGHTS_U=%s", WEIGHTS_U)
    LOGGER.info("WEIGHTS_PT=%s", WEIGHTS_PT)
    LOGGER.info(
        "parameters: u=%d, pT=%d", flat_u.size, flat_pt.size
    )
    LOGGER.info(
        "GFS iterations=%d, ENGD steps pT=%d, ENGD steps u=%d",
        FS_ITERATIONS,
        ENGD_STEPS_PT,
        ENGD_STEPS_U,
    )
    LOGGER.info(
        "ENGD solver=%s, refinement=%d, relative_cutoff=%.3e, "
        "automatic FP64 cutoff=enabled, Gram ridge=%.3e, "
        "compare_lstsq=%s",
        ENGD_SOLVER,
        REFINEMENT_STEPS,
        RELATIVE_CUTOFF,
        GRAM_RIDGE,
        COMPARE_LSTSQ_REFERENCE,
    )
    LOGGER.info(
        "Jacobian backend=%s; point blocks pT=%d u=%d; fused blocks=%s; "
        "microbenchmark=%s",
        JACOBIAN_BACKEND,
        POINT_BLOCK_PT,
        POINT_BLOCK_U,
        FUSE_GRAM_BLOCKS,
        JACOBIAN_MICROBENCH,
    )
    LOGGER.info(
        "vectorized grid search: %d positive levels + implicit rejection; "
        "cached frozen terms=%s",
        LINE_SEARCH_LEVELS,
        JACOBIAN_BACKEND == "pointwise" and CACHE_LINE_SEARCH,
    )
    LOGGER.info(
        "adaptive GFS stop: enabled=%s min_iter=%d patience=%d "
        "outer<=%.3e coupled-pT<=%.3e u-loss<=%.3e",
        ENABLE_FS_EARLY_STOP,
        FS_MIN_ITERATIONS,
        FS_EARLY_STOP_PATIENCE,
        FS_EARLY_STOP_OUTER_TOL,
        FS_EARLY_STOP_COUPLED_PT_TOL,
        FS_EARLY_STOP_U_LOSS_TOL,
    )
    LOGGER.info(
        "TTA thresholds=%s patience=%d completed sweeps; "
        "fixed-budget checkpoints=%s",
        TIME_TO_ACCURACY_THRESHOLDS,
        TTA_PATIENCE,
        FIXED_BUDGET_CHECKPOINTS,
    )

    timing.jit_warmup_seconds_excluded = warmup_training_kernels(
        flat_u, flat_pt
    )
    LOGGER.info(
        "JIT warmup excluded from training time: %.3f s",
        timing.jit_warmup_seconds_excluded,
    )
    LOGGER.info("============================================================")

    solver_wall_start = time.perf_counter()
    fs_stop_counter = 0
    fs_stop_reason = "maximum_iterations"

    for fs_iteration in range(1, FS_ITERATIONS + 1):
        LOGGER.info(
            "--- generalized FS iteration %d/%d ---",
            fs_iteration,
            FS_ITERATIONS,
        )

        # ------------------------------------------------------------------
        # Step 1: joint flow-heat subproblem by exact ENGD
        # ------------------------------------------------------------------
        residual_pt_fn = lambda z: residual_pt_flat(
            z, flat_u_old, flat_pt_old
        )
        jacobian_pt_fn = lambda z: JACOBIAN_PT_FLAT(
            z, flat_u_old, flat_pt_old
        )
        candidate_pt_fn = lambda candidates: candidate_losses_pt(
            candidates, flat_u_old, flat_pt_old
        )
        gram_pt_fn = None
        if JACOBIAN_BACKEND == "pointwise":
            context_start = time.perf_counter()
            pt_pointwise_context = build_pointwise_pt_context(
                flat_u_old, flat_pt_old, POINT_BLOCK_PT
            )
            timing.context_seconds += (
                time.perf_counter() - context_start
            )
            pde_frozen_pt = pt_pointwise_context.pde_frozen.reshape(-1, 2)[
                : PDE_POINTS.shape[0]
            ]
            if CACHE_LINE_SEARCH:
                candidate_pt_fn = (
                    lambda candidates: candidate_losses_pt_cached(
                        candidates, pde_frozen_pt
                    )
                )
            gram_pt_fn = lambda z: pointwise_pt_gram_gradient(
                z, pt_pointwise_context
            )

        for engd_iteration in range(1, ENGD_STEPS_PT + 1):
            flat_pt, stats = explicit_engd_step(
                flat_pt,
                residual_pt_fn,
                jacobian_pt_fn,
                candidate_pt_fn,
                ENGD_CONFIG_PT,
                gram_gradient_fn=gram_pt_fn,
            )
            flat_pt.block_until_ready()

            timing.assembly_seconds += stats.assembly_seconds
            timing.solve_seconds += stats.solve_seconds
            timing.line_search_seconds += stats.line_search_seconds
            timing.reference_validation_seconds_excluded += (
                stats.reference_seconds
            )

            append_profile_row(
                "pT", fs_iteration, engd_iteration, stats
            )
            LOGGER.info(
                "pT ENGD %02d: accepted=%s loss %.4e -> %.4e "
                "eta=%.3e |g|=%.3e |d|=%.3e "
                "rank=%d cutoff=%.3e cond=%.3e linres=%.3e "
                "neg=%d times[A=%.3f S=%.3f LS=%.3f]",
                engd_iteration,
                stats.accepted,
                stats.loss_before,
                stats.loss_after,
                stats.step_size,
                stats.gradient_norm,
                stats.natural_gradient_norm,
                stats.effective_rank,
                stats.spectral_cutoff,
                stats.effective_condition,
                stats.solve_relative_residual,
                stats.negative_count,
                stats.assembly_seconds,
                stats.solve_seconds,
                stats.line_search_seconds,
            )
            if COMPARE_LSTSQ_REFERENCE:
                LOGGER.info(
                    "pT same-Gram LSTSQ check: rank=%d cond=%.3e "
                    "linres=%.3e refined-vs-lstsq "
                    "relL2=%.3e cos=%.12f relE=%.3e",
                    stats.lstsq_rank,
                    stats.lstsq_effective_condition,
                    stats.lstsq_solve_relative_residual,
                    stats.direction_relative_l2_error,
                    stats.direction_cosine,
                    stats.direction_relative_energy_error,
                )

            if stats.step_size == 0.0:
                LOGGER.info(
                    "pT ENGD stopped early: no decreasing grid-search step."
                )
                break

        # ------------------------------------------------------------------
        # Step 2: mechanics subproblem by exact ENGD
        # ------------------------------------------------------------------
        residual_u_fn = lambda z: residual_u_flat(z, flat_pt)
        jacobian_u_fn = lambda z: JACOBIAN_U_FLAT(z, flat_pt)
        candidate_u_fn = lambda candidates: candidate_losses_u(
            candidates, flat_pt
        )
        gram_u_fn = None
        if JACOBIAN_BACKEND == "pointwise":
            context_start = time.perf_counter()
            u_pointwise_context = build_pointwise_u_context(
                flat_pt, POINT_BLOCK_U
            )
            timing.context_seconds += (
                time.perf_counter() - context_start
            )
            pde_frozen_u = u_pointwise_context.pde_frozen.reshape(-1, 2)[
                : PDE_POINTS.shape[0]
            ]
            neu_frozen_u = u_pointwise_context.neu_frozen.reshape(-1, 2)[
                : U_NEU_POINTS.shape[0]
            ]
            if CACHE_LINE_SEARCH:
                candidate_u_fn = (
                    lambda candidates: candidate_losses_u_cached(
                        candidates, pde_frozen_u, neu_frozen_u
                    )
                )
            gram_u_fn = lambda z: pointwise_u_gram_gradient(
                z, u_pointwise_context
            )

        for engd_iteration in range(1, ENGD_STEPS_U + 1):
            flat_u, stats = explicit_engd_step(
                flat_u,
                residual_u_fn,
                jacobian_u_fn,
                candidate_u_fn,
                ENGD_CONFIG_U,
                gram_gradient_fn=gram_u_fn,
            )
            flat_u.block_until_ready()

            timing.assembly_seconds += stats.assembly_seconds
            timing.solve_seconds += stats.solve_seconds
            timing.line_search_seconds += stats.line_search_seconds
            timing.reference_validation_seconds_excluded += (
                stats.reference_seconds
            )

            append_profile_row(
                "u", fs_iteration, engd_iteration, stats
            )
            LOGGER.info(
                "u  ENGD %02d: accepted=%s loss %.4e -> %.4e "
                "eta=%.3e |g|=%.3e |d|=%.3e "
                "rank=%d cutoff=%.3e cond=%.3e linres=%.3e "
                "neg=%d times[A=%.3f S=%.3f LS=%.3f]",
                engd_iteration,
                stats.accepted,
                stats.loss_before,
                stats.loss_after,
                stats.step_size,
                stats.gradient_norm,
                stats.natural_gradient_norm,
                stats.effective_rank,
                stats.spectral_cutoff,
                stats.effective_condition,
                stats.solve_relative_residual,
                stats.negative_count,
                stats.assembly_seconds,
                stats.solve_seconds,
                stats.line_search_seconds,
            )
            if COMPARE_LSTSQ_REFERENCE:
                LOGGER.info(
                    "u same-Gram LSTSQ check: rank=%d cond=%.3e "
                    "linres=%.3e refined-vs-lstsq "
                    "relL2=%.3e cos=%.12f relE=%.3e",
                    stats.lstsq_rank,
                    stats.lstsq_effective_condition,
                    stats.lstsq_solve_relative_residual,
                    stats.direction_relative_l2_error,
                    stats.direction_cosine,
                    stats.direction_relative_energy_error,
                )

            if stats.step_size == 0.0:
                LOGGER.info(
                    "u ENGD stopped early: no decreasing grid-search step."
                )
                break

        # ------------------------------------------------------------------
        # Convergence diagnostics: included because they control termination.
        # ------------------------------------------------------------------
        convergence_start = time.perf_counter()
        pt_breakdown = flow_heat_loss_breakdown(
            flat_pt, flat_u_old, flat_pt_old
        )
        u_breakdown = mechanics_loss_breakdown(flat_u, flat_pt)
        loss_pt_coupled_pde = coupled_flow_heat_pde_loss(
            flat_u, flat_pt
        )
        u_outer_change = relative_prediction_change(
            flat_u, flat_u_old, UNRAVEL_U
        )
        pt_outer_change = relative_prediction_change(
            flat_pt, flat_pt_old, UNRAVEL_PT
        )
        timing.convergence_check_seconds += (
            time.perf_counter() - convergence_start
        )

        # Test-grid accuracy is reporting only and excluded from training time.
        evaluation_start = time.perf_counter()
        errors = evaluate_errors(flat_u, flat_pt)
        timing.evaluation_seconds_excluded += (
            time.perf_counter() - evaluation_start
        )

        loss_pt_total = sum(pt_breakdown.values())
        loss_u_total = sum(u_breakdown.values())
        u_mean_rel_l2 = 0.5 * (
            errors["u1_rel_l2"] + errors["u2_rel_l2"]
        )
        pt_mean_rel_l2 = 0.5 * (
            errors["p_rel_l2"] + errors["T_rel_l2"]
        )
        common_rel_l2_max = max(
            errors["u1_rel_l2"],
            errors["u2_rel_l2"],
            errors["p_rel_l2"],
            errors["T_rel_l2"],
        )

        LOGGER.info(
            "FS %02d GFS pT loss=%.4e [PDE=%.4e Dir=%.4e IC=%.4e]",
            fs_iteration,
            loss_pt_total,
            pt_breakdown["pde"],
            pt_breakdown["dirichlet"],
            pt_breakdown["ic"],
        )
        LOGGER.info(
            "FS %02d original coupled pT PDE loss=%.4e",
            fs_iteration,
            loss_pt_coupled_pde,
        )
        LOGGER.info(
            "FS %02d u loss=%.4e [PDE=%.4e Dir=%.4e Neu=%.4e IC=%.4e]",
            fs_iteration,
            loss_u_total,
            u_breakdown["pde"],
            u_breakdown["dirichlet"],
            u_breakdown["neumann"],
            u_breakdown["ic"],
        )
        LOGGER.info(
            "FS %02d errors: u RelL2=%.4e H1=%.4e | "
            "p RelL2=%.4e H1=%.4e | T RelL2=%.4e H1=%.4e",
            fs_iteration,
            errors["u_rel_l2"],
            errors["u_h1"],
            errors["p_rel_l2"],
            errors["p_h1"],
            errors["T_rel_l2"],
            errors["T_h1"],
        )
        LOGGER.info(
            "FS %02d displacement components: "
            "u1 L2=%.4e RelL2=%.4e H1=%.4e | "
            "u2 L2=%.4e RelL2=%.4e H1=%.4e",
            fs_iteration,
            errors["u1_l2"],
            errors["u1_rel_l2"],
            errors["u1_h1"],
            errors["u2_l2"],
            errors["u2_rel_l2"],
            errors["u2_h1"],
        )
        LOGGER.info(
            "FS %02d comparison metrics: u-mean=%.4e pT-mean=%.4e "
            "E_common=max(u1,u2,p,T)=%.4e",
            fs_iteration,
            u_mean_rel_l2,
            pt_mean_rel_l2,
            common_rel_l2_max,
        )
        LOGGER.info(
            "FS %02d outer changes: u=%.4e, pT=%.4e",
            fs_iteration,
            u_outer_change,
            pt_outer_change,
        )
        LOGGER.info(
            "effective training time=%.2f s "
            "[context=%.2f assembly=%.2f solve=%.2f "
            "line-search=%.2f convergence=%.2f]; "
            "evaluation excluded=%.2f s",
            timing.effective_training_seconds,
            timing.context_seconds,
            timing.assembly_seconds,
            timing.solve_seconds,
            timing.line_search_seconds,
            timing.convergence_check_seconds,
            timing.evaluation_seconds_excluded,
        )

        outer_change_max = max(u_outer_change, pt_outer_change)
        fs_stop_condition = (
            fs_iteration >= FS_MIN_ITERATIONS
            and outer_change_max <= FS_EARLY_STOP_OUTER_TOL
            and loss_pt_coupled_pde
            <= FS_EARLY_STOP_COUPLED_PT_TOL
            and loss_u_total <= FS_EARLY_STOP_U_LOSS_TOL
        )
        if ENABLE_FS_EARLY_STOP and fs_stop_condition:
            fs_stop_counter += 1
        else:
            fs_stop_counter = 0

        LOGGER.info(
            "FS %02d adaptive-stop: outer=%.4e/%.4e "
            "coupled-pT=%.4e/%.4e u-loss=%.4e/%.4e "
            "eligible=%s patience=%d/%d",
            fs_iteration,
            outer_change_max,
            FS_EARLY_STOP_OUTER_TOL,
            loss_pt_coupled_pde,
            FS_EARLY_STOP_COUPLED_PT_TOL,
            loss_u_total,
            FS_EARLY_STOP_U_LOSS_TOL,
            fs_stop_condition,
            fs_stop_counter,
            FS_EARLY_STOP_PATIENCE,
        )

        HISTORY.fs_iteration.append(fs_iteration)
        HISTORY.loss_pt.append(loss_pt_total)
        HISTORY.loss_pt_coupled_pde.append(loss_pt_coupled_pde)
        HISTORY.loss_u.append(loss_u_total)
        HISTORY.u_rel_l2.append(errors["u_rel_l2"])
        HISTORY.u1_rel_l2.append(errors["u1_rel_l2"])
        HISTORY.u2_rel_l2.append(errors["u2_rel_l2"])
        HISTORY.p_rel_l2.append(errors["p_rel_l2"])
        HISTORY.T_rel_l2.append(errors["T_rel_l2"])
        HISTORY.u_mean_rel_l2.append(u_mean_rel_l2)
        HISTORY.pT_mean_rel_l2.append(pt_mean_rel_l2)
        HISTORY.common_rel_l2_max.append(common_rel_l2_max)
        HISTORY.u_h1.append(errors["u_h1"])
        HISTORY.u1_h1.append(errors["u1_h1"])
        HISTORY.u2_h1.append(errors["u2_h1"])
        HISTORY.p_h1.append(errors["p_h1"])
        HISTORY.T_h1.append(errors["T_h1"])
        HISTORY.u_outer_change.append(u_outer_change)
        HISTORY.pt_outer_change.append(pt_outer_change)
        HISTORY.stop_condition.append(fs_stop_condition)
        HISTORY.stop_counter.append(fs_stop_counter)
        HISTORY.training_time.append(
            timing.effective_training_seconds
        )

        flat_u_old = jnp.array(flat_u)
        flat_pt_old = jnp.array(flat_pt)

        if (
            ENABLE_FS_EARLY_STOP
            and fs_stop_counter >= FS_EARLY_STOP_PATIENCE
        ):
            fs_stop_reason = "adaptive_convergence"
            LOGGER.info(
                "GFS stopped early at FS %d: all convergence "
                "conditions held for %d consecutive iterations.",
                fs_iteration,
                FS_EARLY_STOP_PATIENCE,
            )
            break

    timing.total_solver_wall_seconds = (
        time.perf_counter() - solver_wall_start
    )

    LOGGER.info(
        "GFS termination: reason=%s iterations=%d/%d",
        fs_stop_reason,
        len(HISTORY.fs_iteration),
        FS_ITERATIONS,
    )
    LOGGER.info(
        "Training complete: effective=%.3f s, optimizer=%.3f s, "
        "convergence=%.3f s, evaluation(excluded)=%.3f s, "
        "JIT warmup(excluded)=%.3f s, total solver wall=%.3f s",
        timing.effective_training_seconds,
        timing.optimizer_seconds,
        timing.convergence_check_seconds,
        timing.evaluation_seconds_excluded,
        timing.jit_warmup_seconds_excluded,
        timing.total_solver_wall_seconds,
    )

    for subproblem in ("pT", "u"):
        rows = [
            row
            for row in PROFILE_ROWS
            if row["subproblem"] == subproblem
        ]
        if not rows:
            continue
        LOGGER.info(
            "%s ENGD profile over %d updates: "
            "assembly=%.3f s solve=%.3f s line-search=%.3f s "
            "reference(excluded)=%.3f s mean-rank=%.1f",
            subproblem,
            len(rows),
            float(np.sum([row["assembly_seconds"] for row in rows])),
            float(np.sum([row["solve_seconds"] for row in rows])),
            float(np.sum([row["line_search_seconds"] for row in rows])),
            float(np.sum([row["reference_seconds"] for row in rows])),
            float(np.mean([row["effective_rank"] for row in rows])),
        )

    return flat_u, flat_pt, timing

# =============================================================================
# 12. Plotting and saving
# =============================================================================


def save_figure(fig: plt.Figure, stem: str) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(
            RESULT_DIR / f"{stem}.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )


def plot_final_fields(
    flat_u: Array,
    flat_pt: Array,
    t_value: float,
    n_plot: int = 100,
) -> None:
    params_u = UNRAVEL_U(flat_u)
    params_pt = UNRAVEL_PT(flat_pt)

    x = np.linspace(0.0, 1.0, n_plot)
    y = np.linspace(0.0, 1.0, n_plot)
    x_grid, y_grid = np.meshgrid(x, y)
    t_grid = np.full_like(x_grid, t_value)
    points = jnp.stack(
        [x_grid.ravel(), y_grid.ravel(), t_grid.ravel()], axis=1
    )

    u_prediction = np.asarray(V_PREDICT(params_u, points))
    pt_prediction = np.asarray(V_PREDICT(params_pt, points))
    u_reference = np.asarray(V_U_EXACT(points))
    pt_reference = np.asarray(V_PT_EXACT(points))

    predicted = [
        u_prediction[:, 0],
        u_prediction[:, 1],
        pt_prediction[:, 0],
        pt_prediction[:, 1],
    ]
    exact = [
        u_reference[:, 0],
        u_reference[:, 1],
        pt_reference[:, 0],
        pt_reference[:, 1],
    ]
    names = [r"$u_1$", r"$u_2$", r"$p$", r"$T$"]

    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    for column, (pred, ref, name) in enumerate(zip(predicted, exact, names)):
        fields = [pred, ref, np.abs(pred - ref)]
        row_names = ["prediction", "exact", "absolute error"]
        for row, (field, row_name) in enumerate(zip(fields, row_names)):
            image = axes[row, column].imshow(
                field.reshape(n_plot, n_plot),
                origin="lower",
                extent=(0.0, 1.0, 0.0, 1.0),
                aspect="auto",
            )
            axes[row, column].set_title(
                f"{name}: {row_name}, t={t_value:g}"
            )
            axes[row, column].set_xlabel("x")
            axes[row, column].set_ylabel("y")
            fig.colorbar(image, ax=axes[row, column])

    fig.tight_layout()
    save_figure(fig, "final_fields")
    plt.close(fig)


def plot_history() -> None:
    if not HISTORY.fs_iteration:
        return

    iterations = np.asarray(HISTORY.fs_iteration)

    fig_error, ax_error = plt.subplots(figsize=(9, 6))
    ax_error.semilogy(
        iterations, HISTORY.u_rel_l2, marker="o", label=r"$u$ Rel. $L^2$"
    )
    ax_error.semilogy(
        iterations, HISTORY.p_rel_l2, marker="s", label=r"$p$ Rel. $L^2$"
    )
    ax_error.semilogy(
        iterations, HISTORY.T_rel_l2, marker="^", label=r"$T$ Rel. $L^2$"
    )
    ax_error.set_xlabel("Generalized fixed-stress iteration")
    ax_error.set_ylabel("Relative error")
    ax_error.grid(True, which="both", alpha=0.3)
    ax_error.legend()
    save_figure(fig_error, "history_relative_l2")
    plt.close(fig_error)

    fig_components, ax_components = plt.subplots(figsize=(9, 6))
    for values, label, marker in (
        (HISTORY.u1_rel_l2, r"$u_1$", "o"),
        (HISTORY.u2_rel_l2, r"$u_2$", "s"),
        (HISTORY.p_rel_l2, r"$p$", "^"),
        (HISTORY.T_rel_l2, r"$T$", "D"),
        (HISTORY.common_rel_l2_max, r"$E_{common}^{max}$", "x"),
    ):
        ax_components.semilogy(iterations, values, marker=marker, label=label)
    ax_components.set_xlabel("Generalized fixed-stress iteration")
    ax_components.set_ylabel(r"Relative $L^2$ error")
    ax_components.grid(True, which="both", alpha=0.3)
    ax_components.legend()
    save_figure(fig_components, "history_component_relative_l2")
    plt.close(fig_components)

    fig_time, ax_time = plt.subplots(figsize=(9, 6))
    ax_time.semilogy(
        HISTORY.training_time,
        HISTORY.common_rel_l2_max,
        marker="o",
        label=r"$E_{common}^{max}$",
    )
    ax_time.set_xlabel("Effective training time (s)")
    ax_time.set_ylabel(r"Relative $L^2$ error")
    ax_time.grid(True, which="both", alpha=0.3)
    ax_time.legend()
    save_figure(fig_time, "history_error_vs_effective_time")
    plt.close(fig_time)

    fig_loss, ax_loss = plt.subplots(figsize=(9, 6))
    ax_loss.semilogy(
        iterations, HISTORY.loss_u, marker="o", label="mechanics loss"
    )
    ax_loss.semilogy(
        iterations, HISTORY.loss_pt, marker="s", label="GFS flow-heat loss"
    )
    ax_loss.semilogy(
        iterations,
        HISTORY.loss_pt_coupled_pde,
        marker="^",
        label="original coupled pT PDE loss",
    )
    ax_loss.set_xlabel("Generalized fixed-stress iteration")
    ax_loss.set_ylabel("Weighted loss")
    ax_loss.grid(True, which="both", alpha=0.3)
    ax_loss.legend()
    save_figure(fig_loss, "history_loss")
    plt.close(fig_loss)

    fig_change, ax_change = plt.subplots(figsize=(9, 6))
    ax_change.semilogy(
        iterations, HISTORY.u_outer_change, marker="o", label="u outer change"
    )
    ax_change.semilogy(
        iterations,
        HISTORY.pt_outer_change,
        marker="s",
        label="pT outer change",
    )
    ax_change.set_xlabel("Generalized fixed-stress iteration")
    ax_change.set_ylabel("Relative prediction change")
    ax_change.grid(True, which="both", alpha=0.3)
    ax_change.legend()
    save_figure(fig_change, "history_outer_change")
    plt.close(fig_change)


def save_checkpoint(flat_u: Array, flat_pt: Array) -> None:
    np.savez(
        RESULT_DIR / "final_model.npz",
        flat_u=np.asarray(flat_u),
        flat_pt=np.asarray(flat_pt),
        net_u_sizes=np.asarray(NET_U_SIZES),
        net_pt_sizes=np.asarray(NET_PT_SIZES),
    )


def save_history() -> None:
    history_columns = {
        name: np.asarray(values)
        for name, values in vars(HISTORY).items()
    }
    np.savez(RESULT_DIR / "training_history.npz", **history_columns)

    if HISTORY.fs_iteration:
        history_rows = [
            {
                name: (
                    value.item()
                    if isinstance(value, np.generic)
                    else value
                )
                for name, value in zip(history_columns, row)
            }
            for row in zip(*(history_columns[name] for name in history_columns))
        ]
        with open(
            RESULT_DIR / "training_history.csv",
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(file, fieldnames=list(history_columns))
            writer.writeheader()
            writer.writerows(history_rows)
        with open(
            RESULT_DIR / "training_history.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(history_rows, file, indent=2, ensure_ascii=False)



def _comparison_run_identity() -> Dict[str, object]:
    """Identifiers repeated in fixed-budget and TTA tables."""
    return {
        "run_name": RESULT_DIR.name,
        "method": "THP-three-field-GFS-JS-ENGD",
        "exact_case": EXACT_CASE,
        "seed": SEED,
        "E": E,
        "nu": NU,
        "lambda": LAMBDA,
        "mu": MU,
        "alpha": ALPHA,
        "beta": BETA,
        "a0": A0,
        "b0": B0,
        "c0": C0,
        "K": KAPPA,
        "Theta": THETA,
        "gram_ridge": GRAM_RIDGE,
        "weight_u_pde": float(WEIGHTS_U["pde"]),
        "weight_u_dirichlet": float(WEIGHTS_U["dirichlet"]),
        "weight_u_neumann": float(WEIGHTS_U["neumann"]),
        "weight_u_ic": float(WEIGHTS_U["ic"]),
        "weight_pt_pde": float(WEIGHTS_PT["pde"]),
        "weight_pt_dirichlet": float(WEIGHTS_PT["dirichlet"]),
        "weight_pt_ic": float(WEIGHTS_PT["ic"]),
        "engd_steps_pt_per_fs": ENGD_STEPS_PT,
        "engd_steps_u_per_fs": ENGD_STEPS_U,
    }


def _comparison_values_at(index: int) -> Dict[str, object]:
    """Accuracy, loss, update count, and time at one completed GFS sweep."""
    fs_iteration = int(HISTORY.fs_iteration[index])
    pt_updates_completed = sum(
        1
        for row in PROFILE_ROWS
        if row["subproblem"] == "pT"
        and int(row["fs_iteration"]) <= fs_iteration
    )
    u_updates_completed = sum(
        1
        for row in PROFILE_ROWS
        if row["subproblem"] == "u"
        and int(row["fs_iteration"]) <= fs_iteration
    )
    return {
        "fs_iteration": fs_iteration,
        "pT_engd_updates_completed": pt_updates_completed,
        "u_engd_updates_completed": u_updates_completed,
        "total_engd_updates_completed": (
            pt_updates_completed + u_updates_completed
        ),
        "effective_training_seconds": float(HISTORY.training_time[index]),
        "loss_pt": float(HISTORY.loss_pt[index]),
        "loss_pt_coupled_pde": float(HISTORY.loss_pt_coupled_pde[index]),
        "loss_u": float(HISTORY.loss_u[index]),
        "u_rel_l2": float(HISTORY.u_rel_l2[index]),
        "u1_rel_l2": float(HISTORY.u1_rel_l2[index]),
        "u2_rel_l2": float(HISTORY.u2_rel_l2[index]),
        "p_rel_l2": float(HISTORY.p_rel_l2[index]),
        "T_rel_l2": float(HISTORY.T_rel_l2[index]),
        "u_mean_rel_l2": float(HISTORY.u_mean_rel_l2[index]),
        "pT_mean_rel_l2": float(HISTORY.pT_mean_rel_l2[index]),
        "E_common_max": float(HISTORY.common_rel_l2_max[index]),
        "u_h1": float(HISTORY.u_h1[index]),
        "u1_h1": float(HISTORY.u1_h1[index]),
        "u2_h1": float(HISTORY.u2_h1[index]),
        "p_h1": float(HISTORY.p_h1[index]),
        "T_h1": float(HISTORY.T_h1[index]),
    }


def build_fixed_budget_accuracy_rows() -> List[Dict[str, object]]:
    """Build equal-outer-budget records at predefined checkpoints."""
    completed_iterations = len(HISTORY.fs_iteration)
    index_by_iteration = {
        int(iteration): index
        for index, iteration in enumerate(HISTORY.fs_iteration)
    }
    value_names = list(_comparison_values_at(0)) if HISTORY.fs_iteration else [
        "fs_iteration",
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
        "p_rel_l2",
        "T_rel_l2",
        "u_mean_rel_l2",
        "pT_mean_rel_l2",
        "E_common_max",
        "u_h1",
        "u1_h1",
        "u2_h1",
        "p_h1",
        "T_h1",
    ]
    unavailable_values = {name: None for name in value_names}
    rows: List[Dict[str, object]] = []
    for checkpoint in FIXED_BUDGET_CHECKPOINTS:
        index = index_by_iteration.get(checkpoint)
        row: Dict[str, object] = _comparison_run_identity()
        row.update(
            {
                "requested_fs_iteration": int(checkpoint),
                "checkpoint_reached": index is not None,
                "completed_fs_iterations": completed_iterations,
            }
        )
        row.update(
            _comparison_values_at(index)
            if index is not None
            else unavailable_values
        )
        rows.append(row)
    return rows


def build_time_to_accuracy_rows() -> List[Dict[str, object]]:
    """Build patience-confirmed TTA records from completed outer sweeps."""
    if not HISTORY.fs_iteration:
        return []

    metric_histories = (
        ("u1_rel_l2", "u1_rel_l2", HISTORY.u1_rel_l2),
        ("u2_rel_l2", "u2_rel_l2", HISTORY.u2_rel_l2),
        ("p_rel_l2", "p_rel_l2", HISTORY.p_rel_l2),
        ("T_rel_l2", "T_rel_l2", HISTORY.T_rel_l2),
        (
            "u_mean_rel_l2",
            "0.5*(u1_rel_l2+u2_rel_l2)",
            HISTORY.u_mean_rel_l2,
        ),
        (
            "pT_mean_rel_l2",
            "0.5*(p_rel_l2+T_rel_l2)",
            HISTORY.pT_mean_rel_l2,
        ),
        (
            "E_common_max",
            "max(u1_rel_l2,u2_rel_l2,p_rel_l2,T_rel_l2)",
            HISTORY.common_rel_l2_max,
        ),
    )
    final_index = len(HISTORY.fs_iteration) - 1
    rows: List[Dict[str, object]] = []
    for metric_name, metric_definition, metric_values in metric_histories:
        values = np.asarray(metric_values, dtype=float)
        for threshold in TIME_TO_ACCURACY_THRESHOLDS:
            hit: Optional[int] = None
            for index in range(len(values) - TTA_PATIENCE + 1):
                if np.all(values[index : index + TTA_PATIENCE] <= threshold):
                    hit = index
                    break
            reached = hit is not None
            record_index = hit if hit is not None else final_index
            confirmation_index = (
                hit + TTA_PATIENCE - 1 if hit is not None else None
            )
            row: Dict[str, object] = _comparison_run_identity()
            row.update(
                {
                    "metric_name": metric_name,
                    "metric_definition": metric_definition,
                    "threshold": float(threshold),
                    "patience": TTA_PATIENCE,
                    "reached": reached,
                    "first_fs_iteration": (
                        int(HISTORY.fs_iteration[record_index])
                        if reached
                        else None
                    ),
                    "confirmation_fs_iteration": (
                        int(HISTORY.fs_iteration[confirmation_index])
                        if confirmation_index is not None
                        else None
                    ),
                    "time_to_accuracy_seconds": (
                        float(HISTORY.training_time[record_index])
                        if reached
                        else None
                    ),
                    "effective_training_lower_bound_seconds": (
                        None
                        if reached
                        else float(HISTORY.training_time[final_index])
                    ),
                    "metric_value_at_first_reach": (
                        float(values[record_index]) if reached else None
                    ),
                    "final_metric_value": float(values[final_index]),
                    "record_kind": (
                        "patience_confirmed_first_reach"
                        if reached
                        else "final_not_reached"
                    ),
                }
            )
            row.update(_comparison_values_at(record_index))
            rows.append(row)
    return rows


def save_accuracy_reports() -> None:
    """Save fixed-budget and patience-confirmed TTA comparison tables."""
    fixed_budget_rows = build_fixed_budget_accuracy_rows()
    time_to_accuracy_rows = build_time_to_accuracy_rows()
    if not fixed_budget_rows or not time_to_accuracy_rows:
        raise RuntimeError("Accuracy reports require at least one completed sweep.")

    for stem, rows in (
        ("fixed_budget_accuracy", fixed_budget_rows),
        ("time_to_accuracy", time_to_accuracy_rows),
    ):
        with open(
            RESULT_DIR / f"{stem}.csv",
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with open(
            RESULT_DIR / f"{stem}.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(rows, file, indent=2, ensure_ascii=False)

    comparison_summary = {
        "definitions": {
            "primary_metric": (
                "E_common_max="
                "max(u1_rel_l2,u2_rel_l2,p_rel_l2,T_rel_l2)"
            ),
            "group_mean_metrics": (
                "u_mean_rel_l2=0.5*(u1_rel_l2+u2_rel_l2); "
                "pT_mean_rel_l2=0.5*(p_rel_l2+T_rel_l2)"
            ),
            "time_to_accuracy": (
                "first completed FS iteration starting a run of "
                f"{TTA_PATIENCE} consecutive values at or below the threshold"
            ),
            "time_source": (
                "effective_training_seconds="
                "optimizer_seconds+convergence_check_seconds"
            ),
            "jit_warmup_excluded": True,
            "test_grid_evaluation_excluded": True,
        },
        "thresholds": list(TIME_TO_ACCURACY_THRESHOLDS),
        "TTA_patience_completed_sweeps": TTA_PATIENCE,
        "fixed_budget_checkpoints": list(FIXED_BUDGET_CHECKPOINTS),
        "fixed_budget_accuracy": fixed_budget_rows,
        "time_to_accuracy": time_to_accuracy_rows,
    }
    with open(
        RESULT_DIR / "comparison_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(comparison_summary, file, indent=2, ensure_ascii=False)

    for row in fixed_budget_rows:
        if row["checkpoint_reached"]:
            LOGGER.info(
                "Fixed-budget FS %d: time=%.3f s E_common=%.4e",
                row["requested_fs_iteration"],
                row["effective_training_seconds"],
                row["E_common_max"],
            )
        else:
            LOGGER.info(
                "Fixed-budget FS %d: NR (completed FS %d)",
                row["requested_fs_iteration"],
                row["completed_fs_iterations"],
            )

    for row in time_to_accuracy_rows:
        if row["reached"]:
            LOGGER.info(
                "Time-to-accuracy %s<=%.3e: FS=%d confirm-FS=%d "
                "time=%.3f s value=%.4e",
                row["metric_name"],
                row["threshold"],
                row["first_fs_iteration"],
                row["confirmation_fs_iteration"],
                row["time_to_accuracy_seconds"],
                row["metric_value_at_first_reach"],
            )
        else:
            LOGGER.info(
                "Time-to-accuracy %s<=%.3e: NR, time>%.3f s, final=%.4e",
                row["metric_name"],
                row["threshold"],
                row["effective_training_lower_bound_seconds"],
                row["final_metric_value"],
            )


def save_profile_rows() -> None:
    """Save one machine-readable row for every ENGD inner update."""
    if not PROFILE_ROWS:
        return

    fieldnames = list(PROFILE_ROWS[0].keys())
    output_path = RESULT_DIR / "engd_profile.csv"

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(PROFILE_ROWS)

    with open(
        RESULT_DIR / "engd_profile.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(PROFILE_ROWS, file, indent=2, ensure_ascii=False)


def memory_snapshot() -> Dict[str, object]:
    """Return portable host/device peak-memory diagnostics when available."""

    snapshot: Dict[str, object] = {}
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # Linux reports ru_maxrss in KiB; macOS reports bytes.
    rss_scale = 1 if sys.platform == "darwin" else 1024
    snapshot["process_peak_rss_bytes"] = int(usage.ru_maxrss * rss_scale)

    try:
        device = jax.devices()[0]
        snapshot["device_platform"] = device.platform
        snapshot["device_kind"] = device.device_kind
        stats = device.memory_stats() or {}
        for source_key, target_key in (
            ("bytes_in_use", "device_bytes_in_use"),
            ("peak_bytes_in_use", "device_peak_bytes_in_use"),
            ("bytes_limit", "device_bytes_limit"),
        ):
            if source_key in stats:
                snapshot[target_key] = int(stats[source_key])
    except Exception as error:
        snapshot["device_memory_stats_error"] = repr(error)
    return snapshot


def save_metadata(timing: TrainingTiming) -> None:
    metadata = {
        "method": "THP-GFS-ENGD-three-field-compressible",
        "exact_case": EXACT_CASE,
        "profiling": {
            "solver": ENGD_SOLVER,
            "compare_lstsq_reference": COMPARE_LSTSQ_REFERENCE,
            "relative_cutoff": RELATIVE_CUTOFF,
            "automatic_fp64_cutoff": True,
            "refinement_steps": REFINEMENT_STEPS,
            "gram_ridge": GRAM_RIDGE,
            "lstsq_rcond": LSTSQ_RCOND,
            "line_search": "vectorized-logarithmic-grid",
            "line_search_positive_levels": LINE_SEARCH_LEVELS,
            "line_search_implicit_zero_rejection": True,
            "line_search_cached_frozen_terms": (
                JACOBIAN_BACKEND == "pointwise" and CACHE_LINE_SEARCH
            ),
            "profile_rows": len(PROFILE_ROWS),
            "jacobian_backend": JACOBIAN_BACKEND,
            "point_block_pt": POINT_BLOCK_PT,
            "point_block_u": POINT_BLOCK_U,
            "fuse_gram_blocks": FUSE_GRAM_BLOCKS,
            "cache_line_search": CACHE_LINE_SEARCH,
        },
        "seed": SEED,
        "physical_parameters": {
            "E": E,
            "nu": NU,
            "lambda": LAMBDA,
            "mu": MU,
            "alpha": ALPHA,
            "beta": BETA,
            "a0": A0,
            "b0": B0,
            "c0": C0,
            "K": KAPPA,
            "Theta": THETA,
            "t_final": T_FINAL,
        },
        "loss_weighting": {
            "objective_definition": (
                "Each block approximates "
                "0.5 * weight * integral(||residual||^2)."
            ),
            "block_scale_definition": "sqrt(weight * measure / n_points)",
            "weights": {
                "u": {
                    name: float(value)
                    for name, value in WEIGHTS_U.items()
                },
                "pt": {
                    name: float(value)
                    for name, value in WEIGHTS_PT.items()
                },
            },
            "residual_block_scales": {
                "u": {
                    "pde": float(SCALE_U_PDE),
                    "dirichlet": float(SCALE_U_DIR),
                    "neumann": float(SCALE_U_NEU),
                    "ic": float(SCALE_U_IC),
                },
                "pt": {
                    "pde": float(SCALE_PT_PDE),
                    "dirichlet": float(SCALE_PT_DIR),
                    "ic": float(SCALE_PT_IC),
                },
            },
        },
        "gfs": {
            "gamma": GAMMA_FS,
            "S11": S11,
            "S12": S12,
            "S21": S21,
            "S22": S22,
            "iterations_requested": FS_ITERATIONS,
            "iterations_completed": len(HISTORY.fs_iteration),
            "adaptive_stop_enabled": ENABLE_FS_EARLY_STOP,
            "adaptive_stop_min_iterations": FS_MIN_ITERATIONS,
            "adaptive_stop_patience": FS_EARLY_STOP_PATIENCE,
            "adaptive_stop_outer_tolerance": FS_EARLY_STOP_OUTER_TOL,
            "adaptive_stop_coupled_pt_tolerance": FS_EARLY_STOP_COUPLED_PT_TOL,
            "adaptive_stop_u_loss_tolerance": FS_EARLY_STOP_U_LOSS_TOL,
            "adaptive_stop_triggered": bool(
                ENABLE_FS_EARLY_STOP
                and HISTORY.stop_counter
                and HISTORY.stop_counter[-1] >= FS_EARLY_STOP_PATIENCE
            ),
            "final_stop_counter": int(HISTORY.stop_counter[-1])
            if HISTORY.stop_counter else 0,
        },
        "network": {
            "u_sizes": NET_U_SIZES,
            "pt_sizes": NET_PT_SIZES,
        },
        "sampling": {
            "fixed_uniform": True,
            "grid_shape": {
                "pde": [N_XY_PDE, N_XY_PDE, N_T_PDE],
                "ic": [N_XY_IC, N_XY_IC],
                "boundary_per_side": [N_SIDE, N_T_BOUNDARY],
                "evaluation": [N_XY_EVAL, N_XY_EVAL, N_T_EVAL],
            },
            "pde_points": int(PDE_POINTS.shape[0]),
            "ic_points": int(IC_POINTS.shape[0]),
            "u_dirichlet_points": int(U_DIR_POINTS.shape[0]),
            "u_neumann_points": int(U_NEU_POINTS.shape[0]),
            "pt_dirichlet_points": int(PT_DIR_POINTS.shape[0]),
            "evaluation_points": int(EVAL_POINTS.shape[0]),
        },
        "engd_pt": {
            **asdict(ENGD_CONFIG_PT),
            "steps_per_gfs_iteration": ENGD_STEPS_PT,
        },
        "engd_u": {
            **asdict(ENGD_CONFIG_U),
            "steps_per_gfs_iteration": ENGD_STEPS_U,
        },
        "timing": {
            **asdict(timing),
            "optimizer_seconds": timing.optimizer_seconds,
            "effective_training_seconds": timing.effective_training_seconds,
            "effective_training_definition": (
                "optimizer_seconds + convergence_check_seconds"
            ),
            "jit_warmup_excluded": True,
            "test_grid_evaluation_excluded": True,
            "lstsq_reference_validation_excluded": True,
        },
        "accuracy_reporting": {
            "primary_metric": (
                "E_common_max="
                "max(u1_rel_l2,u2_rel_l2,p_rel_l2,T_rel_l2)"
            ),
            "u_mean_metric": "0.5*(u1_rel_l2+u2_rel_l2)",
            "pT_mean_metric": "0.5*(p_rel_l2+T_rel_l2)",
            "time_to_accuracy_thresholds": list(
                TIME_TO_ACCURACY_THRESHOLDS
            ),
            "time_to_accuracy_patience_completed_sweeps": TTA_PATIENCE,
            "fixed_budget_checkpoints": list(FIXED_BUDGET_CHECKPOINTS),
            "time_to_accuracy_uses_effective_training_seconds": True,
            "time_to_accuracy_is_discrete_at_fs_checkpoints": True,
        },
        "memory": memory_snapshot(),
        "runtime": {
            "python": sys.version,
            "jax": jax.__version__,
            "jax_backend": jax.default_backend(),
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
            "xla_python_client_preallocate": os.environ.get(
                "XLA_PYTHON_CLIENT_PREALLOCATE"
            ),
            "xla_python_client_mem_fraction": os.environ.get(
                "XLA_PYTHON_CLIENT_MEM_FRACTION"
            ),
        },
        "save_plots": SAVE_PLOTS,
        "fast_debug": FAST_DEBUG,
    }

    with open(RESULT_DIR / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)


# =============================================================================
# 13. Exact Gram microbenchmark
# =============================================================================


def _global_pt_gram_gradient(
    flat_pt: Array,
    flat_u_old: Array,
    flat_pt_old: Array,
) -> Tuple[Array, Array, Array]:
    residual = residual_pt_flat(flat_pt, flat_u_old, flat_pt_old)
    jacobian = JACOBIAN_PT_FLAT(flat_pt, flat_u_old, flat_pt_old)
    loss = half_squared_norm(residual)
    gram = jacobian.T @ jacobian
    gradient = jacobian.T @ residual
    return loss, gram, gradient


def _global_u_gram_gradient(
    flat_u: Array,
    flat_pt: Array,
) -> Tuple[Array, Array, Array]:
    residual = residual_u_flat(flat_u, flat_pt)
    jacobian = JACOBIAN_U_FLAT(flat_u, flat_pt)
    loss = half_squared_norm(residual)
    gram = jacobian.T @ jacobian
    gradient = jacobian.T @ residual
    return loss, gram, gradient


def _synchronize_triplet(values: Tuple[Array, Array, Array]) -> None:
    for value in values:
        value.block_until_ready()


def _relative_error(candidate: Array, reference: Array) -> float:
    return float(
        jnp.linalg.norm(candidate - reference)
        / jnp.maximum(jnp.linalg.norm(reference), 1.0e-30)
    )


def _timed_repetitions(
    function: Callable[[], Tuple[Array, Array, Array]],
    warmup: int,
    repeats: int,
) -> Tuple[List[float], Tuple[Array, Array, Array]]:
    result = function()
    _synchronize_triplet(result)
    for _ in range(max(warmup - 1, 0)):
        result = function()
        _synchronize_triplet(result)

    timings: List[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        _synchronize_triplet(result)
        timings.append(time.perf_counter() - start)
    return timings, result


def run_jacobian_microbenchmark() -> None:
    """Compare exact global jacfwd and pointwise-jacrev Gram construction."""
    rows: List[Dict[str, object]] = []
    flat_u = FLAT_U
    flat_pt = FLAT_PT
    flat_u_old = jnp.array(flat_u)
    flat_pt_old = jnp.array(flat_pt)

    LOGGER.info("============================================================")
    LOGGER.info("Exact Jacobian/Gram microbenchmark")
    LOGGER.info(
        "point blocks=%s, warmup=%d, repeats=%d, fast_debug=%s",
        MICROBENCH_BLOCK_SIZES,
        MICROBENCH_WARMUP,
        MICROBENCH_REPEATS,
        FAST_DEBUG,
    )

    # Verify the cached production line-search residual against the original
    # full residual path before benchmarking throughput.
    validation_block = MICROBENCH_BLOCK_SIZES[0]
    validation_pt_context = build_pointwise_pt_context(
        flat_u_old, flat_pt_old, validation_block
    )
    validation_u_context = build_pointwise_u_context(
        flat_pt, validation_block
    )
    validation_pt_frozen = validation_pt_context.pde_frozen.reshape(-1, 2)[
        : PDE_POINTS.shape[0]
    ]
    validation_u_pde_frozen = validation_u_context.pde_frozen.reshape(-1, 2)[
        : PDE_POINTS.shape[0]
    ]
    validation_u_neu_frozen = validation_u_context.neu_frozen.reshape(-1, 2)[
        : U_NEU_POINTS.shape[0]
    ]

    pt_offset = 1.0e-7 * jnp.sin(
        jnp.arange(flat_pt.size, dtype=flat_pt.dtype)
    )
    u_offset = 1.0e-7 * jnp.cos(
        jnp.arange(flat_u.size, dtype=flat_u.dtype)
    )
    pt_candidates = jnp.stack(
        (flat_pt, flat_pt + pt_offset, flat_pt - pt_offset)
    )
    u_candidates = jnp.stack(
        (flat_u, flat_u + u_offset, flat_u - u_offset)
    )
    pt_losses_reference = candidate_losses_pt(
        pt_candidates, flat_u_old, flat_pt_old
    )
    pt_losses_cached = candidate_losses_pt_cached(
        pt_candidates, validation_pt_frozen
    )
    u_losses_reference = candidate_losses_u(u_candidates, flat_pt)
    u_losses_cached = candidate_losses_u_cached(
        u_candidates,
        validation_u_pde_frozen,
        validation_u_neu_frozen,
    )
    pt_cache_error = _relative_error(
        pt_losses_cached, pt_losses_reference
    )
    u_cache_error = _relative_error(u_losses_cached, u_losses_reference)
    LOGGER.info(
        "cached line-search equivalence: pT=%.3e u=%.3e",
        pt_cache_error,
        u_cache_error,
    )
    if max(pt_cache_error, u_cache_error) > 1.0e-10:
        raise RuntimeError(
            "Cached line-search residual is not equivalent to the reference "
            f"path: pT={pt_cache_error:.3e}, u={u_cache_error:.3e}."
        )

    reference_pt_timings, reference_pt = _timed_repetitions(
        lambda: _global_pt_gram_gradient(flat_pt, flat_u_old, flat_pt_old),
        MICROBENCH_WARMUP,
        MICROBENCH_REPEATS,
    )
    reference_u_timings, reference_u = _timed_repetitions(
        lambda: _global_u_gram_gradient(flat_u, flat_pt),
        MICROBENCH_WARMUP,
        MICROBENCH_REPEATS,
    )

    for subproblem, timings in (
        ("pT", reference_pt_timings),
        ("u", reference_u_timings),
    ):
        row = {
            "subproblem": subproblem,
            "backend": "global_jacfwd",
            "block_size": 0,
            "mean_seconds": float(np.mean(timings)),
            "median_seconds": float(np.median(timings)),
            "std_seconds": float(np.std(timings)),
            "min_seconds": float(np.min(timings)),
            "max_seconds": float(np.max(timings)),
            "loss_relative_error": 0.0,
            "gram_relative_error": 0.0,
            "gradient_relative_error": 0.0,
            "speedup_vs_jacfwd": 1.0,
        }
        rows.append(row)
        LOGGER.info(
            "%s global jacfwd: median=%.6fs mean=%.6fs std=%.6fs",
            subproblem,
            row["median_seconds"],
            row["mean_seconds"],
            row["std_seconds"],
        )

    for block_size in MICROBENCH_BLOCK_SIZES:
        context_pt = build_pointwise_pt_context(
            flat_u_old, flat_pt_old, block_size
        )
        context_u = build_pointwise_u_context(flat_pt, block_size)

        pt_timings, pt_result = _timed_repetitions(
            lambda: pointwise_pt_gram_gradient(flat_pt, context_pt),
            MICROBENCH_WARMUP,
            MICROBENCH_REPEATS,
        )
        u_timings, u_result = _timed_repetitions(
            lambda: pointwise_u_gram_gradient(flat_u, context_u),
            MICROBENCH_WARMUP,
            MICROBENCH_REPEATS,
        )

        for subproblem, timings, result, reference, reference_timings in (
            ("pT", pt_timings, pt_result, reference_pt, reference_pt_timings),
            ("u", u_timings, u_result, reference_u, reference_u_timings),
        ):
            loss_error = abs(float(result[0] - reference[0])) / max(
                abs(float(reference[0])), 1.0e-30
            )
            gram_error = _relative_error(result[1], reference[1])
            gradient_error = _relative_error(result[2], reference[2])
            median_seconds = float(np.median(timings))
            reference_median = float(np.median(reference_timings))
            row = {
                "subproblem": subproblem,
                "backend": "pointwise_jacrev_streaming",
                "block_size": block_size,
                "mean_seconds": float(np.mean(timings)),
                "median_seconds": median_seconds,
                "std_seconds": float(np.std(timings)),
                "min_seconds": float(np.min(timings)),
                "max_seconds": float(np.max(timings)),
                "loss_relative_error": loss_error,
                "gram_relative_error": gram_error,
                "gradient_relative_error": gradient_error,
                "speedup_vs_jacfwd": reference_median / median_seconds,
            }
            rows.append(row)
            LOGGER.info(
                "%s pointwise block=%d: median=%.6fs speedup=%.3f "
                "lossErr=%.3e GramErr=%.3e gradErr=%.3e",
                subproblem,
                block_size,
                median_seconds,
                row["speedup_vs_jacfwd"],
                loss_error,
                gram_error,
                gradient_error,
            )

    for subproblem in ("pT", "u"):
        candidates = [
            row
            for row in rows
            if row["subproblem"] == subproblem
            and row["backend"] == "pointwise_jacrev_streaming"
        ]
        best = min(candidates, key=lambda row: row["median_seconds"])
        LOGGER.info(
            "%s recommended point block on this device: %d "
            "(median %.6f s)",
            subproblem,
            best["block_size"],
            best["median_seconds"],
        )

    csv_path = RESULT_DIR / "jacobian_microbenchmark.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(
        RESULT_DIR / "jacobian_microbenchmark.json", "w", encoding="utf-8"
    ) as file:
        json.dump(rows, file, indent=2, ensure_ascii=False)

    LOGGER.info("Microbenchmark outputs saved to %s", RESULT_DIR)


# =============================================================================
# 14. Main entry point
# =============================================================================


if __name__ == "__main__":
    if JACOBIAN_MICROBENCH:
        run_jacobian_microbenchmark()
    else:
        FINAL_FLAT_U, FINAL_FLAT_PT, TRAINING_TIMING = train()

        save_checkpoint(FINAL_FLAT_U, FINAL_FLAT_PT)
        save_history()
        save_accuracy_reports()
        save_profile_rows()
        save_metadata(TRAINING_TIMING)

        if not FAST_DEBUG and SAVE_PLOTS:
            try:
                plot_final_fields(FINAL_FLAT_U, FINAL_FLAT_PT, T_FINAL)
                plot_history()
            except Exception:
                LOGGER.exception(
                    "Plotting failed, but model, history, and metadata were saved."
                )

        LOGGER.info("All outputs saved to %s", RESULT_DIR)
