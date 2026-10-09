"""Canonical four-field THP generalized fixed-stress energy-natural core.

Two networks represent (u1,u2,xi) and (p,T), with
xi=-lambda*div(u)+alpha*p+beta*T. Each sweep updates p/T with the previous
u/xi frozen, then u/xi with the new p/T frozen. Positive fixed Gram damping
gives the ENLM direction used in the paper; zero damping gives ENGD.
Historical ENGD environment-variable and output-key spellings are retained.

Group 5 paper runs use raw residuals or modal_both_scaled with
parallel_scale=sigma_parallel and perpendicular_scale=d_perp. The modal
rotation diagonalizes storage; the exact cross-diffusion term is retained
when K and Theta differ. Generalized residual maps and rank-safe ridge remain
available only for historical exploratory controls, outside the paper matrix.

Historical internal S10,S11,S12,S13 correspond to paper S8,S9,S10,S11.
THP_GROUP5_PAPER_CASE_ID supplies the paper label for output while preserving
the original case guards. The comparable-transport guard accepts the paper
FS50 budget and historical FS100 supplements. Frozen data are never rewritten.
Raw S7 paper runs also report the storage-aligned modal solution errors.

Examples from the repository root:
    python thp-four-field-core.py
    THP_FAST_DEBUG=1 THP_SAVE_PLOTS=0 python thp-four-field-core.py

Use the group-specific shell launchers for complete controlled configurations.
New runs default to runs/ and never merge an existing run directory.
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

# Default new runs go under runs/; THP_RESULT_ROOT overrides this destination.

RESULT_ROOT = Path(os.environ.get("THP_RESULT_ROOT", str(SCRIPT_DIR / "runs"))).resolve()


# The default run name includes seed, timestamp, and PID.
RUN_NAME = os.environ.get(
    "THP_RUN_NAME",
    f"thp_four_field_engd_seed{SEED}_{TIME_STAMP}_pid{os.getpid()}",
)


RESULT_DIR = (RESULT_ROOT / RUN_NAME).resolve()



# Frozen scientific data are never a training destination.
for _frozen in SCRIPT_DIR.glob("group*/results"):
    if RESULT_DIR == _frozen.resolve() or _frozen.resolve() in RESULT_DIR.parents:
        raise ValueError("Training output must be outside all frozen results directories.")

# Scientific run directories are never merged or overwritten.
RESULT_DIR.mkdir(parents=True, exist_ok=False)


def build_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("THP-four-field-ENGD")
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
# 2. Physical parameters and four-field coefficients
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

# Optional archive identifiers.  Existing Group 1--4 launchers need not set
# them; Group 5 uses them to keep residual scaling and loss weighting auditable.
GROUP5_CASE_ID = os.environ.get("THP_GROUP5_CASE_ID", "").strip()
GROUP5_VARIANT = os.environ.get("THP_GROUP5_VARIANT", "").strip()
IS_GROUP5_RUN = bool(GROUP5_CASE_ID and GROUP5_VARIANT)
GROUP5_PAPER_CASE_ID = os.environ.get("THP_GROUP5_PAPER_CASE_ID", "").strip()
GROUP5_LEGACY_TO_PAPER = {"S10": "S8", "S11": "S9", "S12": "S10", "S13": "S11"}
if GROUP5_PAPER_CASE_ID and (
    not IS_GROUP5_RUN or GROUP5_PAPER_CASE_ID !=
    GROUP5_LEGACY_TO_PAPER.get(GROUP5_CASE_ID, GROUP5_CASE_ID)
):
    raise ValueError("Group 5 paper and historical case identifiers disagree.")


SPACE_DIMENSION = 2

if E <= 0.0:
    raise ValueError("THP_E must be positive.")
if not 0.0 < NU < 0.5:
    raise ValueError("THP_NU must lie in (0, 0.5).")
LAMBDA = E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
MU = E / (2.0 * (1.0 + NU))
if LAMBDA <= 0.0 or MU <= 0.0:
    raise ValueError("The four-field implementation requires lambda, mu > 0.")
if min(ALPHA, BETA, C0, A0, B0, KAPPA, THETA) < 0.0:
    raise ValueError("THP physical coefficients must be non-negative.")
if KAPPA <= 0.0 or THETA <= 0.0:
    raise ValueError("THP_KAPPA and THP_THETA must be positive.")

INV_LAMBDA = 1.0 / LAMBDA

# Coefficients in the transformed p/T storage matrix:
# [[c0+alpha^2/lambda, alpha*beta/lambda-b0],
#  [alpha*beta/lambda-b0, a0+beta^2/lambda]].
C_ALPHA = C0 + ALPHA * ALPHA * INV_LAMBDA
C_ALPHA_BETA = ALPHA * BETA * INV_LAMBDA - B0
C_BETA = A0 + BETA * BETA * INV_LAMBDA
FOUR_FIELD_STORAGE_MATRIX = jnp.array(
    [[C_ALPHA, C_ALPHA_BETA], [C_ALPHA_BETA, C_BETA]]
)
XI_COUPLING_VECTOR = jnp.array(
    [-ALPHA * INV_LAMBDA, -BETA * INV_LAMBDA]
)

# Hydraulic and thermal diffusion coefficients.
DIFFUSIVITY = jnp.array([KAPPA, THETA])

# p/T PDE residual metric used during training.
#
# raw:
#     use the original residual vector [r_p, r_T].
#
# modal:
#     apply only the orthogonal transformation
#     [r_parallel, r_perpendicular].  Because the transformation is
#     orthogonal, this mode is mathematically equivalent to raw.
#
# modal_perp_scaled:
#     additionally divide the storage-null perpendicular residual by its
#     modal diffusion coefficient d_perp.  For Kappa=Theta this reduces to
#     division by the common diffusivity.
#
# modal_both_scaled:
#     divide the parallel residual by its storage coefficient and the
#     storage-null perpendicular residual by d_perp.  For Kappa!=Theta the
#     transformed equations retain the exact cross-diffusion terms; this mode
#     therefore tests whether simple storage-aligned modal scaling is already
#     sufficient in the jointly-small unequal-transport regime.
#
# diffusion_whitened:
#     apply D^(-1/2) to the original residual.  This is an ablation for the
#     unequal-diffusion experiments, not the recommended complete map.
#
# generalized_modal_balanced:
#     apply M^(-1) U^T D^(-1/2), where U diagonalizes the diffusion-whitened
#     storage matrix and M_ii=max(sigma_i,1).  This isolates spectral balancing
#     from the final symmetric overall normalization.
#
# generalized_modal_scaled:
#     apply eta M^(-1) U^T D^(-1/2), eta=(K*Theta)^(-1/4).  This is the complete
#     modal-coordinate generalized residual metric for K != Theta.
#
# generalized_physical_scaled:
#     apply the symmetric physical-variable pullback
#     D^(-1/2) U M^(-1) U^T D^(-1/2).  This candidate removes eta and returns
#     the balanced generalized equations to the original (p,T) coordinates.
PT_RESIDUAL_MODE = os.environ.get(
    "THP_PT_RESIDUAL_MODE", "raw"
).strip().lower()

if PT_RESIDUAL_MODE not in {
    "raw",
    "modal",
    "modal_perp_scaled",
    "modal_both_scaled",
    "diffusion_whitened",
    "generalized_modal_balanced",
    "generalized_modal_scaled",
    "generalized_physical_scaled",
}:
    raise ValueError(
        "THP_PT_RESIDUAL_MODE must be one of "
        "'raw', 'modal', 'modal_perp_scaled', 'modal_both_scaled', "
        "'diffusion_whitened', 'generalized_modal_balanced', "
        "'generalized_modal_scaled', or 'generalized_physical_scaled', "
        f"but received {PT_RESIDUAL_MODE!r}."
    )

PT_MODAL_NORM = math.sqrt(ALPHA * ALPHA + BETA * BETA)

PT_LEGACY_MODAL_MODES = {
    "modal",
    "modal_perp_scaled",
    "modal_both_scaled",
}
PT_GENERALIZED_RESIDUAL_MODES = {
    "diffusion_whitened",
    "generalized_modal_balanced",
    "generalized_modal_scaled",
    "generalized_physical_scaled",
}

# Preserve the validation behavior of the three pre-existing generalized
# modes.  The new physical pullback intentionally does not use eta.
PT_GENERALIZED_ETA_MODES = {
    "diffusion_whitened",
    "generalized_modal_balanced",
    "generalized_modal_scaled",
}

# Keep Group 1--4 raw outputs byte-schema compatible with the original core.
# Group 5 modal diagnostics require its archive identifiers. Non-raw modes
# and paper S7 raw controls report them; Group 1--4 raw outputs are unchanged.
REPORT_PT_MODAL_DIAGNOSTICS = (
    IS_GROUP5_RUN and (PT_RESIDUAL_MODE != "raw" or GROUP5_PAPER_CASE_ID == "S7")
)

if PT_RESIDUAL_MODE in PT_LEGACY_MODAL_MODES and PT_MODAL_NORM <= 0.0:
    raise ValueError(
        "Modal p/T residuals require alpha^2+beta^2 > 0."
    )

if PT_MODAL_NORM > 0.0:
    PT_PARALLEL_STORAGE = (
        ALPHA * ALPHA * C_ALPHA
        + 2.0 * ALPHA * BETA * C_ALPHA_BETA
        + BETA * BETA * C_BETA
    ) / (PT_MODAL_NORM * PT_MODAL_NORM)

    PT_PERP_STORAGE = (
        BETA * BETA * C_ALPHA
        - 2.0 * ALPHA * BETA * C_ALPHA_BETA
        + ALPHA * ALPHA * C_BETA
    ) / (PT_MODAL_NORM * PT_MODAL_NORM)

    PT_MODAL_STORAGE_COUPLING = (
        ALPHA * BETA * (C_ALPHA - C_BETA)
        + (BETA * BETA - ALPHA * ALPHA) * C_ALPHA_BETA
    ) / (PT_MODAL_NORM * PT_MODAL_NORM)
else:
    PT_PARALLEL_STORAGE = math.nan
    PT_PERP_STORAGE = math.nan
    PT_MODAL_STORAGE_COUPLING = math.nan

# Generalized unequal-diffusion block metric.  NumPy is used only to construct
# a deterministic 2x2 constant; the training residual itself remains a JAX
# operation and therefore works in both global and pointwise streamed backends.
_PT_DIFFUSION_INVERSE_SQRT_NP = np.diag(
    [KAPPA ** -0.5, THETA ** -0.5]
)
_PT_STORAGE_MATRIX_NP = np.array(
    [[C_ALPHA, C_ALPHA_BETA], [C_ALPHA_BETA, C_BETA]],
    dtype=np.float64,
)
_PT_WHITENED_STORAGE_NP = (
    _PT_DIFFUSION_INVERSE_SQRT_NP
    @ _PT_STORAGE_MATRIX_NP
    @ _PT_DIFFUSION_INVERSE_SQRT_NP
)
(
    _pt_generalized_eigenvalues,
    _pt_generalized_eigenvectors,
) = np.linalg.eigh(_PT_WHITENED_STORAGE_NP)
_pt_generalized_order = np.argsort(_pt_generalized_eigenvalues)[::-1]
_pt_generalized_eigenvalues = _pt_generalized_eigenvalues[
    _pt_generalized_order
]
_pt_generalized_eigenvectors = _pt_generalized_eigenvectors[
    :, _pt_generalized_order
]

# Canonicalize eigenvector signs so metadata are reproducible across LAPACK
# implementations.  Signs/order do not affect the squared residual norm.
for _column in range(_pt_generalized_eigenvectors.shape[1]):
    _pivot = int(
        np.argmax(np.abs(_pt_generalized_eigenvectors[:, _column]))
    )
    if _pt_generalized_eigenvectors[_pivot, _column] < 0.0:
        _pt_generalized_eigenvectors[:, _column] *= -1.0

_pt_generalized_psd_tolerance = 1.0e-12 * max(
    1.0, float(np.linalg.norm(_PT_WHITENED_STORAGE_NP, ord=2))
)
if (
    PT_RESIDUAL_MODE in PT_GENERALIZED_RESIDUAL_MODES
    and float(np.min(_pt_generalized_eigenvalues))
    < -_pt_generalized_psd_tolerance
):
    raise ValueError(
        "Generalized p/T residual scaling requires the four-field storage "
        "matrix to be positive semidefinite.  The minimum eigenvalue of "
        "D^(-1/2) C D^(-1/2) is "
        f"{float(np.min(_pt_generalized_eigenvalues)):.6e}."
    )

# Small negative eigenvalues caused only by roundoff are set to zero.
_pt_generalized_eigenvalues = np.maximum(
    _pt_generalized_eigenvalues, 0.0
)
_pt_generalized_balance = np.maximum(
    _pt_generalized_eigenvalues, 1.0
)
_PT_GENERALIZED_PHYSICAL_MATRIX_NP = (
    _PT_DIFFUSION_INVERSE_SQRT_NP
    @ _pt_generalized_eigenvectors
    @ np.diag(1.0 / _pt_generalized_balance)
    @ _pt_generalized_eigenvectors.T
    @ _PT_DIFFUSION_INVERSE_SQRT_NP
)
_pt_generalized_physical_eigenvalues = np.linalg.eigvalsh(
    _PT_GENERALIZED_PHYSICAL_MATRIX_NP
)
_pt_generalized_physical_condition = float(
    np.linalg.cond(_PT_GENERALIZED_PHYSICAL_MATRIX_NP)
)
PT_GENERALIZED_ETA = float(
    os.environ.get(
        "THP_PT_GENERALIZED_ETA",
        str((KAPPA * THETA) ** -0.25),
    )
)
if (
    PT_RESIDUAL_MODE in PT_GENERALIZED_ETA_MODES
    and (
        not math.isfinite(PT_GENERALIZED_ETA)
        or PT_GENERALIZED_ETA <= 0.0
    )
):
    raise ValueError(
        "THP_PT_GENERALIZED_ETA must be positive and finite."
    )

PT_DIFFUSION_INVERSE_SQRT = jnp.asarray(
    _PT_DIFFUSION_INVERSE_SQRT_NP
)
PT_GENERALIZED_WHITENED_STORAGE = jnp.asarray(
    _PT_WHITENED_STORAGE_NP
)
PT_GENERALIZED_EIGENVALUES = jnp.asarray(
    _pt_generalized_eigenvalues
)
PT_GENERALIZED_EIGENVECTORS = jnp.asarray(
    _pt_generalized_eigenvectors
)
PT_GENERALIZED_BALANCE = jnp.asarray(_pt_generalized_balance)
PT_GENERALIZED_PHYSICAL_MATRIX = jnp.asarray(
    _PT_GENERALIZED_PHYSICAL_MATRIX_NP
)

PT_PARALLEL_RESIDUAL_SCALE = float(
    os.environ.get(
        "THP_PT_PARALLEL_RESIDUAL_SCALE",
        str(
            PT_PARALLEL_STORAGE
            if PT_RESIDUAL_MODE == "modal_both_scaled"
            and math.isfinite(PT_PARALLEL_STORAGE)
            and PT_PARALLEL_STORAGE > 0.0
            else 1.0
        ),
    )
)

# Diffusion coefficients in the storage-aligned modal basis.
# For y_parallel=(alpha*p+beta*T)/sqrt(alpha^2+beta^2) and
# y_perp=(beta*p-alpha*T)/sqrt(alpha^2+beta^2),
# D_m=[[d_parallel,d_cross],[d_cross,d_perp]].  Unequal Kappa/Theta therefore
# keep an exact cross-diffusion term rather than invalidating the modal basis.
if PT_MODAL_NORM > 0.0:
    PT_MODAL_PERP_DIFFUSION = (
        BETA * BETA * KAPPA + ALPHA * ALPHA * THETA
    ) / (PT_MODAL_NORM * PT_MODAL_NORM)
    PT_MODAL_PARALLEL_DIFFUSION = (
        ALPHA * ALPHA * KAPPA + BETA * BETA * THETA
    ) / (PT_MODAL_NORM * PT_MODAL_NORM)
    PT_MODAL_CROSS_DIFFUSION = (
        ALPHA * BETA * (THETA - KAPPA)
    ) / (PT_MODAL_NORM * PT_MODAL_NORM)
else:
    PT_MODAL_PERP_DIFFUSION = math.nan
    PT_MODAL_PARALLEL_DIFFUSION = math.nan
    PT_MODAL_CROSS_DIFFUSION = math.nan

PT_PERP_RESIDUAL_SCALE = float(
    os.environ.get(
        "THP_PT_PERP_RESIDUAL_SCALE",
        str(PT_MODAL_PERP_DIFFUSION),
    )
)

if PT_RESIDUAL_MODE in {
    "modal_perp_scaled",
    "modal_both_scaled",
}:
    # Unequal diffusivities are allowed.  The modal transform diagonalizes the
    # storage block, not the diffusion block, so d_cross is retained exactly
    # inside the transformed residual.  Scaling only changes the residual
    # metric and therefore leaves the exact PDE solution unchanged.
    if (
        not math.isfinite(PT_PERP_STORAGE)
        or abs(PT_PERP_STORAGE) > 1.0e-12
    ):
        raise ValueError(
            f"{PT_RESIDUAL_MODE} is currently restricted to the "
            "storage-null perpendicular mode. "
            f"Computed perpendicular storage={PT_PERP_STORAGE:.6e}."
        )

    if (
        not math.isfinite(PT_MODAL_STORAGE_COUPLING)
        or abs(PT_MODAL_STORAGE_COUPLING) > 1.0e-12
    ):
        raise ValueError(
            f"{PT_RESIDUAL_MODE} requires storage-aligned modal equations. "
            "The modal storage off-diagonal must vanish, but it is "
            f"{PT_MODAL_STORAGE_COUPLING:.6e}."
        )

    if (
        not math.isfinite(PT_PERP_RESIDUAL_SCALE)
        or PT_PERP_RESIDUAL_SCALE <= 0.0
    ):
        raise ValueError(
            "THP_PT_PERP_RESIDUAL_SCALE must be positive and finite."
        )

if PT_RESIDUAL_MODE == "modal_both_scaled":
    if (
        not math.isfinite(PT_PARALLEL_RESIDUAL_SCALE)
        or PT_PARALLEL_RESIDUAL_SCALE <= 0.0
    ):
        raise ValueError(
            "THP_PT_PARALLEL_RESIDUAL_SCALE must be "
            "positive and finite."
        )

    if (
        not math.isfinite(PT_PARALLEL_STORAGE)
        or PT_PARALLEL_STORAGE <= 0.0
    ):
        raise ValueError(
            "modal_both_scaled requires a positive "
            "parallel storage coefficient."
        )

T_FINAL = float(os.environ.get("THP_T_FINAL", "1"))

# Fixed production budget used for optimizer and formulation comparisons.
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

# The primary TTA metric contains u,p,T; the four-field diagnostic also includes xi.

TIME_TO_ACCURACY_THRESHOLDS = tuple(
    float(value.strip())
    for value in os.environ.get(
        "THP_TIME_TO_ACCURACY_THRESHOLDS",
        "1e-2,1e-3,8e-4,6e-4,4e-4,2e-4,1e-4",
    ).split(",")
    if value.strip()
)

# TTA is confirmed only after the threshold is satisfied for this many
# consecutive completed outer sweeps.
TTA_PATIENCE = int(os.environ.get("THP_TTA_PATIENCE", "3"))

# Report field errors at fixed outer-iteration budgets.
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
VERIFY_MANUFACTURED = (
    os.environ.get("THP_VERIFY_MANUFACTURED", "1") == "1"
)

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
        
        #     0, T_final/39, ..., T_final。
        t = jnp.linspace(0.0, t_final, n_t)
    else:
        # PDE points exclude t=0 because the initial condition is separate.
        # Exclude t=0 for PDE points; initial conditions are enforced separately.

        
        
        #
        
        #     T_final/n_t, 2*T_final/n_t, ..., T_final。
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

# The first network represents (u1,u2,xi); the legacy ``U`` variable names are
# retained internally so the validated ENGD backend remains minimally changed.
NET_U_SIZES = [3, 32, 32, 3]
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
#
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
def xi_exact(xyt: Array) -> Array:
    """Pseudo-total pressure xi=-lambda*div(u)+alpha*p+beta*T."""
    jac_u = U_EXACT_JAC(xyt)
    div_u = jac_u[0, 0] + jac_u[1, 1]
    return (
        -LAMBDA * div_u
        + ALPHA * p_exact(xyt)
        + BETA * temperature_exact(xyt)
    )


@jit
def uxi_exact(xyt: Array) -> Array:
    """Return [u1,u2,xi]."""
    return jnp.concatenate((u_exact(xyt), jnp.array([xi_exact(xyt)])))


XI_EXACT_GRAD = grad(xi_exact)
XI_EXACT_HESS = hessian(xi_exact)
UXI_EXACT_JAC = jacfwd(uxi_exact)


@jit
def four_field_mechanics_operator(
    hess_u: Array,
    grad_xi: Array,
) -> Array:
    """Return -2*mu*div(eps(u))+grad(xi)."""
    u1_xx = hess_u[0, 0, 0]
    u1_yy = hess_u[0, 1, 1]
    u1_xy = hess_u[0, 0, 1]

    u2_xx = hess_u[1, 0, 0]
    u2_yy = hess_u[1, 1, 1]
    u2_xy = hess_u[1, 0, 1]

    component_1 = -(
        2.0 * MU * u1_xx + MU * (u1_yy + u2_xy)
    ) + grad_xi[0]
    component_2 = -(
        MU * (u2_xx + u1_xy) + 2.0 * MU * u2_yy
    ) + grad_xi[1]
    return jnp.array([component_1, component_2])


@jit
def body_force_exact(xyt: Array) -> Array:
    """Manufactured force in four-field form."""
    return four_field_mechanics_operator(
        U_EXACT_HESS(xyt), XI_EXACT_GRAD(xyt)[:2]
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
    """Evaluate the exact total traction sigma_total @ n.

    sigma_total = 2*mu*eps(u) + lambda*div(u)*I - (alpha*p+beta*T)*I,
    where n is the outward unit normal."""
    jac_u = U_EXACT_JAC(xyt)
    xi_value = xi_exact(xyt)

    du1_dx, du1_dy = jac_u[0, 0], jac_u[0, 1]
    du2_dx, du2_dy = jac_u[1, 0], jac_u[1, 1]

    sigma11 = 2.0 * MU * du1_dx - xi_value
    sigma22 = 2.0 * MU * du2_dy - xi_value
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


U_JAC = jacfwd(u_value, argnums=1)     # shape (3,3): (u1,u2,xi) by (x,y,t)
U_HESS = hessian(u_value, argnums=1)   # shape (3,3,3)
PT_JAC = jacfwd(pt_value, argnums=1)   # Spatial-time Jacobian has shape (2,3).
PT_HESS = hessian(pt_value, argnums=1)  # Spatial-time Hessian has shape (2,3,3).


@jit
def mechanics_pde_residual_one(
    params_u: Params,
    params_pt: Params,
    xyt: Array,
) -> Array:
    uxi = u_value(params_u, xyt)
    jac_u = U_JAC(params_u, xyt)
    hess_u = U_HESS(params_u, xyt)
    p_value, t_value = pt_value(params_pt, xyt)

    momentum = (
        four_field_mechanics_operator(hess_u[:2], jac_u[2, :2])
        - body_force_exact(xyt)
    )
    constraint = (
        jac_u[0, 0]
        + jac_u[1, 1]
        + INV_LAMBDA * uxi[2]
        - ALPHA * INV_LAMBDA * p_value
        - BETA * INV_LAMBDA * t_value
    )
    return jnp.concatenate((momentum, jnp.array([constraint])))


@jit
def total_traction_pred_one(
    params_u: Params,
    params_pt: Params,
    xyt: Array,
) -> Array:
    jac_u = U_JAC(params_u, xyt)
    xi_value = u_value(params_u, xyt)[2]

    du1_dx, du1_dy = jac_u[0, 0], jac_u[0, 1]
    du2_dx, du2_dy = jac_u[1, 0], jac_u[1, 1]

    sigma11 = 2.0 * MU * du1_dx - xi_value
    sigma22 = 2.0 * MU * du2_dy - xi_value
    sigma12 = MU * (du1_dy + du2_dx)

    sigma_total = jnp.array(
        [[sigma11, sigma12], [sigma12, sigma22]]
    )
    # Predicted total traction.
    return sigma_total @ outward_normal(xyt)

@jit
def transform_flow_heat_training_residual(
    raw_residual: Array,
) -> Array:
    """
    Transform the raw [r_p,r_T] PDE residual into the selected metric.

    The parallel direction is aligned with alpha*p+beta*T.
    The perpendicular direction is beta*p-alpha*T.
    """
    if PT_RESIDUAL_MODE == "raw":
        return raw_residual

    if PT_RESIDUAL_MODE == "diffusion_whitened":
        return PT_DIFFUSION_INVERSE_SQRT @ raw_residual

    if PT_RESIDUAL_MODE == "generalized_physical_scaled":
        return PT_GENERALIZED_PHYSICAL_MATRIX @ raw_residual

    if PT_RESIDUAL_MODE in {
        "generalized_modal_balanced",
        "generalized_modal_scaled",
    }:
        transformed = (
            PT_GENERALIZED_EIGENVECTORS.T
            @ (PT_DIFFUSION_INVERSE_SQRT @ raw_residual)
        ) / PT_GENERALIZED_BALANCE
        if PT_RESIDUAL_MODE == "generalized_modal_scaled":
            transformed = PT_GENERALIZED_ETA * transformed
        return transformed

    residual_p = raw_residual[0]
    residual_t = raw_residual[1]

    residual_parallel = (
        ALPHA * residual_p + BETA * residual_t
    ) / PT_MODAL_NORM

    residual_perpendicular = (
        BETA * residual_p - ALPHA * residual_t
    ) / PT_MODAL_NORM

    if PT_RESIDUAL_MODE == "modal":
        return jnp.array(
            [
                residual_parallel,
                residual_perpendicular,
            ]
        )

    if PT_RESIDUAL_MODE == "modal_perp_scaled":
        return jnp.array(
            [
                residual_parallel,
                residual_perpendicular
                / PT_PERP_RESIDUAL_SCALE,
            ]
        )

    return jnp.array(
        [
            residual_parallel
            / PT_PARALLEL_RESIDUAL_SCALE,
            residual_perpendicular
            / PT_PERP_RESIDUAL_SCALE,
        ]
    )

@jit
def flow_heat_pde_residual_one(
    params_pt: Params,
    params_u_old: Params,
    xyt: Array,
) -> Array:
    """Evaluate the split four-field pressure-temperature PDE at one point.

    Current p,T are optimized while the previous outer iterate xi is frozen:
    C_eff @ (p_t,T_t) - (alpha,beta)*xi_old_t/lambda - D @ Laplacian(p,T)."""
    """GFS p/T subproblem residual at one point."""
    jac_q = PT_JAC(params_pt, xyt)
    hess_q = PT_HESS(params_pt, xyt)

    p_t, t_t = jac_q[0, 2], jac_q[1, 2]
    lap_p = hess_q[0, 0, 0] + hess_q[0, 1, 1]
    lap_t = hess_q[1, 0, 0] + hess_q[1, 1, 1]

    jac_uxi_old = U_JAC(params_u_old, xyt)
    xi_t_old = jac_uxi_old[2, 2]

    residual_p = (
        C_ALPHA * p_t
        + C_ALPHA_BETA * t_t
        - KAPPA * lap_p
        - ALPHA * INV_LAMBDA * xi_t_old
        - mass_source_exact(xyt)
    )

    residual_t = (
        C_ALPHA_BETA * p_t
        + C_BETA * t_t
        - THETA * lap_t
        - BETA * INV_LAMBDA * xi_t_old
        - heat_source_exact(xyt)
    )

    raw_residual = jnp.array(
        [
            residual_p,
            residual_t,
        ]
    )

    return transform_flow_heat_training_residual(raw_residual)


@jit
def coupled_flow_heat_pde_residual_one(
    params_pt: Params,
    params_u: Params,
    xyt: Array,
) -> Array:
    """Evaluate the original coupled flow-heat PDE using all current fields.

    No previous xi is frozen. This diagnostic checks convergence to the
    coupled equations rather than only to the current split subproblem."""
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

    jac_uxi = U_JAC(params_u, xyt)
    xi_t = jac_uxi[2, 2]

    sources = jnp.array(
        [mass_source_exact(xyt), heat_source_exact(xyt)]
    )

    return (
        FOUR_FIELD_STORAGE_MATRIX @ q_t
        - DIFFUSIVITY * lap_q
        + XI_COUPLING_VECTOR * xi_t
        - sources
    )


V_MECH_RES = jit(
    vmap(mechanics_pde_residual_one, in_axes=(None, None, 0))
)
V_FLOW_HEAT_RES = jit(
    vmap(
        flow_heat_pde_residual_one,
        in_axes=(None, None, 0),
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
V_XI_EXACT = jit(vmap(xi_exact))
V_UXI_EXACT = jit(vmap(uxi_exact))
V_PT_EXACT = jit(vmap(pt_exact))
V_U_EXACT_JAC = jit(vmap(U_EXACT_JAC))
V_UXI_EXACT_JAC = jit(vmap(UXI_EXACT_JAC))

# Manufactured data and geometry never change during training.  Materialize
# them once and reuse them in every GFS context and line-search candidate.
# This removes repeated high-order AD of the analytic solution from the hot
# path without changing any residual value.
V_BODY_FORCE_EXACT = jit(vmap(body_force_exact))
V_MASS_SOURCE_EXACT = jit(vmap(mass_source_exact))
V_HEAT_SOURCE_EXACT = jit(vmap(heat_source_exact))
V_OUTWARD_NORMAL = jit(vmap(outward_normal))


@jit
def manufactured_four_field_residual_one(xyt: Array) -> Array:
    """Five strong-form residuals evaluated on the manufactured fields."""
    jac_u = U_EXACT_JAC(xyt)
    hess_u = U_EXACT_HESS(xyt)
    grad_xi = XI_EXACT_GRAD(xyt)
    momentum = (
        four_field_mechanics_operator(hess_u, grad_xi[:2])
        - body_force_exact(xyt)
    )
    constraint = (
        jac_u[0, 0]
        + jac_u[1, 1]
        + INV_LAMBDA * xi_exact(xyt)
        - ALPHA * INV_LAMBDA * p_exact(xyt)
        - BETA * INV_LAMBDA * temperature_exact(xyt)
    )

    grad_p = P_EXACT_GRAD(xyt)
    grad_t = T_EXACT_GRAD(xyt)
    hess_p = P_EXACT_HESS(xyt)
    hess_t = T_EXACT_HESS(xyt)
    lap_p = hess_p[0, 0] + hess_p[1, 1]
    lap_t = hess_t[0, 0] + hess_t[1, 1]
    residual_p = (
        C_ALPHA * grad_p[2]
        + C_ALPHA_BETA * grad_t[2]
        - KAPPA * lap_p
        - ALPHA * INV_LAMBDA * grad_xi[2]
        - mass_source_exact(xyt)
    )
    residual_t = (
        C_ALPHA_BETA * grad_p[2]
        + C_BETA * grad_t[2]
        - THETA * lap_t
        - BETA * INV_LAMBDA * grad_xi[2]
        - heat_source_exact(xyt)
    )
    return jnp.concatenate(
        (momentum, jnp.array([constraint, residual_p, residual_t]))
    )


V_MANUFACTURED_FOUR_FIELD_RESIDUAL = jit(
    vmap(manufactured_four_field_residual_one)
)

BODY_FORCE_PDE = V_BODY_FORCE_EXACT(PDE_POINTS)
MASS_SOURCE_PDE = V_MASS_SOURCE_EXACT(PDE_POINTS)
HEAT_SOURCE_PDE = V_HEAT_SOURCE_EXACT(PDE_POINTS)
U_NEU_NORMALS = V_OUTWARD_NORMAL(U_NEU_POINTS)
U_NEU_EXACT_TRACTION = V_EXACT_TRACTION(U_NEU_POINTS)

PT_DIR_TARGET = V_PT_EXACT(PT_DIR_POINTS)
PT_IC_TARGET = V_PT_EXACT(IC_POINTS)
U_DIR_TARGET = V_U_EXACT(U_DIR_POINTS)
U_IC_TARGET = V_UXI_EXACT(IC_POINTS)


# =============================================================================
# 7. Weighted residual-vector construction
# =============================================================================

WEIGHTS_U: Dict[str, float] = {
    "pde": float(os.environ.get("THP_WEIGHT_U_PDE", "1.0")),
    "dirichlet": float(
        os.environ.get("THP_WEIGHT_U_DIRICHLET", "1.0")
    ),
    "neumann": float(
        os.environ.get("THP_WEIGHT_U_NEUMANN", "1.0")
    ),
    "ic": float(os.environ.get("THP_WEIGHT_U_IC", "1.0")),
}
WEIGHTS_PT: Dict[str, float] = {
    "pde": float(os.environ.get("THP_WEIGHT_PT_PDE", "1.0")),
    "dirichlet": float(
        os.environ.get("THP_WEIGHT_PT_DIRICHLET", "1.0")
    ),
    "ic": float(os.environ.get("THP_WEIGHT_PT_IC", "1.0")),
}

for field_name, field_weights in (
    ("WEIGHTS_U", WEIGHTS_U),
    ("WEIGHTS_PT", WEIGHTS_PT),
):
    for weight_name, weight_value in field_weights.items():
        if not math.isfinite(weight_value) or weight_value <= 0.0:
            raise ValueError(
                f"{field_name}[{weight_name!r}] must be finite and positive, "
                f"but received {weight_value}."
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
        V_PREDICT(params_u, U_DIR_POINTS)[:, :2]
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
) -> Array:
    params_pt = UNRAVEL_PT(flat_pt)
    params_u_old = UNRAVEL_U(flat_u_old)

    r_pde = V_FLOW_HEAT_RES(
        params_pt,
        params_u_old,
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
) -> Array:
    """Terms in the GFS p/T PDE that are frozen during one inner solve."""
    params_u_old = UNRAVEL_U(flat_u_old)
    jac_uxi_old = vmap(U_JAC, in_axes=(None, 0))(
        params_u_old, PDE_POINTS
    )
    xi_t_old = jac_uxi_old[:, 2, 2]

    frozen_p = (
        -ALPHA * INV_LAMBDA * xi_t_old
        - MASS_SOURCE_PDE
    )
    frozen_t = (
        -BETA * INV_LAMBDA * xi_t_old
        - HEAT_SOURCE_PDE
    )
    return jnp.stack([frozen_p, frozen_t], axis=1)


BUILD_PT_FROZEN_PDE_TERMS = jit(build_pt_frozen_pde_terms)


def build_u_frozen_terms(flat_pt: Array) -> Tuple[Array, Array]:
    """Current p/T contributions frozen during one mechanics inner solve."""
    params_pt = UNRAVEL_PT(flat_pt)

    pt_pde = V_PREDICT(params_pt, PDE_POINTS)
    constraint_frozen = (
        -ALPHA * INV_LAMBDA * pt_pde[:, 0]
        - BETA * INV_LAMBDA * pt_pde[:, 1]
    )
    pde_frozen = jnp.concatenate(
        (-BODY_FORCE_PDE, constraint_frozen[:, None]), axis=1
    )

    # In four-field variables sigma_total=2*mu*eps(u)-xi*I, so the
    # Neumann operator depends only on the (u,xi) network.
    frozen_neu = -U_NEU_EXACT_TRACTION
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

    raw_residual = jnp.array(
        [
            C_ALPHA * p_t
            + C_ALPHA_BETA * t_t
            - KAPPA * lap_p
            + frozen_term[0],

            C_ALPHA_BETA * p_t
            + C_BETA * t_t
            - THETA * lap_t
            + frozen_term[1],
        ]
    )

    return (
        SCALE_PT_PDE
        * transform_flow_heat_training_residual(raw_residual)
    )


def pt_dir_local_flat(flat_pt: Array, xyt: Array, target: Array) -> Array:
    return SCALE_PT_DIR * (predict(UNRAVEL_PT(flat_pt), xyt) - target)


def pt_ic_local_flat(flat_pt: Array, xyt: Array, target: Array) -> Array:
    return SCALE_PT_IC * (predict(UNRAVEL_PT(flat_pt), xyt) - target)


def elastic_traction_u_only_flat(flat_u: Array, xyt: Array) -> Array:
    '''Compute (2*mu*eps(u)-xi*I)n at one point.'''
    params_u = UNRAVEL_U(flat_u)
    jac_u = U_JAC(params_u, xyt)
    xi_value = predict(params_u, xyt)[2]
    du1_dx, du1_dy = jac_u[0, 0], jac_u[0, 1]
    du2_dx, du2_dy = jac_u[1, 0], jac_u[1, 1]

    sigma11 = 2.0 * MU * du1_dx - xi_value
    sigma22 = 2.0 * MU * du2_dy - xi_value
    sigma12 = MU * (du1_dy + du2_dx)
    sigma = jnp.array([[sigma11, sigma12], [sigma12, sigma22]])
    return sigma @ outward_normal(xyt)


def u_pde_local_flat(
    flat_u: Array,
    xyt: Array,
    frozen_term: Array,
) -> Array:
    params_u = UNRAVEL_U(flat_u)
    uxi = predict(params_u, xyt)
    jac_u = U_JAC(params_u, xyt)
    hess_u = U_HESS(params_u, xyt)
    momentum = four_field_mechanics_operator(
        hess_u[:2], jac_u[2, :2]
    )
    constraint_u_xi = (
        jac_u[0, 0] + jac_u[1, 1] + INV_LAMBDA * uxi[2]
    )
    return SCALE_U_PDE * (
        jnp.concatenate((momentum, jnp.array([constraint_u_xi])))
        + frozen_term
    )


def u_dir_local_flat(flat_u: Array, xyt: Array, target: Array) -> Array:
    return SCALE_U_DIR * (
        predict(UNRAVEL_U(flat_u), xyt)[:2] - target
    )


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
    block_size: int,
) -> PointwisePTContext:
    """Build pressure-temperature blocks from the frozen previous mechanics state."""
    frozen = BUILD_PT_FROZEN_PDE_TERMS(flat_u_old)                  
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
# Rank-safe is deliberately mechanics-only and opt-in.  A positive safety
# factor gamma replaces the fixed mechanics ridge mu_0 by
# max(mu_0, gamma * eps * P_u * max(lambda_max(G_u), 1)).  Keeping the default
# at zero sends Group 1--4 through the historical fixed-ridge path.
U_RANK_SAFE_RIDGE_SAFETY = float(
    os.environ.get("THP_U_RANK_SAFE_RIDGE_SAFETY", "0.0")
)
U_RANK_SAFE_RIDGE_ACTIVE = U_RANK_SAFE_RIDGE_SAFETY > 0.0
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
if (
    not math.isfinite(U_RANK_SAFE_RIDGE_SAFETY)
    or U_RANK_SAFE_RIDGE_SAFETY < 0.0
):
    raise ValueError(
        "THP_U_RANK_SAFE_RIDGE_SAFETY must be finite and non-negative."
    )
if U_RANK_SAFE_RIDGE_ACTIVE and ENGD_SOLVER not in {
    "eigh-refined",
    "eigh",
}:
    raise ValueError(
        "The mechanics rank-safe ridge requires THP_ENGD_SOLVER=eigh-refined "
        "or eigh."
    )
if U_RANK_SAFE_RIDGE_ACTIVE and not (
    IS_GROUP5_RUN and PT_RESIDUAL_MODE == "generalized_physical_scaled"
):
    raise ValueError(
        "THP_U_RANK_SAFE_RIDGE_SAFETY is a controlled final Group-5 "
        "option. Set THP_GROUP5_CASE_ID, THP_GROUP5_VARIANT, and "
        "THP_PT_RESIDUAL_MODE=generalized_physical_scaled, or leave the "
        "safety factor at its default zero."
    )

# Final Group-5 extension: two mirror points where both diffusivities are
# genuinely small but unequal.  This is a configuration-only experiment: the
# exact fields, residual definitions, Gram assembly, EIGH solve, and line
# search functions above are unchanged.  The request detector also catches a
# partially specified S10/S11 command and fails before expensive compilation.
_DOUBLE_SMALL_CASES = {
    "S10": (1.0e-6, 2.0e-6),
    "S11": (2.0e-6, 1.0e-6),
}
_DOUBLE_SMALL_RAW_VARIANT = "raw_double_small_unweighted"
_DOUBLE_SMALL_PHYSICAL_VARIANT = (
    "physical_ranksafe_double_small_unweighted"
)
_DOUBLE_SMALL_MODAL_VARIANT = "modal_both_double_small_unweighted"
_DOUBLE_SMALL_VARIANTS = {
    _DOUBLE_SMALL_RAW_VARIANT,
    _DOUBLE_SMALL_PHYSICAL_VARIANT,
    _DOUBLE_SMALL_MODAL_VARIANT,
}
DOUBLE_SMALL_EXTENSION_REQUESTED = (
    GROUP5_CASE_ID in _DOUBLE_SMALL_CASES
    or GROUP5_VARIANT in _DOUBLE_SMALL_VARIANTS
)
DOUBLE_SMALL_EXTENSION_ACTIVE = False
if DOUBLE_SMALL_EXTENSION_REQUESTED:
    if not (
        IS_GROUP5_RUN
        and GROUP5_CASE_ID in _DOUBLE_SMALL_CASES
        and GROUP5_VARIANT in _DOUBLE_SMALL_VARIANTS
    ):
        raise ValueError(
            "The unequal-double-small extension requires a matched Group-5 "
            "S10/S11 case id and dedicated variant."
        )
    expected_kappa, expected_theta = _DOUBLE_SMALL_CASES[GROUP5_CASE_ID]
    if not (
        math.isclose(KAPPA, expected_kappa, rel_tol=0.0, abs_tol=1.0e-18)
        and math.isclose(
            THETA, expected_theta, rel_tol=0.0, abs_tol=1.0e-18
        )
    ):
        raise ValueError(
            f"{GROUP5_CASE_ID} requires K={expected_kappa:.1e} and "
            f"Theta={expected_theta:.1e}."
        )
    if EXACT_CASE != "distinct":
        raise ValueError(
            "The unequal-double-small extension keeps the historical "
            "distinct manufactured solution."
        )
    if not (
        math.isclose(NU, 0.3, rel_tol=0.0, abs_tol=1.0e-15)
        and math.isclose(ALPHA, 0.1, rel_tol=0.0, abs_tol=1.0e-15)
        and math.isclose(BETA, 0.1, rel_tol=0.0, abs_tol=1.0e-15)
        and A0 == 0.0
        and B0 == 0.0
        and C0 == 0.0
    ):
        raise ValueError(
            "S10/S11 require nu=0.3, alpha=beta=0.1, and a0=b0=c0=0."
        )
    if any(
        value != 1.0
        for value in [*WEIGHTS_U.values(), *WEIGHTS_PT.values()]
    ):
        raise ValueError("S10/S11 fix every external loss weight to 1.")
    if not FAST_DEBUG and (
        FS_ITERATIONS not in {50, 100}
        or 50 not in FIXED_BUDGET_CHECKPOINTS
        or (FS_ITERATIONS == 100 and 100 not in FIXED_BUDGET_CHECKPOINTS)
        or ENABLE_FS_EARLY_STOP
    ):
        raise ValueError(
            "Historical S10/S11 (paper S8/S9) runs require FS50 or FS100, retained FS50, and no "
            "adaptive outer stopping."
        )
    if not math.isclose(
        GRAM_RIDGE, 1.0e-7, rel_tol=0.0, abs_tol=1.0e-20
    ):
        raise ValueError("S10/S11 fix the base Gram ridge at 1e-7.")
    if GROUP5_VARIANT == _DOUBLE_SMALL_RAW_VARIANT:
        if PT_RESIDUAL_MODE != "raw" or U_RANK_SAFE_RIDGE_ACTIVE:
            raise ValueError(
                "The S10/S11 raw variant requires raw p/T residuals and the "
                "historical fixed mechanics ridge."
            )
    elif GROUP5_VARIANT == _DOUBLE_SMALL_MODAL_VARIANT:
        if not (
            PT_RESIDUAL_MODE == "modal_both_scaled"
            and not U_RANK_SAFE_RIDGE_ACTIVE
            and math.isclose(
                PT_PARALLEL_RESIDUAL_SCALE,
                PT_PARALLEL_STORAGE,
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            )
            and math.isclose(
                PT_PERP_RESIDUAL_SCALE,
                PT_MODAL_PERP_DIFFUSION,
                rel_tol=1.0e-12,
                abs_tol=1.0e-18,
            )
        ):
            raise ValueError(
                "The S10/S11 modal variant requires modal_both_scaled, "
                "parallel_scale=sigma_parallel, perpendicular_scale=d_perp, "
                "and the historical fixed mechanics ridge."
            )
    else:
        if not (
            PT_RESIDUAL_MODE == "generalized_physical_scaled"
            and U_RANK_SAFE_RIDGE_ACTIVE
            and math.isclose(
                U_RANK_SAFE_RIDGE_SAFETY,
                3.0,
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
        ):
            raise ValueError(
                "The S10/S11 physical variant requires "
                "generalized_physical_scaled and mechanics rank-safe "
                "safety=3."
            )
    DOUBLE_SMALL_EXTENSION_ACTIVE = True

# Final S7 raw supplement.  The dedicated variant name prevents this stricter
# FS100 protocol from changing any historical S7/Group-1--4 command.  It uses
# exactly the old raw objective and fixed-ridge mechanics path; only the fixed
# budget is extended to 100 with the FS50 checkpoint retained.
_S7_RAW_FS100_VARIANT = "raw_fs100_final"
S7_RAW_FS100_ACTIVE = GROUP5_VARIANT == _S7_RAW_FS100_VARIANT
if S7_RAW_FS100_ACTIVE:
    if not (IS_GROUP5_RUN and GROUP5_CASE_ID == "S7"):
        raise ValueError(
            "The raw_fs100_final variant is reserved for Group-5 S7."
        )
    if EXACT_CASE != "distinct":
        raise ValueError("Final S7 keeps the historical distinct exact fields.")
    if not (
        math.isclose(NU, 0.3, rel_tol=0.0, abs_tol=1.0e-15)
        and math.isclose(ALPHA, 0.1, rel_tol=0.0, abs_tol=1.0e-15)
        and math.isclose(BETA, 0.1, rel_tol=0.0, abs_tol=1.0e-15)
        and A0 == 0.0
        and B0 == 0.0
        and C0 == 0.0
        and math.isclose(KAPPA, 1.0e-6, rel_tol=0.0, abs_tol=1.0e-18)
        and math.isclose(THETA, 1.0e-6, rel_tol=0.0, abs_tol=1.0e-18)
    ):
        raise ValueError(
            "Final S7 requires nu=0.3, alpha=beta=0.1, a0=b0=c0=0, "
            "and K=Theta=1e-6."
        )
    if PT_RESIDUAL_MODE != "raw" or U_RANK_SAFE_RIDGE_ACTIVE:
        raise ValueError(
            "Final S7 raw requires raw p/T residuals and the historical "
            "fixed mechanics ridge."
        )
    if any(
        value != 1.0
        for value in [*WEIGHTS_U.values(), *WEIGHTS_PT.values()]
    ):
        raise ValueError("Final S7 raw fixes every external loss weight to 1.")
    if not FAST_DEBUG and (
        FS_ITERATIONS != 100
        or 50 not in FIXED_BUDGET_CHECKPOINTS
        or 100 not in FIXED_BUDGET_CHECKPOINTS
        or ENABLE_FS_EARLY_STOP
    ):
        raise ValueError(
            "Production S7 raw supplement requires FS100, retained FS50, "
            "and no adaptive outer stopping."
        )
    if not math.isclose(
        GRAM_RIDGE, 1.0e-7, rel_tol=0.0, abs_tol=1.0e-20
    ):
        raise ValueError("Final S7 raw fixes the Gram ridge at 1e-7.")
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
    """Immutable configuration; zero rank_safe_ridge_safety retains fixed damping."""
    line_search_levels: int = 31           
    gram_ridge: float = 0.0                
    rank_safe_ridge_safety: float = 0.0    # gamma=0 keeps the fixed-ridge path
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
    applied_gram_ridge: float   
    rank_safe_fp_floor: float   # eps*P*max(lambda_max(G),1)
    ridge_to_rank_safe_floor: float # applied ridge / rank-safe FP floor
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
    applied_gram_ridge: float
    rank_safe_fp_floor: float
    ridge_to_rank_safe_floor: float

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
) -> Array:
    """Evaluate all p/T grid-search candidates in one compiled vmap."""
    """Evaluate uncached pressure-temperature residual losses at all candidates."""
    return vmap(
        lambda candidate: half_squared_norm(
            residual_pt_flat(candidate, flat_u_old)
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
    base_damping: float,
    rank_safe_ridge_safety: float,
    relative_cutoff: float,
    refinement_steps: int,
) -> Tuple[Array, ...]:
    """Fixed-shape EIGH pseudoinverse + residual refinement."""
    gram = 0.5 * (gram + gram.T)                            
    eigenvalues, eigenvectors = jnp.linalg.eigh(gram)       
    base_damping_array = jnp.asarray(
        base_damping, dtype=gram.dtype
    )
    rank_safe_safety_array = jnp.asarray(
        rank_safe_ridge_safety, dtype=gram.dtype
    )
    unshifted_spectral_scale = jnp.maximum(
        jnp.max(eigenvalues),
        jnp.array(1.0, dtype=eigenvalues.dtype),
    )
    rank_safe_fp_floor = (
        jnp.finfo(eigenvalues.dtype).eps
        * eigenvalues.shape[0]
        * unshifted_spectral_scale
    )
    damping_array = jnp.maximum(
        base_damping_array,
        rank_safe_safety_array * rank_safe_fp_floor,
    )
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
    ridge_to_rank_safe_floor = damping_array / jnp.maximum(
        rank_safe_fp_floor,
        jnp.array(1.0e-300, dtype=rank_safe_fp_floor.dtype),
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
        damping_array,
        rank_safe_fp_floor,
        ridge_to_rank_safe_floor,
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
    rank_safe_ridge_safety: float,
    relative_cutoff: float,
    refinement_steps: int,
) -> DenseSolveResult:
    if damping < 0.0:
        raise ValueError("ENGD damping/ridge must be non-negative.")
    if relative_cutoff < 0.0:
        raise ValueError("Relative EIGH cutoff must be non-negative.")
    if rank_safe_ridge_safety < 0.0:
        raise ValueError("Rank-safe ridge safety must be non-negative.")
    if refinement_steps < 0:
        raise ValueError("Refinement steps must be non-negative.")

    outputs = _EIGH_REFINED_KERNEL(
        gram,
        gradient_flat,
        damping,
        rank_safe_ridge_safety,
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
        applied_gram_ridge,
        rank_safe_fp_floor,
        ridge_to_rank_safe_floor,
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
            applied_gram_ridge,
            rank_safe_fp_floor,
            ridge_to_rank_safe_floor,
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
        applied_gram_ridge,
        rank_safe_fp_floor,
        ridge_to_rank_safe_floor,
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
        applied_gram_ridge=float(applied_gram_ridge),
        rank_safe_fp_floor=float(rank_safe_fp_floor),
        ridge_to_rank_safe_floor=float(ridge_to_rank_safe_floor),
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
        applied_gram_ridge=float(damping),
        rank_safe_fp_floor=math.nan,
        ridge_to_rank_safe_floor=math.nan,
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
            applied_gram_ridge=config.gram_ridge,
            rank_safe_fp_floor=nan,
            ridge_to_rank_safe_floor=nan,
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
            rank_safe_ridge_safety=config.rank_safe_ridge_safety,
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
            damping=solve_result.applied_gram_ridge,
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
        applied_gram_ridge=solve_result.applied_gram_ridge,
        rank_safe_fp_floor=solve_result.rank_safe_fp_floor,
        ridge_to_rank_safe_floor=(
            solve_result.ridge_to_rank_safe_floor
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
    rank_safe_ridge_safety=0.0,
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
    rank_safe_ridge_safety=U_RANK_SAFE_RIDGE_SAFETY,
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
    stats_values = asdict(stats)
    # Preserve the historical Group 1--4 profile schema.  These fields are
    # emitted only for the explicitly enabled Rank-safe experiment.
    if not U_RANK_SAFE_RIDGE_ACTIVE:
        for field_name in (
            "applied_gram_ridge",
            "rank_safe_fp_floor",
            "ridge_to_rank_safe_floor",
        ):
            stats_values.pop(field_name)
    row.update(stats_values)
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
        V_PREDICT(params_u, U_DIR_POINTS)[:, :2]
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
) -> Dict[str, float]:
    params_pt = UNRAVEL_PT(flat_pt)
    params_u_old = UNRAVEL_U(flat_u_old)

    r_pde = V_FLOW_HEAT_RES(
        params_pt,
        params_u_old,
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

    uxi_prediction = V_PREDICT(params_u, EVAL_POINTS)
    u_prediction = uxi_prediction[:, :2]
    xi_prediction = uxi_prediction[:, 2]
    pt_prediction = V_PREDICT(params_pt, EVAL_POINTS)
    u_reference = V_U_EXACT(EVAL_POINTS)
    xi_reference = V_XI_EXACT(EVAL_POINTS)
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
    xi_l2, xi_rel_l2 = absolute_relative_l2_scalar(
        xi_prediction, xi_reference
    )
    p_l2, p_rel_l2 = absolute_relative_l2_scalar(
        pt_prediction[:, 0], pt_reference[:, 0]
    )
    t_l2, t_rel_l2 = absolute_relative_l2_scalar(
        pt_prediction[:, 1], pt_reference[:, 1]
    )
    modal_errors: Dict[str, float] = {}
    if REPORT_PT_MODAL_DIAGNOSTICS:
        q_parallel_prediction = (
            ALPHA * pt_prediction[:, 0] + BETA * pt_prediction[:, 1]
        ) / PT_MODAL_NORM
        q_parallel_reference = (
            ALPHA * pt_reference[:, 0] + BETA * pt_reference[:, 1]
        ) / PT_MODAL_NORM
        q_perp_prediction = (
            BETA * pt_prediction[:, 0] - ALPHA * pt_prediction[:, 1]
        ) / PT_MODAL_NORM
        q_perp_reference = (
            BETA * pt_reference[:, 0] - ALPHA * pt_reference[:, 1]
        ) / PT_MODAL_NORM
        q_parallel_l2, q_parallel_rel_l2 = absolute_relative_l2_scalar(
            q_parallel_prediction, q_parallel_reference
        )
        q_perp_l2, q_perp_rel_l2 = absolute_relative_l2_scalar(
            q_perp_prediction, q_perp_reference
        )
        modal_errors = {
            "q_parallel_l2": q_parallel_l2,
            "q_parallel_rel_l2": q_parallel_rel_l2,
            "q_perp_l2": q_perp_l2,
            "q_perp_rel_l2": q_perp_rel_l2,
        }

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
                    u_jac_prediction[:, :2, :2]
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
    xi_grad_reference = vmap(XI_EXACT_GRAD)(EVAL_POINTS)
    rank_safe_errors: Dict[str, float] = {}
    if U_RANK_SAFE_RIDGE_ACTIVE:
        xi_t_l2, xi_t_rel_l2 = absolute_relative_l2_scalar(
            u_jac_prediction[:, 2, 2],
            xi_grad_reference[:, 2],
        )
        rank_safe_errors = {
            "xi_t_l2": xi_t_l2,
            "xi_t_rel_l2": xi_t_rel_l2,
        }
    xi_grad_error_sq = MEASURE_PDE * float(
        jnp.mean(
            jnp.sum(
                (
                    u_jac_prediction[:, 2, :2]
                    - xi_grad_reference[:, :2]
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

    errors = {
        "u_l2": u_l2,
        "u_rel_l2": u_rel_l2,
        "u_h1": math.sqrt(u_l2 * u_l2 + u_grad_error_sq),
        "u1_l2": u1_l2,
        "u1_rel_l2": u1_rel_l2,
        "u1_h1": math.sqrt(u1_l2 * u1_l2 + u1_grad_error_sq),
        "u2_l2": u2_l2,
        "u2_rel_l2": u2_rel_l2,
        "u2_h1": math.sqrt(u2_l2 * u2_l2 + u2_grad_error_sq),
        "xi_l2": xi_l2,
        "xi_rel_l2": xi_rel_l2,
        "xi_h1": math.sqrt(xi_l2 * xi_l2 + xi_grad_error_sq),
        "p_l2": p_l2,
        "p_rel_l2": p_rel_l2,
        "p_h1": math.sqrt(p_l2 * p_l2 + p_grad_error_sq),
        "T_l2": t_l2,
        "T_rel_l2": t_rel_l2,
        "T_h1": math.sqrt(t_l2 * t_l2 + t_grad_error_sq),
    }
    errors.update(modal_errors)
    errors.update(rank_safe_errors)
    return errors

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
        self.xi_rel_l2: List[float] = []
        if U_RANK_SAFE_RIDGE_ACTIVE:
            self.xi_t_rel_l2: List[float] = []
        self.p_rel_l2: List[float] = []
        self.T_rel_l2: List[float] = []
        if REPORT_PT_MODAL_DIAGNOSTICS:
            self.q_parallel_rel_l2: List[float] = []
            self.q_perp_rel_l2: List[float] = []
        self.u_mean_rel_l2: List[float] = []
        self.pT_mean_rel_l2: List[float] = []
        self.common_rel_l2_max: List[float] = []
        self.four_field_component_rel_l2_max: List[float] = []
        self.physical_rel_l2_max: List[float] = []
        self.four_field_rel_l2_max: List[float] = []
        self.u_h1: List[float] = []
        self.u1_h1: List[float] = []
        self.u2_h1: List[float] = []
        self.xi_h1: List[float] = []
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

    residual_pt_fn = lambda z: residual_pt_flat(z, flat_u_old)
    jacobian_pt_fn = lambda z: JACOBIAN_PT_FLAT(z, flat_u_old)
    candidate_pt_fn = lambda candidates: candidate_losses_pt(
        candidates, flat_u_old
    )
    gram_pt_fn = None
    if JACOBIAN_BACKEND == "pointwise":
        context_pt = build_pointwise_pt_context(
            flat_u_old, POINT_BLOCK_PT
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
        context_u = build_pointwise_u_context(flat_pt, POINT_BLOCK_U)
        pde_frozen_u = context_u.pde_frozen.reshape(-1, 3)[
            : PDE_POINTS.shape[0]
        ]
        neu_frozen_u = context_u.neu_frozen.reshape(-1, 2)[
            : U_NEU_POINTS.shape[0]
        ]
        if CACHE_LINE_SEARCH:
            candidate_u_fn = lambda candidates: candidate_losses_u_cached(
                candidates, pde_frozen_u, neu_frozen_u
            )
        gram_u_fn = lambda z: pointwise_u_gram_gradient(z, context_u)

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
    flow_heat_loss_breakdown(flat_pt, flat_u_old)
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
        "Four-field THP (u,xi,p,T): iterative decoupling + streamed ENGD"
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
        "four-field storage=[[%.6e, %.6e], [%.6e, %.6e]]",
        C_ALPHA,
        C_ALPHA_BETA,
        C_ALPHA_BETA,
        C_BETA,
    )
    LOGGER.info(
        "p/T PDE residual metric: mode=%s, "
        "parallel_storage=%.6e, modal_storage_coupling=%.6e, "
        "parallel_scale=%.6e, "
        "perpendicular_storage=%.6e, perpendicular_scale=%.6e",
        PT_RESIDUAL_MODE,
        PT_PARALLEL_STORAGE,
        PT_MODAL_STORAGE_COUPLING,
        PT_PARALLEL_RESIDUAL_SCALE,
        PT_PERP_STORAGE,
        PT_PERP_RESIDUAL_SCALE,
    )
    if REPORT_PT_MODAL_DIAGNOSTICS:
        LOGGER.info(
            "generalized p/T metric: eta=%.6e, sigma=[%.6e, %.6e], "
            "balance=[%.6e, %.6e]",
            PT_GENERALIZED_ETA,
            float(_pt_generalized_eigenvalues[0]),
            float(_pt_generalized_eigenvalues[1]),
            float(_pt_generalized_balance[0]),
            float(_pt_generalized_balance[1]),
        )
    if PT_RESIDUAL_MODE == "generalized_physical_scaled":
        LOGGER.info(
            "physical pullback (eta unused): "
            "W=[[%.6e, %.6e], [%.6e, %.6e]], "
            "eig=[%.6e, %.6e], cond=%.6e",
            float(_PT_GENERALIZED_PHYSICAL_MATRIX_NP[0, 0]),
            float(_PT_GENERALIZED_PHYSICAL_MATRIX_NP[0, 1]),
            float(_PT_GENERALIZED_PHYSICAL_MATRIX_NP[1, 0]),
            float(_PT_GENERALIZED_PHYSICAL_MATRIX_NP[1, 1]),
            float(_pt_generalized_physical_eigenvalues[0]),
            float(_pt_generalized_physical_eigenvalues[1]),
            _pt_generalized_physical_condition,
        )
    LOGGER.info(
        "loss weights: "
        "u[pde=%.6e, dirichlet=%.6e, neumann=%.6e, ic=%.6e]; "
        "pT[pde=%.6e, dirichlet=%.6e, ic=%.6e]",
        WEIGHTS_U["pde"],
        WEIGHTS_U["dirichlet"],
        WEIGHTS_U["neumann"],
        WEIGHTS_U["ic"],
        WEIGHTS_PT["pde"],
        WEIGHTS_PT["dirichlet"],
        WEIGHTS_PT["ic"],
    )
    LOGGER.info(
        "weighted residual block scales sqrt(weight*measure/N): "
        "u[pde=%.6e, dirichlet=%.6e, neumann=%.6e, ic=%.6e]; "
        "pT[pde=%.6e, dirichlet=%.6e, ic=%.6e]",
        SCALE_U_PDE,
        SCALE_U_DIR,
        SCALE_U_NEU,
        SCALE_U_IC,
        SCALE_PT_PDE,
        SCALE_PT_DIR,
        SCALE_PT_IC,
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
    LOGGER.info(
        "parameters: uXi=%d, pT=%d", flat_u.size, flat_pt.size
    )
    LOGGER.info(
        "outer iterations=%d, ENGD steps pT=%d, ENGD steps uXi=%d",
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
    if DOUBLE_SMALL_EXTENSION_ACTIVE:
        LOGGER.info(
            "Group-5 unequal-double-small extension: case=%s; "
            "K=%.6e Theta=%.6e; variant=%s; exact_case=distinct; "
            "paper FS50 evaluation with the configured training budget",
            GROUP5_CASE_ID,
            KAPPA,
            THETA,
            GROUP5_VARIANT,
        )
    if S7_RAW_FS100_ACTIVE:
        LOGGER.info(
            "Group-5 S7 raw FS100 supplement: K=Theta=1e-6; "
            "unit weights; fixed mechanics ridge; FS50 retained"
        )
    if U_RANK_SAFE_RIDGE_ACTIVE:
        LOGGER.info(
            "Final Group-5 mechanics-only rank-safe ridge enabled; "
            "mu_u=max(%.3e, %.6g*eps*P_u*max(lambda_max(G_u),1)); "
            "pT ridge remains fixed at %.3e",
            GRAM_RIDGE,
            U_RANK_SAFE_RIDGE_SAFETY,
            GRAM_RIDGE,
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
        "accuracy reporting: E_common=max(u1,u2,p,T RelL2), "
        "E_4F=max(u1,u2,xi,p,T RelL2); thresholds=%s; "
        "TTA patience=%d completed sweeps; fixed-budget checkpoints=%s",
        TIME_TO_ACCURACY_THRESHOLDS,
        TTA_PATIENCE,
        FIXED_BUDGET_CHECKPOINTS,
    )

    if VERIFY_MANUFACTURED:
        exact_residual = V_MANUFACTURED_FOUR_FIELD_RESIDUAL(PDE_POINTS)
        exact_residual.block_until_ready()
        exact_residual_max = float(jnp.max(jnp.abs(exact_residual)))
        LOGGER.info(
            "manufactured four-field strong residual max|r|=%.3e",
            exact_residual_max,
        )
        if not math.isfinite(exact_residual_max) or exact_residual_max > 1.0e-7:
            raise RuntimeError(
                "Manufactured four-field residual check failed: "
                f"max|r|={exact_residual_max:.3e}."
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
            "--- four-field outer iteration %d/%d ---",
            fs_iteration,
            FS_ITERATIONS,
        )

        # ------------------------------------------------------------------
        # Step 1: joint flow-heat subproblem by exact ENGD
        # ------------------------------------------------------------------
        residual_pt_fn = lambda z: residual_pt_flat(z, flat_u_old)
        jacobian_pt_fn = lambda z: JACOBIAN_PT_FLAT(z, flat_u_old)
        candidate_pt_fn = lambda candidates: candidate_losses_pt(
            candidates, flat_u_old
        )
        gram_pt_fn = None
        if JACOBIAN_BACKEND == "pointwise":
            context_start = time.perf_counter()
            pt_pointwise_context = build_pointwise_pt_context(
                flat_u_old, POINT_BLOCK_PT
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
            pde_frozen_u = u_pointwise_context.pde_frozen.reshape(-1, 3)[
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
            if U_RANK_SAFE_RIDGE_ACTIVE:
                LOGGER.info(
                    "u  Rank-safe ridge: applied=%.6e fp_floor=%.6e "
                    "ratio=%.6f",
                    stats.applied_gram_ridge,
                    stats.rank_safe_fp_floor,
                    stats.ridge_to_rank_safe_floor,
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
            flat_pt, flat_u_old
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
        physical_rel_l2_max = max(
            errors["u_rel_l2"],
            errors["p_rel_l2"],
            errors["T_rel_l2"],
        )
        four_field_rel_l2_max = max(
            physical_rel_l2_max,
            errors["xi_rel_l2"],
        )
        u_mean_rel_l2 = 0.5 * (
            errors["u1_rel_l2"] + errors["u2_rel_l2"]
        )
        pT_mean_rel_l2 = 0.5 * (
            errors["p_rel_l2"] + errors["T_rel_l2"]
        )
        common_rel_l2_max = max(
            errors["u1_rel_l2"],
            errors["u2_rel_l2"],
            errors["p_rel_l2"],
            errors["T_rel_l2"],
        )
        four_field_component_rel_l2_max = max(
            common_rel_l2_max,
            errors["xi_rel_l2"],
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
            "xi RelL2=%.4e H1=%.4e | p RelL2=%.4e H1=%.4e | "
            "T RelL2=%.4e H1=%.4e",
            fs_iteration,
            errors["u_rel_l2"],
            errors["u_h1"],
            errors["xi_rel_l2"],
            errors["xi_h1"],
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
        if REPORT_PT_MODAL_DIAGNOSTICS:
            LOGGER.info(
                "FS %02d storage-aligned modes: "
                "q_parallel RelL2=%.4e | q_perp RelL2=%.4e",
                fs_iteration,
                errors["q_parallel_rel_l2"],
                errors["q_perp_rel_l2"],
            )
        if U_RANK_SAFE_RIDGE_ACTIVE:
            LOGGER.info(
                "FS %02d Rank-safe transfer diagnostic: "
                "xi_t L2=%.4e RelL2=%.4e",
                fs_iteration,
                errors["xi_t_l2"],
                errors["xi_t_rel_l2"],
            )
        LOGGER.info(
            "FS %02d comparison metrics: u_mean=%.4e pT_mean=%.4e "
            "E_common=%.4e E_4F=%.4e",
            fs_iteration,
            u_mean_rel_l2,
            pT_mean_rel_l2,
            common_rel_l2_max,
            four_field_component_rel_l2_max,
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
        HISTORY.xi_rel_l2.append(errors["xi_rel_l2"])
        if U_RANK_SAFE_RIDGE_ACTIVE:
            HISTORY.xi_t_rel_l2.append(errors["xi_t_rel_l2"])
        HISTORY.p_rel_l2.append(errors["p_rel_l2"])
        HISTORY.T_rel_l2.append(errors["T_rel_l2"])
        if REPORT_PT_MODAL_DIAGNOSTICS:
            HISTORY.q_parallel_rel_l2.append(
                errors["q_parallel_rel_l2"]
            )
            HISTORY.q_perp_rel_l2.append(errors["q_perp_rel_l2"])
        HISTORY.u_mean_rel_l2.append(u_mean_rel_l2)
        HISTORY.pT_mean_rel_l2.append(pT_mean_rel_l2)
        HISTORY.common_rel_l2_max.append(common_rel_l2_max)
        HISTORY.four_field_component_rel_l2_max.append(
            four_field_component_rel_l2_max
        )
        HISTORY.physical_rel_l2_max.append(physical_rel_l2_max)
        HISTORY.four_field_rel_l2_max.append(four_field_rel_l2_max)
        HISTORY.u_h1.append(errors["u_h1"])
        HISTORY.u1_h1.append(errors["u1_h1"])
        HISTORY.u2_h1.append(errors["u2_h1"])
        HISTORY.xi_h1.append(errors["xi_h1"])
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

    uxi_prediction = np.asarray(V_PREDICT(params_u, points))
    pt_prediction = np.asarray(V_PREDICT(params_pt, points))
    uxi_reference = np.asarray(V_UXI_EXACT(points))
    pt_reference = np.asarray(V_PT_EXACT(points))

    predicted = [
        uxi_prediction[:, 0],
        uxi_prediction[:, 1],
        uxi_prediction[:, 2],
        pt_prediction[:, 0],
        pt_prediction[:, 1],
    ]
    exact = [
        uxi_reference[:, 0],
        uxi_reference[:, 1],
        uxi_reference[:, 2],
        pt_reference[:, 0],
        pt_reference[:, 1],
    ]
    names = [r"$u_1$", r"$u_2$", r"$\xi$", r"$p$", r"$T$"]

    fig, axes = plt.subplots(3, 5, figsize=(23, 12))
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
        iterations, HISTORY.xi_rel_l2, marker="D", label=r"$\xi$ Rel. $L^2$"
    )
    ax_error.semilogy(
        iterations, HISTORY.p_rel_l2, marker="s", label=r"$p$ Rel. $L^2$"
    )
    ax_error.semilogy(
        iterations, HISTORY.T_rel_l2, marker="^", label=r"$T$ Rel. $L^2$"
    )
    ax_error.set_xlabel("Four-field outer iteration")
    ax_error.set_ylabel("Relative error")
    ax_error.grid(True, which="both", alpha=0.3)
    ax_error.legend()
    save_figure(fig_error, "history_relative_l2")
    plt.close(fig_error)

    fig_loss, ax_loss = plt.subplots(figsize=(9, 6))
    ax_loss.semilogy(
        iterations, HISTORY.loss_u, marker="o", label="mechanics loss"
    )
    ax_loss.semilogy(
        iterations, HISTORY.loss_pt, marker="s", label="split flow-heat loss"
    )
    ax_loss.semilogy(
        iterations,
        HISTORY.loss_pt_coupled_pde,
        marker="^",
        label="original coupled pT PDE loss",
    )
    ax_loss.set_xlabel("Four-field outer iteration")
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
    ax_change.set_xlabel("Four-field outer iteration")
    ax_change.set_ylabel("Relative prediction change")
    ax_change.grid(True, which="both", alpha=0.3)
    ax_change.legend()
    save_figure(fig_change, "history_outer_change")
    plt.close(fig_change)

    fig_time, ax_time = plt.subplots(figsize=(9, 6))
    ax_time.semilogy(
        HISTORY.training_time,
        HISTORY.common_rel_l2_max,
        marker="o",
        label=r"$E_{\mathrm{common}}=\max(E_{u_1},E_{u_2},E_p,E_T)$",
    )
    ax_time.semilogy(
        HISTORY.training_time,
        HISTORY.four_field_component_rel_l2_max,
        marker="D",
        label=r"$E_{4F}=\max(E_{u_1},E_{u_2},E_\xi,E_p,E_T)$",
    )
    for threshold in TIME_TO_ACCURACY_THRESHOLDS:
        ax_time.axhline(
            threshold,
            color="gray",
            linewidth=0.6,
            alpha=0.25,
        )
    ax_time.set_xlabel("Effective training time (s)")
    ax_time.set_ylabel("Maximum relative $L^2$ error")
    ax_time.grid(True, which="both", alpha=0.3)
    ax_time.legend()
    save_figure(fig_time, "history_accuracy_vs_time")
    plt.close(fig_time)


def save_checkpoint(flat_u: Array, flat_pt: Array) -> None:
    np.savez(
        RESULT_DIR / "final_model.npz",
        flat_uxi=np.asarray(flat_u),
        flat_u=np.asarray(flat_u),
        flat_pt=np.asarray(flat_pt),
        net_uxi_sizes=np.asarray(NET_U_SIZES),
        net_u_sizes=np.asarray(NET_U_SIZES),
        net_pt_sizes=np.asarray(NET_PT_SIZES),
    )


def save_history() -> None:
    history_arrays = {
        "fs_iteration": np.asarray(HISTORY.fs_iteration),
        "loss_pt": np.asarray(HISTORY.loss_pt),
        "loss_pt_coupled_pde": np.asarray(HISTORY.loss_pt_coupled_pde),
        "loss_u": np.asarray(HISTORY.loss_u),
        "u_rel_l2": np.asarray(HISTORY.u_rel_l2),
        "u1_rel_l2": np.asarray(HISTORY.u1_rel_l2),
        "u2_rel_l2": np.asarray(HISTORY.u2_rel_l2),
        "xi_rel_l2": np.asarray(HISTORY.xi_rel_l2),
        "p_rel_l2": np.asarray(HISTORY.p_rel_l2),
        "T_rel_l2": np.asarray(HISTORY.T_rel_l2),
        "u_mean_rel_l2": np.asarray(HISTORY.u_mean_rel_l2),
        "pT_mean_rel_l2": np.asarray(HISTORY.pT_mean_rel_l2),
        "common_rel_l2_max": np.asarray(HISTORY.common_rel_l2_max),
        "four_field_component_rel_l2_max": np.asarray(
            HISTORY.four_field_component_rel_l2_max
        ),
        "physical_rel_l2_max": np.asarray(HISTORY.physical_rel_l2_max),
        "four_field_rel_l2_max": np.asarray(
            HISTORY.four_field_rel_l2_max
        ),
        "u_h1": np.asarray(HISTORY.u_h1),
        "u1_h1": np.asarray(HISTORY.u1_h1),
        "u2_h1": np.asarray(HISTORY.u2_h1),
        "xi_h1": np.asarray(HISTORY.xi_h1),
        "p_h1": np.asarray(HISTORY.p_h1),
        "T_h1": np.asarray(HISTORY.T_h1),
        "u_outer_change": np.asarray(HISTORY.u_outer_change),
        "pt_outer_change": np.asarray(HISTORY.pt_outer_change),
        "stop_condition": np.asarray(HISTORY.stop_condition, dtype=bool),
        "stop_counter": np.asarray(HISTORY.stop_counter, dtype=int),
        "training_time": np.asarray(HISTORY.training_time),
    }
    if REPORT_PT_MODAL_DIAGNOSTICS:
        history_arrays.update(
            {
                "q_parallel_rel_l2": np.asarray(
                    HISTORY.q_parallel_rel_l2
                ),
                "q_perp_rel_l2": np.asarray(HISTORY.q_perp_rel_l2),
            }
        )
    if U_RANK_SAFE_RIDGE_ACTIVE:
        history_arrays["xi_t_rel_l2"] = np.asarray(
            HISTORY.xi_t_rel_l2
        )
    np.savez(RESULT_DIR / "training_history.npz", **history_arrays)

    # CSV/JSON mirrors make the complete outer history inspectable without
    # NumPy-specific tooling and keep all experiment families auditable.
    history_columns = vars(HISTORY)
    history_rows = [
        {
            name: values[index]
            for name, values in history_columns.items()
        }
        for index in range(len(HISTORY.fs_iteration))
    ]
    if history_rows:
        with open(
            RESULT_DIR / "training_history.csv",
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(history_columns.keys()),
            )
            writer.writeheader()
            writer.writerows(history_rows)
        with open(
            RESULT_DIR / "training_history.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(history_rows, file, indent=2, ensure_ascii=False)


def _comparison_run_identity() -> Dict[str, object]:
    """Return identifiers repeated in standalone comparison tables."""
    identity = {
        "run_name": RESULT_DIR.name,
        "method": "THP-four-field-iterative-decoupling-ENGD",
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
        "pt_residual_mode": PT_RESIDUAL_MODE,
        "pt_parallel_storage": PT_PARALLEL_STORAGE,
        "pt_modal_storage_coupling": PT_MODAL_STORAGE_COUPLING,
        "pt_parallel_residual_scale": PT_PARALLEL_RESIDUAL_SCALE,
        "pt_perp_storage": PT_PERP_STORAGE,
        "pt_perp_residual_scale": PT_PERP_RESIDUAL_SCALE,
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
    if IS_GROUP5_RUN:
        identity.update(
            {
                "group5_case_id": GROUP5_CASE_ID,
                "group5_variant": GROUP5_VARIANT,
            }
        )
    if IS_GROUP5_RUN and GROUP5_PAPER_CASE_ID:
        identity["group5_legacy_case_id"] = GROUP5_CASE_ID
        identity["group5_case_id"] = GROUP5_PAPER_CASE_ID
    if DOUBLE_SMALL_EXTENSION_ACTIVE:
        identity.update(
            {
                "group5_extension": "unequal_double_small",
                "double_small_protocol": (
                    "raw"
                    if GROUP5_VARIANT == _DOUBLE_SMALL_RAW_VARIANT
                    else (
                        "modal_both_scaled"
                        if GROUP5_VARIANT == _DOUBLE_SMALL_MODAL_VARIANT
                        else "physical_rank_safe"
                    )
                ),
            }
        )
    if S7_RAW_FS100_ACTIVE:
        identity.update(
            {
                "group5_extension": "equal_double_small",
                "s7_protocol": "raw",
                "fs100_with_fs50_retained": True,
            }
        )
    if U_RANK_SAFE_RIDGE_ACTIVE:
        identity.update(
            {
                "u_rank_safe_ridge_safety": (
                    U_RANK_SAFE_RIDGE_SAFETY
                ),
                "u_rank_safe_ridge_definition": (
                    "max(base_ridge, safety*eps*P_u*"
                    "max(lambda_max(G_u),1))"
                ),
                "u_base_gram_ridge": GRAM_RIDGE,
                "pt_gram_ridge_policy": "fixed",
            }
        )
    if REPORT_PT_MODAL_DIAGNOSTICS:
        identity.update(
            {
                "pt_generalized_eta": PT_GENERALIZED_ETA,
                "pt_generalized_sigma_1": float(
                    _pt_generalized_eigenvalues[0]
                ),
                "pt_generalized_sigma_2": float(
                    _pt_generalized_eigenvalues[1]
                ),
                "pt_generalized_balance_1": float(
                    _pt_generalized_balance[0]
                ),
                "pt_generalized_balance_2": float(
                    _pt_generalized_balance[1]
                ),
                "pt_generalized_eta_used": (
                    PT_RESIDUAL_MODE == "generalized_modal_scaled"
                ),
                "pt_physical_pullback_11": float(
                    _PT_GENERALIZED_PHYSICAL_MATRIX_NP[0, 0]
                ),
                "pt_physical_pullback_12": float(
                    _PT_GENERALIZED_PHYSICAL_MATRIX_NP[0, 1]
                ),
                "pt_physical_pullback_21": float(
                    _PT_GENERALIZED_PHYSICAL_MATRIX_NP[1, 0]
                ),
                "pt_physical_pullback_22": float(
                    _PT_GENERALIZED_PHYSICAL_MATRIX_NP[1, 1]
                ),
            }
        )
    return identity


def _comparison_values_at(index: int) -> Dict[str, object]:
    """Return all accuracy and loss values at one recorded FS iteration."""
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
    values = {
        "fs_iteration": fs_iteration,
        "pT_engd_updates_completed": pt_updates_completed,
        "u_engd_updates_completed": u_updates_completed,
        "total_engd_updates_completed": (
            pt_updates_completed + u_updates_completed
        ),
        "effective_training_seconds": float(HISTORY.training_time[index]),
        "loss_pt": float(HISTORY.loss_pt[index]),
        "loss_pt_coupled_pde": float(
            HISTORY.loss_pt_coupled_pde[index]
        ),
        "loss_u": float(HISTORY.loss_u[index]),
        "u_rel_l2": float(HISTORY.u_rel_l2[index]),
        "u1_rel_l2": float(HISTORY.u1_rel_l2[index]),
        "u2_rel_l2": float(HISTORY.u2_rel_l2[index]),
        "xi_rel_l2": float(HISTORY.xi_rel_l2[index]),
        "p_rel_l2": float(HISTORY.p_rel_l2[index]),
        "T_rel_l2": float(HISTORY.T_rel_l2[index]),
        "u_mean_rel_l2": float(HISTORY.u_mean_rel_l2[index]),
        "pT_mean_rel_l2": float(HISTORY.pT_mean_rel_l2[index]),
        "E_common_max": float(HISTORY.common_rel_l2_max[index]),
        "E_4F_max": float(
            HISTORY.four_field_component_rel_l2_max[index]
        ),
        "physical_rel_l2_max": float(
            HISTORY.physical_rel_l2_max[index]
        ),
        "four_field_rel_l2_max": float(
            HISTORY.four_field_rel_l2_max[index]
        ),
        "component_rel_l2_max": float(
            max(
                HISTORY.u1_rel_l2[index],
                HISTORY.u2_rel_l2[index],
                HISTORY.p_rel_l2[index],
                HISTORY.T_rel_l2[index],
            )
        ),
        "u_h1": float(HISTORY.u_h1[index]),
        "u1_h1": float(HISTORY.u1_h1[index]),
        "u2_h1": float(HISTORY.u2_h1[index]),
        "xi_h1": float(HISTORY.xi_h1[index]),
        "p_h1": float(HISTORY.p_h1[index]),
        "T_h1": float(HISTORY.T_h1[index]),
    }
    if REPORT_PT_MODAL_DIAGNOSTICS:
        values.update(
            {
                "q_parallel_rel_l2": float(
                    HISTORY.q_parallel_rel_l2[index]
                ),
                "q_perp_rel_l2": float(
                    HISTORY.q_perp_rel_l2[index]
                ),
            }
        )
    if U_RANK_SAFE_RIDGE_ACTIVE:
        values["xi_t_rel_l2"] = float(
            HISTORY.xi_t_rel_l2[index]
        )
    return values


def build_fixed_budget_accuracy_rows() -> List[Dict[str, object]]:
    """Build equal-outer-budget accuracy rows at predefined checkpoints."""
    completed_iterations = len(HISTORY.fs_iteration)
    index_by_iteration = {
        int(iteration): index
        for index, iteration in enumerate(HISTORY.fs_iteration)
    }
    unavailable_values = {
        "fs_iteration": None,
        "pT_engd_updates_completed": None,
        "u_engd_updates_completed": None,
        "total_engd_updates_completed": None,
        "effective_training_seconds": None,
        "loss_pt": None,
        "loss_pt_coupled_pde": None,
        "loss_u": None,
        "u_rel_l2": None,
        "u1_rel_l2": None,
        "u2_rel_l2": None,
        "xi_rel_l2": None,
        "p_rel_l2": None,
        "T_rel_l2": None,
        "u_mean_rel_l2": None,
        "pT_mean_rel_l2": None,
        "E_common_max": None,
        "E_4F_max": None,
        "physical_rel_l2_max": None,
        "four_field_rel_l2_max": None,
        "component_rel_l2_max": None,
        "u_h1": None,
        "u1_h1": None,
        "u2_h1": None,
        "xi_h1": None,
        "p_h1": None,
        "T_h1": None,
    }
    if REPORT_PT_MODAL_DIAGNOSTICS:
        unavailable_values.update(
            {
                "q_parallel_rel_l2": None,
                "q_perp_rel_l2": None,
            }
        )
    if U_RANK_SAFE_RIDGE_ACTIVE:
        unavailable_values["xi_t_rel_l2"] = None
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

    metric_histories = [
        ("u1_rel_l2", "u1_rel_l2", HISTORY.u1_rel_l2),
        ("u2_rel_l2", "u2_rel_l2", HISTORY.u2_rel_l2),
        ("xi_rel_l2", "xi_rel_l2", HISTORY.xi_rel_l2),
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
        (
            "E_4F_max",
            "max(u1_rel_l2,u2_rel_l2,xi_rel_l2,p_rel_l2,T_rel_l2)",
            HISTORY.four_field_component_rel_l2_max,
        ),
    ]
    if REPORT_PT_MODAL_DIAGNOSTICS:
        metric_histories[5:5] = [
            (
                "q_parallel_rel_l2",
                "q_parallel_rel_l2",
                HISTORY.q_parallel_rel_l2,
            ),
            (
                "q_perp_rel_l2",
                "q_perp_rel_l2",
                HISTORY.q_perp_rel_l2,
            ),
        ]
    if U_RANK_SAFE_RIDGE_ACTIVE:
        metric_histories.insert(
            5,
            (
                "xi_t_rel_l2",
                "relative_L2(xi_t_prediction-xi_t_exact)",
                HISTORY.xi_t_rel_l2,
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
                        float(values[record_index])
                        if reached
                        else None
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
    """Save fixed-budget accuracy and time-to-accuracy reports."""
    fixed_budget_rows = build_fixed_budget_accuracy_rows()
    time_to_accuracy_rows = build_time_to_accuracy_rows()

    with open(
        RESULT_DIR / "fixed_budget_accuracy.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(fixed_budget_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(fixed_budget_rows)

    with open(
        RESULT_DIR / "fixed_budget_accuracy.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(fixed_budget_rows, file, indent=2, ensure_ascii=False)

    with open(
        RESULT_DIR / "time_to_accuracy.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(time_to_accuracy_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(time_to_accuracy_rows)

    with open(
        RESULT_DIR / "time_to_accuracy.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(time_to_accuracy_rows, file, indent=2, ensure_ascii=False)

    comparison_definitions = {
            "primary_metric": (
                "E_common_max="
                "max(u1_rel_l2,u2_rel_l2,p_rel_l2,T_rel_l2)"
            ),
            "auxiliary_four_field_metric": (
                "E_4F_max="
                "max(u1_rel_l2,u2_rel_l2,xi_rel_l2,p_rel_l2,T_rel_l2)"
            ),
            "group_mean_metrics": (
                "u_mean_rel_l2=0.5*(u1_rel_l2+u2_rel_l2); "
                "pT_mean_rel_l2=0.5*(p_rel_l2+T_rel_l2)"
            ),
            "componentwise_fixed_damping_selection": (
                "compare u1_rel_l2, u2_rel_l2, p_rel_l2, and T_rel_l2 "
                "at the same budget; prefer a damping value that wins at "
                "least three of the four components"
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
    }
    if REPORT_PT_MODAL_DIAGNOSTICS:
        comparison_definitions["storage_aligned_mode_metrics"] = (
            "q_parallel=(alpha*p+beta*T)/sqrt(alpha^2+beta^2); "
            "q_perp=(beta*p-alpha*T)/sqrt(alpha^2+beta^2)"
        )

    comparison_summary = {
        "definitions": comparison_definitions,
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
                "Fixed-budget FS %d: time=%.3f s E_common=%.4e "
                "E_4F=%.4e",
                row["requested_fs_iteration"],
                row["effective_training_seconds"],
                row["E_common_max"],
                row["E_4F_max"],
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
                "Time-to-accuracy %s<=%.3e: FS=%d time=%.3f s "
                "value=%.4e",
                row["metric_name"],
                row["threshold"],
                row["first_fs_iteration"],
                row["time_to_accuracy_seconds"],
                row["metric_value_at_first_reach"],
            )
        else:
            LOGGER.info(
                "Time-to-accuracy %s<=%.3e: NR, time>%.3f s, "
                "final=%.4e",
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


def pt_residual_metric_metadata() -> Dict[str, object]:
    """Describe the active p/T metric without polluting raw archives."""
    metric: Dict[str, object] = {
        "mode": PT_RESIDUAL_MODE,
        "parallel_variable": (
            "(alpha*p+beta*T)/sqrt(alpha^2+beta^2)"
        ),
        "perpendicular_variable": (
            "(beta*p-alpha*T)/sqrt(alpha^2+beta^2)"
        ),
        "parallel_storage": PT_PARALLEL_STORAGE,
        "modal_storage_coupling": PT_MODAL_STORAGE_COUPLING,
        "parallel_scale": PT_PARALLEL_RESIDUAL_SCALE,
        "perpendicular_storage": PT_PERP_STORAGE,
        "perpendicular_scale": PT_PERP_RESIDUAL_SCALE,
        "changes_exact_pde_solution": False,
    }
    if GROUP5_VARIANT == _DOUBLE_SMALL_MODAL_VARIANT:
        metric.update(
            {
                "modal_parallel_diffusion": PT_MODAL_PARALLEL_DIFFUSION,
                "modal_perpendicular_diffusion": PT_MODAL_PERP_DIFFUSION,
                "modal_cross_diffusion": PT_MODAL_CROSS_DIFFUSION,
            }
        )
    if REPORT_PT_MODAL_DIAGNOSTICS:
        active_transform_definitions = {
            "raw": "Identity*[r_p,r_T]",
            "modal": "R*[r_p,r_T]",
            "modal_perp_scaled": (
                "diag(1,perpendicular_scale^(-1))*R*[r_p,r_T]"
            ),
            "modal_both_scaled": (
                "diag(parallel_scale^(-1),perpendicular_scale^(-1))"
                "*R*[r_p,r_T]"
            ),
            "diffusion_whitened": "D^(-1/2)*[r_p,r_T]",
            "generalized_modal_balanced": (
                "M^(-1)*U^T*D^(-1/2)*[r_p,r_T]"
            ),
            "generalized_modal_scaled": (
                "eta*M^(-1)*U^T*D^(-1/2)*[r_p,r_T]"
            ),
            "generalized_physical_scaled": (
                "D^(-1/2)*U*M^(-1)*U^T*D^(-1/2)*[r_p,r_T]"
            ),
        }
        metric.update(
            {
                "raw_residual_order": ["r_p", "r_T"],
                "active_transform_definition": (
                    active_transform_definitions[PT_RESIDUAL_MODE]
                ),
                "generalized_definition": (
                    "D^(-1/2)*U*M^(-1)*U^T*D^(-1/2)*[r_p,r_T] "
                    "for generalized_physical_scaled; "
                    "eta*M^(-1)*U^T*D^(-1/2)*[r_p,r_T] for "
                    "generalized_modal_scaled, "
                    "D=diag(K,Theta), "
                    "U*diag(sigma)*U^T=D^(-1/2)*C*D^(-1/2), "
                    "M=diag(max(sigma,1))"
                ),
                "generalized_eta_default": "(K*Theta)^(-1/4)",
                "generalized_eta": PT_GENERALIZED_ETA,
                "generalized_eta_used": (
                    PT_RESIDUAL_MODE == "generalized_modal_scaled"
                ),
                "diffusion_inverse_sqrt": (
                    _PT_DIFFUSION_INVERSE_SQRT_NP.tolist()
                ),
                "whitened_storage_matrix": (
                    _PT_WHITENED_STORAGE_NP.tolist()
                ),
                "generalized_eigenvalues_descending": (
                    _pt_generalized_eigenvalues.tolist()
                ),
                "generalized_eigenvectors_columns": (
                    _pt_generalized_eigenvectors.tolist()
                ),
                "generalized_balance_diagonal": (
                    _pt_generalized_balance.tolist()
                ),
                "generalized_physical_matrix": (
                    _PT_GENERALIZED_PHYSICAL_MATRIX_NP.tolist()
                ),
                "generalized_physical_eigenvalues_ascending": (
                    _pt_generalized_physical_eigenvalues.tolist()
                ),
                "generalized_physical_condition_number": (
                    _pt_generalized_physical_condition
                ),
            }
        )
    return metric


def accuracy_reporting_metadata() -> Dict[str, object]:
    """Return the old raw schema plus opt-in Group-5 modal metrics."""
    reporting: Dict[str, object] = {
        "primary_metric": (
            "E_common_max="
            "max(u1_rel_l2,u2_rel_l2,p_rel_l2,T_rel_l2)"
        ),
        "auxiliary_four_field_metric": (
            "E_4F_max="
            "max(u1_rel_l2,u2_rel_l2,xi_rel_l2,p_rel_l2,T_rel_l2)"
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
    }
    if REPORT_PT_MODAL_DIAGNOSTICS:
        reporting.update(
            {
                "q_parallel_metric": (
                    "relative_L2((alpha*p+beta*T)/sqrt(alpha^2+beta^2))"
                ),
                "q_perp_metric": (
                    "relative_L2((beta*p-alpha*T)/sqrt(alpha^2+beta^2))"
                ),
            }
        )
    if U_RANK_SAFE_RIDGE_ACTIVE:
        reporting["xi_t_metric"] = "relative_L2(xi_t_prediction-xi_t_exact)"
    return reporting


def engd_config_metadata(config: ENGDConfig) -> Dict[str, object]:
    """Serialize Rank-safe fields only when the opt-in experiment is active."""
    values = asdict(config)
    if not U_RANK_SAFE_RIDGE_ACTIVE:
        values.pop("rank_safe_ridge_safety")
    return values


def save_metadata(timing: TrainingTiming) -> None:
    metadata = {
        "method": "THP-four-field-iterative-decoupling-ENGD",
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
        "four_field_split": {
            "xi_definition": "-lambda*div(u)+alpha*p+beta*T",
            "C_alpha": C_ALPHA,
            "C_alpha_beta": C_ALPHA_BETA,
            "C_beta": C_BETA,
            "pt_pde_residual_metric": pt_residual_metric_metadata(),
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
            "uxi_sizes": NET_U_SIZES,
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
            **engd_config_metadata(ENGD_CONFIG_PT),
            "steps_per_gfs_iteration": ENGD_STEPS_PT,
        },
        "engd_u": {
            **engd_config_metadata(ENGD_CONFIG_U),
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
        "accuracy_reporting": accuracy_reporting_metadata(),
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

    if IS_GROUP5_RUN:
        metadata["experiment_design"] = {
            "group": 5,
            "case_id": GROUP5_CASE_ID,
            "variant": GROUP5_VARIANT,
        }
    if DOUBLE_SMALL_EXTENSION_ACTIVE:
        metadata["experiment_design"].update(
            {
                "extension": "unequal_double_small",
                "protocol": (
                    "raw"
                    if GROUP5_VARIANT == _DOUBLE_SMALL_RAW_VARIANT
                    else (
                        "modal_both_scaled"
                        if GROUP5_VARIANT == _DOUBLE_SMALL_MODAL_VARIANT
                        else "physical_rank_safe"
                    )
                ),
                "mirror_case": "S11"
                if GROUP5_CASE_ID == "S10"
                else "S10",
                "fs100_with_fs50_retained": True,
                "changes_group1_to_group4_defaults": False,
            }
        )
    if S7_RAW_FS100_ACTIVE:
        metadata["experiment_design"].update(
            {
                "extension": "equal_double_small",
                "protocol": "raw",
                "fs100_with_fs50_retained": True,
                "changes_group1_to_group4_defaults": False,
            }
        )
    if U_RANK_SAFE_RIDGE_ACTIVE:
        metadata["profiling"].update(
            {
                "u_rank_safe_ridge_enabled": True,
                "u_rank_safe_ridge_safety": U_RANK_SAFE_RIDGE_SAFETY,
                "u_rank_safe_ridge_definition": (
                    "max(base_ridge, safety*eps*P_u*"
                    "max(lambda_max(G_u),1))"
                ),
                "pt_rank_safe_ridge_enabled": False,
            }
        )
    if IS_GROUP5_RUN and GROUP5_PAPER_CASE_ID:
        design = metadata["experiment_design"]
        design["legacy_case_id"] = GROUP5_CASE_ID
        design["case_id"] = GROUP5_PAPER_CASE_ID
        if "mirror_case" in design:
            design["legacy_mirror_case"] = design["mirror_case"]
            design["mirror_case"] = GROUP5_LEGACY_TO_PAPER.get(
                design["mirror_case"], design["mirror_case"]
            )
    if DOUBLE_SMALL_EXTENSION_ACTIVE:
        metadata["experiment_design"]["fs100_with_fs50_retained"] = FS_ITERATIONS == 100
        metadata["experiment_design"]["paper_evaluation_sweep"] = 50
    with open(RESULT_DIR / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)


# =============================================================================
# 13. Exact Gram microbenchmark
# =============================================================================


def _global_pt_gram_gradient(
    flat_pt: Array,
    flat_u_old: Array,
) -> Tuple[Array, Array, Array]:
    residual = residual_pt_flat(flat_pt, flat_u_old)
    jacobian = JACOBIAN_PT_FLAT(flat_pt, flat_u_old)
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
        flat_u_old, validation_block
    )
    validation_u_context = build_pointwise_u_context(
        flat_pt, validation_block
    )
    validation_pt_frozen = validation_pt_context.pde_frozen.reshape(-1, 2)[
        : PDE_POINTS.shape[0]
    ]
    validation_u_pde_frozen = validation_u_context.pde_frozen.reshape(-1, 3)[
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
        pt_candidates, flat_u_old
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
        lambda: _global_pt_gram_gradient(flat_pt, flat_u_old),
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
            flat_u_old, block_size
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
