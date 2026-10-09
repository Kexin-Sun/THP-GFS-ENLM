#!/usr/bin/env bash
set -euo pipefail

# Usage: bash run_group1_gfs.sh 0
# Run seven required methods serially on one visible GPU.
# A fresh batch tag protects frozen results from accidental overwrites.

GPU_ID="${GPU_ID:-${1:-0}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CANONICAL_CORE="${REPO_ROOT}/thp-four-field-core.py"
DRY_RUN="${THP_DRY_RUN:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RESULT_ROOT="${RESULT_ROOT:-${THP_GROUP1_RESULT_ROOT:-${SCRIPT_DIR}/runs}}"
BATCH_TAG="${BATCH_TAG:-${THP_GROUP1_BATCH_TAG:-$(date +%Y%m%d_%H%M%S)}}"
PREFIX="group1_gfs_${BATCH_TAG}_"
METHODS="${THP_GROUP1_METHODS:-adam,adam_lbfgs,engd_fjls,js_lstsq0,js_eigh0,js_eigh1e7,full_eigh1e7}"
FAST_DEBUG="${THP_FAST_DEBUG:-0}"

# Resolve destinations and protect every frozen experiment group.
RESULT_ROOT="$("${PYTHON_BIN}" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "${RESULT_ROOT}")"
"${PYTHON_BIN}" - "${RESULT_ROOT}" "${REPO_ROOT}" <<'PY'
from pathlib import Path
import sys
target = Path(sys.argv[1]).resolve()
for frozen in Path(sys.argv[2]).glob("group*/results"):
    frozen = frozen.resolve()
    if target == frozen or frozen in target.parents:
        raise SystemExit("Refusing to train inside frozen results: " + str(target))
PY
[[ -f "${CANONICAL_CORE}" ]] || { echo "Missing canonical root core." >&2; exit 2; }

# Reject misspellings instead of silently skipping requested methods.
IFS=',' read -r -a REQUESTED_METHODS <<< "${METHODS}"
for method in "${REQUESTED_METHODS[@]}"; do
    case "${method}" in
        adam|adam_lbfgs|engd_fjls|js_lstsq0|js_eigh0|js_eigh1e7|full_eigh1e7|full_eigh0) ;;
        *) echo "Unknown Group-1 method: ${method}" >&2; exit 2 ;;
    esac
    if [[ -e "${RESULT_ROOT}/${PREFIX}${method}" ]]; then
        echo "Run directory already exists: ${RESULT_ROOT}/${PREFIX}${method}" >&2
        echo "Choose a fresh THP_GROUP1_BATCH_TAG to preserve the existing run." >&2
        exit 2
    fi
done

if [[ "${DRY_RUN}" == "1" ]]; then
    for method in "${REQUESTED_METHODS[@]}"; do
        echo "Planned Group 1: method=${method}; FS100; output=${RESULT_ROOT}/${PREFIX}${method}"
    done
    exit 0
fi

mkdir -p "${RESULT_ROOT}"
MPL_CONFIG_DIR="${RESULT_ROOT}/matplotlib_config"
mkdir -p "${MPL_CONFIG_DIR}"

env \
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    MPLCONFIGDIR="${MPL_CONFIG_DIR}" \
    THP_FAST_DEBUG="${FAST_DEBUG}" \
    "${PYTHON_BIN}" - <<'PY'
import jax
import scipy

jax.config.update("jax_enable_x64", True)
devices = jax.devices()
print("JAX version:", jax.__version__)
print("SciPy version:", scipy.__version__)
print("JAX x64 enabled:", jax.config.jax_enable_x64)
print("Visible JAX devices:", devices)

if len(devices) != 1:
    raise SystemExit(
        "Exactly one device must be visible inside every Group-1 process."
    )
if not jax.config.jax_enable_x64:
    raise SystemExit("Group-1 requires JAX FP64.")
PY

if [[ "${FAST_DEBUG}" != "1" ]]; then
    env CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" - <<'PY'
import jax
if jax.default_backend() != "gpu":
    raise SystemExit(
        "Production Group-1 runs require a JAX GPU backend. "
        "Set THP_FAST_DEBUG=1 only for a smoke test."
    )
PY
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi \
        --query-gpu=index,name,uuid,driver_version,memory.total \
        --format=csv,noheader \
        > "${RESULT_ROOT}/${PREFIX}hardware.txt"

    REQUIRE_ALL_GPUS_IDLE="${THP_REQUIRE_ALL_GPUS_IDLE:-1}"
    GPU_COMPUTE_PROCESSES="$(
        nvidia-smi \
            --query-compute-apps=pid,gpu_uuid,process_name \
            --format=csv,noheader,nounits 2>/dev/null || true
    )"
    if [[ "${REQUIRE_ALL_GPUS_IDLE}" == "1" && -n "${GPU_COMPUTE_PROCESSES//[[:space:]]/}" ]]; then
        echo "Refusing to start: at least one GPU already has a compute process." >&2
        echo "${GPU_COMPUTE_PROCESSES}" >&2
        echo "Wait until every GPU is idle, or set THP_REQUIRE_ALL_GPUS_IDLE=0 for an intentional exception." >&2
        exit 3
    fi
fi

COMMON_ENV=(
    "CUDA_VISIBLE_DEVICES=${GPU_ID}"
    "MPLCONFIGDIR=${MPL_CONFIG_DIR}"
    "THP_RESULT_ROOT=${RESULT_ROOT}"
    "THP_FOUR_FIELD_CORE=${CANONICAL_CORE}"
    "THP_FAST_DEBUG=${FAST_DEBUG}"
    "THP_SEED=${THP_SEED:-12345}"
    "THP_E=1.0"
    "THP_NU=0.3"
    "THP_ALPHA=0.1"
    "THP_BETA=0.1"
    "THP_T_FINAL=1"
    "THP_N_XY_PDE=16"
    "THP_N_XY_IC=22"
    "THP_N_SIDE=24"
    "THP_N_XY_EVAL=40"
    "THP_N_T_EVAL=40"
    "THP_A0=0.2"
    "THP_B0=0.1"
    "THP_C0=0.2"
    "THP_KAPPA=0.1"
    "THP_THETA=0.1"
    "THP_EXACT_CASE=${THP_EXACT_CASE:-distinct}"
    "THP_PT_RESIDUAL_MODE=raw"
    "THP_GROUP5_CASE_ID="
    "THP_GROUP5_PAPER_CASE_ID="
    "THP_GROUP5_VARIANT="
    "THP_U_RANK_SAFE_RIDGE_SAFETY=0.0"
    "THP_WEIGHT_PT_PDE=1.0"
    "THP_WEIGHT_PT_DIRICHLET=1.0"
    "THP_WEIGHT_PT_IC=1.0"
    "THP_N_T_PDE=${THP_N_T_PDE:-16}"
    "THP_N_T_BOUNDARY=${THP_N_T_BOUNDARY:-17}"
    "THP_FS_ITERATIONS=100"
    "THP_GROUP1_FS_ITERATIONS=100"
    "THP_ENABLE_FS_EARLY_STOP=0"
    "THP_GROUP1_ENGD_STEPS_PER_BLOCK=20"
    "THP_GROUP1_ADAM_STEPS_PER_BLOCK=1000"
    "THP_GROUP1_ADAM_LR=${THP_GROUP1_ADAM_LR:-1e-3}"
    "THP_GROUP1_ADAM_LBFGS_SWITCH_FS=30"
    "THP_GROUP1_LBFGS_MAXITER_PER_BLOCK=200"
    "THP_GROUP1_LBFGS_MAXFUN_PER_BLOCK=10000"
    "THP_GROUP1_LBFGS_FTOL=${THP_GROUP1_LBFGS_FTOL:-1e-15}"
    "THP_GROUP1_LBFGS_GTOL=${THP_GROUP1_LBFGS_GTOL:-1e-10}"
    "THP_LINE_SEARCH_LEVELS=31"
    "THP_POINT_JAC_BLOCK_PT=${THP_POINT_JAC_BLOCK_PT:-512}"
    "THP_POINT_JAC_BLOCK_U=${THP_POINT_JAC_BLOCK_U:-512}"
    "THP_FULL_JAC_POINT_BLOCK_PT=${THP_FULL_JAC_POINT_BLOCK_PT:-512}"
    "THP_FULL_JAC_POINT_BLOCK_U=${THP_FULL_JAC_POINT_BLOCK_U:-512}"
    "THP_EIGH_RELATIVE_CUTOFF=0.0"
    "THP_EIGH_REFINEMENT_STEPS=2"
    "THP_TTA_PATIENCE=3"
    "THP_TIME_TO_ACCURACY_THRESHOLDS=1e-2,1e-3,8e-4,6e-4,4e-4,2e-4,1e-4"
    "THP_GROUP1_FIXED_FS_CHECKPOINTS=0,50,100"
    "THP_FIXED_TIME_BUDGETS_SECONDS=${THP_FIXED_TIME_BUDGETS_SECONDS:-10,30,60,120,300,600,1200,1800,3600}"
    "THP_XLA_PREALLOCATE=${THP_XLA_PREALLOCATE:-true}"
    "THP_XLA_MEM_FRACTION=${THP_XLA_MEM_FRACTION:-0.90}"
    "THP_VERIFY_MANUFACTURED=${THP_VERIFY_MANUFACTURED:-1}"
    "THP_SAVE_PLOTS=0"
)

method_selected() {
    local candidate="$1"
    [[ ",${METHODS}," == *",${candidate},"* ]]
}

run_method() {
    local method="$1"
    local run_name="${PREFIX}${method}"
    echo "============================================================"
    echo "Starting ${method} on physical GPU ${GPU_ID}"
    echo "Run name: ${run_name}"
    env \
        "${COMMON_ENV[@]}" \
        "THP_RUN_NAME=${run_name}" \
        "THP_GROUP1_METHOD=${method}" \
        "${PYTHON_BIN}" -u "${SCRIPT_DIR}/thp_group1_gfs_comparison.py"
}

for method in \
    adam \
    adam_lbfgs \
    engd_fjls \
    js_lstsq0 \
    js_eigh0 \
    js_eigh1e7 \
    full_eigh1e7 \
    full_eigh0
do
    if method_selected "${method}"; then
        run_method "${method}"
    fi
done

if [[ "${FAST_DEBUG}" == "1" ]]; then
    echo "Reduced-budget smoke runs completed; production reports are skipped."
    exit 0
fi

SUMMARY_DIR="${SCRIPT_DIR}/reports/${BATCH_TAG}"
ANALYZE_ARGS=()
if method_selected full_eigh0; then
    ANALYZE_ARGS+=(--include-optional)
fi
if [[ "${FAST_DEBUG}" == "1" || "${METHODS}" != "adam,adam_lbfgs,engd_fjls,js_lstsq0,js_eigh0,js_eigh1e7,full_eigh1e7" ]]; then
    ANALYZE_ARGS+=(--allow-incomplete)
fi
"${PYTHON_BIN}" -u "${SCRIPT_DIR}/analyze_group1_gfs.py" \
    --root "${RESULT_ROOT}" \
    --prefix "${PREFIX}" \
    --output "${SUMMARY_DIR}" \
    "${ANALYZE_ARGS[@]}"

if [[ "${FAST_DEBUG}" != "1" ]]; then
    HAVE_PAPER_METHODS=1
    for method in adam adam_lbfgs engd_fjls js_eigh0 js_eigh1e7; do
        if ! method_selected "${method}"; then HAVE_PAPER_METHODS=0; fi
    done
    if [[ "${HAVE_PAPER_METHODS}" == "1" ]]; then
        "${PYTHON_BIN}" "${SCRIPT_DIR}/plot_group1_paper_figures.py" \
            --root "${RESULT_ROOT}" --prefix "${PREFIX}" \
            --output "${SUMMARY_DIR}/paper_figures"
    fi
fi

echo "============================================================"
echo "All requested Group-1 4F-GFS runs are complete."
echo "Result root: ${RESULT_ROOT}"
echo "Batch prefix: ${PREFIX}"
echo "Summary: ${SUMMARY_DIR}"
echo "Production protocol: 100 GFS sweeps for every method."
echo "ENGD: 20 updates/subproblem/sweep; Adam: 1000; Adam->L-BFGS: FS30."
