#!/usr/bin/env bash
set -euo pipefail

# Group 4: archive the proposed four-field GFS method at exactly five physical
# parameter settings. Three-field experiments belong only to Groups 2 and 3.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
canonical_core="${repo_root}/thp-four-field-core.py"
three_field_program="${repo_root}/group2_formulation/thp-three-field.py"
python_bin="${PYTHON_BIN:-python}"
dry_run="${THP_DRY_RUN:-0}"
fast_debug="${THP_FAST_DEBUG:-0}"
gpu_id="${GPU_ID:-1}"
result_root="${RESULT_ROOT:-${script_dir}/runs}"
batch_tag="${BATCH_TAG:-$(date +%Y%m%d-%H%M%S)}"
seed_csv="${SEEDS:-12345}"

# Each entry is nu,K,Theta. Keep this order in the publication table.
case_matrix="${CASE_MATRIX:-0.3,0.1,0.1;0.3,0.1,1e-6;0.3,1e-6,0.1;0.3,1e-6,1e-6;0.4999999,1e-6,1e-6}"

required_files=(
  "${canonical_core}"
  "${script_dir}/summarize_group4_small_transport.py"
)
for required_file in "${required_files[@]}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Missing required file: ${required_file}" >&2
    exit 1
  fi
done

# Protect frozen results across every group in this repository.
"${python_bin}" - "${result_root}" "${repo_root}" <<'PY'
from pathlib import Path
import sys
target = Path(sys.argv[1]).resolve()
for frozen in Path(sys.argv[2]).glob("group*/results"):
    frozen = frozen.resolve()
    if target == frozen or frozen in target.parents:
        raise SystemExit("Refusing to train inside frozen results: " + str(target))
PY

# Frozen results are input data; new training uses runs/ by default.
result_root="$("${python_bin}" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "${result_root}")"
frozen_root="${script_dir}/results"
case "${result_root}/" in
  "${frozen_root}/"*) echo "Refusing to train inside frozen results: ${result_root}" >&2; exit 2 ;;
esac
if [[ "${dry_run}" != "1" ]]; then mkdir -p "${result_root}"; fi
IFS=',' read -r -a seed_values <<< "${seed_csv}"
IFS=';' read -r -a case_values <<< "${case_matrix}"

common_env=(
  "CUDA_VISIBLE_DEVICES=${gpu_id}"
  "THP_RESULT_ROOT=${result_root}"
  "THP_FOUR_FIELD_CORE=${canonical_core}"
  "THP_GROUP5_CASE_ID="
    "THP_GROUP5_PAPER_CASE_ID="
  "THP_GROUP5_VARIANT="
  "THP_U_RANK_SAFE_RIDGE_SAFETY=0.0"
  "THP_XLA_PREALLOCATE=true"
  "THP_XLA_MEM_FRACTION=${MEM_FRACTION:-0.90}"
  "THP_EXACT_CASE=distinct"
  "THP_E=1.0"
  "THP_ALPHA=0.1"
  "THP_BETA=0.1"
  "THP_A0=0.2"
  "THP_B0=0.1"
  "THP_C0=0.2"
  "THP_T_FINAL=1"
  "THP_N_XY_PDE=16"
  "THP_N_T_PDE=16"
  "THP_N_XY_IC=22"
  "THP_N_SIDE=24"
  "THP_N_T_BOUNDARY=17"
  "THP_N_XY_EVAL=40"
  "THP_N_T_EVAL=40"
  "THP_WEIGHT_U_PDE=1"
  "THP_WEIGHT_U_DIRICHLET=1"
  "THP_WEIGHT_U_NEUMANN=1"
  "THP_WEIGHT_U_IC=1"
  "THP_WEIGHT_PT_PDE=1"
  "THP_WEIGHT_PT_DIRICHLET=1"
  "THP_WEIGHT_PT_IC=1"
  # Modal/scaled variants belong to the separate stabilization ablation.
  "THP_PT_RESIDUAL_MODE=raw"
  "THP_GRAM_RIDGE=1e-7"
  "THP_ENGD_SOLVER=eigh-refined"
  "THP_EIGH_RELATIVE_CUTOFF=0"
  "THP_EIGH_REFINEMENT_STEPS=2"
  "THP_EIGH_COMPARE_LSTSQ=0"
  "THP_LINE_SEARCH_LEVELS=31"
  "THP_JACOBIAN_BACKEND=pointwise"
  "THP_FUSE_GRAM_BLOCKS=1"
  "THP_CACHE_LINE_SEARCH=1"
  "THP_POINT_JAC_BLOCK_U=512"
  "THP_POINT_JAC_BLOCK_PT=512"
  "THP_FS_ITERATIONS=50"
  "THP_ENGD_STEPS_U=20"
  "THP_ENGD_STEPS_PT=20"
  "THP_ENABLE_FS_EARLY_STOP=0"
  "THP_ENABLE_COUPLED_EARLY_STOP=0"
  "THP_TIME_TO_ACCURACY_THRESHOLDS=1e-2,1e-3,8e-4,6e-4,4e-4,2e-4"
  "THP_TTA_PATIENCE=3"
  "THP_FIXED_BUDGET_CHECKPOINTS=20,30,40,50"
  "THP_VERIFY_MANUFACTURED=1"
  "THP_SAVE_PLOTS=0"
)

value_tag() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  value="${value//+/p}"
  printf '%s' "${value}"
}

run_one() {
  local seed="$1"
  local nu="$2"
  local kappa="$3"
  local theta="$4"
  local run_name run_dir

  run_name="group4_${batch_tag}_nu$(value_tag "${nu}")_K$(value_tag "${kappa}")_Theta$(value_tag "${theta}")_seed${seed}_gfs_four_field"
  run_dir="${result_root}/${run_name}"

  if [[ -e "${run_dir}" ]]; then
    echo "Refusing to overwrite existing run directory: ${run_dir}" >&2
    echo "Use a new BATCH_TAG for the rerun." >&2
    exit 1
  fi

  if [[ "${dry_run}" == "1" ]]; then
    echo "Planned Group 4: nu=${nu} K=${kappa} Theta=${theta} seed=${seed}; FS50"
    return
  fi

  echo "============================================================"
  echo "Group 4: 4F-GFS nu=${nu} K=${kappa} Theta=${theta} seed=${seed}"
  echo "Output: ${run_dir}"
  echo "============================================================"

  env "${common_env[@]}" \
    "THP_NU=${nu}" \
    "THP_KAPPA=${kappa}" \
    "THP_THETA=${theta}" \
    "THP_SEED=${seed}" \
    "THP_RUN_NAME=${run_name}" \
    "${python_bin}" "${canonical_core}"
}

for seed in "${seed_values[@]}"; do
  seed="${seed//[[:space:]]/}"
  [[ -n "${seed}" ]] || continue
  for case_entry in "${case_values[@]}"; do
    case_entry="${case_entry//[[:space:]]/}"
    [[ -n "${case_entry}" ]] || continue
    IFS=',' read -r nu kappa theta extra <<< "${case_entry}"
    if [[ -z "${nu:-}" || -z "${kappa:-}" || -z "${theta:-}" || -n "${extra:-}" ]]; then
      echo "Invalid CASE_MATRIX entry '${case_entry}'; expected nu,K,Theta." >&2
      exit 1
    fi
    run_one "${seed}" "${nu}" "${kappa}" "${theta}"
  done
done

if [[ "${dry_run}" == "1" || "${fast_debug}" == "1" ]]; then
  echo "Training plan/smoke test complete; production aggregation is skipped."
  exit 0
fi

report_dir="${script_dir}/reports/${batch_tag}"
echo "===== Generating Group-4 tables (no figures) ====="
"${python_bin}" "${script_dir}/summarize_group4_small_transport.py" \
  "${result_root}" \
  --batch-tag "${batch_tag}" \
  --seeds "${seed_csv}" \
  --cases "${case_matrix}" \
  --output-dir "${report_dir}"

echo "All five Group-4 runs and tables completed."
echo "Results: ${result_root}"
echo "Tables:  ${report_dir}"

