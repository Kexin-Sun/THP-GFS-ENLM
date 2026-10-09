#!/usr/bin/env bash
set -euo pipefail

# Group 3: near-incompressible comparison between three-field and four-field
# generalized fixed-stress formulations.  Both formulations use exactly the
# same seed, collocation sets, loss weights, JS-EIGH configuration and fixed
# update budget.  The only swept physical parameter is Poisson's ratio nu.

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
nu_csv="${NUS:-0.49,0.4999,0.49999,0.4999999}"

required_files=(
  "${three_field_program}"
  "${canonical_core}"
  "${script_dir}/plot_group3_incompressible.py"
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
IFS=',' read -r -a nu_values <<< "${nu_csv}"

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
  "THP_KAPPA=0.1"
  "THP_THETA=0.1"
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
  # Group 3 uses aggregate cross-nu plots, so per-run field/history plots are
  # disabled.  This does not change training or effective training time.
  "THP_SAVE_PLOTS=0"
)

nu_tag() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  printf '%s' "${value}"
}

run_method() {
  local seed="$1"
  local nu="$2"
  local method="$3"
  local program="$4"
  local tag
  tag="$(nu_tag "${nu}")"
  local run_name="group3_${batch_tag}_nu${tag}_seed${seed}_${method}"
  local run_dir="${result_root}/${run_name}"

  # Never overwrite or merge an existing scientific run.  Use a new BATCH_TAG
  # for an interrupted/repeated run.
  if [[ -e "${run_dir}" ]]; then
    echo "Refusing to overwrite existing run directory: ${run_dir}" >&2
    echo "Choose a new BATCH_TAG and rerun." >&2
    exit 1
  fi

  if [[ "${dry_run}" == "1" ]]; then
    echo "Planned Group 3: method=${method} nu=${nu} seed=${seed}; program=${program}; FS50"
    return
  fi

  echo "============================================================"
  echo "Group 3: method=${method} nu=${nu} seed=${seed}"
  echo "Output: ${run_dir}"
  echo "============================================================"
  env "${common_env[@]}" \
    "THP_NU=${nu}" \
    "THP_SEED=${seed}" \
    "THP_RUN_NAME=${run_name}" \
    "${python_bin}" "${program}"
}

for seed in "${seed_values[@]}"; do
  seed="${seed//[[:space:]]/}"
  [[ -n "${seed}" ]] || continue
  for nu in "${nu_values[@]}"; do
    nu="${nu//[[:space:]]/}"
    [[ -n "${nu}" ]] || continue
    run_method "${seed}" "${nu}" "gfs_three_field" "${three_field_program}"
    run_method "${seed}" "${nu}" "gfs_four_field" "${canonical_core}"
  done
done

if [[ "${dry_run}" == "1" || "${fast_debug}" == "1" ]]; then
  echo "Training plan/smoke test complete; production aggregation is skipped."
  exit 0
fi

output_dir="${script_dir}/reports/${batch_tag}"
echo "===== Generating Group-3 tables and aggregate figures ====="
"${python_bin}" "${script_dir}/plot_group3_incompressible.py" \
  "${result_root}" \
  --batch-tag "${batch_tag}" \
  --seeds "${seed_csv}" \
  --nus "${nu_csv}" \
  --output-dir "${output_dir}"

echo "All Group-3 runs and reports completed."
echo "Results: ${result_root}"
echo "Reports: ${output_dir}"

