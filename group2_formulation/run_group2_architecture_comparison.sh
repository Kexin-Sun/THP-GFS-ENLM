#!/usr/bin/env bash
set -euo pipefail

# Group 2: one-network coupled, two-network coupled, three-field GFS, and
# four-field GFS.  All four runs use the same physics, collocation sets,
# initialization seed, loss weights, streamed fixed-damping ENLM configuration, and
# equivalent-iteration budget.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
canonical_core="${repo_root}/thp-four-field-core.py"
three_field_program="${repo_root}/group2_formulation/thp-three-field.py"
python_bin="${PYTHON_BIN:-python}"
dry_run="${THP_DRY_RUN:-0}"
fast_debug="${THP_FAST_DEBUG:-0}"
gpu_id="${GPU_ID:-0}"
result_root="${RESULT_ROOT:-${script_dir}/runs}"
batch_tag="${BATCH_TAG:-$(date +%Y%m%d-%H%M%S)}"
seed_csv="${SEEDS:-12345,23456,34567,45678,56789}"

required_files=(
  "${script_dir}/thp-coupled.py"
  "${three_field_program}"
  "${canonical_core}"
  "${script_dir}/plot_group2_comparison.py"
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
  "THP_NU=0.3"
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
  "THP_POINT_JAC_BLOCK_COUPLED=${COUPLED_BLOCK_SIZE:-128}"
  "THP_FS_ITERATIONS=50"
  "THP_ENGD_STEPS_U=20"
  "THP_ENGD_STEPS_PT=20"
  "THP_COUPLED_ROUNDS=50"
  "THP_COUPLED_ENGD_STEPS=20"
  "THP_ENABLE_FS_EARLY_STOP=0"
  "THP_ENABLE_COUPLED_EARLY_STOP=0"
  "THP_TIME_TO_ACCURACY_THRESHOLDS=1e-2,1e-3,8e-4,6e-4,4e-4,2e-4"
  "THP_TTA_PATIENCE=3"
  "THP_FIXED_BUDGET_CHECKPOINTS=10,20,30,40,50"
  "THP_VERIFY_MANUFACTURED=1"
  "THP_SAVE_PLOTS=1"
)

# Check the complete batch before launching any method.
for seed in "${seed_values[@]}"; do
  seed="${seed//[[:space:]]/}"
  [[ "${seed}" =~ ^[0-9]+$ ]] || { echo "Invalid seed: ${seed}" >&2; exit 2; }
  for method in coupled_single coupled_dual gfs_three_field gfs_four_field; do
    run_dir="${result_root}/group2_${batch_tag}_seed${seed}_${method}"
    if [[ -e "${run_dir}" ]]; then
      echo "Refusing to overwrite existing run directory: ${run_dir}" >&2
      echo "Choose a fresh BATCH_TAG." >&2; exit 2
    fi
  done
done

run_one_seed() {
  local seed="$1"
  if [[ "${dry_run}" == "1" ]]; then
    echo "Planned Group 2: seed=${seed}; coupled-single, coupled-dual, 3F-GFS, 4F-GFS; FS50"
    return
  fi

  echo "===== Group 2 seed=${seed}: coupled single network ====="
  env "${common_env[@]}" \
    "THP_SEED=${seed}" \
    "THP_COUPLED_ARCH=single" \
    "THP_RUN_NAME=group2_${batch_tag}_seed${seed}_coupled_single" \
    "${python_bin}" "${script_dir}/thp-coupled.py"

  echo "===== Group 2 seed=${seed}: coupled dual network ====="
  env "${common_env[@]}" \
    "THP_SEED=${seed}" \
    "THP_COUPLED_ARCH=dual" \
    "THP_RUN_NAME=group2_${batch_tag}_seed${seed}_coupled_dual" \
    "${python_bin}" "${script_dir}/thp-coupled.py"

  echo "===== Group 2 seed=${seed}: three-field GFS ====="
  env "${common_env[@]}" \
    "THP_SEED=${seed}" \
    "THP_RUN_NAME=group2_${batch_tag}_seed${seed}_gfs_three_field" \
    "${python_bin}" "${three_field_program}"

  echo "===== Group 2 seed=${seed}: four-field GFS ====="
  env "${common_env[@]}" \
    "THP_SEED=${seed}" \
    "THP_RUN_NAME=group2_${batch_tag}_seed${seed}_gfs_four_field" \
    "${python_bin}" "${canonical_core}"
}

for seed in "${seed_values[@]}"; do
  seed="${seed//[[:space:]]/}"
  if [[ -z "${seed}" ]]; then
    continue
  fi
  run_one_seed "${seed}"
done

if [[ "${dry_run}" == "1" || "${fast_debug}" == "1" ]]; then
  echo "Training plan/smoke test complete; production aggregation is skipped."
  exit 0
fi

echo "===== Generating cross-method Group-2 comparison figures ====="
"${python_bin}" "${script_dir}/plot_group2_comparison.py" \
  "${result_root}" \
  --batch-tag "${batch_tag}" \
  --seeds "${seed_csv}" \
  --output-dir "${script_dir}/reports/${batch_tag}"

echo "All Group-2 runs completed under: ${result_root}"
