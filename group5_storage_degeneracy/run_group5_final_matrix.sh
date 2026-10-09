#!/usr/bin/env bash
set -euo pipefail

# Paper matrix: 12 runs per seed, paper cases S1--S11, evaluated at FS50.
# Original FS100 budgets and exploratory controls remain explicit options.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
canonical_core="${repo_root}/thp-four-field-core.py"
python_bin="${PYTHON_BIN:-python}"
gpu_id="${GPU_ID:-1}"
result_root="${RESULT_ROOT:-${script_dir}/runs}"
batch_tag="${BATCH_TAG:-$(date +%Y%m%d-%H%M%S)}"
seed_csv="${SEEDS:-12345}"
run_only="${RUN_ONLY:-ALL}"
budget="${BUDGET:-paper}"
dry_run="${THP_DRY_RUN:-${DRY_RUN:-0}}"
fast_debug="${THP_FAST_DEBUG:-0}"
resume="${RESUME:-0}"

[[ -f "${canonical_core}" ]] || { echo "Missing canonical root core." >&2; exit 2; }
for flag in dry_run fast_debug resume; do
    [[ "${!flag}" == 0 || "${!flag}" == 1 ]] || {
        echo "${flag} must be 0 or 1." >&2; exit 2;
    }
done

# Build the entire plan before starting any optimizer. No JAX imports here.
plan="$("${python_bin}" - "${script_dir}/experiment_matrix.json" \
    "${result_root}" "${repo_root}" "${run_only}" "${budget}" \
    "${seed_csv}" "${batch_tag}" "${resume}" <<'PY'
import json, sys
from pathlib import Path
matrix_path, destination, root, selection, budget, seed_csv, batch, resume = sys.argv[1:]
target = Path(destination).resolve()
for frozen in Path(root).glob("group*/results"):
    frozen = frozen.resolve()
    if target == frozen or frozen in target.parents:
        raise SystemExit("Refusing to train inside frozen results: " + str(target))
if budget not in {"paper", "historical"}:
    raise SystemExit("BUDGET must be paper or historical.")
if not batch or any(c in batch for c in "/\\") or batch in {".", ".."}:
    raise SystemExit("BATCH_TAG must be a nonempty directory-safe label.")
matrix = json.loads(Path(matrix_path).read_text())
by_key = {row["key"]: row for row in matrix}
keys = [x.strip() for x in selection.split(",") if x.strip()]
if keys == ["ALL"]:
    keys = [row["key"] for row in matrix if row["role"] == "paper"]
if not keys or len(set(keys)) != len(keys) or any(key not in by_key for key in keys):
    raise SystemExit("RUN_ONLY must be ALL or distinct keys from experiment_matrix.json.")
seeds = [x.strip() for x in seed_csv.split(",") if x.strip()]
if not seeds or len(set(seeds)) != len(seeds) or any(not x.isdigit() for x in seeds):
    raise SystemExit("SEEDS must contain distinct nonnegative integer seeds.")
for seed in seeds:
    for key in keys:
        row = by_key[key]
        fs = row["default_training_budget"] if budget == "paper" else row["original_budget"]
        name = f"group5_{batch}_{key}_FS{fs}_seed{seed}_4f_gfs"
        run = target / name
        if run.exists():
            if resume == "1" and (run / "metadata.json").is_file() and (run / "final_model.npz").is_file():
                m = json.loads((run / "metadata.json").read_text())
                hpath = run / "training_history.json"
                h = json.loads(hpath.read_text()) if hpath.is_file() else []
                if (m.get("four_field_split", {}).get("iterations_completed") == fs
                        and [r["fs_iteration"] for r in h] == list(range(1, fs + 1))):
                    print("Skipping complete run: " + name, file=sys.stderr)
                    continue
            raise SystemExit("Refusing to overwrite existing or incomplete run: " + str(run))
        weights = row["pt_weights"]
        columns = [seed, key, row["paper_case_id"], row["legacy_case_id"],
                   row["variant"], row["a0"], row["b0"], row["c0"], row["K"],
                   row["Theta"], row["mode"], row["parallel_scale"],
                   row["perpendicular_scale"], row["rank_safe_safety"], fs,
                   weights["pde"], weights["dirichlet"], weights["ic"], name]
        print("\t".join(map(str, columns)))
PY
)"

common_env=(
    "CUDA_VISIBLE_DEVICES=${gpu_id}" "THP_RESULT_ROOT=${result_root}"
    "THP_FOUR_FIELD_CORE=${canonical_core}" "THP_FAST_DEBUG=${fast_debug}"
    "THP_XLA_PREALLOCATE=true" "THP_XLA_MEM_FRACTION=${MEM_FRACTION:-0.90}"
    "THP_EXACT_CASE=distinct" "THP_E=1.0" "THP_NU=0.3"
    "THP_ALPHA=0.1" "THP_BETA=0.1" "THP_T_FINAL=1"
    "THP_N_XY_PDE=16" "THP_N_T_PDE=16" "THP_N_XY_IC=22"
    "THP_N_SIDE=24" "THP_N_T_BOUNDARY=17" "THP_N_XY_EVAL=40" "THP_N_T_EVAL=40"
    "THP_WEIGHT_U_PDE=1" "THP_WEIGHT_U_DIRICHLET=1"
    "THP_WEIGHT_U_NEUMANN=1" "THP_WEIGHT_U_IC=1"
    "THP_GRAM_RIDGE=1e-7" "THP_ENGD_SOLVER=eigh-refined"
    "THP_EIGH_RELATIVE_CUTOFF=0" "THP_EIGH_REFINEMENT_STEPS=2"
    "THP_EIGH_COMPARE_LSTSQ=0" "THP_LINE_SEARCH_LEVELS=31"
    "THP_JACOBIAN_BACKEND=pointwise" "THP_FUSE_GRAM_BLOCKS=1" "THP_CACHE_LINE_SEARCH=1"
    "THP_JACOBIAN_MICROBENCH=0" "THP_POINT_JAC_BLOCK_U=512" "THP_POINT_JAC_BLOCK_PT=512"
    "THP_ENGD_STEPS_U=20" "THP_ENGD_STEPS_PT=20"
    "THP_ENABLE_FS_EARLY_STOP=0" "THP_ENABLE_COUPLED_EARLY_STOP=0"
    "THP_TIME_TO_ACCURACY_THRESHOLDS=1e-1,5e-2,2e-2,1e-2,8e-3,6e-3,5e-3,4e-3,3e-3,2e-3,1e-3"
    "THP_TTA_PATIENCE=3" "THP_VERIFY_MANUFACTURED=1" "THP_SAVE_PLOTS=0"
)

if [[ -z "${plan}" ]]; then
    echo "No new runs to launch."
    exit 0
fi
while IFS=$'\t' read -r seed key paper_case legacy_case variant a0 b0 c0 kappa theta mode parallel perpendicular rank fs wpde wdir wic name; do
    echo "Planned Group 5: ${key}; paper=${paper_case}; historical=${legacy_case}; FS${fs}; seed=${seed}"
    [[ "${dry_run}" == 1 ]] && continue
    checkpoints="20,30,40,50"
    [[ "${fs}" == 100 ]] && checkpoints="20,30,40,50,60,80,100"
    env "${common_env[@]}" \
        "THP_GROUP5_CASE_ID=${legacy_case}" "THP_GROUP5_PAPER_CASE_ID=${paper_case}" \
        "THP_GROUP5_VARIANT=${variant}" "THP_SEED=${seed}" "THP_RUN_NAME=${name}" \
        "THP_A0=${a0}" "THP_B0=${b0}" "THP_C0=${c0}" "THP_KAPPA=${kappa}" "THP_THETA=${theta}" \
        "THP_PT_RESIDUAL_MODE=${mode}" "THP_PT_PARALLEL_RESIDUAL_SCALE=${parallel}" \
        "THP_PT_PERP_RESIDUAL_SCALE=${perpendicular}" "THP_U_RANK_SAFE_RIDGE_SAFETY=${rank}" \
        "THP_WEIGHT_PT_PDE=${wpde}" "THP_WEIGHT_PT_DIRICHLET=${wdir}" "THP_WEIGHT_PT_IC=${wic}" \
        "THP_FS_ITERATIONS=${fs}" "THP_FIXED_BUDGET_CHECKPOINTS=${checkpoints}" \
        "${python_bin}" "${canonical_core}"
done <<< "${plan}"

if [[ "${dry_run}" == 1 || "${fast_debug}" == 1 ]]; then
    echo "Training plan/smoke test complete; production reports are skipped."
    exit 0
fi
"${python_bin}" "${script_dir}/summarize_group5_storage_degeneracy.py" \
    --results "${result_root}" --batch-tag "${batch_tag}" \
    --output "${script_dir}/reports/${batch_tag}" --allow-subset
