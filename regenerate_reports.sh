#!/usr/bin/env bash
set -euo pipefail

# Reproduce all five groups from frozen records; never import training code.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-python}"
"${python_bin}" "${repo_root}/verify_frozen_results.py"
"${python_bin}" "${repo_root}/group1_optimizer/analyze_group1_gfs.py"
"${python_bin}" "${repo_root}/group1_optimizer/plot_group1_paper_figures.py"
"${python_bin}" "${repo_root}/group2_formulation/plot_group2_comparison.py" \
    --batch-tag multiseed_v1 --seeds 12345,23456,34567,45678,56789
"${python_bin}" "${repo_root}/group3_incompressible/plot_group3_incompressible.py" --batch-tag v1
"${python_bin}" "${repo_root}/group4_small_transport/summarize_group4_small_transport.py" --batch-tag v1
"${python_bin}" "${repo_root}/export_component_tables.py"
"${python_bin}" "${repo_root}/group5_storage_degeneracy/summarize_group5_storage_degeneracy.py"
"${python_bin}" "${repo_root}/verify_frozen_results.py"
