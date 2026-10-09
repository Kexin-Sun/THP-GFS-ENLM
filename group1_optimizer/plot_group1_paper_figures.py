"""Reproduce the five representative Group-1 paper curves (Figures 2 and 3).

Seven methods are required by Tables 1--3. js_lstsq0 and full_eigh1e7 are
controlled-comparison runs used by Table 3; they are intentionally absent
from the five-curve main-text figures. Use analyze_group1_gfs.py for all seven.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from analyze_group1_gfs import assert_common_protocol

PAPER_METHODS = (
    'adam',
    'adam_lbfgs',
    'engd_fjls',
    'js_eigh0',
    'js_eigh1e7',
)

METHOD_LABELS = {
    'adam': 'Adam',
    'adam_lbfgs': 'Adam+L-BFGS',
    'engd_fjls': r'ENGD (FJ+LS, $\gamma=0$)',
    'js_eigh0': r'ENGD (S+EIGH, $\gamma=0$)',
    'js_eigh1e7': r'ENLM (S+EIGH, $\gamma=10^{-7}$)',
}

COLORS = {
    'adam': '#1f77b4',
    'adam_lbfgs': '#2ca02c',
    'engd_fjls': '#8c564b',
    'js_eigh0': '#ff7f0e',
    'js_eigh1e7': '#17becf',
}

MARKERS = {
    'adam': 'o',
    'adam_lbfgs': 's',
    'engd_fjls': 'D',
    'js_eigh0': 'v',
    'js_eigh1e7': 'P',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot paper-ready Group-1 figures.')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent / 'results', help='Directory containing per-method result folders.')
    parser.add_argument('--prefix', type=str, default='group1_gfs_distinct_seed12345_20260824_022014_', help='Common run prefix before the method suffix.')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'archive_summary/paper_figures', help='Output directory for figures.')
    return parser.parse_args()


def read_json(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def discover_runs(root: Path, prefix: str) -> Dict[str, Dict[str, object]]:
    runs: Dict[str, Dict[str, object]] = {}
    for method in PAPER_METHODS:
        candidates = sorted(root.glob(f'{prefix}{method}'))
        if not candidates:
            raise FileNotFoundError(f'Cannot find run folder for method {method!r} under {root}.')
        directory = candidates[0]
        history_path = directory / 'training_history.json'
        metadata_path = directory / 'metadata.json'
        if not history_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f'Missing history/metadata in {directory}')
        runs[method] = {
            'directory': directory,
            'history': read_json(history_path),
            'metadata': read_json(metadata_path),
        }
    assert_common_protocol(runs)
    return runs


def history_at_fs(history: List[Dict[str, object]], fs_iteration: int):
    for row in history:
        if int(row['fs_iteration']) == fs_iteration:
            return row
    return None


def plot_component_figure(
    runs: Dict[str, Dict[str, object]],
    fields: Sequence[Tuple[str, str]],
    x_key: str,
    x_label: str,
    output_stem: Path,
    y_label: str = r'Relative $L^2$ error',
) -> None:
    fig, axes = plt.subplots(1, len(fields), figsize=(5.0 * len(fields), 4.0), sharey=False)
    if len(fields) == 1:
        axes = [axes]

    for axis_idx, (field, title) in enumerate(fields):
        ax = axes[axis_idx]
        for method in PAPER_METHODS:
            history = runs[method]['history']
            x = np.asarray([float(row[x_key]) for row in history], dtype=float)
            y = np.asarray([float(row[field]) for row in history], dtype=float)
            ax.plot(
                x, y,
                label=METHOD_LABELS[method],
                color=COLORS[method],
                marker=MARKERS[method],
                markevery=max(1, len(x) // 10),
                markersize=4,
                linewidth=1.9,
            )
        ax.set_yscale('log')
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(x_label, fontsize=12)
        ax.grid(True, which='both', alpha=0.28)
        if axis_idx == 0:
            ax.set_ylabel(y_label, fontsize=12)
        ax.tick_params(labelsize=10)

        if x_key == 'fs_iteration':
            ax.axvline(30, color=COLORS['adam_lbfgs'], linestyle=':', linewidth=1.2, alpha=0.8)
        elif x_key == 'effective_training_seconds':
            switch = history_at_fs(runs['adam_lbfgs']['history'], 30)
            if switch is not None:
                ax.axvline(float(switch['effective_training_seconds']), color=COLORS['adam_lbfgs'], linestyle=':', linewidth=1.2, alpha=0.8)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.10), frameon=True, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(output_stem.with_suffix('.png'), dpi=250, bbox_inches='tight')
    fig.savefig(output_stem.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def make_all_plots(runs: Dict[str, Dict[str, object]], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)

    mechanics = (
        ('u1_rel_l2', r'$u_1$'),
        ('u2_rel_l2', r'$u_2$'),
        ('xi_rel_l2', r'$\xi$'),
    )
    flow_thermal = (
        ('p_rel_l2', r'$p$'),
        ('T_rel_l2', r'$T$'),
    )

    plot_component_figure(
        runs, mechanics,
        'effective_training_seconds', 'Effective training time (s)',
        output / 'mechanics_errors_vs_effective_time_paper'
    )
    plot_component_figure(
        runs, flow_thermal,
        'effective_training_seconds', 'Effective training time (s)',
        output / 'flow_thermal_errors_vs_effective_time_paper'
    )
    plot_component_figure(
        runs, mechanics,
        'fs_iteration', 'GFS outer iteration',
        output / 'mechanics_errors_vs_fs_iteration_paper'
    )
    plot_component_figure(
        runs, flow_thermal,
        'fs_iteration', 'GFS outer iteration',
        output / 'flow_thermal_errors_vs_fs_iteration_paper'
    )
    plot_component_figure(
        runs,
        (( 'E_common_max', r'$E_{\mathrm{common}}^{\max}$' ),),
        'effective_training_seconds', 'Effective training time (s)',
        output / 'common_max_error_vs_effective_time_paper'
    )


def main() -> None:
    args = parse_args()
    args.root = args.root.resolve()
    args.output = args.output.resolve()
    protected = [args.root, *Path(__file__).resolve().parents[1].glob("group*/results")]
    if any(args.output == path.resolve() or path.resolve() in args.output.parents
           for path in protected):
        raise ValueError("Choose a report output directory outside all frozen results.")
    runs = discover_runs(args.root, args.prefix)
    make_all_plots(runs, args.output)
    print('Paper figures written to:', args.output)
    for method in PAPER_METHODS:
        print(method, '->', runs[method]['directory'])


if __name__ == '__main__':
    main()
