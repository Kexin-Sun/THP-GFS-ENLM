#!/usr/bin/env python3
"""Reproduce Group 5 Tables 8--11 directly from frozen FS50 histories.

This program never imports the training core or JAX. The explicit matrix
maps historical S10--S13 to paper S8--S11 while preserving original bytes.
FS100 and exploratory controls are separate supplementary exports.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
FIELDS = ('u1', 'u2', 'xi', 'p', 'T')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=HERE / 'results')
    parser.add_argument('--output', type=Path, default=HERE / 'archive_summary')
    parser.add_argument('--batch-tag', help='Select new runs with group5_<tag>_ prefix.')
    parser.add_argument('--seed', type=int, default=12345,
                        help='Seed used for the single-seed manuscript tables.')
    parser.add_argument('--allow-subset', action='store_true',
                        help='Export available cases when analyzing selected new runs.')
    return parser.parse_args()


def write_csv(path, rows):
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_display(stem, title, headers, rows, note):
    """Write readable rounded tables; full-precision numeric exports are CSV."""
    md = [f'# {title}', '', note, '', '| ' + ' | '.join(headers) + ' |',
          '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    md += ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows]
    stem.with_suffix('.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    def tex(value):
        return str(value).replace('_', r'\_').replace('%', r'\%')
    lines = [r'\begin{tabular}{' + 'l' * len(headers) + '}', r'\hline',
             ' & '.join(map(tex, headers)) + r' \\', r'\hline']
    lines += [' & '.join(map(tex, row)) + r' \\' for row in rows]
    lines += [r'\hline', r'\end{tabular}', '']
    stem.with_suffix('.tex').write_text('\n'.join(lines), encoding='utf-8')


def match_config(directory, metadata, matrix):
    exact = [row for row in matrix if row['original_run_name'] == directory.name]
    design = metadata.get('experiment_design', {})
    if exact:
        candidates = exact
    else:
        candidates = [row for row in matrix
                      if row['legacy_case_id'] == design.get('legacy_case_id', design.get('case_id'))
                      and row['variant'] == design.get('variant')]
    if len(candidates) != 1:
        raise ValueError(f'Cannot uniquely map run to the archived matrix: {directory}')
    config = candidates[0]
    physics = metadata['physical_parameters']
    for key in ('a0', 'b0', 'c0', 'K', 'Theta'):
        if not math.isclose(physics[key], config[key], rel_tol=1e-12, abs_tol=0):
            raise ValueError(f'{directory}: inconsistent physical parameter {key}')
    for key, value in [('nu', 0.3), ('E', 1), ('alpha', 0.1), ('beta', 0.1)]:
        if not math.isclose(physics[key], value, rel_tol=1e-12, abs_tol=0):
            raise ValueError(f'{directory}: inconsistent reference parameter {key}')
    metric = metadata['four_field_split']['pt_pde_residual_metric']
    if metric['mode'] != config['mode']:
        raise ValueError(f'{directory}: residual mode does not match the matrix')
    for key, config_key in [('parallel_scale', 'parallel_scale'),
                            ('perpendicular_scale', 'perpendicular_scale')]:
        if not math.isclose(metric[key], config[config_key], rel_tol=1e-12, abs_tol=0):
            raise ValueError(f'{directory}: residual scale {key} does not match the matrix')
    if metadata['loss_weighting']['weights']['pt'] != config['pt_weights']:
        raise ValueError(f'{directory}: pressure-temperature loss weights differ')
    if not math.isclose(metadata['profiling']['gram_ridge'], 1e-7, rel_tol=1e-12):
        raise ValueError(f'{directory}: unexpected base damping')
    safety = metadata['profiling'].get('u_rank_safe_ridge_safety', 0.0)
    if safety != config['rank_safe_safety']:
        raise ValueError(f'{directory}: unexpected rank-safe ridge configuration')
    if metadata.get('exact_case') != 'distinct' or metadata.get('fast_debug'):
        raise ValueError(f'{directory}: expected a completed production distinct-case run')
    return config


def discover(results, matrix, batch):
    runs = []
    for directory in sorted(results.iterdir()):
        if batch and not directory.name.startswith(f'group5_{batch}_'):
            continue
        if not (directory / 'metadata.json').is_file():
            continue
        metadata = read_json(directory / 'metadata.json')
        config = match_config(directory, metadata, matrix)
        history = read_json(directory / 'training_history.json')
        budget = metadata['four_field_split']['iterations_completed']
        if budget not in (50, 100) or [row['fs_iteration'] for row in history] != list(range(1, budget + 1)):
            raise ValueError(f'{directory}: incomplete or inconsistent production history')
        if not (directory / 'final_model.npz').is_file():
            raise ValueError(f'{directory}: missing final checkpoint')
        runs.append(dict(directory=directory, metadata=metadata, config=config,
                         history={row['fs_iteration']: row for row in history}))
    if not runs:
        raise ValueError(f'No completed Group 5 runs found under {results}')
    return runs


def identity(run):
    c = run['config']
    return dict(key=c['key'], paper_case_id=c['paper_case_id'],
                legacy_case_id=c['legacy_case_id'], role=c['role'],
                original_run_name=run['directory'].name, seed=run['metadata']['seed'],
                completed_sweeps=max(run['history']), residual_mode=c['mode'])


def accuracy_row(run, sweep):
    result = identity(run)
    row = run['history'][sweep]
    c = run['config']
    result.update({key: c[key] for key in ('a0', 'b0', 'c0', 'K', 'Theta')})
    result.update(fs_iteration=sweep, effective_training_seconds=row['training_time'],
                  E_common_max=max(row[f'{f}_rel_l2'] for f in ('u1', 'u2', 'p', 'T')),
                  E_4F_max=max(row[f'{f}_rel_l2'] for f in FIELDS))
    for field in FIELDS:
        for suffix in ('rel_l2', 'h1'):
            result[f'{field}_{suffix}'] = row[f'{field}_{suffix}']
    for key in ('loss_pt', 'loss_pt_coupled_pde', 'q_parallel_rel_l2', 'q_perp_rel_l2'):
        result[key] = row.get(key, '')
    return result


def export_components(output, number, title, runs):
    rows, display = [], []
    for run in runs:
        config = run['config']
        row = run['history'][50]
        label = config['paper_case_id']
        if number == 10:
            label += ' raw' if config['mode'] == 'raw' else ' modal'
        for field in FIELDS:
            rows.append(dict(**identity(run), fs_iteration=50, field=field,
                             relative_L2=row[f'{field}_rel_l2'], spatial_H1=row[f'{field}_h1']))
        for metric, suffix in [('Relative L2', 'rel_l2'), ('Spatial H1', 'h1')]:
            display.append([label, metric, *[f'{row[f"{field}_{suffix}"]:.2e}' for field in FIELDS]])
    stem = output / f'table{number}_component_errors'
    write_csv(stem.with_suffix('.csv'), rows)
    write_display(stem, title, ['Case', 'Metric', *FIELDS], display,
                  'Values come directly from the archived FS50 history. L2 is relative; '
                  'spatial H1 is absolute. Paper case labels are used.')


def solver_rows(run):
    with (run['directory'] / 'engd_profile.csv').open(newline='') as handle:
        profile = list(csv.DictReader(handle))
    result = []
    for block in sorted({row['subproblem'] for row in profile}):
        first = [row for row in profile if row['subproblem'] == block and int(row['fs_iteration']) <= 50]
        tail = [row for row in first if 41 <= int(row['fs_iteration']) <= 50]
        if not first:
            continue
        record = dict(**identity(run), subproblem=block, budget_sweep=50,
                      accepted_fraction=sum(row['accepted'].lower() in ('true', '1') for row in first) / len(first))
        for key in ('effective_condition', 'effective_rank', 'solve_relative_residual'):
            values = [float(row[key]) for row in tail if row[key] and math.isfinite(float(row[key]))]
            record[f'{key}_median_FS41_FS50'] = statistics.median(values) if values else ''
        result.append(record)
    return result


def main():
    args = parse_args()
    args.results = args.results.resolve()
    args.output = args.output.resolve()
    protected = [args.results, *HERE.parent.glob('group*/results')]
    if any(args.output == path.resolve() or path.resolve() in args.output.parents for path in protected):
        raise ValueError('Choose a report output directory outside all frozen/input results.')
    matrix = read_json(HERE / 'experiment_matrix.json')
    runs = discover(args.results, matrix, args.batch_tag)
    by_key = {}
    for run in runs:
        if run['metadata']['seed'] == args.seed:
            key = run['config']['key']
            if key in by_key:
                raise ValueError(f'Duplicate key {key} for seed {args.seed}; select --batch-tag.')
            by_key[key] = run
    required = [row['key'] for row in matrix if row['role'] == 'paper']
    missing = [key for key in required if key not in by_key]
    if missing and not args.allow_subset:
        raise ValueError(f'Missing paper keys at seed {args.seed}: {missing}')
    args.output.mkdir(parents=True, exist_ok=True)
    parameter_rows = []
    for number in range(1, 12):
        config = next(row for row in matrix if row['paper_case_id'] == f'S{number}' and row['role'] == 'paper')
        det = config['a0'] * config['c0'] - config['b0'] ** 2
        rank = 0 if config['a0'] == config['b0'] == config['c0'] == 0 else (1 if abs(det) < 1e-14 else 2)
        parameter_rows.append(dict(paper_case_id=f'S{number}', legacy_case_id=config['legacy_case_id'],
                                   **{k:config[k] for k in ('a0', 'b0', 'c0', 'K', 'Theta')},
                                   storage_determinant=det, storage_rank=rank, paper_evaluation_sweep=50))
    stem = args.output / 'table8_parameter_configurations'
    write_csv(stem.with_suffix('.csv'), parameter_rows)
    write_display(stem, 'Table 8: storage/transport configurations',
                  ['Case', 'a0', 'b0', 'c0', 'K', 'Theta', 'Storage rank'],
                  [[row['paper_case_id'], *[f'{row[k]:.7g}' for k in ('a0', 'b0', 'c0', 'K', 'Theta')], row['storage_rank']]
                   for row in parameter_rows],
                  'S2 and S5 have positive-definite physical storage; S4 has rank one and S6--S11 have zero storage.')
    selection = lambda keys: [by_key[key] for key in keys if key in by_key]
    export_components(args.output, 9, 'Table 9: storage degeneracy at moderate transport',
                      selection([f'S{i}_RAW' for i in range(1, 7)]))
    export_components(args.output, 10, 'Table 10: S7 raw and modal residuals',
                      selection(['S7_RAW', 'S7_MODAL']))
    export_components(args.output, 11, 'Table 11: unequal weak transport with modal residuals',
                      selection([f'S{i}_MODAL' for i in range(8, 12)]))
    diagnostic_rows = []
    for run in selection(['S7_RAW', 'S7_MODAL']):
        row = run['history'][50]
        for field in ('q_parallel_rel_l2', 'q_perp_rel_l2'):
            if field not in row:
                raise ValueError(f'{run["directory"]}: stored S7 modal diagnostic {field} is missing')
        diagnostic_rows.append(dict(**identity(run), fs_iteration=50,
                                    loss_pt=row['loss_pt'], raw_coupled_pde_loss=row['loss_pt_coupled_pde'],
                                    y_parallel_relative_L2=row['q_parallel_rel_l2'],
                                    y_perpendicular_relative_L2=row['q_perp_rel_l2']))
    stem = args.output / 'table10_s7_modal_diagnostics'
    write_csv(stem.with_suffix('.csv'), diagnostic_rows)
    write_display(stem, 'S7 FS50 modal diagnostics and objective values',
                  ['Method', 'Loss pT', 'Raw coupled PDE loss', 'y_parallel relative L2', 'y_perp relative L2'],
                  [[row['key'], *[f'{row[k]:.2e}' for k in ('loss_pt', 'raw_coupled_pde_loss',
                                                        'y_parallel_relative_L2', 'y_perpendicular_relative_L2')]]
                   for row in diagnostic_rows],
                  'The raw and modal loss_pt values use different residual metrics and are not directly comparable. '
                  'Stored q_parallel and q_perp are the manuscript y_parallel and y_perp variables.')
    write_csv(args.output / 'group5_all_runs_FS50.csv', [accuracy_row(run, 50) for run in runs])
    write_csv(args.output / 'group5_FS100_supplement.csv',
              [accuracy_row(run, 100) for run in runs if 100 in run['history']])
    maps = [dict(**identity(run), paper_evaluation_sweep=50 if run['config']['role'] == 'paper' else '',
                 historical_source=run['config']['original_run_name']) for run in runs]
    write_csv(args.output / 'group5_run_map.csv', maps)
    write_csv(args.output / 'group5_solver_diagnostics_FS50.csv',
              [row for run in runs for row in solver_rows(run)])
    tta = []
    for run in runs:
        with (run['directory'] / 'time_to_accuracy.csv').open(newline='') as handle:
            for row in csv.DictReader(handle):
                row.update(archive_key=run['config']['key'], paper_case_id=run['config']['paper_case_id'],
                           legacy_case_id=run['config']['legacy_case_id'], archive_role=run['config']['role'])
                tta.append(row)
    write_csv(args.output / 'group5_time_to_accuracy.csv', tta)
    report = dict(seed_for_paper_tables=args.seed, complete_runs=len(runs),
                  selected_paper_runs=len(selection(required)), missing_paper_keys=missing,
                  paper_evaluation_sweep=50, historical_case_map={'S10':'S8','S11':'S9','S12':'S10','S13':'S11'},
                  original_runs=maps, source='Stored histories only; no training/core imports.',
                  relative_L2=True, spatial_H1_relative=False)
    (args.output / 'group5_report_manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Group 5: validated {len(runs)} runs; exported Tables 8--11 at FS50 -> {args.output}')


if __name__ == '__main__':
    main()
