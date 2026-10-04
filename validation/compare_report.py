"""Compare a completed Python run with the transcribed assignment tables.

python validation/compare_report.py results/summary.json --output validation
This compares rounded report values, not fresh MATLAB output. It records all
mismatches without changing the numerical results to satisfy the reference.
"""
import argparse
import csv
import json
from pathlib import Path


def compare(summary, reference):
    rows = []
    def add(section, metric, model, asset, expected, actual):
        difference = actual-expected if actual is not None else None
        tolerance = 0 if metric in ('observations', 'violations') else 5e-5
        rows.append(dict(section=section, metric=metric, model=model, asset=asset,
                         report=expected, python=actual, difference=difference,
                         matches_report_precision=difference is not None and abs(difference) <= tolerance))
    add('data', 'observations', '', '', reference['observations'], summary['observations'])
    for metric in ('variance', 'kurtosis'):
        for asset, expected, actual in zip(reference['asset_order'], reference['moments'][metric], summary['moments'][metric]):
            add('moments', metric, '', asset, expected, actual)
    add('moments', 'skewness', '', 'BMW.DE', reference['moments']['bmw_skewness'], summary['moments']['skewness'][0])
    for model, expected in reference['aic'].items():
        add('univariate', 'aic', model, 'BMW.DE', expected, summary['univariate'][model]['aic'])
    for key, expected in reference['garch_parameters_as_printed'].items():
        add('univariate', key, 'GARCH', 'BMW.DE', expected, summary['univariate']['GARCH']['parameters'][key])
    for key, expected in reference['dcc'].items():
        add('dcc', key, '', '', expected, summary['dcc'][key])
    for pair, values in reference['copulas'].items():
        for metric, expected in values.items():
            add('copula', metric, '', pair, expected, summary['copulas'][pair][metric])
    for period, models in reference['backtests'].items():
        for model, metrics in models.items():
            for metric, values in metrics.items():
                for i, expected in enumerate(values):
                    add(period, metric, model, reference['asset_order'][i], expected,
                        summary['backtests'][period][model][metric][i])
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary', type=Path)
    parser.add_argument('--output', type=Path, default=Path('validation'))
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    reference = json.loads(Path(__file__).with_name('matlab_reference.json').read_text())
    rows = compare(summary, reference)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    (args.output/'python_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    matched = sum(r['matches_report_precision'] for r in rows)
    print(f'{matched}/{len(rows)} values match report precision; see comparison.csv for differences.')
