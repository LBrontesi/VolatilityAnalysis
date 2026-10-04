"""Re-evaluate specific source expressions to explain report discrepancies.

This is a Python emulation of the original MATLAB formulas, not a MATLAB run.
It deliberately recreates errors only in this validation script; production
analysis retains the documented corrections.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy import optimize, stats
from code_assignment import load_prices, make_model, fit_model
from fDCC_LogLikelihood import fDCC_LogLikelihood, dcc_objective_gradient


def verify(results, output):
    root = Path(__file__).resolve().parents[1]
    _, returns = load_prices([root/'data'/name for name in ('BMW.DE.csv', 'ISP.MI.csv', 'ENI.MI.csv')], alignment='rows')
    summary = json.loads((results/'summary.json').read_text())
    reference = json.loads((root/'validation/matlab_reference.json').read_text())
    if summary['alignment'] != 'rows' or len(returns)-3500 != len(pd.read_csv(results/'VaR_RiskMetrics.csv')):
        raise ValueError('Requires the default 3500-observation, row-aligned analysis')
    evidence = {'method': 'Python emulation of source expressions; not a fresh MATLAB run'}
    r = returns.to_numpy()
    first = fit_model(r[:, 0])
    # Source infer(EstMdl_asset(:,1), r_t(:,i)) uses the first asset's model.
    variances = np.column_stack([make_model(r[:, i]).fix(first.params).conditional_volatility**2 for i in range(3)])
    z = r/np.sqrt(variances)
    qbar = np.cov(z, rowvar=False, bias=True)
    fit = optimize.minimize(lambda psi: dcc_objective_gradient(z, qbar, psi),
                            [np.log(.06/.94), np.log(.94/.06)], jac=True,
                            method='BFGS', options={'gtol': 1e-4, 'maxiter': 1000})
    _, a, b = fDCC_LogLikelihood(z, qbar, fit.x)
    np.testing.assert_allclose([a, b], [.0160, .9782], atol=5e-5, rtol=0)
    evidence['dcc_reused_first_model'] = {'a': a, 'b': b, 'converged': bool(fit.success)}
    evidence['source_tail_expression'] = {}
    for pair, copula in summary['copulas'].items():
        rho, nu = copula['rho'], copula['nu']
        w = -np.sqrt((nu+1)*np.sqrt(1-rho)/np.sqrt(1+rho))
        tail = float(2*stats.t.pdf(w, nu+1))
        np.testing.assert_allclose(tail, reference['copulas'][pair]['tail_dependence'], atol=5e-5, rtol=0)
        evidence['source_tail_expression'][pair] = tail
    variance = pd.read_csv(results/'forecast_variance_RiskMetrics.csv', index_col=0).to_numpy()
    var = pd.read_csv(results/'VaR_RiskMetrics.csv', index_col=0).to_numpy()
    actual = r[3500:]
    loss = (.05-(actual < -var))*(actual+var)
    evidence['source_crisis_indexing'] = {}
    for name, start, end in [('covid', '2020-02-01', '2020-09-01'), ('war', '2022-02-01', '2022-09-01')]:
        mask = (returns.index >= start) & (returns.index <= end)
        count = int(mask.sum())
        # Source k(k) is an all-true short mask, so it selects the first
        # count forecasts. MSFE separately uses actual crisis observations.
        values = {'observations': count,
                  'wrong_forecast_start': str(returns.index[3500].date()),
                  'wrong_forecast_end': str(returns.index[3500+count-1].date()),
                  'violations': (actual[:count] < -var[:count]).sum(axis=0).tolist(),
                  'total_loss': loss[:count].sum(axis=0).tolist(),
                  'mixed_msfe': np.mean((r[mask]**2-variance[:count])**2, axis=0).tolist()}
        for key, source_key in [('violations', 'violations'), ('total_loss', 'total_loss'), ('mixed_msfe', 'msfe')]:
            np.testing.assert_allclose(values[key], reference['backtests'][name]['RiskMetrics'][source_key], atol=5e-5, rtol=0)
        evidence['source_crisis_indexing'][name] = values
    output.write_text(json.dumps(evidence, indent=2)+'\n')
    print('Source-error emulation matches report DCC, all tail coefficients, and crisis RiskMetrics tables.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('--output', type=Path, default=Path('validation/source_bug_checks.json'))
    args = parser.parse_args()
    verify(args.results, args.output)
