import numpy as np
import pandas as pd
from scipy.special import logit
from fDCC_LogLikelihood import fDCC_LogLikelihood, dcc_correlations
from code_assignment import riskmetrics, dm_test, fit_copula, rolling_forecasts, load_prices, evaluate


def test_dcc_matches_direct_matlab_recursion():
    z = np.random.default_rng(7).normal(size=(60, 3))
    qbar = np.cov(z, rowvar=False, bias=True)
    a, b = .07, .85
    psi = [logit(a), logit(b/(1-a))]
    actual, da, db = fDCC_LogLikelihood(z, qbar, psi)
    q, expected = qbar.copy(), 0
    for i, row in enumerate(z):
        if i: q = (1-a-b)*qbar+a*np.outer(z[i-1], z[i-1])+b*q
        scale = np.diag(1/np.sqrt(np.diag(q)))
        p = scale@q@scale
        expected += .5*(np.log(np.linalg.det(p))+row@np.linalg.inv(p)@row)
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose([da, db], [a, b])
    np.testing.assert_allclose(np.diagonal(dcc_correlations(z, qbar, a, b), axis1=1, axis2=2), 1)


def test_ewma_recursion():
    r = np.array([[1., 2.], [-1., 3.]])
    h = riskmetrics(r)
    np.testing.assert_allclose(h[0], .06*np.outer(r[0], r[0]))
    np.testing.assert_allclose(h[1], .94*h[0]+.06*np.outer(r[1], r[1]))


def test_missing_prices_do_not_bridge_and_dates_align(tmp_path):
    dates = pd.bdate_range('2020-01-01', periods=50)
    for i in range(3):
        prices = np.exp(np.arange(50)*.01+i)
        prices[10] = np.nan
        pd.DataFrame({'Date': dates, 'Close': prices}).to_csv(tmp_path/f'{i}.csv', index=False)
    _, r = load_prices([tmp_path/f'{i}.csv' for i in range(3)], 0)
    assert dates[10] not in r.index and dates[11] not in r.index
    np.testing.assert_allclose(r, 1.)


def test_rolling_forecasts_have_no_future_leakage():
    r = np.random.default_rng(12).normal(size=(66, 1))
    baseline = rolling_forecasts(r, 60, 4)
    changed = r.copy(); changed[-1] = 100
    modified = rolling_forecasts(changed, 60, 4)
    for name, (h, var) in baseline.items():
        assert h.shape == var.shape == (6, 1)
        assert np.all(np.isfinite(h)) and np.all(h > 0)
        np.testing.assert_allclose(h, modified[name][0])
        np.testing.assert_allclose(var, modified[name][1])
    metrics = evaluate(r[60:], baseline)
    assert metrics['RiskMetrics']['violations'].shape == (1,)


def test_dm_and_copula():
    statistic, p = dm_test(np.zeros((30, 2)))
    assert np.isnan(statistic).all()
    u = np.random.default_rng(9).uniform(.01, .99, (100, 2))
    rho, nu, tail = fit_copula(u)
    assert -1 < rho < 1 and nu > 2 and 0 <= tail <= 1
