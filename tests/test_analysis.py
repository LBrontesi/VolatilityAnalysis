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
    innovations = np.random.default_rng(12).normal(size=606)
    r = np.empty((606, 1))
    variance, previous = 1., 0.
    for i, innovation in enumerate(innovations):
        variance = .1+.12*previous**2+.78*variance
        previous = r[i, 0] = np.sqrt(variance)*innovation
    baseline = rolling_forecasts(r, 600, 4)
    changed = r.copy(); changed[-1] = 100
    modified = rolling_forecasts(changed, 600, 4)
    for name, (h, var) in baseline.items():
        assert h.shape == var.shape == (6, 1)
        assert np.all(np.isfinite(h)) and np.all(h > 0)
        np.testing.assert_allclose(h, modified[name][0])
        np.testing.assert_allclose(var, modified[name][1])
    metrics = evaluate(r[600:], baseline)
    assert metrics['RiskMetrics']['violations'].shape == (1,)


def test_dm_and_copula():
    statistic, p = dm_test(np.zeros((30, 2)))
    assert np.isnan(statistic).all()
    u = np.random.default_rng(9).uniform(.01, .99, (100, 2))
    rho, nu, tail = fit_copula(u)
    assert -1 < rho < 1 and nu > 2 and 0 <= tail <= 1


def test_report_data_and_matlab_moments():
    from pathlib import Path
    from code_assignment import matlab_moments
    paths = [Path(__file__).resolve().parents[1]/'data'/name
             for name in ('BMW.DE.csv', 'ISP.MI.csv', 'ENI.MI.csv')]
    _, r = load_prices(paths, alignment='rows')
    assert len(r) == 4937
    moments = matlab_moments(r.to_numpy())
    # Reference: assignment PDF pages 3 and 21, rounded to four decimals.
    np.testing.assert_allclose(moments['variance'], [3.7183, 5.8282, 3.0879], atol=5e-5, rtol=0)
    np.testing.assert_allclose(moments['kurtosis'], [8.6460, 12.6799, 20.0530], atol=5e-5, rtol=0)
    np.testing.assert_allclose(moments['skewness'][0], -.0920, atol=5e-5, rtol=0)
    _, date_aligned = load_prices(paths, alignment='dates')
    assert len(date_aligned) == 4895


def test_matlab_presample_variance_recursions():
    from matlab_initialization import AssignmentGARCH, AssignmentEGARCH
    r = np.array([1., -2., .5, 1.5, -.8])
    v0 = np.mean(r*r)
    for process, parameters in [(AssignmentGARCH(p=1, o=0, q=1), [.1, .1, .8]),
                                (AssignmentGARCH(p=1, o=1, q=1), [.1, .1, .05, .8])]:
        omega, alpha, beta = parameters[0], parameters[1], parameters[-1]
        gamma = parameters[2] if len(parameters) == 4 else 0
        h, e = v0, np.sqrt(v0)
        expected = []
        for observation in r:
            h = omega+(alpha+gamma*(e < 0))*e**2+beta*h
            expected.append(h); e = observation
        actual = process.compute_variance(np.array(parameters), r, np.empty(len(r)),
                                          process.backcast(r), process.variance_bounds(r))
        np.testing.assert_allclose(actual, expected)
    for beta in (.9, 0.):
        parameters = np.array([.05, .12, -.04, beta])
        h, z = v0, 0.
        expected = []
        for observation in r:
            h = np.exp(.05+.12*(abs(z)-np.sqrt(2/np.pi))-.04*z+beta*np.log(h))
            expected.append(h); z = observation/np.sqrt(h)
        process = AssignmentEGARCH(p=1, o=1, q=1)
        actual = process.compute_variance(parameters, r, np.empty(len(r)),
                                          process.backcast(r), process.variance_bounds(r))
        np.testing.assert_allclose(actual, expected)


def test_dcc_analytic_gradient():
    from scipy.optimize import check_grad
    from fDCC_LogLikelihood import dcc_objective_gradient
    z = np.random.default_rng(51).normal(size=(40, 3))
    qbar = np.cov(z, rowvar=False, bias=True)
    psi = np.array([-3., 2.])
    value, gradient = dcc_objective_gradient(z, qbar, psi)
    np.testing.assert_allclose(value, fDCC_LogLikelihood(z, qbar, psi)[0])
    error = check_grad(lambda x: fDCC_LogLikelihood(z, qbar, x)[0],
                       lambda x: dcc_objective_gradient(z, qbar, x)[1], psi, epsilon=1e-6)
    assert error < 1e-5
