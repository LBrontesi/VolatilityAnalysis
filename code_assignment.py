"""Python port of the financial econometrics assignment.

Run python code_assignment.py --help for data and rolling-window options.
All returns and VaR use percentage units. Covariance arrays are time-first.
"""
import argparse
import json
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch import arch_model
from scipy import optimize, signal, stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.nonparametric.smoothers_lowess import lowess
from fDCC_LogLikelihood import fDCC_LogLikelihood, dcc_correlations


def fit_model(r, kind="GARCH", distribution="normal", p=1, mean=False):
    model = arch_model(np.asarray(r), mean="Constant" if mean else "Zero",
                       vol="EGARCH" if kind == "EGARCH" else "GARCH",
                       p=p, o=int(kind in ("GJR", "EGARCH")),
                       q=0 if kind == "ARCH" else 1, dist=distribution, rescale=False)
    result = model.fit(disp="off", show_warning=False)
    if result.convergence_flag:
        warnings.warn(f"{kind}/{distribution} optimizer did not converge")
    return result


def fgarch11t_fit(r):
    result = fit_model(r, distribution="t")
    return result, np.asarray(result.std_resid), np.asarray(result.conditional_volatility)


def load_prices(paths, start_row=1307, end_row=6244):
    series = []
    for path in paths:
        table = pd.read_csv(path)
        date_column = next((c for c in table if str(c).lower() in ("date", "dates")), table.columns[0])
        close_column = next((c for c in table if str(c).lower() == "close"), table.columns[min(4, len(table.columns)-1)])
        table = table.iloc[start_row-2:end_row-1] if start_row else table
        dates = pd.to_datetime(table[date_column], errors="coerce")
        prices = pd.to_numeric(table[close_column], errors="coerce")
        if (prices.dropna() <= 0).any():
            raise ValueError(f"{path}: prices must be positive")
        s = pd.Series(prices.to_numpy(), index=dates, name=Path(path).stem)
        s = s.loc[~s.index.isna()].sort_index()
        if s.index.has_duplicates:
            raise ValueError(f"{path}: duplicate dates")
        series.append(s)
    # Keep missing prices through differencing to avoid invented multi-day returns.
    prices = pd.concat(series, axis=1).sort_index()
    returns = (100*np.log(prices).diff()).dropna()
    if len(returns) < 30:
        raise ValueError("Need at least 30 aligned returns; check row range")
    return prices, returns


def variance_ratio(log_prices):
    y = np.asarray(log_prices)
    increments = np.diff(y)
    n = len(increments)
    centered = increments-increments.mean()
    denominator = centered @ centered
    rows = []
    for k in range(2, int(len(y)**(1/3))+1):
        overlapping = y[k:]-y[:-k]-k*increments.mean()
        variance = overlapping @ overlapping / (k*(n-k+1)*(1-k/n))
        vr = variance/(denominator/(n-1))
        theta = sum((2*(k-j)/k)**2 *
                    np.sum(centered[j:]**2*centered[:-j]**2)/denominator**2
                    for j in range(1, k))
        statistic = (vr-1)/np.sqrt(theta) if theta > 0 else np.nan
        rows.append((k, vr, statistic, 2*stats.norm.sf(abs(statistic))))
    return pd.DataFrame(rows, columns=["period", "ratio", "statistic", "pvalue"])


def riskmetrics(r, weight=.06):
    r = np.asarray(r)
    products = r[:, :, None]*r[:, None, :]
    return signal.lfilter([weight], [1, -(1-weight)], products, axis=0)


def correlations(covariance):
    sd = np.sqrt(np.diagonal(covariance, axis1=1, axis2=2))
    return covariance/(sd[:, :, None]*sd[:, None, :])


def fit_copula(u):
    u = np.clip(np.asarray(u), 1e-10, 1-1e-10)
    def objective(parameters):
        rho, nu = np.tanh(parameters[0]), 2+np.exp(parameters[1])
        x = stats.t.ppf(u, nu)
        joint = stats.multivariate_t.logpdf(x, shape=[[1, rho], [rho, 1]], df=nu)
        return -np.sum(joint-stats.t.logpdf(x, nu).sum(axis=1))
    initial = np.clip(np.corrcoef(stats.norm.ppf(u).T)[0, 1], -.95, .95)
    fit = optimize.minimize(objective, [np.arctanh(initial), np.log(8)],
                            method="L-BFGS-B", bounds=[(-4, 4), (-5, np.log(198))])
    if not fit.success:
        warnings.warn(f"Copula optimization: {fit.message}")
    rho, nu = np.tanh(fit.x[0]), 2+np.exp(fit.x[1])
    tail = 2*stats.t.cdf(-np.sqrt((nu+1)*(1-rho)/(1+rho)), nu+1)
    return float(rho), float(nu), float(tail)


def dm_test(difference):
    d = np.asarray(difference)
    n = len(d)
    if n < 3:
        return np.full(d.shape[1:], np.nan), np.full(d.shape[1:], np.nan)
    centered = d-d.mean(axis=0)
    q = max(1, int(n**(1/3)))
    lrv = np.mean(centered**2, axis=0)
    for lag in range(1, q):
        lrv += 2*(1-lag/q)*np.sum(centered[lag:]*centered[:-lag], axis=0)/n
    statistic = np.divide(d.mean(axis=0), np.sqrt(np.maximum(lrv, 0)/n),
                          out=np.full_like(lrv, np.nan), where=lrv > 0)
    return statistic, 2*stats.norm.sf(abs(statistic))


ROLLING_MODELS = {"GARCH_normal": ("GARCH", "normal", True),
                  "EGARCH_normal": ("EGARCH", "normal", True),
                  "GARCH_t": ("GARCH", "t", True),
                  "GJR_t": ("GJR", "t", True),
                  "GJR_normal": ("GJR", "normal", False)}


def rolling_forecasts(r, window=3500, refit=22, probability=.05):
    r = np.asarray(r)
    if not 30 <= window < len(r) or refit < 1 or not 0 < probability < .5:
        raise ValueError("Need 30 <= window < sample size, positive refit, and 0 < probability < .5")
    n, assets = len(r)-window, r.shape[1]
    output = {}
    for name, (kind, distribution, mean) in ROLLING_MODELS.items():
        variance, value_at_risk = np.empty((n, assets)), np.empty((n, assets))
        for asset in range(assets):
            for block in range(0, n, refit):
                fit = fit_model(r[block:block+window, asset], kind, distribution, mean=mean)
                for offset in range(block, min(block+refit, n)):
                    # Refilter each moving window with parameters fixed until next refit.
                    model = arch_model(r[offset:offset+window, asset],
                        mean="Constant" if mean else "Zero", vol="EGARCH" if kind == "EGARCH" else "GARCH",
                        p=1, o=int(kind in ("GJR", "EGARCH")), q=1, dist=distribution, rescale=False)
                    filtered = model.fix(fit.params)
                    h = float(filtered.forecast(horizon=1, reindex=False).variance.iloc[-1, 0])
                    mu = float(fit.params.get("mu", 0))
                    quantile = stats.norm.ppf(probability)
                    if distribution == "t":
                        nu = fit.params["nu"]
                        quantile = stats.t.ppf(probability, nu)*np.sqrt((nu-2)/nu)
                    variance[offset, asset] = h
                    value_at_risk[offset, asset] = -(mu+quantile*np.sqrt(h))
        output[name] = (variance, value_at_risk)
    h = riskmetrics(r)
    # Retain the source's extra EWMA update when forming its benchmark.
    variance = .06*r[window-1:-1]**2+.94*np.diagonal(h[window-1:-1], axis1=1, axis2=2)
    output["RiskMetrics"] = (variance, -stats.norm.ppf(probability)*np.sqrt(variance))
    return output


def evaluate(actual, forecasts, probability=.05):
    losses = {name: (probability-(actual < -var))*(actual+var)
              for name, (_, var) in forecasts.items()}
    result = {}
    for name, (variance, var) in forecasts.items():
        dm, pvalue = dm_test(losses[name]-losses["RiskMetrics"])
        result[name] = {"violations": (actual < -var).sum(axis=0),
                        "expected_violations": probability*len(actual),
                        "total_loss": losses[name].sum(axis=0),
                        "msfe": np.mean((actual**2-variance)**2, axis=0),
                        "dm": dm, "dm_pvalue": pvalue}
    return result


def save_plot(output, name):
    plt.tight_layout()
    plt.savefig(output/f"{name}.png", dpi=150)
    plt.close()


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    prices, returns = load_prices(args.csv, args.start_row, args.end_row)
    r = returns.iloc[:, 0].to_numpy()
    summary = {"assets": list(returns.columns), "observations": len(returns),
               "moments": {"mean": returns.mean().to_numpy(), "variance": returns.var(ddof=0).to_numpy(),
                           "skewness": stats.skew(returns, axis=0), "kurtosis": stats.kurtosis(returns, axis=0, fisher=False)},
               "jarque_bera": list(stats.jarque_bera(r))}
    prices.plot(title="Prices"); save_plot(output, "prices")
    returns.plot(title="Percentage log returns"); save_plot(output, "returns")
    for name, data in (("returns", r), ("absolute", abs(r)), ("squared", r**2)):
        plot_acf(data, lags=min(20, len(r)-1), title=f"ACF {name}"); save_plot(output, f"acf_{name}")
    plt.hist(r, bins=40, density=True)
    grid = np.linspace(r.min(), r.max(), 200)
    plt.plot(grid, stats.norm.pdf(grid, r.mean(), r.std()))
    save_plot(output, "histogram")
    vr = variance_ratio(np.log(prices.iloc[:, 0].dropna()))
    vr.to_csv(output/"variance_ratio.csv", index=False)
    vr.plot(x="period", y="pvalue", style="r*"); plt.axhline(.05); save_plot(output, "variance_ratio")
    arch_fits = [fit_model(r, "ARCH", p=p) for p in range(1, args.max_arch+1)]
    aic = np.array([fit.aic for fit in arch_fits])
    pd.DataFrame({"p": np.arange(1, len(aic)+1), "aic": aic}).to_csv(output/"arch_order.csv", index=False)
    plt.plot(np.arange(1, len(aic)+1), aic, "r*"); save_plot(output, "arch_order")
    models = {"ARCH": arch_fits[int(aic.argmin())], "GARCH": fit_model(r)}
    for kind in ("GJR", "EGARCH"):
        for distribution in ("normal", "t"):
            models[f"{kind}_{distribution}"] = fit_model(r, kind, distribution)
    summary["univariate"] = {}
    for name, fit in models.items():
        volatility = np.asarray(fit.conditional_volatility)
        residuals = np.asarray(fit.std_resid)
        forecast = fit.forecast(horizon=22, method="simulation" if "EGARCH" in name else "analytic",
                                simulations=5000, rng=(lambda size, nu=fit.params.get("nu", None), generator=np.random.default_rng(args.seed): generator.standard_normal(size) if nu is None else generator.standard_t(nu, size)*np.sqrt((nu-2)/nu)),
                                reindex=False).variance.iloc[-1].to_numpy()
        summary["univariate"][name] = {"parameters": fit.params.to_dict(), "aic": fit.aic,
                                       "forecast_variance": forecast, "jarque_bera": list(stats.jarque_bera(residuals))}
        pd.DataFrame({"variance": volatility**2, "standardized_residual": residuals}, index=returns.index).to_csv(output/f"{name}.csv")
        plt.plot(returns.index, volatility**2)
        plt.plot(pd.bdate_range(returns.index[-1]+pd.offsets.BDay(), periods=22), forecast)
        plt.title(f"{name} conditional variance and forecast"); save_plot(output, f"variance_{name}")
        fig, axes = plt.subplots(2, 1)
        plot_acf(residuals, ax=axes[0]); plot_acf(residuals**2, ax=axes[1]); save_plot(output, f"diagnostics_{name}")
        stats.probplot(residuals*np.sqrt(fit.params["nu"]/(fit.params["nu"]-2)) if "_t" in name else residuals, dist=stats.t if "_t" in name else stats.norm,
                       sparams=(fit.params["nu"],) if "_t" in name else (), plot=plt)
        save_plot(output, f"qq_{name}")
    smooth = lowess(r[1:]**2, r[:-1], frac=.95)
    plt.scatter(r[:-1], r[1:]**2, s=2); plt.plot(smooth[:, 0], smooth[:, 1], "r")
    save_plot(output, "volatility_smile")
    cross = signal.correlate(r**2-np.mean(r**2), r-r.mean(), mode="full")/(len(r)*r.std()*np.std(r**2))
    lags = signal.correlation_lags(len(r), len(r)); select = abs(lags) <= 20
    plt.stem(lags[select], cross[select]); save_plot(output, "leverage_crosscorrelation")
    matrix = returns.to_numpy()
    fits = [fit_model(matrix[:, i]) for i in range(matrix.shape[1])]
    variances = np.column_stack([f.conditional_volatility**2 for f in fits])
    standardized = matrix/np.sqrt(variances)
    qbar = np.cov(standardized, rowvar=False, bias=True)
    dcc = optimize.minimize(lambda psi: fDCC_LogLikelihood(standardized, qbar, psi)[0],
                            [np.log(.06/.94), np.log(.94/.06)], method="BFGS",
                            options={"maxiter": 1000, "gtol": 1e-4})
    if not dcc.success:
        warnings.warn(f"DCC optimizer: {dcc.message}")
    likelihood, a, b = fDCC_LogLikelihood(standardized, qbar, dcc.x)
    corr = dcc_correlations(standardized, qbar, a, b)
    cov = corr*np.sqrt(variances[:, :, None]*variances[:, None, :])
    _, vectors = np.linalg.eigh(np.cov(matrix, rowvar=False))
    vectors = vectors[:, ::-1]
    factors = matrix @ vectors
    factor_variance = np.column_stack([fit_model(factors[:, i]).conditional_volatility**2 for i in range(matrix.shape[1])])
    ogarch = np.einsum("ik,tk,jk->tij", vectors, factor_variance, vectors)
    summary["dcc"] = {"a": a, "b": b, "negative_loglikelihood": likelihood, "converged": bool(dcc.success)}
    np.savez_compressed(output/"multivariate.npz", riskmetrics=riskmetrics(matrix), dcc=cov, dcc_correlations=corr, ogarch=ogarch)
    for name, covariance in (("RiskMetrics", riskmetrics(matrix)), ("DCC", cov), ("OGARCH", ogarch)):
        for metric, values in (("variance", np.diagonal(covariance, axis1=1, axis2=2)),
                               ("covariance", np.column_stack([covariance[:, i, j] for i in range(matrix.shape[1]) for j in range(i+1, matrix.shape[1])])),
                               ("correlation", np.column_stack([correlations(covariance)[:, i, j] for i in range(matrix.shape[1]) for j in range(i+1, matrix.shape[1])]))):
            plt.plot(returns.index, values); plt.title(f"{name} {metric}"); save_plot(output, f"{name}_{metric}")
    uniform = []
    for i in range(matrix.shape[1]):
        fit, residual, _ = fgarch11t_fit(matrix[:, i])
        nu = fit.params["nu"]
        uniform.append(stats.t.cdf(residual*np.sqrt(nu/(nu-2)), nu))
    uniform = np.column_stack(uniform)
    summary["copulas"] = {f"{returns.columns[i]} / {returns.columns[j]}": dict(zip(("rho", "nu", "tail_dependence"), fit_copula(uniform[:, [i, j]])))
                           for i in range(matrix.shape[1]) for j in range(i+1, matrix.shape[1])}
    forecasts = rolling_forecasts(matrix, args.window, args.refit, args.probability)
    actual = matrix[args.window:]
    dates = returns.index[args.window:]
    summary["backtests"] = {"full": evaluate(actual, forecasts, args.probability)}
    for name, start, end in (("covid", "2020-02-01", "2020-09-01"), ("war", "2022-02-01", "2022-09-01")):
        mask = (dates >= start) & (dates <= end)
        summary["backtests"][name] = evaluate(actual[mask], {k: (h[mask], v[mask]) for k, (h, v) in forecasts.items()}, args.probability) if mask.any() else {"observations": 0}
    for name, (h, var) in forecasts.items():
        pd.DataFrame(h, index=dates, columns=returns.columns).to_csv(output/f"forecast_variance_{name}.csv")
        pd.DataFrame(var, index=dates, columns=returns.columns).to_csv(output/f"VaR_{name}.csv")
    plt.plot(dates, actual)
    for start, end in (("2020-02-01", "2020-09-01"), ("2022-02-01", "2022-09-01")):
        plt.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=.15)
    save_plot(output, "crisis_returns")
    def serializable(value):
        if isinstance(value, np.ndarray): return [serializable(v) for v in value.tolist()]
        if isinstance(value, dict): return {k: serializable(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)): return [serializable(v) for v in value]
        if isinstance(value, (float, np.floating)): return float(value) if np.isfinite(value) else None
        if isinstance(value, np.integer): return int(value)
        return value
    (output/"summary.json").write_text(json.dumps(serializable(summary), indent=2, allow_nan=False))
    print(f"Saved analysis to {output.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", nargs=3, required=True, help="Three asset CSVs, first asset used for univariate analysis")
    parser.add_argument("--start-row", type=int, default=1307, help="First physical CSV line, including header; 0 uses all rows")
    parser.add_argument("--end-row", type=int, default=6244)
    parser.add_argument("--window", type=int, default=3500)
    parser.add_argument("--refit", type=int, default=22)
    parser.add_argument("--max-arch", type=int, default=20)
    parser.add_argument("--probability", type=float, default=.05)
    parser.add_argument("--seed", type=int, default=349063)
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    if args.max_arch < 1: parser.error("--max-arch must be positive")
    run(args)
