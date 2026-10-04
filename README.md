# Financial Econometrics: Volatility, Dependence & Risk Analysis

This project applies financial econometric models to analyze volatility dynamics, cross-asset dependence, and market risk. It implements GARCH-family models, DCC-GARCH, copulas, and Value-at-Risk backtesting on BMW, Intesa SanPaolo, and ENI stock returns (2005–2024).

---

## 🔍 Project Summary

### 1️⃣ Univariate Volatility Modelling
- Explored stylized facts of returns: non-normality, volatility clustering & leverage.
- Estimated and compared: ARCH(20), GARCH(1,1), GJR-GARCH, EGARCH (Gaussian & Student-t).
- **Best model:** Student-t **EGARCH(1,1)** — captures leverage and fat tails with smoother volatility.

### 2️⃣ Multivariate Dependence
- Models: **RiskMetrics (EWMA)**, **DCC-GARCH**, **O-GARCH**.
- DCC-GARCH provides the most realistic time-varying correlations.
- Strongest dependence found between **ENI ↔ Intesa SanPaolo**.
- Student-t copula used to estimate non-linear and tail dependence.

### 3️⃣ Value-at-Risk Forecasting
- Daily VaR forecasting with GARCH, EGARCH, Student-t GARCH, and GJR-GARCH (rolling estimation).
- Benchmarked against RiskMetrics using:
  - Violations count
  - Total Loss function
  - Diebold-Mariano test
  - MSFE for volatility forecast
- **Gaussian EGARCH** showed strongest overall performance, especially during COVID-19 and the Russia-Ukraine crisis.

---

## 🧠 Key Skills
- Time-series modelling (GARCH family)
- DCC-GARCH & copula dependence modelling
- VaR estimation, backtesting & forecast evaluation
- Model comparison (AIC, DM test, MSFE, diagnostics)

## Python version

The `python-conversion` branch replaces both MATLAB files with Python. The
original MATLAB sources remain available in Git history. The assignment PDF is
retained as a reference. Its empty four-byte placeholder has been replaced with
the original readable report found alongside the local assignment.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python code_assignment.py --csv BMW.csv INTESA.csv ENI.csv
pytest -q
```

The required original inputs are bundled in `data/`: `BMW.DE.csv`,
`ISP.MI.csv`, `ENI.MI.csv`, and `EuroStoxx50.xlsx`. Run the full analysis with:

```sh
python code_assignment.py --csv data/BMW.DE.csv data/ISP.MI.csv data/ENI.MI.csv
python validation/compare_report.py results/summary.json
```

Alternative CSVs can use `Date` (or `Dates`) and `Close` columns; otherwise the
first and fifth columns are used. The first asset is the univariate subject.
Explicit file selection replaces MATLAB's random spreadsheet selection, since
NumPy and MATLAB do not sample identically from the same seed. The CLI defaults
to `--alignment rows`, matching MATLAB's positional pairing: 4,937 returns and
1,437 forecasts. Because the exchange calendars differ, row pairing can combine
different dates. `--alignment dates` uses same-date observations and retains
4,895 returns. Dates are sorted, with missing prices excluded without bridging
missing observations. The Python `load_prices()` API defaults to date alignment;
pass `alignment="rows"` to reproduce the assignment.
Defaults select physical CSV lines 1307–6244 (header counted), matching the
assignment's intended range. Use `--start-row 0` for all rows.

The default rolling estimation window is 3500 observations, with refitting every
22 observations. For shorter datasets, use e.g. `--window 250 --refit 22`.
`--max-arch 20`, `--probability .05`, `--seed 349063`, and `--output results`
control the remaining analysis settings. A full run fits many models and may
require substantial time. No market data are downloaded automatically.

Outputs include PNG plots, univariate conditional variances and residuals,
ARCH order selection, robust variance-ratio tests, rolling forecast variances,
VaR CSVs, a time-first multivariate covariance NPZ file, and `summary.json`
containing model parameters, moments, copula estimates, and full/COVID/war
backtests (violations, quantile loss, Diebold–Mariano tests, and MSFE).

### Translation details and corrections

- `arch` replaces MATLAB Econometrics Toolbox; SciPy supplies likelihood
  optimization, Student-t copulas, and distribution functions. The absent
  `fgarch11t_fit` helper is implemented in Python.
- Univariate estimation uses zero mean, while the four primary rolling VaR
  models estimate a constant mean, as in the source's `Offset = NaN` setup.
- DCC uses each asset's own fitted variance instead of reusing asset 1's model.
  The helper preserves the source's logistic parameter transform and omitted
  likelihood constant, using linear solves for numerical stability.
- Student-t tail dependence uses the correct CDF formula rather than the
  source's PDF and nested square roots. Copulas use joint maximum likelihood
  rather than MATLAB's approximate ML, so fitted parameters may differ.
- Rolling Gaussian GJR forecasts use the fitted GJR model (the source used a
  stale GARCH model). Crisis masks apply to actual out-of-sample dates, and
  crisis DM statistics use their own sample sizes and an unrestricted HAC lag
  calculation. Degenerate DM comparisons are represented as JSON null.
- The source's extra EWMA update in its VaR benchmark is retained explicitly.
- EGARCH multi-step forecasts use seeded simulation (5000 paths); other
  multi-step forecasts are analytic. Business-day forecast dates exclude
  weekends but do not incorporate exchange-specific holidays.
- LOWESS replaces MATLAB LOESS. Distribution diagnostics use fitted Student-t
  degrees of freedom with variance normalization instead of a fixed value 6.
  `matlab_initialization.py` reproduces MATLAB's average-square presample
  variance, positive GARCH/GJR presample shocks, and zero EGARCH presample
  shocks. Solver tolerances, constraints, and copula methods still differ, so
  not every estimate is numerically identical. DCC uses an analytic gradient
  to avoid finite-difference precision-loss termination.

Validation uses independent DCC and EWMA recursions, missing-data alignment,
rolling forecasts without future-data leakage, copula parameter bounds, and a
synthetic end-to-end run. The full bundled-data run is compared with the report in
`validation/MATLAB_COMPARISON.md`; every transcribed numerical comparison is in
`validation/comparison.csv`. The reference JSON preserves the printed report
values, including apparent typos. MATLAB was not available locally, so this
checks the saved report rather than a fresh MATLAB execution.
