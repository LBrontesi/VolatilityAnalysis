# Comparison with the MATLAB assignment report

Checked on 2026-10-04 using the original BMW, Intesa Sanpaolo, and ENI CSVs.
The report is `Brontesi_Assignment.pdf`. The repository's four-byte placeholder
was replaced with the readable original report recovered alongside the local
MATLAB assignment. No MATLAB executable was available, so this is a comparison
against its printed tables, not a fresh MATLAB execution.

**Verdict: the underlying inputs and main backtest results reproduce the report;
the complete corrected Python analysis is not numerically identical.** All 15
full-sample VaR violation counts match exactly. The return moments, RiskMetrics
loss/MSFE tables, and most Gaussian/Student-t GARCH metrics match the report's
precision. Remaining differences are listed rather than hidden or overwritten.

## Reproduce

```sh
pip install -r requirements.txt
python code_assignment.py --csv data/BMW.DE.csv data/ISP.MI.csv data/ENI.MI.csv
python validation/compare_report.py results/summary.json
python validation/verify_source_bugs.py results
python -m pytest -q
```

`matlab_reference.json` is a transcription of the printed values.
`python_summary.json` contains the full Python run.
`comparison.csv` lists all 192 scalar comparisons with signed differences;
56 match at the strict report-precision tolerance (5e-5, or exact for counts).
Most nonmatches are the intentionally corrected crisis-period tables; close
estimated values are still recorded as nonmatches if they exceed that tolerance.
`source_bug_checks.json` demonstrates several discrepancies by emulating the
original source expressions separately. `environment.txt` records installed
versions for the checked run (Python 3.14).

## Inputs and initial conditions

The default physical CSV lines 1307-6244 produce 4,938 prices, 4,937 returns,
and 1,437 rolling forecasts, using the same asset order as the assignment.
MATLAB combines the assets by row, and the CLI now defaults to `--alignment rows`
to reproduce that convention. Their trading calendars differ: the alternative
`--alignment dates` pairs actual dates and retains 4,895 returns. Row pairing
must not be mistaken for a same-date multivariate dataset.

The initial port used date alignment and arch's weighted backcast. Both changed
its inputs/initial conditions relative to MATLAB. The checked version uses
MATLAB's average-square initial variance and its positive GARCH/GJR and zero
EGARCH initial innovations. Independent recurrence tests verify this behavior.
The defaults are described in MathWorks' documentation for
[estimate](https://www.mathworks.com/help/econ/garch.estimate.html) and
[infer](https://www.mathworks.com/help/econ/garch.infer.html).

Report pages 3 and 21:

| Metric | Asset | Report | Python |
|---|---|---:|---:|
| Variance | BMW | 3.7183 | 3.7183117004 |
| Variance | Intesa | 5.8282 | 5.8282290980 |
| Variance | ENI | 3.0879 | 3.0878569716 |
| Kurtosis | BMW | 8.6460 | 8.6459828902 |
| Kurtosis | Intesa | 12.6799 | 12.6799297125 |
| Kurtosis | ENI | 20.0530 | 20.0529611871 |
| Skewness | BMW | -0.0920 | -0.0919685114 |

Moments use MATLAB's sample-standard-deviation normalization; population
variance remains the mean squared centered returns, as in the source.

## Univariate models

ARCH(20) is the AIC-minimizing order in both implementations. Report page 10:

| Model | Report AIC | Python AIC | Difference |
|---|---:|---:|---:|
| ARCH(20) | 19335.8591 | 19335.8591337 | +0.0000337 |
| GARCH(1,1) | 19260.1805 | 19260.1803642 | -0.0001358 |

The fitted GARCH coefficients are omega=0.0286563885, alpha=0.0479169773,
beta=0.9442450696. The report prints alpha=0.047917 and beta=0.94425.
It prints omega=0.208656, but that is inconsistent with its own standard error
0.0043889 and t-statistic 6.5292: their product is approximately 0.028656.
That suggests a transcription error; the reference file preserves 0.208656.

## Full-sample VaR backtesting

Report pages 36-38. Each cell lists BMW / Intesa / ENI:

| Model | Report violations | Python violations |
|---|---|---|
| Gaussian GARCH | 73 / 62 / 79 | 73 / 62 / 79 |
| Gaussian EGARCH | 70 / 70 / 77 | 70 / 70 / 77 |
| Student-t GARCH | 77 / 68 / 86 | 77 / 68 / 86 |
| Student-t GJR | 75 / 71 / 83 | 75 / 71 / 83 |
| RiskMetrics | 80 / 74 / 84 | 80 / 74 / 84 |

RiskMetrics losses are 323.6212514 / 338.0456695 / 323.3674568, and MSFEs
114.3819362 / 186.6263829 / 375.3183510. They match all six printed values
at four decimals. Student-t GARCH losses also match all three printed values.
Gaussian GARCH MSFEs match all three printed values at four decimals.

For Gaussian GARCH, Intesa's Python total loss is 334.9951355, while the report
prints 335.9951. The other two losses reproduce the printed values. The Intesa
DM p-value and MSFE also reproduce the report's values; a transcription error
in this one loss cell is plausible, but cannot be established without original
MATLAB workspace output. The discrepancy is retained in `comparison.csv`.

EGARCH and Student-t GJR have small estimation differences. For example, BMW
EGARCH total loss is 317.2172853 vs 317.2043 (difference +0.0129853), and its
MSFE is 113.1530115 vs 113.1300 (+0.0230115). Its DM p-value is about 0.0391
vs 0.0383, preserving the 5% rejection. These are not exact numerical matches.

The expected violation count is 71.85; the source floors it to 71, while Python
reports the unfloored expectation.

## Differences explained by original source errors

### DCC, report page 23

The source infers every asset's variance using the first asset's GARCH model.
Python uses each asset's own model. The corrected fit is a=0.0199074650,
b=0.9711907080. Recreating the original reuse separately gives a=0.0159831482,
b=0.9781978260, which round to the report's 0.0160 and 0.9782. Both validation
fits converge. The production DCC objective now uses an analytic gradient,
verified against finite differences, to avoid precision-loss termination.

### Student-t tail dependence, report pages 28-29

The source uses a PDF and nested square roots where the tail coefficient needs
a CDF. Python uses `2*t.cdf(-sqrt((nu+1)*(1-rho)/(1+rho)), nu+1)`.

| Pair | Report tail dependence | Corrected Python | Source expression using Python copula parameters |
|---|---:|---:|---:|
| BMW / Intesa | 0.0014 | 0.00094968 | 0.00139132 |
| BMW / ENI | 0.0104 | 0.00735009 | 0.01037383 |
| Intesa / ENI | 0.1363 | 0.19881689 | 0.13632185 |

All three source-expression values round to the report's coefficients.
Copula correlations themselves are close but not identical: 0.09520225,
0.09209448, 0.56212770 vs 0.0949, 0.0919, 0.5618. Python joint ML differs
from MATLAB's ApproximateML method; the report does not publish copula degrees
of freedom, so full parameter equality cannot be checked.

### COVID and war indexing, report pages 39-42

In the source, `u=k(k)` and `g=h(h)` produce short all-true masks. Applied to
out-of-sample forecast arrays, they select their first 148/151 rows, rather
than the crisis dates. Their MSFE expressions then mix crisis returns with
those unrelated forecasts. Python selects the actual out-of-sample dates and
uses each subset's own sample size for DM statistics.

Recreating those source expressions reproduces **all** crisis RiskMetrics
violations, loss, and MSFE cells at the report's precision. For example:

| Period, BMW RiskMetrics | Report | Source-expression emulation | Corrected Python |
|---|---:|---:|---:|
| COVID total loss | 24.9523 | 24.9523086 | 65.0198967 |
| COVID MSFE | 923.9818 | 923.9817781 | 815.7043498 |
| War total loss | 25.2064 | 25.2064169 | 43.4648223 |
| War MSFE | 115.9010 | 115.9010095 | 101.9132533 |

These differences are substantive and cannot be described as numerical parity.
The report's crisis conclusions should be reevaluated using the corrected
subsets. The extra EWMA update used by the original full-sample VaR benchmark
is deliberately retained and documented for reproducibility.

## Validation limits

Eight tests pass: independent DCC/EWMA/presample recursions, analytic-gradient
validation, missing-data alignment, bundled-data report moments, copula
parameter bounds, and rolling forecasts without future-data leakage. The full
real-data run finishes without optimizer warnings and produces 38 plots,
forecast/residual CSVs, multivariate arrays, and strict JSON summaries.

This validates the saved report's published scalars and explains specific
source errors. It does not compare every conditional variance or forecast to
fresh MATLAB output; those arrays are not included in the report. No numerical
values were changed to force an apparent match.
