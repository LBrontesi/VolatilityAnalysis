# Original assignment inputs

These files were recovered from the local financial econometrics assignment's
`dataset` directory and copied without modification. The three market-price
CSVs contain 6,243 observations, dated 2000-01-03 through 2024-05-31, in the
original quantmod/Yahoo format (`Index`, Open, High, Low, Close, Volume, Adjusted).
`EuroStoxx50.xlsx` is the assignment's original stock-selection workbook.

The analysis uses the fifth column (Close), not Adjusted. Default physical CSV
lines 1307-6244 yield 4,938 prices and 4,937 returns per asset. In MATLAB row
alignment mode, asset 1's dates label all three series. Use `--alignment dates`
for an analysis that pairs observations by actual trading date instead.

The accompanying local `ESTX50_quantmod.R` script identifies Yahoo Finance as
the download source and requests data from 2000-01-01 to 2024-06-01. The raw CSV
files are retained for assignment reproducibility, without filling missing data,
changing prices, or re-downloading revised history.
