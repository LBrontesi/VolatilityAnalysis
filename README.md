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

---

## 👥 Authors
Lorenzo Brontesi, Giuseppe De Santis, Geanina A. S. Hosszu, Benedetta Marchettini  
