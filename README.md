# HJM Interest Rate Framework

This repository provides a robust implementation of the **Heath-Jarrow-Morton (HJM)** framework for modeling the evolution of the instantaneous forward rate curve. The project leverages **Nelson-Siegel-Svensson (NSS)** parameters for curve fitting and **Principal Component Analysis (PCA)** to drive stochastic simulations.

## 1. The HJM-NSS Engine (`HJM_NSS.py`)

The core of this framework is the HJM engine, which models the entire yield curve as a stochastic process. Unlike short-rate models, the HJM approach ensures the initial term structure is perfectly fitted by construction.

### Data Acquisition & NSS Fitting
* The engine fetches historical yield curve parameters from the Federal Reserve (feds200628).
* It implements the **Nelson-Siegel-Svensson (NSS)** model to generate instantaneous forward rates $f(t, T)$.
* The NSS forward rate formulation used is:
    $$f(T) = \beta_0 + \beta_1 e^{-T/\tau_1} + \beta_2 \frac{T}{\tau_1} e^{-T/\tau_1} + \beta_3 \frac{T}{\tau_2} e^{-T/\tau_2}$$
    where $\beta_0$ represents the long-run level, $\beta_1$ the slope, and $\beta_2/\beta_3$ the curvature components.

### Volatility Modeling via PCA
* The model extracts volatility structures by performing PCA on the daily changes of the forward rates.
* It captures the characteristic "Level," "Slope," and "Curvature" movements by retaining components that explain a specified variance threshold (e.g., 95%).
* These discrete PCA loadings are transformed into continuous functions using **Smoothing Splines**, allowing for precise drift and diffusion calculations at any maturity.

### Simulation
* The engine supports both standard and **vectorized Monte Carlo simulations**.
* The simulation accounts for the HJM no-arbitrage condition, where the drift of the forward rate is uniquely determined by its volatility and the market price of risk (assumed zero in the risk-neutral measure).

---

## 2. Swaption Pricing (`HJM_swaption.py`)

This module focuses on the valuation of European Payer Swaptions within the HJM framework.

* **Monte Carlo Pricing:** Simulates the forward curve until the option's expiry and calculates the discounted payoff based on the simulated swap rate and annuity.
* **Black Model Comparison:** For validation, the module estimates the implied Black volatility of the underlying swap rate through HJM simulations and compares the results against the analytical Black swaption formula.
* **Annuity Calculation:** Includes utilities to derive par swap rates and annuity factors from the simulated spot curves.

---

## 3. Risk Management (`HJM_risk.py`)

The risk module provides tools for assessing the market risk of fixed-income portfolios over a defined horizon.

* **Horizon Risk:** Calculates **Value-at-Risk (VaR)** and **Expected Shortfall (ES)** by simulating forward curve paths to a future date (e.g., one quarter).
* **Bond & Swaption Risk:**
    * For bonds, it accounts for coupon reinvestment and funding accrual (carry) over the risk horizon.
    * For swaptions, it performs a nested "simulation-within-a-simulation" to re-price the instrument at the horizon.
* **Performance:** Utilizes vectorized simulation paths to efficiently handle the large-scale computations required for P&L distributions.

---

## Dependencies
* `numpy`, `pandas`, `matplotlib`
* `scipy` (Interpolation and Integration)
* `sklearn` (PCA)
* `pandas_datareader` (FRED data access)

---
