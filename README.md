# 📈 Personal Quant Dashboard

An interactive, high-performance financial analytics tool built with **Streamlit** and **Python**. This dashboard enables investors to perform professional-grade quantitative analysis on custom portfolios, visualize historical performance against major benchmarks, and simulate future outcomes using statistical modeling.

---

## 🚀 Key Features

* **Portfolio Tracking:** Input custom tickers and weights to see real-time performance.
* **Dual View Toggle:** Seamlessly switch between **Growth of $100** and **Percentage Return** views.
* **Multi-Benchmark Comparison:** Stack your portfolio against `SPY`, `QQQ`, `DIA`, `IWM`, and even `BTC-USD`.
* **Statistical Analysis:** * **Sharpe Ratio:** Measures risk-adjusted return.
    * **Volatility:** Annualized standard deviation of returns.
    * **Max Drawdown:** The largest peak-to-trough decline.
* **Correlation Heatmap:** Identify asset relationships to optimize diversification.
* **Monte Carlo Simulations:** Run thousands of scenarios to project future portfolio values and calculate **Value at Risk (VaR)**.



---

## 🛠️ Technical Stack

* **Core Logic:** `Python 3.9+`
* **Web Framework:** `Streamlit`
* **Data Source:** `yfinance` (Yahoo Finance API)
* **Data Wrangling:** `Pandas`, `NumPy`
* **Interactive Charts:** `Plotly`

---

## 📝 Quantitative Methodology

The dashboard utilizes standard financial formulas to ensure accuracy:

### Sharpe Ratio
Calculated to evaluate the risk-adjusted return:
$$S_p = \frac{R_p - R_f}{\sigma_p}$$
*Where $R_p$ is expected portfolio return, $R_f$ is the risk-free rate, and $\sigma_p$ is the portfolio volatility.*

### Log Returns
Used for statistical simulations to ensure time-additivity:
$$r = \ln\left(\frac{P_t}{P_{t-1}}\right)$$

### Monte Carlo Simulation
The simulation uses a **Multivariate Normal Distribution** based on the historical covariance matrix. This preserves the correlations between assets (e.g., how AAPL moves in relation to MSFT) during future projections rather than treating each stock in isolation.



---

## 💻 Installation & Setup

1. **Clone the repo:**
   ```bash
   git clone [https://github.com/your-username/personal-quant-dashboard.git](https://github.com/your-username/personal-quant-dashboard.git)
   cd personal-quant-dashboard
