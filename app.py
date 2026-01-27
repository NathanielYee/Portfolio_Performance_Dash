import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging

# ==========================================
# 1. Page Configuration
# ==========================================
st.set_page_config(
    page_title="Personal Quant Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Personal Quantitative Finance Dashboard")
st.markdown("---")

# ==========================================
# 2. Sidebar Configuration
# ==========================================
st.sidebar.header("🔧 Portfolio Configuration")

# Ticker & Weights Input
default_tickers = "AAPL, MSFT, GOOG, AMZN"
tickers_input = st.sidebar.text_input("Tickers (comma-separated)", value=default_tickers)
weights_input = st.sidebar.text_input("Weights (comma-separated, leave blank for equal)", value="")

# Date Range
col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("Start Date", value=datetime.now() - timedelta(days=365*2))
end_date = col2.date_input("End Date", value=datetime.now())

# --- NEW: Benchmark & View Toggles ---
benchmark_options = ["SPY", "QQQ", "DIA", "IWM", "BTC-USD"]
benchmark_tickers = st.sidebar.multiselect("Compare against Benchmarks", benchmark_options, default=["SPY"])
use_percentage = st.sidebar.toggle("Show as % Return", value=False)
# -------------------------------------

risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%)", value=4.5, step=0.1) / 100

# ==========================================
# 3. Data Processing & Functions
# ==========================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_data
def get_data(tickers, start, end):
    try:
        df = yf.download(tickers, start=start, end=end, auto_adjust=True, multi_level_index=False)
        if df.empty:
            return pd.DataFrame()
        # If multiple tickers, yf returns 'Close' as a DF with ticker columns
        # If single ticker, it's just 'Close'
        if 'Close' in df.columns:
            return df['Close']
        return df
    except Exception as e:
        logger.error(f"Error fetching {tickers}: {str(e)}")
        return pd.DataFrame()

def process_inputs(ticker_str, weight_str):
    tickers = [t.strip().upper() for t in ticker_str.split(',') if t.strip()]
    if weight_str.strip():
        weights = [float(w.strip()) for w in weight_str.split(',') if w.strip()]
    else:
        weights = [1.0 / len(tickers)] * len(tickers)
    
    if len(weights) != len(tickers):
        weights = [1.0 / len(tickers)] * len(tickers)
    else:
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
    return tickers, np.array(weights)

def calculate_metrics(daily_returns, risk_free_rate):
    TRADING_DAYS = 252
    annual_return = daily_returns.mean() * TRADING_DAYS
    annual_vol = daily_returns.std() * np.sqrt(TRADING_DAYS)
    sharpe_ratio = (annual_return - risk_free_rate) / annual_vol
    return annual_return, annual_vol, sharpe_ratio

def calculate_max_drawdown(cumulative_returns):
    roll_max = cumulative_returns.cummax()
    drawdown = cumulative_returns / roll_max - 1.0
    return drawdown.min()

# ==========================================
# 4. Main Execution Logic
# ==========================================
tickers_list, weights = process_inputs(tickers_input, weights_input)

if tickers_list:
    df_prices = get_data(tickers_list, start_date, end_date)
    df_benchmark = get_data(benchmark_tickers, start_date, end_date) if benchmark_tickers else pd.DataFrame()

    if not df_prices.empty:
        # Portfolio Returns
        simple_returns = df_prices.pct_change().dropna()
        log_returns = np.log(df_prices / df_prices.shift(1)).dropna()
        portfolio_daily_returns = (simple_returns * weights).sum(axis=1)
        
        # --- SECTION 1: METRICS ---
        st.subheader("1. Portfolio Performance Metrics")
        p_ret, p_vol, p_sharpe = calculate_metrics(portfolio_daily_returns, risk_free_rate)
        cum_ret_series = (1 + portfolio_daily_returns).cumprod()
        max_dd = calculate_max_drawdown(cum_ret_series)

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Expected Annual Return", f"{p_ret:.2%}")
        col_m2.metric("Annual Volatility", f"{p_vol:.2%}")
        col_m3.metric("Sharpe Ratio", f"{p_sharpe:.2f}")
        col_m4.metric("Max Drawdown", f"{max_dd:.2%}")
        
        st.markdown("---")

        # --- SECTION 2: VISUALIZATIONS ---
        st.subheader("2. Portfolio Visualizations")
        tab1, tab2 = st.tabs(["Performance Comparison", "Correlation Heatmap"])
        
        with tab1:
            y_label = "Percentage Return" if use_percentage else "Value ($100 Invested)"
            
            # Prep Portfolio Plot Data
            if use_percentage:
                portfolio_plot = (1 + portfolio_daily_returns).cumprod() - 1
            else:
                portfolio_plot = 100 * (1 + portfolio_daily_returns).cumprod()

            fig_cum = go.Figure()
            fig_cum.add_trace(go.Scatter(x=portfolio_plot.index, y=portfolio_plot, name='My Portfolio', line=dict(width=3, color='#00CC96')))
            
            # Add Benchmarks
            if not df_benchmark.empty:
                # Handle single vs multi-benchmark DataFrame structure
                for ticker in benchmark_tickers:
                    b_data = df_benchmark[ticker] if len(benchmark_tickers) > 1 else df_benchmark
                    b_returns = b_data.pct_change().dropna()
                    
                    if use_percentage:
                        b_plot = (1 + b_returns).cumprod() - 1
                    else:
                        b_plot = 100 * (1 + b_returns).cumprod()
                        
                    fig_cum.add_trace(go.Scatter(x=b_plot.index, y=b_plot, name=ticker, line=dict(dash='dash', width=1.5)))

            fig_cum.update_layout(
                title=f"Portfolio vs Benchmarks ({y_label})",
                yaxis_title=y_label,
                template="plotly_white",
                yaxis_tickformat=".2%" if use_percentage else "$.2f"
            )
            st.plotly_chart(fig_cum, use_container_width=True)
            
        with tab2:
            corr_matrix = simple_returns.corr()
            fig_corr = px.imshow(corr_matrix, text_auto=True, title="Asset Correlation Matrix", color_continuous_scale="RdBu_r")
            st.plotly_chart(fig_corr, use_container_width=True)

        # --- SECTION 3: MONTE CARLO ---
        st.markdown("---")
        st.subheader("3. Monte Carlo Simulation")
        with st.expander("Run Simulation Scenario"):
            col_sim1, col_sim2 = st.columns(2)
            n_sims = col_sim1.slider("Number of Simulations", 200, 2000, 500)
            time_horizon = col_sim2.slider("Time Horizon (Days)", 30, 365, 252)
            
            if st.button("Run Simulation"):
                mean_returns = log_returns.mean().values
                cov_matrix = log_returns.cov().values
                daily_log_returns_sim = np.random.multivariate_normal(mean_returns, cov_matrix, size=(time_horizon, n_sims))
                portfolio_sim_returns = np.dot(np.exp(daily_log_returns_sim) - 1, weights)
                portfolio_sim_paths = 100 * np.cumprod(1 + portfolio_sim_returns, axis=0)
                
                fig_mc = go.Figure()
                x_axis = list(range(1, time_horizon + 1))
                for i in range(min(n_sims, 50)):
                    fig_mc.add_trace(go.Scatter(x=x_axis, y=portfolio_sim_paths[:, i], mode='lines', line=dict(width=1, color='rgba(100, 100, 255, 0.1)'), showlegend=False))
                
                fig_mc.add_trace(go.Scatter(x=x_axis, y=np.mean(portfolio_sim_paths, axis=1), name='Mean Outcome', line=dict(width=3, color='orange')))
                fig_mc.update_layout(title="Monte Carlo Projected Value", xaxis_title="Days", yaxis_title="Value ($)")
                st.plotly_chart(fig_mc, use_container_width=True)
                
                var_95 = np.percentile((portfolio_sim_paths[-1, :] / 100) - 1, 5)
                st.info(f"**95% Value at Risk (VaR):** {var_95:.2%}")

    else:
        st.warning("No data found. Check your tickers.")
else:
    st.info("👈 Enter stock tickers in the sidebar to begin.")