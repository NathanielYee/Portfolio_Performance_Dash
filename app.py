import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging

# MUST BE FIRST
st.set_page_config(page_title="Personal Quant Dashboard", page_icon="📈", layout="wide")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# Shared Utility Functions
# ==========================================

def get_valid_start_date(d):
    if d.weekday() == 5: return d - timedelta(days=1)
    if d.weekday() == 6: return d - timedelta(days=2)
    return d

@st.cache_data
def get_data(tickers, start, end):
    try:
        df = yf.download(tickers, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'), auto_adjust=True, threads=True)
        if df.empty: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            if 'Close' in df.columns.levels[0]: return df['Close']
        else:
            if 'Close' in df.columns: return df[['Close']]
        return df
    except Exception as e:
        logger.error(f"Error fetching {tickers}: {e}")
        return pd.DataFrame()

def calculate_metrics(daily_returns, rf):
    """Compute annualized metrics dynamically based on actual holding period."""
    n_days = len(daily_returns)
    if n_days == 0:
        return {}

    # Annualization factor derived from actual trading days observed
    years_held = n_days / 252
    ann_factor = 252 / n_days  # scales to 1 year

    # Total cumulative return (actual, not annualized)
    cum_return = (1 + daily_returns).prod() - 1

    # CAGR: geometrically correct annualized return
    ending_growth = 1 + cum_return
    if years_held > 0 and ending_growth > 0:
        cagr = ending_growth ** (1 / years_held) - 1
    else:
        cagr = 0.0

    # Annualized volatility (sqrt-time scaling is standard for daily → annual)
    ann_vol = daily_returns.std() * np.sqrt(252)

    # Sharpe using CAGR (not simple mean scaling)
    sharpe = (cagr - rf) / ann_vol if ann_vol != 0 else 0.0

    # Sortino: downside deviation only
    downside = daily_returns[daily_returns < 0]
    downside_std = downside.std() * np.sqrt(252) if len(downside) > 0 else 0.0
    sortino = (cagr - rf) / downside_std if downside_std != 0 else 0.0

    # Calmar: CAGR / |Max Drawdown|
    cum_series = (1 + daily_returns).cumprod()
    max_dd = calculate_max_drawdown(cum_series)
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    return {
        'n_days': n_days,
        'years_held': years_held,
        'total_return': cum_return,
        'cagr': cagr,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_dd': max_dd,
        'calmar': calmar,
    }

def calculate_max_drawdown(cum):
    return (cum / cum.cummax() - 1.0).min()


# ==========================================
# Shared Sidebar: Portfolio Editor
# ==========================================

def render_sidebar():
    """Renders the shared sidebar and returns portfolio config."""
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.radio("Go to", ["Portfolio Overview", "Volatility & Options Lab"])

    st.sidebar.markdown("---")
    st.sidebar.header("🔧 Portfolio Holdings")

    raw_start = datetime.now().date() - timedelta(days=365)
    valid_start = get_valid_start_date(raw_start)

    default_portfolio = pd.DataFrame({
        "Ticker": ["AAPL", "MSFT", "GOOG"],
        "Shares": [10.0, 15.0, 20.0],
        "Start Date": [valid_start] * 3
    })

    # --- CSV Upload ---
    uploaded_file = st.sidebar.file_uploader("Upload Holdings CSV", type=["csv"], help="CSV with columns: Ticker, Shares, Start Date")

    if uploaded_file is not None:
        try:
            csv_df = pd.read_csv(uploaded_file)
            csv_df.columns = csv_df.columns.str.strip()

            # Flexible column matching (case-insensitive)
            col_map = {}
            for col in csv_df.columns:
                cl = col.lower()
                if 'tick' in cl or 'symbol' in cl:
                    col_map[col] = 'Ticker'
                elif 'share' in cl or 'qty' in cl or 'quantity' in cl or 'amount' in cl:
                    col_map[col] = 'Shares'
                elif 'date' in cl or 'start' in cl:
                    col_map[col] = 'Start Date'
            csv_df = csv_df.rename(columns=col_map)

            required = {'Ticker', 'Shares', 'Start Date'}
            if not required.issubset(csv_df.columns):
                missing = required - set(csv_df.columns)
                st.sidebar.error(f"CSV missing columns: {', '.join(missing)}. Need: Ticker, Shares, Start Date")
                initial_portfolio = default_portfolio
            else:
                csv_df['Shares'] = pd.to_numeric(csv_df['Shares'], errors='coerce')
                csv_df['Start Date'] = pd.to_datetime(csv_df['Start Date'], errors='coerce').dt.date
                csv_df = csv_df.dropna(subset=['Ticker', 'Shares', 'Start Date'])
                if csv_df.empty:
                    st.sidebar.warning("CSV parsed but no valid rows found.")
                    initial_portfolio = default_portfolio
                else:
                    st.sidebar.success(f"Loaded {len(csv_df)} holdings from CSV.")
                    initial_portfolio = csv_df[['Ticker', 'Shares', 'Start Date']]
        except Exception as e:
            st.sidebar.error(f"Error reading CSV: {e}")
            initial_portfolio = default_portfolio
    else:
        initial_portfolio = default_portfolio

    st.sidebar.caption("Edit positions below (or upload a CSV above):")
    portfolio_df = st.sidebar.data_editor(
        initial_portfolio,
        num_rows="dynamic",
        hide_index=True,
        column_config={
            "Start Date": st.column_config.DateColumn("Start Date", required=True),
            "Shares": st.column_config.NumberColumn("Shares", min_value=0.0001, required=True),
            "Ticker": st.column_config.TextColumn("Ticker", required=True)
        }
    )

    # --- CSV Download (save current state) ---
    csv_export = portfolio_df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button("💾 Export Holdings CSV", csv_export, "portfolio_holdings.csv", "text/csv")

    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Settings")
    end_date = st.sidebar.date_input("Analysis End Date", value=datetime.now())
    benchmark_options = ["SPY", "QQQ", "DIA", "IWM", "BTC-USD"]
    benchmark_tickers = st.sidebar.multiselect("Compare against Benchmarks", benchmark_options, default=["SPY"])
    use_pct = st.sidebar.toggle("Show as % Return", value=True)
    rf = st.sidebar.number_input("Risk-Free Rate (%)", value=4.5, step=0.1) / 100

    # Clean portfolio
    portfolio_df = portfolio_df.dropna(subset=['Ticker', 'Shares', 'Start Date'])
    portfolio_df['Ticker'] = portfolio_df['Ticker'].astype(str).str.upper().str.strip()
    portfolio_df = portfolio_df[portfolio_df['Ticker'] != ""]

    return page, portfolio_df, end_date, benchmark_tickers, use_pct, rf


# ==========================================
# Page 1: Portfolio Overview
# ==========================================

def show_main_page(portfolio_df, end_date, benchmark_tickers, use_pct, rf):
    st.title("📈 Personal Portfolio Overview")
    st.markdown("---")

    tickers_list = portfolio_df['Ticker'].tolist()
    if not tickers_list:
        st.info("👈 Enter your holdings in the sidebar to begin.")
        return

    earliest_start = portfolio_df['Start Date'].min()
    df_prices = get_data(tickers_list, earliest_start, end_date)
    df_benchmark = get_data(benchmark_tickers, earliest_start, end_date) if benchmark_tickers else pd.DataFrame()

    if df_prices.empty:
        st.warning("No data found. Check your tickers.")
        return

    if len(tickers_list) == 1:
        df_prices = df_prices.to_frame(name=tickers_list[0])

    asset_daily_returns = df_prices.pct_change().dropna()
    daily_values = pd.DataFrame(index=df_prices.index, columns=tickers_list)

    for _, row in portfolio_df.iterrows():
        t, s, sd = row['Ticker'], row['Shares'], pd.to_datetime(row['Start Date']).date()
        daily_values[t] = df_prices[t] * s
        daily_values.loc[daily_values.index.date < sd, t] = 0.0

    prev_daily_values = daily_values.shift(1)
    total_prev = prev_daily_values.sum(axis=1)
    daily_weights = prev_daily_values.div(total_prev, axis=0).fillna(0)

    common_idx = asset_daily_returns.index.intersection(daily_weights.index)
    portfolio_daily_returns = (asset_daily_returns.loc[common_idx] * daily_weights.loc[common_idx]).sum(axis=1)
    portfolio_daily_returns = portfolio_daily_returns[total_prev.loc[common_idx] > 0]

    current_dollar_values = daily_values.iloc[-1]
    current_weights = (current_dollar_values / current_dollar_values.sum()).values

    # --- METRICS ---
    st.subheader("Portfolio Performance Metrics")
    if len(portfolio_daily_returns) == 0:
        st.warning("Not enough return data to compute metrics.")
        return

    m = calculate_metrics(portfolio_daily_returns, rf)
    cum_ret_series = (1 + portfolio_daily_returns).cumprod()

    period_label = f"{m['n_days']} trading days ({m['years_held']:.1f} yrs)"
    st.caption(f"Metrics computed over **{period_label}** of live portfolio data.")

    row1 = st.columns(5)
    row1[0].metric("Portfolio Value", f"${current_dollar_values.sum():,.2f}",
                    help="Current market value of all holdings (shares × latest close price).")
    row1[1].metric("Total Return", f"{m['total_return']:.2%}",
                    help="Cumulative gain/loss since the portfolio became active. Not annualized.")
    row1[2].metric("CAGR", f"{m['cagr']:.2%}",
                    help="Compound Annual Growth Rate — the geometrically annualized return, accounting for compounding over your actual holding period.")
    row1[3].metric("Annualized Volatility", f"{m['ann_vol']:.2%}",
                    help="Standard deviation of daily returns scaled to one year (×√252). Measures total dispersion of returns — both up and down.")
    row1[4].metric("Max Drawdown", f"{m['max_dd']:.2%}",
                    help="Largest peak-to-trough decline in portfolio value. Represents the worst-case loss experienced during the holding period.")

    row2 = st.columns(5)
    row2[0].metric("Sharpe Ratio", f"{m['sharpe']:.2f}",
                    help="(CAGR − Risk-Free Rate) / Volatility. Measures excess return per unit of total risk. Higher is better; above 1.0 is generally considered good.")
    row2[1].metric("Sortino Ratio", f"{m['sortino']:.2f}",
                    help="Like Sharpe, but only penalizes downside volatility. Uses the standard deviation of negative returns only, so it doesn't punish you for upside swings.")
    row2[2].metric("Calmar Ratio", f"{m['calmar']:.2f}",
                    help="CAGR / |Max Drawdown|. Measures return per unit of tail risk. A higher Calmar means you're being compensated well for the worst loss you experienced.")
    row2[3].metric("Holding Period", f"{m['n_days']} days",
                    help="Number of actual trading days used to compute all metrics. All ratios dynamically scale to this period.")
    row2[4].metric("Risk-Free Rate", f"{rf:.2%}",
                    help="The baseline 'riskless' return (e.g. T-bills) subtracted from your return when computing Sharpe and Sortino. Adjustable in the sidebar.")

    st.markdown("---")

    # --- VISUALIZATIONS ---
    st.subheader("Portfolio Visualizations")
    tab1, tab2, tab3, tab4 = st.tabs(["Performance & Value", "Current Allocation", "Correlation", "Risk Analysis"])

    with tab1:
        total_val_hist = daily_values.sum(axis=1)
        total_val_hist = total_val_hist[total_val_hist > 0]

        fig_val = go.Figure()
        fig_val.add_trace(go.Scatter(
            x=total_val_hist.index, y=total_val_hist,
            fill='tozeroy', name='Total Portfolio Value', line=dict(color='#636EFA', width=2)
        ))
        fig_val.update_layout(title="Total Portfolio Value Over Time ($)", yaxis_tickformat="$,.2f", template="plotly_white")
        st.plotly_chart(fig_val, use_container_width=True)

        y_label = "Percentage Return" if use_pct else "Value ($100 Invested)"
        port_plot = (1 + portfolio_daily_returns).cumprod() - 1 if use_pct else 100 * (1 + portfolio_daily_returns).cumprod()

        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(x=port_plot.index, y=port_plot, name='My Portfolio', line=dict(width=3, color='#00CC96')))

        if not df_benchmark.empty:
            first_active = portfolio_daily_returns.index[0]
            df_bench = df_benchmark.to_frame(name=benchmark_tickers[0]) if isinstance(df_benchmark, pd.Series) else df_benchmark
            for tk in benchmark_tickers:
                try:
                    b_prices = df_bench[tk].loc[first_active:]
                    b_ret = b_prices.pct_change().fillna(0)
                    b_plot = (1 + b_ret).cumprod() - 1 if use_pct else 100 * (1 + b_ret).cumprod()
                    fig_cum.add_trace(go.Scatter(x=b_plot.index, y=b_plot, name=f"Benchmark: {tk}", line=dict(dash='dash', width=1.5)))
                except KeyError:
                    continue

        fig_cum.update_layout(title=f"Relative Performance ({y_label})", yaxis_tickformat=".2%" if use_pct else "$.2f", template="plotly_white")
        st.plotly_chart(fig_cum, use_container_width=True)

    with tab2:
        fig_pie = px.pie(names=current_dollar_values.index, values=current_dollar_values.values, title="Current Allocation", hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)

    with tab3:
        fig_corr = px.imshow(asset_daily_returns.corr(), text_auto=True, title="Correlation Matrix", color_continuous_scale="RdBu_r")
        st.plotly_chart(fig_corr, use_container_width=True)

    with tab4:
        st.subheader("Deep Risk Analytics")
        total_val_hist = daily_values.sum(axis=1)
        total_val_hist = total_val_hist[total_val_hist > 0]
        running_max = total_val_hist.cummax()
        dd_series = (total_val_hist / running_max) - 1

        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=dd_series.index, y=dd_series, fill='tozeroy', name='Drawdown', line=dict(color='red', width=1)))
        fig_dd.update_layout(title="Underwater Drawdown Plot", yaxis_title="Decline from Peak (%)", yaxis_tickformat=".2%", template="plotly_white")
        st.plotly_chart(fig_dd, use_container_width=True)

        cr1, cr2 = st.columns(2)
        with cr1:
            mkt = "SPY"
            has_spy = mkt in df_benchmark.columns if not df_benchmark.empty else False
            if has_spy:
                mkt_ret = df_benchmark[mkt].loc[portfolio_daily_returns.index].pct_change().dropna()
                cd = portfolio_daily_returns.index.intersection(mkt_ret.index)
                cov = np.cov(portfolio_daily_returns.loc[cd], mkt_ret.loc[cd])[0][1]
                beta = cov / np.var(mkt_ret.loc[cd])
                st.metric("Portfolio Beta (vs SPY)", f"{beta:.2f}")
                st.caption(f"Beta of {beta:.2f} → portfolio is {abs(beta-1):.0%} {'more' if beta > 1 else 'less'} volatile than S&P 500.")
            else:
                st.warning("Select 'SPY' in benchmarks to calculate Beta.")

        with cr2:
            n_obs = len(portfolio_daily_returns)
            w = min(60, max(10, n_obs // 5))  # dynamic window: 20% of history, clamped [10, 60]
            r_mean = portfolio_daily_returns.rolling(window=w).mean()
            r_std = portfolio_daily_returns.rolling(window=w).std()
            r_sharpe = (r_mean / r_std) * np.sqrt(252)
            fig_sh = go.Figure()
            fig_sh.add_trace(go.Scatter(x=r_sharpe.index, y=r_sharpe, name=f'{w}D Rolling Sharpe', line=dict(color='orange')))
            fig_sh.update_layout(title=f"Rolling {w}-Day Sharpe Ratio (window auto-sized to data)", yaxis_title="Sharpe Ratio", template="plotly_white")
            st.plotly_chart(fig_sh, use_container_width=True)

    # --- MONTE CARLO ---
    st.markdown("---")
    st.subheader("Monte Carlo Simulation")
    with st.expander("Run Simulation Scenario"):
        sc1, sc2 = st.columns(2)
        n_sims = sc1.slider("Number of Simulations", 200, 2000, 500)
        horizon = sc2.slider("Time Horizon (Days)", 30, 365, 252)

        if st.button("Run Simulation"):
            log_ret = np.log(1 + asset_daily_returns)
            sim = np.random.multivariate_normal(log_ret.mean().values, log_ret.cov().values, size=(horizon, n_sims))
            port_sim = np.dot(np.exp(sim) - 1, current_weights)
            sv = current_dollar_values.sum()
            paths = sv * np.cumprod(1 + port_sim, axis=0)

            fig_mc = go.Figure()
            for i in range(min(n_sims, 50)):
                fig_mc.add_trace(go.Scatter(x=list(range(1, horizon + 1)), y=paths[:, i], mode='lines', line=dict(width=1, color='rgba(100,100,255,0.1)'), showlegend=False))
            fig_mc.add_trace(go.Scatter(x=list(range(1, horizon + 1)), y=np.mean(paths, axis=1), name='Mean Outcome', line=dict(width=3, color='orange')))
            fig_mc.update_layout(title="Monte Carlo Dollar Projection", yaxis_tickformat="$,.2f", template="plotly_white")
            st.plotly_chart(fig_mc, use_container_width=True)

            var95 = np.percentile((paths[-1, :] / sv) - 1, 5)
            st.info(f"**95% VaR:** {var95:.2%} (Potential loss of ${sv * abs(var95):,.2f})")


# ==========================================
# Page 2: Volatility & Options Lab
# ==========================================

def show_volatility_page(portfolio_df):
    st.title("🌋 Volatility & Options Lab")
    st.markdown("---")

    tickers_list = portfolio_df['Ticker'].tolist()
    if not tickers_list:
        st.info("👈 Add tickers in the sidebar first.")
        return

    ticker = st.selectbox("Select Ticker from Portfolio", tickers_list)
    tk = yf.Ticker(ticker)

    try:
        exps = tk.options
    except Exception:
        st.error(f"Could not fetch options data for {ticker}.")
        return

    if not exps:
        st.warning(f"No listed options found for {ticker}.")
        return

    selected_exp = st.selectbox("Expiration Date", exps)
    opt = tk.option_chain(selected_exp)

    st.subheader(f"Options Chain: {ticker} — {selected_exp}")

    tab_calls, tab_puts, tab_vol = st.tabs(["Calls", "Puts", "Volatility Analysis"])

    with tab_calls:
        st.dataframe(opt.calls, use_container_width=True)

    with tab_puts:
        st.dataframe(opt.puts, use_container_width=True)

    with tab_vol:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Call IV Smile")
            calls = opt.calls[opt.calls['impliedVolatility'] > 0].copy()
            if not calls.empty:
                fig_c = px.line(calls, x='strike', y='impliedVolatility', title=f"Call IV Smile ({selected_exp})")
                fig_c.update_layout(yaxis_tickformat=".2%", template="plotly_white")
                st.plotly_chart(fig_c, use_container_width=True)
            else:
                st.caption("No IV data available for calls.")

        with col2:
            st.subheader("Put IV Smile")
            puts = opt.puts[opt.puts['impliedVolatility'] > 0].copy()
            if not puts.empty:
                fig_p = px.line(puts, x='strike', y='impliedVolatility', title=f"Put IV Smile ({selected_exp})")
                fig_p.update_layout(yaxis_tickformat=".2%", template="plotly_white")
                st.plotly_chart(fig_p, use_container_width=True)
            else:
                st.caption("No IV data available for puts.")

        # --- Volatility Term Structure ---
        st.markdown("---")
        st.subheader("Volatility Term Structure (ATM Approximation)")
        st.caption("ATM IV approximated as the mean IV of strikes nearest the last close price.")

        try:
            last_price = tk.info.get('currentPrice') or tk.info.get('regularMarketPrice')
        except Exception:
            last_price = None

        if last_price:
            term_data = []
            for exp in exps[:12]:  # limit to first 12 expirations
                try:
                    chain = tk.option_chain(exp)
                    c = chain.calls.copy()
                    c['dist'] = (c['strike'] - last_price).abs()
                    atm = c.nsmallest(3, 'dist')
                    avg_iv = atm['impliedVolatility'].mean()
                    dte = (pd.to_datetime(exp) - pd.Timestamp.now()).days
                    if avg_iv > 0:
                        term_data.append({"Expiration": exp, "DTE": dte, "ATM IV": avg_iv})
                except Exception:
                    continue

            if term_data:
                term_df = pd.DataFrame(term_data)
                fig_term = px.line(term_df, x='DTE', y='ATM IV', text='Expiration',
                                   title="IV Term Structure (Days to Expiry)")
                fig_term.update_traces(textposition="top center")
                fig_term.update_layout(yaxis_tickformat=".2%", template="plotly_white",
                                       xaxis_title="Days to Expiration", yaxis_title="Implied Volatility")
                st.plotly_chart(fig_term, use_container_width=True)
            else:
                st.caption("Could not build term structure.")
        else:
            st.caption("Could not determine current price for ATM approximation.")

        # --- Historical vs Implied Vol ---
        st.markdown("---")
        st.subheader("Historical Volatility vs Current ATM IV")

        hist_prices = get_data([ticker], datetime.now().date() - timedelta(days=365), datetime.now().date())
        if not hist_prices.empty:
            if isinstance(hist_prices, pd.DataFrame) and ticker in hist_prices.columns:
                hp = hist_prices[ticker]
            else:
                hp = hist_prices.squeeze()

            log_ret = np.log(hp / hp.shift(1)).dropna()
            windows = [20, 40, 60, 90]
            fig_hv = go.Figure()
            for w in windows:
                rv = log_ret.rolling(window=w).std() * np.sqrt(252)
                fig_hv.add_trace(go.Scatter(x=rv.index, y=rv, name=f'{w}D Realized Vol'))

            # Overlay current near-term ATM IV as a horizontal line
            if term_data:
                nearest_iv = term_data[0]['ATM IV']
                fig_hv.add_hline(y=nearest_iv, line_dash="dash", line_color="red",
                                 annotation_text=f"Current ATM IV ({nearest_iv:.1%})")

            fig_hv.update_layout(title=f"{ticker} — Realized Vol vs Implied Vol",
                                 yaxis_tickformat=".2%", template="plotly_white",
                                 yaxis_title="Annualized Volatility")
            st.plotly_chart(fig_hv, use_container_width=True)


# ==========================================
# Main Navigation Controller
# ==========================================

def main():
    page, portfolio_df, end_date, benchmarks, use_pct, rf = render_sidebar()

    if page == "Portfolio Overview":
        show_main_page(portfolio_df, end_date, benchmarks, use_pct, rf)
    elif page == "Volatility & Options Lab":
        show_volatility_page(portfolio_df)

if __name__ == "__main__":
    main()