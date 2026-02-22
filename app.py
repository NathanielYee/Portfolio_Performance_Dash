import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from scipy import stats
import logging, warnings
warnings.filterwarnings('ignore')

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

@st.cache_data(ttl=3600)
def get_risk_free_rate():
    try:
        tnx = yf.Ticker("^TNX")
        hist = tnx.history(period="5d")
        if not hist.empty:
            return hist['Close'].dropna().iloc[-1] / 100
    except Exception:
        pass
    return None

@st.cache_data(ttl=600)
def get_data(tickers, start, end):
    try:
        s = start.strftime('%Y-%m-%d') if hasattr(start, 'strftime') else str(start)
        e = end.strftime('%Y-%m-%d') if hasattr(end, 'strftime') else str(end)
        df = yf.download(tickers, start=s, end=e, auto_adjust=True, threads=True)
        if df.empty: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            if 'Close' in df.columns.levels[0]: return df['Close']
        else:
            if 'Close' in df.columns: return df[['Close']]
        return df
    except Exception as e:
        logger.error(f"Error fetching {tickers}: {e}")
        return pd.DataFrame()

def calculate_max_drawdown(cum):
    return (cum / cum.cummax() - 1.0).min()

def calculate_metrics(daily_returns, rf):
    n = len(daily_returns)
    if n == 0: return {}
    yrs = n / 252
    cum_ret = (1 + daily_returns).prod() - 1
    eg = 1 + cum_ret
    cagr = eg ** (1 / yrs) - 1 if yrs > 0 and eg > 0 else 0.0
    vol = daily_returns.std() * np.sqrt(252)
    sharpe = (cagr - rf) / vol if vol != 0 else 0.0
    ds = daily_returns[daily_returns < 0]
    ds_std = ds.std() * np.sqrt(252) if len(ds) > 0 else 0.0
    sortino = (cagr - rf) / ds_std if ds_std != 0 else 0.0
    cs = (1 + daily_returns).cumprod()
    mdd = calculate_max_drawdown(cs)
    calmar = cagr / abs(mdd) if mdd != 0 else 0.0
    return {'n_days': n, 'years_held': yrs, 'total_return': cum_ret, 'cagr': cagr,
            'ann_vol': vol, 'sharpe': sharpe, 'sortino': sortino, 'max_dd': mdd, 'calmar': calmar}

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


# ==========================================
# Shared Sidebar
# ==========================================

def render_sidebar():
    st.sidebar.title("🧭 Navigation")
    pages = ["Portfolio Overview", "Volatility & Options Lab", "Factor Exposure & Attribution",
             "Screener & Signal Scanner", "Trade Journal & Post-Mortem", "Macro Regime Dashboard"]
    page = st.sidebar.radio("Go to", pages)

    st.sidebar.markdown("---")
    st.sidebar.header("🔧 Portfolio Holdings")

    raw_start = datetime.now().date() - timedelta(days=365)
    valid_start = get_valid_start_date(raw_start)
    default_portfolio = pd.DataFrame({"Ticker": ["AAPL", "MSFT", "GOOG"], "Shares": [10.0, 15.0, 20.0], "Start Date": [valid_start] * 3})

    uploaded_file = st.sidebar.file_uploader("Upload Holdings CSV", type=["csv"], help="CSV with columns: Ticker, Shares, Start Date")
    if uploaded_file is not None:
        try:
            csv_df = pd.read_csv(uploaded_file)
            csv_df.columns = csv_df.columns.str.strip()
            col_map = {}
            for col in csv_df.columns:
                cl = col.lower()
                if 'tick' in cl or 'symbol' in cl: col_map[col] = 'Ticker'
                elif 'share' in cl or 'qty' in cl or 'quantity' in cl or 'amount' in cl: col_map[col] = 'Shares'
                elif 'date' in cl or 'start' in cl: col_map[col] = 'Start Date'
            csv_df = csv_df.rename(columns=col_map)
            required = {'Ticker', 'Shares', 'Start Date'}
            if not required.issubset(csv_df.columns):
                st.sidebar.error(f"CSV missing columns: {required - set(csv_df.columns)}")
                initial = default_portfolio
            else:
                csv_df['Shares'] = pd.to_numeric(csv_df['Shares'], errors='coerce')
                csv_df['Start Date'] = pd.to_datetime(csv_df['Start Date'], errors='coerce').dt.date
                csv_df = csv_df.dropna(subset=['Ticker', 'Shares', 'Start Date'])
                if csv_df.empty:
                    st.sidebar.warning("CSV parsed but no valid rows.")
                    initial = default_portfolio
                else:
                    st.sidebar.success(f"Loaded {len(csv_df)} holdings.")
                    initial = csv_df[['Ticker', 'Shares', 'Start Date']]
        except Exception as e:
            st.sidebar.error(f"CSV error: {e}")
            initial = default_portfolio
    else:
        initial = default_portfolio

    st.sidebar.caption("Edit positions below (or upload CSV above):")
    portfolio_df = st.sidebar.data_editor(initial, num_rows="dynamic", hide_index=True, column_config={
        "Start Date": st.column_config.DateColumn("Start Date", required=True),
        "Shares": st.column_config.NumberColumn("Shares", min_value=0.0001, required=True),
        "Ticker": st.column_config.TextColumn("Ticker", required=True)
    })
    csv_export = portfolio_df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button("💾 Export Holdings CSV", csv_export, "portfolio_holdings.csv", "text/csv")

    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Settings")
    end_date = st.sidebar.date_input("Analysis End Date", value=datetime.now())
    benchmark_options = ["SPY", "QQQ", "DIA", "IWM", "BTC-USD"]
    benchmark_tickers = st.sidebar.multiselect("Compare against Benchmarks", benchmark_options, default=["SPY"])
    use_pct = st.sidebar.toggle("Show as % Return", value=True)

    live_rf = get_risk_free_rate()
    if live_rf is not None:
        st.sidebar.markdown(f"**10Y Treasury Yield:** {live_rf:.2%} *(live)*")
        if st.sidebar.toggle("Override risk-free rate manually", value=False):
            rf = st.sidebar.number_input("Manual RF Rate (%)", value=live_rf * 100, step=0.1) / 100
        else:
            rf = live_rf
    else:
        st.sidebar.warning("Could not fetch live 10Y yield.")
        rf = st.sidebar.number_input("Risk-Free Rate (%)", value=4.5, step=0.1) / 100

    portfolio_df = portfolio_df.dropna(subset=['Ticker', 'Shares', 'Start Date'])
    portfolio_df['Ticker'] = portfolio_df['Ticker'].astype(str).str.upper().str.strip()
    portfolio_df = portfolio_df[portfolio_df['Ticker'] != ""]
    return page, portfolio_df, end_date, benchmark_tickers, use_pct, rf


# ==========================================
# Helper: Build portfolio returns (shared across pages)
# ==========================================

def build_portfolio_returns(portfolio_df, end_date):
    """Returns (portfolio_daily_returns, asset_daily_returns, current_weights, current_dollar_values, df_prices) or Nones."""
    tickers = portfolio_df['Ticker'].tolist()
    if not tickers: return None, None, None, None, None
    earliest = portfolio_df['Start Date'].min()
    df_prices = get_data(tickers, earliest, end_date)
    if df_prices.empty: return None, None, None, None, None
    if len(tickers) == 1: df_prices = df_prices.to_frame(name=tickers[0])

    asset_ret = df_prices.pct_change().dropna()
    dv = pd.DataFrame(index=df_prices.index, columns=tickers)
    for _, r in portfolio_df.iterrows():
        t, s, sd = r['Ticker'], r['Shares'], pd.to_datetime(r['Start Date']).date()
        dv[t] = df_prices[t] * s
        dv.loc[dv.index.date < sd, t] = 0.0
    prev = dv.shift(1)
    tot_prev = prev.sum(axis=1)
    wts = prev.div(tot_prev, axis=0).fillna(0)
    ci = asset_ret.index.intersection(wts.index)
    port_ret = (asset_ret.loc[ci] * wts.loc[ci]).sum(axis=1)
    port_ret = port_ret[tot_prev.loc[ci] > 0]
    cur_dv = dv.iloc[-1]
    cur_wts = (cur_dv / cur_dv.sum()).values
    return port_ret, asset_ret, cur_wts, cur_dv, df_prices


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

    port_ret, asset_ret, cur_wts, cur_dv, df_prices = build_portfolio_returns(portfolio_df, end_date)
    if port_ret is None or len(port_ret) == 0:
        st.warning("No data found. Check your tickers.")
        return

    df_benchmark = get_data(benchmark_tickers, portfolio_df['Start Date'].min(), end_date) if benchmark_tickers else pd.DataFrame()

    # Reconstruct daily values for charts
    dv = pd.DataFrame(index=df_prices.index, columns=tickers_list)
    for _, r in portfolio_df.iterrows():
        t, s, sd = r['Ticker'], r['Shares'], pd.to_datetime(r['Start Date']).date()
        dv[t] = df_prices[t] * s
        dv.loc[dv.index.date < sd, t] = 0.0

    m = calculate_metrics(port_ret, rf)
    cum_ret_series = (1 + port_ret).cumprod()
    period_label = f"{m['n_days']} trading days ({m['years_held']:.1f} yrs)"

    st.subheader("Portfolio Performance Metrics")
    st.caption(f"Metrics computed over **{period_label}** of live portfolio data.")
    r1 = st.columns(5)
    r1[0].metric("Portfolio Value", f"${cur_dv.sum():,.2f}", help="Current market value of all holdings.")
    r1[1].metric("Total Return", f"{m['total_return']:.2%}", help="Cumulative gain/loss since portfolio became active.")
    r1[2].metric("CAGR", f"{m['cagr']:.2%}", help="Compound Annual Growth Rate — geometrically annualized return over your actual holding period.")
    r1[3].metric("Annualized Volatility", f"{m['ann_vol']:.2%}", help="Daily return std dev scaled to one year (×√252).")
    r1[4].metric("Max Drawdown", f"{m['max_dd']:.2%}", help="Largest peak-to-trough decline experienced.")
    r2 = st.columns(5)
    r2[0].metric("Sharpe Ratio", f"{m['sharpe']:.2f}", help="(CAGR − RF) / Volatility. Excess return per unit of total risk.")
    r2[1].metric("Sortino Ratio", f"{m['sortino']:.2f}", help="Like Sharpe but only penalizes downside volatility.")
    r2[2].metric("Calmar Ratio", f"{m['calmar']:.2f}", help="CAGR / |Max Drawdown|. Return per unit of tail risk.")
    r2[3].metric("Holding Period", f"{m['n_days']} days", help="Trading days used to compute all metrics.")
    r2[4].metric("Risk-Free Rate", f"{rf:.2%}", help="Baseline riskless return used in Sharpe/Sortino.")
    st.markdown("---")

    st.subheader("Portfolio Visualizations")
    tab1, tab2, tab3, tab4 = st.tabs(["Performance & Value", "Current Allocation", "Correlation", "Risk Analysis"])

    with tab1:
        tvh = dv.sum(axis=1); tvh = tvh[tvh > 0]
        fig_val = go.Figure()
        fig_val.add_trace(go.Scatter(x=tvh.index, y=tvh, fill='tozeroy', name='Portfolio Value', line=dict(color='#636EFA', width=2)))
        fig_val.update_layout(title="Total Portfolio Value Over Time ($)", yaxis_tickformat="$,.2f", template="plotly_white")
        st.plotly_chart(fig_val, use_container_width=True)

        yl = "Percentage Return" if use_pct else "Value ($100 Invested)"
        pp = (1 + port_ret).cumprod() - 1 if use_pct else 100 * (1 + port_ret).cumprod()
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(x=pp.index, y=pp, name='My Portfolio', line=dict(width=3, color='#00CC96')))
        if not df_benchmark.empty:
            fa = port_ret.index[0]
            db = df_benchmark.to_frame(name=benchmark_tickers[0]) if isinstance(df_benchmark, pd.Series) else df_benchmark
            for tk in benchmark_tickers:
                try:
                    bp = db[tk].loc[fa:]
                    br = bp.pct_change().fillna(0)
                    bpl = (1 + br).cumprod() - 1 if use_pct else 100 * (1 + br).cumprod()
                    fig_cum.add_trace(go.Scatter(x=bpl.index, y=bpl, name=f"Benchmark: {tk}", line=dict(dash='dash', width=1.5)))
                except KeyError: continue
        fig_cum.update_layout(title=f"Relative Performance ({yl})", yaxis_tickformat=".2%" if use_pct else "$.2f", template="plotly_white")
        st.plotly_chart(fig_cum, use_container_width=True)

    with tab2:
        fig_pie = px.pie(names=cur_dv.index, values=cur_dv.values, title="Current Allocation", hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)

    with tab3:
        fig_corr = px.imshow(asset_ret.corr(), text_auto=True, title="Correlation Matrix", color_continuous_scale="RdBu_r")
        st.plotly_chart(fig_corr, use_container_width=True)

    with tab4:
        st.subheader("Deep Risk Analytics")
        tvh = dv.sum(axis=1); tvh = tvh[tvh > 0]
        dd_s = (tvh / tvh.cummax()) - 1
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=dd_s.index, y=dd_s, fill='tozeroy', name='Drawdown', line=dict(color='red', width=1)))
        fig_dd.update_layout(title="Underwater Drawdown Plot", yaxis_tickformat=".2%", template="plotly_white")
        st.plotly_chart(fig_dd, use_container_width=True)

        cr1, cr2 = st.columns(2)
        with cr1:
            has_spy = "SPY" in df_benchmark.columns if not df_benchmark.empty else False
            if has_spy:
                mr = df_benchmark["SPY"].loc[port_ret.index].pct_change().dropna()
                cd = port_ret.index.intersection(mr.index)
                beta = np.cov(port_ret.loc[cd], mr.loc[cd])[0][1] / np.var(mr.loc[cd])
                st.metric("Portfolio Beta (vs SPY)", f"{beta:.2f}")
                st.caption(f"Beta {beta:.2f} → {abs(beta-1):.0%} {'more' if beta > 1 else 'less'} volatile than S&P 500.")
            else:
                st.warning("Select 'SPY' in benchmarks to calculate Beta.")
        with cr2:
            n_obs = len(port_ret)
            w = min(60, max(10, n_obs // 5))
            rs = (port_ret.rolling(w).mean() / port_ret.rolling(w).std()) * np.sqrt(252)
            fig_sh = go.Figure()
            fig_sh.add_trace(go.Scatter(x=rs.index, y=rs, name=f'{w}D Rolling Sharpe', line=dict(color='orange')))
            fig_sh.update_layout(title=f"Rolling {w}-Day Sharpe (auto-sized)", yaxis_title="Sharpe", template="plotly_white")
            st.plotly_chart(fig_sh, use_container_width=True)

    st.markdown("---")
    st.subheader("Monte Carlo Simulation")
    with st.expander("Run Simulation Scenario"):
        sc1, sc2 = st.columns(2)
        n_sims = sc1.slider("Simulations", 200, 2000, 500)
        horizon = sc2.slider("Horizon (Days)", 30, 365, 252)
        if st.button("Run Simulation"):
            lr = np.log(1 + asset_ret)
            sim = np.random.multivariate_normal(lr.mean().values, lr.cov().values, size=(horizon, n_sims))
            ps = np.dot(np.exp(sim) - 1, cur_wts)
            sv = cur_dv.sum()
            paths = sv * np.cumprod(1 + ps, axis=0)
            fig_mc = go.Figure()
            for i in range(min(n_sims, 50)):
                fig_mc.add_trace(go.Scatter(x=list(range(1, horizon+1)), y=paths[:, i], mode='lines', line=dict(width=1, color='rgba(100,100,255,0.1)'), showlegend=False))
            fig_mc.add_trace(go.Scatter(x=list(range(1, horizon+1)), y=np.mean(paths, axis=1), name='Mean', line=dict(width=3, color='orange')))
            fig_mc.update_layout(title="Monte Carlo Projection", yaxis_tickformat="$,.2f", template="plotly_white")
            st.plotly_chart(fig_mc, use_container_width=True)
            v95 = np.percentile((paths[-1] / sv) - 1, 5)
            st.info(f"**95% VaR:** {v95:.2%} (Potential loss of ${sv * abs(v95):,.2f})")


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
        st.error(f"Could not fetch options for {ticker}.")
        return
    if not exps:
        st.warning(f"No listed options for {ticker}.")
        return

    selected_exp = st.selectbox("Expiration Date", exps)
    opt = tk.option_chain(selected_exp)
    st.subheader(f"Options Chain: {ticker} — {selected_exp}")

    tc, tp, tv = st.tabs(["Calls", "Puts", "Volatility Analysis"])
    with tc: st.dataframe(opt.calls, use_container_width=True)
    with tp: st.dataframe(opt.puts, use_container_width=True)

    with tv:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Call IV Smile")
            calls = opt.calls[opt.calls['impliedVolatility'] > 0]
            if not calls.empty:
                fig = px.line(calls, x='strike', y='impliedVolatility', title=f"Call IV Smile ({selected_exp})")
                fig.update_layout(yaxis_tickformat=".2%", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.subheader("Put IV Smile")
            puts = opt.puts[opt.puts['impliedVolatility'] > 0]
            if not puts.empty:
                fig = px.line(puts, x='strike', y='impliedVolatility', title=f"Put IV Smile ({selected_exp})")
                fig.update_layout(yaxis_tickformat=".2%", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("Volatility Term Structure")
        try:
            last_price = tk.info.get('currentPrice') or tk.info.get('regularMarketPrice')
        except Exception:
            last_price = None

        term_data = []
        if last_price:
            for exp in exps[:12]:
                try:
                    ch = tk.option_chain(exp)
                    c = ch.calls.copy()
                    c['dist'] = (c['strike'] - last_price).abs()
                    avg_iv = c.nsmallest(3, 'dist')['impliedVolatility'].mean()
                    dte = (pd.to_datetime(exp) - pd.Timestamp.now()).days
                    if avg_iv > 0: term_data.append({"Expiration": exp, "DTE": dte, "ATM IV": avg_iv})
                except Exception: continue
            if term_data:
                tdf = pd.DataFrame(term_data)
                fig = px.line(tdf, x='DTE', y='ATM IV', text='Expiration', title="IV Term Structure")
                fig.update_traces(textposition="top center")
                fig.update_layout(yaxis_tickformat=".2%", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("Historical Vol vs Implied Vol")
        hp = get_data([ticker], datetime.now().date() - timedelta(days=365), datetime.now().date())
        if not hp.empty:
            s = hp[ticker] if isinstance(hp, pd.DataFrame) and ticker in hp.columns else hp.squeeze()
            lr = np.log(s / s.shift(1)).dropna()
            fig = go.Figure()
            for w in [20, 40, 60, 90]:
                rv = lr.rolling(w).std() * np.sqrt(252)
                fig.add_trace(go.Scatter(x=rv.index, y=rv, name=f'{w}D Realized Vol'))
            if term_data:
                fig.add_hline(y=term_data[0]['ATM IV'], line_dash="dash", line_color="red", annotation_text=f"ATM IV ({term_data[0]['ATM IV']:.1%})")
            fig.update_layout(title=f"{ticker} — Realized vs Implied Vol", yaxis_tickformat=".2%", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)


# ==========================================
# Page 3: Factor Exposure & Attribution
# ==========================================

def show_factor_page(portfolio_df, end_date, rf):
    st.title("🧬 Factor Exposure & Attribution")
    st.markdown("---")
    st.caption("Regresses your portfolio returns against common factor proxies to decompose performance into systematic exposures and residual alpha.")

    port_ret, asset_ret, _, _, _ = build_portfolio_returns(portfolio_df, end_date)
    if port_ret is None or len(port_ret) < 60:
        st.warning("Need at least 60 trading days of portfolio data for factor regression.")
        return

    # Factor proxies via ETFs
    factor_map = {
        "Market (SPY)": "SPY", "Size-SMB (IWM-SPY)": ("IWM", "SPY"),
        "Value-HML (IWD-IWF)": ("IWD", "IWF"), "Momentum (MTUM)": "MTUM",
        "Quality (QUAL)": "QUAL", "Low Vol (USMV)": "USMV"
    }

    st.info("Using ETF proxies for factor returns: SPY (Market), IWM−SPY (Size), IWD−IWF (Value), MTUM (Momentum), QUAL (Quality), USMV (Low Vol).")

    all_etfs = list(set(v if isinstance(v, str) else v[0] for v in factor_map.values()) |
                     set(v[1] for v in factor_map.values() if isinstance(v, tuple)))

    earliest = portfolio_df['Start Date'].min()
    etf_prices = get_data(all_etfs, earliest, end_date)
    if etf_prices.empty:
        st.error("Could not fetch factor ETF data.")
        return

    etf_ret = etf_prices.pct_change().dropna()

    # Build factor return series
    factors = pd.DataFrame(index=etf_ret.index)
    for name, proxy in factor_map.items():
        try:
            if isinstance(proxy, str):
                factors[name] = etf_ret[proxy]
            else:
                factors[name] = etf_ret[proxy[0]] - etf_ret[proxy[1]]
        except KeyError:
            continue

    # Align
    ci = port_ret.index.intersection(factors.index)
    y = port_ret.loc[ci] - rf / 252  # excess returns
    X = factors.loc[ci]
    X = X.dropna(axis=1, how='all').dropna()
    y = y.loc[X.index]

    if len(y) < 30:
        st.warning("Not enough overlapping data for regression.")
        return

    # OLS regression
    X_c = add_constant(X)
    model = OLS(y, X_c).fit()

    st.subheader("Factor Regression Results")

    # Summary metrics
    c1, c2, c3 = st.columns(3)
    alpha_ann = model.params.get('const', 0) * 252
    c1.metric("Annualized Alpha", f"{alpha_ann:.2%}",
              help="Intercept of the regression, annualized. Represents return not explained by factor exposures.")
    c2.metric("R-squared", f"{model.rsquared:.2%}",
              help="Fraction of portfolio return variance explained by factor exposures.")
    c3.metric("Adj. R-squared", f"{model.rsquared_adj:.2%}",
              help="R-squared adjusted for the number of factors used.")

    # Betas table
    st.subheader("Factor Betas (Loadings)")
    betas = pd.DataFrame({
        "Beta": model.params.drop('const', errors='ignore'),
        "t-stat": model.tvalues.drop('const', errors='ignore'),
        "p-value": model.pvalues.drop('const', errors='ignore')
    })
    betas['Significant'] = betas['p-value'] < 0.05
    st.dataframe(betas.style.format({"Beta": "{:.4f}", "t-stat": "{:.2f}", "p-value": "{:.4f}"}), use_container_width=True)

    # Factor beta bar chart
    fig_beta = go.Figure()
    colors = ['#00CC96' if b > 0 else '#EF553B' for b in betas['Beta']]
    fig_beta.add_trace(go.Bar(x=betas.index, y=betas['Beta'], marker_color=colors))
    fig_beta.update_layout(title="Factor Exposures (Betas)", yaxis_title="Beta Loading", template="plotly_white")
    st.plotly_chart(fig_beta, use_container_width=True)

    # Return attribution
    st.subheader("Return Attribution")
    st.caption("Decomposes your portfolio's total excess return into contributions from each factor plus alpha.")
    contrib = (X.mean() * model.params.drop('const', errors='ignore')) * 252
    contrib['Alpha'] = alpha_ann
    fig_attr = go.Figure()
    colors_attr = ['#00CC96' if v > 0 else '#EF553B' for v in contrib]
    fig_attr.add_trace(go.Bar(x=contrib.index, y=contrib.values, marker_color=colors_attr))
    fig_attr.update_layout(title="Annualized Return Attribution", yaxis_title="Contribution to Return", yaxis_tickformat=".2%", template="plotly_white")
    st.plotly_chart(fig_attr, use_container_width=True)

    # Rolling alpha
    st.subheader("Rolling Alpha (60-Day)")
    w = min(60, max(20, len(y) // 5))
    rolling_alpha = []
    for i in range(w, len(y)):
        chunk_y = y.iloc[i-w:i]
        chunk_X = add_constant(X.iloc[i-w:i])
        try:
            rm = OLS(chunk_y, chunk_X).fit()
            rolling_alpha.append({'date': y.index[i], 'alpha': rm.params.get('const', 0) * 252})
        except Exception:
            continue
    if rolling_alpha:
        ra_df = pd.DataFrame(rolling_alpha).set_index('date')
        fig_ra = go.Figure()
        fig_ra.add_trace(go.Scatter(x=ra_df.index, y=ra_df['alpha'], name='Rolling Alpha', line=dict(color='#636EFA')))
        fig_ra.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_ra.update_layout(title=f"Rolling {w}-Day Annualized Alpha", yaxis_tickformat=".2%", template="plotly_white")
        st.plotly_chart(fig_ra, use_container_width=True)


# ==========================================
# Page 4: Screener & Signal Scanner
# ==========================================

def show_screener_page():
    st.title("🔍 Screener & Signal Scanner")
    st.markdown("---")

    tab_signals, tab_coint = st.tabs(["Technical Signals", "Cointegration Scanner"])

    with tab_signals:
        st.subheader("Multi-Ticker Signal Scanner")
        universe_input = st.text_input("Ticker Universe (comma-separated)",
                                        "AAPL, MSFT, GOOG, AMZN, META, NVDA, TSLA, JPM, V, MA, UNH, XOM, JNJ, PG, HD")
        universe = [t.strip().upper() for t in universe_input.split(",") if t.strip()]
        lookback = st.slider("Lookback Period (Days)", 30, 365, 120)

        if st.button("Scan Signals") and universe:
            prices = get_data(universe, datetime.now().date() - timedelta(days=lookback + 50), datetime.now().date())
            if prices.empty:
                st.error("Could not fetch data.")
                return

            results = []
            for tk in universe:
                try:
                    s = prices[tk].dropna()
                    if len(s) < 30: continue
                    ret = s.pct_change().dropna()

                    # RSI
                    rsi = compute_rsi(s).iloc[-1]

                    # Z-score (mean reversion signal)
                    ma = s.rolling(20).mean()
                    std = s.rolling(20).std()
                    zscore = ((s - ma) / std).iloc[-1]

                    # Momentum: cumulative return over last 20 days
                    mom_20 = (s.iloc[-1] / s.iloc[-21] - 1) if len(s) > 21 else np.nan

                    # Volatility (20D annualized)
                    vol_20 = ret.tail(20).std() * np.sqrt(252)

                    # Volume trend (if available, use price as proxy)
                    results.append({
                        "Ticker": tk, "Last Price": s.iloc[-1], "RSI (14)": rsi,
                        "Z-Score (20D)": zscore, "20D Momentum": mom_20,
                        "20D Ann. Vol": vol_20,
                        "Signal": "Oversold" if rsi < 30 else ("Overbought" if rsi > 70 else
                                  ("Mean-Rev Long" if zscore < -2 else ("Mean-Rev Short" if zscore > 2 else "Neutral")))
                    })
                except Exception:
                    continue

            if results:
                df_res = pd.DataFrame(results)
                # Color the signal column
                st.dataframe(df_res.style.format({
                    "Last Price": "${:.2f}", "RSI (14)": "{:.1f}", "Z-Score (20D)": "{:.2f}",
                    "20D Momentum": "{:.2%}", "20D Ann. Vol": "{:.2%}"
                }).applymap(lambda v: 'color: green' if v in ['Oversold', 'Mean-Rev Long'] else
                            ('color: red' if v in ['Overbought', 'Mean-Rev Short'] else ''),
                            subset=['Signal']),
                    use_container_width=True)

                # RSI chart
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Bar(x=df_res['Ticker'], y=df_res['RSI (14)'],
                                          marker_color=['green' if r < 30 else 'red' if r > 70 else 'gray' for r in df_res['RSI (14)']]))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                fig_rsi.update_layout(title="RSI Heatmap", template="plotly_white")
                st.plotly_chart(fig_rsi, use_container_width=True)

    with tab_coint:
        st.subheader("Cointegration Pair Scanner")
        st.caption("Scans all pairs in your universe for cointegration, ranks by p-value and estimated half-life.")
        pair_universe = st.text_input("Pair Universe (comma-separated)", "XOM, CVX, COP, SLB, EOG, MPC, VLO, PSX", key="coint_univ")
        pair_tickers = [t.strip().upper() for t in pair_universe.split(",") if t.strip()]
        coint_lookback = st.slider("Lookback (Days)", 60, 500, 252, key="coint_lb")

        if st.button("Scan Pairs") and len(pair_tickers) >= 2:
            prices = get_data(pair_tickers, datetime.now().date() - timedelta(days=coint_lookback + 10), datetime.now().date())
            if prices.empty or len(prices.columns) < 2:
                st.error("Need at least 2 tickers with data.")
                return

            pairs = []
            tks = [c for c in prices.columns if prices[c].notna().sum() > 30]
            for i in range(len(tks)):
                for j in range(i + 1, len(tks)):
                    t1, t2 = tks[i], tks[j]
                    s1, s2 = prices[t1].dropna(), prices[t2].dropna()
                    ci = s1.index.intersection(s2.index)
                    if len(ci) < 30: continue
                    s1, s2 = s1.loc[ci], s2.loc[ci]

                    # Cointegration test
                    _, pval, _ = coint(s1, s2)

                    # Spread and half-life (OLS hedge ratio)
                    X_hr = add_constant(s2.values)
                    hr_model = OLS(s1.values, X_hr).fit()
                    spread = s1 - hr_model.params[1] * s2

                    # Half-life via AR(1)
                    sp_lag = spread.shift(1).dropna()
                    sp_diff = spread.diff().dropna()
                    ci2 = sp_lag.index.intersection(sp_diff.index)
                    if len(ci2) < 10: continue
                    ar_model = OLS(sp_diff.loc[ci2], add_constant(sp_lag.loc[ci2])).fit()
                    hl = -np.log(2) / ar_model.params.iloc[1] if ar_model.params.iloc[1] < 0 else np.nan

                    # Current z-score of spread
                    z = (spread.iloc[-1] - spread.mean()) / spread.std()

                    # Hurst exponent (simplified R/S)
                    ts = spread.values
                    n = len(ts)
                    if n > 20:
                        max_k = min(n // 2, 100)
                        rs_list = []
                        for k in [20, 50, max_k]:
                            if k > n: continue
                            rs_vals = []
                            for start in range(0, n - k, k):
                                chunk = ts[start:start+k]
                                m_c = chunk.mean()
                                dev = np.cumsum(chunk - m_c)
                                r_s = (dev.max() - dev.min()) / (chunk.std() + 1e-10)
                                rs_vals.append(r_s)
                            if rs_vals:
                                rs_list.append((np.log(k), np.log(np.mean(rs_vals))))
                        if len(rs_list) >= 2:
                            xs, ys = zip(*rs_list)
                            hurst = np.polyfit(xs, ys, 1)[0]
                        else:
                            hurst = np.nan
                    else:
                        hurst = np.nan

                    pairs.append({"Pair": f"{t1}/{t2}", "Coint p-val": pval, "Half-Life": hl,
                                  "Spread Z": z, "Hurst": hurst, "Hedge Ratio": hr_model.params[1],
                                  "Mean-Reverting": "✅" if pval < 0.05 and (not np.isnan(hurst) and hurst < 0.5) else "❌"})

            if pairs:
                pdf = pd.DataFrame(pairs).sort_values("Coint p-val")
                st.dataframe(pdf.style.format({
                    "Coint p-val": "{:.4f}", "Half-Life": "{:.1f}", "Spread Z": "{:.2f}",
                    "Hurst": "{:.3f}", "Hedge Ratio": "{:.4f}"
                }), use_container_width=True)

                # Show spread chart for top pair
                top = pdf.iloc[0]
                t1, t2 = top['Pair'].split('/')
                s1, s2 = prices[t1].dropna(), prices[t2].dropna()
                ci = s1.index.intersection(s2.index)
                spread = s1.loc[ci] - top['Hedge Ratio'] * s2.loc[ci]
                mu, sigma = spread.mean(), spread.std()

                fig_sp = go.Figure()
                fig_sp.add_trace(go.Scatter(x=spread.index, y=spread, name='Spread', line=dict(color='#636EFA')))
                fig_sp.add_hline(y=mu, line_dash="solid", line_color="gray", annotation_text="Mean")
                fig_sp.add_hline(y=mu + 2*sigma, line_dash="dash", line_color="red", annotation_text="+2σ")
                fig_sp.add_hline(y=mu - 2*sigma, line_dash="dash", line_color="green", annotation_text="−2σ")
                fig_sp.update_layout(title=f"Top Pair Spread: {top['Pair']}", template="plotly_white")
                st.plotly_chart(fig_sp, use_container_width=True)


# ==========================================
# Page 5: Trade Journal & Post-Mortem
# ==========================================

def show_journal_page():
    st.title("📓 Trade Journal & Post-Mortem")
    st.markdown("---")

    # Initialize session state
    if 'trades' not in st.session_state:
        st.session_state.trades = []

    tab_log, tab_analysis, tab_data = st.tabs(["Log Trade", "Post-Mortem Analytics", "Trade History"])

    with tab_log:
        st.subheader("Log a New Trade")
        with st.form("trade_form"):
            c1, c2, c3 = st.columns(3)
            ticker = c1.text_input("Ticker", "AAPL")
            direction = c2.selectbox("Direction", ["Long", "Short"])
            shares = c3.number_input("Shares", min_value=0.01, value=10.0)

            c4, c5, c6 = st.columns(3)
            entry_price = c4.number_input("Entry Price", min_value=0.01, value=150.0)
            exit_price = c5.number_input("Exit Price (0 = still open)", min_value=0.0, value=0.0)
            entry_date = c6.date_input("Entry Date", value=datetime.now().date() - timedelta(days=7))

            c7, c8 = st.columns(2)
            exit_date = c7.date_input("Exit Date", value=datetime.now().date())
            thesis = c8.text_area("Trade Thesis", placeholder="Why did you enter this trade?")

            tags = st.multiselect("Tags", ["Momentum", "Mean Reversion", "Breakout", "Earnings", "Macro", "Technical", "Fundamental", "Pairs"])

            submitted = st.form_submit_button("Log Trade")
            if submitted:
                pnl_mult = 1 if direction == "Long" else -1
                is_closed = exit_price > 0
                pnl = pnl_mult * (exit_price - entry_price) * shares if is_closed else 0
                pnl_pct = pnl_mult * (exit_price / entry_price - 1) if is_closed else 0
                holding = (exit_date - entry_date).days if is_closed else (datetime.now().date() - entry_date).days

                trade = {
                    "Ticker": ticker.upper(), "Direction": direction, "Shares": shares,
                    "Entry": entry_price, "Exit": exit_price if is_closed else None,
                    "Entry Date": str(entry_date), "Exit Date": str(exit_date) if is_closed else None,
                    "P&L ($)": pnl, "P&L (%)": pnl_pct, "Holding Days": holding,
                    "Status": "Closed" if is_closed else "Open", "Thesis": thesis, "Tags": tags
                }
                st.session_state.trades.append(trade)
                st.success(f"Logged {'closed' if is_closed else 'open'} {direction} trade: {ticker.upper()}")

        # CSV upload for trade history
        st.markdown("---")
        uploaded = st.file_uploader("Import Trade History (CSV)", type=["csv"], key="trade_csv")
        if uploaded:
            try:
                imp = pd.read_csv(uploaded)
                st.session_state.trades = imp.to_dict('records')
                st.success(f"Imported {len(imp)} trades.")
            except Exception as e:
                st.error(f"Import error: {e}")

    with tab_analysis:
        st.subheader("Post-Mortem Analytics")
        trades = st.session_state.trades
        closed = [t for t in trades if t.get('Status') == 'Closed']

        if not closed:
            st.info("Log some closed trades to see analytics.")
            return

        df_t = pd.DataFrame(closed)
        wins = df_t[df_t['P&L ($)'] > 0]
        losses = df_t[df_t['P&L ($)'] <= 0]

        # Core stats
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Trades", len(df_t))
        c2.metric("Win Rate", f"{len(wins)/len(df_t):.1%}",
                   help="Percentage of trades with positive P&L.")
        avg_win = wins['P&L ($)'].mean() if len(wins) > 0 else 0
        avg_loss = abs(losses['P&L ($)'].mean()) if len(losses) > 0 else 0
        profit_factor = avg_win / avg_loss if avg_loss > 0 else float('inf')
        c3.metric("Profit Factor", f"{profit_factor:.2f}",
                   help="Average win / Average loss. Above 1.5 is solid.")
        c4.metric("Total P&L", f"${df_t['P&L ($)'].sum():,.2f}")
        expectancy = df_t['P&L ($)'].mean()
        c5.metric("Expectancy ($/trade)", f"${expectancy:,.2f}",
                   help="Average P&L per trade. Positive = your edge in dollar terms.")

        c6, c7, c8 = st.columns(3)
        c6.metric("Avg Win", f"${avg_win:,.2f}")
        c7.metric("Avg Loss", f"-${avg_loss:,.2f}")
        c8.metric("Avg Holding Period", f"{df_t['Holding Days'].mean():.0f} days")

        # P&L distribution
        st.subheader("P&L Distribution")
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(x=df_t['P&L ($)'], nbinsx=20, marker_color='#636EFA'))
        fig_dist.add_vline(x=0, line_dash="dash", line_color="red")
        fig_dist.update_layout(title="Trade P&L Distribution", xaxis_title="P&L ($)", template="plotly_white")
        st.plotly_chart(fig_dist, use_container_width=True)

        # Equity curve
        st.subheader("Cumulative P&L (Equity Curve)")
        df_t_sorted = df_t.sort_values('Exit Date')
        df_t_sorted['Cum P&L'] = df_t_sorted['P&L ($)'].cumsum()
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(x=df_t_sorted['Exit Date'], y=df_t_sorted['Cum P&L'],
                                     fill='tozeroy', line=dict(color='#00CC96', width=2)))
        fig_eq.update_layout(title="Cumulative P&L", yaxis_tickformat="$,.2f", template="plotly_white")
        st.plotly_chart(fig_eq, use_container_width=True)

        # Performance by direction
        if len(df_t['Direction'].unique()) > 1:
            st.subheader("Long vs Short Performance")
            dir_stats = df_t.groupby('Direction').agg(
                Trades=('P&L ($)', 'count'), WinRate=('P&L ($)', lambda x: (x > 0).mean()),
                AvgPnL=('P&L ($)', 'mean'), TotalPnL=('P&L ($)', 'sum')
            ).reset_index()
            st.dataframe(dir_stats.style.format({"WinRate": "{:.1%}", "AvgPnL": "${:,.2f}", "TotalPnL": "${:,.2f}"}),
                          use_container_width=True)

        # Performance by tag
        if any(t.get('Tags') for t in closed):
            st.subheader("Performance by Strategy Tag")
            tag_rows = []
            for t in closed:
                for tag in (t.get('Tags') or []):
                    tag_rows.append({"Tag": tag, "P&L": t['P&L ($)']})
            if tag_rows:
                tag_df = pd.DataFrame(tag_rows)
                tag_stats = tag_df.groupby('Tag').agg(Trades=('P&L', 'count'), AvgPnL=('P&L', 'mean'), TotalPnL=('P&L', 'sum')).reset_index()
                st.dataframe(tag_stats.style.format({"AvgPnL": "${:,.2f}", "TotalPnL": "${:,.2f}"}), use_container_width=True)

    with tab_data:
        st.subheader("Full Trade History")
        if st.session_state.trades:
            df_all = pd.DataFrame(st.session_state.trades)
            st.dataframe(df_all, use_container_width=True)
            csv = df_all.to_csv(index=False).encode('utf-8')
            st.download_button("💾 Export Trade History", csv, "trade_journal.csv", "text/csv")
            if st.button("🗑️ Clear All Trades", type="secondary"):
                st.session_state.trades = []
                st.rerun()
        else:
            st.info("No trades logged yet.")


# ==========================================
# Page 6: Macro Regime Dashboard
# ==========================================

@st.cache_data(ttl=3600)
def fetch_macro_data():
    """Fetch macro indicators from Yahoo Finance."""
    tickers = {
        "^TNX": "10Y Yield", "^FVX": "5Y Yield", "^IRX": "3M Yield",
        "^VIX": "VIX", "SPY": "SPY", "DX-Y.NYB": "DXY", "GC=F": "Gold",
        "HYG": "HY Corp Bond", "LQD": "IG Corp Bond", "TLT": "20Y Treasury"
    }
    end = datetime.now().date()
    start = end - timedelta(days=756)  # ~3 years
    data = {}
    for tk, name in tickers.items():
        try:
            d = yf.download(tk, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'), auto_adjust=True, progress=False)
            if not d.empty:
                data[name] = d['Close'].squeeze() if isinstance(d['Close'], pd.DataFrame) else d['Close']
        except Exception:
            continue
    return pd.DataFrame(data)

def show_macro_page():
    st.title("🌍 Macro Regime Dashboard")
    st.markdown("---")
    st.caption("Real-time macro regime indicators to contextualize your portfolio's environment.")

    macro = fetch_macro_data()
    if macro.empty:
        st.error("Could not fetch macro data.")
        return

    # --- Current Readings ---
    st.subheader("Current Macro Snapshot")
    c1, c2, c3, c4, c5 = st.columns(5)

    latest = macro.iloc[-1]
    prev = macro.iloc[-2] if len(macro) > 1 else latest

    if '10Y Yield' in macro.columns and '3M Yield' in macro.columns:
        spread_10y3m = latest.get('10Y Yield', 0) - latest.get('3M Yield', 0)
        prev_spread = prev.get('10Y Yield', 0) - prev.get('3M Yield', 0)
        c1.metric("10Y-3M Spread", f"{spread_10y3m:.2f}%", delta=f"{spread_10y3m - prev_spread:.2f}%",
                   help="Yield curve slope. Negative = inverted (recession signal).")

    if 'VIX' in macro.columns:
        vix = latest['VIX']
        c2.metric("VIX", f"{vix:.1f}", delta=f"{vix - prev.get('VIX', vix):.1f}",
                   help="CBOE Volatility Index. >20 = elevated fear, >30 = crisis-level.")

    if 'HY Corp Bond' in macro.columns and 'IG Corp Bond' in macro.columns:
        hy_ret = macro['HY Corp Bond'].pct_change().tail(20).mean() * 252
        ig_ret = macro['IG Corp Bond'].pct_change().tail(20).mean() * 252
        credit_momentum = hy_ret - ig_ret
        c3.metric("HY-IG Momentum", f"{credit_momentum:.2%}",
                   help="Relative performance of high-yield vs investment-grade bonds. Negative = credit stress.")

    if 'DXY' in macro.columns:
        dxy = latest['DXY']
        c4.metric("Dollar Index (DXY)", f"{dxy:.1f}", delta=f"{dxy - prev.get('DXY', dxy):.1f}",
                   help="US Dollar strength. Rising DXY = tightening financial conditions globally.")

    if 'Gold' in macro.columns:
        gold = latest['Gold']
        c5.metric("Gold", f"${gold:,.0f}", delta=f"${gold - prev.get('Gold', gold):,.0f}",
                   help="Safe haven demand indicator.")

    # --- Regime Classification ---
    st.markdown("---")
    st.subheader("Regime Classification")

    regime_signals = {}
    if '10Y Yield' in macro.columns and '3M Yield' in macro.columns:
        curve = macro['10Y Yield'] - macro['3M Yield']
        regime_signals['Yield Curve'] = "Inverted ⚠️" if curve.iloc[-1] < 0 else ("Flat" if curve.iloc[-1] < 0.5 else "Normal ✅")
    if 'VIX' in macro.columns:
        v = macro['VIX'].iloc[-1]
        regime_signals['Vol Regime'] = "Crisis 🔴" if v > 30 else ("Elevated 🟡" if v > 20 else "Low Vol 🟢")
    if 'HY Corp Bond' in macro.columns and 'IG Corp Bond' in macro.columns:
        regime_signals['Credit'] = "Stress 🔴" if credit_momentum < -0.05 else ("Neutral 🟡" if credit_momentum < 0.02 else "Risk-On 🟢")
    if 'DXY' in macro.columns:
        dxy_ma = macro['DXY'].rolling(50).mean()
        regime_signals['Dollar Trend'] = "Strengthening 📈" if macro['DXY'].iloc[-1] > dxy_ma.iloc[-1] else "Weakening 📉"

    # Overall regime
    risk_score = 0
    for k, v in regime_signals.items():
        if '🔴' in v or '⚠️' in v: risk_score += 2
        elif '🟡' in v: risk_score += 1
    overall = "RISK-OFF 🔴" if risk_score >= 4 else ("CAUTIOUS 🟡" if risk_score >= 2 else "RISK-ON 🟢")

    rc1, rc2 = st.columns([1, 2])
    with rc1:
        st.metric("Overall Regime", overall,
                   help="Composite signal from yield curve, VIX, credit, and dollar. Higher risk score = more defensive positioning warranted.")
        for k, v in regime_signals.items():
            st.caption(f"**{k}:** {v}")

    with rc2:
        # Yield curve history
        if '10Y Yield' in macro.columns and '3M Yield' in macro.columns:
            curve = macro['10Y Yield'] - macro['3M Yield']
            fig_yc = go.Figure()
            fig_yc.add_trace(go.Scatter(x=curve.index, y=curve, fill='tozeroy', name='10Y-3M Spread',
                                         line=dict(color=np.where(curve < 0, 'red', '#636EFA').tolist()[-1])))
            fig_yc.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Inversion")
            fig_yc.update_layout(title="Yield Curve (10Y − 3M)", yaxis_title="Spread (%)", template="plotly_white")
            st.plotly_chart(fig_yc, use_container_width=True)

    # --- VIX Term Structure Proxy ---
    st.markdown("---")
    st.subheader("Volatility & Risk Indicators")
    vc1, vc2 = st.columns(2)

    with vc1:
        if 'VIX' in macro.columns:
            fig_vix = go.Figure()
            vix_s = macro['VIX'].dropna()
            fig_vix.add_trace(go.Scatter(x=vix_s.index, y=vix_s, name='VIX', line=dict(color='orange')))
            fig_vix.add_hline(y=20, line_dash="dash", line_color="yellow", annotation_text="Elevated")
            fig_vix.add_hline(y=30, line_dash="dash", line_color="red", annotation_text="Crisis")
            fig_vix.update_layout(title="VIX History", template="plotly_white")
            st.plotly_chart(fig_vix, use_container_width=True)

    with vc2:
        if 'HY Corp Bond' in macro.columns and 'IG Corp Bond' in macro.columns:
            hy_ig = (macro['HY Corp Bond'] / macro['IG Corp Bond'])
            hy_ig = hy_ig / hy_ig.iloc[0]  # normalize
            fig_credit = go.Figure()
            fig_credit.add_trace(go.Scatter(x=hy_ig.index, y=hy_ig, name='HY/IG Ratio', line=dict(color='#EF553B')))
            fig_credit.update_layout(title="Credit Risk Appetite (HY/IG Ratio)", template="plotly_white",
                                      yaxis_title="Ratio (normalized)")
            st.plotly_chart(fig_credit, use_container_width=True)

    # --- Dollar & Gold ---
    st.subheader("Dollar & Safe Haven Flows")
    dc1, dc2 = st.columns(2)
    with dc1:
        if 'DXY' in macro.columns:
            fig_dxy = go.Figure()
            dxy_s = macro['DXY'].dropna()
            ma50 = dxy_s.rolling(50).mean()
            fig_dxy.add_trace(go.Scatter(x=dxy_s.index, y=dxy_s, name='DXY', line=dict(color='#636EFA')))
            fig_dxy.add_trace(go.Scatter(x=ma50.index, y=ma50, name='50D MA', line=dict(dash='dash', color='gray')))
            fig_dxy.update_layout(title="US Dollar Index (DXY)", template="plotly_white")
            st.plotly_chart(fig_dxy, use_container_width=True)
    with dc2:
        if 'Gold' in macro.columns:
            fig_gold = go.Figure()
            gs = macro['Gold'].dropna()
            fig_gold.add_trace(go.Scatter(x=gs.index, y=gs, name='Gold', line=dict(color='goldenrod')))
            fig_gold.update_layout(title="Gold ($/oz)", yaxis_tickformat="$,.0f", template="plotly_white")
            st.plotly_chart(fig_gold, use_container_width=True)


# ==========================================
# Main Navigation Controller
# ==========================================

def main():
    page, portfolio_df, end_date, benchmarks, use_pct, rf = render_sidebar()

    if page == "Portfolio Overview":
        show_main_page(portfolio_df, end_date, benchmarks, use_pct, rf)
    elif page == "Volatility & Options Lab":
        show_volatility_page(portfolio_df)
    elif page == "Factor Exposure & Attribution":
        show_factor_page(portfolio_df, end_date, rf)
    elif page == "Screener & Signal Scanner":
        show_screener_page()
    elif page == "Trade Journal & Post-Mortem":
        show_journal_page()
    elif page == "Macro Regime Dashboard":
        show_macro_page()

if __name__ == "__main__":
    main()