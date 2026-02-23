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
from scipy.stats import norm
import requests, re, logging, warnings, math
warnings.filterwarnings('ignore')

# Optional: better sentiment if installed
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER = SentimentIntensityAnalyzer()
    HAS_VADER = True
except ImportError:
    HAS_VADER = False

try:
    import feedparser
    HAS_FEEDPARSER = True
except ImportError:
    HAS_FEEDPARSER = False

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
        tnx = yf.Ticker("^TNX"); hist = tnx.history(period="5d")
        if not hist.empty: return hist['Close'].dropna().iloc[-1] / 100
    except Exception: pass
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
    yrs = n / 252; cum_ret = (1 + daily_returns).prod() - 1; eg = 1 + cum_ret
    cagr = eg ** (1 / yrs) - 1 if yrs > 0 and eg > 0 else 0.0
    vol = daily_returns.std() * np.sqrt(252)
    sharpe = (cagr - rf) / vol if vol != 0 else 0.0
    ds = daily_returns[daily_returns < 0]; ds_std = ds.std() * np.sqrt(252) if len(ds) > 0 else 0.0
    sortino = (cagr - rf) / ds_std if ds_std != 0 else 0.0
    cs = (1 + daily_returns).cumprod(); mdd = calculate_max_drawdown(cs)
    calmar = cagr / abs(mdd) if mdd != 0 else 0.0
    return {'n_days': n, 'years_held': yrs, 'total_return': cum_ret, 'cagr': cagr,
            'ann_vol': vol, 'sharpe': sharpe, 'sortino': sortino, 'max_dd': mdd, 'calmar': calmar}

def compute_rsi(series, window=14):
    delta = series.diff(); gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean(); rs = gain / loss
    return 100 - (100 / (1 + rs))


# ==========================================
# Options Pricing
# ==========================================

def bs_price(S, K, T, r, sigma, option_type='call'):
    if T <= 0 or sigma <= 0: return max(S - K, 0) if option_type == 'call' else max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T)); d2 = d1 - sigma * np.sqrt(T)
    return (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)) if option_type == 'call' else (K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))

def bs_greeks(S, K, T, r, sigma, option_type='call'):
    if T <= 0 or sigma <= 0:
        return {'delta': 1.0 if option_type == 'call' else -1.0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0, 'price': bs_price(S, K, T, r, sigma, option_type)}
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T)); d2 = d1 - sigma * np.sqrt(T)
    price = bs_price(S, K, T, r, sigma, option_type); gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    if option_type == 'call':
        delta = norm.cdf(d1); theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        delta = norm.cdf(d1) - 1; theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    return {'price': price, 'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega, 'rho': rho}

def binomial_greeks(S, K, T, r, sigma, option_type='call', steps=200, american=True):
    def _bp(S, K, T, r, sigma, ot, n, am):
        if T <= 0 or sigma <= 0: return max(S - K, 0) if ot == 'call' else max(K - S, 0)
        dt = T / n; u = np.exp(sigma * np.sqrt(dt)); d = 1 / u; p = (np.exp(r * dt) - d) / (u - d); disc = np.exp(-r * dt)
        pr = S * u ** np.arange(n, -1, -1) * d ** np.arange(0, n + 1)
        v = np.maximum(pr - K, 0) if ot == 'call' else np.maximum(K - pr, 0)
        for i in range(n - 1, -1, -1):
            v = disc * (p * v[:-1] + (1 - p) * v[1:])
            if am:
                pi = S * u ** np.arange(i, -1, -1) * d ** np.arange(0, i + 1)
                v = np.maximum(v, np.maximum(pi - K, 0) if ot == 'call' else np.maximum(K - pi, 0))
        return v[0]
    price = _bp(S, K, T, r, sigma, option_type, steps, american); dS = S * 0.01
    pu = _bp(S + dS, K, T, r, sigma, option_type, steps, american); pd_v = _bp(S - dS, K, T, r, sigma, option_type, steps, american)
    return {'price': price, 'delta': (pu - pd_v) / (2 * dS), 'gamma': (pu - 2 * price + pd_v) / dS**2,
            'theta': _bp(S, K, max(T - 1/365, 1e-6), r, sigma, option_type, steps, american) - price,
            'vega': _bp(S, K, T, r, sigma + 0.01, option_type, steps, american) - price,
            'rho': _bp(S, K, T, r + 0.01, sigma, option_type, steps, american) - price}


# ==========================================
# News & Sentiment Engine
# ==========================================

# Loughran-McDonald financial sentiment lexicon (curated subset)
_POS_WORDS = {
    'beat', 'beats', 'beating', 'exceeded', 'exceeds', 'upgrade', 'upgraded', 'upgrades', 'upside',
    'bullish', 'rally', 'rallies', 'rallied', 'surge', 'surges', 'surged', 'soar', 'soars', 'soared',
    'gain', 'gains', 'gained', 'profit', 'profits', 'profitable', 'growth', 'growing', 'grew',
    'outperform', 'outperforms', 'outperformed', 'positive', 'strong', 'stronger', 'strongest',
    'record', 'high', 'highs', 'boom', 'booming', 'breakout', 'momentum', 'accelerate', 'accelerating',
    'optimism', 'optimistic', 'confidence', 'confident', 'recovery', 'recovering', 'rebound', 'rebounds',
    'innovation', 'innovative', 'dividend', 'buyback', 'expansion', 'expanding', 'opportunity',
    'overweight', 'buy', 'accumulate', 'attractive', 'favorable', 'robust', 'resilient', 'success',
    'winner', 'winning', 'jumps', 'jumped', 'climbs', 'climbed', 'rises', 'rose', 'risen', 'up'
}
_NEG_WORDS = {
    'miss', 'missed', 'misses', 'missing', 'downgrade', 'downgraded', 'downgrades', 'downside',
    'bearish', 'crash', 'crashes', 'crashed', 'plunge', 'plunges', 'plunged', 'tumble', 'tumbles',
    'loss', 'losses', 'losing', 'decline', 'declines', 'declined', 'declining', 'fall', 'falls', 'fell',
    'underperform', 'underperforms', 'negative', 'weak', 'weaker', 'weakest', 'weakness',
    'low', 'lows', 'bust', 'recession', 'recessionary', 'slowdown', 'slowing', 'decelerate',
    'pessimism', 'pessimistic', 'fear', 'fears', 'worried', 'worry', 'concern', 'concerns', 'concerned',
    'risk', 'risky', 'volatile', 'volatility', 'uncertainty', 'uncertain', 'crisis', 'default',
    'sell', 'underweight', 'avoid', 'cut', 'cuts', 'layoff', 'layoffs', 'fired', 'bankruptcy',
    'fraud', 'scandal', 'investigation', 'lawsuit', 'fine', 'fined', 'penalty', 'warning', 'warns',
    'drops', 'dropped', 'sinks', 'sank', 'slips', 'slipped', 'down', 'plummets', 'tanks', 'tanked',
    'debt', 'deficit', 'inflation', 'tariff', 'tariffs', 'sanctions', 'shutdown', 'war'
}
_INTENSIFIERS = {'very', 'extremely', 'significantly', 'sharply', 'dramatically', 'massive', 'huge', 'major'}
_NEGATORS = {'not', 'no', "n't", 'never', 'neither', 'nor', 'hardly', 'barely', 'without'}

def score_headline_lexicon(text):
    """Financial-specific sentiment scoring using Loughran-McDonald style lexicon."""
    words = re.findall(r"[a-z']+", text.lower())
    pos_score = 0; neg_score = 0; intensity = 1.0
    for i, w in enumerate(words):
        if w in _INTENSIFIERS: intensity = 1.5; continue
        negated = any(words[max(0, i-j)] in _NEGATORS for j in range(1, 4) if i - j >= 0)
        if w in _POS_WORDS:
            if negated: neg_score += intensity
            else: pos_score += intensity
        elif w in _NEG_WORDS:
            if negated: pos_score += intensity
            else: neg_score += intensity
        intensity = 1.0
    total = pos_score + neg_score
    if total == 0: return 0.0
    return (pos_score - neg_score) / total  # [-1, 1]

def score_headline(text):
    """Use VADER if available, else fall back to lexicon."""
    if HAS_VADER:
        return VADER.polarity_scores(text)['compound']  # [-1, 1]
    return score_headline_lexicon(text)

def sentiment_label(score):
    if score >= 0.3: return "Bullish 🟢"
    elif score <= -0.3: return "Bearish 🔴"
    elif score >= 0.1: return "Slightly Bullish 🟡"
    elif score <= -0.1: return "Slightly Bearish 🟠"
    return "Neutral ⚪"

def sentiment_color(score):
    if score >= 0.3: return "#00CC96"
    elif score <= -0.3: return "#EF553B"
    elif score >= 0.1: return "#FFA15A"
    elif score <= -0.1: return "#FF6692"
    return "#636EFA"

@st.cache_data(ttl=900)
def fetch_ticker_news(ticker, max_articles=25):
    """Multi-source news aggregation for a single ticker."""
    articles = []

    # Source 1: yfinance news
    try:
        tk = yf.Ticker(ticker)
        news = tk.news
        if news:
            for n in news[:max_articles]:
                title = n.get('title', '')
                if not title: continue
                ts = n.get('providerPublishTime', 0)
                articles.append({
                    'title': title, 'publisher': n.get('publisher', 'Yahoo Finance'),
                    'link': n.get('link', ''), 'source': 'Yahoo Finance',
                    'timestamp': datetime.fromtimestamp(ts) if ts else datetime.now(),
                    'thumbnail': n.get('thumbnail', {}).get('resolutions', [{}])[0].get('url', '') if n.get('thumbnail') else ''
                })
    except Exception as e:
        logger.debug(f"yfinance news error for {ticker}: {e}")

    # Source 2: Google News RSS
    if HAS_FEEDPARSER:
        try:
            url = f"https://news.google.com/rss/search?q={ticker}+stock+when:7d&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_articles]:
                title = entry.get('title', '')
                # Google News appends " - Publisher" to titles
                parts = title.rsplit(' - ', 1)
                clean_title = parts[0] if len(parts) > 1 else title
                publisher = parts[1] if len(parts) > 1 else 'Google News'
                try:
                    ts = datetime(*entry.published_parsed[:6])
                except Exception:
                    ts = datetime.now()
                articles.append({
                    'title': clean_title, 'publisher': publisher,
                    'link': entry.get('link', ''), 'source': 'Google News',
                    'timestamp': ts, 'thumbnail': ''
                })
        except Exception as e:
            logger.debug(f"Google News error for {ticker}: {e}")

    # Source 3: Finviz headlines (scrape-light, public page)
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        resp = requests.get(f"https://finviz.com/quote.ashx?t={ticker}", headers=headers, timeout=5)
        if resp.status_code == 200:
            # Extract news table rows
            rows = re.findall(r'class="tab-link-news"[^>]*>(.*?)</a>', resp.text)
            dates = re.findall(r'<td[^>]*width="130"[^>]*>(.*?)</td>', resp.text)
            links = re.findall(r'class="tab-link-news"[^>]*href="(.*?)"', resp.text)
            for i, title in enumerate(rows[:15]):
                title = re.sub(r'<[^>]+>', '', title).strip()
                if not title: continue
                articles.append({
                    'title': title, 'publisher': 'Finviz',
                    'link': links[i] if i < len(links) else '',
                    'source': 'Finviz', 'timestamp': datetime.now(),
                    'thumbnail': ''
                })
    except Exception as e:
        logger.debug(f"Finviz error for {ticker}: {e}")

    # Deduplicate by title similarity
    seen = set(); deduped = []
    for a in articles:
        key = a['title'][:60].lower()
        if key not in seen:
            seen.add(key); deduped.append(a)

    # Score sentiment
    for a in deduped:
        a['sentiment'] = score_headline(a['title'])
        a['label'] = sentiment_label(a['sentiment'])

    # Sort by timestamp (most recent first)
    deduped.sort(key=lambda x: x['timestamp'], reverse=True)
    return deduped

@st.cache_data(ttl=900)
def fetch_market_news():
    """Fetch broad market news from RSS feeds."""
    articles = []
    feeds = {
        'MarketWatch': 'https://feeds.marketwatch.com/marketwatch/topstories/',
        'Reuters Business': 'https://feeds.reuters.com/reuters/businessNews',
        'CNBC': 'https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114',
    }
    if not HAS_FEEDPARSER: return articles
    for name, url in feeds.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:10]:
                try:
                    ts = datetime(*entry.published_parsed[:6])
                except Exception:
                    ts = datetime.now()
                title = entry.get('title', '')
                articles.append({
                    'title': title, 'publisher': name,
                    'link': entry.get('link', ''), 'source': name,
                    'timestamp': ts, 'sentiment': score_headline(title),
                    'label': sentiment_label(score_headline(title))
                })
        except Exception:
            continue
    articles.sort(key=lambda x: x['timestamp'], reverse=True)
    return articles

def aggregate_sentiment(articles):
    """Compute aggregate sentiment metrics from a list of scored articles."""
    if not articles: return {'mean': 0, 'median': 0, 'bullish_pct': 0, 'bearish_pct': 0, 'count': 0, 'label': 'No Data'}
    scores = [a['sentiment'] for a in articles]
    mean_s = np.mean(scores); med_s = np.median(scores)
    bull = sum(1 for s in scores if s >= 0.1) / len(scores)
    bear = sum(1 for s in scores if s <= -0.1) / len(scores)
    return {'mean': mean_s, 'median': med_s, 'bullish_pct': bull, 'bearish_pct': bear,
            'count': len(scores), 'label': sentiment_label(mean_s)}


# ==========================================
# Signal Aggregation (includes sentiment)
# ==========================================

@st.cache_data(ttl=1800)
def compute_all_signals(tickers_list, portfolio_val, rf):
    signals = {'macro': {}, 'vol': {}, 'portfolio': {}, 'ticker_signals': {}, 'news': {}}

    # Macro
    try:
        md = yf.download(["^TNX", "^IRX", "^VIX", "DX-Y.NYB", "HYG", "LQD"], period="60d", auto_adjust=True, progress=False)
        if not md.empty:
            cl = md['Close'] if isinstance(md.columns, pd.MultiIndex) else md[['Close']]
            if isinstance(cl.columns, pd.MultiIndex): cl = cl.droplevel(0, axis=1)
            if '^TNX' in cl.columns and '^IRX' in cl.columns:
                curve = cl['^TNX'].iloc[-1] - cl['^IRX'].iloc[-1]
                signals['macro']['yield_curve'] = curve
                signals['macro']['curve_regime'] = 'inverted' if curve < 0 else ('flat' if curve < 0.5 else 'normal')
            if '^VIX' in cl.columns:
                vix = cl['^VIX'].iloc[-1]; signals['macro']['vix'] = vix
                signals['macro']['vix_regime'] = 'crisis' if vix > 30 else ('elevated' if vix > 20 else 'low')
            if 'HYG' in cl.columns and 'LQD' in cl.columns:
                cm = cl['HYG'].pct_change().tail(20).mean() * 252 - cl['LQD'].pct_change().tail(20).mean() * 252
                signals['macro']['credit_momentum'] = cm
                signals['macro']['credit_regime'] = 'stress' if cm < -0.05 else ('neutral' if cm < 0.02 else 'risk_on')
        score = 0
        if signals['macro'].get('curve_regime') == 'inverted': score += 2
        elif signals['macro'].get('curve_regime') == 'flat': score += 1
        if signals['macro'].get('vix_regime') == 'crisis': score += 2
        elif signals['macro'].get('vix_regime') == 'elevated': score += 1
        if signals['macro'].get('credit_regime') == 'stress': score += 2
        elif signals['macro'].get('credit_regime') == 'neutral': score += 1
        signals['macro']['regime'] = 'risk_off' if score >= 4 else ('cautious' if score >= 2 else 'risk_on')
        signals['macro']['risk_score'] = score
    except Exception:
        signals['macro']['regime'] = 'unknown'

    # Market-wide news sentiment
    try:
        mkt_news = fetch_market_news()
        signals['news']['market'] = aggregate_sentiment(mkt_news)
    except Exception:
        signals['news']['market'] = {'mean': 0, 'label': 'No Data', 'count': 0}

    # Per ticker
    for tk_sym in tickers_list:
        try:
            tk_obj = yf.Ticker(tk_sym); tk_s = {}
            try: spot = tk_obj.info.get('currentPrice') or tk_obj.info.get('regularMarketPrice')
            except Exception: spot = None
            tk_s['spot'] = spot

            hp = yf.download(tk_sym, period="260d", auto_adjust=True, progress=False)
            if not hp.empty:
                cs = hp['Close'].squeeze() if isinstance(hp['Close'], pd.DataFrame) else hp['Close']
                lr = np.log(cs / cs.shift(1)).dropna()
                tk_s['rv_20'] = lr.tail(20).std() * np.sqrt(252)
                tk_s['rv_60'] = lr.tail(60).std() * np.sqrt(252)
                tk_s['rsi'] = compute_rsi(cs).iloc[-1]
                ma20 = cs.rolling(20).mean(); std20 = cs.rolling(20).std()
                tk_s['zscore'] = ((cs - ma20) / std20).iloc[-1]
                tk_s['momentum_20'] = cs.iloc[-1] / cs.iloc[-21] - 1 if len(cs) > 21 else 0

            try:
                exps = tk_obj.options
                if exps and spot:
                    chain = tk_obj.option_chain(exps[0]); c = chain.calls.copy()
                    c['dist'] = (c['strike'] - spot).abs()
                    atm_iv = c.nsmallest(3, 'dist')['impliedVolatility'].mean()
                    tk_s['atm_iv'] = atm_iv; tk_s['iv_rv_gap'] = atm_iv - tk_s.get('rv_20', atm_iv)
                    if not hp.empty:
                        rvs = lr.rolling(20).std().dropna() * np.sqrt(252)
                        if len(rvs) > 20:
                            tk_s['iv_rank'] = (atm_iv - rvs.min()) / (rvs.max() - rvs.min()) if rvs.max() != rvs.min() else 0.5
                            tk_s['iv_percentile'] = (rvs < atm_iv).mean()
                    pt, ct = spot * 0.95, spot * 1.05
                    p = chain.puts[chain.puts['impliedVolatility'] > 0].copy()
                    if len(p) > 0: p['d'] = (p['strike'] - pt).abs(); piv = p.nsmallest(1, 'd')['impliedVolatility'].values[0]
                    else: piv = atm_iv
                    cc = chain.calls[chain.calls['impliedVolatility'] > 0].copy()
                    if len(cc) > 0: cc['d'] = (cc['strike'] - ct).abs(); civ = cc.nsmallest(1, 'd')['impliedVolatility'].values[0]
                    else: civ = atm_iv
                    tk_s['skew'] = piv - civ; tk_s['nearest_exp'] = exps[0]
                    tk_s['call_chain'] = chain.calls[['strike', 'lastPrice', 'bid', 'ask', 'impliedVolatility', 'openInterest']].copy()
                    tk_s['put_chain'] = chain.puts[['strike', 'lastPrice', 'bid', 'ask', 'impliedVolatility', 'openInterest']].copy()
            except Exception: pass

            # News sentiment for this ticker
            try:
                tk_news = fetch_ticker_news(tk_sym)
                tk_s['news_sentiment'] = aggregate_sentiment(tk_news)
                tk_s['news_articles'] = tk_news
            except Exception:
                tk_s['news_sentiment'] = {'mean': 0, 'label': 'No Data', 'count': 0}

            signals['ticker_signals'][tk_sym] = tk_s
        except Exception as e:
            logger.error(f"Signal error {tk_sym}: {e}"); continue

    signals['portfolio']['value'] = portfolio_val; signals['portfolio']['rf'] = rf
    return signals


def generate_trade_ideas(signals, portfolio_df, rf):
    ideas = []
    regime = signals['macro'].get('regime', 'unknown')
    port_val = signals['portfolio'].get('value', 0)
    max_risk = port_val * 0.02
    mkt_sentiment = signals.get('news', {}).get('market', {}).get('mean', 0)

    for tk_sym, ts in signals['ticker_signals'].items():
        spot = ts.get('spot')
        if not spot: continue
        iv = ts.get('atm_iv', 0.25); iv_rank = ts.get('iv_rank', 0.5)
        rv = ts.get('rv_20', iv); gap = ts.get('iv_rv_gap', 0); skew = ts.get('skew', 0)
        rsi = ts.get('rsi', 50); zscore = ts.get('zscore', 0)
        exp = ts.get('nearest_exp', 'N/A')
        calls = ts.get('call_chain', pd.DataFrame()); puts = ts.get('put_chain', pd.DataFrame())
        news_sent = ts.get('news_sentiment', {}).get('mean', 0)
        news_label = ts.get('news_sentiment', {}).get('label', 'No Data')

        holding = portfolio_df[portfolio_df['Ticker'] == tk_sym]
        shares = holding['Shares'].values[0] if len(holding) > 0 else 0
        pos_val = spot * shares

        def ns(chain, target, d='above'):
            if chain.empty: return None, None, None
            sub = chain[chain['strike'] >= target] if d == 'above' else chain[chain['strike'] <= target]
            if sub.empty: return None, None, None
            row = sub.iloc[0] if d == 'above' else sub.iloc[-1]
            mid = (row['bid'] + row['ask']) / 2 if row['bid'] > 0 and row['ask'] > 0 else row['lastPrice']
            return row['strike'], mid, row['impliedVolatility']

        # 1. Protective Put — risk_off + concentrated + bearish news amplifies urgency
        if regime in ['risk_off', 'cautious'] and pos_val > port_val * 0.05 and not puts.empty:
            pk, pp, _ = ns(puts, spot * 0.95, 'below')
            if pk and pp and pp > 0:
                n_c = max(1, int(shares / 100)); cost = pp * 100 * n_c
                urg = 'High' if (regime == 'risk_off' or news_sent < -0.2) else 'Medium'
                ideas.append({'ticker': tk_sym, 'strategy': 'Protective Put', 'direction': 'Hedge', 'urgency': urg,
                    'legs': [{'type': 'Buy Put', 'strike': pk, 'price': pp, 'contracts': n_c}],
                    'expiry': exp, 'cost': cost, 'max_loss': cost, 'max_gain': (spot - pk) * 100 * n_c - cost,
                    'rationale': f"Regime: {regime}. {shares:.0f} shares (${pos_val:,.0f}, {pos_val/port_val:.1%}). "
                                 f"News sentiment: {news_label} ({news_sent:+.2f}). "
                                 f"5% OTM put costs ${cost:,.0f}.",
                    'risk_reward': ((spot - pk) * 100 * n_c - cost) / cost if cost > 0 else 0})

        # 2. Iron Condor — high IV rank + neutral news
        if iv_rank > 0.70 and abs(news_sent) < 0.3 and not calls.empty and not puts.empty:
            spk, spp, _ = ns(puts, spot * 0.95, 'below'); bpk, bpp, _ = ns(puts, spot * 0.90, 'below')
            sck, scp, _ = ns(calls, spot * 1.05, 'above'); bck, bcp, _ = ns(calls, spot * 1.10, 'above')
            if spk and bpk and sck and bck and spp and bpp and scp and bcp and spk != bpk and sck != bck:
                tc = (spp - bpp) + (scp - bcp); mr = max(spk - bpk, bck - sck) - tc
                n_c = max(1, int(max_risk / (mr * 100))) if mr > 0 else 1; tcd = tc * 100 * n_c
                ideas.append({'ticker': tk_sym, 'strategy': 'Iron Condor', 'direction': 'Neutral / Sell Vol', 'urgency': 'High' if iv_rank > 0.85 else 'Medium',
                    'legs': [{'type': 'Sell Put', 'strike': spk, 'price': spp, 'contracts': n_c},
                             {'type': 'Buy Put', 'strike': bpk, 'price': bpp, 'contracts': n_c},
                             {'type': 'Sell Call', 'strike': sck, 'price': scp, 'contracts': n_c},
                             {'type': 'Buy Call', 'strike': bck, 'price': bcp, 'contracts': n_c}],
                    'expiry': exp, 'cost': -tcd, 'max_loss': mr * 100 * n_c, 'max_gain': tcd,
                    'rationale': f"IV rank {iv_rank:.0%}. News: {news_label} (neutral enough for premium selling). "
                                 f"Collect ${tcd:,.0f}, max risk ${mr * 100 * n_c:,.0f}.",
                    'risk_reward': tcd / (mr * 100 * n_c) if mr > 0 else 0})

        # 3. Long Straddle — cheap vol + high news activity (catalyst expected)
        news_count = ts.get('news_sentiment', {}).get('count', 0)
        if iv_rank < 0.20 and news_count > 5 and not calls.empty and not puts.empty:
            ack, acp, _ = ns(calls, spot, 'above'); apk, app, _ = ns(puts, spot, 'below')
            if ack and apk and acp and app:
                sc = acp + app; n_c = max(1, int(max_risk / (sc * 100)))
                ideas.append({'ticker': tk_sym, 'strategy': 'Long Straddle', 'direction': 'Buy Vol', 'urgency': 'Medium',
                    'legs': [{'type': 'Buy Call', 'strike': ack, 'price': acp, 'contracts': n_c},
                             {'type': 'Buy Put', 'strike': apk, 'price': app, 'contracts': n_c}],
                    'expiry': exp, 'cost': sc * 100 * n_c, 'max_loss': sc * 100 * n_c, 'max_gain': float('inf'),
                    'rationale': f"IV rank {iv_rank:.0%} (cheap vol) with {news_count} recent articles — potential catalyst. "
                                 f"News tone: {news_label}. Straddle costs ${sc * 100 * n_c:,.0f}, needs {sc/spot:.1%} move.",
                    'risk_reward': 0})

        # 4. Bull Call Spread — oversold + bullish news confirmation
        if rsi < 30 and zscore < -1.5 and news_sent > -0.1 and not calls.empty:
            bk, bp, _ = ns(calls, spot, 'above'); sk, sp2, _ = ns(calls, spot * 1.05, 'above')
            if bk and sk and bp and sp2 and bk < sk:
                db = bp - sp2; mp = (sk - bk) - db
                n_c = max(1, int(max_risk / (db * 100))) if db > 0 else 1
                ideas.append({'ticker': tk_sym, 'strategy': 'Bull Call Spread', 'direction': 'Bullish', 'urgency': 'High' if news_sent > 0.2 else 'Medium',
                    'legs': [{'type': 'Buy Call', 'strike': bk, 'price': bp, 'contracts': n_c},
                             {'type': 'Sell Call', 'strike': sk, 'price': sp2, 'contracts': n_c}],
                    'expiry': exp, 'cost': db * 100 * n_c, 'max_loss': db * 100 * n_c, 'max_gain': mp * 100 * n_c,
                    'rationale': f"RSI {rsi:.0f}, z-score {zscore:.1f} (oversold). News: {news_label} ({news_sent:+.2f}) — "
                                 f"{'confirms' if news_sent > 0 else 'not contradicting'} mean reversion thesis. "
                                 f"${bk:.0f}/{sk:.0f} spread, ${db * 100 * n_c:,.0f} debit.",
                    'risk_reward': mp / db if db > 0 else 0})

        # 5. Bear Put Spread — overbought + bearish news confirmation
        elif rsi > 70 and zscore > 1.5 and news_sent < 0.1 and not puts.empty:
            bk, bp, _ = ns(puts, spot, 'below'); sk, sp2, _ = ns(puts, spot * 0.95, 'below')
            if bk and sk and bp and sp2 and bk > sk:
                db = bp - sp2; mp = (bk - sk) - db
                n_c = max(1, int(max_risk / (db * 100))) if db > 0 else 1
                ideas.append({'ticker': tk_sym, 'strategy': 'Bear Put Spread', 'direction': 'Bearish', 'urgency': 'High' if news_sent < -0.2 else 'Medium',
                    'legs': [{'type': 'Buy Put', 'strike': bk, 'price': bp, 'contracts': n_c},
                             {'type': 'Sell Put', 'strike': sk, 'price': sp2, 'contracts': n_c}],
                    'expiry': exp, 'cost': db * 100 * n_c, 'max_loss': db * 100 * n_c, 'max_gain': mp * 100 * n_c,
                    'rationale': f"RSI {rsi:.0f}, z-score {zscore:.1f} (overbought). News: {news_label} ({news_sent:+.2f}). "
                                 f"${bk:.0f}/{sk:.0f} put spread, ${db * 100 * n_c:,.0f} debit.",
                    'risk_reward': mp / db if db > 0 else 0})

        # 6. Collar — concentrated position
        if pos_val > port_val * 0.15 and shares >= 100 and not calls.empty and not puts.empty:
            sck, scp, _ = ns(calls, spot * 1.05, 'above'); bpk, bpp, _ = ns(puts, spot * 0.95, 'below')
            if sck and bpk and scp and bpp:
                n_c = int(shares / 100); net = (bpp - scp) * 100 * n_c
                ideas.append({'ticker': tk_sym, 'strategy': 'Collar', 'direction': 'Hedge (Neutral)', 'urgency': 'High' if regime == 'risk_off' or news_sent < -0.3 else 'Medium',
                    'legs': [{'type': 'Sell Call', 'strike': sck, 'price': scp, 'contracts': n_c},
                             {'type': 'Buy Put', 'strike': bpk, 'price': bpp, 'contracts': n_c}],
                    'expiry': exp, 'cost': net, 'max_loss': (spot - bpk) * 100 * n_c + net,
                    'max_gain': (sck - spot) * 100 * n_c - net,
                    'rationale': f"{tk_sym} is {pos_val/port_val:.1%} of portfolio. News: {news_label}. "
                                 f"Collar caps at ${sck:.0f}, floors at ${bpk:.0f}. Net {'cost' if net > 0 else 'credit'}: ${abs(net):,.0f}.",
                    'risk_reward': ((sck - spot) * 100 * n_c - net) / ((spot - bpk) * 100 * n_c + net) if ((spot - bpk) * 100 * n_c + net) > 0 else 0})

        # 7. Vol sell — IV >> RV
        if gap > 0.08 and abs(news_sent) < 0.25 and not calls.empty:
            ak, ap, _ = ns(calls, spot, 'above')
            if ak and ap:
                n_c = max(1, int(max_risk / (ap * 100)))
                covered = "Covered by your shares." if shares >= n_c * 100 else "⚠️ NAKED — need shares or long call."
                ideas.append({'ticker': tk_sym, 'strategy': 'Covered Call / Vol Sell', 'direction': 'Sell Vol', 'urgency': 'Medium',
                    'legs': [{'type': 'Sell Call', 'strike': ak, 'price': ap, 'contracts': n_c}],
                    'expiry': exp, 'cost': -ap * 100 * n_c, 'max_loss': float('inf'), 'max_gain': ap * 100 * n_c,
                    'rationale': f"IV ({iv:.1%}) > RV ({rv:.1%}) by {gap:.1%}. News: {news_label} (calm enough to sell). "
                                 f"${ap * 100 * n_c:,.0f} credit. {covered}",
                    'risk_reward': 0})

        # 8. NEW: News-Driven Momentum — strong bullish/bearish news without technical confirmation yet
        if abs(news_sent) > 0.4 and 35 < rsi < 65 and abs(zscore) < 1 and not calls.empty and not puts.empty:
            if news_sent > 0.4:
                bk, bp, _ = ns(calls, spot * 1.02, 'above'); sk, sp2, _ = ns(calls, spot * 1.07, 'above')
                if bk and sk and bp and sp2 and bk < sk:
                    db = bp - sp2; mp = (sk - bk) - db
                    n_c = max(1, int(max_risk / (db * 100))) if db > 0 else 1
                    ideas.append({'ticker': tk_sym, 'strategy': 'News Momentum (Bull Spread)', 'direction': 'Bullish (News)', 'urgency': 'Medium',
                        'legs': [{'type': 'Buy Call', 'strike': bk, 'price': bp, 'contracts': n_c},
                                 {'type': 'Sell Call', 'strike': sk, 'price': sp2, 'contracts': n_c}],
                        'expiry': exp, 'cost': db * 100 * n_c, 'max_loss': db * 100 * n_c, 'max_gain': mp * 100 * n_c,
                        'rationale': f"Strong bullish news ({news_sent:+.2f}) before technical signals catch up. "
                                     f"RSI {rsi:.0f}, z-score {zscore:.1f} still neutral — potential momentum setup. "
                                     f"{news_count} recent articles. ${bk:.0f}/{sk:.0f} call spread.",
                        'risk_reward': mp / db if db > 0 else 0})
            elif news_sent < -0.4:
                bk, bp, _ = ns(puts, spot * 0.98, 'below'); sk, sp2, _ = ns(puts, spot * 0.93, 'below')
                if bk and sk and bp and sp2 and bk > sk:
                    db = bp - sp2; mp = (bk - sk) - db
                    n_c = max(1, int(max_risk / (db * 100))) if db > 0 else 1
                    ideas.append({'ticker': tk_sym, 'strategy': 'News Momentum (Bear Spread)', 'direction': 'Bearish (News)', 'urgency': 'Medium',
                        'legs': [{'type': 'Buy Put', 'strike': bk, 'price': bp, 'contracts': n_c},
                                 {'type': 'Sell Put', 'strike': sk, 'price': sp2, 'contracts': n_c}],
                        'expiry': exp, 'cost': db * 100 * n_c, 'max_loss': db * 100 * n_c, 'max_gain': mp * 100 * n_c,
                        'rationale': f"Strong bearish news ({news_sent:+.2f}) before technicals confirm. "
                                     f"RSI {rsi:.0f} still neutral. {news_count} articles. ${bk:.0f}/{sk:.0f} put spread.",
                        'risk_reward': mp / db if db > 0 else 0})

    urgency_order = {'High': 0, 'Medium': 1, 'Low': 2}
    ideas.sort(key=lambda x: (urgency_order.get(x['urgency'], 3), -x.get('risk_reward', 0)))
    return ideas


# ==========================================
# Sidebar
# ==========================================

def render_sidebar():
    st.sidebar.title("🧭 Navigation")
    pages = ["Portfolio Overview", "Volatility & Options Lab", "Factor Exposure & Attribution",
             "Screener & Signal Scanner", "Trade Journal & Post-Mortem", "Macro Regime Dashboard",
             "Strategy Engine", "News & Sentiment"]
    page = st.sidebar.radio("Go to", pages)
    st.sidebar.markdown("---")
    st.sidebar.header("🔧 Portfolio Holdings")
    raw_start = datetime.now().date() - timedelta(days=365); valid_start = get_valid_start_date(raw_start)
    default_portfolio = pd.DataFrame({"Ticker": ["AAPL", "MSFT", "GOOG"], "Shares": [10.0, 15.0, 20.0], "Start Date": [valid_start] * 3})
    uploaded = st.sidebar.file_uploader("Upload Holdings CSV", type=["csv"])
    if uploaded:
        try:
            csv_df = pd.read_csv(uploaded); csv_df.columns = csv_df.columns.str.strip()
            cm = {}
            for col in csv_df.columns:
                cl = col.lower()
                if 'tick' in cl or 'symbol' in cl: cm[col] = 'Ticker'
                elif 'share' in cl or 'qty' in cl: cm[col] = 'Shares'
                elif 'date' in cl or 'start' in cl: cm[col] = 'Start Date'
            csv_df = csv_df.rename(columns=cm)
            if {'Ticker', 'Shares', 'Start Date'}.issubset(csv_df.columns):
                csv_df['Shares'] = pd.to_numeric(csv_df['Shares'], errors='coerce')
                csv_df['Start Date'] = pd.to_datetime(csv_df['Start Date'], errors='coerce').dt.date
                csv_df = csv_df.dropna(subset=['Ticker', 'Shares', 'Start Date'])
                initial = csv_df[['Ticker', 'Shares', 'Start Date']] if not csv_df.empty else default_portfolio
            else: initial = default_portfolio
        except Exception: initial = default_portfolio
    else: initial = default_portfolio
    st.sidebar.caption("Edit positions:")
    portfolio_df = st.sidebar.data_editor(initial, num_rows="dynamic", hide_index=True, column_config={
        "Start Date": st.column_config.DateColumn("Start Date", required=True),
        "Shares": st.column_config.NumberColumn("Shares", min_value=0.0001, required=True),
        "Ticker": st.column_config.TextColumn("Ticker", required=True)})
    st.sidebar.download_button("💾 Export CSV", portfolio_df.to_csv(index=False).encode('utf-8'), "portfolio.csv", "text/csv")
    st.sidebar.markdown("---"); st.sidebar.header("⚙️ Settings")
    end_date = st.sidebar.date_input("End Date", value=datetime.now())
    benchmarks = st.sidebar.multiselect("Benchmarks", ["SPY", "QQQ", "DIA", "IWM", "BTC-USD"], default=["SPY"])
    use_pct = st.sidebar.toggle("Show as % Return", value=True)
    live_rf = get_risk_free_rate()
    if live_rf:
        st.sidebar.markdown(f"**10Y Yield:** {live_rf:.2%} *(live)*")
        rf = st.sidebar.number_input("RF (%)", value=live_rf * 100, step=0.1) / 100 if st.sidebar.toggle("Override RF", value=False) else live_rf
    else: rf = st.sidebar.number_input("Risk-Free Rate (%)", value=4.5, step=0.1) / 100
    portfolio_df = portfolio_df.dropna(subset=['Ticker', 'Shares', 'Start Date'])
    portfolio_df['Ticker'] = portfolio_df['Ticker'].astype(str).str.upper().str.strip()
    portfolio_df = portfolio_df[portfolio_df['Ticker'] != ""]
    return page, portfolio_df, end_date, benchmarks, use_pct, rf


def build_portfolio_returns(portfolio_df, end_date):
    tickers = portfolio_df['Ticker'].tolist()
    if not tickers: return None, None, None, None, None
    df_prices = get_data(tickers, portfolio_df['Start Date'].min(), end_date)
    if df_prices.empty: return None, None, None, None, None
    if len(tickers) == 1: df_prices = df_prices.to_frame(name=tickers[0])
    asset_ret = df_prices.pct_change().dropna()
    dv = pd.DataFrame(index=df_prices.index, columns=tickers)
    for _, r in portfolio_df.iterrows():
        t, s, sd = r['Ticker'], r['Shares'], pd.to_datetime(r['Start Date']).date()
        dv[t] = df_prices[t] * s; dv.loc[dv.index.date < sd, t] = 0.0
    prev = dv.shift(1); tot = prev.sum(axis=1); wts = prev.div(tot, axis=0).fillna(0)
    ci = asset_ret.index.intersection(wts.index)
    port_ret = (asset_ret.loc[ci] * wts.loc[ci]).sum(axis=1); port_ret = port_ret[tot.loc[ci] > 0]
    cur_dv = dv.iloc[-1]; cur_wts = (cur_dv / cur_dv.sum()).values
    return port_ret, asset_ret, cur_wts, cur_dv, df_prices


# ==========================================
# Page 1: Portfolio Overview (Full)
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
    daily_values = pd.DataFrame(index=df_prices.index, columns=tickers_list)
    for _, row in portfolio_df.iterrows():
        t, s, sd = row['Ticker'], row['Shares'], pd.to_datetime(row['Start Date']).date()
        daily_values[t] = df_prices[t] * s
        daily_values.loc[daily_values.index.date < sd, t] = 0.0

    # --- METRICS ---
    st.subheader("Portfolio Performance Metrics")

    m = calculate_metrics(port_ret, rf)
    cum_ret_series = (1 + port_ret).cumprod()
    period_label = f"{m['n_days']} trading days ({m['years_held']:.1f} yrs)"
    st.caption(f"Metrics computed over **{period_label}** of live portfolio data.")

    row1 = st.columns(5)
    row1[0].metric("Portfolio Value", f"${cur_dv.sum():,.2f}",
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
        # --- Dollar Value History ---
        total_value_history = daily_values.sum(axis=1)
        total_value_history = total_value_history[total_value_history > 0]

        fig_val = go.Figure()
        fig_val.add_trace(go.Scatter(
            x=total_value_history.index, y=total_value_history,
            fill='tozeroy', name='Total Portfolio Value',
            line=dict(color='#636EFA', width=2)
        ))
        fig_val.update_layout(
            title="Total Portfolio Value Over Time ($)",
            yaxis_tickformat="$,.2f", template="plotly_white"
        )
        st.plotly_chart(fig_val, use_container_width=True)

        # --- Relative Performance ---
        y_label = "Percentage Return" if use_pct else "Value ($100 Invested)"
        portfolio_plot = (1 + port_ret).cumprod() - 1 if use_pct else 100 * (1 + port_ret).cumprod()

        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(
            x=portfolio_plot.index, y=portfolio_plot,
            name='My Portfolio', line=dict(width=3, color='#00CC96')
        ))

        if not df_benchmark.empty:
            first_active_date = port_ret.index[0]

            if isinstance(df_benchmark, pd.Series):
                df_bench_clean = df_benchmark.to_frame(name=benchmark_tickers[0])
            else:
                df_bench_clean = df_benchmark

            for ticker in benchmark_tickers:
                try:
                    b_prices = df_bench_clean[ticker].loc[first_active_date:]
                    b_returns = b_prices.pct_change().fillna(0)

                    if use_pct:
                        b_plot = (1 + b_returns).cumprod() - 1
                    else:
                        b_plot = 100 * (1 + b_returns).cumprod()

                    fig_cum.add_trace(go.Scatter(
                        x=b_plot.index, y=b_plot,
                        name=f"Benchmark: {ticker}",
                        line=dict(dash='dash', width=1.5)
                    ))
                except KeyError:
                    continue

        fig_cum.update_layout(
            title=f"Relative Performance ({y_label})",
            yaxis_tickformat=".2%" if use_pct else "$.2f",
            template="plotly_white"
        )
        st.plotly_chart(fig_cum, use_container_width=True)

    with tab2:
        fig_pie = px.pie(
            names=cur_dv.index, values=cur_dv.values,
            title="Current Allocation", hole=0.4
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with tab3:
        fig_corr = px.imshow(
            asset_ret.corr(), text_auto=True,
            title="Correlation Matrix", color_continuous_scale="RdBu_r"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    with tab4:
        st.subheader("Deep Risk Analytics")

        # 1. UNDERWATER DRAWDOWN PLOT
        total_value_history = daily_values.sum(axis=1)
        total_value_history = total_value_history[total_value_history > 0]
        running_max = total_value_history.cummax()
        drawdown_series = (total_value_history / running_max) - 1

        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=drawdown_series.index, y=drawdown_series,
            fill='tozeroy', name='Drawdown',
            line=dict(color='red', width=1)
        ))
        fig_dd.update_layout(
            title="Underwater Drawdown Plot (Peak-to-Trough)",
            yaxis_title="Decline from Peak (%)",
            yaxis_tickformat=".2%",
            template="plotly_white"
        )
        st.plotly_chart(fig_dd, use_container_width=True)

        col_risk1, col_risk2 = st.columns(2)

        with col_risk1:
            # 2. BETA ANALYSIS
            market_ticker = "SPY"
            has_spy = market_ticker in df_benchmark.columns if not df_benchmark.empty else False
            if has_spy:
                market_returns = df_benchmark[market_ticker].loc[port_ret.index].pct_change().dropna()
                common_dates = port_ret.index.intersection(market_returns.index)
                p_ret_aligned = port_ret.loc[common_dates]
                m_ret_aligned = market_returns.loc[common_dates]

                covariance = np.cov(p_ret_aligned, m_ret_aligned)[0][1]
                market_variance = np.var(m_ret_aligned)
                beta = covariance / market_variance

                st.metric("Portfolio Beta (vs SPY)", f"{beta:.2f}",
                           help="Measures portfolio sensitivity to market moves. Beta > 1 means more volatile than the market.")
                st.caption(f"A Beta of {beta:.2f} means your portfolio is historically "
                           f"{abs(beta - 1):.0%} {'more' if beta > 1 else 'less'} volatile than the S&P 500.")
            else:
                st.warning("Select 'SPY' in benchmarks to calculate Beta Analysis.")

        with col_risk2:
            # 3. ROLLING SHARPE RATIO (auto-sized window)
            n_obs = len(port_ret)
            window = min(60, max(10, n_obs // 5))

            rolling_mean = port_ret.rolling(window=window).mean()
            rolling_std = port_ret.rolling(window=window).std()
            rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)

            fig_sharpe = go.Figure()
            fig_sharpe.add_trace(go.Scatter(
                x=rolling_sharpe.index, y=rolling_sharpe,
                name=f'{window}D Rolling Sharpe',
                line=dict(color='orange')
            ))
            fig_sharpe.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_sharpe.update_layout(
                title=f"Rolling {window}-Day Sharpe Ratio (window auto-sized to data)",
                yaxis_title="Sharpe Ratio",
                template="plotly_white"
            )
            st.plotly_chart(fig_sharpe, use_container_width=True)

    # --- MONTE CARLO ---
    st.markdown("---")
    st.subheader("Monte Carlo Simulation")
    with st.expander("Run Simulation Scenario"):
        col_sim1, col_sim2 = st.columns(2)
        n_sims = col_sim1.slider("Number of Simulations", 200, 2000, 500)
        time_horizon = col_sim2.slider("Time Horizon (Days)", 30, 365, 252)

        if st.button("Run Simulation"):
            log_returns = np.log(1 + asset_ret)
            daily_log_returns_sim = np.random.multivariate_normal(
                log_returns.mean().values,
                log_returns.cov().values,
                size=(time_horizon, n_sims)
            )
            portfolio_sim_returns = np.dot(np.exp(daily_log_returns_sim) - 1, cur_wts)

            start_val = cur_dv.sum()
            portfolio_sim_paths = start_val * np.cumprod(1 + portfolio_sim_returns, axis=0)

            fig_mc = go.Figure()
            for i in range(min(n_sims, 50)):
                fig_mc.add_trace(go.Scatter(
                    x=list(range(1, time_horizon + 1)),
                    y=portfolio_sim_paths[:, i],
                    mode='lines',
                    line=dict(width=1, color='rgba(100, 100, 255, 0.1)'),
                    showlegend=False
                ))

            fig_mc.add_trace(go.Scatter(
                x=list(range(1, time_horizon + 1)),
                y=np.mean(portfolio_sim_paths, axis=1),
                name='Mean Outcome',
                line=dict(width=3, color='orange')
            ))

            # Add percentile bands
            p5 = np.percentile(portfolio_sim_paths, 5, axis=1)
            p25 = np.percentile(portfolio_sim_paths, 25, axis=1)
            p75 = np.percentile(portfolio_sim_paths, 75, axis=1)
            p95 = np.percentile(portfolio_sim_paths, 95, axis=1)

            fig_mc.add_trace(go.Scatter(
                x=list(range(1, time_horizon + 1)), y=p95,
                name='95th Percentile', line=dict(width=1, dash='dot', color='green')
            ))
            fig_mc.add_trace(go.Scatter(
                x=list(range(1, time_horizon + 1)), y=p75,
                name='75th Percentile', line=dict(width=1, dash='dot', color='lightgreen')
            ))
            fig_mc.add_trace(go.Scatter(
                x=list(range(1, time_horizon + 1)), y=p25,
                name='25th Percentile', line=dict(width=1, dash='dot', color='lightsalmon')
            ))
            fig_mc.add_trace(go.Scatter(
                x=list(range(1, time_horizon + 1)), y=p5,
                name='5th Percentile', line=dict(width=1, dash='dot', color='red')
            ))

            fig_mc.update_layout(
                title="Monte Carlo Dollar Projection",
                yaxis_tickformat="$,.2f",
                xaxis_title="Trading Days",
                template="plotly_white"
            )
            st.plotly_chart(fig_mc, use_container_width=True)

            # Terminal value stats
            terminal_values = portfolio_sim_paths[-1, :]
            terminal_returns = (terminal_values / start_val) - 1
            var_95_percent = np.percentile(terminal_returns, 5)
            cvar_95 = terminal_returns[terminal_returns <= var_95_percent].mean()

            st.markdown("---")
            st.subheader("Simulation Results")

            sr1, sr2, sr3, sr4 = st.columns(4)
            sr1.metric("Starting Value", f"${start_val:,.2f}")
            sr2.metric("Mean Terminal Value", f"${np.mean(terminal_values):,.2f}",
                        help="Average ending portfolio value across all simulations.")
            sr3.metric("Median Terminal Value", f"${np.median(terminal_values):,.2f}",
                        help="50th percentile ending value — more robust than mean to outliers.")
            sr4.metric("Std Dev of Terminal", f"${np.std(terminal_values):,.2f}",
                        help="Dispersion of outcomes — higher = wider range of possible endings.")

            sr5, sr6, sr7, sr8 = st.columns(4)
            sr5.metric("95% VaR", f"{var_95_percent:.2%}",
                        help="95% Value at Risk — the loss that would only be exceeded 5% of the time.")
            sr6.metric("95% VaR ($)", f"${start_val * abs(var_95_percent):,.2f}")
            sr7.metric("95% CVaR", f"{cvar_95:.2%}",
                        help="Conditional VaR (Expected Shortfall) — the average loss in the worst 5% of scenarios. More conservative than VaR.")
            sr8.metric("Prob of Loss", f"{(terminal_returns < 0).mean():.1%}",
                        help="Percentage of simulations that ended with a loss.")

            # Terminal value distribution
            fig_term = go.Figure()
            fig_term.add_trace(go.Histogram(
                x=terminal_values, nbinsx=50,
                marker_color='#636EFA', opacity=0.7
            ))
            fig_term.add_vline(x=start_val, line_dash="dash", line_color="red",
                                annotation_text=f"Start: ${start_val:,.0f}")
            fig_term.add_vline(x=np.mean(terminal_values), line_dash="dash", line_color="orange",
                                annotation_text=f"Mean: ${np.mean(terminal_values):,.0f}")
            fig_term.update_layout(
                title=f"Terminal Value Distribution ({n_sims} simulations, {time_horizon} days)",
                xaxis_title="Portfolio Value ($)", yaxis_title="Frequency",
                xaxis_tickformat="$,.0f", template="plotly_white"
            )
            st.plotly_chart(fig_term, use_container_width=True)


# ==========================================
# Page 2: Volatility & Options Lab (Full)
# ==========================================

def show_volatility_page(portfolio_df, rf):
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

    try:
        spot = tk.info.get('currentPrice') or tk.info.get('regularMarketPrice') or tk.history(period="1d")['Close'].iloc[-1]
    except Exception:
        spot = None

    tabs = st.tabs(["Options Chain", "Vol Surface 3D", "Put-Call Skew", "Greeks Calculator",
                     "IV Rank & Regime", "Vol Arb Scanner", "Options Flow & OI"])

    # ================================================================
    # TAB 1: OPTIONS CHAIN
    # ================================================================
    with tabs[0]:
        selected_exp = st.selectbox("Expiration Date", exps, key="chain_exp")
        opt = tk.option_chain(selected_exp)

        st.subheader(f"Options Chain: {ticker} — {selected_exp}")

        tc, tp = st.columns(2)
        with tc:
            st.subheader("Calls")
            st.dataframe(opt.calls, use_container_width=True)
        with tp:
            st.subheader("Puts")
            st.dataframe(opt.puts, use_container_width=True)

        # IV Smiles
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Call IV Smile")
            calls = opt.calls[opt.calls['impliedVolatility'] > 0].copy()
            if not calls.empty:
                fig = px.line(calls, x='strike', y='impliedVolatility', title=f"Call IV Smile ({selected_exp})")
                fig.update_layout(yaxis_tickformat=".2%", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("No IV data available for calls.")

        with c2:
            st.subheader("Put IV Smile")
            puts = opt.puts[opt.puts['impliedVolatility'] > 0].copy()
            if not puts.empty:
                fig = px.line(puts, x='strike', y='impliedVolatility', title=f"Put IV Smile ({selected_exp})")
                fig.update_layout(yaxis_tickformat=".2%", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("No IV data available for puts.")

        # --- Volatility Term Structure (inside chain tab) ---
        st.markdown("---")
        st.subheader("Volatility Term Structure (ATM Approximation)")
        st.caption("ATM IV approximated as the mean IV of strikes nearest the last close price.")

        try:
            last_price = tk.info.get('currentPrice') or tk.info.get('regularMarketPrice')
        except Exception:
            last_price = None

        term_data = []
        if last_price:
            for exp in exps[:12]:
                try:
                    chain_td = tk.option_chain(exp)
                    c_td = chain_td.calls.copy()
                    c_td['dist'] = (c_td['strike'] - last_price).abs()
                    avg_iv = c_td.nsmallest(3, 'dist')['impliedVolatility'].mean()
                    dte = (pd.to_datetime(exp) - pd.Timestamp.now()).days
                    if avg_iv > 0:
                        term_data.append({"Expiration": exp, "DTE": dte, "ATM IV": avg_iv})
                except Exception:
                    continue

            if term_data:
                term_df = pd.DataFrame(term_data)
                fig_term = px.line(
                    term_df, x='DTE', y='ATM IV', text='Expiration',
                    title="IV Term Structure (Days to Expiry)"
                )
                fig_term.update_traces(textposition="top center")
                fig_term.update_layout(
                    yaxis_tickformat=".2%", template="plotly_white",
                    xaxis_title="Days to Expiration", yaxis_title="Implied Volatility"
                )
                st.plotly_chart(fig_term, use_container_width=True)
            else:
                st.caption("Could not build term structure.")
        else:
            st.caption("Could not determine current price for ATM approximation.")

        # --- Historical vs Implied Vol (inside chain tab) ---
        st.markdown("---")
        st.subheader("Historical Volatility vs Current ATM IV")

        hist_prices_chain = get_data([ticker], datetime.now().date() - timedelta(days=365), datetime.now().date())
        if not hist_prices_chain.empty:
            if isinstance(hist_prices_chain, pd.DataFrame) and ticker in hist_prices_chain.columns:
                hp_chain = hist_prices_chain[ticker]
            else:
                hp_chain = hist_prices_chain.squeeze()

            log_ret_chain = np.log(hp_chain / hp_chain.shift(1)).dropna()
            windows = [20, 40, 60, 90]
            fig_hv = go.Figure()
            for w in windows:
                rv = log_ret_chain.rolling(window=w).std() * np.sqrt(252)
                fig_hv.add_trace(go.Scatter(x=rv.index, y=rv, name=f'{w}D Realized Vol'))

            if term_data:
                nearest_iv = term_data[0]['ATM IV']
                fig_hv.add_hline(
                    y=nearest_iv, line_dash="dash", line_color="red",
                    annotation_text=f"Current ATM IV ({nearest_iv:.1%})"
                )

            fig_hv.update_layout(
                title=f"{ticker} — Realized Vol vs Implied Vol",
                yaxis_tickformat=".2%", template="plotly_white",
                yaxis_title="Annualized Volatility"
            )
            st.plotly_chart(fig_hv, use_container_width=True)

    # ================================================================
    # TAB 2: VOLATILITY SURFACE 3D
    # ================================================================
    with tabs[1]:
        st.subheader("Implied Volatility Surface")
        st.caption("3D surface of implied volatility across strikes and expirations.")
        n_exps = st.slider("Number of expirations to include", 3, min(len(exps), 15), min(8, len(exps)), key="surf_n")

        surface_data = []
        for exp in exps[:n_exps]:
            try:
                chain = tk.option_chain(exp)
                dte = max((pd.to_datetime(exp) - pd.Timestamp.now()).days, 1)
                for _, row in chain.calls.iterrows():
                    if row['impliedVolatility'] > 0.001:
                        moneyness = row['strike'] / spot if spot else row['strike']
                        if 0.7 < moneyness < 1.3:
                            surface_data.append({
                                'Strike': row['strike'], 'DTE': dte,
                                'IV': row['impliedVolatility'], 'Moneyness': moneyness
                            })
            except Exception:
                continue

        if surface_data:
            sdf = pd.DataFrame(surface_data)

            # Pivot for surface
            pivot = sdf.pivot_table(values='IV', index='Strike', columns='DTE', aggfunc='mean')
            pivot = pivot.interpolate(method='linear', axis=0).interpolate(method='linear', axis=1).dropna()

            if not pivot.empty:
                # 3D Surface
                fig_surf = go.Figure(data=[go.Surface(
                    z=pivot.values, x=pivot.columns.values, y=pivot.index.values,
                    colorscale='Viridis', colorbar_title='IV'
                )])
                fig_surf.update_layout(
                    title=f"{ticker} Implied Volatility Surface",
                    scene=dict(xaxis_title='DTE', yaxis_title='Strike', zaxis_title='IV'),
                    width=800, height=600
                )
                st.plotly_chart(fig_surf, use_container_width=True)

                # 2D Heatmap alternative
                fig_hm = px.imshow(
                    pivot.values,
                    x=[str(d) for d in pivot.columns],
                    y=[f"${s:.0f}" for s in pivot.index],
                    color_continuous_scale='Viridis', aspect='auto',
                    labels=dict(x="DTE", y="Strike", color="IV")
                )
                fig_hm.update_layout(title="IV Heatmap (Strike × DTE)", template="plotly_white")
                st.plotly_chart(fig_hm, use_container_width=True)
        else:
            st.warning("Could not build vol surface — insufficient data.")

    # ================================================================
    # TAB 3: PUT-CALL SKEW MONITOR
    # ================================================================
    with tabs[2]:
        st.subheader("Put-Call Skew Monitor")
        st.caption("Measures the difference between OTM put IV and OTM call IV. Rising skew = increasing demand for downside protection.")

        if not spot:
            st.warning("Could not determine spot price.")
        else:
            skew_pct = st.slider("OTM distance (%)", 2, 20, 5, key="skew_pct",
                                  help="How far OTM to measure. 5% means comparing 95% strike puts vs 105% strike calls.")
            put_target = spot * (1 - skew_pct / 100)
            call_target = spot * (1 + skew_pct / 100)

            skew_data = []
            for exp in exps[:12]:
                try:
                    chain = tk.option_chain(exp)
                    dte = max((pd.to_datetime(exp) - pd.Timestamp.now()).days, 1)

                    # Nearest OTM put
                    p = chain.puts[chain.puts['impliedVolatility'] > 0].copy()
                    p['dist'] = (p['strike'] - put_target).abs()
                    put_iv = p.nsmallest(1, 'dist')['impliedVolatility'].values[0] if len(p) > 0 else np.nan

                    # Nearest OTM call
                    c = chain.calls[chain.calls['impliedVolatility'] > 0].copy()
                    c['dist'] = (c['strike'] - call_target).abs()
                    call_iv = c.nsmallest(1, 'dist')['impliedVolatility'].values[0] if len(c) > 0 else np.nan

                    skew = put_iv - call_iv
                    skew_data.append({
                        'Expiration': exp, 'DTE': dte,
                        'Put IV': put_iv, 'Call IV': call_iv, 'Skew': skew
                    })
                except Exception:
                    continue

            if skew_data:
                skdf = pd.DataFrame(skew_data).dropna()

                c1, c2 = st.columns(2)
                with c1:
                    fig_sk = go.Figure()
                    fig_sk.add_trace(go.Bar(
                        x=skdf['Expiration'], y=skdf['Skew'],
                        marker_color=['red' if s > 0 else 'green' for s in skdf['Skew']]
                    ))
                    fig_sk.update_layout(
                        title=f"Put-Call Skew by Expiration ({skew_pct}% OTM)",
                        yaxis_tickformat=".2%", template="plotly_white",
                        yaxis_title="Put IV − Call IV"
                    )
                    st.plotly_chart(fig_sk, use_container_width=True)

                with c2:
                    fig_pc = go.Figure()
                    fig_pc.add_trace(go.Scatter(
                        x=skdf['DTE'], y=skdf['Put IV'],
                        name=f'{skew_pct}% OTM Put IV', line=dict(color='red')
                    ))
                    fig_pc.add_trace(go.Scatter(
                        x=skdf['DTE'], y=skdf['Call IV'],
                        name=f'{skew_pct}% OTM Call IV', line=dict(color='green')
                    ))
                    fig_pc.update_layout(
                        title="OTM Put vs Call IV Term Structure",
                        yaxis_tickformat=".2%", xaxis_title="DTE", template="plotly_white"
                    )
                    st.plotly_chart(fig_pc, use_container_width=True)

                # Data table
                st.dataframe(skdf.style.format({
                    "Put IV": "{:.2%}", "Call IV": "{:.2%}", "Skew": "{:.2%}"
                }), use_container_width=True)

    # ================================================================
    # TAB 4: GREEKS CALCULATOR & SCENARIO P&L
    # ================================================================
    with tabs[3]:
        st.subheader("Greeks Calculator & Scenario P&L")
        st.caption("Black-Scholes (European) vs Binomial Tree (American). Compare pricing and Greeks side by side.")

        gc1, gc2, gc3 = st.columns(3)
        g_spot = gc1.number_input("Spot Price", value=float(spot) if spot else 150.0, step=1.0, key="g_spot")
        g_strike = gc2.number_input("Strike Price", value=float(round(spot)) if spot else 150.0, step=1.0, key="g_strike")
        g_dte = gc3.number_input("Days to Expiry", value=30, min_value=1, step=1, key="g_dte")

        gc4, gc5, gc6 = st.columns(3)
        g_vol = gc4.number_input("Implied Vol (%)", value=25.0, step=0.5, key="g_vol") / 100
        g_rf = gc5.number_input("Risk-Free Rate (%)", value=rf * 100, step=0.1, key="g_rf") / 100
        g_type = gc6.selectbox("Option Type", ["Call", "Put"], key="g_type")
        g_contracts = st.number_input("Number of Contracts", value=1, min_value=1, step=1, key="g_contracts")

        T = g_dte / 365
        ot = g_type.lower()

        bs = bs_greeks(g_spot, g_strike, T, g_rf, g_vol, ot)
        bn = binomial_greeks(g_spot, g_strike, T, g_rf, g_vol, ot, steps=200, american=True)

        # Model Comparison Table
        st.subheader("Model Comparison")
        comp_df = pd.DataFrame({
            'Metric': ['Price', 'Delta', 'Gamma', 'Theta (daily)', 'Vega (per 1%)', 'Rho (per 1%)'],
            'Black-Scholes (Euro)': [bs['price'], bs['delta'], bs['gamma'], bs['theta'], bs['vega'], bs['rho']],
            'Binomial Tree (Amer)': [bn['price'], bn['delta'], bn['gamma'], bn['theta'], bn['vega'], bn['rho']]
        })
        comp_df['Difference'] = comp_df['Binomial Tree (Amer)'] - comp_df['Black-Scholes (Euro)']
        st.dataframe(comp_df.style.format({
            'Black-Scholes (Euro)': '{:.4f}',
            'Binomial Tree (Amer)': '{:.4f}',
            'Difference': '{:.4f}'
        }), use_container_width=True)

        # Position summary
        pos_val = bs['price'] * 100 * g_contracts
        st.markdown(f"**Position Value:** ${pos_val:,.2f} ({g_contracts} contract{'s' if g_contracts > 1 else ''} × 100 shares × ${bs['price']:.2f})")

        # Scenario P&L Matrix
        st.subheader("Scenario P&L Heatmap")
        st.caption("Shows P&L for a range of spot price and IV changes from the current position.")
        spot_range = np.linspace(g_spot * 0.85, g_spot * 1.15, 15)
        vol_range = np.linspace(max(g_vol * 0.5, 0.05), g_vol * 1.5, 11)
        entry_price = bs['price']

        pnl_matrix = np.zeros((len(vol_range), len(spot_range)))
        for i, v in enumerate(vol_range):
            for j, s in enumerate(spot_range):
                new_price = bs_price(s, g_strike, T, g_rf, v, ot)
                pnl_matrix[i, j] = (new_price - entry_price) * 100 * g_contracts

        fig_pnl = go.Figure(data=go.Heatmap(
            z=pnl_matrix,
            x=[f"${s:.0f}" for s in spot_range],
            y=[f"{v:.0%}" for v in vol_range],
            colorscale='RdYlGn', zmid=0, colorbar_title='P&L ($)'
        ))
        fig_pnl.update_layout(
            title="P&L Scenario Matrix (Spot × IV)",
            xaxis_title="Spot Price", yaxis_title="Implied Volatility",
            template="plotly_white"
        )
        st.plotly_chart(fig_pnl, use_container_width=True)

        # Greeks over spot range
        st.subheader("Greeks Across Spot Price")
        greek_names = ['delta', 'gamma', 'theta', 'vega']
        greeks_over_spot = {g: [] for g in greek_names}
        for s in spot_range:
            gr = bs_greeks(s, g_strike, T, g_rf, g_vol, ot)
            for g in greek_names:
                greeks_over_spot[g].append(gr[g])

        fig_gr = make_subplots(rows=2, cols=2, subplot_titles=[g.capitalize() for g in greek_names])
        for idx, g in enumerate(greek_names):
            r, c = idx // 2 + 1, idx % 2 + 1
            fig_gr.add_trace(go.Scatter(
                x=spot_range, y=greeks_over_spot[g],
                name=g.capitalize(), line=dict(width=2)
            ), row=r, col=c)
            fig_gr.add_vline(x=g_spot, line_dash="dash", line_color="gray", row=r, col=c)
        fig_gr.update_layout(template="plotly_white", height=500, showlegend=False, title_text="Greeks Sensitivity")
        st.plotly_chart(fig_gr, use_container_width=True)

    # ================================================================
    # TAB 5: IV PERCENTILE RANK & REGIME
    # ================================================================
    with tabs[4]:
        st.subheader("IV Percentile Rank & Regime")
        st.caption("Where does current IV sit relative to its trailing 1-year range?")

        hist_prices = get_data([ticker], datetime.now().date() - timedelta(days=400), datetime.now().date())
        if hist_prices.empty:
            st.warning("No historical data available.")
        else:
            hp = hist_prices[ticker] if isinstance(hist_prices, pd.DataFrame) and ticker in hist_prices.columns else hist_prices.squeeze()
            log_ret = np.log(hp / hp.shift(1)).dropna()

            # Trailing 20D realized vol
            rv_20 = log_ret.rolling(20).std() * np.sqrt(252)
            rv_20 = rv_20.dropna()

            # Get current near-term ATM IV
            current_iv = None
            try:
                chain = tk.option_chain(exps[0])
                if spot:
                    c = chain.calls.copy()
                    c['dist'] = (c['strike'] - spot).abs()
                    current_iv = c.nsmallest(3, 'dist')['impliedVolatility'].mean()
            except Exception:
                pass

            if current_iv and len(rv_20) > 20:
                # IV Rank = (Current - 52wk Low) / (52wk High - 52wk Low)
                iv_high = rv_20.max()
                iv_low = rv_20.min()
                iv_rank = (current_iv - iv_low) / (iv_high - iv_low) if iv_high != iv_low else 0.5
                iv_percentile = (rv_20 < current_iv).mean()

                c1, c2, c3 = st.columns(3)
                c1.metric("Current ATM IV", f"{current_iv:.1%}",
                           help="Near-term at-the-money implied vol.")
                c2.metric("IV Rank (1Y)", f"{iv_rank:.1%}",
                           help="(Current IV − 1Y Low) / (1Y High − 1Y Low). High rank = vol is expensive relative to recent range.")
                c3.metric("IV Percentile (1Y)", f"{iv_percentile:.1%}",
                           help="% of days in the past year where realized vol was below current IV.")

                # Regime classification
                if iv_percentile > 0.8:
                    regime = "Elevated 🔴 — Favor selling premium"
                elif iv_percentile > 0.5:
                    regime = "Normal 🟡 — Neutral strategies"
                elif iv_percentile > 0.2:
                    regime = "Low 🟢 — Consider buying premium"
                else:
                    regime = "Compressed ⚪ — Vol expansion likely"
                st.info(f"**Vol Regime:** {regime}")

                # Historical distribution chart
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=rv_20, nbinsx=50, name='20D Realized Vol',
                    marker_color='#636EFA', opacity=0.7
                ))
                fig_hist.add_vline(
                    x=current_iv, line_color="red", line_width=3,
                    annotation_text=f"Current IV ({current_iv:.1%})"
                )
                fig_hist.update_layout(
                    title="1-Year Volatility Distribution vs Current IV",
                    xaxis_title="Annualized Volatility", yaxis_title="Frequency",
                    template="plotly_white"
                )
                st.plotly_chart(fig_hist, use_container_width=True)

                # Time series with rank zones
                fig_ts = go.Figure()
                fig_ts.add_trace(go.Scatter(
                    x=rv_20.index, y=rv_20,
                    name='20D Realized Vol', line=dict(color='#636EFA')
                ))
                fig_ts.add_hline(y=current_iv, line_dash="dash", line_color="red",
                                  annotation_text="Current IV")
                fig_ts.add_hline(y=rv_20.quantile(0.8), line_dash="dot", line_color="orange",
                                  annotation_text="80th pctl")
                fig_ts.add_hline(y=rv_20.quantile(0.2), line_dash="dot", line_color="green",
                                  annotation_text="20th pctl")
                fig_ts.update_layout(
                    title="Realized Vol Time Series with Current IV",
                    yaxis_tickformat=".2%", template="plotly_white"
                )
                st.plotly_chart(fig_ts, use_container_width=True)
            else:
                st.warning("Could not compute IV rank — insufficient data.")

    # ================================================================
    # TAB 6: VOL ARBITRAGE SCANNER
    # ================================================================
    with tabs[5]:
        st.subheader("Volatility Arbitrage Scanner")
        st.caption("Compares realized volatility (backward-looking) against implied volatility (forward-looking). Large IV−RV gaps = potential opportunity.")

        scan_universe = st.text_input(
            "Scan Universe (comma-separated)",
            ", ".join(portfolio_df['Ticker'].tolist()), key="vol_arb_univ"
        )
        scan_tickers = [t.strip().upper() for t in scan_universe.split(",") if t.strip()]

        if st.button("Scan Vol Arb", key="vol_arb_btn") and scan_tickers:
            results = []
            progress = st.progress(0)
            for idx, stk in enumerate(scan_tickers):
                progress.progress((idx + 1) / len(scan_tickers))
                try:
                    stk_tk = yf.Ticker(stk)
                    stk_exps = stk_tk.options
                    if not stk_exps:
                        continue

                    # Get spot price
                    try:
                        stk_spot = stk_tk.info.get('currentPrice') or stk_tk.info.get('regularMarketPrice')
                    except Exception:
                        stk_spot = None
                    if not stk_spot:
                        continue

                    # ATM IV from nearest expiry
                    chain = stk_tk.option_chain(stk_exps[0])
                    c = chain.calls.copy()
                    c['dist'] = (c['strike'] - stk_spot).abs()
                    atm_iv = c.nsmallest(3, 'dist')['impliedVolatility'].mean()

                    # Realized vol (20D)
                    hp = yf.download(stk, period="60d", auto_adjust=True, progress=False)
                    if hp.empty:
                        continue
                    close = hp['Close'].squeeze() if isinstance(hp['Close'], pd.DataFrame) else hp['Close']
                    lr = np.log(close / close.shift(1)).dropna()
                    rv_20 = lr.tail(20).std() * np.sqrt(252)

                    gap = atm_iv - rv_20
                    results.append({
                        "Ticker": stk, "Spot": stk_spot,
                        "ATM IV": atm_iv, "20D RV": rv_20,
                        "IV−RV Gap": gap,
                        "IV/RV Ratio": atm_iv / rv_20 if rv_20 > 0 else np.nan,
                        "Signal": "Sell Vol 🔴" if gap > 0.05 else ("Buy Vol 🟢" if gap < -0.05 else "Neutral ⚪")
                    })
                except Exception:
                    continue
            progress.empty()

            if results:
                rdf = pd.DataFrame(results).sort_values("IV−RV Gap", ascending=False)
                st.dataframe(rdf.style.format({
                    "Spot": "${:.2f}", "ATM IV": "{:.2%}", "20D RV": "{:.2%}",
                    "IV−RV Gap": "{:.2%}", "IV/RV Ratio": "{:.2f}"
                }), use_container_width=True)

                fig_gap = go.Figure()
                colors = ['red' if g > 0 else 'green' for g in rdf['IV−RV Gap']]
                fig_gap.add_trace(go.Bar(x=rdf['Ticker'], y=rdf['IV−RV Gap'], marker_color=colors))
                fig_gap.add_hline(y=0, line_dash="dash")
                fig_gap.update_layout(
                    title="IV − RV Gap by Ticker",
                    yaxis_tickformat=".2%", template="plotly_white",
                    yaxis_title="IV minus Realized Vol"
                )
                st.plotly_chart(fig_gap, use_container_width=True)
            else:
                st.warning("No results. Check tickers.")

    # ================================================================
    # TAB 7: OPTIONS FLOW & OPEN INTEREST ANALYSIS
    # ================================================================
    with tabs[6]:
        st.subheader("Options Flow & Open Interest Analysis")

        oi_exp = st.selectbox("Expiration", exps, key="oi_exp")
        opt = tk.option_chain(oi_exp)

        # --- OI Distribution by Strike ---
        st.subheader("Open Interest by Strike")
        oi_calls = opt.calls[['strike', 'openInterest', 'volume']].copy()
        oi_calls.columns = ['Strike', 'Call OI', 'Call Volume']
        oi_puts = opt.puts[['strike', 'openInterest', 'volume']].copy()
        oi_puts.columns = ['Strike', 'Put OI', 'Put Volume']
        oi_merged = pd.merge(oi_calls, oi_puts, on='Strike', how='outer').fillna(0)

        fig_oi = go.Figure()
        fig_oi.add_trace(go.Bar(
            x=oi_merged['Strike'], y=oi_merged['Call OI'],
            name='Call OI', marker_color='green', opacity=0.7
        ))
        fig_oi.add_trace(go.Bar(
            x=oi_merged['Strike'], y=-oi_merged['Put OI'],
            name='Put OI', marker_color='red', opacity=0.7
        ))
        if spot:
            fig_oi.add_vline(x=spot, line_dash="dash", annotation_text=f"Spot ${spot:.0f}")
        fig_oi.update_layout(
            title=f"Open Interest Distribution ({oi_exp})",
            barmode='relative',
            yaxis_title="Open Interest (Puts shown negative)",
            template="plotly_white"
        )
        st.plotly_chart(fig_oi, use_container_width=True)

        # --- Unusual Activity (Volume / OI Ratio) ---
        st.subheader("Unusual Activity (Volume / OI Ratio)")
        st.caption("High Vol/OI ratio indicates new positioning. Ratio > 1 means today's volume exceeds all existing open interest.")

        all_opts = pd.concat([
            opt.calls[['strike', 'volume', 'openInterest', 'impliedVolatility']].assign(Type='Call'),
            opt.puts[['strike', 'volume', 'openInterest', 'impliedVolatility']].assign(Type='Put')
        ])
        all_opts['Vol/OI'] = np.where(
            all_opts['openInterest'] > 0,
            all_opts['volume'] / all_opts['openInterest'], 0
        )
        unusual = all_opts[all_opts['Vol/OI'] > 0.5].sort_values('Vol/OI', ascending=False).head(20)

        if not unusual.empty:
            st.dataframe(unusual.style.format({
                "strike": "${:.0f}", "volume": "{:,.0f}", "openInterest": "{:,.0f}",
                "impliedVolatility": "{:.2%}", "Vol/OI": "{:.2f}"
            }), use_container_width=True)

            fig_unusual = go.Figure()
            for otype, color in [('Call', 'green'), ('Put', 'red')]:
                sub = unusual[unusual['Type'] == otype]
                if not sub.empty:
                    fig_unusual.add_trace(go.Bar(
                        x=[f"${s:.0f} {otype}" for s in sub['strike']],
                        y=sub['Vol/OI'], name=otype, marker_color=color, opacity=0.8
                    ))
            fig_unusual.add_hline(y=1, line_dash="dash", line_color="orange", annotation_text="Vol = OI")
            fig_unusual.update_layout(
                title="Top Unusual Activity Strikes",
                yaxis_title="Volume / Open Interest",
                template="plotly_white"
            )
            st.plotly_chart(fig_unusual, use_container_width=True)
        else:
            st.caption("No significant unusual activity detected.")

        # --- Put/Call OI Ratio by Expiration ---
        st.subheader("Put/Call OI Ratio by Expiration")
        pcr_data = []
        for exp in exps[:12]:
            try:
                ch = tk.option_chain(exp)
                total_call_oi = ch.calls['openInterest'].sum()
                total_put_oi = ch.puts['openInterest'].sum()
                pcr = total_put_oi / total_call_oi if total_call_oi > 0 else np.nan
                dte = max((pd.to_datetime(exp) - pd.Timestamp.now()).days, 1)
                pcr_data.append({
                    "Expiration": exp, "DTE": dte, "Put/Call OI": pcr,
                    "Call OI": total_call_oi, "Put OI": total_put_oi
                })
            except Exception:
                continue

        if pcr_data:
            pcr_df = pd.DataFrame(pcr_data)
            fig_pcr = go.Figure()
            fig_pcr.add_trace(go.Bar(
                x=pcr_df['Expiration'], y=pcr_df['Put/Call OI'],
                marker_color=['red' if p > 1 else 'green' for p in pcr_df['Put/Call OI']]
            ))
            fig_pcr.add_hline(y=1, line_dash="dash", annotation_text="Neutral (P/C = 1)")
            fig_pcr.update_layout(
                title="Put/Call OI Ratio by Expiration",
                yaxis_title="Put/Call Ratio",
                template="plotly_white"
            )
            st.plotly_chart(fig_pcr, use_container_width=True)

        # --- Max Pain ---
        st.subheader("Max Pain Estimate")
        st.caption("The strike where total option holder losses are maximized (and writer profits maximized).")

        mp_calls = opt.calls[['strike', 'openInterest']].copy()
        mp_puts = opt.puts[['strike', 'openInterest']].copy()
        strikes = sorted(set(mp_calls['strike'].tolist() + mp_puts['strike'].tolist()))

        pain = []
        for s in strikes:
            call_pain = mp_calls.apply(
                lambda r: r['openInterest'] * max(s - r['strike'], 0), axis=1
            ).sum()
            put_pain = mp_puts.apply(
                lambda r: r['openInterest'] * max(r['strike'] - s, 0), axis=1
            ).sum()
            pain.append({'Strike': s, 'Total Pain': call_pain + put_pain})

        pain_df = pd.DataFrame(pain)
        max_pain_strike = pain_df.loc[pain_df['Total Pain'].idxmin(), 'Strike']

        fig_mp = go.Figure()
        fig_mp.add_trace(go.Scatter(
            x=pain_df['Strike'], y=pain_df['Total Pain'],
            fill='tozeroy', name='Total Pain',
            line=dict(color='#636EFA')
        ))
        fig_mp.add_vline(
            x=max_pain_strike, line_dash="dash", line_color="red",
            annotation_text=f"Max Pain: ${max_pain_strike:.0f}"
        )
        if spot:
            fig_mp.add_vline(
                x=spot, line_dash="dot", line_color="green",
                annotation_text=f"Spot: ${spot:.0f}"
            )
        fig_mp.update_layout(
            title=f"Max Pain Analysis ({oi_exp})",
            xaxis_title="Strike", yaxis_title="Total $ Pain",
            yaxis_tickformat="$,.0f", template="plotly_white"
        )
        st.plotly_chart(fig_mp, use_container_width=True)

# ==========================================
# Page 3: Factor Exposure & Attribution
# ==========================================

def show_factor_page(portfolio_df, end_date, rf):
    st.title("🧬 Factor Exposure & Attribution")
    st.markdown("---")
    st.caption("Regresses portfolio returns against factor proxies to decompose performance into systematic exposures and residual alpha.")

    port_ret, _, _, _, _ = build_portfolio_returns(portfolio_df, end_date)
    if port_ret is None or len(port_ret) < 60:
        st.warning("Need at least 60 trading days for factor regression.")
        return

    factor_map = {
        "Market (SPY)": "SPY", "Size-SMB (IWM-SPY)": ("IWM", "SPY"),
        "Value-HML (IWD-IWF)": ("IWD", "IWF"), "Momentum (MTUM)": "MTUM",
        "Quality (QUAL)": "QUAL", "Low Vol (USMV)": "USMV"
    }
    st.info("ETF factor proxies: SPY (Mkt), IWM−SPY (Size), IWD−IWF (Value), MTUM, QUAL, USMV.")

    all_etfs = list(set(v if isinstance(v, str) else v[0] for v in factor_map.values()) |
                     set(v[1] for v in factor_map.values() if isinstance(v, tuple)))
    etf_prices = get_data(all_etfs, portfolio_df['Start Date'].min(), end_date)
    if etf_prices.empty:
        st.error("Could not fetch factor ETF data.")
        return
    etf_ret = etf_prices.pct_change().dropna()

    factors = pd.DataFrame(index=etf_ret.index)
    for name, proxy in factor_map.items():
        try:
            factors[name] = etf_ret[proxy] if isinstance(proxy, str) else etf_ret[proxy[0]] - etf_ret[proxy[1]]
        except KeyError:
            continue

    ci = port_ret.index.intersection(factors.index)
    y = port_ret.loc[ci] - rf / 252
    X = factors.loc[ci].dropna(axis=1, how='all').dropna()
    y = y.loc[X.index]
    if len(y) < 30:
        st.warning("Insufficient overlapping data for regression.")
        return

    model = OLS(y, add_constant(X)).fit()
    alpha_ann = model.params.get('const', 0) * 252

    st.subheader("Factor Regression Results")
    c1, c2, c3 = st.columns(3)
    c1.metric("Annualized Alpha", f"{alpha_ann:.2%}",
              help="Return not explained by factor exposures, annualized.")
    c2.metric("R-squared", f"{model.rsquared:.2%}",
              help="Fraction of portfolio return variance explained by factors.")
    c3.metric("Adj. R-squared", f"{model.rsquared_adj:.2%}",
              help="R-squared adjusted for number of factors.")

    st.subheader("Factor Betas (Loadings)")
    betas = pd.DataFrame({
        "Beta": model.params.drop('const', errors='ignore'),
        "t-stat": model.tvalues.drop('const', errors='ignore'),
        "p-value": model.pvalues.drop('const', errors='ignore')
    })
    betas['Significant'] = betas['p-value'] < 0.05
    st.dataframe(betas.style.format({"Beta": "{:.4f}", "t-stat": "{:.2f}", "p-value": "{:.4f}"}),
                  use_container_width=True)

    fig_beta = go.Figure()
    colors = ['#00CC96' if b > 0 else '#EF553B' for b in betas['Beta']]
    fig_beta.add_trace(go.Bar(x=betas.index, y=betas['Beta'], marker_color=colors))
    fig_beta.update_layout(title="Factor Exposures (Betas)", yaxis_title="Beta Loading", template="plotly_white")
    st.plotly_chart(fig_beta, use_container_width=True)

    st.subheader("Return Attribution")
    st.caption("Decomposes your portfolio's total excess return into contributions from each factor plus alpha.")
    contrib = (X.mean() * model.params.drop('const', errors='ignore')) * 252
    contrib['Alpha'] = alpha_ann
    fig_attr = go.Figure()
    colors_attr = ['#00CC96' if v > 0 else '#EF553B' for v in contrib]
    fig_attr.add_trace(go.Bar(x=contrib.index, y=contrib.values, marker_color=colors_attr))
    fig_attr.update_layout(title="Annualized Return Attribution", yaxis_title="Contribution to Return",
                            yaxis_tickformat=".2%", template="plotly_white")
    st.plotly_chart(fig_attr, use_container_width=True)

    st.subheader("Rolling Alpha")
    w = min(60, max(20, len(y) // 5))
    rolling_alpha = []
    for i in range(w, len(y)):
        try:
            rm = OLS(y.iloc[i - w:i], add_constant(X.iloc[i - w:i])).fit()
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
            for tk_s in universe:
                try:
                    s = prices[tk_s].dropna()
                    if len(s) < 30:
                        continue
                    ret = s.pct_change().dropna()
                    rsi = compute_rsi(s).iloc[-1]
                    ma = s.rolling(20).mean()
                    std = s.rolling(20).std()
                    zscore = ((s - ma) / std).iloc[-1]
                    mom_20 = (s.iloc[-1] / s.iloc[-21] - 1) if len(s) > 21 else np.nan
                    vol_20 = ret.tail(20).std() * np.sqrt(252)
                    signal = "Oversold" if rsi < 30 else ("Overbought" if rsi > 70 else
                             ("Mean-Rev Long" if zscore < -2 else ("Mean-Rev Short" if zscore > 2 else "Neutral")))
                    results.append({
                        "Ticker": tk_s, "Last Price": s.iloc[-1], "RSI (14)": rsi,
                        "Z-Score (20D)": zscore, "20D Momentum": mom_20,
                        "20D Ann. Vol": vol_20, "Signal": signal
                    })
                except Exception:
                    continue

            if results:
                df_res = pd.DataFrame(results)
                st.dataframe(df_res.style.format({
                    "Last Price": "${:.2f}", "RSI (14)": "{:.1f}", "Z-Score (20D)": "{:.2f}",
                    "20D Momentum": "{:.2%}", "20D Ann. Vol": "{:.2%}"
                }).applymap(lambda v: 'color: green' if v in ['Oversold', 'Mean-Rev Long'] else
                            ('color: red' if v in ['Overbought', 'Mean-Rev Short'] else ''),
                            subset=['Signal']),
                    use_container_width=True)

                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Bar(x=df_res['Ticker'], y=df_res['RSI (14)'],
                                          marker_color=['green' if r < 30 else 'red' if r > 70 else 'gray' for r in df_res['RSI (14)']]))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                fig_rsi.update_layout(title="RSI Overview", template="plotly_white")
                st.plotly_chart(fig_rsi, use_container_width=True)

    with tab_coint:
        st.subheader("Cointegration Pair Scanner")
        st.caption("Scans all pairs for cointegration, ranks by p-value and estimated half-life.")
        pair_universe = st.text_input("Pair Universe (comma-separated)",
                                       "XOM, CVX, COP, SLB, EOG, MPC, VLO, PSX", key="coint_univ")
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
                    idx = s1.index.intersection(s2.index)
                    if len(idx) < 30:
                        continue
                    s1c, s2c = s1.loc[idx], s2.loc[idx]

                    # Cointegration test
                    _, pval, _ = coint(s1c, s2c)

                    # Hedge ratio via OLS
                    hr_model = OLS(s1c.values, add_constant(s2c.values)).fit()
                    spread = s1c - hr_model.params[1] * s2c

                    # Half-life via AR(1) on spread
                    sp_lag = spread.shift(1).dropna()
                    sp_diff = spread.diff().dropna()
                    ci2 = sp_lag.index.intersection(sp_diff.index)
                    if len(ci2) < 10:
                        continue
                    ar_model = OLS(sp_diff.loc[ci2], add_constant(sp_lag.loc[ci2])).fit()
                    hl = -np.log(2) / ar_model.params.iloc[1] if ar_model.params.iloc[1] < 0 else np.nan

                    # Current spread z-score
                    z = (spread.iloc[-1] - spread.mean()) / spread.std()

                    # Hurst exponent (simplified R/S method)
                    ts_vals = spread.values
                    n_pts = len(ts_vals)
                    hurst = np.nan
                    if n_pts > 20:
                        max_k = min(n_pts // 2, 100)
                        rs_list = []
                        for k in [20, 50, max_k]:
                            if k > n_pts:
                                continue
                            rs_vals = []
                            for start_idx in range(0, n_pts - k, k):
                                chunk = ts_vals[start_idx:start_idx + k]
                                m_c = chunk.mean()
                                dev = np.cumsum(chunk - m_c)
                                r_s = (dev.max() - dev.min()) / (chunk.std() + 1e-10)
                                rs_vals.append(r_s)
                            if rs_vals:
                                rs_list.append((np.log(k), np.log(np.mean(rs_vals))))
                        if len(rs_list) >= 2:
                            xs, ys = zip(*rs_list)
                            hurst = np.polyfit(xs, ys, 1)[0]

                    pairs.append({
                        "Pair": f"{t1}/{t2}", "Coint p-val": pval, "Half-Life": hl,
                        "Spread Z": z, "Hurst": hurst, "Hedge Ratio": hr_model.params[1],
                        "Mean-Reverting": "✅" if pval < 0.05 and (not np.isnan(hurst) and hurst < 0.5) else "❌"
                    })

            if pairs:
                pdf = pd.DataFrame(pairs).sort_values("Coint p-val")
                st.dataframe(pdf.style.format({
                    "Coint p-val": "{:.4f}", "Half-Life": "{:.1f}", "Spread Z": "{:.2f}",
                    "Hurst": "{:.3f}", "Hedge Ratio": "{:.4f}"
                }), use_container_width=True)

                # Spread chart for top pair
                top = pdf.iloc[0]
                t1, t2 = top['Pair'].split('/')
                s1, s2 = prices[t1].dropna(), prices[t2].dropna()
                idx = s1.index.intersection(s2.index)
                spread = s1.loc[idx] - top['Hedge Ratio'] * s2.loc[idx]
                mu, sigma = spread.mean(), spread.std()

                fig_sp = go.Figure()
                fig_sp.add_trace(go.Scatter(x=spread.index, y=spread, name='Spread', line=dict(color='#636EFA')))
                fig_sp.add_hline(y=mu, line_dash="solid", line_color="gray", annotation_text="Mean")
                fig_sp.add_hline(y=mu + 2 * sigma, line_dash="dash", line_color="red", annotation_text="+2σ")
                fig_sp.add_hline(y=mu - 2 * sigma, line_dash="dash", line_color="green", annotation_text="−2σ")
                fig_sp.update_layout(title=f"Top Pair Spread: {top['Pair']}", template="plotly_white")
                st.plotly_chart(fig_sp, use_container_width=True)


# ==========================================
# Page 5: Trade Journal & Post-Mortem (Full Upgraded)
# ==========================================

def show_journal_page():
    st.title("📓 Trade Journal & Post-Mortem")
    st.markdown("---")

    if 'trades' not in st.session_state:
        st.session_state.trades = []
    if 'export_reminder' not in st.session_state:
        st.session_state.export_reminder = 0

    # Persistence reminder: every 5 new trades
    st.session_state.export_reminder = len(st.session_state.trades)
    if st.session_state.export_reminder > 0 and st.session_state.export_reminder % 5 == 0:
        st.warning(f"💾 You have **{st.session_state.export_reminder}** trades in session. Consider exporting to CSV to avoid losing data.")

    tab_log, tab_open, tab_analysis, tab_replay, tab_data = st.tabs([
        "Log Trade", "Open Positions", "Post-Mortem Analytics", "Trade Replay", "Trade History"
    ])

    # ================================================================
    # TAB 1: LOG TRADE
    # ================================================================
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

            tags = st.multiselect("Tags", ["Momentum", "Mean Reversion", "Breakout", "Earnings",
                                            "Macro", "Technical", "Fundamental", "Pairs", "Options", "Swing", "Day Trade"])

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

        # CSV upload
        st.markdown("---")
        uploaded = st.file_uploader("Import Trade History (CSV)", type=["csv"], key="trade_csv")
        if uploaded:
            try:
                imp = pd.read_csv(uploaded)
                st.session_state.trades = imp.to_dict('records')
                st.success(f"Imported {len(imp)} trades.")
            except Exception as e:
                st.error(f"Import error: {e}")

    # ================================================================
    # TAB 2: OPEN POSITIONS TRACKER
    # ================================================================
    with tab_open:
        st.subheader("Open Positions — Live P&L")
        open_trades = [t for t in st.session_state.trades if t.get('Status') == 'Open']

        if not open_trades:
            st.info("No open positions. Log a trade with Exit Price = 0 to track it here.")
        else:
            live_data = []
            for t in open_trades:
                tk_sym = t['Ticker']
                try:
                    tk_obj = yf.Ticker(tk_sym)
                    live_price = tk_obj.info.get('currentPrice') or tk_obj.info.get('regularMarketPrice')
                    if not live_price:
                        hp = tk_obj.history(period="1d")
                        live_price = hp['Close'].iloc[-1] if not hp.empty else None
                except Exception:
                    live_price = None

                entry = t['Entry']
                shares_held = t['Shares']
                direction = t['Direction']
                entry_date = t['Entry Date']
                days_held = (datetime.now().date() - pd.to_datetime(entry_date).date()).days

                if live_price:
                    mult = 1 if direction == "Long" else -1
                    unrealized_pnl = mult * (live_price - entry) * shares_held
                    unrealized_pct = mult * (live_price / entry - 1)
                    cost_basis = entry * shares_held
                    market_val = live_price * shares_held
                else:
                    unrealized_pnl = 0
                    unrealized_pct = 0
                    cost_basis = entry * shares_held
                    market_val = 0

                live_data.append({
                    'Ticker': tk_sym, 'Direction': direction, 'Shares': shares_held,
                    'Entry': entry, 'Current': live_price or 0,
                    'Cost Basis': cost_basis, 'Market Value': market_val,
                    'Unrealized P&L': unrealized_pnl, 'Return': unrealized_pct,
                    'Days Held': days_held, 'Thesis': t.get('Thesis', '')
                })

            if live_data:
                live_df = pd.DataFrame(live_data)

                # Summary metrics
                total_unrealized = live_df['Unrealized P&L'].sum()
                total_cost = live_df['Cost Basis'].sum()
                total_mkt = live_df['Market Value'].sum()

                lc1, lc2, lc3, lc4 = st.columns(4)
                lc1.metric("Open Positions", len(live_df))
                lc2.metric("Total Cost Basis", f"${total_cost:,.2f}")
                lc3.metric("Total Market Value", f"${total_mkt:,.2f}")
                color = "normal" if total_unrealized >= 0 else "inverse"
                lc4.metric("Total Unrealized P&L", f"${total_unrealized:,.2f}",
                            delta=f"{total_unrealized / total_cost:.2%}" if total_cost > 0 else "N/A")

                st.dataframe(live_df.style.format({
                    'Entry': '${:.2f}', 'Current': '${:.2f}',
                    'Cost Basis': '${:,.2f}', 'Market Value': '${:,.2f}',
                    'Unrealized P&L': '${:,.2f}', 'Return': '{:.2%}'
                }).applymap(lambda v: 'color: green' if isinstance(v, (int, float)) and v > 0 else
                            ('color: red' if isinstance(v, (int, float)) and v < 0 else ''),
                            subset=['Unrealized P&L', 'Return']),
                    use_container_width=True)

                # P&L bar chart
                fig_open = go.Figure()
                colors = ['#00CC96' if p >= 0 else '#EF553B' for p in live_df['Unrealized P&L']]
                fig_open.add_trace(go.Bar(x=live_df['Ticker'], y=live_df['Unrealized P&L'], marker_color=colors))
                fig_open.add_hline(y=0, line_dash="dash", line_color="gray")
                fig_open.update_layout(title="Unrealized P&L by Position", yaxis_tickformat="$,.2f",
                                        template="plotly_white")
                st.plotly_chart(fig_open, use_container_width=True)

    # ================================================================
    # TAB 3: POST-MORTEM ANALYTICS
    # ================================================================
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

        # --- Core Stats ---
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Trades", len(df_t))
        win_rate = len(wins) / len(df_t) if len(df_t) > 0 else 0
        c2.metric("Win Rate", f"{win_rate:.1%}",
                   help="Percentage of trades with positive P&L.")
        avg_win = wins['P&L ($)'].mean() if len(wins) > 0 else 0
        avg_loss = abs(losses['P&L ($)'].mean()) if len(losses) > 0 else 0
        profit_factor = avg_win / avg_loss if avg_loss > 0 else float('inf')
        c3.metric("Profit Factor", f"{profit_factor:.2f}",
                   help="Average win / Average loss. Above 1.5 is solid.")
        c4.metric("Total P&L", f"${df_t['P&L ($)'].sum():,.2f}")
        expectancy = df_t['P&L ($)'].mean()
        c5.metric("Expectancy ($/trade)", f"${expectancy:,.2f}",
                   help="Average P&L per trade.")

        c6, c7, c8 = st.columns(3)
        c6.metric("Avg Win", f"${avg_win:,.2f}")
        c7.metric("Avg Loss", f"-${avg_loss:,.2f}")
        if 'Holding Days' in df_t.columns:
            c8.metric("Avg Holding Period", f"{df_t['Holding Days'].mean():.0f} days")

        # --- Risk-Adjusted Metrics ---
        st.markdown("---")
        st.subheader("Risk-Adjusted Metrics")

        # Kelly Criterion
        if win_rate > 0 and avg_loss > 0:
            payoff_ratio = avg_win / avg_loss
            kelly = win_rate - ((1 - win_rate) / payoff_ratio)
            half_kelly = kelly / 2
        else:
            kelly = 0
            half_kelly = 0
            payoff_ratio = 0

        # Sharpe on trade P&L
        trade_sharpe = df_t['P&L ($)'].mean() / df_t['P&L ($)'].std() if df_t['P&L ($)'].std() > 0 else 0

        # Max consecutive wins/losses
        pnl_series = df_t['P&L ($)'].values
        streaks = []
        current_streak = 0
        for pnl in pnl_series:
            if pnl > 0:
                current_streak = current_streak + 1 if current_streak > 0 else 1
            else:
                current_streak = current_streak - 1 if current_streak < 0 else -1
            streaks.append(current_streak)
        max_win_streak = max(streaks) if streaks else 0
        max_loss_streak = abs(min(streaks)) if streaks else 0

        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("Kelly Criterion", f"{kelly:.1%}",
                    help=f"Optimal bet size: {kelly:.1%} of capital. Half-Kelly (safer): {half_kelly:.1%}. "
                         f"Based on {win_rate:.0%} win rate and {payoff_ratio:.1f}x payoff ratio.")
        rc2.metric("Half-Kelly", f"{half_kelly:.1%}",
                    help="More conservative — half of full Kelly. Widely used in practice.")
        rc3.metric("Trade Sharpe", f"{trade_sharpe:.2f}",
                    help="Mean P&L / Std Dev of P&L. Measures consistency of returns across trades.")
        rc4.metric("Payoff Ratio", f"{payoff_ratio:.2f}x",
                    help="Average win size / Average loss size.")

        # --- Win/Loss Streaks & Tilt Detection ---
        st.markdown("---")
        st.subheader("Win/Loss Streaks & Tilt Detection")

        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Max Win Streak", f"{max_win_streak}")
        sc2.metric("Max Loss Streak", f"{max_loss_streak}")
        current_streak_val = streaks[-1] if streaks else 0
        streak_label = f"{abs(current_streak_val)} {'Wins' if current_streak_val > 0 else 'Losses'}" if current_streak_val != 0 else "Even"
        sc3.metric("Current Streak", streak_label)

        # Streak chart
        fig_streak = go.Figure()
        streak_colors = ['#00CC96' if s > 0 else '#EF553B' if s < 0 else 'gray' for s in streaks]
        fig_streak.add_trace(go.Bar(
            x=list(range(1, len(streaks) + 1)), y=streaks,
            marker_color=streak_colors, name='Streak'
        ))
        fig_streak.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_streak.update_layout(title="Win/Loss Streak History",
                                  xaxis_title="Trade #", yaxis_title="Streak (+ = wins, − = losses)",
                                  template="plotly_white")
        st.plotly_chart(fig_streak, use_container_width=True)

        # Tilt detection
        if max_loss_streak >= 3:
            # Check if position sizes increased during the losing streak
            streak_start = None
            for i in range(len(streaks)):
                if streaks[i] <= -3 and streak_start is None:
                    streak_start = max(0, i - abs(streaks[i]) + 1)
            if streak_start is not None:
                pre_streak_avg_size = df_t.iloc[:streak_start]['Shares'].mean() if streak_start > 0 else df_t['Shares'].mean()
                during_streak = df_t.iloc[streak_start:streak_start + max_loss_streak]
                during_avg_size = during_streak['Shares'].mean() if not during_streak.empty else 0
                if during_avg_size > pre_streak_avg_size * 1.3:
                    st.error(f"⚠️ **Tilt Detected:** During your {max_loss_streak}-trade losing streak, "
                             f"average position size increased from {pre_streak_avg_size:.1f} to {during_avg_size:.1f} shares "
                             f"({(during_avg_size / pre_streak_avg_size - 1):.0%} larger). This is a classic revenge-trading pattern.")
                elif during_avg_size > pre_streak_avg_size * 1.1:
                    st.warning(f"⚠️ **Mild Tilt Warning:** Position sizes slightly elevated during your losing streak "
                               f"({pre_streak_avg_size:.1f} → {during_avg_size:.1f} shares).")
                else:
                    st.success("✅ No tilt detected — position sizing stayed disciplined during losing streaks.")

        # Rolling Expectancy Curve
        st.subheader("Rolling Expectancy Curve")
        st.caption("Shows how your edge (average P&L per trade) evolves over time. Uptrend = improving, downtrend = edge decay.")
        if len(df_t) >= 10:
            roll_w = min(20, len(df_t) // 3)
            rolling_exp = df_t['P&L ($)'].rolling(roll_w).mean()
            fig_exp = go.Figure()
            fig_exp.add_trace(go.Scatter(x=list(range(1, len(rolling_exp) + 1)), y=rolling_exp,
                                          name=f'{roll_w}-Trade Rolling Expectancy', line=dict(color='#636EFA', width=2)))
            fig_exp.add_hline(y=0, line_dash="dash", line_color="red")
            fig_exp.add_hline(y=expectancy, line_dash="dot", line_color="orange",
                              annotation_text=f"Overall Avg: ${expectancy:,.0f}")
            fig_exp.update_layout(title=f"Rolling {roll_w}-Trade Expectancy",
                                   yaxis_tickformat="$,.2f", xaxis_title="Trade #", template="plotly_white")
            st.plotly_chart(fig_exp, use_container_width=True)

        # --- P&L Distribution ---
        st.markdown("---")
        st.subheader("P&L Distribution")
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(x=df_t['P&L ($)'], nbinsx=20, marker_color='#636EFA'))
        fig_dist.add_vline(x=0, line_dash="dash", line_color="red")
        fig_dist.add_vline(x=expectancy, line_dash="dot", line_color="orange",
                            annotation_text=f"Avg: ${expectancy:,.0f}")
        fig_dist.update_layout(title="Trade P&L Distribution", xaxis_title="P&L ($)", template="plotly_white")
        st.plotly_chart(fig_dist, use_container_width=True)

        # --- Equity Curve with Drawdown ---
        st.markdown("---")
        st.subheader("Equity Curve & Drawdown")
        df_t_sorted = df_t.sort_values('Exit Date').copy()
        df_t_sorted['Cum P&L'] = df_t_sorted['P&L ($)'].cumsum()
        df_t_sorted['Peak'] = df_t_sorted['Cum P&L'].cummax()
        df_t_sorted['Drawdown'] = df_t_sorted['Cum P&L'] - df_t_sorted['Peak']

        max_dd = df_t_sorted['Drawdown'].min()
        max_dd_idx = df_t_sorted['Drawdown'].idxmin()
        # Recovery: trades after max DD to get back to peak
        post_dd = df_t_sorted.loc[max_dd_idx:]
        recovered = post_dd[post_dd['Cum P&L'] >= df_t_sorted.loc[max_dd_idx, 'Peak']]
        recovery_trades = len(post_dd.loc[:recovered.index[0]]) if not recovered.empty else None

        fig_eq = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                                vertical_spacing=0.05)
        fig_eq.add_trace(go.Scatter(
            x=df_t_sorted['Exit Date'], y=df_t_sorted['Cum P&L'],
            fill='tozeroy', line=dict(color='#00CC96', width=2), name='Equity Curve'
        ), row=1, col=1)
        fig_eq.add_trace(go.Scatter(
            x=df_t_sorted['Exit Date'], y=df_t_sorted['Peak'],
            line=dict(color='gray', dash='dot', width=1), name='High Water Mark'
        ), row=1, col=1)
        fig_eq.add_trace(go.Scatter(
            x=df_t_sorted['Exit Date'], y=df_t_sorted['Drawdown'],
            fill='tozeroy', line=dict(color='red', width=1), name='Drawdown',
            fillcolor='rgba(239,85,59,0.2)'
        ), row=2, col=1)
        fig_eq.update_yaxes(tickformat="$,.2f", row=1, col=1)
        fig_eq.update_yaxes(tickformat="$,.2f", title_text="Drawdown", row=2, col=1)
        fig_eq.update_layout(title="Equity Curve with Drawdown", template="plotly_white", height=500)
        st.plotly_chart(fig_eq, use_container_width=True)

        dd1, dd2, dd3 = st.columns(3)
        dd1.metric("Max Drawdown", f"${max_dd:,.2f}",
                    help="Largest peak-to-trough decline in cumulative P&L.")
        dd2.metric("Max DD as % of Peak", f"{max_dd / df_t_sorted['Peak'].max():.1%}" if df_t_sorted['Peak'].max() != 0 else "N/A",
                    help="Drawdown relative to the highest equity point.")
        dd3.metric("Recovery (Trades)", f"{recovery_trades}" if recovery_trades else "Not recovered",
                    help="Number of trades to recover from max drawdown back to high water mark.")

        # --- Performance Heatmaps ---
        st.markdown("---")
        st.subheader("Performance Heatmaps")

        # Parse exit dates
        df_t_sorted['Exit Date Parsed'] = pd.to_datetime(df_t_sorted['Exit Date'], errors='coerce')
        df_with_dates = df_t_sorted.dropna(subset=['Exit Date Parsed'])

        if not df_with_dates.empty:
            df_with_dates['Weekday'] = df_with_dates['Exit Date Parsed'].dt.day_name()
            df_with_dates['Month'] = df_with_dates['Exit Date Parsed'].dt.month_name()

            hm1, hm2 = st.columns(2)

            with hm1:
                # By weekday
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
                day_stats = df_with_dates.groupby('Weekday')['P&L ($)'].agg(['mean', 'sum', 'count']).reindex(day_order).dropna()
                if not day_stats.empty:
                    fig_day = go.Figure()
                    fig_day.add_trace(go.Bar(
                        x=day_stats.index, y=day_stats['mean'],
                        marker_color=['#00CC96' if v >= 0 else '#EF553B' for v in day_stats['mean']],
                        text=[f"n={int(c)}" for c in day_stats['count']], textposition='outside'
                    ))
                    fig_day.add_hline(y=0, line_dash="dash", line_color="gray")
                    fig_day.update_layout(title="Avg P&L by Day of Week", yaxis_tickformat="$,.2f",
                                           template="plotly_white")
                    st.plotly_chart(fig_day, use_container_width=True)

            with hm2:
                # By month
                month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                               'July', 'August', 'September', 'October', 'November', 'December']
                month_stats = df_with_dates.groupby('Month')['P&L ($)'].agg(['mean', 'sum', 'count']).reindex(month_order).dropna()
                if not month_stats.empty:
                    fig_month = go.Figure()
                    fig_month.add_trace(go.Bar(
                        x=month_stats.index, y=month_stats['mean'],
                        marker_color=['#00CC96' if v >= 0 else '#EF553B' for v in month_stats['mean']],
                        text=[f"n={int(c)}" for c in month_stats['count']], textposition='outside'
                    ))
                    fig_month.add_hline(y=0, line_dash="dash", line_color="gray")
                    fig_month.update_layout(title="Avg P&L by Month", yaxis_tickformat="$,.2f",
                                             template="plotly_white")
                    st.plotly_chart(fig_month, use_container_width=True)

            # P&L vs Holding Period
            if 'Holding Days' in df_t.columns:
                st.subheader("P&L vs Holding Period")
                fig_hold = go.Figure()
                fig_hold.add_trace(go.Scatter(
                    x=df_t['Holding Days'], y=df_t['P&L ($)'],
                    mode='markers', marker=dict(
                        size=10, color=df_t['P&L ($)'],
                        colorscale='RdYlGn', cmid=0, showscale=True,
                        colorbar=dict(title='P&L ($)')
                    ),
                    text=df_t['Ticker'], name='Trades'
                ))
                fig_hold.add_hline(y=0, line_dash="dash", line_color="gray")
                fig_hold.update_layout(title="P&L vs Holding Period",
                                        xaxis_title="Holding Days", yaxis_title="P&L ($)",
                                        yaxis_tickformat="$,.2f", template="plotly_white")
                st.plotly_chart(fig_hold, use_container_width=True)

                # Optimal holding period analysis
                if len(df_t) >= 10:
                    short_trades = df_t[df_t['Holding Days'] <= 5]
                    medium_trades = df_t[(df_t['Holding Days'] > 5) & (df_t['Holding Days'] <= 20)]
                    long_trades = df_t[df_t['Holding Days'] > 20]
                    hp1, hp2, hp3 = st.columns(3)
                    if not short_trades.empty:
                        hp1.metric("Day Trades (≤5D)", f"${short_trades['P&L ($)'].mean():,.2f}/trade",
                                    help=f"{len(short_trades)} trades, {(short_trades['P&L ($)'] > 0).mean():.0%} win rate")
                    if not medium_trades.empty:
                        hp2.metric("Swing Trades (5-20D)", f"${medium_trades['P&L ($)'].mean():,.2f}/trade",
                                    help=f"{len(medium_trades)} trades, {(medium_trades['P&L ($)'] > 0).mean():.0%} win rate")
                    if not long_trades.empty:
                        hp3.metric("Position Trades (>20D)", f"${long_trades['P&L ($)'].mean():,.2f}/trade",
                                    help=f"{len(long_trades)} trades, {(long_trades['P&L ($)'] > 0).mean():.0%} win rate")

        # --- Long vs Short ---
        if len(df_t['Direction'].unique()) > 1:
            st.markdown("---")
            st.subheader("Long vs Short Performance")
            dir_stats = df_t.groupby('Direction').agg(
                Trades=('P&L ($)', 'count'),
                WinRate=('P&L ($)', lambda x: (x > 0).mean()),
                AvgPnL=('P&L ($)', 'mean'),
                TotalPnL=('P&L ($)', 'sum')
            ).reset_index()
            st.dataframe(dir_stats.style.format({
                "WinRate": "{:.1%}", "AvgPnL": "${:,.2f}", "TotalPnL": "${:,.2f}"
            }), use_container_width=True)

        # --- By Tag ---
        if any(t.get('Tags') for t in closed):
            st.markdown("---")
            st.subheader("Performance by Strategy Tag")
            tag_rows = []
            for t in closed:
                for tag in (t.get('Tags') or []):
                    tag_rows.append({"Tag": tag, "P&L": t['P&L ($)']})
            if tag_rows:
                tag_df = pd.DataFrame(tag_rows)
                tag_stats = tag_df.groupby('Tag').agg(
                    Trades=('P&L', 'count'),
                    WinRate=('P&L', lambda x: (x > 0).mean()),
                    AvgPnL=('P&L', 'mean'),
                    TotalPnL=('P&L', 'sum')
                ).reset_index()
                st.dataframe(tag_stats.style.format({
                    "WinRate": "{:.1%}", "AvgPnL": "${:,.2f}", "TotalPnL": "${:,.2f}"
                }), use_container_width=True)

                fig_tag = go.Figure()
                fig_tag.add_trace(go.Bar(
                    x=tag_stats['Tag'], y=tag_stats['AvgPnL'],
                    marker_color=['#00CC96' if v >= 0 else '#EF553B' for v in tag_stats['AvgPnL']],
                    text=[f"n={int(n)}" for n in tag_stats['Trades']], textposition='outside'
                ))
                fig_tag.add_hline(y=0, line_dash="dash", line_color="gray")
                fig_tag.update_layout(title="Avg P&L by Strategy Tag", yaxis_tickformat="$,.2f",
                                       template="plotly_white")
                st.plotly_chart(fig_tag, use_container_width=True)

    # ================================================================
    # TAB 4: TRADE REPLAY
    # ================================================================
    with tab_replay:
        st.subheader("Trade Replay — Visual Review")
        st.caption("Select a closed trade to see the actual price action during your holding period with entry/exit marked.")

        closed_trades = [t for t in st.session_state.trades if t.get('Status') == 'Closed']
        if not closed_trades:
            st.info("No closed trades to replay.")
        else:
            # Build selection labels
            trade_labels = []
            for i, t in enumerate(closed_trades):
                pnl = t.get('P&L ($)', 0)
                emoji = "🟢" if pnl > 0 else "🔴"
                trade_labels.append(
                    f"{emoji} #{i+1}: {t['Ticker']} {t['Direction']} | "
                    f"{t['Entry Date']} → {t['Exit Date']} | P&L: ${pnl:,.2f}"
                )

            selected_idx = st.selectbox("Select Trade to Replay", range(len(trade_labels)),
                                         format_func=lambda i: trade_labels[i], key="replay_sel")
            trade = closed_trades[selected_idx]

            # Trade details
            tc1, tc2, tc3, tc4 = st.columns(4)
            tc1.metric("Ticker", trade['Ticker'])
            tc2.metric("Direction", trade['Direction'])
            tc3.metric("P&L", f"${trade['P&L ($)']:,.2f}")
            tc4.metric("Holding", f"{trade.get('Holding Days', 0)} days")

            if trade.get('Thesis'):
                st.markdown(f"**Thesis:** {trade['Thesis']}")

            # Fetch price data for the trade period (with buffer)
            entry_date = pd.to_datetime(trade['Entry Date']).date()
            exit_date = pd.to_datetime(trade['Exit Date']).date()
            buffer_days = max(10, (exit_date - entry_date).days // 2)
            chart_start = entry_date - timedelta(days=buffer_days)
            chart_end = exit_date + timedelta(days=buffer_days)

            chart_data = get_data([trade['Ticker']], chart_start, chart_end)

            if not chart_data.empty:
                if isinstance(chart_data, pd.DataFrame) and trade['Ticker'] in chart_data.columns:
                    price_series = chart_data[trade['Ticker']]
                else:
                    price_series = chart_data.squeeze()

                fig_replay = go.Figure()

                # Price line
                fig_replay.add_trace(go.Scatter(
                    x=price_series.index, y=price_series,
                    name='Price', line=dict(color='#636EFA', width=2)
                ))

                # Holding period highlight
                mask = (price_series.index >= pd.Timestamp(entry_date)) & (price_series.index <= pd.Timestamp(exit_date))
                holding_prices = price_series[mask]
                if not holding_prices.empty:
                    fig_replay.add_trace(go.Scatter(
                        x=holding_prices.index, y=holding_prices,
                        name='Holding Period', line=dict(color='orange', width=3),
                        fill='tozeroy', fillcolor='rgba(255,165,0,0.1)'
                    ))

                # Entry marker
                fig_replay.add_trace(go.Scatter(
                    x=[pd.Timestamp(entry_date)], y=[trade['Entry']],
                    mode='markers+text', name='Entry',
                    marker=dict(size=15, color='green', symbol='triangle-up'),
                    text=[f"Entry: ${trade['Entry']:.2f}"], textposition='top center'
                ))

                # Exit marker
                fig_replay.add_trace(go.Scatter(
                    x=[pd.Timestamp(exit_date)], y=[trade['Exit']],
                    mode='markers+text', name='Exit',
                    marker=dict(size=15, color='red', symbol='triangle-down'),
                    text=[f"Exit: ${trade['Exit']:.2f}"], textposition='bottom center'
                ))

                # High/low during holding
                if not holding_prices.empty:
                    high_during = holding_prices.max()
                    low_during = holding_prices.min()
                    fig_replay.add_hline(y=high_during, line_dash="dot", line_color="green",
                                          annotation_text=f"High: ${high_during:.2f}")
                    fig_replay.add_hline(y=low_during, line_dash="dot", line_color="red",
                                          annotation_text=f"Low: ${low_during:.2f}")

                pnl_color = "green" if trade['P&L ($)'] > 0 else "red"
                fig_replay.update_layout(
                    title=f"Trade Replay: {trade['Ticker']} {trade['Direction']} — "
                          f"<span style='color:{pnl_color}'>${trade['P&L ($)']:,.2f}</span>",
                    yaxis_tickformat="$,.2f", template="plotly_white", height=500
                )
                st.plotly_chart(fig_replay, use_container_width=True)

                # Trade analysis
                if not holding_prices.empty:
                    st.subheader("Trade Analysis")
                    ta1, ta2, ta3, ta4 = st.columns(4)
                    ta1.metric("Entry to High", f"{(high_during / trade['Entry'] - 1):.2%}",
                                help="Max favorable excursion — how much the trade went in your favor.")
                    ta2.metric("Entry to Low", f"{(low_during / trade['Entry'] - 1):.2%}",
                                help="Max adverse excursion — how much heat you took.")
                    if trade['Direction'] == 'Long':
                        money_left = (high_during - trade['Exit']) * trade['Shares']
                        ta3.metric("Money Left on Table", f"${money_left:,.2f}",
                                    help="Difference between the high during holding and your exit.")
                    else:
                        money_left = (trade['Exit'] - low_during) * trade['Shares']
                        ta3.metric("Money Left on Table", f"${money_left:,.2f}",
                                    help="Difference between your exit and the low during holding.")
                    ta4.metric("Captured", f"{abs(trade['P&L ($)']) / (abs(high_during - low_during) * trade['Shares']):.0%}" if (high_during - low_during) > 0 else "N/A",
                                help="What % of the total move (high-low) did you capture?")
            else:
                st.warning(f"Could not fetch price data for {trade['Ticker']}.")

    # ================================================================
    # TAB 5: TRADE HISTORY
    # ================================================================
    with tab_data:
        st.subheader("Full Trade History")
        if st.session_state.trades:
            df_all = pd.DataFrame(st.session_state.trades)
            st.dataframe(df_all, use_container_width=True)
            csv = df_all.to_csv(index=False).encode('utf-8')
            st.download_button("💾 Export Trade History", csv, "trade_journal.csv", "text/csv")

            dc1, dc2 = st.columns(2)
            with dc1:
                if st.button("🗑️ Clear All Trades", type="secondary"):
                    st.session_state.trades = []
                    st.rerun()
            with dc2:
                st.caption(f"Total: {len(df_all)} trades ({len([t for t in st.session_state.trades if t.get('Status') == 'Closed'])} closed, "
                           f"{len([t for t in st.session_state.trades if t.get('Status') == 'Open'])} open)")
        else:
            st.info("No trades logged yet.")

# ==========================================
# Page 6: Macro Regime Dashboard (Full)
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
            d = yf.download(tk, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'),
                            auto_adjust=True, progress=False)
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

    # ================================================================
    # CURRENT MACRO SNAPSHOT
    # ================================================================
    st.subheader("Current Macro Snapshot")
    c1, c2, c3, c4, c5 = st.columns(5)

    latest = macro.iloc[-1]
    prev = macro.iloc[-2] if len(macro) > 1 else latest

    credit_momentum = 0
    spread_10y3m = 0

    if '10Y Yield' in macro.columns and '3M Yield' in macro.columns:
        spread_10y3m = latest['10Y Yield'] - latest['3M Yield']
        prev_spread = prev['10Y Yield'] - prev['3M Yield']
        c1.metric("10Y-3M Spread", f"{spread_10y3m:.2f}%",
                   delta=f"{spread_10y3m - prev_spread:.2f}%",
                   help="Yield curve slope. Negative = inverted (recession signal).")

    if 'VIX' in macro.columns:
        vix = latest['VIX']
        c2.metric("VIX", f"{vix:.1f}",
                   delta=f"{vix - prev.get('VIX', vix):.1f}",
                   help="CBOE Volatility Index. >20 = elevated fear, >30 = crisis-level.")

    if 'HY Corp Bond' in macro.columns and 'IG Corp Bond' in macro.columns:
        hy_ret = macro['HY Corp Bond'].pct_change().tail(20).mean() * 252
        ig_ret = macro['IG Corp Bond'].pct_change().tail(20).mean() * 252
        credit_momentum = hy_ret - ig_ret
        c3.metric("HY-IG Momentum", f"{credit_momentum:.2%}",
                   help="Relative performance of high-yield vs investment-grade bonds. Negative = credit stress.")

    if 'DXY' in macro.columns:
        dxy = latest['DXY']
        c4.metric("Dollar Index (DXY)", f"{dxy:.1f}",
                   delta=f"{dxy - prev.get('DXY', dxy):.1f}",
                   help="US Dollar strength. Rising DXY = tightening financial conditions globally.")

    if 'Gold' in macro.columns:
        gold = latest['Gold']
        c5.metric("Gold", f"${gold:,.0f}",
                   delta=f"${gold - prev.get('Gold', gold):,.0f}",
                   help="Safe haven demand indicator.")

    # Second row of metrics
    c6, c7, c8, c9, c10 = st.columns(5)

    if '10Y Yield' in macro.columns:
        c6.metric("10Y Yield", f"{latest['10Y Yield']:.2f}%",
                   delta=f"{latest['10Y Yield'] - prev.get('10Y Yield', latest['10Y Yield']):.2f}%",
                   help="10-Year US Treasury yield.")

    if '3M Yield' in macro.columns:
        c7.metric("3M Yield", f"{latest['3M Yield']:.2f}%",
                   delta=f"{latest['3M Yield'] - prev.get('3M Yield', latest['3M Yield']):.2f}%",
                   help="3-Month T-bill yield.")

    if '5Y Yield' in macro.columns:
        c8.metric("5Y Yield", f"{latest['5Y Yield']:.2f}%",
                   delta=f"{latest['5Y Yield'] - prev.get('5Y Yield', latest['5Y Yield']):.2f}%",
                   help="5-Year US Treasury yield.")

    if '20Y Treasury' in macro.columns:
        tlt_ret_20d = macro['20Y Treasury'].pct_change().tail(20).sum()
        c9.metric("TLT (20D Return)", f"{tlt_ret_20d:.2%}",
                   help="20-day return on 20Y Treasury ETF. Positive = flight to safety / rates falling.")

    if 'SPY' in macro.columns:
        spy_ret_20d = macro['SPY'].pct_change().tail(20).sum()
        c10.metric("SPY (20D Return)", f"{spy_ret_20d:.2%}",
                    help="20-day S&P 500 return. Context for equity market direction.")

    # ================================================================
    # REGIME CLASSIFICATION
    # ================================================================
    st.markdown("---")
    st.subheader("Regime Classification")

    regime_signals = {}
    if '10Y Yield' in macro.columns and '3M Yield' in macro.columns:
        curve_val = macro['10Y Yield'].iloc[-1] - macro['3M Yield'].iloc[-1]
        regime_signals['Yield Curve'] = "Inverted ⚠️" if curve_val < 0 else ("Flat" if curve_val < 0.5 else "Normal ✅")
    if 'VIX' in macro.columns:
        v = macro['VIX'].iloc[-1]
        regime_signals['Vol Regime'] = "Crisis 🔴" if v > 30 else ("Elevated 🟡" if v > 20 else "Low Vol 🟢")
    if credit_momentum != 0:
        regime_signals['Credit'] = "Stress 🔴" if credit_momentum < -0.05 else ("Neutral 🟡" if credit_momentum < 0.02 else "Risk-On 🟢")
    if 'DXY' in macro.columns:
        dxy_ma = macro['DXY'].rolling(50).mean()
        regime_signals['Dollar Trend'] = "Strengthening 📈" if macro['DXY'].iloc[-1] > dxy_ma.iloc[-1] else "Weakening 📉"
    if 'Gold' in macro.columns:
        gold_ma = macro['Gold'].rolling(50).mean()
        regime_signals['Gold Trend'] = "Rising (Risk-Off) 📈" if macro['Gold'].iloc[-1] > gold_ma.iloc[-1] else "Falling (Risk-On) 📉"
    if 'SPY' in macro.columns:
        spy_ma200 = macro['SPY'].rolling(200).mean()
        spy_ma50 = macro['SPY'].rolling(50).mean()
        if not spy_ma200.empty and not spy_ma50.empty:
            spy_last = macro['SPY'].iloc[-1]
            if spy_last < spy_ma200.iloc[-1]:
                regime_signals['Equity Trend'] = "Below 200D MA 🔴"
            elif spy_ma50.iloc[-1] < spy_ma200.iloc[-1]:
                regime_signals['Equity Trend'] = "Death Cross 🟡"
            else:
                regime_signals['Equity Trend'] = "Above 200D MA 🟢"

    # Overall regime score
    risk_score = 0
    for k, v in regime_signals.items():
        if '🔴' in v or '⚠️' in v:
            risk_score += 2
        elif '🟡' in v:
            risk_score += 1
    overall = "RISK-OFF 🔴" if risk_score >= 5 else ("CAUTIOUS 🟡" if risk_score >= 3 else "RISK-ON 🟢")

    rc1, rc2 = st.columns([1, 2])
    with rc1:
        st.metric("Overall Regime", overall,
                   help="Composite signal from yield curve, VIX, credit, dollar, gold, and equity trend. Higher score = more defensive.")
        st.metric("Risk Score", f"{risk_score} / {len(regime_signals) * 2}",
                   help="Sum of all signal scores. Each signal contributes 0 (green), 1 (yellow), or 2 (red).")
        st.markdown("---")
        for k, v in regime_signals.items():
            st.caption(f"**{k}:** {v}")

    with rc2:
        # Yield curve history
        if '10Y Yield' in macro.columns and '3M Yield' in macro.columns:
            curve = macro['10Y Yield'] - macro['3M Yield']
            fig_yc = go.Figure()
            # Color segments: red when inverted, blue when normal
            fig_yc.add_trace(go.Scatter(
                x=curve.index, y=curve, fill='tozeroy', name='10Y-3M Spread',
                line=dict(color='#636EFA', width=1.5)
            ))
            fig_yc.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Inversion Line")
            fig_yc.add_hline(y=0.5, line_dash="dot", line_color="orange", annotation_text="Flat Zone")
            fig_yc.update_layout(
                title="Yield Curve History (10Y − 3M)",
                yaxis_title="Spread (%)", template="plotly_white"
            )
            st.plotly_chart(fig_yc, use_container_width=True)

    # ================================================================
    # FULL YIELD CURVE SNAPSHOT
    # ================================================================
    st.markdown("---")
    st.subheader("Yield Curve Snapshot")
    st.caption("Current yield curve shape across maturities vs 30 days ago.")

    yield_cols = {'3M Yield': 0.25, '5Y Yield': 5, '10Y Yield': 10}
    available_yields = {k: v for k, v in yield_cols.items() if k in macro.columns}

    if len(available_yields) >= 2:
        current_yields = {mat: latest[col] for col, mat in available_yields.items()}
        prev_30d_idx = max(0, len(macro) - 22)
        prev_30d = macro.iloc[prev_30d_idx]
        past_yields = {mat: prev_30d[col] for col, mat in available_yields.items()}

        fig_yc_snap = go.Figure()
        fig_yc_snap.add_trace(go.Scatter(
            x=list(current_yields.keys()), y=list(current_yields.values()),
            name='Current', mode='lines+markers', line=dict(color='#636EFA', width=3),
            marker=dict(size=10)
        ))
        fig_yc_snap.add_trace(go.Scatter(
            x=list(past_yields.keys()), y=list(past_yields.values()),
            name='30 Days Ago', mode='lines+markers', line=dict(color='gray', width=2, dash='dash'),
            marker=dict(size=8)
        ))
        fig_yc_snap.update_layout(
            title="Yield Curve Shape (Current vs 30D Ago)",
            xaxis_title="Maturity (Years)", yaxis_title="Yield (%)",
            template="plotly_white"
        )
        st.plotly_chart(fig_yc_snap, use_container_width=True)

    # ================================================================
    # VOLATILITY & RISK INDICATORS
    # ================================================================
    st.markdown("---")
    st.subheader("Volatility & Risk Indicators")
    vc1, vc2 = st.columns(2)

    with vc1:
        if 'VIX' in macro.columns:
            fig_vix = go.Figure()
            vix_s = macro['VIX'].dropna()
            # Rolling percentile bands
            vix_p25 = vix_s.rolling(252).quantile(0.25)
            vix_p75 = vix_s.rolling(252).quantile(0.75)
            fig_vix.add_trace(go.Scatter(
                x=vix_p75.index, y=vix_p75, name='75th Pctl (1Y)',
                line=dict(width=0), showlegend=False
            ))
            fig_vix.add_trace(go.Scatter(
                x=vix_p25.index, y=vix_p25, name='25th-75th Range',
                fill='tonexty', fillcolor='rgba(255,165,0,0.1)',
                line=dict(width=0)
            ))
            fig_vix.add_trace(go.Scatter(
                x=vix_s.index, y=vix_s, name='VIX',
                line=dict(color='orange', width=2)
            ))
            fig_vix.add_hline(y=20, line_dash="dash", line_color="yellow", annotation_text="Elevated")
            fig_vix.add_hline(y=30, line_dash="dash", line_color="red", annotation_text="Crisis")
            fig_vix.update_layout(title="VIX History with 1Y Percentile Range", template="plotly_white")
            st.plotly_chart(fig_vix, use_container_width=True)

    with vc2:
        if 'HY Corp Bond' in macro.columns and 'IG Corp Bond' in macro.columns:
            hy_ig = (macro['HY Corp Bond'] / macro['IG Corp Bond'])
            hy_ig = hy_ig / hy_ig.iloc[0]  # normalize
            hy_ig_ma = hy_ig.rolling(50).mean()
            fig_credit = go.Figure()
            fig_credit.add_trace(go.Scatter(
                x=hy_ig.index, y=hy_ig, name='HY/IG Ratio',
                line=dict(color='#EF553B', width=2)
            ))
            fig_credit.add_trace(go.Scatter(
                x=hy_ig_ma.index, y=hy_ig_ma, name='50D MA',
                line=dict(color='gray', dash='dash', width=1)
            ))
            fig_credit.update_layout(
                title="Credit Risk Appetite (HY/IG Ratio)", template="plotly_white",
                yaxis_title="Ratio (normalized)"
            )
            st.plotly_chart(fig_credit, use_container_width=True)

    # ================================================================
    # DOLLAR & SAFE HAVENS
    # ================================================================
    st.markdown("---")
    st.subheader("Dollar & Safe Haven Flows")
    dc1, dc2 = st.columns(2)

    with dc1:
        if 'DXY' in macro.columns:
            fig_dxy = go.Figure()
            dxy_s = macro['DXY'].dropna()
            ma50 = dxy_s.rolling(50).mean()
            ma200 = dxy_s.rolling(200).mean()
            fig_dxy.add_trace(go.Scatter(
                x=dxy_s.index, y=dxy_s, name='DXY',
                line=dict(color='#636EFA', width=2)
            ))
            fig_dxy.add_trace(go.Scatter(
                x=ma50.index, y=ma50, name='50D MA',
                line=dict(dash='dash', color='orange', width=1)
            ))
            fig_dxy.add_trace(go.Scatter(
                x=ma200.index, y=ma200, name='200D MA',
                line=dict(dash='dot', color='gray', width=1)
            ))
            fig_dxy.update_layout(title="US Dollar Index (DXY)", template="plotly_white")
            st.plotly_chart(fig_dxy, use_container_width=True)

    with dc2:
        if 'Gold' in macro.columns:
            fig_gold = go.Figure()
            gs = macro['Gold'].dropna()
            gold_ma50 = gs.rolling(50).mean()
            fig_gold.add_trace(go.Scatter(
                x=gs.index, y=gs, name='Gold',
                line=dict(color='goldenrod', width=2)
            ))
            fig_gold.add_trace(go.Scatter(
                x=gold_ma50.index, y=gold_ma50, name='50D MA',
                line=dict(dash='dash', color='gray', width=1)
            ))
            fig_gold.update_layout(
                title="Gold ($/oz)", yaxis_tickformat="$,.0f",
                template="plotly_white"
            )
            st.plotly_chart(fig_gold, use_container_width=True)

    # ================================================================
    # TREASURY & EQUITY
    # ================================================================
    st.markdown("---")
    st.subheader("Treasury & Equity Context")
    te1, te2 = st.columns(2)

    with te1:
        if '20Y Treasury' in macro.columns:
            fig_tlt = go.Figure()
            tlt_s = macro['20Y Treasury'].dropna()
            tlt_ma = tlt_s.rolling(50).mean()
            fig_tlt.add_trace(go.Scatter(
                x=tlt_s.index, y=tlt_s, name='TLT (20Y Treasury ETF)',
                line=dict(color='#AB63FA', width=2)
            ))
            fig_tlt.add_trace(go.Scatter(
                x=tlt_ma.index, y=tlt_ma, name='50D MA',
                line=dict(dash='dash', color='gray', width=1)
            ))
            fig_tlt.update_layout(
                title="Long-Duration Treasury (TLT)",
                yaxis_tickformat="$,.2f", template="plotly_white"
            )
            st.plotly_chart(fig_tlt, use_container_width=True)

    with te2:
        if 'SPY' in macro.columns:
            fig_spy = go.Figure()
            spy_s = macro['SPY'].dropna()
            spy_ma50 = spy_s.rolling(50).mean()
            spy_ma200 = spy_s.rolling(200).mean()
            fig_spy.add_trace(go.Scatter(
                x=spy_s.index, y=spy_s, name='SPY',
                line=dict(color='#00CC96', width=2)
            ))
            fig_spy.add_trace(go.Scatter(
                x=spy_ma50.index, y=spy_ma50, name='50D MA',
                line=dict(dash='dash', color='orange', width=1)
            ))
            fig_spy.add_trace(go.Scatter(
                x=spy_ma200.index, y=spy_ma200, name='200D MA',
                line=dict(dash='dot', color='red', width=1)
            ))
            fig_spy.update_layout(
                title="S&P 500 (SPY) with Moving Averages",
                yaxis_tickformat="$,.2f", template="plotly_white"
            )
            st.plotly_chart(fig_spy, use_container_width=True)

    # ================================================================
    # CROSS-ASSET CORRELATION
    # ================================================================
    st.markdown("---")
    st.subheader("Cross-Asset Correlation (60D Rolling)")
    st.caption("Shows how macro assets are moving relative to each other over the trailing 60 days.")

    corr_assets = ['VIX', 'SPY', 'DXY', 'Gold', '20Y Treasury', 'HY Corp Bond']
    available_corr = [a for a in corr_assets if a in macro.columns]

    if len(available_corr) >= 3:
        corr_returns = macro[available_corr].pct_change().dropna()
        corr_60d = corr_returns.tail(60).corr()

        fig_corr = px.imshow(
            corr_60d, text_auto='.2f',
            color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
            title="60-Day Return Correlation Matrix"
        )
        fig_corr.update_layout(template="plotly_white")
        st.plotly_chart(fig_corr, use_container_width=True)

    # ================================================================
    # HISTORICAL REGIME TIMELINE
    # ================================================================
    st.markdown("---")
    st.subheader("Historical Regime Timeline")
    st.caption("Rolling regime classification based on the same scoring system, applied historically.")

    if 'VIX' in macro.columns and '10Y Yield' in macro.columns and '3M Yield' in macro.columns:
        # Compute rolling regime scores
        regime_history = pd.DataFrame(index=macro.index)

        # Yield curve signal
        curve_hist = macro['10Y Yield'] - macro['3M Yield']
        regime_history['curve_score'] = np.where(curve_hist < 0, 2, np.where(curve_hist < 0.5, 1, 0))

        # VIX signal
        regime_history['vix_score'] = np.where(macro['VIX'] > 30, 2, np.where(macro['VIX'] > 20, 1, 0))

        # Credit signal (rolling 20D)
        if 'HY Corp Bond' in macro.columns and 'IG Corp Bond' in macro.columns:
            hy_roll = macro['HY Corp Bond'].pct_change().rolling(20).mean() * 252
            ig_roll = macro['IG Corp Bond'].pct_change().rolling(20).mean() * 252
            credit_roll = hy_roll - ig_roll
            regime_history['credit_score'] = np.where(credit_roll < -0.05, 2, np.where(credit_roll < 0.02, 1, 0))
        else:
            regime_history['credit_score'] = 0

        # DXY signal
        if 'DXY' in macro.columns:
            dxy_above_ma = macro['DXY'] > macro['DXY'].rolling(50).mean()
            regime_history['dxy_score'] = np.where(dxy_above_ma, 1, 0)
        else:
            regime_history['dxy_score'] = 0

        regime_history['total_score'] = regime_history.sum(axis=1)
        regime_history['regime'] = np.where(
            regime_history['total_score'] >= 5, 'Risk-Off',
            np.where(regime_history['total_score'] >= 3, 'Cautious', 'Risk-On')
        )

        # Color map
        color_map = {'Risk-On': '#00CC96', 'Cautious': '#FFA15A', 'Risk-Off': '#EF553B'}

        fig_regime = go.Figure()

        # Score line
        fig_regime.add_trace(go.Scatter(
            x=regime_history.index, y=regime_history['total_score'],
            name='Regime Score', line=dict(color='#636EFA', width=2),
            fill='tozeroy', fillcolor='rgba(99,110,250,0.1)'
        ))
        fig_regime.add_hline(y=5, line_dash="dash", line_color="red", annotation_text="Risk-Off Threshold")
        fig_regime.add_hline(y=3, line_dash="dash", line_color="orange", annotation_text="Cautious Threshold")

        fig_regime.update_layout(
            title="Historical Regime Score",
            yaxis_title="Composite Risk Score",
            template="plotly_white"
        )
        st.plotly_chart(fig_regime, use_container_width=True)

        # Regime distribution
        regime_counts = regime_history['regime'].value_counts()
        fig_rdist = go.Figure()
        fig_rdist.add_trace(go.Pie(
            labels=regime_counts.index.tolist(),
            values=regime_counts.values.tolist(),
            marker_colors=[color_map.get(r, 'gray') for r in regime_counts.index],
            hole=0.4, textinfo='label+percent'
        ))
        fig_rdist.update_layout(title="Time Spent in Each Regime (~3 Years)", template="plotly_white")

        # SPY returns by regime
        if 'SPY' in macro.columns:
            spy_ret = macro['SPY'].pct_change()
            regime_history['spy_ret'] = spy_ret

            regime_perf = regime_history.groupby('regime')['spy_ret'].agg(
                ['mean', 'std', 'count']
            ).reset_index()
            regime_perf['ann_return'] = regime_perf['mean'] * 252
            regime_perf['ann_vol'] = regime_perf['std'] * np.sqrt(252)
            regime_perf.columns = ['Regime', 'Daily Mean', 'Daily Std', 'Days', 'Ann. Return', 'Ann. Vol']

            rd1, rd2 = st.columns(2)
            with rd1:
                st.plotly_chart(fig_rdist, use_container_width=True)
            with rd2:
                st.subheader("SPY Performance by Regime")
                st.dataframe(regime_perf[['Regime', 'Days', 'Ann. Return', 'Ann. Vol']].style.format({
                    'Ann. Return': '{:.2%}', 'Ann. Vol': '{:.2%}'
                }), use_container_width=True)

                fig_regime_ret = go.Figure()
                fig_regime_ret.add_trace(go.Bar(
                    x=regime_perf['Regime'], y=regime_perf['Ann. Return'],
                    marker_color=[color_map.get(r, 'gray') for r in regime_perf['Regime']]
                ))
                fig_regime_ret.update_layout(
                    title="Annualized SPY Return by Regime",
                    yaxis_tickformat=".2%", template="plotly_white"
                )
                st.plotly_chart(fig_regime_ret, use_container_width=True)

    # ================================================================
    # INTER-MARKET DIVERGENCES
    # ================================================================
    st.markdown("---")
    st.subheader("Inter-Market Divergence Monitor")
    st.caption("Tracks when risk indicators disagree — divergences often precede regime shifts.")

    divergences = []

    # Stock/bond divergence: SPY up but TLT up too = confusion
    if 'SPY' in macro.columns and '20Y Treasury' in macro.columns:
        spy_20d = macro['SPY'].pct_change(20).iloc[-1]
        tlt_20d = macro['20Y Treasury'].pct_change(20).iloc[-1]
        if spy_20d > 0.02 and tlt_20d > 0.02:
            divergences.append({
                'Pair': 'SPY vs TLT', 'Signal': 'Both Rising ⚠️',
                'Detail': f"SPY +{spy_20d:.1%}, TLT +{tlt_20d:.1%} over 20D. Unusual — suggests mixed risk signals.",
                'Severity': 'Medium'
            })
        elif spy_20d < -0.02 and tlt_20d < -0.02:
            divergences.append({
                'Pair': 'SPY vs TLT', 'Signal': 'Both Falling 🔴',
                'Detail': f"SPY {spy_20d:.1%}, TLT {tlt_20d:.1%}. Correlated sell-off — potential liquidity crisis.",
                'Severity': 'High'
            })

    # VIX vs SPY: VIX elevated but SPY rising = complacency
    if 'VIX' in macro.columns and 'SPY' in macro.columns:
        vix_current = macro['VIX'].iloc[-1]
        spy_20d = macro['SPY'].pct_change(20).iloc[-1]
        if vix_current > 22 and spy_20d > 0.03:
            divergences.append({
                'Pair': 'VIX vs SPY', 'Signal': 'VIX Elevated + SPY Rising ⚠️',
                'Detail': f"VIX at {vix_current:.1f} while SPY up {spy_20d:.1%}. Markets may be complacent.",
                'Severity': 'Medium'
            })

    # Dollar vs Gold: both rising = conflicting safe-haven signals
    if 'DXY' in macro.columns and 'Gold' in macro.columns:
        dxy_20d = macro['DXY'].pct_change(20).iloc[-1]
        gold_20d = macro['Gold'].pct_change(20).iloc[-1]
        if dxy_20d > 0.01 and gold_20d > 0.02:
            divergences.append({
                'Pair': 'DXY vs Gold', 'Signal': 'Both Rising ⚠️',
                'Detail': f"Dollar +{dxy_20d:.1%}, Gold +{gold_20d:.1%}. Conflicting safe-haven flows.",
                'Severity': 'Medium'
            })

    if divergences:
        div_df = pd.DataFrame(divergences)
        st.dataframe(div_df, use_container_width=True)
    else:
        st.success("No significant inter-market divergences detected. Markets are behaving consistently with the current regime.")


# ==========================================
# Page 7: Strategy Engine — Trade Ideas (Full)
# ==========================================

def show_strategy_page(portfolio_df, end_date, rf):
    st.title("🧠 Strategy Engine — Trade Ideas")
    st.markdown("---")
    st.caption("Aggregates signals from macro regime, volatility surface, factor exposure, news sentiment, and technicals to generate actionable, sized trade ideas.")

    tickers = portfolio_df['Ticker'].tolist()
    if not tickers:
        st.info("👈 Add holdings to generate ideas.")
        return

    # Estimate portfolio value
    port_ret, _, _, cur_dv, _ = build_portfolio_returns(portfolio_df, end_date)
    port_val = cur_dv.sum() if cur_dv is not None else 0
    if port_val <= 0:
        st.warning("Could not compute portfolio value.")
        return

    # Settings
    st.sidebar.markdown("---")
    st.sidebar.header("🧠 Strategy Settings")
    risk_budget = st.sidebar.slider("Max Risk per Trade (% of Portfolio)", 0.5, 5.0, 2.0, 0.5) / 100

    with st.spinner("Scanning all signals across your portfolio..."):
        signals = compute_all_signals(tickers, port_val, rf)
        ideas = generate_trade_ideas(signals, portfolio_df, rf)

    # --- Signal Summary ---
    st.subheader("Signal Dashboard")
    regime = signals['macro'].get('regime', 'unknown')
    regime_display = {"risk_on": "RISK-ON 🟢", "cautious": "CAUTIOUS 🟡", "risk_off": "RISK-OFF 🔴"}.get(regime, "UNKNOWN")
    mkt_news = signals.get('news', {}).get('market', {})

    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
    sc1.metric("Macro Regime", regime_display,
               help="Composite of yield curve, VIX, and credit signals.")
    sc2.metric("VIX", f"{signals['macro'].get('vix', 0):.1f}",
               help=f"Regime: {signals['macro'].get('vix_regime', 'N/A')}")
    sc3.metric("Market News", mkt_news.get('label', 'N/A'),
               help=f"Avg sentiment: {mkt_news.get('mean', 0):+.2f} from {mkt_news.get('count', 0)} articles")
    sc4.metric("Portfolio Value", f"${port_val:,.0f}")
    sc5.metric("Ideas Generated", f"{len(ideas)}")

    # Per-ticker signal summary
    st.subheader("Per-Ticker Signal Summary")
    tk_summary = []
    for tk, ts in signals['ticker_signals'].items():
        ns_data = ts.get('news_sentiment', {})
        tk_summary.append({
            'Ticker': tk,
            'Spot': ts.get('spot', 0),
            'ATM IV': ts.get('atm_iv', 0),
            '20D RV': ts.get('rv_20', 0),
            'IV-RV Gap': ts.get('iv_rv_gap', 0),
            'IV Rank': ts.get('iv_rank', 0),
            'RSI': ts.get('rsi', 50),
            'Z-Score': ts.get('zscore', 0),
            'Skew': ts.get('skew', 0),
            'News': ns_data.get('label', 'N/A'),
            'Sentiment': ns_data.get('mean', 0),
            'Articles': ns_data.get('count', 0)
        })
    if tk_summary:
        tkdf = pd.DataFrame(tk_summary)
        st.dataframe(tkdf.style.format({
            'Spot': '${:.2f}', 'ATM IV': '{:.1%}', '20D RV': '{:.1%}', 'IV-RV Gap': '{:.1%}',
            'IV Rank': '{:.0%}', 'RSI': '{:.0f}', 'Z-Score': '{:.2f}', 'Skew': '{:.1%}',
            'Sentiment': '{:+.2f}'
        }), use_container_width=True)

    # --- No ideas guard ---
    st.markdown("---")
    if not ideas:
        st.info("No actionable ideas based on current signals. This means your portfolio is well-positioned for the current environment.")
        return

    # ================================================================
    # TOP RECOMMENDATION
    # ================================================================

    def score_idea(idea):
        urgency_score = {'High': 3, 'Medium': 2, 'Low': 1}.get(idea['urgency'], 0)
        rr = idea.get('risk_reward', 0)
        rr_score = min(rr, 5) / 5 if rr > 0 else 0.1
        risk_penalty = 0 if idea['max_loss'] == float('inf') else 1
        cost_eff = 1 - min(abs(idea['cost']) / port_val, 1) if port_val > 0 else 0.5
        return (urgency_score * 2) + (rr_score * 3) + (risk_penalty * 1.5) + (cost_eff * 0.5)

    scored_ideas = [(score_idea(i), i) for i in ideas]
    scored_ideas.sort(key=lambda x: x[0], reverse=True)
    top_idea = scored_ideas[0][1]

    st.subheader("🏆 Top Recommendation")
    with st.container():
        tc1, tc2, tc3, tc4, tc5 = st.columns(5)
        tc1.metric("Ticker", top_idea['ticker'])
        tc2.metric("Strategy", top_idea['strategy'])
        tc3.metric("Direction", top_idea['direction'])
        cost_label = "Credit" if top_idea['cost'] < 0 else "Cost"
        tc4.metric(cost_label, f"${abs(top_idea['cost']):,.2f}")
        if top_idea.get('risk_reward', 0) > 0:
            tc5.metric("Risk/Reward", f"{top_idea['risk_reward']:.1f}x")
        elif top_idea['max_gain'] == float('inf'):
            tc5.metric("Upside", "Unlimited")
        else:
            tc5.metric("Max Gain", f"${top_idea['max_gain']:,.2f}" if top_idea['max_gain'] != float('inf') else "∞")

        # Legs table
        top_legs = pd.DataFrame(top_idea['legs'])
        top_legs['total'] = top_legs['price'] * 100 * top_legs['contracts']
        top_legs.columns = ['Action', 'Strike', 'Price/Share', 'Contracts', 'Total $']
        st.dataframe(top_legs.style.format({
            'Strike': '${:.2f}', 'Price/Share': '${:.2f}', 'Total $': '${:,.2f}'
        }), use_container_width=True)

        # Risk metrics
        tr1, tr2, tr3, tr4 = st.columns(4)
        tr1.metric("Max Loss",
                    f"${top_idea['max_loss']:,.2f}" if top_idea['max_loss'] != float('inf') else "∞")
        tr2.metric("Max Gain",
                    f"${top_idea['max_gain']:,.2f}" if top_idea['max_gain'] != float('inf') else "∞")
        tr3.metric("Risk % of Portfolio",
                    f"{abs(top_idea['max_loss']) / port_val:.2%}" if top_idea['max_loss'] != float('inf') else "N/A")
        tr4.metric("Expiry", top_idea['expiry'])

        st.markdown(f"**Why this trade:** {top_idea['rationale']}")

        # Top recommendation payoff diagram
        def _render_payoff(idea, title_prefix=""):
            """Render a payoff diagram for any idea (multi-leg or single-leg)."""
            if len(idea['legs']) >= 2:
                all_strikes = [l['strike'] for l in idea['legs']]
                spot_range = np.linspace(min(all_strikes) * 0.92, max(all_strikes) * 1.08, 150)
                pnl_at_exp = np.zeros_like(spot_range)
                for leg in idea['legs']:
                    k, p, n = leg['strike'], leg['price'], leg['contracts']
                    if 'Buy Call' in leg['type']:
                        pnl_at_exp += (np.maximum(spot_range - k, 0) - p) * 100 * n
                    elif 'Sell Call' in leg['type']:
                        pnl_at_exp += (p - np.maximum(spot_range - k, 0)) * 100 * n
                    elif 'Buy Put' in leg['type']:
                        pnl_at_exp += (np.maximum(k - spot_range, 0) - p) * 100 * n
                    elif 'Sell Put' in leg['type']:
                        pnl_at_exp += (p - np.maximum(k - spot_range, 0)) * 100 * n

                # Breakevens (OUTSIDE leg loop)
                sign_changes = np.where(np.diff(np.sign(pnl_at_exp)))[0]
                breakevens = []
                for sc_idx in sign_changes:
                    x1, x2 = spot_range[sc_idx], spot_range[sc_idx + 1]
                    y1, y2 = pnl_at_exp[sc_idx], pnl_at_exp[sc_idx + 1]
                    if y2 != y1:
                        breakevens.append(x1 - y1 * (x2 - x1) / (y2 - y1))

                fig = go.Figure()
                profit_y = np.where(pnl_at_exp >= 0, pnl_at_exp, 0)
                loss_y = np.where(pnl_at_exp < 0, pnl_at_exp, 0)
                fig.add_trace(go.Scatter(
                    x=spot_range, y=profit_y, fill='tozeroy',
                    line=dict(color='#00CC96', width=0), fillcolor='rgba(0,204,150,0.3)',
                    name='Profit Zone'))
                fig.add_trace(go.Scatter(
                    x=spot_range, y=loss_y, fill='tozeroy',
                    line=dict(color='#EF553B', width=0), fillcolor='rgba(239,85,59,0.3)',
                    name='Loss Zone'))
                fig.add_trace(go.Scatter(
                    x=spot_range, y=pnl_at_exp,
                    line=dict(color='white', width=2), name='P&L', showlegend=False))
                fig.add_hline(y=0, line_dash="dash", line_color="gray")

                for s in all_strikes:
                    fig.add_vline(x=s, line_dash="dot", line_color="orange",
                                  annotation_text=f"${s:.0f}")
                for be in breakevens:
                    fig.add_vline(x=be, line_dash="dash", line_color="blue",
                                  annotation_text=f"BE: ${be:.2f}")

                max_pnl = pnl_at_exp.max()
                min_pnl = pnl_at_exp.min()
                fig.add_annotation(x=spot_range[np.argmax(pnl_at_exp)], y=max_pnl,
                                    text=f"Max Profit: ${max_pnl:,.0f}", showarrow=True,
                                    arrowhead=2, font=dict(color='green'))
                fig.add_annotation(x=spot_range[np.argmin(pnl_at_exp)], y=min_pnl,
                                    text=f"Max Loss: ${min_pnl:,.0f}", showarrow=True,
                                    arrowhead=2, font=dict(color='red'))
                fig.update_layout(
                    title=f"{title_prefix}Payoff at Expiry",
                    xaxis_title="Underlying Price", yaxis_title="P&L ($)",
                    yaxis_tickformat="$,.0f", xaxis_tickformat="$,.0f",
                    template="plotly_white", height=450)
                st.plotly_chart(fig, use_container_width=True)

                if breakevens:
                    st.caption(f"**Breakeven(s):** {', '.join([f'${be:,.2f}' for be in breakevens])}")

            elif len(idea['legs']) == 1:
                leg = idea['legs'][0]
                k, p, n = leg['strike'], leg['price'], leg['contracts']
                spot_range = np.linspace(k * 0.80, k * 1.20, 100)
                pnl_at_exp = np.zeros_like(spot_range)
                if 'Buy Call' in leg['type']:
                    pnl_at_exp = (np.maximum(spot_range - k, 0) - p) * 100 * n
                elif 'Sell Call' in leg['type']:
                    pnl_at_exp = (p - np.maximum(spot_range - k, 0)) * 100 * n
                elif 'Buy Put' in leg['type']:
                    pnl_at_exp = (np.maximum(k - spot_range, 0) - p) * 100 * n
                elif 'Sell Put' in leg['type']:
                    pnl_at_exp = (p - np.maximum(k - spot_range, 0)) * 100 * n

                fig = go.Figure()
                profit_y = np.where(pnl_at_exp >= 0, pnl_at_exp, 0)
                loss_y = np.where(pnl_at_exp < 0, pnl_at_exp, 0)
                fig.add_trace(go.Scatter(x=spot_range, y=profit_y, fill='tozeroy',
                    line=dict(color='#00CC96', width=0), fillcolor='rgba(0,204,150,0.3)', name='Profit'))
                fig.add_trace(go.Scatter(x=spot_range, y=loss_y, fill='tozeroy',
                    line=dict(color='#EF553B', width=0), fillcolor='rgba(239,85,59,0.3)', name='Loss'))
                fig.add_trace(go.Scatter(x=spot_range, y=pnl_at_exp,
                    line=dict(color='white', width=2), showlegend=False))
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                fig.add_vline(x=k, line_dash="dot", line_color="orange",
                              annotation_text=f"Strike: ${k:.0f}")
                fig.update_layout(
                    title=f"{title_prefix}Payoff at Expiry — {leg['type']}",
                    xaxis_title="Underlying Price", yaxis_title="P&L ($)",
                    yaxis_tickformat="$,.0f", xaxis_tickformat="$,.0f",
                    template="plotly_white", height=450)
                st.plotly_chart(fig, use_container_width=True)

        _render_payoff(top_idea, title_prefix="🏆 Top Recommendation — ")

    # ================================================================
    # ALL TRADE IDEAS
    # ================================================================
    st.markdown("---")
    st.subheader(f"All Trade Ideas ({len(ideas)})")

    # Filters
    fc1, fc2, fc3 = st.columns(3)
    urgency_filter = fc1.multiselect("Urgency", ["High", "Medium", "Low"], default=["High", "Medium"])
    direction_filter = fc2.multiselect("Direction",
                                        list(set(i['direction'] for i in ideas)),
                                        default=list(set(i['direction'] for i in ideas)))
    strategy_filter = fc3.multiselect("Strategy",
                                       list(set(i['strategy'] for i in ideas)),
                                       default=list(set(i['strategy'] for i in ideas)))

    filtered = [i for i in ideas
                if i['urgency'] in urgency_filter
                and i['direction'] in direction_filter
                and i['strategy'] in strategy_filter]

    for idx, idea in enumerate(filtered):
        urgency_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(idea['urgency'], "⚪")
        with st.expander(
            f"{urgency_color} {idea['ticker']} — {idea['strategy']} ({idea['direction']})",
            expanded=(idea['urgency'] == 'High')
        ):
            # Header metrics
            ic1, ic2, ic3, ic4 = st.columns(4)
            ic1.metric("Strategy", idea['strategy'])
            ic2.metric("Urgency", idea['urgency'])
            cost_label = "Credit" if idea['cost'] < 0 else "Cost"
            ic3.metric(cost_label, f"${abs(idea['cost']):,.2f}")
            if idea.get('risk_reward', 0) > 0:
                ic4.metric("Risk/Reward", f"{idea['risk_reward']:.1f}x")
            elif idea['max_gain'] == float('inf'):
                ic4.metric("Risk/Reward", "Unlimited upside")
            else:
                ic4.metric("Max Gain", f"${idea['max_gain']:,.2f}" if idea['max_gain'] != float('inf') else "∞")

            # Legs table
            st.markdown("**Trade Structure:**")
            legs_df = pd.DataFrame(idea['legs'])
            legs_df['total'] = legs_df['price'] * 100 * legs_df['contracts']
            legs_df.columns = ['Action', 'Strike', 'Price/Share', 'Contracts', 'Total $']
            st.dataframe(legs_df.style.format({
                'Strike': '${:.2f}', 'Price/Share': '${:.2f}', 'Total $': '${:,.2f}'
            }), use_container_width=True)

            st.markdown(f"**Expiry:** {idea['expiry']}")

            # Risk summary
            rc1, rc2, rc3 = st.columns(3)
            rc1.metric("Max Loss",
                        f"${idea['max_loss']:,.2f}" if idea['max_loss'] != float('inf') else "∞",
                        help="Worst case scenario.")
            rc2.metric("Max Gain",
                        f"${idea['max_gain']:,.2f}" if idea['max_gain'] != float('inf') else "∞")
            rc3.metric("Risk as % of Portfolio",
                        f"{abs(idea['max_loss']) / port_val:.2%}" if idea['max_loss'] != float('inf') else "N/A")

            # Rationale
            st.markdown(f"**Rationale:** {idea['rationale']}")

            # Payoff diagram (reuse the shared function)
            _render_payoff(idea)

    # ================================================================
    # SUMMARY & ANALYTICS
    # ================================================================
    st.markdown("---")
    st.subheader("Ideas Summary")
    summary = pd.DataFrame([{
        'Ticker': i['ticker'],
        'Strategy': i['strategy'],
        'Direction': i['direction'],
        'Urgency': i['urgency'],
        'Cost/Credit': i['cost'],
        'Max Loss': i['max_loss'] if i['max_loss'] != float('inf') else np.nan,
        'Max Gain': i['max_gain'] if i['max_gain'] != float('inf') else np.nan,
        'R/R': i.get('risk_reward', 0)
    } for i in filtered])

    if not summary.empty:
        st.dataframe(summary.style.format({
            'Cost/Credit': '${:,.0f}',
            'Max Loss': '${:,.0f}',
            'Max Gain': '${:,.0f}',
            'R/R': '{:.1f}x'
        }), use_container_width=True)

    # Capital & Risk Summary
    total_debit = sum(i['cost'] for i in filtered if i['cost'] > 0)
    total_credit = sum(abs(i['cost']) for i in filtered if i['cost'] < 0)
    total_max_risk = sum(i['max_loss'] for i in filtered if i['max_loss'] != float('inf'))

    st.subheader("Capital & Risk Summary")
    fc1, fc2, fc3, fc4 = st.columns(4)
    fc1.metric("Total Capital Required", f"${total_debit:,.0f}",
               help="Sum of all debit trades.")
    fc2.metric("Total Credit Received", f"${total_credit:,.0f}",
               help="Sum of all credit trades.")
    fc3.metric("Total Max Risk", f"${total_max_risk:,.0f}",
               help=f"That's {total_max_risk / port_val:.1%} of your portfolio." if port_val > 0 else "")
    fc4.metric("Net Capital Deployment", f"${total_debit - total_credit:,.0f}",
               help="Debit minus credit — net cash outflow if all trades execute.")

    # Risk Budget Visualization
    if total_max_risk > 0:
        st.subheader("Risk Budget Allocation")
        risk_by_strategy = {}
        for i in filtered:
            strat = i['strategy']
            ml = i['max_loss'] if i['max_loss'] != float('inf') else 0
            risk_by_strategy[strat] = risk_by_strategy.get(strat, 0) + ml

        if risk_by_strategy:
            fig_risk = go.Figure()
            fig_risk.add_trace(go.Pie(
                labels=list(risk_by_strategy.keys()),
                values=list(risk_by_strategy.values()),
                hole=0.4, textinfo='label+percent'
            ))
            fig_risk.update_layout(title="Max Risk by Strategy Type", template="plotly_white")
            st.plotly_chart(fig_risk, use_container_width=True)

    # Ideas by Ticker
    st.subheader("Ideas by Ticker")
    ideas_by_ticker = {}
    for i in filtered:
        tk = i['ticker']
        if tk not in ideas_by_ticker:
            ideas_by_ticker[tk] = []
        ideas_by_ticker[tk].append(i)

    ticker_summary = []
    for tk, tk_ideas in ideas_by_ticker.items():
        total_cost = sum(i['cost'] for i in tk_ideas)
        total_risk = sum(i['max_loss'] for i in tk_ideas if i['max_loss'] != float('inf'))
        ticker_summary.append({
            'Ticker': tk,
            'Ideas': len(tk_ideas),
            'Strategies': ", ".join(set(i['strategy'] for i in tk_ideas)),
            'Net Cost/Credit': total_cost,
            'Total Max Risk': total_risk,
            'Risk % of Portfolio': total_risk / port_val if port_val > 0 else 0
        })

    if ticker_summary:
        ts_df = pd.DataFrame(ticker_summary)
        st.dataframe(ts_df.style.format({
            'Net Cost/Credit': '${:,.0f}',
            'Total Max Risk': '${:,.0f}',
            'Risk % of Portfolio': '{:.2%}'
        }), use_container_width=True)

    # Urgency Distribution
    st.subheader("Urgency Distribution")
    urg_counts = {'High': 0, 'Medium': 0, 'Low': 0}
    for i in filtered:
        urg_counts[i['urgency']] = urg_counts.get(i['urgency'], 0) + 1

    fig_urg = go.Figure()
    fig_urg.add_trace(go.Bar(
        x=list(urg_counts.keys()),
        y=list(urg_counts.values()),
        marker_color=['#EF553B', '#FFA15A', '#00CC96']
    ))
    fig_urg.update_layout(title="Trade Ideas by Urgency", yaxis_title="Count", template="plotly_white")
    st.plotly_chart(fig_urg, use_container_width=True)

    # Direction Distribution
    dir_counts = {}
    for i in filtered:
        d = i['direction']
        dir_counts[d] = dir_counts.get(d, 0) + 1

    fig_dir = go.Figure()
    fig_dir.add_trace(go.Pie(
        labels=list(dir_counts.keys()),
        values=list(dir_counts.values()),
        hole=0.4, textinfo='label+percent'
    ))
    fig_dir.update_layout(title="Ideas by Direction", template="plotly_white")
    st.plotly_chart(fig_dir, use_container_width=True)
# ==========================================
# Page 8: News & Sentiment
# ==========================================

def show_news_page(portfolio_df, rf):
    st.title("📰 News & Sentiment Monitor"); st.markdown("---")
    tickers = portfolio_df['Ticker'].tolist()
    if not tickers: st.info("👈 Add tickers."); return

    # Dependency check
    dep_msgs = []
    if not HAS_VADER: dep_msgs.append("`pip install vaderSentiment` for better NLP sentiment (using keyword lexicon fallback)")
    if not HAS_FEEDPARSER: dep_msgs.append("`pip install feedparser` for Google News & RSS feeds")
    if dep_msgs:
        st.info("**Optional dependencies:** " + " | ".join(dep_msgs))

    tab_portfolio, tab_ticker, tab_market = st.tabs(["Portfolio Sentiment", "Ticker Deep Dive", "Market News"])

    # --- Tab 1: Portfolio-wide sentiment heatmap ---
    with tab_portfolio:
        st.subheader("Portfolio News Sentiment Heatmap")
        st.caption("Aggregated sentiment from multiple sources for each holding.")

        with st.spinner("Fetching news for all tickers..."):
            all_sentiments = {}
            for tk in tickers:
                articles = fetch_ticker_news(tk)
                all_sentiments[tk] = {'articles': articles, 'agg': aggregate_sentiment(articles)}

        # Heatmap data
        hm_data = []
        for tk, data in all_sentiments.items():
            agg = data['agg']
            hm_data.append({'Ticker': tk, 'Avg Sentiment': agg['mean'], 'Articles': agg['count'],
                            'Bullish %': agg['bullish_pct'], 'Bearish %': agg['bearish_pct'], 'Label': agg['label']})

        hm_df = pd.DataFrame(hm_data)
        if not hm_df.empty:
            # Sentiment bar chart
            colors = [sentiment_color(s) for s in hm_df['Avg Sentiment']]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=hm_df['Ticker'], y=hm_df['Avg Sentiment'], marker_color=colors, text=hm_df['Label'], textposition='outside'))
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.update_layout(title="News Sentiment by Ticker", yaxis_title="Sentiment Score (-1 to +1)",
                               yaxis_range=[-1, 1], template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

            # Summary table
            st.dataframe(hm_df.style.format({
                'Avg Sentiment': '{:+.3f}', 'Bullish %': '{:.0%}', 'Bearish %': '{:.0%}'
            }).applymap(lambda v: f'background-color: {sentiment_color(v)}22' if isinstance(v, (int, float)) and -1 <= v <= 1 else '', subset=['Avg Sentiment']),
                use_container_width=True)

            # Bull/Bear breakdown
            fig_bb = go.Figure()
            fig_bb.add_trace(go.Bar(x=hm_df['Ticker'], y=hm_df['Bullish %'], name='Bullish', marker_color='#00CC96'))
            fig_bb.add_trace(go.Bar(x=hm_df['Ticker'], y=-hm_df['Bearish %'], name='Bearish', marker_color='#EF553B'))
            fig_bb.update_layout(title="Bullish vs Bearish Article Ratio", barmode='relative', yaxis_tickformat=".0%", template="plotly_white")
            st.plotly_chart(fig_bb, use_container_width=True)

    # --- Tab 2: Ticker deep dive ---
    with tab_ticker:
        st.subheader("Ticker News Deep Dive")
        sel_tk = st.selectbox("Select Ticker", tickers, key="news_tk")
        articles = fetch_ticker_news(sel_tk)

        if not articles:
            st.warning(f"No news found for {sel_tk}.")
        else:
            agg = aggregate_sentiment(articles)

            # Summary metrics
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Avg Sentiment", f"{agg['mean']:+.3f}")
            c2.metric("Label", agg['label'])
            c3.metric("Articles", f"{agg['count']}")
            c4.metric("Bullish", f"{agg['bullish_pct']:.0%}")
            c5.metric("Bearish", f"{agg['bearish_pct']:.0%}")

            # Sentiment distribution
            scores = [a['sentiment'] for a in articles]
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(x=scores, nbinsx=20, marker_color='#636EFA'))
            fig_dist.add_vline(x=0, line_dash="dash", line_color="gray")
            fig_dist.add_vline(x=agg['mean'], line_dash="solid", line_color="red", annotation_text=f"Avg: {agg['mean']:+.2f}")
            fig_dist.update_layout(title="Headline Sentiment Distribution", xaxis_title="Sentiment (-1 to +1)", template="plotly_white")
            st.plotly_chart(fig_dist, use_container_width=True)

            # Article feed
            st.subheader(f"Recent Headlines ({len(articles)})")
            source_filter = st.multiselect("Filter by Source", list(set(a['source'] for a in articles)),
                                            default=list(set(a['source'] for a in articles)), key="src_filter")
            sent_filter = st.slider("Sentiment Range", -1.0, 1.0, (-1.0, 1.0), 0.1, key="sent_range")

            for a in articles:
                if a['source'] not in source_filter: continue
                if not (sent_filter[0] <= a['sentiment'] <= sent_filter[1]): continue

                s_color = sentiment_color(a['sentiment'])
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.markdown(f"**[{a['title']}]({a['link']})** — *{a['publisher']}*")
                    st.caption(f"{a['timestamp'].strftime('%b %d, %H:%M') if isinstance(a['timestamp'], datetime) else ''} | Source: {a['source']}")
                with col2:
                    st.markdown(f"<div style='text-align:center; padding:8px; border-radius:8px; background-color:{s_color}22; color:{s_color}; font-weight:bold;'>"
                                f"{a['sentiment']:+.2f}<br><small>{a['label']}</small></div>", unsafe_allow_html=True)
                st.markdown("---")

    # --- Tab 3: Market-wide news ---
    with tab_market:
        st.subheader("Market-Wide News Feed")
        if not HAS_FEEDPARSER:
            st.warning("Install `feedparser` for market news: `pip install feedparser`")
        else:
            mkt_articles = fetch_market_news()
            if mkt_articles:
                agg = aggregate_sentiment(mkt_articles)
                c1, c2, c3 = st.columns(3)
                c1.metric("Market Sentiment", f"{agg['mean']:+.3f}"); c2.metric("Tone", agg['label']); c3.metric("Articles", agg['count'])

                for a in mkt_articles[:30]:
                    sc = sentiment_color(a['sentiment'])
                    col1, col2 = st.columns([5, 1])
                    with col1:
                        st.markdown(f"**[{a['title']}]({a['link']})** — *{a['publisher']}*")
                        st.caption(f"{a['timestamp'].strftime('%b %d, %H:%M') if isinstance(a['timestamp'], datetime) else ''}")
                    with col2:
                        st.markdown(f"<div style='text-align:center; padding:4px; border-radius:6px; background-color:{sc}22; color:{sc};'>"
                                    f"{a['sentiment']:+.2f}</div>", unsafe_allow_html=True)
                    st.markdown("---")
            else:
                st.info("No market news available.")


# ==========================================
# Main
# ==========================================

def main():
    page, portfolio_df, end_date, benchmarks, use_pct, rf = render_sidebar()
    if page == "Portfolio Overview": show_main_page(portfolio_df, end_date, benchmarks, use_pct, rf)
    elif page == "Volatility & Options Lab": show_volatility_page(portfolio_df, rf)
    elif page == "Factor Exposure & Attribution": show_factor_page(portfolio_df, end_date, rf)
    elif page == "Screener & Signal Scanner": show_screener_page()
    elif page == "Trade Journal & Post-Mortem": show_journal_page()
    elif page == "Macro Regime Dashboard": show_macro_page()
    elif page == "Strategy Engine": show_strategy_page(portfolio_df, end_date, rf)
    elif page == "News & Sentiment": show_news_page(portfolio_df, rf)

if __name__ == "__main__":
    main()