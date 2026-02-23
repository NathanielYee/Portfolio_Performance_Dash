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
# Page 1: Portfolio Overview
# ==========================================

def show_main_page(portfolio_df, end_date, benchmarks, use_pct, rf):
    st.title("📈 Personal Portfolio Overview"); st.markdown("---")
    tickers = portfolio_df['Ticker'].tolist()
    if not tickers: st.info("👈 Enter holdings."); return
    port_ret, asset_ret, cur_wts, cur_dv, df_prices = build_portfolio_returns(portfolio_df, end_date)
    if port_ret is None or len(port_ret) == 0: st.warning("No data."); return
    df_bench = get_data(benchmarks, portfolio_df['Start Date'].min(), end_date) if benchmarks else pd.DataFrame()
    dv = pd.DataFrame(index=df_prices.index, columns=tickers)
    for _, r in portfolio_df.iterrows():
        t, s, sd = r['Ticker'], r['Shares'], pd.to_datetime(r['Start Date']).date()
        dv[t] = df_prices[t] * s; dv.loc[dv.index.date < sd, t] = 0.0
    m = calculate_metrics(port_ret, rf)
    st.subheader("Performance Metrics"); st.caption(f"Over **{m['n_days']}d ({m['years_held']:.1f}y)**")
    r1 = st.columns(5)
    r1[0].metric("Value", f"${cur_dv.sum():,.2f}"); r1[1].metric("Return", f"{m['total_return']:.2%}")
    r1[2].metric("CAGR", f"{m['cagr']:.2%}"); r1[3].metric("Vol", f"{m['ann_vol']:.2%}"); r1[4].metric("Max DD", f"{m['max_dd']:.2%}")
    r2 = st.columns(5)
    r2[0].metric("Sharpe", f"{m['sharpe']:.2f}"); r2[1].metric("Sortino", f"{m['sortino']:.2f}")
    r2[2].metric("Calmar", f"{m['calmar']:.2f}"); r2[3].metric("Days", f"{m['n_days']}"); r2[4].metric("RF", f"{rf:.2%}")
    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(["Performance", "Allocation", "Correlation", "Risk"])
    with tab1:
        tvh = dv.sum(axis=1); tvh = tvh[tvh > 0]
        fig = go.Figure(); fig.add_trace(go.Scatter(x=tvh.index, y=tvh, fill='tozeroy', line=dict(color='#636EFA', width=2)))
        fig.update_layout(title="Portfolio Value", yaxis_tickformat="$,.2f", template="plotly_white"); st.plotly_chart(fig, use_container_width=True)
        pp = (1 + port_ret).cumprod() - 1 if use_pct else 100 * (1 + port_ret).cumprod()
        fig2 = go.Figure(); fig2.add_trace(go.Scatter(x=pp.index, y=pp, name='Portfolio', line=dict(width=3, color='#00CC96')))
        if not df_bench.empty:
            db = df_bench.to_frame(name=benchmarks[0]) if isinstance(df_bench, pd.Series) else df_bench
            for bk in benchmarks:
                try:
                    bp = db[bk].loc[port_ret.index[0]:]; br = bp.pct_change().fillna(0)
                    bpl = (1 + br).cumprod() - 1 if use_pct else 100 * (1 + br).cumprod()
                    fig2.add_trace(go.Scatter(x=bpl.index, y=bpl, name=bk, line=dict(dash='dash', width=1.5)))
                except KeyError: continue
        fig2.update_layout(title="Relative Performance", yaxis_tickformat=".2%" if use_pct else "$.2f", template="plotly_white")
        st.plotly_chart(fig2, use_container_width=True)
    with tab2: st.plotly_chart(px.pie(names=cur_dv.index, values=cur_dv.values, hole=0.4, title="Allocation"), use_container_width=True)
    with tab3: st.plotly_chart(px.imshow(asset_ret.corr(), text_auto=True, title="Correlation", color_continuous_scale="RdBu_r"), use_container_width=True)
    with tab4:
        tvh = dv.sum(axis=1); tvh = tvh[tvh > 0]; dd = (tvh / tvh.cummax()) - 1
        fig = go.Figure(); fig.add_trace(go.Scatter(x=dd.index, y=dd, fill='tozeroy', line=dict(color='red', width=1)))
        fig.update_layout(title="Drawdown", yaxis_tickformat=".2%", template="plotly_white"); st.plotly_chart(fig, use_container_width=True)
        cr1, cr2 = st.columns(2)
        with cr1:
            if "SPY" in df_bench.columns if not df_bench.empty else False:
                mr = df_bench["SPY"].loc[port_ret.index].pct_change().dropna(); cd = port_ret.index.intersection(mr.index)
                beta = np.cov(port_ret.loc[cd], mr.loc[cd])[0][1] / np.var(mr.loc[cd]); st.metric("Beta", f"{beta:.2f}")
        with cr2:
            w = min(60, max(10, len(port_ret) // 5))
            rs = (port_ret.rolling(w).mean() / port_ret.rolling(w).std()) * np.sqrt(252)
            fig = go.Figure(); fig.add_trace(go.Scatter(x=rs.index, y=rs, line=dict(color='orange')))
            fig.update_layout(title=f"Rolling {w}D Sharpe", template="plotly_white"); st.plotly_chart(fig, use_container_width=True)
    st.markdown("---"); st.subheader("Monte Carlo")
    with st.expander("Simulate"):
        sc1, sc2 = st.columns(2); ns_mc = sc1.slider("Sims", 200, 2000, 500); hz = sc2.slider("Horizon", 30, 365, 252)
        if st.button("Run"):
            lr = np.log(1 + asset_ret); sim = np.random.multivariate_normal(lr.mean().values, lr.cov().values, size=(hz, ns_mc))
            ps = np.dot(np.exp(sim) - 1, cur_wts); sv = cur_dv.sum(); paths = sv * np.cumprod(1 + ps, axis=0)
            fig = go.Figure()
            for i in range(min(ns_mc, 50)):
                fig.add_trace(go.Scatter(x=list(range(1, hz+1)), y=paths[:, i], mode='lines', line=dict(width=1, color='rgba(100,100,255,0.1)'), showlegend=False))
            fig.add_trace(go.Scatter(x=list(range(1, hz+1)), y=np.mean(paths, axis=1), name='Mean', line=dict(width=3, color='orange')))
            fig.update_layout(yaxis_tickformat="$,.2f", template="plotly_white"); st.plotly_chart(fig, use_container_width=True)


# ==========================================
# Page 2: Volatility & Options Lab (condensed — same 7 tabs)
# ==========================================

def show_volatility_page(portfolio_df, rf):
    st.title("🌋 Volatility & Options Lab"); st.markdown("---")
    tickers = portfolio_df['Ticker'].tolist()
    if not tickers: st.info("👈 Add tickers."); return
    ticker = st.selectbox("Ticker", tickers); tk = yf.Ticker(ticker)
    try: exps = tk.options
    except Exception: st.error("No options."); return
    if not exps: st.warning("No options listed."); return
    try: spot = tk.info.get('currentPrice') or tk.info.get('regularMarketPrice') or tk.history(period="1d")['Close'].iloc[-1]
    except Exception: spot = None

    tabs = st.tabs(["Chain", "Surface", "Skew", "Greeks", "IV Rank", "Vol Arb", "Flow"])
    # Tab 0: Chain
    with tabs[0]:
        exp_sel = st.selectbox("Expiry", exps, key="ch_e"); opt = tk.option_chain(exp_sel)
        c1, c2 = st.columns(2)
        with c1: st.dataframe(opt.calls, use_container_width=True)
        with c2: st.dataframe(opt.puts, use_container_width=True)
    # Tab 1: Surface
    with tabs[1]:
        ne = st.slider("Expirations", 3, min(len(exps), 15), min(8, len(exps)), key="sn"); sd = []
        for exp in exps[:ne]:
            try:
                ch = tk.option_chain(exp); dte = max((pd.to_datetime(exp) - pd.Timestamp.now()).days, 1)
                for _, row in ch.calls.iterrows():
                    if row['impliedVolatility'] > 0.001 and spot and 0.7 < row['strike']/spot < 1.3:
                        sd.append({'Strike': row['strike'], 'DTE': dte, 'IV': row['impliedVolatility']})
            except Exception: continue
        if sd:
            sdf = pd.DataFrame(sd); piv = sdf.pivot_table(values='IV', index='Strike', columns='DTE', aggfunc='mean')
            piv = piv.interpolate(axis=0).interpolate(axis=1).dropna()
            if not piv.empty:
                fig = go.Figure(data=[go.Surface(z=piv.values, x=piv.columns.values, y=piv.index.values, colorscale='Viridis')])
                fig.update_layout(scene=dict(xaxis_title='DTE', yaxis_title='Strike', zaxis_title='IV'), width=800, height=600)
                st.plotly_chart(fig, use_container_width=True)
    # Tab 2: Skew
    with tabs[2]:
        if spot:
            skp = st.slider("OTM %", 2, 20, 5, key="skp"); skd = []
            for exp in exps[:12]:
                try:
                    ch = tk.option_chain(exp); dte = max((pd.to_datetime(exp) - pd.Timestamp.now()).days, 1)
                    p = ch.puts[ch.puts['impliedVolatility'] > 0].copy(); p['d'] = (p['strike'] - spot*(1-skp/100)).abs()
                    pv = p.nsmallest(1, 'd')['impliedVolatility'].values[0] if len(p) > 0 else np.nan
                    c = ch.calls[ch.calls['impliedVolatility'] > 0].copy(); c['d'] = (c['strike'] - spot*(1+skp/100)).abs()
                    cv = c.nsmallest(1, 'd')['impliedVolatility'].values[0] if len(c) > 0 else np.nan
                    skd.append({'Exp': exp, 'DTE': dte, 'Put IV': pv, 'Call IV': cv, 'Skew': pv - cv})
                except Exception: continue
            if skd:
                skdf = pd.DataFrame(skd).dropna()
                fig = go.Figure(); fig.add_trace(go.Bar(x=skdf['Exp'], y=skdf['Skew'], marker_color=['red' if s > 0 else 'green' for s in skdf['Skew']]))
                fig.update_layout(title=f"Skew ({skp}% OTM)", yaxis_tickformat=".2%", template="plotly_white"); st.plotly_chart(fig, use_container_width=True)
    # Tab 3: Greeks
    with tabs[3]:
        gc = st.columns(3)
        gs = gc[0].number_input("Spot", value=float(spot) if spot else 150.0, key="gs"); gk = gc[1].number_input("Strike", value=float(round(spot)) if spot else 150.0, key="gk")
        gd = gc[2].number_input("DTE", value=30, min_value=1, key="gd")
        gc2 = st.columns(3)
        gv = gc2[0].number_input("IV%", value=25.0, step=0.5, key="gv") / 100; gr = gc2[1].number_input("RF%", value=rf*100, step=0.1, key="gr") / 100
        gt = gc2[2].selectbox("Type", ["Call", "Put"], key="gt")
        T = gd / 365; ot = gt.lower(); bs = bs_greeks(gs, gk, T, gr, gv, ot); bn = binomial_greeks(gs, gk, T, gr, gv, ot)
        cdf = pd.DataFrame({'Metric': ['Price', 'Delta', 'Gamma', 'Theta', 'Vega', 'Rho'],
            'BS': [bs['price'], bs['delta'], bs['gamma'], bs['theta'], bs['vega'], bs['rho']],
            'Binomial': [bn['price'], bn['delta'], bn['gamma'], bn['theta'], bn['vega'], bn['rho']]})
        st.dataframe(cdf.style.format({'BS': '{:.4f}', 'Binomial': '{:.4f}'}), use_container_width=True)
        # Scenario P&L
        sr = np.linspace(gs*0.85, gs*1.15, 15); vr = np.linspace(max(gv*0.5, 0.05), gv*1.5, 11)
        pnl = np.array([[(bs_price(s, gk, T, gr, v, ot) - bs['price']) * 100 for s in sr] for v in vr])
        fig = go.Figure(data=go.Heatmap(z=pnl, x=[f"${s:.0f}" for s in sr], y=[f"{v:.0%}" for v in vr], colorscale='RdYlGn', zmid=0))
        fig.update_layout(title="P&L (Spot × IV)", template="plotly_white"); st.plotly_chart(fig, use_container_width=True)
    # Tab 4: IV Rank
    with tabs[4]:
        hp = get_data([ticker], datetime.now().date() - timedelta(days=400), datetime.now().date())
        if not hp.empty:
            s = hp[ticker] if isinstance(hp, pd.DataFrame) and ticker in hp.columns else hp.squeeze()
            lr = np.log(s / s.shift(1)).dropna(); rv20 = lr.rolling(20).std().dropna() * np.sqrt(252)
            civ = None
            try:
                ch = tk.option_chain(exps[0]); c = ch.calls.copy()
                if spot: c['d'] = (c['strike'] - spot).abs(); civ = c.nsmallest(3, 'd')['impliedVolatility'].mean()
            except Exception: pass
            if civ and len(rv20) > 20:
                ivr = (civ - rv20.min()) / (rv20.max() - rv20.min()) if rv20.max() != rv20.min() else 0.5
                c1, c2 = st.columns(2); c1.metric("ATM IV", f"{civ:.1%}"); c2.metric("IV Rank", f"{ivr:.1%}")
                fig = go.Figure(); fig.add_trace(go.Histogram(x=rv20, nbinsx=50, marker_color='#636EFA', opacity=0.7))
                fig.add_vline(x=civ, line_color="red", annotation_text=f"IV {civ:.1%}")
                fig.update_layout(title="Vol Distribution", template="plotly_white"); st.plotly_chart(fig, use_container_width=True)
    # Tab 5: Vol Arb
    with tabs[5]:
        vu = st.text_input("Universe", ", ".join(tickers), key="va"); vtk = [t.strip().upper() for t in vu.split(",") if t.strip()]
        if st.button("Scan Vol", key="vab") and vtk:
            res = []; prog = st.progress(0)
            for i, stk in enumerate(vtk):
                prog.progress((i+1)/len(vtk))
                try:
                    to = yf.Ticker(stk); se = to.options
                    if not se: continue
                    sp2 = to.info.get('currentPrice') or to.info.get('regularMarketPrice')
                    if not sp2: continue
                    ch = to.option_chain(se[0]); c = ch.calls.copy(); c['d'] = (c['strike'] - sp2).abs()
                    aiv = c.nsmallest(3, 'd')['impliedVolatility'].mean()
                    hp2 = yf.download(stk, period="60d", auto_adjust=True, progress=False)
                    cl = hp2['Close'].squeeze() if isinstance(hp2['Close'], pd.DataFrame) else hp2['Close']
                    rv = np.log(cl / cl.shift(1)).dropna().tail(20).std() * np.sqrt(252)
                    res.append({"Ticker": stk, "IV": aiv, "RV": rv, "Gap": aiv - rv})
                except Exception: continue
            prog.empty()
            if res:
                rdf = pd.DataFrame(res).sort_values("Gap", ascending=False)
                st.dataframe(rdf.style.format({"IV": "{:.2%}", "RV": "{:.2%}", "Gap": "{:.2%}"}), use_container_width=True)
    # Tab 6: Flow
    with tabs[6]:
        oi_e = st.selectbox("Expiry", exps, key="oie"); opt = tk.option_chain(oi_e)
        oc = opt.calls[['strike', 'openInterest']].copy(); op = opt.puts[['strike', 'openInterest']].copy()
        om = pd.merge(oc.rename(columns={'openInterest': 'Call OI'}), op.rename(columns={'openInterest': 'Put OI'}), on='strike', how='outer').fillna(0)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=om['strike'], y=om['Call OI'], name='Call OI', marker_color='green', opacity=0.7))
        fig.add_trace(go.Bar(x=om['strike'], y=-om['Put OI'], name='Put OI', marker_color='red', opacity=0.7))
        if spot: fig.add_vline(x=spot, line_dash="dash", annotation_text=f"${spot:.0f}")
        fig.update_layout(title="OI", barmode='relative', template="plotly_white"); st.plotly_chart(fig, use_container_width=True)
        # Max Pain
        strikes = sorted(set(oc['strike'].tolist() + op['strike'].tolist()))
        pain = [{'S': s, 'P': oc.apply(lambda r: r['openInterest'] * max(s - r['strike'], 0), axis=1).sum() +
                 op.apply(lambda r: r['openInterest'] * max(r['strike'] - s, 0), axis=1).sum()} for s in strikes]
        pdf = pd.DataFrame(pain); mpk = pdf.loc[pdf['P'].idxmin(), 'S']
        fig = go.Figure(); fig.add_trace(go.Scatter(x=pdf['S'], y=pdf['P'], fill='tozeroy'))
        fig.add_vline(x=mpk, line_dash="dash", line_color="red", annotation_text=f"Max Pain ${mpk:.0f}")
        fig.update_layout(title="Max Pain", yaxis_tickformat="$,.0f", template="plotly_white"); st.plotly_chart(fig, use_container_width=True)


# ==========================================
# Pages 3-6 (Factor, Screener, Journal, Macro) — same logic, kept compact
# ==========================================

def show_factor_page(portfolio_df, end_date, rf):
    st.title("🧬 Factor Exposure"); st.markdown("---")
    port_ret, _, _, _, _ = build_portfolio_returns(portfolio_df, end_date)
    if port_ret is None or len(port_ret) < 60: st.warning("Need 60+ days."); return
    fm = {"Market": "SPY", "Size": ("IWM", "SPY"), "Value": ("IWD", "IWF"), "Momentum": "MTUM", "Quality": "QUAL", "LowVol": "USMV"}
    all_e = list(set(v if isinstance(v, str) else v[0] for v in fm.values()) | set(v[1] for v in fm.values() if isinstance(v, tuple)))
    ep = get_data(all_e, portfolio_df['Start Date'].min(), end_date)
    if ep.empty: st.error("No data."); return
    er = ep.pct_change().dropna(); factors = pd.DataFrame(index=er.index)
    for n, p in fm.items():
        try: factors[n] = er[p] if isinstance(p, str) else er[p[0]] - er[p[1]]
        except KeyError: continue
    ci = port_ret.index.intersection(factors.index); y = port_ret.loc[ci] - rf/252; X = factors.loc[ci].dropna(axis=1, how='all').dropna(); y = y.loc[X.index]
    if len(y) < 30: return
    model = OLS(y, add_constant(X)).fit(); alpha = model.params.get('const', 0) * 252
    c1, c2, c3 = st.columns(3); c1.metric("Alpha", f"{alpha:.2%}"); c2.metric("R²", f"{model.rsquared:.2%}"); c3.metric("Adj R²", f"{model.rsquared_adj:.2%}")
    betas = pd.DataFrame({'Beta': model.params.drop('const', errors='ignore'), 't': model.tvalues.drop('const', errors='ignore'), 'p': model.pvalues.drop('const', errors='ignore')})
    st.dataframe(betas.style.format({"Beta": "{:.4f}", "t": "{:.2f}", "p": "{:.4f}"}), use_container_width=True)
    fig = go.Figure(); fig.add_trace(go.Bar(x=betas.index, y=betas['Beta'], marker_color=['#00CC96' if b > 0 else '#EF553B' for b in betas['Beta']]))
    fig.update_layout(title="Factor Betas", template="plotly_white"); st.plotly_chart(fig, use_container_width=True)

def show_screener_page():
    st.title("🔍 Screener"); st.markdown("---")
    t1, t2 = st.tabs(["Signals", "Cointegration"])
    with t1:
        ui = st.text_input("Universe", "AAPL, MSFT, GOOG, AMZN, META, NVDA, TSLA, JPM")
        u = [t.strip().upper() for t in ui.split(",") if t.strip()]; lb = st.slider("Lookback", 30, 365, 120)
        if st.button("Scan") and u:
            pr = get_data(u, datetime.now().date() - timedelta(days=lb+50), datetime.now().date())
            if pr.empty: return
            res = []
            for tk in u:
                try:
                    s = pr[tk].dropna()
                    if len(s) < 30: continue
                    rsi = compute_rsi(s).iloc[-1]; z = ((s - s.rolling(20).mean()) / s.rolling(20).std()).iloc[-1]
                    sig = "Oversold" if rsi < 30 else ("Overbought" if rsi > 70 else ("MR Long" if z < -2 else ("MR Short" if z > 2 else "Neutral")))
                    res.append({"Ticker": tk, "RSI": rsi, "Z": z, "Signal": sig})
                except Exception: continue
            if res:
                st.dataframe(pd.DataFrame(res).style.format({"RSI": "{:.1f}", "Z": "{:.2f}"}).applymap(
                    lambda v: 'color: green' if v in ['Oversold', 'MR Long'] else ('color: red' if v in ['Overbought', 'MR Short'] else ''), subset=['Signal']), use_container_width=True)
    with t2:
        pi = st.text_input("Pairs", "XOM, CVX, COP, SLB, EOG", key="cu"); pt = [t.strip().upper() for t in pi.split(",") if t.strip()]
        if st.button("Scan Pairs") and len(pt) >= 2:
            pr = get_data(pt, datetime.now().date() - timedelta(days=262), datetime.now().date())
            if pr.empty: return
            pairs = []
            for i in range(len(pt)):
                for j in range(i+1, len(pt)):
                    try:
                        s1, s2 = pr[pt[i]].dropna(), pr[pt[j]].dropna(); ci = s1.index.intersection(s2.index)
                        if len(ci) < 30: continue
                        _, pv, _ = coint(s1.loc[ci], s2.loc[ci])
                        pairs.append({"Pair": f"{pt[i]}/{pt[j]}", "p": pv})
                    except Exception: continue
            if pairs: st.dataframe(pd.DataFrame(pairs).sort_values("p").style.format({"p": "{:.4f}"}), use_container_width=True)

def show_journal_page():
    st.title("📓 Trade Journal"); st.markdown("---")
    if 'trades' not in st.session_state: st.session_state.trades = []
    t1, t2, t3 = st.tabs(["Log", "Analytics", "History"])
    with t1:
        with st.form("tf"):
            c1, c2, c3 = st.columns(3); tk = c1.text_input("Ticker", "AAPL"); dr = c2.selectbox("Dir", ["Long", "Short"]); sh = c3.number_input("Shares", value=10.0, min_value=0.01)
            c4, c5 = st.columns(2); ep = c4.number_input("Entry $", value=150.0, min_value=0.01); xp = c5.number_input("Exit $ (0=open)", min_value=0.0, value=0.0)
            th = st.text_area("Thesis")
            if st.form_submit_button("Log"):
                m = 1 if dr == "Long" else -1; cl = xp > 0; pnl = m * (xp - ep) * sh if cl else 0
                st.session_state.trades.append({"Ticker": tk.upper(), "Direction": dr, "Entry": ep, "Exit": xp if cl else None, "P&L ($)": pnl, "Status": "Closed" if cl else "Open", "Thesis": th})
                st.success(f"Logged {tk.upper()}")
    with t2:
        cl = [t for t in st.session_state.trades if t.get('Status') == 'Closed']
        if cl:
            df = pd.DataFrame(cl); st.metric("Win Rate", f"{(df['P&L ($)'] > 0).mean():.1%}"); st.metric("Total P&L", f"${df['P&L ($)'].sum():,.2f}")
    with t3:
        if st.session_state.trades: st.dataframe(pd.DataFrame(st.session_state.trades), use_container_width=True)
        st.download_button("💾 Export", pd.DataFrame(st.session_state.trades).to_csv(index=False).encode('utf-8') if st.session_state.trades else b"", "journal.csv")

@st.cache_data(ttl=3600)
def fetch_macro_data():
    tks = {"^TNX": "10Y", "^IRX": "3M", "^VIX": "VIX", "DX-Y.NYB": "DXY", "GC=F": "Gold", "HYG": "HYG", "LQD": "LQD"}
    data = {}
    for t, n in tks.items():
        try:
            d = yf.download(t, period="2y", auto_adjust=True, progress=False)
            if not d.empty: data[n] = d['Close'].squeeze() if isinstance(d['Close'], pd.DataFrame) else d['Close']
        except Exception: continue
    return pd.DataFrame(data)

def show_macro_page():
    st.title("🌍 Macro Regime"); st.markdown("---")
    m = fetch_macro_data()
    if m.empty: st.error("No data."); return
    la = m.iloc[-1]; c1, c2, c3, c4 = st.columns(4)
    if '10Y' in m.columns and '3M' in m.columns: c1.metric("10Y-3M", f"{la['10Y']-la['3M']:.2f}%")
    if 'VIX' in m.columns: c2.metric("VIX", f"{la['VIX']:.1f}")
    if 'DXY' in m.columns: c3.metric("DXY", f"{la['DXY']:.1f}")
    if 'Gold' in m.columns: c4.metric("Gold", f"${la['Gold']:,.0f}")
    if '10Y' in m.columns and '3M' in m.columns:
        cs = m['10Y'] - m['3M']; fig = go.Figure(); fig.add_trace(go.Scatter(x=cs.index, y=cs, fill='tozeroy'))
        fig.add_hline(y=0, line_dash="dash", line_color="red"); fig.update_layout(title="Yield Curve", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    if 'VIX' in m.columns:
        fig = go.Figure(); fig.add_trace(go.Scatter(x=m.index, y=m['VIX'], line=dict(color='orange')))
        fig.add_hline(y=20, line_dash="dash"); fig.add_hline(y=30, line_dash="dash", line_color="red")
        fig.update_layout(title="VIX", template="plotly_white"); st.plotly_chart(fig, use_container_width=True)


# ==========================================
# Page 7: Strategy Engine
# ==========================================

def show_strategy_page(portfolio_df, end_date, rf):
    st.title("🧠 Strategy Engine"); st.markdown("---")
    tickers = portfolio_df['Ticker'].tolist()
    if not tickers: st.info("👈 Add holdings."); return
    port_ret, _, _, cur_dv, _ = build_portfolio_returns(portfolio_df, end_date)
    port_val = cur_dv.sum() if cur_dv is not None else 0
    if port_val <= 0: st.warning("No portfolio value."); return

    with st.spinner("Scanning signals + news across portfolio..."):
        signals = compute_all_signals(tickers, port_val, rf)
        ideas = generate_trade_ideas(signals, portfolio_df, rf)

    regime = signals['macro'].get('regime', 'unknown')
    mkt_news = signals.get('news', {}).get('market', {})
    rd = {"risk_on": "RISK-ON 🟢", "cautious": "CAUTIOUS 🟡", "risk_off": "RISK-OFF 🔴"}.get(regime, "UNKNOWN")

    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
    sc1.metric("Macro Regime", rd); sc2.metric("VIX", f"{signals['macro'].get('vix', 0):.1f}")
    sc3.metric("Market News", mkt_news.get('label', 'N/A'), help=f"Avg sentiment: {mkt_news.get('mean', 0):+.2f} from {mkt_news.get('count', 0)} articles")
    sc4.metric("Portfolio", f"${port_val:,.0f}"); sc5.metric("Ideas", f"{len(ideas)}")

    # Per-ticker signal + sentiment summary
    st.subheader("Signal Dashboard")
    tks = []
    for tk, ts in signals['ticker_signals'].items():
        ns_data = ts.get('news_sentiment', {})
        tks.append({'Ticker': tk, 'Spot': ts.get('spot', 0), 'IV': ts.get('atm_iv', 0), 'RV': ts.get('rv_20', 0),
                     'IV Rank': ts.get('iv_rank', 0), 'RSI': ts.get('rsi', 50), 'Z': ts.get('zscore', 0),
                     'News': ns_data.get('label', 'N/A'), 'Sent.': ns_data.get('mean', 0), 'Articles': ns_data.get('count', 0)})
    if tks:
        tkdf = pd.DataFrame(tks)
        st.dataframe(tkdf.style.format({'Spot': '${:.2f}', 'IV': '{:.1%}', 'RV': '{:.1%}', 'IV Rank': '{:.0%}',
                                          'RSI': '{:.0f}', 'Z': '{:.2f}', 'Sent.': '{:+.2f}'}), use_container_width=True)

    st.markdown("---"); st.subheader(f"Trade Ideas ({len(ideas)})")
    if not ideas: st.info("No actionable ideas — portfolio is well-positioned."); return

    fc1, fc2 = st.columns(2)
    uf = fc1.multiselect("Urgency", ["High", "Medium", "Low"], default=["High", "Medium"])
    df_filter = fc2.multiselect("Direction", list(set(i['direction'] for i in ideas)), default=list(set(i['direction'] for i in ideas)))
    filtered = [i for i in ideas if i['urgency'] in uf and i['direction'] in df_filter]

    for idea in filtered:
        uc = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(idea['urgency'], "⚪")
        with st.expander(f"{uc} {idea['ticker']} — {idea['strategy']} ({idea['direction']})", expanded=(idea['urgency'] == 'High')):
            ic = st.columns(4)
            ic[0].metric("Strategy", idea['strategy']); ic[1].metric("Urgency", idea['urgency'])
            cl = "Credit" if idea['cost'] < 0 else "Cost"; ic[2].metric(cl, f"${abs(idea['cost']):,.2f}")
            if idea.get('risk_reward', 0) > 0: ic[3].metric("R/R", f"{idea['risk_reward']:.1f}x")
            elif idea['max_gain'] == float('inf'): ic[3].metric("Upside", "Unlimited")
            else: ic[3].metric("Max Gain", f"${idea['max_gain']:,.2f}")

            legs_df = pd.DataFrame(idea['legs']); legs_df['total'] = legs_df['price'] * 100 * legs_df['contracts']
            legs_df.columns = ['Action', 'Strike', '$/sh', 'Contracts', 'Total $']
            st.dataframe(legs_df.style.format({'Strike': '${:.2f}', '$/sh': '${:.2f}', 'Total $': '${:,.2f}'}), use_container_width=True)
            st.markdown(f"**Expiry:** {idea['expiry']}")
            rc = st.columns(3)
            rc[0].metric("Max Loss", f"${idea['max_loss']:,.2f}" if idea['max_loss'] != float('inf') else "∞")
            rc[1].metric("Max Gain", f"${idea['max_gain']:,.2f}" if idea['max_gain'] != float('inf') else "∞")
            rc[2].metric("Risk %", f"{abs(idea['max_loss'])/port_val:.2%}" if idea['max_loss'] != float('inf') else "N/A")
            st.markdown(f"**Rationale:** {idea['rationale']}")

            # P&L at expiry
            if len(idea['legs']) >= 2 and idea['max_loss'] != float('inf'):
                ks = [l['strike'] for l in idea['legs']]; sr = np.linspace(min(ks)*0.95, max(ks)*1.05, 100)
                pnl = np.zeros_like(sr)
                for l in idea['legs']:
                    k, p, n = l['strike'], l['price'], l['contracts']
                    if 'Buy Call' in l['type']: pnl += (np.maximum(sr - k, 0) - p) * 100 * n
                    elif 'Sell Call' in l['type']: pnl += (p - np.maximum(sr - k, 0)) * 100 * n
                    elif 'Buy Put' in l['type']: pnl += (np.maximum(k - sr, 0) - p) * 100 * n
                    elif 'Sell Put' in l['type']: pnl += (p - np.maximum(k - sr, 0)) * 100 * n
                fig = go.Figure(); fig.add_trace(go.Scatter(x=sr, y=pnl, fill='tozeroy', line=dict(color='#00CC96')))
                fig.add_hline(y=0, line_dash="dash")
                for s in ks: fig.add_vline(x=s, line_dash="dot", line_color="orange")
                fig.update_layout(title="P&L at Expiry", yaxis_tickformat="$,.0f", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)


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