import os
import re
import json
import time
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import warnings

warnings.filterwarnings("ignore")

try:
    import requests
except Exception:
    requests = None

logging.getLogger("yahooquery").setLevel(logging.ERROR)

# ============================================================
# CONFIG
# ============================================================
FMP_API_KEY = os.environ.get("FMP_API_KEY")
if not FMP_API_KEY:
    raise RuntimeError(
        "Set the FMP_API_KEY environment variable first, e.g.\n"
        "  export FMP_API_KEY='your-key-here'"
    )


FMP_CACHE_FILE = "fmp_cache.json"       # kept for possible future use
YAHOO_CACHE_FILE = "yahoo_cache.json"   # kept for possible future use

EXCEL_PATH = os.environ.get("TICKER_LIST_PATH", "data/approved_tickers.xlsx")
VALID_TICKER_PATTERN = re.compile(r"^[A-Z]+$")  # only letters, no digits

TOP_N_ENRICH = 300       # how many by market cap to fully enrich with FMP
MAX_TRAIN = 250          # max tickers used for factor model
FACTOR_YEARS = 5         # yfinance history in years
FACTOR_INTERVAL = "1mo"  # yfinance interval

FINAL_MAX_RESULTS = 50   # <<< TOP 50 FINAL NAMES

# NEW: stronger Philly & environment emphasis knobs
PHILLY_TICKERS = ["CMCSA", "UGI", "PPL", "BK", "FIS", "IVZ", "LIN", "CECO", "INFI"]
ENV_KEYWORDS = [
    "community", "sustainable", "sustainability", "green",
    "environmental", "environment", "climate", "renewable",
    "clean energy", "solar", "wind", "esg", "social", "impact",
    "urban", "park", "playground", "public space"
]

print("=" * 80)
print("WHARTON COMPETITION - MTWB FOUNDATION PORTFOLIO")
print("Connor Barwin's Philadelphia & Community-Focused Investments")
print("=" * 80)


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_val(d, *keys, default=None):
    """
    Try keys in order from dict d, convert to float when possible, else default.
    Handles things like '12%' or '1,234.5'.
    """
    if not isinstance(d, dict):
        return default
    for k in keys:
        v = d.get(k)
        if v is None:
            continue
        try:
            return float(v)
        except Exception:
            try:
                s = str(v).replace("%", "").replace(",", "")
                return float(s)
            except Exception:
                continue
    return default


def load_yahoo_cache():
    if os.path.exists(YAHOO_CACHE_FILE):
        try:
            with open(YAHOO_CACHE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_yahoo_cache(cache):
    try:
        with open(YAHOO_CACHE_FILE, "w") as f:
            json.dump(cache, f)
    except Exception:
        pass


YAHOO_CACHE = load_yahoo_cache()


def _extract_fmp_data(resp_json):
    """
    Normalize FMP responses:
      - [ {...}, {...} ]
      - { "data": [ ... ] }
      - { "symbol": "...", ... }
      - error payloads with "Error Message"
    """
    if isinstance(resp_json, dict):
        if "Error Message" in resp_json or "error" in resp_json:
            return []
        data = resp_json.get("data")
        if isinstance(data, list):
            return data
        return [resp_json]
    elif isinstance(resp_json, list):
        return resp_json
    else:
        return []


# ============================================================
# FMP: BATCH QUOTES
# ============================================================

def batch_get_fmp_quotes(tickers, batch_size=200):
    """
    Fetch price + marketCap for many tickers using the STABLE batch-quote endpoint.
    Example:
      https://financialmodelingprep.com/stable/batch-quote?symbols=AAPL,MSFT&apikey=...
    """
    if requests is None:
        raise RuntimeError("The 'requests' library is required for FMP calls.")

    results = {}
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        symbols = ",".join(batch)
        url = (
            "https://financialmodelingprep.com/stable/batch-quote"
            f"?symbols={symbols}&apikey={FMP_API_KEY}"
        )
        print(f" → Batch quotes {i // batch_size + 1}: {len(batch)} tickers")
        try:
            resp = requests.get(url, timeout=15)
            resp_json = resp.json()
            data = _extract_fmp_data(resp_json)
            for item in data:
                t = item.get("symbol")
                if t:
                    results[t] = {
                        "price": item.get("price"),
                        "marketCap": item.get("marketCap"),
                    }
        except Exception as e:
            print("Batch quote error:", e)
        time.sleep(0.15)
    return results


# ============================================================
# FMP: SINGLE-TICKER RATIOS
# ============================================================

def get_fmp_ratios_for_ticker(ticker):
    """
    Uses the working endpoint:
      https://financialmodelingprep.com/stable/ratios?symbol=AAPL&apikey=...
    Returns key ratios for MTWB scoring and prescreen.
    """
    if requests is None:
        raise RuntimeError("The 'requests' library is required for FMP calls.")

    url = (
        f"https://financialmodelingprep.com/stable/ratios"
        f"?symbol={ticker}&apikey={FMP_API_KEY}"
    )
    try:
        resp = requests.get(url, timeout=10)
        resp_json = resp.json()

        if isinstance(resp_json, dict) and "Error Message" in resp_json:
            print(f"Ratios error for {ticker}: {resp_json.get('Error Message')}")
            return {}

        data = _extract_fmp_data(resp_json)
        if not data:
            return {}

        item = data[0]  # most recent

        pe = safe_val(item, "priceToEarningsRatio")
        debt_to_equity = safe_val(item, "debtToEquityRatio")
        current_ratio = safe_val(item, "currentRatio")

        net_ni_ps = safe_val(item, "netIncomePerShare")
        equity_ps = safe_val(item, "shareholdersEquityPerShare")
        roe = None
        if equity_ps not in (None, 0) and net_ni_ps is not None:
            roe = net_ni_ps / equity_ps  # e.g. 0.18 = 18%

        return {
            "trailingPE": pe,
            "returnOnEquity": roe,
            "debtToEquity": debt_to_equity,
            "currentRatio": current_ratio,
            "forwardPE": None,  # not available from this endpoint
        }
    except Exception as e:
        print(f"Ratios error for {ticker}:", e)
        return {}


def batch_get_fmp_ratios(tickers, sleep_sec=0.1):
    results = {}
    total = len(tickers)
    for idx, t in enumerate(tickers, start=1):
        print(f"  Ratios {idx}/{total}: {t}")
        res = get_fmp_ratios_for_ticker(t)
        if res:
            results[t] = res
        time.sleep(sleep_sec)
    return results


# ============================================================
# FMP: SINGLE-TICKER INCOME STATEMENT (REVENUE GROWTH)
# ============================================================

def get_fmp_income_growth_for_ticker(ticker):
    """
    Uses:
      https://financialmodelingprep.com/stable/income-statement?symbol=AAPL&apikey=...
    Computes:
      revenueGrowth = (latestRevenue - previousRevenue) / previousRevenue
    """
    if requests is None:
        raise RuntimeError("The 'requests' library is required for FMP calls.")

    url = (
        "https://financialmodelingprep.com/stable/income-statement"
        f"?symbol={ticker}&apikey={FMP_API_KEY}"
    )
    try:
        resp = requests.get(url, timeout=10)
        resp_json = resp.json()

        if isinstance(resp_json, dict) and "Error Message" in resp_json:
            print(f"Income statement error for {ticker}: {resp_json.get('Error Message')}")
            return None

        data = _extract_fmp_data(resp_json)
        if not data or len(data) < 2:
            return None

        r0 = safe_val(data[0], "revenue")
        r1 = safe_val(data[1], "revenue")
        if r1 is not None and r1 > 0 and r0 is not None:
            return (r0 - r1) / r1
        return None
    except Exception as e:
        print(f"Income statement error for {ticker}:", e)
        return None


def batch_get_fmp_income_growth(tickers, sleep_sec=0.1):
    results = {}
    total = len(tickers)
    for idx, t in enumerate(tickers, start=1):
        print(f"  Income {idx}/{total}: {t}")
        growth = get_fmp_income_growth_for_ticker(t)
        results[t] = growth
        time.sleep(sleep_sec)
    return results


# ============================================================
# OPTIONAL: FMP PROFILE (FOR FINAL WINNERS ONLY)
# ============================================================

def get_fmp_profile(ticker):
    """
    Single-ticker profile from STABLE endpoint.
    Only used for FINAL screened tickers to fill name / sector.
    """
    if requests is None:
        raise RuntimeError("The 'requests' library is required for FMP calls.")

    url = (
        f"https://financialmodelingprep.com/stable/profile"
        f"?symbol={ticker}&apikey={FMP_API_KEY}"
    )
    try:
        resp = requests.get(url, timeout=10)
        resp_json = resp.json()

        if isinstance(resp_json, dict) and "Error Message" in resp_json:
            print(f"Profile error for {ticker}: {resp_json.get('Error Message')}")
            return {}

        data = _extract_fmp_data(resp_json)
        if not data:
            return {}
        p = data[0]
        return {
            "longName": p.get("companyName") or p.get("companyNameLong") or ticker,
            "sector": p.get("sector"),
            "industry": p.get("industry"),
            "exchangeShortName": p.get("exchangeShortName"),
            "isEtf": p.get("isEtf"),
            "country": p.get("country"),
        }
    except Exception:
        return {}


# ============================================================
# TICKER CLEANING (REMOVE FOREIGN / OTC PATTERNS)
# ============================================================

def is_reasonable_wharton_ticker(ticker):
    """
    Keep:
      - 1–5 letters
      - Only A–Z
    Remove:
      - any '.' (foreign, dual-listing)
      - any digits
      - length > 5
      - ending with 'F' or 'Y' (foreign / ADR)
    """
    if not isinstance(ticker, str):
        return False
    t = ticker.strip().upper()
    if "." in t:
        return False
    if any(ch.isdigit() for ch in t):
        return False
    if not VALID_TICKER_PATTERN.match(t):
        return False
    if len(t) > 5:
        return False
    if t.endswith("F") or t.endswith("Y"):
        return False
    return True


# ============================================================
# PRESCREEN (HARD FILTER)
# ============================================================

def passes_prescreen(info):
    """
    HARD filter:
      - price, marketCap, PE, ROE, revenueGrowth must exist
      - price >= 5
      - marketCap >= 300M
      - ROE > 5%
      - revenueGrowth > 0
      - 0 < PE <= 80
    """
    price = safe_val(info, "price", default=None)
    mcap = safe_val(info, "marketCap", default=None)
    pe = safe_val(info, "trailingPE", "forwardPE", default=None)
    roe = info.get("returnOnEquity")
    growth = info.get("revenueGrowth")

    if price is None or mcap is None or pe is None or roe is None or growth is None:
        return False

    if price < 5:
        return False
    if mcap < 3e8:
        return False
    if roe <= 0.05:
        return False
    if growth <= 0:
        return False
    if pe <= 0 or pe > 80:
        return False

    return True


def run_fundamental_prescreen(all_tickers):
    """
    Pipeline:
      1) Basic cleaning of tickers (remove foreign/OTC patterns).
      2) Batch quotes for all remaining tickers.
      3) Basic quote filter on price & marketCap.
      4) Take TOP_N_ENRICH by market cap for full FMP ratios + income.
      5) Apply HARD prescreen on enriched data.
    """
    print("\nApplying fundamental prescreen (HARD FILTER)...")

    clean_tickers = [t for t in all_tickers if is_reasonable_wharton_ticker(t)]
    print(
        f"✓ {len(clean_tickers)} tickers after removing foreign/OTC patterns "
        f"(from {len(all_tickers)})"
    )

    if not clean_tickers:
        print("❌ No valid tickers after cleaning.")
        return [], {}

    quote_cache = batch_get_fmp_quotes(clean_tickers)

    basic_pass = []
    for t in clean_tickers:
        q = quote_cache.get(t, {}) or {}
        price = safe_val(q, "price")
        mcap = safe_val(q, "marketCap")
        if price is None or mcap is None:
            continue
        if price < 5:
            continue
        if mcap < 3e8:
            continue
        basic_pass.append((t, mcap))

    print(
        f"✓ {len(basic_pass)} tickers passed BASIC quote screen "
        f"(out of {len(clean_tickers)})"
    )

    if not basic_pass:
        print("❌ No tickers passed basic screen.")
        return [], {}

    basic_pass_sorted = sorted(basic_pass, key=lambda x: x[1], reverse=True)
    enrich_tickers = [t for t, _ in basic_pass_sorted[:TOP_N_ENRICH]]

    print(f"\nEnriching TOP {len(enrich_tickers)} tickers with ratios + income...")
    ratios_cache = batch_get_fmp_ratios(enrich_tickers)
    growth_cache = batch_get_fmp_income_growth(enrich_tickers)

    fundamentals_cache = {}
    filtered = []

    for t in enrich_tickers:
        quote = quote_cache.get(t, {}) or {}
        ratios = ratios_cache.get(t, {}) or {}
        rev_growth = growth_cache.get(t)

        info = {
            "symbol": t,
            "longName": t,
            "sector": None,
            "industry": None,
            "price": safe_val(quote, "price"),
            "marketCap": safe_val(quote, "marketCap"),
            "trailingPE": ratios.get("trailingPE"),
            "returnOnEquity": ratios.get("returnOnEquity"),
            "currentRatio": ratios.get("currentRatio"),
            "debtToEquity": ratios.get("debtToEquity"),
            "revenueGrowth": rev_growth,
            "beta": 1.0,  # placeholder
            "source": "fmp-enriched",
        }

        fundamentals_cache[t] = info

        if passes_prescreen(info):
            filtered.append(t)

    print(
        f"✓ {len(filtered)} tickers passed ENRICHED hard filter "
        f"(from {len(enrich_tickers)})"
    )
    return filtered, fundamentals_cache


# ============================================================
# MTWB SCORING  (UPGRADED FOR PHILLY + ENVIRONMENT)
# ============================================================

def calculate_comprehensive_scores(info, ticker, market_cap):
    """
    MTWB scoring with stronger emphasis on:
      - Philly-connected companies
      - Environmental / community oriented companies

    Still preserves your original structure:
      - Values (Connor / community / sports)
      - Financial quality (ROE, growth, valuation, health)
      - Risk adjustment (beta, size, sector)
    """
    scores = {
        "mtwb": 0,
        "community": 5,
        "sports": 3,
        "quality": 50,
    }

    # 1. CONNOR'S VALUES (PHILLY / COMMUNITY / SPORTS)
    # Stronger Philly emphasis
    if ticker in PHILLY_TICKERS:
        scores["mtwb"] += 25   # was 15 before – now stronger
        scores["community"] += 4

    sports_companies = [
        "NKE", "UA", "LULU", "DKS", "FL",
        "HIBB", "VFC", "PLNT", "PTON", "YETI"
    ]
    if ticker in sports_companies:
        scores["mtwb"] += 10
        scores["sports"] = 8 if ticker in ["NKE", "UA"] else 6

    # Environmental / community keywords (upgraded, more weight)
    company_desc = f"{info.get('longName') or ''} {info.get('industry') or ''}".lower()
    community_bonus = 0
    for keyword in ENV_KEYWORDS:
        if keyword in company_desc:
            community_bonus += 4   # slightly higher per hit
            if community_bonus >= 20:  # cap so it doesn't blow up
                break
    scores["mtwb"] += community_bonus
    scores["community"] += min(5, community_bonus // 4)

    # 2. FINANCIAL QUALITY (same logic as original, just slightly cleaned)
    financial_score = 0

    # ROE (10)
    roe = info.get("returnOnEquity", 0)
    roe_pct = roe * 100 if roe is not None else 0
    if roe_pct > 20:
        financial_score += 10
        scores["quality"] += 20
    elif roe_pct > 15:
        financial_score += 8
        scores["quality"] += 16
    elif roe_pct > 10:
        financial_score += 6
        scores["quality"] += 12
    elif roe_pct > 5:
        financial_score += 4
        scores["quality"] += 8
    elif roe_pct > 0:
        financial_score += 2
        scores["quality"] += 4

    # Growth (10)
    growth_raw = info.get("revenueGrowth")
    growth = growth_raw * 100 if growth_raw is not None else None

    if growth is not None:
        if market_cap > 1e10:
            if growth > 10:
                financial_score += 10
            elif growth > 5:
                financial_score += 7
            elif growth > 0:
                financial_score += 4
        else:
            if growth > 20:
                financial_score += 10
            elif growth > 10:
                financial_score += 8
            elif growth > 5:
                financial_score += 6
            elif growth > 0:
                financial_score += 4

    # Valuation (10)
    pe = safe_val(info, "trailingPE", "forwardPE", default=None)
    if pe is not None:
        if market_cap > 1e10:
            if pe <= 15:
                financial_score += 10
            elif pe <= 25:
                financial_score += 7
            elif pe <= 35:
                financial_score += 4
        else:
            if pe <= 20:
                financial_score += 10
            elif pe <= 30:
                financial_score += 8
            elif pe <= 40:
                financial_score += 6
            elif pe <= 50:
                financial_score += 4

    # Financial Health (10)
    debt_equity = info.get("debtToEquity")
    current_ratio = info.get("currentRatio")
    if debt_equity is not None and current_ratio is not None:
        if debt_equity < 0.5 and current_ratio > 1.5:
            financial_score += 10
        elif debt_equity < 1.0 and current_ratio > 1.0:
            financial_score += 7
        elif debt_equity < 2.0:
            financial_score += 4

    scores["mtwb"] += financial_score

    # 3. RISK ADJUSTMENT (20)
    risk_score = 0
    beta = info.get("beta", 1.0)

    if market_cap > 1e10:
        if beta < 0.8:
            risk_score += 10
        elif beta <= 1.2:
            risk_score += 8
        elif beta <= 1.5:
            risk_score += 5
    else:
        if beta < 1.0:
            risk_score += 10
        elif beta <= 1.5:
            risk_score += 8
        elif beta <= 2.0:
            risk_score += 6

    if market_cap < 2e9:
        risk_score += 5

    stable = ["Utilities", "Consumer Defensive", "Healthcare", "Real Estate"]
    if info.get("sector", "") in stable:
        risk_score += 10

    scores["mtwb"] += risk_score

    # Cap scores
    scores["mtwb"] = min(100, scores["mtwb"])
    scores["quality"] = min(100, scores["quality"])
    scores["community"] = min(10, scores["community"])
    scores["sports"] = min(10, scores["sports"])

    return scores


def get_philly_connection(ticker):
    return "Philadelphia" if ticker in PHILLY_TICKERS else "None"


def get_category(info, ticker):
    return info.get("sector") or "Unknown"


# ============================================================
# FACTOR MODEL: PRICE HISTORY (YFINANCE)
# ============================================================

def download_monthly_price_history_yf(tickers, batch_size=10):
    print(
        f"\n[FACTOR MODEL] Downloading {FACTOR_YEARS}-year monthly prices "
        "using YFinance (batch mode)..."
    )
    all_prices = {}

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        print(f"Downloading batch {i // batch_size + 1} ({len(batch)} tickers)...")
        try:
            df = yf.download(
                tickers=batch,
                period=f"{FACTOR_YEARS}y",
                interval=FACTOR_INTERVAL,
                group_by="ticker",
                threads=False,
                timeout=10,
            )
        except Exception as e:
            print("Batch download error / timeout:", e)
            time.sleep(2)
            continue

        for t in batch:
            try:
                if isinstance(df.columns, pd.MultiIndex):
                    close = df[t]["Close"].dropna()
                else:
                    close = df["Close"].dropna()
                if len(close) >= 60:  # ~5 years of monthly data
                    all_prices[t] = close
            except Exception:
                continue

        time.sleep(0.5)

    print(f"✓ Got valid monthly history for {len(all_prices)} tickers.")
    return all_prices


def compute_max_drawdown(prices):
    running_max = prices.cummax()
    drawdown = (prices - running_max) / running_max
    return drawdown.min()


def split_first3_last2(series):
    if len(series) < 60:
        return series, pd.Series([], index=series.index)
    first3 = series.iloc[:36]
    last2 = series.iloc[36:36 + 24]
    return first3, last2


def build_factor_training_data(price_history):
    rows = []
    y_vals = []

    for ticker, prices in price_history.items():
        first3, last2 = split_first3_last2(prices)
        if len(first3) < 36 or len(last2) < 24:
            continue

        r_first = first3.pct_change().dropna()
        r_last = last2.pct_change().dropna()
        if r_first.empty or r_last.empty:
            continue

        mean_ret = r_first.mean()
        vol = r_first.std()
        mdd = compute_max_drawdown(first3)

        six_months_ago = first3.index[-1] - pd.DateOffset(months=6)
        first3_last6 = first3[first3.index >= six_months_ago]
        if len(first3_last6) > 1 and first3_last6.iloc[0] > 0:
            mom6 = np.log(first3_last6.iloc[-1] / first3_last6.iloc[0])
        else:
            mom6 = 0.0

        future_return = last2.iloc[-1] / last2.iloc[0] - 1.0

        rows.append(
            {
                "ticker": ticker,
                "mean_ret": mean_ret,
                "vol": vol,
                "max_dd": mdd,
                "mom6": mom6,
            }
        )
        y_vals.append(future_return)

    if rows:
        print("Example factor row:", rows[0])

    X = pd.DataFrame(rows).set_index("ticker")
    y = pd.Series(y_vals, index=X.index)
    print(f"[FACTOR MODEL] Built training set with {len(X)} stocks.")
    return X, y


def train_factor_weights_from_history(price_history):
    X, y = build_factor_training_data(price_history)
    print(f"Using {len(X)} stocks to train statistical factor weights.")

    if len(X) < 50:
        print("Not enough training data. Skipping factor model.")
        return None, None, None

    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    y = y.loc[X.index]

    X = X.loc[:, X.std() > 0]
    if X.empty:
        print("All factor columns removed due to zero variance.")
        return None, None, None

    mins = X.min()
    maxs = X.max()
    denom = (maxs - mins)
    denom[denom == 0] = 1e-9

    X_norm = (X - mins) / denom
    X_norm = X_norm.clip(0, 1)

    lambda_ridge = 0.01
    XtX = X_norm.T @ X_norm + lambda_ridge * np.eye(X_norm.shape[1])
    Xty = X_norm.T @ y

    try:
        W = np.linalg.solve(XtX, Xty)
    except Exception as e:
        print("Ridge regression failed:", e)
        return None, None, None

    weights = pd.Series(W, index=X_norm.columns)

    print("\n[FACTOR MODEL] Trained factor weights (ridge regularized):")
    print(weights)

    return weights, mins, maxs


def compute_current_factors_for_ticker(prices):
    three_years_ago = prices.index[-1] - pd.DateOffset(years=3)
    window = prices[prices.index >= three_years_ago]

    if len(window) < 36:
        return None

    r = window.pct_change().dropna()
    if r.empty:
        return None

    mean_ret = r.mean()
    vol = r.std()
    mdd = compute_max_drawdown(window)

    six_months_ago = window.index[-1] - pd.DateOffset(months=6)
    w_last6 = window[window.index >= six_months_ago]
    if len(w_last6) > 1 and w_last6.iloc[0] > 0:
        mom6 = np.log(w_last6.iloc[-1] / w_last6.iloc[0])
    else:
        mom6 = 0.0

    return pd.Series(
        {
            "mean_ret": mean_ret,
            "vol": vol,
            "max_dd": mdd,
            "mom6": mom6,
        }
    )


def compute_factor_scores_for_universe(tickers, price_history, weights, mins, maxs):
    raw_scores = {}

    for t in tickers:
        if t not in price_history:
            continue
        series = price_history[t]
        factors = compute_current_factors_for_ticker(series)
        if factors is None:
            continue

        common = weights.index.intersection(factors.index)
        if len(common) == 0:
            continue

        denom = (maxs[common] - mins[common]).replace(0, 1e-9)
        norm = (factors[common] - mins[common]) / denom
        norm = norm.clip(0, 1)

        score = float(np.dot(norm.values, weights[common].values))
        raw_scores[t] = score

    if not raw_scores:
        return {}

    vals = np.array(list(raw_scores.values()))
    s_min, s_max = float(vals.min()), float(vals.max())

    if abs(s_max - s_min) < 1e-9:
        # avoid collapse: give everyone mid score
        return {t: 0.5 for t in raw_scores}

    factor_scores = {
        t: (raw_scores[t] - s_min) / (s_max - s_min) for t in raw_scores
    }
    return factor_scores


# ============================================================
# GROWTH PROJECTION (OPTIONAL FOR STORY) – LEFT AS IS
# ============================================================

def calculate_growth_projection(total_invested, avg_mtwb_score, years=10):
    print("\n" + "=" * 80)
    print("10-YEAR GROWTH PROJECTION TO $1.5M")
    print("=" * 80)

    expected_return = 0.08 + (avg_mtwb_score - 60) * 0.00175

    portfolio_value = total_invested
    print(f"Starting Portfolio: ${portfolio_value:,.0f}")
    print("Target by 2034: $1,500,000")
    print(f"Projected Annual Return: {expected_return:.2%}")
    print("Annual Withdrawals: $10,000 starting Year 3 (2026)")
    print("\nYear-by-Year Projection:")
    print("-" * 50)

    for year in range(1, years + 1):
        current_year = 2024 + year
        portfolio_value *= (1 + expected_return)
        if current_year >= 2026:
            portfolio_value -= 10000
        print(f"Year {current_year}: ${portfolio_value:,.0f}")

    print("-" * 50)
    if portfolio_value >= 1500000:
        print(f"✓ TARGET ACHIEVED: ${portfolio_value:,.0f}")
        print(f"  (${portfolio_value - 1500000:,.0f} above target)")
    else:
        print(f"✗ SHORTFALL: ${1500000 - portfolio_value:,.0f} below target")

    return expected_return, portfolio_value


# ============================================================
# SCREENING LOGIC
# ============================================================

def get_approved_tickers_from_excel(path):
    xls = pd.ExcelFile(path)
    tickers = []
    possible_cols = [
        "Ticker", "Symbol", "Ticker Symbol",
        "tickers", "ticker", "symbol",
    ]

    for sheet in xls.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet)
        for col in df.columns:
            if str(col).strip() in possible_cols:
                col_tickers = (
                    df[col]
                    .dropna()
                    .astype(str)
                    .str.replace(".", "-", regex=False)
                    .str.strip()
                    .str.upper()
                    .tolist()
                )
                tickers.extend(col_tickers)

    tickers = sorted(set(tickers))
    print(f"✓ Loaded {len(tickers)} raw tickers from ALL sheets")
    return tickers

def screen_all_companies_with_factors(
    all_tickers,
    price_history,
    factor_scores,
    fundamentals_cache,
    max_results=None,  # None now means include ALL
):
    print("\n" + "=" * 80)
    print("INTELLIGENT SCREENING WITH FUNDAMENTAL + FACTOR MODEL")
    print("=" * 80)

    screened = []

    for ticker in all_tickers:
        info = fundamentals_cache.get(ticker)
        if info is None:
            continue

        price = safe_val(info, "price", default=None)
        market_cap = safe_val(info, "marketCap", default=None)
        if price is None or market_cap is None:
            continue

        factor_score = factor_scores.get(ticker)
        if factor_score is None:
            continue

        scores = calculate_comprehensive_scores(info, ticker, market_cap)
        mtwb = scores["mtwb"]
        community_score = scores["community"]
        sports_score = scores["sports"]

        mtwb_norm = mtwb / 100.0
        final_score = 0.6 * mtwb_norm + 0.4 * factor_score

        screened.append(
            {
                "Ticker": ticker,
                "Company_Name": info.get("longName") or ticker,
                "Sector": info.get("sector") or "Unknown",
                "Market_Cap": market_cap,
                "Price": round(price, 2),
                "MTWB_Score": mtwb,
                "Factor_Score": round(factor_score, 3),
                "Final_Combined_Score": round(final_score, 5),
                "Community_Score": community_score,
                "Sports_Score": sports_score,
                "Philly_Connection": get_philly_connection(ticker),
                "Category": get_category(info, ticker),
            }
        )

    if not screened:
        print("❌ No companies passed screening after factor model.")
        return pd.DataFrame([])

    df = pd.DataFrame(screened).sort_values(
        "Final_Combined_Score", ascending=False
    ).reset_index(drop=True)

    # Enrich all winners (no top N limit)
    print(f"\nEnriching {len(df)} winners with FMP profile (name / sector)...")
    for i, row in df.iterrows():
        t = row["Ticker"]
        p = get_fmp_profile(t)
        if p:
            df.at[i, "Company_Name"] = p.get("longName", row["Company_Name"])
            if p.get("sector"):
                df.at[i, "Sector"] = p.get("sector")
                df.at[i, "Category"] = p.get("sector")
        time.sleep(0.05)

    print(f"\n✓ Completed screening. Total companies: {len(df)}")
    return df

# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 80)
    print("STEP 1: FUNDAMENTAL PRESCREEN WITH FMP")
    print("=" * 80)

    all_tickers = get_approved_tickers_from_excel(EXCEL_PATH)

    filtered_tickers, fundamentals_cache = run_fundamental_prescreen(all_tickers)

    if len(filtered_tickers) == 0:
        print("❌ ERROR: No stocks passed prescreen.")
        return

    filtered_sorted = sorted(
        filtered_tickers,
        key=lambda t: safe_val(fundamentals_cache[t], "marketCap", default=0),
        reverse=True,
    )
    train_tickers = filtered_sorted[:MAX_TRAIN]
    print(f"Using {len(train_tickers)} tickers for factor model.")

    print("\n" + "=" * 80)
    print("STEP 2: TRAIN FACTOR MODEL FROM 5-YEAR HISTORY")
    print("=" * 80)

    price_history = download_monthly_price_history_yf(train_tickers)
    if len(price_history) == 0:
        print("❌ ERROR: No valid price history.")
        return

    factor_weights, factor_mins, factor_maxs = train_factor_weights_from_history(
        price_history
    )
    if factor_weights is None:
        print("❌ Factor model training failed.")
        return

    factor_scores = compute_factor_scores_for_universe(
        train_tickers, price_history, factor_weights, factor_mins, factor_maxs
    )

    print("\n" + "=" * 80)
    print("STEP 3: SCREEN COMPANIES")
    print("=" * 80)

    df = screen_all_companies_with_factors(
        train_tickers,
        price_history,
        factor_scores,
        fundamentals_cache,
        max_results=FINAL_MAX_RESULTS,
    )

    if df.empty:
        print("❌ No companies passed screening.")
        return

    print(df.head(FINAL_MAX_RESULTS))
    df.to_csv("screened_stocks.csv", index=False)
    print("✓ Saved screened_stocks.csv")


if __name__ == "__main__":
    main()
