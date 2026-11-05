import cvxpy as cp
import numpy as np
import pandas as pd
import yfinance as yf

# ============================================
# CONFIG
# ============================================
TOTAL_INVESTMENT = 500000   # <-- FIXED TO $500,000
MIN_WEIGHT = 0.01           # 1% minimum portfolio weight after optimization


# ============================================
# HELPER FUNCTIONS
# ============================================

def allocation_table(tickers, weights, total_investment):
    dollars = weights * total_investment
    return pd.DataFrame({
        "Ticker": tickers,
        "Weight": weights,
        "Dollar_Amount": dollars
    })


def weights_to_integer_shares(tickers, weights, last_prices, total_investment):
    """
    Convert continuous weights into whole shares.
    """
    weights = np.asarray(weights)
    last_prices = np.asarray(last_prices)

    # Target allocation
    target_dollars = weights * total_investment

    # Whole-share conversion
    shares = np.floor(target_dollars / last_prices).astype(int)

    # Actual invested dollars after flooring
    invested = shares * last_prices
    total_invested = invested.sum()

    # Realized weight after rounding
    realized_weights = invested / total_invested if total_invested > 0 else np.zeros_like(invested)

    df = pd.DataFrame({
        "Ticker": tickers,
        "Price": last_prices,
        "Target_Weight": weights,
        "Shares": shares,
        "Invested_Dollars": invested,
        "Realized_Weight": realized_weights
    })

    print(f"\nTotal Intended Investment: ${total_investment:,.2f}")
    print(f"Total Actually Invested: ${total_invested:,.2f}")
    print(f"Uninvested Cash: ${total_investment - total_invested:,.2f}")

    return df


# ============================================
# MARKOWITZ OPTIMIZATION
# ============================================

def markowitz_min_variance(cov_matrix):
    n = cov_matrix.shape[0]
    w = cp.Variable(n)

    obj = cp.Minimize(cp.quad_form(w, cov_matrix))
    cons = [cp.sum(w) == 1, w >= 0]

    prob = cp.Problem(obj, cons)
    prob.solve()

    w_val = np.array(w.value).flatten()
    w_val[w_val < 0] = 0
    return w_val / w_val.sum()


def markowitz_target_return(cov_matrix, mu, target):
    n = cov_matrix.shape[0]
    w = cp.Variable(n)

    cons = [
        cp.sum(w) == 1,
        w >= 0,
        mu @ w >= target
    ]

    prob = cp.Problem(cp.Minimize(cp.quad_form(w, cov_matrix)), cons)
    prob.solve()

    if w.value is None:
        raise RuntimeError("Target return infeasible")

    w_val = np.array(w.value).flatten()
    w_val[w_val < 0] = 0
    return w_val / w_val.sum()


def post_process_weights(weights, min_weight=MIN_WEIGHT):
    w = np.array(weights)
    w[w < min_weight] = 0
    return w / w.sum() if w.sum() > 0 else weights


def allocate_portfolio(tickers, cov_matrix, mu=None, target_return=None):
    if target_return is None:
        raw = markowitz_min_variance(cov_matrix)
    else:
        raw = markowitz_target_return(cov_matrix, mu, target_return)

    cleaned = post_process_weights(raw, MIN_WEIGHT)
    return pd.DataFrame({"Ticker": tickers, "Weight": cleaned})


# ============================================
# MAIN EXECUTION
# ============================================

def run_mpt_from_screened(
        screened_csv,
        period="3y",
        interval="1d",
        target_return=None,
        allocation_output="allocation.csv",
        total_investment=TOTAL_INVESTMENT):

    print("\n=========== LOADING SCREENED STOCKS ===========")
    df = pd.read_csv(screened_csv)
    tickers = df["Ticker"].tolist()
    print(f"Loaded {len(tickers)} stocks")

    print("\n=========== DOWNLOADING PRICE HISTORY ===========")
    prices = yf.download(tickers=tickers, period=period, interval=interval)["Close"].dropna()
    print(f"Prices shape: {prices.shape}")

    print("\n=========== COMPUTING RETURNS ===========")
    returns = prices.pct_change().dropna()
    cov_matrix = returns.cov().values
    mu = returns.mean().values

    print("\n=========== RUNNING MARKOWITZ OPTIMIZATION ===========")
    allocation_df = allocate_portfolio(
        tickers=tickers,
        cov_matrix=cov_matrix,
        mu=mu,
        target_return=target_return
    )

    print("\n=========== SAVING CONTINUOUS WEIGHTS ===========")
    allocation_df.to_csv(allocation_output, index=False)
    print(f"Saved: {allocation_output}")

    print("\n=========== CONVERTING TO WHOLE SHARES ===========")
    last_prices = prices.iloc[-1][tickers].values
    weights = allocation_df["Weight"].values

    shares_df = weights_to_integer_shares(
        tickers,
        weights,
        last_prices,
        total_investment
    )

    # Remove zero-share positions
    shares_df = shares_df[shares_df["Shares"] > 0].reset_index(drop=True)

    shares_output = "allocation_with_shares.csv"
    shares_df.to_csv(shares_output, index=False)
    print(f"Saved: {shares_output}")

    print("\n=========== FINAL WHOLE-SHARE PORTFOLIO ===========")
    print(shares_df)

    return allocation_df, shares_df


# ============================================
# RUN SCRIPT
# ============================================

if __name__ == "__main__":
    target_annual = 0.08
    target_daily = (1 + target_annual)**(1/252) - 1

    run_mpt_from_screened(
        "screened_stocks.csv",
        target_return=target_daily,
        total_investment=TOTAL_INVESTMENT
    )
