# Wharton-Investment-Competition-Stock-Screener

## Context
Python stock screener and portfolio optimizer: API fundamentals, ridge-regression factor model, Markowitz allocation. Built for the Wharton Global High School Investment Competition.
For the 2025-26 case competition, students were roleplaying as a consultant for the company Make the World Better and instructed to create a stock portfolio for Connor Barwin, a previous football player from Philadelphia. 

## Methods
This model has two parts: a screener and an allocator. 

The screener scans through 5,000+ stocks and performs an initial filter, focusing on traits such as price, market cap, and ROE and filtering out foreign and OTC ticker formats, leaving up to 250 stocks. From there, each stock receives a score based on its quantitative financial performance and qualitative traits that align with Barwin's values. Next, a ridge regression analysis is run based on the past 5 years of monthly prices. For months 1-36, the model finds the average monthly return, volatility, max drawdown, and 6 month momentum. The model then finds the weights that best predict each stock's next 2 year return from its first 3 year factors, across all 250 stocks at once. For each stock, the same data points are calculated by its 3 most recent year and multiplied by the weights to get its predicted score. Both the scores and regression results are combined into a final score for a stock, and the top 50 scoring stocks are put into the Markowitz model which determines the allocation of them.

The allocation model uses Markowitz portfolio theory to determine how much Barwin should invest into each of the top 50 stocks. It first gathers average daily returns of a stock from the past 3 years, and puts it into the covariance matrix which measures how volatile each stock is and how interlinked its price is with another stock. Then, the model finds the weights that give the lowest total risk, where the weights add up to 100%, must be >= 0, and have a minimum expected return of 8%/yr. Weights under 1% are dropped before each weight is multiplied by the total portfolio value of $500,000 and divided by the stock's price. Note that share counts are also rounded down to whole shares, which is why there is some cash remaining. 

## Results 

Ultimately there were 15 stocks in the portfolio. $497,159 was invested out of the $500,000 due to the remaining cash. 

The final results can be found here: https://docs.google.com/spreadsheets/d/146H4YLEB6lLGWOUzxvIHWqphVSH1wNEnMtfPNydLz7A/edit?usp=sharing


## Limitations 

Right now the factor model only learns from stocks that pass today's filter, meaning that companies that did badly over the past five years are filtered out. This leads to an inherent survivorship bias. Some steps forward would be to train the model on stocks that later failed. 

Additionally, there is no cap on a single stock and official qualitative scoring system is still in beta. 

## How to run
Requires Python 3.9+ and a paid FMP subscription https://site.financialmodelingprep.com/ 

1. **Install the packages**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set your API key and ticker list**
   ```bash
   export FMP_API_KEY="your-key-here"
   export TICKER_LIST_PATH="path/to/approved_tickers.xlsx"
   ```
   The approved ticker list was provided by the competition and isn't included here.

3. **Run the screener** (takes a while; it makes several hundred API calls)
   ```bash
   python screener.py
   ```
   Output: `screened_stocks.csv`

4. **Run the allocator**
   ```bash
   python mpt.py
   ```
   Output: `allocation.csv` and `allocation_with_shares.csv`
