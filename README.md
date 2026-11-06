# Wharton-Investment-Competition-Stock-Screener

## Context
Python stock screener and portfolio optimizer: API fundamentals, ridge-regression factor model, Markowitz allocation. Built for the Wharton Global High School Investment Competition.
For the 2025-26 case competition, students were roleplaying as a consultant for the company Make the World Better and instructed to create a stock portfolio for Connor Barwin, a previous football player from Philadelphia. 

## How it works
This model has two parts: a screener and an allocator. 

The screener scans through 5,000+ of stocks and performs an initial filter, focusing on traits such as price, market cap, and ROE and filtering out foreign and OTC ticker formats, leaving 250 stocks. From there, each stock receives a score based on its quantitative financial performance and qualitative traits that align with Barwin's values. Next, a ridge regression analysis is run based on the past 5 years of monthly prices. For months 1-36, the model finds the average monthly return, volatility, max drawdown, and 6 month momentum and weights each factor. The model then finds the weights that best predict each stock's next 2 year return from its first 3 year factors, across all 250 stocks at once. For each stock, the same data points are calculated by its 3 most recent years, multiplied by the weights, and gets its predicted score. Both the scores and regression results are combined into a final score for a stock, and the top 50 scoring stocks are put into the Markowitz model which determines the allocation of them.

The allocation model uses Markowitz portfolio theory to determine how much Barwin should invest into each of the top 50 stocks. It first gathers average daily returns of a stock from the past 3 years, and puts it into the covariance matrix which measures how volatile each stock is and how interlinked its price is with another stock. Then, the model finds the weights that give the lowest total risk, where the weights add up to 100%, must be >= 0, and have a minimum expected return of 8%/yr. Each weight is multiplied by the total portfolio value of $500,000 and divided by the stock's price. Weights under 1% are dropped and share counts are also rounded down to whole shares, which is why there is some cash remaining. 
