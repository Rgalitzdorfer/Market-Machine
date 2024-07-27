# Market Machine Overview
As the Co-founder of Market Machine along with Kieran Griperay, this project comprehensively backtested 3,300 stocks over 4 years of data on technical indicators, descriptive statistics, and the stock's sector performance to find a consistent edge to trade within the stock market. The early stages of Market Machine relied heavily on quality data collection where stocks were filtered by a $300 million market cap to include all small to large-cap stocks. After this, 4 years of historical data was collected on each stock through TA-Lib and Yfinance, as well as a customized sector performance indicator that tracked technical indicators from each sector ETF. Different combinations of these indicators were then used in the backtest to evaluate how the strategy would've performed over the last 1,000 trading days, showing the win rate, average percentage change, and several other metrics used to evaluate if that strategy provided a true edge in the market. If the strategy appeared profitable from the backtest, a portfolio simulation was then conducted to evaluate how a real account would have performed over the given time period. This portfolio was then plotted to visually show advanced volatility metrics, ensuring the strategy's performance would have been realistic if traded in a real account. Lastly, if the strategy was deemed profitable, a stock filtration was performed using that strategy's specific trading criteria at the end of every day in the trading market. All of these stocks are then placed into TensorFlow's LSTM Model, utilizing machine learning to identify and capitalize on some of the largest predicted moves in the stock market for the following day. Risk management varied slightly for different strategies, but one stock never exceeded more than 10% or less than 5% of the trading account. Stops and targets varied by strategy as well, but exits were typically placed once one of the indicators suggesting a trade entry had changed, indicating that the original hypothesis for entering the trade was no longer valid, suggesting an exit. Market Machine found and actively trades 3 successful strategies that outperform several traditional indexes by over 100% for YOY growth.
 
# Code Breakdown
## Libraries Used 
### Panas (DataFrame Manipulation)
### Numpy (Arrays)
### Yfinance (Historical Stock Data)
### TA-Lib (Calculate Technical Indicators)
### Backtrader (Portfolio Simulation)
### Matplotlib (Portfolio Visualization)
### Skicit-Learn (Machine Learning)
### TensorFlow (Deep Learning)
### Multiprocessing (Parallel Processing)

## Data Collection (1)
This file reads a list of NASDAQ and NYSE stocks from a CSV file, maps their sectors to corresponding ETFs, and filters them by a $300 million market cap using parallel processing. The filtered data, including tickers, sectors, market caps, and corresponding sector ETFs containing 3,300 stocks is then saved to a new CSV file.

## Individual Stock Data (2)
Individual Stock Data reads the list of small-large cap stocks saved from Data Collection, retrieves historical sector ETF data, calculates technical indicators, and identifies trending ETFs, saving this information to a CSV file. It then processes each stock, fetching historical data, calculating technical indicators, merging with ETF trend data, and saving the processed data. Each of the 3,300 stocks now has 4 years of historical data with the following indicators:

1. Average Directional Index
2. Relative Strength Index
3. Exponential Moving Averages (20, 50, 100)
4. Bollinger Bands
5. Moving Average Convergence Divergence
6. Average True Range
7. Parabolic SAR
8. Stochastic Oscillator
9. Volume
10. ETF Sector Performance (Customized)

## Backtest (3)
This code performs a backtest over the last 4 years on all 3,300 stocks by applying different buy and sell conditions from the indicators listed above, calculating the percentage change and duration for each trade, and saving the results to CSV files. It then aggregates results from all stocks, calculates descriptive statistics of strategy performance, and prints the average percentage change, average trade duration, and other evaluation metrics to provide a systematic and objective way to determine the profitability and risk attached to each strategy.

## BT Portfolio (4)
This loads the trade results from the Backtest file, pairs buy and sell trades, and creates a simplified trades Dataframe. It is then used to plot the strategy with Backtrader showing how a real investment portfolio would have performed, logging portfolio metrics, and calculating performance statistics. The backtest results, including portfolio values, are saved to a CSV file, and the portfolio performance over time is visualized and saved as a plot creating a visual that highlights max drawdown, unprofitable periods, and overall volatility to assess advanced metrics which are ultimately used to decide if a strategy creates enough of a consistent edge to be traded.

## TensorFlow Model (5)
This file uses a TensorFlow LSTM model to predict future stock prices as a validation step for the above stock filtration. It begins by downloading historical data for a list of stock tickers and calculates various technical indicators. The data is then used to build and train LSTM models for each stock, predicting future prices. These predictions are compared to current prices to determine whether to 'Buy' or 'Hold' the stock. The results, including the predicted percentage increase, are stored and the top predictions are visualized and saved. This process adds a layer of confidence to the trading strategy by providing an additional check from the backtested results using machine learning predictions.







