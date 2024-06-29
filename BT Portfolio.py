#Import Libraries
import pandas as pd #Data Manipulation
import backtrader as bt #Backtesting
import matplotlib.pyplot as plt #Visualizations
from collections import defaultdict #Managing Active Trades

#Load Results
file_path = '/Users/ryangalitzdorfer/Downloads/Market Machine/Stock Filtration/Backtest/Total Results/All_Results.csv'  
all_results_df = pd.read_csv(file_path)  
all_results_df['Date'] = pd.to_datetime(all_results_df['Date']) #Convert to Datetime Format
all_results_df.sort_values(by=['Stock', 'Date'], inplace=True)  #Sorting the DataFrame by 'Stock' & 'Date'

# Initialize a list to hold paired trades
paired_trades = []  # List to store paired buy and sell trades

# Dictionary to hold buys until a corresponding sell is found
open_trades = {}  # Dictionary to keep track of open buy trades

# Iterate through each row in the DataFrame
for index, row in all_results_df.iterrows():  # Looping through each row of the DataFrame
    stock = row['Stock']  # Getting the stock name
    signal = row['Signal']  # Getting the signal (Buy/Sell)
    
    if signal == 'Buy':  # If the signal is a Buy
        open_trades[stock] = row  # Store the buy information in open_trades
    elif signal == 'Sell' and stock in open_trades:  # If the signal is a Sell and there is an open buy trade for this stock
        buy = open_trades.pop(stock)  # Remove the buy trade from open_trades
        
        # Create a paired trade dictionary
        paired_trade = {  # Dictionary to store paired trade details
            'Stock': stock,
            'Buy_Date': buy['Date'],
            'Buy_Price': buy['Price'],
            'Sell_Date': row['Date'],
            'Sell_Price': row['Price'],
            'Percentage_Change': row['Percentage Change']
        }
        
        paired_trades.append(paired_trade)  # Append the paired trade to paired_trades

# Convert paired trades to a DataFrame
simplified_trades_df = pd.DataFrame(paired_trades)  # Creating a DataFrame from the paired trades list
simplified_trades_df.sort_values(by='Buy_Date', inplace=True)  # Sorting the DataFrame by 'Buy_Date'
simplified_trades_df['Buy_Date'] = pd.to_datetime(simplified_trades_df['Buy_Date'])  # Ensuring 'Buy_Date' is in datetime format
simplified_trades_df['Sell_Date'] = pd.to_datetime(simplified_trades_df['Sell_Date'])  # Ensuring 'Sell_Date' is in datetime format
# Save the simplified trades DataFrame
save_path = '/Users/ryangalitzdorfer/Downloads/Market Machine/Stock Filtration/BT Portfolio/Simplified_Trades.csv'  # Path to save the simplified trades
simplified_trades_df.to_csv(save_path, index=False)  # Saving the simplified trades DataFrame to a CSV file

print(f"Simplified trades DataFrame saved to: {save_path}")  # Printing the save path

# Backtrader setup
import backtrader as bt  # Importing backtrader library
import pandas as pd  # Importing pandas library
import matplotlib.pyplot as plt  # Importing matplotlib library
from collections import defaultdict  # Importing defaultdict from collections module

# Assume simplified_trades_df is already defined and cleaned
start_date = simplified_trades_df['Buy_Date'].min()  # Getting the earliest buy date
end_date = simplified_trades_df['Sell_Date'].max()  # Getting the latest sell date

class SimplifiedTradeStrategy(bt.Strategy):  # Defining a custom trading strategy
    def __init__(self):  # Initialization method
        self.active_trades = defaultdict(list)  # Dictionary to keep track of active trades
        self.portfolio_values = []  # List to store portfolio values over time
        self.initial_value = self.broker.getvalue()  # Initial portfolio value
        self.total_return = 0  # Total return of the portfolio
        self.total_percentage_change = 0  # Total percentage change of the portfolio
        self.total_trades = 0  # Total number of trades executed
        self.daily_returns = []  # List to store daily returns
        self.active_trades_count = 0  # Number of active trades
        self.buy_count = 0  # Count of buy trades
        self.sell_count = 0  # Count of sell trades

    def log(self, txt, dt=None):  # Method to log messages
        """ Logging function for this strategy"""
        dt = dt or self.datas[0].datetime.date(0)  # Getting the current date
        print(f'{dt.isoformat()} - {txt}')  # Printing the log message

    def next(self):  # Method called for each bar of data
        current_date = self.datas[0].datetime.date(0)  # Getting the current date
        portfolio_value = self.broker.getvalue()  # Getting the current portfolio value
        cash = self.broker.get_cash()  # Getting the current cash balance
        active_trades_count = sum(len(trades) for trades in self.active_trades.values())  # Counting the number of active trades

        self.log(f"Date: {current_date}, Portfolio Value: {portfolio_value:.2f}, Cash: {cash:.2f}, Active Trades: {active_trades_count}")  # Logging portfolio details

        # Reset daily buy and sell counts
        self.buy_count = 0  # Resetting buy count for the day
        self.sell_count = 0  # Resetting sell count for the day

        # Process Buy Trades
        buy_rows = simplified_trades_df[simplified_trades_df['Buy_Date'] == pd.Timestamp(current_date)]  # Getting rows with buy trades for the current date
        for index, row in buy_rows.iterrows():  # Looping through each buy trade
            size = 100  # Assuming a fixed trade size
            self.active_trades[row['Stock']].append({'buy_price': row['Buy_Price'], 'size': size, 'buy_date': current_date})  # Adding the trade to active trades
            self.buy_count += 1  # Incrementing buy count
            self.log(f"Opening trade for {row['Stock']} at {row['Buy_Price']} with size {size}")  # Logging the buy trade

        # Process Sell Trades
        sell_rows = simplified_trades_df[simplified_trades_df['Sell_Date'] == pd.Timestamp(current_date)]  # Getting rows with sell trades for the current date
        closed_trades = []  # List to store closed trades
        total_return = 0  # Resetting total return for the day
        for index, row in sell_rows.iterrows():  # Looping through each sell trade
            stock = row['Stock']  # Getting the stock name
            if stock in self.active_trades:  # Checking if there are active trades for the stock
                for trade in self.active_trades[stock]:  # Looping through each active trade
                    buy_price = trade['buy_price']  # Getting the buy price
                    size = trade['size']  # Getting the trade size
                    sell_price = row['Sell_Price']  # Getting the sell price
                    trade_return = (sell_price - buy_price) / buy_price  # Calculating the trade return
                    total_return += trade_return * size  # Adding the trade return to total return
                    self.total_return += trade_return * size  # Adding the trade return to total return for the strategy
                    self.total_percentage_change += trade_return * 100  # Adding the percentage change to total percentage change
                    self.total_trades += 1  # Incrementing the total trades count
                    self.sell_count += 1  # Incrementing the sell count
                    closed_trades.append(trade)  # Adding the trade to closed trades
                    self.log(f"Closing trade for {stock} at {sell_price} with return {trade_return:.2%}")  # Logging the sell trade

                # Remove closed trades
                self.active_trades[stock] = [trade for trade in self.active_trades[stock] if trade not in closed_trades]  # Removing closed trades from active trades
                if not self.active_trades[stock]:  # Checking if there are no active trades left for the stock
                    del self.active_trades[stock]  # Removing the stock from active trades

        # Calculate Daily Portfolio Return
        daily_return = 0  # Initializing daily return
        if active_trades_count > 0:  # Checking if there are active trades
            daily_return = total_return / self.initial_value  # Calculating daily return
            self.daily_returns.append(daily_return)  # Adding the daily return to daily returns

        # Update Portfolio Value
        new_portfolio_value = portfolio_value * (1 + daily_return)  # Calculating new portfolio value
        self.broker.set_cash(cash * (1 + daily_return))  # Updating cash balance
        percentage_change = (new_portfolio_value - portfolio_value) / portfolio_value * 100  # Calculating percentage change

        # Append portfolio metrics
        self.portfolio_values.append(
            (current_date, new_portfolio_value, active_trades_count, self.buy_count, self.sell_count, round(percentage_change, 2))
        )  # Appending portfolio metrics

    def stop(self):  # Method called at the end of the backtest
        # Calculate average percentage change per trade
        average_percentage_change = self.total_percentage_change / self.total_trades if self.total_trades > 0 else 0  # Calculating average percentage change

        # Calculate cumulative return
        final_value = self.broker.getvalue()  # Getting the final portfolio value
        total_return = (final_value - self.initial_value) / self.initial_value * 100  # Calculating cumulative return

        # Calculate maximum drawdown
        portfolio_values_df = pd.DataFrame(self.portfolio_values, columns=['Date', 'Portfolio_Value', 'Active_Trades', 'Buys', 'Sells', 'Daily_Percentage_Change'])  # Creating a DataFrame from portfolio values
        portfolio_values_df.set_index('Date', inplace=True)  # Setting the date as the index
        rolling_max = portfolio_values_df['Portfolio_Value'].cummax()  # Calculating the rolling maximum portfolio value
        drawdown = portfolio_values_df['Portfolio_Value'] / rolling_max - 1  # Calculating drawdown
        max_drawdown = drawdown.min() * 100  # Calculating the maximum drawdown in percentage

        print(f"Total Trades: {self.total_trades}")  # Printing the total number of trades
        print(f"Average Percentage Change per Trade: {average_percentage_change:.2f}%")  # Printing the average percentage change per trade
        print(f"Total Return: {total_return:.2f}%")  # Printing the cumulative return
        print(f"Maximum Drawdown: {max_drawdown:.2f}%")  # Printing the maximum drawdown

# Function to create and add data to Cerebro
def add_data_to_cerebro(cerebro, stock):  # Function to add stock data to the backtest engine
    stock_data = simplified_trades_df[simplified_trades_df['Stock'] == stock]  # Filtering data for the specific stock
    stock_data = stock_data[['Buy_Date', 'Buy_Price']].set_index('Buy_Date').rename(columns={'Buy_Price': 'close'})  # Preparing the data
    stock_data = stock_data.reindex(pd.date_range(start=start_date, end=end_date, freq='B')).ffill()  # Reindexing and forward-filling missing data
    stock_data['open'] = stock_data['close']  # Setting open prices
    stock_data['high'] = stock_data['close']  # Setting high prices
    stock_data['low'] = stock_data['close']  # Setting low prices
    stock_data['volume'] = 1000  # Dummy volume data
    stock_data['openinterest'] = 0  # Dummy open interest data

    # Create Data Feed
    data_feed = bt.feeds.PandasData(dataname=stock_data)  # Creating a data feed for backtrader
    cerebro.adddata(data_feed, name=stock)  # Adding the data feed to cerebro

# Create a cerebro instance
cerebro = bt.Cerebro()  # Creating a backtrader engine instance

# Add strategy to cerebro
cerebro.addstrategy(SimplifiedTradeStrategy)  # Adding the custom strategy to cerebro

# Add data for each stock
for stock in simplified_trades_df['Stock'].unique():  # Looping through each unique stock
    add_data_to_cerebro(cerebro, stock)  # Adding the stock data to cerebro

# Set starting cash
cerebro.broker.set_cash(100000.0)  # Setting the starting cash for the backtest

# Run the backtest
results = cerebro.run()  # Running the backtest
strategy = results[0]  # Extracting the strategy instance from the results

# Extract the portfolio values with new columns
portfolio_values = pd.DataFrame(
    strategy.portfolio_values, 
    columns=['Date', 'Portfolio_Value', 'Active_Trades', 'Buys', 'Sells', 'Daily_Percentage_Change']
)  # Creating a DataFrame from the portfolio values
portfolio_values.set_index('Date', inplace=True)  # Setting the date as the index

# Save the portfolio values to a CSV file
output_path = '/Users/ryangalitzdorfer/Downloads/Market Machine/Stock Filtration/BT Portfolio/Portfolio_Values.csv'  # Path to save the portfolio values
portfolio_values.to_csv(output_path)  # Saving the portfolio values to a CSV file
print(f"Portfolio values saved to: {output_path}")  # Printing the save path

# Plot the portfolio performance
plt.figure(figsize=(10, 6))  # Setting the figure size for the plot
plt.plot(portfolio_values.index, portfolio_values['Portfolio_Value'], label='Portfolio Value')  # Plotting the portfolio value over time
plt.xlabel('Date')  # Setting the x-axis label
plt.ylabel('Portfolio Value ($)')  # Setting the y-axis label
plt.title('Portfolio Performance Over Time')  # Setting the plot title
plt.legend()  # Adding a legend to the plot
plt.grid()  # Adding a grid to the plot

# Save the plot
plot_path = '/Users/ryangalitzdorfer/Downloads/Market Machine/Stock Filtration/BT Portfolio/Portfolio_Performance.png'  # Path to save the plot
plt.savefig(plot_path)  # Saving the plot to a file
print(f"Portfolio performance plot saved to: {plot_path}")  # Printing the save path

plt.show()  # Displaying the plot
