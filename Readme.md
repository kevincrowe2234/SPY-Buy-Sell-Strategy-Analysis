# SPY Buy/Sell Strategy Analysis

This software analyzes historical SPY (S&P 500 ETF) price data to evaluate and optimize moving average-based trading strategies. It provides:

- **Moving Average Optimization:** Searches for the best short and long moving average durations to maximize final wealth using a buy/sell signal strategy.
- **Annualized Gain Table:** Generates a CSV file (`annualized_gain_table.csv`) showing the annualized percentage gain for each (short, long) moving average combination.
- **Trade Count Table:** Generates a CSV file (`trade_count_table.csv`) showing the number of trades for each (short, long) moving average combination.
- **Buy-and-Hold Analysis:** Reports results for standard buy-and-hold and buy/sell signal-based buy-and-hold, including annualized percentage gain.
- **Custom MA Case:** Evaluates a user-defined moving average pair (e.g., short=50, long=150).

## Features
- Loads and cleans SPY price history from CSV
- Calculates moving averages and generates buy/sell signals
- Simulates trading and calculates final wealth, profit, and annualized gain
- Optimizes moving average durations over user-defined ranges
- Outputs results to terminal and CSV files for further analysis

## Usage
1. Place your SPY price history CSV file in the project folder.
2. Run `SPY.py` to perform the analysis and generate output files.
3. Review the terminal output and CSV files for strategy performance details.

## Output Files
- `annualized_gain_table.csv`: Annualized % gain for each (short, long) MA combination
- `trade_count_table.csv`: Number of trades for each (short, long) MA combination

## Requirements
- Python 3.7+
- pandas, numpy

## Customization
- Adjust the short and long moving average ranges in `SPY.py` as needed
- Modify the initial investment or custom MA case parameters for different scenarios

---
For questions or improvements, please contact the author or open an issue.
