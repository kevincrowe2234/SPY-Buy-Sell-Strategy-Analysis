import pandas as pd
import numpy as np
from itertools import product
import os
from multiprocessing import Pool, cpu_count
from functools import partial

def load_data(file_path):
    try:
        df = pd.read_csv(file_path, header=None, names=['Date', 'Price'])
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
        df['Price'] = df['Price'].replace(r'[\$,]', '', regex=True).astype(float)
        df = df.dropna().reset_index(drop=True)
        if df.empty:
            raise ValueError("DataFrame is empty after cleaning.")
        return df
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def calculate_moving_averages(df, short_window, long_window):
    df = df.copy()
    df['SMA'] = df['Price'].rolling(window=short_window).mean()
    df['LMA'] = df['Price'].rolling(window=long_window).mean()
    df['SMA_Slope'] = df['SMA'].diff()
    df['LMA_Slope'] = df['LMA'].diff()
    return df

def generate_signals(df):
    df = df.copy()
    df['Signal'] = 0
    for i in range(1, len(df)):
        prev_sma = df['SMA'].iloc[i-1]
        prev_lma = df['LMA'].iloc[i-1]
        curr_sma = df['SMA'].iloc[i]
        curr_lma = df['LMA'].iloc[i]
        sma_slope = df['SMA_Slope'].iloc[i]
        lma_slope = df['LMA_Slope'].iloc[i]
        if (prev_sma <= prev_lma and curr_sma > curr_lma and sma_slope >= 0 and lma_slope >= 0):
            df.loc[i, 'Signal'] = 1
        elif (prev_sma >= prev_lma and curr_sma < curr_lma and sma_slope <= 0 and lma_slope <= 0):
            df.loc[i, 'Signal'] = -1
    return df

def calculate_wealth(df, initial_investment=10000):
    cash = initial_investment
    shares = 0
    position = 0
    trades = []
    for i in range(len(df)):
        signal = df['Signal'].iloc[i]
        price = df['Price'].iloc[i]
        if signal == 1 and position == 0:
            shares = cash / price
            cash = 0
            position = 1
            trades.append(('Buy', df['Date'].iloc[i], price))
        elif signal == -1 and position == 1:
            cash = shares * price
            shares = 0
            position = 0
            trades.append(('Sell', df['Date'].iloc[i], price))
    if position == 1:
        cash = shares * df['Price'].iloc[-1]
        trades.append(('Sell', df['Date'].iloc[-1], df['Price'].iloc[-1]))
    final_wealth = cash
    return final_wealth, trades

def evaluate_combination(params, df, initial_investment):
    short_window, long_window = params
    if short_window >= long_window:
        return short_window, long_window, 0, []
    df_copy = df.copy()
    df_copy = calculate_moving_averages(df_copy, short_window, long_window)
    df_copy = df_copy.dropna().reset_index(drop=True)
    if df_copy.empty:
        return short_window, long_window, 0, []
    df_copy = generate_signals(df_copy)
    final_wealth, trades = calculate_wealth(df_copy, initial_investment)
    return short_window, long_window, final_wealth, trades

def optimize_moving_averages(df, short_windows, long_windows, initial_investment=10000):
    best_wealth = 0
    best_short_window = 0
    best_long_window = 0
    best_trades = []
    combinations = list(product(short_windows, long_windows))
    func = partial(evaluate_combination, df=df, initial_investment=initial_investment)
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(func, combinations)
    for short_window, long_window, wealth, trades in results:
        if wealth > best_wealth:
            best_wealth = wealth
            best_short_window = short_window
            best_long_window = long_window
            best_trades = trades
    return best_short_window, best_long_window, best_wealth, best_trades

def main():
    file_path = r"c:\Users\Kcrow\OneDrive\Documents\SPY Buy Sell Strategy Analysis\SPY Price History.csv"
    df = load_data(file_path)
    if df is None:
        print("Error: Could not load data from CSV. Please check the file format and contents.")
        return
    short_windows = range(5, 101, 1)
    long_windows = range(20, 201, 1)
    initial_investment = 10000
    def annualized_gain(initial, final, start_date, end_date):
        num_days = (end_date - start_date).days
        num_years = num_days / 365.25
        if num_years > 0:
            return ((final / initial) ** (1 / num_years) - 1) * 100
        else:
            return 0.0

    try:
        # Calculate annualized gain % and number of trades for all (short, long) combinations
        gain_table = pd.DataFrame(index=long_windows, columns=short_windows, dtype=float)
        trade_count_table = pd.DataFrame(index=long_windows, columns=short_windows, dtype=int)
        # Cache for optimization: (short, long) -> (final_wealth, trades)
        result_cache = {}
        for long_ma in long_windows:
            for short_ma in short_windows:
                if short_ma >= long_ma:
                    gain_table.at[long_ma, short_ma] = np.nan
                    trade_count_table.at[long_ma, short_ma] = np.nan
                    continue
                df_tmp = calculate_moving_averages(df.copy(), short_ma, long_ma)
                df_tmp = df_tmp.dropna().reset_index(drop=True)
                if df_tmp.empty:
                    gain_table.at[long_ma, short_ma] = np.nan
                    trade_count_table.at[long_ma, short_ma] = np.nan
                    result_cache[(short_ma, long_ma)] = (0, [])
                    continue
                df_tmp = generate_signals(df_tmp)
                final_wealth, trades = calculate_wealth(df_tmp, initial_investment)
                if trades:
                    first_trade_date = trades[0][1]
                    last_trade_date = trades[-1][1]
                    annualized_pct = annualized_gain(initial_investment, final_wealth, first_trade_date, last_trade_date)
                    trade_count = len(trades)
                else:
                    annualized_pct = np.nan
                    trade_count = np.nan
                gain_table.at[long_ma, short_ma] = annualized_pct
                trade_count_table.at[long_ma, short_ma] = trade_count
                result_cache[(short_ma, long_ma)] = (final_wealth, trades)
        # Save the tables to CSV
        gain_table.to_csv("annualized_gain_table.csv")
        trade_count_table.to_csv("trade_count_table.csv")

        # Find optimum using cached results
        best_wealth = 0
        best_short = 0
        best_long = 0
        best_trades = []
        for (short_ma, long_ma), (wealth, trades) in result_cache.items():
            if wealth > best_wealth:
                best_wealth = wealth
                best_short = short_ma
                best_long = long_ma
                best_trades = trades
        start_date = df['Date'].iloc[0]
        end_date = df['Date'].iloc[-1]
        buy_price = df['Price'].iloc[0]
        sell_price = df['Price'].iloc[-1]
        buy_and_hold_profit = (sell_price - buy_price) * (initial_investment / buy_price)
        buy_and_hold_final = initial_investment + buy_and_hold_profit
        annualized_profit_pct = annualized_gain(initial_investment, buy_and_hold_final, start_date, end_date)

        # Buy and hold using first buy and last sell signals
        df_signals = calculate_moving_averages(df.copy(), best_short, best_long)
        df_signals = df_signals.dropna().reset_index(drop=True)
        df_signals = generate_signals(df_signals)
        buy_indices = df_signals.index[df_signals['Signal'] == 1].tolist()
        sell_indices = df_signals.index[df_signals['Signal'] == -1].tolist()
        if buy_indices and sell_indices:
            first_buy_idx = buy_indices[0]
            last_sell_idx = sell_indices[-1] if sell_indices[-1] > first_buy_idx else buy_indices[-1]
            buy_date = df_signals['Date'].iloc[first_buy_idx]
            buy_price_signal = df_signals['Price'].iloc[first_buy_idx]
            sell_date = df_signals['Date'].iloc[last_sell_idx]
            sell_price_signal = df_signals['Price'].iloc[last_sell_idx]
            shares_signal = initial_investment / buy_price_signal
            final_wealth_signal = shares_signal * sell_price_signal
            profit_signal = final_wealth_signal - initial_investment
            annualized_signal_pct = annualized_gain(initial_investment, final_wealth_signal, buy_date, sell_date)
        else:
            buy_date = sell_date = None
            buy_price_signal = sell_price_signal = None
            final_wealth_signal = profit_signal = annualized_signal_pct = 0.0

        print(f"Start Date: {start_date.strftime('%Y-%m-%d')}")
        print(f"End Date: {end_date.strftime('%Y-%m-%d')}")
        print(f"Buy and Hold (Full Period):")
        print(f"  Profit: ${buy_and_hold_profit:.2f}")
        print(f"  Final Wealth: ${buy_and_hold_final:.2f}")
        print(f"  Annualized % Gain: {annualized_profit_pct:.2f}%")
        if buy_date and sell_date:
            print(f"Buy and Hold (First Buy/Last Sell Signals):")
            print(f"  Buy on {buy_date.strftime('%Y-%m-%d')} at ${buy_price_signal:.2f}")
            print(f"  Sell on {sell_date.strftime('%Y-%m-%d')} at ${sell_price_signal:.2f}")
            print(f"  Profit: ${profit_signal:.2f}")
            print(f"  Final Wealth: ${final_wealth_signal:.2f}")
            print(f"  Annualized % Gain: {annualized_signal_pct:.2f}%")
        else:
            print("Buy and Hold (First Buy/Last Sell Signals): Not enough signals to compute.")

        # Optimized strategy annualized % profit
        if best_trades:
            first_trade_date = best_trades[0][1]
            last_trade_date = best_trades[-1][1]
            annualized_best_pct = annualized_gain(initial_investment, best_wealth, first_trade_date, last_trade_date)
        else:
            annualized_best_pct = 0.0

        print(f"Best Short MA Window: {best_short} days")
        print(f"Best Long MA Window: {best_long} days")
        print(f"Final Wealth: ${best_wealth:.2f}")
        print(f"Profit: ${best_wealth - initial_investment:.2f}")
        print(f"Annualized % Gain (Optimized Strategy): {annualized_best_pct:.2f}%")
        print(f"Number of Trades: {len(best_trades)}")
        print("\nTrade Log:")
        for trade in best_trades:
            action, date, price = trade
            print(f"{action} on {date.strftime('%Y-%m-%d')} at ${price:.2f}")

        # Custom MA case: short=50, long=150
        custom_short = 50
        custom_long = 150
        df_custom = df.copy()
        df_custom = calculate_moving_averages(df_custom, custom_short, custom_long)
        df_custom = df_custom.dropna().reset_index(drop=True)
        if not df_custom.empty:
            df_custom = generate_signals(df_custom)
            custom_wealth, custom_trades = calculate_wealth(df_custom, initial_investment)
            if custom_trades:
                custom_first_trade_date = custom_trades[0][1]
                custom_last_trade_date = custom_trades[-1][1]
                annualized_custom_pct = annualized_gain(initial_investment, custom_wealth, custom_first_trade_date, custom_last_trade_date)
            else:
                annualized_custom_pct = 0.0
            print(f"\n--- Custom MA Case: Short={custom_short} Long={custom_long} ---")
            print(f"Final Wealth: ${custom_wealth:.2f}")
            print(f"Profit: ${custom_wealth - initial_investment:.2f}")
            print(f"Annualized % Gain (Custom MA): {annualized_custom_pct:.2f}%")
            print(f"Number of Trades: {len(custom_trades)}")
            print("Trade Log:")
            for trade in custom_trades:
                action, date, price = trade
                print(f"{action} on {date.strftime('%Y-%m-%d')} at ${price:.2f}")
    except Exception as e:
        print(f"Error during optimization: {e}")

if __name__ == "__main__":
    main()
   