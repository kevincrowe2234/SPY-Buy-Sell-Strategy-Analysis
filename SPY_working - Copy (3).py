import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import tkinter as tk
from tkinter import messagebox, filedialog, ttk, scrolledtext
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from datetime import datetime, timedelta
import numpy as np
import os

# Global variables
all_trades = []  # Store all trades for the detailed view
cancel_backtest = False  # Flag to cancel running backtest

def read_csv(file_path):
    try:
        print(f"Reading CSV file: {file_path}")
        
        # Looking at the file, we have "DD/MM/YYYY, $PRICE" format with no header
        # Using a custom reading approach specifically for this file
        df = pd.read_csv(file_path, header=None, names=['Date', 'Price'], sep=',', engine='python')
        print(f"Read data with shape: {df.shape}")
        print(f"First few rows: {df.head().to_string()}")
        
        # Clean the price column - remove $ and any spaces
        df['Price'] = df['Price'].str.strip().str.replace('$', '').str.replace(' ', '').astype(float)
        print(f"Price column cleaned: {df['Price'].head().tolist()}")
        
        # Convert dates - first try day/month/year (European format)
        try:
            df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
            print("Date conversion successful with day/month/year format")
        except:
            # If that fails, try month/day/year (US format)
            try:
                df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
                print("Date conversion successful with month/day/year format")
            except Exception as e:
                print(f"Date conversion error: {e}")
                # Last resort - try the default parser
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                print("Using default date parser")
        
        # Check if we have any NaT values after date conversion
        if df['Date'].isna().any():
            print(f"Warning: {df['Date'].isna().sum()} rows have invalid dates")
            
        # Sorting data by date
        df = df.sort_values('Date')
        print(f"Data sorted by date. Range: {df['Date'].min()} to {df['Date'].max()}")
        
        return df
    except Exception as e:
        messagebox.showerror("Error", f"Error reading CSV file: {e}")
        print(f"Error in read_csv: {e}")
        return None
        
        # Sorting data by date
        df = df.sort_values('Date')
        print(f"Data sorted by date. Range: {df['Date'].min()} to {df['Date'].max()}")
        
        return df
    except Exception as e:
        messagebox.showerror("Error", f"Error reading CSV file: {e}")
        print(f"Error in read_csv: {e}")
        return None
        
        # Sorting data by date
        df = df.sort_values('Date')
        
        # Drop any rows with invalid dates
        df = df.dropna(subset=['Date'])
        
        return df
    except Exception as e:
        messagebox.showerror("Error", f"Error reading CSV file: {e}")
        return None

def calculate_moving_averages(df, short_period, long_period):
    df_copy = df.copy()
    df_copy['Short_MA'] = df_copy['Price'].rolling(window=short_period).mean()
    df_copy['Long_MA'] = df_copy['Price'].rolling(window=long_period).mean()
    return df_copy

def calculate_slopes(df, lookback=3):
    """Calculate the slopes of Short_MA and Long_MA using vectorized operations for better performance."""
    # Create new columns for slopes with default value 0
    df['Short_MA_Slope'] = 0.0
    df['Long_MA_Slope'] = 0.0
    
    # Need at least lookback+1 points to calculate a meaningful slope
    if len(df) <= lookback:
        return df
    
    # Helper function to calculate slopes efficiently
    def calculate_slope_vector(series, window):
        # Handle missing values by returning 0 for those positions
        result = np.zeros(len(series))
        # Create a 2D array with rolling windows
        # This is faster than looping through each point
        valid_mask = ~pd.isna(series)
        
        if valid_mask.sum() <= window:
            return result
            
        # Use only valid data for calculations
        valid_series = series[valid_mask].values
        # Prepare the x values (just indices)
        x = np.arange(window)
        
        # Iterate through each window but much less frequently
        for i in range(window, len(valid_series)):
            y = valid_series[i-window:i]
            # Use np.polyfit directly on the window - faster than full output
            if len(y) == window and not np.isnan(y).any():
                slope = np.polyfit(x, y, 1)[0]
                # Map back to original series positions
                orig_idx = valid_mask.index[valid_mask][i]
                result[orig_idx] = slope
                
        return result
        
    # Calculate slopes only for non-NaN values (much faster)
    short_ma_series = df['Short_MA']
    long_ma_series = df['Long_MA']
    
    # Only calculate slopes if we have enough valid data
    if (~pd.isna(short_ma_series)).sum() > lookback:
        df['Short_MA_Slope'] = calculate_slope_vector(short_ma_series, lookback)
    
    if (~pd.isna(long_ma_series)).sum() > lookback:
        df['Long_MA_Slope'] = calculate_slope_vector(long_ma_series, lookback)
    
    return df

def calculate_slope_at_point(series, index, lookback=3):
    """Calculate the slope of a single point using the previous lookback days."""
    if index < lookback:
        return 0.0  # Not enough data for slope calculation
    
    # Get values for the lookback window
    values = series.iloc[index-lookback:index].values
    
    # Check if we have enough valid data
    if len(values) < lookback or np.isnan(values).any():
        return 0.0
    
    # Calculate slope using linear regression
    x = np.arange(len(values))
    slope = np.polyfit(x, values, 1)[0]
    
    return slope

def generate_signals(df, include_slope_check=False, lookback=3):
    """Generate buy and sell signals based on moving average crossovers and optionally slope direction.
    
    Optimized version that only calculates slopes at crossover points."""
    # Initialize Signal column with 0 (no signal)
    df['Signal'] = 0
    
    # We don't pre-calculate slopes for all points; instead, we'll calculate them only when needed
    # This is much more efficient when crossovers are rare
    
    # We need at least one value in both MAs to start
    if len(df.dropna()) > 0:
        # Look for crossover points (short MA crosses above long MA = buy, below = sell)
        for i in range(1, len(df)):
            # Skip rows with NaN values
            if pd.isna(df['Short_MA'].iloc[i-1]) or pd.isna(df['Long_MA'].iloc[i-1]) or \
               pd.isna(df['Short_MA'].iloc[i]) or pd.isna(df['Long_MA'].iloc[i]):
                continue
                
            # Buy signal: Short MA crosses above Long MA
            buy_crossover = (df['Short_MA'].iloc[i-1] <= df['Long_MA'].iloc[i-1] and 
                             df['Short_MA'].iloc[i] > df['Long_MA'].iloc[i])
            
            # Sell signal: Short MA crosses below Long MA
            sell_crossover = (df['Short_MA'].iloc[i-1] >= df['Long_MA'].iloc[i-1] and 
                             df['Short_MA'].iloc[i] < df['Long_MA'].iloc[i])
            
            # Only calculate slopes if we have a crossover and slope check is enabled
            if include_slope_check and (buy_crossover or sell_crossover):
                # Calculate slopes at this specific point only
                short_ma_slope = calculate_slope_at_point(df['Short_MA'], i, lookback)
                long_ma_slope = calculate_slope_at_point(df['Long_MA'], i, lookback)
                
                # For buy signals, check if both slopes are positive
                if buy_crossover and short_ma_slope > 0 and long_ma_slope > 0:
                    df.loc[df.index[i], 'Signal'] = 1
                    
                # For sell signals, check if both slopes are negative
                elif sell_crossover and short_ma_slope < 0 and long_ma_slope < 0:
                    df.loc[df.index[i], 'Signal'] = -1
            elif not include_slope_check:
                # If slope check is disabled, just use the crossover signals
                if buy_crossover:
                    df.loc[df.index[i], 'Signal'] = 1
                elif sell_crossover:
                    df.loc[df.index[i], 'Signal'] = -1
    
    return df

def calculate_annualized_gain(initial, final, start_date, end_date):
    """Calculate the annualized percentage gain given initial and final values and dates."""
    num_days = (end_date - start_date).days
    num_years = num_days / 365.25
    if num_years > 0 and initial > 0:
        return ((final / initial) ** (1 / num_years) - 1) * 100
    else:
        return 0.0

def calculate_wealth(df, initial_investment=10000, allow_short_selling=False, short_sell_only=False):
    """Calculate wealth over time based on trading signals."""
    # We need valid data with signals
    df_valid = df.dropna().copy()
    
    if len(df_valid) < 2:
        return initial_investment, 0, []
    
    # Initialize variables
    cash = initial_investment
    shares = 0
    position = 0  # 0: no position, 1: long position, -1: short position
    short_entry_price = 0  # Price at which short position was entered
    trade_list = []  # List to track trades for detailed view
    
    # Loop through each row to simulate trading
    for i in range(len(df_valid)):
        signal = df_valid['Signal'].iloc[i]
        price = df_valid['Price'].iloc[i]
        date = df_valid['Date'].iloc[i]
        
        # Long trades (Buy signal opens, Sell signal closes)
        if not short_sell_only and signal == 1 and (position == 0 or (allow_short_selling and position == -1)):
            if position == -1:  # Close short position if exists
                profit = shares * (short_entry_price - price)  # Profit from short trade
                cash += profit + shares * short_entry_price  # Return initial investment plus profit
                shares = 0
                position = 0
                trade_list.append(('Cover Short', date, price))
            
            # Open long position
            shares = cash / price
            cash = 0
            position = 1
            trade_list.append(('Buy', date, price))
        
        # Sell signal - close long or open short
        elif signal == -1:
            if position == 1:  # Close long position
                cash = shares * price
                shares = 0
                position = 0
                trade_list.append(('Sell', date, price))
            
            # If short selling is allowed and we don't have a position, open a short position
            if allow_short_selling and position == 0:
                shares = cash / price  # Number of shares to short
                short_entry_price = price
                position = -1
                cash = 0  # Cash is held as collateral
                trade_list.append(('Short Sell', date, price))
    
    # Close any open position at the end
    final_price = df_valid['Price'].iloc[-1]
    final_date = df_valid['Date'].iloc[-1]
    
    if position == 1:  # Long position
        cash = shares * final_price
    elif position == -1:  # Short position
        profit = shares * (short_entry_price - final_price)
        cash += profit + shares * short_entry_price
        
    # Calculate the total number of trades
    total_trades = len(trade_list)
    
    # Calculate annualized gain
    start_date = df_valid['Date'].iloc[0]
    annualized_gain = calculate_annualized_gain(initial_investment, cash, start_date, final_date)
    
    # Return final wealth value and the trade list
    return cash, annualized_gain, trade_list

def run_backtest():
    """Run the backtest with the specified parameters."""
    try:
        # These variables need to be accessed from the global scope
        global file_path_var, start_date_var, end_date_var
        global short_min_var, short_max_var, long_min_var, long_max_var
        global allow_short_selling_var, short_sell_only_var, include_slope_check_var, slope_lookback_var
        global progress_bar, progress_label, root, heatmap_button, progress_var, trades_heatmap_button
        global cancel_backtest
        
        # Reset cancel flag
        cancel_backtest = False
        
        # Create a separate thread for running the backtest
        import threading
        backtest_thread = threading.Thread(target=run_backtest_worker)
        backtest_thread.daemon = True  # Allow the thread to be terminated when the program exits
        backtest_thread.start()
        
        return
        
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")
        # Reset progress bar on error
        progress_var.set(0)  # Update the variable instead of the widget directly
        progress_label.config(text="")
        
def run_backtest_worker():
    """Worker function that runs the backtest in a separate thread."""
    try:
        # These variables need to be accessed from the global scope
        global file_path_var, start_date_var, end_date_var
        global short_min_var, short_max_var, long_min_var, long_max_var
        global allow_short_selling_var, short_sell_only_var, include_slope_check_var, slope_lookback_var
        global progress_bar, progress_label, root, heatmap_button, progress_var, trades_heatmap_button
        global cancel_backtest
        
        # Get parameters from the UI
        file_path = file_path_var.get()
        start_date_str = start_date_var.get()
        end_date_str = end_date_var.get()
        short_min = short_min_var.get()
        short_max = short_max_var.get()
        long_min = long_min_var.get()
        long_max = long_max_var.get()
        allow_short_selling = allow_short_selling_var.get()
        short_sell_only = short_sell_only_var.get() if allow_short_selling else False
        include_slope_check = include_slope_check_var.get()
        slope_lookback = slope_lookback_var.get()
        
        # Validate inputs
        try:
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
            
            if start_date >= end_date:
                messagebox.showerror("Error", "Start date must be before end date.")
                return
                
            if short_min <= 0 or short_max <= 0 or long_min <= 0 or long_max <= 0:
                messagebox.showerror("Error", "MA periods must be positive integers.")
                return
                
            if short_min > short_max or long_min > long_max:
                messagebox.showerror("Error", "Min values must be less than or equal to max values.")
                return
                
            if short_max >= long_min:
                messagebox.showerror("Error", "Short MA max must be less than Long MA min.")
                return
        except ValueError:
            messagebox.showerror("Error", "Invalid date format. Use YYYY-MM-DD.")
            return
            
        # Read the data
        df = read_csv(file_path)
        if df is None:
            return
            
        # Filter by date range
        df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
        
        if df.empty:
            messagebox.showerror("Error", "No data found in the specified date range.")
            return
            
        # Generate combinations of short and long MA periods
        short_periods = range(short_min, short_max + 1)
        long_periods = range(long_min, long_max + 1)
        
        # Setup output
        results = []
        total_combinations = len(short_periods) * len(long_periods)
        
        # Setup progress bar
        progress_bar['maximum'] = total_combinations
        progress_var.set(0)  # Update the variable instead of the widget directly
        progress_label.config(text="0% Complete")
        root.update_idletasks()
        
        # Record start time
        start_time = datetime.now()
        
        # Process each combination
        combination_count = 0
        
        # Check if the backtest has been cancelled before we start any calculations
        if cancel_backtest:
            messagebox.showinfo("Cancelled", "Backtest was cancelled by user.")
            # Reset progress bar
            progress_var.set(0)
            progress_label.config(text="")
            return
            
        # For slope check, we can optimize by pre-calculating MAs for each period once
        if include_slope_check:
            # Pre-calculate all needed MAs only once
            ma_cache = {}
            
            # Check for cancellation more frequently, even during pre-calculation
            for short_period in short_periods:
                # Check cancellation for each short period
                if cancel_backtest:
                    messagebox.showinfo("Cancelled", "Backtest was cancelled by user.")
                    progress_var.set(0)
                    progress_label.config(text="")
                    return
                    
                for long_period in long_periods:
                    # Check cancellation even more frequently
                    if cancel_backtest:
                        messagebox.showinfo("Cancelled", "Backtest was cancelled by user.")
                        progress_var.set(0)
                        progress_label.config(text="")
                        return
                        
                    # Calculate moving averages and store in cache
                    df_with_ma = calculate_moving_averages(df, short_period, long_period)
                    ma_cache[(short_period, long_period)] = df_with_ma
                    
            # Now process using the cached values
            for short_period in short_periods:
                # Check cancellation for each short period
                if cancel_backtest:
                    messagebox.showinfo("Cancelled", "Backtest was cancelled by user.")
                    progress_var.set(0)
                    progress_label.config(text="")
                    return
                    
                for long_period in long_periods:
                    # Check if the backtest has been cancelled
                    if cancel_backtest:
                        messagebox.showinfo("Cancelled", "Backtest was cancelled by user.")
                        # Reset progress bar
                        progress_var.set(0)
                        progress_label.config(text="")
                        return
                        
                    # Get pre-calculated MA data
                    df_with_ma = ma_cache[(short_period, long_period)]
                    
                    # Generate trading signals with custom lookback
                    df_with_signals = generate_signals(df_with_ma, include_slope_check, lookback=slope_lookback)
                    
                    # Calculate final wealth and gain
                    final_wealth, annualized_gain, trade_list = calculate_wealth(
                        df_with_signals, 10000, allow_short_selling, short_sell_only
                    )
                    
                    # Count the number of trades
                    trade_count = len(trade_list) if trade_list else 0
                    
                    # Store results
                    results.append({
                        'short_ma': short_period,
                        'long_ma': long_period,
                        'final_wealth': final_wealth,
                        'annualized_gain': annualized_gain,
                        'trade_count': trade_count
                    })
                    
                    # Update progress bar
                    combination_count += 1
                    progress = int((combination_count / total_combinations) * 100)
                    progress_var.set(combination_count)
                    progress_label.config(text=f"{progress}% Complete")
                    root.update_idletasks()
        else:
            # Original approach for when slope check is disabled
            for short_period in short_periods:
                # Check cancellation for each short period
                if cancel_backtest:
                    messagebox.showinfo("Cancelled", "Backtest was cancelled by user.")
                    progress_var.set(0)
                    progress_label.config(text="")
                    return
                    
                for long_period in long_periods:
                    # Check if the backtest has been cancelled
                    if cancel_backtest:
                        messagebox.showinfo("Cancelled", "Backtest was cancelled by user.")
                        # Reset progress bar
                        progress_var.set(0)
                        progress_label.config(text="")
                        return
                        
                    # Calculate moving averages
                    df_with_ma = calculate_moving_averages(df, short_period, long_period)
                    
                    # Generate trading signals
                    df_with_signals = generate_signals(df_with_ma, include_slope_check)
                    
                    # Calculate final wealth and gain
                    final_wealth, annualized_gain, trade_list = calculate_wealth(
                        df_with_signals, 10000, allow_short_selling, short_sell_only
                    )
                    
                    # Count the number of trades
                    trade_count = len(trade_list) if trade_list else 0
                    
                    # Store results
                    results.append({
                        'short_ma': short_period,
                        'long_ma': long_period,
                        'final_wealth': final_wealth,
                        'annualized_gain': annualized_gain,
                        'trade_count': trade_count
                    })
                    
                    # Update progress bar
                    combination_count += 1
                    progress = int((combination_count / total_combinations) * 100)
                    progress_var.set(combination_count)
                    progress_label.config(text=f"{progress}% Complete")
                    root.update_idletasks()
                
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Compute the best combination
        best_result = results_df.loc[results_df['annualized_gain'].idxmax()]
        
        # Create a pivot table for the heatmap
        gain_table = results_df.pivot(index='long_ma', columns='short_ma', values='annualized_gain')
        
        # Also create a pivot table for trade counts
        trade_count_table = results_df.pivot(index='long_ma', columns='short_ma', values='trade_count')
        
        # Display results in a message box
        elapsed_time = (datetime.now() - start_time).total_seconds()
        result_message = (
            f"Analysis Complete!\n\n"
            f"Best Combination:\n"
            f"Short MA: {best_result['short_ma']}\n"
            f"Long MA: {best_result['long_ma']}\n"
            f"Annualized %Gain: {best_result['annualized_gain']:.2f}%\n"
            f"Final Wealth: ${best_result['final_wealth']:.2f}\n\n"
            f"Processed {total_combinations} combinations in {elapsed_time:.2f} seconds."
        )
        messagebox.showinfo("Results", result_message)
        
        # Save results to CSV
        filename = "ma_strategy_results.csv"
        
        # Add metadata to the file
        with open(filename, 'w') as f:
            # Write metadata header
            f.write(f"# Moving Average Crossover Strategy Results\n")
            f.write(f"# Date Range: {start_date_str} to {end_date_str}\n")
            f.write(f"# Allow Short Selling: {allow_short_selling}\n")
            f.write(f"# Short Sell Only: {short_sell_only}\n")
            f.write(f"# Include Slope Check: {include_slope_check}\n")
            f.write(f"# Slope Lookback Period: {slope_lookback}\n")
            f.write(f"# Short MA Range: {short_min} to {short_max}\n")
            f.write(f"# Long MA Range: {long_min} to {long_max}\n")
            f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Best Combination: Short MA={best_result['short_ma']}, Long MA={best_result['long_ma']}, Annualized %Gain={best_result['annualized_gain']:.2f}%\n")
            f.write(f"#\n")  # Empty line before the CSV data
        
        # Append the actual CSV data
        results_df.to_csv(filename, mode='a', index=False)
        
        # The pivot tables are no longer saved as separate files since they can be regenerated
        # from the ma_strategy_results.csv file when needed
        
        # Reset progress bar
        progress_var.set(0)  # Update the variable instead of the widget directly
        progress_label.config(text="")
        
        # Enable the heatmap buttons now that we have results
        heatmap_button.config(state=tk.NORMAL)
        trades_heatmap_button.config(state=tk.NORMAL)
        
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")
        # Reset progress bar on error
        progress_var.set(0)  # Update the variable instead of the widget directly
        progress_label.config(text="")

def create_heatmap_window(root, data, title, value_label, cmap, bounds, colors):
    """Create a window with a heatmap visualization of the strategy results."""
    # Create a new window
    heatmap_window = tk.Toplevel(root)
    heatmap_window.title(title)
    heatmap_window.state('zoomed')  # Maximize window
    
    # Create a frame for the window contents
    main_frame = tk.Frame(heatmap_window)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Status frame for displaying cell information
    status_frame = tk.Frame(main_frame, relief=tk.SUNKEN, borderwidth=1)
    status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
    
    status_var = tk.StringVar(value="Hover over cells to see details")
    status_label = tk.Label(status_frame, textvariable=status_var, anchor='w', 
                          font=("Arial", 11), bg='#f0f0f0', padx=10, pady=5)
    status_label.pack(fill=tk.X)
    
    # Create a figure and axes for the heatmap
    fig = plt.Figure(figsize=(12, 8), dpi=100)
    ax = fig.add_subplot(111)
    
    # Create the heatmap
    im = ax.imshow(data, cmap=cmap, aspect='auto', interpolation='nearest')
    
    # Set the colorbar with custom bounds
    if bounds is not None:
        # Create a BoundaryNorm to map colors correctly with the boundaries
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        
        # Use all bounds except the last for ticks, which gives us one tick per color
        tick_locations = bounds[:-1] + (bounds[1:] - bounds[:-1]) / 2  # Center of each color band
        
        # Create colorbar with proper tick marks for each color segment
        cbar = fig.colorbar(
            im, 
            ax=ax, 
            norm=norm, 
            boundaries=bounds, 
            ticks=tick_locations,
            spacing='proportional'
        )
        
        # Set tick labels to show the range for each color segment
        tick_labels = [f"{bounds[i]:.1f} - {bounds[i+1]:.1f}" for i in range(len(bounds)-1)]
        cbar.ax.set_yticklabels(tick_labels)
    else:
        cbar = fig.colorbar(im, ax=ax)
    
    # Adjust the colorbar label position
    cbar.ax.set_ylabel(value_label, rotation=270, labelpad=20)
    
    # Set tick labels - with smart reduction of labels for readability
    # Define max number of ticks we want to show
    max_ticks = 30
    
    # Handle x-axis (columns) labels
    col_values = np.array(data.columns)
    col_length = len(col_values)
    
    if col_length > max_ticks:
        # Find a nice interval that's a multiple of 2, 5, or 10
        raw_interval = col_length / max_ticks
        
        # Calculate potential nice intervals
        nice_intervals = [1, 2, 5, 10, 20, 25, 50, 100]
        interval = 1
        
        # Find the smallest nice interval that gives us <= max_ticks
        for nice in nice_intervals:
            if col_length / nice <= max_ticks:
                interval = nice
                break
            interval = nice
        
        # Create ticks at regular intervals based on values, not just indices
        min_val = col_values.min()
        max_val = col_values.max()
        
        # Create nice tick locations
        tick_values = np.arange(
            (min_val // interval) * interval,  # Start at floor of min_val
            max_val + interval,                # End after max_val
            interval                          # Use our nice interval
        )
        
        # Filter to only include values in our range
        tick_values = tick_values[(tick_values >= min_val) & (tick_values <= max_val)]
        
        # Find the closest indices for our tick values
        tick_indices = []
        for val in tick_values:
            idx = np.abs(col_values - val).argmin()
            if idx not in tick_indices:  # Avoid duplicates
                tick_indices.append(idx)
        
        ax.set_xticks(tick_indices)
        ax.set_xticklabels([col_values[i] for i in tick_indices])
        
        print(f"X-axis: Using {len(tick_indices)} ticks with interval {interval}")
    else:
        # If we have few enough labels, show them all
        ax.set_xticks(np.arange(len(data.columns)))
        ax.set_xticklabels(data.columns)
    
    # Handle y-axis (rows) labels
    row_values = np.array(data.index)
    row_length = len(row_values)
    
    if row_length > max_ticks:
        # Find a nice interval that's a multiple of 2, 5, or 10
        raw_interval = row_length / max_ticks
        
        # Calculate potential nice intervals
        nice_intervals = [1, 2, 5, 10, 20, 25, 50, 100]
        interval = 1
        
        # Find the smallest nice interval that gives us <= max_ticks
        for nice in nice_intervals:
            if row_length / nice <= max_ticks:
                interval = nice
                break
            interval = nice
        
        # Create ticks at regular intervals based on values, not just indices
        min_val = row_values.min()
        max_val = row_values.max()
        
        # Create nice tick locations
        tick_values = np.arange(
            (min_val // interval) * interval,  # Start at floor of min_val
            max_val + interval,                # End after max_val
            interval                          # Use our nice interval
        )
        
        # Filter to only include values in our range
        tick_values = tick_values[(tick_values >= min_val) & (tick_values <= max_val)]
        
        # Find the closest indices for our tick values
        tick_indices = []
        for val in tick_values:
            idx = np.abs(row_values - val).argmin()
            if idx not in tick_indices:  # Avoid duplicates
                tick_indices.append(idx)
        
        ax.set_yticks(tick_indices)
        ax.set_yticklabels([row_values[i] for i in tick_indices])
        
        print(f"Y-axis: Using {len(tick_indices)} ticks with interval {interval}")
    else:
        # If we have few enough labels, show them all
        ax.set_yticks(np.arange(len(data.index)))
        ax.set_yticklabels(data.index)
    
    # Add labels with interval information
    col_min, col_max = data.columns.min(), data.columns.max()
    row_min, row_max = data.index.min(), data.index.max()
    
    # Add axis labels with range information
    ax.set_xlabel(f'Short Moving Average Period (Range: {col_min} to {col_max})')
    ax.set_ylabel(f'Long Moving Average Period (Range: {row_min} to {row_max})')
    
    # Set title based on the value label
    if "Gain" in value_label:
        ax.set_title(f"{title} - Higher Values (Redder) Are Better\n(Hover over cells for exact values)")
    elif "Trades" in value_label:
        ax.set_title(f"{title} - Shows Trading Activity (More Trades = Redder)\n(Hover over cells for exact values)")
    else:
        ax.set_title(f"{title}\n(Hover over cells for exact values)")
    
    # Rotate the x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add a grid to help with readability
    ax.grid(True, alpha=0.3)
    
    # Add a text note about axis ticks if they've been reduced
    if len(data.columns) > max_ticks or len(data.index) > max_ticks:
        fig.text(0.5, 0.01, 
                 "Note: Axis labels are shown at intervals for clarity. Hover over cells for exact values.", 
                 ha='center', fontsize=9, fontstyle='italic', alpha=0.7)
    
    # Note: We're no longer adding text annotations to the cells to avoid clutter
    # The mouseover functionality will provide all necessary information
    
    # Embed the plot in the Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=main_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    # Add navigation toolbar
    toolbar = NavigationToolbar2Tk(canvas, main_frame)
    toolbar.update()
    
    # Function to handle mouseover events
    def motion_notify_event(event):
        if event.inaxes == ax:
            # Get the indices of the current cell
            i = int(event.ydata + 0.5) if event.ydata is not None else -1
            j = int(event.xdata + 0.5) if event.xdata is not None else -1
            
            # Check if the indices are valid
            if 0 <= i < len(data.index) and 0 <= j < len(data.columns):
                long_ma = data.index[i]
                short_ma = data.columns[j]
                value = data.iloc[i, j]
                
                if pd.isna(value):
                    status_var.set("No data available for this cell")
                    return
                
                # Create different tooltips based on heatmap type
                if "Gain" in value_label:
                    tooltip_text = (f"Long MA: {long_ma}  |  Short MA: {short_ma}  |  "
                                   f"Annualized Gain: {value:.2f}%")
                    
                    # Try to get additional data if available
                    try:
                        # Try to load data from the combined results file
                        combined_file = "ma_strategy_results.csv"
                        if os.path.exists(combined_file):
                            # Skip the metadata header lines
                            with open(combined_file, 'r') as f:
                                line = f.readline()
                                skiprows = 0
                                while not line.startswith("long_ma,short_ma"):
                                    line = f.readline()
                                    skiprows += 1
                                    if skiprows > 100:  # Safety check
                                        break
                            
                            combined_data = pd.read_csv(combined_file, skiprows=skiprows)
                            row = combined_data[(combined_data['long_ma'] == long_ma) & 
                                              (combined_data['short_ma'] == short_ma)]
                            
                            if not row.empty:
                                tooltip_text += f"  |  Total Trades: {int(row['trade_count'].iloc[0])}"
                                if 'long_trades' in row.columns and 'short_trades' in row.columns:
                                    tooltip_text += f"  |  Long: {int(row['long_trades'].iloc[0])}  |  Short: {int(row['short_trades'].iloc[0])}"
                    except Exception as e:
                        print(f"Error getting additional data: {e}")
                        
                elif "Trades" in value_label:
                    tooltip_text = (f"Long MA: {long_ma}  |  Short MA: {short_ma}  |  "
                                   f"Total Trades: {int(value) if not pd.isna(value) else 'N/A'}")
                    
                    # Try to get additional data if available
                    try:
                        # Try to load data from the combined results file
                        combined_file = "ma_strategy_results.csv"
                        if os.path.exists(combined_file):
                            # Skip the metadata header lines
                            with open(combined_file, 'r') as f:
                                line = f.readline()
                                skiprows = 0
                                while not line.startswith("long_ma,short_ma"):
                                    line = f.readline()
                                    skiprows += 1
                                    if skiprows > 100:  # Safety check
                                        break
                            
                            combined_data = pd.read_csv(combined_file, skiprows=skiprows)
                            row = combined_data[(combined_data['long_ma'] == long_ma) & 
                                              (combined_data['short_ma'] == short_ma)]
                            
                            if not row.empty:
                                tooltip_text += f"  |  Annualized Gain: {row['annualized_gain'].iloc[0]:.2f}%"
                                if 'long_trades' in row.columns and 'short_trades' in row.columns:
                                    tooltip_text += f"  |  Long: {int(row['long_trades'].iloc[0])}  |  Short: {int(row['short_trades'].iloc[0])}"
                    except Exception as e:
                        print(f"Error getting additional data: {e}")
                
                else:
                    tooltip_text = f"Long MA: {long_ma}  |  Short MA: {short_ma}  |  Value: {value}"
                
                # Update the status bar
                status_var.set(tooltip_text)
            else:
                status_var.set("Hover over cells to see details")
        else:
            status_var.set("Hover over cells to see details")
    
    # Connect the mouseover event to the handler function
    canvas.mpl_connect('motion_notify_event', motion_notify_event)
    
    # Create a dedicated frame for the close button to ensure it's visible
    button_frame = tk.Frame(main_frame)
    button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5, before=status_frame)
    
    # Add a more prominent close button
    close_button = tk.Button(
        button_frame, 
        text="Close", 
        command=heatmap_window.destroy, 
        width=15,
        height=2,  # Make button taller
        bg="#f44336",  # Red background
        fg="white",    # White text
        font=("Arial", 10, "bold")
    )
    close_button.pack(pady=5)
    
    # Initialize all widgets before returning
    heatmap_window.update_idletasks()
    
    return heatmap_window

def ensure_maximized(window):
    """Ensure the window is maximized (workaround for some Tkinter issues)."""
    window.state('zoomed')  # Try to maximize again after window is fully created

def show_price_chart(file_path, start_date, end_date):
    """Show a price chart with moving averages and buy/sell signals."""
    # We need access to global variables for the controls in this function
    global allow_short_selling_var, short_sell_only_var, include_slope_check_var, slope_lookback_var, file_path_var, all_trades
    global trades_count_var, long_trades_var, short_trades_var, annualized_gain_var
    try:
        # Convert string dates to datetime objects
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Read data
        df = read_csv(file_path)
        if df is None:
            return
            
        # Filter by date range
        filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
        
        if filtered_df.empty:
            messagebox.showerror("Error", "No data found in the specified date range.")
            return
        
        # Create a new window
        chart_window = tk.Toplevel()
        chart_window.title("Price Chart with Moving Averages")
        chart_window.state('zoomed')  # Maximize the window by default
        
        # Create main frame
        main_frame = tk.Frame(chart_window)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create chart frame on the left
        chart_frame = tk.Frame(main_frame)
        chart_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create a frame with scrollbar for controls
        controls_container = tk.Frame(main_frame)
        controls_container.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Add scrollbar to the controls container
        controls_scroll = tk.Scrollbar(controls_container, orient=tk.VERTICAL)
        controls_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create canvas for scrolling
        controls_canvas = tk.Canvas(controls_container, width=375, bg='#f0f0f0', 
                                   yscrollcommand=controls_scroll.set)
        controls_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        controls_scroll.config(command=controls_canvas.yview)
        
        # Create controls frame on the right (increased width from 250 to 375)
        controls_frame = tk.Frame(controls_canvas, width=375, padx=15, pady=15, bg='#f0f0f0')
        controls_canvas.create_window((0, 0), window=controls_frame, anchor=tk.NW, width=375)
        
        # Update the scrollregion when the size of the frame changes
        def update_scrollregion(event):
            controls_canvas.configure(scrollregion=controls_canvas.bbox(tk.ALL))
        
        # Function to handle mousewheel scrolling
        def on_mousewheel(event):
            controls_canvas.yview_scroll(-1 * int(event.delta/120), "units")
        
        # Bind events for scrolling
        controls_frame.bind("<Configure>", update_scrollregion)
        controls_canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        # Add controls for moving averages
        tk.Label(controls_frame, text="Moving Averages", font=("Arial", 14, "bold"), bg='#f0f0f0').pack(pady=(0, 10))
        
        # Short MA control with frame
        short_ma_frame = tk.Frame(controls_frame, bg='#f0f0f0')
        short_ma_frame.pack(fill=tk.X, pady=(10, 0))
        tk.Label(short_ma_frame, text="Short Moving Average:", bg='#f0f0f0').pack(anchor='w')
        short_ma_var = tk.StringVar(value="50")
        short_ma_entry = tk.Entry(short_ma_frame, textvariable=short_ma_var, width=10, font=("Arial", 10))
        short_ma_entry.pack(pady=(0, 10), anchor='w')
        
        # Long MA control with frame
        long_ma_frame = tk.Frame(controls_frame, bg='#f0f0f0')
        long_ma_frame.pack(fill=tk.X, pady=(5, 0))
        tk.Label(long_ma_frame, text="Long Moving Average:", bg='#f0f0f0').pack(anchor='w')
        long_ma_var = tk.StringVar(value="150")
        long_ma_entry = tk.Entry(long_ma_frame, textvariable=long_ma_var, width=10, font=("Arial", 10))
        long_ma_entry.pack(side=tk.LEFT, pady=(0, 5))
        
        # Function to update the chart with moving averages and buy/sell signals
        def update_chart():
            global file_path_var, allow_short_selling_var, short_sell_only_var, include_slope_check_var, trades_count_var, long_trades_var, short_trades_var, annualized_gain_var, all_trades
            ax.clear()
            
            # Get MA values
            try:
                short_ma = int(short_ma_var.get())
                long_ma = int(long_ma_var.get())
                
                if short_ma <= 0 or long_ma <= 0:
                    messagebox.showerror("Error", "Moving average periods must be positive integers.")
                    return
            except ValueError:
                messagebox.showerror("Error", "Moving average periods must be positive integers.")
                return
                
            # Function to calculate annualized gain
            def calculate_annualized_gain(initial, final, start_date, end_date):
                num_days = (end_date - start_date).days
                num_years = num_days / 365.25
                if num_years > 0 and initial > 0:
                    return ((final / initial) ** (1 / num_years) - 1) * 100
                else:
                    return 0.0
                    
            # Track start time for performance monitoring
            chart_start_time = datetime.now()
            
            # Calculate moving averages
            global df_copy
            df_copy = filtered_df.copy()
            df_copy['Short_MA'] = df_copy['Price'].rolling(window=short_ma).mean()
            df_copy['Long_MA'] = df_copy['Price'].rolling(window=long_ma).mean()
            
            # Get the include slope check option
            include_slope_check = include_slope_check_var.get()
            
            # Track time for slope calculations
            slope_start_time = datetime.now()
            
            # Get the lookback period
            lookback = slope_lookback_var.get()
            
            # Generate signals using the optimized function that calculates slopes only at crossovers
            df_copy = generate_signals(df_copy, include_slope_check, lookback=lookback)
            # Always fill slope columns for all rows
            df_copy = calculate_slopes(df_copy, lookback)
            
            if include_slope_check:
                print(f"Optimized slope calculation took: {(datetime.now() - slope_start_time).total_seconds():.2f} seconds")
            
            # Plot the data - price in black, short MA in blue, long MA in green
            ax.plot(df_copy['Date'], df_copy['Price'], 'k-', label='Price')
            ax.plot(df_copy['Date'], df_copy['Short_MA'], 'b-', label=f'Short MA ({short_ma})')
            ax.plot(df_copy['Date'], df_copy['Long_MA'], 'g-', label=f'Long MA ({long_ma})')
            
            # Calculate min and max prices for vertical positioning
            min_price = filtered_df['Price'].min()
            max_price = filtered_df['Price'].max()
            
            # Calculate vertical position for buy/sell markers
            # We'll place them much further away from the price line to avoid being obscured by any lines
            price_range = max_price - min_price
            buy_offset = price_range * 0.08  # 8% of price range above price (doubled from 4%)
            sell_offset = price_range * 0.08  # 8% of price range below price (doubled from 4%)
            
            # Add buy signals (green arrows pointing up)
            buy_signals = df_copy[df_copy['Signal'] == 1]
            if not buy_signals.empty:
                for idx, row in buy_signals.iterrows():
                    # Place buy markers (green B)
                    price = row['Price']
                    marker_y = price + buy_offset  # Place above price
                    ax.annotate('B', xy=(row['Date'], marker_y),
                                xytext=(0, 0), textcoords='offset points',
                                fontsize=12, fontweight='bold', color='green',
                                horizontalalignment='center', backgroundcolor='white',
                                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
                    # Add vertical line at buy point
                    ax.axvline(x=row['Date'], color='green', linestyle='--', alpha=0.3)
            
            # Add sell signals (red arrows pointing down)
            sell_signals = df_copy[df_copy['Signal'] == -1]
            if not sell_signals.empty:
                for idx, row in sell_signals.iterrows():
                    # Place sell markers (red S)
                    price = row['Price'] 
                    marker_y = price - sell_offset  # Place below price
                    ax.annotate('S', xy=(row['Date'], marker_y),
                                xytext=(0, 0), textcoords='offset points',
                                fontsize=12, fontweight='bold', color='red',
                                horizontalalignment='center', backgroundcolor='white',
                                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
                    # Add vertical line at sell point
                    ax.axvline(x=row['Date'], color='red', linestyle='--', alpha=0.3)
            
            # Format the x-axis to show dates properly
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            fig.autofmt_xdate()  # Rotates and formats date labels
            
            # Add labels and title
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.set_title(f'Price Chart with Buy/Sell Signals ({start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")})')
            
            # Set y-axis range based on the min and max values with even more padding for signals
            # Add 12% padding (increased from 8%)
            padding = (max_price - min_price) * 0.12
            ax.set_ylim(min_price - padding*3.0, max_price + padding*3.0)  # Further increased padding for buy/sell markers
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Create a clean, simple legend for lines only
            # This avoids any issues with the Buy/Sell signals in the legend
            legend = ax.legend(
                ['Price', f'Short MA ({short_ma})', f'Long MA ({long_ma})'],
                loc='upper left',
                fontsize='small',
                frameon=True,
                fancybox=True
            )
            
            # Calculate strategy performance metrics for display
            # Calculate wealth and trade count
            initial_investment = 10000  # Standard initial investment
            
            # First, find valid signals (non-NaN)
            valid_df = df_copy.dropna()
            
            # Count buy and sell signals
            buy_signals_count = len(valid_df[valid_df['Signal'] == 1])
            sell_signals_count = len(valid_df[valid_df['Signal'] == -1])
            total_signals = buy_signals_count + sell_signals_count
            
            # Get short selling options
            allow_short_selling = allow_short_selling_var.get()
            short_sell_only = short_sell_only_var.get() if allow_short_selling else False
            
            # Track trades for detailed statistics
            long_trade_count = 0
            short_trade_count = 0
            
            # Update the trades count displays
            trades_count_var.set(f"{total_signals}")
            
            # Update long and short trade counts based on strategy options
            if short_sell_only:
                long_trade_count = 0
                short_trade_count = sell_signals_count
                long_trades_var.set("0 (disabled)")
            else:
                long_trade_count = buy_signals_count
                short_trade_count = sell_signals_count if allow_short_selling else 0
                long_trades_var.set(f"{long_trade_count}")
            
            short_trades_var.set(f"{short_trade_count}" if allow_short_selling else "0 (disabled)")
            
            # Simulate trading to calculate performance
            if len(valid_df) > 0:
                # Get the first and last valid dates
                first_date = valid_df['Date'].iloc[0]
                last_date = valid_df['Date'].iloc[-1]
                
                # Run trading simulation based on signals
                cash = initial_investment
                shares = 0
                position = 0  # 0: no position, 1: long position, -1: short position
                short_entry_price = 0  # Price at which short position was entered
                
                # Get short selling options
                allow_short_selling = allow_short_selling_var.get()
                short_sell_only = short_sell_only_var.get() if allow_short_selling else False
                
                for i in range(len(valid_df)):
                    signal = valid_df['Signal'].iloc[i]
                    price = valid_df['Price'].iloc[i]
                    
                    # Long trades (Buy signal opens, Sell signal closes)
                    if not short_sell_only and signal == 1 and (position == 0 or (allow_short_selling and position == -1)):
                        if position == -1:  # Close short position if exists
                            profit = shares * (short_entry_price - price)  # Profit from short trade
                            cash += profit + shares * short_entry_price  # Return initial investment plus profit
                            shares = 0
                            position = 0
                        
                        # Open long position
                        shares = cash / price
                        cash = 0
                        position = 1
                    
                    # Sell signal - close long or open short
                    elif signal == -1:
                        if position == 1:  # Close long position
                            cash = shares * price
                            shares = 0
                            position = 0
                        
                        # If short selling is allowed and we don't have a position, open a short position
                        if allow_short_selling and position == 0:
                            shares = cash / price  # Number of shares to short
                            short_entry_price = price
                            position = -1
                            cash = 0  # Cash is held as collateral
                
                # Close any open position at the end
                final_price = valid_df['Price'].iloc[-1]
                if position == 1:  # Long position
                    cash = shares * final_price
                elif position == -1:  # Short position
                    profit = shares * (short_entry_price - final_price)
                    cash += profit + shares * short_entry_price
                
                # Calculate final wealth and annualized gain
                final_wealth = cash
                annualized_gain = calculate_annualized_gain(initial_investment, final_wealth, first_date, last_date)
                
                # Update the annualized gain display
                annualized_gain_var.set(f"{annualized_gain:.2f}%")
            else:
                # No valid data for calculation
                annualized_gain_var.set("N/A")
                
            # Now gather trade information and update the trade list
            # First, reset the all_trades list
            global all_trades
            all_trades = []  # Create a new list rather than just clearing it
            
            # Collect all trades based on signals
            allow_short = allow_short_selling_var.get()
            short_only = short_sell_only_var.get() if allow_short else False
            
            position = 0  # 0: no position, 1: long, -1: short
            
            # Process each signal to create trade entries
            for i in range(len(valid_df)):
                signal = valid_df['Signal'].iloc[i]
                date = valid_df['Date'].iloc[i]
                price = valid_df['Price'].iloc[i]
                
                # Buy signal
                if signal == 1:
                    if not short_only and (position == 0 or (allow_short and position == -1)):
                        if position == -1:
                            # Cover short before buying
                            all_trades.append(('Cover Short', date, price))
                        # Open long position
                        all_trades.append(('Buy', date, price))
                        position = 1
                
                # Sell signal
                elif signal == -1:
                    if position == 1:
                        # Close long position
                        all_trades.append(('Sell', date, price))
                        position = 0
                    
                    if allow_short and position == 0:
                        # Open short position
                        all_trades.append(('Short Sell', date, price))
                        position = -1
                        
            # Debug: Print trade information after processing
            print(f"After collecting, all_trades has {len(all_trades)} trades")
                        
            # Update the trade list display
            trades_text.config(state=tk.NORMAL)  # Enable editing
            trades_text.delete(1.0, tk.END)  # Clear current content
            
            # Display trades from the global variable
            if all_trades:
                # Display a limited number of most recent trades (last 5)
                # BUT CHANGE: Show the first and last trades to ensure context is clear
                if len(all_trades) <= 5:
                    display_trades = all_trades  # Display all if 5 or fewer
                else:
                    # Mix of first and last trades for better context
                    first_trades = all_trades[:2]  # First 2 trades
                    last_trades = all_trades[-3:]  # Last 3 trades
                    display_trades = first_trades + last_trades
                
                # Add header to make it clear what's being shown
                if len(all_trades) > 5:
                    trades_text.insert(tk.END, "First and last trades (for context):\n")
                    trades_text.insert(tk.END, "-------------------------------\n")
                
                for trade in display_trades:
                    action, date, price = trade[:3]
                    # Reduce action width from 15 to 8 characters
                    trades_text.insert(tk.END, f"{action:<8} {date.strftime('%Y-%m-%d')}  ${price:.2f}\n")
                
                if len(all_trades) > 5:
                    trades_text.insert(tk.END, f"\n... and {len(all_trades) - 5} more trades")
                
                # Debug information
                print(f"Updated trade display with {len(all_trades)} trades")
            else:
                trades_text.insert(tk.END, "No trades to display")
            
            trades_text.config(state=tk.DISABLED)  # Make it read-only again
            
            # Update the canvas
            fig.tight_layout()
            canvas.draw()
        
        # Add update button next to the moving averages
        update_button = tk.Button(long_ma_frame, text="Update Chart", command=update_chart, 
                                 bg="#4CAF50", fg="white", font=("Arial", 9, "bold"), padx=5, pady=1)
        update_button.pack(side=tk.RIGHT, padx=5, pady=(0, 5))
        
        # Add signal explanation - more compact layout
        signal_frame = tk.Frame(controls_frame, bg='#f0f0f0', relief=tk.GROOVE, bd=1)
        signal_frame.pack(fill=tk.X, pady=5)
        tk.Label(signal_frame, text="Signal Legend", font=("Arial", 10, "bold"), bg='#f0f0f0').pack(pady=3)
        
        # Add a green line for Buy signals
        buy_frame = tk.Frame(signal_frame, bg='#f0f0f0')
        buy_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Create a colored line to represent the Buy signal
        buy_line_canvas = tk.Canvas(buy_frame, width=15, height=15, bg='#f0f0f0', highlightthickness=0)
        buy_line_canvas.pack(side=tk.LEFT)
        buy_line_canvas.create_line(2, 7, 13, 7, fill="green", width=2)
        
        # Add the letter B
        tk.Label(buy_frame, text="B", font=("Arial", 12, "bold"), fg="green", bg='#f0f0f0').pack(side=tk.LEFT, padx=2)
        
        # Add explanation text
        tk.Label(buy_frame, text="= Buy (Short MA crosses above Long MA)", bg='#f0f0f0').pack(side=tk.LEFT, padx=5)
        
        # Add a red line for Sell signals
        sell_frame = tk.Frame(signal_frame, bg='#f0f0f0')
        sell_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Create a colored line to represent the Sell signal
        sell_line_canvas = tk.Canvas(sell_frame, width=15, height=15, bg='#f0f0f0', highlightthickness=0)
        sell_line_canvas.pack(side=tk.LEFT)
        sell_line_canvas.create_line(2, 7, 13, 7, fill="red", width=2)
        
        # Add the letter S
        tk.Label(sell_frame, text="S", font=("Arial", 12, "bold"), fg="red", bg='#f0f0f0').pack(side=tk.LEFT, padx=2)
        
        # Add explanation text
        tk.Label(sell_frame, text="= Sell (Short MA crosses below Long MA)", bg='#f0f0f0').pack(side=tk.LEFT, padx=5)
        
        # Create the figure - reduced size to leave more room for the wider controls panel
        fig = plt.Figure(figsize=(8, 5.5), dpi=100)
        ax = fig.add_subplot(111)
        
        # Add short selling options frame - more compact layout
        short_selling_frame = tk.Frame(controls_frame, bg='#f0f0f0', relief=tk.GROOVE, bd=1)
        short_selling_frame.pack(fill=tk.X, pady=5, padx=5)
        
        # Add title for short selling options - smaller font
        tk.Label(short_selling_frame, text="Short Selling Options", font=("Arial", 10, "bold"), 
                bg='#f0f0f0').pack(pady=(3, 1))        # Create variables for short selling checkboxes
        allow_short_selling_var = tk.BooleanVar(value=False)
        short_sell_only_var = tk.BooleanVar(value=False)
        include_slope_check_var = tk.BooleanVar(value=False)
        
        # Function to toggle short sell only visibility
        def toggle_short_sell_only_visibility():
            if allow_short_selling_var.get():
                # Show short sell only checkbox when allow short selling is checked
                short_sell_only_cb.pack(pady=2, anchor=tk.CENTER)
            else:
                # Hide short sell only checkbox when allow short selling is unchecked
                short_sell_only_cb.pack_forget()
                short_sell_only_var.set(False)  # Reset to unchecked when hidden
            # Update chart to reflect the new settings
            update_chart()
        
        # Add the allow short selling checkbox
        short_selling_cb = tk.Checkbutton(short_selling_frame, text="Allow Short Selling",
                                        variable=allow_short_selling_var,
                                        onvalue=True, offvalue=False,
                                        command=toggle_short_sell_only_visibility,
                                        bg='#f0f0f0')
        short_selling_cb.pack(pady=2, anchor=tk.CENTER)
        
        # Add the short sell only checkbox (initially hidden) - renamed for clarity
        short_sell_only_cb = tk.Checkbutton(short_selling_frame, text="Suppress Long Trades",
                                          variable=short_sell_only_var,
                                          onvalue=True, offvalue=False,
                                          command=update_chart,  # Update chart when this changes
                                          bg='#f0f0f0')
                                          
        # Add a frame for slope options
        slope_frame = tk.Frame(short_selling_frame, bg='#f0f0f0')
        slope_frame.pack(pady=2, fill=tk.X)
        
        # Add the include slope check checkbox
        include_slope_check_cb = tk.Checkbutton(slope_frame, text="Include Slope Check",
                                          variable=include_slope_check_var,
                                          onvalue=True, offvalue=False,
                                          command=update_chart,  # Update chart when this changes
                                          bg='#f0f0f0')
        include_slope_check_cb.pack(side=tk.LEFT, padx=(20, 5))
        
        # Add a label for the lookback period
        tk.Label(slope_frame, text="Lookback:", bg='#f0f0f0').pack(side=tk.LEFT)
        
        # Add a spinbox for the lookback period
        lookback_spinbox = tk.Spinbox(slope_frame, from_=1, to=10, width=3, 
                                     textvariable=slope_lookback_var, 
                                     command=lambda: update_chart() if include_slope_check_var.get() else None)
        lookback_spinbox.pack(side=tk.LEFT, padx=5)
        
        # Create a frame for displaying statistics - more compact layout
        stats_frame = tk.Frame(controls_frame, bg='#f0f0f0', relief=tk.GROOVE, bd=1)
        stats_frame.pack(fill=tk.X, pady=5, padx=5)
        
        # Add a heading for the statistics section - smaller font
        tk.Label(stats_frame, text="Strategy Performance", font=("Arial", 10, "bold"), 
                bg='#f0f0f0').pack(pady=(3, 0))
        
        # Create variables for the stats that will be updated
        annualized_gain_var = tk.StringVar(value="---.--")
        trades_count_var = tk.StringVar(value="--")
        long_trades_var = tk.StringVar(value="--")
        short_trades_var = tk.StringVar(value="--")
        
        # Add Annualized %Gain display
        gain_frame = tk.Frame(stats_frame, bg='#f0f0f0')
        gain_frame.pack(fill=tk.X, padx=10, pady=2)
        tk.Label(gain_frame, text="Annualized %Gain:", bg='#f0f0f0', 
                font=("Arial", 9, "bold")).pack(side=tk.LEFT)
        tk.Label(gain_frame, textvariable=annualized_gain_var, bg='#f0f0f0', 
                fg="#0066cc", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        
        # Add Total Number of Trades display
        trades_frame = tk.Frame(stats_frame, bg='#f0f0f0')
        trades_frame.pack(fill=tk.X, padx=10, pady=2)
        tk.Label(trades_frame, text="Number of Trades:", bg='#f0f0f0',
                font=("Arial", 9, "bold")).pack(side=tk.LEFT)
        tk.Label(trades_frame, textvariable=trades_count_var, bg='#f0f0f0',
                fg="#0066cc", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        
        # Add Long Trades display
        long_trades_frame = tk.Frame(stats_frame, bg='#f0f0f0')
        long_trades_frame.pack(fill=tk.X, padx=10, pady=2)
        tk.Label(long_trades_frame, text="Long Trades:", bg='#f0f0f0',
                font=("Arial", 9, "bold")).pack(side=tk.LEFT)
        tk.Label(long_trades_frame, textvariable=long_trades_var, bg='#f0f0f0',
                fg="#006600", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        
        # Add Short Trades display
        short_trades_frame = tk.Frame(stats_frame, bg='#f0f0f0')
        short_trades_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
        tk.Label(short_trades_frame, text="Short Trades:", bg='#f0f0f0',
                font=("Arial", 9, "bold")).pack(side=tk.LEFT)
        tk.Label(short_trades_frame, textvariable=short_trades_var, bg='#f0f0f0',
                fg="#990000", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        
        # Add a frame for displaying recent trades - more compact layout
        trades_list_frame = tk.Frame(controls_frame, bg='#f0f0f0', relief=tk.GROOVE, bd=1)
        trades_list_frame.pack(fill=tk.X, pady=5, padx=5)
        
        # Add a title for the trades list - smaller font
        tk.Label(trades_list_frame, text="Recent Trades", font=("Arial", 10, "bold"), 
                bg='#f0f0f0').pack(pady=(3, 1))
        
        # Create a frame for the trade entries with reduced height to ensure button visibility
        trade_entries_frame = tk.Frame(trades_list_frame, bg='#f0f0f0', height=100)
        trade_entries_frame.pack(fill=tk.X, padx=10, pady=3)
        
        # Variable to store all trades for the detailed view
        all_trades = []
        
        # Create a frame to hold text and scrollbar side by side
        text_scroll_frame = tk.Frame(trade_entries_frame, bg='#f0f0f0')
        text_scroll_frame.pack(fill=tk.X, expand=True)
        
        # Create a Text widget to display recent trades (limited to a few)
        trades_text = tk.Text(text_scroll_frame, height=5, width=45, font=("Courier", 10),
                             bg='#f5f5f5', relief=tk.SUNKEN, bd=1)
        trades_text.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Add vertical scrollbar
        trades_scrollbar = tk.Scrollbar(text_scroll_frame, orient=tk.VERTICAL, command=trades_text.yview)
        trades_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        trades_text.config(yscrollcommand=trades_scrollbar.set)
        
        trades_text.insert(tk.END, "No trades to display")
        trades_text.config(state=tk.DISABLED)  # Make it read-only
        
        # Function to display the full list of trades in a new window
        def show_detailed_trades():
            global all_trades, df_copy
            # Get the final price from the valid dataframe for potential end-of-simulation closing prices
            final_price = None
            try:
                final_price = df_copy.dropna()['Price'].iloc[-1]
            except:
                final_price = None
                
            # Debug: Check the contents of all_trades
            print(f"All Trades count: {len(all_trades)}")
            for trade in all_trades:
                print(trade)
                
            if not all_trades:
                messagebox.showinfo("No Trades", "No trades available for the current settings.")
                return
                
            # Create a new window for detailed trades
            trades_window = tk.Toplevel()
            trades_window.title("Detailed Trade List")
            trades_window.geometry("500x500")  # Reasonable size for the trades list
            
            # Create a frame for the window contents
            main_frame = tk.Frame(trades_window, padx=10, pady=10)
            main_frame.pack(fill=tk.BOTH, expand=True)
            
            # Add a title
            tk.Label(main_frame, text="Complete Trade History", 
                    font=("Arial", 14, "bold")).pack(pady=(0, 10))
            
            # Create a frame with scrollbar for the trades list
            list_frame = tk.Frame(main_frame)
            list_frame.pack(fill=tk.BOTH, expand=True)
            
            # Add scrollbars
            v_scrollbar = tk.Scrollbar(list_frame, orient=tk.VERTICAL)
            v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            h_scrollbar = tk.Scrollbar(list_frame, orient=tk.HORIZONTAL)
            h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
            
            # Create a text widget with scrollbars for the trades
            trades_detail = tk.Text(list_frame, font=("Courier", 10), wrap=tk.NONE,
                                  yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
            trades_detail.pack(fill=tk.BOTH, expand=True)
            
            # Connect scrollbars to the text widget
            v_scrollbar.config(command=trades_detail.yview)
            h_scrollbar.config(command=trades_detail.xview)
            
            # Add column headers (fixed width for alignment)
            header = f"{'Action':<15} {'Date':<12} {'Price':>10} {'Prev Short MA':>14} {'Short MA':>12} {'Prev Long MA':>14} {'Long MA':>12} {'Slope':>10} {'Comments':<25}"
            trades_detail.insert(tk.END, header + "\n")
            trades_detail.insert(tk.END, "-" * len(header) + "\n")
            
            # Debug information before displaying trades
            print(f"Showing {len(all_trades)} trades in detailed view")
            
            # Add all trades to the text widget with position tracking for better context
            position = 0  # 0: no position, 1: long position, -1: short position
            last_action = None
            
            print("DEBUG: all_trades:", all_trades)
            for i, trade in enumerate(all_trades):
                # Check if slope check is enabled
                slope_enabled = include_slope_check_var.get() if 'include_slope_check_var' in globals() else False

                action, date, price = trade[:3]
                print(f"Adding trade: {action} on {date} at ${price:.2f}")

                # Get MA values for this date from df_copy
                ma_row = df_copy[df_copy['Date'] == date]
                print(f"DEBUG: ma_row for date {date}:", ma_row)
                short_ma = ma_row['Short_MA'].values[0] if not ma_row.empty else float('nan')
                long_ma = ma_row['Long_MA'].values[0] if not ma_row.empty else float('nan')

                # Get previous day's MA values
                idx_list = ma_row.index.tolist()
                if idx_list and idx_list[0] > 0:
                    prev_row = df_copy.iloc[idx_list[0] - 1]
                    prev_short_ma = prev_row['Short_MA']
                    prev_long_ma = prev_row['Long_MA']
                else:
                    prev_short_ma = float('nan')
                    prev_long_ma = float('nan')

                # Format MA values
                short_ma_str = f"{short_ma:.2f}" if not np.isnan(short_ma) else "N/A"
                long_ma_str = f"{long_ma:.2f}" if not np.isnan(long_ma) else "N/A"
                prev_short_ma_str = f"{prev_short_ma:.2f}" if not np.isnan(prev_short_ma) else "N/A"
                prev_long_ma_str = f"{prev_long_ma:.2f}" if not np.isnan(prev_long_ma) else "N/A"
                if slope_enabled:
                    slope_val = ma_row['Long_MA_Slope'].values[0] if not ma_row.empty and 'Long_MA_Slope' in ma_row.columns else float('nan')
                    slope_str = f"{slope_val:.4f}" if not np.isnan(slope_val) else "N/A"
                else:
                    slope_str = "NA"

                # Add position information for Comments column
                position_info = ""
                if action == "Buy":
                    position = 1
                    position_info = "(Opening long position)"
                elif action == "Sell":
                    position = 0
                    if last_action == "Buy":
                        position_info = "(Closing long position)"
                    else:
                        position_info = "(Selling)"
                elif action == "Short Sell":
                    position = -1
                    position_info = "(Opening short position)"
                elif action == "Cover Short":
                    position = 0
                    position_info = "(Closing short position)"

                # Format aligned row with Comments and previous MA columns and Slope
                row_text = f"{action:<15} {date.strftime('%Y-%m-%d'):<12} {price:>10.2f} {prev_short_ma_str:>14} {short_ma_str:>12} {prev_long_ma_str:>14} {long_ma_str:>12} {slope_str:>10} {position_info:<25}"
                if action == "Buy":
                    trades_detail.insert(tk.END, row_text + "\n", "buy")
                elif action == "Sell":
                    trades_detail.insert(tk.END, row_text + "\n", "sell")
                elif action == "Short Sell":
                    trades_detail.insert(tk.END, row_text + "\n", "short")
                elif action == "Cover Short":
                    trades_detail.insert(tk.END, row_text + "\n", "cover")
                else:
                    trades_detail.insert(tk.END, row_text + "\n")

                # If this is the last trade and position is not 0, add a note
                if i == len(all_trades) - 1 and position != 0:
                    trades_detail.insert(tk.END, f"\nNote: Position still open at end of simulation.\n", "note")
                    if position == 1:
                        trades_detail.insert(tk.END, f"Long position was closed at final price (not shown in trade list)\n", "note")
                    elif position == -1:
                        trades_detail.insert(tk.END, f"Short position was covered at final price (not shown in trade list)\n", "note")

                last_action = action
            
            # Add notes block after the trade list
            trades_detail.insert(tk.END, "\nNote: Trades typically occur in pairs:\n", "note")
            trades_detail.insert(tk.END, "- Buy → Sell (for long trades)\n", "note")
            trades_detail.insert(tk.END, "- Short Sell → Cover Short (for short trades)\n", "note")
            trades_detail.insert(tk.END, "The last trade may not have a matching closing trade if the position\n", "note") 
            trades_detail.insert(tk.END, "is still open at the end of the simulation.\n\n", "note")

            # Configure tags for colors
            trades_detail.tag_configure("buy", foreground="green")
            trades_detail.tag_configure("sell", foreground="red")
            trades_detail.tag_configure("short", foreground="purple")
            trades_detail.tag_configure("cover", foreground="blue") 
            trades_detail.tag_configure("note", foreground="gray", font=("Arial", 9, "italic"))

            # Make it read-only
            trades_detail.config(state=tk.DISABLED)

            # Add a close button
            tk.Button(main_frame, text="Close", command=trades_window.destroy, 
                     width=15, bg="#f0f0f0").pack(pady=10)
        
        # Add a button frame to ensure button is always visible
        button_frame = tk.Frame(trades_list_frame, bg='#f0f0f0')
        button_frame.pack(fill=tk.X, pady=2)
        
        # Add a button to show detailed trades - more compact styling
        more_button = tk.Button(button_frame, text="View All Trades", command=show_detailed_trades,
                              width=15, font=("Arial", 8), padx=2, pady=1)
        more_button.pack()
        
        # Embed the plot in the Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize chart with default values
        update_chart()
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to create chart: {str(e)}")

def show_trades_heatmap():
    """Show a heatmap visualization of the number of trades for different MA combinations."""
    import pandas as pd
    import numpy as np
    
    # Check if results file exists
    combined_file = "ma_strategy_results.csv"
    
    if not os.path.exists(combined_file):
        messagebox.showinfo("No Data Available", 
                          "No trade count data found. Please run a backtest first using the 'Run Backtest' button.")
        return
    
    try:
        # Debug message
        print(f"Attempting to load trade data from {combined_file}")
        
        # The file has metadata lines starting with '#' at the top
        # We'll use pandas comment parameter to skip these lines
        combined_data = pd.read_csv(combined_file, comment='#')
        
        # Check if there's any data
        if combined_data.empty:
            messagebox.showinfo("No Data", "The results file exists but contains no data. Please run a backtest first.")
            return
            
        # Check if we have the required columns
        required_columns = ['short_ma', 'long_ma', 'trade_count']
        if not all(col in combined_data.columns for col in required_columns):
            messagebox.showerror("Error", f"The results file is missing required columns. Found: {', '.join(combined_data.columns)}")
            return
            
        # Pivot the data to get the trade_count_table format
        print(f"Creating pivot table from data with {len(combined_data)} rows")
        trade_count_table = combined_data.pivot(index='long_ma', columns='short_ma', values='trade_count')
        
        # Check if pivot table has data
        if trade_count_table.empty:
            messagebox.showinfo("No Data", "Could not create heatmap - no valid data after processing.")
            return
            
        print(f"Created trade count table with shape {trade_count_table.shape}")
        
    except Exception as e:
        error_message = f"Could not load trade count data: {str(e)}"
        print(error_message)
        messagebox.showerror("Error", error_message)
        return
        
    # Define color bins and colormap for trades (adjust these values based on your data)
    try:
        max_trades = trade_count_table.max().max() if not trade_count_table.empty else 10
        
        # Check if max_trades is valid
        if pd.isna(max_trades) or max_trades <= 0:
            max_trades = 10  # Set a default if invalid
            
        print(f"Max trades value: {max_trades}")
        
        # Create bounds with better distribution for the trades
        min_trades = 0  # Minimum is always 0 for trades
        
        # Create more granular bounds for trade counts
        bounds = np.linspace(min_trades, max_trades, 6)  # 6 bounds means 5 color segments
        
        # Make sure bounds are integers for trade counts
        bounds = np.round(bounds).astype(int)
        
        # Ensure bounds are increasing and unique
        bounds = np.unique(bounds)
        if len(bounds) < 6:  # If we lost some bounds due to uniqueness, add more
            additional = np.arange(bounds[-1] + 1, bounds[-1] + 1 + (6 - len(bounds)))
            bounds = np.append(bounds, additional)
        
        # Add one to the max for proper boundary norm
        if bounds[-1] == max_trades:
            bounds = np.append(bounds, max_trades + 1)
            
        colors = ["#FFFFFF", "#FFCCCC", "#FF9999", "#FF6666", "#FF3333", "#FF0000"]
        cmap = mcolors.ListedColormap(colors)
        
        print(f"Trade count heatmap color bounds: {bounds}")
            
        # Create the heatmap window
        heatmap_window = create_heatmap_window(
            root=root,
            data=trade_count_table,
            title="Trade Count Heatmap",
            value_label="Number of Trades",
            cmap=cmap,
            bounds=bounds,
            colors=colors
        )
        
        # Add a failsafe to ensure window is maximized after a short delay
        heatmap_window.after(100, lambda: ensure_maximized(heatmap_window))
        
        # Add protocol handler to ensure proper closing
        heatmap_window.protocol("WM_DELETE_WINDOW", heatmap_window.destroy)
        
    except Exception as e:
        error_message = f"Error creating trades heatmap: {str(e)}"
        print(error_message)
        messagebox.showerror("Error", error_message)
        return
        
    heatmap_window.mainloop()

def ensure_maximized(window):
    """Ensure a window is maximized properly."""
    try:
        # Try platform-specific approaches
        window.state('zoomed')  # Windows
    except:
        # Fallback - set a very large size
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        window.geometry(f"{screen_width}x{screen_height}+0+0")

def view_results():
    """View the ma_strategy_results.csv file in a scrollable window."""
    # Check if the results file exists
    results_file = "ma_strategy_results.csv"
    if not os.path.exists(results_file):
        messagebox.showinfo("No Results", "No results file found. Please run a backtest first.")
        return
        
    try:
        # Create a new window
        results_window = tk.Toplevel()
        results_window.title("Backtest Results")
        results_window.geometry("800x600")
        
        # Create a frame for the window contents
        main_frame = tk.Frame(results_window, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Add a title
        tk.Label(main_frame, text="Backtest Results", 
                font=("Arial", 14, "bold")).pack(pady=(0, 10))
        
        # Create a frame with scrollbar for the results
        text_frame = tk.Frame(main_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        # Add scrollbars
        v_scrollbar = tk.Scrollbar(text_frame, orient=tk.VERTICAL)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        h_scrollbar = tk.Scrollbar(text_frame, orient=tk.HORIZONTAL)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Create a text widget with scrollbars for the results
        results_text = tk.Text(text_frame, font=("Courier", 10), wrap=tk.NONE,
                             yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        results_text.pack(fill=tk.BOTH, expand=True)
        
        # Connect scrollbars to the text widget
        v_scrollbar.config(command=results_text.yview)
        h_scrollbar.config(command=results_text.xview)
        
        # Read the CSV file
        with open(results_file, 'r') as f:
            results_content = f.read()
        
        # Insert the file content into the text widget
        results_text.insert(tk.END, results_content)
        
        # Make the text read-only
        results_text.config(state=tk.DISABLED)
        
        # Add a close button
        tk.Button(main_frame, text="Close", command=results_window.destroy, 
                width=15).pack(pady=10)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error opening results file: {str(e)}")

def cancel_backtest_callback():
    """Cancel a running backtest."""
    global cancel_backtest
    
    # First check if a backtest is actually running
    if progress_var.get() > 0:
        cancel_backtest = True
        print("Backtest cancellation requested")
        
        # Reset progress bar immediately to give visual feedback
        progress_var.set(0)
        progress_label.config(text="Cancelling...")
        root.update_idletasks()
        
        # Change the button text to show it's working
        for widget in root.winfo_children():
            if isinstance(widget, tk.Frame):
                for child in widget.winfo_children():
                    if isinstance(child, tk.Button) and child['text'] == "Cancel":
                        child.config(text="Cancelling...", state=tk.DISABLED)
                        break
        
        # Show cancellation message
        messagebox.showinfo("Cancelling", "Backtest cancellation has been requested. The process will stop at the next check point.")
        
        # Reset the button text after a short delay
        root.after(1000, lambda: reset_cancel_button())
    else:
        messagebox.showinfo("No Backtest Running", "There is no backtest currently running to cancel.")

def reset_cancel_button():
    """Reset the cancel button text and state after cancellation."""
    for widget in root.winfo_children():
        if isinstance(widget, tk.Frame):
            for child in widget.winfo_children():
                if isinstance(child, tk.Button) and child['text'] == "Cancelling...":
                    child.config(text="Cancel", state=tk.NORMAL)
                    break
    
    # Update the progress label
    progress_label.config(text="Cancelled")
    root.update_idletasks()

def launch_gui():
    global root, file_path_var, start_date_var, end_date_var
    global short_min_var, short_max_var, long_min_var, long_max_var
    global allow_short_selling_var, short_sell_only_var, include_slope_check_var, slope_lookback_var
    global progress_bar, progress_label, heatmap_button, progress_var, trades_heatmap_button
    global trades_count_var, long_trades_var, short_trades_var, annualized_gain_var, all_trades
    
    root = tk.Tk()
    root.title("Moving Average Trading Strategy Analysis")
    root.geometry("600x650")  # Adjusted dimensions for better display of all elements
    
    # Create progress bar variable (widget will be created after buttons)
    progress_var = tk.DoubleVar()
    
    def show_gain_heatmap():
        import pandas as pd
        import numpy as np
        
        # Check if results file exists
        combined_file = "ma_strategy_results.csv"
        
        if not os.path.exists(combined_file):
            messagebox.showinfo("No Data Available", 
                              "No gain data found. Please run a backtest first using the 'Run Backtest' button.")
            return
        
        try:
            # Debug message
            print(f"Attempting to load gain data from {combined_file}")
            
            # The file has metadata lines starting with '#' at the top
            # We'll use pandas comment parameter to skip these lines
            combined_data = pd.read_csv(combined_file, comment='#')
            
            # Check if there's any data
            if combined_data.empty:
                messagebox.showinfo("No Data", "The results file exists but contains no data. Please run a backtest first.")
                return
                
            # Check if we have the required columns
            required_columns = ['short_ma', 'long_ma', 'annualized_gain']
            if not all(col in combined_data.columns for col in required_columns):
                messagebox.showerror("Error", f"The results file is missing required columns. Found: {', '.join(combined_data.columns)}")
                return
                
            # Pivot the data to get the gain_table format
            print(f"Creating pivot table from data with {len(combined_data)} rows")
            gain_table = combined_data.pivot(index='long_ma', columns='short_ma', values='annualized_gain')
            
            # Check if pivot table has data
            if gain_table.empty:
                messagebox.showinfo("No Data", "Could not create heatmap - no valid data after processing.")
                return
                
            print(f"Created gain table with shape {gain_table.shape}")
            
        except Exception as e:
            error_message = f"Could not load annualized %gain data: {str(e)}"
            print(error_message)
            messagebox.showerror("Error", error_message)
            return
            
        # Define color bins and colormap
        try:
            max_gain = gain_table.max().max() if not gain_table.empty else 12
            
            # Check if max_gain is valid
            if pd.isna(max_gain) or max_gain <= 0:
                max_gain = 12  # Set a default if invalid
                
            print(f"Max gain value: {max_gain}")
            
            # Create bounds for gain - making sure to include all color ranges
            min_gain = min(0, gain_table.min().min() if not gain_table.empty else 0)
            max_gain = max(12, gain_table.max().max() if not gain_table.empty else 12)
            
            # Create more granular bounds to ensure all colors are represented
            bounds = np.linspace(min_gain, max_gain, 7)  # 7 bounds means 6 color segments
            
            # Make sure bounds are rounded to nearest 0.5 for cleaner display
            bounds = np.round(bounds * 2) / 2
            
            # Ensure bounds are increasing and unique
            bounds = np.unique(bounds)
            if len(bounds) < 6:  # If we lost some bounds due to uniqueness, add more
                additional = np.linspace(bounds[-1], bounds[-1] + 3, 7 - len(bounds))
                bounds = np.append(bounds, additional[1:])  # Skip the first as it's duplicate
            
            colors = ["#87CEEB", "#90EE90", "#FFFF00", "#FFD700", "#FF8C00", "#FF0000"]
            cmap = mcolors.ListedColormap(colors)
            
            print(f"Gain heatmap color bounds: {bounds}")
            
            # Use the helper function to create the heatmap window
            heatmap_window = create_heatmap_window(
                root=root, 
                data=gain_table, 
                title="Annualized %Gain Heatmap", 
                value_label="Annualized %Gain", 
                cmap=cmap, 
                bounds=bounds, 
                colors=colors
            )
            
            # Add a failsafe to ensure window is maximized after a short delay
            heatmap_window.after(100, lambda: ensure_maximized(heatmap_window))
            
            # Add protocol handler to ensure proper closing
            heatmap_window.protocol("WM_DELETE_WINDOW", heatmap_window.destroy)
            
        except Exception as e:
            error_message = f"Error creating gain heatmap: {str(e)}"
            print(error_message)
            messagebox.showerror("Error", error_message)
            return
            
        heatmap_window.mainloop()
    
    # Create entry widgets and other UI elements with default values
    file_path_var = tk.StringVar(value="C:/Users/Kcrow/OneDrive/Documents/SPY Buy Sell Strategy Analysis/SPY Price History.csv")
    start_date_var = tk.StringVar(value="2000-01-01")
    end_date_var = tk.StringVar(value="2005-01-01")
    short_min_var = tk.IntVar(value=5)
    short_max_var = tk.IntVar(value=20)
    long_min_var = tk.IntVar(value=120)
    long_max_var = tk.IntVar(value=170)
    allow_short_selling_var = tk.BooleanVar(value=False)  # Default is unchecked
    short_sell_only_var = tk.BooleanVar(value=False)  # Default is unchecked
    include_slope_check_var = tk.BooleanVar(value=False)  # Default is unchecked
    slope_lookback_var = tk.IntVar(value=3)  # Default lookback period

    def browse_file():
        filename = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if filename:
            file_path_var.set(filename)

    tk.Label(root, text="Select CSV File:").pack(pady=5)
    # Wider CSV file input box (increased from width=40 to width=60)
    tk.Entry(root, textvariable=file_path_var, width=60).pack(pady=2, padx=10, fill=tk.X)
    # Make browse button smaller with less horizontal padding
    browse_frame = tk.Frame(root)
    browse_frame.pack(pady=2)
    tk.Button(browse_frame, text="Browse", command=browse_file, width=15).pack()

    # Start and End Date fields with consistent layout
    tk.Label(root, text="Start Date (YYYY-MM-DD):").pack(pady=5)
    tk.Entry(root, textvariable=start_date_var, width=20).pack(pady=2)
    tk.Label(root, text="End Date (YYYY-MM-DD):").pack(pady=5)
    tk.Entry(root, textvariable=end_date_var, width=20).pack(pady=2)
    
    # Add Chart button frame
    chart_frame = tk.Frame(root)
    chart_frame.pack(pady=5)
    tk.Button(chart_frame, text="Chart", command=lambda: show_price_chart(file_path_var.get(), 
                                                                     start_date_var.get(), 
                                                                     end_date_var.get()),
              width=15).pack()

    # Short MA Range with improved layout
    tk.Label(root, text="Short MA Range (min, max):").pack(pady=5)
    short_ma_frame = tk.Frame(root)
    short_ma_frame.pack(pady=2)
    tk.Entry(short_ma_frame, textvariable=short_min_var, width=10).pack(side=tk.LEFT, padx=5)
    tk.Label(short_ma_frame, text="to").pack(side=tk.LEFT)
    tk.Entry(short_ma_frame, textvariable=short_max_var, width=10).pack(side=tk.LEFT, padx=5)

    # Long MA Range with improved layout
    tk.Label(root, text="Long MA Range (min, max):").pack(pady=5)
    long_ma_frame = tk.Frame(root)
    long_ma_frame.pack(pady=2)
    tk.Entry(long_ma_frame, textvariable=long_min_var, width=10).pack(side=tk.LEFT, padx=5)
    tk.Label(long_ma_frame, text="to").pack(side=tk.LEFT)
    tk.Entry(long_ma_frame, textvariable=long_max_var, width=10).pack(side=tk.LEFT, padx=5)
    
    # Short selling options frame
    options_frame = tk.Frame(root, pady=5)
    options_frame.pack()
    
    # Allow short selling checkbox
    short_selling_cb = tk.Checkbutton(options_frame, text="Allow Short Selling", 
                                     variable=allow_short_selling_var, 
                                     command=lambda: short_sell_only_cb.configure(
                                         state=tk.NORMAL if allow_short_selling_var.get() else tk.DISABLED))
    short_selling_cb.pack(side=tk.LEFT, padx=5)
    
    # Short sell only checkbox (initially disabled)
    short_sell_only_cb = tk.Checkbutton(options_frame, text="Short Sell Only", 
                                      variable=short_sell_only_var,
                                      state=tk.DISABLED)
    short_sell_only_cb.pack(side=tk.LEFT, padx=5)
    
    # Advanced options frame
    advanced_options_frame = tk.Frame(root, pady=5)
    advanced_options_frame.pack()
    
    # Add a frame with a slider to control the lookback period for slope calculation
    options_slider_frame = tk.Frame(advanced_options_frame, pady=2)
    options_slider_frame.pack(fill=tk.X, padx=2)
    
    # Add a label for the slider
    tk.Label(options_slider_frame, text="Slope Lookback (days):").pack(side=tk.LEFT, padx=5)
    
    # Create a variable for the slope lookback period
    slope_lookback_var = tk.IntVar(value=3)  # Default value
    
    # Add a scale widget (slider)
    slope_slider = tk.Scale(options_slider_frame, from_=1, to=10, orient=tk.HORIZONTAL,
                           variable=slope_lookback_var, length=150)
    slope_slider.pack(side=tk.LEFT, padx=5)
    
    # Include Slope Check checkbox
    slope_check_cb = tk.Checkbutton(advanced_options_frame, text="Include Slope Check", 
                                   variable=include_slope_check_var)
    slope_check_cb.pack(padx=5)
    
    # Create a frame for the run and cancel buttons
    buttons_frame = tk.Frame(root)
    buttons_frame.pack(pady=10)
    
    # Run Backtest button
    run_button = tk.Button(buttons_frame, text="Run Backtest", command=run_backtest, 
                         width=15, bg="#4CAF50", fg="white", font=("Arial", 10, "bold"))
    run_button.pack(side=tk.LEFT, padx=5)
    
    # Cancel button
    cancel_button = tk.Button(buttons_frame, text="Cancel", command=cancel_backtest_callback,
                            width=10, bg="#F44336", fg="white", font=("Arial", 10, "bold"))
    cancel_button.pack(side=tk.LEFT, padx=5)
    
    # View Results button
    view_results_button = tk.Button(root, text="View Results", command=view_results,
                                  width=20, bg="#9C27B0", fg="white", font=("Arial", 10, "bold"))
    view_results_button.pack(pady=5)
    
    # Add Show Heatmap button (initially disabled)
    heatmap_button = tk.Button(root, text="Show Gain Heatmap", command=show_gain_heatmap, 
                             width=20, bg="#2196F3", fg="white", font=("Arial", 10, "bold"),
                             state=tk.DISABLED)
    heatmap_button.pack(pady=5)
    
    # Add Show Trades Heatmap button (initially disabled)
    trades_heatmap_button = tk.Button(root, text="Show Trades Heatmap", command=show_trades_heatmap, 
                                    width=20, bg="#FF9800", fg="white", font=("Arial", 10, "bold"),
                                    state=tk.DISABLED)
    trades_heatmap_button.pack(pady=5)
    
    # Add progress bar for long-running operations
    progress_frame = tk.Frame(root)
    progress_frame.pack(fill=tk.X, padx=20, pady=10)
    
    progress_bar = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, length=550, 
                                  mode='determinate', variable=progress_var)
    progress_bar.pack(side=tk.TOP, fill=tk.X)
    
    progress_label = tk.Label(progress_frame, text="")
    progress_label.pack(side=tk.TOP)
    
    # Add a credits section
    credits_frame = tk.Frame(root, bg='#f0f0f0')
    credits_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=5)
    
    tk.Label(credits_frame, text="Moving Average Trading Strategy Analysis Tool", 
            font=("Arial", 8), fg="#555555", bg='#f0f0f0').pack(pady=(5, 1))
    
    # Try to activate the heatmap button if results file exists
    try:
        # Check if results file exists
        if os.path.exists("ma_strategy_results.csv"):
            heatmap_button.config(state=tk.NORMAL)
            trades_heatmap_button.config(state=tk.NORMAL)
    except:
        # If file doesn't exist, leave buttons disabled
        pass
    
    # Start the main event loop
    root.mainloop()

if __name__ == "__main__":
    launch_gui()
