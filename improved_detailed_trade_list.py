"""
Improved Detailed Trade List Function for SPY Buy Sell Strategy Analysis

This file contains an improved version of the show_detailed_trades function
that adds comprehensive information to the detailed trade list view, including:

1. Moving Average Values (current and previous day)
2. Slope information (when enabled)
3. Gain/Loss calculations for each closing trade
4. Final price trade simulation for open positions
5. Better organization and explanations

Instructions:
1. Replace your show_detailed_trades function in SPY.py with this version
2. Make sure to add the include_slope_check_var and slope_lookback_var variables
   to your globals if you want to use the slope check feature
"""

import tkinter as tk
from tkinter import messagebox, Toplevel
import pandas as pd
import numpy as np
from datetime import datetime

def show_detailed_trades(all_trades, filtered_df, trades_count_var, long_trades_var, 
                         short_trades_var, annualized_gain_var, short_ma_var, long_ma_var,
                         include_slope_check_var=None, slope_lookback_var=None):
    """
    Enhanced function to show detailed trade list with comprehensive information.
    
    Parameters:
    - all_trades: List of trade tuples in format (action, date, price)
    - filtered_df: DataFrame with price data
    - trades_count_var, long_trades_var, short_trades_var, annualized_gain_var: StringVars for stats
    - short_ma_var, long_ma_var: StringVars containing MA period values
    - include_slope_check_var: BooleanVar for enabling slope check (optional)
    - slope_lookback_var: IntVar for slope lookback period (optional)
    """
    
    # Enable slope checking if the variables exist
    slope_check_enabled = False
    slope_lookback = 3  # Default value
    
    if include_slope_check_var is not None:
        try:
            slope_check_enabled = include_slope_check_var.get()
            if slope_check_enabled and slope_lookback_var is not None:
                slope_lookback = slope_lookback_var.get()
        except:
            slope_check_enabled = False
    
    if not all_trades:
        messagebox.showinfo("No Trades", "No trades available for the current settings.")
        return
    
    # Get current chart dataframe with MAs for the detailed trade info
    try:
        # Get the current short and long MA periods from the entry fields
        short_ma = int(short_ma_var.get())
        long_ma = int(long_ma_var.get())
        
        # Recalculate MAs with these periods
        df_with_ma = filtered_df.copy()
        df_with_ma['Short_MA'] = df_with_ma['Price'].rolling(window=short_ma).mean()
        df_with_ma['Long_MA'] = df_with_ma['Price'].rolling(window=long_ma).mean()
        
        # Calculate slopes if needed
        if slope_check_enabled:
            for i in range(len(df_with_ma)):
                if i >= slope_lookback:
                    # Calculate short MA slope
                    short_vals = df_with_ma['Short_MA'].iloc[i-slope_lookback:i+1].values
                    if not np.isnan(short_vals).any():
                        x = np.arange(len(short_vals))
                        df_with_ma.loc[df_with_ma.index[i], 'Short_MA_Slope'] = np.polyfit(x, short_vals, 1)[0]
                    
                    # Calculate long MA slope
                    long_vals = df_with_ma['Long_MA'].iloc[i-slope_lookback:i+1].values
                    if not np.isnan(long_vals).any():
                        x = np.arange(len(long_vals))
                        df_with_ma.loc[df_with_ma.index[i], 'Long_MA_Slope'] = np.polyfit(x, long_vals, 1)[0]
    except Exception as e:
        print(f"Error preparing MA data for detailed trades: {e}")
        df_with_ma = None
        
    # Create a new window for detailed trades
    trades_window = tk.Toplevel()
    trades_window.title("Detailed Trade List")
    trades_window.geometry("800x600")  # Larger size to accommodate more columns
    
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
    
    # Function to look up MA values for a specific date
    def get_ma_values(date):
        if df_with_ma is None:
            return (float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'))
        
        # Find this date in the dataframe
        trade_row = df_with_ma[df_with_ma['Date'] == date]
        
        if not trade_row.empty:
            short_ma = trade_row['Short_MA'].values[0]
            long_ma = trade_row['Long_MA'].values[0]
            
            # Get previous day's values if possible
            idx = df_with_ma.index[df_with_ma['Date'] == date].tolist()
            if idx and idx[0] > 0:
                prev_row = df_with_ma.iloc[idx[0] - 1]
                prev_short_ma = prev_row['Short_MA'] 
                prev_long_ma = prev_row['Long_MA']
            else:
                prev_short_ma = float('nan')
                prev_long_ma = float('nan')
            
            # Get slope values if enabled
            if slope_check_enabled:
                if 'Short_MA_Slope' in trade_row.columns:
                    short_slope = trade_row['Short_MA_Slope'].values[0]
                    long_slope = trade_row['Long_MA_Slope'].values[0]
                else:
                    # Default values if not calculated
                    short_slope = float('nan')
                    long_slope = float('nan')
            else:
                short_slope = float('nan')
                long_slope = float('nan')
        
            return (short_ma, long_ma, prev_short_ma, prev_long_ma, short_slope, long_slope)
        else:
            return (float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'))
    
    # Add column headers with expanded information
    header = "Action          Date            Price      "
    header += "Short MA    Long MA     "
    header += "Prev Short  Prev Long   "
    
    if slope_check_enabled:
        header += "Short Slope Long Slope  "
    
    header += "% Gain/Loss"
    trades_detail.insert(tk.END, header + "\n")
    
    # Add a separator line matching the header width
    separator = "-" * len(header)
    trades_detail.insert(tk.END, separator + "\n")
    
    # Add all trades to the text widget with position tracking for better context
    position = 0  # 0: no position, 1: long position, -1: short position
    last_action = None
    entry_price = 0  # Track entry price for calculating gain/loss
    entry_date = None  # Track entry date
    
    for i, trade in enumerate(all_trades):
        action, date, price = trade[:3]
        
        # Get MA values for this date
        short_ma, long_ma, prev_short_ma, prev_long_ma, short_slope, long_slope = get_ma_values(date)
        
        # Calculate gain/loss for closing trades
        gain_loss = "N/A"
        if (action == "Sell" and last_action == "Buy") or (action == "Cover Short" and last_action == "Short Sell"):
            if entry_price > 0:
                if action == "Sell":
                    pct_gain = ((price - entry_price) / entry_price) * 100
                else:  # Cover Short
                    pct_gain = ((entry_price - price) / entry_price) * 100
                gain_loss = f"{pct_gain:+.2f}%" if not np.isnan(pct_gain) else "N/A"
        
        # Format MA values and slopes
        short_ma_str = f"{short_ma:.2f}" if not np.isnan(short_ma) else "N/A"
        long_ma_str = f"{long_ma:.2f}" if not np.isnan(long_ma) else "N/A"
        prev_short_str = f"{prev_short_ma:.2f}" if not np.isnan(prev_short_ma) else "N/A"
        prev_long_str = f"{prev_long_ma:.2f}" if not np.isnan(prev_long_ma) else "N/A"
        
        # Format slope values if enabled
        if slope_check_enabled:
            short_slope_str = f"{short_slope:.4f}" if not np.isnan(short_slope) else "N/A"
            long_slope_str = f"{long_slope:.4f}" if not np.isnan(long_slope) else "N/A"
            slope_info = f"{short_slope_str:<11} {long_slope_str:<11} "
        else:
            slope_info = ""
        
        # Build the row with all the data
        trade_line = f"{action:<15} {date.strftime('%Y-%m-%d'):<15} ${price:<9.2f} "
        trade_line += f"{short_ma_str:<11} {long_ma_str:<11} "
        trade_line += f"{prev_short_str:<11} {prev_long_str:<11} "
        trade_line += slope_info
        trade_line += f"{gain_loss:<10}"
        
        # Add position information for context
        position_info = ""
        
        if action == "Buy":
            position = 1
            entry_price = price
            entry_date = date
            position_info = "(Opening long position)"
            tag = "buy"
        elif action == "Sell":
            position = 0
            # Check if this closes a long position
            if last_action == "Buy":
                position_info = "(Closing long position)"
            else:
                position_info = "(Selling)"
            tag = "sell"
            entry_price = 0
        elif action == "Short Sell":
            position = -1
            entry_price = price
            entry_date = date
            position_info = "(Opening short position)"
            tag = "short"
        elif action == "Cover Short":
            position = 0
            position_info = "(Closing short position)"
            tag = "cover"
            entry_price = 0
        else:
            tag = ""
        
        # Insert the line with appropriate color
        trades_detail.insert(tk.END, trade_line + f" {position_info}\n", tag)
        
        last_action = action
    
    # If this is the last trade and position is not 0, add a simulated closing trade
    if position != 0:
        # Get the final price from the dataframe
        final_price = df_with_ma['Price'].iloc[-1] if df_with_ma is not None and not df_with_ma.empty else 0
        final_date = df_with_ma['Date'].iloc[-1] if df_with_ma is not None and not df_with_ma.empty else None
        
        if final_date is not None:
            # Get MA values for final date
            short_ma, long_ma, prev_short_ma, prev_long_ma, short_slope, long_slope = get_ma_values(final_date)
            
            # Calculate gain/loss
            if entry_price > 0:
                if position == 1:  # Long position
                    pct_gain = ((final_price - entry_price) / entry_price) * 100
                else:  # Short position
                    pct_gain = ((entry_price - final_price) / entry_price) * 100
                gain_loss = f"{pct_gain:+.2f}%" if not np.isnan(pct_gain) else "N/A"
            
            # Format MA values
            short_ma_str = f"{short_ma:.2f}" if not np.isnan(short_ma) else "N/A"
            long_ma_str = f"{long_ma:.2f}" if not np.isnan(long_ma) else "N/A"
            prev_short_str = f"{prev_short_ma:.2f}" if not np.isnan(prev_short_ma) else "N/A"
            prev_long_str = f"{prev_long_ma:.2f}" if not np.isnan(prev_long_ma) else "N/A"
            
            # Format slope values if enabled
            if slope_check_enabled:
                short_slope_str = f"{short_slope:.4f}" if not np.isnan(short_slope) else "N/A"
                long_slope_str = f"{long_slope:.4f}" if not np.isnan(long_slope) else "N/A"
                slope_info = f"{short_slope_str:<11} {long_slope_str:<11} "
            else:
                slope_info = ""
            
            # Add a simulated closing trade
            action = "Sell*" if position == 1 else "Cover Short*"
            
            # Build the row with all the data
            trade_line = f"{action:<15} {final_date.strftime('%Y-%m-%d'):<15} ${final_price:<9.2f} "
            trade_line += f"{short_ma_str:<11} {long_ma_str:<11} "
            trade_line += f"{prev_short_str:<11} {prev_long_str:<11} "
            trade_line += slope_info
            trade_line += f"{gain_loss:<10}"
            
            position_info = "(Simulated closing trade)"
            
            # Use appropriate tag color
            tag = "sell" if position == 1 else "cover"
            trades_detail.insert(tk.END, trade_line + f" {position_info}\n", tag)
            
            # Add note explaining the asterisk
            trades_detail.insert(tk.END, f"\n* Position was closed at final price to calculate annual gain/loss.\n", "note")
    
    # Add a header explaining the trade pairs (moved from top to here)
    trades_detail.insert(tk.END, "\n\nNote: Trades typically occur in pairs:\n", "note")
    trades_detail.insert(tk.END, "- Buy → Sell (for long trades)\n", "note")
    trades_detail.insert(tk.END, "- Short Sell → Cover Short (for short trades)\n", "note")
    
    # Add a summary section to explain the relationship between trades and performance
    trades_detail.insert(tk.END, "\n--------------------------------------\n", "header")
    trades_detail.insert(tk.END, "SUMMARY\n", "header")
    trades_detail.insert(tk.END, "--------------------------------------\n", "header")
    
    # Get the current values from the strategy performance display
    trades_detail.insert(tk.END, f"Total Trades: {trades_count_var.get()}\n", "summary")
    trades_detail.insert(tk.END, f"Long Trades: {long_trades_var.get()}\n", "summary")
    trades_detail.insert(tk.END, f"Short Trades: {short_trades_var.get()}\n", "summary")
    trades_detail.insert(tk.END, f"Annualized Gain: {annualized_gain_var.get()}\n\n", "summary")
    
    # Add explanation about trade counts and display
    trades_detail.insert(tk.END, "NOTE ABOUT TRADE COUNTS:\n", "note")
    trades_detail.insert(tk.END, "The 'Recent Trades' section in the main window only shows a \n", "note")
    trades_detail.insert(tk.END, "selection of trades - typically the first and last few trades \n", "note")
    trades_detail.insert(tk.END, "for context. This can make it appear that there are fewer trades \n", "note")
    trades_detail.insert(tk.END, "than indicated by the strategy performance metrics.\n\n", "note")
    
    trades_detail.insert(tk.END, "This detailed view shows ALL trades that occurred during the simulation.\n", "note")
    trades_detail.insert(tk.END, "The performance metrics are calculated based on ALL trades, not just \n", "note")
    trades_detail.insert(tk.END, "the ones shown in the 'Recent Trades' section.\n", "note")
    
    # Configure tags for colors
    trades_detail.tag_configure("buy", foreground="green")
    trades_detail.tag_configure("sell", foreground="red")
    trades_detail.tag_configure("short", foreground="purple")
    trades_detail.tag_configure("cover", foreground="blue") 
    trades_detail.tag_configure("note", foreground="gray", font=("Arial", 9, "italic"))
    trades_detail.tag_configure("header", font=("Arial", 10, "bold"))
    trades_detail.tag_configure("summary", font=("Arial", 10))
    
    # Make it read-only
    trades_detail.config(state=tk.DISABLED)
    
    # Add a close button
    tk.Button(main_frame, text="Close", command=trades_window.destroy, 
             width=15, bg="#f0f0f0").pack(pady=10)


# Example of how to call this function:
"""
# In your show_price_chart function, replace the show_detailed_trades function with:

def show_detailed_trades():
    # Import the improved function
    from improved_detailed_trade_list import show_detailed_trades as improved_show_detailed_trades
    
    if not all_trades:
        messagebox.showinfo("No Trades", "No trades available for the current settings.")
        return
    
    # Call the improved function with all necessary parameters
    improved_show_detailed_trades(
        all_trades=all_trades,
        filtered_df=filtered_df, 
        trades_count_var=trades_count_var,
        long_trades_var=long_trades_var, 
        short_trades_var=short_trades_var,
        annualized_gain_var=annualized_gain_var,
        short_ma_var=short_ma_var,
        long_ma_var=long_ma_var,
        include_slope_check_var=include_slope_check_var if 'include_slope_check_var' in globals() else None,
        slope_lookback_var=slope_lookback_var if 'slope_lookback_var' in globals() else None
    )
"""
