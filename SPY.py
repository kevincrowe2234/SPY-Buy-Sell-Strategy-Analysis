import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import threading
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

def save_results_with_metadata(results_df, filename, params):
    """
    Save results DataFrame with metadata header information.
    
    Args:
        results_df: DataFrame containing the analysis results
        filename: Output filename
        params: Dictionary containing parameters to include in metadata
    """
    with open(filename, 'w') as f:
        # Write metadata section
        f.write("# Moving Average Trading Strategy Analysis\n")
        f.write(f"# Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("# \n")
        f.write(f"# Input file: {params['input_file']}\n")
        f.write(f"# Date range: {params['start_date']} to {params['end_date']}\n")
        f.write(f"# Short MA range: {params['short_min']} to {params['short_max']}\n")
        f.write(f"# Long MA range: {params['long_min']} to {params['long_max']}\n")
        f.write(f"# Initial investment: ${params['initial_investment']:,.2f}\n")
        f.write("# \n")
        
        if 'best_short' in params and 'best_long' in params:
            f.write(f"# Best combination: Short MA={params['best_short']}, Long MA={params['best_long']}, ")
            f.write(f"Annual Gain={params['best_gain']:.2f}%, Trades={params['best_trades']}\n")
        
        if 'buy_hold_gain' in params:
            f.write(f"# Buy-and-hold performance: Annual Gain={params['buy_hold_gain']:.2f}%\n")
        
        f.write("# \n")
        f.write("# Reserved for future use\n")
        f.write("# Reserved for future use\n")
        f.write("# Reserved for future use\n")
        f.write("# \n")
        f.write("# DATA START\n")
        
        # Write the DataFrame without the index
        results_df.to_csv(f, index=False)

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


# --- GUI Section ---
def run_analysis(file_path, start_date, end_date, short_min, short_max, long_min, long_max, progress_var=None, root=None, run_button=None, progress_bar=None):
    df = load_data(file_path)
    if df is None:
        messagebox.showerror("Error", "Could not load data from CSV. Please check the file format and contents.")
        return
    # Filter by date
    df = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))].reset_index(drop=True)
    if df.empty:
        messagebox.showerror("Error", "No data in selected date range.")
        return
    short_windows = range(short_min, short_max + 1)
    long_windows = range(long_min, long_max + 1)
    initial_investment = 10000
    def annualized_gain(initial, final, start_date, end_date):
        num_days = (end_date - start_date).days
        num_years = num_days / 365.25
        if num_years > 0:
            return ((final / initial) ** (1 / num_years) - 1) * 100
        else:
            return 0.0

    try:
        gain_table = pd.DataFrame(index=long_windows, columns=short_windows, dtype=float)
        trade_count_table = pd.DataFrame(index=long_windows, columns=short_windows, dtype=int)
        result_cache = {}
        
        # Calculate total combinations for progress tracking
        total_combinations = len(long_windows) * len(short_windows)
        completed_combinations = 0
        
        for long_ma in long_windows:
            for short_ma in short_windows:
                # Update progress bar
                if progress_var is not None and root is not None:
                    completed_combinations += 1
                    progress_percent = (completed_combinations / total_combinations) * 100
                    progress_var.set(progress_percent)
                    root.update_idletasks()
                    
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
        
        # For backward compatibility, still save the individual tables
        gain_table.to_csv("annualized_gain_table.csv")
        trade_count_table.to_csv("trade_count_table.csv")
        
        # Create a combined results DataFrame with MultiIndex
        # First, prepare the data in a format suitable for a combined DataFrame
        combined_data = []
        for long_ma in long_windows:
            for short_ma in short_windows:
                if short_ma < long_ma:  # Skip invalid combinations
                    combined_data.append({
                        'long_ma': long_ma,
                        'short_ma': short_ma,
                        'annualized_gain': gain_table.at[long_ma, short_ma],
                        'trade_count': trade_count_table.at[long_ma, short_ma]
                    })
        
        # Create the combined DataFrame
        combined_results = pd.DataFrame(combined_data)
        
        # Find the best combination
        best_combination = max(combined_data, key=lambda x: x['annualized_gain'] if not pd.isna(x['annualized_gain']) else -float('inf'))
        
        # Calculate buy and hold annualized gain
        buy_price = df['Price'].iloc[0]
        sell_price = df['Price'].iloc[-1]
        buy_and_hold_profit = (sell_price - buy_price) * (initial_investment / buy_price)
        buy_and_hold_final = initial_investment + buy_and_hold_profit
        buy_hold_annualized = annualized_gain(initial_investment, buy_and_hold_final, df['Date'].iloc[0], df['Date'].iloc[-1])
        
        # Prepare parameters dictionary
        params_dict = {
            'input_file': file_path,
            'start_date': start_date,
            'end_date': end_date,
            'short_min': short_min,
            'short_max': short_max,
            'long_min': long_min,
            'long_max': long_max,
            'initial_investment': initial_investment,
            'best_short': best_combination['short_ma'],
            'best_long': best_combination['long_ma'],
            'best_gain': best_combination['annualized_gain'],
            'best_trades': best_combination['trade_count'],
            'buy_hold_gain': buy_hold_annualized
        }
        
        # Save the combined results with metadata
        save_results_with_metadata(combined_results, "ma_strategy_results.csv", params_dict)

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
        start_date_df = df['Date'].iloc[0]
        end_date_df = df['Date'].iloc[-1]
        buy_price = df['Price'].iloc[0]
        sell_price = df['Price'].iloc[-1]
        buy_and_hold_profit = (sell_price - buy_price) * (initial_investment / buy_price)
        buy_and_hold_final = initial_investment + buy_and_hold_profit
        annualized_profit_pct = annualized_gain(initial_investment, buy_and_hold_final, start_date_df, end_date_df)

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

        output = []
        output.append(f"Start Date: {start_date_df.strftime('%Y-%m-%d')}")
        output.append(f"End Date: {end_date_df.strftime('%Y-%m-%d')}")
        output.append(f"Buy and Hold (Full Period):")
        output.append(f"  Profit: ${buy_and_hold_profit:.2f}")
        output.append(f"  Final Wealth: ${buy_and_hold_final:.2f}")
        output.append(f"  Annualized % Gain: {annualized_profit_pct:.2f}%")
        if buy_date and sell_date:
            output.append(f"Buy and Hold (First Buy/Last Sell Signals):")
            output.append(f"  Buy on {buy_date.strftime('%Y-%m-%d')} at ${buy_price_signal:.2f}")
            output.append(f"  Sell on {sell_date.strftime('%Y-%m-%d')} at ${sell_price_signal:.2f}")
            output.append(f"  Profit: ${profit_signal:.2f}")
            output.append(f"  Final Wealth: ${final_wealth_signal:.2f}")
            output.append(f"  Annualized % Gain: {annualized_signal_pct:.2f}%")
        else:
            output.append("Buy and Hold (First Buy/Last Sell Signals): Not enough signals to compute.")

        # Optimized strategy annualized % profit
        if best_trades:
            first_trade_date = best_trades[0][1]
            last_trade_date = best_trades[-1][1]
            annualized_best_pct = annualized_gain(initial_investment, best_wealth, first_trade_date, last_trade_date)
        else:
            annualized_best_pct = 0.0

        output.append(f"Best Short MA Window: {best_short} days")
        output.append(f"Best Long MA Window: {best_long} days")
        output.append(f"Final Wealth: ${best_wealth:.2f}")
        output.append(f"Profit: ${best_wealth - initial_investment:.2f}")
        output.append(f"Annualized % Gain (Optimized Strategy): {annualized_best_pct:.2f}%")
        output.append(f"Number of Trades: {len(best_trades)}")
        output.append("\nTrade Log:")
        for trade in best_trades:
            action, date, price = trade
            output.append(f"{action} on {date.strftime('%Y-%m-%d')} at ${price:.2f}")

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
            output.append(f"\n--- Custom MA Case: Short={custom_short} Long={custom_long} ---")
            output.append(f"Final Wealth: ${custom_wealth:.2f}")
            output.append(f"Profit: ${custom_wealth - initial_investment:.2f}")
            output.append(f"Annualized % Gain (Custom MA): {annualized_custom_pct:.2f}%")
            output.append(f"Number of Trades: {len(custom_trades)}")
            output.append("Trade Log:")
            for trade in custom_trades:
                action, date, price = trade
                output.append(f"{action} on {date.strftime('%Y-%m-%d')} at ${price:.2f}")
        
        # Hide progress bar and restore run button when complete
        if root is not None and progress_bar is not None and run_button is not None:
            # We need to use a small delay to ensure this happens after the progress bar has been updated
            def restore_ui():
                progress_var.set(0)
                progress_bar.pack_forget()  # Hide progress bar
                run_button.pack(pady=5)  # Show run button again with consistent styling
            
            root.after(100, restore_ui)
            
        messagebox.showinfo("Analysis Complete", "\n".join(output))
    except Exception as e:
        messagebox.showerror("Error during optimization", str(e))

def create_heatmap_window(root, data, title, value_label, cmap=None, bounds=None, colors=None):
    """Helper function to create heatmap windows with consistent layout and sizing"""
    import pandas as pd
    import numpy as np
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    
    # Create a new Toplevel window for the heatmap
    heatmap_window = tk.Toplevel(root)
    heatmap_window.title(title)
    
    # Set window state to zoomed (maximized) by default
    heatmap_window.state('zoomed')
    
    # Create the main container frame with padding
    main_frame = tk.Frame(heatmap_window, padx=10, pady=10)
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # We'll use a different approach with grid layout for better control
    main_frame.grid_rowconfigure(0, weight=1)  # Plot area takes all available space
    main_frame.grid_rowconfigure(1, weight=0)  # Info label has fixed height
    main_frame.grid_columnconfigure(0, weight=1)  # Both widgets span full width
    
    # Create the figure frame first - it will take row 0
    fig_frame = tk.Frame(main_frame)
    fig_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
    
    # Create a fixed height frame for the info label at the bottom in row 1
    info_frame = tk.Frame(main_frame, height=80, bg='#f0f0f0')
    info_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=(0, 10))
    info_frame.grid_propagate(False)  # Prevents the frame from shrinking to label size
    
    # Create the info label with more prominent styling
    info_label = tk.Label(info_frame, text="Hover over cells to see details", 
                         font=("Arial", 14, "bold"), bg="white", relief="ridge", 
                         padx=15, pady=10, borderwidth=2)
    # Center the label within its frame
    info_frame.grid_columnconfigure(0, weight=1)
    info_frame.grid_rowconfigure(0, weight=1)
    info_label.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
    
    # Create matplotlib figure - better sizing for maximized window
    fig, ax = plt.subplots(figsize=(14, 9.0))
    
    # Generate the heatmap
    if cmap is None:
        cmap = plt.cm.viridis
        im = ax.imshow(data.values, aspect='auto', cmap=cmap, origin='lower')
    else:
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        im = ax.imshow(data.values, aspect='auto', cmap=cmap, norm=norm, origin='lower')
    
    # Configure axes and labels
    ax.set_xlabel("Short duration simple moving average")
    ax.set_ylabel("Longer duration simple moving average")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(data.columns)))
    ax.set_xticklabels(data.columns)
    ax.set_yticks(np.arange(len(data.index)))
    ax.set_yticklabels(data.index)
    plt.xticks(rotation=90)
    
    # Add colorbar or legend
    if bounds is not None and colors is not None:
        from matplotlib.patches import Patch
        legend_labels = []
        for i in range(len(bounds) - 1):
            if i == len(bounds) - 2:
                legend_labels.append(f"{value_label} > {bounds[i]}")
            else:
                legend_labels.append(f"{value_label} between {bounds[i]} and {bounds[i+1]}")
        
        legend_patches = [Patch(color=colors[i], label=legend_labels[i]) for i in range(len(legend_labels))]
        ax.legend(handles=legend_patches, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize='small')
    else:
        cbar = plt.colorbar(im)
        cbar.set_label(value_label)
    
    # Adjust layout for maximized window
    plt.tight_layout(rect=[0, 0, 0.95, 0.98])
    
    # Embed the matplotlib figure in the Tkinter window using grid for better control
    canvas = FigureCanvasTkAgg(fig, master=fig_frame)
    canvas.draw()
    
    # Configure the figure frame to expand properly
    fig_frame.grid_rowconfigure(0, weight=1)
    fig_frame.grid_columnconfigure(0, weight=1)
    
    # Place the canvas in the grid
    canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
    
    # Function to handle mouse hover events
    def on_hover(event):
        if event.inaxes == ax:
            col = int(event.xdata + 0.5)
            row = int(event.ydata + 0.5)
            if 0 <= row < len(data.index) and 0 <= col < len(data.columns):
                short_ma = data.columns[col]
                long_ma = data.index[row]
                value = data.iloc[row, col]
                if not np.isnan(value):
                    info_text = f"Short MA: {short_ma}, Long MA: {long_ma}, {value_label}: {value:.2f}"
                    if "%" not in value_label and "Gain" in value_label:
                        info_text += "%"
                    info_label.config(text=info_text)
    
    # Function to handle mouse click events
    def on_click(event):
        if event.inaxes == ax:
            col = int(event.xdata + 0.5)
            row = int(event.ydata + 0.5)
            if 0 <= row < len(data.index) and 0 <= col < len(data.columns):
                short_ma = data.columns[col]
                long_ma = data.index[row]
                value = data.iloc[row, col]
                if not np.isnan(value):
                    detail_text = f"Detailed Information:\n\n"
                    detail_text += f"Short Moving Average: {short_ma} days\n"
                    detail_text += f"Long Moving Average: {long_ma} days\n"
                    
                    if "Gain" in title:
                        detail_text += f"Annualized Gain: {value:.2f}%\n"
                        # Try to get trade count data
                        try:
                            trade_count_table = pd.read_csv("trade_count_table.csv", index_col=0)
                            trade_count = trade_count_table.iloc[row, col]
                            if not np.isnan(trade_count):
                                detail_text += f"Number of Trades: {int(trade_count)}\n"
                        except:
                            pass
                    elif "Trade" in title:
                        detail_text += f"Number of Trades: {int(value)}\n"
                        # Try to get gain data
                        try:
                            gain_table = pd.read_csv("annualized_gain_table.csv", index_col=0)
                            gain = gain_table.iloc[row, col]
                            if not np.isnan(gain):
                                detail_text += f"Annualized Gain: {gain:.2f}%\n"
                        except:
                            pass
                    
                    messagebox.showinfo(f"Cell Details (Short MA: {short_ma}, Long MA: {long_ma})", detail_text)
    
    # Connect the event handlers to the canvas
    fig.canvas.mpl_connect('motion_notify_event', on_hover)
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    # Make sure the window appears in the right state
    heatmap_window.update_idletasks()  # Process pending events
    
    # Ensure window is maximized (zoomed state) and has focus
    heatmap_window.state('zoomed')
    heatmap_window.lift()
    heatmap_window.focus_force()
    
    return heatmap_window

def launch_gui():
    root = tk.Tk()
    root.title("Moving Average Trading Strategy Analysis")
    root.geometry("600x650")  # Adjusted dimensions for better display of all elements
    
    # Create progress bar variable (widget will be created after buttons)
    progress_var = tk.DoubleVar()
    
    def show_gain_heatmap():
        import pandas as pd
        import numpy as np
        
        # Try to load from the new combined file format first
        try:
            # Skip the metadata header lines
            with open("ma_strategy_results.csv", 'r') as f:
                line = f.readline()
                skiprows = 0
                while not line.startswith("long_ma,short_ma"):
                    line = f.readline()
                    skiprows += 1
                    if skiprows > 100:  # Safety check
                        break
            
            # Read the combined file, skipping metadata
            combined_data = pd.read_csv("ma_strategy_results.csv", skiprows=skiprows)
            
            # Pivot the data to get the gain_table format
            gain_table = combined_data.pivot(index='long_ma', columns='short_ma', values='annualized_gain')
            
        except Exception:
            # Fall back to the old file format if new format isn't available or has issues
            try:
                gain_table = pd.read_csv("annualized_gain_table.csv", index_col=0)
            except Exception as e:
                messagebox.showerror("Error", f"Could not load annualized gain data: {e}")
                return
            
        # Define color bins and colormap
        bounds = [6, 7, 8, 9, 10, gain_table.max().max()+1]
        colors = ["#87CEEB", "#90EE90", "#FFFF00", "#FFD700", "#FF8C00", "#FF0000"]
        cmap = mcolors.ListedColormap(colors)
        
        # Use the helper function to create the heatmap window
        heatmap_window = create_heatmap_window(
            root=root, 
            data=gain_table, 
            title="Annualized Gain Heatmap", 
            value_label="Annualized Gain", 
            cmap=cmap, 
            bounds=bounds, 
            colors=colors
        )
        
        # Add a failsafe to ensure window is maximized after a short delay
        heatmap_window.after(100, lambda: ensure_maximized(heatmap_window))
        
        heatmap_window.mainloop()
    
    # Create entry widgets and other UI elements with default values
    file_path_var = tk.StringVar(value="C:/Users/Kcrow/OneDrive/Documents/SPY Buy Sell Strategy Analysis/SPY Price History.csv")
    start_date_var = tk.StringVar(value="2000-01-01")
    end_date_var = tk.StringVar(value="2005-01-01")
    short_min_var = tk.IntVar(value=5)
    short_max_var = tk.IntVar(value=20)
    long_min_var = tk.IntVar(value=120)
    long_max_var = tk.IntVar(value=170)

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

    # Short MA Range with improved layout
    tk.Label(root, text="Short MA Range (min, max):").pack(pady=5)
    short_ma_frame = tk.Frame(root)
    short_ma_frame.pack(pady=2)
    tk.Entry(short_ma_frame, textvariable=short_min_var, width=10).pack(side=tk.LEFT, padx=5)
    tk.Entry(short_ma_frame, textvariable=short_max_var, width=10).pack(side=tk.LEFT, padx=5)
    
    # Long MA Range with improved layout
    tk.Label(root, text="Long MA Range (min, max):").pack(pady=5)
    long_ma_frame = tk.Frame(root)
    long_ma_frame.pack(pady=2)
    tk.Entry(long_ma_frame, textvariable=long_min_var, width=10).pack(side=tk.LEFT, padx=5)
    tk.Entry(long_ma_frame, textvariable=long_max_var, width=10).pack(side=tk.LEFT, padx=5)
    
    # Create a frame for the buttons and progress bar to ensure consistent layout
    button_frame = tk.Frame(root)
    button_frame.pack(pady=10, fill=tk.X)
    
    # Create inner frame to control button width
    run_inner_frame = tk.Frame(button_frame)
    run_inner_frame.pack()
    
    # Create progress bar widget but don't make it visible yet
    progress_bar = ttk.Progressbar(button_frame, variable=progress_var, maximum=100)
    
    def run():
        file_path = file_path_var.get()
        start_date = start_date_var.get()
        end_date = end_date_var.get()
        short_min = short_min_var.get()
        short_max = short_max_var.get()
        long_min = long_min_var.get()
        long_max = long_max_var.get()
        if not file_path or not start_date or not end_date:
            messagebox.showerror("Missing Input", "Please fill in all fields.")
            return
            
        # Show the progress bar
        run_button.pack_forget()  # Remove the Run button
        progress_bar.pack(fill=tk.X, padx=20)  # Show the progress bar
        progress_bar.update_idletasks()
        
        # Start analysis in a separate thread
        threading.Thread(target=run_analysis, args=(file_path, start_date, end_date, short_min, short_max, long_min, long_max, progress_var, root, run_button, progress_bar)).start()

    # Create the Run Analysis button - narrower with width parameter
    run_button = tk.Button(run_inner_frame, text="Run Analysis", command=run, width=20)
    run_button.pack(pady=5)
    
    # Helper function to ensure window is maximized
    def ensure_maximized(window):
        # Try different approaches to maximize the window
        try:
            # For Windows
            window.state('zoomed')
        except:
            try:
                # For some Linux/Mac systems
                window.attributes('-zoomed', True)
            except:
                # Fallback - set a very large size
                screen_width = window.winfo_screenwidth()
                screen_height = window.winfo_screenheight()
                window.geometry(f"{screen_width}x{screen_height}+0+0")
    
    # Create function for showing trades heatmap
    def show_trades_heatmap():
        import pandas as pd
        import numpy as np
        
        # Try to load from the new combined file format first
        try:
            # Skip the metadata header lines
            with open("ma_strategy_results.csv", 'r') as f:
                line = f.readline()
                skiprows = 0
                while not line.startswith("long_ma,short_ma"):
                    line = f.readline()
                    skiprows += 1
                    if skiprows > 100:  # Safety check
                        break
            
            # Read the combined file, skipping metadata
            combined_data = pd.read_csv("ma_strategy_results.csv", skiprows=skiprows)
            
            # Pivot the data to get the trade_count_table format
            trade_count_table = combined_data.pivot(index='long_ma', columns='short_ma', values='trade_count')
            
        except Exception:
            # Fall back to the old file format if new format isn't available or has issues
            try:
                trade_count_table = pd.read_csv("trade_count_table.csv", index_col=0)
            except Exception as e:
                messagebox.showerror("Error", f"Could not load trade count data: {e}")
                return
            
        # Use the helper function to create the heatmap window
        heatmap_window = create_heatmap_window(
            root=root,
            data=trade_count_table,
            title="Trade Count Heatmap",
            value_label="Number of Trades"
        )
        
        # Add a failsafe to ensure window is maximized after a short delay
        heatmap_window.after(100, lambda: ensure_maximized(heatmap_window))
        
        heatmap_window.mainloop()
    
    # Create a frame for heatmap buttons
    heatmap_frame = tk.Frame(root)
    heatmap_frame.pack(pady=10, fill=tk.X)
    
    # Create inner frames to control button width
    gain_button_frame = tk.Frame(heatmap_frame)
    gain_button_frame.pack(pady=2)
    trades_button_frame = tk.Frame(heatmap_frame)
    trades_button_frame.pack(pady=2)
    
    # Add both heatmap buttons - narrower with width parameter
    tk.Button(gain_button_frame, text="Show Gain Heatmap", command=show_gain_heatmap, width=20).pack()
    tk.Button(trades_button_frame, text="Show Trades Heatmap", command=show_trades_heatmap, width=20).pack()
    
    root.mainloop()

if __name__ == "__main__":
    launch_gui()