"""
Trading Strategy Analysis Tool

This tool provides a comprehensive analysis of moving average crossover trading strategies.
It reads CSV files with date and price data, performs backtesting of MA crossover strategies,
and displays results through interactive GUI with heatmaps and price charts.

Features:
- Modular design with TradingStrategy, Visualization, and TradingGUI classes
- Support for slope-based signal filtering
- Interactive heatmaps for strategy performance visualization
- Detailed trade analysis and charting
- Configurable moving average ranges and trading parameters

Author: Generated from MA_TradingStrategy.py
Date: September 9, 2025
"""

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
import logging
import unittest
import threading
import csv
import difflib
from dateutil import parser as date_parser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_strategy.log'),
        logging.StreamHandler()
    ]
)

# === Trading Strategy Class ===

class TradingStrategy:
    """
    Handles data loading, moving average calculations, signal generation, and backtesting.

    This class encapsulates all trading strategy logic including data processing,
    technical indicator calculations, and performance evaluation.
    """

    def __init__(self):
        """Initialize the trading strategy with empty cache."""
        self.ma_cache = {}  # Cache for moving average calculations

    def _detect_csv_structure(self, file_path):
        """
        Detect CSV structure using built-in csv.Sniffer.

        Args:
            file_path (str): Path to the CSV file

        Returns:
            dict: Detection results with delimiter, quotechar, has_header, etc.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                sample = f.read(1024)  # Read first 1KB for detection

            # Use csv.Sniffer to detect structure
            sniffer = csv.Sniffer()

            # Detect delimiter
            try:
                delimiter = sniffer.sniff(sample, delimiters=',;\t|').delimiter
            except csv.Error:
                delimiter = ','  # Default fallback

            # Detect quote character
            try:
                quotechar = sniffer.sniff(sample).quotechar
            except csv.Error:
                quotechar = '"'  # Default fallback

            # Check for headers
            try:
                has_header = sniffer.has_header(sample)
            except csv.Error:
                has_header = True  # Default assumption

            # Additional check: if first column looks like dates, likely no headers
            if has_header:
                lines = sample.split('\n')[:5]  # Check first 5 lines
                if lines:
                    first_line = lines[0].strip()
                    if first_line:
                        first_values = first_line.split(delimiter)
                        if len(first_values) >= 2:
                            first_col_value = first_values[0].strip()
                            # Check if first column value looks like a date
                            try:
                                date_parser.parse(first_col_value)
                                # If we can parse it as a date, likely no headers
                                has_header = False
                                logging.info(f"Detected first column as date '{first_col_value}', setting has_header=False")
                            except (ValueError, TypeError):
                                pass

            return {
                'delimiter': delimiter,
                'quotechar': quotechar,
                'has_header': has_header,
                'confidence': 0.8 if has_header else 0.6  # Base confidence
            }

        except Exception as e:
            logging.warning(f"CSV structure detection failed: {e}")
            return {
                'delimiter': ',',
                'quotechar': '"',
                'has_header': False,  # Default to no headers for safety
                'confidence': 0.3  # Low confidence fallback
            }

    def _detect_date_format(self, date_samples):
        """
        Detect date format from sample date strings with proper disambiguation.

        Args:
            date_samples (list): List of date strings to analyze

        Returns:
            dict: Detected date format info or None if no format detected
        """
        if not date_samples:
            return None

        successful_parses = []
        format_counts = {}
        
        # Take more samples to improve detection accuracy
        sample_size = min(20, len(date_samples))

        for date_str in date_samples[:sample_size]:
            try:
                # Strip the date string
                date_str = date_str.strip()
                # Strip BOM if present
                date_str = date_str.lstrip('\ufeff')
                
                if not date_str:
                    continue

                # Convert back to string to detect format
                if '/' in date_str:
                    if date_str.count('/') == 2:
                        # Try different common formats
                        for fmt in ['%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d']:
                            try:
                                datetime.strptime(date_str, fmt)
                                format_counts[fmt] = format_counts.get(fmt, 0) + 1
                                break
                            except ValueError:
                                continue
                elif '-' in date_str:
                    if date_str.count('-') == 2:
                        for fmt in ['%Y-%m-%d', '%d-%m-%Y', '%m-%d-%Y']:
                            try:
                                datetime.strptime(date_str, fmt)
                                format_counts[fmt] = format_counts.get(fmt, 0) + 1
                                break
                            except ValueError:
                                continue

                successful_parses.append(date_str)

            except (ValueError, TypeError):
                continue

        if not format_counts:
            return None

        # Special disambiguation logic for ambiguous formats
        if '/' in date_samples[0] and len(format_counts) > 1:
            # Check for disambiguating dates (where day > 12 or month > 12)
            for date_str in date_samples[:sample_size]:
                try:
                    parts = date_str.strip().split('/')
                    if len(parts) == 3:
                        first_part = int(parts[0])
                        second_part = int(parts[1])
                        
                        # If first part > 12, it must be day (so format is d/m/y)
                        if first_part > 12:
                            if '%d/%m/%Y' in format_counts:
                                return {
                                    'format': '%d/%m/%Y',
                                    'confidence': 0.9,  # High confidence due to disambiguation
                                    'samples_parsed': len(successful_parses)
                                }
                        
                        # If second part > 12, it must be month (so format is m/d/y)
                        elif second_part > 12:
                            if '%m/%d/%Y' in format_counts:
                                return {
                                    'format': '%m/%d/%Y',
                                    'confidence': 0.9,  # High confidence due to disambiguation
                                    'samples_parsed': len(successful_parses)
                                }
                except (ValueError, IndexError):
                    continue

        # Return most common format if no disambiguation found
        most_common = max(format_counts.items(), key=lambda x: x[1])
        confidence = most_common[1] / len(successful_parses) if successful_parses else 0

        return {
            'format': most_common[0],
            'confidence': confidence,
            'samples_parsed': len(successful_parses)
        }

    def _map_columns(self, headers, expected_columns=['Date', 'Price']):
        """
        Map detected headers to expected columns using fuzzy string matching.

        Args:
            headers (list): List of detected header names
            expected_columns (list): List of expected column names

        Returns:
            dict: Mapping of expected columns to detected headers
        """
        mapping = {}
        used_headers = set()

        for expected in expected_columns:
            best_match = None
            best_ratio = 0

            for header in headers:
                if header in used_headers:
                    continue

                # Use difflib for fuzzy matching
                ratio = difflib.SequenceMatcher(None, expected.lower(), header.lower()).ratio()

                # Also check for common variations
                variations = {
                    'Date': ['date', 'datetime', 'time', 'timestamp', 'day'],
                    'Price': ['price', 'close', 'closing_price', 'value', 'amount', 'cost']
                }

                for variation in variations.get(expected, []):
                    variation_ratio = difflib.SequenceMatcher(None, variation, header.lower()).ratio()
                    ratio = max(ratio, variation_ratio)

                if ratio > best_ratio and ratio > 0.6:  # Minimum similarity threshold
                    best_match = header
                    best_ratio = ratio

            if best_match:
                mapping[expected] = best_match
                used_headers.add(best_match)

        return mapping

    def _calculate_detection_confidence(self, structure_result, date_result, mapping_result):
        """
        Calculate overall confidence in the detection results.

        Args:
            structure_result (dict): CSV structure detection results
            date_result (dict): Date format detection results
            mapping_result (dict): Column mapping results

        Returns:
            float: Overall confidence score (0-1)
        """
        structure_conf = structure_result.get('confidence', 0.5)
        date_conf = date_result.get('confidence', 0) if date_result else 0
        mapping_conf = len(mapping_result) / 2.0  # 0.5 per column found

        # Weighted average
        weights = [0.4, 0.4, 0.2]  # Structure, date format, column mapping
        confidence = (structure_conf * weights[0] +
                     date_conf * weights[1] +
                     mapping_conf * weights[2])

        return min(confidence, 1.0)

    def _create_fallback_dialog(self, file_path, detection_results):
        """
        Create a fallback dialog for manual CSV configuration when auto-detection confidence is low.

        Args:
            file_path (str): Path to the CSV file
            detection_results (dict): Results from auto-detection

        Returns:
            dict or None: Manual configuration or None if cancelled
        """
        # Create a standalone dialog window
        dialog = tk.Tk()
        dialog.title("CSV Configuration Required")
        dialog.geometry("500x400")
        dialog.resizable(False, False)

        result = {}

        def on_ok():
            result.update({
                'delimiter': delimiter_var.get(),
                'has_header': has_header_var.get(),
                'date_column': date_col_var.get(),
                'price_column': price_col_var.get(),
                'date_format': date_format_var.get()
            })
            dialog.quit()  # Use quit instead of destroy to allow result retrieval
            dialog.destroy()

        def on_cancel():
            result.clear()
            dialog.quit()
            dialog.destroy()

        # Show detection results
        tk.Label(dialog, text="Auto-detection Results:", font=("Arial", 12, "bold")).pack(pady=10)

        results_text = tk.Text(dialog, height=6, width=60, font=("Courier", 9))
        results_text.pack(pady=5)

        confidence = detection_results.get('confidence', 0)
        results_text.insert(tk.END, f"Detection Confidence: {confidence:.1%}\n")
        results_text.insert(tk.END, f"Detected Delimiter: '{detection_results.get('delimiter', ',')}'\n")
        results_text.insert(tk.END, f"Has Header: {detection_results.get('has_header', True)}\n")

        if 'date_format' in detection_results:
            results_text.insert(tk.END, f"Date Format: {detection_results['date_format']}\n")

        if 'column_mapping' in detection_results:
            mapping = detection_results['column_mapping']
            results_text.insert(tk.END, f"Column Mapping: {mapping}\n")

        results_text.config(state=tk.DISABLED)

        # Manual configuration
        tk.Label(dialog, text="Manual Configuration:", font=("Arial", 12, "bold")).pack(pady=10)

        # Delimiter
        delim_frame = tk.Frame(dialog)
        delim_frame.pack(fill=tk.X, padx=20, pady=2)
        tk.Label(delim_frame, text="Delimiter:").pack(side=tk.LEFT)
        delimiter_var = tk.StringVar(value=detection_results.get('delimiter', ','))
        tk.Entry(delim_frame, textvariable=delimiter_var, width=5).pack(side=tk.RIGHT)

        # Has header
        header_frame = tk.Frame(dialog)
        header_frame.pack(fill=tk.X, padx=20, pady=2)
        tk.Label(header_frame, text="Has Header Row:").pack(side=tk.LEFT)
        has_header_var = tk.BooleanVar(value=detection_results.get('has_header', True))
        tk.Checkbutton(header_frame, variable=has_header_var).pack(side=tk.RIGHT)

        # Column names
        col_frame = tk.Frame(dialog)
        col_frame.pack(fill=tk.X, padx=20, pady=5)

        tk.Label(col_frame, text="Date Column:").grid(row=0, column=0, sticky='w')
        date_col_var = tk.StringVar(value=detection_results.get('column_mapping', {}).get('Date', 'Date'))
        tk.Entry(col_frame, textvariable=date_col_var).grid(row=0, column=1, padx=5)

        tk.Label(col_frame, text="Price Column:").grid(row=1, column=0, sticky='w')
        price_col_var = tk.StringVar(value=detection_results.get('column_mapping', {}).get('Price', 'Price'))
        tk.Entry(col_frame, textvariable=price_col_var).grid(row=1, column=1, padx=5)

        # Date format
        format_frame = tk.Frame(dialog)
        format_frame.pack(fill=tk.X, padx=20, pady=5)
        tk.Label(format_frame, text="Date Format:").pack(side=tk.LEFT)
        date_format_var = tk.StringVar(value=detection_results.get('date_format', '%Y-%m-%d'))
        tk.Entry(format_frame, textvariable=date_format_var).pack(side=tk.RIGHT)

        # Buttons
        button_frame = tk.Frame(dialog)
        button_frame.pack(fill=tk.X, padx=20, pady=20)
        tk.Button(button_frame, text="OK", command=on_ok, width=10).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Cancel", command=on_cancel, width=10).pack(side=tk.RIGHT, padx=5)

        dialog.mainloop()  # Wait for user input
        return result if result else None

    def read_csv(self, file_path):
        """
        Read and validate CSV file with smart auto-detection of format, headers, and columns.

        Uses built-in Python libraries (csv.Sniffer, dateutil.parser, difflib) to automatically
        detect CSV structure, date formats, and column mappings. Falls back to manual
        configuration only when auto-detection confidence is low.

        Args:
            file_path (str): Path to the CSV file

        Returns:
            pd.DataFrame or None: Processed DataFrame with Date and Price columns, or None if error

        Raises:
            FileNotFoundError: If the file does not exist
            pd.errors.ParserError: If CSV parsing fails after all attempts
            ValueError: If data validation fails
        """
        try:
            logging.info(f"Reading CSV file with smart detection: {file_path}")

            # Step 1: Auto-detect CSV structure
            structure = self._detect_csv_structure(file_path)
            logging.info(f"Detected structure: delimiter='{structure['delimiter']}', "
                        f"has_header={structure['has_header']}, confidence={structure['confidence']:.1%}")

            # Step 2: Read sample data to analyze content
            try:
                sample_df = pd.read_csv(file_path, sep=structure['delimiter'],
                                      nrows=20, header=0 if structure['has_header'] else None)
                logging.info(f"Sample data shape: {sample_df.shape}")
                logging.info(f"Sample columns: {list(sample_df.columns)}")
            except Exception as e:
                logging.warning(f"Failed to read sample with detected structure: {e}")
                # Fallback to basic reading
                sample_df = pd.read_csv(file_path, sep=',', nrows=20, header=None)
                structure['delimiter'] = ','
                structure['has_header'] = False

            # Step 3: Detect column mapping
            if structure['has_header']:
                headers = list(sample_df.columns)
                column_mapping = self._map_columns(headers)
                logging.info(f"Column mapping: {column_mapping}")
            else:
                # For headerless files, assume first two columns are Date and Price
                column_mapping = {'Date': sample_df.columns[0], 'Price': sample_df.columns[1]}
                logging.info("No headers detected, using positional mapping")

            # Step 4: Detect date format from sample data
            date_samples = []
            date_col_name = column_mapping.get('Date')
            logging.info(f"Date column name from mapping: {date_col_name}")
            logging.info(f"Sample DF columns: {list(sample_df.columns)}")
            
            if date_col_name is not None and date_col_name in sample_df.columns:
                date_samples = sample_df[date_col_name].dropna().astype(str).tolist()
                logging.info(f"Extracted {len(date_samples)} date samples: {date_samples[:5]}")
            else:
                logging.warning(f"Date column '{date_col_name}' not found in sample data")

            date_detection = self._detect_date_format(date_samples)
            if date_detection:
                logging.info(f"Detected date format: {date_detection['format']} "
                           f"(confidence: {date_detection['confidence']:.1%})")
            else:
                logging.warning("Could not detect date format, will use pandas default parsing")

            # Step 5: Calculate overall confidence
            confidence = self._calculate_detection_confidence(structure, date_detection, column_mapping)
            logging.info(f"Overall detection confidence: {confidence:.1%}")

            # Step 6: Decide whether to use auto-detection or ask for manual input
            manual_config = None
            if confidence < 0.7:  # Low confidence threshold
                logging.info("Low confidence in auto-detection, requesting manual configuration")

                detection_results = {
                    'confidence': confidence,
                    'delimiter': structure['delimiter'],
                    'has_header': structure['has_header'],
                    'column_mapping': column_mapping,
                    'date_format': date_detection['format'] if date_detection else '%Y-%m-%d'
                }

                # Try to create fallback dialog (only if we have a GUI context)
                try:
                    manual_config = self._create_fallback_dialog(file_path, detection_results)
                    if manual_config:
                        logging.info("User provided manual configuration")
                        structure.update({
                            'delimiter': manual_config['delimiter'],
                            'has_header': manual_config['has_header']
                        })
                        column_mapping = {
                            'Date': manual_config['date_column'],
                            'Price': manual_config['price_column']
                        }
                        date_format = manual_config['date_format']
                    else:
                        logging.info("User cancelled manual configuration, using auto-detection")
                except Exception as e:
                    logging.warning(f"Could not create fallback dialog: {e}, using auto-detection")

            # Step 7: Read the full CSV with detected/manual configuration
            try:
                df = pd.read_csv(file_path, sep=structure['delimiter'],
                               header=0 if structure['has_header'] else None)
                logging.info(f"Successfully read CSV with shape: {df.shape}")
            except Exception as e:
                logging.error(f"Failed to read CSV with detected parameters: {e}")
                raise pd.errors.ParserError(f"Could not parse CSV file: {e}")

            # Step 8: Apply column mapping
            logging.info(f"Before column mapping - columns: {list(df.columns)}")
            logging.info(f"Column mapping check: Date={column_mapping.get('Date')}, Price={column_mapping.get('Price')}")
            
            if column_mapping.get('Date') is not None and column_mapping.get('Price') is not None:
                if structure['has_header']:
                    # Rename columns
                    df = df.rename(columns={
                        column_mapping['Date']: 'Date',
                        column_mapping['Price']: 'Price'
                    })
                    logging.info("Renamed columns using header mapping")
                else:
                    # For headerless files, assign positional names
                    original_columns = list(df.columns)
                    df.columns = ['Date', 'Price'] + list(df.columns[2:])
                    logging.info(f"Assigned positional column names: {original_columns} -> {list(df.columns)}")
            else:
                logging.warning("Could not map required columns, assuming first two columns")
                original_columns = list(df.columns)
                df.columns = ['Date', 'Price'] + list(df.columns[2:])
                logging.info(f"Applied fallback column names: {original_columns} -> {list(df.columns)}")
            
            logging.info(f"After column mapping - columns: {list(df.columns)}")

            # Step 9: Clean and validate price column  
            # After column mapping, always use the standardized 'Price' column name
            if 'Price' in df.columns:
                df['Price'] = (df['Price'].astype(str)
                              .str.strip()
                              .str.replace('$', '', regex=False)
                              .str.replace(',', '', regex=False)
                              .str.replace(' ', '', regex=False)
                              .astype(float))
                logging.info(f"Price column cleaned: {df['Price'].head().tolist()}")
            else:
                raise ValueError("Price column not found in data")

            # Step 10: Parse dates with detected format
            # After column mapping, always use the standardized 'Date' column name
            if 'Date' in df.columns:
                if manual_config and 'date_format' in manual_config:
                    # Use manual date format
                    try:
                        df['Date'] = pd.to_datetime(df['Date'], format=manual_config['date_format'])
                        logging.info("Date conversion successful with manual format")
                    except ValueError as e:
                        logging.warning(f"Manual date format failed: {e}, trying auto-parse")
                        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                elif date_detection and date_detection['confidence'] > 0.8:
                    # Use detected date format
                    try:
                        df['Date'] = pd.to_datetime(df['Date'], format=date_detection['format'])
                        logging.info(f"Date conversion successful with detected format: {date_detection['format']}")
                    except ValueError as e:
                        logging.warning(f"Detected date format failed: {e}, trying auto-parse")
                        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                else:
                    # Use pandas default parsing
                    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                    logging.info("Using default date parser")

                # Check for invalid dates
                invalid_dates = df['Date'].isna().sum()
                if invalid_dates > 0:
                    logging.warning(f"{invalid_dates} rows have invalid dates")
            else:
                raise ValueError("Date column not found in data")

            # Step 11: Final validation and sorting
            df = df.dropna(subset=['Date', 'Price'])  # Remove rows with missing data
            df = df.sort_values('Date')
            logging.info(f"Data sorted by date. Range: {df['Date'].min()} to {df['Date'].max()}")
            logging.info(f"Final data shape: {df.shape}")

            return df

        except FileNotFoundError:
            error_msg = f"File not found: {file_path}"
            logging.error(error_msg)
            messagebox.showerror("File Error", error_msg)
            return None
        except pd.errors.ParserError as e:
            error_msg = f"CSV parsing error: {str(e)}"
            logging.error(error_msg)
            messagebox.showerror("Parse Error", error_msg)
            return None
        except ValueError as e:
            error_msg = f"Data validation error: {str(e)}"
            logging.error(error_msg)
            messagebox.showerror("Data Error", error_msg)
            return None
        except Exception as e:
            error_msg = f"Unexpected error reading CSV: {str(e)}"
            logging.error(error_msg)
            messagebox.showerror("Error", error_msg)
            return None

    def calculate_moving_averages(self, df, short_period, long_period):
        """
        Calculate short and long moving averages using vectorized operations.

        Args:
            df (pd.DataFrame): Input DataFrame with Price column
            short_period (int): Short MA period
            long_period (int): Long MA period

        Returns:
            pd.DataFrame: DataFrame with added Short_MA and Long_MA columns
        """
        df_copy = df.copy()

        # Vectorized rolling mean calculations
        df_copy['Short_MA'] = df_copy['Price'].rolling(window=short_period).mean()
        df_copy['Long_MA'] = df_copy['Price'].rolling(window=long_period).mean()

        return df_copy

    def calculate_slopes(self, df, lookback=3):
        """
        Calculate slopes of Short_MA and Long_MA using vectorized operations.

        Args:
            df (pd.DataFrame): DataFrame with Short_MA and Long_MA columns
            lookback (int): Number of periods to look back for slope calculation

        Returns:
            pd.DataFrame: DataFrame with added slope columns
        """
        df = df.copy()
        df['Short_MA_Slope'] = 0.0
        df['Long_MA_Slope'] = 0.0

        if len(df) <= lookback:
            return df

        # Vectorized slope calculation
        def calculate_slope_vector(series, window):
            result = np.zeros(len(series))
            valid_mask = ~pd.isna(series)

            if valid_mask.sum() <= window:
                return result

            valid_series = series[valid_mask].values

            for i in range(window, len(valid_series)):
                y = valid_series[i-window:i]
                if len(y) == window and not np.isnan(y).any():
                    slope = np.polyfit(np.arange(window), y, 1)[0]
                    orig_idx = valid_mask.index[valid_mask][i]
                    result[orig_idx] = slope

            return result

        if (~pd.isna(df['Short_MA'])).sum() > lookback:
            df['Short_MA_Slope'] = calculate_slope_vector(df['Short_MA'], lookback)

        if (~pd.isna(df['Long_MA'])).sum() > lookback:
            df['Long_MA_Slope'] = calculate_slope_vector(df['Long_MA'], lookback)

        return df

    def generate_signals(self, df, include_slope_check=False, lookback=3):
        """
        Generate buy/sell signals based on MA crossovers and optional slope checks.

        Args:
            df (pd.DataFrame): DataFrame with MA columns
            include_slope_check (bool): Whether to include slope filtering
            lookback (int): Lookback period for slope calculation

        Returns:
            pd.DataFrame: DataFrame with added Signal column
        """
        df = df.copy()
        df['Signal'] = 0

        if len(df.dropna()) < 2:
            return df

        for i in range(1, len(df)):
            if (pd.isna(df['Short_MA'].iloc[i-1]) or pd.isna(df['Long_MA'].iloc[i-1]) or
                pd.isna(df['Short_MA'].iloc[i]) or pd.isna(df['Long_MA'].iloc[i])):
                continue

            buy_crossover = (df['Short_MA'].iloc[i-1] <= df['Long_MA'].iloc[i-1] and
                           df['Short_MA'].iloc[i] > df['Long_MA'].iloc[i])

            sell_crossover = (df['Short_MA'].iloc[i-1] >= df['Long_MA'].iloc[i-1] and
                            df['Short_MA'].iloc[i] < df['Long_MA'].iloc[i])

            if include_slope_check and (buy_crossover or sell_crossover):
                short_slope = self._calculate_slope_at_point(df['Short_MA'], i, lookback)
                long_slope = self._calculate_slope_at_point(df['Long_MA'], i, lookback)

                if buy_crossover and short_slope > 0 and long_slope > 0:
                    df.loc[df.index[i], 'Signal'] = 1
                elif sell_crossover and short_slope < 0 and long_slope < 0:
                    df.loc[df.index[i], 'Signal'] = -1
            elif not include_slope_check:
                if buy_crossover:
                    df.loc[df.index[i], 'Signal'] = 1
                elif sell_crossover:
                    df.loc[df.index[i], 'Signal'] = -1

        return df

    def _calculate_slope_at_point(self, series, index, lookback=3):
        """
        Calculate slope at a specific point in the series.

        Args:
            series (pd.Series): Time series data
            index (int): Index to calculate slope at
            lookback (int): Number of periods to look back

        Returns:
            float: Slope value
        """
        if index < lookback:
            return 0.0

        values = series.iloc[index-lookback:index].values

        if len(values) < lookback or np.isnan(values).any():
            return 0.0

        return np.polyfit(np.arange(len(values)), values, 1)[0]

    def calculate_annualized_gain(self, initial, final, start_date, end_date):
        """
        Calculate annualized percentage gain.

        Args:
            initial (float): Initial investment
            final (float): Final value
            start_date (datetime): Start date
            end_date (datetime): End date

        Returns:
            float: Annualized gain percentage
        """
        num_days = (end_date - start_date).days
        num_years = num_days / 365.25

        if num_years > 0 and initial > 0:
            return ((final / initial) ** (1 / num_years) - 1) * 100
        return 0.0

    def calculate_wealth(self, df, initial_investment=10000, allow_short_selling=False,
                        short_sell_only=False):
        """
        Calculate wealth over time based on trading signals.

        Args:
            df (pd.DataFrame): DataFrame with signals
            initial_investment (float): Starting investment amount
            allow_short_selling (bool): Whether short selling is allowed
            short_sell_only (bool): Whether to only do short trades

        Returns:
            tuple: (final_wealth, annualized_gain, trade_list)
        """
        df_valid = df.dropna().copy()

        if len(df_valid) < 2:
            return initial_investment, 0, []

        cash = initial_investment
        shares = 0
        position = 0  # 0: no position, 1: long, -1: short
        short_entry_price = 0
        trade_list = []

        for i in range(len(df_valid)):
            signal = df_valid['Signal'].iloc[i]
            price = df_valid['Price'].iloc[i]
            date = df_valid['Date'].iloc[i]

            # Long trades
            if not short_sell_only and signal == 1 and (position == 0 or
               (allow_short_selling and position == -1)):
                if position == -1:  # Close short
                    profit = shares * (short_entry_price - price)
                    cash += profit + shares * short_entry_price
                    shares = 0
                    position = 0
                    trade_list.append(('Cover Short', date, price))

                # Open long
                shares = cash / price
                cash = 0
                position = 1
                trade_list.append(('Buy', date, price))

            # Sell signals
            elif signal == -1:
                if position == 1:  # Close long
                    cash = shares * price
                    shares = 0
                    position = 0
                    trade_list.append(('Sell', date, price))

                # Open short if allowed
                if allow_short_selling and position == 0:
                    shares = cash / price
                    short_entry_price = price
                    position = -1
                    cash = 0
                    trade_list.append(('Short Sell', date, price))

        # Close final position
        final_price = df_valid['Price'].iloc[-1]
        final_date = df_valid['Date'].iloc[-1]

        if position == 1:
            cash = shares * final_price
        elif position == -1:
            profit = shares * (short_entry_price - final_price)
            cash += profit + shares * short_entry_price

        total_trades = len(trade_list)
        start_date = df_valid['Date'].iloc[0]
        annualized_gain = self.calculate_annualized_gain(initial_investment, cash,
                                                        start_date, final_date)

        return cash, annualized_gain, trade_list

    def run_backtest(self, file_path, start_date, end_date, short_min, short_max,
                    long_min, long_max, allow_short_selling, short_sell_only,
                    include_slope_check, slope_lookback, progress_callback=None):
        """
        Run comprehensive backtest across MA parameter ranges.

        Args:
            file_path (str): Path to CSV file
            start_date (str): Start date string
            end_date (str): End date string
            short_min (int): Minimum short MA period
            short_max (int): Maximum short MA period
            long_min (int): Minimum long MA period
            long_max (int): Maximum long MA period
            allow_short_selling (bool): Allow short selling
            short_sell_only (bool): Only short trades
            include_slope_check (bool): Include slope filtering
            slope_lookback (int): Slope lookback period
            progress_callback (callable): Progress update callback

        Returns:
            list: List of result dictionaries
        """
        logging.info("Starting backtest")

        df = self.read_csv(file_path)
        if df is None:
            return []

        # Filter date range
        df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
        if df.empty:
            messagebox.showerror("Error", "No data in specified date range")
            return []

        short_periods = range(short_min, short_max + 1)
        long_periods = range(long_min, long_max + 1)
        total_combinations = len(short_periods) * len(long_periods)
        results = []

        if include_slope_check:
            # Pre-calculate MAs for efficiency
            for short_period in short_periods:
                for long_period in long_periods:
                    key = (short_period, long_period)
                    if key not in self.ma_cache:
                        self.ma_cache[key] = self.calculate_moving_averages(df, short_period, long_period)

        combination_count = 0
        for short_period in short_periods:
            for long_period in long_periods:
                if include_slope_check:
                    df_with_ma = self.ma_cache[(short_period, long_period)]
                else:
                    df_with_ma = self.calculate_moving_averages(df, short_period, long_period)

                df_with_signals = self.generate_signals(df_with_ma, include_slope_check, slope_lookback)
                final_wealth, annualized_gain, trade_list = self.calculate_wealth(
                    df_with_signals, 10000, allow_short_selling, short_sell_only)

                results.append({
                    'short_ma': short_period,
                    'long_ma': long_period,
                    'final_wealth': final_wealth,
                    'annualized_gain': annualized_gain,
                    'trade_count': len(trade_list)
                })

                combination_count += 1
                if progress_callback:
                    progress = int((combination_count / total_combinations) * 100)
                    progress_callback(combination_count, progress)

        logging.info(f"Backtest completed with {len(results)} results")
        return results

# === Visualization Class ===

class Visualization:
    """
    Handles all visualization tasks including heatmaps and price charts.

    This class provides methods for creating interactive visualizations
    of trading strategy performance and price data.
    """

    def __init__(self, root):
        """
        Initialize visualization handler.

        Args:
            root (tk.Tk): Root Tkinter window
        """
        self.root = root

    def create_heatmap_window(self, data, title, value_label, cmap, bounds, colors):
        """
        Create an interactive heatmap window.

        Args:
            data (pd.DataFrame): Pivot table data for heatmap
            title (str): Window title
            value_label (str): Label for colorbar
            cmap: Matplotlib colormap
            bounds (list): Color boundaries
            colors (list): Color list
        """
        heatmap_window = tk.Toplevel(self.root)
        heatmap_window.title(title)
        heatmap_window.state('zoomed')

        main_frame = tk.Frame(heatmap_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Status bar
        status_frame = tk.Frame(main_frame, relief=tk.SUNKEN, borderwidth=1)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        status_var = tk.StringVar(value="Hover over cells to see details")
        status_label = tk.Label(status_frame, textvariable=status_var, anchor='w',
                              font=("Arial", 11), bg='#f0f0f0', padx=10, pady=5)
        status_label.pack(fill=tk.X)

        # Create heatmap
        fig = plt.Figure(figsize=(12, 8), dpi=100)
        ax = fig.add_subplot(111)

        im = ax.imshow(data, cmap=cmap, aspect='auto', interpolation='nearest')

        # Configure colorbar
        if bounds is not None:
            norm = mcolors.BoundaryNorm(bounds, cmap.N)
            tick_locations = bounds[:-1] + (bounds[1:] - bounds[:-1]) / 2
            cbar = fig.colorbar(im, ax=ax, norm=norm, boundaries=bounds,
                              ticks=tick_locations, spacing='proportional')
            tick_labels = [f"{bounds[i]:.1f} - {bounds[i+1]:.1f}" for i in range(len(bounds)-1)]
            cbar.ax.set_yticklabels(tick_labels)
        else:
            cbar = fig.colorbar(im, ax=ax)

        cbar.ax.set_ylabel(value_label, rotation=270, labelpad=20)

        # Configure axes
        self._configure_heatmap_axes(ax, data)

        # Set labels and title
        ax.set_xlabel('Short Moving Average Period')
        ax.set_ylabel('Long Moving Average Period')
        ax.set_title(f"{title} - Higher Values Are Better" if "Gain" in value_label
                    else f"{title} - Shows Trading Activity")

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        ax.grid(True, alpha=0.3)

        # Embed plot
        canvas = FigureCanvasTkAgg(fig, master=main_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add toolbar
        toolbar = NavigationToolbar2Tk(canvas, main_frame)
        toolbar.update()

        # Mouse hover functionality
        def motion_notify_event(event):
            if event.inaxes == ax:
                i = int(event.ydata + 0.5) if event.ydata is not None else -1
                j = int(event.xdata + 0.5) if event.xdata is not None else -1

                if 0 <= i < len(data.index) and 0 <= j < len(data.columns):
                    long_ma = data.index[i]
                    short_ma = data.columns[j]
                    value = data.iloc[i, j]

                    if pd.isna(value):
                        status_var.set("No data available")
                        return

                    if "Gain" in value_label:
                        tooltip = (f"Long MA: {long_ma} | Short MA: {short_ma} | "
                                 f"Annualized Gain: {value:.2f}%")
                    else:
                        tooltip = (f"Long MA: {long_ma} | Short MA: {short_ma} | "
                                 f"Total Trades: {int(value)}")

                    status_var.set(tooltip)
                else:
                    status_var.set("Hover over cells to see details")
            else:
                status_var.set("Hover over cells to see details")

        canvas.mpl_connect('motion_notify_event', motion_notify_event)

        # Close button
        button_frame = tk.Frame(main_frame)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        tk.Button(button_frame, text="Close", command=heatmap_window.destroy,
                 width=15, height=2, bg="#f44336", fg="white",
                 font=("Arial", 10, "bold")).pack(pady=5)

        heatmap_window.update_idletasks()

    def _configure_heatmap_axes(self, ax, data):
        """Configure heatmap axes with smart tick spacing."""
        max_ticks = 30

        # X-axis configuration
        col_values = np.array(data.columns)
        col_length = len(col_values)

        if col_length > max_ticks:
            interval = self._find_nice_interval(col_length, max_ticks)
            tick_values = np.arange(col_values.min(),
                                  col_values.max() + interval, interval)
            tick_values = tick_values[(tick_values >= col_values.min()) &
                                    (tick_values <= col_values.max())]

            tick_indices = []
            for val in tick_values:
                idx = np.abs(col_values - val).argmin()
                if idx not in tick_indices:
                    tick_indices.append(idx)

            ax.set_xticks(tick_indices)
            ax.set_xticklabels([col_values[i] for i in tick_indices])
        else:
            ax.set_xticks(np.arange(len(data.columns)))
            ax.set_xticklabels(data.columns)

        # Y-axis configuration
        row_values = np.array(data.index)
        row_length = len(row_values)

        if row_length > max_ticks:
            interval = self._find_nice_interval(row_length, max_ticks)
            tick_values = np.arange(row_values.min(),
                                  row_values.max() + interval, interval)
            tick_values = tick_values[(tick_values >= row_values.min()) &
                                    (tick_values <= row_values.max())]

            tick_indices = []
            for val in tick_values:
                idx = np.abs(row_values - val).argmin()
                if idx not in tick_indices:
                    tick_indices.append(idx)

            ax.set_yticks(tick_indices)
            ax.set_yticklabels([row_values[i] for i in tick_indices])
        else:
            ax.set_yticks(np.arange(len(data.index)))
            ax.set_yticklabels(data.index)

    def _find_nice_interval(self, length, max_ticks):
        """Find a nice interval for axis ticks."""
        raw_interval = length / max_ticks
        nice_intervals = [1, 2, 5, 10, 20, 25, 50, 100]

        for nice in nice_intervals:
            if length / nice <= max_ticks:
                return nice
        return nice_intervals[-1]

    def show_price_chart(self, file_path, start_date, end_date, strategy):
        """
        Show interactive price chart with moving averages and signals.

        Args:
            file_path (str): Path to CSV file
            start_date (str): Start date string
            end_date (str): End date string
            strategy (TradingStrategy): Strategy instance for calculations
        """
        try:
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
            end_date = datetime.strptime(end_date, '%Y-%m-%d')

            df = strategy.read_csv(file_path)
            if df is None:
                return

            filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
            if filtered_df.empty:
                messagebox.showerror("Error", "No data in specified date range")
                return

            # Create chart window
            chart_window = tk.Toplevel(self.root)
            chart_window.title("Price Chart with Moving Averages")
            chart_window.state('zoomed')

            main_frame = tk.Frame(chart_window)
            main_frame.pack(fill=tk.BOTH, expand=True)

            # Chart and controls layout
            chart_frame = tk.Frame(main_frame)
            chart_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            controls_container = tk.Frame(main_frame)
            controls_container.pack(side=tk.RIGHT, fill=tk.Y)

            # Scrollable controls
            controls_scroll = tk.Scrollbar(controls_container, orient=tk.VERTICAL)
            controls_scroll.pack(side=tk.RIGHT, fill=tk.Y)

            controls_canvas = tk.Canvas(controls_container, width=375, bg='#f0f0f0',
                                       yscrollcommand=controls_scroll.set)
            controls_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            controls_scroll.config(command=controls_canvas.yview)

            controls_frame = tk.Frame(controls_canvas, width=375, padx=15, pady=15, bg='#f0f0f0')
            controls_canvas.create_window((0, 0), window=controls_frame, anchor=tk.NW)

            # Update scroll region
            def update_scrollregion(event):
                controls_canvas.configure(scrollregion=controls_canvas.bbox(tk.ALL))
            controls_frame.bind("<Configure>", update_scrollregion)

            # Mouse wheel scrolling
            def on_mousewheel(event):
                controls_canvas.yview_scroll(-1 * int(event.delta/120), "units")
            controls_canvas.bind_all("<MouseWheel>", on_mousewheel)

            # Control variables for chart options (moved to top)
            allow_short_selling_var = tk.BooleanVar(value=False)
            short_sell_only_var = tk.BooleanVar(value=False)
            include_slope_check_var = tk.BooleanVar(value=False)
            slope_lookback_var = tk.IntVar(value=3)

            # MA variables (moved to top)
            short_ma_var = tk.StringVar(value="50")
            long_ma_var = tk.StringVar(value="150")

            # Stats variables
            annualized_gain_var = tk.StringVar(value="---.--")
            trades_count_var = tk.StringVar(value="--")
            long_trades_var = tk.StringVar(value="--")
            short_trades_var = tk.StringVar(value="--")

            # Global-like variables for chart function
            all_trades = []
            df_copy = None

            # Update chart function (moved before UI controls)
            def update_chart():
                nonlocal df_copy, all_trades
                ax.clear()

                try:
                    short_ma = int(short_ma_var.get())
                    long_ma = int(long_ma_var.get())

                    if short_ma <= 0 or long_ma <= 0:
                        messagebox.showerror("Error", "MA periods must be positive")
                        return
                except ValueError:
                    messagebox.showerror("Error", "MA periods must be integers")
                    return

                # Track performance
                chart_start_time = datetime.now()

                # Calculate moving averages
                df_copy = filtered_df.copy()
                df_copy['Short_MA'] = df_copy['Price'].rolling(window=short_ma).mean()
                df_copy['Long_MA'] = df_copy['Price'].rolling(window=long_ma).mean()

                # Get options
                include_slope_check = include_slope_check_var.get()
                lookback = slope_lookback_var.get()

                # Track slope calculation time
                slope_start_time = datetime.now()

                # Generate signals
                df_copy = strategy.generate_signals(df_copy, include_slope_check, lookback=lookback)
                # Always fill slope columns for display
                df_copy = strategy.calculate_slopes(df_copy, lookback)

                if include_slope_check:
                    logging.info(f"Optimized slope calculation took: {(datetime.now() - slope_start_time).total_seconds():.2f} seconds")

                # Plot data
                ax.plot(df_copy['Date'], df_copy['Price'], 'k-', label='Price')
                ax.plot(df_copy['Date'], df_copy['Short_MA'], 'b-', label=f'Short MA ({short_ma})')
                ax.plot(df_copy['Date'], df_copy['Long_MA'], 'g-', label=f'Long MA ({long_ma})')

                # Calculate positioning for signals
                min_price = filtered_df['Price'].min()
                max_price = filtered_df['Price'].max()
                price_range = max_price - min_price
                buy_offset = price_range * 0.08
                sell_offset = price_range * 0.08

                # Add buy signals
                buy_signals = df_copy[df_copy['Signal'] == 1]
                if not buy_signals.empty:
                    for idx, row in buy_signals.iterrows():
                        marker_y = row['Price'] + buy_offset
                        ax.annotate('B', xy=(row['Date'], marker_y),
                                   fontsize=12, fontweight='bold', color='green',
                                   ha='center', va='center',
                                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

                # Add sell signals
                sell_signals = df_copy[df_copy['Signal'] == -1]
                if not sell_signals.empty:
                    for idx, row in sell_signals.iterrows():
                        marker_y = row['Price'] - sell_offset
                        ax.annotate('S', xy=(row['Date'], marker_y),
                                   fontsize=12, fontweight='bold', color='red',
                                   ha='center', va='center',
                                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

                # Format axes
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                fig.autofmt_xdate()

                ax.set_xlabel('Date')
                ax.set_ylabel('Price')
                ax.set_title(f'Price Chart with Buy/Sell Signals ({start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")})')

                padding = (max_price - min_price) * 0.12
                ax.set_ylim(min_price - padding, max_price + padding)
                ax.grid(True, alpha=0.3)

                ax.legend(['Price', f'Short MA ({short_ma})', f'Long MA ({long_ma})'],
                         loc='upper left', fontsize='small')

                # Calculate strategy performance
                initial_investment = 10000
                allow_short_selling = allow_short_selling_var.get()
                short_sell_only = short_sell_only_var.get() if allow_short_selling else False

                final_wealth, annualized_gain, trade_list = strategy.calculate_wealth(
                    df_copy, initial_investment=initial_investment,
                    allow_short_selling=allow_short_selling,
                    short_sell_only=short_sell_only)

                # Update stats
                annualized_gain_var.set(f"{annualized_gain:.2f}%")
                trades_count_var.set(str(len(trade_list)))
                long_trades = sum(1 for t in trade_list if t[0] == 'Buy')
                short_trades = sum(1 for t in trade_list if t[0] == 'Short Sell')
                long_trades_var.set(str(long_trades))
                short_trades_var.set(str(short_trades) if allow_short_selling else "0 (disabled)")

                # Update all_trades for detailed view
                all_trades = trade_list.copy()

                # Update canvas
                fig.tight_layout()
                canvas.draw()

            # MA controls
            tk.Label(controls_frame, text="Moving Averages", font=("Arial", 14, "bold"),
                    bg='#f0f0f0').pack(pady=(0, 10))

            # Create horizontal frame for MA and slope controls
            ma_and_slope_frame = tk.Frame(controls_frame, bg='#f0f0f0')
            ma_and_slope_frame.pack(fill=tk.X, pady=(10, 0))

            # Left side: MA controls
            ma_controls_frame = tk.Frame(ma_and_slope_frame, bg='#f0f0f0')
            ma_controls_frame.pack(side=tk.LEFT, padx=(0, 20))

            # Short MA
            short_ma_frame = tk.Frame(ma_controls_frame, bg='#f0f0f0')
            short_ma_frame.pack(fill=tk.X, pady=(0, 10))
            tk.Label(short_ma_frame, text="Short Moving Average:", bg='#f0f0f0').pack(anchor='w')
            tk.Entry(short_ma_frame, textvariable=short_ma_var, width=10,
                    font=("Arial", 10)).pack(pady=(0, 10), anchor='w')

            # Long MA
            long_ma_frame = tk.Frame(ma_controls_frame, bg='#f0f0f0')
            long_ma_frame.pack(fill=tk.X, pady=(5, 0))
            tk.Label(long_ma_frame, text="Long Moving Average:", bg='#f0f0f0').pack(anchor='w')
            tk.Entry(long_ma_frame, textvariable=long_ma_var, width=10,
                    font=("Arial", 10)).pack(side=tk.LEFT, pady=(0, 5))

            # Right side: Slope controls
            slope_controls_frame = tk.Frame(ma_and_slope_frame, bg='#f0f0f0')
            slope_controls_frame.pack(side=tk.RIGHT, padx=(20, 0))

            tk.Label(slope_controls_frame, text="Slope Check Options", font=("Arial", 12, "bold"),
                    bg='#f0f0f0').pack(pady=(0, 10))

            # Include slope check
            include_slope_check_cb = tk.Checkbutton(slope_controls_frame, text="Include Slope Check",
                                                  variable=include_slope_check_var,
                                                  onvalue=True, offvalue=False,
                                                  command=update_chart,
                                                  bg='#f0f0f0')
            include_slope_check_cb.pack(anchor='w', pady=(0, 5))

            # Lookback entry
            lookback_frame = tk.Frame(slope_controls_frame, bg='#f0f0f0')
            lookback_frame.pack(anchor='w')
            tk.Label(lookback_frame, text="Lookback:", bg='#f0f0f0').pack(side=tk.LEFT)
            lookback_spinbox = tk.Spinbox(lookback_frame, from_=1, to=10, width=3,
                                         textvariable=slope_lookback_var,
                                         command=lambda: update_chart() if include_slope_check_var.get() else None)
            lookback_spinbox.pack(side=tk.LEFT, padx=5)

            # Update button (moved above Strategy Performance)
            update_button_frame = tk.Frame(controls_frame, bg='#f0f0f0')
            update_button_frame.pack(fill=tk.X, pady=(10, 0))
            tk.Button(update_button_frame, text="Update Chart", command=update_chart,
                     bg="#4CAF50", fg="white", font=("Arial", 9, "bold"),
                     padx=5, pady=1).pack(anchor='center')

            # Signal explanation
            signal_frame = tk.Frame(controls_frame, bg='#f0f0f0', relief=tk.GROOVE, bd=1)
            signal_frame.pack(fill=tk.X, pady=5)
            tk.Label(signal_frame, text="Signal Legend", font=("Arial", 10, "bold"), bg='#f0f0f0').pack(pady=3)

            # Buy signal legend
            buy_frame = tk.Frame(signal_frame, bg='#f0f0f0')
            buy_frame.pack(fill=tk.X, padx=10, pady=5)

            buy_line_canvas = tk.Canvas(buy_frame, width=15, height=15, bg='#f0f0f0', highlightthickness=0)
            buy_line_canvas.pack(side=tk.LEFT)
            buy_line_canvas.create_line(2, 7, 13, 7, fill="green", width=2)

            tk.Label(buy_frame, text="B", font=("Arial", 12, "bold"), fg="green", bg='#f0f0f0').pack(side=tk.LEFT, padx=2)
            tk.Label(buy_frame, text="= Buy (Short MA crosses above Long MA)", bg='#f0f0f0').pack(side=tk.LEFT, padx=5)

            # Sell signal legend
            sell_frame = tk.Frame(signal_frame, bg='#f0f0f0')
            sell_frame.pack(fill=tk.X, padx=10, pady=5)

            sell_line_canvas = tk.Canvas(sell_frame, width=15, height=15, bg='#f0f0f0', highlightthickness=0)
            sell_line_canvas.pack(side=tk.LEFT)
            sell_line_canvas.create_line(2, 7, 13, 7, fill="red", width=2)

            tk.Label(sell_frame, text="S", font=("Arial", 12, "bold"), fg="red", bg='#f0f0f0').pack(side=tk.LEFT, padx=2)
            tk.Label(sell_frame, text="= Sell (Short MA crosses below Long MA)", bg='#f0f0f0').pack(side=tk.LEFT, padx=5)

            # Create plot
            fig = plt.Figure(figsize=(8, 5.5), dpi=100)
            ax = fig.add_subplot(111)

            # Short selling options
            short_selling_frame = tk.Frame(controls_frame, bg='#f0f0f0', relief=tk.GROOVE, bd=1)
            short_selling_frame.pack(fill=tk.X, pady=5, padx=5)

            tk.Label(short_selling_frame, text="Short Selling Options", font=("Arial", 10, "bold"),
                    bg='#f0f0f0').pack(pady=(3, 1))

            # Toggle function for short sell only visibility
            def toggle_short_sell_only_visibility():
                if allow_short_selling_var.get():
                    short_sell_only_cb.pack(pady=2, anchor=tk.W)
                else:
                    short_sell_only_cb.pack_forget()
                    short_sell_only_var.set(False)
                update_chart()

            # Allow short selling checkbox
            short_selling_cb = tk.Checkbutton(short_selling_frame, text="Allow Short Selling",
                                            variable=allow_short_selling_var,
                                            onvalue=True, offvalue=False,
                                            command=toggle_short_sell_only_visibility,
                                            bg='#f0f0f0')
            short_selling_cb.pack(pady=2, anchor=tk.W)

            # Short sell only checkbox (initially hidden)
            short_sell_only_cb = tk.Checkbutton(short_selling_frame, text="Suppress Long Trades",
                                              variable=short_sell_only_var,
                                              onvalue=True, offvalue=False,
                                              command=update_chart,
                                              bg='#f0f0f0')
            short_sell_only_cb.pack(pady=2, anchor=tk.W)

            # Strategy performance stats
            stats_frame = tk.Frame(controls_frame, bg='#f0f0f0', relief=tk.GROOVE, bd=1)
            stats_frame.pack(fill=tk.X, pady=5, padx=5)

            tk.Label(stats_frame, text="Strategy Performance", font=("Arial", 10, "bold"),
                    bg='#f0f0f0').pack(pady=(3, 0))

            # Annualized gain
            gain_frame = tk.Frame(stats_frame, bg='#f0f0f0')
            gain_frame.pack(fill=tk.X, padx=10, pady=2)
            tk.Label(gain_frame, text="Annualized %Gain:", bg='#f0f0f0',
                    font=("Arial", 9, "bold")).pack(side=tk.LEFT)
            tk.Label(gain_frame, textvariable=annualized_gain_var, bg='#f0f0f0',
                    fg="#0066cc", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)

            # Total trades
            trades_frame = tk.Frame(stats_frame, bg='#f0f0f0')
            trades_frame.pack(fill=tk.X, padx=10, pady=2)
            tk.Label(trades_frame, text="Number of Trades:", bg='#f0f0f0',
                    font=("Arial", 9, "bold")).pack(side=tk.LEFT)
            tk.Label(trades_frame, textvariable=trades_count_var, bg='#f0f0f0',
                    fg="#0066cc", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)

            # Long trades
            long_trades_frame = tk.Frame(stats_frame, bg='#f0f0f0')
            long_trades_frame.pack(fill=tk.X, padx=10, pady=2)
            tk.Label(long_trades_frame, text="Long Trades:", bg='#f0f0f0',
                    font=("Arial", 9, "bold")).pack(side=tk.LEFT)
            tk.Label(long_trades_frame, textvariable=long_trades_var, bg='#f0f0f0',
                    fg="#006600", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)

            # Short trades
            short_trades_frame = tk.Frame(stats_frame, bg='#f0f0f0')
            short_trades_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
            tk.Label(short_trades_frame, text="Short Trades:", bg='#f0f0f0',
                    font=("Arial", 9, "bold")).pack(side=tk.LEFT)
            tk.Label(short_trades_frame, textvariable=short_trades_var, bg='#f0f0f0',
                    fg="#990000", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)

            # Detailed trades function
            def show_detailed_trades():
                nonlocal all_trades, df_copy

                trades_window = tk.Toplevel(chart_window)
                trades_window.title("Detailed Trade List")
                trades_window.geometry("500x500")

                main_frame = tk.Frame(trades_window, padx=10, pady=10)
                main_frame.pack(fill=tk.BOTH, expand=True)

                tk.Label(main_frame, text="Complete Trade History",
                        font=("Arial", 14, "bold")).pack(pady=(0, 10))

                list_frame = tk.Frame(main_frame)
                list_frame.pack(fill=tk.BOTH, expand=True)

                v_scrollbar = tk.Scrollbar(list_frame, orient=tk.VERTICAL)
                v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                h_scrollbar = tk.Scrollbar(list_frame, orient=tk.HORIZONTAL)
                h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

                trades_detail = tk.Text(list_frame, font=("Courier", 10), wrap=tk.NONE,
                                        yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
                trades_detail.pack(fill=tk.BOTH, expand=True)

                v_scrollbar.config(command=trades_detail.yview)
                h_scrollbar.config(command=trades_detail.xview)

                # Add detailed trade information
                header = (
                    f"{'Action':<15}|{'Date':<12}|{'Price':>10}|{'Gain%':>8}|{'Prev Short MA':>14}|{'Short MA':>12}|"
                    f"{'Prev Long MA':>14}|{'Long MA':>12}|{'Short MA Slope':>14}|{'Long MA Slope':>14}|{'Comments':<25}"
                )
                trades_detail.insert(tk.END, header + "\n")
                trades_detail.insert(tk.END, "|".join(["-" * 15, "-" * 12, "-" * 10, "-" * 8, "-" * 14, "-" * 12, "-" * 14, "-" * 12, "-" * 14, "-" * 14, "-" * 25]) + "\n")

                entry_price = None
                entry_action = None
                position = 0
                last_action = None
                simulated_trade_added = False

                for i, trade in enumerate(all_trades):
                    action, date, price = trade[:3]
                    gain_str = ""
                    if action in ('Sell', 'Sell*', 'Cover Short'):
                        if entry_price is not None:
                            gain = ((price - entry_price) / entry_price) * 100 if entry_price != 0 else 0
                            if entry_action in ('Short Sell',):
                                gain = ((entry_price - price) / entry_price) * 100 if entry_price != 0 else 0
                            gain_str = f"{gain:+.2f}"
                        entry_price = None
                        entry_action = None
                    elif action in ('Buy', 'Short Sell'):
                        entry_price = price
                        entry_action = action

                    # Get MA values
                    ma_row = df_copy[df_copy['Date'] == date]
                    short_ma = ma_row['Short_MA'].values[0] if not ma_row.empty else float('nan')
                    long_ma = ma_row['Long_MA'].values[0] if not ma_row.empty else float('nan')

                    # Get previous MA values
                    idx_list = ma_row.index.tolist()
                    if idx_list and idx_list[0] > 0:
                        prev_row = df_copy.iloc[idx_list[0] - 1]
                        prev_short_ma = prev_row['Short_MA']
                        prev_long_ma = prev_row['Long_MA']
                    else:
                        prev_short_ma = float('nan')
                        prev_long_ma = float('nan')

                    # Format values
                    short_ma_str = f"{short_ma:.2f}" if not np.isnan(short_ma) else "N/A"
                    long_ma_str = f"{long_ma:.2f}" if not np.isnan(long_ma) else "N/A"
                    prev_short_ma_str = f"{prev_short_ma:.2f}" if not np.isnan(prev_short_ma) else "N/A"
                    prev_long_ma_str = f"{prev_long_ma:.2f}" if not np.isnan(prev_long_ma) else "N/A"

                    slope_enabled = include_slope_check_var.get()
                    if slope_enabled:
                        long_slope_val = ma_row['Long_MA_Slope'].values[0] if not ma_row.empty and 'Long_MA_Slope' in ma_row.columns else float('nan')
                        short_slope_val = ma_row['Short_MA_Slope'].values[0] if not ma_row.empty and 'Short_MA_Slope' in ma_row.columns else float('nan')
                        long_slope_str = f"{long_slope_val:.4f}" if not np.isnan(long_slope_val) else "N/A"
                        short_slope_str = f"{short_slope_val:.4f}" if not np.isnan(short_slope_val) else "N/A"
                    else:
                        long_slope_str = "NA"
                        short_slope_str = "NA"

                    # Position info
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
                        position = 0
                        position_info = "(Opening short position)"
                    elif action == "Cover Short":
                        position = 0
                        position_info = "(Closing short position)"

                    row_text = (
                        f"{action:<15}|{date.strftime('%Y-%m-%d'):<12}|{price:>10.2f}|{gain_str:>8}|{prev_short_ma_str:>14}|"
                        f"{short_ma_str:>12}|{prev_long_ma_str:>14}|{long_ma_str:>12}|{short_slope_str:>14}|{long_slope_str:>14}|{position_info:<25}"
                    )
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
                    last_action = action

                # Add simulated closing trade if needed
                if position != 0 and not simulated_trade_added:
                    simulated_trade_added = True
                    final_row = df_copy.dropna(subset=['Price']).iloc[-1]
                    final_date = final_row['Date']
                    final_price = final_row['Price']
                    short_ma = final_row['Short_MA'] if 'Short_MA' in final_row else float('nan')
                    long_ma = final_row['Long_MA'] if 'Long_MA' in final_row else float('nan')

                    idx = df_copy.index[df_copy['Date'] == final_date].tolist()
                    if idx and idx[0] > 0:
                        prev_row = df_copy.iloc[idx[0] - 1]
                        prev_short_ma = prev_row['Short_MA']
                        prev_long_ma = prev_row['Long_MA']
                    else:
                        prev_short_ma = float('nan')
                        prev_long_ma = float('nan')

                    slope_enabled = include_slope_check_var.get()
                    if slope_enabled:
                        long_slope_val = final_row['Long_MA_Slope'] if 'Long_MA_Slope' in final_row else float('nan')
                        short_slope_val = final_row['Short_MA_Slope'] if 'Short_MA_Slope' in final_row else float('nan')
                        long_slope_str = f"{long_slope_val:.4f}" if not np.isnan(long_slope_val) else "N/A"
                        short_slope_str = f"{short_slope_val:.4f}" if not np.isnan(short_slope_val) else "N/A"
                    else:
                        long_slope_str = "NA"
                        short_slope_str = "NA"

                    short_ma_str = f"{short_ma:.2f}" if not np.isnan(short_ma) else "N/A"
                    long_ma_str = f"{long_ma:.2f}" if not np.isnan(long_ma) else "N/A"
                    prev_short_ma_str = f"{prev_short_ma:.2f}" if not np.isnan(prev_short_ma) else "N/A"
                    prev_long_ma_str = f"{prev_long_ma:.2f}" if not np.isnan(prev_long_ma) else "N/A"

                    gain_str = ""
                    if entry_price is not None:
                        if position == 1:
                            gain = ((final_price - entry_price) / entry_price) * 100 if entry_price != 0 else 0
                        else:
                            gain = ((entry_price - final_price) / entry_price) * 100 if entry_price != 0 else 0
                        gain_str = f"{gain:+.2f}"

                    if position == 1:
                        action = "Sell*"
                        tag = "sell"
                        position_info = "(Simulated closing trade)"
                    else:
                        action = "Cover Short*"
                        tag = "cover"
                        position_info = "(Simulated closing trade)"

                    row_text = (
                        f"{action:<15}|{final_date.strftime('%Y-%m-%d'):<12}|{final_price:>10.2f}|{gain_str:>8}|{prev_short_ma_str:>14}|"
                        f"{short_ma_str:>12}|{prev_long_ma_str:>14}|{long_ma_str:>12}|{short_slope_str:>14}|{long_slope_str:>14}|{position_info:<25}"
                    )
                    trades_detail.insert(tk.END, row_text + "\n", tag)
                    trades_detail.insert(tk.END, f"\nNote: Position still open at end of simulation.\n", "note")

                # Add notes
                trades_detail.insert(tk.END, "\nNote: Trades typically occur in pairs:\n", "note")
                trades_detail.insert(tk.END, "- Buy → Sell (for long trades)\n", "note")
                trades_detail.insert(tk.END, "- Short Sell → Cover Short (for short trades)\n", "note")
                trades_detail.insert(tk.END, "The last trade may not have a matching closing trade if the position\n", "note")
                trades_detail.insert(tk.END, "is still open at the end of the simulation. If so, a simulated closing trade marked with an asterisk (*) is shown.\n\n", "note")
                trades_detail.insert(tk.END, "*Note: Trades with an * are not actual trades, they indicate a trade is still open at the end of the simulation.\n", "note")
                trades_detail.insert(tk.END, "Note: positive slope means the MA is trending upward, while a negative slope means it is trending downward.\n", "note")

                # Configure tags
                trades_detail.tag_configure("buy", foreground="green")
                trades_detail.tag_configure("sell", foreground="red")
                trades_detail.tag_configure("short", foreground="purple")
                trades_detail.tag_configure("cover", foreground="blue")
                trades_detail.tag_configure("note", foreground="gray", font=("Arial", 9, "italic"))

                trades_detail.config(state=tk.DISABLED)

                tk.Button(main_frame, text="Close", command=trades_window.destroy,
                         width=15, bg="#f0f0f0").pack(pady=10)

            # View trades button (moved outside Recent Trades box)
            view_trades_frame = tk.Frame(controls_frame, bg='#f0f0f0')
            view_trades_frame.pack(fill=tk.X, pady=5, padx=5)
            tk.Button(view_trades_frame, text="View Trades", command=show_detailed_trades,
                     width=15, font=("Arial", 10), bg="#f0f0f0").pack()

            # Embed plot
            canvas = FigureCanvasTkAgg(fig, master=chart_frame)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # Initialize chart
            update_chart()

        except Exception as e:
            logging.error(f"Error creating chart: {str(e)}")
            messagebox.showerror("Error", f"Failed to create chart: {str(e)}")

# === Trading GUI Class ===

class TradingGUI:
    """
    Main GUI class for the trading strategy analysis tool.

    This class manages the Tkinter interface, handles user interactions,
    and coordinates between the trading strategy and visualization components.
    """

    def __init__(self):
        """Initialize the GUI with state variables."""
        self.root = tk.Tk()
        self.root.title("Moving Average Trading Strategy Analysis")
        self.root.geometry("700x750")

        # State variables (eliminating globals)
        self.file_path_var = tk.StringVar(value=r"C:\Users\Kcrow\OneDrive\Documents\SPY Buy Sell Strategy Analysis\SPY Price History.csv")
        self.start_date_var = tk.StringVar(value="2000-01-01")
        self.end_date_var = tk.StringVar(value="2005-01-01")
        self.short_min_var = tk.IntVar(value=5)
        self.short_max_var = tk.IntVar(value=20)
        self.long_min_var = tk.IntVar(value=120)
        self.long_max_var = tk.IntVar(value=170)
        self.allow_short_selling_var = tk.BooleanVar(value=False)
        self.short_sell_only_var = tk.BooleanVar(value=False)
        self.include_slope_check_var = tk.BooleanVar(value=False)
        self.slope_lookback_var = tk.IntVar(value=3)

        # Progress variables
        self.progress_var = tk.DoubleVar()
        self.cancel_backtest = False

        # Initialize components
        self.strategy = TradingStrategy()
        self.visualization = Visualization(self.root)

        # UI elements (will be created in setup_ui)
        self.progress_bar = None
        self.progress_label = None
        self.heatmap_button = None
        self.trades_heatmap_button = None
        self.file_text = None

        self.setup_ui()

    def setup_ui(self):
        """Set up the main user interface."""
        # File selection
        tk.Label(self.root, text="Select CSV File:").pack(pady=5)
        
        # Text box for file path (centered with wrapping for long paths)
        file_entry_frame = tk.Frame(self.root)
        file_entry_frame.pack(pady=2, padx=10, fill=tk.X)
        
        # Create a Text widget instead of Entry for better handling of long paths
        file_text_frame = tk.Frame(file_entry_frame)
        file_text_frame.pack(expand=True)
        
        self.file_text = tk.Text(file_text_frame, height=2, width=80, wrap=tk.WORD, 
                                font=('TkDefaultFont', 9), relief=tk.SUNKEN, bd=1)
        self.file_text.pack(pady=2)
        
        # Bind the Text widget to update the StringVar
        def on_text_change(event=None):
            content = self.file_text.get("1.0", tk.END).strip()
            self.file_path_var.set(content)
        
        self.file_text.bind('<KeyRelease>', on_text_change)
        self.file_text.bind('<FocusOut>', on_text_change)
        
        # Update Text widget when StringVar changes
        def on_var_change(*args):
            current_content = self.file_text.get("1.0", tk.END).strip()
            new_content = self.file_path_var.get()
            if current_content != new_content:
                self.file_text.delete("1.0", tk.END)
                self.file_text.insert("1.0", new_content)
        
        self.file_path_var.trace('w', on_var_change)
        
        # Browse button (centered below the text box)
        browse_frame = tk.Frame(self.root)
        browse_frame.pack(pady=5)
        tk.Button(browse_frame, text="Browse", command=self.browse_file, width=15).pack()

        # Date inputs
        tk.Label(self.root, text="Start Date (YYYY-MM-DD):").pack(pady=5)
        tk.Entry(self.root, textvariable=self.start_date_var, width=20).pack(pady=2)
        tk.Label(self.root, text="End Date (YYYY-MM-DD):").pack(pady=5)
        tk.Entry(self.root, textvariable=self.end_date_var, width=20).pack(pady=2)

        # Chart button
        chart_frame = tk.Frame(self.root)
        chart_frame.pack(pady=5)
        tk.Button(chart_frame, text="Chart", command=self.show_price_chart,
                 width=15).pack()

        # Controls frame for MA ranges and slope options
        controls_frame = tk.Frame(self.root)
        controls_frame.pack(pady=10)

        # Moving Average Ranges frame (left side)
        ma_frame = tk.Frame(controls_frame, relief=tk.GROOVE, bd=2, padx=10, pady=10)
        ma_frame.pack(side=tk.LEFT, padx=5)

        tk.Label(ma_frame, text="Moving Average Ranges", font=("Arial", 12, "bold")).pack(pady=(0, 10))

        # Short MA range
        tk.Label(ma_frame, text="Short MA Range (min, max):").pack(pady=5)
        short_frame = tk.Frame(ma_frame)
        short_frame.pack(pady=2)
        tk.Entry(short_frame, textvariable=self.short_min_var, width=10).pack(side=tk.LEFT, padx=5)
        tk.Label(short_frame, text="to").pack(side=tk.LEFT)
        tk.Entry(short_frame, textvariable=self.short_max_var, width=10).pack(side=tk.LEFT, padx=5)

        # Long MA range
        tk.Label(ma_frame, text="Long MA Range (min, max):").pack(pady=5)
        long_frame = tk.Frame(ma_frame)
        long_frame.pack(pady=2)
        tk.Entry(long_frame, textvariable=self.long_min_var, width=10).pack(side=tk.LEFT, padx=5)
        tk.Label(long_frame, text="to").pack(side=tk.LEFT)
        tk.Entry(long_frame, textvariable=self.long_max_var, width=10).pack(side=tk.LEFT, padx=5)

        # Slope Check Options frame (right side)
        slope_frame = tk.Frame(controls_frame, relief=tk.GROOVE, bd=2, padx=10, pady=10)
        slope_frame.pack(side=tk.RIGHT, padx=5)

        tk.Label(slope_frame, text="Slope Check Options", font=("Arial", 12, "bold")).pack(pady=(0, 10))

        # Include slope check
        tk.Checkbutton(slope_frame, text="Include Slope Check",
                      variable=self.include_slope_check_var).pack(pady=2)

        # Lookback entry
        lookback_frame = tk.Frame(slope_frame)
        lookback_frame.pack(pady=2)
        tk.Label(lookback_frame, text="Lookback:").pack(side=tk.LEFT)
        tk.Entry(lookback_frame, textvariable=self.slope_lookback_var, width=5).pack(side=tk.LEFT, padx=5)

        # Short selling options
        options_frame = tk.Frame(self.root, pady=5)
        options_frame.pack()

        tk.Checkbutton(options_frame, text="Allow Short Selling",
                      variable=self.allow_short_selling_var,
                      command=self.toggle_short_sell_only).pack(side=tk.LEFT, padx=5)

        self.short_sell_cb = tk.Checkbutton(options_frame, text="Short Sell Only",
                                          variable=self.short_sell_only_var,
                                          state=tk.DISABLED)
        self.short_sell_cb.pack(side=tk.LEFT, padx=5)

        # Buttons frame
        buttons_frame = tk.Frame(self.root)
        buttons_frame.pack(pady=10)

        tk.Button(buttons_frame, text="Run Backtest", command=self.run_backtest,
                 width=15, bg="#4CAF50", fg="white", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)

        tk.Button(buttons_frame, text="Cancel", command=self.cancel_backtest_callback,
                 width=10, bg="#F44336", fg="white", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)

        # Results buttons
        tk.Button(self.root, text="View Results", command=self.view_results,
                 width=20, bg="#9C27B0", fg="white", font=("Arial", 10, "bold")).pack(pady=5)

        self.heatmap_button = tk.Button(self.root, text="Show Gain Heatmap",
                                      command=self.show_gain_heatmap, width=20,
                                      bg="#2196F3", fg="white", font=("Arial", 10, "bold"),
                                      state=tk.DISABLED)
        self.heatmap_button.pack(pady=5)

        self.trades_heatmap_button = tk.Button(self.root, text="Show Trades Heatmap",
                                             command=self.show_trades_heatmap, width=20,
                                             bg="#FF9800", fg="white", font=("Arial", 10, "bold"),
                                             state=tk.DISABLED)
        self.trades_heatmap_button.pack(pady=5)

        # Progress bar
        progress_frame = tk.Frame(self.root)
        progress_frame.pack(fill=tk.X, padx=20, pady=10)

        self.progress_bar = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL,
                                          length=550, mode='determinate',
                                          variable=self.progress_var)
        self.progress_bar.pack(side=tk.TOP, fill=tk.X)

        self.progress_label = tk.Label(progress_frame, text="")
        self.progress_label.pack(side=tk.TOP)

        # Credits
        credits_frame = tk.Frame(self.root, bg='#f0f0f0')
        credits_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=5)
        tk.Label(credits_frame, text="Moving Average Trading Strategy Analysis Tool",
                font=("Arial", 8), fg="#555555", bg='#f0f0f0').pack(pady=(5, 1))

        # Enable heatmap buttons if results exist
        self.check_existing_results()

    def browse_file(self):
        """Handle file browsing."""
        filename = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if filename:
            self.file_path_var.set(filename)

    def toggle_short_sell_only(self):
        """Toggle short sell only checkbox based on allow short selling."""
        if self.allow_short_selling_var.get():
            self.short_sell_cb.config(state=tk.NORMAL)
        else:
            self.short_sell_cb.config(state=tk.DISABLED)
            self.short_sell_only_var.set(False)

    def show_price_chart(self):
        """Show price chart using visualization component."""
        self.visualization.show_price_chart(
            self.file_path_var.get(),
            self.start_date_var.get(),
            self.end_date_var.get(),
            self.strategy
        )

    def run_backtest(self):
        """Run the backtest in a separate thread."""
        try:
            # Validate inputs
            start_date = datetime.strptime(self.start_date_var.get(), '%Y-%m-%d')
            end_date = datetime.strptime(self.end_date_var.get(), '%Y-%m-%d')

            if start_date >= end_date:
                messagebox.showerror("Error", "Start date must be before end date")
                return

            if (self.short_min_var.get() <= 0 or self.short_max_var.get() <= 0 or
                self.long_min_var.get() <= 0 or self.long_max_var.get() <= 0):
                messagebox.showerror("Error", "MA periods must be positive")
                return

            if (self.short_min_var.get() > self.short_max_var.get() or
                self.long_min_var.get() > self.long_max_var.get()):
                messagebox.showerror("Error", "Min values must be <= max values")
                return

            if self.short_max_var.get() >= self.long_min_var.get():
                messagebox.showerror("Error", "Short MA max must be < Long MA min")
                return

        except ValueError:
            messagebox.showerror("Error", "Invalid date format")
            return

        # Reset cancel flag and start backtest thread
        self.cancel_backtest = False
        backtest_thread = threading.Thread(target=self._run_backtest_worker)
        backtest_thread.daemon = True
        backtest_thread.start()

    def _run_backtest_worker(self):
        """Worker function for running backtest in separate thread."""
        try:
            logging.info("Starting backtest execution")

            # Track start time for duration calculation
            start_time = datetime.now()

            # Get parameters
            params = {
                'file_path': self.file_path_var.get(),
                'start_date': self.start_date_var.get(),
                'end_date': self.end_date_var.get(),
                'short_min': self.short_min_var.get(),
                'short_max': self.short_max_var.get(),
                'long_min': self.long_min_var.get(),
                'long_max': self.long_max_var.get(),
                'allow_short_selling': self.allow_short_selling_var.get(),
                'short_sell_only': self.short_sell_only_var.get(),
                'include_slope_check': self.include_slope_check_var.get(),
                'slope_lookback': self.slope_lookback_var.get()
            }

            # Progress callback
            def progress_callback(count, progress):
                if self.cancel_backtest:
                    raise Exception("Backtest cancelled")
                self.progress_var.set(progress)
                self.progress_label.config(text=f"{progress}% Complete")
                self.root.update_idletasks()

            # Run backtest
            results = self.strategy.run_backtest(progress_callback=progress_callback, **params)

            if not results:
                return

            # Process results
            results_df = pd.DataFrame(results)
            best_result = results_df.loc[results_df['annualized_gain'].idxmax()]

            # Create pivot tables
            gain_table = results_df.pivot(index='long_ma', columns='short_ma', values='annualized_gain')
            trade_count_table = results_df.pivot(index='long_ma', columns='short_ma', values='trade_count')

            # Calculate duration and metadata for results
            end_time = datetime.now()
            elapsed_time = (end_time - start_time).total_seconds()
            hours, remainder = divmod(int(elapsed_time), 3600)
            minutes, seconds = divmod(remainder, 60)
            duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            num_combinations = len(results)

            # Save results with metadata
            self._save_results(results_df, params, best_result, duration_str, num_combinations)

            # Show results
            result_message = (
                f"Analysis Complete!\n\n"
                f"Best Combination:\n"
                f"Short MA: {best_result['short_ma']}\n"
                f"Long MA: {best_result['long_ma']}\n"
                f"Annualized %Gain: {best_result['annualized_gain']:.2f}%\n"
                f"Final Wealth: ${best_result['final_wealth']:.2f}\n\n"
                f"Duration: {duration_str}\n"
                f"Processed {len(results)} combinations."
            )
            messagebox.showinfo("Results", result_message)

            # Reset progress and enable buttons
            self.progress_var.set(0)
            self.progress_label.config(text="")
            self.heatmap_button.config(state=tk.NORMAL)
            self.trades_heatmap_button.config(state=tk.NORMAL)

        except Exception as e:
            logging.error(f"Backtest error: {str(e)}")
            messagebox.showerror("Error", f"Backtest failed: {str(e)}")
            self.progress_var.set(0)
            self.progress_label.config(text="")

    def _save_results(self, results_df, params, best_result, duration_str=None, num_combinations=None):
        """Save backtest results to CSV file."""
        filename = "ma_strategy_results.csv"

        with open(filename, 'w') as f:
            f.write(f"# Moving Average Crossover Strategy Results\n")
            f.write(f"# Date Range: {params['start_date']} to {params['end_date']}\n")
            f.write(f"# Allow Short Selling: {params['allow_short_selling']}\n")
            f.write(f"# Short Sell Only: {params['short_sell_only']}\n")
            f.write(f"# Include Slope Check: {params['include_slope_check']}\n")
            f.write(f"# Slope Lookback Period: {params['slope_lookback']}\n")
            f.write(f"# Short MA Range: {params['short_min']} to {params['short_max']}\n")
            f.write(f"# Long MA Range: {params['long_min']} to {params['long_max']}\n")
            f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            if duration_str:
                f.write(f"# Duration: {duration_str}\n")
            if num_combinations:
                f.write(f"# Processed {num_combinations} combinations\n")
            f.write(f"# Initial Balance: $10,000\n")
            f.write(f"# Best Combination: Short MA={best_result['short_ma']}, Long MA={best_result['long_ma']}, Annualized %Gain={best_result['annualized_gain']:.2f}%\n")
            f.write(f"# Final Wealth: ${best_result['final_wealth']:.2f} (excluding brokerage)\n")
            f.write("#\n")

        results_df.to_csv(filename, mode='a', index=False)

    def cancel_backtest_callback(self):
        """Handle backtest cancellation."""
        if self.progress_var.get() > 0:
            self.cancel_backtest = True
            self.progress_label.config(text="Cancelling...")
            messagebox.showinfo("Cancelling", "Backtest cancellation requested")
        else:
            messagebox.showinfo("No Backtest Running", "No backtest currently running")

    def show_gain_heatmap(self):
        """Show gain heatmap."""
        try:
            combined_data = pd.read_csv("ma_strategy_results.csv", comment='#')
            gain_table = combined_data.pivot(index='long_ma', columns='short_ma', values='annualized_gain')

            max_gain = gain_table.max().max()
            min_gain = min(0, gain_table.min().min())
            bounds = np.linspace(min_gain, max_gain, 7)
            bounds = np.round(bounds * 2) / 2

            colors = ["#87CEEB", "#90EE90", "#FFFF00", "#FFD700", "#FF8C00", "#FF0000"]
            cmap = mcolors.ListedColormap(colors)

            self.visualization.create_heatmap_window(
                gain_table, "Annualized %Gain Heatmap", "Annualized %Gain",
                cmap, bounds, colors
            )
        except Exception as e:
            logging.error(f"Error showing gain heatmap: {str(e)}")
            messagebox.showerror("Error", f"Failed to show heatmap: {str(e)}")

    def show_trades_heatmap(self):
        """Show trades heatmap."""
        try:
            combined_data = pd.read_csv("ma_strategy_results.csv", comment='#')
            trade_count_table = combined_data.pivot(index='long_ma', columns='short_ma', values='trade_count')

            max_trades = trade_count_table.max().max()
            bounds = np.linspace(0, max_trades, 6)
            bounds = np.round(bounds).astype(int)
            if bounds[-1] == max_trades:
                bounds = np.append(bounds, max_trades + 1)

            colors = ["#FFFFFF", "#FFCCCC", "#FF9999", "#FF6666", "#FF3333", "#FF0000"]
            cmap = mcolors.ListedColormap(colors)

            self.visualization.create_heatmap_window(
                trade_count_table, "Trade Count Heatmap", "Number of Trades",
                cmap, bounds, colors
            )
        except Exception as e:
            logging.error(f"Error showing trades heatmap: {str(e)}")
            messagebox.showerror("Error", f"Failed to show heatmap: {str(e)}")

    def view_results(self):
        """View results file in a scrollable window."""
        results_file = "ma_strategy_results.csv"
        if not os.path.exists(results_file):
            messagebox.showinfo("No Results", "No results file found")
            return

        try:
            results_window = tk.Toplevel(self.root)
            results_window.title("Backtest Results")
            results_window.geometry("800x600")

            main_frame = tk.Frame(results_window, padx=10, pady=10)
            main_frame.pack(fill=tk.BOTH, expand=True)

            tk.Label(main_frame, text="Backtest Results", font=("Arial", 14, "bold")).pack(pady=(0, 10))

            text_frame = tk.Frame(main_frame)
            text_frame.pack(fill=tk.BOTH, expand=True)

            v_scrollbar = tk.Scrollbar(text_frame, orient=tk.VERTICAL)
            v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            h_scrollbar = tk.Scrollbar(text_frame, orient=tk.HORIZONTAL)
            h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

            results_text = tk.Text(text_frame, font=("Courier", 10), wrap=tk.NONE,
                                 yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
            results_text.pack(fill=tk.BOTH, expand=True)

            v_scrollbar.config(command=results_text.yview)
            h_scrollbar.config(command=results_text.xview)

            with open(results_file, 'r') as f:
                results_content = f.read()
            results_text.insert(tk.END, results_content)
            results_text.config(state=tk.DISABLED)

            tk.Button(main_frame, text="Close", command=results_window.destroy,
                     width=15).pack(pady=10)

        except Exception as e:
            messagebox.showerror("Error", f"Error opening results: {str(e)}")

    def check_existing_results(self):
        """Check if results file exists and enable buttons accordingly."""
        try:
            if os.path.exists("ma_strategy_results.csv"):
                self.heatmap_button.config(state=tk.NORMAL)
                self.trades_heatmap_button.config(state=tk.NORMAL)
        except:
            pass

    def run(self):
        """Start the GUI main loop."""
        self.root.mainloop()

# === Unit Tests ===

class TestTradingStrategy(unittest.TestCase):
    """
    Unit tests for the TradingStrategy class.

    Tests cover critical methods including data loading, signal generation,
    and edge cases.
    """

    def setUp(self):
        """Set up test fixtures."""
        self.strategy = TradingStrategy()
        # Create sample data
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        prices = np.random.uniform(100, 200, 100)
        self.sample_df = pd.DataFrame({'Date': dates, 'Price': prices})

    def test_read_csv_file_not_found(self):
        """Test read_csv with non-existent file."""
        result = self.strategy.read_csv("nonexistent.csv")
        self.assertIsNone(result)

    def test_calculate_moving_averages(self):
        """Test moving average calculation."""
        result = self.strategy.calculate_moving_averages(self.sample_df, 5, 10)
        self.assertIn('Short_MA', result.columns)
        self.assertIn('Long_MA', result.columns)
        # Check that MAs are calculated correctly
        self.assertTrue(result['Short_MA'].isna().sum() < len(result))

    def test_generate_signals_basic(self):
        """Test basic signal generation without slope check."""
        df_with_ma = self.strategy.calculate_moving_averages(self.sample_df, 5, 10)
        result = self.strategy.generate_signals(df_with_ma, False, 3)
        self.assertIn('Signal', result.columns)
        # Should have some signals
        self.assertTrue((result['Signal'] != 0).any())

    def test_calculate_wealth(self):
        """Test wealth calculation."""
        df_with_ma = self.strategy.calculate_moving_averages(self.sample_df, 5, 10)
        df_with_signals = self.strategy.generate_signals(df_with_ma, False, 3)
        wealth, gain, trades = self.strategy.calculate_wealth(df_with_signals)
        self.assertIsInstance(wealth, (int, float))
        self.assertIsInstance(gain, (int, float))
        self.assertIsInstance(trades, list)

    def test_slope_parameters_passed(self):
        """Test that slope parameters are correctly passed to run_backtest."""
        # This is a mock test - in real scenario would need to mock file operations
        # For now, just test that the method exists and accepts parameters
        self.assertTrue(hasattr(self.strategy, 'run_backtest'))
        # Test method signature
        import inspect
        sig = inspect.signature(self.strategy.run_backtest)
        params = list(sig.parameters.keys())
        self.assertIn('include_slope_check', params)
        self.assertIn('slope_lookback', params)

    def test_empty_csv_handling(self):
        """Test handling of empty CSV data."""
        empty_df = pd.DataFrame(columns=['Date', 'Price'])
        df_with_ma = self.strategy.calculate_moving_averages(empty_df, 5, 10)
        result = self.strategy.generate_signals(df_with_ma, False, 3)
        self.assertEqual(len(result), 0)

    def test_invalid_dates(self):
        """Test handling of invalid date formats."""
        invalid_df = pd.DataFrame({
            'Date': ['invalid_date', '2020-01-01'],
            'Price': [100, 101]
        })
        # This would normally be handled in read_csv, but testing the core logic
        df_copy = invalid_df.copy()
        df_copy['Date'] = pd.to_datetime(df_copy['Date'], errors='coerce')
        result = self.strategy.calculate_moving_averages(df_copy, 5, 10)
        self.assertTrue(result['Date'].isna().iloc[0])

# === Main Entry Point ===

def main():
    """Main entry point for the application."""
    logging.info("Starting Trading Strategy Analysis Tool")
    gui = TradingGUI()
    gui.run()

if __name__ == "__main__":
    main()
