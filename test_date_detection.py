#!/usr/bin/env python3
"""
Test script to verify the date detection functionality
"""

import pandas as pd
from trading_strategy import TradingStrategy
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_date_detection():
    print("Testing date detection functionality...")
    
    # Create strategy instance
    strategy = TradingStrategy()
    
    # Read sample data
    print("\n1. Reading sample CSV data...")
    sample_df = pd.read_csv('SPY Price History.csv', nrows=20, header=None)
    print(f"Sample data shape: {sample_df.shape}")
    print(f"Sample columns: {list(sample_df.columns)}")
    print(f"First few rows:\n{sample_df.head()}")
    
    # Test column mapping for headerless file
    print("\n2. Testing column mapping...")
    column_mapping = {'Date': sample_df.columns[0], 'Price': sample_df.columns[1]}
    print(f"Column mapping: {column_mapping}")
    
    # Extract date samples
    print("\n3. Extracting date samples...")
    date_col_name = column_mapping.get('Date')
    print(f"Date column name: {date_col_name}")
    print(f"Date column exists in sample_df: {date_col_name in sample_df.columns}")
    
    if date_col_name is not None and date_col_name in sample_df.columns:
        date_samples = sample_df[date_col_name].dropna().astype(str).tolist()
        print(f"Extracted {len(date_samples)} date samples: {date_samples}")
        
        # Test date detection
        print("\n4. Testing date format detection...")
        date_detection = strategy._detect_date_format(date_samples)
        print(f"Date detection result: {date_detection}")
        
        if date_detection:
            print(f"✓ Detected format: {date_detection['format']}")
            print(f"✓ Confidence: {date_detection['confidence']:.1%}")
            print(f"✓ Samples parsed: {date_detection['samples_parsed']}")
        else:
            print("✗ Date detection failed")
    else:
        print("✗ Could not extract date samples")

if __name__ == "__main__":
    test_date_detection()
