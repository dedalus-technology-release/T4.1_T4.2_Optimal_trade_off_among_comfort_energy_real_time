# -*- coding: utf-8 -*-
"""
Data Preprocessing with sPMV Calculation - Fixed Version
Focus: Reading data, calculating sPMV, and creating features
"""

import pandas as pd
import numpy as np
import os
import traceback
import sys
from datetime import datetime

# Import sPMV calculation
try:
    from scripts.Calculate_sPMV_v1 import sPMV_calculation
    print("Successfully imported sPMV_calculation function")
except ImportError:
    print("ERROR: Could not import sPMV_calculation function from sPMV_v1.py")
    sys.exit(1)

def preprocess_data(df):
    """
    Process data to extract required features:
    - sPMV
    - Baseline Energy Consumption (from input)
    - Flexibility Below (from input)
    - Energy Average
    - Indoor Temperature (t_r)
    - Indoor Relative Humidity (rh_r)
    - Time information
    """
    try:
        print("\nProcessing data...")
        
        # Convert time column to datetime
        df['time'] = pd.to_datetime(df['time'])
        
        # Filter data starting from 2024-07-04 21:00:00
        #start_date = pd.to_datetime('2024-07-04 21:00:00+00:00')
        #df = df[df['time'] >= start_date].copy()
        #print(f"Filtered data to start from {start_date}")
        
        # Handle missing values in required columns
        for col in ['t_r', 'rh_r']:
            if col in df.columns:
                df[col] = df[col].ffill().bfill()
            else:
                print(f"ERROR: Required column {col} not found")
                sys.exit(1)
        
        # Create time-based features
        df['hour'] = df['time'].dt.hour
        df['day_of_week'] = df['time'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Calculate sPMV
        print("Calculating sPMV...")
        spmv_df = sPMV_calculation(df['t_r'], df['rh_r'], pd.Series(df['time']))
        df = pd.merge(df, spmv_df[['DATE', 'sPMV']], left_on='time', right_on='DATE', how='left')
        df['sPMV'] = df['sPMV'].ffill().bfill()
        
        # Calculate energy average from baseline and flexibility_below
        df['energy_average'] = (df['baseline'] + df['flexibility_below']) / 2
        
        # Select required features
        features = [
            'time',
            'sPMV',
            'baseline',
            'flexibility_below',
            'energy_average',
            't_r',
            'rh_r',
            'hour',
            'day_of_week',
            'is_weekend'
        ]
        
        df_final = df[features].copy()
        print("\nFinal features summary:")
        print(f"Shape: {df_final.shape}")
        print(f"Columns: {df_final.columns.tolist()}")
        
        return df_final
        
    except Exception as e:
        print(f"ERROR in data preprocessing: {e}")
        traceback.print_exc()
        sys.exit(1)
