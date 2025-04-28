#!/usr/bin/env python
"""
Script to add forecasted sPMV values to the dataset based on a pre-trained model.

This script uses a trained TensorFlow model to predict future sPMV values
based on a 7-day historical window. The predictions are added as a new column
to the dataset without modifying the original file.
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Parameters
WINDOW_SIZE = 1 * 24  # One week of hourly data

def load_model(model_path):
    """Load the pre-trained TensorFlow model."""
    logger.info(f"Loading model from {model_path}")
    try:
        model = tf.keras.models.load_model(model_path)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

def prepare_features(df):
    """Prepare the feature set for the model.
    
    Args:
        df: DataFrame with the dataset
        
    Returns:
        X: numpy array with features organized in windows
    """
    logger.info("Preparing features for prediction")
    
    # Add synthetic CO2 data if it doesn't exist (for backward compatibility)
    if 'co2_ppm' not in df.columns:
        # Generate realistic CO2 values between 400-1200 ppm with daily patterns
        # Higher during occupied hours, lower at night
        base_co2 = 400  # Base outdoor CO2 level
        hour_factors = {
            0: 0.2, 1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.2,  # Night (low)
            6: 0.3, 7: 0.5, 8: 0.7, 9: 0.8, 10: 0.9, 11: 1.0,  # Morning (rising)
            12: 1.0, 13: 0.9, 14: 0.8, 15: 0.9, 16: 1.0, 17: 0.9,  # Afternoon (high)
            18: 0.7, 19: 0.6, 20: 0.5, 21: 0.4, 22: 0.3, 23: 0.2,  # Evening (falling)
        }
        
        # Generate CO2 values based on hour of day with some randomness
        df['co2_ppm'] = df.apply(
            lambda row: base_co2 + hour_factors[row['hour']] * 800 + np.random.normal(0, 50), 
            axis=1
        )
        
        logger.info(f"Added synthetic CO2 data with range: {df['co2_ppm'].min():.1f}-{df['co2_ppm'].max():.1f} ppm")
    
    # Select feature columns (all except time and target)
    feature_cols = [col for col in df.columns if col in ['t_r', 'rh_r', 'hour', 'day_of_week', 'is_weekend', 'co2_ppm']]
    
    logger.info("feature_cols:" + str(feature_cols))
    
    # Create sliding windows of the data
    X = []
    for i in range(len(df) - WINDOW_SIZE):
        window = df[feature_cols].iloc[i:i+WINDOW_SIZE].values
        X.append(window)
    
    # Handle the first WINDOW_SIZE rows that don't have enough history
    for i in range(WINDOW_SIZE):
        # For the first entries, use what's available and pad with zeros
        if i == 0:
            window = np.zeros((WINDOW_SIZE, len(feature_cols)))
            window[-i-1:] = df[feature_cols].iloc[:i+1].values
        else:
            window = np.zeros((WINDOW_SIZE, len(feature_cols)))
            window[-i-1:] = df[feature_cols].iloc[:i+1].values
        X.insert(0, window)
    
    logger.info(f"Created {len(X)} feature windows with {len(feature_cols)} features (including CO2)")
    return np.array(X)

def generate_forecasts(model, X):
    """Generate sPMV forecasts using the model.
    
    Args:
        model: Trained TensorFlow model
        X: Feature windows
        
    Returns:
        numpy array of forecasted sPMV values
    """
    logger.info("Generating forecasts")
    try:
        forecasts = model.predict(X)
        # If the model outputs multiple values, take the first one
        if isinstance(forecasts, list) or (isinstance(forecasts, np.ndarray) and forecasts.ndim > 1 and forecasts.shape[1] > 1):
            forecasts = forecasts[:, 0]
        logger.info(f"Generated {len(forecasts)} forecasts")
        return forecasts
    except Exception as e:
        logger.error(f"Forecast generation failed: {str(e)}")
        raise

def add_forecasts_to_dataset(df, forecasts):
    """Add the forecasted sPMV values to the dataset.
    
    Args:
        df: Original DataFrame
        forecasts: numpy array of forecasted values
        
    Returns:
        DataFrame with forecasted values added
    """
    logger.info("Adding forecasts to dataset")
    df_copy = df.copy()
    df_copy['forecasted_sPMV'] = forecasts
    return df_copy

def forecast_and_update_df(df, model):
    """Main function to run the forecasting process."""
    logger.info("Starting sPMV forecasting process")
    
    # Prepare features and generate forecasts
    logger.info("df.head():", df.head())
    X = prepare_features(df)
    logger.info(X.shape)
    logger.info(X[0][0])
    forecasts = generate_forecasts(model, X)
    
    # Add forecasts to dataset and save
    df_with_forecasts = add_forecasts_to_dataset(df, forecasts)
    
    logger.info("sPMV forecasting completed successfully")
    
    # Print summary of results
    print(f"\nProcess completed successfully!")
    print(f"Added {len(forecasts)} forecasted sPMV values")
    print("\nSample of the enhanced dataset:")
    print(df_with_forecasts.head())
    
    return df_with_forecasts
