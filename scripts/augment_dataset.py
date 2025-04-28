import pandas as pd
import numpy as np
import os
import traceback
import sys
from datetime import datetime

from scripts.Calculate_sPMV_v1 import sPMV_calculation
#from data_preprocessing import load_data, preprocess_data, save_features_file

def augment_dataset(initial_df, sensor_name):
    """
    Augment the dataset with new data.

    Parameters:
    initial_df (pd.DataFrame): The initial DataFrame to augment.

    Returns:
    pd.DataFrame: The augmented DataFrame.
    """
    try:
        # Load the initial dataset
        df = initial_df.copy()

        # Concatenate the new data with the existing dataset
        #df = pd.concat([df, new_data], ignore_index=True)

        # Recalculate sPMV for the augmented dataset
        df['sPMV'] = sPMV_calculation(df)

        return df

    except Exception as e:
        print(f"Error during dataset augmentation: {e}")
        traceback.print_exc()