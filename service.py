import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta, timezone

from scripts.Calculate_sPMV_v1 import sPMV_calculation
from scripts.augment_dataset import augment_dataset
from scripts.control_dbf_apis import get_token, get_device_data_grouped
from scripts.data_preprocessing import preprocess_data
from scripts.forecast_spmv import forecast_and_update_df, prepare_features, generate_forecasts
import tensorflow as tf
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

load_dotenv()  # take environment variables from .env

BASE_DIR = "data"
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'final_model.h5')
USERNAME = os.getenv('DOMX_USERNAME')
PASSWORD = os.getenv('DOMX_PASSWORD')
DEFAULT_START_DAYS_AGO = 7  # if there is no CSV file, start from 7 days ago

MODEL  = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"mse": tf.keras.metrics.MeanSquaredError()}
)

app = FastAPI()

def iso_date(dt):
    return dt.replace(hour=0, minute=0, second=0, microsecond=0).isoformat().replace("+00:00", "Z")

def get_last_date_in_csv(csv_path):
    df = pd.read_csv(csv_path)
    if 'time' in df.columns and not df.empty:
        return pd.to_datetime(df['time']).max().replace(tzinfo=timezone.utc)
    return None

def convert_response_to_dataframe(data):
    
    dfs = []
    for key, entries in data.items():
        df = pd.DataFrame(entries)
        df.rename(columns={"value": key}, inplace=True)
        dfs.append(df)

    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = pd.merge(merged_df, df, on="time")

    merged_df["time"] = pd.to_datetime(merged_df["time"])
    
    return merged_df

def add_random_columns(df):
    np.random.seed(42)
    df["baseline"] = np.random.uniform(0, 361, size=len(df))
    df["flexibility_below"] = np.random.uniform(0, 253, size=len(df))
    return df

def update_datasets():
    token = get_token(USERNAME, PASSWORD)
    
    if not token:
        print("Token retrieval failed.")
        return
    
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    
    for building, apartment in config.items():
        for exx, sensors in apartment.items():
            sensor_id = sensors.get("air_quality")
            if not sensor_id:
                continue

            folder_path = os.path.join(DATASETS_DIR, building, exx)
            os.makedirs(folder_path, exist_ok=True)

            csv_path = os.path.join(folder_path, f"{exx}_dataset.csv")

            # Calcola la data di partenza
            if os.path.exists(csv_path):
                last_date = get_last_date_in_csv(csv_path)
                if last_date:
                    date_from = last_date + timedelta(days=1)
                else:
                    date_from = today - timedelta(days=DEFAULT_START_DAYS_AGO)
            else:
                date_from = today - timedelta(days=DEFAULT_START_DAYS_AGO)

            # check empty request
            if date_from >= today:
                print(f"No new data for {sensor_id} ({building} - {exx})")
                continue

            # download new data
            print(f"[{building} - {exx}] download from {iso_date(date_from)} to {iso_date(today)}")
            data_json = get_device_data_grouped(
                token,
                sensor_id,
                iso_date(date_from),
                iso_date(today),
                metrics=["t_r", "rh_r"],
                interval="1h"
            )

            try:
                new_df = convert_response_to_dataframe(data_json)
                new_df = add_random_columns(new_df) # To be replaced with actual energy data
                new_df = preprocess_data(new_df)
                
                X = prepare_features(new_df[-24:])
                forecasts = generate_forecasts(MODEL, X)
                forecast_df = pd.DataFrame(forecasts, columns=["forecasted_sPMV"])
                forecast_dates = pd.date_range(start=today, periods=24, freq='H', tz='UTC')
                forecast_df['time'] = forecast_dates
                forecast_df = forecast_df[['time', 'forecasted_sPMV']]
                forecast_df.to_csv(os.path.join( folder_path, "forecasted_sPMV.csv"), index=False)
                
                new_df = forecast_and_update_df(new_df, MODEL)
                print(new_df.head())
                if os.path.exists(csv_path):
                    existing_df = pd.read_csv(csv_path)
                    existing_df["time"] = pd.to_datetime(existing_df["time"])
                    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                    combined_df.drop_duplicates(subset=["time"], inplace=True)
                else:
                    combined_df = new_df

                combined_df.sort_values(by="time", inplace=True)
                combined_df.to_csv(csv_path, index=False)
                print(f"Updated data saved in {csv_path}")
            except Exception as e:
                print(f"Error during saving {sensor_id}: {e}")

@app.get("/forecast_sPMV")
def forecast_sPMV():
    update_datasets()
    return {"message": "Forecasting completed!"}

@app.get("/sPMV/{building}/{exx}")
def GET_forecasted_sPMV(building: str, exx: str):
    folder_path = os.path.join(DATASETS_DIR, building, exx)
    csv_path = os.path.join(folder_path, f"forecasted_sPMV.csv")
    
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail=f"File not found for building '{building}' and apartment '{exx}'.")

    try:
        df = pd.read_csv(csv_path)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading CSV file: {str(e)}")

#if __name__ == "__main__":
#    update_datasets()
