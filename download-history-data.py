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
START_DATE = datetime(2024, 7, 1, tzinfo=timezone.utc)

MODEL  = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"mse": tf.keras.metrics.MeanSquaredError()}
)

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

def fill_missing_hours_with_hourly_means(df):
    df["time"] = pd.to_datetime(df["time"])
    df["hour"] = df["time"].dt.hour
    df["date"] = df["time"].dt.date
    df["synthetic"] = 0  # real data

    complete_df = df.copy()

    # Calculates historical averages for each hour
    hourly_means = df.groupby("hour").mean(numeric_only=True)

    start_date = df["time"].min().normalize()
    tz = df["time"].dt.tz if df["time"].dt.tz is not None else None
    end_date = (datetime.now(tz=tz).replace(hour=0, minute=0, second=0, microsecond=0)) - timedelta(days=1)


    all_days = pd.date_range(start=start_date, end=end_date, freq="D")

    for day in all_days:
        day_data = df[df["time"].dt.date == day.date()]
        if len(day_data) == 24:
            continue  # Full day, skip

        existing_hours = set(day_data["hour"])
        missing_hours = set(range(24)) - existing_hours

        for hour in missing_hours:
            timestamp = datetime.combine(day.date(), datetime.min.time()) + timedelta(hours=hour)
            mean_row = hourly_means.loc[hour].copy()
            row = mean_row.to_dict()
            row["time"] = pd.Timestamp(timestamp, tz="UTC")
            row["hour"] = hour
            row["date"] = day.date()
            row["synthetic"] = 1
            complete_df = pd.concat([complete_df, pd.DataFrame([row])], ignore_index=True)

    complete_df.drop(columns=["hour", "date"], errors="ignore", inplace=True)
    complete_df.sort_values("time", inplace=True)
    complete_df.reset_index(drop=True, inplace=True)

    return complete_df


def download_history_data():
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

            if START_DATE >= today:
                print(f"No data to retrieve: START_DATE >= today")
                continue

            all_data = []

            # Itera mese per mese
            current_start = START_DATE
            while current_start < today:
                current_end = min(current_start + timedelta(days=30), today)

                print(f"[{building} - {exx}] Downloading from {iso_date(current_start)} to {iso_date(current_end)}")

                try:
                    data_json = get_device_data_grouped(
                        token,
                        sensor_id,
                        iso_date(current_start),
                        iso_date(current_end),
                        metrics=["t_r", "rh_r"],
                        interval="1h"
                    )

                    temp_df = convert_response_to_dataframe(data_json)
                    all_data.append(temp_df)

                except Exception as e:
                    print(f"⚠️  Error fetching data from {current_start.date()} to {current_end.date()}: {e}")

                current_start = current_end

            # Unisci tutti i dati
            if not all_data:
                print(f"⚠️  No data retrieved for {sensor_id} ({building} - {exx})")
                continue

            new_df_before = pd.concat(all_data, ignore_index=True)

            new_df_before = fill_missing_hours_with_hourly_means(new_df_before)

            try:
                #new_df = convert_response_to_dataframe(data_json)
                new_df = add_random_columns(new_df_before) # To be replaced with actual energy data
                new_df = preprocess_data(new_df_before)
                new_df['synthetic'] = new_df_before['synthetic']
                
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

if __name__ == "__main__":
    download_history_data()
