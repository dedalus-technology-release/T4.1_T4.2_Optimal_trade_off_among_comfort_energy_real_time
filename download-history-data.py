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
from scripts.flexibility_heating_service_apis import get_flexibility_heating, EnergyEntry
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


#def fill_missing_with_weekday_hourly_means(df, columns):
#    # Estrai giorno della settimana e orario (HH:MM)
#    df["weekday"] = df["time"].dt.weekday       # 0=Monday, 6=Sunday
#    df["hour_minute"] = df["time"].dt.strftime("%H:%M")
#
#    # Per ciascuna colonna da riempire
#    for col in columns:
#        # Calcola la media per combinazione (weekday, hour_minute)
#        group_means = df.groupby(["weekday", "hour_minute"])[col].transform("mean")
#        
#        # Riempie solo i valori NaN con la media corrispondente
#        df[col] = df[col].fillna(group_means)
#
#    # Rimuove colonne temporanee
#    df.drop(columns=["weekday", "hour_minute"], inplace=True)
    
def fill_missing_hours_with_weekday_hour_means(df, columns_to_average):
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df["hour"] = df["time"].dt.hour
    df["weekday"] = df["time"].dt.dayofweek  # 0 = lunedì
    df["date"] = df["time"].dt.date
    df["synthetic"] = 0  # 0 = dati reali

    complete_df = df.copy()

    # Calcola medie solo per le colonne specificate
    mean_by_hour_day = df.groupby(["weekday", "hour"])[columns_to_average].mean()

    # Definizione intervallo temporale
    start_date = df["time"].min().normalize()
    tz = df["time"].dt.tz if df["time"].dt.tz is not None else None
    end_date = (datetime.now(tz=tz).replace(hour=0, minute=0, second=0, microsecond=0)) - timedelta(days=1)
    all_days = pd.date_range(start=start_date, end=end_date, freq="D")

    for day in all_days:
        day_data = df[df["time"].dt.date == day.date()]
        if len(day_data) == 24:
            continue  # Giorno completo

        existing_hours = set(day_data["hour"])
        missing_hours = set(range(24)) - existing_hours
        weekday = day.dayofweek

        for hour in missing_hours:
            timestamp = datetime.combine(day.date(), datetime.min.time()) + timedelta(hours=hour)
            if (weekday, hour) not in mean_by_hour_day.index:
                continue  # Nessuna media disponibile

            # Crea una nuova riga solo con le colonne specificate
            mean_values = mean_by_hour_day.loc[(weekday, hour)].to_dict()
            row = {col: mean_values.get(col, 0) for col in columns_to_average}
            row["time"] = pd.Timestamp(timestamp, tz="UTC")
            row["hour"] = hour
            row["weekday"] = weekday
            row["date"] = day.date()
            row["synthetic"] = 1
            complete_df = pd.concat([complete_df, pd.DataFrame([row])], ignore_index=True)

    complete_df.drop(columns=["hour", "weekday", "date"], errors="ignore", inplace=True)
    complete_df.sort_values("time", inplace=True)
    complete_df.reset_index(drop=True, inplace=True)

    return complete_df


def sliding_two_month_windows(df, step_days=30, window_days=60):
    #end_date = df["time"].max().replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = df["time"].max()
    start_date = df["time"].min()
    windows = []

    current_end = end_date
    while current_end > start_date:
        current_start = current_end - timedelta(days=window_days)
        #mask = (df["time"] >= current_start) & (df["time"] < current_end)
        mask = (df["time"] >= current_start) & (df["time"] <= current_end)
        window_df = df.loc[mask]
        if not window_df.empty:
            windows.append(window_df)  # Aggiungiamo normalmente
        current_end -= timedelta(days=step_days)

    return windows

def update_all_data_with_results(all_data_df, results):
    # Assicura che le colonne esistano
    for col in ["baseline", "flexibility_above", "flexibility_below"]:
        if col not in all_data_df.columns:
            all_data_df[col] = pd.NA
        syn_col = f"synthetic_{col}"
        if syn_col not in all_data_df.columns:
            all_data_df[syn_col] = 1
    
    for entry in results:
        time = pd.to_datetime(entry.get("time")).tz_convert("UTC") if pd.to_datetime(entry.get("time")).tzinfo else pd.to_datetime(entry.get("time")).tz_localize("UTC")
        mask = all_data_df["time"] == time
        if mask.any():
            for key in ["baseline", "flexibility_above", "flexibility_below"]:
                value = entry.get(key)
                if value is not None:
                    all_data_df.loc[mask, key] = value
                    all_data_df.loc[mask, f"synthetic_{key}"] = 0  # Marca come reale

                    
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
            smart_meter_id = sensors.get("smart_meters")
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
                    
                    
                    energy_json = get_device_data_grouped(
                        token,
                        smart_meter_id,
                        iso_date(current_start),
                        iso_date(current_end),
                        metrics=["energy_a"],
                        interval="1h"
                    )

                    energy_df = pd.DataFrame(energy_json["energy_a"])     
                    energy_df.rename(columns={"value": "energy_average"}, inplace=True)
                    energy_df["time"] = pd.to_datetime(energy_df["time"])
                    
                    temp_df = pd.merge(temp_df, energy_df, on="time", how="left")
                    
                    all_data.append(temp_df)             

                except Exception as e:
                    print(f"⚠️  Error fetching data from {current_start.date()} to {current_end.date()}: {e}")

                current_start = current_end

            all_data_df = pd.concat(all_data).sort_values("time").reset_index(drop=True)
            
            for col in ["baseline", "flexibility_above", "flexibility_below"]:
                if col not in all_data_df.columns:
                    all_data_df[col] = None
            
            print("all_data_df.tail(3):")
            print(all_data_df.tail(3)) #2025-05-29
            
            #all_data_df = fill_missing_with_weekday_hourly_means(all_data_df)
            all_data_df = fill_missing_hours_with_weekday_hour_means(all_data_df, ["energy_average","t_r","rh_r"])
            
            # Genera finestre temporali da 2 mesi con overlap
            windows = sliding_two_month_windows(all_data_df)

            windows[1].to_csv("./last_window.csv", index=False)  # Save the first window for debugging
            
            for window_df in windows:
                energy_entries = [
                    EnergyEntry(time=row["time"].to_pydatetime(), energy_average=row["energy_average"])
                    for _, row in window_df.iterrows()
                    if not pd.isna(row["energy_average"])
                ]
                if energy_entries:
                    result = get_flexibility_heating(energy_entries)
                    update_all_data_with_results(all_data_df, result)       
            
            #fill_missing_with_weekday_hourly_means(
            #    all_data_df,
            #    ["baseline", "flexibility_above", "flexibility_below"]
            #)
            
            all_data_df = fill_missing_hours_with_weekday_hour_means(all_data_df, ["baseline", "flexibility_above", "flexibility_below"])
            
            #print("AFTER update_all_data_with_results Final column order:")
            #print(all_data_df.columns.tolist())
            #
            #print("AFTER update_all_data_with_results Sample rows:")
            #print(all_data_df.tail(3)[['time', 'baseline', 'flexibility_below', 'flexibility_above']])
            
            # Unisci tutti i dati
            if all_data_df.empty:
                print(f"⚠️  No data retrieved for {sensor_id} ({building} - {exx})")
                continue

            #new_df_before = pd.concat(all_data_df, ignore_index=True)
            new_df_before = all_data_df.copy()
                
            try:
                #new_df = convert_response_to_dataframe(data_json)
                #new_df = add_random_columns(new_df_before) # To be replaced with actual energy data
                new_df = preprocess_data(new_df_before)
                #new_df['synthetic'] = new_df_before['synthetic']
                
                for col in ["baseline", "flexibility_above", "flexibility_below"]:
                    syn_col = f"synthetic_{col}"
                    if syn_col not in new_df.columns:
                        new_df[syn_col] = new_df_before[syn_col]
                
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

                print("Final column order:")
                print(combined_df.columns.tolist())

                print("Sample rows:")
                print(combined_df.tail(3)[['time', 'forecasted_sPMV', 'flexibility_above']])
                
                combined_df.to_csv(csv_path, index=False)
                print(f"Updated data saved in {csv_path}")
                
            except Exception as e:
                print(f"Error during saving {sensor_id}: {e}")

if __name__ == "__main__":
    download_history_data()
