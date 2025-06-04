import pandas as pd
import numpy as np
import os
import logging
import json
from datetime import datetime, timedelta, timezone
from fastapi.middleware.cors import CORSMiddleware

from scripts.control_dbf_apis import get_token, get_device_data_grouped
from scripts.data_preprocessing import preprocess_data
from scripts.forecast_spmv import forecast_and_update_df, prepare_features, generate_forecasts
from scripts.pareto_optimization_runner import run_pareto_optimization
from scripts.flexibility_heating_service_apis import get_flexibility_heating, EnergyEntry
import tensorflow as tf
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, Form, Response
from fastapi.security import OAuth2PasswordRequestForm
from models import create_user, authenticate_user, is_admin
from auth import create_access_token, verify_token

from typing import Optional
from datetime import timedelta

load_dotenv()  # take environment variables from .env

BASE_DIR = os.getenv('DATA_DIR')
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'final_model.h5')
USERNAME = os.getenv('DOMX_USERNAME')
PASSWORD = os.getenv('DOMX_PASSWORD')
APP_FE_URL = os.getenv('APP_FE_URL')
DEFAULT_START_DAYS_AGO = 15

logging.basicConfig(
    level=logging.INFO,  # Livello minimo di registrazione
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Formato del messaggio di log
    datefmt='%Y-%m-%d %H:%M:%S'  # Formato della data e dell'ora
)

app = FastAPI()

MODEL  = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"mse": tf.keras.metrics.MeanSquaredError()}
)

app = FastAPI()

origins = [APP_FE_URL]

app.add_middleware(
     CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
        if "time" not in df.columns:
            continue
        merged_df = pd.merge(merged_df, df, on="time")

    if "time" in merged_df.columns:
        merged_df["time"] = pd.to_datetime(merged_df["time"])
    
    return merged_df

def add_random_columns(df):
    np.random.seed(42)
    df["baseline"] = np.random.uniform(0, 361, size=len(df))
    df["flexibility_below"] = np.random.uniform(0, 253, size=len(df))
    return df

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

def fill_missing_flexibility_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Riempi NaN in baseline, flexibility_above e flexibility_below
    con medie giornaliere/orarie e aggiungi flag synthetic_*.
    """
    df = df.copy()
    target_cols = ["baseline", "flexibility_above", "flexibility_below"]
    
    # Aggiunge colonne temporanee per il raggruppamento
    df["hour"] = df["time"].dt.hour
    df["dayofweek"] = df["time"].dt.dayofweek

    for col in target_cols:
        synthetic_col = f"synthetic_{col}"
        df[synthetic_col] = 0  # default a reale

        if df[col].isna().any():
            # Calcola medie per giorno della settimana e ora
            mean_values = (
                df.groupby(["dayofweek", "hour"])[col]
                .mean()
                .reset_index()
                .rename(columns={col: f"{col}_mean"})
            )

            # Merge per assegnare media corretta a ciascuna riga
            df = df.merge(mean_values, on=["dayofweek", "hour"], how="left")

            # Riempie valori mancanti
            na_mask = df[col].isna()
            df.loc[na_mask, col] = df.loc[na_mask, f"{col}_mean"]
            df.loc[na_mask, synthetic_col] = 1

            # Rimuove colonna temporanea
            df.drop(columns=[f"{col}_mean"], inplace=True)

    # Rimuove colonne ausiliarie
    df.drop(columns=["hour", "dayofweek"], inplace=True, errors="ignore")
    return df

def fill_missing_raw_data(data_json, energy_json, date_range, historical_df):
    full_range = pd.date_range(start=date_range[0], end=date_range[1], freq="1H", tz="UTC")

    # Prepara DataFrame anche se vuoti
    t_r_df = pd.DataFrame(data_json.get("t_r", []))
    rh_r_df = pd.DataFrame(data_json.get("rh_r", []))
    energy_df = pd.DataFrame(energy_json.get("energy_a", []))

    # Se mancano, crea colonna 'time' vuota per evitare KeyError
    for df in [t_r_df, rh_r_df, energy_df]:
        if "time" not in df.columns:
            df["time"] = pd.to_datetime([])

    # Converti le date
    for df in [t_r_df, rh_r_df, energy_df]:
        df["time"] = pd.to_datetime(df["time"])

    # Prepara medie orarie dallo storico
    if not historical_df.empty:
        historical_df["hour"] = historical_df["time"].dt.hour
        avg_by_hour = historical_df.groupby("hour")[["t_r", "rh_r", "energy_average"]].mean()
    else:
        avg_by_hour = pd.DataFrame(columns=["t_r", "rh_r", "energy_average"])

    synthetic_flags = {
        "synthetic_t_r": [],
        "synthetic_rh_r": [],
        "synthetic_energy": []
    }

    completed_t_r, completed_rh_r, completed_energy = [], [], []

    for t in full_range:
        hour = t.hour

        # === t_r ===
        if t in t_r_df["time"].values:
            val = t_r_df[t_r_df["time"] == t]["value"].values[0]
            completed_t_r.append({"time": t, "value": val})
            synthetic_flags["synthetic_t_r"].append(0)
        else:
            synthetic_val = avg_by_hour.at[hour, "t_r"] if hour in avg_by_hour.index else None
            completed_t_r.append({"time": t, "value": synthetic_val})
            synthetic_flags["synthetic_t_r"].append(1)

        # === rh_r ===
        if t in rh_r_df["time"].values:
            val = rh_r_df[rh_r_df["time"] == t]["value"].values[0]
            completed_rh_r.append({"time": t, "value": val})
            synthetic_flags["synthetic_rh_r"].append(0)
        else:
            synthetic_val = avg_by_hour.at[hour, "rh_r"] if hour in avg_by_hour.index else None
            completed_rh_r.append({"time": t, "value": synthetic_val})
            synthetic_flags["synthetic_rh_r"].append(1)

        # === energy ===
        if t in energy_df["time"].values:
            val = energy_df[energy_df["time"] == t]["value"].values[0]
            completed_energy.append({"time": t, "value": val})
            synthetic_flags["synthetic_energy"].append(0)
        else:
            synthetic_val = avg_by_hour.at[hour, "energy_average"] if hour in avg_by_hour.index else None
            completed_energy.append({"time": t, "value": synthetic_val})
            synthetic_flags["synthetic_energy"].append(1)

    new_data_json = {
        "t_r": completed_t_r,
        "rh_r": completed_rh_r
    }
    new_energy_json = {
        "energy_a": completed_energy
    }

    synthetic_df = pd.DataFrame({
        "time": full_range,
        **synthetic_flags
    })

    return new_data_json, new_energy_json, synthetic_df

                 
def update_datasets():
    token = get_token(USERNAME, PASSWORD)
    
    if not token:
        logging.error("Token retrieval failed.")
        return
    
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday = today - timedelta(days=1)

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

            if os.path.exists(csv_path):
                historical_df = pd.read_csv(csv_path, parse_dates=["time"])
                historical_df.sort_values("time", inplace=True)
                
                # Riempi i NaN nelle colonne sintetiche con 0
                synthetic_cols = [
                    "synthetic_baseline", "synthetic_flexibility_above", "synthetic_flexibility_below",
                    "synthetic_t_r", "synthetic_rh_r", "synthetic_energy"
                ]
                for col in synthetic_cols:
                    if col in historical_df.columns:
                        historical_df[col].fillna(0, inplace=True)
                
                last_date = historical_df["time"].max().normalize()
                date_from = last_date + timedelta(days=1)
            else:
                historical_df = pd.DataFrame()
                date_from = today - timedelta(days=DEFAULT_START_DAYS_AGO)
            
            date_to = yesterday
            
            # check empty request
            if date_from > date_to:
                logging.info(f"No new data to fetch for {sensor_id} ({building} - {exx})")
                continue

            # download new data
            logging.info(f"[{building} - {exx}] download from {iso_date(date_from)} to {iso_date(today)}")
            
            data_json = get_device_data_grouped(
                token, sensor_id,
                iso_date(date_from), iso_date(date_to + timedelta(days=1)),
                metrics=["t_r", "rh_r"], interval="1h"
            )
            
            energy_json = get_device_data_grouped(
                token, smart_meter_id,
                iso_date(date_from), iso_date(date_to + timedelta(days=1)),
                metrics=["energy_a"], interval="1h"
            )
            
            # Riempie i buchi nei dati grezzi usando medie orarie
            full_range_end = datetime.combine(date_to, datetime.max.time()).replace(tzinfo=timezone.utc)
            data_json, energy_json, synthetic_flags_df = fill_missing_raw_data(
                data_json, energy_json, (date_from, full_range_end), historical_df
            )

            temp_df = convert_response_to_dataframe(data_json)
            
            # Aggiunge energy
            energy_df = pd.DataFrame(energy_json["energy_a"])
            energy_df.rename(columns={"value": "energy_average"}, inplace=True)
            energy_df["time"] = pd.to_datetime(energy_df["time"])

            # Merge con temperatura e umidità
            new_df = pd.merge(temp_df, energy_df, on="time", how="left")

            # Merge con i flag sintetici
            new_df = pd.merge(new_df, synthetic_flags_df, on="time", how="left")
            
            # === Calcolo Flexibility per blocco intero ===
            window_start = date_from - timedelta(days=31)
            window_end = date_to + timedelta(days=1)

            history_window_df = historical_df[
                (historical_df["time"] >= window_start) & (historical_df["time"] < date_from)
            ] if not historical_df.empty else pd.DataFrame()
            
            api_input_df = pd.concat([history_window_df, new_df], ignore_index=True)
            
            energy_entries = [
                EnergyEntry(time=row["time"].to_pydatetime(), energy_average=row["energy_average"])
                for _, row in api_input_df.iterrows()
                if not pd.isna(row["energy_average"])
            ]
            
            hours_today = pd.date_range(start=today, periods=24, freq='H', tz='UTC')
            energy_entries += [
                EnergyEntry(time=dt.to_pydatetime(), energy_average=0.0)
                for dt in hours_today
            ]
            
            flexibility_results = []
            
            if energy_entries:
                flexibility_results = get_flexibility_heating(energy_entries)
                
                if isinstance(flexibility_results, list) and flexibility_results:
                    flex_df = pd.DataFrame(flexibility_results)
                    flex_df["time"] = pd.to_datetime(flex_df["time"])

                    today_mask = flex_df["time"].dt.normalize() == today
                    today_df = flex_df[today_mask].copy()

                    # Salva i risultati di oggi
                    if not today_df.empty:
                        flexibility_path = os.path.join(folder_path, "flexibility_heating.csv")
                        today_df.to_csv(flexibility_path, index=False)
                        logging.info(f"Saved flexibility results for today in {flexibility_path}")
                    else:
                        logging.warning(f"No flexibility results for today to save in {building} - {exx}")

                    # Rimuovi i risultati di oggi dal dataset da usare per update_all_data_with_results
                    flexibility_results = flex_df[~today_mask].to_dict(orient="records")
                else:
                    logging.warning(f"No valid flexibility results to process for {building} - {exx}")
                
                update_all_data_with_results(new_df, flexibility_results)
            
            new_df = fill_missing_flexibility_values(new_df)
            
            try:
                #new_df = convert_response_to_dataframe(data_json)
                #new_df = add_random_columns(new_df) # To be replaced with actual energy data
                new_df = preprocess_data(new_df)
                
                pre_forecast_df = new_df.copy()
                
                X = prepare_features(new_df[-24:])
                forecasts = generate_forecasts(MODEL, X)
                forecast_df = pd.DataFrame(forecasts, columns=["forecasted_sPMV"])
                forecast_dates = pd.date_range(start=today, periods=24, freq='H', tz='UTC')
                forecast_df['time'] = forecast_dates
                forecast_df = forecast_df[['time', 'forecasted_sPMV']]
                forecast_df.to_csv(os.path.join( folder_path, "forecasted_sPMV.csv"), index=False)
                
                forecasted_df  = forecast_and_update_df(pre_forecast_df, MODEL)
            
                combined_df = pd.concat([historical_df, forecasted_df], ignore_index=True)
                combined_df.drop_duplicates(subset=["time"], inplace=True)
                combined_df.sort_values(by="time", inplace=True)
                combined_df.to_csv(csv_path, index=False)
                
                logging.info(f"Updated data saved in {csv_path}")
            except Exception as e:
                logging.error(f"Error during saving {sensor_id}: {e}")


def optimize_all():
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    
    for building, apartment in config.items():
        for exx, sensors in apartment.items():
            folder_path = os.path.join(DATASETS_DIR, building, exx)
            csv_path = os.path.join(folder_path, f"{exx}_dataset.csv")
            
            if not os.path.exists(csv_path):
                logging.error(f"File not found for building '{building}' and apartment '{exx}'.")
                continue
            
            run_pareto_optimization(csv_path, folder_path)

# Endpoint: login (token)
@app.post("/token")
def login( response: Response, form_data: OAuth2PasswordRequestForm = Depends()):
    if not authenticate_user(form_data.username, form_data.password):
        raise HTTPException(status_code=400, detail="Invalid username or password")
    
    access_token_expires = timedelta(minutes=60)
    access_token = create_access_token(data={"sub": form_data.username}, expires_delta=access_token_expires)

    # return {"access_token": access_token, "token_type": "bearer"}
    response.set_cookie(
        key="token",
         value= access_token,
          httponly=True, 
          secure=False,  # use True in production with HTTP,
          samesite="lax",
          max_age=60*15, # 15 minutes
          path="/"
    )
    return {"message": "Login  successful"}

# Endpoint: create new user (only admin)
@app.post("/admin/create_user")
def api_create_user(
    username: str = Form(...),
    password: str = Form(...),
    is_admin_flag: Optional[bool] = Form(False),
    current_user: str = Depends(verify_token)
):
    if not is_admin(current_user):
        raise HTTPException(status_code=403, detail="Admin privileges required")
    
    try:
        create_user(username, password, is_admin=is_admin_flag)
        return {"message": f"User '{username}' created."}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/validate")
def validate_user(current_user: str = Depends(verify_token)):
    return {"user": current_user}

@app.post("/logout")
def logout(response:Response):
    response.delete_cookie("token")
    return {"message": "logged out"}

@app.get("/forecast_sPMV")
def forecast_sPMV(current_user: str = Depends(verify_token)):
    if not is_admin(current_user):
        raise HTTPException(status_code=403, detail="Admin privileges required")
    update_datasets()
    return {"message": "Forecasting and optimization completed!"}

@app.get("/run_optimization")
def run_optimization(current_user: str = Depends(verify_token)):
    if not is_admin(current_user):
        raise HTTPException(status_code=403, detail="Admin privileges required")
    optimize_all()
    return {"message": "Optimization completed!"}

@app.get("/sPMV/{building}/{exx}")
def GET_forecasted_sPMV(current_user: str = Depends(verify_token), building: str = "CASA MADDALENA", exx: str = "E144"):
    folder_path = os.path.join(DATASETS_DIR, building, exx)
    csv_path = os.path.join(folder_path, f"forecasted_sPMV.csv")
    
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail=f"File not found for building '{building}' and apartment '{exx}'.")

    try:
        df = pd.read_csv(csv_path)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading CSV file: {str(e)}")

@app.get("/optimization/{building}/{exx}")
def GET_optimization(current_user: str = Depends(verify_token), building: str = "CASA MADDALENA", exx: str = "E144"):
    
    folder_path = os.path.join(DATASETS_DIR, building, exx, "optimization_results")
    csv_path = os.path.join(folder_path, f"solutions_summary.csv")
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail=f"File not found for building '{building}' and apartment '{exx}'.")

    try:
        df = pd.read_csv(csv_path)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading CSV file: {str(e)}")

@app.get("/energy/{building}/{exx}")
def GET_energy(current_user: str = Depends(verify_token), building: str = "CASA MADDALENA", exx: str = "E144"):
    
    date_from = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    date_end = date_from + timedelta(days=1)
    
    token = get_token(USERNAME, PASSWORD)
    
    if not token:
        logging.error("Token retrieval failed.")
        return
    
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    
    for building_name, apartment in config.items():
        if building_name == building:
            for exx_code, sensors in apartment.items():
                if(exx_code == exx):
                    sensor_id = sensors.get("smart_meters")
    
    data_json = get_device_data_grouped(
        token,
        sensor_id,
        iso_date(date_from),
        iso_date(date_end),
        metrics=["energy_a", "power_a"],
        interval="1h"
    )
    
    return data_json

@app.get("/flexibility_heating/{building}/{exx}")
def GET_forecasted_sPMV(current_user: str = Depends(verify_token), building: str = "CASA MADDALENA", exx: str = "E144"):
    folder_path = os.path.join(DATASETS_DIR, building, exx)
    csv_path = os.path.join(folder_path, f"flexibility_heating.csv")
    
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail=f"File not found for building '{building}' and apartment '{exx}'.")

    try:
        df = pd.read_csv(csv_path)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading CSV file: {str(e)}")