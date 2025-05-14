import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta, timezone
from fastapi.middleware.cors import CORSMiddleware

from scripts.Calculate_sPMV_v1 import sPMV_calculation
from scripts.augment_dataset import augment_dataset
from scripts.control_dbf_apis import get_token, get_device_data_grouped
from scripts.data_preprocessing import preprocess_data
from scripts.forecast_spmv import forecast_and_update_df, prepare_features, generate_forecasts
from scripts.pareto_optimization_runner import run_pareto_optimization
import tensorflow as tf
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, status, Form, Response
from fastapi.security import OAuth2PasswordRequestForm
from models import create_user, authenticate_user, is_admin
from auth import create_access_token, verify_token
from database import init_db

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
DEFAULT_START_DAYS_AGO = 7

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

def optimize_all():
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    
    for building, apartment in config.items():
        for exx, sensors in apartment.items():
            folder_path = os.path.join(DATASETS_DIR, building, exx)
            csv_path = os.path.join(folder_path, f"{exx}_dataset.csv")
            
            if not os.path.exists(csv_path):
                print(f"File not found for building '{building}' and apartment '{exx}'.")
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
    return {"message": "Forecasting completed!"}

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

#if __name__ == "__main__":
#    update_datasets()
