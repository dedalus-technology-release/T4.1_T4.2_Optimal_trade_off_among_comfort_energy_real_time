import os
import requests
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List
import json
from dotenv import load_dotenv

load_dotenv()

FLEXIBILITY_HEATING_URL = os.getenv('FLEXIBILITY_HEATING_URL')

class EnergyEntry(BaseModel):
    time: datetime
    energy_average: float

class FlexibilityEntry(EnergyEntry):
    baseline: Optional[float] = None
    flexibility_above: Optional[float] = None
    flexibility_below: Optional[float] = None
    
def get_flexibility_heating(data: List[EnergyEntry]):

    headers = {
        'Content-Type': 'application/json'
    }
    
    json_data = json.dumps([
        {
            "time": entry.time.isoformat(),
            "energy_average": entry.energy_average
        } for entry in data
    ])
    
    #if not os.path.isfile("./last_flexibility_input.json"):
    #    with open("./last_flexibility_input.json", "w") as f:
    #        json.dump(json_data, f, indent=2)
    
    #response = requests.request("POST", FLEXIBILITY_HEATING_URL + "/flexibility/calculate", headers=headers, data=json_data)
    response = requests.post("http://localhost:8555/flexibility/calculate", headers=headers, data=json_data)
    
    return response.json()