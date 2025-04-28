import requests
   
def get_token(username, password):
    
    # Endpoint token
    token_url = "https://sso.domx.online/auth/realms/gastools/protocol/openid-connect/token"

    # Client ID e secret
    client_id = "gas-tools-control-externalapi-app"
    client_secret = ""  # se vuoto, lascia stringa vuota

    # Richiesta del token
    data = {
        "grant_type": "password",
        "client_id": client_id,
        "client_secret": client_secret,
        "username": username,
        "password": password,
        "scope": "openid"
    }

    response = requests.post(token_url, data=data)

    if response.ok:
        token = response.json()['access_token']
        return token
    else:
        print("Errore:", response.status_code, response.text)
        return None
    
def get_device_data_grouped(token, sensor_name, date_from, date_to, metrics = ["t_r", "rh_r"], interval = "1h"):
    
    metrics_query = "&" + "&".join([f"metric[]={m}" for m in metrics])
    url = "https://api.control.business.domx.online/api/v1/devices/" + sensor_name + "/export/data/grouped?time_from=" + date_from + "&time_to=" + date_to + metrics_query + "&interval=" + interval

    print(url)
    
    headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {token}'
    }

    response = requests.request("GET", url, headers=headers)

    # print(response.text)

    return response.json()