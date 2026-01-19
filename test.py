import requests
import json

# =============================
# API endpoint
# =============================
url = "http://127.0.0.1:9696/predict"

# =============================
# Sample patient JSON (matches FEATURES)
# =============================
patient_data = {
    "Gender": 1,
    "Age": 28,
    "Height": 172,
    "Weight": 75,
    "family_history_with_overweight": 1,
    "FAVC": 0,
    "FCVC": 2,
    "NCP": 3,
    "CAEC": 1,
    "SMOKE": 0,
    "CH2O": 2,
    "SCC": 1,
    "FAF": 1,
    "TUE": 2,
    "CALC": 0,
    "MTRANS": 3
}

# =============================
# Send POST request
# =============================
response = requests.post(url, json=patient_data)

# =============================
# Print results
# =============================
if response.status_code == 200:
    print("Response JSON:")
    print(json.dumps(response.json(), indent=4))
else:
    print(f"Error {response.status_code}: {response.text}")
