import requests
import json

# URL of the Flask server running locally
url = "http://127.0.0.1:5000/predict"

# Your input data (features) to send to the server
# New test data to send
new_data = {
    "features": [0, 90, 60, 20, 85, 25, 0.3, 25]  # Replace with your own test data
}

response = requests.post(url, headers={'Content-Type': 'application/json'}, data=json.dumps(new_data))

if response.status_code == 200:
    print("New Prediction:", response.json())
else:
    print(f"Failed to get response from server. Status code: {response.status_code}")
