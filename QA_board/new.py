import requests
import json

BASE_URL = 'http://164.52.200.229:9486'

def test_get_interactions():
    url = f"{BASE_URL}/interactions"
    response = requests.get(url)
    print(f"GET {url} - Status Code: {response.status_code}")
    
    try:
        response_data = response.json()
        print(json.dumps(response_data, indent=2))
    except json.JSONDecodeError:
        print("Failed to decode JSON response.")
        print(response.text)
    
    if response.status_code != 200:
        print("Error:", response.text)
    elif 'total_calls' in response_data and response_data['total_calls'] == 0:
        print("Warning: The total_calls is 0, but the database should have data.")
       

if __name__ == "__main__":
    test_get_interactions()
