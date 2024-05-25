import requests
import json

BASE_URL = 'http://164.52.200.229:9486'

def test_get_interactions():
    url = f"{BASE_URL}/interactions"
    response = requests.get(url)
    print(f"GET {url} - Status Code: {response.status_code}")
    if response.status_code == 200:
        interactions = response.json()
        print(json.dumps(interactions, indent=2))
    else:
        print(response.text)

def test_get_trends():
    url = f"{BASE_URL}/trends?period=week"
    response = requests.get(url)
    print(f"GET {url} - Status Code: {response.status_code}")
    if response.status_code == 200:
        trends = response.json()
        print(json.dumps(trends, indent=2))
    else:
        print(response.text)

def test_get_client_interactions(client_id):
    url = f"{BASE_URL}/interactions/client?client_id={client_id}"
    response = requests.get(url)
    print(f"GET {url} - Status Code: {response.status_code}")
    if response.status_code == 200:
        interactions = response.json()
        print(json.dumps(interactions, indent=2))
    else:
        print(response.text)

if __name__ == "__main__":
    # Test the /interactions endpoint
    test_get_interactions()

    # Test the /trends endpoint
    # test_get_trends()

    # Replace 'your-client-id' with an actual client ID from your database to test
    # test_get_client_interactions('your-client-id')
