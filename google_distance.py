import numpy as np
import requests
import time

def get_google_distance_matrices(coords, api_key):
    n = len(coords)
    dist_matrix = np.zeros((n, n))
    time_matrix = np.zeros((n, n))
    
    # Google's limit for the classic API is 100 elements per request
    # We use a batch size of 10 to be safe (10x10 = 100)
    batch_size = 10 

    for i in range(0, n, batch_size):
        for j in range(0, n, batch_size):
            origins = coords[i : i + batch_size]
            destinations = coords[j : j + batch_size]
            
            origins_str = "|".join([f"{c[0]},{c[1]}" for c in origins])
            dest_str = "|".join([f"{c[0]},{c[1]}" for c in destinations])
            
            url = "https://maps.googleapis.com/maps/api/distancematrix/json"
            params = {
                "origins": origins_str,
                "destinations": dest_str,
                "key": api_key,
                "mode": "driving"
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if data.get("status") != "OK":
                raise Exception(f"Google API Error: {data.get('status')} - {data.get('error_message', 'Check Billing/Enablement')}")

            for row_idx, row in enumerate(data["rows"]):
                for col_idx, element in enumerate(row["elements"]):
                    if element["status"] == "OK":
                        dist_matrix[i + row_idx][j + col_idx] = element["distance"]["value"] / 1000.0
                        time_matrix[i + row_idx][j + col_idx] = element["duration"]["value"] / 60.0
            
            # Short sleep to avoid hitting per-second rate limits
            time.sleep(0.1)

    return dist_matrix, time_matrix
