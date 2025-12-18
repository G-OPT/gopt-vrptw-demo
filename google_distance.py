import numpy as np
import requests
import json
from math import radians, cos, sin, asin, sqrt

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance in km between two points."""
    CIRCUITRY_FACTOR = 1.25 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 
    return c * r * CIRCUITRY_FACTOR

def get_google_distance_matrices(coords, api_key):
    n = len(coords)
    
    # --- IF LARGE DATASET: USE HAVERSINE (FREE & FAST) ---
    if n > 25:
        print(f"Large dataset ({n} nodes). Using Haversine.")
        dist_matrix = np.zeros((n, n))
        time_matrix = np.zeros((n, n))
        AVG_SPEED_KMPH = 60 

        for i in range(n):
            for j in range(n):
                if i == j:
                    dist_matrix[i][j] = 0
                    time_matrix[i][j] = 0
                else:
                    d = haversine_distance(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
                    dist_matrix[i][j] = d
                    time_matrix[i][j] = (d / AVG_SPEED_KMPH) * 60
        return dist_matrix, time_matrix

    # --- IF SMALL DATASET: USE CLASSIC GOOGLE DISTANCE MATRIX ---
    print(f"Small dataset ({n} nodes). Calling Classic Google API...")
    
    # Using the classic, highly-compatible endpoint
    url = "https://maps.googleapis.com/maps/api/distancematrix/json"
    
    # Format coordinates as "lat,lng|lat,lng"
    locations_str = "|".join([f"{c[0]},{c[1]}" for c in coords])
    
    params = {
        "origins": locations_str,
        "destinations": locations_str,
        "key": api_key,
        "mode": "driving"
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        raise Exception(f"Google API Error {response.status_code}: {response.text}")

    data = response.json()
    
    if data.get("status") != "OK":
        raise Exception(f"Google API Logic Error: {data.get('status')} - {data.get('error_message', 'No message')}")

    dist_matrix = np.zeros((n, n))
    time_matrix = np.zeros((n, n))

    for i, row in enumerate(data["rows"]):
        for j, element in enumerate(row["elements"]):
            if element["status"] == "OK":
                # distance is in meters, convert to km
                dist_matrix[i][j] = element["distance"]["value"] / 1000.0
                # duration is in seconds, convert to minutes
                time_matrix[i][j] = element["duration"]["value"] / 60.0
            else:
                # Fallback to Haversine if Google can't find a road
                d = haversine_distance(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
                dist_matrix[i][j] = d
                time_matrix[i][j] = (d / 60) * 60

    return dist_matrix, time_matrix
