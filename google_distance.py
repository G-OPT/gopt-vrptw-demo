import numpy as np
import requests
import json
from math import radians, cos, sin, asin, sqrt

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance in km between two points with a Finnish circuitry factor."""
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
    
    # --- OPTION 2: FALLBACK FOR LARGE DATASETS ---
    if n > 25:
        print(f"Large dataset detected ({n} nodes). Falling back to Haversine logic.")
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

    # --- OPTION 1: GOOGLE ROUTES API FOR SMALL DATASETS ---
    print(f"Small dataset detected ({n} nodes). Calling Google Routes API...")
    url = "https://routes.googleapis.com/distanceMatrix/v1:computeRouteMatrix"
    
    waypoint_list = []
    for c in coords:
        waypoint_list.append({
            "waypoint": {
                "location": {
                    "latLng": {"latitude": c[0], "longitude": c[1]}
                }
            }
        })

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "originIndex,destinationIndex,distanceMeters,duration"
    }

    body = {
        "origins": waypoint_list,
        "destinations": waypoint_list,
        "travelMode": "DRIVE",
        "routingPreference": "TRAFFIC_UNAWARE"
    }

    response = requests.post(url, headers=headers, data=json.dumps(body))

    if response.status_code != 200:
        raise Exception(f"Google Routes API Error {response.status_code}: {response.text}")

    data = response.json()
    dist_matrix = np.zeros((n, n))
    time_matrix = np.zeros((n, n))

    for entry in data:
        i = entry.get("originIndex", 0)
        j = entry.get("destinationIndex", 0)
        
        # Convert meters to km
        meters = entry.get("distanceMeters", 0)
        dist_matrix[i][j] = meters / 1000.0
        
        # Convert '123s' string to float minutes
        duration_str = entry.get("duration", "0s")
        seconds = float(duration_str.replace("s", ""))
        time_matrix[i][j] = seconds / 60.0

    return dist_matrix, time_matrix
