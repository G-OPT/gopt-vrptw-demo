import requests
import json
import streamlit as st

def get_google_distance_matrices(coords, api_key):
    """
    Uses the NEW Google Routes API - ComputeRouteMatrix
    Returns:
        distance_matrix_km  (NxN)
        time_matrix_min     (NxN)
    """

    origins = []
    destinations = []

    # Build origins/destinations in proper Schema
    for (lat, lon) in coords:
        waypoint = {
            "waypoint": {
                "location": {
                    "latLng": {
                        "latitude": lat,
                        "longitude": lon
                    }
                }
            }
        }
        origins.append(waypoint)
        destinations.append(waypoint)

    url = "https://routes.googleapis.com/distanceMatrix/v2:computeRouteMatrix"

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        # Only return fields we need
        "X-Goog-FieldMask": "originIndex,destinationIndex,duration,distanceMeters"
    }

    body = {
        "origins": origins,
        "destinations": destinations,
        "travelMode": "DRIVE",
        "routingPreference": "TRAFFIC_AWARE"
    }

    response = requests.post(url, headers=headers, data=json.dumps(body))
    
    if response.status_code != 200:
        raise Exception(f"Google Routes API Error {response.status_code}: {response.text}")


    if response.status_code != 200:
        raise Exception(f"Google Routes API Error {response.status_code}: {response.text}")

    data = response.json()

    n = len(coords)
    distance_matrix_km = [[0]*n for _ in range(n)]
    time_matrix_min = [[0]*n for _ in range(n)]

    # Parse results
    for element in data:
        i = element["originIndex"]
        j = element["destinationIndex"]

        dist = element.get("distanceMeters", None)
        dur = element.get("duration", None)

        if dist is None or dur is None:
            distance_matrix_km[i][j] = 9999
            time_matrix_min[i][j] = 9999
        else:
            distance_matrix_km[i][j] = dist / 1000.0
            # duration is string like "120s"
            seconds = float(dur.replace("s", ""))
            time_matrix_min[i][j] = seconds / 60.0

    return distance_matrix_km, time_matrix_min
