import requests
import json
import polyline

def get_route_polyline(origin, destination, api_key):
    """
    Calls Google Routes API (ComputeRoutes) to get the REAL road path.
    Returns: list of (lat, lon) coordinates forming the full road path.
    """
    url = "https://routes.googleapis.com/directions/v2:computeRoutes"

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "routes.polyline.encodedPolyline"
    }

    body = {
        "origin": {
            "location": {
                "latLng": {
                    "latitude": origin[0],
                    "longitude": origin[1]
                }
            }
        },
        "destination": {
            "location": {
                "latLng": {
                    "latitude": destination[0],
                    "longitude": destination[1]
                }
            }
        },
        "travelMode": "DRIVE",
        "routingPreference": "TRAFFIC_AWARE"
    }

    r = requests.post(url, headers=headers, data=json.dumps(body))

    if r.status_code != 200:
        print("Google ComputeRoutes Error:", r.text)
        return None

    data = r.json()

    if "routes" not in data or len(data["routes"]) == 0:
        return None

    encoded = data["routes"][0]["polyline"]["encodedPolyline"]

    # Decode polyline into (lat, lon)
    coords = polyline.decode(encoded)

    return coords
