import numpy as np
import requests
import time

def get_google_distance_matrices(coords, api_key):
    """
    Fetches distance and duration matrices from Google Distance Matrix API.
    Uses batching to handle more than 10 locations and avoid MAX_ELEMENTS_EXCEEDED.
    """
    n = len(coords)
    dist_matrix = np.zeros((n, n))
    time_matrix = np.zeros((n, n))
    
    # Google Classic API limits:
    # Max elements per request: 100 (for standard) or 625 (for premium).
    # We use a batch size of 10x10 to stay safely within the 100-element limit.
    batch_size = 10 

    print(f"Generating matrix for {n} nodes ({n*n} elements) via Google API...")

    for i in range(0, n, batch_size):
        for j in range(0, n, batch_size):
            # Define the current batch of origins and destinations
            origins_batch = coords[i : i + batch_size]
            destinations_batch = coords[j : j + batch_size]
            
            # Format coordinates for the URL: "lat,lng|lat,lng"
            origins_str = "|".join([f"{c[0]},{c[1]}" for c in origins_batch])
            dest_str = "|".join([f"{c[0]},{c[1]}" for c in destinations_batch])
            
            url = "https://maps.googleapis.com/maps/api/distancematrix/json"
            params = {
                "origins": origins_str,
                "destinations": dest_str,
                "key": api_key,
                "mode": "driving",
                "units": "metric"
            }
            
            try:
                response = requests.get(url, params=params)
                data = response.json()
                
                if data.get("status") != "OK":
                    error_msg = data.get("error_message", "Check if Distance Matrix API is enabled in Google Cloud Console.")
                    raise Exception(f"Google API Error: {data.get('status')} - {error_msg}")

                # Populate the master matrices with the results of this batch
                for row_idx, row in enumerate(data["rows"]):
                    for col_idx, element in enumerate(row["elements"]):
                        if element["status"] == "OK":
                            # Convert meters to kilometers
                            dist_matrix[i + row_idx][j + col_idx] = element["distance"]["value"] / 1000.0
                            # Convert seconds to minutes
                            time_matrix[i + row_idx][j + col_idx] = element["duration"]["value"] / 60.0
                        else:
                            # If no road is found, we set a high distance to penalize this route
                            dist_matrix[i + row_idx][j + col_idx] = 9999.0
                            time_matrix[i + row_idx][j + col_idx] = 9999.0
                
                # Small delay to respect rate limits (Queries Per Second)
                time.sleep(0.05)

            except Exception as e:
                print(f"Error during API batch processing: {e}")
                raise e

    return dist_matrix, time_matrix
