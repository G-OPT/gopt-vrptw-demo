import numpy as np
from math import radians, cos, sin, asin, sqrt

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance in km between two points."""
    # Finland-specific circuitry factor (accounts for roads not being straight)
    CIRCUITRY_FACTOR = 1.25 
    
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers.
    
    return c * r * CIRCUITRY_FACTOR

def get_google_distance_matrices(coords, api_key):
    n = len(coords)
    
    # CHECK: If locations > 25, use Haversine to avoid API limits and high costs
    if n > 25:
        print(f"Large dataset detected ({n} nodes). Falling back to Haversine logic.")
        dist_matrix = np.zeros((n, n))
        time_matrix = np.zeros((n, n))
        
        # Average speed in Finland (combination of urban and highway)
        AVG_SPEED_KMPH = 60 

        for i in range(n):
            for j in range(n):
                if i == j:
                    dist_matrix[i][j] = 0
                    time_matrix[i][j] = 0
                else:
                    d = haversine_distance(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
                    dist_matrix[i][j] = d
                    # Convert distance to minutes
                    time_matrix[i][j] = (d / AVG_SPEED_KMPH) * 60
        
        return dist_matrix, time_matrix

    # ELSE: Use your existing Google API logic for small, high-precision requests
    # [Keep your existing Google request code here...]
