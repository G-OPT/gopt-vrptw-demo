import math
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import numpy as np

from google_distance import get_google_distance_matrices


# --------------------------------------------------------
# Helper: compute Euclidean distance in kilometers
# --------------------------------------------------------
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


# --------------------------------------------------------
# Main VRPTW Solver
# --------------------------------------------------------
def solve_vrp(
    coords,
    demands,
    vehicle_capacity,
    ready_times,
    due_times,
    service_times,
    num_vehicles,
    use_google=False,
    api_key=None,
):

    n = len(coords)
    unreachable_indices = []

    # --------------------------------------------------------
    # Build distance & time matrices
    # --------------------------------------------------------
    if use_google and api_key:
        print("Using Google Distance Matrix API...")
        distance_matrix_km, time_matrix_min = get_google_distance_matrices(
            coords, api_key
        )
        
        # SMART DETECTION: Identify indices that have the 9999 penalty
        for i in range(n):
            # If a location cannot reach the depot or be reached by the depot
            # we flag it as unreachable.
            if distance_matrix_km[0][i] >= 9000 or distance_matrix_km[i][0] >= 9000:
                unreachable_indices.append(i)
    else:
        print("Using Euclidean distances...")
        distance_matrix_km = [
            [haversine_distance(*coords[i], *coords[j]) for j in range(n)]
            for i in range(n)
        ]
        time_matrix_min = [[int(d * 60 / 35) for d in row] for row in distance_matrix_km]

    distance_matrix_m = [[int(d * 1000) for d in row] for row in distance_matrix_km]

    # --------------------------------------------------------
    # OR-Tools Routing Model
    # --------------------------------------------------------
    manager = pywrapcp.RoutingIndexManager(n, num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_cb(from_index, to_index):
        f = manager.IndexToNode(from_index)
        t = manager.IndexToNode(to_index)
        return distance_matrix_m[f][t]

    transit_cb_idx = routing.RegisterTransitCallback(distance_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb_idx)

    def demand_cb(from_index):
        f = manager.IndexToNode(from_index)
        return demands[f]

    demand_cb_idx = routing.RegisterUnaryTransitCallback(demand_cb)
    routing.AddDimensionWithVehicleCapacity(
        demand_cb_idx, 0, [vehicle_capacity] * num_vehicles, True, "Capacity"
    )

    def time_cb(from_index, to_index):
        f = manager.IndexToNode(from_index)
        t = manager.IndexToNode(to_index)
        return time_matrix_min[f][t] + service_times[f]

    time_cb_idx = routing.RegisterTransitCallback(time_cb)
    routing.AddDimension(time_cb_idx, 30_000, 30_000, False, "Time")
    time_dim = routing.GetDimensionOrDie("Time")

    for node in range(n):
        idx = manager.NodeToIndex(node)
        time_dim.CumulVar(idx).SetRange(ready_times[node], due_times[node])

    for v in range(num_vehicles):
        routing.Start(v)
        routing.End(v)
        time_dim.CumulVar(routing.Start(v)).SetRange(ready_times[0], due_times[0])
        time_dim.CumulVar(routing.End(v)).SetRange(ready_times[0], due_times[0])

    # --------------------------------------------------------
    # Search Parameters
    # --------------------------------------------------------
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_params.time_limit.seconds = 5

    try:
        solution = routing.SolveWithParameters(search_params)
    except Exception as e:
        return None, None, []

    if solution is None:
        return None, None, []

    # --------------------------------------------------------
    # Extract Routes
    # --------------------------------------------------------
    routes = []
    total_km = 0.0

    for v in range(num_vehicles):
        idx = routing.Start(v)
        route = []
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            route.append(node)
            prev_idx = idx
            idx = solution.Value(routing.NextVar(idx))
            next_node = manager.IndexToNode(idx)
            total_km += distance_matrix_km[node][next_node]
        route.append(0)
        routes.append(route)

    # Return the three required values
    return routes, total_km, list(set(unreachable_indices))


# --------------------------------------------------------
# DataFrame Converter
# --------------------------------------------------------
def routes_to_dataframe(df, routes):
    results = []
    for vid, route in enumerate(routes):
        for order, node in enumerate(route):
            results.append({
                "vehicle": vid,
                "stop_order": order,
                "name": df.iloc[node]["name"],
                "latitude": df.iloc[node]["latitude"],
                "longitude": df.iloc[node]["longitude"],
                "demand": df.iloc[node]["demand"],
                "ready_time": df.iloc[node]["ready_time"],
                "due_time": df.iloc[node]["due_time"],
                "service_time": df.iloc[node]["service_time"],
            })
    return __import__("pandas").DataFrame(results).sort_values(["vehicle", "stop_order"]).reset_index(drop=True)
