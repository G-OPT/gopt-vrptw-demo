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

    # --------------------------------------------------------
    # Build distance & time matrices
    # --------------------------------------------------------
    if use_google and api_key:
        print("Using Google Distance Matrix API...")
        distance_matrix_km, time_matrix_min = get_google_distance_matrices(
            coords, api_key
        )
    else:
        print("Using Euclidean distances...")
        distance_matrix_km = [
            [haversine_distance(*coords[i], *coords[j]) for j in range(n)]
            for i in range(n)
        ]
        # assume ~35 km/h average
        time_matrix_min = [[int(d * 60 / 35) for d in row] for row in distance_matrix_km]

    # Convert km matrix to meters for OR-Tools (optional)
    distance_matrix_m = [[int(d * 1000) for d in row] for row in distance_matrix_km]

    # --------------------------------------------------------
    # OR-Tools Routing Model
    # --------------------------------------------------------
    manager = pywrapcp.RoutingIndexManager(n, num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    # --------------------------------------------------------
    # Distance Callback
    # --------------------------------------------------------
    def distance_cb(from_index, to_index):
        f = manager.IndexToNode(from_index)
        t = manager.IndexToNode(to_index)
        return distance_matrix_m[f][t]

    transit_cb_idx = routing.RegisterTransitCallback(distance_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb_idx)

    # --------------------------------------------------------
    # Capacity Constraint
    # --------------------------------------------------------
    def demand_cb(from_index):
        f = manager.IndexToNode(from_index)
        return demands[f]

    demand_cb_idx = routing.RegisterUnaryTransitCallback(demand_cb)

    routing.AddDimensionWithVehicleCapacity(
        demand_cb_idx,
        0,                     # no slack
        [vehicle_capacity] * num_vehicles,
        True,
        "Capacity",
    )

    # --------------------------------------------------------
    # Time Window Constraint (Correct VRPTW Model)
    # --------------------------------------------------------
    def time_cb(from_index, to_index):
        f = manager.IndexToNode(from_index)
        t = manager.IndexToNode(to_index)
        travel = time_matrix_min[f][t]
        service = service_times[f]
        return travel + service

    time_cb_idx = routing.RegisterTransitCallback(time_cb)

    routing.AddDimension(
        time_cb_idx,
        30_000,                # large slack allowed
        30_000,                # max travel time
        False,
        "Time",
    )

    time_dim = routing.GetDimensionOrDie("Time")

    # Set time windows: CumulVar(node) in [ready_time, due_time]
    for node in range(n):
        idx = manager.NodeToIndex(node)
        time_dim.CumulVar(idx).SetRange(ready_times[node], due_times[node])

    # Depot must also have time window
    for v in range(num_vehicles):
        start = routing.Start(v)
        end = routing.End(v)
        time_dim.CumulVar(start).SetRange(ready_times[0], due_times[0])
        time_dim.CumulVar(end).SetRange(ready_times[0], due_times[0])

    # --------------------------------------------------------
    # Search Parameters
    # --------------------------------------------------------
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_params.time_limit.seconds = 5
    search_params.log_search = False

    # --------------------------------------------------------
    # Solve
    # --------------------------------------------------------
    #solution = routing.SolveWithParameters(search_params)
    try:
        solution = routing.SolveWithParameters(search_params)
    except Exception as e:
        import streamlit as st
        st.error("❌ OR-Tools crashed inside SolveWithParameters:")
        st.code(str(e))
        return None, None

    if solution is None:
        import streamlit as st
        st.error("❌ No feasible solution found. The solver returned None.")
        st.write("Possible reasons:")
        st.write("- Time windows impossible")
        st.write("- Capacity too small")
        st.write("- Too few vehicles")
        st.write("- Incorrect distance matrix (Google API issue)")
        st.write("- Coordinates too far / unreachable")
        return None, None


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

            # accumulate km
            total_km += distance_matrix_km[node][next_node]

        route.append(0)   # return to depot
        routes.append(route)

    return routes, total_km


# --------------------------------------------------------
# DataFrame Converter
# --------------------------------------------------------
def routes_to_dataframe(df, routes):
    """Convert OR-Tools routes into a readable DataFrame for Streamlit."""
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

    return (
        __import__("pandas")
        .DataFrame(results)
        .sort_values(["vehicle", "stop_order"])
        .reset_index(drop=True)
    )
