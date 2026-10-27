"""
Last-Mile Routing Optimisation — Module 3B (revised: variable fleet)
AI-Driven Supply Chain Resilience in African E-Commerce
Input:  instance_n50_seed42.json (Kigali CVRP benchmark)
Output: Table 5 — distance and route count: baseline vs OR-Tools
"""

import json
import numpy as np
import math
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# ============================================================
# 1. LOAD INSTANCE
# ============================================================
with open("instance_n200_seed42.json") as f:
    data = json.load(f)

depot = data["depot"]
customers = data["customers"]
capacity = data["vehicle_capacity"]
n_vehicles_stated = data["n_vehicles"]
n_customers = data["n_customers"]

total_demand = sum(c["demand_kg"] for c in customers)
print("=== Instance summary ===")
print(f"Customers:              {n_customers}")
print(f"Vehicles (as stated):   {n_vehicles_stated}")
print(f"Capacity/vehicle:       {capacity} kg")
print(f"Total demand:           {total_demand:.1f} kg")
print(f"Fleet capacity stated:  {n_vehicles_stated * capacity} kg")
print(f"Feasible as stated:     {total_demand <= n_vehicles_stated * capacity}")
print(f"Minimum vehicles needed: {math.ceil(total_demand / capacity)}")

# ============================================================
# 2. COORDINATES, DEMANDS
# ============================================================
coords = [(depot["lat"], depot["lon"])] + [(c["lat"], c["lon"]) for c in customers]
demands = [0] + [int(round(c["demand_kg"] * 10)) for c in customers]
CAPACITY_SCALED = int(capacity * 10)

# ============================================================
# 3. HAVERSINE DISTANCE MATRIX
# ============================================================
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

n = len(coords)
dist_matrix = np.zeros((n, n), dtype=int)
for i in range(n):
    for j in range(n):
        if i != j:
            d = haversine_km(*coords[i], *coords[j])
            dist_matrix[i, j] = int(round(d * 1000))

# ============================================================
# 4. BASELINE: SWEEP HEURISTIC
# ============================================================
def baseline_sweep(coords, demands, capacity):
    depot_lat, depot_lon = coords[0]
    angles = []
    for i in range(1, len(coords)):
        lat, lon = coords[i]
        dx = (lon - depot_lon) * math.cos(math.radians(depot_lat))
        dy = lat - depot_lat
        angles.append((math.atan2(dy, dx), i))
    angles.sort()

    routes, current, load = [], [], 0
    for _, idx in angles:
        d = demands[idx]
        if load + d > capacity:
            routes.append(current); current, load = [], 0
        current.append(idx); load += d
    if current:
        routes.append(current)
    return routes

def route_distance(route, dist_matrix):
    if not route: return 0
    total = dist_matrix[0, route[0]]
    for k in range(len(route) - 1):
        total += dist_matrix[route[k], route[k+1]]
    return total + dist_matrix[route[-1], 0]

baseline_routes = baseline_sweep(coords, demands, CAPACITY_SCALED)
baseline_total = sum(route_distance(r, dist_matrix) for r in baseline_routes)
baseline_n = len(baseline_routes)

print(f"\n=== Baseline (sweep heuristic) ===")
print(f"Routes:         {baseline_n}")
print(f"Total distance: {baseline_total/1000:.2f} km")

# ============================================================
# 5. OR-TOOLS CVRP (variable fleet)
# ============================================================
MAX_VEHICLES = n_customers  # effectively unlimited

manager = pywrapcp.RoutingIndexManager(n, MAX_VEHICLES, 0)
routing = pywrapcp.RoutingModel(manager)

def distance_callback(from_index, to_index):
    return dist_matrix[manager.IndexToNode(from_index),
                        manager.IndexToNode(to_index)]

transit_idx = routing.RegisterTransitCallback(distance_callback)
routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

# Fixed cost per vehicle: strongly discourage using extra vehicles
for v in range(MAX_VEHICLES):
    routing.SetFixedCostOfVehicle(1000000, v)   # 1000 km penalty

def demand_callback(from_index):
    return demands[manager.IndexToNode(from_index)]

demand_idx = routing.RegisterUnaryTransitCallback(demand_callback)
routing.AddDimensionWithVehicleCapacity(
    demand_idx, 0,
    [CAPACITY_SCALED] * MAX_VEHICLES,
    True, "Capacity"
)

search_params = pywrapcp.DefaultRoutingSearchParameters()
search_params.first_solution_strategy = (
    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
)
search_params.local_search_metaheuristic = (
    routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
)
search_params.time_limit.seconds = 30

print("\n=== Solving OR-Tools CVRP (30s time limit, variable fleet) ===")
solution = routing.SolveWithParameters(search_params)

# ============================================================
# 6. EXTRACT RESULTS
# ============================================================
if solution:
    ortools_total = 0
    ortools_routes = []
    for v in range(MAX_VEHICLES):
        index = routing.Start(v)
        route = []
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        route.append(manager.IndexToNode(index))
        if len(route) > 2:
            ortools_routes.append(route)
            ortools_total += route_distance(route[1:-1], dist_matrix)

    ortools_n = len(ortools_routes)
    print(f"Routes:         {ortools_n}")
    print(f"Total distance: {ortools_total/1000:.2f} km")

    table5 = pd.DataFrame([
        {"Method": "Sweep heuristic (baseline)",
         "Routes": baseline_n,
         "Total distance (km)": round(baseline_total/1000, 2),
         "Distance reduction (%)": 0.0},
        {"Method": "OR-Tools CVRP",
         "Routes": ortools_n,
         "Total distance (km)": round(ortools_total/1000, 2),
         "Distance reduction (%)": round((1 - ortools_total/baseline_total)*100, 1)},
    ])
    print("\n=== Table 5 (actual numbers) ===")
    print(table5.to_string(index=False))
    table5.to_csv("table5_vrp_routing.csv", index=False)

    # Capacity sanity check
    for r in ortools_routes:
        load = sum(demands[i] for i in r[1:-1])
        assert load <= CAPACITY_SCALED, f"Overloaded: {load}"
    print("All OR-Tools routes satisfy capacity.")
else:
    print("No solution found.")