import pandas as pd
import os

# Create a directory for the project data to keep things organized
output_dir = "campus_transport_data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"📂 Generatings datasets in '{output_dir}/'...")

# ==========================================
# 1. NODE DATA (Locations)
# Used for: Visualization, Graph Nodes
# ==========================================
nodes_data = [
    {"node_id": 0, "name": "Main Gate",      "type": "Entry/Exit",  "x": 0,   "y": 0,    "elevation_m": 10},
    {"node_id": 1, "name": "Admin Block",    "type": "Academic",    "x": 200, "y": 300,  "elevation_m": 12},
    {"node_id": 2, "name": "Library",        "type": "Academic",    "x": 400, "y": 300,  "elevation_m": 15},
    {"node_id": 3, "name": "Student Center", "type": "Social",      "x": 300, "y": 100,  "elevation_m": 10},
    {"node_id": 4, "name": "Dorm A (North)", "type": "Residential", "x": 200, "y": 600,  "elevation_m": 20},
    {"node_id": 5, "name": "Dorm B (South)", "type": "Residential", "x": 300, "y": -200, "elevation_m": 8},
    {"node_id": 6, "name": "Sports Complex", "type": "Social",      "x": 600, "y": 0,    "elevation_m": 10},
    {"node_id": 7, "name": "Engg Dept",      "type": "Academic",    "x": 500, "y": 400,  "elevation_m": 18},
    {"node_id": 8, "name": "Cafeteria",      "type": "Social",      "x": 400, "y": 0,    "elevation_m": 10},
    {"node_id": 9, "name": "Research Park",  "type": "Industry",    "x": 800, "y": 300,  "elevation_m": 25}
]

df_nodes = pd.DataFrame(nodes_data)
df_nodes.to_csv(f"{output_dir}/campus_locations.csv", index=False)
print("✅ Created campus_locations.csv")


# ==========================================
# 2. EDGE DATA (The Network)
# Used for: Dijkstra, MST, Max Flow
# Note: 'capacity' is people per minute. 
#       'mode_type' determines if cars can enter.
# ==========================================
edges_data = [
    {"source": 0, "target": 3, "distance_m": 350, "mode_type": "road", "capacity_pax_min": 120, "condition": "good"},
    {"source": 0, "target": 1, "distance_m": 400, "mode_type": "road", "capacity_pax_min": 150, "condition": "good"},
    {"source": 3, "target": 8, "distance_m": 150, "mode_type": "path", "capacity_pax_min": 80,  "condition": "narrow"}, # Walk only
    {"source": 1, "target": 2, "distance_m": 200, "mode_type": "road", "capacity_pax_min": 200, "condition": "excellent"},
    {"source": 2, "target": 7, "distance_m": 150, "mode_type": "road", "capacity_pax_min": 150, "condition": "good"},
    {"source": 1, "target": 4, "distance_m": 300, "mode_type": "path", "capacity_pax_min": 60,  "condition": "stairs"}, # Walk only
    {"source": 4, "target": 7, "distance_m": 450, "mode_type": "road", "capacity_pax_min": 100, "condition": "poor"},
    {"source": 3, "target": 5, "distance_m": 300, "mode_type": "road", "capacity_pax_min": 120, "condition": "good"},
    {"source": 5, "target": 8, "distance_m": 250, "mode_type": "path", "capacity_pax_min": 80,  "condition": "flat"}, # Walk only
    {"source": 8, "target": 6, "distance_m": 200, "mode_type": "road", "capacity_pax_min": 300, "condition": "excellent"},
    {"source": 7, "target": 9, "distance_m": 400, "mode_type": "road", "capacity_pax_min": 500, "condition": "highway"},
    {"source": 6, "target": 9, "distance_m": 500, "mode_type": "road", "capacity_pax_min": 500, "condition": "highway"}
]

df_edges = pd.DataFrame(edges_data)
df_edges.to_csv(f"{output_dir}/transport_network.csv", index=False)
print("✅ Created transport_network.csv")


# ==========================================
# 3. DRIVER BIDS (Assignment Problem)
# Used for: Hungarian Algorithm (Minimizing Cost)
# Values represent Overtime Cost ($) for that shift
# ==========================================
bids_data = [
    {"driver": "D1", "morning_loop": 20, "lunch_express": 15, "evening_loop": 40, "night_safety": 60, "weekend_special": 50},
    {"driver": "D2", "morning_loop": 25, "lunch_express": 20, "evening_loop": 35, "night_safety": 55, "weekend_special": 55},
    {"driver": "D3", "morning_loop": 18, "lunch_express": 22, "evening_loop": 30, "night_safety": 70, "weekend_special": 45},
    {"driver": "D4", "morning_loop": 30, "lunch_express": 25, "evening_loop": 25, "night_safety": 40, "weekend_special": 60},
    {"driver": "D5", "morning_loop": 22, "lunch_express": 18, "evening_loop": 28, "night_safety": 50, "weekend_special": 40}
]

df_bids = pd.DataFrame(bids_data)
df_bids.to_csv(f"{output_dir}/driver_bids.csv", index=False)
print("✅ Created driver_bids.csv")


# ==========================================
# 4. ENVIRONMENTAL FACTORS
# Used for: CO2 Analysis & Strategic Recommendations
# ==========================================
eco_data = [
    {"mode": "Walking",          "avg_speed_kmh": 5,  "co2_g_per_km": 0,   "space_sqm": 0.8},
    {"mode": "Cycling",          "avg_speed_kmh": 15, "co2_g_per_km": 0,   "space_sqm": 1.2},
    {"mode": "Electric Shuttle", "avg_speed_kmh": 25, "co2_g_per_km": 40,  "space_sqm": 1.5},
    {"mode": "Diesel Bus",       "avg_speed_kmh": 25, "co2_g_per_km": 90,  "space_sqm": 1.5},
    {"mode": "Private Car",      "avg_speed_kmh": 30, "co2_g_per_km": 170, "space_sqm": 8.0}
]

df_eco = pd.DataFrame(eco_data)
df_eco.to_csv(f"{output_dir}/eco_stats.csv", index=False)
print("✅ Created eco_stats.csv")

print("\n🎉 All datasets generated successfully!")