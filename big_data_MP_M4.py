import pandas as pd
import numpy as np
import random
import os

# Configuration: How big do you want the project to be?
NUM_NODES = 50           # Scale this up to 100 or 500 if you want "Big Data"
AREA_SIZE = 2000         # 2km x 2km campus
CONNECTIVITY_RADIUS = 400 # Connect nodes if they are within 400m
RANDOM_SEED = 42         # Keep this fixed so every student gets the same "Random" city

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

output_dir = "large_campus_data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"🏗️ Generating Large Scale Campus ({NUM_NODES} Buildings)...")

# ==========================================
# 1. GENERATE NODES (Procedural Locations)
# ==========================================
node_types = ["Academic", "Residential", "Social", "Admin", "Industry"]
node_data = []

for i in range(NUM_NODES):
    # Cluster logic: Residential tends to be North, Industry East, etc.
    # We add some randomness to typical coordinates
    n_type = np.random.choice(node_types, p=[0.4, 0.3, 0.15, 0.1, 0.05])
    
    x = np.random.randint(0, AREA_SIZE)
    y = np.random.randint(0, AREA_SIZE)
    
    # Elevation noise (Perlin noise simplified)
    elevation = 10 + (x/100) + np.random.normal(0, 5)
    
    node_data.append({
        "node_id": i,
        "name": f"{n_type}_Bldg_{i}",
        "type": n_type,
        "x": x,
        "y": y,
        "elevation_m": round(elevation, 1)
    })

df_nodes = pd.DataFrame(node_data)
df_nodes.to_csv(f"{output_dir}/large_locations.csv", index=False)
print(f"✅ Generated {NUM_NODES} locations.")

# ==========================================
# 2. GENERATE EDGES (Geometric Graph Logic)
# Logic: If two buildings are close, a road likely exists.
# ==========================================
edges_data = []

# Calculate all-pairs distances (Vectorized for speed)
coords = df_nodes[['x', 'y']].values
from scipy.spatial.distance import pdist, squareform
dist_matrix = squareform(pdist(coords))

count_roads = 0
count_paths = 0

for i in range(NUM_NODES):
    for j in range(i + 1, NUM_NODES): # Avoid duplicates and self-loops
        dist = dist_matrix[i][j]
        
        # PROBABILITY LOGIC:
        # 1. High chance of connection if close (Local Roads)
        # 2. Small chance of connection if far (Highways/Express routes)
        
        connected = False
        mode = "road"
        
        if dist < 200:
            # Very close neighbors: Almost always connected
            if random.random() < 0.8: connected = True
            
        elif dist < CONNECTIVITY_RADIUS:
            # Medium distance: Moderate chance
            if random.random() < 0.3: connected = True
            
        elif dist < 1000 and random.random() < 0.02:
            # Long distance: Rare "Highway" connection
            connected = True
            mode = "highway"
        
        if connected:
            # Determine attributes
            if mode != "highway":
                # Short links might be walking paths vs roads
                mode = "path" if (dist < 150 and random.random() < 0.5) else "road"
            
            capacity = 500 if mode == "highway" else (200 if mode == "road" else 50)
            
            edges_data.append({
                "source": i,
                "target": j,
                "distance_m": round(dist, 1),
                "mode_type": mode,
                "capacity_pax_min": capacity
            })
            
            if mode == "path": count_paths += 1
            else: count_roads += 1

df_edges = pd.DataFrame(edges_data)
df_edges.to_csv(f"{output_dir}/large_network.csv", index=False)
print(f"✅ Generated Network: {count_roads} Roads, {count_paths} Paths.")


# ==========================================
# 3. SCALED DRIVER BIDS (Assignment Problem)
# Generate a 50x50 Matrix (50 Drivers, 50 Routes)
# ==========================================
num_drivers = 20
num_routes = 20 # Must match for square assignment or be padded

bids = []
for d in range(num_drivers):
    driver_bids = {"driver": f"D_{d}"}
    for r in range(num_routes):
        # Cost is random but correlated (some drivers are just expensive)
        base_rate = np.random.randint(15, 30)
        route_difficulty = np.random.randint(0, 20)
        driver_bids[f"Route_{r}"] = base_rate + route_difficulty
    bids.append(driver_bids)

df_bids = pd.DataFrame(bids)
df_bids.to_csv(f"{output_dir}/large_driver_bids.csv", index=False)
print(f"✅ Generated Bids for {num_drivers} drivers.")

print(f"\n🚀 HUGE DATASET READY in '{output_dir}/'")