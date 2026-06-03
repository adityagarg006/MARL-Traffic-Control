# --- STANDALONE GRAPH DIAGNOSTIC CELL ---
import json
import torch
import gym
import cityflow
import gym_cityflow
import os
from torch_geometric.utils import add_self_loops

config_path = "Intersections_4/sample_config.json"

# 1. Ask the environment for the exact nodes it controls
print("Initializing environment to detect active nodes...")
env = gym.make(id='cityflow-v0', configPath=config_path, episodeSteps=10)
raw_obs = env.reset()
active_nodes = sorted(raw_obs.keys())
env.close()

# 2. Parse the roadnet 
cfg = json.load(open(config_path, "r"))
roadnet_path = os.path.join(cfg.get("dir", ""), cfg["roadnetFile"])
roadnet = json.load(open(roadnet_path, "r"))

# 3. Create and print the mapping
node_to_index = {node: idx for idx, node in enumerate(active_nodes)}
print("\nNode mapping (intersection_id → node_idx):")
for inter, idx in node_to_index.items():
    print(f"  {inter} → {idx}")

# 4. Map the physical roads
edge_list = []
for road in roadnet["roads"]:
    start_node = road.get("startIntersection")
    end_node = road.get("endIntersection")
    if start_node in node_to_index and end_node in node_to_index:
        u, v = node_to_index[start_node], node_to_index[end_node]
        edge_list.append([u, v])
        edge_list.append([v, u]) # Bidirectional

tensor_edges = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

# --- OPTION 1 vs OPTION 2 CHECK ---
# If you chose Option 2 (Strict Replication), uncomment the line below to match the referral code perfectly.
# If you chose Option 1 (Mathematically Pure), leave it commented out!
tensor_edges, _ = add_self_loops(tensor_edges, num_nodes=len(active_nodes))

print("\nEdge index tensor:")
print(tensor_edges)