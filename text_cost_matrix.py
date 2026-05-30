import json
import numpy as np
from datetime import datetime, timezone
from cost_matrix import build_cost_matrix, get_altitudes_from_tles

# Load TLEs
with open("debris_tles.json") as f:
    data = json.load(f)

# Use first 10 objects
sample = data[:10]
t = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

# Get altitudes
objects = get_altitudes_from_tles(sample, t)
print(f"Successfully propagated {len(objects)} objects\n")

names     = [o[1] for o in objects]
altitudes = [o[2] for o in objects]

for name, alt in zip(names, altitudes):
    print(f"  {name:<35} {alt:.1f} km")

# Build matrix
matrix = build_cost_matrix(altitudes)

print(f"\nDelta-v cost matrix ({len(names)}×{len(names)}) in km/s:")
print(np.round(matrix, 4))

print(f"\nMatrix stats:")
print(f"  Min non-zero delta-v : {matrix[matrix > 0].min():.4f} km/s")
print(f"  Max delta-v          : {matrix.max():.4f} km/s")
print(f"  Mean non-zero        : {matrix[matrix > 0].mean():.4f} km/s")

np.save("cost_matrix.npy", matrix)
with open("object_names.json", "w") as f:
    json.dump(names, f)
print("\nSaved cost_matrix.npy and object_names.json")
print("Ready for greedy baseline tomorrow ✅")