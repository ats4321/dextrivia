import numpy as np
import json
from greedy import greedy_solve

# Load saved matrix and names
matrix = np.load("cost_matrix.npy")
with open("object_names.json") as f:
    names = json.load(f)

n = len(names)
print(f"Running greedy solver on {n} debris objects...\n")

# Try starting from every object, keep best result
best_result = None
best_dv     = float('inf')

for start in range(n):
    result = greedy_solve(matrix, start=start)
    if result['total_deltav'] < best_dv:
        best_dv     = result['total_deltav']
        best_result = result
        best_start  = start

print(f"Best greedy solution (starting from object {best_start}):")
print(f"  Total delta-v : {best_result['total_deltav']:.4f} km/s\n")
print(f"  Sequence:")
for step, (idx, cost) in enumerate(
    zip(best_result['sequence'], [0.0] + best_result['step_costs'])
):
    marker = " ← start" if step == 0 else f"  +{cost:.4f} km/s"
    print(f"    Step {step+1:2d}: {names[idx]:<35}{marker}")

print(f"\nThis is your baseline.")
print(f"Your QUBO solver this weekend needs to beat {best_result['total_deltav']:.4f} km/s")

# Save baseline for comparison
import json
with open("greedy_baseline.json", "w") as f:
    json.dump({
        "total_deltav": best_result['total_deltav'],
        "sequence": best_result['sequence'],
        "names": [names[i] for i in best_result['sequence']],
    }, f, indent=2)

print("Saved greedy_baseline.json ✅")