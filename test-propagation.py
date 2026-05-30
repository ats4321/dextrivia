"""
Loads your real CelesTrak debris_tles.json and propagates
the first 5 objects at epoch, +10 min, +30 min.
All altitudes should be 200-2000 km if working correctly.
"""

import json
from datetime import datetime, timezone, timedelta
from propagation import propagate

# Load your real TLEs from CelesTrak
with open("debris_tles.json") as f:
    data = json.load(f)

# Take first 5 objects
sample = data[:5]

base_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
offsets   = [timedelta(0), timedelta(minutes=10), timedelta(minutes=30)]

print(f"{'Name':<30} {'Offset':>8} {'Alt (km)':>10} {'|r| (km)':>10} {'Err':>5}")
print("─" * 70)

all_ok = True

for obj in sample:
    name = obj["name"]
    l1   = obj["line1"]
    l2   = obj["line2"]

    for offset in offsets:
        t      = base_time + offset
        result = propagate(l1, l2, t)
        alt    = result["altitude_km"]
        mag    = float(sum(x**2 for x in result["position_km"]) ** 0.5)
        err    = result["error"]
        label  = f"+{int(offset.total_seconds()//60)}min"

        status = "✅" if err == 0 and 200 <= alt <= 2000 else "❌"
        print(f"{name:<30} {label:>8} {alt:>10.1f} {mag:>10.1f} {err:>5}  {status}")

        if err != 0 or not (200 <= alt <= 2000):
            all_ok = False

print()
print("All checks passed ✅ — ready for delta-v matrix tomorrow." if all_ok
      else "Some checks failed ❌ — see rows above.")