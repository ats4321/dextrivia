import json
from pathlib import Path

import requests

# /pub/TLE/*.txt often returns 403; gp.php returns the same Celestrak GROUP.
URL = (
    "https://celestrak.org/NORAD/elements/gp.php"
    "?GROUP=iridium-33-debris&FORMAT=tle"
)
HEADERS = {
    "User-Agent": (
        "OrbitalDebrisStudy/1.0 (Iridium-Cosmos debris TLEs; "
        "https://celestrak.org/)"
    ),
}

script_dir = Path(__file__).resolve().parent
out_path = script_dir / "debris_tles.json"

response = requests.get(URL, headers=HEADERS, timeout=60)
response.raise_for_status()

lines = [ln.strip() for ln in response.text.strip().splitlines() if ln.strip()]

objects = []
for i in range(0, len(lines) - 2, 3):
    objects.append(
        {
            "name": lines[i],
            "line1": lines[i + 1],
            "line2": lines[i + 2],
        }
    )

print(f"Got {len(objects)} debris objects")
for obj in objects[:3]:
    print(f"  {obj['name']}")

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(objects, f, indent=2)

print(f"Saved to {out_path}")
