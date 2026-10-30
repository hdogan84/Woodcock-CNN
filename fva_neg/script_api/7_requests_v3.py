import requests
import pandas as pd

url = "https://xeno-canto.org/api/3/recordings"

MAX_RECORDINGS = 3000

page = 1
found = []
metadata = []
recorder = "Working group for animal ecology and planning"


while len(found) < MAX_RECORDINGS:

    params = {
        #"query": 'cnt:Germany',
        "query": 'rec:"Working group for animal ecology and planning"',
        "key": "c5581f6264b1fbe8123550c6490977ca857a1d7d",
        "page": page,
        "per_page": 250
    }

    response = requests.get(url, params=params)
    response.raise_for_status()

    data = response.json()
    recordings = data.get("recordings", [])

    print(f"Page {page}: {len(recordings)} recordings")

    for rec in recordings:

        metadata.append(rec)
        
        if len(found) >= MAX_RECORDINGS:
            break

    num_pages = data.get("numPages", 1)

    if page >= num_pages:
        break

    page += 1


df = pd.DataFrame(metadata)

df.to_csv(
    "xc_FVA.csv",
    index=False
)

print()
print("=" * 60)
print(f"Found {len(df)} recordings")
print("Saved to: xc_FVA.csv")
print("=" * 60)