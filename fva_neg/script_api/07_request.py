import requests
import pandas as pd
import time

# --------------------------------------------------
# 1. Read input CSV and extract unique XC IDs
# --------------------------------------------------

input_file = "../../data/annotation_xc/eurwoo.csv"

df = pd.read_csv(
    input_file,
    header=None
)

xc_ids = df.iloc[:, 1].dropna().astype(int).unique()

print(f"Found {len(xc_ids)} unique Xeno-canto IDs")


# --------------------------------------------------
# 2. Query Xeno-canto API
# --------------------------------------------------

api_url = "https://xeno-canto.org/api/3/recordings"

metadata = []


for i, xc_id in enumerate(xc_ids, start=1):

    params = {
        "query": f"nr:{xc_id}",
        "key": "c5581f6264b1fbe8123550c6490977ca857a1d7d",
        "per_page": 10
    }

    response = requests.get(
        api_url,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    recordings = data.get("recordings", [])

    print(
        f"{i}/{len(xc_ids)} | "
        f"Requested: {xc_id} | "
        f"Returned: {len(recordings)}"
    )

    # --------------------------------------------------
    # Verify that exactly the requested recording came back
    # --------------------------------------------------

    if len(recordings) != 1:
        print(f"WARNING: Expected 1 recording for {xc_id}, got {len(recordings)}")
        continue

    rec = recordings[0]

    returned_id = str(rec.get("id"))

    if returned_id != str(xc_id):
        print(
            f"WARNING: ID mismatch! "
            f"Requested {xc_id}, got {returned_id}"
        )
        continue

    # Correct recording
    metadata.append(rec)

    print(
        f"    OK: {rec.get('id')} | "
        f"{rec.get('en')} | "
        f"{rec.get('cnt')} | "
        f"{rec.get('length')}"
    )

    time.sleep(0.1)


# --------------------------------------------------
# 3. Save metadata
# --------------------------------------------------

metadata_df = pd.DataFrame(metadata)

metadata_df.to_csv(
    "xc_metadata.csv",
    index=False
)

print()
print("=" * 60)
print(f"Unique XC IDs queried:      {len(xc_ids)}")
print(f"Metadata records retrieved: {len(metadata_df)}")
print(f"Missing/failed:              {len(xc_ids) - len(metadata_df)}")
print("Saved to: xc_metadata.csv")
print("=" * 60)