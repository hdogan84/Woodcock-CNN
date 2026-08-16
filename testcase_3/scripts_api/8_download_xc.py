import pandas as pd
import requests
from pathlib import Path

CSV_FILE = "xc_audiomoth_Eurasian_pt2.csv"
OUTPUT_DIR = Path("audiomoth_Eurasian")

OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(CSV_FILE)

for i, row in df.iterrows():

    recording_id = str(row["recording_id"])
    output_file = OUTPUT_DIR / f"{recording_id}.mp3"

    if output_file.exists():
        print(f"[{i+1}/{len(df)}] Already exists: {output_file.name}")
        continue

    url = f"https://xeno-canto.org/{recording_id}/download"

    print(f"[{i+1}/{len(df)}] Downloading {recording_id}...")

    try:
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()

        with open(output_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

        print(f"    Saved: {output_file}")

    except Exception as e:
        print(f"    ERROR: {e}")

print("Done.")