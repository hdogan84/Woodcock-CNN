import requests
import pandas as pd

url = "https://xeno-canto.org/api/3/recordings"

MAX_RECORDINGS = 300

page = 10
found = []

while len(found) < MAX_RECORDINGS:

    params = {
        #"query": 'cnt:Germany',
        "query": "fam:scolopacidae",
        "key": "c5581f6264b1fbe8123550c6490977ca857a1d7d",
        "page": page,
        "per_page": "250"
    }

    response = requests.get(url, params=params)
    response.raise_for_status()

    data = response.json()
    recordings = data.get("recordings", [])

    print(f"Page {page}: {len(recordings)} recordings")

    for rec in recordings:

        mic = rec.get("mic", "") or ""

        if "audiomoth" in mic.lower():

            found.append({
                "recording_id": rec.get("id"),
                "mic": mic,
                "species": rec.get("en"),
                "country": rec.get("cnt"),
                "date": rec.get("date"),
                "year": rec.get("year"),
                "duration": rec.get("length"),
                "recording_url": f"https://xeno-canto.org/{rec.get('id')}"
            })

            print(
                f"FOUND {len(found)}/{MAX_RECORDINGS}: "
                f"ID={rec.get('id')} | "
                f"Mic={mic} | "
                f"Species={rec.get('en')} | "
                f"Country={rec.get('cnt')} | "
                f"Date={rec.get('date')}"
            )

            if len(found) >= MAX_RECORDINGS:
                break

    num_pages = data.get("numPages", 1)

    if page >= num_pages:
        break

    page += 1


df = pd.DataFrame(found)

df.to_csv(
    "xc_audiomoth_Eurasian_pt2.csv",
    index=False
)

print()
print("=" * 60)
print(f"Found {len(df)} AudioMoth recordings")
print("Saved to: xeno_canto_audiomoth_before_2023_50.csv")
print("=" * 60)