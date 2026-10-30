import requests
import pandas as pd

url = "https://xeno-canto.org/api/3/recordings"

MAX_RECORDINGS = 25000

page = 1
found = []

while len(found) < MAX_RECORDINGS:

    params = {
        #  fam:Scolopicadie or something like that
        "query": "box:47.68,7.60,49.19,10.19",
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

        duration = rec.get("length")
        duration_str = str(duration)

        parts = duration_str.split(":")

        if len(parts) == 2:          # MM:SS
            minutes, seconds = parts
            duration_seconds = int(minutes) * 60 + float(seconds)

        elif len(parts) == 3:        # HH:MM:SS
            hours, minutes, seconds = parts
            duration_seconds = int(hours) * 3600 + int(minutes) * 60 + float(seconds)

        else:                        # already seconds
            duration_seconds = float(duration_str)

        if duration_seconds <3.0:
            continue

        if "eurasian woodcock" in rec.get("en").lower():
            continue

     
        found.append({
            "recording_id": rec.get("id"),
            "mic": rec.get("mic"),
            "species": rec.get("en"),
            "country": rec.get("cnt"),
            "date": rec.get("date"),
            "duration": rec.get("length"),
            "user": rec.get("rec"),
            "recording_url": f"https://xeno-canto.org/{rec.get('id')}"
        })

        print(
            f"FOUND {len(found)}/{MAX_RECORDINGS}: "
            f"ID={rec.get('id')} | "
            #f"Mic={mic} | "
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
    "xc_BadenWurttem.csv",
    index=False
)

print()
print("=" * 60)
print(f"Found {len(df)}recordings")
print("Saved to: csv")
print("=" * 60)