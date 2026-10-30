import requests
import json

xc_id = 891756

api_url = "https://xeno-canto.org/api/3/recordings"

params = {
    "query": f"nr:{xc_id}",
    "key": "c5581f6264b1fbe8123550c6490977ca857a1d7d",
    "per_page": 10
}

response = requests.get(api_url, params=params, timeout=30)

print("STATUS:", response.status_code)
print("URL:", response.url)

response.raise_for_status()

data = response.json()

print(json.dumps(data, indent=2)[:5000])

recordings = data.get("recordings", [])

print("\nNumber returned:", len(recordings))

for rec in recordings:
    print(
        "Returned ID:", rec.get("id"),
        "| Species:", rec.get("en"),
        "| URL:", rec.get("url")
    )

if len(recordings) == 1 and str(recordings[0].get("id")) == str(xc_id):
    print("\nSUCCESS: Exactly one recording returned, and the ID matches.")
else:
    print("\nFAILED: Did not get exactly the requested recording.")