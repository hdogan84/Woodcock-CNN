import requests

url = "https://xeno-canto.org/api/3/recordings"

params = {
    "query": "fam:laridae",
    "key": "demo"
}

response = requests.get(url, params=params)
response.raise_for_status()

data = response.json()

print("Total recordings returned:", len(data["recordings"]))
print()

for rec in data["recordings"]:
    mic = rec.get("mic", "")

    if mic and mic.strip():
        print(
            f'ID: {rec.get("id")} | '
            f'Mic: {mic}'
        )