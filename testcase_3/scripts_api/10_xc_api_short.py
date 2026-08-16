import requests

url = "https://xeno-canto.org/api/3/recordings"

params = {
    "query": "fam:scolopacidae",
    "key": "demo",
    "page": 1
}

r = requests.get(url, params=params)

print("Status:", r.status_code)
print("URL:", r.url)
print("Response:")
print(r.text[:2000])