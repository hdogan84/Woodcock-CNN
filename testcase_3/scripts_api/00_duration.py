import pandas as pd

# Read CSV
df = pd.read_csv("events.csv")

# Calculate duration
df["duration"] = df["end"] - df["start"]

# Extract date from sound.files
"""df["date"] = pd.to_datetime(
    df["sound.files"].str.extract(r"^(\d{8})")[0],
    format="%Y%m%d"
)"""

# Extract date from deploy.id
df["date"] = pd.to_datetime(
    df["deploy.id"].str.extract(r"(\d{4}_\d{2}_\d{2})")[0],
    format="%Y_%m_%d"
)

# Sort from longest to shortest
df = df.sort_values("duration", ascending=False)

# Save
df.to_csv("events_long_2_short.csv", index=False)

print(df[["sound.files", "start", "end", "duration", "date"]].head(20))