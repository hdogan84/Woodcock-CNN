import pandas as pd

# --------------------------------------------------
# File paths
# --------------------------------------------------

main_file = "selections.csv"
short_file = "selec_long_neg.csv"
output_file = "selecs_positive_test_samedate.csv"


# --------------------------------------------------
# Load files
# --------------------------------------------------

df = pd.read_csv(main_file)
df["site.id"] = df["deploy.id"].astype(str).str.split("_").str[0]
df["duration"] = df["end"] - df["start"]

df["date"] = pd.to_datetime(
    df["sound.files"].str.extract(r"^(\d{8})")[0],
    format="%Y%m%d"
).dt.date

df["start.time"] = pd.to_datetime(
    df["sound.files"].str.extract(r"_(\d{6})\.")[0],
    format="%H%M%S"
).dt.time

df_short = pd.read_csv(short_file)



print(f"Main CSV: {len(df)} rows")
print(f"First 30 rows from long2short: {len(df_short)} rows")


# --------------------------------------------------
# Get the properties of the negative selections
# --------------------------------------------------
selec_ids = df_short["selec"].unique()
recording_ids = df[df["selec"].isin(selec_ids)]["recording.id"].unique()
deploy_ids = df[df["selec"].isin(selec_ids)]["deploy.id"].unique()
dates = df[df["selec"].isin(selec_ids)]["date"].unique()
site_ids = df[df["selec"].isin(selec_ids)]["site.id"].unique()

print(f"\nRecording IDs represented negs:")
print(recording_ids)
print(f"Number of unique recording IDs: {len(recording_ids)}")

# --------------------------------------------------
# Slice main CSV to those recording IDs
# --------------------------------------------------

df_sliced = df[
    ~df["selec"].isin(selec_ids)
].copy()

print(f"\nDF length after excluding Negative selections: {len(df_sliced)}")

# --------------------------------------------------
# Slice main CSV to those recording IDs
# --------------------------------------------------

df_sliced = df_sliced[df_sliced["date"].isin(dates)].copy()
#df_sliced = df_sliced[df_sliced["site.id"].isin(site_ids)].copy()
df_sliced = df_sliced[df_sliced["duration"] < 3.05]

print(f"\nRows belonging to positive test set: {len(df_sliced)}")

print("Number of rows with duration > 3 seconds:", (df_sliced["duration"] < 3.05).sum())

# --------------------------------------------------
# Save
# --------------------------------------------------

# df_sliced.to_csv(output_file, index=False)

print(f"\nFinal rows: {len(df_sliced)}")
print(f"Saved to: {output_file}")