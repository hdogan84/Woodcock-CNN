import pandas as pd

# Read CSV
df = pd.read_csv("selections.csv")

# ---------------------------------------------------------
# 1. Find whistle annotations
# ---------------------------------------------------------
df_whistle = df[
    df["annotation"].str.contains("whistle", case=False, na=False)
].copy()

# Extract date from sound.files (first 8 digits)
df_whistle["date"] = pd.to_datetime(
    df_whistle["sound.files"].str.extract(r"^(\d{8})")[0],
    format="%Y%m%d"
)

# Sort by date, then sound.files, then selec
df_whistle = df_whistle.sort_values(
    ["date", "sound.files", "selec"]
)

# Determine unique whistle files in chronological order
files = (
    df_whistle[["sound.files", "date"]]
    .drop_duplicates()
    .sort_values(["date", "sound.files"])
)

# ---------------------------------------------------------
# 2. Select the most recent 25% of whistle files as test
# ---------------------------------------------------------
n_files = len(files)
n_train = int(0.75 * n_files)

train_files = set(files.iloc[:n_train]["sound.files"])
test_files = set(files.iloc[n_train:]["sound.files"])

# ---------------------------------------------------------
# 3. Test set:
#    ONLY whistle annotations from the test files
# ---------------------------------------------------------
df_test = (
    df_whistle[
        df_whistle["sound.files"].isin(test_files)
    ]
    .sort_values(
        by=["date", "sound.files", "selec"],
        ascending=[False, False, True]
    )
)

# ---------------------------------------------------------
# 4. Train/validation:
#    EVERYTHING that is NOT in df_test
# ---------------------------------------------------------
df_train_val = df.drop(index=df_test.index).copy()

# Optional: sort train/val
df_train_val = df_train_val.sort_values(
    by=["sound.files", "selec"]
)

# ---------------------------------------------------------
# 5. Save
# ---------------------------------------------------------
df_test.to_csv("selections_whistle_test.csv", index=False)
df_train_val.to_csv("selections_train_val.csv", index=False)

print(f"Saved {len(df_test)} rows to selections_whistle_test.csv")
print(f"Saved {len(df_train_val)} rows to df_train_val.csv")

print(f"\nUnique whistle files: {n_files}")
print(f"Train files: {len(train_files)}")
print(f"Test files : {len(test_files)}")

print("\nTest annotations:")
print(df_test["annotation"].value_counts())

print("\nTrain/validation annotations:")
print(df_train_val["annotation"].value_counts())