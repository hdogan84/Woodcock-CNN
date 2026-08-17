import pandas as pd

# Load CSVs
df = pd.read_csv("selections.csv")
df_pos_test = pd.read_csv("selecs_positive_test_samedate_samesite.csv")
df_neg = pd.read_csv("selec_long_neg.csv")

# Make sure selec IDs have the same type
df["selec"] = pd.to_numeric(df["selec"], errors="coerce")
df_pos_test["selec"] = pd.to_numeric(df_pos_test["selec"], errors="coerce")
df_neg["selec"] = pd.to_numeric(df_neg["selec"], errors="coerce")

# IDs to exclude
positive_test_ids = set(df_pos_test["selec"].dropna())
negative_ids = set(df_neg["selec"].dropna())

# Remove positive test and negative IDs from main dataset
df_train = df[
    ~df["selec"].isin(positive_test_ids) &
    ~df["selec"].isin(negative_ids)
].copy()

# Calculate duration in seconds
df_train["duration"] = df_train["end"] - df_train["start"]


# Exclude recordings longer than 3 seconds
df_train = df_train[df_train["duration"] <= 3].copy()

# Save
df_train.to_csv("selections_train.csv", index=False)

# Report
print(f"Original rows:       {len(df)}")
print(f"Positive test IDs:   {len(positive_test_ids)}")
print(f"Negative IDs:        {len(negative_ids)}")
print(f"Final training rows: {len(df_train)}")
print()
print("Saved as: selections_train.csv")