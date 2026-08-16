import pandas as pd

# --------------------------------------------------
# File paths
# --------------------------------------------------

main_file = "../data/Holderried/selections.csv"
short_file = "selec_long_neg.csv"
output_file = "selecs_positive_test.csv"


# --------------------------------------------------
# Load files
# --------------------------------------------------

df = pd.read_csv(main_file)
df_short = pd.read_csv(short_file)


print(f"Main CSV: {len(df)} rows")
print(f"Length of Negative selections: {len(df_short)} rows")


# --------------------------------------------------
# Get recording IDs from the Negatives csv file
# --------------------------------------------------
selec_ids = df_short["selec"].unique()
recording_ids = df[
    df["selec"].isin(selec_ids)
]["recording.id"].unique()

print(f"\nRecording IDs present in Negative selections:")
print(recording_ids)
print(f"Number of unique recording IDs: {len(recording_ids)}")


# --------------------------------------------------
# Slice main CSV to those recording IDs
# --------------------------------------------------

df_sliced = df[
    df["recording.id"].isin(recording_ids)
].copy()

print(f"\nRows belonging to those recording IDs: {len(df_sliced)}")


# --------------------------------------------------
# Remove the particular selec rows from the 30 rows
#
# Match using BOTH recording.id and selec
# --------------------------------------------------


df_sliced = df_sliced[
    ~df_sliced["selec"].isin(selec_ids)
].copy()


# --------------------------------------------------
# Save
# --------------------------------------------------

df_sliced.to_csv(output_file, index=False)

print(f"\nFinal rows: {len(df_sliced)}")
print(f"Saved to: {output_file}")