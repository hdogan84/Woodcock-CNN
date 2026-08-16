import pandas as pd
from pathlib import Path

# Read the test CSV (already sorted from newest to oldest)
df = pd.read_csv("selections_whistle_test.csv")

test_dir = Path("selections_test")

renamed = 0
missing = 0

# Rename each unique snippet according to its position in the CSV
for i, selec in enumerate(df["selec"].drop_duplicates()):

    old_file = test_dir / f"{selec}.wav"
    new_file = test_dir / f"{i}_{selec}.wav"

    if old_file.exists():
        old_file.rename(new_file)
        renamed += 1
    else:
        print(f"Missing: {old_file}")
        missing += 1

print(f"Renamed {renamed} files")
print(f"Missing {missing} files")