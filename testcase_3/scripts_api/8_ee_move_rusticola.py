import shutil
from pathlib import Path
import pandas as pd

# Paths
csv_file = Path("audiomoth_rusticola.csv")
source_folder = Path("audiomoth_Eurasian")
target_folder = Path("audiomoth_rusticola")

# Create target folder
target_folder.mkdir(exist_ok=True)

# Load CSV
df = pd.read_csv(csv_file, header=None)

# First column contains XC IDs
xc_ids = set(df.iloc[:, 0].astype(str).str.strip())

print(f"XC IDs in CSV: {len(xc_ids)}")

# Find and move matching MP3 files
moved = 0
not_found = []

for xc_id in xc_ids:
    source_file = source_folder / f"{xc_id}.mp3"
    target_file = target_folder / f"{xc_id}.mp3"

    if source_file.exists():
        shutil.move(str(source_file), str(target_file))
        print(f"Moved: {source_file.name}")
        moved += 1
    else:
        not_found.append(xc_id)

print("\n--- Summary ---")
print(f"Moved: {moved}")
print(f"Not found: {len(not_found)}")

if not_found:
    print("\nFiles not found:")
    for xc_id in not_found:
        print(xc_id)