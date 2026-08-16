import pandas as pd
import shutil
from pathlib import Path

# Read CSV
df = pd.read_csv("selections_whistle.csv")

src_dir = Path("selections")
dst_dir = Path("selections_test")
dst_dir.mkdir(exist_ok=True)

# Only test snippets
df_test = df[df["split"] == "test"]

moved = 0
missing = 0

for selec in df_test["selec"].unique():

    src = src_dir / f"{selec}.wav"
    dst = dst_dir / src.name

    if src.exists():
        shutil.copy2(src, dst)   # use shutil.move(...) if you want to move
        moved += 1
    else:
        print(f"Missing: {src}")
        missing += 1

print(f"Copied {moved} files")
print(f"Missing {missing} files")