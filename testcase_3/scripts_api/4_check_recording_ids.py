import pandas as pd

# --------------------------------------------------
# Settings
# --------------------------------------------------

csv_path = "selections.csv"


# --------------------------------------------------
# Load CSV
# --------------------------------------------------

df = pd.read_csv(csv_path)

print(f"Loaded {len(df)} rows")
print(f"Unique sound.files: {df['sound.files'].nunique()}")
print(f"Unique recording.id: {df['recording.id'].nunique()}")
print()


# --------------------------------------------------
# Check recording.id -> sound.files
# --------------------------------------------------

recording_to_sound = (
    df.groupby("recording.id")["sound.files"]
    .nunique()
)

bad_recording_ids = recording_to_sound[
    recording_to_sound > 1
]

print("==============================================")
print("CHECK: recording.id -> sound.files")
print("==============================================")

if len(bad_recording_ids) == 0:
    print("OK: Every recording.id maps to exactly one sound.files")
else:
    print(f"PROBLEM: {len(bad_recording_ids)} recording.id values map to multiple sound.files")
    print()
    print(bad_recording_ids)

    print("\nDetails:")
    print(
        df[df["recording.id"].isin(bad_recording_ids.index)]
        [["recording.id", "sound.files", "deploy.id"]]
        .drop_duplicates()
        .sort_values(["recording.id", "sound.files"])
        .to_string(index=False)
    )

print()


# --------------------------------------------------
# Check sound.files -> recording.id
# --------------------------------------------------

sound_to_recording = (
    df.groupby("sound.files")["recording.id"]
    .nunique()
)

bad_sound_files = sound_to_recording[
    sound_to_recording > 1
]

print("==============================================")
print("CHECK: sound.files -> recording.id")
print("==============================================")

if len(bad_sound_files) == 0:
    print("OK: Every sound.files maps to exactly one recording.id")
else:
    print(f"PROBLEM: {len(bad_sound_files)} sound.files map to multiple recording.id values")
    print()
    print(bad_sound_files)

    print("\nDetails:")
    print(
        df[df["sound.files"].isin(bad_sound_files.index)]
        [["sound.files", "recording.id", "deploy.id"]]
        .drop_duplicates()
        .sort_values(["sound.files", "recording.id"])
        .to_string(index=False)
    )

print()


# --------------------------------------------------
# Final one-to-one check
# --------------------------------------------------

print("==============================================")
print("FINAL RESULT")
print("==============================================")

if len(bad_recording_ids) == 0 and len(bad_sound_files) == 0:
    print("✓ PERFECT ONE-TO-ONE MAPPING")
    print("  Every recording.id corresponds to exactly one sound.files")
    print("  Every sound.files corresponds to exactly one recording.id")
else:
    print("✗ NOT A ONE-TO-ONE MAPPING")
    print("  See the problems listed above.")