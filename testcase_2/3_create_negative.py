import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
import numpy as np

# --------------------------------------------------
# Paths
# --------------------------------------------------

csv_file = "selections_whistle_test_annot.csv"

input_dir = Path("selections_test/1")
output_dir = Path("selections_test/0")
output_dir.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Load CSV
# --------------------------------------------------

df = pd.read_csv(csv_file)

# Remove rows without a positive annotation
df = df.dropna(subset=["t_start", "t_end"]).copy()

print(f"Processing {len(df)} annotated snippets")


# --------------------------------------------------
# Process each annotation
# --------------------------------------------------

for _, row in df.iterrows():

    selec = int(row["selec"])
    t_start = float(row["t_start"])
    t_end = float(row["t_end"])

    # Find corresponding renamed file, e.g. 0_8095.wav
    matches = list(input_dir.glob(f"*_{selec}.wav"))

    if not matches:
        # Also try uppercase WAV
        matches = list(input_dir.glob(f"*_{selec}.WAV"))

    if not matches:
        print(f"WARNING: No audio found for selec {selec}")
        continue

    audio_file = matches[0]

    # --------------------------------------------------
    # Load audio
    # --------------------------------------------------

    y, sr = librosa.load(audio_file, sr=None, mono=True)

    duration = len(y) / sr

    # Make sure annotation is inside the audio
    t_start = max(0, t_start)
    t_end = min(duration, t_end)

    start_sample = int(t_start * sr)
    end_sample = int(t_end * sr)

    # --------------------------------------------------
    # Remove positive section
    # --------------------------------------------------

    negative_before = y[:start_sample]
    negative_after = y[end_sample:]

    # Concatenate everything except the positive call
    negative = np.concatenate([
        negative_before,
        negative_after
    ])

    # --------------------------------------------------
    # Make exactly 3 seconds
    # --------------------------------------------------

    target_length = int(3 * sr)

    if len(negative) >= target_length:

        # Enough negative audio already
        negative = negative[:target_length]

    else:

        # Repeat available negative audio until >= 3 sec
        if len(negative) == 0:
            print(f"WARNING: No negative audio available for selec {selec}")
            continue

        repetitions = int(np.ceil(target_length / len(negative)))

        negative = np.tile(negative, repetitions)

        # Exactly 3 seconds
        negative = negative[:target_length]

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    output_file = output_dir / f"{selec}_neg.wav"

    sf.write(output_file, negative, sr)

    print(
        f"{selec}: "
        f"positive={t_end-t_start:.3f}s → "
        f"saved {output_file.name}"
    )

print("\nDone.")
