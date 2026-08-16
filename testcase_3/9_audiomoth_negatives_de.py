from pathlib import Path
import librosa
import soundfile as sf
import numpy as np

# --------------------------------------------------
# Settings
# --------------------------------------------------

INPUT_DIR = Path("audiomoth_recordings_DE")
OUTPUT_DIR = Path("audiomoth_negatives_DE")

SEGMENT_LENGTH = 3.0
MAX_SEGMENTS = 8

OUTPUT_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# Process files
# --------------------------------------------------

mp3_files = sorted(INPUT_DIR.glob("*.mp3"))

print(f"Found {len(mp3_files)} MP3 files")

for file_idx, mp3_file in enumerate(mp3_files, start=1):

    recording_id = mp3_file.stem

    print(
        f"\n[{file_idx}/{len(mp3_files)}] "
        f"{mp3_file.name}"
    )

    try:
        # Load audio
        audio, sr = librosa.load(
            mp3_file,
            sr=48000,
            mono=True
        )

    except Exception as e:
        print(f"  ERROR reading file: {e}")
        continue

    duration = len(audio) / sr

    print(f"  Duration: {duration:.1f} s")
    print(f"  Sample rate: {sr} Hz")

    # Need at least 3 seconds
    if duration < SEGMENT_LENGTH:
        print("  Too short -- skipping")
        continue

    # Maximum number of complete 3-sec segments
    possible_segments = int(duration // SEGMENT_LENGTH)

    n_segments = min(
        MAX_SEGMENTS,
        possible_segments
    )

    # Choose evenly distributed segment positions
    if n_segments == 1:
        starts = [0]
    else:
        max_start = duration - SEGMENT_LENGTH

        starts = np.linspace(
            0,
            max_start,
            n_segments
        )

    # --------------------------------------------------
    # Extract segments
    # --------------------------------------------------

    for segment_idx, start_time in enumerate(starts, start=1):

        start_sample = int(start_time * sr)
        end_sample = start_sample + int(SEGMENT_LENGTH * sr)

        segment = audio[start_sample:end_sample]

        # Make sure segment is exactly 3 seconds
        if len(segment) != int(SEGMENT_LENGTH * sr):
            continue

        output_file = OUTPUT_DIR / (
            f"{recording_id}_{segment_idx}.wav"
        )

        sf.write(
            output_file,
            segment,
            sr,
            subtype="PCM_16"
        )

        print(
            f"  Saved: {output_file.name} "
            f"({start_time:.1f} s)"
        )

print("\nDone.")