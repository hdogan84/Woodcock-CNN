from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def predict_directory(
    audio_dir: Path,
    output_dir: Path,
    species: str,
    analyzer: Analyzer,
) -> dict:
    """Run BirdNET on all audio files in a directory."""

    if not audio_dir.is_dir():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    audio_files = sorted(
        f for f in audio_dir.iterdir()
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
    )

    if not audio_files:
        raise ValueError(f"No audio files found in {audio_dir}")

    predictions = {}

    print(f"\nProcessing {len(audio_files)} audio files")
    print(f"Input:  {audio_dir}")
    print(f"Output: {output_dir}")
    print(f"Species: {species}\n")

    for i, audio_file in enumerate(audio_files, start=1):

        recording = Recording(
            analyzer,
            audio_file,
            min_conf=0.0,
        )

        recording.analyze()

        # Extract the maximum confidence for the target species.
        confidence = max(
            (
                detection["confidence"]
                for detection in recording.detections
                if detection["scientific_name"] == species
            ),
            default=0.0,
        )

        predictions[audio_file.name] = confidence

        print(
            f"[{i}/{len(audio_files)}] "
            f"{audio_file.name}: {confidence:.4f}"
        )

    # Save predictions in NumPy format (backwards compatible).
    np.save(output_dir / "predictions.npy", predictions)

    # Also save as CSV for convenient inspection.
    pd.DataFrame(
        predictions.items(),
        columns=["filename", "confidence"],
    ).to_csv(output_dir / "predictions.csv", index=False)

    print(f"\nPredictions saved to {output_dir}")

    return predictions


def main():

    parser = argparse.ArgumentParser(
        description="Run BirdNET predictions on an audio dataset."
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Root directory containing the audio folder.",
    )

    parser.add_argument(
        "--class-id",
        type=str,
        required=True,
        help="Class folder to process (e.g. 0 or 1).",
    )

    parser.add_argument(
        "--species",
        type=str,
        default="Scolopax rusticola",
        help="Target species (scientific name).",
    )

    args = parser.parse_args()

    data_dir = args.data_dir

    # Automatically construct input and output paths.
    audio_dir = data_dir / "audio" / args.class_id
    output_dir = data_dir / "birdnet_probs" / args.class_id

    # Initialize BirdNET once.
    print("Initializing BirdNET...")
    analyzer = Analyzer()

    predict_directory(
        audio_dir=audio_dir,
        output_dir=output_dir,
        species=args.species,
        analyzer=analyzer,
    )


if __name__ == "__main__":
    main()
