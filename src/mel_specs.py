
import argparse
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


# ==========================================
# Configuration
# ==========================================

TARGET_SR = 44100

MAX_DURATION = 16.0  # Maximum audio length in seconds

N_FFT = 4096
HOP_LENGTH = 512
N_MELS = 256

FMIN = 20
FMAX = 20000

DB_MIN = -80
DB_MAX = 0

FIGURE_DPI = 180


def create_mel_spectrogram(audio_path, output_dir):

    audio_path = Path(audio_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing: {audio_path.name}")

    # Load audio at original sample rate
    y, original_sr = librosa.load(
        audio_path,
        sr=None,
        mono=True
    )

    # Check original sample rate
    if original_sr < 40000:
        raise ValueError(
            f"Sample rate {original_sr} Hz is too low "
            "to represent frequencies up to 20 kHz."
        )

    # Limit audio to first 16 seconds
    max_samples = int(MAX_DURATION * original_sr)
    y = y[:max_samples]

    duration = len(y) / original_sr

    # Resample to target sample rate
    if original_sr != TARGET_SR:
        y = librosa.resample(
            y,
            orig_sr=original_sr,
            target_sr=TARGET_SR
        )

    sr = TARGET_SR

    print(f"Original sample rate: {original_sr} Hz")
    print(f"Processed duration: {duration:.2f} seconds")

    # ======================================
    # Calculate Mel spectrogram
    # ======================================

    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0
    )

    mel_spec_db = librosa.power_to_db(
        mel_spec,
        ref=np.max
    )

    # ======================================
    # Plot spectrogram
    # ======================================

    fig, ax = plt.subplots(figsize=(14, 7))

    img = librosa.display.specshow(
        mel_spec_db,
        sr=sr,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="mel",
        fmin=FMIN,
        fmax=FMAX,
        cmap="inferno",
        vmin=DB_MIN,
        vmax=DB_MAX,
        ax=ax
    )

    ax.set_title(
        f"Mel Spectrogram - {audio_path.stem}",
        fontsize=14
    )

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Frequency (Hz)")

    fig.colorbar(
        img,
        ax=ax,
        format="%+2.0f dB",
        label="Power (dB)"
    )

    plt.tight_layout()

    # ======================================
    # Save image
    # ======================================

    output_path = output_dir / (
        audio_path.stem + "_mel.png"
    )

    plt.savefig(
        output_path,
        dpi=FIGURE_DPI,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved: {output_path}\n")


def main():

    parser = argparse.ArgumentParser(
        description="Generate Mel spectrograms from WAV and MP3 files"
    )

    parser.add_argument(
        "input",
        help="Directory containing audio files"
    )

    parser.add_argument(
        "-o",
        "--output",
        default="mel_images",
        help="Output directory"
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.is_dir():
        print("Please provide a valid input directory.")
        return

    # Find WAV and MP3 files
    audio_files = sorted(
        list(input_path.glob("*.wav")) +
        list(input_path.glob("*.mp3"))
    )

    if not audio_files:
        print("No WAV or MP3 files found.")
        return

    print(f"\nFound {len(audio_files)} audio files.")
    print(f"Maximum duration: {MAX_DURATION} seconds\n")

    successful = 0
    failed = 0

    for audio_file in audio_files:

        try:
            create_mel_spectrogram(
                audio_file,
                args.output
            )
            successful += 1

        except Exception as e:
            print(f"ERROR: {audio_file.name}: {e}\n")
            failed += 1

    print("=" * 45)
    print("PROCESSING COMPLETE")
    print(f"Successful: {successful}")
    print(f"Failed:     {failed}")
    print(f"Total:      {len(audio_files)}")
    print("=" * 45)


if __name__ == "__main__":
    main()
