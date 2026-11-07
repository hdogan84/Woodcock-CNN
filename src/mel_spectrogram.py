
import argparse
from pathlib import Path
import warnings

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


# ==========================================
# Configuration
# ==========================================

TARGET_SR = 44100

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

    # Load audio, preserving original sample rate
    y, original_sr = librosa.load(
        audio_path,
        sr=None,
        mono=True
    )

    # Check whether 20 kHz can be represented
    if original_sr < 40000:
        raise ValueError(
            f"Sample rate {original_sr} Hz is too low "
            "to represent frequencies up to 20 kHz."
        )

    # Resample to standard sample rate
    if original_sr != TARGET_SR:
        y = librosa.resample(
            y,
            orig_sr=original_sr,
            target_sr=TARGET_SR
        )

    sr = TARGET_SR

    duration = len(y) / sr

    print(f"Duration: {duration:.2f} seconds")
    print(f"Sample rate: {sr} Hz")

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

    # Convert power to decibels
    mel_spec_db = librosa.power_to_db(
        mel_spec,
        ref=np.max
    )

    # ======================================
    # Plot spectrogram
    # ======================================

    # Increase image width for longer recordings
    figure_width = max(14, min(36, duration * 0.3))

    fig, ax = plt.subplots(
        figsize=(figure_width, 7)
    )

    img = librosa.display.specshow(
        mel_spec_db,
        sr=sr,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="mel",
        fmin=FMIN,
        fmax=FMAX,
        cmap="magma",
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
        description="Generate Mel spectrogram images"
    )

    parser.add_argument(
        "input",
        help="Audio file or directory containing audio files"
    )

    parser.add_argument(
        "-o",
        "--output",
        default="mel_images",
        help="Output directory (default: mel_images)"
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_file():

        create_mel_spectrogram(
            input_path,
            args.output
        )

    elif input_path.is_dir():

        audio_files = (
            list(input_path.glob("*.wav")) +
            list(input_path.glob("*.mp3"))
        )

        if not audio_files:
            print("No WAV or MP3 files found.")
            return

        for audio_file in audio_files:

            try:
                create_mel_spectrogram(
                    audio_file,
                    args.output
                )

            except Exception as e:
                print(f"Error processing {audio_file.name}: {e}")

    else:
        print("Input path does not exist.")


if __name__ == "__main__":
    main()
