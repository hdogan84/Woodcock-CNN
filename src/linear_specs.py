
import argparse
from pathlib import Path

import librosa
import numpy as np
import matplotlib.pyplot as plt


# ==========================================
# Configuration: Publication-quality figures
# ==========================================

TARGET_SR = 44100
MAX_DURATION = 16.0

N_FFT = 2048
HOP_LENGTH = 128

FMIN = 20
FMAX = 20000

DB_MIN = -80
DB_MAX = 0

FIGURE_WIDTH = 7
FIGURE_HEIGHT = 4
FIGURE_DPI = 600

TITLE_SIZE = 15
LABEL_SIZE = 14
TICK_SIZE = 12
COLORBAR_SIZE = 12


def create_linear_spectrogram(audio_path, output_dir):

    audio_path = Path(audio_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing: {audio_path.name}")

    # Load audio
    y, original_sr = librosa.load(
        audio_path,
        sr=None,
        mono=True
    )

    if original_sr < 40000:
        raise ValueError(
            f"Sample rate {original_sr} Hz is too low "
            "for a 20 kHz upper frequency."
        )

    # Limit duration
    y = y[:int(MAX_DURATION * original_sr)]

    duration = len(y) / original_sr

    # Resample
    if original_sr != TARGET_SR:
        y = librosa.resample(
            y,
            orig_sr=original_sr,
            target_sr=TARGET_SR
        )

    sr = TARGET_SR

    print(f"Duration: {duration:.2f} seconds")

    # ======================================
    # STFT
    # ======================================

    stft = librosa.stft(
        y,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        window="hann"
    )

    spectrogram = np.abs(stft) ** 2

    spectrogram_db = librosa.power_to_db(
        spectrogram,
        ref=np.max
    )

    frequencies = librosa.fft_frequencies(
        sr=sr,
        n_fft=N_FFT
    )

    freq_mask = (
        (frequencies >= FMIN) &
        (frequencies <= FMAX)
    )

    spectrogram_db = spectrogram_db[freq_mask, :]
    frequencies = frequencies[freq_mask]

    times = librosa.frames_to_time(
        np.arange(spectrogram_db.shape[1]),
        sr=sr,
        hop_length=HOP_LENGTH
    )

    # ======================================
    # Publication-quality figure
    # ======================================

    fig, ax = plt.subplots(
        figsize=(FIGURE_WIDTH, FIGURE_HEIGHT)
    )

    img = ax.pcolormesh(
        times,
        frequencies/1000,
        spectrogram_db,
        cmap="inferno",
        vmin=DB_MIN,
        vmax=DB_MAX,
        shading="auto",
        rasterized=True
    )

    ax.set_xlim(0, duration)
    ax.set_ylim(FMIN / 1000, FMAX / 1000)

    ax.set_xlabel(
        "Time (s)",
        fontsize=LABEL_SIZE,
        fontweight="bold"
    )

    ax.set_ylabel(
        "Frequency (kHz)",
        fontsize=LABEL_SIZE,
        fontweight="bold"
    )

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_SIZE,
        width=1.2,
        length=5
    )

    # Colorbar
    cbar = fig.colorbar(
        img,
        ax=ax,
        pad=0.02
    )

    cbar.set_label(
        "Power (dB)",
        fontsize=COLORBAR_SIZE,
        fontweight="bold"
    )

    cbar.ax.tick_params(
        labelsize=TICK_SIZE,
        width=1.2,
        length=4
    )

    # Axis borders
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    # No title: suitable for paper figures
    # Add figure labels (a), (b), etc. in paper layout instead.

    fig.tight_layout()

    # ======================================
    # Save
    # ======================================

    output_path = output_dir / (
        audio_path.stem + "_linear.png"
    )

    fig.savefig(
        output_path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(fig)

    print(f"Saved: {output_path}\n")


def main():

    parser = argparse.ArgumentParser(
        description="Generate publication-quality linear spectrograms"
    )

    parser.add_argument(
        "input",
        help="Directory containing WAV and MP3 files"
    )

    parser.add_argument(
        "-o",
        "--output",
        default="linear_images",
        help="Output directory"
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.is_dir():
        print("Please provide a valid input directory.")
        return

    audio_files = sorted(
        list(input_path.glob("*.wav")) +
        list(input_path.glob("*.mp3"))
    )

    if not audio_files:
        print("No WAV or MP3 files found.")
        return

    print(f"\nFound {len(audio_files)} files.\n")

    successful = 0
    failed = 0

    for audio_file in audio_files:

        try:
            create_linear_spectrogram(
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
