import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from scipy.interpolate import interp1d
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
)

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MODEL_PATH = "../models/BirdNET-Analyzer-V2.2/V2.2/BirdNET_GLOBAL_3K_V2.2_Model_FP32.tflite"
LABEL_PATH = "../models/BirdNET-Analyzer-V2.2/V2.2/BirdNET_GLOBAL_3K_V2.2_Labels.txt"

POS_FOLDER = "../data/Holderried/selections_wavs/1"
#POS_CSV = "selecs_positive_test.csv"
POS_CSV = "selecs_positive_test_samedate_samesite.csv"

NEG_FOLDER = "../data/test_data_3/audio/0"

OUTPUT_PREDICTIONS = "testset3_predictions.npy"
OUTPUT_F1 = "testset3_f1.npz"

TARGET_SPECIES = "Scolopax rusticola"

THRESHOLD = 0.5


# ---------------------------------------------------------------------
# Helper function: predict selected positive files from CSV
# ---------------------------------------------------------------------

def predict_csv_files(csv_file, folder, analyzer):
    """
    Predict BirdNET confidence for WAV files specified by the
    'selec' column in a CSV.

    Returns
    -------
    predictions : dict
        filename -> confidence
    labels : dict
        filename -> true label (1)
    """

    folder = Path(folder)

    df = pd.read_csv(csv_file)

    if "selec" not in df.columns:
        raise ValueError("CSV does not contain a 'selec' column.")

    predictions = {}
    labels = {}

    for selec in df["selec"].dropna():

        # Convert e.g. 8050.0 -> 8050
        selec = int(selec)

        filename = f"{selec}.wav"
        audio_path = folder / filename

        if not audio_path.exists():
            print(f"WARNING: File not found: {audio_path}")
            continue

        print(f"Analyzing positive: {filename}")

        recording = Recording(
            analyzer,
            audio_path,
            min_conf=0.0,
        )

        recording.analyze()

        confidence = 0.0

        for detection in recording.detections:
            if detection["scientific_name"] == TARGET_SPECIES:
                confidence = max(
                    confidence,
                    detection["confidence"]
                )

        predictions[filename] = confidence
        labels[filename] = 1

    return predictions, labels


# ---------------------------------------------------------------------
# Helper function: predict all files in negative folder
# ---------------------------------------------------------------------

def predict_folder(folder, analyzer):
    """
    Predict BirdNET confidence for all WAV files in a folder.

    Returns
    -------
    predictions : dict
        filename -> confidence
    labels : dict
        filename -> true label (0)
    """

    folder = Path(folder)
    label = int(folder.name)

    predictions = {}
    labels = {}

    wav_files = sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith(".wav")
    )

    for file in wav_files:

        audio_path = folder / file

        recording = Recording(
            analyzer,
            audio_path,
            min_conf=0.0,
        )

        recording.analyze()

        confidence = 0.0

        for detection in recording.detections:
            if detection["scientific_name"] == TARGET_SPECIES:
                confidence = max(
                    confidence,
                    detection["confidence"]
                )

        predictions[file] = confidence
        labels[file] = label

    return predictions, labels


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    analyzer = Analyzer(
        classifier_model_path=MODEL_PATH,
        classifier_labels_path=LABEL_PATH,
    )

    # -------------------------------------------------------------
    # Positive predictions: ONLY files listed in CSV
    # -------------------------------------------------------------

    print("Running predictions on positive segments from CSV...")

    pred_pos, labels_pos = predict_csv_files(
        POS_CSV,
        POS_FOLDER,
        analyzer,
    )

    # -------------------------------------------------------------
    # Negative predictions: ALL WAV files in negative folder
    # -------------------------------------------------------------

    print("Running predictions on negative segments...")

    pred_neg, labels_neg = predict_folder(
        NEG_FOLDER,
        analyzer,
    )

    # -------------------------------------------------------------
    # Combine
    # -------------------------------------------------------------

    predictions = {**pred_pos, **pred_neg}
    labels = {**labels_pos, **labels_neg}

    np.save(OUTPUT_PREDICTIONS, predictions)

    sorted_keys = sorted(predictions.keys())

    y_pred = np.array([
        predictions[k]
        for k in sorted_keys
    ])

    y_true = np.array([
        labels[k]
        for k in sorted_keys
    ])

    # -------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------

    ap = average_precision_score(
        y_true,
        y_pred
    )

    y_pred_binary = (
        y_pred >= THRESHOLD
    ).astype(int)

    acc = accuracy_score(
        y_true,
        y_pred_binary
    )

    print(f"\nAverage Precision : {ap:.4f}")
    print(f"Accuracy          : {acc:.4f}")

    pos = y_pred[y_true == 1]
    neg = y_pred[y_true == 0]

    print(
        f"Positive range: "
        f"{pos.min():.4f} - {pos.max():.4f}"
    )

    print(
        f"Negative range: "
        f"{neg.min():.4f} - {neg.max():.4f}"
    )

    print(f"Positive files analyzed: {len(pos)}")
    print(f"Negative files analyzed: {len(neg)}")

    # -------------------------------------------------------------
    # Precision-Recall curve
    # -------------------------------------------------------------

    precision, recall, thr = precision_recall_curve(
        y_true,
        y_pred
    )

    # -------------------------------------------------------------
    # F1
    # -------------------------------------------------------------

    f1 = (
        2 * precision * recall /
        (precision + recall + 1e-12)
    )

    print(f"Best F1: {f1.max():.4f}")

    thr_uniform = np.linspace(
        thr.min(),
        thr.max(),
        100
    )

    interpolator = interp1d(
        thr,
        f1[:-1],
        kind="linear"
    )

    f1_uniform = interpolator(thr_uniform)

    np.savez(
        OUTPUT_F1,
        thr=thr_uniform,
        f1=f1_uniform,
    )

    # -------------------------------------------------------------
    # Confusion Matrix
    # -------------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred_binary
    )

    fig, ax = plt.subplots(
        figsize=(5, 5)
    )

    im = ax.imshow(cm)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels([
        "Negative",
        "Positive"
    ])

    ax.set_yticklabels([
        "Negative",
        "Positive"
    ])

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    ax.set_title(
        f"Confusion Matrix "
        f"(threshold = {THRESHOLD})"
    )

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                fontsize=14
            )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()