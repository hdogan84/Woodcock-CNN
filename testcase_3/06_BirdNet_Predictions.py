import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
NEG_FOLDER = "../data/test_data_3/audio/0"

OUTPUT_PREDICTIONS = "testset3_predictions.npy"
OUTPUT_F1 = "testset3_f1.npz"

TARGET_SPECIES = "Scolopax rusticola"

THRESHOLD = 0.5


# ---------------------------------------------------------------------
# Helper function
# ---------------------------------------------------------------------

def predict_folder(folder, analyzer):
    """
    Predict BirdNET confidence for all wav files in a folder.

    Returns
    -------
    predictions : dict
        filename -> confidence
    labels : dict
        filename -> true label (1 or 0 based on folder name)
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
                confidence = max(confidence, detection["confidence"])

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

    #analyzer = Analyzer()

    print("Running predictions on positive segments...")
    pred_pos, labels_pos = predict_folder(POS_FOLDER, analyzer)

    print("Running predictions on negative segments...")
    pred_neg, labels_neg = predict_folder(NEG_FOLDER, analyzer)

    predictions = {**pred_pos, **pred_neg}
    labels = {**labels_pos, **labels_neg}

    np.save(OUTPUT_PREDICTIONS, predictions)

    sorted_keys = sorted(predictions.keys())

    y_pred = np.array([predictions[k] for k in sorted_keys])
    y_true = np.array([labels[k] for k in sorted_keys])

    # -------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------

    ap = average_precision_score(y_true, y_pred)

    y_pred_binary = (y_pred >= THRESHOLD).astype(int)

    acc = accuracy_score(y_true, y_pred_binary)

    print(f"\nAverage Precision : {ap:.4f}")
    print(f"Accuracy          : {acc:.4f}")

    pos = y_pred[y_true == 1]
    neg = y_pred[y_true == 0]

    print(f"Positive range: {pos.min():.4f} - {pos.max():.4f}")
    print(f"Negative range: {neg.min():.4f} - {neg.max():.4f}")

    # -------------------------------------------------------------
    # Precision-Recall curve
    # -------------------------------------------------------------

    precision, recall, thr = precision_recall_curve(y_true, y_pred)



    # -------------------------------------------------------------
    # F1
    # -------------------------------------------------------------

    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    print(f"Best F1: {f1.max():.4f}")


    thr_uniform = np.linspace(thr.min(), thr.max(), 100)

    interpolator = interp1d(thr, f1[:-1], kind="linear")

    f1_uniform = interpolator(thr_uniform)

    np.savez(
        OUTPUT_F1,
        thr=thr_uniform,
        f1=f1_uniform,
    )

    # -------------------------------------------------------------
    # Confusion Matrix
    # -------------------------------------------------------------

    cm = confusion_matrix(y_true, y_pred_binary)

    fig, ax = plt.subplots(figsize=(5, 5))

    im = ax.imshow(cm)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Negative", "Positive"])
    ax.set_yticklabels(["Negative", "Positive"])

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix (threshold = {THRESHOLD})")

    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, cm[i, j],
                ha="center",
                va="center",
                fontsize=14
            )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()