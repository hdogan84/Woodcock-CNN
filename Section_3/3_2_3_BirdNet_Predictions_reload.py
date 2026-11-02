import matplotlib.pyplot as plt
import numpy as np
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

INPUT_RESULTS = "testset3_results_samedate.npz"
OUTPUT_F1 = "testset3_f1_samedate.npz"

THRESHOLD = 0.5


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    # -------------------------------------------------------------
    # Load saved predictions and labels
    # -------------------------------------------------------------

    print(f"Loading results from: {INPUT_RESULTS}")

    data = np.load(
        INPUT_RESULTS,
        allow_pickle=True
    )

    predictions = data["predictions"].item()
    labels = data["labels"].item()

    print(f"Loaded predictions for {len(predictions)} files.")
    print(f"Loaded labels for {len(labels)} files.")

    # -------------------------------------------------------------
    # Check that predictions and labels match
    # -------------------------------------------------------------

    prediction_files = set(predictions.keys())
    label_files = set(labels.keys())

    missing_labels = prediction_files - label_files
    missing_predictions = label_files - prediction_files

    if missing_labels:
        print("\nWARNING: Predictions without labels:")
        for f in sorted(missing_labels):
            print(f"  {f}")

    if missing_predictions:
        print("\nWARNING: Labels without predictions:")
        for f in sorted(missing_predictions):
            print(f"  {f}")

    common_files = sorted(
        prediction_files.intersection(label_files)
    )

    if len(common_files) == 0:
        raise RuntimeError(
            "No matching filenames found between "
            "predictions and labels."
        )

    print(f"Using {len(common_files)} files.")

    # -------------------------------------------------------------
    # Construct arrays
    # -------------------------------------------------------------

    y_pred = np.array(
        [predictions[k] for k in common_files]
    )

    y_true = np.array(
        [labels[k] for k in common_files]
    )

    # -------------------------------------------------------------
    # Dataset information
    # -------------------------------------------------------------

    print("\nDataset:")
    print(f"  Total samples    : {len(y_true)}")
    print(f"  Positive samples : {np.sum(y_true == 1)}")
    print(f"  Negative samples : {np.sum(y_true == 0)}")

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

    cm = confusion_matrix(
        y_true,
        y_pred_binary
    )

    print("\nMetrics:")
    print(f"Average Precision : {ap:.4f}")
    print(f"Accuracy          : {acc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    # -------------------------------------------------------------
    # Positive / negative confidence ranges
    # -------------------------------------------------------------

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
        2 * precision * recall
        / (precision + recall + 1e-12)
    )

    print(
        f"\nBest F1: {f1.max():.4f}"
    )

    # -------------------------------------------------------------
    # Best F1 threshold
    # -------------------------------------------------------------

    if len(thr) > 0:

        best_index = np.argmax(f1[:-1])
        best_threshold = thr[best_index]

        print(
            f"Best F1 threshold: "
            f"{best_threshold:.6f}"
        )


    # -------------------------------------------------------------
    # Interpolated F1 curve
    # -------------------------------------------------------------

    if len(thr) > 1:

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

        f1_uniform = interpolator(
            thr_uniform
        )

        np.savez(
            OUTPUT_F1,
            thr=thr_uniform,
            f1=f1_uniform,
        )

        print(
            f"\nSaved interpolated F1 curve to: "
            f"{OUTPUT_F1}"
        )

        # ---------------------------------------------------------
        # Plot interpolated F1
        # ---------------------------------------------------------

       


if __name__ == "__main__":
    main()