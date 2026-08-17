import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
)

# ---------------------------------------------------------------------
# Add project source folder
# ---------------------------------------------------------------------

parent_folder = Path().resolve().parent
src_path = parent_folder / "src"
sys.path.append(str(src_path))

from tools import get_embedding_birdnet


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

ROOT_FOLDER = "../data/test_data/embedding/birdnet/"
DATASET = "ds1"

MODEL_FOLDER = "../notebooks/saved_models/"

classifiers = ["svc", "rfc", "xgb"]


# ---------------------------------------------------------------------
# Load test data
# ---------------------------------------------------------------------

df_pos = get_embedding_birdnet(ROOT_FOLDER, 1)
df_neg = get_embedding_birdnet(ROOT_FOLDER, 0)

df_pos["target"] = 1
df_neg["target"] = 0

df_test = pd.concat(
    [df_pos, df_neg],
    ignore_index=True
)

X_test = np.vstack(df_test["embedding"].values)
y_test = df_test["target"].values


# ---------------------------------------------------------------------
# Store results
# ---------------------------------------------------------------------

results = []


# ---------------------------------------------------------------------
# Evaluate all 15 models
# ---------------------------------------------------------------------

for fold in range(0, 5):

    for classifier in classifiers:

        model_path = (
            f"{MODEL_FOLDER}"
            f"{classifier}_model_negfold_{fold}.joblib"
        )

        print("\n" + "=" * 60)
        print(f"Fold: {fold} | Classifier: {classifier}")
        print(f"Loading: {model_path}")

        model = joblib.load(model_path)

        # -------------------------------------------------------------
        # Prediction
        # -------------------------------------------------------------

        y_pred = model.predict(X_test)

        # SVM uses decision_function.
        # RFC and XGB use predict_proba.
        if hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
        else:
            y_score = model.predict_proba(X_test)[:, 1]

        # -------------------------------------------------------------
        # Metrics
        # -------------------------------------------------------------

        acc = accuracy_score(y_test, y_pred)
        ap = average_precision_score(y_test, y_score)

        cm = confusion_matrix(y_test, y_pred)

        # -------------------------------------------------------------
        # Precision-Recall curve
        # -------------------------------------------------------------

        precision, recall, thresholds = precision_recall_curve(
            y_test,
            y_score
        )

        # F1 score for each threshold
        f1 = (
            2 * precision * recall
            / (precision + recall + 1e-12)
        )

        # Last precision/recall value has no corresponding threshold
        best_idx = np.argmax(f1[:-1])

        best_f1 = f1[best_idx]
        best_threshold = thresholds[best_idx]

        # -------------------------------------------------------------
        # Store result
        # -------------------------------------------------------------

        results.append({
            "fold": fold,
            "classifier": classifier,
            "accuracy": acc,
            "AP": ap,
            "best_F1": best_f1,
            "best_threshold": best_threshold,
        })

        # -------------------------------------------------------------
        # Print results
        # -------------------------------------------------------------

        print(f"Accuracy      : {acc:.4f}")
        print(f"AP            : {ap:.4f}")
        print(f"Best F1       : {best_f1:.4f}")
        print(f"Best threshold: {best_threshold:.4f}")
        print("Confusion matrix:")
        print(cm)


# ---------------------------------------------------------------------
# Results dataframe
# ---------------------------------------------------------------------

df_results = pd.DataFrame(results)


# ---------------------------------------------------------------------
# Print all results
# ---------------------------------------------------------------------

print("\n\n" + "=" * 80)
print("ALL RESULTS")
print("=" * 80)

print(
    df_results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ---------------------------------------------------------------------
# Summary by classifier
# ---------------------------------------------------------------------

print("\n\n" + "=" * 80)
print("SUMMARY BY CLASSIFIER")
print("=" * 80)

for classifier in classifiers:

    df_classifier = df_results[
        df_results["classifier"] == classifier
    ]

    print(f"\n===== {classifier.upper()} =====")

    for metric in [
        "accuracy",
        "AP",
        "best_F1",
        "best_threshold",
    ]:

        values = df_classifier[metric].values

        print(
            f"{metric:16s}: "
            f"{np.mean(values):.4f} ± "
            f"{np.std(values, ddof=1):.4f}"
        )


# ---------------------------------------------------------------------
# Summary by fold
# ---------------------------------------------------------------------

print("\n\n" + "=" * 80)
print("SUMMARY BY FOLD")
print("=" * 80)

for fold in range(1, 6):

    df_fold = df_results[
        df_results["fold"] == fold
    ]

    print(f"\n===== Fold {fold} =====")

    print(
        df_fold[
            [
                "classifier",
                "accuracy",
                "AP",
                "best_F1",
                "best_threshold",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )


# ---------------------------------------------------------------------
# Optional: save results
# ---------------------------------------------------------------------

df_results.to_csv(
    f"{MODEL_FOLDER}{DATASET}_all_15_inference_results.csv",
    index=False
)

print(
    f"\nResults saved to: "
    f"{MODEL_FOLDER}{DATASET}_all_15_inference_results.csv"
)