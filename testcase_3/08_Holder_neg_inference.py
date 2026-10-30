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

MODEL_DIR = Path("../models/negative_subset_svc/")
ROOT_FOLDER = "../data/test_data_3/embedding/"


# ---------------------------------------------------------------------
# Load test negatives
# ---------------------------------------------------------------------

df_neg = get_embedding_birdnet(ROOT_FOLDER, 0)
df_neg["target"] = 0

print(f"Test negatives: {len(df_neg)}")


# ---------------------------------------------------------------------
# Load held-out positives
# ---------------------------------------------------------------------

ROOT_FOLDER = "../data/train_data/embedding/holderried/"

df_pos = get_embedding_birdnet(ROOT_FOLDER, 1)

df_pos["selec"] = (
    df_pos["embed_name"]
    .str.extract(r"^(\d+)")
    .astype(int)
)

df_pos["target"] = 1


# ---------------------------------------------------------------------
# Select held-out positive recordings
# ---------------------------------------------------------------------

# df_heldout_pos = pd.read_csv(
#     "selecs_positive_test_samedate_samesite.csv"
# )

df_heldout_pos = pd.read_csv(
    "selecs_positive_test_samedate.csv"
)

selec_ids = df_heldout_pos["selec"].unique()

df_pos = df_pos[
    df_pos["selec"].isin(selec_ids)
].copy()

print(f"Test positives: {len(df_pos)}")


# ---------------------------------------------------------------------
# Combine test data
# ---------------------------------------------------------------------

df_test = pd.concat(
    [df_pos, df_neg],
    ignore_index=True
)

X_test = np.vstack(
    df_test["embedding"].values
)

y_test = df_test["target"].values

print(f"Total test samples: {len(df_test)}")
print(f"X_test shape: {X_test.shape}")


# ---------------------------------------------------------------------
# Models to evaluate
# ---------------------------------------------------------------------

model_files = [
    "svc_neg1_seed3.joblib",
    "svc_neg2_xc.joblib",
    "svc_neg3_holderried.joblib",
    "svc_neg4_audiomoth_DE.joblib",
    "svc_neg5_audiomoth_Eurasian.joblib",
    "svc_neg6_baden_baden.joblib",
    "svc_neg7_redmi.joblib",
]


# ---------------------------------------------------------------------
# Evaluate every model
# ---------------------------------------------------------------------

results = []

for model_file in model_files:

    print("\n" + "=" * 70)
    print(f"MODEL: {model_file}")
    print("=" * 70)

    model_path = MODEL_DIR / model_file

    model = joblib.load(model_path)

    # -------------------------------------------------------------
    # Predictions
    # -------------------------------------------------------------

    y_pred = model.predict(X_test)

    y_score = model.predict_proba(X_test)[:, 1]

    # -------------------------------------------------------------
    # Standard metrics
    # -------------------------------------------------------------

    acc = accuracy_score(
        y_test,
        y_pred
    )

    ap = average_precision_score(
        y_test,
        y_score
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    # -------------------------------------------------------------
    # Precision-recall curve
    # -------------------------------------------------------------

    precision, recall, thresholds = precision_recall_curve(
        y_test,
        y_score
    )

    # F1 for each threshold
    f1 = (
        2 * precision * recall
        / (precision + recall + 1e-12)
    )

    # Last precision/recall value has no corresponding threshold
    best_idx = np.argmax(f1[:-1])

    best_f1 = f1[best_idx]
    best_threshold = thresholds[best_idx]

    # -------------------------------------------------------------
    # Store results
    # -------------------------------------------------------------

    results.append({
        "model": model_file,
        "accuracy": acc,
        "average_precision": ap,
        "best_f1": best_f1,
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
# Summary
# ---------------------------------------------------------------------

results_df = pd.DataFrame(results)

print("\n\n")
print("=" * 90)
print("SUMMARY")
print("=" * 90)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)