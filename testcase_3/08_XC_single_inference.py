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

ROOT_FOLDER = "../data/test_data_3/embedding/"
DATASET = "ds1"

# ---------------------------------------------------------------------
# Load test data
# ---------------------------------------------------------------------

df_pos = get_embedding_birdnet(ROOT_FOLDER, 1)
df_neg = get_embedding_birdnet(ROOT_FOLDER, 0)

df_pos["target"] = 1
df_neg["target"] = 0

df_test = pd.concat([df_pos, df_neg], ignore_index=True)

X_test = np.vstack(df_test["embedding"].values)
y_test = df_test["target"].values

# ---------------------------------------------------------------------
# Evaluate saved models
# ---------------------------------------------------------------------

accuracies = []
aps = []
best_f1s = []
best_thresholds = []


model = joblib.load(
    f"../notebooks/rfc_model_negfold_1.joblib"
)

y_pred = model.predict(X_test)
y_score = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
ap = average_precision_score(y_test, y_score)
cm = confusion_matrix(y_test, y_pred)

# Precision-Recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_score)

# F1 score for each threshold
f1 = 2 * precision * recall / (precision + recall + 1e-12)

best_idx = np.argmax(f1[:-1])  # last value has no threshold

best_f1 = f1[best_idx]
best_threshold = thresholds[best_idx]

accuracies.append(acc)
aps.append(ap)
best_f1s.append(best_f1)
best_thresholds.append(best_threshold)


print(f"Accuracy      : {acc:.4f}")
print(f"AP            : {ap:.4f}")
print(f"Best F1       : {best_f1:.4f}")
print(f"Best threshold: {best_threshold:.4f}")
print(cm)

# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

print("\n==========================")

print("Accuracy per model :", np.round(accuracies, 4))
print(
    f"Accuracy           : {np.mean(accuracies):.4f} ± {np.std(accuracies, ddof=1):.4f}"
)

print()

print("AP per model       :", np.round(aps, 4))
print(f"AP                 : {np.mean(aps):.4f} ± {np.std(aps, ddof=1):.4f}")

print()

print("Best F1 per model  :", np.round(best_f1s, 4))
print(
    f"Best F1            : {np.mean(best_f1s):.4f} ± {np.std(best_f1s, ddof=1):.4f}"
)

print()

print("Best thresholds    :", np.round(best_thresholds, 4))
print(f"Mean threshold     : {np.mean(best_thresholds):.4f}")