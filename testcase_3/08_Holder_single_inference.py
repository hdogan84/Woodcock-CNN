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

# --------------
# Env to use: CLEF

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

model_type = "xgb"

ROOT_FOLDER = "../data/test_data_3/embedding/"
df_neg = get_embedding_birdnet(ROOT_FOLDER, 0)
df_neg["target"] = 0

# ---------------------------------------------------------------------
# Load test data - Positives
# ---------------------------------------------------------------------
ROOT_FOLDER = "../data/train_data/embedding/holderried/"
df_pos = get_embedding_birdnet(ROOT_FOLDER, 1)

df_pos["selec"] = (
    df_pos["embed_name"]
    .str.extract(r"^(\d+)")
    .astype(int)
)

df_pos["target"] = 1

# You need to slice out for the held-out part of the positives 
#df_heldout_pos = pd.read_csv("selecs_positive_test_samedate_samesite.csv") 
df_heldout_pos = pd.read_csv("selecs_positive_test_samedate.csv") 
#df_heldout_pos = pd.read_csv("selecs_positive_test.csv") 

selec_ids = df_heldout_pos["selec"].unique()
df_pos = df_pos[df_pos["selec"].isin(selec_ids)]


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


model = joblib.load(f"../Holderried/full_models_backup/{model_type}_model.joblib")

y_pred = model.predict(X_test)
y_score = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
ap = average_precision_score(y_test, y_score)
cm = confusion_matrix(y_test, y_pred)

# Precision-Recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_score)

# F1 score for each threshold
f1 = 2 * precision * recall / (precision + recall + 1e-12)

best_idx = np.argmax(f1[:-1])

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

