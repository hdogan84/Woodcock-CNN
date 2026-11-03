from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RFC
import xgboost as XGB
import matplotlib.pyplot as plt
import pandas as pd
import sys, os
from pathlib import Path
import numpy as np

from sklearn.metrics import (
    average_precision_score,
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# ============================================================
# SETTINGS
# ============================================================

N_FOLDS = 2

parent_folder = Path().resolve().parent
sys.path.append(str(parent_folder / "src"))
from tools import get_embedding_birdnet


# ============================================================
# LOAD DATA
# ============================================================

df_full = pd.read_csv("../testcase_3/selections.csv")

df_pos = pd.read_csv("../testcase_3/selecs_positive_test_samedate.csv")
df_pos["date"] = df_pos["sound.files"].str.split("_").str[0]
df_pos["recording.id"] = df_pos["recording.id"].astype(int)
df_pos["target"] = 1

preds = np.load('holderried_predictions_selections.npy', allow_pickle=True)
preds_pos = preds.item()
preds_pos = pd.DataFrame(preds_pos.items(),columns=["filename", "prediction"])
preds_pos["selec"] = preds_pos["filename"].str.replace(".wav", "", regex=False).astype(int)

df_pos = df_pos.merge(
    preds_pos[["selec", "prediction"]],
    on="selec",
    how="left"
)
df_pos = df_pos.dropna()

df_neg = get_embedding_birdnet("../data/test_data_3/embedding/", 0)
df_neg["selec"] = df_neg["embed_name"].str.split("_").str[0].astype(int)
df_neg["filename"] = df_neg["embed_name"].str.split(".", n=1).str[0]
df_neg = df_neg.merge(df_full, on="selec", how="left").dropna()
df_neg["date"] = df_neg["sound.files"].str.split("_").str[0]
df_neg["target"] = 0

preds = np.load('holderried_preds_testdata3_neg.npy', allow_pickle=True)
preds_neg = preds.item()
preds_neg = pd.DataFrame(preds_neg.items(),columns=["filename", "prediction"])
preds_neg["filename"] = preds_neg["filename"].str.replace(".wav", "", regex=False)


df_neg = df_neg.merge(
    preds_neg[["filename", "prediction"]],
    on="filename",
    how="left"
)




df_combined = pd.concat([df_pos, df_neg], ignore_index=True)
df_combined["date"] = df_combined["date"].astype(int)
df_combined = df_combined.sort_values(
    ["date", "recording.id", "selec"], na_position="last"
).reset_index(drop=True)


# ============================================================
# CHRONOLOGICAL DATE-BASED FOLDS
# ============================================================
# Each unique date stays completely within one fold.
# Fold sizes are made approximately equal by SAMPLE count.

date_counts = df_combined.groupby("date").size().sort_index()
dates = date_counts.index.to_numpy()
counts = date_counts.to_numpy()
cumulative = np.cumsum(counts)
total = len(df_combined)

# Find boundaries closest to equal sample counts,
# while never splitting a date.
boundaries = []
for i in range(1, N_FOLDS):
    target = total * i / N_FOLDS
    pos = np.argmin(np.abs(cumulative[:-1] - target)) + 1

    # Ensure every fold gets at least one date
    min_pos = boundaries[-1] + 1 if boundaries else 1
    max_pos = len(dates) - (N_FOLDS - i)
    pos = max(min_pos, min(pos, max_pos))
    boundaries.append(pos)

date_to_fold = {}
start = 0
for fold, end in enumerate(boundaries + [len(dates)]):
    for date in dates[start:end]:
        date_to_fold[date] = fold
    start = end

df_combined["fold"] = df_combined["date"].map(date_to_fold).astype(int)


# ============================================================
# CHECK SPLIT
# ============================================================

print("\n" + "=" * 60)
print("CHRONOLOGICAL DATE-BASED SPLIT")
print("=" * 60)

for fold in range(N_FOLDS):
    d = df_combined[df_combined["fold"] == fold]
    print(
        f"\nFold {fold}: "
        f"samples={len(d)}, "
        f"dates={d['date'].nunique()}, "
        f"range={d['date'].min()}-{d['date'].max()}, "
        f"positive={(d['target'] == 1).sum()}, "
        f"negative={(d['target'] == 0).sum()}"
    )

# Verify that no date occurs in multiple folds
assert df_combined.groupby("date")["fold"].nunique().max() == 1


# ============================================================
# FEATURES / LABELS / SPLITS
# ============================================================

X = df_combined["prediction"].values
y = df_combined["target"].astype(int).values

fold_splits = []

for fold in range(N_FOLDS):
    test_idx = np.where(df_combined["fold"].values == fold)[0]
    train_idx = np.where(df_combined["fold"].values != fold)[0]
    fold_splits.append((train_idx, test_idx))


# ============================================================
# MODEL EVALUATION
# ============================================================

os.makedirs("confusion_matrices", exist_ok=True)

threshold = 0.5

results = []
confusion_matrices = []
accuracies = []
aps = []
cms = []

for fold in sorted(df_combined["fold"].unique()):

    df_fold = df_combined[df_combined["fold"] == fold]

    y_true = df_fold["target"].values
    y_score = df_fold["prediction"].values
    y_pred = (y_score >= threshold).astype(int)

    # Metrics
    ap = average_precision_score(y_true, y_score)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0) 
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_score)

    cm = confusion_matrix(y_true, y_pred)
    confusion_matrices.append(cm)
    accuracies.append(accuracy)
    aps.append(ap)
    cms.append(cm)

    results.append({
        "fold": fold,
        "n_samples": len(df_fold),
        "n_positive": (y_true == 1).sum(),
        "n_negative": (y_true == 0).sum(),
        "average_precision": ap,
        "roc_auc": auc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    })

results_df = pd.DataFrame(results)

print(results_df)

results = {
        "model": "Baseline",
        "accuracies": accuracies,
        "aps": aps,
        "accuracy_mean": np.mean(accuracies),
        "accuracy_std": np.std(accuracies, ddof=1),
        "ap_mean": np.mean(aps),
        "ap_std": np.std(aps, ddof=1),
        "cms": cms
    }

   
# ============================================================
# SUMMARY
#

# ============================================================
# SAVE CONFUSION MATRIX PNGs
# ============================================================

def save_confusion_matrices(result, model_name):

    fig, axes = plt.subplots(
        1, N_FOLDS,
        figsize=(5 * N_FOLDS, 4)
    )

    axes = np.atleast_1d(axes)

    for i, cm in enumerate(result["cms"]):

        ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Negative", "Positive"]
        ).plot(
            ax=axes[i],
            colorbar=False
        )

        fold_dates = df_combined.loc[
            df_combined["fold"] == i, "date"
        ]

        axes[i].set_title(
            f"{model_name} - Fold {i + 1}\n"
            f"{fold_dates.min()} - {fold_dates.max()}"
        )

    plt.tight_layout()

    path = (
        f"confusion_matrices/"
        f"{model_name.lower()}_chronological_{N_FOLDS}_fold.png"
    )

    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


save_confusion_matrices(results, "Baseline")


