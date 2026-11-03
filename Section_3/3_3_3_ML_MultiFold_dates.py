
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RFC
import xgboost as XGB
import matplotlib.pyplot as plt
import pandas as pd
import sys, os
from pathlib import Path
import numpy as np

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

df_neg = get_embedding_birdnet("../data/test_data_3/embedding/", 0)
df_neg["selec"] = df_neg["embed_name"].str.split("_").str[0].astype(int)
df_neg["filename"] = df_neg["embed_name"].str.split(".", n=1).str[0]
df_neg = df_neg.merge(df_full, on="selec", how="left").dropna()
df_neg["date"] = df_neg["sound.files"].str.split("_").str[0]
df_neg["target"] = 0

df_train = pd.read_csv("../testcase_3/selecs_positive_test_samedate.csv")

df_pos = get_embedding_birdnet("../data/train_data/embedding/holderried/", 1)
df_pos["selec"] = df_pos["embed_name"].str.extract(r"^(\d+)").astype(int)
df_pos = df_pos.merge(
    df_train[["selec", "sound.files", "recording.id"]],
    on="selec",
    how="left"
).dropna()
df_pos["date"] = df_pos["sound.files"].str.split("_").str[0]
df_pos["recording.id"] = df_pos["recording.id"].astype(int)
df_pos["target"] = 1

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

X = np.vstack(df_combined["embedding"].values)
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


def evaluate_model(model, model_name, score_method):

    accuracies = []
    aps = []
    cms = []

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    for fold, (train_idx, test_idx) in enumerate(fold_splits, 1):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        if score_method == "decision_function":
            y_score = model.decision_function(X_test)
        else:
            y_score = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        ap = average_precision_score(y_test, y_score)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

        accuracies.append(accuracy)
        aps.append(ap)
        cms.append(cm)

        print(f"\nFold {fold}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  AP:       {ap:.4f}")
        print("  Confusion matrix:")
        print(cm)

    print("\nPer-fold accuracy:", np.round(accuracies, 4))
    print(
        f"Accuracy: {np.mean(accuracies):.4f} ± "
        f"{np.std(accuracies, ddof=1):.4f}"
    )

    print("\nPer-fold AP:", np.round(aps, 4))
    print(
        f"AP:       {np.mean(aps):.4f} ± "
        f"{np.std(aps, ddof=1):.4f}"
    )

    return {
        "model": model_name,
        "accuracies": accuracies,
        "aps": aps,
        "accuracy_mean": np.mean(accuracies),
        "accuracy_std": np.std(accuracies, ddof=1),
        "ap_mean": np.mean(aps),
        "ap_std": np.std(aps, ddof=1),
        "cms": cms
    }


# ============================================================
# MODELS
# ============================================================

svc_results = evaluate_model(
    SVC(kernel="rbf", cache_size=500),
    "SVC",
    "decision_function"
)

rfc_results = evaluate_model(
    RFC(n_jobs=-1, random_state=24),
    "RFC",
    "predict_proba"
)

xgb_results = evaluate_model(
    XGB.XGBClassifier(
        objective="binary:logistic",
        random_state=24,
        n_jobs=-1
    ),
    "XGB",
    "predict_proba"
)


# ============================================================
# SUMMARY
# ============================================================

results = pd.DataFrame([
    {
        "model": r["model"],
        "accuracy_fold_1": r["accuracies"][0],
        "accuracy_fold_2": r["accuracies"][1],
        "accuracy_mean ± std":
            f"{r['accuracy_mean']:.4f} ± {r['accuracy_std']:.4f}",
        "AP_fold_1": r["aps"][0],
        "AP_fold_2": r["aps"][1],
        "AP_mean ± std":
            f"{r['ap_mean']:.4f} ± {r['ap_std']:.4f}"
    }
    for r in [svc_results, rfc_results, xgb_results]
])

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)
print(results.to_string(index=False))


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


save_confusion_matrices(svc_results, "SVC")
save_confusion_matrices(rfc_results, "RFC")
save_confusion_matrices(xgb_results, "XGB")

