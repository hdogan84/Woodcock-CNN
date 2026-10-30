from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix
)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RFC
import xgboost as XGB
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import sys, os
from sklearn.model_selection import GroupKFold
from pathlib import Path
import numpy as np

parent_folder = Path().resolve().parent
src_path = parent_folder / 'src'
sys.path.append(str(src_path))

from tools import get_embedding_birdnet


# #env to use: clef

df_full = pd.read_csv("../testcase_3/selections.csv")


root_folder='../data/test_data_3/embedding/'
df_neg = get_embedding_birdnet(root_folder, 0)
df_neg["selec"] = df_neg["embed_name"].str.split("_").str[0].astype(int)

df_neg["filename"] = df_neg["embed_name"].str.split(".", n=1).str[0]

df_neg = df_neg.merge(
    df_full,
    on="selec",
    how="left"
)

df_neg = df_neg.dropna()

df_neg["date"] = df_neg["sound.files"].str.split("_").str[0]


df_train = pd.read_csv("../testcase_3/selecs_positive_test_samedate.csv")


root_folder='../data/train_data/embedding/holderried/'
df_pos = get_embedding_birdnet(root_folder, 1)

df_pos["selec"] = (
    df_pos["embed_name"]
    .str.extract(r"^(\d+)")
    .astype(int)
)

df_pos = df_pos.merge(
    df_train[["selec", "sound.files", "recording.id"]],
    on="selec",
    how="left"
)

df_pos["date"] = df_pos["sound.files"].str.split("_").str[0]

df_pos = df_pos.dropna() # this removes extra "selec" columns


df_pos["recording.id"] = df_pos["recording.id"].astype(int)

df_pos.head()

df_pos['target'] = 1
df_neg['target'] = 0



# ============================================================
# YEAR-BASED TWO-FOLD SPLIT
# ============================================================

# Combine positive and negative data
df_combined = pd.concat([df_pos, df_neg], ignore_index=True)

# Keep a deterministic ordering
df_combined = df_combined.sort_values(
    ["date", "recording.id", "selec"]
).reset_index(drop=True)

# Make sure date is numeric/integer-like
df_combined["date"] = df_combined["date"].astype(int)

# Features and labels
X = np.vstack(df_combined["embedding"].values)
y = df_combined["target"].astype(int).values

# ------------------------------------------------------------
# Explicit chronological folds
#
# Fold 0 = 2020
# Fold 1 = 2021
# ------------------------------------------------------------

df_combined["fold"] = -1

df_combined.loc[df_combined["date"] < 20210000, "fold"] = 0
df_combined.loc[df_combined["date"] >= 20210000, "fold"] = 1

# Check that only the expected years are present
print("\nDates by fold:")
print(
    df_combined.groupby("fold")["date"]
    .agg(["min", "max", "nunique"])
)

print("\nSamples per fold:")
print(df_combined["fold"].value_counts().sort_index())

print("\nTarget counts per fold:")
print(
    pd.crosstab(
        df_combined["fold"],
        df_combined["target"]
    )
)

# ------------------------------------------------------------
# Save fold assignment
# ------------------------------------------------------------

output_csv = "Testset_year_based_two_fold.csv"

df_combined[
    [
        "selec",
        "filename",
        "recording.id",
        "date",
        "target",
        "fold"
    ]
].to_csv(output_csv, index=False)

print(f"\nSaved fold information to: {output_csv}")


# ============================================================
# DEFINE THE TWO CHRONOLOGICAL SPLITS
# ============================================================

fold_splits = []

for fold in [0, 1]:

    test_idx = np.where(df_combined["fold"].values == fold)[0]
    train_idx = np.where(df_combined["fold"].values != fold)[0]

    fold_splits.append((train_idx, test_idx))

    print(
        f"\nFold {fold}: "
        f"train={len(train_idx)}, "
        f"test={len(test_idx)}"
    )


# ============================================================
# MODEL EVALUATION FUNCTION
# ============================================================

os.makedirs("saved_models", exist_ok=True)
os.makedirs("confusion_matrices", exist_ok=True)


def evaluate_model(model, model_name, score_method):

    accuracies = []
    aps = []
    confusion_matrices = []

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    for fold, (train_idx, test_idx) in enumerate(
        fold_splits,
        start=1
    ):

        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        # ----------------------------------------------------
        # Fit
        # ----------------------------------------------------

        model.fit(X_train, y_train)

        # ----------------------------------------------------
        # Save model
        # ----------------------------------------------------

        model_path = (
            f"saved_models/"
            f"{model_name.lower()}_year_fold_{fold}.joblib"
        )

        joblib.dump(model, model_path)

        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        y_pred = model.predict(X_test)

        if score_method == "decision_function":
            y_score = model.decision_function(X_test)

        elif score_method == "predict_proba":
            y_score = model.predict_proba(X_test)[:, 1]

        else:
            raise ValueError(
                f"Unknown score method: {score_method}"
            )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        accuracy = accuracy_score(y_test, y_pred)

        ap = average_precision_score(
            y_test,
            y_score
        )

        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1]
        )

        accuracies.append(accuracy)
        aps.append(ap)
        confusion_matrices.append(cm)

        # ----------------------------------------------------
        # Save confusion matrix as CSV
        # ----------------------------------------------------

        cm_csv = (
            f"confusion_matrices/"
            f"{model_name.lower()}_year_fold_{fold}.csv"
        )

        np.savetxt(
            cm_csv,
            cm,
            fmt="%d",
            delimiter=","
        )

        print(
            f"\nFold {fold}"
            f"\n  Accuracy: {accuracy:.4f}"
            f"\n  AP:       {ap:.4f}"
        )

        print("  Confusion matrix:")
        print(cm)

    # --------------------------------------------------------
    # Overall CV results
    # --------------------------------------------------------

    print("\nPer-fold accuracy:")
    print(np.round(accuracies, 4))

    print(
        f"Accuracy: "
        f"{np.mean(accuracies):.4f} ± "
        f"{np.std(accuracies, ddof=1):.4f}"
    )

    print("\nPer-fold AP:")
    print(np.round(aps, 4))

    print(
        f"AP: "
        f"{np.mean(aps):.4f} ± "
        f"{np.std(aps, ddof=1):.4f}"
    )

    # --------------------------------------------------------
    # Save combined confusion matrices
    # --------------------------------------------------------

    combined_cm = np.vstack(confusion_matrices)

    combined_cm_path = (
        f"confusion_matrices/"
        f"{model_name.lower()}_year_based_two_fold.csv"
    )

    np.savetxt(
        combined_cm_path,
        combined_cm,
        fmt="%d",
        delimiter=","
    )

    return {
        "model": model_name,
        "accuracy_fold_1": accuracies[0],
        "accuracy_fold_2": accuracies[1],
        "accuracy_mean": np.mean(accuracies),
        "accuracy_std": np.std(accuracies, ddof=1),
        "ap_fold_1": aps[0],
        "ap_fold_2": aps[1],
        "ap_mean": np.mean(aps),
        "ap_std": np.std(aps, ddof=1),
        "confusion_matrices": confusion_matrices
    }



# ============================================================
# SVC
# ============================================================

svc_results = evaluate_model(
    model=SVC(
        kernel="rbf",
        cache_size=500
    ),
    model_name="SVC",
    score_method="decision_function"
)



# ============================================================
# RANDOM FOREST
# ============================================================

rfc_results = evaluate_model(
    model=RFC(
        n_jobs=-1,
        random_state=24
    ),
    model_name="RFC",
    score_method="predict_proba"
)



# ============================================================
# XGBOOST
# ============================================================

xgb_results = evaluate_model(
    model=XGB.XGBClassifier(
        objective="binary:logistic",
        random_state=24,
        n_jobs=-1
    ),
    model_name="XGB",
    score_method="predict_proba"
)


# ============================================================
# SUMMARY TABLE
# ============================================================

results = pd.DataFrame([
    {
        "model": svc_results["model"],
        "accuracy_fold_1": svc_results["accuracy_fold_1"],
        "accuracy_fold_2": svc_results["accuracy_fold_2"],
        "accuracy_mean": svc_results["accuracy_mean"],
        "accuracy_std": svc_results["accuracy_std"],
        "AP_fold_1": svc_results["ap_fold_1"],
        "AP_fold_2": svc_results["ap_fold_2"],
        "AP_mean": svc_results["ap_mean"],
        "AP_std": svc_results["ap_std"],
    },
    {
        "model": rfc_results["model"],
        "accuracy_fold_1": rfc_results["accuracy_fold_1"],
        "accuracy_fold_2": rfc_results["accuracy_fold_2"],
        "accuracy_mean": rfc_results["accuracy_mean"],
        "accuracy_std": rfc_results["accuracy_std"],
        "AP_fold_1": rfc_results["ap_fold_1"],
        "AP_fold_2": rfc_results["ap_fold_2"],
        "AP_mean": rfc_results["ap_mean"],
        "AP_std": rfc_results["ap_std"],
    },
    {
        "model": xgb_results["model"],
        "accuracy_fold_1": xgb_results["accuracy_fold_1"],
        "accuracy_fold_2": xgb_results["accuracy_fold_2"],
        "accuracy_mean": xgb_results["accuracy_mean"],
        "accuracy_std": xgb_results["accuracy_std"],
        "AP_fold_1": xgb_results["ap_fold_1"],
        "AP_fold_2": xgb_results["ap_fold_2"],
        "AP_mean": xgb_results["ap_mean"],
        "AP_std": xgb_results["ap_std"],
    }
])

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)
print(results.to_string(index=False))

results.to_csv(
    "year_based_two_fold_model_results.csv",
    index=False
)


# ============================================================
# SAVE CONFUSION MATRIX FIGURES AS PNG
# Nothing is displayed
# ============================================================

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def save_confusion_matrix_figures(results_dict, model_name):

    cms = results_dict["confusion_matrices"]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 4)
    )

    for i, cm in enumerate(cms):

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Negative", "Positive"]
        )

        disp.plot(
            ax=axes[i],
            colorbar=False
        )

        axes[i].set_title(
            f"{model_name} - "
            f"{'2020' if i == 0 else '2021'}"
        )

    plt.tight_layout()

    output_path = (
        f"confusion_matrices/"
        f"{model_name.lower()}_year_based_two_fold.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    # Do NOT display
    plt.close(fig)

    print(f"Saved: {output_path}")


# Save PNGs
save_confusion_matrix_figures(svc_results, "SVC")
save_confusion_matrix_figures(rfc_results, "RFC")
save_confusion_matrix_figures(xgb_results, "XGB")






