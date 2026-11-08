#!/usr/bin/env python3
"""Train and compare SVM, Random Forest, and XGBoost on BirdNET embeddings.

The script's workflow is:
- Load negative embeddings from test_data_3 and positive Holderried embeddings.
- Join them to the corresponding selection metadata and derive recording dates.
- Use 4-fold GroupKFold, keeping recordings from the same date together.
- Compare SVM, Random Forest, and XGBoost using accuracy and average precision.
- Train one model per grouped-CV fold, with fold confusion matrices.

Run from the project working directory (the directory from which the notebook
was run), so the default relative paths resolve as in the notebook.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def load_embedding_source(folder: Path, label: int) -> pd.DataFrame:
    """Load BirdNET embeddings using the project's get_embedding_birdnet helper."""
    if not folder.is_dir():
        raise FileNotFoundError(f"Embedding directory not found: {folder}")

    from tools import get_embedding_birdnet
    frame = get_embedding_birdnet(str(folder) + "/", label)

    required = {"embed_name", "embedding"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{folder}: missing columns {sorted(missing)}")
    return frame


def load_dataset(
    data_root: Path,
    testcase_dir: Path,
) -> pd.DataFrame:
    """Load positive/negative embeddings, attach metadata, and combine."""
    full_metadata_path = testcase_dir / "selections.csv"
    positive_metadata_path = testcase_dir / "selecs_positive_test_samedate.csv"

    for path in (full_metadata_path, positive_metadata_path):
        if not path.is_file():
            raise FileNotFoundError(f"Metadata file not found: {path}")

    full_metadata = pd.read_csv(full_metadata_path)
    positive_metadata = pd.read_csv(positive_metadata_path)

    if "selec" not in full_metadata.columns:
        raise ValueError(f"{full_metadata_path} must contain 'selec'.")
    positive_columns = ["selec", "sound.files", "recording.id", "deploy.id"]
    missing = set(positive_columns) - set(positive_metadata.columns)
    if missing:
        raise ValueError(
            f"{positive_metadata_path} is missing columns: {sorted(missing)}"
        )

    # Negative samples: selection ID comes from the part before the first "_".
    negative = load_embedding_source(
        data_root / "test_data_3" / "embedding", 0
    )
    negative["selec"] = (
        negative["embed_name"].astype(str).str.split("_").str[0]
    )
    negative["selec"] = pd.to_numeric(negative["selec"], errors="coerce")
    if negative["selec"].isna().any():
        raise ValueError("Could not extract selection IDs from negative embed_name.")
    negative["selec"] = negative["selec"].astype(int)
    negative["filename"] = (
        negative["embed_name"].astype(str).str.split(".", n=1).str[0]
    )
    negative = negative.merge(
        full_metadata, on="selec", how="left", validate="many_to_one"
    ).dropna()
    negative["date"] = (
        negative["sound.files"].astype(str).str.split("_").str[0]
    )
    negative["target"] = 0

    # Positive samples: Holderried selection ID is the leading integer.
    positive = load_embedding_source(
        data_root / "train_data" / "embedding" / "holderried", 1
    )
    positive["selec"] = pd.to_numeric(
        positive["embed_name"].astype(str).str.extract(r"^(\d+)")[0],
        errors="coerce",
    )
    if positive["selec"].isna().any():
        raise ValueError("Could not extract selection IDs from positive embed_name.")
    positive["selec"] = positive["selec"].astype(int)
    positive = positive.merge(
        positive_metadata[positive_columns],
        on="selec",
        how="left",
        validate="many_to_one",
    )
    positive["date"] = (
        positive["sound.files"].astype(str).str.split("_").str[0]
    )
    positive = positive.dropna()
    positive["recording.id"] = positive["recording.id"].astype(int)
    positive["target"] = 1

    combined = pd.concat([positive, negative], ignore_index=True, sort=False)
    if combined.empty:
        raise ValueError("No embedding samples remain after metadata matching.")
    return combined


def build_features(data: pd.DataFrame):
    """Build a numeric embedding matrix, labels, and date-based groups."""
    try:
        X = np.vstack(data["embedding"].to_numpy())
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Embedding vectors must be numeric and have equal dimensions."
        ) from exc
    if X.ndim != 2 or X.shape[0] != len(data):
        raise ValueError("Expected one embedding vector per sample.")
    if not np.isfinite(X).all():
        raise ValueError("Embedding matrix contains NaN or infinite values.")

    y = data["target"].to_numpy(dtype=int)
    groups = data["date"].astype(str).to_numpy()
    return X.astype(np.float32, copy=False), y, groups


def make_models(random_state: int):
    models = {
        "SVM": SVC(kernel="rbf", cache_size=300),
        "Random Forest": RandomForestClassifier(
            n_jobs=-1, random_state=random_state
        ),
    }
    if XGBClassifier is None:
        print("Warning: xgboost is not installed; skipping XGBoost.")
    else:
        models["XGBoost"] = XGBClassifier(
            objective="binary:logistic",
            random_state=random_state,
            n_jobs=-1,
        )
    return models


def compare_models(models, X, y, groups, cv):
    """Run the notebook's separate cross_val_score calls for both metrics."""
    summaries = []
    for name, model in models.items():
        print(f"\n===== {name} =====")
        accuracy = cross_val_score(
            model, X, y, cv=cv, groups=groups, scoring="accuracy"
        )
        average_precision = cross_val_score(
            model, X, y, cv=cv, groups=groups,
            scoring="average_precision",
        )
        print(
            f"Accuracy: {accuracy.mean():.4f} ± "
            f"{accuracy.std(ddof=1):.4f}"
        )
        print(
            f"AP:       {average_precision.mean():.4f} ± "
            f"{average_precision.std(ddof=1):.4f}"
        )
        summaries.append({
            "model": name,
            "accuracy_mean": accuracy.mean(),
            "accuracy_std": accuracy.std(ddof=1),
            "average_precision_mean": average_precision.mean(),
            "average_precision_std": average_precision.std(ddof=1),
        })
    return pd.DataFrame(summaries)

def save_fold_confusion_matrices(
    models, X, y, groups, n_splits: int, output_dir: Path
):
    """Fit each model on each training fold and save its validation confusion matrix."""

    output_dir.mkdir(parents=True, exist_ok=True)
    cv = GroupKFold(n_splits=n_splits)

    for name, estimator in models.items():
        model_slug = name.lower().replace(" ", "_")

        print(f"\nSaving {name} confusion matrices...")

        for fold, (train_idx, test_idx) in enumerate(
            cv.split(X, y, groups), start=1
        ):
            # Clone the estimator to ensure a fresh model for each fold
            model = clone(estimator)

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Fit and predict
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Calculate confusion matrix
            matrix = confusion_matrix(
                y_test,
                y_pred,
                labels=[0, 1]
            )

            # Plot confusion matrix
            display = ConfusionMatrixDisplay(
                confusion_matrix=matrix,
                display_labels=["Negative", "Positive"]
            )

            fig, ax = plt.subplots(figsize=(5, 4))

            display.plot(
                ax=ax,
                cmap="Blues",
                colorbar=False,
                values_format="d"
            )

            ax.set_title(f"{name} — Fold {fold}")

            fig.tight_layout()

            # Save PNG
            output_path = (
                output_dir
                / f"{model_slug}_confusion_matrix_fold_{fold}.png"
            )

            fig.savefig(
                output_path,
                dpi=200,
                bbox_inches="tight"
            )

            plt.close(fig)

            print(f"Saved: {output_path}")

    print(f"\nAll confusion matrices saved to: {output_dir}")



def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path, default=Path.cwd().parent,
        help="Project root containing src/tools.py and data/.",
    )
    parser.add_argument(
        "--testcase-dir", type=Path, default=Path("../testcase_3"),
        help="Directory containing the two testcase metadata CSV files.",
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("../data"),
        help="Root directory containing test_data_3 and train_data.",
    )
    parser.add_argument("--n-splits", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=24)
    parser.add_argument(
        "--fold-metadata-output", type=Path,
        default=Path("df_comb_site_testcase3.csv"),
    )
    parser.add_argument(
        "--results-output", type=Path, default=None,
        help="Optional CSV path for model comparison results.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    if args.n_splits < 2:
        raise ValueError("--n-splits must be at least 2.")

    project_src = args.project_root / "src"
    
    if str(project_src) not in sys.path:
        sys.path.insert(0, str(project_src))

    data = load_dataset(args.data_root, args.testcase_dir)
    data = data.sample(frac=1, random_state=args.random_state).reset_index(drop=True)
    X, y, groups = build_features(data)

    
    unique_groups = np.unique(groups)
    if len(unique_groups) < args.n_splits:
        raise ValueError(
            f"GroupKFold with {args.n_splits} splits requires at least "
            f"{args.n_splits} unique dates; found {len(unique_groups)}."
        )

    print(f"Positive samples: {(y == 1).sum()}")
    print(f"Negative samples: {(y == 0).sum()}")
    print(f"Embedding matrix shape: {X.shape}")
    print(f"Unique recording dates: {len(unique_groups)}")

    cv = GroupKFold(n_splits=args.n_splits)
    data["fold"] = -1
    for fold, (_, validation_idx) in enumerate(cv.split(X, y, groups)):
        data.loc[validation_idx, "fold"] = fold

    fold_output = args.fold_metadata_output
    fold_output.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        col for col in
        ["selec", "filename", "recording.id", "date", "target", "fold"]
        if col in data.columns
    ]
    data[columns].to_csv(fold_output, index=False)
    print(f"Fold assignments saved to: {fold_output}")
    print("Samples per fold:")
    print(data["fold"].value_counts().sort_index().to_string())

    results = compare_models(make_models(args.random_state), X, y, groups, cv)
    if args.results_output:
        args.results_output.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.results_output, index=False)
        print(f"\nModel comparison saved to: {args.results_output}")

    save_fold_confusion_matrices(
        make_models(args.random_state),
        X,
        y,
        groups,
        args.n_splits,
        Path("saved_models")
    )


if __name__ == "__main__":
    main()
