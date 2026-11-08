#!/usr/bin/env python3
"""Train and compare SVM, Random Forest, and XGBoost classifiers.

The classifiers are trained on BirdNET embeddings. Model performance is
estimated with filename-grouped cross-validation.

Run from the project directory:
    python 3_1_2_train_cross_val.py

"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


N_SPLITS = 5
RANDOM_STATE = 22


def load_source(data_root: Path, relative_path: str, label: int, rule: str) -> pd.DataFrame:
    """Load embeddings from one source and derive the filename grouping key."""
    from tools import get_embedding_birdnet

    folder = data_root / relative_path
    if not folder.is_dir():
        raise FileNotFoundError(f"Embedding directory not found: {folder}")

    frame = get_embedding_birdnet(str(folder) + "/", label)
    required = {"embed_name", "embedding"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{folder}: missing columns {sorted(missing)}")

    names = frame["embed_name"].astype("string")
    if rule == "holderried":
        filename = names.str.split(".", n=1).str[0]
    elif rule == "hyphen_drop_last":
        filename = names.str.split("-", n=1).str[0].str[:-1]
    elif rule == "underscore":
        filename = names.str.split("_", n=1).str[0]
    else:
        raise ValueError(f"Unknown filename rule: {rule}")

    frame = frame.copy()
    frame["filename"] = filename
    frame["target"] = label
    return frame


def load_positive(data_root: Path, metadata_path: Path) -> pd.DataFrame:
    """Load and combine the three positive embedding sources."""
    holderried = load_source(
        data_root, "train_data/embedding/holderried", 1, "holderried"
    )
    holderried["selec"] = pd.to_numeric(holderried["filename"], errors="coerce")
    holderried["selec"] = holderried["selec"].astype("int64")

    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_path}")
    metadata = pd.read_csv(metadata_path)
    columns = ["selec", "sound.files", "recording.id"]    
    metadata = metadata[columns].copy()

    holderried = holderried.merge(
        metadata, on="selec", how="left", validate="many_to_one"
    ).dropna(subset=["sound.files", "recording.id"])

    xc_long = load_source(data_root, "XC_long/embedding", 1, "hyphen_drop_last")
    xc_initial = load_source(
        data_root, "train_data/xc/embedding", 1, "hyphen_drop_last"
    )
    return pd.concat([holderried, xc_long, xc_initial], ignore_index=True, sort=False)


def load_negative(data_root: Path) -> pd.DataFrame:
    """Load and combine the six negative embedding sources."""
    sources = [
        ("FVA_non_Eurasian_segs/seed2/embedding", "underscore"),
        ("XC_long/embedding", "hyphen_drop_last"),
        ("train_data/embedding/northwest_DE", "underscore"),
        ("audiomoth_DE/embedding", "underscore"),
        ("audiomoth_Eurasian/embedding", "underscore"),
        # The original notebook removes the final character after splitting.
        ("train_data_5/seed1/embedding", "hyphen_drop_last"),
    ]
    frames = [load_source(data_root, path, 0, rule) for path, rule in sources]
    return pd.concat(frames, ignore_index=True, sort=False)


def make_embedding_matrix(data: pd.DataFrame) -> np.ndarray:
    """Convert the embedding column into a validated 2D feature matrix."""
    try:
        X = np.vstack(data["embedding"].to_numpy())
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Embeddings must be numeric vectors of equal length."
        ) from exc

    if X.ndim != 2 or X.shape[0] != len(data):
        raise ValueError("Expected one embedding vector per row.")
    if not np.isfinite(X).all():
        raise ValueError("Embedding matrix contains NaN or infinite values.")
    return X.astype(np.float32, copy=False)


def evaluate_model(name, model, X, y, groups, cv) -> dict:
    """Evaluate a classifier with the notebook's two cross_val_score calls."""
    accuracy_scores = cross_val_score(
        model, X, y, groups=groups, cv=cv, scoring="accuracy"
    )
    average_precision_scores = cross_val_score(
        model, X, y, groups=groups, cv=cv, scoring="average_precision"
    )

    result = {
        "model": name,
        "accuracy_mean": accuracy_scores.mean(),
        "accuracy_std": accuracy_scores.std(),
        "average_precision_mean": average_precision_scores.mean(),
        "average_precision_std": average_precision_scores.std(),
    }

    print(
        f"{name}: "
        f"Accuracy = {result['accuracy_mean']:.4f} ± "
        f"{result['accuracy_std']:.4f}; "
        f"Average precision = {result['average_precision_mean']:.4f} ± "
        f"{result['average_precision_std']:.4f}"
    )
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("../data"))
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd().parent,
        help="Project root containing src/tools.py.",
    )
    parser.add_argument(
        "--train-metadata",
        type=Path,
        default=Path("../data/Holderried/Holderried_pos_train_1956.csv"),
    )
    parser.add_argument("--n-splits", type=int, default=N_SPLITS)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--save-combined", type=Path, default=None)
    parser.add_argument("--save-results", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    
    src = args.project_root / "src"
    if not src.is_dir():
        raise FileNotFoundError(f"Could not find project source directory: {src}")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    positive = load_positive(args.data_root, args.train_metadata)
    negative = load_negative(args.data_root)
    combined = pd.concat([positive, negative], ignore_index=True, sort=False)
    combined = combined.sample(
        frac=1, random_state=args.random_state
    ).reset_index(drop=True)

    print(f"Positive samples: {len(positive)}")
    print(f"Negative samples: {len(negative)}")
    print(f"Total samples:    {len(combined)}")

    X = make_embedding_matrix(combined)
    y = combined["target"].to_numpy(dtype=int)
    groups = combined["filename"].astype(str).to_numpy()

    
    n_groups = pd.Series(groups).nunique()
    if n_groups < args.n_splits:
        raise ValueError(
            f"Need {args.n_splits} unique filename groups; found {n_groups}."
        )

    cv = GroupKFold(n_splits=args.n_splits)

    models = {
        "SVM (RBF)": SVC(kernel="rbf", cache_size=300),
        "Random Forest": RandomForestClassifier(n_jobs=-1),
        "XGBoost" : XGBClassifier(objective="binary:logistic"),
    }

    print(f"\nEmbedding matrix shape: {X.shape}")
    print(f"Unique filename groups: {n_groups}\n")

    results = []
    for name, model in models.items():
        print(f"===== Training and evaluating {name} =====")
        results.append(evaluate_model(name, model, X, y, groups, cv))

    results_df = pd.DataFrame(results)

    if args.save_combined:
        args.save_combined.parent.mkdir(parents=True, exist_ok=True)
        if args.save_combined.suffix.lower() in {".pkl", ".pickle"}:
            combined.to_pickle(args.save_combined)
        else:
            combined.to_csv(args.save_combined, index=False)
        print(f"Combined data saved to: {args.save_combined}")

    if args.save_results:
        args.save_results.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(args.save_results, index=False)
        print(f"Model results saved to: {args.save_results}")


if __name__ == "__main__":
    main()
