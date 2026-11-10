#!/usr/bin/env python3
"""Train full-dataset BirdNET embedding classifiers and save the fitted models.

- Load three positive embedding sources and seven negative sources.
- Attach Holderried metadata to the first positive source.
- Combine and shuffle all samples (random_state=22).
- Fit a SVM, Random Forest, and XGBoost classifier on the full dataset.
- Save each fitted model as a joblib file and print its training accuracy.

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
except ImportError as exc:
    raise ImportError(
        "XGBoost is required for this script. Install it with "
        "`pip install xgboost`."
    ) from exc


def load_embedding_source(folder: Path, label: int) -> pd.DataFrame:
    """Load embeddings using the project's get_embedding_birdnet helper."""
    if not folder.is_dir():
        raise FileNotFoundError(f"Embedding directory not found: {folder}")

    from tools import get_embedding_birdnet

    frame = get_embedding_birdnet(str(folder) + "/", label)
    required = {"embed_name", "embedding"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{folder}: missing columns {sorted(missing)}")
    return frame


def load_dataset(data_root: Path, metadata_file: Path) -> pd.DataFrame:
    """Load and combine the positive and negative embedding sources."""
    if not metadata_file.is_file():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    holderried_metadata = pd.read_csv(metadata_file)
    required_metadata = {"selec", "sound.files", "recording.id"}
    missing = required_metadata - set(holderried_metadata.columns)
    if missing:
        raise ValueError(
            f"{metadata_file} is missing columns: {sorted(missing)}"
        )

    # Positive source 1: Holderried selections, with metadata.
    pos1 = load_embedding_source(
        data_root / "train_data" / "embedding" / "holderried", 1
    )
    pos1["filename"] = pos1["embed_name"].astype(str).str.split(".").str[0]
    pos1["selec"] = pd.to_numeric(pos1["filename"], errors="coerce")
    
    pos1["selec"] = pos1["selec"].astype(int)
    pos1 = pos1.merge(
        holderried_metadata[["selec", "sound.files", "recording.id"]],
        on="selec",
        how="left",
    ).dropna()

    # Positive source 2: XC Long.
    pos2 = load_embedding_source(data_root / "XC_long" / "embedding", 1)
    pos2["filename"] = (
        pos2["embed_name"].astype(str).str.split("-").str[0].str[:-1]
    )

    # Positive source 3: XC initial submission.
    pos3 = load_embedding_source(
        data_root / "train_data" / "xc" / "embedding", 1
    )
    pos3["filename"] = (
        pos3["embed_name"].astype(str).str.split("-").str[0].str[:-1]
    )

    positives = pd.concat([pos1, pos2, pos3], ignore_index=True)

    # Negative source 1: FVA non-Eurasian segments.
    neg1 = load_embedding_source(
        data_root / "FVA_non_Eurasian_segs" / "seed2" / "embedding", 0
    )
    neg1["filename"] = (
        neg1["embed_name"].astype(str).str.split("_").str[0]
    )

    # Negative source 2: XC Long.
    neg2 = load_embedding_source(data_root / "XC_long" / "embedding", 0)
    neg2["filename"] = (
        neg2["embed_name"].astype(str).str.split("-").str[0].str[:-1]
    )

    # Negative source 3: North DE.
    neg3 = load_embedding_source(data_root / "train_data" / "embedding" / "northwest_DE", 0)
    neg3["filename"] = (neg3["embed_name"].astype(str).str.split("_").str[0])

    # Negative source 4: Audiomoth DE.
    neg4 = load_embedding_source(data_root / "audiomoth_DE" / "embedding", 0)
    neg4["filename"] = (neg4["embed_name"].astype(str).str.split("_").str[0])

    # Negative source 5: Audiomoth Eurasian.
    neg5 = load_embedding_source(data_root / "audiomoth_Eurasian" / "embedding", 0)
    neg5["filename"] = (
        neg5["embed_name"].astype(str).str.split("_").str[0]
    )

    # Negative source 6: Baden-Baden.
    neg6 = load_embedding_source(data_root / "train_data_5" / "seed1" / "embedding", 0)
    neg6["filename"] = (
        neg6["embed_name"].astype(str).str.split("-").str[0].str[:-1]
    )

    # Negative source 7: locally generated negatives.
    neg7 = load_embedding_source(data_root / "test_data_2" / "embedding", 0)
    neg7["filename"] = (
        neg7["embed_name"].astype(str).str.split(".").str[0]
    )

    negatives = pd.concat(
        [neg1, neg2, neg3, neg4, neg5, neg6, neg7],
        ignore_index=True,
    )

    positives["target"] = 1
    negatives["target"] = 0

    combined = pd.concat([positives, negatives], ignore_index=True, sort=False)

    if combined.empty:
        raise ValueError("No embedding samples were loaded.")
    
    return combined


def build_features(data: pd.DataFrame):
    """Create the feature matrix and target vector."""
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

    return X, y


def train_and_save_models(X, y, output_dir: Path):
    """Fit the three notebook classifiers on all samples and save them."""
    output_dir.mkdir(parents=True, exist_ok=True)

    models = {
        "svc_model.joblib": SVC(
            kernel="linear",
            probability=True,
            random_state=22,
        ),
        "rfc_model.joblib": RandomForestClassifier(n_jobs=-1),
        "xgb_model.joblib": XGBClassifier(objective="binary:logistic"),
    }

    for filename, model in models.items():
        print(f"\nTraining {model.__class__.__name__} on the full dataset...")
        model.fit(X, y)

        model_path = output_dir / filename
        joblib.dump(model, model_path)

        train_accuracy = model.score(X, y)
        print(f"Model saved to: {model_path}")
        print(f"Train accuracy: {train_accuracy:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd().parent,
        help="Project root containing src/tools.py.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("../data"),
        help="Root directory containing the embedding source folders.",
    )
    parser.add_argument(
        "--metadata-file",
        type=Path,
        default=Path("../data/Holderried/Holderried_pos_train_1956.csv"),
        help="Holderried positive-selection metadata CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory in which the three joblib models are saved.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    project_src = args.project_root / "src"
    if not project_src.is_dir():
        raise FileNotFoundError(f"Could not find source directory: {project_src}")
    if str(project_src) not in sys.path:
        sys.path.insert(0, str(project_src))

    data = load_dataset(args.data_root, args.metadata_file)
    data = data.sample(frac=1, random_state=22).reset_index(drop=True)
    X, y = build_features(data)

    print(f"Positive samples: {(y == 1).sum()}")
    print(f"Negative samples: {(y == 0).sum()}")
    print(f"Embedding matrix shape: {X.shape}")

    train_and_save_models(X, y, args.output_dir)


if __name__ == "__main__":
    main()
