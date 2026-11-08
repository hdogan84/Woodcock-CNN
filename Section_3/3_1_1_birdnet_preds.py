#!/usr/bin/env python3
"""Evaluate BirdNET confidence scores with filename-grouped cross-validation
   for the Train/validation datasets in sec. 3.1

Run from the project directory, for example:
    python evaluate_birdnet.py

Override paths when needed:
    python evaluate_birdnet.py --data-root ../data \
        --train-metadata Holderried_pos_train_1956.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.model_selection import GroupKFold


RANDOM_STATE = 22
N_SPLITS = 5
THRESHOLD = 0.5


def read_predictions(data_root: Path, relative_dir: str, class_name: str) -> pd.DataFrame:
    """Read BirdNET predictions for one source and normalize its filename key."""
    csv_path = data_root / relative_dir / "birdnet_probs" / class_name / "predictions.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Prediction file not found: {csv_path}")

    frame = pd.read_csv(csv_path)
    required = {"filename", "confidence"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")
    return frame


def load_positive_data(data_root: Path, train_metadata_path: Path) -> pd.DataFrame:
    """Load and combine the three positive-record sources."""
    # Holderried selections: map selection IDs to their source recording metadata.
    holderried = read_predictions(
        data_root, "Holderried/selections_wavs", "1"
    )
    selection_text = holderried["filename"].astype("string").str.split(".", n=1).str[0]
    holderried["selec"] = pd.to_numeric(selection_text, errors="coerce")
    holderried["selec"] = holderried["selec"].astype("int64")

    if not train_metadata_path.is_file():
        raise FileNotFoundError(f"Training metadata file not found: {train_metadata_path}")
    
    metadata = pd.read_csv(train_metadata_path)
    metadata_columns = {"selec", "sound.files", "recording.id"}
    metadata = metadata[list(metadata_columns)].copy()

    holderried = holderried.merge(metadata, on="selec", how="left", validate="many_to_one")
    # Drop rows that are part of test metadata.
    holderried = holderried.dropna(subset=["sound.files", "recording.id"])
    

    # XC Long: retain the original filename for grouping and add the 
    # normalized wavname field.
    xc_long = read_predictions(data_root, "XC_long", "1")
    xc_long["wavname"] = (
        xc_long["filename"].astype("string").str.split("-", n=1).str[0].str[:-1]
    )

    # Initial XC submission: normalize the filename 
    xc_initial = read_predictions(data_root, "train_data/xc", "1")
    xc_initial["filename"] = (
        xc_initial["filename"].astype("string").str.split("-", n=1).str[0].str[:-1]
    )

    return pd.concat([holderried, xc_long, xc_initial], ignore_index=True, sort=False)


def load_negative_data(data_root: Path) -> pd.DataFrame:
    """Load and combine the six negative-record sources."""
    sources = [
        ("FVA_non_Eurasian_segs/seed2", "underscore"),
        ("XC_long", "xc_long"),
        ("train_data/embedding/northwest_DE", "underscore"),
        ("audiomoth_DE", "underscore"),
        ("audiomoth_Eurasian", "underscore"),
        ("train_data_5/seed1", "underscore"),
    ]

    frames = []
    for relative_dir, filename_rule in sources:
        frame = read_predictions(data_root, relative_dir, "0")
        filename = frame["filename"].astype("string")
        if filename_rule == "xc_long":
            frame["filename"] = filename.str.split("-", n=1).str[0].str[:-1]
        else:
            frame["filename"] = filename.str.split("_", n=1).str[0]
        frames.append(frame)

    return pd.concat(frames, ignore_index=True, sort=False)


def evaluate_grouped_cv(
        data: pd.DataFrame,
        n_splits: int = N_SPLITS,
        threshold: float = THRESHOLD,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:

    """Evaluate BirdNET confidence scores """
    required = {"filename", "confidence", "target"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Combined data is missing columns: {sorted(missing)}")

    data = data.copy()
    data["confidence"] = pd.to_numeric(data["confidence"], errors="coerce")
    data["target"] = data["target"].astype(int)

    X = data["confidence"].to_numpy()
    y = data["target"].to_numpy()
    groups = data["filename"].astype(str).to_numpy()

    n_groups = pd.Series(groups).nunique()
    if n_groups < n_splits:
        raise ValueError(
            f"GroupKFold with {n_splits} splits requires at least {n_splits} "
            f"unique filename groups; found {n_groups}."
        )

    cv = GroupKFold(n_splits=n_splits)
    data["fold"] = -1
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), start=1):
        # Record the held-out fold for every row.
        data.iloc[test_idx, data.columns.get_loc("fold")] = fold - 1

        y_true = y[test_idx]
        y_score = X[test_idx]
        y_pred = (y_score >= threshold).astype(int)

        accuracy = accuracy_score(y_true, y_pred)
        average_precision = average_precision_score(y_true, y_score)
        fold_results.append(
            {
                "fold": fold,
                "accuracy": accuracy,
                "average_precision": average_precision,
                "n_test": len(test_idx),
            }
        )

        print(
            f"Fold {fold}: Accuracy = {accuracy:.4f}, "
            f"AP = {average_precision:.4f}"
        )

    results = pd.DataFrame(fold_results)
    print("\n===== Grouped cross-validation results =====")
    for metric, label in (
        ("accuracy", "Accuracy"),
        ("average_precision", "Average Precision"),
    ):
        values = results[metric]
        print(f"{label}: {values.mean():.4f} ± {values.std(ddof=1):.4f}")

    return data, results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("../data"),
        help="Directory containing the source data folders (default: ../data).",
    )
    parser.add_argument(
        "--train-metadata",
        type=Path,
        default=Path("../data/Holderried/Holderried_pos_train_1956.csv"),
        help="CSV mapping Holderried selection IDs to recording metadata.",
    )
    parser.add_argument(
        "--save-combined",
        type=Path,
        default=None,
        help="Optional path to save the combined data, including its CV fold.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=THRESHOLD,
        help=f"Confidence threshold for accuracy (default: {THRESHOLD}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.threshold <= 1:
        raise ValueError("--threshold must be between 0 and 1.")

    positive = load_positive_data(args.data_root, args.train_metadata)
    negative = load_negative_data(args.data_root)

    positive["target"] = 1
    negative["target"] = 0
    combined = pd.concat([positive, negative], ignore_index=True, sort=False)

    # Deterministic shuffle.
    combined = combined.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    print(f"Positive rows: {len(positive)}")
    print(f"Negative rows: {len(negative)}")
    print(f"Total rows:    {len(combined)}")

    combined, _ = evaluate_grouped_cv(combined, threshold=args.threshold)

    if args.save_combined is not None:
        args.save_combined.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(args.save_combined, index=False)
        print(f"\nCombined data with fold assignments saved to: {args.save_combined}")


if __name__ == "__main__":
    main()
