#!/usr/bin/env python3
"""Match saved BirdNET predictions to test-set folds and calculate metrics.

This script the prediction dictionaries saved as .npy files, joins them to the
test-case metadata, and reports metrics per fold and across folds.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)


def load_prediction_dictionary(path: Path, label: str) -> pd.DataFrame:
    """Load a trusted NumPy object file containing a prediction dictionary."""
    if not path.is_file():
        raise FileNotFoundError(f"{label} prediction file not found: {path}")
    loaded = np.load(path, allow_pickle=True)
    try:
        predictions = loaded.item()
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"{path} must contain a single dictionary object.") from exc
    finally:
        if hasattr(loaded, "close"):
            loaded.close()
    if not isinstance(predictions, dict):
        raise TypeError(f"{path} must contain a dictionary.")
    
    frame = pd.DataFrame(predictions.items(), columns=["filename", "prediction"])
    frame["filename"] = frame["filename"].astype(str)
    frame["prediction"] = pd.to_numeric(frame["prediction"], errors="coerce")
    if frame["prediction"].isna().any():
        raise ValueError(f"{path} contains non-numeric predictions.")
    
    return frame


def load_metadata(path: Path) -> pd.DataFrame:
    """Load metadata and apply the notebook's positive filename convention."""
    if not path.is_file():
        raise FileNotFoundError(f"Metadata CSV not found: {path}")
    df = pd.read_csv(path)
    required = {"selec", "filename", "target", "fold"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    df = df.copy()
    for col in ("target", "fold", "selec"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    df["target"] = df["target"].astype(int)
    df["fold"] = df["fold"].astype(int)
    pos = df["target"] == 1
    df.loc[pos, "filename"] = df.loc[pos, "selec"].astype("Int64").astype(str)
    df.loc[~pos, "filename"] = df.loc[~pos, "filename"].astype(str)
    return df


def match_predictions(pos_preds: pd.DataFrame, neg_preds: pd.DataFrame,
                      metadata: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Join positive predictions by selection ID and negatives by filename."""
    pos = pos_preds.copy()
    pos["selec"] = pd.to_numeric(
        pos["filename"].str.replace(".wav", "", regex=False), errors="coerce"
    )
    
    pos["selec"] = pos["selec"].astype("int64")
    pos_meta = metadata.loc[metadata["target"] == 1]
    pos = pos.merge(pos_meta, on="selec", how="left",
                    suffixes=("_prediction", "_metadata"))
    pos = pos.dropna(subset=["target", "fold"])

    neg = neg_preds.copy()
    neg["filename"] = neg["filename"].str.replace(".wav", "", regex=False)
    neg_meta = metadata.loc[metadata["target"] == 0]
    neg = neg.merge(neg_meta, on="filename", how="left",
                    suffixes=("_prediction", "_metadata"))
    unmatched_negatives = int(neg["target"].isna().sum())

    combined = pd.concat([pos, neg], ignore_index=True, sort=False)
    combined = combined.dropna(subset=["target", "fold"]).copy()
    if combined.empty:
        raise ValueError("No predictions matched the metadata and fold assignments.")
    combined["target"] = combined["target"].astype(int)
    combined["fold"] = combined["fold"].astype(int)
    return combined, unmatched_negatives


def calculate_fold_metrics(data: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Calculate the same metrics as the notebook, separately for each fold."""
    rows = []
    for fold in sorted(data["fold"].unique()):
        part = data.loc[data["fold"] == fold]
        y_true = part["target"].to_numpy(dtype=int)
        y_score = part["prediction"].to_numpy(dtype=float)
        y_pred = (y_score >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).ravel()
        auc = roc_auc_score(y_true, y_score) if np.unique(y_true).size == 2 else np.nan
        rows.append({
            "fold": fold,
            "n_samples": len(part),
            "n_positive": int((y_true == 1).sum()),
            "n_negative": int((y_true == 0).sum()),
            "average_precision": average_precision_score(y_true, y_score),
            "roc_auc": auc,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        })
    return pd.DataFrame(rows)


def print_summary(results: pd.DataFrame) -> None:
    metrics = ["average_precision", "roc_auc", "accuracy",
               "precision", "recall", "f1"]
    print("\nMean ± standard deviation across folds:")
    for metric in metrics:
        print(f"{metric}: {results[metric].mean():.4f} ± {results[metric].std():.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path,
                        default=Path("../data/Holderried/df_comb_date_testcase3.csv"))
    parser.add_argument("--positive-predictions", type=Path,
                        default=Path("preds/holderried_predictions_selections.npy"))
    parser.add_argument("--negative-predictions", type=Path,
                        default=Path("preds/holderried_preds_testdata3_neg.npy"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Optional directory for matched predictions and metrics.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 <= args.threshold <= 1:
        raise ValueError("--threshold must be between 0 and 1.")
    metadata = load_metadata(args.metadata)
    pos_preds = load_prediction_dictionary(args.positive_predictions, "Positive")
    neg_preds = load_prediction_dictionary(args.negative_predictions, "Negative")
    matched, unmatched_neg = match_predictions(pos_preds, neg_preds, metadata)

    n_pos = int((matched["target"] == 1).sum())
    n_neg = int((matched["target"] == 0).sum())
    print(f"Positive predictions matched: {n_pos}")
    print(f"Negative predictions matched: {n_neg}")
    print(f"Unmatched negative predictions: {unmatched_neg}")
    print(f"Total matched predictions: {len(matched)}")
    print(f"Classification threshold: {args.threshold:.3f}")

    results = calculate_fold_metrics(matched, args.threshold)
    print("\nMetrics by fold:")
    print(results.to_string(index=False))
    print_summary(results)

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        matched.to_csv(args.output_dir / "matched_predictions.csv", index=False)
        results.to_csv(args.output_dir / "metrics_by_fold.csv", index=False)
        print(f"\nOutput files saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
