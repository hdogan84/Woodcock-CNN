
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.svm import SVC


# =============================================================================
# Paths / imports
# =============================================================================

parent_folder = Path().resolve().parent
src_path = parent_folder / "src"
sys.path.append(str(src_path))

from tools import get_embedding_birdnet


# =============================================================================
# Configuration
# =============================================================================

RANDOM_STATE = 22

MODEL_DIR = Path("../models/negative_subset_svc")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Load positive datasets
# =============================================================================

# -----------------------------------------------------------------------------
# Positive 1: Holderried
# -----------------------------------------------------------------------------

root_folder = "../data/train_data/embedding/holderried/"

df_pos1 = get_embedding_birdnet(root_folder, 1)

df_pos1["filename"] = (
    df_pos1["embed_name"]
    .str.split(".")
    .str[0]
)

df_pos1["selec"] = df_pos1["filename"].astype(int)


# Add metadata from selections_train.csv
df_train = pd.read_csv("../testcase_3/selections_train.csv")

df_pos1 = df_pos1.merge(
    df_train[["selec", "sound.files", "recording.id"]],
    on="selec",
    how="left"
)

df_pos1 = df_pos1.dropna()


# -----------------------------------------------------------------------------
# Positive 2: XC long positive records
# -----------------------------------------------------------------------------

root_folder = "../data/test_data/embedding/"

df_pos2 = get_embedding_birdnet(root_folder, 1)

df_pos2["filename"] = (
    df_pos2["embed_name"]
    .str.split("-")
    .str[0]
    .str[:-1]
)


# -----------------------------------------------------------------------------
# Positive 3: XC initial submission positive records
# -----------------------------------------------------------------------------

root_folder = "../data/train_data/embedding/xc/"

df_pos3 = get_embedding_birdnet(root_folder, 1)

df_pos3["filename"] = (
    df_pos3["embed_name"]
    .str.split("-")
    .str[0]
    .str[:-1]
)


# -----------------------------------------------------------------------------
# Combine all positives
# -----------------------------------------------------------------------------

df_pos = pd.concat(
    [df_pos1, df_pos2, df_pos3],
    ignore_index=True
)

df_pos["target"] = 1

print(f"Total positive samples: {len(df_pos)}")


# =============================================================================
# Load negative datasets
# =============================================================================

# Each entry contains:
#   name         -> name used for the saved model
#   folder       -> embedding folder
#   filename_fn  -> how filename is extracted
#   description  -> optional human-readable description

negative_datasets = [

    {
        "name": "neg1_seed3",
        "folder": "../data/FVA_non_Eurasian_segs/embedding/seed3/",
        "filename_fn": lambda x: x.str.split("_").str[0],
        "description": "FVA non-Eurasian seed3",
    },

    {
        "name": "neg2_xc",
        "folder": "../data/test_data/embedding/",
        "filename_fn": lambda x: (
            x.str.split("-").str[0].str[:-1]
        ),
        "description": "XC long recordings",
    },

    {
        "name": "neg3_holderried",
        "folder": "../data/train_data/embedding/holderried/",
        "filename_fn": lambda x: x.str.split("_").str[0],
        "description": "Schönow recordings",
    },

    {
        "name": "neg4_audiomoth_DE",
        "folder": "../data/audiomoth_DE/embedding/",
        "filename_fn": lambda x: x.str.split("_").str[0],
        "description": "Audiomoth DE",
    },

    {
        "name": "neg5_audiomoth_Eurasian",
        "folder": "../data/audiomoth_Eurasian/embedding/",
        "filename_fn": lambda x: x.str.split("_").str[0],
        "description": "Audiomoth Eurasian",
    },

    {
        "name": "neg6_baden_baden",
        "folder": "../data/test_data_5/embedding/",
        "filename_fn": lambda x: (
            x.str.split("-").str[0].str[:-1]
        ),
        "description": "Baden Baden recordings",
    },

]


# =============================================================================
# Load all negative subsets
# =============================================================================

negative_dataframes = {}

for neg in negative_datasets:

    print(f"\nLoading {neg['name']}: {neg['description']}")

    df_neg = get_embedding_birdnet(
        neg["folder"],
        0
    )

    if neg["filename_fn"] is not None:
        df_neg["filename"] = neg["filename_fn"](
            df_neg["embed_name"]
        )

    df_neg["target"] = 0

    negative_dataframes[neg["name"]] = df_neg

    print(f"  Samples: {len(df_neg)}")


# =============================================================================
# Train one SVC per negative subset
# =============================================================================

print("\n" + "=" * 70)
print("TRAINING SVC MODELS")
print("=" * 70)

for neg_name, df_neg in negative_dataframes.items():

    print(f"\n{'-' * 70}")
    print(f"Training: {neg_name}")
    print(f"{'-' * 70}")

    # -------------------------------------------------------------------------
    # Combine fixed positives with this negative subset
    # -------------------------------------------------------------------------

    df_combined = pd.concat(
        [df_pos, df_neg],
        ignore_index=True
    )

    print(f"Positive samples : {len(df_pos)}")
    print(f"Negative samples : {len(df_neg)}")
    print(f"Total samples    : {len(df_combined)}")

    # -------------------------------------------------------------------------
    # Extract features and labels
    # -------------------------------------------------------------------------

    X = np.vstack(
        df_combined["embedding"].values
    )

    y = df_combined["target"].values

    print(f"Feature matrix   : {X.shape}")

    # -------------------------------------------------------------------------
    # Create SVC
    # -------------------------------------------------------------------------

    model = SVC(
        kernel="rbf",
        probability=True,
        random_state=RANDOM_STATE
    )

    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------

    model.fit(X, y)

    # -------------------------------------------------------------------------
    # Training accuracy
    # -------------------------------------------------------------------------

    train_accuracy = model.score(X, y)

    print(f"Train accuracy   : {train_accuracy:.4f}")

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    model_path = MODEL_DIR / f"svc_{neg_name}.joblib"

    joblib.dump(
        model,
        model_path
    )

    print(f"Saved            : {model_path}")


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)