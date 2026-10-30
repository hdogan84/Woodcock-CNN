import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

INPUT_CSV = "testset1_segments.csv"
OUTPUT_CSV = "testset1_xc_results.csv"

THRESHOLD = 0.5


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

df = pd.read_csv(INPUT_CSV)

results = []

for xc_id in df["XC_ID"].unique():

    # Select all segments belonging to this XC recording
    df_xc = df[df["XC_ID"] == xc_id]

    predictions = df_xc["prediction"].values
    labels = df_xc["label"].values

    # Binary predictions for accuracy
    binary_predictions = (
        predictions >= THRESHOLD
    ).astype(int)

    # Metrics
    accuracy = accuracy_score(
        labels,
        binary_predictions
    )

    avg_precision = average_precision_score(
        labels,
        predictions
    )

    # Number of negative and positive segments
    n_zeros = (labels == 0).sum()
    n_ones = (labels == 1).sum()

    results.append({
        "XC_ID": xc_id,
        "accuracy": accuracy,
        "avg. precision": avg_precision,
        "n_0": n_zeros,
        "n_1": n_ones,
    })


# ---------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------

df_results = pd.DataFrame(results)

df_results.to_csv(
    OUTPUT_CSV,
    index=False
)

print(f"Saved {len(df_results)} XC recordings to:")
print(OUTPUT_CSV)

print("\nResult:")
print(df_results)