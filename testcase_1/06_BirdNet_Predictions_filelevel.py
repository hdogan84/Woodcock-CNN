import csv
import numpy as np


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

INPUT_RESULTS = "testset1_results.npz"
OUTPUT_CSV = "testset1_segments.csv"


# ---------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------

def extract_xc_id(filename):
    """
    Example:
        XC664087 - Eurasian Woodcock - Scolopax rusticola_3.wav

    Returns:
        XC664087
    """

    return filename.split(" - ")[0]


def extract_t_start(filename):
    """
    Example:
        XC664087 - Eurasian Woodcock - Scolopax rusticola_3.wav

    Returns:
        3
    """

    # Remove .wav
    stem = filename.rsplit(".", 1)[0]

    # Take everything after the final underscore
    t_start = stem.rsplit("_", 1)[-1]

    return int(t_start)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    # -------------------------------------------------------------
    # Load predictions and labels
    # -------------------------------------------------------------

    print(f"Loading: {INPUT_RESULTS}")

    data = np.load(
        INPUT_RESULTS,
        allow_pickle=True
    )

    predictions = data["predictions"].item()
    labels = data["labels"].item()

    print(f"Number of predictions: {len(predictions)}")
    print(f"Number of labels     : {len(labels)}")

    # -------------------------------------------------------------
    # Create CSV
    # -------------------------------------------------------------

    rows = []

    for filename, prediction in predictions.items():

        # Get the label for THIS EXACT segment
        if filename not in labels:
            print(
                f"WARNING: No label found for {filename}"
            )
            continue

        label = labels[filename]

        # Extract XC ID
        xc_id = extract_xc_id(filename)

        # Extract segment start/time index
        t_start = extract_t_start(filename)

        rows.append({
            "XC_ID": xc_id,
            "t_start": t_start,
            "prediction": float(prediction),
            "label": int(label),
        })

    # -------------------------------------------------------------
    # Save CSV
    # -------------------------------------------------------------

    fieldnames = [
        "XC_ID",
        "t_start",
        "prediction",
        "label",
    ]

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    # -------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------

    print("\nDone.")
    print(f"Rows written: {len(rows)}")
    print(f"Output file : {OUTPUT_CSV}")


if __name__ == "__main__":
    main()