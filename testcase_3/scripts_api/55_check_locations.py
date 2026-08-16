import pandas as pd

# Input CSV
input_csv = "selections.csv"

df = pd.read_csv(input_csv)

# Split deploy.id
parts = df["deploy.id"].astype(str).str.split("_")

df["prefix"] = parts.str[0]
df["suffix"] = parts.str[-1]

# ---------------------------------------------------------
# 1. Check: one prefix -> exactly one suffix
# ---------------------------------------------------------

prefix_to_suffix = (
    df.groupby("prefix")["suffix"]
    .nunique()
)

prefix_violations = prefix_to_suffix[prefix_to_suffix > 1]

print("\n=== PREFIX -> SUFFIX ===")

if prefix_violations.empty:
    print("OK: Every prefix maps to exactly one suffix.")
else:
    print("WARNING: Some prefixes map to multiple suffixes:")
    print(prefix_violations)

    print("\nDetails:")
    print(
        df[df["prefix"].isin(prefix_violations.index)]
        [["deploy.id", "prefix", "suffix"]]
        .drop_duplicates()
        .sort_values(["prefix", "suffix"])
        .to_string(index=False)
    )


# ---------------------------------------------------------
# 2. Check: one suffix -> exactly one prefix
# ---------------------------------------------------------

suffix_to_prefix = (
    df.groupby("suffix")["prefix"]
    .nunique()
)

suffix_violations = suffix_to_prefix[suffix_to_prefix > 1]

print("\n=== SUFFIX -> PREFIX ===")

if suffix_violations.empty:
    print("OK: Every suffix maps to exactly one prefix.")
else:
    print("WARNING: Some suffixes map to multiple prefixes:")
    print(suffix_violations)

    print("\nDetails:")
    print(
        df[df["suffix"].isin(suffix_violations.index)]
        [["deploy.id", "prefix", "suffix"]]
        .drop_duplicates()
        .sort_values(["suffix", "prefix"])
        .to_string(index=False)
    )


# ---------------------------------------------------------
# 3. Summary of all mappings
# ---------------------------------------------------------

mapping = (
    df[["prefix", "suffix"]]
    .drop_duplicates()
    .sort_values(["prefix", "suffix"])
)

print("\n=== UNIQUE PREFIX/SUFFIX MAPPINGS ===")
print(mapping.to_string(index=False))

print(f"\nNumber of unique prefixes: {mapping['prefix'].nunique()}")
print(f"Number of unique suffixes: {mapping['suffix'].nunique()}")
print(f"Number of unique prefix/suffix pairs: {len(mapping)}")
