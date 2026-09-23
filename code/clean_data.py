"""
Data pipeline

Builds bank-by-sector exposure matrix from EBA 2025 EU-wide
Transparency Exercise (tr_cre.csv), 

normalizes it into exposure shares,
and then computes the cosine similarity matrix used as contagion network's
edge weights.

Outputs (output/processed/):
  bank_sector_exposure.csv  - raw exposure amounts, banks x NACE sectors
  bank_sector_shares.csv    - row-normalized exposure shares (each row sums to 1)
  similarity_matrix.csv     - cosine similarity, banks x banks
  bank_metadata.csv         - LEI_Code, Name, Country for surviving banks
"""

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = "data"  # folder holding the raw input files (tr_cre.csv, tr_oth.csv, TR_Metadata.xlsx)
OUTPUT_DIR = "output/processed"  # folder where the cleaned/derived CSVs get written

ITEM = "2521301"        # EBA's code for "credit risk exposure by NACE sector" (the raw file mixes many item types together)
PERIOD = "202506"       # June 2025, the latest quarter available 
NACE_EXCLUDE = "0"      # EBA's placeholder for "no sector breakdown reported"


def load_exposures():
    cols = ["LEI_Code", "Period", "Item", "NACE_codes", "Amount"]
    df = pd.read_csv(f"{DATA_DIR}/tr_cre.csv", usecols=cols, dtype={
        "LEI_Code": str, "Period": str, "Item": str, "NACE_codes": str,
    })
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0.0)

    df = df[(df["Item"] == ITEM) & (df["Period"] == PERIOD)]
    df = df[df["NACE_codes"] != NACE_EXCLUDE]
    # "XXXXXXXXXXXXXXXXXXXX" is EBA's placeholder LEI for the residual "Other
    # banks" aggregate, not an individually identifiable institution.
    df = df[df["LEI_Code"] != "XXXXXXXXXXXXXXXXXXXX"]
    return df


def build_exposure_matrix(df):
    matrix = df.pivot_table(
        index="LEI_Code", columns="NACE_codes", values="Amount",
        aggfunc="sum", fill_value=0.0,
    )
    # sort sector columns numerically (1..19) rather than lexically
    matrix = matrix[sorted(matrix.columns, key=int)]
    return matrix


def load_metadata(bank_ids):
    inst = pd.read_excel(
        f"{DATA_DIR}/TR_Metadata.xlsx", sheet_name="List of Institutions", header=1,
    )
    inst = inst.rename(columns={"Country": "Country_code", "Desc_country": "Country"})
    inst = inst[["LEI_Code", "Name", "Country", "Country_code"]]
    return inst[inst["LEI_Code"].isin(bank_ids)].reset_index(drop=True)


def main():
    raw = load_exposures()
    n_reporting_banks = raw["LEI_Code"].nunique()
    print(f"Item {ITEM}, period {PERIOD}: {n_reporting_banks} banks report a nonzero-NACE exposure.")

    matrix = build_exposure_matrix(raw)

    row_totals = matrix.sum(axis=1)
    zero_rows = row_totals[row_totals == 0].index
    if len(zero_rows):
        print(f"Dropping {len(zero_rows)} bank(s) with zero total sector exposure: {list(zero_rows)}")
        matrix = matrix.drop(index=zero_rows)
        row_totals = row_totals.drop(index=zero_rows)

    print(f"Final exposure matrix: {matrix.shape[0]} banks x {matrix.shape[1]} sectors.")

    shares = matrix.div(row_totals, axis=0)

    sim = cosine_similarity(shares.values)
    sim_df = pd.DataFrame(sim, index=matrix.index, columns=matrix.index)

    meta = load_metadata(matrix.index)
    missing_meta = set(matrix.index) - set(meta["LEI_Code"])
    if missing_meta:
        print(f"Warning: {len(missing_meta)} bank(s) in the exposure matrix have no metadata match: {missing_meta}")

    matrix.to_csv(f"{OUTPUT_DIR}/bank_sector_exposure.csv")
    shares.to_csv(f"{OUTPUT_DIR}/bank_sector_shares.csv")
    sim_df.to_csv(f"{OUTPUT_DIR}/similarity_matrix.csv")
    meta.to_csv(f"{OUTPUT_DIR}/bank_metadata.csv", index=False)

    print(f"Wrote exposure matrix, shares, similarity matrix, and metadata to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
