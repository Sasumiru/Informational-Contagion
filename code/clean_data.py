"""
Data pipeline

Builds bank-by-sector exposure matrix from EBA 2025 EU-wide
Transparency Exercise (tr_cre.csv), 

normalizes it into exposure shares,
and then computes the cosine similarity matrix used as contagion network's
edge weights.

Files Outputted:
  bank_sector_exposure.csv  - raw exposure amounts, banks x NACE sectors
  bank_sector_shares.csv    - row-normalized exposure shares (each row sums to 1)
  similarity_matrix.csv     - cosine similarity, banks x banks
  bank_metadata.csv         - LEI_Code, Name, Country for surviving banks
"""

import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
ITEM = "2521301"        # EBA's code for "credit risk exposure by NACE sector" (the raw file mixes many item types together)
PERIOD = "202506"       # June 2025, the latest quarter available 
NACE_EXCLUDE = "0"      # EBA's placeholder for "no sector breakdown reported"
DATA_DIR = "data"  # folder holding the raw input files (tr_cre.csv, tr_oth.csv, TR_Metadata.xlsx)
OUTPUT_DIR = "output/processed"  # folder where the cleaned/derived CSVs get written


# raw data isn't tracked in git (see README) - fail with a clear message
# instead of a confusing pandas error if it hasn't been downloaded yet
def check_data_files():
    required = ["tr_cre.csv", "TR_Metadata.xlsx"]
    missing = [f for f in required if not os.path.exists(f"{DATA_DIR}/{f}")]
    if missing:
        raise FileNotFoundError(
            f"Missing file(s) in '{DATA_DIR}/': {missing}. "
            f"Raw data isn't tracked in git - see README for where to get it."
        )


# reads and filters raw exposure data down to one item/period, only on sector rows
def filterExpose():
    column = ["LEI_Code", "Period", "Item", "NACE_codes", "Amount"]

    # reading columns as text 
    strType = {
        "LEI_Code": str,
        "Period": str,
        "Item": str,
        "NACE_codes": str,
    }
    df = pd.read_csv(f"{DATA_DIR}/tr_cre.csv", usecols=column, dtype=strType)

    # turn things that cant be read as a number into 0
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    df["Amount"] = df["Amount"].fillna(0.0)

    # keep only rows for the sector-exposure item, in the one quarter we care about
    is_right_item = df["Item"] == ITEM
    is_right_period = df["Period"] == PERIOD
    df = df[is_right_item & is_right_period]

    # drop rows where no sector breakdown was given
    df = df[df["NACE_codes"] != NACE_EXCLUDE]
    # "XXXXXXXXXXXXXXXXXXXX" is EBA's placeholder LEI for a residual "Other banks"
    # aggregate row, not a real individual bank - drop it so it isn't treated as one
    df = df[df["LEI_Code"] != "XXXXXXXXXXXXXXXXXXXX"]
    return df


# reshapes new rows into one row per bank and one column per sector
def buildMatrix(df):
    # turns bank into table with one row per bank, one column per sector, filled with the exposure amount
    matrix = df.pivot_table(
        index="LEI_Code",       # each row = one bank
        columns="NACE_codes",   # each column = one sector
        values="Amount",        # the number that goes in each cell
        aggfunc="sum",          # if a bank has more than one row for the same sector, add them up
        fill_value=0.0,         # if a bank has no exposure to a sector, put 0 instead of leaving it blank
    )

    # pivot_table sorts columns as text by default (1, 10, 11, ... 2, 3, ...),
    # so re-sort them as numbers instead to get 1, 2, 3, ... 19
    sector_columns_in_order = sorted(matrix.columns, key=int)
    matrix = matrix[sector_columns_in_order]

    return matrix

# looks up bank name/country for given LEI codes
def load_metadata(bank_ids):
    # read the "List of Institutions" sheet - row 1 is a title, so the real
    # column headers start on the next row (header=1 means "skip 1 row")
    inst = pd.read_excel(
        f"{DATA_DIR}/TR_Metadata.xlsx",
        sheet_name="List of Institutions",
        header=1,
    )

    # the file calls these columns "Country" (a code) and "Desc_country" (the
    # full name) - rename them so it's obvious which is which
    inst = inst.rename(columns={
        "Country": "Country_code",
        "Desc_country": "Country",
    })

    # keep only the columns we actually need
    inst = inst[["LEI_Code", "Name", "Country", "Country_code"]]

    # only keep the banks that are actually in our exposure matrix
    mask = inst["LEI_Code"].isin(bank_ids)                    # True/False: is this LEI in our list?
    matching_banks = inst[mask]                                # keep only the True rows
    matching_banks = matching_banks.reset_index(drop=True)     # renumber rows 0,1,2... cleanly

    return matching_banks

# runs the full pipeline and writes all four output CSVs
def main():
    check_data_files()

    # step 1: load and filter the raw data
    raw = filterExpose()
    n_reporting_banks = raw["LEI_Code"].nunique()
    print(f"Item {ITEM}, period {PERIOD}: {n_reporting_banks} banks report a nonzero-NACE exposure.")

    # step 2: reshape it into one row per bank, one column per sector
    matrix = buildMatrix(raw)

    # step 3: drop any bank with no exposure at all - you can't turn a row of
    # all zeros into percentages, that would mean dividing by zero
    row_totals = matrix.sum(axis=1)
    has_no_exposure = row_totals == 0
    zero_rows = row_totals[has_no_exposure].index
    if len(zero_rows):
        print(f"Dropping {len(zero_rows)} bank(s) with zero total sector exposure: {list(zero_rows)}")
        matrix = matrix.drop(index=zero_rows)
        row_totals = row_totals.drop(index=zero_rows)

    print(f"Final exposure matrix: {matrix.shape[0]} banks x {matrix.shape[1]} sectors.")

    # step 4: turn raw amounts into shares - divide each bank's row by its
    # own total, so every bank's sectors add up to 1 (100%)
    shares = matrix.div(row_totals, axis=0)

    # step 5: compare every bank's share profile to every other bank's -
    # this gives a bank x bank table of similarity scores
    sim = cosine_similarity(shares.values)
    sim_df = pd.DataFrame(sim, index=matrix.index, columns=matrix.index)

    # step 6: attach bank names/countries, and warn if any bank has no match
    meta = load_metadata(matrix.index)
    all_banks = set(matrix.index)
    known_banks = set(meta["LEI_Code"])
    missing_meta = all_banks - known_banks
    if missing_meta:
        print(f"Warning: {len(missing_meta)} bank(s) in the exposure matrix have no metadata match: {missing_meta}")

    # step 7: save all four results as CSV files
    matrix.to_csv(f"{OUTPUT_DIR}/bank_sector_exposure.csv")
    shares.to_csv(f"{OUTPUT_DIR}/bank_sector_shares.csv")
    sim_df.to_csv(f"{OUTPUT_DIR}/similarity_matrix.csv")
    meta.to_csv(f"{OUTPUT_DIR}/bank_metadata.csv", index=False)

    print(f"Wrote exposure matrix, shares, similarity matrix, and metadata to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
