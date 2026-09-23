# Network-Based Informational Contagion in EU Banking Systems

Dissertation project modelling informational contagion between EU banks: bad
news about one bank raises perceived risk in other banks based on similarity
of sector exposure, not direct financial linkage.

## Data

Source: EBA 2025 EU-wide Transparency Exercise (120 banks, 25 countries).
Raw files live in `data/` and are **not** tracked in git (see `.gitignore`) —
they're available at [OneDrive link — add before submission].

- `tr_cre.csv` — credit risk exposures by NACE sector (filtered to item 2521301)
- `tr_oth.csv` — capital ratios, total assets, other bank-level figures
- `TR_Metadata.xlsx` — bank names, countries, NACE sector labels

## Pipeline

1. `code/clean_data.py` — filters `tr_cre.csv` to the sector-exposure item
   (latest period, 202506), builds a bank x NACE-sector exposure matrix,
   normalizes rows to exposure shares, and computes the cosine similarity
   matrix used as network edge weights.
2. `code/network_construction.py` — builds a complete weighted graph from
   the similarity matrix (NetworkX), computes centrality metrics (weighted
   degree, betweenness, eigenvector, PageRank), and merges in capital ratio
   / total assets from `tr_oth.csv`.
3. `code/attack_simulation.py` — implements the interaction rule
   (`risk_transmitted(A -> B) = similarity(A, B) * sensitivity`, with bad
   news at the trigger bank modeled as a standardized shock of 1) and runs
   one single-bank attack as an end-to-end test case, flagging banks whose
   transmitted risk exceeds a stated criterion (`CONTAGION_THRESHOLD`, a
   fraction of the full shock magnitude). Also defines the sensitivity grid
   (`SENSITIVITY_GRID`) to be swept in later robustness testing.
4. `code/statistical_analysis.py` — builds a trigger-side contagion
   severity measure (banks crossing the threshold when each bank in turn
   is the trigger, across all 107 banks), then runs OLS (severity ~
   centrality + CET1 ratio + log total assets, one model per centrality
   measure) and a logistic classifier with leave-one-out cross-validation
   on a median-split high/low severity label (RQ2/RQ3). Both include a
   robustness spec excluding 4 public-sector/municipal funding agencies
   with structurally elevated capital ratios.

Run in order from the project root:

```
venv/Scripts/python.exe code/clean_data.py
venv/Scripts/python.exe code/network_construction.py
venv/Scripts/python.exe code/attack_simulation.py
venv/Scripts/python.exe code/statistical_analysis.py
```

Outputs land in `output/processed/` (tracked in git — small derived CSVs,
not raw data).

## Setup

```
python -m venv venv
venv/Scripts/python.exe -m pip install -r requirements.txt
```

## Notes

- 120 banks are in the EBA population; 107 remain after filtering to banks
  that report the sector-exposure item and dropping EBA's placeholder
  "Other banks" aggregate row (LEI `XXXXXXXXXXXXXXXXXXXX`).
- Cosine similarity is computed over exposure shares, but is mathematically
  identical to computing it over raw exposure amounts (cosine similarity is
  invariant to positive per-row scaling) — shares are kept as an output
  because they're the more interpretable object.
