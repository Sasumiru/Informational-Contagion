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

Run in order from the project root:

```
venv/Scripts/python.exe code/clean_data.py
venv/Scripts/python.exe code/network_construction.py
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
