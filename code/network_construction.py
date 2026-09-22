"""
Week 2: Network construction.

Builds the weighted similarity network from output/processed/similarity_matrix.csv
(produced by clean_data.py), computes centrality metrics for each bank, and merges
in capital ratio / size data from tr_oth.csv.

Output (output/processed/):
  network_metrics.csv - one row per bank: centrality metrics + capital ratio + size
"""

import sys

import networkx as nx
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = "data"
OUTPUT_DIR = "output/processed"
PERIOD = "202506"

# tr_oth.csv items (see data/TR_Metadata.xlsx for the full item dictionary)
ITEM_CET1_RATIO = "2520140"   # CET1 capital ratio (transitional)
ITEM_TOTAL_CAPITAL_RATIO = "2520142"  # Total capital ratio (transitional)
ITEM_TOTAL_ASSETS = "2521010"  # Total assets (size)


def build_graph(sim: pd.DataFrame) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(sim.index)
    banks = list(sim.index)
    for i, a in enumerate(banks):
        for b in banks[i + 1:]:
            g.add_edge(a, b, weight=sim.loc[a, b])
    return g


def compute_centralities(g: nx.Graph) -> pd.DataFrame:
    weighted_degree = dict(g.degree(weight="weight"))
    # betweenness treats "weight" as a distance/cost, so higher similarity
    # must mean a shorter path -> convert similarity to a distance
    distance_g = nx.Graph()
    distance_g.add_nodes_from(g.nodes)
    for u, v, data in g.edges(data=True):
        distance_g.add_edge(u, v, distance=1.0 - data["weight"])
    betweenness = nx.betweenness_centrality(distance_g, weight="distance")
    eigenvector = nx.eigenvector_centrality(g, weight="weight", max_iter=1000)
    pagerank = nx.pagerank(g, weight="weight")

    return pd.DataFrame({
        "weighted_degree": weighted_degree,
        "betweenness": betweenness,
        "eigenvector": eigenvector,
        "pagerank": pagerank,
    })


def load_capital_and_size():
    df = pd.read_csv(
        f"{DATA_DIR}/tr_oth.csv", dtype={"LEI_Code": str, "Period": str, "Item": str},
        usecols=["LEI_Code", "Period", "Item", "Amount"],
    )
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    df = df[df["Period"] == PERIOD]

    items = {
        ITEM_CET1_RATIO: "cet1_ratio",
        ITEM_TOTAL_CAPITAL_RATIO: "total_capital_ratio",
        ITEM_TOTAL_ASSETS: "total_assets",
    }
    df = df[df["Item"].isin(items)]
    wide = df.pivot_table(index="LEI_Code", columns="Item", values="Amount", aggfunc="first")
    wide = wide.rename(columns=items)
    return wide


def main():
    sim = pd.read_csv(f"{OUTPUT_DIR}/similarity_matrix.csv", index_col=0)
    sim.columns = sim.index  # column labels read back as strings, matches index already

    g = build_graph(sim)
    print(f"Graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges (complete, no cutoff).")

    centralities = compute_centralities(g)
    capital_size = load_capital_and_size()

    metrics = centralities.join(capital_size, how="left")
    missing = metrics[metrics[["cet1_ratio", "total_capital_ratio", "total_assets"]].isna().any(axis=1)]
    if len(missing):
        print(f"Warning: {len(missing)} bank(s) missing capital/size data: {list(missing.index)}")

    meta = pd.read_csv(f"{OUTPUT_DIR}/bank_metadata.csv", index_col="LEI_Code")
    metrics = metrics.join(meta[["Name", "Country", "Country_code"]], how="left")

    metrics.index.name = "LEI_Code"
    metrics = metrics.sort_values("weighted_degree", ascending=False)
    metrics.to_csv(f"{OUTPUT_DIR}/network_metrics.csv")

    print(f"Wrote network metrics for {len(metrics)} banks to {OUTPUT_DIR}/network_metrics.csv")
    print(metrics[["Name", "weighted_degree", "eigenvector", "pagerank", "cet1_ratio"]].head(10).to_string())


if __name__ == "__main__":
    main()
