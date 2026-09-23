"""
Statistical analysis.

Builds a trigger-side contagion severity measure:

    severity_i(sensitivity) = number of banks j != i for which
        similarity(i, j) * sensitivity > CONTAGION_THRESHOLD

i.e. how many banks cross the contagion threshold (code/attack_simulation.py)
when bank i is the attack trigger, computed for all 107 possible triggers.
This is the trigger-side "systemic importance" framing of contagion severity
(RQ2/RQ3: does network position predict a bank's potential to trigger
widespread contagion, vs. traditional measures like size and capital
ratio?), not a target/victim-side measure - a target-side severity measure
(risk received, averaged over all possible triggers) would reduce to a
fixed multiple of the receiving bank's own weighted-degree centrality and
so would be tautological against that predictor.

Sensitivity basis: PRIMARY_SENSITIVITY was chosen empirically from
SENSITIVITY_GRID (code/attack_simulation.py), excluding the confirmed
floor value 0.1 (below CONTAGION_THRESHOLD, guaranteed zero for every
possible trigger - see attack_simulation.py), by picking the value that
maximizes the variance of severity_i across banks - i.e. where the model
best discriminates high- from low-severity triggers. This is verified at
runtime (see report_sensitivity_search) rather than hard-coded blind: 0.5
has the highest variance among [0.26, 0.3, 0.5, 0.7, 0.9], with no ceiling
saturation. 0.7 and 0.9 show a genuine ceiling effect (47/107 and 74/107
banks respectively land within 90% of the maximum possible severity),
reported as a finding, not silently excluded. ROBUSTNESS_SENSITIVITIES
re-run the same models at 0.3, 0.7 and 0.9 to check whether conclusions
hold away from the primary value; 0.9 is flagged as ceiling-affected.

Centrality measures: weighted_degree, eigenvector and pagerank are
near-perfectly collinear in this network (r > 0.999 - verified; expected
for a complete weighted graph dominated by each node's total similarity
mass). Only betweenness is meaningfully distinct (r ~ 0.40 with the
others). Each centrality measure is therefore modeled separately against
severity rather than combined into one multivariate model, which would be
a textbook multicollinearity violation.

Public-sector agencies: identified by institution type (reviewed all 107
names for public-sector/municipal-funding-agency business models), not by
an arbitrary CET1 cutoff - defining the group by a ratio threshold would
describe the symptom (near-zero risk-weighted assets from lending almost
entirely to government-backed entities) rather than the structural cause.
Four banks match: Kommuninvest and Kuntarahoitus (Nordic municipal funding
agencies, CET1 353% and 89%), plus SFIL and BNG Bank (French/Dutch
local-government financing agencies, CET1 44% and 40%) - the latter two
were missed by an earlier CET1 > 50% filter, which is exactly why type is
the defining criterion and elevated CET1 is only corroborating evidence
(all four sit well above the sample's 75th percentile of ~20%). These are
genuine, not data errors, but would be extreme leverage points on the
cet1_ratio coefficient if left unflagged. Handled two ways in every model:
(1) included in the full N=107 sample with an is_public_sector_agency
control flag, and (2) a robustness spec excluding them entirely (N=103,
flag dropped as it would be constant). Total assets is log-transformed
(log_total_assets) as the size predictor - standard practice given its
heavy right skew in this sample (max/median ratio ~27x).

OLS: severity ~ centrality_measure + cet1_ratio + log_total_assets
  (+ is_public_sector_agency in the full-sample spec), one model per
  centrality measure x sensitivity x spec.

Classifier: severity at the primary sensitivity, binarized via median
split (high vs. low systemic-importance trigger - the same DV as the OLS,
just binarized, not a separate construct), logistic regression with the
same predictors (features standardized, since cet1_ratio and
log_total_assets are on very different scales and cet1_ratio still has
two outliers post-flagging), evaluated with leave-one-out cross-validation
(appropriate given N=107/105).

Output (output/processed/):
  contagion_severity.csv - per-bank severity at each analysed sensitivity,
    plus the high/low classification label
  ols_results.csv - coefficient/p-value/R2 per centrality measure x
    sensitivity x spec (full-sample-with-flag vs. excl-public-sector)
  classifier_results.csv - LOOCV accuracy per centrality measure x spec
"""

import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from attack_simulation import CONTAGION_THRESHOLD, SENSITIVITY_GRID

sys.stdout.reconfigure(encoding="utf-8")

OUTPUT_DIR = "output/processed"

CENTRALITY_MEASURES = ["weighted_degree", "betweenness", "eigenvector", "pagerank"]
CAPITAL_COL = "cet1_ratio"

PRIMARY_SENSITIVITY = 0.5
ROBUSTNESS_SENSITIVITIES = [0.3, 0.7, 0.9]  # 0.9 is ceiling-affected, see module docstring

PUBLIC_SECTOR_AGENCIES = [
    "EV2XZWMLLXF2QRX0CD47",  # Kommuninvest - Grupp (Sweden) - municipal funding agency
    "529900HEKOENJHPNN480",  # Kuntarahoitus Oyj (Finland) - municipal funding agency
    "549300HFEHJOXGE4ZE63",  # SFIL S.A. (France) - local-government financing agency
    "529900GGYMNGRQTDOO93",  # BNG Bank N.V. (Netherlands) - municipal funding agency
]


def compute_severity(sim: pd.DataFrame, sensitivity: float, threshold: float) -> pd.Series:
    above = (sim.values * sensitivity) > threshold
    np.fill_diagonal(above, False)
    return pd.Series(above.sum(axis=1), index=sim.index, name=f"severity_s{sensitivity}")


def report_sensitivity_search(sim: pd.DataFrame, threshold: float, candidates: list) -> float:
    print("Sensitivity search (excludes the confirmed floor 0.1):")
    n_others = len(sim) - 1
    variances = {}
    for s in candidates:
        sev = compute_severity(sim, s, threshold)
        near_ceiling = int((sev >= 0.9 * n_others).sum())
        variances[s] = sev.var()
        print(
            f"  s={s}: mean={sev.mean():.1f} var={sev.var():.1f} "
            f"range=[{sev.min()},{sev.max()}] near_ceiling={near_ceiling}/{len(sev)}"
        )
    best = max(variances, key=variances.get)
    print(f"  -> highest-variance sensitivity: {best}\n")
    return best


def build_features(metrics: pd.DataFrame) -> pd.DataFrame:
    df = metrics.copy()
    df["log_total_assets"] = np.log(df["total_assets"])
    df["is_public_sector_agency"] = df.index.isin(PUBLIC_SECTOR_AGENCIES).astype(int)
    return df


def run_ols(y: pd.Series, X: pd.DataFrame, label: str) -> sm.regression.linear_model.RegressionResultsWrapper:
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    print(f"--- OLS: {label} (N={int(model.nobs)}) ---")
    for name in X.columns:
        print(f"  {name:28s} coef={model.params[name]: .4f}  p={model.pvalues[name]:.4f}")
    print(f"  R2={model.rsquared:.4f}  Adj R2={model.rsquared_adj:.4f}\n")
    return model


def run_ols_pair(df: pd.DataFrame, y_col: str, centrality: str, sensitivity: float) -> tuple:
    X_main = df[[centrality, CAPITAL_COL, "log_total_assets", "is_public_sector_agency"]]
    model_main = run_ols(df[y_col], X_main, f"s={sensitivity} {centrality} (full N=107, w/ public-sector flag)")

    mask = df["is_public_sector_agency"] == 0
    X_robust = df.loc[mask, [centrality, CAPITAL_COL, "log_total_assets"]]
    model_robust = run_ols(df.loc[mask, y_col], X_robust, f"s={sensitivity} {centrality} (excl. public-sector agencies)")

    return model_main, model_robust


def run_logistic_loocv(df: pd.DataFrame, y_col: str, feature_cols: list, label: str) -> float:
    X = df[feature_cols].values
    y = df[y_col].values
    pipe = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])
    preds = cross_val_predict(pipe, X, y, cv=LeaveOneOut())
    acc = accuracy_score(y, preds)
    baseline = max(y.mean(), 1 - y.mean())
    print(f"--- Logistic + LOOCV: {label} --- accuracy={acc:.3f} (majority-class baseline={baseline:.3f}, N={len(y)})")
    return acc


def main():
    sim = pd.read_csv(f"{OUTPUT_DIR}/similarity_matrix.csv", index_col=0)
    sim.columns = sim.index
    metrics = pd.read_csv(f"{OUTPUT_DIR}/network_metrics.csv", index_col="LEI_Code")
    df = build_features(metrics)

    search_candidates = [s for s in SENSITIVITY_GRID if s >= CONTAGION_THRESHOLD]
    best = report_sensitivity_search(sim, CONTAGION_THRESHOLD, search_candidates)
    if best != PRIMARY_SENSITIVITY:
        print(f"WARNING: highest-variance sensitivity ({best}) != documented PRIMARY_SENSITIVITY ({PRIMARY_SENSITIVITY}) - re-check.\n")

    all_sensitivities = [PRIMARY_SENSITIVITY] + ROBUSTNESS_SENSITIVITIES
    for s in all_sensitivities:
        df[f"severity_s{s}"] = compute_severity(sim, s, CONTAGION_THRESHOLD).reindex(df.index)

    primary_col = f"severity_s{PRIMARY_SENSITIVITY}"
    median = df[primary_col].median()
    df["high_severity"] = (df[primary_col] > median).astype(int)
    print(f"Median split on {primary_col} (median={median}): {df['high_severity'].value_counts().to_dict()}\n")

    print("Correlation of primary severity with centrality measures (checking against tautology):")
    print(df[[primary_col] + CENTRALITY_MEASURES].corr()[primary_col].to_string())
    print()

    ols_rows = []
    for s in all_sensitivities:
        y_col = f"severity_s{s}"
        for c in CENTRALITY_MEASURES:
            model_main, model_robust = run_ols_pair(df, y_col, c, s)
            ols_rows.append({
                "sensitivity": s, "centrality_measure": c, "spec": "full_with_flag",
                "coef": model_main.params[c], "pvalue": model_main.pvalues[c],
                "r_squared": model_main.rsquared, "n": int(model_main.nobs),
            })
            ols_rows.append({
                "sensitivity": s, "centrality_measure": c, "spec": "excl_public_sector",
                "coef": model_robust.params[c], "pvalue": model_robust.pvalues[c],
                "r_squared": model_robust.rsquared, "n": int(model_robust.nobs),
            })
    ols_df = pd.DataFrame(ols_rows)
    ols_df.to_csv(f"{OUTPUT_DIR}/ols_results.csv", index=False)

    print("=== Classifier (logistic + LOOCV, primary sensitivity only) ===")
    clf_rows = []
    mask = df["is_public_sector_agency"] == 0
    for c in CENTRALITY_MEASURES:
        acc_main = run_logistic_loocv(
            df, "high_severity", [c, CAPITAL_COL, "log_total_assets", "is_public_sector_agency"],
            f"{c} (full N=107, w/ flag)",
        )
        acc_robust = run_logistic_loocv(
            df[mask], "high_severity", [c, CAPITAL_COL, "log_total_assets"],
            f"{c} (excl. public-sector agencies)",
        )
        clf_rows.append({"centrality_measure": c, "spec": "full_with_flag", "loocv_accuracy": acc_main})
        clf_rows.append({"centrality_measure": c, "spec": "excl_public_sector", "loocv_accuracy": acc_robust})
    clf_df = pd.DataFrame(clf_rows)
    clf_df.to_csv(f"{OUTPUT_DIR}/classifier_results.csv", index=False)

    severity_cols = [f"severity_s{s}" for s in all_sensitivities]
    out = df[severity_cols + ["high_severity", "Name", "Country"]]
    out.to_csv(f"{OUTPUT_DIR}/contagion_severity.csv")

    print(f"\nWrote severity, OLS, and classifier results to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
