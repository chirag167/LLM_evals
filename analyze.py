"""
analyze.py  (steps 6-8: statistical error, systematic error, conclusions)

Reads results.csv from run_eval.py and compares two models.

The central object is a table D of per-question differences:
    D[config, question] = correct_A - correct_B      (each entry -1, 0 or +1)
One row per knob setting ("config"), one column per question.
Every number below is computed from D.

Run:  python analyze.py results.csv model_A model_B
"""

import sys
import numpy as np
import pandas as pd

KNOBS = ["template", "n_shots", "option_order", "extraction"]


# =====================================================================
# Building the difference table
# =====================================================================

def difference_table(df, model_a, model_b):
    """Returns D (DataFrame: configs x questions) and the subject of each
    question (Series indexed by question_id)."""
    wide = df.pivot_table(index=KNOBS + ["question_id"], columns="model",
                          values="correct")
    d = wide[model_a] - wide[model_b]           # pairing happens here
    D = d.unstack("question_id")
    subjects = df.drop_duplicates("question_id").set_index("question_id")["subject"]
    return D, subjects.loc[D.columns]


# =====================================================================
# Step 6: statistical error for each configuration
# =====================================================================

def paired_se(d):
    """Miller Eq. 7: SE of the mean of per-question differences."""
    return d.std(ddof=1) / np.sqrt(len(d))


def clustered_paired_se(d, clusters):
    """Miller Eq. 8: paired SE allowing questions in the same subject
    to be correlated."""
    resid = d - d.mean()
    sums = resid.groupby(clusters.values).sum()
    return np.sqrt((sums ** 2).sum()) / len(d)


def per_config_table(D, subjects):
    """One row per configuration: the difference and its two error bars."""
    rows = []
    for config, d in D.iterrows():
        rows.append({**dict(zip(KNOBS, config)),
                     "delta": d.mean(),
                     "se_paired": paired_se(d),
                     "se_clustered": clustered_paired_se(d, subjects)})
    table = pd.DataFrame(rows)
    table["z"] = table["delta"] / table["se_clustered"]
    return table


# =====================================================================
# Step 7: systematic error
# =====================================================================

def systematic_spread(table):
    """Standard deviation of delta across all configurations."""
    return table["delta"].std(ddof=1)


def knob_budget(table):
    """For each knob: average delta at each of its settings (averaging over
    all other knobs), and the shift = (largest - smallest) / 2.
    This splits the systematic error by source."""
    out = []
    for knob in KNOBS:
        means = table.groupby(knob)["delta"].mean()
        out.append({"knob": knob,
                    "delta_by_setting": {str(k): round(100 * v, 2)
                                         for k, v in means.items()},
                    "shift_points": 100 * (means.max() - means.min()) / 2})
    return pd.DataFrame(out)


def bootstrap(D, n_boot=2000, seed=0):
    """Resample QUESTIONS with replacement (the same draw for every config,
    so pairing is kept). For each draw, record:
      - the systematic spread across configs  (how stable is step 7?)
      - delta for one randomly chosen config  (stat + sys together)"""
    rng = np.random.default_rng(seed)
    M = D.to_numpy()                       # configs x questions
    n_cfg, n_q = M.shape
    spreads = np.empty(n_boot)
    combined = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_q, size=n_q)
        deltas = M[:, idx].mean(axis=1)
        spreads[b] = deltas.std(ddof=1)
        combined[b] = deltas[rng.integers(n_cfg)]
    return spreads, combined


# =====================================================================
# Step 8: the questions the project asks
# =====================================================================

def report(table, budget, spreads, combined, model_a, model_b):
    pts = lambda x: f"{100 * x:+.2f}"
    print(f"\nComparison: {model_a} minus {model_b}, "
          f"{len(table)} configurations\n")

    print("Per-configuration results (points):")
    show = table.copy()
    for col in ["delta", "se_paired", "se_clustered"]:
        show[col] = (100 * show[col]).round(2)
    print(show.round({"z": 2}).to_string(index=False))

    stat = table["se_clustered"].median()
    sys_ = systematic_spread(table)
    print(f"\nMean delta over configs : {pts(table['delta'].mean())} points")
    print(f"Statistical error (median clustered SE): {100*stat:.2f}")
    print(f"Systematic error (spread across configs): {100*sys_:.2f}"
          f"   [bootstrap 68% range "
          f"{100*np.percentile(spreads,16):.2f} to "
          f"{100*np.percentile(spreads,84):.2f}]")

    print("\nSystematic budget by knob (shift in points):")
    print(budget.round({"shift_points": 2}).to_string(index=False))

    print(f"\nConfigs where {model_b} comes out ahead: "
          f"{(table['delta'] < 0).sum()} of {len(table)}")
    print(f"Configs where the gap is significant (|z| > 1.96): "
          f"{(table['z'].abs() > 1.96).sum()} of {len(table)}")
    print(f"P({model_a} better), stat + sys combined: "
          f"{(combined > 0).mean():.3f}")
    lo, hi = np.percentile(combined, [2.5, 97.5])
    print(f"95% range of delta, stat + sys combined: "
          f"{pts(lo)} to {pts(hi)} points")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
    model_a = sys.argv[2] if len(sys.argv) > 2 else "model_A"
    model_b = sys.argv[3] if len(sys.argv) > 3 else "model_B"

    df = pd.read_csv(path)
    D, subjects = difference_table(df, model_a, model_b)
    table = per_config_table(D, subjects)
    budget = knob_budget(table)
    spreads, combined = bootstrap(D)
    report(table, budget, spreads, combined, model_a, model_b)
