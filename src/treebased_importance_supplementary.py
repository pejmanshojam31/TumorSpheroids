"""
Supplementary feature-importance analysis for the TreeBasedSelector pipeline.

Refits TreeBasedSelector (a RandomForestClassifier wrapped in
sklearn.feature_selection.SelectFromModel) on the saved per-range training
data, extracts its Gini feature importances, aggregates each base radiomic
feature's importance across its day-resolved variants, ranks features within
each observation range, and averages the rank across short/mid/long.

Produces:
    Results/rf_top10.csv                        top 10 features by mean rank
    Results/rf_top15.csv                        top 15 features (used by the figure)
    Results/rf_n_selected.json                  features retained per range
    Results/rf_importance_robust_features.pdf   contribution-rank heatmap
    Results/rf_importance_robust_features.png   (same, PNG)

Run from anywhere; paths are resolved relative to this file's location in
src/, not hardcoded to any one machine.
"""
import os
import re
import sys
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WORK, "src"))

from MLTimeSeriesModel import MLTimeSeriesModel  # noqa: E402

RES = os.path.join(WORK, "Results")

CLASS_MAP = {
    "firstorder": "FO", "glcm": "GLCM", "gldm": "GLDM",
    "glrlm": "GLRLM", "glszm": "GLSZM", "ngtdm": "NGTDM",
}

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 150,
})


def prettify(base_name: str) -> str:
    """'HighGrayLevelRunEmphasis' -> 'High Gray Level Run Emphasis'."""
    return re.sub(r'(?<=[a-z0-9])(?=[A-Z])', ' ', base_name)


def base_feature_and_class(col: str):
    """'original_firstorder_90Percentile_day0' -> ('FO', '90Percentile')."""
    m = re.match(r'original_([a-z]+)_(.+?)(?:_day\d+)?$', col)
    if not m:
        return None, None
    cls_key, base = m.group(1), m.group(2)
    return CLASS_MAP.get(cls_key, cls_key.upper()), base


def compute_importances():
    """Refit TreeBasedSelector per range and rank aggregated feature importances."""
    range_results = {}
    n_selected = {}

    for label in ["short", "mid", "long"]:
        train_df = pd.read_csv(os.path.join(RES, f"{label}_train.csv"))
        X_train = train_df.drop(columns=["diagnosis"])
        y_train = train_df["diagnosis"]

        m = MLTimeSeriesModel()
        m.select_feature_selection("TreeBasedSelector")
        m.select_model("MLPClassifier")  # required by build_pipeline(), unused here
        m.build_pipeline()
        m.fit(X_train, y_train)

        var_support = m.pipeline.named_steps['low_variance_filter'].get_support()
        post_var_cols = X_train.columns[var_support]

        sfm = m.pipeline.named_steps['feature_selection']
        importances = sfm.estimator_.feature_importances_
        selected_mask = sfm.get_support()
        assert len(importances) == len(post_var_cols)
        n_selected[label] = int(selected_mask.sum())

        imp_df = pd.DataFrame({"column": post_var_cols, "importance": importances})

        # Aggregate (sum) importance across day-suffixed variants of the same base feature.
        agg, cls_of = {}, {}
        for _, row in imp_df.iterrows():
            cls, base = base_feature_and_class(row["column"])
            if base is None:
                continue
            agg[base] = agg.get(base, 0.0) + row["importance"]
            cls_of[base] = cls

        agg_df = pd.DataFrame({
            "base": list(agg.keys()),
            "class": [cls_of[b] for b in agg.keys()],
            "importance": list(agg.values()),
        })
        agg_df["rank"] = agg_df["importance"].rank(ascending=False, method="min").astype(int)
        range_results[label] = agg_df.sort_values("rank").set_index("base")

        print(f"[{label}] post-variance-filter features: {len(post_var_cols)}, "
              f"selected by TreeBasedSelector (importance >= mean): {n_selected[label]}")

    all_bases = (set(range_results["short"].index)
                 & set(range_results["mid"].index)
                 & set(range_results["long"].index))
    rows = []
    for base in all_bases:
        r_short = range_results["short"].loc[base, "rank"]
        r_mid = range_results["mid"].loc[base, "rank"]
        r_long = range_results["long"].loc[base, "rank"]
        rows.append({
            "feature": prettify(base), "class": range_results["short"].loc[base, "class"],
            "short": int(r_short), "mid": int(r_mid), "long": int(r_long),
            "mean_rank": (r_short + r_mid + r_long) / 3,
        })
    combined = pd.DataFrame(rows).sort_values("mean_rank").reset_index(drop=True)
    return combined, n_selected


def fig_robust(top15: pd.DataFrame):
    """Contribution-rank heatmap: one row per feature, one column per observation
    range, colour and annotation showing that feature's rank within the range
    (1 = most important)."""
    top = top15.iloc[::-1]
    mat = top[["short", "mid", "long"]].to_numpy().astype(int)
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    im = ax.imshow(mat, cmap="viridis_r", aspect="auto", vmin=1, vmax=max(30, mat.max()))
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Short", "Mid", "Long"])
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"].tolist(), fontsize=8)
    ax.set_title("Contribution rank of the most robust features\n(1 = most influential)")
    for i in range(mat.shape[0]):
        for k in range(mat.shape[1]):
            ax.text(k, i, f"{mat[i, k]}", ha="center", va="center",
                    color="white" if mat[i, k] > 15 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Rank (1 = top)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RES, f"rf_importance_robust_features.{ext}"), bbox_inches="tight")
    plt.close(fig)


def main():
    combined, n_selected = compute_importances()
    top10, top15 = combined.head(10), combined.head(15)

    top10.to_csv(os.path.join(RES, "rf_top10.csv"), index=False)
    top15.to_csv(os.path.join(RES, "rf_top15.csv"), index=False)
    with open(os.path.join(RES, "rf_n_selected.json"), "w") as f:
        json.dump(n_selected, f, indent=2)

    fig_robust(top15)

    print("\n=== TOP 10 (mean rank across short/mid/long) ===")
    print(top10[["feature", "class", "short", "mid", "long", "mean_rank"]].to_string(index=False))
    print(f"\nWrote: {RES}/rf_top10.csv, rf_top15.csv, rf_n_selected.json,")
    print(f"       rf_importance_robust_features.pdf/.png")


if __name__ == "__main__":
    main()
