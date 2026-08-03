#!/usr/bin/env python3
"""
fairness_eval.py
================
Fairness across skin tone (ITA) for the SegFormer-B5 baseline, on all 185
consensus test images. Ported VERBATIM from the core-model analysis notebooks
(`fairness_analysis` in the saved-analysis generator) so the CSVs drop straight
into the existing fairness comparison (`results_final/fairness_per_group.csv`
schema): per-ITA-group median Dice + IQR + bootstrap 95% CI, mean recall,
complete-miss rate, Kruskal-Wallis across groups, Bonferroni-corrected pairwise
Mann-Whitney.

Runs on the per-image CSVs written by train_segformer_b5_baseline.py (averaged
over seeds if several were trained) -- no GPU needed.

    python fairness_eval.py \
        --out-dir results_segformer_b5 \
        --ita-csv ita_labels/wl_test_per_image_ita.csv \
        --seeds 42

CAVEAT: exploratory at n=28 subjects -- each ITA group has only ~9-17 subjects,
so read direction, not significance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as _st


def _bootstrap_ci(values, n=2000, seed=0):
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    meds = [np.median(rng.choice(values, size=len(values), replace=True)) for _ in range(n)]
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def fairness_analysis(per_image_df: pd.DataFrame, ita: pd.DataFrame, model_name: str) -> dict:
    df = per_image_df.merge(ita, on="stem", how="left", validate="one_to_one")
    assert df["ita_group_index_5"].notna().all(), "stems missing ITA labels"
    per_group, samples = [], []
    for gidx, g in sorted(df.groupby("ita_group_index_5"), key=lambda kv: kv[0]):
        vals = g["dice"].to_numpy()
        lo, hi = _bootstrap_ci(vals)
        per_group.append({"model": model_name, "ita_group_index_5": int(gidx),
                          "skin_tone_category": g["skin_tone_category"].iloc[0], "n_images": len(g),
                          "median_dice": float(np.median(vals)),
                          "iqr_dice": float(np.percentile(vals, 75) - np.percentile(vals, 25)),
                          "ci95_lo": lo, "ci95_hi": hi, "mean_recall": float(g["recall"].mean()),
                          "miss_rate": float(((g["pred_positive_pixels"] == 0) & (g["gt_positive_pixels"] > 0)).mean())})
        samples.append(vals)
    H, p = _st.kruskal(*samples)
    pairs = [(i, j) for i in range(len(samples)) for j in range(i + 1, len(samples))]
    pairwise = []
    for i, j in pairs:
        pv = _st.mannwhitneyu(samples[i], samples[j], alternative="two-sided").pvalue
        adj = min(1.0, pv * len(pairs))
        pairwise.append({"model": model_name, "group_a": per_group[i]["skin_tone_category"],
                         "group_b": per_group[j]["skin_tone_category"], "pvalue": pv,
                         "bonferroni_p": adj, "significant": bool(adj < 0.05)})
    pg = pd.DataFrame(per_group)
    best, worst = pg.loc[pg["median_dice"].idxmax()], pg.loc[pg["median_dice"].idxmin()]
    stat = {"model": model_name, "kruskal_H": float(H), "kruskal_p": float(p), "significant": bool(p < 0.05),
            "fairness_gap": float(best["median_dice"] - worst["median_dice"]),
            "best_group": best["skin_tone_category"], "worst_group": worst["skin_tone_category"],
            "max_miss_rate_gap": float(pg["miss_rate"].max() - pg["miss_rate"].min())}
    return {"per_group": pg, "pairwise": pd.DataFrame(pairwise), "stats": stat}


def main():
    p = argparse.ArgumentParser(description="ITA fairness eval for the SegFormer-B5 baseline.")
    p.add_argument("--out-dir", required=True, help="results_segformer_b5 dir (contains runs/ + results/).")
    p.add_argument("--ita-csv", required=True, help="ita_labels/wl_test_per_image_ita.csv (bundled).")
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--model-name", default="segformer_b5")
    a = p.parse_args()

    out_dir = Path(a.out_dir)
    res_dir = out_dir / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    # per-image test results, averaged over seeds (same as the baselines notebook §10)
    frames = []
    for s in a.seeds:
        f = out_dir / "runs" / f"{a.model_name}__seed{s}" / "test_per_image.csv"
        if f.exists():
            frames.append(pd.read_csv(f))
        else:
            print(f"  !! no test_per_image.csv for seed {s} ({f}) -- skipping")
    if not frames:
        raise SystemExit("no per-image test CSVs found -- run training/eval first.")
    per_image = (pd.concat(frames).groupby("stem", as_index=False)
                 .agg({"dice": "mean", "recall": "mean",
                       "pred_positive_pixels": "mean", "gt_positive_pixels": "first"}))
    assert len(per_image) == 185, f"expected 185 test images, got {len(per_image)}"

    ita = pd.read_csv(a.ita_csv)[["stem", "skin_tone_category", "ita_group_index_5"]]
    out = fairness_analysis(per_image, ita, a.model_name)

    out["per_group"].to_csv(res_dir / f"{a.model_name}_fairness_per_group.csv", index=False)
    out["pairwise"].to_csv(res_dir / f"{a.model_name}_fairness_pairwise.csv", index=False)
    pd.DataFrame([out["stats"]]).to_csv(res_dir / f"{a.model_name}_fairness_stats.csv", index=False)

    print("\n" + "=" * 72 + f"\nFairness by ITA group -- {a.model_name} "
          f"({len(frames)} seed[s], 185 test images)\n" + "=" * 72)
    print(out["per_group"][["skin_tone_category", "n_images", "median_dice", "iqr_dice",
                            "ci95_lo", "ci95_hi", "mean_recall", "miss_rate"]].to_string(index=False))
    s = out["stats"]
    print(f"\nKruskal-Wallis H={s['kruskal_H']:.2f} p={s['kruskal_p']:.4f} (significant={s['significant']})")
    print(f"fairness gap (median Dice): {s['fairness_gap']:.4f}  best={s['best_group']}  worst={s['worst_group']}")
    print(f"max miss-rate gap: {s['max_miss_rate_gap']*100:.2f}%")
    sig = out["pairwise"][out["pairwise"].significant]
    print("pairwise (Bonferroni) significant:", "none" if sig.empty else f"\n{sig.to_string(index=False)}")

    # optional bar chart (matplotlib present on most envs; skip silently if not)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        order = ["Light (II-III)", "Intermediate (III-IV)", "Tan (IV)", "Brown (V)", "Dark (VI)"]
        pg = out["per_group"].set_index("skin_tone_category").reindex(
            [g for g in order if g in set(out["per_group"].skin_tone_category)])
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(pg.index, pg["median_dice"],
               yerr=[pg["median_dice"] - pg["ci95_lo"], pg["ci95_hi"] - pg["median_dice"]],
               capsize=4, color="#4477aa")
        ax.set_ylabel("median Dice"); ax.set_ylim(0, 1); ax.grid(axis="y", alpha=0.3)
        ax.set_title(f"{a.model_name} median Dice by ITA group (bootstrap 95% CI, exploratory n=28)")
        plt.xticks(rotation=15); plt.tight_layout()
        png = res_dir / f"{a.model_name}_fairness_by_group.png"
        plt.savefig(png, dpi=140, bbox_inches="tight")
        print("chart ->", png)
    except ImportError:
        print("(matplotlib not installed -- table only)")

    print("outputs ->", res_dir)


if __name__ == "__main__":
    main()
