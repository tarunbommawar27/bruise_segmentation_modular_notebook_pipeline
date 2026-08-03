#!/usr/bin/env python3
"""
aggregate_report.py — score every arm against the reference with the success
hierarchy from the review (point 3):

  1. NON-INFERIORITY in subject-level Dice   (primary gate; margin --ni-margin)
  2. reduction in complete-miss + low-recall failures  (miss, %recall<0.10, %Dice<0.20)
  3. improved small-bruise recall
  4. improved worst-ITA-group recall  (exploratory: few subjects/group)
  5. lower deployment cost  (params/latency — reported, not bootstrapped)

All effects use a paired subject-cluster bootstrap (28 subjects). Dice is NOT
framed as ceiling-limited; the claim is "the compact student preserves overall
segmentation quality while reducing high-consequence failures and cost."

Verdicts:
  REPORTABLE WIN  = non-inferior subject Dice AND >=1 failure/subgroup endpoint
                    improves with CI clear of null
  NON-INFERIOR    = non-inferior subject Dice, no failure-tail gain
  INFERIOR        = fails subject-Dice non-inferiority

    python aggregate_report.py --ref <ref_test_per_image.csv> --ita <ita.csv> \
        --arms name1=path1/test_per_image.csv ... --reps 5000 \
        --ni-margin 0.03 --out aggregate_out
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd


def load(path):
    d = pd.read_csv(path)
    cols = ["stem", "dice", "recall", "pred_positive_pixels", "gt_positive_pixels"]
    d = d[[c for c in cols if c in d.columns]].copy()
    d["miss"] = ((d["pred_positive_pixels"] == 0) & (d["gt_positive_pixels"] > 0)).astype(float)
    return d


def merged(ref, arm, ita):
    m = ref.merge(arm, on="stem", suffixes=("_ref", "_arm")).merge(ita, on="stem")
    m["gt_positive_pixels"] = m["gt_positive_pixels_ref"]
    return m


def boot(df, small_mask, reps, rng):
    subj = df["subject"].to_numpy(); uniq = np.unique(subj)
    idxby = {s: np.where(subj == s)[0] for s in uniq}
    d_a, d_r = df["dice_arm"].to_numpy(), df["dice_ref"].to_numpy()
    r_a, r_r = df["recall_arm"].to_numpy(), df["recall_ref"].to_numpy()
    m_a, m_r = df["miss_arm"].to_numpy(), df["miss_ref"].to_numpy()
    ita = df["ita_group_index_5"].to_numpy(); sm = small_mask.to_numpy()
    subj_arr = subj

    def subj_median_dice(ix, d):
        s = subj_arr[ix]
        order = np.argsort(s, kind="stable")
        s2, d2 = s[order], d[ix][order]
        # per-subject mean, then median over subjects
        uniq2, start = np.unique(s2, return_index=True)
        means = np.array([d2[a:b].mean() for a, b in zip(start, list(start[1:]) + [len(d2)])])
        return np.median(means)

    def stat(ix):
        o = {}
        o["subj_median_dice"] = subj_median_dice(ix, d_a) - subj_median_dice(ix, d_r)
        o["median_dice"] = np.median(d_a[ix]) - np.median(d_r[ix])
        o["miss"] = m_a[ix].mean() - m_r[ix].mean()
        o["pct_rec10"] = (r_a[ix] < 0.10).mean() - (r_r[ix] < 0.10).mean()
        o["pct_dice20"] = (d_a[ix] < 0.20).mean() - (d_r[ix] < 0.20).mean()
        smx = ix[sm[ix]]
        o["small_recall"] = (r_a[smx].mean() - r_r[smx].mean()) if smx.size else np.nan
        # p5 subject dice (arm - ref)
        def p5(d):
            s = subj_arr[ix]; order = np.argsort(s, kind="stable")
            s2, d2 = s[order], d[ix][order]
            _, start = np.unique(s2, return_index=True)
            means = np.array([d2[a:b].mean() for a, b in zip(start, list(start[1:]) + [len(d2)])])
            return np.percentile(means, 5)
        o["p5_subj_dice"] = p5(d_a) - p5(d_r)
        worst = []
        for g in range(5):
            gx = ix[ita[ix] == g]
            if gx.size:
                worst.append((r_r[gx].mean(), r_a[gx].mean() - r_r[gx].mean()))
        o["worst_group_recall"] = (min(worst, key=lambda t: t[0])[1] if worst else np.nan)
        return o

    full = np.arange(len(df)); point = stat(full); keys = list(point.keys())
    B = {k: np.empty(reps) for k in keys}
    for r in range(reps):
        ix = np.concatenate([idxby[s] for s in rng.choice(uniq, uniq.size, replace=True)])
        s = stat(ix)
        for k in keys:
            B[k][r] = s[k]
    res = {}
    lower_better = {"miss", "pct_rec10", "pct_dice20"}
    for k in keys:
        arr = B[k][~np.isnan(B[k])]
        lo, hi = np.percentile(arr, [2.5, 97.5])
        p = np.mean(arr < 0) if k in lower_better else np.mean(arr > 0)
        res[k] = dict(point=float(point[k]), lo=float(lo), hi=float(hi), p_better=float(p))
    return res


FAILURE_LOWER = ["miss", "pct_rec10", "pct_dice20"]           # improvement = CI hi < 0
FAILURE_HIGHER = ["small_recall", "worst_group_recall", "p5_subj_dice"]  # improvement = CI lo > 0


def verdict(res, ni_margin):
    ni = res["subj_median_dice"]
    noninferior = ni["lo"] > -ni_margin
    superior = ni["lo"] > 0
    gains = [k for k in FAILURE_LOWER if res[k]["hi"] < 0]
    gains += [k for k in FAILURE_HIGHER if res[k]["lo"] > 0]
    regress = [k for k in FAILURE_LOWER if res[k]["lo"] > 0]
    regress += [k for k in FAILURE_HIGHER if res[k]["hi"] < 0]
    if not noninferior:
        return f"INFERIOR (subject-Dice CI lo {ni['lo']:+.3f} <= -{ni_margin})"
    tag = "SUPERIOR-Dice" if superior else "non-inferior Dice"
    if gains:
        v = f"REPORTABLE WIN: {tag} + improves {', '.join(gains)}"
    else:
        v = f"NON-INFERIOR ({tag}, no failure-tail gain)"
    if regress:
        v += f"  [!] regresses {', '.join(regress)}"
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--ita", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--reps", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ni-margin", type=float, default=0.03,
                    help="subject-Dice non-inferiority margin (default 0.03)")
    ap.add_argument("--small-quantile", type=float, default=1 / 3)
    ap.add_argument("--out", default="aggregate_out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    ita = pd.read_csv(a.ita)[["stem", "subject", "ita_group_index_5"]]
    ref = load(a.ref)
    report = {"reference": a.ref, "reps": a.reps, "ni_margin": a.ni_margin,
              "hierarchy": ["subject-Dice non-inferiority", "complete-miss + low-recall tail",
                            "small-bruise recall", "worst-group recall (exploratory)",
                            "deployment cost (separate)"], "arms": {}}
    print("=" * 80)
    print("ARMS vs REFERENCE — hierarchy: (1) subject-Dice non-inferiority "
          f"[margin {a.ni_margin}] -> (2) failure tail -> (3) small recall -> (4) worst group")
    print("=" * 80)
    for spec in a.arms:
        name, path = spec.split("=", 1)
        if not os.path.exists(path):
            print(f"\n[{name}] MISSING {path}"); continue
        df = merged(ref, load(path), ita)
        thr = df["gt_positive_pixels"].quantile(a.small_quantile)
        res = boot(df, df["gt_positive_pixels"] <= thr, a.reps, rng)
        v = verdict(res, a.ni_margin)
        report["arms"][name] = {"verdict": v, "n": int(len(df)), "results": res}
        print(f"\n[{name}]  ->  {v}")
        for lab, k in [("subj median Dice", "subj_median_dice"), ("complete-miss", "miss"),
                       ("%recall<0.10", "pct_rec10"), ("%Dice<0.20", "pct_dice20"),
                       ("small recall", "small_recall"), ("worst-grp recall", "worst_group_recall"),
                       ("p5 subj Dice", "p5_subj_dice")]:
            d = res[k]
            print(f"    {lab:<18s} {d['point']:+.4f}  [{d['lo']:+.4f},{d['hi']:+.4f}]  P(better)={d['p_better']:.3f}")
    with open(os.path.join(a.out, "aggregate_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("\n" + "=" * 80 + "\nGATE SUMMARY\n" + "=" * 80)
    for name, v in report["arms"].items():
        print(f"  {name:<40s} {v['verdict']}")
    print(f"\n[done] {os.path.join(a.out, 'aggregate_report.json')}")


if __name__ == "__main__":
    main()
