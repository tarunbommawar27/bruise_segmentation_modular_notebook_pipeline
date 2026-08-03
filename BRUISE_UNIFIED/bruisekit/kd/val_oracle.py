#!/usr/bin/env python3
"""
val_oracle.py — B2/B5 complementarity on VALIDATION (review point 4).
=====================================================================
Decides whether the adaptive multi-teacher method is worth running, WITHOUT
touching the test set. Uses B2 and B5 per-image Dice on VALIDATION (each at its
own val-fitted operating point, produced by the LOCKED eval) and reports, with a
subject-cluster bootstrap:

  - correlation of per-image Dice (B2 vs B5)
  - #images where each teacher beats the other by >0.10
  - the ORACLE (per-image best teacher) mean/median Dice
  - oracle GAIN over the best single teacher, with 95% CI and P(gain>0)
  - also derives rel_b: a validation reliability weight for teacher B (B5) in
    [0,1] = share of val images where B5 >= B2, for the adaptive gate.

GATE RULE (printed): run the adaptive multi-teacher arm (Exp C / Phase 3) only if
the oracle gain CI is clear of zero on validation. The TEST oracle is computed
ONLY later, as a frozen-method diagnostic — never here.

    python val_oracle.py --b2-val B2_val_per_image.csv --b5-val B5_val_per_image.csv \
        --reps 5000 --out distill_out/val_oracle
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd


def spearman(a, b):
    return float(np.corrcoef(pd.Series(a).rank(), pd.Series(b).rank())[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b2-val", required=True)
    ap.add_argument("--b5-val", required=True)
    ap.add_argument("--reps", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    b2 = pd.read_csv(a.b2_val)[["stem", "dice"]].rename(columns={"dice": "d_b2"})
    b5 = pd.read_csv(a.b5_val)
    subj_col = "subject" if "subject" in b5.columns else None
    keep = ["stem", "dice"] + ([subj_col] if subj_col else [])
    b5 = b5[keep].rename(columns={"dice": "d_b5"})
    df = b2.merge(b5, on="stem")
    if subj_col is None:
        df["subject"] = df["stem"].astype(str).str.split("_").str[0]
    d2, d5 = df["d_b2"].to_numpy(), df["d_b5"].to_numpy()
    orc = np.maximum(d2, d5)

    subj = df["subject"].to_numpy(); uniq = np.unique(subj)
    idxby = {s: np.where(subj == s)[0] for s in uniq}

    def gain(ix):
        return orc[ix].mean() - max(d2[ix].mean(), d5[ix].mean())

    point = gain(np.arange(len(df)))
    boot = np.empty(a.reps)
    for r in range(a.reps):
        ix = np.concatenate([idxby[s] for s in rng.choice(uniq, uniq.size, replace=True)])
        boot[r] = gain(ix)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p_pos = float(np.mean(boot > 0))
    rel_b = float((d5 >= d2).mean())          # validation reliability of B5 vs B2

    res = {
        "n_val_images": int(len(df)), "n_subjects": int(uniq.size),
        "spearman_dice_b2_b5": spearman(d2, d5),
        "b5_beats_b2_by_gt_0.10": int(np.sum(d5 - d2 > 0.10)),
        "b2_beats_b5_by_gt_0.10": int(np.sum(d2 - d5 > 0.10)),
        "b2_mean_dice": float(d2.mean()), "b5_mean_dice": float(d5.mean()),
        "oracle_mean_dice": float(orc.mean()), "oracle_median_dice": float(np.median(orc)),
        "oracle_gain_over_best_single": float(point),
        "oracle_gain_ci95": [float(lo), float(hi)], "p_gain_positive": p_pos,
        "rel_b_for_adaptive_gate": rel_b,
        "GATE_run_adaptive": bool(lo > 0),
    }
    with open(os.path.join(a.out, "val_oracle.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    print("\nGATE:", "RUN adaptive multi-teacher (val oracle gain clears 0)" if res["GATE_run_adaptive"]
          else "SKIP adaptive — uniform ensemble is the ceiling (val oracle gain within noise)")
    print(f"Pass --rel-b {rel_b:.3f} to distill_segformer's adaptive arm.")


if __name__ == "__main__":
    main()
