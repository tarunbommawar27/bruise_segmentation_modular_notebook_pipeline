#!/usr/bin/env python3
"""
Paired subject-cluster bootstrap comparison for the B5 baseline vs the
final-ipynb saved models (B2 teacher, B0 distilled).

WHAT IT DOES
------------
1. Paired subject-level cluster bootstrap (28 subjects = clusters, B>=4000 reps)
   for  B5 - B2  and  B5 - B0distilled, reporting for each:
       mean Dice diff, median Dice diff, recall diff, complete-miss diff,
       small-bruise Dice diff, and each ITA-group median Dice diff,
   with 95% CIs and the bootstrap probability that B5 is better.
2. Per-image error / complementarity analysis (B5 vs B2):
       Pearson + Spearman correlation of per-image Dice,
       #cases B5 beats B2 by >0.10, #cases B2 beats B5 by >0.10,
       success/fail contingency (one model finds the bruise, the other misses),
       ORACLE (per-image best-teacher) mean/median Dice and its gain over the
       best single teacher, WITH a subject-cluster bootstrap CI on the gain.
3. (Optional) Teacher probability-map disagreement, if --prob_b2 / --prob_b5
   directories of per-stem probability maps (.npy) are supplied.

All differences are (B5 - other): positive Dice/recall diff => B5 better;
negative complete-miss diff => B5 better (fewer misses).

INPUT FILES  (all share columns: stem,dice,iou,precision,recall,
              pred_positive_pixels,gt_positive_pixels)
    --b5   results_segformer_b5/runs/segformer_b5__seed0/test_per_image.csv
    --b2   <B2 teacher test_per_image.csv>
    --b0   <B0 distilled test_per_image.csv>
    --ita  ita_labels/wl_test_per_image_ita.csv   (stem,subject,ita_group_index_5,skin_tone_category)

USAGE
    python paired_bootstrap_b5_compare.py \
        --b5 test_per_image_b5.csv \
        --b2 test_per_image_b2.csv \
        --b0 test_per_image_b0distilled.csv \
        --ita wl_test_per_image_ita.csv \
        --reps 5000 --seed 0 --out b5_comparison_out

Only numpy + pandas required (scipy used if present, else pure-numpy fallback).
"""
import argparse
import json
import os
import numpy as np
import pandas as pd

ITA_NAMES = {0: "Light (II-III)", 1: "Intermediate (III-IV)", 2: "Tan (IV)",
             3: "Brown (V)", 4: "Dark (VI)"}


# ----------------------------------------------------------------------------- helpers
def load_pi(path, tag):
    df = pd.read_csv(path)
    need = {"stem", "dice", "recall", "pred_positive_pixels", "gt_positive_pixels"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns {missing}")
    df = df[["stem", "dice", "iou", "recall",
             "pred_positive_pixels", "gt_positive_pixels"]].copy()
    df["miss"] = ((df["pred_positive_pixels"] == 0) &
                  (df["gt_positive_pixels"] > 0)).astype(float)
    df = df.rename(columns={c: f"{c}_{tag}" for c in
                            ["dice", "iou", "recall", "pred_positive_pixels",
                             "gt_positive_pixels", "miss"]})
    return df


def spearman(a, b):
    ar = pd.Series(a).rank().to_numpy()
    br = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ar, br)[0, 1])


def pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1])


# ----------------------------------------------------------------------------- bootstrap
def cluster_bootstrap(df, a, b, small_mask, reps, rng):
    """
    Paired subject-cluster bootstrap of (metric_a - metric_b).
    df has columns: subject, ita5, gt_positive_pixels, and per-model
    dice_<a/b>, recall_<a/b>, miss_<a/b>.
    Returns dict metric -> (point, lo, hi, p_b5_better).
    """
    subjects = df["subject"].to_numpy()
    uniq = np.unique(subjects)
    # precompute row indices per subject
    idx_by_subj = {s: np.where(subjects == s)[0] for s in uniq}

    da, db = df[f"dice_{a}"].to_numpy(), df[f"dice_{b}"].to_numpy()
    ra, rb = df[f"recall_{a}"].to_numpy(), df[f"recall_{b}"].to_numpy()
    ma, mb = df[f"miss_{a}"].to_numpy(), df[f"miss_{b}"].to_numpy()
    ita = df["ita5"].to_numpy()
    sm = small_mask.to_numpy() if hasattr(small_mask, "to_numpy") else small_mask

    def stats_on(ix):
        out = {}
        out["mean_dice"] = da[ix].mean() - db[ix].mean()
        out["median_dice"] = np.median(da[ix]) - np.median(db[ix])
        out["recall"] = ra[ix].mean() - rb[ix].mean()
        out["complete_miss"] = ma[ix].mean() - mb[ix].mean()
        smx = ix[sm[ix]]
        out["small_dice"] = (np.median(da[smx]) - np.median(db[smx])
                             if smx.size else np.nan)
        for g in range(5):
            gx = ix[ita[ix] == g]
            out[f"ita{g}_dice"] = (np.median(da[gx]) - np.median(db[gx])
                                   if gx.size else np.nan)
        return out

    # point estimate on the real sample
    full = np.arange(len(df))
    point = stats_on(full)

    keys = list(point.keys())
    boot = {k: np.empty(reps) for k in keys}
    for r in range(reps):
        drawn = rng.choice(uniq, size=uniq.size, replace=True)
        ix = np.concatenate([idx_by_subj[s] for s in drawn])
        s = stats_on(ix)
        for k in keys:
            boot[k][r] = s[k]

    res = {}
    for k in keys:
        arr = boot[k]
        valid = arr[~np.isnan(arr)]
        lo, hi = np.percentile(valid, [2.5, 97.5])
        if k == "complete_miss":
            p_better = float(np.mean(valid < 0))   # fewer misses = better
        else:
            p_better = float(np.mean(valid > 0))   # higher dice/recall = better
        res[k] = dict(point=float(point[k]), lo=float(lo), hi=float(hi),
                      p_b5_better=p_better, n_valid=int(valid.size))
    return res


def oracle_gain_bootstrap(df, a, b, reps, rng):
    """Bootstrap CI on (oracle_mean_dice - best_single_mean_dice)."""
    subjects = df["subject"].to_numpy()
    uniq = np.unique(subjects)
    idx_by_subj = {s: np.where(subjects == s)[0] for s in uniq}
    da, db = df[f"dice_{a}"].to_numpy(), df[f"dice_{b}"].to_numpy()
    orc = np.maximum(da, db)

    def gain(ix):
        best_single = max(da[ix].mean(), db[ix].mean())
        return orc[ix].mean() - best_single

    full = np.arange(len(df))
    point = gain(full)
    boot = np.empty(reps)
    for r in range(reps):
        drawn = rng.choice(uniq, size=uniq.size, replace=True)
        ix = np.concatenate([idx_by_subj[s] for s in drawn])
        boot[r] = gain(ix)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(point=float(point), lo=float(lo), hi=float(hi),
                p_gain_positive=float(np.mean(boot > 0)))


# ----------------------------------------------------------------------------- prob maps
def prob_disagreement(stems, dir_a, dir_b):
    """Mean per-pixel |p_a - p_b| and thresholded-mask IoU-disagreement."""
    abs_diffs, mask_ious = [], []
    used = 0
    for st in stems:
        pa = os.path.join(dir_a, f"{st}.npy")
        pb = os.path.join(dir_b, f"{st}.npy")
        if not (os.path.exists(pa) and os.path.exists(pb)):
            continue
        A, Bp = np.load(pa).astype(np.float32), np.load(pb).astype(np.float32)
        if A.shape != Bp.shape:
            continue
        abs_diffs.append(float(np.mean(np.abs(A - Bp))))
        ma, mb = A > 0.5, Bp > 0.5
        union = np.logical_or(ma, mb).sum()
        iou = 1.0 if union == 0 else np.logical_and(ma, mb).sum() / union
        mask_ious.append(float(iou))
        used += 1
    if used == 0:
        return None
    return dict(n_used=used,
                mean_abs_prob_diff=float(np.mean(abs_diffs)),
                mean_pred_mask_iou=float(np.mean(mask_ious)))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b5", required=True)
    ap.add_argument("--b2", required=True)
    ap.add_argument("--b0", required=True, help="B0 distilled test_per_image.csv")
    ap.add_argument("--ita", required=True)
    ap.add_argument("--reps", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--small_quantile", type=float, default=1 / 3,
                    help="bottom fraction of gt pixels = small-bruise stratum")
    ap.add_argument("--prob_b2", default=None, help="dir of B2 prob maps (.npy)")
    ap.add_argument("--prob_b5", default=None, help="dir of B5 prob maps (.npy)")
    ap.add_argument("--out", default="b5_comparison_out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    b5 = load_pi(args.b5, "b5")
    b2 = load_pi(args.b2, "b2")
    b0 = load_pi(args.b0, "b0")
    ita = pd.read_csv(args.ita)[["stem", "subject", "ita_group_index_5",
                                 "skin_tone_category"]]
    ita = ita.rename(columns={"ita_group_index_5": "ita5"})

    df = (b5.merge(b2, on="stem", how="inner", suffixes=("", "_dup"))
             .merge(b0, on="stem", how="inner")
             .merge(ita, on="stem", how="inner"))
    # gt pixels are identical across models; keep one
    df["gt_positive_pixels"] = df["gt_positive_pixels_b5"]
    n = len(df)
    n_subj = df["subject"].nunique()
    print(f"[info] merged images: {n}   subjects: {n_subj}   reps: {args.reps}")
    if n_subj != 28:
        print(f"[warn] expected 28 subjects, got {n_subj}")

    thr = df["gt_positive_pixels"].quantile(args.small_quantile)
    small_mask = df["gt_positive_pixels"] <= thr
    print(f"[info] small-bruise stratum: gt_pixels <= {thr:.0f}  "
          f"({int(small_mask.sum())} images)")

    def fmt(d):
        return (f"{d['point']:+.4f}  [{d['lo']:+.4f}, {d['hi']:+.4f}]  "
                f"P(B5 better)={d['p_b5_better']:.3f}")

    report = {"n_images": n, "n_subjects": int(n_subj), "reps": args.reps,
              "small_pixel_threshold": float(thr)}

    for other, label in [("b2", "B5 - B2 teacher"),
                         ("b0", "B5 - B0 distilled")]:
        print("\n" + "=" * 72)
        print(f"PAIRED SUBJECT-CLUSTER BOOTSTRAP:  {label}")
        print("=" * 72)
        res = cluster_bootstrap(df, "b5", other, small_mask, args.reps, rng)
        rows = [("Mean Dice diff", "mean_dice"),
                ("Median Dice diff", "median_dice"),
                ("Recall diff", "recall"),
                ("Complete-miss diff", "complete_miss"),
                ("Small-bruise Dice diff", "small_dice"),
                ("ITA Light (II-III)", "ita0_dice"),
                ("ITA Intermediate (III-IV)", "ita1_dice"),
                ("ITA Tan (IV)", "ita2_dice"),
                ("ITA Brown (V)", "ita3_dice"),
                ("ITA Dark (VI)", "ita4_dice")]
        for name, key in rows:
            print(f"  {name:<28s} {fmt(res[key])}")
        report[label] = res

    # -------- per-image complementarity (B5 vs B2) --------
    print("\n" + "=" * 72)
    print("PER-IMAGE ERROR / COMPLEMENTARITY  (B5 vs B2)")
    print("=" * 72)
    d5, d2 = df["dice_b5"].to_numpy(), df["dice_b2"].to_numpy()
    m5, m2 = df["miss_b5"].to_numpy(), df["miss_b2"].to_numpy()
    delta = d5 - d2
    succ = 0.5  # "found the bruise" threshold on Dice
    b5_only = int(np.sum((d5 >= succ) & (d2 < succ)))
    b2_only = int(np.sum((d2 >= succ) & (d5 < succ)))
    comp = {
        "pearson_dice": pearson(d5, d2),
        "spearman_dice": spearman(d5, d2),
        "b5_beats_b2_by_gt_0.10": int(np.sum(delta > 0.10)),
        "b2_beats_b5_by_gt_0.10": int(np.sum(delta < -0.10)),
        "b5_finds_b2_misses(dice>=.5 vs <.5)": b5_only,
        "b2_finds_b5_misses(dice>=.5 vs <.5)": b2_only,
        "b5_miss_only": int(np.sum((m5 == 1) & (m2 == 0))),
        "b2_miss_only": int(np.sum((m2 == 1) & (m5 == 0))),
        "both_miss": int(np.sum((m5 == 1) & (m2 == 1))),
    }
    for k, v in comp.items():
        print(f"  {k:<40s} {v}")
    report["per_image_b5_vs_b2"] = comp

    # -------- oracle (best teacher per image) --------
    print("\n" + "-" * 72)
    print("ORACLE (per-image best of B2/B5) -- potential of multi-teacher")
    print("-" * 72)
    orc = np.maximum(d5, d2)
    orc_miss = float(np.mean((m5 == 1) & (m2 == 1)))  # oracle misses only if both do
    oracle = {
        "b5_mean_dice": float(d5.mean()), "b5_median_dice": float(np.median(d5)),
        "b2_mean_dice": float(d2.mean()), "b2_median_dice": float(np.median(d2)),
        "oracle_mean_dice": float(orc.mean()), "oracle_median_dice": float(np.median(orc)),
        "oracle_gain_over_best_single_mean": float(orc.mean() - max(d5.mean(), d2.mean())),
        "b2_miss_rate": float(m2.mean()), "b5_miss_rate": float(m5.mean()),
        "oracle_miss_rate": orc_miss,
    }
    og = oracle_gain_bootstrap(df, "b5", "b2", args.reps, rng)
    oracle["oracle_gain_bootstrap"] = og
    for k, v in oracle.items():
        if k != "oracle_gain_bootstrap":
            print(f"  {k:<38s} {v:.4f}")
    print(f"  oracle gain (mean Dice) vs best single teacher: "
          f"{og['point']:+.4f}  [{og['lo']:+.4f}, {og['hi']:+.4f}]  "
          f"P(gain>0)={og['p_gain_positive']:.3f}")
    report["oracle_b2_b5"] = oracle

    # -------- optional prob-map disagreement --------
    if args.prob_b2 and args.prob_b5:
        print("\n" + "-" * 72)
        print("TEACHER PROBABILITY-MAP DISAGREEMENT (B2 vs B5)")
        print("-" * 72)
        pd_res = prob_disagreement(df["stem"].tolist(), args.prob_b2, args.prob_b5)
        if pd_res is None:
            print("  [warn] no matching .npy prob maps found; skipped.")
        else:
            for k, v in pd_res.items():
                print(f"  {k:<24s} {v}")
            report["prob_disagreement_b2_b5"] = pd_res
    else:
        print("\n[info] prob-map disagreement skipped "
              "(pass --prob_b2 DIR --prob_b5 DIR of per-stem .npy maps to enable).")

    out_json = os.path.join(args.out, "b5_comparison_report.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    df.to_csv(os.path.join(args.out, "merged_per_image.csv"), index=False)
    print(f"\n[done] wrote {out_json}")
    print(f"[done] wrote {os.path.join(args.out, 'merged_per_image.csv')}")


if __name__ == "__main__":
    main()
