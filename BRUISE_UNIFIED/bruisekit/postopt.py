"""Reduce complete misses WITHOUT retraining. Everything here is fitted on val and
applied to test; nothing changes a trained weight.

THE PROBLEM THIS SOLVES
------------------------
A single global threshold couples two failure modes: pushing it down to stop blank
masks (complete misses) also floods the easy images with false positives, so miss-%
and Dice move together -- two sides of one knob. Lowering the threshold slides ALONG
the miss-vs-Dice curve; it cannot move the curve. The three techniques here try to
move the curve (fewer misses at the SAME Dice), and one deliberately games the miss
metric so its cost is visible for comparison:

  ensemble   -- average the 3 seeds' probability maps. A miss needs ALL three seeds
                to blank the same image, which is rare (the per-seed misses are
                different images). Free: no retraining, just averaging maps we can
                already produce. This is the honest, recommended lever.
  TTA        -- average probs over horizontal+vertical flips. Raises the probability
                on borderline images so they clear the threshold without lowering it.
  no-blank   -- if a mask is still empty, recover the most-confident region instead of
                returning blank. This GAMES the miss metric (guarantees a non-zero
                prediction whether or not anything real was found), so it is reported
                as a separate floor, never as the main method.

HOW TO READ THE RESULT (this is the whole point of the question)
----------------------------------------------------------------
Plot miss-% against Dice. A real improvement sits BELOW-AND-LEFT of the single-model
threshold-sweep curve -- lower miss at equal-or-better Dice. A point that just slid
down the same curve (Dice fell as much as miss) is the threshold in disguise, not an
improvement. Everything below fits its threshold on VAL with the same miss-tie-break
rule as the baseline, so the comparison is like-for-like.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from bruisekit.metrics import compute_image_row, summarize
from bruisekit.sweep import select_cut


@torch.no_grad()
def probs_plain(model, loader, device, amp: bool):
    """One forward pass per image -> sigmoid probability maps [N,H,W] (fp16), GT
    (bool), stems. fp16 storage keeps the whole split in memory; the >= comparison
    the sweep does is exact regardless of storage dtype."""
    model.eval()
    P, G, S = [], [], []
    use_amp = amp and device.type == "cuda"
    for x, y, s in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            z = model(x)
        P.append(torch.sigmoid(z.float())[:, 0].half().cpu())
        G.append((y[:, 0] > 0.5).cpu())
        S.extend(s)
    return torch.cat(P), torch.cat(G), S


@torch.no_grad()
def probs_tta(model, loader, device, amp: bool, flips=("none", "h", "v")):
    """TTA: average sigmoid probs over identity + horizontal + vertical flip.

    Each flip is applied to the INPUT and UNDONE on the OUTPUT before averaging, so
    the maps stay pixel-aligned -- averaging misaligned maps would blur the boundary
    rather than sharpen the confidence. TTA is used identically on val (to fit the
    threshold) and test (to score); mismatching them would fit a threshold for a
    distribution the test pass never sees.
    """
    model.eval()
    P, G, S = [], [], []
    use_amp = amp and device.type == "cuda"
    for x, y, s in loader:
        x = x.to(device, non_blocking=True)
        acc = torch.zeros(x.shape[0], x.shape[2], x.shape[3], device=x.device)
        for f in flips:
            xf = torch.flip(x, [3]) if f == "h" else torch.flip(x, [2]) if f == "v" else x
            with torch.amp.autocast("cuda", enabled=use_amp):
                z = model(xf)
            p = torch.sigmoid(z.float())[:, 0]
            p = torch.flip(p, [2]) if f == "h" else torch.flip(p, [1]) if f == "v" else p
            acc = acc + p
        P.append((acc / len(flips)).half().cpu())
        G.append((y[:, 0] > 0.5).cpu())
        S.extend(s)
    return torch.cat(P), torch.cat(G), S


def mean_over_seeds(prob_list, stem_list):
    """Average probability maps across runs, aligned by stem.

    Loaders are shuffle=False so every seed already returns images in the same order,
    but this re-indexes by stem anyway rather than trusting that -- a silent order
    mismatch would average seed A's image 5 onto seed B's image 6 and quietly corrupt
    every ensemble number. Asserts the image SETS are identical first.
    """
    ref = stem_list[0]
    ref_set = set(ref)
    for sl in stem_list:
        if set(sl) != ref_set:
            raise ValueError("ensemble seeds cover different image sets")
    acc = None
    for probs, sl in zip(prob_list, stem_list):
        pos = {s: i for i, s in enumerate(sl)}
        reordered = torch.stack([probs[pos[s]] for s in ref]).float()
        acc = reordered if acc is None else acc + reordered
    return (acc / len(prob_list)).half(), list(ref)


def sweep_prob(probs, gts, thresholds):
    """Per-threshold mean Dice / SE / complete-miss on probability maps.

    Same exact-integer pixel counting as sweep.sweep_cuts (bool masks summed to
    int64), so the numbers are directly comparable to the logit-cut sweep. Emits the
    columns select_cut expects, with `cut` == `threshold` (probability space).

    The comparison is done in float32, NOT the fp16 the maps are stored in, so the
    threshold this sweep FITS on val is applied at the exact same numeric boundary
    score_prob_at() uses on test (it also compares in float32). Comparing fp16 here
    would put the val fit and the test apply on slightly different boundaries for any
    pixel sitting right at the threshold -- the same "the sweep must match the score"
    trap the logit sweep already guards against.
    """
    probs = probs.float()
    gts = gts.bool()
    gt_sum = gts.sum(dim=(1, 2))
    gt_has = gt_sum > 0
    n = len(gt_sum)
    rows = []
    for t in thresholds:
        pred = probs >= t
        inter = (pred & gts).sum(dim=(1, 2))
        ps = pred.sum(dim=(1, 2))
        den = ps + gt_sum
        dice = torch.where(den > 0, 2.0 * inter.double() / den.double().clamp_min(1.0),
                           torch.ones_like(den, dtype=torch.float64))
        miss = ((ps == 0) & gt_has).double()
        rows.append({"cut": float(t), "threshold": float(t),
                     "mean_dice": float(dice.mean()),
                     "se_dice": float(dice.std(unbiased=True) / np.sqrt(n)),
                     "complete_miss_rate": float(miss.mean())})
    return pd.DataFrame(rows)


def score_prob_at(probs, gts, stems, thr, no_blank=False, rel=0.5):
    """Score probability maps at a fixed threshold; optional no-blank floor.

    no_blank: when the thresholded mask is empty on an image that has a bruise, fall
    back to the region at >= rel * max-probability -- the most-confident blob the
    model saw. This never returns blank, which is exactly why it must be reported
    separately: it converts a genuine miss into a (possibly wrong) small prediction.
    """
    p_np = probs.float().numpy()
    g_np = gts.numpy()
    rows = []
    for i, s in enumerate(stems):
        p = p_np[i]
        pred = (p >= thr).astype("uint8")
        if no_blank and pred.sum() == 0 and p.max() > 0:
            pred = (p >= rel * float(p.max())).astype("uint8")
        rows.append(compute_image_row(pred, g_np[i].astype("uint8"), s))
    return pd.DataFrame(rows), summarize(rows)


def fit_on_val_apply_to_test(val_probs, val_gts, test_probs, test_gts, test_stems,
                             thresholds, no_blank=False):
    """The honest protocol: sweep val -> select_cut (miss-tie-break) -> score test.

    Returns (operating_point_dict, test_per_image_df, test_summary). The threshold is
    chosen ONLY from val, then applied once to test, exactly like the baseline in the
    main notebook -- so any miss/Dice difference is the technique, not a re-tuned cut.
    """
    grid = sweep_prob(val_probs, val_gts, thresholds)
    op = select_cut(grid)
    per_img, summ = score_prob_at(test_probs, test_gts, test_stems, op["threshold"], no_blank=no_blank)
    return op, per_img, summ