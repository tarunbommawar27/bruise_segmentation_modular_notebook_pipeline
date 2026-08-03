"""Fit each model's operating point on val. Never on test."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch


@torch.no_grad()
def cache_logits(model, loader, device, amp: bool):
    """One forward pass per image; keep the logits.

    The sweep tries 481 cuts. Re-running the model for each would be 481 x 134 =
    64,454 forward passes; caching the logits once makes all 481 cuts pure tensor
    arithmetic on data already on the GPU. fp16 keeps 134x640x640 at ~110 MB --
    the quantisation is ~3 decimal places on a logit, far below the resolution any
    of this can distinguish.
    """
    model.eval()
    logits, gts, stems = [], [], []
    for x, y, s in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp):
            z = model(x)
        logits.append(z.float().half()[:, 0])
        # GT as BOOL, never float: the sweep counts pixels, and counting is what
        # bool+int64 do exactly. See sweep_cuts for why that matters.
        gts.append((y[:, 0] > 0.5).to(device))
        stems.extend(s)
    return torch.cat(logits), torch.cat(gts), stems


def sweep_cuts(logits, gts, cuts):
    """Per-cut mean Dice, its standard error, and complete-miss rate. Vectorised.

    THE REDUCTIONS ARE EXACT, ON PURPOSE. Dice is a ratio of PIXEL COUNTS, so the
    intersection and the two cardinalities are computed as boolean masks summed to
    int64 -- exactly what metrics.dice_np does in numpy, and exact by construction.

    Doing the same reductions in fp16 (tempting, since the cached logits are fp16)
    is wrong: fp16 carries an 11-bit mantissa, and a 640x640 image sums ~410k terms,
    so the running total stops being able to represent +1 long before the sum
    finishes. Measured against the numpy implementation that drifted by ~1.5e-4 per
    cut -- small, but the tie band is selected by comparing cuts whose Dice differ
    by ~1e-3, so the error is a tenth of the signal it is used to rank. The logits
    stay fp16 (storage only; the >= comparison is exact regardless).
    """
    rows = []
    gts = gts.bool()
    gt_sum = gts.sum(dim=(1, 2))              # int64, exact
    gt_has = gt_sum > 0
    n = len(gt_sum)
    for c in cuts:
        pred = logits >= c                    # bool; comparison is exact in fp16
        inter = (pred & gts).sum(dim=(1, 2))
        pred_sum = pred.sum(dim=(1, 2))
        denom = pred_sum + gt_sum
        # denom == 0 means both prediction and GT are empty: a correct agreement,
        # scored 1.0 -- matching metrics.dice_np so the sweep and the report agree.
        dice = torch.where(denom > 0,
                           2.0 * inter.double() / denom.double().clamp_min(1.0),
                           torch.ones_like(denom, dtype=torch.float64))
        miss = ((pred_sum == 0) & gt_has).double()
        d = dice
        rows.append({
            "cut": float(c), "threshold": float(torch.sigmoid(torch.tensor(c))),
            "mean_dice": float(d.mean()),
            "se_dice": float(d.std(unbiased=True) / np.sqrt(n)),
            "complete_miss_rate": float(miss.mean()),
        })
    return pd.DataFrame(rows)


def select_cut(df: pd.DataFrame) -> dict:
    """Pick the operating point: the tie band's LOWEST-MISS cut.

    WHY A TIE BAND AT ALL
    ----------------------
    These sweeps are extraordinarily flat -- on the previous models, B2's val Dice
    moved by 0.009 across thresholds from 0.154 to 0.959. That is not a peak, it is
    noise on a plateau, and taking argmax of it fits the val set's sampling error.
    Every cut within ONE STANDARD ERROR of the peak is statistically tied, so the
    band -- not the argmax -- is the honest answer to "which cut is best?".

    WHY MISS RATE BREAKS THE TIE
    -----------------------------
    Cuts in the band are Dice-equivalent but they are NOT miss-equivalent: a lower
    cut predicts more pixels, so fewer images come back completely blank, at
    statistically identical Dice. The old rule took the band's MEDIAN cut, which
    optimises for stability -- but a blank mask is a missed injury, and the miss
    rate is the one metric that separates these models by more than label noise.
    So: minimise miss rate within the band; break remaining ties on Dice; break
    what is left on the median cut, for reproducibility.
    """
    peak = df.loc[df["mean_dice"].idxmax()]
    band = df[df["mean_dice"] >= peak["mean_dice"] - peak["se_dice"]]
    best_miss = band["complete_miss_rate"].min()
    tied = band[band["complete_miss_rate"] <= best_miss + 1e-12]
    top_dice = tied["mean_dice"].max()
    finalists = tied[tied["mean_dice"] >= top_dice - 1e-12]
    chosen = finalists.iloc[len(finalists) // 2]
    return {
        "cut": float(chosen["cut"]), "threshold": float(chosen["threshold"]),
        "val_dice_at_cut": float(chosen["mean_dice"]),
        "val_miss_at_cut": float(chosen["complete_miss_rate"]),
        "val_peak_dice": float(peak["mean_dice"]),
        "peak_cut": float(peak["cut"]),
        "band_lo_threshold": float(band["threshold"].min()),
        "band_hi_threshold": float(band["threshold"].max()),
        "band_width_cuts": int(len(band)),
        "n_cuts": int(len(df)),
    }