"""Metrics: Dice/IoU per image, and threshold-free AP for model selection."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def dice_np(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    denom = pred.sum() + gt.sum()
    return 1.0 if denom == 0 else float(2 * np.logical_and(pred, gt).sum() / denom)


def iou_np(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    union = np.logical_or(pred, gt).sum()
    return 1.0 if union == 0 else float(np.logical_and(pred, gt).sum() / union)


def compute_image_row(pred: np.ndarray, gt: np.ndarray, stem: str) -> dict:
    pred_b, gt_b = pred.astype(bool), gt.astype(bool)
    tp = int(np.logical_and(pred_b, gt_b).sum())
    fp = int(np.logical_and(pred_b, ~gt_b).sum())
    fn = int(np.logical_and(~pred_b, gt_b).sum())
    return {
        "stem": stem,
        "dice": dice_np(pred, gt), "iou": iou_np(pred, gt),
        "precision": 1.0 if tp + fp == 0 else tp / (tp + fp),
        "recall": 1.0 if tp + fn == 0 else tp / (tp + fn),
        "pred_positive_pixels": int(pred_b.sum()),
        "gt_positive_pixels": int(gt_b.sum()),
    }


def summarize(rows: list[dict]) -> dict:
    df = pd.DataFrame(rows)
    # "Complete miss" = the model output ZERO pixels on an image that has a bruise.
    # This is the metric that separates the models by more than label noise, and for
    # an injury-documentation tool it is the one that actually matters: a blank mask
    # is a missed injury. Reported as a first-class number, not buried in the tail.
    miss = (df["pred_positive_pixels"] == 0) & (df["gt_positive_pixels"] > 0)
    return {
        "n_images": int(len(df)),
        "mean_dice": float(df["dice"].mean()),
        "median_dice": float(df["dice"].median()),
        "mean_iou": float(df["iou"].mean()),
        "median_iou": float(df["iou"].median()),
        "mean_precision": float(df["precision"].mean()),
        "mean_recall": float(df["recall"].mean()),
        "complete_miss_count": int(miss.sum()),
        "complete_miss_rate": float(miss.mean()),
    }


class BinnedAP:
    """Threshold-free average precision over pixels, via probability histograms.

    WHY AP IS THE MODEL-SELECTION METRIC
    -------------------------------------
    The old pipeline saved best_model.pt by val Dice AT A FIXED 0.5 -- but the
    threshold is re-fitted afterwards anyway, and the fitted operating points are
    nowhere near 0.5 (YOLO's lands around 0.18). So 0.5-Dice selection asks "which
    epoch is best at an operating point we will not use?" and can pick the wrong
    epoch for any model whose calibration drifts during training. AP integrates
    over ALL thresholds, so the epoch choice cannot be biased by one arbitrary cut.

    WHY HISTOGRAMS AND NOT sklearn.average_precision_score
    -------------------------------------------------------
    134 val images x 640 x 640 = 55M pixels. Sorting 55M floats per epoch costs
    seconds of wall-clock and ~450 MB. Binning into 4096 buckets on the GPU makes
    the whole thing O(bins) in memory and effectively free, at a quantisation
    error of ~1/4096 in probability -- three orders of magnitude below the
    epoch-to-epoch differences it has to rank.
    """

    def __init__(self, bins: int = 4096, device: str = "cuda"):
        self.bins = bins
        self.pos = torch.zeros(bins, dtype=torch.float64, device=device)
        self.neg = torch.zeros(bins, dtype=torch.float64, device=device)

    @torch.no_grad()
    def update(self, prob: torch.Tensor, gt: torch.Tensor) -> None:
        p = prob.reshape(-1).float().clamp(0, 1)
        g = gt.reshape(-1) > 0.5
        idx = (p * (self.bins - 1)).round().long()
        self.pos += torch.bincount(idx[g], minlength=self.bins).double()
        self.neg += torch.bincount(idx[~g], minlength=self.bins).double()

    def compute(self) -> float:
        """AP = sum over thresholds of (recall_k - recall_{k-1}) * precision_k."""
        total_pos = self.pos.sum()
        if total_pos == 0:
            return float("nan")
        # Walk bins from the highest probability downward: each step admits one more
        # bucket as "predicted positive", which is exactly sweeping the threshold down.
        tp = torch.cumsum(self.pos.flip(0), dim=0)
        fp = torch.cumsum(self.neg.flip(0), dim=0)
        precision = tp / (tp + fp).clamp_min(1e-12)
        recall = tp / total_pos
        d_recall = torch.diff(recall, prepend=torch.zeros(1, dtype=recall.dtype, device=recall.device))
        return float((d_recall * precision).sum())