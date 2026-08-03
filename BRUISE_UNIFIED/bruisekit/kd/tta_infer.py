#!/usr/bin/env python3
"""
tta_infer.py — flip-TTA miss recovery, reported SEPARATELY (review point 8).
============================================================================
TTA is an INFERENCE strategy, not distillation. This scores any trained student
with 4-way flip TTA and reports Dice, complete-miss, recall, latency, and the
forward-pass count, so a gain from extra inference compute is never confused with
a training/distillation gain. Threshold is swept on val (same locked eval).

    python tta_infer.py --ckpt distill_out/expA_.../best_model.pt \
        --pretrained pretrained_weights/segformer_mit_b0 \
        --train-manifest manifests/train_manifest.csv \
        --test-manifest manifests/test_manifest.csv --data-root . \
        --out distill_out/tta/expA
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import kd_core as K


@torch.no_grad()
def score_tta(model, loader, device, threshold, amp, tta):
    model.eval(); rows = []; t0 = time.time(); n = 0
    for x, y, stems in loader:
        x = x.to(device, non_blocking=True); n += x.shape[0]
        prob = (K.tta_flip_prob(model, x, amp) if tta
                else torch.sigmoid(model(x).float()))
        prob = prob.cpu().numpy(); gt = y.numpy()
        for i, s in enumerate(stems):
            pred = (prob[i, 0] >= threshold).astype("uint8")
            rows.append(K.compute_image_row(pred, (gt[i, 0] > 0.5).astype("uint8"), str(s)))
    dt = time.time() - t0
    return pd.DataFrame(rows), K.summarize(rows), dt / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pretrained", default="pretrained_weights/segformer_mit_b0")
    ap.add_argument("--train-manifest", required=True)
    ap.add_argument("--test-manifest", required=True)
    ap.add_argument("--data-root", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-amp", action="store_true")
    a = ap.parse_args()
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    amp = not a.no_amp
    cfg = dict(K.DEFAULTS)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    full = K.load_manifest(a.train_manifest, a.data_root)
    test_df = K.load_manifest(a.test_manifest, a.data_root)
    _, val_df = K.resolve_train_val(full, cfg)
    model = K.SegformerWrapper(K.build_segformer(a.pretrained, num_labels=1)).to(device)
    model.load_state_dict(torch.load(a.ckpt, map_location=device, weights_only=True))
    val_loader = K.make_loader(val_df, cfg["img_size"], 8, False, cfg["workers"], 0)
    test_loader = K.make_loader(test_df, cfg["img_size"], 8, False, cfg["workers"], 0)
    _, thr = K.threshold_sweep(model, val_loader, device, cfg["thresholds"], amp)

    rows = []
    for tag, tta, passes in [("baseline", False, 1), ("tta_flip4", True, 4)]:
        pi, s, lat = score_tta(model, test_loader, device, thr, amp, tta)
        pi.to_csv(out / f"{tag}_test_per_image.csv", index=False)
        rows.append({"mode": tag, "forward_passes": passes, "sec_per_image": round(lat, 4),
                     "mean_dice": s["mean_dice"], "median_dice": s["median_dice"],
                     "mean_recall": s["mean_recall"], "complete_miss_rate": s["complete_miss_rate"]})
    tbl = pd.DataFrame(rows)
    tbl.to_csv(out / "tta_comparison.csv", index=False)
    (out / "tta_summary.json").write_text(json.dumps({"threshold": thr, "rows": rows}, indent=2))
    print(tbl.to_string(index=False))
    print("\nReported separately from distillation results (inference-only).")


if __name__ == "__main__":
    main()
