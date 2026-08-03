#!/usr/bin/env python3
"""
score_reference.py — score a teacher/reference checkpoint with the LOCKED eval.
==============================================================================
Review point 1: every number in the study must come from ONE evaluation
implementation. This scores a (converted, SegformerWrapper-layout) checkpoint —
B2 teacher or B0-distilled — with kd_core's exact eval: sweep the threshold on
val, then score val AND test at that threshold. Writes val_per_image.csv and
test_per_image.csv so the reference rows are directly comparable to every
distilled student (same code path) and usable by the VALIDATION oracle.

    python score_reference.py --name segformer_b2_teacher \
        --ckpt teachers/segformer_b2_teacher/best_model.pt \
        --pretrained pretrained_weights/segformer_mit_b2 \
        --train-manifest manifests/train_manifest.csv \
        --test-manifest  manifests/test_manifest.csv --data-root . \
        --test-ita ita_labels/wl_test_per_image_ita.csv --out-dir distill_out/reference
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import kd_core as K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pretrained", required=True)
    ap.add_argument("--train-manifest", required=True)
    ap.add_argument("--test-manifest", required=True)
    ap.add_argument("--data-root", default="")
    ap.add_argument("--test-ita", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-amp", action="store_true")
    a = ap.parse_args()
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    amp = not a.no_amp
    cfg = dict(K.DEFAULTS)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    full = K.load_manifest(a.train_manifest, a.data_root)
    test_df = K.load_manifest(a.test_manifest, a.data_root)
    _, val_df = K.resolve_train_val(full, cfg)

    model = K.SegformerWrapper(K.build_segformer(a.pretrained, num_labels=1)).to(device)
    model.load_state_dict(torch.load(a.ckpt, map_location=device, weights_only=True))

    val_loader = K.make_loader(val_df, a.img_size, 8, False, a.workers, 0)
    test_loader = K.make_loader(test_df, a.img_size, 8, False, a.workers, 0)
    thr_df, thr = K.threshold_sweep(model, val_loader, device, cfg["thresholds"], amp)

    vmap = dict(zip(val_df["stem"].astype(str), val_df["subject"].astype(str)))
    tmap = dict(zip(test_df["stem"].astype(str), test_df["subject"].astype(str)))
    val_pi, _ = K.evaluate(model, val_loader, device, thr, amp)
    val_pi["subject"] = val_pi["stem"].map(vmap)
    test_pi, summ = K.evaluate(model, test_loader, device, thr, amp)
    test_pi["subject"] = test_pi["stem"].map(tmap)
    val_pi.to_csv(out / f"{a.name}_val_per_image.csv", index=False)
    test_pi.to_csv(out / f"{a.name}_test_per_image.csv", index=False)
    tail = K.tail_metrics(test_pi)

    fair = {}
    if a.test_ita and Path(a.test_ita).exists():
        _, fair = K.fairness_by_group(test_pi, a.test_ita, a.name,
                                      out_csv=out / f"{a.name}_fairness_per_group.csv")
    (out / f"{a.name}_summary.json").write_text(json.dumps(
        {"name": a.name, "threshold": thr, **summ, **tail, **fair}, indent=2))
    print(f"[{a.name}] thr={thr:.2f} test dice={summ['mean_dice']:.4f} "
          f"median={summ['median_dice']:.4f} miss={summ['complete_miss_rate']*100:.2f}% "
          f"rec<.1={tail['pct_recall_below_0.10']*100:.1f}%")


if __name__ == "__main__":
    main()
