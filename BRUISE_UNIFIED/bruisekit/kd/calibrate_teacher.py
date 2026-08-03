#!/usr/bin/env python3
"""
calibrate_teacher.py — temperature-calibrate a trained SegFormer teacher.
=========================================================================
Ports scripts/02_calibrate_teacher.py so B5 (and any teacher lacking one) gets
its own temperature.json. The distillation trainers distill temperature-SCALED
soft targets, so an uncalibrated teacher (T=1.0) transfers over-confident,
near-binary probabilities — defeating the point of soft-label KD.

Writes temperature.json into the teacher run dir (next to best_model.pt).

    python calibrate_teacher.py \
        --teacher-dir results_segformer_b5/runs/segformer_b5__seed42 \
        --pretrained  pretrained_weights/segformer_mit_b5 \
        --train-manifest manifests/train_manifest.csv --data-root .
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import kd_core as K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-dir", required=True)
    ap.add_argument("--pretrained", required=True)
    ap.add_argument("--train-manifest", required=True)
    ap.add_argument("--data-root", default="")
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    tdir = Path(a.teacher_dir)
    out = tdir / "temperature.json"
    if out.exists() and not a.force:
        print(f"[skip] {out} exists (T={json.loads(out.read_text()).get('temperature')})")
        return
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")

    cfg = dict(K.DEFAULTS)
    full_train = K.load_manifest(a.train_manifest, a.data_root)
    _, val_df = K.resolve_train_val(full_train, cfg)

    model = K.SegformerWrapper(K.build_segformer(a.pretrained, num_labels=1)).to(device)
    model.load_state_dict(torch.load(str(tdir / "best_model.pt"),
                                     map_location=device, weights_only=True))
    val_loader = K.make_loader(val_df, a.img_size, 8, False, a.workers, 0)
    T, nll_b, nll_a = K.calibrate_temperature(model, val_loader, device, amp=not a.no_amp)
    out.write_text(json.dumps({"temperature": T, "nll_before": nll_b, "nll_after": nll_a}, indent=2))
    print(f"[done] {tdir.name}  T={T:.4f}  nll {nll_b:.4f} -> {nll_a:.4f}")
    if T < 1.0:
        print("  [warn] T<1.0 is unusual for BCE-trained models; check training.")


if __name__ == "__main__":
    main()
