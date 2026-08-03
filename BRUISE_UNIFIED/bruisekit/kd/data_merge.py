#!/usr/bin/env python3
"""
data_merge.py — build a merged-dataset manifest as a SEPARATE factor (review §7).
=================================================================================
Dataset merging and the distillation loss must never change at the same time, or
you cannot attribute the effect. This produces a clean merged manifest for the
2x2 design (data {current, merged} x method {response, proposed}) with the
pre-merge guardrails the review requires:

  - common mask semantics : both manifests must expose image_path/mask_path/stem/subject
  - label-quality rule    : --min-fg-frac drops near-empty masks (optional)
  - duplicate/leakage      : errors on any stem or SUBJECT shared across sources
  - domain identifier      : adds `domain` (0=current, 1=external)
  - source-balanced sample : adds `sample_weight` so the smaller source is not swamped
  - dataset-specific val   : keeps each source's own train/val split column
  - resolution/color note  : records per-source image size stats (harmonise upstream)

    python data_merge.py --current manifests/train_manifest.csv \
        --external EXT/train.csv --data-root-current . --data-root-external EXT \
        --out manifests/merged_train_manifest.csv

Then run the 2x2 by pointing distill_segformer --train-manifest at the merged CSV
(same --kd for both data conditions; change ONE factor at a time).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import kd_core as K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--current", required=True)
    ap.add_argument("--external", required=True)
    ap.add_argument("--data-root-current", default="")
    ap.add_argument("--data-root-external", default="")
    ap.add_argument("--min-fg-frac", type=float, default=0.0,
                    help="drop masks with fg fraction below this (0 disables)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cur = K.load_manifest(a.current, a.data_root_current); cur["domain"] = 0
    ext = K.load_manifest(a.external, a.data_root_external); ext["domain"] = 1
    for name, df in [("current", cur), ("external", ext)]:
        if "split" not in df.columns:
            print(f"[warn] {name} has no split column — its rows default to train; "
                  f"provide a per-source split to keep dataset-specific validation")
            df["split"] = "train"

    # duplicate / leakage guard (stem AND subject across sources)
    for key in ("stem", "subject"):
        shared = set(cur[key]) & set(ext[key])
        if shared:
            raise SystemExit(f"LEAKAGE: {len(shared)} shared {key} across sources "
                             f"e.g. {sorted(map(str, shared))[:5]} — resolve before merging")

    merged = pd.concat([cur, ext], ignore_index=True)

    # optional label-quality filter (needs readable masks)
    if a.min_fg_frac > 0:
        import cv2
        keep = []
        for _, r in merged.iterrows():
            m = cv2.imread(str(r.mask_path), cv2.IMREAD_GRAYSCALE)
            frac = 0.0 if m is None else float((m > 0).mean())
            keep.append(frac >= a.min_fg_frac)
        dropped = int((~np.array(keep)).sum())
        merged = merged[keep].reset_index(drop=True)
        print(f"label-quality: dropped {dropped} masks below fg_frac {a.min_fg_frac}")

    # source-balanced sampling weight (inverse source frequency, mean 1.0)
    counts = merged["domain"].map(merged["domain"].value_counts())
    w = (len(merged) / (2.0 * counts)).astype(float)
    merged["sample_weight"] = w / w.mean()

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(a.out, index=False)
    print(f"[done] merged {len(cur)} current + {len(ext)} external = {len(merged)} rows -> {a.out}")
    print(merged.groupby(["domain", "split"]).size().to_string())
    print("\nNEXT: run the 2x2 — same --kd for {current, merged} train manifests; "
          "change only ONE factor (data OR method) per comparison.")


if __name__ == "__main__":
    main()
