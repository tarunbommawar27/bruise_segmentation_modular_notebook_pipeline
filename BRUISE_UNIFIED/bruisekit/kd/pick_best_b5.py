#!/usr/bin/env python3
"""
pick_best_b5.py — select the best B5 seed on val, calibrate, bundle as teacher.
===============================================================================
The B5 baseline trains 3 seeds. This picks the best ON VALIDATION with the LOCKED
eval (kd_core threshold sweep), calibrates its temperature, scores val+test, and
writes teachers/segformer_b5_teacher/{best_model.pt, temperature.json,
val_per_image.csv, test_per_image.csv, seed_selection.csv}. B5 checkpoints are
already SegformerWrapper layout (trained by this suite) — no conversion needed.

    python pick_best_b5.py --runs results_segformer_b5/runs --seeds 42 123 2026 \
        --pretrained pretrained_weights/segformer_mit_b5 \
        --train-manifest manifests/train_manifest.csv \
        --test-manifest manifests/test_manifest.csv --data-root . \
        --test-ita ita_labels/wl_test_per_image_ita.csv \
        --out teachers/segformer_b5_teacher
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

import kd_core as K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="dir with segformer_b5__seed<N> subdirs")
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--pretrained", required=True)
    ap.add_argument("--train-manifest", required=True)
    ap.add_argument("--test-manifest", required=True)
    ap.add_argument("--data-root", default="")
    ap.add_argument("--test-ita", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
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
    val_loader = K.make_loader(val_df, cfg["img_size"], 8, False, a.workers, 0)
    test_loader = K.make_loader(test_df, cfg["img_size"], 8, False, a.workers, 0)

    rows, best = [], None
    for sd in a.seeds:
        ck = Path(a.runs) / f"segformer_b5__seed{sd}" / "best_model.pt"
        if not ck.exists():
            print("  missing", ck); continue
        model = K.SegformerWrapper(K.build_segformer(a.pretrained, num_labels=1)).to(device)
        model.load_state_dict(torch.load(str(ck), map_location=device, weights_only=True))
        thr_df, thr = K.threshold_sweep(model, val_loader, device, cfg["thresholds"], amp)
        vdice = float(thr_df.iloc[0]["mean_dice"])
        rows.append({"seed": sd, "val_mean_dice": vdice, "threshold": thr, "ckpt": str(ck)})
        print(f"  B5 seed{sd}: val_dice={vdice:.4f} @thr={thr:.2f}")
        if best is None or vdice > best["val_mean_dice"]:
            if best is not None:
                del best["model"]
            best = {"seed": sd, "val_mean_dice": vdice, "threshold": thr, "ckpt": str(ck), "model": model}
        else:
            del model
        torch.cuda.empty_cache()
    if best is None:
        raise SystemExit(f"no B5 checkpoints under {a.runs} for seeds {a.seeds}")

    model = best["model"]; thr = best["threshold"]
    T, _, _ = K.calibrate_temperature(model, val_loader, device, amp)
    vmap = dict(zip(val_df["stem"].astype(str), val_df["subject"].astype(str)))
    tmap = dict(zip(test_df["stem"].astype(str), test_df["subject"].astype(str)))
    val_pi, _ = K.evaluate(model, val_loader, device, thr, amp); val_pi["subject"] = val_pi["stem"].map(vmap)
    test_pi, summ = K.evaluate(model, test_loader, device, thr, amp); test_pi["subject"] = test_pi["stem"].map(tmap)

    torch.save(model.state_dict(), out / "best_model.pt")
    (out / "temperature.json").write_text(json.dumps({"temperature": T}, indent=2))
    (out / "threshold.json").write_text(json.dumps({"threshold": thr, "selected_seed": best["seed"]}, indent=2))
    val_pi.to_csv(out / "val_per_image.csv", index=False)
    test_pi.to_csv(out / "test_per_image.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "seed_selection.csv", index=False)
    if a.test_ita and Path(a.test_ita).exists():
        K.fairness_by_group(test_pi, a.test_ita, "segformer_b5_teacher", out_csv=out / "fairness_per_group.csv")
    tail = K.tail_metrics(test_pi)
    print(f"[done] BEST B5 seed {best['seed']}  T={T:.3f}  test dice={summ['mean_dice']:.4f} "
          f"median={summ['median_dice']:.4f} miss={summ['complete_miss_rate']*100:.2f}% "
          f"rec<.1={tail['pct_recall_below_0.10']*100:.1f}% -> {out}")


if __name__ == "__main__":
    main()
