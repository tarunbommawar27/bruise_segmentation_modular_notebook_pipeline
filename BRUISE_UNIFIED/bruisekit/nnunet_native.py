"""Native nnU-Net v2 baseline: drive its own CLI end to end, then score with the
same metrics as everything else so the number sits in one comparable table.

Adapted from EXTRA/train_nnunet_baseline.py. The one change vs that script: the
fold-0 split is taken DIRECTLY from the package's 697/134 train/val manifests
(not re-derived from a val_fraction), so it is bit-identical to the split the
SegFormer/SMP models use. nnU-Net trains on train+val cases with fold 0 deciding
which are validation.

WHY NATIVE: nnU-Net fingerprints the dataset and self-configures preprocessing,
architecture, patch/batch size and LR schedule. Forcing it onto the shared recipe
would discard exactly what makes it a strong baseline -- so it runs its own way,
the same reason YOLO is trained natively in the final notebook.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from bruisekit.metrics import compute_image_row, summarize


def _run(cmd: list[str]) -> None:
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def set_env(root: Path) -> dict:
    """nnU-Net reads three env vars; point them at local SSD (fast, wiped on disconnect)."""
    raw = root / "raw"; pre = root / "preprocessed"; res = root / "results"
    for d in (raw, pre, res):
        d.mkdir(parents=True, exist_ok=True)
    os.environ["nnUNet_raw"] = str(raw)
    os.environ["nnUNet_preprocessed"] = str(pre)
    os.environ["nnUNet_results"] = str(res)
    return {"raw": raw, "pre": pre, "res": res}


def dataset_dirname(dataset_id: int, name: str) -> str:
    return f"Dataset{int(dataset_id):03d}_{name}"


def _case_id(stem: str, idx: int) -> str:
    safe = "".join(c if c.isalnum() else "_" for c in str(stem))
    return f"c{idx:04d}_{safe}"[:64]


def _write_case(work_root: Path, image_rel: str, out_img_dir: Path, case: str) -> None:
    """RGB image -> three single-channel PNGs case_0000/_0001/_0002 (nnU-Net RGB layout)."""
    bgr = cv2.imread(str(work_root / image_rel), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"cannot read image {image_rel}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    for ch in range(3):
        cv2.imwrite(str(out_img_dir / f"{case}_{ch:04d}.png"), rgb[:, :, ch])


def _write_label(work_root: Path, mask_rel: str, out_lbl_dir: Path, case: str) -> None:
    m = cv2.imread(str(work_root / mask_rel), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f"cannot read mask {mask_rel}")
    if m.ndim == 3:
        m = m[..., 0]
    cv2.imwrite(str(out_lbl_dir / f"{case}.png"), (m > 0).astype(np.uint8))     # 0/1, not 0/255


def convert(env, ds_dir, work_root, train_df, val_df, test_df):
    """Stage 1: manifests -> nnU-Net raw. fold-0 split comes from the train/val manifests."""
    raw = env["raw"] / ds_dir
    imagesTr, labelsTr = raw / "imagesTr", raw / "labelsTr"
    imagesTs = raw / "imagesTs"
    for d in (imagesTr, labelsTr, imagesTs):
        d.mkdir(parents=True, exist_ok=True)

    val_subjects = set(val_df["subject"])
    combined = pd.concat([train_df, val_df], ignore_index=True)
    mapping = []
    for i, r in combined.iterrows():
        case = _case_id(r.stem, i)
        _write_case(work_root, r.image_path, imagesTr, case)
        _write_label(work_root, r.mask_path, labelsTr, case)
        mapping.append({"case": case, "stem": r.stem, "subject": r.subject,
                        "split": "val" if r.subject in val_subjects else "train"})

    test_map = []
    for i, r in test_df.iterrows():
        case = _case_id(r.stem, 100000 + i)
        _write_case(work_root, r.image_path, imagesTs, case)
        test_map.append({"case": case, "stem": r.stem, "gt_mask": str(work_root / r.mask_path)})

    (raw / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "R", "1": "G", "2": "B"},
        "labels": {"background": 0, "bruise": 1},
        "numTraining": len(mapping), "file_ending": ".png",
    }, indent=2))
    pd.DataFrame(mapping).to_csv(raw / "train_case_mapping.csv", index=False)
    pd.DataFrame(test_map).to_csv(raw / "test_case_mapping.csv", index=False)
    n_tr = sum(m["split"] == "train" for m in mapping); n_va = len(mapping) - n_tr
    print(f"  raw -> {raw}  ({n_tr} train / {n_va} val / {len(test_map)} test cases)")


def plan(cfg):
    _run(["nnUNetv2_plan_and_preprocess", "-d", cfg["nnunet_dataset_id"], "--verify_dataset_integrity"])


def write_splits(env, ds_dir):
    """Stage 3: subject-grouped splits_final.json (fold 0) from our mapping."""
    raw = env["raw"] / ds_dir
    pre = env["pre"] / ds_dir
    if not pre.exists():
        raise FileNotFoundError(f"{pre} missing; run plan() first.")
    m = pd.read_csv(raw / "train_case_mapping.csv")
    fold0 = {"train": m[m.split == "train"].case.tolist(),
             "val":   m[m.split == "val"].case.tolist()}
    (pre / "splits_final.json").write_text(json.dumps([fold0] * 5, indent=2))
    print(f"  splits_final.json (fold 0: {len(fold0['train'])} train / {len(fold0['val'])} val)")


def train(cfg):
    cmd = ["nnUNetv2_train", cfg["nnunet_dataset_id"], cfg["nnunet_config"], cfg["nnunet_fold"]]
    if cfg.get("nnunet_epochs"):
        cmd += ["-num_epochs", cfg["nnunet_epochs"]]
    # --c continues from the latest checkpoint if a previous session was interrupted.
    cmd += ["--c"]
    try:
        _run(cmd)
    except subprocess.CalledProcessError:
        # first run has no checkpoint to continue from -> retry without --c
        _run([c for c in cmd if c != "--c"])


def predict(cfg, env, ds_dir, out_dir):
    raw = env["raw"] / ds_dir
    out = Path(out_dir) / "nnunet_test_pred"; out.mkdir(parents=True, exist_ok=True)
    _run(["nnUNetv2_predict", "-i", raw / "imagesTs", "-o", out,
          "-d", cfg["nnunet_dataset_id"], "-c", cfg["nnunet_config"], "-f", cfg["nnunet_fold"]])
    return out


def _load_bin(path, size, interp):
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f"cannot read {path}")
    if m.ndim == 3:
        m = m[..., 0]
    return cv2.resize((m > 0).astype(np.uint8), (size, size), interpolation=interp)


def score(env, ds_dir, pred_dir, size=640):
    """Stage 6: pred & GT resized together (nearest) to 640 -- same geometry as everything else."""
    raw = env["raw"] / ds_dir
    tmap = pd.read_csv(raw / "test_case_mapping.csv")
    rows = []
    for _, r in tmap.iterrows():
        pp = Path(pred_dir) / f"{r.case}.png"
        if not pp.exists():
            print(f"  !! missing prediction {pp}; skipping {r.stem}"); continue
        pred = _load_bin(pp, size, cv2.INTER_NEAREST)
        gt = _load_bin(r.gt_mask, size, cv2.INTER_NEAREST)
        rows.append(compute_image_row(pred, gt, str(r.stem)))
    if not rows:
        raise RuntimeError("no scored images; did predict() run?")
    per_image = pd.DataFrame(rows)
    return per_image, summarize(rows)