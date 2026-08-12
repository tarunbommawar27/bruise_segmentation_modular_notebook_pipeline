"""Native Ultralytics YOLO training + the two prediction paths.

This module deliberately uses Ultralytics (mosaic, EMA, letterbox, its own recipe) --
that is the whole point: YOLO is trained on its home turf, not on SegFormer's
transformer recipe. It is the ONLY module that imports ultralytics at train time.

TWO PREDICTION PATHS, and why both:
  native argmax   -- YOLO.predict() letterboxes to 640, argmaxes the 2-class head,
                     returns the mask at native resolution. We bring pred and GT to
                     640 (nearest) together and score. This is YOLO's best case and
                     reproduces its ~0.83 median.
  custom /255     -- pull the raw nn.Module, feed the SAME 640 stretch tensor SegFormer
                     sees (just /255, no ImageNet norm), take sigmoid(z1 - z0), and
                     sweep the threshold on val. Same geometry as SegFormer, so it is
                     the apples-to-apples-on-eval number.
"""
from __future__ import annotations

import copy
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from bruisekit.metrics import compute_image_row, summarize
from bruisekit.sweep import select_cut


def _ul_device() -> str:
    """Ultralytics device string: '0' on GPU (Colab), 'cpu' otherwise (local tests).

    Derived, not hardcoded, so the same module runs on the A100 in Colab and on a CPU
    box for verification without a code change -- a hardcoded '0' raises on CPU.
    """
    return "0" if torch.cuda.is_available() else "cpu"


def _read_native_mask(root: Path, rel: str) -> np.ndarray:
    m = cv2.imread(str(root / rel), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f"Cannot read mask: {rel}")
    if m.ndim == 3:
        m = m[..., 0]
    return (m > 0).astype(np.uint8)


def _to_640(arr: np.ndarray) -> np.ndarray:
    return cv2.resize(arr.astype(np.uint8), (640, 640), interpolation=cv2.INTER_NEAREST)


# ── dataset build (native-res, Ultralytics semantic format) ──────────────────
def build_yolo_dataset(df, work_root: Path, out_dir: Path, split: str,
                       alpha=None, teacher_prob_fn=None, imgsz: int = 640) -> None:
    """images/<split>/ (symlink to native) + masks/<split>/ (0/1 class PNG, native res).

    For a distilled TRAIN split, the pseudo-mask fuses the teacher's native-resolution
    probability: class = (alpha*GT + (1-alpha)*teacher_prob >= 0.5). alpha < 0.5 keeps
    this non-degenerate (alpha > 0.5 would collapse it to plain GT). Val/test always use
    clean GT.
    """
    img_dir = out_dir / "images" / split; img_dir.mkdir(parents=True, exist_ok=True)
    msk_dir = out_dir / "masks" / split;  msk_dir.mkdir(parents=True, exist_ok=True)
    for _, r in df.iterrows():
        src = (work_root / r.image_path).resolve()
        dst = img_dir / Path(r.image_path).name
        if not dst.exists():
            try:
                dst.symlink_to(src)
            except OSError:
                shutil.copy2(src, dst)
        gt = _read_native_mask(work_root, r.mask_path).astype(np.float32)
        if split == "train" and teacher_prob_fn is not None:
            prob = teacher_prob_fn(work_root / r.image_path, gt.shape)
            cls = ((alpha * gt + (1.0 - alpha) * prob) >= 0.5).astype(np.uint8)
        else:
            cls = gt.astype(np.uint8)
        cv2.imwrite(str(msk_dir / f"{r.stem}.png"), cls)


def write_data_yaml(out_dir: Path, run_dir: Path) -> Path:
    import yaml
    data = {"path": str(out_dir.resolve()), "train": "images/train", "val": "images/val",
            "masks_dir": "masks", "names": {0: "background", 1: "bruise"}}
    p = run_dir / "data.yaml"
    with open(p, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return p


def pretrained_for(family: str, paths: dict) -> str:
    """Which pretrained backbone a YOLO family starts from.

    One function rather than a conditional at each call site, because getting it
    wrong is silent: `YOLO("yolo26n-sem.pt")` builds and trains perfectly well
    under a run named `yolo_sem_l_direct`, produces a `best.pt`, scores, and
    lands in a table as the LARGE arm. Nothing raises. The only symptom is a
    28 M-parameter row reporting 1.6 M parameters, in a study whose entire Stage
    Y question is whether capacity fixes the miss rate.

    Nano is the default so that every pre-existing family keeps the exact weights
    it already trained with -- adding Stage Y must not perturb Stage A.
    """
    key = "yolo_l" if _is_large(family) else "yolo"
    p = paths[key]
    if not Path(p).exists():
        raise FileNotFoundError(
            f"{family} needs {Path(p).name}, which is not in pretrained_weights/.\n"
            f"  Fetch it once with:\n"
            f"    from ultralytics import YOLO; YOLO('{Path(p).name}')\n"
            f"  then move the downloaded file to {Path(p).parent}")
    return p


def _is_large(family: str) -> bool:
    """`yolo_sem_l_direct` / `yolo_sem_l_distilled` are large; everything else nano.

    Matched on the `_l_` infix rather than a substring like "l", which would also
    match `yolo_sem_distilled`.
    """
    return "_l_" in family or family.endswith("_l")


def train_native(weights_path: str, data_yaml: Path, run_dir: Path, cfg: dict, seed: int) -> Path:
    """Native Ultralytics training. Skips if best.pt exists; resumes from last.pt if
    interrupted (Ultralytics writes last.pt every epoch, so a disconnect costs <=1 epoch)."""
    from ultralytics import YOLO
    best = run_dir / "ultralytics_runs" / "train" / "weights" / "best.pt"
    last = run_dir / "ultralytics_runs" / "train" / "weights" / "last.pt"
    if best.exists():
        return best
    if last.exists():
        YOLO(str(last)).train(resume=True)
        return best
    YOLO(weights_path).train(
        data=str(data_yaml), task="semantic", imgsz=cfg["img_size"], epochs=cfg["epochs"],
        patience=cfg["patience"], batch=cfg["yolo_batch"], workers=cfg["workers"],
        device=_ul_device(), project=str(run_dir / "ultralytics_runs"), name="train", exist_ok=True,
        optimizer=cfg["yolo_optimizer"], lrf=cfg["yolo_lrf"], cos_lr=True,
        warmup_epochs=cfg["yolo_warmup_epochs"], weight_decay=cfg["yolo_weight_decay"],
        close_mosaic=cfg["yolo_close_mosaic"], seed=seed,
    )
    return best


# ── path (a): native argmax ──────────────────────────────────────────────────
def predict_native_argmax(best_pt: Path, df, work_root: Path, imgsz: int = 640):
    """YOLO.predict() argmax, pred+GT brought to 640 together and scored."""
    from ultralytics import YOLO
    model = YOLO(str(best_pt))
    rows = []
    for _, r in df.iterrows():
        res = model.predict(str(work_root / r.image_path), imgsz=imgsz, device=_ul_device(), verbose=False)[0]
        if getattr(res, "semantic_mask", None) is not None:
            cm = res.semantic_mask.data
            cm = cm.cpu().numpy() if hasattr(cm, "cpu") else np.asarray(cm)
            pred = (cm == 1).astype(np.uint8)
        else:
            pred = np.zeros((imgsz, imgsz), np.uint8)
        gt = _read_native_mask(work_root, r.mask_path)
        rows.append(compute_image_row(_to_640(pred), _to_640(gt), str(r.stem)))
    return pd.DataFrame(rows), summarize(rows)


# ── path (b): custom /255, raw module, val-swept threshold ───────────────────
def _raw_module(best_pt: Path, device):
    from ultralytics import YOLO
    return copy.deepcopy(YOLO(str(best_pt)).model).to(device).eval()


@torch.no_grad()
def _probs_640(model, df640, cache_root: Path, device):
    """sigmoid(z1 - z0) at 640, feeding the 640-stretch /255 tensor (SegFormer geometry)."""
    P, G, S = [], [], []
    for _, r in df640.iterrows():
        bgr = cv2.imread(str(cache_root / r.image_path), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)
        out = model(x)
        if isinstance(out, (list, tuple)):
            out = out[0]
        if out.shape[-2:] != (640, 640):
            out = F.interpolate(out.float(), size=(640, 640), mode="bilinear", align_corners=False)
        z = out[:, 1] - out[:, 0] if out.shape[1] >= 2 else out[:, 0]
        P.append(torch.sigmoid(z)[0].half().cpu())
        m = cv2.imread(str(cache_root / r.mask_path), cv2.IMREAD_GRAYSCALE)
        if m.ndim == 3: m = m[..., 0]
        G.append(torch.from_numpy((m > 0)))
        S.append(str(r.stem))
    return torch.stack(P), torch.stack(G), S


def predict_custom_255(best_pt: Path, val640, test640, cache_root: Path, device, thresholds):
    """Sweep the probability threshold on val, apply once to test. Same scoring as SegFormer."""
    from bruisekit.postopt import sweep_prob, score_prob_at
    model = _raw_module(best_pt, device)
    vp, vg, _ = _probs_640(model, val640, cache_root, device)
    grid = sweep_prob(vp, vg, thresholds)
    op = select_cut(grid)
    tp, tg, ts = _probs_640(model, test640, cache_root, device)
    per_img, summ = score_prob_at(tp, tg, ts, op["threshold"])
    return op, per_img, summ