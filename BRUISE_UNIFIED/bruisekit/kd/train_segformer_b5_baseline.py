#!/usr/bin/env python3
"""
train_segformer_b5_baseline.py
==============================
SegFormer MiT-B5 baseline for WL bruise segmentation, trained with the SAME
custom loop / recipe as the core SegFormer models (B2 teacher, B0 direct) in
`pipeline/trainer.py`, then scored with the SAME Dice / IoU / complete-miss
metrics at matched 640 geometry so the number sits in one comparable table.

WHY THE CUSTOM LOOP (and not nnU-Net's or Ultralytics')
-------------------------------------------------------
MiT-B5 is structurally identical to the B2/B0 models already in the study:
a pretrained MiT encoder + a randomly-initialised all-MLP decode head. So it
takes the reference recipe verbatim: Dice+BCE loss, AdamW with a backbone/head
LR split (6e-5 / 6e-4), no weight decay on norms & biases, linear warmup (1%)
-> poly decay stepped per optimizer step, AMP + grad-clip 1.0, VRAM-probed
micro-batch accumulated to an effective batch of 8, epoch selection on val
mean Dice @ 0.50, threshold swept on VAL (0.05..0.95), scored ONCE on TEST.
Holding the recipe fixed across encoder scales is the whole point.

OFFLINE PRETRAINED WEIGHTS
--------------------------
ORC compute nodes have no internet, so the ImageNet-pretrained `nvidia/mit-b5`
encoder is bundled in `pretrained_weights/segformer_mit_b5/` and loaded from
disk. The checkpoint is a SegformerForImageClassification export; loading it
into SegformerForSemanticSegmentation with ignore_mismatched_sizes=True keeps
the encoder weights and randomly initialises the decode head -- exactly how the
B2 teacher was built from `nvidia/mit-b2`.

SPLIT
-----
The fold-0 train/val membership comes straight from the `split` column in
train_manifest.csv -- the canonical 697 train / 134 val partition shared with
the 5 core models and the nnU-Net baseline. No re-derivation. If the column is
absent a subject-grouped fallback split (val_fraction=0.18, seed=42) is used.

Usage
-----
    pip install "transformers>=4.40" "albumentations>=1.4" torch \
                opencv-python-headless pandas tqdm

    python train_segformer_b5_baseline.py \
        --train-manifest manifests/train_manifest.csv \
        --test-manifest  manifests/test_manifest.csv \
        --data-root . --out-dir results_segformer_b5 \
        --pretrained pretrained_weights/segformer_mit_b5 \
        --seeds 42

Resume: fully automatic — if `resume_checkpoint.pt` exists in the run dir the
loop continues from the last completed epoch (written every epoch, so a 12 h
wall-time kill loses at most one epoch). Re-run the same command.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

# Force offline BEFORE transformers import — ORC compute nodes have no internet
# and a hub lookup would hang for minutes before failing.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# albumentations & transformers are imported lazily inside the functions that
# need them so `--help` and manifest inspection work without a full ML stack.


# ======================================================================================
# §0 · Config defaults (verbatim from configs/common_train.yaml)
# ======================================================================================
DEFAULTS = dict(
    img_size        = 640,
    epochs          = 100,
    patience        = 15,
    workers         = 0,
    amp             = True,
    backbone_lr     = 6e-5,      # pretrained MiT encoder -> conservative
    head_lr         = 6e-4,      # random all-MLP decode head -> 10x
    betas           = (0.9, 0.999),
    weight_decay    = 0.01,
    warmup_fraction = 0.01,
    poly_power      = 1.0,
    gradient_clip   = 1.0,
    effective_batch = 8,
    max_probe_batch = 32,
    vram_target     = 0.75,
    # subject-grouped fallback split (only if the manifest lacks a `split` column)
    val_fraction    = 0.18,
    split_seed      = 42,
    # threshold sweep on VAL (identical grid to pipeline/trainer.py)
    thresholds      = [round(0.05 * k, 2) for k in range(1, 20)],   # 0.05 .. 0.95
)


# ======================================================================================
# §1 · Manifest loading  (robust column detection + canonical split column)
# ======================================================================================
_IMAGE_KEYS   = ["image_path", "img_path", "image", "wl_image_path", "image_file", "filepath", "path", "rgb_path"]
_MASK_KEYS    = ["mask_path", "mask", "label_path", "gt_path", "annotation_path", "mask_file", "seg_path"]
_SUBJECT_KEYS = ["subject", "subject_id", "patient", "patient_id", "case_id", "case", "person_id"]
_STEM_KEYS    = ["stem", "id", "name", "image_id", "sample_id"]
_SPLIT_KEYS   = ["split", "subset", "partition"]


def _pick(df: pd.DataFrame, keys: list[str], required: bool, what: str) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for k in keys:
        if k in lower:
            return lower[k]
    if required:
        raise KeyError(f"Could not find a {what} column. Looked for {keys}; "
                       f"manifest has {list(df.columns)}.")
    return None


def load_manifest(csv_path: str, data_root: str) -> pd.DataFrame:
    """Normalised manifest: stem, subject, image_path, mask_path (absolute), split (if present)."""
    df = pd.read_csv(csv_path)
    img_c   = _pick(df, _IMAGE_KEYS,   True,  "image path")
    mask_c  = _pick(df, _MASK_KEYS,    True,  "mask path")
    subj_c  = _pick(df, _SUBJECT_KEYS, False, "subject")
    stem_c  = _pick(df, _STEM_KEYS,    False, "stem")
    split_c = _pick(df, _SPLIT_KEYS,   False, "split")

    root = Path(data_root) if data_root else None

    def _abs(p):
        p = Path(str(p))
        return str(p) if (p.is_absolute() or root is None) else str(root / p)

    out = pd.DataFrame()
    out["image_path"] = df[img_c].apply(_abs)
    out["mask_path"]  = df[mask_c].apply(_abs)
    out["stem"]       = df[stem_c].astype(str) if stem_c else out["image_path"].apply(lambda p: Path(p).stem)
    out["subject"]    = df[subj_c].astype(str) if subj_c else out["stem"]
    if split_c:
        out["split"] = df[split_c].astype(str)
    return out.drop_duplicates(subset=["image_path"]).reset_index(drop=True)


def subject_val_split(train_df: pd.DataFrame, val_fraction: float, seed: int):
    """Fallback: carve VALIDATION by SUBJECT (never split a subject across train/val)."""
    subjects = sorted(train_df["subject"].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    n_val = max(1, int(round(len(subjects) * val_fraction)))
    val_subjects = set(subjects[:n_val])
    val = train_df[train_df["subject"].isin(val_subjects)].reset_index(drop=True)
    trn = train_df[~train_df["subject"].isin(val_subjects)].reset_index(drop=True)
    assert not (set(trn["subject"]) & set(val["subject"])), "subject leak train/val"
    return trn, val


def resolve_train_val(full_train: pd.DataFrame, cfg: dict):
    """Prefer the CANONICAL `split` column (697/134); subject-grouped fallback otherwise."""
    if "split" in full_train.columns and full_train["split"].isin(["train", "val"]).any():
        trn = full_train[full_train["split"] == "train"].reset_index(drop=True)
        val = full_train[full_train["split"] == "val"].reset_index(drop=True)
        print(f"  using CANONICAL split column: {len(trn)} train / {len(val)} val")
        return trn, val
    print(f"  no split column; subject-grouped fallback "
          f"(val_fraction={cfg['val_fraction']}, seed={cfg['split_seed']})")
    return subject_val_split(full_train, cfg["val_fraction"], cfg["split_seed"])


# ======================================================================================
# §2 · Data  (verbatim from pipeline/data.py: ImageNet norm IN the loader)
# ======================================================================================
def get_augmentation(training: bool, img_size: int):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    imagenet_norm = A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    resize = [A.Resize(height=img_size, width=img_size)]   # bilinear img / nearest mask
    if not training:
        return A.Compose(resize + [imagenet_norm, ToTensorV2()])
    return A.Compose(resize + [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.20, contrast_limit=0.20, p=0.4),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
        A.GaussNoise(p=0.2),
        imagenet_norm,
        ToTensorV2(),
    ])


class BruiseDataset(torch.utils.data.Dataset):
    """Reads native-res image+mask, resizes to img_size.
    Returns (x[3,H,W] ImageNet-normalised, y[1,H,W]{0,1}, stem)."""

    def __init__(self, df: pd.DataFrame, img_size: int, training: bool = False):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.tfm = get_augmentation(training, img_size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        img = cv2.imread(str(r.image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Cannot read image: {r.image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(r.mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Cannot read mask: {r.mask_path}")
        if mask.ndim == 3:               # ultralytics monkey-patches cv2.imread -> (H,W,1)
            mask = mask[..., 0]
        mask = (mask > 0).astype(np.float32)

        aug = self.tfm(image=img, mask=mask)
        x = aug["image"].float()
        y = aug["mask"].unsqueeze(0).float()
        assert y.shape == (1, self.img_size, self.img_size), f"bad mask shape {y.shape} for {r.stem}"
        return x, y, str(r.stem)


def make_loader(df, img_size, batch_size, training, workers, seed=0):
    ds = BruiseDataset(df, img_size, training=training)
    gen = torch.Generator(); gen.manual_seed(seed)

    def _init_worker(worker_id):
        np.random.seed(seed * 1000 + worker_id)

    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=training, drop_last=training,
        num_workers=workers, pin_memory=True,
        persistent_workers=workers > 0,
        worker_init_fn=_init_worker, generator=gen,
    )


# ======================================================================================
# §3 · Model  (verbatim from pipeline/models.py: SegformerWrapper)
# ======================================================================================
def build_segformer(pretrained: str, num_labels: int = 1) -> nn.Module:
    from transformers import SegformerForSemanticSegmentation
    return SegformerForSemanticSegmentation.from_pretrained(
        pretrained, num_labels=num_labels, ignore_mismatched_sizes=True,
    )


class SegformerWrapper(nn.Module):
    """Wraps HF SegformerForSemanticSegmentation; upsamples logits to input resolution
    and exposes the backbone (.segformer) so the optimizer can give it a lower LR."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @property
    def backbone(self) -> nn.Module:
        return self.model.segformer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=x)
        logits = out.logits
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits

    def gradient_checkpointing_enable(self) -> None:
        try:
            self.model.segformer.encoder.gradient_checkpointing = True
        except AttributeError:
            pass


def count_params(model): return sum(p.numel() for p in model.parameters())


def build_param_groups(model, backbone_lr, head_lr, weight_decay):
    """Backbone/head LR split + no weight decay on norms and biases (by id(), not name)."""
    backbone_ids = {id(p) for p in model.backbone.parameters()}
    groups = {
        "backbone_decay":    {"params": [], "lr": backbone_lr, "weight_decay": weight_decay},
        "backbone_no_decay": {"params": [], "lr": backbone_lr, "weight_decay": 0.0},
        "head_decay":        {"params": [], "lr": head_lr,     "weight_decay": weight_decay},
        "head_no_decay":     {"params": [], "lr": head_lr,     "weight_decay": 0.0},
    }
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        where = "backbone" if id(p) in backbone_ids else "head"
        decay = "_no_decay" if (p.ndim <= 1 or "norm" in name.lower() or "bias" in name.lower()) else "_decay"
        groups[where + decay]["params"].append(p)
    out = [g for g in groups.values() if g["params"]]
    n_grouped = sum(len(g["params"]) for g in out)
    n_total = sum(1 for _, p in model.named_parameters() if p.requires_grad)
    assert n_grouped == n_total, f"param grouping lost {n_total - n_grouped} tensors"
    return out


# ======================================================================================
# §4 · Loss & metrics  (verbatim from pipeline/losses.py + pipeline/metrics.py)
# ======================================================================================
class DiceBCELoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__(); self.smooth = smooth

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target)
        prob = torch.sigmoid(logits)
        inter = (prob * target).sum(dim=(1, 2, 3))
        denom = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        return bce + (1.0 - dice.mean())


def dice_np(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    denom = pred.sum() + gt.sum()
    return 1.0 if denom == 0 else float(2 * np.logical_and(pred, gt).sum() / denom)


def iou_np(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    union = np.logical_or(pred, gt).sum()
    return 1.0 if union == 0 else float(np.logical_and(pred, gt).sum() / union)


def compute_image_row(pred, gt, stem):
    pred_b, gt_b = pred.astype(bool), gt.astype(bool)
    tp = int(np.logical_and(pred_b, gt_b).sum())
    fp = int(np.logical_and(pred_b, ~gt_b).sum())
    fn = int(np.logical_and(~pred_b, gt_b).sum())
    return {"stem": stem, "dice": dice_np(pred, gt), "iou": iou_np(pred, gt),
            "precision": 1.0 if tp + fp == 0 else tp / (tp + fp),
            "recall": 1.0 if tp + fn == 0 else tp / (tp + fn),
            "pred_positive_pixels": int(pred_b.sum()), "gt_positive_pixels": int(gt_b.sum())}


def summarize(rows):
    df = pd.DataFrame(rows)
    miss = (df["pred_positive_pixels"] == 0) & (df["gt_positive_pixels"] > 0)
    return {"n_images": int(len(df)),
            "mean_dice": float(df["dice"].mean()), "median_dice": float(df["dice"].median()),
            "mean_iou": float(df["iou"].mean()), "median_iou": float(df["iou"].median()),
            "mean_precision": float(df["precision"].mean()), "mean_recall": float(df["recall"].mean()),
            "complete_miss_count": int(miss.sum()), "complete_miss_rate": float(miss.mean())}


# ======================================================================================
# §5 · Engine  (seed, schedule, VRAM probe, threshold-eval, resumable train loop)
# ======================================================================================
def seed_everything(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def lr_multiplier(step, total_steps, warmup_steps, power=1.0):
    if step <= warmup_steps:
        return step / max(1, warmup_steps)
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return (1.0 - progress) ** power


def find_optimal_micro_batch(model, img_size, device, effective_batch, target_hi, amp, max_probe):
    """Probe increasing micro-batches with a real fwd+bwd+step, measuring PEAK reserved
    memory. Returns (micro_batch, accum_steps, vram_fraction) with accum sized so
    micro_batch * accum_steps ~= effective_batch (pipeline/batch_finder.py logic)."""
    if not torch.cuda.is_available():
        micro_batch = min(effective_batch, max_probe)
        return micro_batch, max(1, effective_batch // micro_batch), 0.0

    total_vram = torch.cuda.get_device_properties(device).total_memory
    scaler = torch.amp.GradScaler("cuda") if amp else None
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)  # throwaway, exercises .step()

    chosen_batch, chosen_frac, batch = 1, 0.0, 1
    while batch <= max_probe:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            x = torch.randn(batch, 3, img_size, img_size, device=device)
            y = torch.randint(0, 2, (batch, 1, img_size, img_size), device=device).float()
            model.train(); optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=amp):
                logits = model(x)
                loss = F.binary_cross_entropy_with_logits(logits, y)
            if scaler is not None:
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); optimizer.step()
            frac = torch.cuda.max_memory_reserved(device) / total_vram
            del x, y, logits, loss
            torch.cuda.empty_cache()
            if frac > target_hi:
                break
            chosen_batch, chosen_frac = batch, frac
            batch *= 2
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            break

    optimizer.zero_grad(set_to_none=True)
    micro_batch = max(1, chosen_batch)
    accum_steps = max(1, effective_batch // micro_batch)
    return micro_batch, accum_steps, chosen_frac


@torch.no_grad()
def evaluate(model, loader, device, threshold, amp):
    """Per-image rows + summary at one threshold (pipeline/trainer.py::evaluate)."""
    model.eval(); rows = []
    for x, y, stems in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp):
            logits = model(x)
        prob = torch.sigmoid(logits.float()).cpu().numpy()   # fp32 before sigmoid: no fp16 overflow
        gt = y.numpy()
        for i, stem in enumerate(stems):
            pred = (prob[i, 0] >= threshold).astype("uint8")
            g = (gt[i, 0] > 0.5).astype("uint8")
            rows.append(compute_image_row(pred, g, str(stem)))
    return pd.DataFrame(rows), summarize(rows)


def threshold_sweep(model, loader, device, thresholds, amp):
    """Sweep candidate thresholds on VAL; best = highest val mean Dice (never on test)."""
    rows = []
    for thr in thresholds:
        _, s = evaluate(model, loader, device, thr, amp)
        rows.append({"threshold": thr, **s})
    df = pd.DataFrame(rows).sort_values("mean_dice", ascending=False)
    return df, float(df.iloc[0]["threshold"])


def _atomic_save(obj, dest: Path):
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    torch.save(obj, tmp); os.replace(tmp, dest)


def train_run(run_id, seed, cfg, manifests, runs_dir, device, pretrained, grad_ckpt):
    """Train one seed of SegFormer-B5. Idempotent & resumable (12 h wall-time safe)."""
    try:
        from tqdm.auto import tqdm
    except Exception:
        def tqdm(x, **k): return x

    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    done_file = run_dir / "DONE.json"
    if done_file.exists():
        return {"run_id": run_id, "status": "skipped", **json.loads(done_file.read_text())}

    seed_everything(seed)
    amp = cfg["amp"]
    model = SegformerWrapper(build_segformer(pretrained, num_labels=1)).to(device)
    if grad_ckpt:
        model.gradient_checkpointing_enable()
        print("  gradient checkpointing: ON")

    resume_path = run_dir / "resume_checkpoint.pt"
    if resume_path.exists():
        # Skip the probe — reuse the batch size found before the wall-time kill
        st_tmp = torch.load(str(resume_path), map_location="cpu", weights_only=False)
        micro, accum, vram_frac = st_tmp["micro_batch"], st_tmp["accum_steps"], st_tmp.get("vram_frac", 0.0)
        del st_tmp
        print(f"  [resume] reusing micro_batch={micro} accum_steps={accum}")
    else:
        micro, accum, vram_frac = find_optimal_micro_batch(
            model, cfg["img_size"], device, cfg["effective_batch"],
            cfg["vram_target"], amp, cfg["max_probe_batch"])
    print(f"  micro_batch={micro} accum_steps={accum} effective={micro*accum} "
          f"vram_frac={vram_frac:.3f}")

    train_loader = make_loader(manifests["train"], cfg["img_size"], micro, True,  cfg["workers"], seed)
    val_loader   = make_loader(manifests["val"],   cfg["img_size"], micro, False, cfg["workers"], seed)

    param_groups = build_param_groups(model, cfg["backbone_lr"], cfg["head_lr"], cfg["weight_decay"])
    optimizer = torch.optim.AdamW(param_groups, betas=tuple(cfg["betas"]))
    peak_lrs = [g["lr"] for g in param_groups]
    scaler = torch.amp.GradScaler("cuda") if amp else None

    steps_per_epoch = max(1, len(train_loader) // accum)
    total_steps = steps_per_epoch * cfg["epochs"]
    warmup_steps = max(1, int(total_steps * cfg["warmup_fraction"]))
    criterion = DiceBCELoss()

    start_epoch, best_dice, patience, global_step, history = 1, float("-inf"), 0, 0, []
    if resume_path.exists():
        st = torch.load(str(resume_path), map_location=device, weights_only=False)
        model.load_state_dict(st["model"]); optimizer.load_state_dict(st["optimizer"])
        if scaler is not None and st.get("scaler"):
            scaler.load_state_dict(st["scaler"])
        start_epoch, best_dice = st["epoch"] + 1, st["best_dice"]
        patience, global_step, history = st["patience"], st["global_step"], st["history"]
        print(f"  [resume] {run_id} from epoch {start_epoch} (best_val_dice={best_dice:.4f})"); del st

    (run_dir / "run_config.json").write_text(json.dumps({
        "run_id": run_id, "model": "segformer_b5", "pretrained": str(pretrained), "seed": seed,
        "micro_batch": micro, "accum_steps": accum, "effective_batch": micro * accum,
        "vram_fraction_at_probe": round(vram_frac, 4),
        "backbone_lr": cfg["backbone_lr"], "head_lr": cfg["head_lr"],
        "weight_decay": cfg["weight_decay"], "total_steps": total_steps,
        "warmup_steps": warmup_steps, "poly_power": cfg["poly_power"],
        "img_size": cfg["img_size"], "params": count_params(model),
        "gradient_checkpointing": bool(grad_ckpt),
        "n_train": int(len(manifests["train"])), "n_val": int(len(manifests["val"])),
    }, indent=2))

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        running, t0 = 0.0, time.time()
        for step, (x, y, _) in enumerate(tqdm(train_loader, desc=f"{run_id} e{epoch}", leave=False)):
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            # LR stepped per OPTIMIZER step (accumulation boundary), like pipeline/trainer.py
            with torch.amp.autocast("cuda", enabled=amp):
                logits = model(x)
                loss = criterion(logits, y) / accum
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            running += loss.item() * accum
            if (step + 1) % accum == 0 or (step + 1) == len(train_loader):
                global_step += 1
                mult = lr_multiplier(global_step, total_steps, warmup_steps, cfg["poly_power"])
                for g, peak in zip(optimizer.param_groups, peak_lrs):
                    g["lr"] = peak * mult
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"])
                    scaler.step(optimizer); scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"])
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        # epoch selection on val mean Dice @ 0.50 (same as the core SegFormer models)
        _, val_summary = evaluate(model, val_loader, device, 0.50, amp)
        val_dice = val_summary["mean_dice"]
        train_loss = running / max(1, len(train_loader))
        cur_lr = peak_lrs[0] * lr_multiplier(global_step, total_steps, warmup_steps, cfg["poly_power"])
        history.append({"epoch": epoch, "train_loss": round(train_loss, 6),
                        "backbone_lr": cur_lr, **val_summary,
                        "sec": round(time.time() - t0, 1)})
        pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)

        if val_dice > best_dice:
            best_dice, patience = val_dice, 0
            _atomic_save(model.state_dict(), run_dir / "best_model.pt"); flag = " *"
        else:
            patience += 1; flag = ""
        print(f"  {run_id} e{epoch:3d} loss={train_loss:.4f} val_dice={val_dice:.4f} "
              f"lr={cur_lr:.2e} {time.time()-t0:.0f}s{flag}")

        # Resume checkpoint every epoch — a wall-time kill loses at most 1 epoch
        _atomic_save({"epoch": epoch, "model": model.state_dict(),
                      "optimizer": optimizer.state_dict(),
                      "scaler": scaler.state_dict() if scaler else None,
                      "best_dice": best_dice, "patience": patience,
                      "global_step": global_step, "history": history,
                      "micro_batch": micro, "accum_steps": accum,
                      "vram_frac": vram_frac}, resume_path)
        if patience >= cfg["patience"]:
            print(f"  early stop at epoch {epoch} (patience={cfg['patience']})"); break

    summary = {"run_id": run_id, "model": "segformer_b5", "seed": seed,
               "best_val_dice": best_dice, "epochs_trained": len(history),
               "params": count_params(model), "micro_batch": micro, "accum_steps": accum}
    done_file.write_text(json.dumps(summary, indent=2))
    if resume_path.exists():
        resume_path.unlink()
    del model; torch.cuda.empty_cache()
    return {"status": "trained", **summary}


# ======================================================================================
# §6 · Main
# ======================================================================================
def parse_args():
    p = argparse.ArgumentParser(description="SegFormer MiT-B5 WL bruise baseline (reference custom loop).")
    p.add_argument("--train-manifest", required=True)
    p.add_argument("--test-manifest",  required=True)
    p.add_argument("--out-dir",        required=True)
    p.add_argument("--data-root", default="", help="Prefix for relative paths in manifests (else use as-is).")
    p.add_argument("--pretrained", default="pretrained_weights/segformer_mit_b5",
                   help="Local dir with the bundled nvidia/mit-b5 checkpoint (loaded offline).")
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    p.add_argument("--patience", type=int, default=DEFAULTS["patience"])
    p.add_argument("--img-size", type=int, default=DEFAULTS["img_size"])
    p.add_argument("--workers", type=int, default=DEFAULTS["workers"])
    p.add_argument("--effective-batch", type=int, default=DEFAULTS["effective_batch"])
    p.add_argument("--grad-checkpoint", action="store_true",
                   help="Enable encoder gradient checkpointing (if B5@640 probes to micro_batch=1).")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--eval-only", action="store_true",
                   help="Skip training; sweep + score existing best_model.pt runs.")
    return p.parse_args()


def main():
    a = parse_args()
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("!! WARNING: no CUDA -> running on CPU. This is only sane for a smoke test.")

    pretrained = Path(a.pretrained)
    assert (pretrained / "config.json").exists(), \
        f"Pretrained weights not found at {pretrained} (need config.json + pytorch_model.bin)."

    cfg = dict(DEFAULTS)
    cfg.update(img_size=a.img_size, epochs=a.epochs, patience=a.patience, workers=a.workers,
               amp=not a.no_amp, effective_batch=a.effective_batch)

    out_dir = Path(a.out_dir); runs_dir = out_dir / "runs"; res_dir = out_dir / "results"
    runs_dir.mkdir(parents=True, exist_ok=True); res_dir.mkdir(parents=True, exist_ok=True)

    # --- manifests: canonical 697/134 split from the `split` column -------------------
    full_train = load_manifest(a.train_manifest, a.data_root)
    test_df    = load_manifest(a.test_manifest,  a.data_root)
    train_df, val_df = resolve_train_val(full_train, cfg)

    # leakage guards vs the consensus test set
    for col in ("subject", "stem"):
        leak = (set(train_df[col]) | set(val_df[col])) & set(test_df[col])
        if leak:
            print(f"!! WARNING: {col} overlap between train/val and TEST: {sorted(leak)[:10]}...")
    manifests = {"train": train_df, "val": val_df, "test": test_df}
    print(f"train {len(train_df)} imgs / {train_df.subject.nunique()} subj | "
          f"val {len(val_df)} imgs / {val_df.subject.nunique()} subj | test {len(test_df)} imgs")

    all_rows = []
    for seed in a.seeds:
        run_id = f"segformer_b5__seed{seed}"
        print(f"\n{'='*72}\n{run_id}\n{'='*72}")
        t0 = time.time()
        if not a.eval_only:
            res = train_run(run_id, seed, cfg, manifests, runs_dir, device,
                            pretrained, a.grad_checkpoint)
            print(f"-> {res['status']} best_val_dice={res.get('best_val_dice', float('nan')):.4f} "
                  f"({(time.time()-t0)/60:.1f} min)")

        rd = runs_dir / run_id
        if not (rd / "best_model.pt").exists():
            print(f"  no best_model.pt for {run_id}; skipping eval"); continue

        # --- sweep threshold on VAL, then score ONCE on TEST -------------------------
        model = SegformerWrapper(build_segformer(pretrained, num_labels=1)).to(device)
        model.load_state_dict(torch.load(str(rd / "best_model.pt"), map_location=device,
                                         weights_only=True))
        val_loader  = make_loader(val_df,  cfg["img_size"], 8, False, cfg["workers"], seed)
        test_loader = make_loader(test_df, cfg["img_size"], 8, False, cfg["workers"], seed)

        thr_df, best_thr = threshold_sweep(model, val_loader, device, cfg["thresholds"], cfg["amp"])
        thr_df.to_csv(rd / "threshold_search.csv", index=False)
        print(f"  best VAL threshold = {best_thr:.2f} "
              f"(val_dice={thr_df.iloc[0]['mean_dice']:.4f})")

        pi, summ = evaluate(model, test_loader, device, best_thr, cfg["amp"])
        pi.to_csv(rd / "test_per_image.csv", index=False)
        row = {"model": "segformer_b5", "seed": seed, "threshold": best_thr, **summ}
        all_rows.append(row)
        print(f"  TEST dice={summ['mean_dice']:.4f} median={summ['median_dice']:.4f} "
              f"miss={summ['complete_miss_rate']*100:.2f}%")
        del model; torch.cuda.empty_cache()

    if all_rows:
        per_seed = pd.DataFrame(all_rows)
        per_seed.to_csv(res_dir / "segformer_b5_test_per_seed.csv", index=False)
        agg = {"model": "segformer_b5", "n_seeds": len(all_rows),
               "mean_dice": float(per_seed["mean_dice"].mean()),
               "std_dice": float(per_seed["mean_dice"].std()) if len(all_rows) > 1 else 0.0,
               "median_dice": float(per_seed["median_dice"].mean()),
               "mean_iou": float(per_seed["mean_iou"].mean()),
               "complete_miss_rate": float(per_seed["complete_miss_rate"].mean()),
               "complete_miss_count": int(round(per_seed["complete_miss_count"].mean())),
               "n_images": int(per_seed["n_images"].iloc[0]),
               "thresholds": per_seed["threshold"].tolist(), "seeds": a.seeds}
        (res_dir / "segformer_b5_FINAL.json").write_text(json.dumps(agg, indent=2))
        print("\n" + "=" * 72 + f"\nFINAL segformer_b5 (mean over {len(all_rows)} seed[s])\n" + "=" * 72)
        print(f"  mean_dice   {agg['mean_dice']:.4f}")
        print(f"  median_dice {agg['median_dice']:.4f}")
        print(f"  mean_iou    {agg['mean_iou']:.4f}")
        print(f"  miss_rate   {agg['complete_miss_rate']*100:.2f}%")
        print("\noutputs ->", res_dir)


if __name__ == "__main__":
    main()
