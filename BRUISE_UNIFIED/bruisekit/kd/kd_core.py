#!/usr/bin/env python3
"""
kd_core.py — shared machinery for the SegFormer distillation suite.
====================================================================
Everything the distillation trainers reuse, lifted VERBATIM (where possible)
from `train_segformer_b5_baseline.py` so the recipe is byte-for-byte the same
as the baselines it is compared against:

  * manifest loading + canonical 697/134 split
  * ImageNet-norm data loader (identical augs)
  * SegformerWrapper + param-group LR split
  * DiceBCE loss, per-image metrics, complete-miss
  * seed / poly-schedule / VRAM probe / threshold sweep / atomic save
  * val-based threshold selection, test scoring

PLUS the new distillation pieces:

  * teacher loading (frozen SegFormer, temperature-scaled soft probs)
  * temperature calibration (L-BFGS on val logits — port of scripts/02)
  * KD losses:   response (calibrated BCE), CWD (channel-wise), DKD (decoupled)
  * teacher fusion:  uniform ensemble, confidence-adaptive per-pixel gate
  * group-worst loss + hard-example / miss-targeted per-pixel weighting
  * fairness-by-ITA-group evaluation (join test_per_image with the ITA CSV)

Nothing here imports the main `pipeline/` package — this file is self-contained
so the SegFormer half of the suite runs anywhere the B5 baseline package runs.
"""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================================================================================
# §0 · Config defaults (verbatim from the B5 baseline / configs/common_train.yaml)
# ======================================================================================
DEFAULTS = dict(
    img_size=640, epochs=100, patience=15, workers=0, amp=True,
    backbone_lr=6e-5, head_lr=6e-4, betas=(0.9, 0.999), weight_decay=0.01,
    warmup_fraction=0.01, poly_power=1.0, gradient_clip=1.0,
    effective_batch=8, max_probe_batch=32, vram_target=0.75,
    val_fraction=0.18, split_seed=42,
    thresholds=[round(0.05 * k, 2) for k in range(1, 20)],
)

ITA_NAMES = {0: "Light (II-III)", 1: "Intermediate (III-IV)", 2: "Tan (IV)",
             3: "Brown (V)", 4: "Dark (VI)"}

# ======================================================================================
# §1 · Manifest loading
# ======================================================================================
_IMAGE_KEYS = ["image_path", "img_path", "image", "wl_image_path", "image_file", "filepath", "path", "rgb_path"]
_MASK_KEYS = ["mask_path", "mask", "label_path", "gt_path", "annotation_path", "mask_file", "seg_path"]
_SUBJECT_KEYS = ["subject", "subject_id", "patient", "patient_id", "case_id", "case", "person_id"]
_STEM_KEYS = ["stem", "id", "name", "image_id", "sample_id"]
_SPLIT_KEYS = ["split", "subset", "partition"]


def _pick(df, keys, required, what):
    lower = {c.lower(): c for c in df.columns}
    for k in keys:
        if k in lower:
            return lower[k]
    if required:
        raise KeyError(f"Could not find a {what} column. Looked for {keys}; has {list(df.columns)}.")
    return None


def load_manifest(csv_path, data_root):
    df = pd.read_csv(csv_path)
    img_c = _pick(df, _IMAGE_KEYS, True, "image path")
    mask_c = _pick(df, _MASK_KEYS, True, "mask path")
    subj_c = _pick(df, _SUBJECT_KEYS, False, "subject")
    stem_c = _pick(df, _STEM_KEYS, False, "stem")
    split_c = _pick(df, _SPLIT_KEYS, False, "split")
    root = Path(data_root) if data_root else None

    def _abs(p):
        p = Path(str(p))
        return str(p) if (p.is_absolute() or root is None) else str(root / p)

    out = pd.DataFrame()
    out["image_path"] = df[img_c].apply(_abs)
    out["mask_path"] = df[mask_c].apply(_abs)
    out["stem"] = df[stem_c].astype(str) if stem_c else out["image_path"].apply(lambda p: Path(p).stem)
    out["subject"] = df[subj_c].astype(str) if subj_c else out["stem"]
    if split_c:
        out["split"] = df[split_c].astype(str)
    return out.drop_duplicates(subset=["image_path"]).reset_index(drop=True)


def subject_val_split(train_df, val_fraction, seed):
    subjects = sorted(train_df["subject"].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    n_val = max(1, int(round(len(subjects) * val_fraction)))
    val_subjects = set(subjects[:n_val])
    val = train_df[train_df["subject"].isin(val_subjects)].reset_index(drop=True)
    trn = train_df[~train_df["subject"].isin(val_subjects)].reset_index(drop=True)
    assert not (set(trn["subject"]) & set(val["subject"])), "subject leak train/val"
    return trn, val


def resolve_train_val(full_train, cfg):
    if "split" in full_train.columns and full_train["split"].isin(["train", "val"]).any():
        trn = full_train[full_train["split"] == "train"].reset_index(drop=True)
        val = full_train[full_train["split"] == "val"].reset_index(drop=True)
        print(f"  using CANONICAL split column: {len(trn)} train / {len(val)} val")
        return trn, val
    print(f"  no split column; subject-grouped fallback (seed={cfg['split_seed']})")
    return subject_val_split(full_train, cfg["val_fraction"], cfg["split_seed"])


# ======================================================================================
# §2 · Data (verbatim ImageNet-norm loader)
# ======================================================================================
def get_augmentation(training, img_size):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    imagenet_norm = A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    resize = [A.Resize(height=img_size, width=img_size)]
    if not training:
        return A.Compose(resize + [imagenet_norm, ToTensorV2()])
    return A.Compose(resize + [
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.20, contrast_limit=0.20, p=0.4),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
        A.GaussNoise(p=0.2), imagenet_norm, ToTensorV2(),
    ])


class BruiseDataset(torch.utils.data.Dataset):
    """Returns (x[3,H,W] ImageNet-normalised, y[1,H,W]{0,1}, stem)."""

    def __init__(self, df, img_size, training=False):
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
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = (mask > 0).astype(np.float32)
        aug = self.tfm(image=img, mask=mask)
        x = aug["image"].float()
        y = aug["mask"].unsqueeze(0).float()
        return x, y, str(r.stem)


def make_loader(df, img_size, batch_size, training, workers, seed=0):
    ds = BruiseDataset(df, img_size, training=training)
    gen = torch.Generator(); gen.manual_seed(seed)

    def _init_worker(worker_id):
        np.random.seed(seed * 1000 + worker_id)

    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=training, drop_last=training,
        num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
        worker_init_fn=_init_worker, generator=gen)


# ======================================================================================
# §3 · Model (verbatim SegformerWrapper)
# ======================================================================================
def build_segformer(pretrained, num_labels=1):
    from transformers import SegformerForSemanticSegmentation
    return SegformerForSemanticSegmentation.from_pretrained(
        pretrained, num_labels=num_labels, ignore_mismatched_sizes=True)


class SegformerWrapper(nn.Module):
    """Wraps HF SegformerForSemanticSegmentation; upsamples logits to input res.
    Optionally returns encoder hidden states (for feature/CWD distillation)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    @property
    def backbone(self):
        return self.model.segformer

    def forward(self, x, return_hidden=False):
        out = self.model(pixel_values=x, output_hidden_states=return_hidden)
        logits = out.logits
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        if return_hidden:
            return logits, out.hidden_states
        return logits

    def gradient_checkpointing_enable(self):
        try:
            self.model.segformer.encoder.gradient_checkpointing = True
        except AttributeError:
            pass


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def build_param_groups(model, backbone_lr, head_lr, weight_decay):
    backbone_ids = {id(p) for p in model.backbone.parameters()}
    groups = {
        "backbone_decay": {"params": [], "lr": backbone_lr, "weight_decay": weight_decay},
        "backbone_no_decay": {"params": [], "lr": backbone_lr, "weight_decay": 0.0},
        "head_decay": {"params": [], "lr": head_lr, "weight_decay": weight_decay},
        "head_no_decay": {"params": [], "lr": head_lr, "weight_decay": 0.0},
    }
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        where = "backbone" if id(p) in backbone_ids else "head"
        decay = "_no_decay" if (p.ndim <= 1 or "norm" in name.lower() or "bias" in name.lower()) else "_decay"
        groups[where + decay]["params"].append(p)
    out = [g for g in groups.values() if g["params"]]
    return out


# ======================================================================================
# §4 · Supervised loss + metrics (verbatim)
# ======================================================================================
class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1.0):
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
# §5 · Distillation losses & teacher fusion
# ======================================================================================
def _to_prob(logits, T):
    return torch.sigmoid(logits.float() / T)


class ResponseKD(nn.Module):
    """Calibrated response KD: alpha*DiceBCE(student,GT) + (1-alpha)*BCE(student,teacher_prob).
    Optional per-pixel weight (hard-example / miss-targeted). teacher_prob is a
    soft target in (0,1); when a per-pixel weight is given, BOTH terms use it."""

    def __init__(self, alpha=0.75):
        super().__init__(); self.alpha = alpha
        self.sup = DiceBCELoss()

    def forward(self, s_logits, gt, teacher_prob, weight=None):
        sup = self.sup(s_logits, gt)
        if weight is None:
            soft = F.binary_cross_entropy_with_logits(s_logits, teacher_prob)
        else:
            soft = F.binary_cross_entropy_with_logits(s_logits, teacher_prob, reduction="none")
            soft = (weight * soft).mean()
        return self.alpha * sup + (1.0 - self.alpha) * soft


def cwd_loss(student_feats, teacher_feats, adapters, T=4.0):
    """Channel-Wise Distillation (Shu et al., ICCV'21) on encoder stages.
    Each channel's spatial map is a softmax distribution; match student->teacher
    per-channel via KL. Scale-invariant, so the 27M/82M teacher vs 3.7M student
    channel-width gap does not turn into a trivial magnitude match. `adapters`
    project student channels to teacher width (1x1 conv per stage)."""
    total = 0.0
    for fs, ft, adapt in zip(student_feats, teacher_feats, adapters):
        fs = adapt(fs)                                   # [B,C,H,W] -> teacher C
        if fs.shape[-2:] != ft.shape[-2:]:
            fs = F.interpolate(fs, size=ft.shape[-2:], mode="bilinear", align_corners=False)
        B, C, H, W = ft.shape
        ps = F.log_softmax(fs.reshape(B, C, -1) / T, dim=-1)
        pt = F.softmax(ft.reshape(B, C, -1).detach() / T, dim=-1)
        total = total + F.kl_div(ps, pt, reduction="batchmean") * (T * T) / C
    return total / max(1, len(adapters))


def dkd_loss(s_logits, t_logits, gt, T=4.0, beta=2.0):
    """Decoupled KD (Zhao et al., CVPR'22), binary-segmentation form.
    Splits the soft transfer into Target-Class KD (foreground vs background
    agreement) and Non-target KD. For a 2-class (fg/bg) pixel problem we build a
    2-logit distribution per pixel [bg, fg] = [0, z]. TCKD matches the fg/bg mass;
    NCKD matches the within-remaining structure. gt selects the target channel."""
    B, _, H, W = s_logits.shape
    npix = B * H * W                                               # normalise per pixel
    zs = torch.cat([torch.zeros_like(s_logits), s_logits], dim=1)   # [B,2,H,W]
    zt = torch.cat([torch.zeros_like(t_logits), t_logits], dim=1).detach()
    ps = F.softmax(zs / T, dim=1)
    pt = F.softmax(zt / T, dim=1)
    tgt = (gt > 0.5).long().squeeze(1)                             # [B,H,W] target channel idx
    onehot = F.one_hot(tgt, 2).permute(0, 3, 1, 2).float()         # [B,2,H,W]
    # TCKD: binary mass on target vs non-target (sum KL over pixels / npix)
    ps_t = (ps * onehot).sum(1, keepdim=True)
    pt_t = (pt * onehot).sum(1, keepdim=True)
    ps_b = torch.cat([ps_t, 1 - ps_t], 1).clamp_min(1e-6)
    pt_b = torch.cat([pt_t, 1 - pt_t], 1).clamp_min(1e-6)
    tckd = F.kl_div(ps_b.log(), pt_b, reduction="sum") / npix * (T * T)
    # NCKD: distribution over NON-target channels (near-zero in the binary case,
    # kept for formula completeness). Mask the target channel out of the softmax.
    ps_n = F.log_softmax(zs / T - 1e4 * onehot, dim=1)
    pt_n = F.softmax(zt / T - 1e4 * onehot, dim=1)
    nckd = F.kl_div(ps_n, pt_n, reduction="sum") / npix * (T * T)
    return tckd + beta * nckd


def fuse_teachers(prob_a, prob_b, mode, rel_b=None):
    """Combine two temperature-calibrated teacher probability maps.

      uniform  : 0.5*(p_a+p_b)
      adaptive : per-pixel RELIABILITY gate. The base gate is the confidence
                 disagreement w0 = |p_b-.5| / (|p_a-.5|+|p_b-.5|) — it leans on the
                 more-confident teacher at each pixel, which also captures B2/B5
                 disagreement (agreement -> either teacher fine; disagreement ->
                 follow the confident one). Optionally biased by a scalar
                 `rel_b` in [0,1], the VALIDATION-derived reliability of teacher B
                 relative to A (from val_oracle.py / reliability fit): the effective
                 weight is w = w0 * rel_b / (w0*rel_b + (1-w0)*(1-rel_b)).

    Leak-free by construction: uses teacher OUTPUTS and a VALIDATION-estimated
    scalar only — never labels, ITA, or the test set. (Point 4/6 of the review:
    the gate is developed on val, not test.)"""
    if prob_b is None:
        return prob_a
    if mode == "uniform":
        return 0.5 * (prob_a + prob_b)
    if mode == "adaptive":
        ca = (prob_a - 0.5).abs()
        cb = (prob_b - 0.5).abs()
        w0 = cb / (ca + cb + 1e-6)                      # base weight on teacher B
        if rel_b is None:
            w = w0
        else:
            rb = float(rel_b)
            w = (w0 * rb) / (w0 * rb + (1.0 - w0) * (1.0 - rb) + 1e-8)
        return w * prob_b + (1.0 - w) * prob_a
    raise ValueError(f"unknown fusion mode {mode}")


def hard_example_weight(teacher_prob, gt, gamma=2.0, small_thr=0.02):
    """Per-pixel weight upweighting the failure modes the project cares about:
      - complete-miss risk: teacher confident foreground (p high) -> weight up
      - small bruises: images with tiny GT area get a global boost
      - boundary/uncertain region: teacher prob near 0.5 -> weight up
    Returns weight in ~[1, 1+gamma]. Estimated only from teacher + GT (train)."""
    conf_fg = teacher_prob                                    # emphasise foreground mass
    uncertain = 1.0 - (2.0 * (teacher_prob - 0.5).abs())      # peaks at p=0.5
    per_pixel = 1.0 + gamma * (0.5 * conf_fg + 0.5 * uncertain)
    # small-bruise global multiplier (per image)
    area = gt.mean(dim=(1, 2, 3), keepdim=True)               # fraction fg
    small_boost = 1.0 + (area < small_thr).float()            # 2x for small bruises
    return per_pixel * small_boost


def group_worst_loss(s_logits, gt, group_idx, n_groups=5):
    """Worst-group DiceBCE: compute supervised loss per ITA group in the batch,
    return the MAX (worst) group loss so optimisation pushes the worst group.
    group_idx: LongTensor [B] with values in [0,n_groups). Falls back to plain
    mean when a batch has a single group."""
    sup = DiceBCELoss()
    losses = []
    for g in range(n_groups):
        m = (group_idx == g)
        if m.any():
            losses.append(sup(s_logits[m], gt[m]))
    if not losses:
        return sup(s_logits, gt)
    return torch.stack(losses).max()


# --- boundary machinery (shared by BPKD and the boundary-weighted KD term) -------------
def edge_band(gt, k=5):
    """Boundary band of the GT mask via morphological gradient (GPU, no scipy).
    A pixel is on the band if a k-neighbourhood contains BOTH fg and bg — i.e.
    maxpool(gt) != minpool(gt). Returns a float mask in {0,1}, shape like gt."""
    pad = k // 2
    mx = F.max_pool2d(gt, k, 1, pad)
    mn = -F.max_pool2d(-gt, k, 1, pad)
    return (mx != mn).float()


def boundary_weight_map(gt, k=5, beta=4.0):
    """Per-pixel weight that peaks on the GT boundary band (for boundary-weighted
    KD). w = 1 + beta * edge_band. Reuses the same edge definition as BPKD so the
    two are consistent."""
    return 1.0 + beta * edge_band(gt, k)


def bpkd_loss(s_logits, teacher_prob, gt, k=5, lam_edge=4.0, lam_body=1.0):
    """Boundary-Privileged KD (Liu et al., WACV'24), binary-segmentation form.
    Separates EDGE knowledge from BODY knowledge instead of weighting every pixel
    equally: distil the teacher's soft prob on the boundary band and on the interior
    with SEPARATE weights (edge emphasised). Bruise boundaries are weak/irregular,
    so edge transfer is where a compact student most needs the teacher."""
    band = edge_band(gt, k)                                   # [B,1,H,W] in {0,1}
    bce = F.binary_cross_entropy_with_logits(s_logits, teacher_prob, reduction="none")
    edge = (bce * band).sum() / band.sum().clamp_min(1.0)
    body = (bce * (1.0 - band)).sum() / (1.0 - band).sum().clamp_min(1.0)
    return lam_edge * edge + lam_body * body


def angular_distillation(student_feats, teacher_feats, adapters):
    """Angular (direction-based) feature distillation (Liu et al., WACV'24 —
    'Rethinking KD with Raw Features'). Matches the DIRECTION of per-pixel channel
    vectors (cosine), not their magnitude, so the large B5->B0 magnitude gap does
    not turn into a trivial scale fit. Cheap and stable. `adapters` map student
    channels to teacher width per stage."""
    total = 0.0
    for fs, ft, adapt in zip(student_feats, teacher_feats, adapters):
        fs = adapt(fs)
        if fs.shape[-2:] != ft.shape[-2:]:
            fs = F.interpolate(fs, size=ft.shape[-2:], mode="bilinear", align_corners=False)
        fsn = F.normalize(fs, dim=1)                          # per-pixel channel direction
        ftn = F.normalize(ft.detach(), dim=1)
        cos = (fsn * ftn).sum(dim=1)                          # [B,H,W]
        total = total + (1.0 - cos).mean()
    return total / max(1, len(adapters))


# ======================================================================================
# §6 · Teacher loading + temperature calibration
# ======================================================================================
def load_temperature(teacher_dir):
    f = Path(teacher_dir) / "temperature.json"
    if not f.exists():
        print(f"  [warn] no temperature.json in {teacher_dir} — using T=1.0")
        return 1.0
    return float(json.loads(f.read_text()).get("temperature", 1.0))


def load_teacher(teacher_dir, pretrained, device, amp=True):
    """Frozen SegFormer teacher -> callable returning temperature-scaled soft probs.
    Also exposes .raw_logits(x) for DKD (needs logits, not probs)."""
    teacher_dir = Path(teacher_dir)
    best = teacher_dir / "best_model.pt"
    if not best.exists():
        raise FileNotFoundError(f"teacher weights not found: {best}")
    model = SegformerWrapper(build_segformer(pretrained, num_labels=1)).to(device)
    model.load_state_dict(torch.load(str(best), map_location=device, weights_only=True))
    model.eval()
    T = load_temperature(teacher_dir)
    print(f"  teacher loaded from {teacher_dir} (T={T:.3f})")

    def prob_fn(x):
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=amp):
            logits = model(x)
        return _to_prob(logits, T)

    def logit_fn(x):
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=amp):
            logits = model(x)
        return (logits.float() / T)

    def hidden_fn(x):
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=amp):
            logits, hs = model(x, return_hidden=True)
        return [h.float() for h in hs]

    prob_fn.raw_logits = logit_fn
    prob_fn.hidden = hidden_fn
    prob_fn.temperature = T
    prob_fn._model = model
    return prob_fn


@torch.no_grad()
def calibrate_temperature(model, val_loader, device, amp=True):
    """L-BFGS temperature fit on val logits (port of scripts/02_calibrate_teacher)."""
    model.eval()
    all_logits, all_targets = [], []
    for x, y, _ in val_loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp):
            all_logits.append(model(x).float().cpu())
        all_targets.append(y.cpu())
    logits = torch.cat(all_logits); targets = torch.cat(all_targets)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.05, max_iter=100)

    def closure():
        opt.zero_grad()
        t = torch.exp(log_t)
        loss = F.binary_cross_entropy_with_logits(logits / t, targets)
        loss.backward()
        return loss

    with torch.enable_grad():
        opt.step(closure)
    T = float(torch.exp(log_t).item())
    nll_before = float(F.binary_cross_entropy_with_logits(logits, targets))
    nll_after = float(F.binary_cross_entropy_with_logits(logits / T, targets))
    return T, nll_before, nll_after


# ======================================================================================
# §7 · Engine (verbatim: seed, schedule, probe, eval, sweep, atomic save)
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


def find_optimal_micro_batch(fwd_fn, params, img_size, device, effective_batch,
                             target_hi, amp, max_probe, teacher_fns=()):
    """Probe micro-batch with a real fwd+bwd+step of the STUDENT plus the teacher
    forward(s) (so the ensemble's two teacher passes are counted in the VRAM budget)."""
    if not torch.cuda.is_available():
        mb = min(effective_batch, max_probe)
        return mb, max(1, effective_batch // mb), 0.0
    total_vram = torch.cuda.get_device_properties(device).total_memory
    scaler = torch.amp.GradScaler("cuda") if amp else None
    optimizer = torch.optim.SGD(params, lr=1e-6)
    chosen, chosen_frac, batch = 1, 0.0, 1
    while batch <= max_probe:
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
            x = torch.randn(batch, 3, img_size, img_size, device=device)
            y = torch.randint(0, 2, (batch, 1, img_size, img_size), device=device).float()
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=amp):
                for tf in teacher_fns:
                    _ = tf(x)
                logits = fwd_fn(x)
                loss = F.binary_cross_entropy_with_logits(logits, y)
            if scaler is not None:
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); optimizer.step()
            frac = torch.cuda.max_memory_reserved(device) / total_vram
            del x, y, logits, loss; torch.cuda.empty_cache()
            if frac > target_hi:
                break
            chosen, chosen_frac = batch, frac
            batch *= 2
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); break
    optimizer.zero_grad(set_to_none=True)
    mb = max(1, chosen)
    return mb, max(1, effective_batch // mb), chosen_frac


@torch.no_grad()
def evaluate(model, loader, device, threshold, amp):
    model.eval(); rows = []
    for x, y, stems in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp):
            logits = model(x)
        prob = torch.sigmoid(logits.float()).cpu().numpy()
        gt = y.numpy()
        for i, stem in enumerate(stems):
            pred = (prob[i, 0] >= threshold).astype("uint8")
            g = (gt[i, 0] > 0.5).astype("uint8")
            rows.append(compute_image_row(pred, g, str(stem)))
    return pd.DataFrame(rows), summarize(rows)


def threshold_sweep(model, loader, device, thresholds, amp):
    rows = []
    for thr in thresholds:
        _, s = evaluate(model, loader, device, thr, amp)
        rows.append({"threshold": thr, **s})
    df = pd.DataFrame(rows).sort_values("mean_dice", ascending=False)
    return df, float(df.iloc[0]["threshold"])


def atomic_save(obj, dest):
    dest = Path(dest)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    torch.save(obj, tmp); os.replace(tmp, dest)


# ======================================================================================
# §8 · Fairness-by-ITA-group evaluation
# ======================================================================================
def fairness_by_group(per_image_df, ita_csv, model_name, out_csv=None, n_boot=2000, seed=0):
    """Join per-image results with the ITA CSV (stem -> ita_group_index_5) and
    summarise Dice/recall/miss per skin-tone group, with a subject-cluster
    bootstrap CI on each group's median Dice. Returns (per_group_df, gap_dict)."""
    ita = pd.read_csv(ita_csv)
    keep = ["stem", "subject", "ita_group_index_5", "skin_tone_category"]
    ita = ita[[c for c in keep if c in ita.columns]].copy()
    df = per_image_df.merge(ita, on="stem", how="inner")
    df["miss"] = ((df["pred_positive_pixels"] == 0) & (df["gt_positive_pixels"] > 0)).astype(float)
    rng = np.random.default_rng(seed)
    rows = []
    for g, sub in df.groupby("ita_group_index_5"):
        subs = sub["subject"].to_numpy() if "subject" in sub else np.arange(len(sub))
        uniq = np.unique(subs)
        idx_by = {s: np.where(subs == s)[0] for s in uniq}
        dice = sub["dice"].to_numpy()
        boot = np.empty(n_boot)
        for b in range(n_boot):
            drawn = rng.choice(uniq, size=uniq.size, replace=True)
            ix = np.concatenate([idx_by[s] for s in drawn])
            boot[b] = np.median(dice[ix])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append({"model": model_name, "ita_group_index_5": int(g),
                     "skin_tone_category": ITA_NAMES.get(int(g), str(g)),
                     "n_images": int(len(sub)), "median_dice": float(np.median(dice)),
                     "ci95_lo": float(lo), "ci95_hi": float(hi),
                     "mean_recall": float(sub["recall"].mean()),
                     "miss_rate": float(sub["miss"].mean())})
    pg = pd.DataFrame(rows).sort_values("ita_group_index_5").reset_index(drop=True)
    if out_csv:
        pg.to_csv(out_csv, index=False)
    best = pg.loc[pg["median_dice"].idxmax()]
    worst = pg.loc[pg["median_dice"].idxmin()]
    gap = {"fairness_gap_median_dice": float(best["median_dice"] - worst["median_dice"]),
           "best_group": best["skin_tone_category"], "worst_group": worst["skin_tone_category"],
           "worst_group_median_dice": float(worst["median_dice"]),
           "worst_group_recall": float(worst["mean_recall"]),
           "max_miss_rate": float(pg["miss_rate"].max())}
    return pg, gap


# ======================================================================================
# §9 · Failure-tail metrics + TTA (review points 3 and 8)
# ======================================================================================
def tail_metrics(per_image_df, subject_col="subject"):
    """Low-recall / failure-tail resolution beyond exact complete-miss (review §3).
    Complete-miss saturates at 0% for all three strong SegFormers, so we add:
      - pct_recall_below_0.10  : fraction of images the model barely finds
      - pct_dice_below_0.20    : fraction of near-total failures
      - p5_subject_dice        : 5th-percentile of per-SUBJECT mean Dice (tail floor)
      - complete_miss_rate     : kept for continuity
    per_image_df needs columns dice, recall, pred/gt_positive_pixels (+subject for p5)."""
    d = per_image_df
    miss = ((d["pred_positive_pixels"] == 0) & (d["gt_positive_pixels"] > 0))
    out = {
        "n_images": int(len(d)),
        "pct_recall_below_0.10": float((d["recall"] < 0.10).mean()),
        "pct_dice_below_0.20": float((d["dice"] < 0.20).mean()),
        "complete_miss_rate": float(miss.mean()),
    }
    if subject_col in d.columns:
        subj_dice = d.groupby(subject_col)["dice"].mean().to_numpy()
        out["p5_subject_dice"] = float(np.percentile(subj_dice, 5))
        out["n_subjects"] = int(len(subj_dice))
    return out


@torch.no_grad()
def tta_flip_prob(model, x, amp=True):
    """Flip-TTA probability map: mean of sigmoid over {identity, hflip, vflip, hvflip}.
    Inference-only (review §8) — reported separately with its forward-pass count (4),
    never folded into a training/distillation result."""
    def _p(inp):
        with torch.amp.autocast("cuda", enabled=amp and torch.cuda.is_available()):
            return torch.sigmoid(model(inp).float())
    p = _p(x)
    p = p + _p(torch.flip(x, dims=[3])).flip(dims=[3])
    p = p + _p(torch.flip(x, dims=[2])).flip(dims=[2])
    p = p + _p(torch.flip(x, dims=[2, 3])).flip(dims=[2, 3])
    return p / 4.0
