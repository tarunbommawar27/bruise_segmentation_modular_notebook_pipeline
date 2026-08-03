"""The two architectures behind one interface.

THE INTERFACE
--------------
    forward_train(x) -> (logits[B,1,H,W], aux_logits[B,1,H,W] | None)
    forward(x)       -> logits[B,1,H,W]

x is RAW [0,1] pixels (see data.py). Each wrapper applies its own pixel scale.
Both models emit ONE bruise logit at full input resolution, so every downstream
consumer -- loss, sweep, metric, benchmark -- is architecture-blind. Nothing
below this line ever branches on a model's name.

WHY YOLO IS BUILT WITH nc=1 (not nc=2)
---------------------------------------
The pretrained yolo26n-sem.pt has nc=19 (Cityscapes). Rebuilding the head with
nc=2 gives [B,2,H,W], from which the bruise logit is z1-z0 (2-class softmax and
sigmoid-of-the-difference are the same function). Rebuilding with nc=1 gives that
single logit directly, and Ultralytics' own loss supports it (nn.BCEWithLogitsLoss
for nc==1). One less transformation, one less place to get the sign wrong, and
structurally identical to SegFormer's 1-channel head. `.load()` transfers 360/364
tensors -- only the head is new, exactly mirroring SegFormer's randomly-initialised
1-class head on a pretrained backbone.

WHY THE HEAD GETS 10x THE BACKBONE'S LR (both architectures)
-------------------------------------------------------------
The backbone is pretrained and already has good features; a conservative LR
preserves them. The head is randomly initialised and has to catch up. This is the
SegFormer paper's recipe (Xie et al. 2021) and it is applied to YOLO here too --
not because YOLO's paper says so, but because holding the recipe fixed across
architectures is the entire point of an apple-to-apple comparison.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SegFormerNet(nn.Module):
    """HuggingFace SegFormer with a 1-class head. Input scale: ImageNet."""

    def __init__(self, pretrained_dir: str):
        super().__init__()
        from transformers import SegformerForSemanticSegmentation
        self.net = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_dir, num_labels=1, ignore_mismatched_sizes=True)
        # Buffers (not constants) so they move with .to(device) and are saved with
        # the module -- the pixel scale travels with the weights it belongs to.
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    @property
    def backbone(self):
        return self.net.segformer

    @property
    def head(self):
        return self.net.decode_head

    def forward_train(self, x):
        x = (x - self.mean) / self.std                      # [0,1] -> ImageNet
        logits = self.net(pixel_values=x).logits            # [B,1,H/4,W/4]
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits, None                                 # SegFormer has no aux head

    def forward(self, x):
        return self.forward_train(x)[0]


class YoloSemNet(nn.Module):
    """Ultralytics YOLO26n-sem with an nc=1 head. Input scale: /255 (i.e. x as-is).

    In train() the underlying module returns (main[B,1,H/8,W/8], aux[B,1,H/16,W/16]);
    in eval() it returns only the main tensor. Both are upsampled to input
    resolution here so callers never see the stride.

    NOTE the stride: YOLO's main head is at H/8 (80x80 for a 640 input) against
    SegFormer's H/4 (160x160). YOLO predicts at a quarter of SegFormer's spatial
    resolution in area, which is an architectural ceiling on boundary precision,
    not something training can fix. Worth stating in the paper.
    """

    def __init__(self, pretrained_pt: str):
        super().__init__()
        from ultralytics import YOLO
        from ultralytics.nn.tasks import SemanticSegmentationModel

        pretrained = YOLO(str(pretrained_pt))
        self.net = SemanticSegmentationModel("yolo26n-sem.yaml", nc=1, ch=3, verbose=False)
        self.net.load(pretrained.model)     # transfers the backbone; head stays random
        del pretrained

    @property
    def backbone(self):
        return self.net.model[:-1]

    @property
    def head(self):
        return self.net.model[-1]

    def forward_train(self, x):
        out = self.net(x)                                   # x already /255
        if isinstance(out, (list, tuple)):
            main, aux = out[0], out[1]
        else:
            main, aux = out, None
        size = x.shape[-2:]
        main = F.interpolate(main.float(), size=size, mode="bilinear", align_corners=False)
        if aux is not None:
            aux = F.interpolate(aux.float(), size=size, mode="bilinear", align_corners=False)
        return main, aux

    def forward(self, x):
        return self.forward_train(x)[0]


def build_model(arch: str, size: str, paths: dict) -> nn.Module:
    if arch == "segformer":
        return SegFormerNet(paths[f"segformer_{size}"])
    if arch == "yolo":
        return YoloSemNet(paths["yolo"])
    raise ValueError(f"unknown arch: {arch}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_param_groups(model, backbone_lr: float, head_lr: float, weight_decay: float):
    """Backbone/head LR split + no weight decay on norms and biases.

    No decay on 1-D params (norm weights, biases): decaying a normalisation scale
    or a bias shrinks it toward zero with no regularising benefit -- standard in
    every transformer recipe including SegFormer's.

    Membership is by id(), not by name: the two architectures name their
    parameters completely differently, and a name-prefix rule would silently put
    every YOLO parameter in the wrong group.
    """
    backbone_ids = {id(p) for p in model.backbone.parameters()}
    groups = {
        "backbone_decay":    {"params": [], "lr": backbone_lr, "weight_decay": weight_decay},
        "backbone_no_decay": {"params": [], "lr": backbone_lr, "weight_decay": 0.0},
        "head_decay":        {"params": [], "lr": head_lr,     "weight_decay": weight_decay},
        "head_no_decay":     {"params": [], "lr": head_lr,     "weight_decay": 0.0},
    }
    for name, p in model.named_parameters():
        if not p.requires_grad or name in ("mean", "std"):
            continue
        where = "backbone" if id(p) in backbone_ids else "head"
        decay = "_no_decay" if (p.ndim <= 1 or "norm" in name.lower() or "bias" in name.lower()) else "_decay"
        groups[where + decay]["params"].append(p)

    out = [g for g in groups.values() if g["params"]]
    n_grouped = sum(len(g["params"]) for g in out)
    n_total = sum(1 for n, p in model.named_parameters() if p.requires_grad and n not in ("mean", "std"))
    assert n_grouped == n_total, f"param grouping lost {n_total - n_grouped} tensors"
    return out