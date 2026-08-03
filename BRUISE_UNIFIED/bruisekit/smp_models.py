"""U-Net / DeepLabV3+ (segmentation_models_pytorch) behind bruisekit's interface.

WHY THESE GO THROUGH THE SHARED LOOP
------------------------------------
An SMP model is a pretrained ImageNet encoder + a randomly-initialised decoder/head
-- structurally identical to SegFormer (pretrained backbone + random 1-class head).
So they take the reference recipe VERBATIM: the loader emits raw [0,1] pixels and the
MODEL applies ImageNet normalisation (exactly like SegFormerNet), a 1-class head, the
encoder/head LR split, Dice+BCE, poly schedule, threshold-free AP selection, and the
val-swept threshold applied once to test. Holding the recipe fixed is the whole point
of a fair baseline.

THE build_model SHIM
--------------------
`engine.train_run` is architecture-blind: it calls `build_model(arch, size, paths)`
and never branches on a model's name. We only need to teach that one function to build
SMP architectures. Because `engine` did `from bruisekit.models import build_model` at
import time (binding the original by value), we reassign the name in BOTH namespaces
-- `bruisekit.models` (for eval cells that import it fresh) and `bruisekit.engine`
(the copy the training loop actually calls). SegFormer/YOLO still route to the
original builder untouched.
"""
from __future__ import annotations

import torch
import torch.nn as nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SMP_ARCHS = {"unet", "deeplabv3plus", "deeplabv3", "unetplusplus", "fpn", "manet"}


class SMPNet(nn.Module):
    """segmentation_models_pytorch model with a 1-class head. Input scale: ImageNet.

        forward_train(x) -> (logits[B,1,H,W], None)   -- x is RAW [0,1]; norm applied here
        forward(x)       -> logits[B,1,H,W]
        .backbone        -> the pretrained encoder (for the encoder/head LR split)
    """

    def __init__(self, arch: str, encoder: str = "resnet50", encoder_weights: str | None = "imagenet"):
        super().__init__()
        import segmentation_models_pytorch as smp
        builders = {
            "unet": smp.Unet, "deeplabv3plus": smp.DeepLabV3Plus, "deeplabv3": smp.DeepLabV3,
            "unetplusplus": smp.UnetPlusPlus, "fpn": smp.FPN, "manet": smp.MAnet,
        }
        if arch not in builders:
            raise ValueError(f"unknown smp arch: {arch}. choices: {list(builders)}")
        self.net = builders[arch](encoder_name=encoder, encoder_weights=encoder_weights,
                                  in_channels=3, classes=1, activation=None)
        # Buffers (not constants) so they move with .to(device) and save with the module.
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    @property
    def backbone(self):
        return self.net.encoder      # decoder + segmentation_head fall into the "head" LR group

    @property
    def head(self):
        return self.net.decoder

    def forward_train(self, x):
        x = (x - self.mean) / self.std               # [0,1] -> ImageNet
        return self.net(x), None                     # smp upsamples to input res -> [B,1,H,W]

    def forward(self, x):
        return self.forward_train(x)[0]


def install_build_model_shim(smp_micro_batch: int = 16):
    """Route SMP archs to SMPNet, and make batch selection safe for DeepLabV3+.

    Two patches into bruisekit.engine's namespace (train_run looks both names up as
    module globals at call time, so reassigning the module attribute takes effect):

    1. build_model -> builds SMP architectures for the SMP arch names.
    2. resolve_micro_batch -> for SMPNet models, SKIP the VRAM probe and return a
       FIXED batch. The probe escalates from batch=1 in TRAIN mode; DeepLabV3+'s ASPP
       image-pooling branch produces a [B, C, 1, 1] tensor whose BatchNorm raises
       "Expected more than 1 value per channel" at B=1. A fixed batch also means every
       SMP baseline (and every seed) trains at the SAME batch, which is cleaner for the
       baseline comparison than per-model probed batches. Other archs are untouched.
    """
    import bruisekit.models as _bm
    import bruisekit.engine as _be
    original = _bm.build_model

    def build_model(arch, size, paths):
        if arch in SMP_ARCHS:
            return SMPNet(arch, encoder=paths.get("smp_encoder", "resnet50"))
        return original(arch, size, paths)

    _bm.build_model = build_model
    _be.build_model = build_model     # the name the training loop already bound

    original_probe = _be.resolve_micro_batch

    def resolve_micro_batch(model, cfg, device, teacher=None):
        if isinstance(model, SMPNet):
            b = max(2, int(cfg.get("smp_micro_batch", smp_micro_batch)))
            return b, 1               # (micro_batch, accum_steps); no probe, no batch-1 forward
        return original_probe(model, cfg, device, teacher)

    _be.resolve_micro_batch = resolve_micro_batch
    return build_model