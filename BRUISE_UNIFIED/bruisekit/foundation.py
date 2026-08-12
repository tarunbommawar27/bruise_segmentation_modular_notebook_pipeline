"""Stage N -- do medical foundation encoders beat ImageNet encoders on bruises?

WHAT THIS STAGE IS, AND WHAT IT DELIBERATELY IS NOT
----------------------------------------------------
It is ONE question: **does the pretraining corpus matter?** Same decoder, same
recipe, same seeds, same split -- only the encoder's pretraining changes.

It is NOT a distillation stage. The proposal this was cut down from wanted a
foundation teacher feeding reliability-gated + multi-teacher + ITA-fairness-aware
KD. Three objections, all from this project's own results:

  1. THERE IS NO TEACHER DEFICIT TO FIX. On the current lineage
     `segformer_b2_teacher`, `segformer_b5_teacher` and `segformer_b0_distilled`
     each have ZERO complete misses on the 185 test images. The pre-registered
     endpoint is already saturated at the top of the field. A better teacher
     cannot improve on zero.
  2. THE BOTTLENECK IS TRANSFER, NOT TEACHERS. Stage M measured an oracle gain of
     +0.048 Dice over the best single teacher and the student captured NONE of
     it, because the router's fused target scored 0.723 -- below every individual
     teacher in the pool. Adding a fifth, larger teacher to a fusion that already
     degrades its inputs changes the wrong variable.
  3. THE FAIRNESS TARGET IS NOT THERE. 20 of the 21 ITA tests run in this study
     are non-significant, and the one that is (`segformer_b2_teacher`, p=0.0105)
     has its WORST group at Tan (IV), not Dark. Where YOLO shows a gap the worst
     group is Light (II-III) and D6 attributes it to a lesion-size confound.
     Stage C already ran ITA-group-weighted KD (`p3_adaptive_group`) and it
     scored 0.7586 against the 0.7680 reference -- one of the worst arms in the
     grid. Re-running that mechanism against a gap the data does not show is not
     a fairness result, it is a third null.

So distillation is out of scope here BY CONSTRUCTION. If the baseline shows
something, one KD arm against a teacher-matched control is the next decision, and
it is a separate one.

THE SCALING PRIOR THIS STAGE IS TESTING AGAINST
------------------------------------------------
    segformer_b0_direct     3.71 M   0.7663
    segformer_b2_teacher   27.35 M   0.7692
    segformer_b5_teacher   ~85   M   0.7727

23x the parameters bought +0.006 Dice, and human annotators disagree with each
other by 0.755 (gbarimah vs erik) to 0.581 (paul vs erik). A 400 M encoder that
lands at ~0.775 has told us nothing we did not already know. That is exactly why
the gate below scores the ENCODER's features rather than the trained model: if
dermatology pretraining is worth anything, it has to show up in the features
before it shows up in a fine-tuned Dice number that the annotation ceiling caps
anyway.

THE PRE-REGISTERED GATE (fixed before any number is looked at)
---------------------------------------------------------------
Freeze each encoder, train ONLY a linear head, score on the 134 validation
images. Then:

    open  iff  (medsiglip - resnet50) val-Dice CI clears zero

Two things are reported ALONGSIDE and are deliberately NOT ANDed in:

  - THE ATTRIBUTION ARM. `dinov2` is a modern self-supervised ViT trained on
    natural images. If MedSigLIP beats ResNet-50 but DINOv2 beats it by the same
    margin, the win is "modern ViT pretraining", NOT "medical pretraining", and
    the paper claim has to change. This is the confound the original proposal
    had no control for.
  - THE MISS ENDPOINT. Misses are what this study is judged on, so a pool that
    contains a miss the others do not is interesting even when Dice says nothing.
    Read both; the printed verdict says which opened.

WHY A FROZEN PROBE AND NOT JUST FINE-TUNING EVERYTHING
--------------------------------------------------------
A fine-tuned 400 M encoder on 697 images from 95 subjects will fit the training
subjects whatever its pretraining was, so a good fine-tuned score is weak
evidence that the pretraining helped. A frozen probe cannot do that: it can only
report what is already in the features. It also costs ~1 GPU-hour against ~20,
which is the whole point of gating (see `multiteacher.oracle_gate`, and Stage C
6.2 before it).

WHERE THE WEIGHTS COME FROM -- READ `SOURCES` AND `download_instructions()`
----------------------------------------------------------------------------
None of these ship with the bundle. MedSigLIP is licence-gated and must be
accepted by a human before it can be fetched; ORC compute nodes are usually
offline. So every encoder is loaded with `local_files_only=True` from
`pretrained_weights/foundation/<dir>/`, and a missing one raises with the exact
command that fixes it rather than silently training from random init -- which on
697 images is a large and completely invisible handicap (`weights.py`, same
reasoning).

THE ONE RECIPE DEVIATION, STATED UP FRONT
-------------------------------------------
A 400 M ViT with gradients at 448x448 does not fit alongside a 40 GB MIG slice at
micro-batch 16. These arms train at micro-batch 4 with accumulation 4, so the
EFFECTIVE batch (16), the optimizer-step count and the LR schedule are identical
to the SMP and mobile baselines -- only the LayerNorm/GroupNorm group size
differs, and both are per-sample normalisers, so unlike the BatchNorm case in
Stage M this changes nothing statistical. Recorded here because handbook 3 says
the shared recipe is the entire basis for comparing these models.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# 1. where the weights come from
# ─────────────────────────────────────────────────────────────────────────────
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# SigLIP-family models are trained on [-1, 1] pixels, not ImageNet statistics.
# This is the same class of mistake as feeding YOLO ImageNet-normalised input,
# which capped it at Dice 0.479 with no threshold able to recover it (data.py).
# Pixel scale belongs to the model, and it is declared per encoder below.
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)


@dataclass
class FoundationSource:
    """One encoder: where to get it, what it is, and what it is licensed as.

    `licence` is not documentation for its own sake. Handbook 7b.1 chose
    DeepLabV3+ over SegFormer as the Stage F teacher specifically to escape
    NVIDIA's non-commercial MiT licence. Re-introducing a restricted encoder
    without recording it would silently undo that.
    """

    key: str
    repo: str
    local_dir: str
    kind: str = "gated"           # "gated" | "open"
    licence: str = ""
    init: str = ""
    note: str = ""
    # Published encoder size, in millions. THE SECOND LOAD CHECK, and the one that
    # catches what a key-count check cannot: a body built from the right config by
    # the WRONG class has plausible dimensions and the wrong parameter total.
    # `efficient_models.self_test` makes the same argument -- a transformer with a
    # wrong head count still returns [B,1,H,W] and still trains, it is simply not
    # the model named in the table.
    expected_params_M: float = 0.0
    aliases: tuple = field(default_factory=tuple)


SOURCES: dict[str, FoundationSource] = {
    "medsiglip": FoundationSource(
        key="medsiglip",
        repo="google/medsiglip-448",
        local_dir="medsiglip-448",
        kind="gated",
        licence="Health AI Developer Foundations terms of use -- NOT a standard OSS "
                "licence. Must be accepted by a human on the model page before the "
                "weights can be downloaded, and it carries use restrictions. VERIFY "
                "the current terms before any commercialisation claim.",
        init="SigLIP vision encoder (~400 M) trained on medical image-text pairs "
             "including dermatology, radiology, histopathology and ophthalmology.",
        note="The repo holds a full vision+text SigLIP. Only the VISION tower is "
             "used here -- there is no text prompt anywhere in this stage, so the "
             "text encoder is never downloaded into memory. Native resolution 448, "
             "patch 14 -- NOT 16, as this note and handbook 7f.9 said until "
             "2026-08-09; the grid at 448 is therefore 32x32, not 28x28. The code "
             "always read patch_size from the config so no run was affected, but "
             "7f.9's confound table understated this arm's grid. Patch size is "
             "read, never assumed, so a 640 input is resized to 448 rather than "
             "having its "
             "position embeddings interpolated.",
        expected_params_M=428.0,      # SoViT-400m vision tower
        aliases=("google/medsiglip-448",),
    ),
    "dinov2": FoundationSource(
        key="dinov2",
        repo="facebook/dinov2-base",
        local_dir="dinov2-base",
        kind="open",
        licence="Apache-2.0.",
        init="DINOv2 ViT-B/14 (~86 M), self-supervised on LVD-142M natural images.",
        note="THE ATTRIBUTION CONTROL, and the most important arm in the gate. It "
             "is a modern self-supervised ViT with NO medical data. If it matches "
             "MedSigLIP, then whatever the probe measures is 'modern ViT features', "
             "not 'medical pretraining', and the headline claim has to change. "
             "Open-licensed and ~350 MB, so there is no reason to skip it.",
        expected_params_M=86.6,       # ViT-B/14
    ),
    "resnet50": FoundationSource(
        key="resnet50",
        repo="torchvision IMAGENET1K_V2",
        local_dir="(torchvision cache)",
        kind="open",
        licence="BSD-3-Clause (torchvision weights).",
        init="ResNet-50, ImageNet-1k supervised.",
        note="THE BASELINE THE GATE IS SCORED AGAINST. Same encoder family as "
             "unet_r50 and deeplabv3plus_r50, so a win here is directly "
             "interpretable against Stage B. Fetched by torchvision, which caches "
             "to ~/.cache/torch -- pre-fetch it on a login node if the compute "
             "node is offline.",
        expected_params_M=23.5,       # conv trunk only; the fc head is dropped
    ),
}

# BiomedParse is NOT in SOURCES: it is a segmentation model, not an encoder, so it
# does not belong in a frozen-feature comparison. It has its own zero-shot path
# (`biomedparse_zeroshot`) because it needs no training at all and answers a
# different, cheaper question -- see that function's docstring.
BIOMEDPARSE_REPO = "microsoft/BiomedParse"


def download_instructions(env=None) -> str:
    """Exactly what to download, from where, and where to put it.

    Returned as text rather than executed, because the gated model needs a human
    to accept a licence and the compute node is usually offline. This is the
    string the notebook prints when an encoder is missing.
    """
    dest = str(Path(env.weights) / "foundation") if env is not None else \
        "<bundle>/pretrained_weights/foundation"
    return f"""\
Download these ON A MACHINE WITH INTERNET (your laptop, or an ORC login node),
then put them under:

    {dest}/

1. MedSigLIP  -- GATED, needs a licence acceptance first
   a) Open https://huggingface.co/{SOURCES['medsiglip'].repo} while signed in and
      accept the Health AI Developer Foundations terms. Without this every
      download returns 401.
   b) huggingface-cli login          (paste a token with 'read' scope)
   c) huggingface-cli download {SOURCES['medsiglip'].repo} \\
        --local-dir {dest}/{SOURCES['medsiglip'].local_dir}
   ~1.6 GB (vision + text towers; only the vision tower is loaded).

2. DINOv2  -- open, no login, the attribution control
   huggingface-cli download {SOURCES['dinov2'].repo} \\
        --local-dir {dest}/{SOURCES['dinov2'].local_dir}
   ~350 MB.

3. ResNet-50  -- torchvision, no manual step IF the node has internet. If it does
   not, pre-fetch on a login node so it lands in the shared torch cache:
     python -c "import torchvision; torchvision.models.resnet50(weights='IMAGENET1K_V2')"
   ~100 MB into ~/.cache/torch/hub/checkpoints/.

Optional, only for the zero-shot cell:
4. BiomedParse
   huggingface-cli download {BIOMEDPARSE_REPO} \\
        --local-dir {dest}/BiomedParse
   Skip it if you only want the gate -- the notebook detects its absence and
   continues.

Then rsync the whole `foundation/` directory to the bundle on ORC. Nothing here
is fetched at run time: every load passes local_files_only=True, so a missing
file raises with this message instead of silently training from random init.
"""


def report_sources(env) -> pd.DataFrame:
    """What is on disk right now. Downloads nothing.

    Mirrors `weights.report`: the provenance table is as much the deliverable as
    the files are, because an encoder that quietly failed to load is the one
    failure mode that still produces a plausible-looking number.
    """
    rows = []
    for k, s in SOURCES.items():
        if k == "resnet50":
            present = True                       # torchvision decides at build time
            where = s.local_dir
        else:
            p = Path(env.weights) / "foundation" / s.local_dir
            present = (p / "config.json").exists()
            where = str(p)
        rows.append({"encoder": k, "repo": s.repo, "kind": s.kind,
                     "present": present, "licence": s.licence.split(".")[0],
                     "init": s.init, "path": where})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 2. the architecture -- one interface, three encoders
# ─────────────────────────────────────────────────────────────────────────────
class LinearProbeHead(nn.Module):
    """1x1 conv on the token grid. THE PROBE HEAD -- nothing else is allowed here.

    A probe must be linear. The moment the head can itself learn spatial structure
    it stops measuring the encoder and starts measuring the head, and a strong
    decoder will paper over a weak encoder well enough to make every arm tie --
    which would read as "no difference between pretraining corpora" when it
    actually means "the experiment could not see one".
    """

    def __init__(self, in_ch: int, num_classes: int = 1):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, num_classes, kernel_size=1)

    def forward(self, feat):
        return self.proj(feat)


class ConvDecodeHead(nn.Module):
    """Small progressive-upsample decoder. THE BASELINE HEAD.

    GroupNorm, not BatchNorm, for two reasons that both bit this project before:
    the engine's VRAM probe escalates from batch 1 and BatchNorm raises "Expected
    more than 1 value per channel" there (handbook 3, and the reason SMP and the
    mobile nets skip the probe); and Stage M had to note a genuine BatchNorm
    group-size deviation when accumulation shrank its micro-batch. GroupNorm is
    per-sample, so neither issue exists and the accumulation deviation documented
    in this module's docstring costs nothing statistically.
    """

    def __init__(self, in_ch: int, num_classes: int = 1, widths=(256, 128, 64)):
        super().__init__()
        blocks, c = [], in_ch
        for w in widths:
            blocks.append(nn.Sequential(
                nn.Conv2d(c, w, 3, padding=1, bias=False),
                nn.GroupNorm(min(32, w), w),
                nn.GELU(),
            ))
            c = w
        self.blocks = nn.ModuleList(blocks)
        self.classifier = nn.Conv2d(c, num_classes, 1)

    def forward(self, feat):
        for b in self.blocks:
            feat = F.interpolate(b(feat), scale_factor=2.0, mode="bilinear",
                                 align_corners=False)
        return self.classifier(feat)


def _transformer_blocks(encoder) -> list:
    """The encoder's transformer layers, whatever the vendor called them.

    SigLIP names them `encoder.layers`, DINOv2 names them `encoder.layer`. Written
    as an explicit search that RAISES when it finds neither, rather than returning
    an empty list: an empty list would make `unfreeze_last_n` a silent no-op and
    the arm would train its head only while every table called it fine-tuned.
    """
    for path in ("encoder.layers", "encoder.layer", "layers", "blocks"):
        obj = encoder
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None and len(list(obj)) > 0:
            return list(obj)
    raise AttributeError(
        f"cannot locate transformer blocks on {type(encoder).__name__}; "
        "unfreeze_last_n cannot be honoured, and silently freezing everything "
        "would mislabel the arm")


class FoundationSegNet(nn.Module):
    """A frozen-or-partly-unfrozen ViT encoder + a segmentation head.

    Implements the study's architecture-blind contract exactly as models.py
    defines it:

        forward_train(x) -> (logits[B,1,H,W], None)
        forward(x)       -> logits[B,1,H,W]
        .backbone        -> the pretrained part, for build_param_groups

    x is RAW [0,1] pixels. The pixel scale is applied HERE, from buffers that move
    with `.to(device)` and are saved with the weights, so the scale travels with
    the model that needs it.

    WHY THE INPUT IS RESIZED TO THE ENCODER'S NATIVE RESOLUTION
    ------------------------------------------------------------
    640 is not a multiple of these patch sizes and is far from what either model
    saw in pretraining. The alternative -- interpolating the position embeddings
    up to 640 -- is standard but it perturbs exactly the thing being measured: a
    probe of pretrained features should show the features at the resolution they
    were pretrained at. So the tensor is resized 640 -> enc_size, the encoder runs
    natively, and the LOGITS are upsampled back to 640. Every model in this study
    is scored on 640 masks, so the output contract is unchanged.

    This is a real limitation and belongs in the writeup: these arms see a
    448-pixel image where SegFormer sees 640, which costs small-bruise detail --
    the exact population the complete-miss metric is about.
    """

    def __init__(self, encoder, mean, std, head: str = "conv",
                 freeze: bool = True, unfreeze_last_n: int = 0,
                 num_classes: int = 1, enc_size: int | None = None):
        super().__init__()
        self.encoder = encoder
        cfg = encoder.config
        self.enc_size = int(enc_size or getattr(cfg, "image_size", 224))
        self.patch = int(getattr(cfg, "patch_size", 16))
        self.embed_dim = int(getattr(cfg, "hidden_size"))
        self.grid = self.enc_size // self.patch

        self.head_kind = head
        self.decode_head = (LinearProbeHead(self.embed_dim, num_classes) if head == "linear"
                            else ConvDecodeHead(self.embed_dim, num_classes))

        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

        self.frozen = bool(freeze)
        self.unfrozen_blocks = 0
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            if unfreeze_last_n:
                blocks = _transformer_blocks(self.encoder)
                n = min(int(unfreeze_last_n), len(blocks))
                for blk in blocks[-n:]:
                    for p in blk.parameters():
                        p.requires_grad_(True)
                self.unfrozen_blocks = n
                self.frozen = False
        self.init_source = ""

    @property
    def backbone(self):
        return self.encoder

    def train(self, mode: bool = True):
        """Keep a fully frozen encoder in eval() even when the module trains.

        Dropout and any stochastic-depth path inside the encoder would otherwise
        inject noise into features that are supposed to be FIXED, which is not a
        probe of the pretrained representation any more. Partially unfrozen arms
        train normally -- there the encoder really is being optimised.
        """
        super().train(mode)
        if mode and self.frozen:
            self.encoder.eval()
        return self

    def _tokens(self, x):
        out = self.encoder(pixel_values=x)
        tok = out.last_hidden_state                       # [B, (prefix +) N, C]
        want = self.grid * self.grid
        if tok.shape[1] < want:
            raise RuntimeError(
                f"encoder returned {tok.shape[1]} tokens, expected at least {want} "
                f"for a {self.enc_size}/{self.patch} grid -- the input was resized "
                f"to the wrong size or the config was misread")
        # CLS and register tokens are always a PREFIX, so taking the last grid*grid
        # tokens is correct for both families and needs no per-vendor branch.
        tok = tok[:, tok.shape[1] - want:, :]
        b, _, c = tok.shape
        return tok.transpose(1, 2).reshape(b, c, self.grid, self.grid)

    def forward_train(self, x):
        size = x.shape[-2:]
        x = (x - self.mean) / self.std
        if x.shape[-1] != self.enc_size or x.shape[-2] != self.enc_size:
            x = F.interpolate(x, size=(self.enc_size, self.enc_size),
                              mode="bilinear", align_corners=False)
        if self.frozen:
            # The single largest memory saving here, and the same reasoning as
            # engine.load_teacher: without it we build a backward graph through
            # 400 M parameters that are never updated.
            with torch.no_grad():
                feat = self._tokens(x)
            feat = feat.detach()
        else:
            feat = self._tokens(x)
        logits = self.decode_head(feat.float())
        return F.interpolate(logits, size=size, mode="bilinear", align_corners=False), None

    def forward(self, x):
        return self.forward_train(x)[0]


class ResNet50ProbeNet(nn.Module):
    """ImageNet ResNet-50 features + the same head. THE CONTROL ARM.

    Same encoder family as `unet_r50` and `deeplabv3plus_r50`, so a probe win over
    it is directly interpretable against Stage B rather than against a number from
    a different lineage. Runs at 640 natively -- a CNN has no position embeddings
    and no native resolution, which is itself part of why it is the fair control:
    it is not being handicapped by the resize the ViTs need.
    """

    def __init__(self, head: str = "conv", freeze: bool = True,
                 unfreeze_last_n: int = 0, num_classes: int = 1,
                 weights: str = "IMAGENET1K_V2"):
        super().__init__()
        from torchvision.models import resnet50
        net = resnet50(weights=weights)
        self.encoder = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool,
                                     net.layer1, net.layer2, net.layer3, net.layer4)
        self.embed_dim = 2048
        self.head_kind = head
        # ResNet's stride-32 output is 20x20 for a 640 input against the ViTs'
        # 28x28 at 448. The conv head's three 2x upsamples then land at 160 and
        # 224 respectively, and both are bilinearly resized to 640 by the same
        # final interpolate -- so neither arm gets a resolution advantage the
        # other does not, which is what the comparison requires.
        self.decode_head = (LinearProbeHead(self.embed_dim, num_classes) if head == "linear"
                            else ConvDecodeHead(self.embed_dim, num_classes))
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

        self.frozen = bool(freeze)
        self.unfrozen_blocks = 0
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            if unfreeze_last_n:
                # layer4 then layer3, matching "unfreeze the last N stages".
                stages = [self.encoder[7], self.encoder[6], self.encoder[5], self.encoder[4]]
                for blk in stages[:int(unfreeze_last_n)]:
                    for p in blk.parameters():
                        p.requires_grad_(True)
                self.unfrozen_blocks = int(unfreeze_last_n)
                self.frozen = False
        self.init_source = "ResNet-50, ImageNet-1k supervised (torchvision)"

    @property
    def backbone(self):
        return self.encoder

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.frozen:
            self.encoder.eval()          # also holds ResNet's BatchNorm statistics fixed
        return self

    def forward_train(self, x):
        size = x.shape[-2:]
        x = (x - self.mean) / self.std
        if self.frozen:
            with torch.no_grad():
                feat = self.encoder(x)
            feat = feat.detach()
        else:
            feat = self.encoder(x)
        logits = self.decode_head(feat.float())
        return F.interpolate(logits, size=size, mode="bilinear", align_corners=False), None

    def forward(self, x):
        return self.forward_train(x)[0]


# ─────────────────────────────────────────────────────────────────────────────
# 3. the arms
# ─────────────────────────────────────────────────────────────────────────────
# `head`, `freeze` and `unfreeze_last_n` are the ONLY things that differ between a
# probe arm and its baseline arm. Everything else -- recipe, split, seeds, loss,
# cut selection -- is shared, which is what makes the contrast attributable.
FOUNDATION_ARCHS: dict[str, dict] = {
    # ── the gate: frozen encoder, LINEAR head, ~15 epochs each ────────────────
    "medsiglip_probe": {"encoder": "medsiglip", "head": "linear",
                        "freeze": True, "unfreeze_last_n": 0},
    "dinov2_probe": {"encoder": "dinov2", "head": "linear",
                     "freeze": True, "unfreeze_last_n": 0},
    "resnet50_probe": {"encoder": "resnet50", "head": "linear",
                       "freeze": True, "unfreeze_last_n": 0},

    # ── the baseline: partial unfreeze, conv decoder, full recipe, 3 seeds ────
    # unfreeze_last_n = 6 is "the few layers" a 697-image training set can
    # support. Fully unfreezing 400 M parameters on 95 training subjects fits the
    # subjects, not the task, and the frozen probe above is what tells us whether
    # the features were any good in the first place.
    "medsiglip_seg": {"encoder": "medsiglip", "head": "conv",
                      "freeze": True, "unfreeze_last_n": 6},
    "dinov2_seg": {"encoder": "dinov2", "head": "conv",
                   "freeze": True, "unfreeze_last_n": 6},
    "resnet50_seg": {"encoder": "resnet50", "head": "conv",
                     "freeze": True, "unfreeze_last_n": 2},
}

PROBE_ARMS = ("medsiglip_probe", "dinov2_probe", "resnet50_probe")
SEG_ARMS = ("medsiglip_seg", "dinov2_seg", "resnet50_seg")

# The gate's contrast and its attribution arm. Written down here, in the module,
# so it cannot be chosen after the numbers are seen.
GATE_TREATMENT = "medsiglip_probe"
GATE_CONTROL = "resnet50_probe"
GATE_ATTRIBUTION = "dinov2_probe"
GATE_MARGIN = 0.01            # Dice points, the same margin Stage C and M used


# `model_type` in a repo's config.json -> the class that can actually load it.
# DISPATCH, NOT TRIAL-AND-ERROR. The first version of this function tried
# SiglipVisionModel, then Dinov2Model, then AutoModel, and returned the first that
# did not raise. `SiglipVisionModel.from_pretrained` does not raise on a DINOv2
# directory: it reads hidden_size / patch_size / image_size out of the config,
# builds a SigLIP body of those dimensions, matches NONE of the checkpoint's
# parameter names, warns, and returns a RANDOMLY INITIALISED model. That arm then
# trains, converges, and reports a number -- a completely invisible failure, and
# exactly the one `weights.load_pretrained`'s docstring warns about. It cost the
# 2026-08-06 gate run its entire attribution arm.
_MODEL_TYPE_TO_CLASS = {
    "siglip": "SiglipModel",
    "siglip_vision_model": "SiglipVisionModel",
    "dinov2": "Dinov2Model",
    "dinov2_with_registers": "Dinov2WithRegistersModel",
    "clip": "CLIPModel",
    "clip_vision_model": "CLIPVisionModel",
    "vit": "ViTModel",
}


def _load_vision_encoder(path: Path, key: str, max_missing: int = 0):
    """Load the VISION tower from a local HuggingFace directory, offline and VERIFIED.

    Two guarantees, both of which the first version of this function lacked:

    1. THE CLASS IS CHOSEN BY `model_type`, not by trying classes until one stops
       raising -- see `_MODEL_TYPE_TO_CLASS` for what that cost.
    2. THE LOAD IS CHECKED. `output_loading_info=True` returns the missing and
       unexpected key lists, and anything beyond `max_missing` raises. A model
       whose weights did not land must fail here, loudly, and not forty minutes
       later as a plausible-looking Dice number.

    `local_files_only=True` everywhere: a silent network fetch is what makes an
    offline compute node stall instead of failing at load.
    """
    import transformers

    cfg_file = path / "config.json"
    if not cfg_file.exists():
        raise FileNotFoundError(
            f"{key}: no encoder at {path}\n\n{download_instructions()}")

    cfg = json.loads(cfg_file.read_text())
    mtype = cfg.get("model_type", "")
    cls_name = _MODEL_TYPE_TO_CLASS.get(mtype)
    if cls_name is None:
        raise RuntimeError(
            f"{key}: config.json says model_type={mtype!r}, which is not in "
            f"_MODEL_TYPE_TO_CLASS. Add it there rather than falling back to "
            f"AutoModel -- a wrong class here loads NOTHING and says nothing.")
    cls = getattr(transformers, cls_name, None)
    if cls is None:
        raise RuntimeError(
            f"{key}: transformers {transformers.__version__} has no {cls_name}; "
            f"upgrade it or add the correct class name for model_type={mtype!r}.")

    model, info = cls.from_pretrained(str(path), local_files_only=True,
                                      output_loading_info=True)
    missing = list(info.get("missing_keys", []))
    unexpected = list(info.get("unexpected_keys", []))

    # A combined vision+text repo: keep only the vision tower. The TEXT tower's
    # keys are then legitimately "unused", not missing -- nothing here ever builds
    # a text prompt, so they are dropped rather than counted against the load.
    vision = getattr(model, "vision_model", model)
    n_loaded = sum(1 for _ in vision.state_dict())

    if len(missing) > max_missing:
        raise RuntimeError(
            f"{key}: loaded {cls_name} from {path} but {len(missing)} parameters "
            f"were NOT in the checkpoint -- they are RANDOM.\n"
            f"  e.g. {missing[:4]}\n"
            f"  This is the failure that produced a random-weights DINOv2 arm on "
            f"2026-08-06. The class and the checkpoint do not match; do not train "
            f"on this.")
    if unexpected and not hasattr(model, "vision_model"):
        print(f"  {key}: NOTE {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}")

    if not hasattr(vision, "config") or not hasattr(vision.config, "hidden_size"):
        raise RuntimeError(f"{key}: {type(vision).__name__} exposes no config.hidden_size")

    print(f"  {key}: model_type={mtype} -> {cls_name} -> {type(vision).__name__}, "
          f"hidden={vision.config.hidden_size}, "
          f"patch={getattr(vision.config, 'patch_size', '?')}, "
          f"image_size={getattr(vision.config, 'image_size', '?')}, "
          f"{n_loaded} tensors, 0 missing")
    return vision


def build_foundation(env, arch: str, num_classes: int = 1, verbose: bool = True):
    """Construct one arm. Raises on a missing encoder -- never falls back to random.

    `weights.load_pretrained` documents why: a partially-matching or absent
    checkpoint is the classic way to end up training a model that REPORTS
    pretrained while most of its weights are random.
    """
    if arch not in FOUNDATION_ARCHS:
        raise ValueError(f"unknown foundation arch {arch!r}; "
                         f"choices: {sorted(FOUNDATION_ARCHS)}")
    spec = FOUNDATION_ARCHS[arch]
    key = spec["encoder"]

    if key == "resnet50":
        m = ResNet50ProbeNet(head=spec["head"], freeze=spec["freeze"],
                             unfreeze_last_n=spec["unfreeze_last_n"],
                             num_classes=num_classes)
    else:
        src = SOURCES[key]
        path = Path(env.weights) / "foundation" / src.local_dir
        enc = _load_vision_encoder(path, key)
        mean, std = ((SIGLIP_MEAN, SIGLIP_STD) if key == "medsiglip"
                     else (IMAGENET_MEAN, IMAGENET_STD))
        m = FoundationSegNet(enc, mean, std, head=spec["head"],
                             freeze=spec["freeze"],
                             unfreeze_last_n=spec["unfreeze_last_n"],
                             num_classes=num_classes)
        m.init_source = src.init

    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    total = sum(p.numel() for p in m.parameters())

    # THE SECOND LOAD CHECK. A key-count check catches a checkpoint that did not
    # load; this catches a body built by the WRONG CLASS, which loads zero keys
    # but has plausible dimensions because it read them from the right config.
    # On 2026-08-06 a DINOv2 directory opened through SiglipVisionModel produced a
    # 93.6 M SigLIP body against DINOv2's published 86.6 M, trained happily, and
    # invalidated the gate's entire attribution arm. 8 % would have caught it.
    want = SOURCES[key].expected_params_M
    enc_M = sum(p.numel() for p in m.backbone.parameters()) / 1e6
    if want and abs(enc_M - want) / want > 0.05:
        raise RuntimeError(
            f"{arch}: encoder has {enc_M:.2f} M parameters, published is "
            f"{want:.2f} M ({100*(enc_M-want)/want:+.1f} %). The architecture built "
            f"is not the one named in the table -- check that the class chosen for "
            f"config.json's model_type is right. Do not train on this.")

    if verbose:
        print(f"  {arch:<18} {total/1e6:7.1f} M params ({enc_M:.1f} M encoder, "
              f"published {want:.1f} M), {trainable/1e6:6.4f} M trainable "
              f"({100*trainable/total:.2f} %), head={spec['head']}, "
              f"unfrozen_blocks={m.unfrozen_blocks}")
    return m


def install_foundation_shim(env, verbose: bool = True):
    """Teach `engine.train_run`'s `build_model` about these arms.

    Mirrors `efficient_models.install_efficient_shim`, including WHY it reassigns
    the name in two modules: `bruisekit.engine` bound `build_model` by value at
    import time, so patching only `bruisekit.models` leaves the training loop
    calling the original. Any arch this shim does not recognise falls through
    untouched.
    """
    import bruisekit.engine as _be
    import bruisekit.models as _bm

    previous = _bm.build_model

    def build_model(arch, size, paths):
        if arch in FOUNDATION_ARCHS:
            return build_foundation(env, arch, num_classes=1, verbose=verbose)
        return previous(arch, size, paths)

    _bm.build_model = build_model
    _be.build_model = build_model

    previous_probe = _be.resolve_micro_batch

    def resolve_micro_batch(model, cfg, device, teacher=None):
        """Fixed micro-batch with accumulation, never the VRAM probe.

        The probe does a real forward/backward from batch 1 and escalates. On a
        400 M ViT that is minutes of wasted GPU time before every run, and its
        answer is already known: it does not fit past 4 at 448 with gradients on
        a 40 GB MIG slice. `micro * accum` is held at 16 so the EFFECTIVE batch,
        the optimizer-step count and the LR schedule match the SMP and mobile
        baselines exactly -- see this module's docstring.
        """
        if isinstance(model, (FoundationSegNet, ResNet50ProbeNet)):
            micro = int(cfg.get("foundation_micro_batch", 4))
            if getattr(model, "frozen", False):
                # No encoder gradients: a frozen arm holds far more per step.
                micro = int(cfg.get("foundation_probe_micro_batch", 8))
            target = int(cfg.get("foundation_effective_batch", 16))
            micro = max(1, min(micro, target))
            while target % micro:
                micro -= 1
            return micro, target // micro
        return previous_probe(model, cfg, device, teacher)

    _be.resolve_micro_batch = resolve_micro_batch
    return build_model


# ─────────────────────────────────────────────────────────────────────────────
# 4. scoring -- val operating point, then test once
# ─────────────────────────────────────────────────────────────────────────────
def fit_operating_point(model, env, cfg, man640, run_dir: Path,
                        n_cuts: int = 481) -> dict:
    """Sweep the logit cut on VALIDATION and write `operating_point.json`.

    Identical policy to the rest of the study and deliberately not re-derived
    here: `sweep.select_cut` takes every cut within one standard error of the
    peak as statistically tied and breaks the tie on complete-miss rate, because
    these sweeps are flat plateaus and the argmax fits the val set's sampling
    error. A checkpoint without this file has an undefined test score.
    """
    from bruisekit.data import make_loader
    from bruisekit.sweep import cache_logits, select_cut, sweep_cuts

    amp = bool(cfg.get("amp", True)) and str(env.device).startswith("cuda")
    loader = make_loader(man640["val"], env.cache640, cfg["img_size"],
                         cfg.get("eval_batch", 8), False, cfg.get("workers", 0), 0)
    logits, gts, _ = cache_logits(model, loader, env.device, amp)
    cuts = np.linspace(-6.0, 6.0, n_cuts)
    df = sweep_cuts(logits, gts, cuts)
    op = select_cut(df)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "operating_point.json").write_text(json.dumps(op, indent=2))
    df.to_csv(run_dir / "val_sweep.csv", index=False)
    del logits, gts
    if str(env.device).startswith("cuda"):
        torch.cuda.empty_cache()
    return op


def score_split(model, env, cfg, man640, split: str, cut: float,
                meta: pd.DataFrame | None = None) -> pd.DataFrame:
    """One pass over a split at an ALREADY-FIXED cut. Returns the normalised table.

    `report.normalize` is applied here rather than by the caller so that
    `complete_miss` is recomputed from `dice` in exactly one place. Handbook 7.2a:
    this study publishes `dice == 0`, not `pred_positive_pixels == 0`, and the two
    differ by the case where a model fires confidently on the wrong region --
    still a missed injury to a clinician.
    """
    from bruisekit.data import make_loader
    from bruisekit.evaluate import evaluate_at_cut
    from bruisekit.report import normalize

    amp = bool(cfg.get("amp", True)) and str(env.device).startswith("cuda")
    loader = make_loader(man640[split], env.cache640, cfg["img_size"],
                         cfg.get("eval_batch", 8), False, cfg.get("workers", 0), 0)
    with torch.inference_mode():
        per_image, _ = evaluate_at_cut(model, loader, env.device, cut, amp)
    return normalize(per_image, meta)


def load_arm(env, arch: str, run_dir: Path):
    """Rebuild an arm and load its trained weights. Returns (model, cut).

    The cut is READ, never re-fitted -- re-sweeping on test would pick a better
    threshold and report a meaningless number, because the threshold would have
    been fitted on the data it is scored on (loaders.py says the same).
    """
    model = build_foundation(env, arch, verbose=False)
    state = torch.load(str(run_dir / "best.pt"), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(env.device).eval()
    op_file = run_dir / "operating_point.json"
    if not op_file.exists():
        raise FileNotFoundError(
            f"{run_dir.name}: no operating_point.json -- its score is undefined. "
            f"Run fit_operating_point first.")
    return model, float(json.loads(op_file.read_text())["cut"])


# ─────────────────────────────────────────────────────────────────────────────
# 5. THE GATE
# ─────────────────────────────────────────────────────────────────────────────
def probe_gate(val_tables: dict[str, pd.DataFrame], n_boot: int = 10000,
               seed: int = 0, margin: float = GATE_MARGIN) -> dict:
    """Is dermatology pretraining worth ~20 GPU-hours of fine-tuning?

    Parameters
    ----------
    val_tables : {arm_name: normalised per-image table on the 134 VAL images}.
        Must contain `GATE_TREATMENT` and `GATE_CONTROL`; `GATE_ATTRIBUTION` is
        used when present.

    THE RULE, fixed in this module before any number was produced:

        open  iff  the (medsiglip - resnet50) val-Dice CI clears zero

    Reported alongside and NOT ANDed in:

      - ATTRIBUTION. delta(medsiglip - dinov2). If DINOv2 -- self-supervised on
        natural images, zero medical data -- matches MedSigLIP, then the probe is
        measuring modern ViT features and not medical pretraining. The gate can
        open and the CLAIM still has to change. There is no honest way to fold
        that into a single boolean, so it is printed separately.
      - MISSES. delta on complete-miss rate, the endpoint this study is judged on.

    WHY VALIDATION AND NOT TEST
    -----------------------------
    A gate scored on test is a decision made on the data the paper reports, and
    every downstream number inherits that. The 134 val images are the same ones
    every operating point in this study is fitted on, so this spends no new
    information.
    """
    from bruisekit.significance import paired_contrast_multi

    for need in (GATE_TREATMENT, GATE_CONTROL):
        if need not in val_tables:
            raise KeyError(f"probe_gate needs {need!r}; have {sorted(val_tables)}")

    main = paired_contrast_multi(val_tables[GATE_TREATMENT], val_tables[GATE_CONTROL],
                                 GATE_TREATMENT, GATE_CONTROL,
                                 n_boot=n_boot, seed=seed, margin=margin)

    per_arm = {k: {"mean_dice": float(v.dice.mean()),
                   "median_dice": float(v.dice.median()),
                   "misses": int(v.complete_miss.sum()),
                   "n_images": int(len(v))}
               for k, v in sorted(val_tables.items())}

    out = {
        "n_val_images": int(len(val_tables[GATE_CONTROL])),
        "n_subjects": int(main["n_subjects"]),
        "per_arm": per_arm,
        "treatment": GATE_TREATMENT, "control": GATE_CONTROL,
        "delta_dice": main["delta_dice"],
        "delta_dice_ci95": [main["lo"], main["hi"]],
        "p_treatment_better": main["p_a_better"],
        "p_two_sided": main["p_two_sided"],
        "verdict_vs_control": main["verdict"],
        "delta_miss_rate": main["delta_miss_rate"],
        "delta_miss_ci95": [main["miss_lo"], main["miss_hi"]],
        "p_treatment_fewer_misses": main["p_a_fewer_misses"],
        "margin": margin,
        "n_boot": n_boot,
    }

    # ── attribution: medical pretraining, or just a modern ViT? ───────────────
    if GATE_ATTRIBUTION in val_tables:
        attr = paired_contrast_multi(val_tables[GATE_TREATMENT],
                                     val_tables[GATE_ATTRIBUTION],
                                     GATE_TREATMENT, GATE_ATTRIBUTION,
                                     n_boot=n_boot, seed=seed, margin=margin)
        dino = paired_contrast_multi(val_tables[GATE_ATTRIBUTION],
                                     val_tables[GATE_CONTROL],
                                     GATE_ATTRIBUTION, GATE_CONTROL,
                                     n_boot=n_boot, seed=seed, margin=margin)
        out["attribution"] = {
            "delta_vs_dinov2": attr["delta_dice"],
            "delta_vs_dinov2_ci95": [attr["lo"], attr["hi"]],
            "dinov2_delta_vs_control": dino["delta_dice"],
            "dinov2_delta_vs_control_ci95": [dino["lo"], dino["hi"]],
            # The honest reading, computed rather than left to the reader: if
            # DINOv2 gets most of the way to MedSigLIP's gain, the medical corpus
            # is not what is doing the work.
            "medical_share_of_gain": (
                float(1.0 - dino["delta_dice"] / main["delta_dice"])
                if abs(main["delta_dice"]) > 1e-9 else float("nan")),
        }

    out["GATE_dice_ci_clears_zero"] = bool(main["lo"] > 0)
    out["GATE_miss_improved"] = bool(main["miss_hi"] < 0)
    out["GATE_run_method"] = bool(main["lo"] > 0)
    return out


def print_gate(gate: dict) -> None:
    """The verdict, in the shape a reader can paste into a meeting note."""
    print("─" * 74)
    print("FROZEN-PROBE GATE -- 134 validation images, linear head, encoder frozen")
    print("─" * 74)
    print(f"{'arm':<20}{'mean Dice':>11}{'median':>10}{'misses':>9}")
    for k, v in gate["per_arm"].items():
        print(f"{k:<20}{v['mean_dice']:>11.4f}{v['median_dice']:>10.4f}{v['misses']:>9d}")

    lo, hi = gate["delta_dice_ci95"]
    print(f"\n{gate['treatment']} - {gate['control']}")
    print(f"  delta Dice        {gate['delta_dice']:+.4f}   CI [{lo:+.4f}, {hi:+.4f}]"
          f"   P(better) = {gate['p_treatment_better']:.3f}")
    mlo, mhi = gate["delta_miss_ci95"]
    print(f"  delta miss rate   {gate['delta_miss_rate']:+.4f}   CI [{mlo:+.4f}, {mhi:+.4f}]"
          f"   (lower is better)")

    if "attribution" in gate:
        a = gate["attribution"]
        alo, ahi = a["delta_vs_dinov2_ci95"]
        dlo, dhi = a["dinov2_delta_vs_control_ci95"]
        print("\n  ATTRIBUTION -- is it MEDICAL pretraining, or just a modern ViT?")
        print(f"    medsiglip - dinov2   {a['delta_vs_dinov2']:+.4f}  CI [{alo:+.4f}, {ahi:+.4f}]")
        print(f"    dinov2    - resnet50 {a['dinov2_delta_vs_control']:+.4f}  "
              f"CI [{dlo:+.4f}, {dhi:+.4f}]")
        share = a["medical_share_of_gain"]
        if np.isfinite(share):
            print(f"    share of the gain attributable to the MEDICAL corpus: {share:6.1%}")
            if share < 0.5:
                print("    -> most of the gain is generic ViT pretraining. Whatever the")
                print("       gate says, do NOT claim dermatology pretraining as the cause.")

    print()
    print(f"  GATE_dice_ci_clears_zero : {gate['GATE_dice_ci_clears_zero']}")
    print(f"  GATE_miss_improved       : {gate['GATE_miss_improved']}  (reported, NOT ANDed)")
    print("─" * 74)
    if gate["GATE_run_method"]:
        print("VERDICT: OPEN -- the frozen features are better. Train the seg arms.")
    else:
        print("VERDICT: CLOSED -- dermatology features do not beat ImageNet features")
        print("         on this task at frozen. That is a RESULT: report it and stop.")
        print("         ~20 GPU-hours saved, and the null is publishable as-is.")
    print("─" * 74)


# ─────────────────────────────────────────────────────────────────────────────
# 6. BiomedParse -- the zero-shot question, for one afternoon and no training
# ─────────────────────────────────────────────────────────────────────────────
def biomedparse_zeroshot(env, test_manifest: pd.DataFrame, predict_fn=None,
                         prompt: str = "bruise", limit: int | None = None):
    """Score a text-prompted biomedical segmenter on the 185 test images, untrained.

    A DIFFERENT AND CHEAPER QUESTION than the gate: not "are these features
    better", but "does a biomedical foundation model see a bruise AT ALL when you
    ask it to". No training, no threshold to fit -- like native-argmax YOLO, the
    absence of a fitted parameter is exactly why the number is trustworthy.

    `predict_fn` is REQUIRED to be supplied by the caller when the packaged
    integration is not importable, and this function does not guess at one.
    BiomedParse ships as a research repo whose inference entry point moves between
    releases; fabricating a call here would produce an import error at best and,
    at worst, a plausible mask from the wrong code path. Signature:

        predict_fn(image_path: str, prompt: str) -> np.ndarray[H, W]  (0/1)

    Returns (per_image_table, note). A None table means it did not run, with the
    reason in `note` -- never a silent skip.
    """
    from bruisekit.metrics import compute_image_row

    if predict_fn is None:
        try:                                        # noqa: SIM105 -- the message matters
            from biomedparse import predict as _p   # type: ignore  # noqa: F401
        except Exception:                           # noqa: BLE001
            return None, (
                "BiomedParse not importable and no predict_fn supplied -- SKIPPED.\n"
                f"  To enable: download {BIOMEDPARSE_REPO} (see download_instructions), "
                "install its requirements, and pass predict_fn=<your callable>.\n"
                "  This cell is optional; the gate does not depend on it.")
        predict_fn = _p

    import cv2

    df = test_manifest if limit is None else test_manifest.head(limit)
    rows = []
    for _, r in df.iterrows():
        pred = np.asarray(predict_fn(str(env.data / r.image_path), prompt))
        gt = cv2.imread(str(env.data / r.mask_path), cv2.IMREAD_GRAYSCALE)
        if gt is None:
            raise FileNotFoundError(f"unreadable mask: {r.mask_path}")
        if gt.ndim == 3:
            gt = gt[..., 0]
        gt = (gt > 0).astype(np.uint8)
        if pred.shape != gt.shape:
            pred = cv2.resize(pred.astype(np.uint8), (gt.shape[1], gt.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        rows.append(compute_image_row(pred.astype(np.uint8), gt, str(r.stem)))
    return pd.DataFrame(rows), f"BiomedParse zero-shot, prompt={prompt!r}, n={len(rows)}"


# ─────────────────────────────────────────────────────────────────────────────
# 7. self test
# ─────────────────────────────────────────────────────────────────────────────
def self_test(verbose: bool = True) -> bool:
    """Check the two things that would silently invalidate every downstream number.

    1. THE INTERFACE. Every arm must return (logits[B,1,640,640], None) from
       forward_train and the identical tensor from forward. A model that returns
       the right shape by accident from the wrong code path still trains and still
       scores -- it is simply not the model in the table.
    2. THE FREEZE. A frozen probe must have ZERO trainable encoder parameters, and
       a partially unfrozen arm must have MORE THAN ZERO. Getting this backwards
       is invisible: the run completes, the loss falls, and the arm is mislabelled.

    Runs on a randomly initialised ViT-shaped stub so it needs no downloaded
    weights and no GPU -- it tests this module's wiring, not the vendors'.
    """
    ok = True

    class _StubCfg:
        hidden_size, patch_size, image_size = 64, 16, 64

    class _StubEncoder(nn.Module):
        """Minimal stand-in: 16 patch tokens + 1 CLS prefix, 2 'layers'."""

        def __init__(self):
            super().__init__()
            self.config = _StubCfg()
            self.embed = nn.Conv2d(3, 64, 16, stride=16)
            self.encoder = nn.Module()
            self.encoder.layers = nn.ModuleList([nn.Linear(64, 64) for _ in range(2)])

        def forward(self, pixel_values):
            t = self.embed(pixel_values).flatten(2).transpose(1, 2)      # [B,16,64]
            for lyr in self.encoder.layers:
                t = lyr(t)
            cls = t[:, :1] * 0
            out = type("O", (), {})()
            out.last_hidden_state = torch.cat([cls, t], dim=1)           # prefix CLS
            return out

    x = torch.rand(2, 3, 640, 640)

    for head in ("linear", "conv"):
        m = FoundationSegNet(_StubEncoder(), IMAGENET_MEAN, IMAGENET_STD,
                             head=head, freeze=True, unfreeze_last_n=0).eval()
        with torch.no_grad():
            logits, aux = m.forward_train(x)
            plain = m(x)
        shape_ok = tuple(logits.shape) == (2, 1, 640, 640)
        enc_trainable = sum(p.numel() for p in m.encoder.parameters() if p.requires_grad)
        same = bool(torch.equal(plain, logits))
        good = shape_ok and aux is None and same and enc_trainable == 0
        ok &= good
        if verbose:
            print(f"  frozen/{head:<7} shape={tuple(logits.shape)} aux_none={aux is None} "
                  f"forward_matches={same} encoder_trainable={enc_trainable}  "
                  f"{'OK' if good else 'FAIL'}")

    m = FoundationSegNet(_StubEncoder(), IMAGENET_MEAN, IMAGENET_STD,
                         head="conv", freeze=True, unfreeze_last_n=1)
    enc_trainable = sum(p.numel() for p in m.encoder.parameters() if p.requires_grad)
    good = enc_trainable > 0 and m.unfrozen_blocks == 1 and not m.frozen
    ok &= good
    if verbose:
        print(f"  unfreeze_last_n=1  encoder_trainable={enc_trainable} "
              f"blocks={m.unfrozen_blocks} frozen={m.frozen}  {'OK' if good else 'FAIL'}")

    # A frozen encoder must stay in eval() when the wrapper is put in train mode,
    # or dropout perturbs features that are supposed to be fixed.
    m2 = FoundationSegNet(_StubEncoder(), IMAGENET_MEAN, IMAGENET_STD,
                          head="linear", freeze=True).train()
    good = (not m2.encoder.training) and m2.decode_head.training
    ok &= good
    if verbose:
        print(f"  train() keeps frozen encoder in eval: encoder.training="
              f"{m2.encoder.training} head.training={m2.decode_head.training}  "
              f"{'OK' if good else 'FAIL'}")

    # build_param_groups must account for every trainable tensor -- it asserts, so
    # a failure here is a hard error rather than a quietly unoptimised head.
    from bruisekit.models import build_param_groups
    groups = build_param_groups(m, 6e-5, 6e-4, 0.01)
    n = sum(len(g["params"]) for g in groups)
    ok &= n > 0
    if verbose:
        print(f"  build_param_groups: {len(groups)} groups, {n} tensors  "
              f"{'OK' if n > 0 else 'FAIL'}")
        print("PASS" if ok else "FAIL")
    return bool(ok)


if __name__ == "__main__":
    raise SystemExit(0 if self_test() else 1)
