"""Stage N2 -- close the grid confound, then test DERMATOLOGY pretraining properly.

WHY THIS STAGE EXISTS
----------------------
Stage N (`foundation.py`, handbook 7f) asked "does medical pretraining help?" and
came back with `medsiglip - dinov2 = -0.166`. Two things are wrong with quoting
that, and this module fixes both.

  1. THE GRID CONFOUND (handbook 7f.9). The three arms scored in exactly the order
     of their feature-grid size -- dinov2 37x37 -> 0.657, medsiglip 28x28 -> 0.491,
     resnet50 20x20 -> 0.123. A linear probe is a 1x1 convolution on that grid,
     bilinearly upsampled to 640, so a finer grid is worth Dice INDEPENDENTLY of
     what the encoder knows. Rank correlation of 1.0 between grid and score is not
     proof of a confound, but it is exactly what one looks like.

  2. MEDSIGLIP IS NOT A DERMATOLOGY MODEL. It is a general-medical vision-language
     encoder -- chest X-ray, histopathology, ophthalmology, with dermatology as one
     slice. "Medical pretraining does not help on bruises" was never tested by it;
     what was tested is "this particular general-hospital encoder does not help".
     The claim the study wants to make needs an encoder trained on CONSUMER-CAMERA
     PHOTOGRAPHS OF SKIN, which is what our images are.

So Stage N2 is two experiments in one notebook, in this order, and the first one
gates how the second is read:

  EXPERIMENT 1 -- GRID CALIBRATION. One encoder (ImageNet ResNet-50), one stage
  (layer3), FOUR input sizes -> grids 20 / 28 / 37 / 40. Identical weights,
  identical depth, identical everything except the grid. Whatever Dice moves is
  the grid, because nothing else changed. Run alongside a PIXEL FLOOR -- a 1x1
  conv on raw RGB average-pooled to the same four grids, zero pretraining -- which
  bounds how much of any score is reachable with no features at all.

  EXPERIMENT 2 -- THE CORPUS ARMS, ALL AT 28x28. Every encoder is probed at the
  SAME grid, so the confound cannot recur by construction. Dermatology-specific
  encoders (DermLIP, DermLIP-PanDerm) against the three Stage N controls
  (DINOv2 = generic self-supervised, MedSigLIP = general-medical, ResNet-50 =
  ImageNet supervised) plus BiomedCLIP as a second non-derm biomedical corpus.

WHAT IS DELIBERATELY NOT HERE
------------------------------
No distillation, no fine-tuning, no seg arms. `foundation.py`'s docstring gives
three reasons distillation is out of scope and every one of them still holds; this
module inherits that decision rather than reopening it. Stage N2 answers ONE
question -- does the pretraining CORPUS matter once the grid is held fixed -- and
if it does, the seg arms are Stage N's 10 onward with a new encoder key, which is
a separate decision.

GOOGLE DERM FOUNDATION IS EXCLUDED, AND THE REASON IS NOT A JUDGEMENT CALL
---------------------------------------------------------------------------
`google/derm-foundation` is the most on-target model in existence for this task:
BiT-M ResNet-101x3, trained on teledermatology photographs from consumer cameras.
It cannot be used here. It ships as a TensorFlow/Keras SavedModel whose only
exported signature returns a SINGLE 6144-dimensional embedding per 448x448 image.
There is no token grid and no intermediate feature map on the public interface, so
there is nothing for a 1x1 convolution to sit on. A probe of it would be a probe
of one vector per image, which cannot produce a segmentation mask at all.

It is registered in `SOURCES` with `spatial=False` so it appears in the provenance
table as an EXPLICIT GAP rather than being quietly omitted -- the same treatment
handbook 5 gives nnU-Net, and for the same reason: "the model we could not run" is
exactly the fact a reader needs. `dermfoundation_tiled_note()` describes the one
honest way to get spatial features out of it if that is ever worth the compute.

THE ONE RECIPE DEVIATION, STATED UP FRONT
------------------------------------------
The ViT arms have their position embeddings RESAMPLED to a 28x28 grid. Stage N
took the opposite decision (handbook 7f.6: run each encoder at its native
resolution, never interpolate) -- and that decision is precisely what produced the
confound in 7f.9, because the native resolutions differ. You cannot have both
"every encoder at its native grid" and "every encoder at the same grid". This
stage picks the second, because the question is about the CORPUS and the grid is
the nuisance variable. Resampling is applied IDENTICALLY to every ViT arm, so it
is not a difference between arms; it is a property of the whole experiment, and it
belongs in the limitations either way.
"""
from __future__ import annotations

import inspect
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from bruisekit.foundation import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    SIGLIP_MEAN,
    SIGLIP_STD,
    LinearProbeHead,
    _load_vision_encoder,
)

# OpenAI-CLIP statistics. open_clip's own preprocessing uses these for every
# model in the family unless a config overrides them, and both DermLIP and
# BiomedCLIP inherit that default. Getting this wrong is the same class of error
# as feeding YOLO ImageNet-normalised pixels (handbook 3): the model still runs,
# still trains, and reports a number that is quietly several points low.
OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# THE GRID EVERY CORPUS ARM IS PROBED AT.
#
# CORRECTED 2026-08-09. This was chosen believing 28 was MedSigLIP's native
# 448/16, making it the one arm that needed no resampling. THAT WAS WRONG:
# MedSigLIP is SoViT-400m/14, patch 14, so its native grid at 448 is 32x32 and
# the run reports its input as 392 (= 28 x 14). NO ARM here is at its native grid.
#
# The choice stands anyway, because what the design requires is that the grid be
# IDENTICAL across arms, not that it be native to any one of them -- and the
# calibration measured what a grid is worth on this task before the corpus arms
# were read: -0.101 Dice per doubling, CI [-0.128, -0.074], i.e. finer is slightly
# WORSE and the effect is small next to the between-corpus spread. A different
# target would move every arm together. Record the error rather than quietly
# re-deriving a rationale that fits: handbook 7f.9 and 7h.3.
TARGET_GRID = 28


# ─────────────────────────────────────────────────────────────────────────────
# 1. where the weights come from
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class DermSource:
    """One encoder: where to get it, what it saw, what it is licensed as.

    `spatial` is the field that decides whether an entry can be an ARM at all. A
    model that only emits a pooled embedding has nothing for a 1x1 convolution to
    sit on, and listing it as an arm that "scored badly" would be a measurement of
    our adapter rather than of the encoder.

    `licence` is carried for the same reason `foundation.FoundationSource` carries
    it: handbook 7b.1 chose DeepLabV3+ over SegFormer as the Stage F teacher
    specifically to escape a non-commercial licence, and re-introducing a
    restricted encoder without recording it would silently undo that. Note that
    BOTH dermatology encoders here are non-commercial.
    """

    key: str
    repo: str
    local_dir: str
    loader: str                    # "open_clip" | "hf" | "torchvision" | "none"
    corpus: str                    # the label the result table groups by
    kind: str = "open"             # "open" | "gated" | "manual"
    licence: str = ""
    init: str = ""
    note: str = ""
    spatial: bool = True           # False -> cannot be an arm; an explicit gap
    native_size: int = 224
    patch: int = 16
    expected_params_M: float = 0.0
    # Tolerance on the parameter check. Tighter is better, but an open_clip vision
    # tower may or may not include its projection matrix depending on how the
    # checkpoint was exported, which is worth ~0.5 M on a ViT-B. 10 % still catches
    # the failure this check exists for -- a body built by the wrong class, which
    # was 8 % off on 2026-08-06 (foundation.py, _MODEL_TYPE_TO_CLASS).
    param_tol: float = 0.05
    aliases: tuple = field(default_factory=tuple)


SOURCES: dict[str, DermSource] = {
    # ── the dermatology arms: the reason this stage exists ────────────────────
    "dermlip": DermSource(
        key="dermlip",
        repo="redlessone/DermLIP_ViT-B-16",
        local_dir="DermLIP_ViT-B-16",
        loader="open_clip",
        corpus="dermatology (image-text)",
        kind="open",
        licence="CC-BY-NC-ND-4.0 -- NON-COMMERCIAL, and no-derivatives. Stricter "
                "than MedSigLIP's terms in one respect and looser in another; "
                "either way this is not a commercialisable encoder.",
        init="ViT-B/16 CLIP vision tower trained on Derm1M -- 1.03 M image-text "
             "pairs over 403 k unique dermatological images, clinical and "
             "dermoscopic, 390 skin conditions.",
        note="THE PRIMARY TREATMENT ARM, fixed by name before any number was "
             "produced. Derm1M is the closest published corpus to our data: "
             "consumer-camera photographs of skin. If dermatology pretraining is "
             "worth anything for bruises, this is where it shows up.",
        native_size=224, patch=16, expected_params_M=86.2, param_tol=0.10,
    ),
    "dermlip_panderm": DermSource(
        key="dermlip_panderm",
        repo="redlessone/DermLIP_PanDerm-base-w-PubMed-256",
        local_dir="DermLIP_PanDerm-base-w-PubMed-256",
        loader="open_clip",
        corpus="dermatology (SSL + image-text)",
        kind="open",
        licence="CC-BY-NC-ND-4.0 -- NON-COMMERCIAL, no-derivatives. The model "
                "card's metadata header says cc-by-4.0 while its body says "
                "cc-by-nc-nd-4.0; the RESTRICTIVE reading is recorded here because "
                "an ambiguous licence resolved in our own favour is not a licence "
                "review. Confirm with the authors before any commercialisation "
                "claim.",
        init="PanDerm-base ViT-B/16 vision tower -- self-supervised on >2 M "
             "skin-disease images across four imaging modalities -- then CLIP-"
             "aligned on Derm1M against a PubMedBERT-256 text encoder. 224 native.",
        note="SECONDARY derm arm, and a genuinely different recipe from `dermlip`: "
             "self-supervised pretraining on skin FIRST, language alignment second. "
             "If the two derm arms disagree, that difference is informative, and "
             "the primary/secondary split above is what stops it becoming a "
             "post-hoc pick.\n"
             "  REACHED THROUGH HUGGINGFACE, not PanDerm's Google Drive .pth: that "
             "checkpoint needs the authors' CAEv2 model definition to load, and a "
             "hand-rewritten architecture is the failure mode handbook 7.1 vendored "
             "two whole files to avoid.\n"
             "  KNOWN RISK: the model card tells you to install the Derm1M package "
             "before loading, so plain open_clip may not be able to construct the "
             "PanDerm vision tower. If it cannot, `_load_open_clip_visual` raises "
             "and this arm is DROPPED -- which costs nothing, because it is "
             "secondary and the gate's primary arm is `dermlip`. Do not substitute "
             "another checkpoint for it and do not hand-build the tower.",
        native_size=224, patch=16, expected_params_M=86.2, param_tol=0.10,
    ),

    # ── the non-derm biomedical control ───────────────────────────────────────
    "biomedclip": DermSource(
        key="biomedclip",
        repo="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        local_dir="BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        loader="open_clip",
        corpus="biomedical figures (image-text)",
        kind="open",
        licence="MIT -- the only permissively-licensed biomedical arm in the pool.",
        init="ViT-B/16 CLIP vision tower trained on PMC-15M: 15 M figure-caption "
             "pairs scraped from PubMed Central articles.",
        note="ARCHITECTURE-MATCHED to both derm arms -- ViT-B/16, 224 native, same "
             "open_clip loader, same resampling. So `dermlip - biomedclip` varies "
             "the CORPUS and nothing else: skin photographs against journal "
             "figures, at identical capacity. That is a cleaner contrast than "
             "anything Stage N could form, where every arm differed in size, "
             "patch, native resolution and vendor at once.",
        native_size=224, patch=16, expected_params_M=86.2, param_tol=0.10,
    ),

    # ── the three Stage N controls, re-run HERE at the matched grid ───────────
    # They are re-run rather than quoted from FOUNDATION_RESULTS/ because the
    # whole point of this stage is that those numbers were taken at three
    # different grids. Comparing a new 28x28 arm against a stored 37x37 one would
    # reproduce the exact confound the stage exists to close.
    "dinov2": DermSource(
        key="dinov2",
        repo="facebook/dinov2-base",
        local_dir="dinov2-base",
        loader="hf",
        corpus="natural images (self-supervised)",
        kind="open",
        licence="Apache-2.0.",
        init="DINOv2 ViT-B/14, self-supervised on LVD-142M natural images.",
        note="THE GENERIC CONTROL, and the arm to beat. It won Stage N outright "
             "(+0.534 over ResNet-50) -- but it also had the finest grid, AND "
             "frozen-feature linear probing is the benchmark DINOv2 was designed "
             "and tuned against (handbook 7f.9). At 392/14 = 28 the first "
             "advantage is removed. The second cannot be removed and stays in the "
             "limitations.",
        native_size=518, patch=14, expected_params_M=86.6,
    ),
    "medsiglip": DermSource(
        key="medsiglip",
        repo="google/medsiglip-448",
        local_dir="medsiglip-448",
        loader="hf",
        corpus="general medical (image-text)",
        kind="gated",
        licence="Health AI Developer Foundations terms of use -- NOT a standard OSS "
                "licence, use-restricted, and a human must accept it on the model "
                "page before the weights download.",
        init="SigLIP SoViT-400m vision tower on medical image-text pairs: "
             "radiology, histopathology, ophthalmology, dermatology.",
        note="THE GENERAL-MEDICAL CONTROL. PATCH 14, NOT 16 -- earlier revisions "
             "of this note and of handbook 7f.3 said 16 and were wrong; the run "
             "settles it, `medsiglip_g28` reports input 392 = 28 x 14. Its native "
             "grid at 448 is 32x32, so this arm is resampled like every other one "
             "and is NOT the untouched reference the TARGET_GRID comment once "
             "claimed. Also 5x the parameters of every other ViT here -- if "
             "capacity mattered on this task it would show up as this arm winning, "
             "and it came LAST at 0.467, behind an 8.5 M ImageNet ResNet trunk.",
        native_size=448, patch=14, expected_params_M=428.0,
    ),
    "resnet50": DermSource(
        key="resnet50",
        repo="torchvision IMAGENET1K_V2",
        local_dir="(torchvision cache)",
        loader="torchvision",
        corpus="ImageNet-1k (supervised)",
        kind="open",
        licence="BSD-3-Clause (torchvision weights).",
        init="ResNet-50, ImageNet-1k supervised.",
        note="THE IMAGENET CONTROL and the grid-calibration encoder. Same family "
             "as `unet_r50` and `deeplabv3plus_r50`, so anything measured against "
             "it is interpretable against Stage B. Probed at layer3 (stride 16) so "
             "448 -> 28 exactly; Stage N used layer4 (stride 32) at 640 -> 20, and "
             "the layer3-vs-layer4 pair is itself one of the calibration points.",
        native_size=448, patch=16, expected_params_M=23.5,
    ),

    # ── registered, and NOT runnable. An explicit gap, not an omission. ───────
    "derm_foundation": DermSource(
        key="derm_foundation",
        repo="google/derm-foundation",
        local_dir="derm-foundation",
        loader="none",
        corpus="dermatology (teledermatology photos)",
        kind="gated",
        licence="Health AI Developer Foundations terms of use -- gated, "
                "use-restricted.",
        init="BiT-M ResNet-101x3, contrastively pretrained then fine-tuned on "
             "clinical dermatology datasets. Input 448x448.",
        note="THE BEST-TARGETED MODEL IN EXISTENCE FOR THIS TASK, AND IT CANNOT BE "
             "PROBED. It ships as a TensorFlow/Keras SavedModel whose only exported "
             "signature returns a single 6144-d embedding per image -- no token "
             "grid, no intermediate feature map, nothing for a 1x1 convolution to "
             "sit on. This is a property of what Google published, not a limitation "
             "of this code. See `dermfoundation_tiled_note()` for the one honest "
             "workaround and what it would cost.",
        spatial=False, native_size=448, patch=0, expected_params_M=0.0,
    ),
    # MedImageInsight is deliberately absent as an ARM and recorded here in prose:
    # it is a DaViT trained on broad medical imaging, i.e. the SAME question
    # MedSigLIP already answers ("does a general-hospital encoder help?"), reached
    # through a different vendor's code. Adding it would be a replication, not a
    # new contrast, and this stage's scarce comparison is dermatology-vs-generic.
}

# Arms that can actually be probed, in the order they should be reported.
PROBEABLE = tuple(k for k, s in SOURCES.items() if s.spatial)


def dermfoundation_tiled_note() -> str:
    """Why `google/derm-foundation` is not an arm, and what running it would take."""
    return """\
google/derm-foundation cannot be a probe arm as published.

  WHAT IT IS      BiT-M ResNet-101x3, trained on teledermatology photographs --
                  by corpus, the closest published model to our images.
  WHAT IT EMITS   one 6144-dimensional vector per 448x448 image. That is the only
                  exported signature of the Keras SavedModel. No token grid, no
                  intermediate feature map.
  WHY THAT ENDS   a linear probe here is a 1x1 convolution ON A GRID. One vector
                  per image is a grid of 1x1. Upsampled to 640 it can only ever
                  predict a constant mask, so its Dice would measure the mean
                  bruise area of the dataset and nothing about the encoder.

THE ONE HONEST WORKAROUND, and its cost:

  Slide a 448x448 window over each 640x640 image on a KxK lattice, take the
  embedding at each position, and assemble a [K, K, 6144] feature volume. That is
  a genuine spatial feature map built only from the published interface.

    K = 8   ->  64 forwards/image  x 831 train+val images  = 53 k forwards
                of a ResNet-101x3 (~380 M params) at 448.
                Order of an hour on an A100, plus a TensorFlow install.
    K = 8   ->  an 8x8 grid, COARSER than the 20x20 that scored 0.12 in Stage N.

  So the workaround is both the most expensive arm in the pool and the one with
  the worst grid, and Experiment 1 of this stage is what tells you the second of
  those is disqualifying on its own. Recommendation: do not run it. Cite the
  architecture as the reason, which is checkable and does not depend on our
  compute budget.

  If it is run anyway, it needs TensorFlow, a Health AI Developer Foundations
  licence acceptance, and it must be reported at ITS grid against the calibration
  curve -- never against the 28x28 arms.
"""


def download_instructions(env=None) -> str:
    """Exactly what to download, from where, and where to put it.

    Text rather than an executed download: two of these are licence-gated and the
    compute node is usually offline, so the notebook prints this and stops instead
    of silently training from random init.
    """
    dest = str(Path(env.weights) / "foundation") if env is not None else \
        "<bundle>/pretrained_weights/foundation"
    return f"""\
Download ON A MACHINE WITH INTERNET (your laptop, or an ORC login node), into:

    {dest}/

Stage N2 shares `pretrained_weights/foundation/` with Stage N, so if you already
ran Stage N then dinov2-base and medsiglip-448 are ALREADY THERE and steps 4-5
are no-ops. Only steps 1-3 are new.

The CLI is `hf` (huggingface_hub >= 1.0). Older environments ship
`huggingface-cli` instead, with the same flags -- `huggingface-cli download ...`
and `huggingface-cli login`. Use whichever the node has.

1. DermLIP  -- THE PRIMARY ARM, open, no login
   hf download {SOURCES['dermlip'].repo} \\
        --local-dir {dest}/{SOURCES['dermlip'].local_dir}
   ~600 MB. Licence CC-BY-NC-ND-4.0 (non-commercial).

2. DermLIP-PanDerm  -- secondary derm arm, open, no login
   hf download {SOURCES['dermlip_panderm'].repo} \\
        --local-dir {dest}/{SOURCES['dermlip_panderm'].local_dir}
   ~600 MB. Licence CC-BY-NC-ND-4.0 (see the SOURCES note -- the card is
   self-contradictory and the restrictive reading is the one recorded).
   IF THIS 404s, OR open_clip CANNOT BUILD IT, DROP IT. Its vision tower is
   PanDerm and the model card says to install the Derm1M package first, so plain
   open_clip may not construct it. That costs nothing: this arm is SECONDARY, the
   gate's primary arm is DermLIP, and the notebook reports whatever is present.
   Do not substitute a different checkpoint for it.

3. BiomedCLIP  -- the non-derm biomedical control, open, MIT
   hf download {SOURCES['biomedclip'].repo} \\
        --local-dir {dest}/{SOURCES['biomedclip'].local_dir}
   ~800 MB.

4. DINOv2  -- the generic control (already present if Stage N ran)
   hf download {SOURCES['dinov2'].repo} \\
        --local-dir {dest}/{SOURCES['dinov2'].local_dir}
   ~350 MB.

5. MedSigLIP  -- GATED, needs a licence acceptance (already present if Stage N ran)
   a) open https://huggingface.co/{SOURCES['medsiglip'].repo} signed in and accept
      the Health AI Developer Foundations terms, or every download returns 401.
   b) hf auth login          (token with 'read' scope)
   c) hf download {SOURCES['medsiglip'].repo} \\
        --local-dir {dest}/{SOURCES['medsiglip'].local_dir}
   ~1.6 GB (only the vision tower is ever loaded).

6. ResNet-50  -- torchvision, automatic IF the node has internet. If not:
     python -c "import torchvision; torchvision.models.resnet50(weights='IMAGENET1K_V2')"
   ~100 MB into ~/.cache/torch/hub/checkpoints/. Run it on a login node.

NOT DOWNLOADED, ON PURPOSE:
  google/derm-foundation -- registered as an explicit gap. It emits one 6144-d
  vector per image and has no token grid, so it cannot be probed. See
  `dermprobe.dermfoundation_tiled_note()`.

PYTHON DEPENDENCY: the three open_clip arms need `open_clip_torch`. The notebook
installs it for you in §2 via `ensure_open_clip()` -- but ALWAYS with --no-deps,
because open_clip_torch declares torch and torchvision as dependencies and a
plain install can drop different wheels into ~/.local, which precedes the system
prefix on sys.path and would silently re-version torch for EVERY stage in this
bundle. To do it by hand instead, from the login node:

    <the kernel's python> -m pip install --user --no-deps \\
        open_clip_torch ftfy regex timm safetensors wcwidth

Use the interpreter the KERNEL runs, not whichever `python` is on $PATH -- home
is shared, so a --user install from the login node lands where the compute node
reads. The notebook prints the right path if the versions do not line up.

Then rsync the whole `foundation/` directory to the bundle on ORC. Nothing is
fetched at run time -- every load passes local_files_only=True.
"""


def report_sources(env) -> pd.DataFrame:
    """What is on disk right now, and what could never be run. Downloads nothing.

    The unrunnable row is INCLUDED with `spatial=False`. A provenance table that
    silently drops the model a reader is most likely to ask about is worse than no
    table -- "the model we could not run, and why" is itself the finding.
    """
    rows = []
    for k, s in SOURCES.items():
        if s.loader == "torchvision":
            present, where = True, s.local_dir
        else:
            p = Path(env.weights) / "foundation" / s.local_dir
            present = p.exists() and any(p.iterdir()) if p.is_dir() else False
            where = str(p)
        rows.append({
            "encoder": k, "corpus": s.corpus, "repo": s.repo, "loader": s.loader,
            "kind": s.kind, "probeable": s.spatial, "present": bool(present),
            "licence": s.licence.split("--")[0].strip().rstrip("."),
            "path": where,
        })
    return pd.DataFrame(rows)


def check_open_clip() -> tuple[bool, str]:
    """Is `open_clip_torch` importable RIGHT NOW? Returns (ok, message), never raises.

    Called from the setup cell AND again at the top of the training cell. That
    looks redundant and is not: a notebook variable set during setup goes stale
    the moment the package is installed mid-session, and a stale False silently
    skips every open_clip arm for a reason that stopped being true.
    """
    import sys
    try:
        import open_clip
    except Exception as e:                                    # noqa: BLE001
        arms = ", ".join(k for k, s in SOURCES.items() if s.loader == "open_clip")
        return False, (f"open_clip is NOT importable ({type(e).__name__}: {e}).\n"
                       f"  The three open_clip arms -- {arms} -- cannot run.\n"
                       f"  Fix with `ensure_open_clip()`, or by hand:\n"
                       f"      {sys.executable} -m pip install --user --no-deps "
                       f"open_clip_torch ftfy regex timm safetensors wcwidth")
    return True, f"open_clip {getattr(open_clip, '__version__', '?')} importable"


# Never auto-installed. If open_clip asks for one of these, the environment is
# broken in a way that installing would make worse: a ~/.local copy SHADOWS the
# system one for every notebook in this bundle, so a fix for one arm silently
# re-versions torch under Stages A through Y.
_NEVER_INSTALL = {"torch", "torchvision", "torchaudio"}
# import name -> pip name, where they differ.
_PIP_NAME = {"PIL": "pillow", "cv2": "opencv-python", "yaml": "pyyaml",
             "pkg_resources": "setuptools"}


def _torch_versions_on_disk() -> str:
    """torch/torchvision versions as a FRESH interpreter sees them.

    Deliberately a subprocess. `torch.__version__` in the running kernel cannot
    change no matter what pip writes to disk, so an in-process before/after
    comparison would report "unchanged" through exactly the failure it exists to
    catch.
    """
    import subprocess
    import sys
    code = ("import torch, torchvision;"
            "print(torch.__version__ + '|' + torchvision.__version__)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else \
        f"PROBE FAILED: {r.stderr.strip()[:200]}"


def ensure_open_clip(install: bool = True, max_rounds: int = 8) -> tuple[bool, str]:
    """Make `open_clip` importable in THIS kernel without disturbing torch.

    Returns (ok, message). Raises only when it would otherwise leave the
    environment in a worse state than it found it.

    WHY THIS IS NOT JUST `pip install open_clip_torch`
    ----------------------------------------------------
    `open_clip_torch` declares torch and torchvision as dependencies. A plain
    install can resolve them to different wheels and drop them into
    `~/.local/lib/pythonX/site-packages`, which precedes the system prefix on
    `sys.path` -- so a fix for three probe arms silently re-versions torch for
    every stage in this bundle, and the symptom appears somewhere else entirely.

    So: `--no-deps` on everything, resolve missing imports one at a time by
    ASKING the failed import what it wants rather than guessing a list that
    drifts between releases, and verify in a fresh subprocess that torch and
    torchvision are byte-identical before and after. If they are not, this raises
    with the uninstall command rather than letting the run continue.
    """
    import importlib
    import site
    import subprocess
    import sys

    ok, msg = check_open_clip()
    if ok or not install:
        return ok, msg

    before = _torch_versions_on_disk()
    if before.startswith("PROBE FAILED"):
        raise RuntimeError(
            f"cannot read the current torch version, so a change could not be "
            f"detected: {before}\n  Fix that before installing anything here.")
    print(f"  torch|torchvision BEFORE : {before}")

    def _pip(pkg: str) -> bool:
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--user", "--no-deps", "-q", pkg],
            capture_output=True, text=True)
        if r.returncode != 0:
            tail = (r.stderr or r.stdout).strip().splitlines()
            print(f"    pip failed for {pkg}: {tail[-1][:180] if tail else '?'}")
        return r.returncode == 0

    print("  installing open_clip_torch (--no-deps)")
    _pip("open_clip_torch")
    site.addsitedir(site.getusersitepackages())
    importlib.invalidate_caches()

    for _ in range(max_rounds):
        try:
            import open_clip                                   # noqa: F401
            break
        except ModuleNotFoundError as e:
            name = (e.name or "").split(".")[0]
            if not name:
                raise
            if name in _NEVER_INSTALL:
                raise RuntimeError(
                    f"open_clip wants {name!r}, which this environment already "
                    f"provides. Installing it would shadow the system copy for "
                    f"every stage in this bundle. Investigate the environment "
                    f"instead of installing.") from e
            pkg = _PIP_NAME.get(name, name)
            print(f"    missing {name!r} -> installing {pkg}")
            if not _pip(pkg):
                raise RuntimeError(
                    f"could not install {pkg}. If this node has no internet, run "
                    f"on the LOGIN node (shared home, so it lands where this "
                    f"kernel reads):\n    {sys.executable} -m pip install --user "
                    f"--no-deps {pkg}") from e
            importlib.invalidate_caches()
    else:
        raise RuntimeError(f"still not importable after {max_rounds} rounds")

    after = _torch_versions_on_disk()
    print(f"  torch|torchvision AFTER  : {after}")
    if after != before:
        raise RuntimeError(
            f"TORCH OR TORCHVISION CHANGED: {before} -> {after}\n"
            f"  A ~/.local copy now shadows the system one and will affect EVERY "
            f"notebook in this bundle. Undo it before running anything else:\n"
            f"      {sys.executable} -m pip uninstall -y torch torchvision\n"
            f"  then re-run this cell and confirm the versions are back to {before}.")

    return check_open_clip()


# ─────────────────────────────────────────────────────────────────────────────
# 2. position-embedding resampling -- the mechanism the whole stage rests on
# ─────────────────────────────────────────────────────────────────────────────
def resample_pos_embed(pos: torch.Tensor, new_grid: int,
                       n_prefix: int = 1) -> torch.Tensor:
    """Bicubically resample a ViT position embedding to a `new_grid` x `new_grid` grid.

    `pos` is [1, n_prefix + G*G, C] or [n_prefix + G*G, C]. Prefix tokens (CLS,
    registers) are always leading and are carried through untouched -- they have no
    spatial meaning, so interpolating them would be mixing a global token into a
    spatial one.

    Written here rather than imported from timm or open_clip because the helper's
    name, signature and prefix-handling have all moved between versions of both,
    and a silently wrong resample produces a model that trains fine and scores
    low -- the failure mode this whole stage was built to detect in the first
    place. Twelve lines is cheaper than a version pin.

    RAISES if the token count is not `n_prefix` plus a perfect square. That is the
    one thing that must not be guessed at: an off-by-one in the prefix count
    shifts every position embedding by one patch and degrades the arm invisibly.
    """
    squeeze = pos.dim() == 2
    if squeeze:
        pos = pos.unsqueeze(0)
    n = pos.shape[1] - n_prefix
    g = int(round(math.sqrt(n)))
    if g * g != n:
        raise ValueError(
            f"position embedding has {pos.shape[1]} tokens and n_prefix={n_prefix}, "
            f"leaving {n} spatial tokens which is not a perfect square. The prefix "
            f"count is wrong; do not guess it -- read it off the model.")
    if g == new_grid:
        return pos.squeeze(0) if squeeze else pos

    prefix, spatial = pos[:, :n_prefix], pos[:, n_prefix:]
    c = spatial.shape[-1]
    spatial = spatial.reshape(1, g, g, c).permute(0, 3, 1, 2)
    spatial = F.interpolate(spatial.float(), size=(new_grid, new_grid),
                            mode="bicubic", align_corners=False)
    spatial = spatial.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, c)
    out = torch.cat([prefix.float(), spatial], dim=1).to(pos.dtype)
    return out.squeeze(0) if squeeze else out


# ─────────────────────────────────────────────────────────────────────────────
# 3. the arm wrappers -- one interface, four very different encoders
# ─────────────────────────────────────────────────────────────────────────────
class _ProbeBase(nn.Module):
    """The study's architecture-blind contract, implemented once.

        forward_train(x) -> (logits[B,1,H,W], None)
        forward(x)       -> logits[B,1,H,W]
        .backbone        -> the pretrained part, for build_param_groups

    `x` is RAW [0,1] pixels and each subclass applies its own scale from buffers,
    exactly as handbook 3 requires. Subclasses implement `_features(x) -> [B,C,g,g]`
    and set `mean` / `std` / `enc_size` / `grid` / `embed_dim`.
    """

    def _features(self, x):                                   # pragma: no cover
        raise NotImplementedError

    def _build_head(self, num_classes: int = 1):
        # LINEAR ONLY, and that is not a style choice. The moment the head can
        # learn spatial structure it stops measuring the encoder and starts
        # measuring the head -- a strong decoder papers over a weak encoder well
        # enough to make every arm tie, which reads as "no difference between
        # corpora" when it means "the experiment could not see one".
        self.decode_head = LinearProbeHead(self.embed_dim, num_classes)

    def _register_scale(self, mean, std):
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    @property
    def backbone(self):
        return self.encoder

    def train(self, mode: bool = True):
        """A frozen encoder stays in eval() even when the wrapper trains.

        Dropout and stochastic depth would otherwise inject noise into features
        that are supposed to be FIXED, which is no longer a probe of the
        pretrained representation.
        """
        super().train(mode)
        if mode and getattr(self, "frozen", True):
            self.encoder.eval()
        return self

    def _freeze(self):
        self.frozen = True
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    def forward_train(self, x):
        size = x.shape[-2:]
        x = (x - self.mean) / self.std
        if x.shape[-1] != self.enc_size or x.shape[-2] != self.enc_size:
            x = F.interpolate(x, size=(self.enc_size, self.enc_size),
                              mode="bilinear", align_corners=False)
        if self.frozen:
            with torch.no_grad():
                feat = self._features(x)
            feat = feat.detach()
        else:
            feat = self._features(x)
        if feat.shape[-1] != self.grid or feat.shape[-2] != self.grid:
            raise RuntimeError(
                f"{type(self).__name__}: features are {tuple(feat.shape[-2:])}, "
                f"declared grid is {self.grid}x{self.grid}. Every number in this "
                f"stage is a comparison AT A FIXED GRID, so a mismatch here "
                f"invalidates the arm rather than merely degrading it.")
        logits = self.decode_head(feat.float())
        return F.interpolate(logits, size=size, mode="bilinear",
                             align_corners=False), None

    def forward(self, x):
        return self.forward_train(x)[0]


class HFViTProbe(_ProbeBase):
    """A HuggingFace ViT vision tower (DINOv2, SigLIP) probed at `grid`.

    `interpolate_pos_encoding=True` is passed when the encoder's forward accepts
    it -- SigLIP needs it to run at anything but its configured `image_size`, and
    DINOv2's embeddings interpolate unconditionally so the flag is a no-op there.
    Detected with `inspect.signature` rather than assumed, because passing an
    unsupported kwarg raises and hardcoding it for one vendor breaks the other.
    """

    def __init__(self, encoder, mean, std, grid: int = TARGET_GRID,
                 num_classes: int = 1):
        super().__init__()
        self.encoder = encoder
        cfg = encoder.config
        self.patch = int(getattr(cfg, "patch_size", 16))
        self.embed_dim = int(getattr(cfg, "hidden_size"))
        self.grid = int(grid)
        self.enc_size = self.grid * self.patch
        self._accepts_interp = "interpolate_pos_encoding" in inspect.signature(
            encoder.forward).parameters
        self._register_scale(mean, std)
        self._build_head(num_classes)
        self._freeze()
        self.init_source = ""

    def _features(self, x):
        kw = {"interpolate_pos_encoding": True} if self._accepts_interp else {}
        tok = self.encoder(pixel_values=x, **kw).last_hidden_state
        want = self.grid * self.grid
        if tok.shape[1] < want:
            raise RuntimeError(
                f"encoder returned {tok.shape[1]} tokens, need at least {want} for "
                f"a {self.grid}x{self.grid} grid at input {self.enc_size}. The "
                f"position encoding was not interpolated "
                f"(accepts_interp={self._accepts_interp}).")
        # CLS and register tokens are always a PREFIX, so the last grid*grid tokens
        # are the spatial ones for both families and no per-vendor branch is needed.
        tok = tok[:, tok.shape[1] - want:, :]
        b, _, c = tok.shape
        return tok.transpose(1, 2).reshape(b, c, self.grid, self.grid)


class OpenClipProbe(_ProbeBase):
    """An open_clip CLIP vision tower (DermLIP, BiomedCLIP, DermLIP-PanDerm).

    open_clip builds its vision tower one of two ways and BOTH are in the wild for
    the checkpoints this stage uses, so both are handled by explicit dispatch on
    what the object actually has:

      * a **timm** trunk at `visual.trunk` -- the tower is a timm VisionTransformer
        and `trunk.forward_features(x)` already returns [B, N, C].
      * open_clip's **native** VisionTransformer -- patch conv at `.conv1`,
        learned `.positional_embedding`, `.transformer`, `.ln_post`. Its own
        `forward` pools and projects, so the token path is reproduced here.

    Neither path is guessed at: `_setup` raises if it recognises neither, for the
    same reason `foundation._MODEL_TYPE_TO_CLASS` refuses to fall back to
    `AutoModel`. A tower that half-loads or silently returns pooled features would
    train, converge, and report a number -- which cost Stage N its entire
    attribution arm on 2026-08-06.
    """

    def __init__(self, visual, mean, std, grid: int = TARGET_GRID,
                 patch: int = 16, num_classes: int = 1):
        super().__init__()
        self.encoder = visual
        self.grid = int(grid)
        self.patch = int(patch)
        self.enc_size = self.grid * self.patch
        self._setup()
        self._register_scale(mean, std)
        self._build_head(num_classes)
        self._freeze()
        self.init_source = ""

    # ── which of the two towers is this? ─────────────────────────────────────
    def _setup(self):
        v = self.encoder
        if hasattr(v, "trunk"):
            self.mode = "timm"
            trunk = v.trunk
            self.embed_dim = int(getattr(trunk, "embed_dim", 0)) or \
                int(trunk.patch_embed.proj.out_channels)
            self.n_prefix = int(getattr(trunk, "num_prefix_tokens", 1))
            self._resample_timm(trunk)
        elif hasattr(v, "transformer") and hasattr(v, "positional_embedding"):
            self.mode = "native"
            self.embed_dim = int(v.positional_embedding.shape[-1])
            self.n_prefix = 1 if hasattr(v, "class_embedding") else 0
            with torch.no_grad():
                v.positional_embedding = nn.Parameter(
                    resample_pos_embed(v.positional_embedding.data, self.grid,
                                       self.n_prefix),
                    requires_grad=False)
        else:
            raise RuntimeError(
                f"unrecognised open_clip vision tower {type(v).__name__}: it has "
                f"neither a timm `.trunk` nor open_clip's native "
                f"`.transformer`/`.positional_embedding`. Add the case explicitly "
                f"rather than falling back to `.forward` -- open_clip's forward "
                f"returns POOLED features, which would silently give this arm a "
                f"1x1 grid and a meaningless score.")

    def _resample_timm(self, trunk):
        """Retarget a timm ViT to `self.enc_size`, preferring its own API.

        `set_input_size` exists in timm >= 1.0 and handles patch_embed bookkeeping
        that manual surgery would miss. When it is absent the fallback resamples
        `pos_embed` and relaxes `patch_embed`'s size assertion -- and then
        `_features` verifies the resulting grid anyway, so a fallback that did not
        work fails loudly instead of scoring low.
        """
        if hasattr(trunk, "set_input_size"):
            trunk.set_input_size(img_size=(self.enc_size, self.enc_size))
            return
        pe = getattr(trunk, "pos_embed", None)
        if pe is None:
            raise RuntimeError(
                "timm trunk has neither set_input_size nor pos_embed; cannot "
                "retarget it to a fixed grid.")
        with torch.no_grad():
            trunk.pos_embed = nn.Parameter(
                resample_pos_embed(pe.data, self.grid, self.n_prefix),
                requires_grad=False)
        pe_mod = trunk.patch_embed
        pe_mod.img_size = (self.enc_size, self.enc_size)
        pe_mod.grid_size = (self.grid, self.grid)
        pe_mod.num_patches = self.grid * self.grid
        pe_mod.strict_img_size = False

    # ── tokens ────────────────────────────────────────────────────────────────
    def _features(self, x):
        v = self.encoder
        if self.mode == "timm":
            tok = v.trunk.forward_features(x)                 # [B, N, C]
        else:
            tok = v.conv1(x)                                  # [B, C, g, g]
            b, c, gh, gw = tok.shape
            tok = tok.reshape(b, c, gh * gw).permute(0, 2, 1)
            if hasattr(v, "class_embedding"):
                cls = v.class_embedding.to(tok.dtype).reshape(1, 1, -1).expand(b, 1, c)
                tok = torch.cat([cls, tok], dim=1)
            tok = tok + v.positional_embedding.to(tok.dtype)
            if hasattr(v, "patch_dropout"):
                tok = v.patch_dropout(tok)
            if hasattr(v, "ln_pre"):
                tok = v.ln_pre(tok)
            tok = v.transformer(tok)
            if hasattr(v, "ln_post"):
                tok = v.ln_post(tok)
        want = self.grid * self.grid
        if tok.dim() != 3 or tok.shape[1] < want:
            raise RuntimeError(
                f"vision tower returned {tuple(tok.shape)}; expected [B, >={want}, C] "
                f"for a {self.grid}x{self.grid} grid. The resample did not take.")
        tok = tok[:, tok.shape[1] - want:, :]
        b, _, c = tok.shape
        return tok.transpose(1, 2).reshape(b, c, self.grid, self.grid)


class ResNetStageProbe(_ProbeBase):
    """ImageNet ResNet-50 truncated after `stage`, probed at a chosen input size.

    THE GRID-CALIBRATION ENCODER, and the reason this class takes `in_size` as a
    free parameter. Holding the weights, the depth and the recipe fixed while
    moving ONLY the input size moves ONLY the feature grid, so the Dice difference
    between two such arms is the grid effect with nothing else in it. That is the
    measurement handbook 7f.9 asks for and could not get from three encoders that
    differed in four ways at once.

        stage="layer2"  stride  8   in_size 224 -> 28x28
        stage="layer3"  stride 16   in_size 320/448/592/640 -> 20/28/37/40
        stage="layer4"  stride 32   in_size 640 -> 20x20   (Stage N's arm)
    """

    STRIDE = {"layer2": 8, "layer3": 16, "layer4": 32}
    CHANNELS = {"layer2": 512, "layer3": 1024, "layer4": 2048}

    def __init__(self, stage: str = "layer3", in_size: int = 448,
                 num_classes: int = 1, weights: str = "IMAGENET1K_V2"):
        super().__init__()
        if stage not in self.STRIDE:
            raise ValueError(f"stage must be one of {sorted(self.STRIDE)}, got {stage!r}")
        from torchvision.models import resnet50

        net = resnet50(weights=weights)
        # THE PARAMETER CHECK, done on the FULL trunk before truncation so it can
        # be compared against a single published figure. After truncation the
        # count depends on the stage and there is nothing to check it against.
        full = sum(p.numel() for p in net.parameters()) - \
            sum(p.numel() for p in net.fc.parameters())
        want = SOURCES["resnet50"].expected_params_M
        if abs(full / 1e6 - want) / want > 0.05:
            raise RuntimeError(
                f"resnet50 trunk has {full/1e6:.2f} M parameters, published "
                f"{want:.2f} M. torchvision returned something other than "
                f"ResNet-50; do not probe it.")

        keep = [net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2]
        if stage in ("layer3", "layer4"):
            keep.append(net.layer3)
        if stage == "layer4":
            keep.append(net.layer4)
        self.encoder = nn.Sequential(*keep)

        self.stage = stage
        self.enc_size = int(in_size)
        self.embed_dim = self.CHANNELS[stage]
        stride = self.STRIDE[stage]
        if self.enc_size % stride:
            raise ValueError(
                f"in_size {in_size} is not divisible by {stage}'s stride {stride}; "
                f"the grid would be non-integral and the arm would not be "
                f"grid-matched to anything.")
        self.grid = self.enc_size // stride

        self._register_scale(IMAGENET_MEAN, IMAGENET_STD)
        self._build_head(num_classes)
        self._freeze()
        self.init_source = (f"ResNet-50 ImageNet-1k supervised (torchvision), "
                            f"{stage} @ {in_size} -> {self.grid}x{self.grid}")

    def _features(self, x):
        return self.encoder(x)

    def train(self, mode: bool = True):
        # eval() also holds ResNet's BatchNorm running statistics fixed, which a
        # frozen encoder requires and `requires_grad_(False)` alone does not give.
        return super().train(mode)


class PixelProbe(_ProbeBase):
    """Raw RGB average-pooled to `grid`, then a 1x1 convolution. THE FLOOR.

    No pretraining, no encoder, three input channels. Whatever Dice this reaches
    at a given grid is reachable with ZERO learned features, purely from colour
    and the resolution at which a constant-per-cell mask can be drawn.

    This is the control that makes Experiment 1 interpretable rather than merely
    suggestive. The ResNet sweep says "grid is worth N Dice points to an ImageNet
    encoder"; the pixel sweep says "grid is worth M Dice points to nothing at
    all". If a corpus arm's advantage over another is smaller than (N - M), the
    encoders are not what is being compared.
    """

    def __init__(self, grid: int = TARGET_GRID, in_size: int = 448,
                 num_classes: int = 1):
        super().__init__()
        self.encoder = nn.Identity()          # named so `.backbone` still resolves
        self.grid = int(grid)
        self.enc_size = int(in_size)
        self.embed_dim = 3
        self._register_scale(IMAGENET_MEAN, IMAGENET_STD)
        self._build_head(num_classes)
        self.frozen = True
        self.init_source = f"raw RGB, adaptive-avg-pooled to {grid}x{grid} -- no pretraining"

    def _features(self, x):
        return F.adaptive_avg_pool2d(x, (self.grid, self.grid))


# ─────────────────────────────────────────────────────────────────────────────
# 4. the arms
# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 1 -- grid calibration. One encoder, one stage, four input sizes; plus
# the pixel floor at the same four grids. `layer4 @ 640` is included because it
# reproduces Stage N's ResNet arm exactly, which is what ties this curve back to
# the 7f.8 table it is meant to reinterpret.
CALIBRATION_ARCHS: dict[str, dict] = {
    "rn50_l3_g20": {"kind": "resnet", "stage": "layer3", "in_size": 320},
    "rn50_l3_g28": {"kind": "resnet", "stage": "layer3", "in_size": 448},
    "rn50_l3_g37": {"kind": "resnet", "stage": "layer3", "in_size": 592},
    "rn50_l3_g40": {"kind": "resnet", "stage": "layer3", "in_size": 640},
    "rn50_l4_g20": {"kind": "resnet", "stage": "layer4", "in_size": 640},
    "pixel_g20":   {"kind": "pixel", "grid": 20, "in_size": 640},
    "pixel_g28":   {"kind": "pixel", "grid": 28, "in_size": 448},
    "pixel_g37":   {"kind": "pixel", "grid": 37, "in_size": 592},
    "pixel_g40":   {"kind": "pixel", "grid": 40, "in_size": 640},
}

# EXPERIMENT 2 -- the corpus arms. EVERY ONE AT 28x28. That is the whole design:
# the nuisance variable is pinned and only the pretraining corpus varies.
CORPUS_ARCHS: dict[str, dict] = {
    "dermlip_g28":         {"kind": "openclip", "encoder": "dermlip"},
    "dermlip_panderm_g28": {"kind": "openclip", "encoder": "dermlip_panderm"},
    "biomedclip_g28":      {"kind": "openclip", "encoder": "biomedclip"},
    "dinov2_g28":          {"kind": "hf", "encoder": "dinov2"},
    "medsiglip_g28":       {"kind": "hf", "encoder": "medsiglip"},
    "rn50_g28":            {"kind": "resnet", "stage": "layer3", "in_size": 448},
}

DERM_ARCHS = tuple(CORPUS_ARCHS)          # everything this module can build
ALL_ARCHS: dict[str, dict] = {**CALIBRATION_ARCHS, **CORPUS_ARCHS}

CALIBRATION_ARMS = tuple(CALIBRATION_ARCHS)
CORPUS_ARMS = tuple(CORPUS_ARCHS)

# ── THE PRE-REGISTERED GATE, fixed here before any number was produced ────────
# The primary treatment is DermLIP BY NAME, not "the best derm arm". Picking the
# better of two arms after seeing them and then testing it is the oldest way to
# manufacture a significant result, and this study already refuses the equivalent
# move on seeds (handbook 15, trap 3).
GATE_TREATMENT = "dermlip_g28"                 # dermatology image-text
GATE_GENERIC = "dinov2_g28"                    # modern SSL, natural images
GATE_MEDICAL = "medsiglip_g28"                 # general-medical image-text
GATE_IMAGENET = "rn50_g28"                     # ImageNet supervised
GATE_BIOMED = "biomedclip_g28"                 # non-derm biomedical image-text
GATE_SECONDARY = ("dermlip_panderm_g28",)      # reported, never promoted
GATE_MARGIN = 0.01                             # the margin Stages C, M and N used

# The grid every corpus arm must report, asserted rather than trusted.
GATE_REQUIRED_GRID = TARGET_GRID


def _encoder_mean_std(key: str):
    if key == "medsiglip":
        return SIGLIP_MEAN, SIGLIP_STD
    if SOURCES[key].loader == "open_clip":
        return OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
    return IMAGENET_MEAN, IMAGENET_STD


def _load_open_clip_visual(env, key: str, max_missing: int = 0):
    """Build the VISION TOWER ONLY from a local open_clip checkpoint, offline.

    WHY NOT `create_model_and_transforms`, WHICH IS WHAT EVERY MODEL CARD SHOWS
    ----------------------------------------------------------------------------
    Two reasons, and the first one is fatal rather than merely inconvenient.

    1. `hf-hub:<path>` DOES NOT ACCEPT A LOCAL DIRECTORY. Everything after the
       prefix is validated as a HuggingFace repo id, so a filesystem path fails
       with "Repo id must be in the form 'repo_name' or 'namespace/repo_name'".
       There is no local-directory form of that call.
    2. BUILDING THE FULL CLIP WOULD GO TO THE NETWORK. BiomedCLIP's `text_cfg`
       names `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract` and
       DermLIP-PanDerm's names `neuml/pubmedbert-base-embeddings`, both of which
       open_clip resolves through `transformers` at construction time. Under
       HF_HUB_OFFLINE=1 that raises; without it, it silently downloads a text
       encoder this stage never uses. Nothing here builds a text prompt.

    So the tower is constructed directly from `open_clip_config.json`'s
    `vision_cfg` and only the `visual.*` keys are loaded out of the checkpoint.
    That is strictly less machinery than the full path, not more: no tokenizer, no
    text tower, no hub call, and the load is verifiable key by key.

    THE VERIFICATION, which the model-card path does not give you
    --------------------------------------------------------------
    `load_state_dict(strict=False)` returns the missing and unexpected key lists
    and anything beyond `max_missing` RAISES. A tower whose weights did not land
    must fail here, loudly, and not forty minutes later as a plausible-looking
    Dice number -- handbook 7f.7a is what happens when it does not.

    THE UNKNOWN-KEY STRIP, which is not defensive programming
    ----------------------------------------------------------
    `redlessone/DermLIP_PanDerm-base-w-PubMed-256` ships `"pretrain_path": null`
    inside `vision_cfg`, and `CLIPVisionCfg` has no such field -- so
    `CLIPVisionCfg(**vision_cfg)` raises TypeError on a checkpoint that is
    otherwise completely fine. Unknown keys are dropped AND REPORTED, never
    dropped silently: a key that changes the architecture would be dropped by the
    same line, and the printout is what lets a reader notice.
    """
    import dataclasses

    import open_clip                                           # noqa: F401
    from open_clip.model import CLIPVisionCfg, _build_vision_tower

    src = SOURCES[key]
    path = Path(env.weights) / "foundation" / src.local_dir
    cfg_file = path / "open_clip_config.json"
    if not cfg_file.exists():
        raise FileNotFoundError(
            f"{key}: no open_clip_config.json at {path}\n"
            f"  A partial `hf download` is the usual cause -- check the directory "
            f"is not empty and re-download if it is.\n\n{download_instructions(env)}")

    blob = json.loads(cfg_file.read_text())
    model_cfg = blob.get("model_cfg", blob)
    if "vision_cfg" not in model_cfg or "embed_dim" not in model_cfg:
        raise RuntimeError(
            f"{key}: {cfg_file} has no model_cfg.vision_cfg / model_cfg.embed_dim. "
            f"This is not an open_clip checkpoint directory.")

    vcfg = dict(model_cfg["vision_cfg"])
    known = {f.name for f in dataclasses.fields(CLIPVisionCfg)}
    dropped = {k: vcfg.pop(k) for k in list(vcfg) if k not in known}
    if dropped:
        print(f"  {key}: vision_cfg keys unknown to this open_clip and DROPPED: "
              f"{dropped} -- confirm none of them changes the architecture")

    try:
        visual = _build_vision_tower(
            int(model_cfg["embed_dim"]), CLIPVisionCfg(**vcfg),
            quick_gelu=bool(model_cfg.get("quick_gelu", False)))
    except Exception as e:                                     # noqa: BLE001
        raise RuntimeError(
            f"{key}: open_clip {open_clip.__version__} could not build the vision "
            f"tower from {cfg_file} ({type(e).__name__}: {e}).\n"
            f"  vision_cfg = {vcfg}\n"
            f"  Do NOT hand-write a replacement architecture -- a tower that is "
            f"close but not identical loads most keys, trains, and reports a "
            f"number for a model that is not the one in the table."
        ) from e

    # The checkpoint. Both filenames are in the wild for open_clip repos.
    ckpt = next((path / n for n in ("open_clip_pytorch_model.safetensors",
                                    "open_clip_pytorch_model.bin",
                                    "model.safetensors",
                                    "pytorch_model.bin")
                 if (path / n).exists()), None)
    if ckpt is None:
        raise FileNotFoundError(
            f"{key}: config.json is present at {path} but no weights file is.\n"
            f"  Looked for open_clip_pytorch_model.safetensors/.bin, "
            f"model.safetensors, pytorch_model.bin.\n"
            f"  Found: {sorted(p.name for p in path.iterdir())[:12]}\n"
            f"  Re-run the `hf download` for this repo.")

    if ckpt.suffix == ".safetensors":
        from safetensors.torch import load_file
        sd = load_file(str(ckpt))
    else:
        sd = torch.load(str(ckpt), map_location="cpu", weights_only=True)
    sd = sd.get("state_dict", sd)
    sd = {k[len("module."):] if k.startswith("module.") else k: v
          for k, v in sd.items()}

    vsd = {k[len("visual."):]: v for k, v in sd.items() if k.startswith("visual.")}
    if not vsd:
        raise RuntimeError(
            f"{key}: {ckpt.name} contains no `visual.*` keys, so there is no "
            f"vision tower to load. Keys start with: "
            f"{sorted({k.split('.')[0] for k in sd})[:8]}")

    info = visual.load_state_dict(vsd, strict=False)
    missing, unexpected = list(info.missing_keys), list(info.unexpected_keys)
    if len(missing) > max_missing:
        raise RuntimeError(
            f"{key}: built the tower from {cfg_file.name} but {len(missing)} of "
            f"its parameters were NOT in {ckpt.name} -- they are RANDOM.\n"
            f"  e.g. {missing[:4]}\n"
            f"  This is the failure that produced a random-weights DINOv2 arm on "
            f"2026-08-06 (handbook 7f.7a). Do not train on this.")
    if unexpected:
        print(f"  {key}: {len(unexpected)} checkpoint keys unused by the tower, "
              f"e.g. {unexpected[:3]}")

    n_par = sum(p.numel() for p in visual.parameters()) / 1e6
    print(f"  {key}: {type(visual).__name__} from {ckpt.name}, "
          f"{len(vsd)} tensors, 0 missing, {n_par:.1f} M params")
    return visual


def build_arm(env, arch: str, num_classes: int = 1, verbose: bool = True):
    """Construct one arm. Raises on a missing encoder -- never falls back to random.

    Two checks run on every pretrained arm before it is returned, and they are the
    two that Stage N's 2026-08-06 failure needed:

      1. THE LOAD. Handled inside the loader -- HF via `output_loading_info`,
         open_clip by raising on any construction error.
      2. THE SIZE. The built encoder's parameter count against a published figure.
         This is the check a key-count cannot make: a body assembled by the wrong
         class from the right config loads zero keys, has plausible dimensions,
         trains happily, and is simply not the model named in the table.
    """
    if arch not in ALL_ARCHS:
        raise ValueError(f"unknown arch {arch!r}; choices: {sorted(ALL_ARCHS)}")
    spec = ALL_ARCHS[arch]
    kind = spec["kind"]

    if kind == "pixel":
        m = PixelProbe(grid=spec["grid"], in_size=spec["in_size"],
                       num_classes=num_classes)
        key = None
    elif kind == "resnet":
        m = ResNetStageProbe(stage=spec["stage"], in_size=spec["in_size"],
                             num_classes=num_classes)
        key = "resnet50"
    elif kind == "hf":
        key = spec["encoder"]
        src = SOURCES[key]
        path = Path(env.weights) / "foundation" / src.local_dir
        enc = _load_vision_encoder(path, key)
        mean, std = _encoder_mean_std(key)
        m = HFViTProbe(enc, mean, std, grid=TARGET_GRID, num_classes=num_classes)
        m.init_source = src.init
    elif kind == "openclip":
        key = spec["encoder"]
        src = SOURCES[key]
        visual = _load_open_clip_visual(env, key)
        mean, std = _encoder_mean_std(key)
        m = OpenClipProbe(visual, mean, std, grid=TARGET_GRID, patch=src.patch,
                          num_classes=num_classes)
        m.init_source = src.init
    else:                                                     # pragma: no cover
        raise ValueError(f"unknown arm kind {kind!r}")

    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    enc_M = sum(p.numel() for p in m.backbone.parameters()) / 1e6

    if key is not None and kind != "resnet":       # resnet checks its own trunk
        want = SOURCES[key].expected_params_M
        tol = SOURCES[key].param_tol
        if want and abs(enc_M - want) / want > tol:
            raise RuntimeError(
                f"{arch}: encoder has {enc_M:.2f} M parameters, published is "
                f"{want:.2f} M ({100*(enc_M-want)/want:+.1f} %, tolerance "
                f"{100*tol:.0f} %). The architecture built is not the one named in "
                f"the table. Do not train on this.")

    if verbose:
        extra = f"mode={m.mode}" if hasattr(m, "mode") else ""
        print(f"  {arch:<22} {enc_M:7.1f} M encoder  grid={m.grid}x{m.grid}  "
              f"input={m.enc_size}  embed={m.embed_dim:<5} "
              f"trainable={trainable/1e3:6.1f} k ({100*trainable/max(total,1):.4f} %) "
              f"{extra}")
    return m


def install_derm_shim(env, verbose: bool = True):
    """Teach `engine.train_run`'s `build_model` about these arms.

    Reassigns the name in BOTH `bruisekit.models` and `bruisekit.engine` because
    engine bound `build_model` by value at import time, so patching models alone
    leaves the training loop calling the original -- the same note
    `foundation.install_foundation_shim` carries. Any arch this shim does not
    recognise falls through untouched, INCLUDING Stage N's, so the two overlays
    can be installed in the same kernel.
    """
    import bruisekit.engine as _be
    import bruisekit.models as _bm

    previous = _bm.build_model

    def build_model(arch, size, paths):
        if arch in ALL_ARCHS:
            return build_arm(env, arch, num_classes=1, verbose=verbose)
        return previous(arch, size, paths)

    _bm.build_model = build_model
    _be.build_model = build_model

    previous_probe = _be.resolve_micro_batch

    def resolve_micro_batch(model, cfg, device, teacher=None):
        """Fixed micro-batch with accumulation, never the VRAM probe.

        Every arm here is FROZEN with a single 1x1 convolution trainable, so the
        probe's forward/backward escalation from batch 1 would spend minutes
        rediscovering an answer that is already known. `micro * accum` is pinned
        to 16 so the effective batch, the optimizer-step count and the LR schedule
        match the SMP and mobile baselines exactly (handbook 3).
        """
        if isinstance(model, _ProbeBase):
            micro = int(cfg.get("derm_probe_micro_batch", 8))
            target = int(cfg.get("derm_effective_batch", 16))
            micro = max(1, min(micro, target))
            while target % micro:
                micro -= 1
            return micro, target // micro
        return previous_probe(model, cfg, device, teacher)

    _be.resolve_micro_batch = resolve_micro_batch
    return build_model


# ─────────────────────────────────────────────────────────────────────────────
# 5. EXPERIMENT 1 -- how much Dice is the grid worth?
# ─────────────────────────────────────────────────────────────────────────────
def arm_grid(arch: str) -> int:
    """The declared feature grid of an arm, without building it.

    Read from the spec rather than from a name suffix: the suffix is a label and
    labels drift, whereas `in_size // stride` is the thing the model will actually
    produce and `_ProbeBase.forward_train` asserts against.
    """
    spec = ALL_ARCHS[arch]
    if spec["kind"] == "pixel":
        return int(spec["grid"])
    if spec["kind"] == "resnet":
        return int(spec["in_size"]) // ResNetStageProbe.STRIDE[spec["stage"]]
    return TARGET_GRID


def grid_calibration(val_tables: dict[str, pd.DataFrame],
                     n_boot: int = 10000, seed: int = 0) -> dict:
    """Fit Dice against log2(grid) for the two calibration families.

    Returns the SLOPE in Dice per doubling of grid, separately for:

      * `resnet` -- an ImageNet encoder whose weights and depth never change, so
        the slope is the grid effect ON REAL FEATURES.
      * `pixel`  -- no encoder at all, so the slope is what the grid buys with
        nothing to say.

    Bootstrapped over SUBJECTS, not images, matching every other interval in this
    study: images from one subject are not independent and an image-level
    resample would give intervals that are too narrow by roughly sqrt(images per
    subject).

    THE NUMBER THIS EXISTS TO PRODUCE is `resnet_dice_20_to_37` -- what an
    ImageNet ResNet-50 gains purely by going from Stage N's 20x20 to DINOv2's
    37x37. Handbook 7f.8 reports that pair as `dinov2 - resnet50 = +0.534` and
    attributes all of it to pretraining. Whatever fraction of it this number
    covers was never pretraining at all.
    """
    rng = np.random.default_rng(seed)
    out: dict = {"n_boot": n_boot, "families": {}}

    for family, prefix in (("resnet", "rn50_l3_"), ("pixel", "pixel_")):
        arms = [a for a in sorted(val_tables) if a.startswith(prefix)]
        if len(arms) < 2:
            out["families"][family] = {"arms": arms, "note": "fewer than 2 arms; skipped"}
            continue

        grids = np.array([arm_grid(a) for a in arms], float)
        x = np.log2(grids)
        # One row per subject per arm, so a resample draws whole subjects and the
        # per-arm means move together -- which is what makes the slope's interval
        # paired rather than independent across arms.
        per_subject = {a: val_tables[a].groupby("subject").dice.mean() for a in arms}
        subjects = sorted(set.intersection(*(set(s.index) for s in per_subject.values())))
        mat = np.array([[per_subject[a][s] for s in subjects] for a in arms])  # [arm, subj]

        point = np.polyfit(x, mat.mean(axis=1), 1)[0]
        boots = np.empty(n_boot)
        n = len(subjects)
        for i in range(n_boot):
            idx = rng.integers(0, n, n)
            boots[i] = np.polyfit(x, mat[:, idx].mean(axis=1), 1)[0]
        lo, hi = np.percentile(boots, [2.5, 97.5])

        out["families"][family] = {
            "arms": arms,
            "grids": grids.tolist(),
            "mean_dice": mat.mean(axis=1).round(4).tolist(),
            "n_subjects": int(n),
            "slope_dice_per_doubling": float(point),
            "slope_ci95": [float(lo), float(hi)],
        }

    # The single number the write-up needs: the 20 -> 37 span, in Dice, for the
    # ImageNet encoder. Extrapolated from the fitted slope AND, where the arms
    # exist, read off directly -- both are reported because a straight line in
    # log2(grid) is an assumption, and the direct read is not.
    rf = out["families"].get("resnet", {})
    if "slope_dice_per_doubling" in rf:
        span = math.log2(37) - math.log2(20)
        out["resnet_dice_20_to_37_fitted"] = float(rf["slope_dice_per_doubling"] * span)
        out["resnet_dice_20_to_37_ci95"] = [float(rf["slope_ci95"][0] * span),
                                            float(rf["slope_ci95"][1] * span)]
    if "rn50_l3_g20" in val_tables and "rn50_l3_g37" in val_tables:
        out["resnet_dice_20_to_37_measured"] = float(
            val_tables["rn50_l3_g37"].dice.mean() - val_tables["rn50_l3_g20"].dice.mean())

    # Stage N's own ResNet arm, reproduced, against the same encoder one stage
    # earlier. Same weights, same input, ONLY the stride changes -- so this pair
    # is the cleanest single statement of the confound there is.
    if "rn50_l4_g20" in val_tables and "rn50_l3_g40" in val_tables:
        out["stage_n_arm_dice"] = float(val_tables["rn50_l4_g20"].dice.mean())
        out["same_input_finer_stage_dice"] = float(val_tables["rn50_l3_g40"].dice.mean())
        out["stride_only_delta"] = (out["same_input_finer_stage_dice"]
                                    - out["stage_n_arm_dice"])
    return out


def print_calibration(cal: dict) -> None:
    """The grid curve, in the shape a reader can paste into a meeting note."""
    print("─" * 78)
    print("EXPERIMENT 1 -- GRID CALIBRATION")
    print("  one encoder, one stage, four input sizes: only the grid changes.")
    print("─" * 78)
    for family, f in cal.get("families", {}).items():
        if "slope_dice_per_doubling" not in f:
            print(f"  {family:<8} {f.get('note', 'skipped')}")
            continue
        lo, hi = f["slope_ci95"]
        print(f"\n  {family.upper()}   ({f['n_subjects']} subjects)")
        for a, g, d in zip(f["arms"], f["grids"], f["mean_dice"]):
            print(f"      {a:<16} grid {int(g):>3}x{int(g):<3}  val Dice {d:.4f}")
        print(f"      slope: {f['slope_dice_per_doubling']:+.4f} Dice per DOUBLING "
              f"of grid   CI [{lo:+.4f}, {hi:+.4f}]")

    if "resnet_dice_20_to_37_fitted" in cal:
        lo, hi = cal["resnet_dice_20_to_37_ci95"]
        print(f"\n  20x20 -> 37x37 on the SAME ImageNet encoder:")
        print(f"      fitted   {cal['resnet_dice_20_to_37_fitted']:+.4f} "
              f"CI [{lo:+.4f}, {hi:+.4f}]")
        if "resnet_dice_20_to_37_measured" in cal:
            print(f"      measured {cal['resnet_dice_20_to_37_measured']:+.4f} "
                  f"(direct, no line fitted)")
        print(f"\n  Handbook 7f.8 reports dinov2 - resnet50 = +0.5342 and reads all")
        print(f"  of it as pretraining. Those two arms differed by 37x37 vs 20x20.")
        print(f"  The number above is how much of that gap the GRID alone explains.")

    if "stride_only_delta" in cal:
        print(f"\n  Same weights, same 640 input, ONLY the stride changes:")
        print(f"      layer4 (20x20, = Stage N's arm)  {cal['stage_n_arm_dice']:.4f}")
        print(f"      layer3 (40x40)                   {cal['same_input_finer_stage_dice']:.4f}")
        print(f"      delta                            {cal['stride_only_delta']:+.4f}")
    print("─" * 78)


# ─────────────────────────────────────────────────────────────────────────────
# 6. EXPERIMENT 2 -- THE GATE
# ─────────────────────────────────────────────────────────────────────────────
def corpus_gate(val_tables: dict[str, pd.DataFrame], n_boot: int = 10000,
                seed: int = 0, margin: float = GATE_MARGIN,
                calibration: dict | None = None) -> dict:
    """Does DERMATOLOGY pretraining beat generic pretraining at a MATCHED grid?

    THE RULE, fixed in this module before any number was produced:

        open  iff  the (dermlip_g28 - dinov2_g28) val-Dice CI clears zero

    `dermlip_g28` is the treatment BY NAME. The second dermatology arm is
    secondary and cannot be promoted into the primary slot after the fact -- that
    substitution is how a two-arm experiment becomes a one-arm experiment with two
    chances, and this study refuses the equivalent move on seeds already.

    Reported ALONGSIDE and deliberately NOT ANDed in -- the same policy Stage C's
    6.2, Stage M's 7e.3 and Stage N's 7f.4 all use:

      - `vs_medical`   dermlip - medsiglip. Is a SKIN corpus better than a
                       GENERAL-HOSPITAL corpus? This is the contrast Stage N
                       thought it was running and was not.
      - `vs_biomed`    dermlip - biomedclip. Architecture-matched to the patch,
                       same loader, same resample: the corpus is the only thing
                       that differs. The cleanest contrast in the pool.
      - `vs_imagenet`  dermlip - rn50_g28. Ties the whole stage back to Stage B.
      - misses         the endpoint handbook 1 says decides.

    WHY VALIDATION AND NOT TEST. A gate scored on test is a decision taken on the
    data the paper reports. The 134 val images are the ones every operating point
    in this study is already fitted on, so this spends no new information.
    """
    from bruisekit.significance import paired_contrast_multi

    for need in (GATE_TREATMENT, GATE_GENERIC):
        if need not in val_tables:
            raise KeyError(f"corpus_gate needs {need!r}; have {sorted(val_tables)}")

    # THE GRID ASSERTION. The entire premise of this stage is that every corpus
    # arm sits at the same grid; if that silently stopped being true the gate
    # would reproduce the confound it exists to close, wearing a new label.
    bad = {a: arm_grid(a) for a in val_tables
           if a in CORPUS_ARCHS and arm_grid(a) != GATE_REQUIRED_GRID}
    if bad:
        raise RuntimeError(
            f"corpus arms are NOT grid-matched: {bad} against the required "
            f"{GATE_REQUIRED_GRID}. This stage's only claim is a comparison at a "
            f"FIXED grid; refusing to score it.")

    main = paired_contrast_multi(val_tables[GATE_TREATMENT], val_tables[GATE_GENERIC],
                                 GATE_TREATMENT, GATE_GENERIC,
                                 n_boot=n_boot, seed=seed, margin=margin)

    per_arm = {}
    for k, v in sorted(val_tables.items()):
        per_arm[k] = {
            "grid": arm_grid(k),
            "mean_dice": float(v.dice.mean()),
            "median_dice": float(v.dice.median()),
            "iqr_dice": float(v.dice.quantile(0.75) - v.dice.quantile(0.25)),
            "misses": int(v.complete_miss.sum()),
            "n_images": int(len(v)),
        }

    out = {
        "n_val_images": int(len(val_tables[GATE_GENERIC])),
        "n_subjects": int(main["n_subjects"]),
        "required_grid": GATE_REQUIRED_GRID,
        "grid_matched": True,
        "per_arm": per_arm,
        "treatment": GATE_TREATMENT, "control": GATE_GENERIC,
        "delta_dice": main["delta_dice"],
        "delta_dice_ci95": [main["lo"], main["hi"]],
        "p_treatment_better": main["p_a_better"],
        "p_two_sided": main["p_two_sided"],
        "verdict_vs_generic": main["verdict"],
        "delta_miss_rate": main["delta_miss_rate"],
        "delta_miss_ci95": [main["miss_lo"], main["miss_hi"]],
        "p_treatment_fewer_misses": main["p_a_fewer_misses"],
        "margin": margin, "n_boot": n_boot,
    }

    contrasts = {}
    for label, other in (("vs_medical", GATE_MEDICAL),
                         ("vs_biomed", GATE_BIOMED),
                         ("vs_imagenet", GATE_IMAGENET)):
        if other in val_tables:
            r = paired_contrast_multi(val_tables[GATE_TREATMENT], val_tables[other],
                                      GATE_TREATMENT, other,
                                      n_boot=n_boot, seed=seed, margin=margin)
            contrasts[label] = {"against": other, "delta_dice": r["delta_dice"],
                                "ci95": [r["lo"], r["hi"]],
                                "p_two_sided": r["p_two_sided"],
                                "verdict": r["verdict"],
                                "delta_miss_rate": r["delta_miss_rate"]}
    out["contrasts"] = contrasts

    # Secondary derm arms: reported against the SAME generic control so they are
    # readable next to the primary, and flagged so nobody quotes one as the result.
    sec = {}
    for a in GATE_SECONDARY:
        if a in val_tables:
            r = paired_contrast_multi(val_tables[a], val_tables[GATE_GENERIC],
                                      a, GATE_GENERIC, n_boot=n_boot, seed=seed,
                                      margin=margin)
            sec[a] = {"delta_dice": r["delta_dice"], "ci95": [r["lo"], r["hi"]],
                      "p_two_sided": r["p_two_sided"], "verdict": r["verdict"],
                      "SECONDARY": "not the pre-registered treatment; do not promote"}
    out["secondary"] = sec

    # ── what Experiment 1 does to Stage N's headline ──────────────────────────
    if calibration:
        out["grid_calibration"] = {
            "resnet_slope_per_doubling":
                calibration.get("families", {}).get("resnet", {})
                .get("slope_dice_per_doubling"),
            "pixel_slope_per_doubling":
                calibration.get("families", {}).get("pixel", {})
                .get("slope_dice_per_doubling"),
            "resnet_dice_20_to_37": calibration.get("resnet_dice_20_to_37_measured",
                                                    calibration.get("resnet_dice_20_to_37_fitted")),
        }
        g = out["grid_calibration"]["resnet_dice_20_to_37"]
        if g is not None:
            # Handbook 7f.8's headline, and how much of it the grid explains. Not
            # a correction applied to anything -- the corpus arms above are already
            # grid-matched and need none. This line exists so the OLD number can be
            # re-read honestly rather than quietly dropped.
            out["stage_n_reinterpretation"] = {
                "published_dinov2_minus_resnet50": 0.5342,
                "grid_explained": float(g),
                "residual_attributable_to_pretraining": float(0.5342 - g),
                "grid_share": float(g / 0.5342),
            }

    lo = main["lo"]
    out["GATE_derm_beats_generic"] = bool(lo > 0)
    out["GATE_derm_beats_medical"] = bool(
        contrasts.get("vs_medical", {}).get("ci95", [-1, 1])[0] > 0)
    out["GATE_miss_improved"] = bool(main["miss_hi"] < 0)
    out["GATE_run_seg_arms"] = bool(lo > 0)
    return out


def print_gate(gate: dict) -> None:
    """The verdict, in the shape a reader can paste into a meeting note."""
    print("─" * 78)
    print("EXPERIMENT 2 -- CORPUS GATE")
    print(f"  134 validation images, frozen encoders, linear head, "
          f"ALL ARMS AT {gate['required_grid']}x{gate['required_grid']}")
    print("─" * 78)
    print(f"{'arm':<24}{'grid':>6}{'mean Dice':>11}{'median':>9}{'IQR':>8}{'misses':>8}")
    for k, v in sorted(gate["per_arm"].items(),
                       key=lambda kv: -kv[1]["mean_dice"]):
        print(f"{k:<24}{v['grid']:>6}{v['mean_dice']:>11.4f}{v['median_dice']:>9.4f}"
              f"{v['iqr_dice']:>8.3f}{v['misses']:>8d}")

    lo, hi = gate["delta_dice_ci95"]
    print(f"\n  PRIMARY  {gate['treatment']} - {gate['control']}")
    print(f"    delta Dice      {gate['delta_dice']:+.4f}   CI [{lo:+.4f}, {hi:+.4f}]"
          f"   P(better) = {gate['p_treatment_better']:.3f}")
    mlo, mhi = gate["delta_miss_ci95"]
    print(f"    delta miss rate {gate['delta_miss_rate']:+.4f}   "
          f"CI [{mlo:+.4f}, {mhi:+.4f}]   (lower is better)")

    if gate.get("contrasts"):
        print("\n  REPORTED ALONGSIDE (not ANDed into the verdict)")
        for label, c in gate["contrasts"].items():
            clo, chi = c["ci95"]
            print(f"    {label:<13} vs {c['against']:<22} {c['delta_dice']:+.4f}  "
                  f"CI [{clo:+.4f}, {chi:+.4f}]  {c['verdict']}")

    if gate.get("secondary"):
        print("\n  SECONDARY derm arms -- reported, NEVER promoted to primary")
        for a, s in gate["secondary"].items():
            slo, shi = s["ci95"]
            print(f"    {a:<24} vs generic {s['delta_dice']:+.4f}  "
                  f"CI [{slo:+.4f}, {shi:+.4f}]")

    if "stage_n_reinterpretation" in gate:
        r = gate["stage_n_reinterpretation"]
        print("\n  WHAT EXPERIMENT 1 DOES TO HANDBOOK 7f.8")
        print(f"    published  dinov2 - resnet50   {r['published_dinov2_minus_resnet50']:+.4f}")
        print(f"    grid alone explains            {r['grid_explained']:+.4f}"
              f"  ({r['grid_share']:.0%} of it)")
        print(f"    residual for pretraining       "
              f"{r['residual_attributable_to_pretraining']:+.4f}")
        if r["grid_share"] > 0.5:
            print("    -> MOST of 7f.8's headline was feature-grid resolution.")
            print("       7f.8 must be rewritten before it is quoted anywhere.")

    print()
    print(f"  GATE_derm_beats_generic : {gate['GATE_derm_beats_generic']}")
    print(f"  GATE_derm_beats_medical : {gate['GATE_derm_beats_medical']}  "
          f"(reported, NOT ANDed)")
    print(f"  GATE_miss_improved      : {gate['GATE_miss_improved']}  "
          f"(reported, NOT ANDed)")
    print("─" * 78)
    if gate["GATE_run_seg_arms"]:
        print("VERDICT: OPEN -- dermatology-pretrained features beat generic")
        print("         self-supervised features at a matched grid. Train the seg")
        print("         arms (Stage N notebook 10 onward, new encoder key).")
    else:
        print("VERDICT: CLOSED -- dermatology pretraining does not beat generic")
        print("         self-supervised pretraining on this task, at a matched grid,")
        print("         with the corpus confound closed. THAT IS THE RESULT, and it")
        print("         is a much stronger claim than 7f.8 could make: it now names")
        print("         a real dermatology corpus rather than a general-medical one,")
        print("         and it cannot be explained by feature-grid resolution.")
    print("─" * 78)


# ─────────────────────────────────────────────────────────────────────────────
# 7. self test
# ─────────────────────────────────────────────────────────────────────────────
def self_test(verbose: bool = True) -> bool:
    """Check the things that would silently invalidate every downstream number.

    1. THE RESAMPLE. A resampled position embedding must have exactly
       n_prefix + grid^2 tokens, must leave the prefix untouched, and must be a
       no-op when the grid already matches. An off-by-one in the prefix count
       shifts every embedding by one patch and degrades an arm invisibly.
    2. THE INTERFACE. Every arm returns (logits[B,1,640,640], None) from
       forward_train and the identical tensor from forward.
    3. THE GRID. Every arm's features come out at its DECLARED grid. This stage's
       only claim is a comparison at a fixed grid, so an arm that quietly runs at
       a different one is worse than an arm that crashes.
    4. THE FREEZE. Zero trainable encoder parameters on every arm.

    Runs on stubs and on the real `PixelProbe`, so it needs no downloaded weights
    and no GPU -- it tests this module's wiring, not the vendors'.
    """
    ok = True

    def check(name, good, detail=""):
        nonlocal ok
        ok &= bool(good)
        if verbose:
            print(f"  {name:<44} {detail:<30} {'OK' if good else 'FAIL'}")

    # ── 1. the resample ──────────────────────────────────────────────────────
    pos = torch.randn(1, 1 + 14 * 14, 32)
    r = resample_pos_embed(pos, 28, n_prefix=1)
    check("resample 14->28 token count", tuple(r.shape) == (1, 1 + 28 * 28, 32),
          str(tuple(r.shape)))
    check("resample leaves the prefix untouched",
          torch.allclose(r[:, :1], pos[:, :1]))
    same = resample_pos_embed(pos, 14, n_prefix=1)
    check("resample to the same grid is a no-op", torch.equal(same, pos))
    try:
        resample_pos_embed(torch.randn(1, 200, 32), 28, n_prefix=1)
        check("resample raises on a non-square token count", False)
    except ValueError:
        check("resample raises on a non-square token count", True)

    # ── 2-4. the arms ────────────────────────────────────────────────────────
    x = torch.rand(2, 3, 640, 640)

    for grid, in_size in ((20, 640), (28, 448), (37, 592)):
        m = PixelProbe(grid=grid, in_size=in_size).eval()
        with torch.no_grad():
            logits, aux = m.forward_train(x)
            plain = m(x)
        check(f"PixelProbe g{grid}",
              tuple(logits.shape) == (2, 1, 640, 640) and aux is None
              and torch.equal(plain, logits),
              f"shape={tuple(logits.shape)}")

    class _StubCfg:
        hidden_size, patch_size, image_size = 48, 16, 448

    class _StubHFEncoder(nn.Module):
        """A ViT-shaped stand-in that emits grid^2 tokens plus a CLS prefix."""

        def __init__(self):
            super().__init__()
            self.config = _StubCfg()
            self.embed = nn.Conv2d(3, 48, 16, stride=16)

        def forward(self, pixel_values, interpolate_pos_encoding=False):
            t = self.embed(pixel_values).flatten(2).transpose(1, 2)
            out = type("O", (), {})()
            out.last_hidden_state = torch.cat([t[:, :1] * 0, t], dim=1)
            return out

    m = HFViTProbe(_StubHFEncoder(), IMAGENET_MEAN, IMAGENET_STD, grid=28).eval()
    with torch.no_grad():
        logits, aux = m.forward_train(x)
    enc_trainable = sum(p.numel() for p in m.encoder.parameters() if p.requires_grad)
    check("HFViTProbe interface + grid + freeze",
          tuple(logits.shape) == (2, 1, 640, 640) and aux is None
          and m.grid == 28 and m.enc_size == 448 and enc_trainable == 0,
          f"grid={m.grid} input={m.enc_size} enc_trainable={enc_trainable}")
    check("HFViTProbe detects interpolate_pos_encoding", m._accepts_interp)

    m2 = HFViTProbe(_StubHFEncoder(), IMAGENET_MEAN, IMAGENET_STD, grid=28).train()
    check("train() keeps a frozen encoder in eval",
          (not m2.encoder.training) and m2.decode_head.training)

    # An arm whose features come out at the wrong grid must RAISE, not score low.
    class _WrongGrid(_StubHFEncoder):
        def forward(self, pixel_values, interpolate_pos_encoding=False):
            out = super().forward(pixel_values)
            out.last_hidden_state = out.last_hidden_state[:, :50]
            return out

    m3 = HFViTProbe(_WrongGrid(), IMAGENET_MEAN, IMAGENET_STD, grid=28).eval()
    try:
        with torch.no_grad():
            m3.forward_train(x)
        check("wrong token count raises", False)
    except RuntimeError:
        check("wrong token count raises", True)

    # ── the native open_clip token path ──────────────────────────────────────
    class _StubNativeVisual(nn.Module):
        """open_clip's native VisionTransformer surface, minimally."""

        def __init__(self, grid=14, width=48):
            super().__init__()
            self.conv1 = nn.Conv2d(3, width, 16, stride=16, bias=False)
            self.class_embedding = nn.Parameter(torch.zeros(width))
            self.positional_embedding = nn.Parameter(
                torch.randn(1 + grid * grid, width) * 0.02)
            self.ln_pre = nn.LayerNorm(width)
            self.transformer = nn.Sequential(nn.Linear(width, width))
            self.ln_post = nn.LayerNorm(width)

    m4 = OpenClipProbe(_StubNativeVisual(), OPENAI_CLIP_MEAN, OPENAI_CLIP_STD,
                       grid=28, patch=16).eval()
    with torch.no_grad():
        logits, aux = m4.forward_train(x)
    check("OpenClipProbe native path",
          m4.mode == "native" and tuple(logits.shape) == (2, 1, 640, 640)
          and aux is None,
          f"mode={m4.mode} pos={tuple(m4.encoder.positional_embedding.shape)}")

    class _NoTower(nn.Module):
        pass

    try:
        OpenClipProbe(_NoTower(), OPENAI_CLIP_MEAN, OPENAI_CLIP_STD, grid=28)
        check("unrecognised tower raises", False)
    except RuntimeError:
        check("unrecognised tower raises", True)

    # ── registry sanity ──────────────────────────────────────────────────────
    check("every corpus arm declares the target grid",
          all(arm_grid(a) == TARGET_GRID for a in CORPUS_ARMS),
          f"{ {a: arm_grid(a) for a in CORPUS_ARMS} }")
    check("calibration arms span >= 3 distinct grids",
          len({arm_grid(a) for a in CALIBRATION_ARMS}) >= 3,
          str(sorted({arm_grid(a) for a in CALIBRATION_ARMS})))
    check("the pre-registered treatment is a real arm", GATE_TREATMENT in CORPUS_ARCHS)
    check("derm_foundation is registered as NOT probeable",
          SOURCES["derm_foundation"].spatial is False
          and "derm_foundation" not in ALL_ARCHS)

    from bruisekit.models import build_param_groups
    groups = build_param_groups(m, 6e-5, 6e-4, 0.01)
    n = sum(len(g["params"]) for g in groups)
    check("build_param_groups accounts for the head", n > 0, f"{n} tensors")

    if verbose:
        print("PASS" if ok else "FAIL")
    return bool(ok)


if __name__ == "__main__":
    raise SystemExit(0 if self_test() else 1)
