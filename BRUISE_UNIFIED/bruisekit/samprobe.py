"""Stage N4 -- does MASK-supervised pretraining beat CAPTION-supervised pretraining?

THE QUESTION
------------
Stages N, N2 and N3 tested four pretrained encoders and every medical one lost:

    frozen probe (7h.7)    dermlip - dinov2   = -0.0913   [-0.1208, -0.0544]
    fine-tuned   (7i.7)    dermlip - dinov2   = -0.0859   [-0.0425, +0.1281]
    frozen probe (7f.8)    medsiglip - dinov2 = -0.1660   [-0.2010, -0.1306]

But every one of those medical arms is CLIP/SigLIP-style: one pooled vector per
image trained against one sentence. A caption describes a picture globally and
never says which pixels are the lesion, so that objective is rewarded for
throwing spatial detail away. DINOv2's objective is patch-level. Handbook 7h.9
item 2 already names this as the likeliest mechanism behind the whole ranking --
and if it is the mechanism, then "medical pretraining does not help" is the wrong
conclusion to draw from those three rows. The right one is narrower: "medical
CAPTION pretraining does not help."

SAM and MedSAM are the arms that separate those two readings. Both were
pretrained with DENSE MASK supervision, which is the objective this task actually
wants. They differ from each other in exactly one thing -- the corpus:

    sam_ft      SA-1B, 11 M natural images, ~1.1 B masks
    medsam_ft   ~1.5 M medical image-mask pairs across ~10 modalities

So `medsam - sam` is the cleanest medical-corpus contrast this project can form.
It holds the architecture (ViT-B/16), the objective (mask supervision), the
capacity and the recipe fixed and varies the corpus alone. Nothing in Stages N,
N2 or N3 managed that: there, corpus and objective moved together.

WHAT THIS IS NOT
----------------
THIS IS NOT A TEST OF MedSAM AS A SEGMENTATION MODEL, and it must not be written
up as one. SAM and MedSAM are PROMPTABLE: you give them a box or a point and they
segment the thing inside it. This stage throws the prompt encoder and the mask
decoder away and keeps only the image encoder, because our pipeline is fully
automatic and has no prompt to give. "MedSAM with a ground-truth box" would score
high and would answer a different question -- it hands the model the answer's
location, which is the hard half of this task (handbook 4: misses, not Dice, are
where our models separate).

So the honest claim shape is "MedSAM's FEATURES", never "MedSAM". The neck IS
kept (see USE_NECK) because it is the last pretrained stage and MedSAM fine-tuned
it; the promptable head is what is discarded.

WHAT IS PRE-REGISTERED (fixed here before any number is produced)
-----------------------------------------------------------------
    UNFREEZE_BLOCKS = 6           last six blocks + the neck, as Stage N3
    TARGET_GRID     = 40          40 * 16 = 640, the pipeline's own img_size
    USE_NECK        = True        the 256-d neck output is the feature map
    SEEDS           = (0,)        screening run, as Stage N3
    CEILING_BAND    = (0.75, 0.78)

    PRIMARY    medsam_ft - sam_ft
        clears zero POSITIVE -> the medical CORPUS buys something once the
            OBJECTIVE is right. Stages N/N2/N3 measured the objective, not the
            corpus, and their conclusion needs narrowing. A foundation teacher
            becomes worth costing out.
        contains zero -> NULL, and a strong one: with the objective held fixed
            and matched capacity, a 1.5 M-pair medical mask corpus buys nothing.
            Combined with N2 and N3 that closes the medical-pretraining question
            on a fourth axis.
        clears zero NEGATIVE -> MedSAM is WORSE than SAM, the same direction
            DermLIP and MedSiGLIP already went. Report it; do not explain it away.

    SECONDARY  best SAM-family arm - dinov2_ft (val 0.7824, STAGE_N3_RESULTS)
        Does mask supervision beat DINOv2's patch-level SSL at all? This is the
        arm that tests 7h.9 item 2 directly.

    SMALL-LESION TRIGGER   D1-D4 zero_dice_rate, either arm vs dinov2_ft
        A Dice null with a miss-rate win IS A RESULT and is reported as the
        headline when it happens. Dice is saturated at this label-noise level
        (Friedman p = 0.61 across the seven headline models); complete misses on
        small bruises are the endpoint that has ever moved and the one a
        clinician cares about. `mask_supervision_gate` REFUSES to print a
        verdict without the miss columns, exactly as `ceiling_gate` does.

FAIRNESS IS SCORED HERE, NOT BOLTED ON AFTERWARDS
--------------------------------------------------
Handbook 8.4: lesion size is confounded with ITA group in this test set, so any
unconditioned fairness number measures both at once. `fairness_report` therefore
runs `lesionsize.fairness_conditioned`, which reports each group MARGINALLY and
WITHIN the small-lesion stratum, and `lesionsize.size_by_ita`, which measures the
confound itself. Reusing that module rather than re-deriving it also means these
arms are binned by the SAME deciles as every other model in the study.

WHY THIS MODULE IS NOT IN THE UNIFIED BUNDLE
---------------------------------------------
Same policy as foundation.py, dermprobe.py, finetune_n3.py, lesionsize.py and
multiteacher.py: experiment modules are authored in bruisekit/ but kept out of
60_build's copy list, and ship as a standalone notebook plus an overlay zip.
Nothing in the reporting pipeline imports this, and `train_arms` writes to
STAGE_N4_RESULTS/runs rather than env.runs so `Registry` cannot pick these arms
up into a table they are not part of.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import dermprobe as _dp
from . import finetune_n3 as _n3

# The decoder is IMPORTED, not re-declared. Stage N4's numbers are only readable
# against Stage N3's if the head is the same object -- a re-typed copy that drifts
# by one channel turns every cross-stage comparison into a confound nobody would
# think to check.
from .finetune_n3 import ConvDecodeHead, DECODER_WIDTH

# ── pre-registered constants ─────────────────────────────────────────────────

TARGET_GRID = 40
UNFREEZE_BLOCKS = 6
CEILING_BAND = (0.75, 0.78)
CEILING_HIGH = 0.79
CEILING_LOW = 0.73

#: Keep SAM's neck (1x1 conv -> LN -> 3x3 conv -> LN, 768 -> 256) and probe its
#: 256-channel output rather than the last block's 768.
#:
#: WHY. The neck is pretrained, it is the tensor SAM's own mask decoder consumes,
#: and MedSAM fine-tuned it on medical masks. Dropping it would discard the part
#: of the encoder that is most specifically about producing masks -- which is the
#: property this whole stage exists to test. Cost: the head's input projection is
#: 256->256 instead of 768->256, so the decoder is 951 k parameters here against
#: 1.08 M in Stage N3. That is 0.3 % of a 44 M trainable budget and is recorded in
#: arm_build_info.csv rather than being silently absorbed.
USE_NECK = True

#: SAM normalises with ImageNet statistics on a 0-255 scale (pixel_mean
#: [123.675, 116.28, 103.53]). Divided through by 255 that is the standard
#: ImageNet mean/std, which is what `_ProbeBase` applies to raw [0,1] pixels.
#: NOT the /255-only path YOLO needs (handbook 3) -- getting this backwards is
#: the error that silently capped YOLO at 0.479 Dice, so it is written out here
#: rather than inherited from whatever the last arm used.
SAM_MEAN = (0.485, 0.456, 0.406)
SAM_STD = (0.229, 0.224, 0.225)

#: The field these arms are being asked to join. Stage N3's VAL numbers, because
#: this gate is val-only for the same reason N3's was (7f.4): a decision taken on
#: test is a decision taken on the data the paper reports.
REFERENCE_DICE_VAL = {
    "dinov2_ft": 0.7824,
    "dermlip_ft": 0.7235,
}

#: Stage N3 TEST numbers, carried for the write-up only. Never used by the gate.
REFERENCE_DICE_TEST = {
    "dinov2_ft": 0.7902,
    "dermlip_ft": 0.7043,
    "segformer_b5_teacher": 0.7727,
    "segformer_b2_teacher": 0.7692,
    "segformer_b0_direct": 0.7663,
}

ARMS: dict[str, dict] = {
    "sam_ft": {
        "source": "sam",
        "corpus": "natural images, mask-supervised",
        "note": "the ATTRIBUTION arm. Same objective as medsam, zero medical "
                "data. Without it a medsam result cannot distinguish 'medical "
                "masks help' from 'mask supervision helps'.",
    },
    "medsam_ft": {
        "source": "medsam",
        "corpus": "medical images, mask-supervised",
        "note": "the TREATMENT arm. Fixed by name before any number was produced.",
    },
}

GATE_PRIMARY = "medsam_ft"
GATE_ATTRIBUTION = "sam_ft"

#: The contrast list, fixed here. `lesionsize.contrast_table` applies
#: Holm-Bonferroni WITHIN the confirmatory family only; the exploratory rows stay
#: uncorrected and LABELLED. Folding them in would either over-penalise the two
#: questions this stage was designed around or launder two post-hoc comparisons
#: into confirmatory ones -- 7g.6's reversal is why that line is drawn here and
#: not after the numbers are in.
CONTRASTS: list[tuple[str, str, str, str]] = [
    ("medsam_ft", "sam_ft", "confirmatory",
     "THE PRIMARY. Objective, architecture, capacity and recipe held fixed; only "
     "the pretraining corpus moves. The cleanest medical-corpus contrast in the "
     "project."),
    ("sam_ft", "dinov2_ft", "confirmatory",
     "Does dense mask supervision beat patch-level SSL? Tests 7h.9 item 2 -- that "
     "the N/N2/N3 ranking was an OBJECTIVE effect wearing a corpus label."),
    ("medsam_ft", "dinov2_ft", "confirmatory",
     "The headline the write-up will be asked for: does the medical mask encoder "
     "beat the study's best foundation arm?"),
    ("medsam_ft", "dermlip_ft", "exploratory",
     "Medical masks against medical captions, both ViT-B/16 at grid 40. POST-HOC "
     "in the sense that it is not what the stage was designed to answer, but it "
     "is the contrast that separates corpus from objective most directly."),
]

#: Endpoints, in the order they are read. zero_dice_rate FIRST -- Dice is
#: saturated and misses are what separates models here (handbook 4).
PRIMARY_ENDPOINTS = ("zero_dice_rate", "mean_recall", "median_dice")

RESULTS_DIRNAME = "STAGE_N4_RESULTS"

#: Subdirectory of `env.weights` these encoders live in. NOT `foundation/` --
#: Stage N2 owns that directory and mixing two stages' checkpoints into one tree
#: is how a `report_sources` table starts reporting another stage's arms.
WEIGHTS_SUBDIR = "sam"


# ─────────────────────────────────────────────────────────────────────────────
# 1. where the weights come from
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SamSource:
    """One SAM-family encoder: where to get it, what it saw, what it is licensed as.

    Deliberately a separate registry from `dermprobe.SOURCES` rather than three
    more entries in it. An experiment module must not mutate a sibling
    experiment's table: `dermprobe.report_sources` and `download_instructions`
    both iterate SOURCES, and appending to it would make Stage N2's notebook print
    encoders it never probed and stop on downloads it never needed.
    """

    key: str
    repo: str
    local_dir: str
    corpus: str
    licence: str = ""
    init: str = ""
    note: str = ""
    url: str = ""                  # the human-readable page, for the write-up
    fallback_url: str = ""         # the original (non-HF) checkpoint, if any
    patch: int = 16
    native_size: int = 1024
    #: Parameter count of the RAW loaded image encoder, BEFORE the position
    #: embedding is resampled 64 -> 40. Checked at that point on purpose: the
    #: resample shrinks pos_embed from 3.15 M to 1.23 M, so a check applied after
    #: it would be comparing against a number no publication reports.
    expected_params_M: float = 89.7
    param_tol: float = 0.05
    aliases: tuple = field(default_factory=tuple)


SOURCES: dict[str, SamSource] = {
    "sam": SamSource(
        key="sam",
        repo="facebook/sam-vit-base",
        local_dir="sam-vit-base",
        corpus="natural images (mask-supervised)",
        licence="Apache-2.0. The only permissive licence among the large "
                "pretrained encoders this project has tried apart from DINOv2, "
                "and the reason a SAM-family win would be commercially usable "
                "where a DermLIP or MedSigLIP win would not (7b.1).",
        init="SAM ViT-B/16 image encoder, trained on SA-1B: 11 M licensed "
             "natural images and ~1.1 B masks, promptable mask prediction.",
        note="THE ATTRIBUTION ARM, and it is the one that makes this stage worth "
             "running. Stage N's 2026-08-06 lesson was that a treatment arm "
             "without a matched control produces an attribution number that is "
             "entirely an artefact -- there, MedSigLIP measured against a "
             "randomly-initialised ViT. Here, a medsam result without sam cannot "
             "tell 'medical masks help' from 'mask supervision helps'.",
        url="https://huggingface.co/facebook/sam-vit-base",
        fallback_url="https://dl.fbaipublicfiles.com/segment_anything/"
                     "sam_vit_b_01ec64.pth",
    ),
    "medsam": SamSource(
        key="medsam",
        repo="wanglab/medsam-vit-base",
        local_dir="medsam-vit-base",
        corpus="medical images (mask-supervised)",
        licence="Apache-2.0 as published. VERIFY THIS ON THE MODEL CARD BEFORE "
                "any commercialisation claim -- it is recorded here from the "
                "release, not from a licence review, and this project has "
                "already been bitten once by a card whose header and body "
                "disagreed (dermprobe.SOURCES['dermlip_panderm']).",
        init="MedSAM: SAM ViT-B/16 image encoder fine-tuned on ~1.5 M medical "
             "image-mask pairs spanning ~10 imaging modalities (Ma et al., "
             "Nature Communications 2024).",
        note="THE TREATMENT ARM. Two things to hold in mind when reading its "
             "score.\n"
             "  1. ITS CORPUS IS MOSTLY RADIOLOGY AND DERMOSCOPY, not clinical "
             "photography. Dermoscopy is contact, polarised and lesion-centred; "
             "our images are standard-lighting photographs at varying distance "
             "across skin tones. That is the same distribution gap that put "
             "MedSigLIP -- trained on 'medical images including dermatology' -- "
             "LAST at 0.4670. Expect a penalty and do not read it as a surprise.\n"
             "  2. IT WAS TRAINED PROMPTED. Every mask in its training signal "
             "came with a box. Nothing guarantees its features are organised for "
             "prompt-free localisation, which is the task here.",
        url="https://huggingface.co/wanglab/medsam-vit-base",
        fallback_url="https://github.com/bowang-lab/MedSAM",
    ),
}


def download_instructions(env=None) -> str:
    """Exact commands to fetch both encoders, with licences and sizes.

    Printed by the notebook when an encoder is absent, so the failure mode of a
    missing checkpoint is a copyable command rather than forty minutes of
    training from random initialisation.
    """
    dest = (Path(env.weights) / WEIGHTS_SUBDIR) if env is not None else \
        Path("<bundle>/pretrained_weights") / WEIGHTS_SUBDIR
    lines = [
        "STAGE N4 ENCODERS -- both are ~375 MB, both are open, neither is in the zip.",
        "=" * 78,
        "",
        f"  mkdir -p {dest}",
        "",
    ]
    for i, (key, s) in enumerate(SOURCES.items(), 1):
        lines += [
            f"{i}. {key.upper()}  --  {s.corpus}",
            f"   page    : {s.url}",
            f"   licence : {s.licence.splitlines()[0]}",
            "",
            f"   hf download {s.repo} \\",
            f"        --local-dir {dest}/{s.local_dir}",
            "",
            f"   fallback (original release, NOT the HuggingFace layout -- this "
            f"module\n   loads the HF layout only): {s.fallback_url}",
            "",
        ]
    lines += [
        "VERIFY THE REPO IDS BEFORE RUNNING A LONG JOB. They are recorded from the",
        "published releases, not from a live check, and a repo that has been renamed",
        "fails at load rather than silently -- but it fails after you have waited.",
        "",
        "WHAT ARRIVES: a HuggingFace directory with config.json whose model_type is",
        "'sam'. `load_sam_encoder` refuses anything else rather than falling back to",
        "AutoModel; 7f.7a is what that rule is for.",
    ]
    return "\n".join(lines)


def report_sources(env) -> pd.DataFrame:
    """One row per encoder: is it on disk, how big, what is it licensed as."""
    rows = []
    for key, s in SOURCES.items():
        path = Path(env.weights) / WEIGHTS_SUBDIR / s.local_dir
        cfg = path / "config.json"
        size_mb = np.nan
        if path.exists():
            size_mb = sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e6
        rows.append({
            "encoder": key, "repo": s.repo, "corpus": s.corpus,
            "present": bool(cfg.exists()), "path": str(path),
            "size_MB": round(size_mb, 1) if size_mb == size_mb else np.nan,
            "expected_M_params": s.expected_params_M,
            "licence": s.licence.splitlines()[0],
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 2. loading -- verified, offline, and never falling back to a random init
# ─────────────────────────────────────────────────────────────────────────────
def load_sam_encoder(path: Path, key: str, max_missing: int = 0):
    """Load a SAM-family VISION encoder from a local HuggingFace directory.

    Three guarantees, and all three exist because of 7f.7a -- where
    `SiglipVisionModel.from_pretrained` did not raise on a DINOv2 directory,
    returned a randomly-initialised ViT, trained happily, and invalidated an
    entire run along with the attribution number computed from it:

      1. THE CLASS IS CHOSEN BY `model_type`. `config.json` must say 'sam'. There
         is no AutoModel fallback -- a wrong class here loads nothing and says
         nothing.
      2. THE LOAD IS CHECKED. `output_loading_info=True` returns the missing-key
         list and anything beyond `max_missing` raises. The prompt encoder and
         mask decoder are DISCARDED, so their keys are legitimately unused rather
         than missing, and they are not counted against the load.
      3. THE VISION TOWER IS LOCATED BY NAME, not by taking whatever attribute
         happens to exist.

    `local_files_only=True`: a silent network fetch is what makes an offline
    compute node stall instead of failing at load.
    """
    import transformers

    cfg_file = Path(path) / "config.json"
    if not cfg_file.exists():
        raise FileNotFoundError(
            f"{key}: no encoder at {path}\n\n{download_instructions()}")

    cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
    mtype = cfg.get("model_type", "")
    if mtype != "sam":
        raise RuntimeError(
            f"{key}: config.json at {path} says model_type={mtype!r}, expected "
            f"'sam'. This module loads the HuggingFace SAM layout only. If you "
            f"downloaded the original .pth release instead, convert it or fetch "
            f"the HF repo -- do NOT point this at a different architecture and "
            f"hope the state dict lands.")

    cls = getattr(transformers, "SamModel", None)
    if cls is None:
        raise RuntimeError(
            f"transformers {transformers.__version__} has no SamModel. Upgrade "
            f"transformers; do not substitute another class.")

    model, info = cls.from_pretrained(str(path), local_files_only=True,
                                      output_loading_info=True)
    missing = list(info.get("missing_keys", []))
    if len(missing) > max_missing:
        raise RuntimeError(
            f"{key}: loaded SamModel from {path} but {len(missing)} parameters "
            f"were NOT in the checkpoint -- they are RANDOM.\n"
            f"  e.g. {missing[:4]}\n"
            f"  This is the 2026-08-06 failure (7f.7a). Do not train on this.")

    enc = getattr(model, "vision_encoder", None)
    if enc is None:
        raise RuntimeError(
            f"{key}: SamModel has no `.vision_encoder` in transformers "
            f"{transformers.__version__}. The attribute has moved; find it and "
            f"name it here rather than reaching for the first ModuleList that "
            f"looks right.")

    n_M = sum(p.numel() for p in enc.parameters()) / 1e6
    want, tol = SOURCES[key].expected_params_M, SOURCES[key].param_tol
    if want and abs(n_M - want) / want > tol:
        raise RuntimeError(
            f"{key}: image encoder has {n_M:.2f} M parameters, published is "
            f"{want:.2f} M ({100 * (n_M - want) / want:+.1f} %, tolerance "
            f"{100 * tol:.0f} %). The architecture built is not the one named in "
            f"the table. Do not train on this.\n"
            f"  NOTE this check runs BEFORE the position embedding is resampled "
            f"64 -> {TARGET_GRID}, which legitimately removes ~1.9 M parameters.")
    return enc


# ─────────────────────────────────────────────────────────────────────────────
# 3. the two structural patches SAM needs that no other arm in this study did
# ─────────────────────────────────────────────────────────────────────────────
def resample_pos_embed_2d(pos: torch.Tensor, new_grid: int) -> torch.Tensor:
    """Bicubically resample a [1, G, G, C] 2-D position embedding to `new_grid`.

    SAM's position embedding is NOT the [1, 1+G*G, C] token sequence every other
    encoder in this project uses, so `dermprobe.resample_pos_embed` is the wrong
    function and calling it here would silently reinterpret a spatial axis as a
    token axis. There is no CLS token to carry through: SAM's ViT has no prefix.

    SAM stores it for a 1024 px input, i.e. a 64x64 grid, and ADDS it directly to
    the patch grid. At 640 px that is a 40x40 grid and the addition is a shape
    error -- so this is not an optimisation, it is the thing that makes the arm
    run at all. Raises rather than guessing if the tensor is not square.
    """
    if pos.dim() != 4:
        raise ValueError(
            f"expected a [1, G, G, C] SAM position embedding, got shape "
            f"{tuple(pos.shape)}. If this is a token-sequence embedding you want "
            f"dermprobe.resample_pos_embed, not this function.")
    _, gh, gw, c = pos.shape
    if gh != gw:
        raise ValueError(
            f"position embedding grid is {gh}x{gw}, not square. Do not guess at "
            f"the intended layout -- read it off the model.")
    if gh == new_grid:
        return pos
    x = pos.permute(0, 3, 1, 2).float()
    x = F.interpolate(x, size=(new_grid, new_grid), mode="bicubic",
                      align_corners=False)
    return x.permute(0, 2, 3, 1).to(pos.dtype)


def find_sam_blocks(encoder: nn.Module) -> tuple[nn.ModuleList, str]:
    """The transformer block list inside a SAM image encoder.

    Two layouts are in the wild for the checkpoints this stage uses:

        HuggingFace SamVisionEncoder     .layers
        original segment_anything        .blocks

    `finetune_n3.find_blocks` does not know either of them, and extending it from
    here would mean an experiment module reaching into a sibling experiment's
    behaviour. So this is a separate function with the same contract -- and the
    same reason for existing.

    RAISES on an unrecognised tower rather than returning empty. An arm that
    silently unfreezes NOTHING trains like a frozen probe, scores low-to-mid, and
    is indistinguishable from a real 'the ceiling holds' result. Unlike 7f.7a's
    silent random init there is no parameter count that gives it away, which is
    why 7i.4 built three independent guards for it and why this is the first.
    """
    for name, get in (("layers", lambda m: m.layers),
                      ("blocks", lambda m: m.blocks)):
        try:
            blocks = get(encoder)
        except AttributeError:
            continue
        if isinstance(blocks, (nn.ModuleList, nn.Sequential)) and len(blocks) > 0:
            return blocks, name
    raise RuntimeError(
        f"cannot find the transformer blocks in {type(encoder).__name__}: it has "
        f"neither `.layers` (HuggingFace) nor `.blocks` (segment_anything). Add "
        f"the case explicitly. An arm that silently unfreezes nothing scores like "
        f"a frozen probe and reads like a confirmed ceiling.")


def find_sam_pos_embed(encoder: nn.Module) -> tuple[nn.Parameter, str]:
    """SAM's 2-D position embedding parameter. Raises if it is not where expected."""
    for name in ("pos_embed", "position_embeddings"):
        p = getattr(encoder, name, None)
        if isinstance(p, (nn.Parameter, torch.Tensor)) and p.dim() == 4:
            return p, name
    raise RuntimeError(
        f"{type(encoder).__name__} has no 4-D position embedding at `.pos_embed` "
        f"or `.position_embeddings`. SAM adds a [1,G,G,C] embedding to the patch "
        f"grid; without resampling it this arm cannot run at anything but 1024 px, "
        f"and guessing at the attribute is how it would run at the wrong one.")


def find_sam_neck(encoder: nn.Module):
    """SAM's neck (768 -> 256), if this tower has one. Returns (module, name)."""
    for name in ("neck",):
        mod = getattr(encoder, name, None)
        if isinstance(mod, nn.Module):
            return mod, name
    return None, ""


# ─────────────────────────────────────────────────────────────────────────────
# 4. the arm
# ─────────────────────────────────────────────────────────────────────────────
class SamViTProbe(_dp._ProbeBase):
    """A SAM-family image encoder, retargeted to `grid` and partially unfrozen.

    Subclasses `dermprobe._ProbeBase` for its architecture-blind contract --
    raw [0,1] pixels in, full-resolution logits out, and the grid assertion in
    `forward_train` that turns a silently-wrong feature map into an exception.
    Everything below is the part that is genuinely different about SAM.

    THREE THINGS SAM DOES THAT NO OTHER ENCODER IN THIS STUDY DOES
    ---------------------------------------------------------------
    1. Its position embedding is 2-D ([1,64,64,C]) and is added to the patch grid,
       not to a token sequence. See `resample_pos_embed_2d`.
    2. It has no CLS or register prefix, so there is no prefix to strip -- the
       `tok[:, n-want:, :]` slice every other probe here performs would be a
       silent off-by-nothing that happens to work and then stops working.
    3. Its encoder emits [B, C, H, W] already (after the neck), not [B, N, C].
       `_features` therefore does no reshaping and instead CHECKS the layout,
       because a tower that returned tokens would otherwise be reshaped into a
       plausible-looking garbage grid.
    """

    def __init__(self, encoder, mean=SAM_MEAN, std=SAM_STD,
                 grid: int = TARGET_GRID, patch: int = 16,
                 use_neck: bool = USE_NECK, num_classes: int = 1):
        super().__init__()
        self.encoder = encoder
        self.grid = int(grid)
        self.patch = int(patch)
        self.enc_size = self.grid * self.patch
        self.use_neck = bool(use_neck)
        self._retarget()
        self._resolve_embed_dim()
        self._register_scale(mean, std)
        self._build_head(num_classes)
        self._freeze()
        self.init_source = ""

    # ── setup ────────────────────────────────────────────────────────────────
    def _retarget(self):
        """Resample the position embedding and relax the patch-embed size check.

        BOTH halves are required and the second is not optional plumbing.
        `SamPatchEmbeddings.forward` RAISES when the input is not exactly
        `config.image_size` (1024), so without the relax this arm cannot run at
        640 at all -- it dies on the first forward pass, after the model has been
        built and reported a perfectly healthy parameter count.

        The projection itself is size-agnostic: kernel = stride = patch, so it
        tiles any multiple of 16 correctly. The check is a guard against the
        position embedding no longer lining up, and that is precisely what the
        resample above fixes -- so relaxing it here is honest rather than a
        bypass. `num_patches` is updated with it so nothing downstream reads a
        stale 64x64 = 4096.
        """
        pos, name = find_sam_pos_embed(self.encoder)
        native = int(pos.shape[1])
        with torch.no_grad():
            setattr(self.encoder, name,
                    nn.Parameter(resample_pos_embed_2d(pos.data, self.grid),
                                 requires_grad=False))
        self.pos_embed_name = name
        self.native_grid = native

        pe = getattr(self.encoder, "patch_embed", None)
        if pe is None:
            raise RuntimeError(
                f"{type(self.encoder).__name__} has no `.patch_embed`. Its input "
                f"size guard cannot be relaxed, so this arm would raise on the "
                f"first forward pass at {self.enc_size}px.")
        if hasattr(pe, "image_size"):
            pe.image_size = (self.enc_size, self.enc_size)
        if hasattr(pe, "num_patches"):
            pe.num_patches = self.grid * self.grid
        if hasattr(pe, "grid_size"):
            pe.grid_size = (self.grid, self.grid)

    def _resolve_embed_dim(self):
        """Channel count of whatever `_features` will return.

        Read off the built module rather than off the config: with `use_neck` the
        answer is the neck's output channels (256) and without it the block width
        (768), and a config field that reports one while the forward pass returns
        the other is exactly the kind of mismatch `forward_train`'s grid check
        cannot catch.
        """
        neck, _ = find_sam_neck(self.encoder)
        if self.use_neck and neck is not None:
            convs = [m for m in neck.modules() if isinstance(m, nn.Conv2d)]
            if not convs:
                raise RuntimeError(
                    "SAM neck contains no Conv2d; cannot determine its output "
                    "width. Set USE_NECK = False rather than guessing 256.")
            self.embed_dim = int(convs[-1].out_channels)
            self.feature_source = "neck"
            return
        cfg = getattr(self.encoder, "config", None)
        dim = int(getattr(cfg, "hidden_size", 0) or getattr(cfg, "output_channels", 0))
        if not dim:
            raise RuntimeError(
                "cannot read the block width off this SAM encoder's config. Name "
                "the field explicitly rather than defaulting to 768.")
        self.embed_dim = dim
        self.feature_source = "blocks"

    def _build_head(self, num_classes: int = 1):
        # Overrides _ProbeBase's LinearProbeHead. Stage N2 needed a linear head so
        # a strong decoder could not paper over a weak encoder; here the encoder
        # adapts, so the head must be a FAIR decoder instead -- and it must be the
        # SAME decoder Stage N3 used, or the two stages are not comparable.
        self.decode_head = ConvDecodeHead(self.embed_dim, DECODER_WIDTH, num_classes)

    # ── unfreezing ───────────────────────────────────────────────────────────
    def unfreeze_last(self, n_blocks: int = UNFREEZE_BLOCKS) -> dict:
        """Unfreeze the last `n_blocks` blocks plus the neck.

        The neck is included for the reason `finetune_n3.find_final_norm` includes
        the final LayerNorm: it sits directly on the unfrozen blocks and its
        statistics are tuned to the frozen ones, so freezing it while they move is
        a mismatch that costs Dice for no reason. For MedSAM there is a second
        reason -- the neck is part of what was fine-tuned on medical masks, and
        this stage is about those weights.

        The position embedding stays FROZEN. It was resampled, not trained, and
        letting 1.2 M interpolated parameters adapt would let the arm partly
        re-learn a grid the pre-registration fixed.
        """
        blocks, where = find_sam_blocks(self.encoder)
        if n_blocks > len(blocks):
            raise ValueError(
                f"asked to unfreeze {n_blocks} blocks but the tower has "
                f"{len(blocks)}. Pre-registration says {UNFREEZE_BLOCKS}; changing "
                f"it changes the experiment, so change it deliberately in the "
                f"config rather than clamping silently here.")

        for p in self.encoder.parameters():
            p.requires_grad_(False)
        for blk in blocks[len(blocks) - n_blocks:]:
            for p in blk.parameters():
                p.requires_grad_(True)

        neck, neck_name = find_sam_neck(self.encoder)
        if neck is not None and self.use_neck:
            for p in neck.parameters():
                p.requires_grad_(True)

        # Drives _ProbeBase.forward_train's no_grad path AND its train() override
        # that pins the encoder to eval(). Both must stop now.
        self.frozen = False

        trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.encoder.parameters())
        if trainable == 0:
            raise RuntimeError(
                "unfreeze_last left every encoder parameter frozen. See "
                "find_sam_blocks' docstring: this arm would silently be a probe, "
                "and nothing downstream could tell.")
        return {
            "blocks_found_at": where,
            "blocks_total": len(blocks),
            "blocks_unfrozen": n_blocks,
            "neck": neck_name or "(none found)",
            "neck_unfrozen": bool(neck is not None and self.use_neck),
            "feature_source": self.feature_source,
            "pos_embed_native_grid": self.native_grid,
            "pos_embed_resampled_to": self.grid,
            "encoder_trainable_params": trainable,
            "encoder_total_params": total,
            "encoder_trainable_fraction": round(trainable / max(total, 1), 4),
        }

    # ── features ─────────────────────────────────────────────────────────────
    def _features(self, x):
        enc = self.encoder
        out = enc(pixel_values=x) if hasattr(enc, "config") else enc(x)
        feat = getattr(out, "last_hidden_state", out)
        if isinstance(feat, (tuple, list)):
            feat = feat[0]

        if feat.dim() != 4:
            raise RuntimeError(
                f"SAM encoder returned {tuple(feat.shape)}; expected a 4-D "
                f"[B, C, g, g] map. A token sequence reshaped here would produce "
                f"a plausible grid of garbage, so this refuses instead.")
        # HuggingFace returns [B, C, g, g] after the neck; a tower without a neck
        # may return [B, g, g, C]. Distinguish by which axis matches the grid
        # rather than by vendor, and raise if neither does.
        if feat.shape[-1] == self.grid and feat.shape[-2] == self.grid:
            pass                                       # already [B, C, g, g]
        elif feat.shape[1] == self.grid and feat.shape[2] == self.grid:
            feat = feat.permute(0, 3, 1, 2).contiguous()
        else:
            raise RuntimeError(
                f"SAM encoder returned {tuple(feat.shape)} at input "
                f"{self.enc_size}px; no axis pair matches the declared "
                f"{self.grid}x{self.grid} grid. The position-embedding resample "
                f"did not take.")
        if feat.shape[1] != self.embed_dim:
            raise RuntimeError(
                f"feature map has {feat.shape[1]} channels, embed_dim was "
                f"resolved as {self.embed_dim}. The head was built for the wrong "
                f"width -- check `use_neck`.")
        return feat


def build_arm(env, arm: str, num_classes: int = 1, verbose: bool = True):
    """Build one Stage N4 arm at TARGET_GRID with the last blocks unfrozen."""
    if arm not in ARMS:
        raise KeyError(f"unknown Stage N4 arm {arm!r}; have {sorted(ARMS)}")
    src = SOURCES[ARMS[arm]["source"]]
    path = Path(env.weights) / WEIGHTS_SUBDIR / src.local_dir

    encoder = load_sam_encoder(path, src.key)
    model = SamViTProbe(encoder, SAM_MEAN, SAM_STD, grid=TARGET_GRID,
                        patch=src.patch, use_neck=USE_NECK,
                        num_classes=num_classes)
    info = model.unfreeze_last(UNFREEZE_BLOCKS)
    model.init_source = src.init
    model.n4_info = {"arm": arm, "grid": TARGET_GRID, "enc_size": model.enc_size,
                     "patch": model.patch, "embed_dim": model.embed_dim,
                     "use_neck": USE_NECK, **info}

    if verbose:
        print(f"  {arm}: grid {model.grid}x{model.grid} at {model.enc_size}px "
              f"(patch {model.patch}), features from {info['feature_source']} "
              f"({model.embed_dim}-d)")
        print(f"    pos_embed resampled {info['pos_embed_native_grid']} -> "
              f"{info['pos_embed_resampled_to']}")
        print(f"    unfrozen {info['blocks_unfrozen']}/{info['blocks_total']} blocks "
              f"at .{info['blocks_found_at']}, neck={info['neck']} "
              f"(unfrozen={info['neck_unfrozen']})")
        print(f"    encoder trainable {info['encoder_trainable_params']:,} / "
              f"{info['encoder_total_params']:,} "
              f"({100 * info['encoder_trainable_fraction']:.1f} %)")
    return model


def install_n4_shim(env, verbose: bool = True) -> None:
    """Teach `engine.train_run`'s `build_model` and `loaders.spec_for` about N4.

    A shim rather than an edit, for the same reason Stages N2 and N3 use one:
    engine.py and models.py are the shared pipeline and an experiment must not be
    able to change what the reporting stages build.

    Reassigns the name in BOTH `bruisekit.models` and `bruisekit.engine`, because
    engine bound `build_model` by value at import time -- patching models alone
    leaves train_run calling the original. Any arch this shim does not recognise
    falls through untouched, so the N2/N3/N4 overlays can share one kernel.
    """
    import bruisekit.engine as _be
    import bruisekit.models as _bm

    from . import loaders

    if getattr(_bm, "_n4_shim", False):
        if verbose:
            print("Stage N4 shim already installed")
        return

    original = _bm.build_model

    def build_model(arch: str, size: str, paths: dict):
        if arch == "n4":
            return build_arm(env, size, verbose=False)
        return original(arch, size, paths)

    _bm.build_model = build_model
    _be.build_model = build_model
    _bm._n4_shim = True

    for arm in ARMS:
        loaders.FAMILY_SPEC[arm] = {"arch": "n4", "size": arm, "distill": False}

    if verbose:
        print(f"Stage N4 shim installed: {', '.join(ARMS)}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. training
# ─────────────────────────────────────────────────────────────────────────────
def train_arms(env, cfg: dict, man640: dict, runs_dir, arms=tuple(ARMS),
               seeds=(0,), verbose: bool = True) -> list:
    """Train each arm through `engine.train_run`, unmodified.

    Going through the shared driver is the entire point: the arms inherit the
    same optimiser, LR split, warmup, early stopping, augmentation and resume
    contract that produced segformer_b0's 0.7663 AND Stage N3's two arms. A
    bespoke loop here would make `medsam - sam` unreadable, because a difference
    could always be the recipe rather than the corpus.

    `runs_dir` is a Stage N4 directory, NOT env.runs. The reporting stages scan
    env.runs by name and an experiment must not be able to inject arms into a
    table it is not part of.
    """
    from bruisekit import loaders as L
    from bruisekit.engine import train_run

    if not str(env.device).startswith("cuda"):
        raise RuntimeError(
            f"train_arms would fine-tune {len(tuple(arms)) * len(tuple(seeds))} "
            f"run(s) but device is {env.device}. Six unfrozen ViT blocks at 640 "
            f"needs a GPU session; on CPU this would not finish.")

    install_n4_shim(env, verbose=verbose)

    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    done = []
    for arm in arms:
        spec = L.spec_for(arm)
        for seed in seeds:
            run_id = f"{arm}__seed{seed}"
            if verbose:
                print(f"\n{'=' * 70}\n{run_id}  ({ARMS[arm]['corpus']})\n{'=' * 70}")
            res = train_run(run_id, spec, seed, cfg, env.paths_for_models(),
                            man640, env.cache640, runs_dir, env.device)
            if verbose:
                print(f"  -> {res.get('status', 'trained')}")
            done.append(run_id)
            if str(env.device).startswith("cuda"):
                torch.cuda.empty_cache()
    return done


def load_trained(env, arm: str, run_dir: Path):
    """Rebuild an arm and load its trained weights. Returns the model, on CPU.

    The checkpoint is loaded with `weights_only=False` because engine writes a
    dict with optimiser state alongside the model; `strict=True` is left on so a
    key mismatch -- the symptom of a changed `use_neck` or `TARGET_GRID` between
    training and scoring -- raises instead of leaving layers at their init.
    """
    model = build_arm(env, arm, verbose=False)
    state = torch.load(str(Path(run_dir) / "best.pt"), map_location="cpu",
                       weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 6. the gate
# ─────────────────────────────────────────────────────────────────────────────
def _bootstrap_ci(values: np.ndarray, groups: np.ndarray, n_boot: int,
                  rng: np.random.Generator) -> tuple[float, float]:
    """Subject-clustered bootstrap CI on a mean. Imported behaviour, not re-derived."""
    return _n3._bootstrap_ci(values, groups, n_boot, rng)


def _paired_delta(a: pd.DataFrame, b: pd.DataFrame, n_boot: int,
                  seed: int) -> dict:
    """Paired subject-clustered bootstrap of mean Dice (a - b) on aligned tables.

    Paired: the SAME resampled subject list is applied to both arms on every draw,
    because both scored the same images. Resampling independently discards the
    pairing and hides real effects of the size this stage is looking for.
    """
    da = a.sort_values("stem").reset_index(drop=True)
    db = b.sort_values("stem").reset_index(drop=True)
    if len(da) != len(db) or not (da.stem.to_numpy() == db.stem.to_numpy()).all():
        raise ValueError(
            "the two arms did not score the same images -- refusing to pair them. "
            "A paired bootstrap over misaligned rows reports an interval that "
            "means nothing.")
    va = da.dice.to_numpy(float)
    vb = db.dice.to_numpy(float)
    subj = da.subject.to_numpy()
    uniq = np.unique(subj)
    idx = {g: np.flatnonzero(subj == g) for g in uniq}
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx[g] for g in pick])
        draws[i] = va[rows].mean() - vb[rows].mean()
    lo, hi = float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))
    return {"delta": float(va.mean() - vb.mean()), "ci_lo": lo, "ci_hi": hi,
            "clears_zero": bool(lo > 0 or hi < 0),
            "n_images": int(len(va)), "n_subjects": int(len(uniq))}


def mask_supervision_gate(val_tables: dict[str, pd.DataFrame],
                          n_boot: int = 10000, seed: int = 0,
                          band: tuple[float, float] = CEILING_BAND) -> dict:
    """Apply the pre-registered reading at the top of this module.

    `val_tables[arm]` must be a per-image frame with `dice` and `subject`
    columns -- i.e. what `evaluate.evaluate_at_cut` returns joined to the
    manifest, or what `foundation.score_split` returns directly. Stage N3's two
    arms may be passed in the same dict under their own names to activate the
    cross-stage contrasts; they are read, never re-scored.

    Misses are counted as `dice == 0`, the definition the handbook publishes --
    NOT the per-seed empty-prediction count. The two differ by the case where a
    model fires confidently on the wrong region, which is still a missed injury,
    and mixing them is a known trap (handbook 4).

    RAISES ON AN EMPTY MATCH. Handed a dict keyed by run_id or filename rather
    than arm name, every arm is skipped and the fall-through reading is a
    considered-looking verdict for a gate that never ran. That is not
    hypothetical: it is what `STAGE_N3_RESULTS/ceiling_gate.json` contains, for a
    stage whose primary arm had in fact scored the highest Dice in the study
    (7i.7a). The check is load-bearing.
    """
    rng = np.random.default_rng(seed)
    lo_band, hi_band = band
    rows, verdicts = [], {}

    matched = [a for a in ARMS if a in val_tables]
    if not matched:
        raise KeyError(
            f"none of the Stage N4 arms {sorted(ARMS)} is in the table dict "
            f"(got keys {sorted(val_tables)}). Key by ARM NAME, not run_id or "
            f"filename -- otherwise this returns a considered-looking reading "
            f"for a gate that never looked at anything (7i.7a).")

    for arm, table in val_tables.items():
        for col in ("dice", "subject"):
            if col not in table.columns:
                raise KeyError(
                    f"{arm}: table needs a {col!r} column. Without subject the CI "
                    f"would be image-clustered and far too narrow to trust.")
        dice = table["dice"].to_numpy(dtype=float)
        mean = float(dice.mean())
        lo, hi = _bootstrap_ci(dice, table["subject"].to_numpy(), n_boot, rng)

        if arm in ARMS:
            if mean > CEILING_HIGH:
                v = "ABOVE_BAND"
            elif mean < CEILING_LOW:
                v = "BELOW_BAND_CHECK_FOR_COLLAPSE"
            elif lo_band <= mean <= hi_band:
                v = "IN_BAND"
            else:
                v = "NEAR_BAND"
            verdicts[arm] = v
        else:
            v = "reference (Stage N3)"

        rows.append({
            "arm": arm,
            "corpus": ARMS.get(arm, {}).get("corpus", "-- Stage N3 --"),
            "dice": round(mean, 4),
            "ci_lo": round(lo, 4),
            "ci_hi": round(hi, 4),
            "median": round(float(np.median(dice)), 4),
            "iqr": round(float(np.percentile(dice, 75) - np.percentile(dice, 25)), 4),
            "misses": int((dice == 0).sum()),
            "miss_rate": round(float((dice == 0).mean()), 4),
            "n": int(len(dice)),
            "verdict": v,
        })

    table = pd.DataFrame(rows)

    # ── the contrasts, in the pre-registered order ───────────────────────────
    contrasts = {}
    for a, b, kind, question in CONTRASTS:
        if a in val_tables and b in val_tables:
            r = _paired_delta(val_tables[a], val_tables[b], n_boot, seed)
            contrasts[f"{a} - {b}"] = {**r, "kind": kind, "question": question}

    # ── the reading ──────────────────────────────────────────────────────────
    primary = contrasts.get(f"{GATE_PRIMARY} - {GATE_ATTRIBUTION}")
    any_collapse = any(v == "BELOW_BAND_CHECK_FOR_COLLAPSE" for v in verdicts.values())
    any_above = any(v == "ABOVE_BAND" for v in verdicts.values())

    if any_collapse:
        reading = (
            "INCONCLUSIVE -- an arm is below 0.73. One seed cannot tell a weak "
            "encoder from a collapsed run (Stage Y seed 2 did exactly this). "
            "Inspect the loss curve and re-run that arm on another seed BEFORE "
            "reading anything into the corpus contrast.")
        corpus = None
    elif primary is None:
        reading = ("PRIMARY CONTRAST NOT COMPUTABLE -- both medsam_ft and sam_ft "
                   "must be present. A medsam number without its attribution arm "
                   "cannot distinguish 'medical masks help' from 'mask "
                   "supervision helps', which is the whole design (7f.7a).")
        corpus = None
    elif not primary["clears_zero"]:
        reading = (
            "NULL ON THE MEDICAL CORPUS. With the OBJECTIVE held fixed (both arms "
            "mask-supervised), the architecture matched (ViT-B/16) and the recipe "
            "identical, a 1.5 M-pair medical mask corpus buys nothing over natural "
            "images. Together with N2 (captions, frozen) and N3 (captions, "
            "fine-tuned) that is the medical-pretraining question closed on a "
            "fourth axis. READ THE SMALL-LESION TABLE BEFORE CALLING IT A NULL -- "
            "a Dice null with a miss-rate win is a result, not an absence of one.")
        corpus = False
    elif primary["delta"] > 0:
        reading = (
            "THE MEDICAL CORPUS HELPS, once the objective is right. This narrows "
            "N/N2/N3: what those stages measured was the CAPTION objective, not "
            "the medical corpus. A foundation teacher becomes worth costing out, "
            "and 7f.2's first objection -- 'there is no teacher deficit to fix' -- "
            "should be re-examined against this arm's miss column rather than its "
            "Dice.")
        corpus = True
    else:
        reading = (
            "THE MEDICAL CORPUS HURTS -- medsam is WORSE than sam, the same "
            "direction DermLIP (-0.0859) and MedSigLIP (-0.1660) already went, "
            "now with the objective held fixed. Report it plainly; three "
            "independent corpora pointing the same way is the finding.")
        corpus = False

    if any_above and not any_collapse:
        reading += ("\n  NOTE: an arm cleared 0.79, which is Stage N3's "
                    "'ceiling not binding' trigger. That reading is independent "
                    "of the corpus verdict above and belongs in the write-up too.")

    return {"table": table, "verdicts": verdicts, "contrasts": contrasts,
            "corpus_helps": corpus, "reading": reading, "band": list(band),
            "reference_dice_val": REFERENCE_DICE_VAL,
            "reference_dice_test": REFERENCE_DICE_TEST,
            "n_boot": n_boot, "seed": seed,
            "unfreeze_blocks": UNFREEZE_BLOCKS, "target_grid": TARGET_GRID,
            "use_neck": USE_NECK}


def print_gate(gate: dict) -> None:
    t = gate["table"]
    if t.empty:
        print("no arms present")
        return
    if "misses" not in t.columns:
        raise RuntimeError("refusing to print a verdict without the miss column")

    print("=" * 92)
    print(f"STAGE N4 -- mask-supervised encoders. band "
          f"{gate['band'][0]:.2f}-{gate['band'][1]:.2f}, "
          f"{gate['n_boot']} subject-clustered resamples")
    print(f"           grid {gate['target_grid']}, last {gate['unfreeze_blocks']} "
          f"blocks + neck unfrozen, use_neck={gate['use_neck']}")
    print("=" * 92)
    print(t.to_string(index=False))

    print("\nCONTRASTS (paired, subject-clustered):")
    for name, c in gate["contrasts"].items():
        star = "*" if c["clears_zero"] else " "
        print(f" {star} {name:<28} {c['delta']:+.4f}  "
              f"[{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}]  {c['kind']}")
    print("   * = interval excludes zero (UNCORRECTED -- see the Holm column in "
          "the small-lesion contrast table)")

    print("\nMISSES are the endpoint that separates models here -- Dice is "
          "saturated\n(Friedman p = 0.61 across the seven headline models). Read "
          "the miss column\nfirst, then the small-lesion table, THEN the Dice "
          "column.")
    print(f"\nREADING: {gate['reading']}")
    print("=" * 92)


# ─────────────────────────────────────────────────────────────────────────────
# 7. fairness and small-lesion recall
# ─────────────────────────────────────────────────────────────────────────────
def fairness_report(tables: dict[str, pd.DataFrame], n_boot: int = 10000,
                    seed: int = 0, verbose: bool = True) -> dict:
    """Size-stratified and ITA-conditioned performance for every arm.

    `tables` must be TEST tables normalised through `report.normalize` with the
    manifest joined, so each carries `subject`, `skin_tone_category` and
    `gt_positive_pixels`. Stage N3's arms and any headline model may be included;
    every table is binned by the SAME deciles, which is what makes the rows
    comparable across stages.

    WHAT IT RETURNS, AND WHY EACH PIECE IS HERE
    --------------------------------------------
    key            the decile assignment. `lesionsize.assign_bins` RAISES if two
                   tables disagree about an image's GT area, which is the check
                   that catches tables taken from different test sets.
    size_by_ita    the CONFOUND itself (8.4): what share of each ITA group's
                   images are small. If that varies across groups -- and in this
                   test set it does -- every unconditioned fairness number in the
                   study is confounded by exactly this much, so it is printed
                   before any fairness number rather than after.
    headline       whole set, the D1-D4 small stratum, and D1 alone, per arm.
                   Carries `wrong_place_n` = zero-Dice minus empty-prediction:
                   the failure where a model outputs pixels and all of them are
                   wrong. The per-seed tables cannot see it and it is the one a
                   clinician would care about most.
    by_bin         the full per-decile breakdown.
    contrasts      the pre-registered list at every endpoint, restricted to the
                   D1-D4 stratum, Holm-corrected within the confirmatory family.
    conditioned    per-ITA-group recall MARGINALLY and WITHIN the small stratum.
                   Compare a model's best-minus-worst gap in "all" against the
                   same gap in "D1_D4": if it shrinks, the marginal gap was
                   partly a size effect wearing a skin-tone label.
    group_dice     per-group median Dice, recall, precision and miss rate, plus
                   the Kruskal-Wallis test, via `report`. The gap is descriptive
                   and the test is inferential and they routinely disagree here;
                   both are returned so neither can be quoted alone.
    """
    from . import lesionsize as _ls
    from . import report as _rp

    key = _ls.assign_bins(tables, verbose=verbose)
    small = key.is_primary.to_numpy()

    out = {
        "key": key,
        "size_by_ita": _ls.size_by_ita(tables, key, verbose=verbose),
        "headline": _ls.headline(tables, key),
        "by_bin": _ls.by_bin(tables, key),
        "contrasts_small": _ls.contrast_table(
            tables, key, small, endpoints=PRIMARY_ENDPOINTS,
            contrasts=CONTRASTS, n_boot=n_boot, seed=seed, verbose=verbose),
        "conditioned_recall": _ls.fairness_conditioned(
            tables, key, endpoint="mean_recall", n_boot=n_boot, seed=seed,
            verbose=verbose),
        "conditioned_miss": _ls.fairness_conditioned(
            tables, key, endpoint="zero_dice_rate", n_boot=n_boot, seed=seed,
            verbose=False),
    }

    rows = []
    for name, df in tables.items():
        g = _rp.fairness_by_group(df)
        g.insert(0, "model", name)
        rows.append(g)
        out.setdefault("gaps", {})[name] = _rp.fairness_gap(df)
    out["group_dice"] = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out["size_quintiles"] = pd.concat(
        [_rp.size_quintiles(df).assign(model=name) for name, df in tables.items()],
        ignore_index=True) if tables else pd.DataFrame()
    return out


def print_fairness(fr: dict) -> None:
    """The three tables to read, in the order they should be read."""
    h = fr["headline"]
    print("=" * 104)
    print("SMALL-LESION PERFORMANCE -- read this BEFORE the Dice table")
    print("=" * 104)
    cols = ["model", "all_zero_dice_n", "all_wrong_place_n", "all_median_dice",
            "D1_D4_zero_dice_n", "D1_D4_mean_recall", "D1_D4_median_dice",
            "D1_zero_dice_n", "D1_mean_recall", "D1_median_dice"]
    print(h[[c for c in cols if c in h.columns]].to_string(index=False))
    print("\n  D1_D4 = the four smallest GT-area deciles (the small-lesion "
          "stratum).\n  wrong_place = zero Dice while predicting SOMETHING: "
          "output pixels, all wrong.\n  A model can tie on Dice and still be the "
          "one that finds small bruises.")

    c = fr["contrasts_small"]
    if not c.empty:
        print("\n" + "=" * 104)
        print("PRE-REGISTERED CONTRASTS, SMALL-LESION STRATUM (Holm within "
              "confirmatory)")
        print("=" * 104)
        cols = ["a", "b", "endpoint", "kind", "delta", "lo", "hi", "clears_zero",
                "p_holm", "survives_holm", "min_detectable"]
        print(c[[x for x in cols if x in c.columns]].to_string(index=False))
        print("\n  `survives_holm` is the ONLY column a confirmatory claim may be "
              "made from.\n  `clears_zero` is the uncorrected interval, kept so "
              "the difference is visible.\n  `min_detectable` says what this "
              "stratum could have detected -- a null with a\n  large "
              "min_detectable is 'underpowered', not 'no effect'.")

    g = fr.get("gaps", {})
    if g:
        print("\n" + "=" * 104)
        print("FAIRNESS -- descriptive gap and inferential test, per arm")
        print("=" * 104)
        for name, d in g.items():
            if not d:
                continue
            sig = "SIGNIFICANT" if d.get("significant") else "not significant"
            print(f"  {name:<24} gap {d['fairness_gap']:+.3f} "
                  f"({d['best_group']} -> {d['worst_group']}), "
                  f"miss-rate gap {d['max_miss_rate_gap']:+.3f}, "
                  f"Kruskal p={d['kruskal_p']:.4f} [{sig}]")
        print("\n  20 of the 21 ITA tests in this study are non-significant "
              "(handbook 8).\n  A visually large gap over 28 subjects in five "
              "groups usually is not one.\n  Read `conditioned_recall` next: if "
              "the gap shrinks inside D1_D4, it was size.")


# ─────────────────────────────────────────────────────────────────────────────
# 8. writing
# ─────────────────────────────────────────────────────────────────────────────
def results_dir(env) -> Path:
    """`STAGE_N4_RESULTS/` at the bundle root. Created on demand.

    Written here and nowhere else. Never `results/`, `FINAL_RESULT/`,
    `FOUNDATION_RESULTS/`, `DERM_PROBE_RESULTS/`, `STAGE_N3_RESULTS/` or
    `_work/runs/`: an experiment that fails must leave no trace in the
    directories the study's published numbers come from.
    """
    p = Path(env.root) / RESULTS_DIRNAME
    p.mkdir(parents=True, exist_ok=True)
    return p


def save(env, name: str, obj, subdir: str = "tables") -> Path:
    """Write a DataFrame (.csv) or a JSON-able object (.json) under the results dir."""
    d = results_dir(env) / subdir
    d.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, pd.DataFrame):
        p = d / f"{name}.csv"
        obj.to_csv(p, index=False)
    else:
        p = d / f"{name}.json"
        p.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    return p


def save_gate(env, gate: dict) -> list:
    """Persist the gate as a table plus a JSON verdict, and return both paths."""
    written = [save(env, "mask_supervision_gate", gate["table"])]
    payload = {k: v for k, v in gate.items() if k != "table"}
    written.append(save(env, "mask_supervision_gate", payload))
    return written


def save_fairness(env, fr: dict) -> list:
    """Persist every fairness / size table. Frames only; `gaps` goes out as JSON."""
    written = []
    for name, obj in fr.items():
        if isinstance(obj, pd.DataFrame):
            written.append(save(env, f"fairness__{name}", obj))
        elif name == "gaps":
            written.append(save(env, "fairness__gaps", obj))
    return written


# ─────────────────────────────────────────────────────────────────────────────
# 9. self-test -- no weights, no GPU, no network
# ─────────────────────────────────────────────────────────────────────────────
def self_test(verbose: bool = True) -> bool:
    """Structural checks. Everything that can be verified without a checkpoint."""
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
                  f"{'  ' + detail if detail else ''}")

    # 1. the 2-D position-embedding resample -- the patch SAM specifically needs
    pos = torch.randn(1, 64, 64, 768)
    out = resample_pos_embed_2d(pos, TARGET_GRID)
    check("pos_embed 64 -> 40 keeps [1,G,G,C]",
          tuple(out.shape) == (1, TARGET_GRID, TARGET_GRID, 768),
          f"got {tuple(out.shape)}")
    check("pos_embed resample is a no-op at the native grid",
          resample_pos_embed_2d(pos, 64).shape == pos.shape)
    try:
        resample_pos_embed_2d(torch.randn(1, 197, 768), TARGET_GRID)
        check("2-D resample REFUSES a token-sequence embedding", False)
    except ValueError:
        check("2-D resample REFUSES a token-sequence embedding", True)

    # 2. block discovery finds both layouts, and RAISES rather than returning empty
    class HFLike(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(12)])

    class SamLike(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(12)])

    for cls, want in ((HFLike, "layers"), (SamLike, "blocks")):
        blocks, where = find_sam_blocks(cls())
        check(f"find_sam_blocks({cls.__name__})",
              len(blocks) == 12 and where == want, where)
    try:
        find_sam_blocks(nn.Linear(4, 4))
        check("find_sam_blocks RAISES on an unknown tower", False)
    except RuntimeError:
        check("find_sam_blocks RAISES on an unknown tower", True)

    # 3. the decoder is Stage N3's, at this stage's width
    head = ConvDecodeHead(256, DECODER_WIDTH, 1)
    y = head(torch.zeros(2, 256, TARGET_GRID, TARGET_GRID))
    check("decoder 40x40 -> 160x160", tuple(y.shape) == (2, 1, 160, 160),
          f"got {tuple(y.shape)}")
    check("decoder stays small (<5 M)",
          sum(p.numel() for p in head.parameters()) < 5e6,
          f"{sum(p.numel() for p in head.parameters()):,} params")

    # 4. unfreezing touches exactly the last N blocks and the neck, and never zero
    class FakePatchEmbed(nn.Module):
        """Carries the `image_size` guard that HF's SamPatchEmbeddings enforces."""

        def __init__(self, native: int = 1024, patch: int = 16):
            super().__init__()
            self.image_size = (native, native)
            self.num_patches = (native // patch) ** 2
            self.projection = nn.Conv2d(3, 768, patch, stride=patch)

    class FakeSam(nn.Module):
        """Minimal SAM-shaped tower: patch embed, 2-D pos embed, 12 blocks, neck."""

        def __init__(self, grid_native: int = 64, dim: int = 768, out: int = 256):
            super().__init__()
            self.patch_embed = FakePatchEmbed()
            self.pos_embed = nn.Parameter(torch.zeros(1, grid_native,
                                                      grid_native, dim))
            self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(12)])
            self.neck = nn.Sequential(nn.Conv2d(dim, out, 1), nn.Conv2d(out, out, 3,
                                                                        padding=1))
            self.patch = 16

        def forward(self, x):                              # pragma: no cover
            raise NotImplementedError("structural fixture only")

    m = SamViTProbe.__new__(SamViTProbe)
    nn.Module.__init__(m)
    m.encoder = FakeSam()
    m.grid, m.patch, m.enc_size, m.use_neck = TARGET_GRID, 16, 640, True
    m._retarget()
    check("retarget resampled pos_embed in place",
          tuple(m.encoder.pos_embed.shape) == (1, TARGET_GRID, TARGET_GRID, 768),
          f"got {tuple(m.encoder.pos_embed.shape)}")
    check("resampled pos_embed stays frozen",
          m.encoder.pos_embed.requires_grad is False)
    # Without this relax, HF's SamPatchEmbeddings raises on the FIRST forward
    # pass at 640 -- after the model has built and reported a healthy parameter
    # count. Caught on real weights 2026-08-11; this is the regression guard.
    check("patch_embed size guard relaxed to the stage's input",
          tuple(m.encoder.patch_embed.image_size) == (640, 640),
          f"got {tuple(m.encoder.patch_embed.image_size)}")
    check("patch_embed num_patches updated",
          m.encoder.patch_embed.num_patches == TARGET_GRID ** 2,
          f"got {m.encoder.patch_embed.num_patches}")

    m._resolve_embed_dim()
    check("embed_dim read off the neck's last conv", m.embed_dim == 256,
          f"got {m.embed_dim}, source={m.feature_source}")

    info = m.unfreeze_last(UNFREEZE_BLOCKS)
    blocks = m.encoder.layers
    check("first 6 blocks stay frozen",
          all(not p.requires_grad for b in blocks[:6] for p in b.parameters()))
    check("last 6 blocks unfrozen",
          all(p.requires_grad for b in blocks[6:] for p in b.parameters()))
    check("neck unfrozen",
          all(p.requires_grad for p in m.encoder.neck.parameters()))
    check("pos_embed NOT unfrozen", not m.encoder.pos_embed.requires_grad)
    check("trainable fraction is real, not zero",
          info["encoder_trainable_fraction"] > 0.1,
          f"{100 * info['encoder_trainable_fraction']:.1f} %")
    check("frozen flag cleared", m.frozen is False)
    try:
        m.unfreeze_last(99)
        check("over-unfreeze RAISES", False)
    except ValueError:
        check("over-unfreeze RAISES", True)

    # 5. the gate refuses a dict it cannot read -- 7i.7a's void verdict
    try:
        mask_supervision_gate({"medsam_ft__seed0": pd.DataFrame(
            {"dice": [0.5], "subject": ["a"]})}, n_boot=10)
        check("gate RAISES when keyed by run_id rather than arm", False)
    except KeyError:
        check("gate RAISES when keyed by run_id rather than arm", True)

    # 6. and it refuses a table with no subject column
    try:
        mask_supervision_gate({"medsam_ft": pd.DataFrame({"dice": [0.5, 0.6]})},
                              n_boot=10)
        check("gate RAISES without a subject column", False)
    except KeyError:
        check("gate RAISES without a subject column", True)

    # 7. the contrast list is well formed and names only registered arms
    known = set(ARMS) | set(REFERENCE_DICE_VAL)
    bad = [(a, b) for a, b, _, _ in CONTRASTS if a not in known or b not in known]
    check("every contrast names a known arm", not bad, str(bad))
    check("the primary contrast is medsam - sam",
          CONTRASTS[0][0] == GATE_PRIMARY and CONTRASTS[0][1] == GATE_ATTRIBUTION)

    if verbose:
        print(f"\n  {'ALL PASS' if ok else 'FAILURES ABOVE -- do not train on this'}")
    return ok
