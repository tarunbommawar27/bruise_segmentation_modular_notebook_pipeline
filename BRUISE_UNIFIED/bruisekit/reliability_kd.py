"""Stage H -- reliability-gated distillation, and the B2 teacher for every student.

WHAT THIS STAGE ADDS, IN ONE PARAGRAPH
---------------------------------------
Two things, and they are separable on purpose. (1) A new KD *mechanism*:
`ReliabilityGatedDistillLoss`, which weights the soft term by an estimate of how
much the teacher can be trusted -- per pixel and per image -- instead of trusting
it uniformly. (2) A new KD *teacher axis*: SegFormer-B2 distilling into every
small student in the study (the four mobile architectures, plus B0 and YOLO26n),
so the Stage F question "does cross-architecture KD help?" is asked with the
study's strongest teacher rather than only with DeepLabV3+.

Crossing the two gives a 2x2 that each arm alone cannot resolve:

                         plain response KD        reliability-gated KD
    DeepLabV3+/R50       Stage F (existing)       --  (not run; see below)
    SegFormer-B2         Stage H `*_b2kd`         Stage H `*_rgkd`

WHY RELIABILITY GATING, AND WHY HERE
-------------------------------------
Handbook 7b.7 recorded a specific, predicted failure: the LR-ASPP student
distilled from DeepLabV3+ gained Dice but went from 3 complete misses to 5, and
the pre-registered explanation was that the teacher misses more than the student
does (12 of 555 vs 3) so the student inherits some of the teacher's miss
behaviour along with its ranking. The Fast-SCNN arm did the same thing harder:
misses 7 -> 13, and the skin-tone fairness gap widened from 0.136 to 0.199.

The mechanism behind that is not subtle. `DistillLoss` regresses the student onto
`sigmoid(teacher_logits / T)` at every pixel of every image with a single fixed
weight `1 - alpha`. On an image the teacher completely misses, that soft target is
not weak -- it is confidently, uniformly *empty*, and it is applied with exactly
the same force as on an image the teacher got right. The student is being told,
with full confidence, that there is no bruise on precisely the images the clinical
metric exists to catch.

Reliability gating removes that specific pathology and nothing else:

  - **Per pixel**, the soft term is down-weighted where the teacher is
    *confidently wrong* and left at full strength where the teacher is *uncertain*.
    Uncertainty is the dark knowledge distillation is for, so it must not be gated;
    confident error is noise wearing the costume of a label.
  - **Per image**, the soft term is faded out as the teacher's own agreement with
    the ground truth falls, reaching zero on an image the teacher misses outright.
    That image then trains on hard labels alone -- which is what it would have got
    from the direct baseline, and the direct baseline does not have the miss.

WHAT MAKES THIS A CLEAN CONTRAST AND NOT ANOTHER KNOB
------------------------------------------------------
The gated loss **reduces exactly to `losses.DistillLoss` in the limit where the
teacher is uniformly reliable** (`r == 1`, `g == 1`). Same alpha, same supervised
term, same BCE against the same calibrated teacher probability. So every `*_rgkd`
arm differs from its plain-KD control in one respect only: the weight the soft
term carries at each pixel. Nothing about the recipe, the teacher, the seed, the
batch or the threshold protocol moves (handbook 3, 11).

That is why `alpha` is deliberately NOT re-tuned for the gated arms. They inherit
`segformer_alpha` / `efficient_alpha` / `yolo_alpha` from their controls. A gated
arm that also had its own alpha would be two changes, and the contrast would be
uninterpretable -- the same reason handbook 3 holds the LR fixed across
architectures.

THE RELIABILITY FUNCTION
-------------------------
For a calibrated teacher probability `p` and ground truth `y` in {0, 1}:

    c = |2p - 1|            teacher confidence: 0 at p = 0.5, 1 at p in {0, 1}
    r = 1 - c * |p - y|     per-pixel reliability, in [0, 1]

Three regimes, and the shape is the whole argument:

    teacher confidently right   c ~ 1, |p-y| ~ 0   ->  r ~ 1   full KD weight
    teacher uncertain           c ~ 0              ->  r ~ 1   full KD weight
    teacher confidently wrong   c ~ 1, |p-y| ~ 1   ->  r ~ 0   KD suppressed

Note carefully what the middle row means: an uncertain teacher is *not* penalised.
A reliability function built on `|p - y|` alone would down-weight every pixel the
teacher was unsure about -- i.e. it would delete precisely the pixels that carry
information the hard label does not. Multiplying by the confidence is what keeps
uncertainty intact and removes only assertive error.

`r` is computed against the ground truth, which is available at training time and
only at training time. That is not leakage: the hard labels are already in the
loss through the supervised term. What the gate uses them for is deciding *how
much to listen to a second opinion*, not manufacturing a target.

THE PER-IMAGE GATE
-------------------
    dice_T = 2 * sum(p * y) / (sum(p) + sum(y))        soft Dice, threshold-free
    g      = clip((dice_T - gate_lo) / (gate_hi - gate_lo), 0, 1)

Defaults `gate_lo = 0.10`, `gate_hi = 0.50`.

**Soft Dice, not thresholded Dice**, so the gate needs no operating point. The
whole project selects models on threshold-free AP for the same reason (handbook
3): a gate that depended on a cut would entangle this arm with threshold choice,
and the cut is not even fitted until after training.

**A ramp, not a step.** Augmentation is re-sampled every epoch, so an image whose
teacher Dice sits near a hard boundary would flip its KD term on and off between
epochs and inject a discontinuity that has nothing to do with the image. The ramp
also means the gate degrades gracefully rather than cliff-edging.

`dice_T` is computed on the *augmented* batch the student is actually seeing, not
on a cached clean-image score. Those differ -- a crop that removes most of the
bruise genuinely is an image the teacher is less reliable on -- and the one that
matters is the one being trained on.

KEEPING alpha HONEST WHEN THE GATE FIRES
-----------------------------------------
Gating removes gradient mass from the soft term. Left alone, that would quietly
lower the *effective* KD weight, and the arm would be confounded with an alpha
change -- the exact confound the fixed recipe exists to prevent. So the freed
weight is handed back to the supervised term:

    coverage  = mean(w)                        w = g * r, in [0, 1]
    alpha_eff = alpha + (1 - alpha) * (1 - coverage)
    loss      = alpha_eff * supervised(GT) + (1 - alpha_eff) * weighted_soft

with `weighted_soft` the w-weighted mean of the per-pixel BCE against the teacher
probability -- i.e. a mean over the pixels that survived gating, on the same scale
as the ungated mean. At `coverage = 1` this is `DistillLoss` term for term. At
`coverage = 0` it is `SupervisedLoss`. Nothing in between changes the total loss
scale, so the LR schedule sees the same gradient magnitudes it was tuned for.

WHY BOTH `*_b2kd` AND `*_rgkd` EXIST FOR THE MOBILE STUDENTS
--------------------------------------------------------------
`*_rgkd` needs a control that shares its teacher. `fastscnn_distilled` uses
DeepLabV3+, so comparing `fastscnn_rgkd` against it would move the teacher and the
gate at once. `fastscnn_b2kd` is plain `DistillLoss` from B2 -- identical to
`fastscnn_distilled` except for the teacher -- so the two contrasts decompose:

    *_b2kd  vs  *_distilled     what changing the teacher does
    *_rgkd  vs  *_b2kd          what the gate does, teacher held fixed

B0 already has a plain-B2 control in Stage A (`segformer_b0_distilled`), so it
needs only the gated arm.

THE YOLO ARM IS BUILT BUT NOT REGISTERED
-----------------------------------------
`build_gated_yolo_dataset` implements the gate for YOLO's offline pseudo-mask
path and is tested, but `yolo_sem_rgkd` is **deliberately absent from
`TEACHER_FOR`, `STAGE_H_FAMILIES`, `CONTROL_FOR` and `CONTRAST_FAMILY_H`**, so
nothing schedules it and Ultralytics is never needed by this stage.

The cost of that is stated rather than hidden: every confirmatory contrast in
`CONTRAST_FAMILY_H` now tests the gate on an ONLINE loss, and the study says
nothing about whether it transfers to the offline pseudo-mask route. Holm
corrects over three contrasts instead of four. Say that when reporting.

To re-enable, three lines and nothing else:

    TEACHER_FOR["yolo_sem_rgkd"]  = "segformer_b2_teacher"
    CONTROL_FOR["yolo_sem_rgkd"]  = "yolo_sem_distilled"
    STAGE_H_FAMILIES             += ("yolo_sem_rgkd",)

plus the matching entries in `registry.STAGE_H_FAMILIES` / `STAGE_H_KIND` and a
`CONTRAST_FAMILY_H` row. The `FAMILY_SPEC` entry in `loaders.py`, the notebook's
YOLO training branch and the native-argmax scoring branch are all still in place
and were smoke-tested, so nothing else has to be rebuilt.

TopFormer and PP-MobileSeg get DeepLabV3+ arms here too (`*_distilled`), which
Stage F never defined. That completes the teacher axis to 2x4 rather than 2x2 and
costs two dict entries; whether they are trained is a seed-budget decision the
registry reports honestly either way.

WHY NOT A GATED DEEPLAB ARM
----------------------------
It would be the scientifically tidiest fourth cell, and it is deliberately left
empty rather than quietly omitted: 2x2x4 students is 24 arms before seeds, and the
question Stage H was built to answer -- does gating fix the inherited-miss failure
mode -- is answerable within one teacher. If the gate works on B2 and the licence
argument for DeepLabV3+ (see `distill_efficient`) still matters for deployment,
that cell is the obvious next run, and `TEACHER_FOR` is one line per arm.

HOW IT IS WIRED, AND WHY NOTHING IS EDITED IN PLACE
----------------------------------------------------
`engine.py` and `losses.py` are build artefacts -- handbook 16 extracts them
verbatim from `bruise_colab_baselines.ipynb`, so an edit is reverted by the next
`60_build_unified_bundle.py`. This module therefore rebinds two names at runtime,
the same pattern `distill_efficient.install_teacher_shim` and
`efficient_models.install_efficient_shim` already use:

    engine.DistillLoss   -> a dispatcher returning the gated loss for gated arms
                            and the untouched original for everything else
    engine.load_teacher  -> resolves the teacher from TEACHER_FOR for Stage H arms
                            and falls through for everything else

Both dispatch on `ACTIVE_ARM`, which the caller sets with the `arm()` context
manager, because `train_run` tells neither the loss nor the teacher loader which
run is training -- it passes only a seed-bearing path. A context manager rather
than an argument keeps the fall-through explicit: outside `arm()`, both shims are
inert and Stages A-F behave exactly as before.

`train_run` itself is untouched. It already implements distillation when the spec
says `distill=True`; Stage H changes only which teacher it loads and how the soft
term is weighted, so the arm inherits the shared recipe, the LR split, early
stopping and the whole resume contract.
"""
from __future__ import annotations

import contextlib
import json
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# the arm tables
# ─────────────────────────────────────────────────────────────────────────────
# Student family -> teacher family. One entry per Stage H arm.
#
# Every teacher here is a shipped, already-calibrated checkpoint: B2 carries the
# calibration.json that Stage A's own students distilled from (engine.train_run
# writes it for any run whose id starts with segformer_b2_teacher), and the
# DeepLabV3+ entries reuse the temperature distill_efficient fits under WORK_DIR.
#
# THE YOLO ARM IS DELIBERATELY NOT REGISTERED -- see `build_gated_yolo_dataset`.
TEACHER_FOR = {
    # ── reliability-gated, SegFormer-B2 teacher ──────────────────────────────
    "segformer_b0_rgkd": "segformer_b2_teacher",
    "fastscnn_rgkd": "segformer_b2_teacher",
    "lraspp_mobilenetv3_rgkd": "segformer_b2_teacher",
    "topformer_tiny_rgkd": "segformer_b2_teacher",
    "ppmobileseg_tiny_rgkd": "segformer_b2_teacher",
    # ── plain response KD, SegFormer-B2 teacher: the gated arms' controls ────
    "fastscnn_b2kd": "segformer_b2_teacher",
    "lraspp_mobilenetv3_b2kd": "segformer_b2_teacher",
    "topformer_tiny_b2kd": "segformer_b2_teacher",
    "ppmobileseg_tiny_b2kd": "segformer_b2_teacher",
    # ── plain response KD, DeepLabV3+ teacher: completes Stage F to 4 students ─
    "topformer_tiny_distilled": "deeplabv3plus_r50",
    "ppmobileseg_tiny_distilled": "deeplabv3plus_r50",
}

# The subset that uses the gated loss. Everything else in TEACHER_FOR trains with
# the untouched `losses.DistillLoss`, which is what makes it a control.
RGKD_FAMILIES = tuple(f for f in TEACHER_FOR if f.endswith("_rgkd"))

# Plain-KD arms defined here rather than in Stage F. Split out so the notebook can
# name them separately in the plan and the cost estimate.
B2KD_FAMILIES = tuple(f for f in TEACHER_FOR if f.endswith("_b2kd"))
DEEPLAB_EXTRA_FAMILIES = ("topformer_tiny_distilled", "ppmobileseg_tiny_distilled")

# Every Stage H family, in table order: gated arms first, then their controls.
STAGE_H_FAMILIES = (
    "segformer_b0_rgkd",
    "lraspp_mobilenetv3_rgkd", "fastscnn_rgkd",
    "topformer_tiny_rgkd", "ppmobileseg_tiny_rgkd",
    "lraspp_mobilenetv3_b2kd", "fastscnn_b2kd",
    "topformer_tiny_b2kd", "ppmobileseg_tiny_b2kd",
    "topformer_tiny_distilled", "ppmobileseg_tiny_distilled",
)

# Stage H family -> the mobile ARCHITECTURE it runs on. Only the efficient-family
# arms appear: `segformer_b0_rgkd` and `yolo_sem_rgkd` resolve through
# `loaders.FAMILY_SPEC` like any other SegFormer/YOLO run and need no alias.
STUDENT_ARCH = {
    "fastscnn_rgkd": "fastscnn",
    "lraspp_mobilenetv3_rgkd": "lraspp_mobilenetv3",
    "topformer_tiny_rgkd": "topformer_tiny",
    "ppmobileseg_tiny_rgkd": "ppmobileseg_tiny",
    "fastscnn_b2kd": "fastscnn",
    "lraspp_mobilenetv3_b2kd": "lraspp_mobilenetv3",
    "topformer_tiny_b2kd": "topformer_tiny",
    "ppmobileseg_tiny_b2kd": "ppmobileseg_tiny",
    "topformer_tiny_distilled": "topformer_tiny",
    "ppmobileseg_tiny_distilled": "ppmobileseg_tiny",
}

# Every gated arm and the control it must be read against. The contrast on the
# LEFT of each pair changes exactly one thing relative to the right.
CONTROL_FOR = {
    "segformer_b0_rgkd": "segformer_b0_distilled",
    "fastscnn_rgkd": "fastscnn_b2kd",
    "lraspp_mobilenetv3_rgkd": "lraspp_mobilenetv3_b2kd",
    "topformer_tiny_rgkd": "topformer_tiny_b2kd",
    "ppmobileseg_tiny_rgkd": "ppmobileseg_tiny_b2kd",
    "fastscnn_b2kd": "fastscnn_distilled",
    "lraspp_mobilenetv3_b2kd": "lraspp_mobilenetv3_distilled",
    "topformer_tiny_b2kd": "topformer_tiny_distilled",
    "ppmobileseg_tiny_b2kd": "ppmobileseg_tiny_distilled",
    "topformer_tiny_distilled": "topformer_tiny",
    "ppmobileseg_tiny_distilled": "ppmobileseg_tiny",
}

# Where each teacher family's shipped checkpoints live, relative to env.ckpt.
# Extends distill_efficient.TEACHER_STAGE_DIR rather than replacing it.
TEACHER_STAGE_DIR = {
    "segformer_b2_teacher": "final",
    "deeplabv3plus_r50": "baselines",
    "unet_r50": "baselines",
}


# ─────────────────────────────────────────────────────────────────────────────
# Stage H's own confirmatory family
# ─────────────────────────────────────────────────────────────────────────────
# A SEPARATE Holm family from `significance.CONTRAST_FAMILY`, deliberately.
#
# Appending these to the existing confirmatory set would re-penalise a contrast
# that has already been conducted and reported (the Stage F LR-ASPP arm, adjusted
# p = 0.0042 at k = 3). Multiplicity control is over the comparisons made to
# answer *one* question; Stage H asks a different question with its own
# pre-specified list, and correcting within it is the honest treatment.
#
# Fixed before any Stage H run existed. Edit ONLY before a run.
CONFIRMATORY = "confirmatory"
EXPLORATORY = "exploratory"

CONTRAST_FAMILY_H: list[tuple[str, str, str, str]] = [
    # ── confirmatory: gate vs its teacher-matched control ────────────────────
    # Three, not four: the YOLO arm is not registered (see build_gated_yolo_dataset),
    # so every confirmatory contrast here tests the gate on the ONLINE loss. That
    # is a narrower claim than the four-contrast version and must be stated as one.
    ("segformer_b0_rgkd", "segformer_b0_distilled", CONFIRMATORY,
     "does reliability gating help the B2->B0 student?"),
    ("lraspp_mobilenetv3_rgkd", "lraspp_mobilenetv3_b2kd", CONFIRMATORY,
     "does gating help a pretrained mobile student, teacher held fixed at B2?"),
    ("fastscnn_rgkd", "fastscnn_b2kd", CONFIRMATORY,
     "does gating help a scratch-init mobile student, teacher held fixed at B2?"),

    # ── exploratory: the teacher axis, and the students without a Stage F arm ─
    ("lraspp_mobilenetv3_b2kd", "lraspp_mobilenetv3_distilled", EXPLORATORY,
     "teacher swap: B2 vs DeepLabV3+ into the strongest mobile student"),
    ("fastscnn_b2kd", "fastscnn_distilled", EXPLORATORY,
     "teacher swap: B2 vs DeepLabV3+ into the scratch student"),
    ("topformer_tiny_rgkd", "topformer_tiny_b2kd", EXPLORATORY,
     "gating on TopFormer -- no Stage F prior, so exploratory"),
    ("ppmobileseg_tiny_rgkd", "ppmobileseg_tiny_b2kd", EXPLORATORY,
     "gating on PP-MobileSeg -- no Stage F prior, so exploratory"),
    ("lraspp_mobilenetv3_rgkd", "lraspp_mobilenetv3", EXPLORATORY,
     "gated KD against no KD at all, best mobile student"),
    ("fastscnn_rgkd", "fastscnn", EXPLORATORY,
     "gated KD against no KD at all, scratch student"),
    ("lraspp_mobilenetv3_rgkd", "segformer_b2_teacher", EXPLORATORY,
     "how much of an 8.5x larger teacher survives the gate?"),
]


# ─────────────────────────────────────────────────────────────────────────────
# the active arm -- what both shims dispatch on
# ─────────────────────────────────────────────────────────────────────────────
ACTIVE_ARM: str | None = None

# Populated by the loss dispatcher so the driver can dump gate diagnostics after
# `train_run` returns. `train_run` constructs the criterion once per run, so there
# is exactly one live instance per arm+seed and no ambiguity about whose stats
# these are.
LIVE_CRITERION = None


def set_active_arm(family: str | None) -> None:
    """Tell the shims which arm is training. `None` makes both inert."""
    global ACTIVE_ARM
    ACTIVE_ARM = family


@contextlib.contextmanager
def arm(family: str | None):
    """Scope one training run to one Stage H arm.

    Used as::

        with RK.arm(r.family):
            train_run(...)

    A context manager rather than a module flag someone sets and forgets: if the
    run raises, the flag is cleared on the way out and the next run does not
    inherit the wrong teacher. Nesting restores the previous value rather than
    clearing to None, so a driver inside a driver still behaves.
    """
    global ACTIVE_ARM
    previous = ACTIVE_ARM
    ACTIVE_ARM = family
    try:
        yield family
    finally:
        ACTIVE_ARM = previous


def is_stage_h(family: str) -> bool:
    return family in TEACHER_FOR


def is_gated(family: str | None) -> bool:
    return family in RGKD_FAMILIES


# ─────────────────────────────────────────────────────────────────────────────
# the reliability function -- shared by the online loss and the offline YOLO path
# ─────────────────────────────────────────────────────────────────────────────
def reliability(teacher_prob, target):
    """Per-pixel reliability `r = 1 - |2p - 1| * |p - y|`, in [0, 1].

    Works on torch tensors and on numpy arrays alike -- both support the four
    operations used -- which is what lets the online loss and the offline YOLO
    pseudo-mask builder share one definition instead of two that can drift.

    See the module docstring for the three regimes and why confidence multiplies
    the error term rather than the error term standing alone.
    """
    return 1.0 - abs(2.0 * teacher_prob - 1.0) * abs(teacher_prob - target)


def image_gate(teacher_prob, target, gate_lo: float = 0.10, gate_hi: float = 0.50,
               dims=(1, 2, 3), eps: float = 1e-6):
    """Per-image KD gate from the teacher's own soft Dice against the ground truth.

    Returns a tensor shaped for broadcasting against `[B, 1, H, W]` (torch path)
    holding `clip((dice_T - gate_lo) / (gate_hi - gate_lo), 0, 1)`.

    Soft Dice, so no threshold is involved -- the operating point is not fitted
    until after training, and a gate that depended on one would entangle this arm
    with threshold choice.

    `dims` defaults to the channel/spatial axes of a `[B, 1, H, W]` batch; pass
    `None` for a single un-batched map.
    """
    import torch

    inter = (teacher_prob * target).sum(dim=dims, keepdim=True)
    denom = teacher_prob.sum(dim=dims, keepdim=True) + target.sum(dim=dims, keepdim=True)
    dice_t = (2.0 * inter) / (denom + eps)
    return torch.clamp((dice_t - gate_lo) / max(gate_hi - gate_lo, eps), 0.0, 1.0), dice_t


# ─────────────────────────────────────────────────────────────────────────────
# the loss
# ─────────────────────────────────────────────────────────────────────────────
def _build_gated_loss_class():
    """Define the loss lazily so importing this module never requires torch.

    Same reason `loaders.py` imports torch inside function bodies: reading a
    Stage H table on a laptop must not need a deep-learning stack installed.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from bruisekit.losses import SupervisedLoss

    class ReliabilityGatedDistillLoss(nn.Module):
        """`alpha_eff * supervised(GT) + (1 - alpha_eff) * reliability-weighted soft`.

        Drop-in for `losses.DistillLoss`: same constructor signature, same call
        signature `(logits, aux_logits, target, teacher_prob)`, same supervised
        term. The only difference is that the soft term is a weighted mean rather
        than a plain one, and that the weight it gives up is returned to the
        supervised term so the total loss scale is unchanged.

        Reduces to `DistillLoss` exactly when every weight is 1. That property is
        what makes `*_rgkd` vs `*_b2kd` a one-variable contrast, so it is asserted
        by `self_test()` rather than left as a claim.

        Diagnostics are accumulated as running sums over optimizer-visible calls
        and read back with `stats()`. They are not optional colour: an arm whose
        gate never fired and an arm whose gate fired constantly are different
        experiments, and the tables cannot tell them apart from Dice alone.
        """

        def __init__(self, alpha: float = 0.6, aux_weight: float = 0.4,
                     gate_lo: float = 0.10, gate_hi: float = 0.50,
                     miss_gate_dice: float = 0.05):
            super().__init__()
            self.alpha = float(alpha)
            self.gate_lo = float(gate_lo)
            self.gate_hi = float(gate_hi)
            self.miss_gate_dice = float(miss_gate_dice)
            self.sup = SupervisedLoss(aux_weight)
            self.reset_stats()

        # ── diagnostics ──────────────────────────────────────────────────────
        def reset_stats(self) -> None:
            self._n_batches = 0
            self._n_images = 0
            self._sum_coverage = 0.0
            self._sum_reliability = 0.0
            self._sum_gate = 0.0
            self._sum_teacher_dice = 0.0
            self._sum_alpha_eff = 0.0
            self._n_gated_off = 0
            self._n_teacher_miss = 0

        def stats(self) -> dict:
            """What the gate actually did over this run. Safe to call at any point."""
            b = max(1, self._n_batches)
            i = max(1, self._n_images)
            return {
                "loss": "ReliabilityGatedDistillLoss",
                "alpha_nominal": self.alpha,
                "gate_lo": self.gate_lo, "gate_hi": self.gate_hi,
                "batches_seen": self._n_batches,
                "images_seen": self._n_images,
                "mean_coverage": self._sum_coverage / b,
                "mean_pixel_reliability": self._sum_reliability / b,
                "mean_image_gate": self._sum_gate / i,
                "mean_teacher_soft_dice": self._sum_teacher_dice / i,
                "mean_alpha_effective": self._sum_alpha_eff / b,
                "images_fully_gated_off": self._n_gated_off,
                "frac_images_fully_gated_off": self._n_gated_off / i,
                "teacher_near_miss_images": self._n_teacher_miss,
                "frac_teacher_near_miss": self._n_teacher_miss / i,
            }

        # ── the loss ─────────────────────────────────────────────────────────
        def forward(self, logits, aux_logits, target, teacher_prob):
            hard = self.sup(logits, aux_logits, target)

            # No gradient flows through the weights: they are a property of the
            # teacher and the label, not of the student. Letting autograd see them
            # would let the student reduce the loss by making the teacher look
            # unreliable, which is not a thing it should be able to optimise.
            with torch.no_grad():
                t = teacher_prob.detach().float()
                y = target.detach().float()
                g, dice_t = image_gate(t, y, self.gate_lo, self.gate_hi)
                r = reliability(t, y)
                w = g * r
                coverage = w.mean()

            # Per-pixel BCE, then a w-weighted mean. `reduction="none"` rather than
            # weighting the scalar: the weight varies per pixel, so it has to be
            # applied before the reduction or it is just a scaled global BCE.
            bce = F.binary_cross_entropy_with_logits(logits, teacher_prob.detach(),
                                                     reduction="none")
            denom = w.sum()
            soft = (w * bce).sum() / denom.clamp_min(1e-6)

            # Hand the gated-away weight back to the supervised term so the total
            # loss scale -- and therefore the gradient magnitude the LR schedule
            # was tuned against -- does not move with the gate.
            alpha_eff = self.alpha + (1.0 - self.alpha) * (1.0 - coverage)
            loss = alpha_eff * hard + (1.0 - alpha_eff) * soft

            with torch.no_grad():
                gv = g.reshape(g.shape[0], -1)[:, 0]
                dv = dice_t.reshape(dice_t.shape[0], -1)[:, 0]
                self._n_batches += 1
                self._n_images += int(gv.numel())
                self._sum_coverage += float(coverage)
                self._sum_reliability += float(r.mean())
                self._sum_gate += float(gv.sum())
                self._sum_teacher_dice += float(dv.sum())
                self._sum_alpha_eff += float(alpha_eff)
                self._n_gated_off += int((gv <= 1e-6).sum())
                self._n_teacher_miss += int((dv <= self.miss_gate_dice).sum())

            return loss

    return ReliabilityGatedDistillLoss


_GATED_CLASS = None


def gated_loss_class():
    """The gated loss class, built on first use."""
    global _GATED_CLASS
    if _GATED_CLASS is None:
        _GATED_CLASS = _build_gated_loss_class()
    return _GATED_CLASS


# ─────────────────────────────────────────────────────────────────────────────
# shim 1: the loss
# ─────────────────────────────────────────────────────────────────────────────
def install_loss_shim(gate_lo: float = 0.10, gate_hi: float = 0.50,
                      verbose: bool = True):
    """Rebind `engine.DistillLoss` to a dispatcher keyed on `ACTIVE_ARM`.

    `train_run` resolves `DistillLoss` as a module global at call time, and the
    global it resolves is the one bound in `bruisekit.engine` -- so that is the
    name that has to move, exactly as `install_efficient_shim` moves
    `build_model` in both `bruisekit.models` and `bruisekit.engine`.

    Outside an `arm()` for a gated family the dispatcher returns the original
    `losses.DistillLoss` unchanged, so Stage A's `segformer_b0_distilled` and
    Stage F's arms are bit-for-bit unaffected by this shim being installed.

    Installing twice does not wrap the wrapper: `_original` is preserved so the
    fall-through does not depend on call order.
    """
    import bruisekit.engine as _be

    previous = getattr(_be.DistillLoss, "_original", _be.DistillLoss)
    Gated = gated_loss_class()

    def DistillLoss(alpha=0.5, aux_weight=0.4):
        global LIVE_CRITERION
        if is_gated(ACTIVE_ARM):
            crit = Gated(alpha=alpha, aux_weight=aux_weight,
                         gate_lo=gate_lo, gate_hi=gate_hi)
            LIVE_CRITERION = crit
            if verbose:
                print(f"  [loss] {ACTIVE_ARM}: reliability-gated KD "
                      f"(alpha={alpha}, gate {gate_lo:.2f}->{gate_hi:.2f})")
            return crit
        LIVE_CRITERION = None
        return previous(alpha, aux_weight)

    DistillLoss._original = previous
    _be.DistillLoss = DistillLoss
    if verbose:
        print(f"engine.DistillLoss patched -> reliability-gated for "
              f"{len(RGKD_FAMILIES)} arm(s); unchanged for everything else.")
    return DistillLoss


def dump_stats(run_dir: Path) -> dict | None:
    """Write the live criterion's gate diagnostics to `run_dir/reliability_gate.json`.

    Returns the stats, or None when the last run was not a gated arm. Called by
    the driver after `train_run` returns, because `train_run` has no idea a gate
    exists and must not be edited to learn (handbook 16).
    """
    crit = LIVE_CRITERION
    if crit is None:
        return None
    s = crit.stats()
    dest = Path(run_dir) / "reliability_gate.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(s, indent=2))
    return s


def gate_report(env, families=None, seeds=(0, 1, 2)):
    """Collect every arm's `reliability_gate.json` into one table.

    An arm's Dice is not interpretable without this: a gate that never fired makes
    `*_rgkd` a relabelled `*_b2kd`, and a gate that fired on half the images is a
    different experiment from one that fired on 2 %. Reported next to the results,
    never in an appendix.
    """
    import pandas as pd

    families = RGKD_FAMILIES if families is None else tuple(families)
    rows = []
    for family in families:
        for seed in seeds:
            p = env.runs / f"{family}__seed{seed}" / "reliability_gate.json"
            if not p.exists():
                continue
            rows.append({"family": family, "seed": seed, **json.loads(p.read_text())})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# shim 2: the teacher
# ─────────────────────────────────────────────────────────────────────────────
def teacher_dir(env, family: str, seed: int) -> Path:
    """The shipped checkpoint directory for one (teacher family, seed).

    Prefers `checkpoints/`, then a fresh run under `WORK_DIR/runs/` -- a session
    that just trained its own B2 should be able to distil from it without the
    checkpoint having to be copied into the read-only bundle.
    """
    sub = TEACHER_STAGE_DIR.get(family)
    if sub is None:
        raise KeyError(f"no shipped location known for teacher family {family!r}; "
                       f"known: {sorted(TEACHER_STAGE_DIR)}")
    for d in (env.ckpt / sub / f"{family}__seed{seed}", env.runs / f"{family}__seed{seed}"):
        if (d / "best.pt").exists():
            return d
    raise FileNotFoundError(
        f"teacher {family}__seed{seed} has no best.pt in checkpoints/{sub}/ or "
        f"{env.runs}. Stage A ships all three B2 seeds and Stage B all three "
        f"DeepLabV3+ seeds; check the bundle.")


def teacher_temperature(env, family: str, seed: int, cfg: dict, man640: dict | None,
                        verbose: bool = True) -> float:
    """The teacher's calibrated temperature, fitted only if it is not already known.

    Two sources, in this order, and the order is the point:

      1. A `calibration.json` NEXT TO the checkpoint. B2 has one because
         `engine.train_run` fits it at the end of any `segformer_b2_teacher` run --
         so a Stage H student distils from exactly the temperature Stage A's own
         students did, and the two arms are comparable without re-deriving
         anything.
      2. Otherwise `distill_efficient.ensure_calibration`, which fits it with
         `engine.calibrate_temperature` on the 134 val images and caches it under
         WORK_DIR. That is the DeepLabV3+ path; those baselines were trained as
         endpoints, not teachers, so they ship no temperature.

    Never falls back to T = 1. Distilling from an uncalibrated teacher regresses
    the student onto a near-binary soft label -- the hard label with extra steps --
    and the arm would silently measure nothing.
    """
    shipped = teacher_dir(env, family, seed) / "calibration.json"
    if shipped.exists():
        t = float(json.loads(shipped.read_text())["temperature"])
        if verbose:
            print(f"  [cal] {family}__seed{seed}: T={t:.4f} (shipped with the checkpoint)")
        return t

    from bruisekit import distill_efficient as DE

    # The cached fit under WORK_DIR is checked BEFORE demanding manifests: the
    # teacher shim runs inside `train_run`, where MAN640 is not in scope, and a
    # temperature fitted earlier in the session is the whole point of caching it.
    cached = DE.calibration_path(env, family, seed)
    if cached.exists():
        t = float(json.loads(cached.read_text())["temperature"])
        if verbose:
            print(f"  [cal] {family}__seed{seed}: T={t:.4f} (cached)")
        return t

    if man640 is None:
        raise RuntimeError(
            f"{family}__seed{seed} has no shipped calibration.json and none was "
            f"fitted earlier in this session, so its temperature is unknown at the "
            f"point the teacher is loaded. Call\n"
            f"    RK.teacher_temperature(env, {family!r}, {seed}, CFG, MAN640)\n"
            f"once before training -- the Stage H setup cell does this for every "
            f"teacher the session will ask for, precisely so this fails in seconds "
            f"rather than after a model has been built.")
    return float(DE.ensure_calibration(env, family, seed, cfg, man640,
                                       verbose=verbose)["temperature"])


def install_teacher_shim(env, cfg: dict, teacher_seed_mode: str = "same",
                         fixed_seed: int = 0, verbose: bool = True):
    """Redirect `engine.load_teacher` for Stage H arms; fall through for the rest.

    `train_run` builds the teacher path itself and hands `load_teacher` a directory
    spelled `.../segformer_b2_teacher__seed{k}` under `WORK_DIR/runs/`. Two things
    are wrong with that for Stage H and this shim fixes both: the B2 checkpoints
    live in `checkpoints/final/`, not the work dir, and half the Stage H arms want
    DeepLabV3+ rather than B2. The seed is the only piece of that path worth
    keeping, and it is taken off the end.

    Dispatches on `ACTIVE_ARM`. Outside a Stage H `arm()` this shim calls whatever
    was bound before it -- which, in a session that also trains Stage F, is
    `distill_efficient`'s shim. Install this one LAST so Stage H arms are resolved
    here and everything else keeps the behaviour it had.

    teacher_seed_mode
        "same"  -- student seed k distils from teacher seed k. Stage A's and
                   Stage F's convention: the student's reported spread then
                   includes the teacher's own variance, which is part of the
                   pipeline being measured.
        "fixed" -- every student distils from `fixed_seed`. Narrower spread that no
                   longer measures teacher variance; say so if you use it.
    """
    if teacher_seed_mode not in ("same", "fixed"):
        raise ValueError(f"teacher_seed_mode must be 'same' or 'fixed', "
                         f"got {teacher_seed_mode!r}")

    import torch
    import bruisekit.engine as _be
    from bruisekit import loaders as L

    previous = getattr(_be.load_teacher, "_original", _be.load_teacher)
    cache: dict[tuple[str, int], float] = {}

    def load_teacher(dir_from_train_run, paths: dict, device, amp: bool):
        family = TEACHER_FOR.get(ACTIVE_ARM)
        if family is None:
            return previous(dir_from_train_run, paths, device, amp)

        name = Path(dir_from_train_run).name
        try:
            student_seed = int(name.rsplit("seed", 1)[1])
        except (IndexError, ValueError):
            return previous(dir_from_train_run, paths, device, amp)

        seed = student_seed if teacher_seed_mode == "same" else fixed_seed
        d = teacher_dir(env, family, seed)
        key = (family, seed)
        if key not in cache:
            cache[key] = teacher_temperature(env, family, seed, cfg, None, verbose=verbose)
        temperature = cache[key]

        # encoder_weights=None on the load path -- handbook 15 trap 5: SMP
        # otherwise fetches ImageNet ResNet-50 only to have every weight of it
        # overwritten by the line below.
        model = L.build_for_load(env, family, cfg.get("smp_encoder", "resnet50"))
        model.load_state_dict(torch.load(str(d / "best.pt"), map_location=device,
                                         weights_only=True))
        model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)

        def teacher_fn(x):
            # no_grad for the same reason as the original: the teacher is frozen,
            # and a backward graph through it is the largest avoidable allocation
            # in the distillation setup.
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=amp):
                logits = model(x)
            return torch.sigmoid(logits.float() / temperature)

        teacher_fn.temperature = temperature
        teacher_fn.source = f"{family}__seed{seed}"
        if verbose:
            print(f"  [teacher] {ACTIVE_ARM} <- {family}__seed{seed}  "
                  f"T={temperature:.4f}  (mode={teacher_seed_mode})")
        return teacher_fn

    load_teacher._original = previous
    _be.load_teacher = load_teacher
    if verbose:
        print(f"engine.load_teacher patched -> Stage H teachers for "
              f"{len(TEACHER_FOR)} arm(s); falls through otherwise.")
    return load_teacher


def register_student_aliases():
    """Make every mobile Stage H arm resolve to its architecture. Idempotent.

    Same mechanism and the same reason as `distill_efficient.register_student_aliases`:
    `EFFICIENT_ARCHS`, `PUBLISHED_PARAMS_M`, `_HEAD_IN_CHANNELS` and
    `weights.SOURCES` are keyed by ARCH at the definition site and looked up by
    FAMILY at the call site, and a distilled variant is a new family on an existing
    architecture. Registered here rather than in `efficient_models` so that dict
    keeps meaning "the four architectures" and `self_test` keeps checking each one
    once.
    """
    from bruisekit import distill_efficient as DE
    return [DE.register_student_alias(family, arch)
            for family, arch in STUDENT_ARCH.items()]


# ─────────────────────────────────────────────────────────────────────────────
# the YOLO arm -- offline, because Ultralytics trains itself
# ─────────────────────────────────────────────────────────────────────────────
def build_gated_yolo_dataset(df, work_root: Path, out_dir: Path, split: str,
                             alpha: float, teacher_prob_fn, imgsz: int = 640,
                             gate_lo: float = 0.10, gate_hi: float = 0.50,
                             stats_out: Path | None = None) -> dict:
    """`yolo_native.build_yolo_dataset`, with the fusion weight gated per pixel.

    YOLO never sees `DistillLoss`: it trains natively under Ultralytics, and the
    teacher reaches it only through the pseudo-mask baked into the training labels
    (handbook 4). So the gate has to be applied where the fusion happens, offline,
    rather than in a loss.

    Stage A's fusion is a single scalar::

        class = (alpha * gt + (1 - alpha) * teacher_prob >= 0.5)

    The gated version makes `alpha` a per-pixel quantity that rises to 1 -- i.e.
    pure ground truth -- exactly where the teacher is unreliable::

        a_pix = alpha + (1 - alpha) * (1 - g * r)
        class = (a_pix * gt + (1 - a_pix) * teacher_prob >= 0.5)

    with the same `r` and `g` the online loss uses. At `g * r == 1` this is Stage
    A's fusion character for character, which is what makes `yolo_sem_rgkd` vs
    `yolo_sem_distilled` a one-variable contrast.

    THIS IS A DELIBERATE COPY, NOT A REFACTOR
    ------------------------------------------
    `yolo_native.py` is extracted verbatim from `bruise_colab_baselines.ipynb` at
    build time (handbook 16), so a fusion change made there is reverted by the next
    build. It is copied rather than shimmed because a shim would have to intercept
    a loop body, and handbook 15 trap 11 is explicit that when a tested call
    sequence exists you copy it rather than write a new one. Everything except the
    two fusion lines is the original: same symlink-then-copy fallback, same
    native-resolution masks (Ultralytics letterboxes internally, so pre-resizing
    would apply the resize twice), same clean GT for val.

    STATUS: IMPLEMENTED AND TESTED, BUT THE ARM IS NOT REGISTERED
    --------------------------------------------------------------
    `yolo_sem_rgkd` is absent from `TEACHER_FOR` / `STAGE_H_FAMILIES` /
    `CONTROL_FOR` / `CONTRAST_FAMILY_H`, so nothing schedules it and this function
    is not currently called. It is kept rather than deleted because it is the only
    implementation of the gate on the OFFLINE pseudo-mask route, and re-enabling
    the arm is three lines (see the module docstring). What the study loses in the
    meantime is stated there and must be repeated when reporting: every
    confirmatory Stage H contrast tests the gate on an online loss only.

    Returns a diagnostics dict, and writes it to `stats_out` when given -- the same
    "an ungated gate is a different experiment" argument as `gate_report`.
    """
    import shutil

    import cv2
    import numpy as np
    from bruisekit.yolo_native import _read_native_mask

    img_dir = out_dir / "images" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    msk_dir = out_dir / "masks" / split
    msk_dir.mkdir(parents=True, exist_ok=True)

    n_img = 0
    n_gated_off = 0
    sum_gate = 0.0
    sum_rel = 0.0
    sum_dice_t = 0.0
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
            prob = teacher_prob_fn(work_root / r.image_path, gt.shape).astype(np.float32)

            dice_t = float(2.0 * (prob * gt).sum() / (prob.sum() + gt.sum() + 1e-6))
            g = float(np.clip((dice_t - gate_lo) / max(gate_hi - gate_lo, 1e-6), 0.0, 1.0))
            rel = reliability(prob, gt)
            a_pix = alpha + (1.0 - alpha) * (1.0 - g * rel)
            cls = ((a_pix * gt + (1.0 - a_pix) * prob) >= 0.5).astype(np.uint8)

            n_img += 1
            n_gated_off += int(g <= 1e-6)
            sum_gate += g
            sum_rel += float(rel.mean())
            sum_dice_t += dice_t
        else:
            cls = gt.astype(np.uint8)
        cv2.imwrite(str(msk_dir / f"{r.stem}.png"), cls)

    n = max(1, n_img)
    stats = {"split": split, "images_fused": n_img,
             "alpha_nominal": alpha, "gate_lo": gate_lo, "gate_hi": gate_hi,
             "mean_image_gate": sum_gate / n,
             "mean_pixel_reliability": sum_rel / n,
             "mean_teacher_soft_dice": sum_dice_t / n,
             "images_fully_gated_off": n_gated_off,
             "frac_images_fully_gated_off": n_gated_off / n}
    if stats_out is not None and n_img:
        Path(stats_out).parent.mkdir(parents=True, exist_ok=True)
        Path(stats_out).write_text(json.dumps(stats, indent=2))
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# the driver
# ─────────────────────────────────────────────────────────────────────────────
def train_arm(env, family: str, seeds, cfg: dict, man640: dict,
              teacher_seed_mode: str = "same", fixed_seed: int = 0,
              verbose: bool = True) -> list:
    """Train one Stage H arm at each seed. Returns the run_ids it touched.

    Uses `engine.train_run` unmodified, so the arm inherits the shared recipe, the
    LR split, early stopping and the whole resume contract (`DONE.json` to skip,
    `resume.pt` every few epochs). Both shims are installed here and the run is
    wrapped in `arm()`, so calling this is sufficient -- no separate setup step
    that could be forgotten.

    YOLO arms are NOT handled here: they train natively under Ultralytics through
    `yolo_native.train_native` on a pseudo-mask dataset, which is a different
    sequence entirely, and the notebook's training cell dispatches on kind. No
    YOLO arm is registered at present -- the guard below is kept so that
    re-enabling one fails with an explanation rather than by silently taking the
    wrong path.

    `train_arm` trains but does not threshold-sweep or score. Run the notebook's
    Stage H threshold cell afterwards, or use `train_missing`, which does both and
    skips whatever already has a `DONE.json`.
    """
    import torch
    from bruisekit import loaders as L
    from bruisekit.engine import train_run

    if family not in TEACHER_FOR:
        raise KeyError(f"{family!r} is not a Stage H arm; known: {sorted(TEACHER_FOR)}")
    if family == "yolo_sem_rgkd":
        raise ValueError(
            "yolo_sem_rgkd trains natively under Ultralytics, not through "
            "train_run. Use the notebook's Stage H training cell, which builds the "
            "gated pseudo-mask dataset with build_gated_yolo_dataset first.")
    if not str(env.device).startswith("cuda"):
        raise RuntimeError(
            f"train_arm would train {len(tuple(seeds))} run(s) but device is "
            f"{env.device}. Distillation runs a frozen teacher on every step; this "
            f"needs a GPU session.")

    register_student_aliases()
    install_loss_shim(cfg.get("rgkd_gate_lo", 0.10), cfg.get("rgkd_gate_hi", 0.50),
                      verbose=verbose)
    install_teacher_shim(env, cfg, teacher_seed_mode=teacher_seed_mode,
                         fixed_seed=fixed_seed, verbose=verbose)

    # Fail in seconds rather than after a model is built if a temperature cannot
    # be resolved. DeepLabV3+ needs fitting; B2 ships one.
    teacher_family = TEACHER_FOR[family]
    needed = sorted(set(seeds)) if teacher_seed_mode == "same" else [fixed_seed]
    for s in needed:
        teacher_temperature(env, teacher_family, s, cfg, man640, verbose=verbose)

    spec = L.spec_for(family)
    alpha = cfg.get("efficient_alpha" if family in STUDENT_ARCH else "segformer_alpha", 0.6)
    run_cfg = {**cfg, "alpha": alpha}

    done = []
    for seed in seeds:
        run_id = f"{family}__seed{seed}"
        if verbose:
            print(f"\n{'=' * 70}\n{run_id}  (teacher: {teacher_family}, alpha={alpha}, "
                  f"gated={is_gated(family)})\n{'=' * 70}")
        with arm(family):
            res = train_run(run_id, spec, seed, run_cfg, env.paths_for_models(),
                            man640, env.cache640, env.runs, env.device)
            s = dump_stats(env.runs / run_id)
        if verbose:
            print(f"  -> {res.get('status', 'trained')}")
            if s:
                print(f"     gate: coverage {s['mean_coverage']:.4f}, "
                      f"{s['images_fully_gated_off']} of {s['images_seen']} image-views "
                      f"fully gated off")
        done.append(run_id)
        if str(env.device).startswith("cuda"):
            torch.cuda.empty_cache()
    return done


# ─────────────────────────────────────────────────────────────────────────────
# verification
# ─────────────────────────────────────────────────────────────────────────────
def self_test(verbose: bool = True):
    """Prove the three properties the Stage H contrasts depend on. Seconds, CPU only.

    1. **Reduction.** With `r == 1` and `g == 1` the gated loss equals
       `losses.DistillLoss` to floating-point tolerance. If this fails, `*_rgkd`
       vs `*_b2kd` is not a one-variable contrast and no Stage H number means what
       the tables say it means.
    2. **Gating.** An image the teacher completely misses contributes no soft term,
       and the loss on it equals the plain supervised loss.
    3. **Uncertainty is preserved.** A maximally uncertain teacher (p = 0.5) keeps
       full weight, which is the property that distinguishes this from
       down-weighting by error alone.

    Returns a DataFrame of checks; raises on failure rather than printing a red row,
    because a silently-broken gate produces plausible numbers.
    """
    import pandas as pd
    import torch

    from bruisekit.losses import DistillLoss

    Gated = gated_loss_class()
    torch.manual_seed(0)
    B, H, W = 4, 32, 32
    logits = torch.randn(B, 1, H, W, requires_grad=True)
    target = (torch.rand(B, 1, H, W) > 0.7).float()

    rows = []

    # 1. reduction: a teacher that is perfectly confident and perfectly right has
    #    r == 1 everywhere and dice_T == 1, so g == 1 and coverage == 1.
    perfect = target.clone().clamp(1e-6, 1 - 1e-6)
    g_loss = Gated(alpha=0.6, aux_weight=0.4)(logits, None, target, perfect)
    p_loss = DistillLoss(alpha=0.6, aux_weight=0.4)(logits, None, target, perfect)
    d = float(abs(g_loss - p_loss).detach())
    rows.append({"check": "reduces to DistillLoss when the teacher is reliable",
                 "value": d, "tol": 2e-3, "pass": d < 2e-3})

    # 2. gating: a teacher asserting empty on an image with a real bruise.
    empty = torch.full_like(target, 1e-6)
    crit = Gated(alpha=0.6, aux_weight=0.4)
    g_loss = crit(logits, None, target, empty)
    from bruisekit.losses import SupervisedLoss
    s_loss = SupervisedLoss(0.4)(logits, None, target)
    d = float(abs(g_loss - s_loss).detach())
    st = crit.stats()
    rows.append({"check": "teacher-miss image falls back to supervised only",
                 "value": d, "tol": 1e-4, "pass": d < 1e-4})
    rows.append({"check": "teacher-miss image is counted as fully gated off",
                 "value": st["frac_images_fully_gated_off"], "tol": 1.0,
                 "pass": st["frac_images_fully_gated_off"] == 1.0})

    # 3. uncertainty preserved: p = 0.5 everywhere -> r == 1 everywhere.
    r = reliability(torch.full_like(target, 0.5), target)
    d = float(abs(r - 1.0).max())
    rows.append({"check": "an uncertain teacher keeps full weight (r == 1 at p=0.5)",
                 "value": d, "tol": 1e-6, "pass": d < 1e-6})

    # 4. confident error is suppressed.
    r_wrong = reliability(torch.ones_like(target) * (1 - target), target)
    d = float(r_wrong.max())
    rows.append({"check": "a confidently wrong teacher is fully suppressed (r == 0)",
                 "value": d, "tol": 1e-6, "pass": d < 1e-6})

    out = pd.DataFrame(rows)
    if verbose:
        for _, row in out.iterrows():
            print(f"  {'PASS' if row['pass'] else 'FAIL'}  {row['check']:<52} "
                  f"|delta| = {row['value']:.3e}")
    bad = out[~out["pass"]]
    if len(bad):
        raise AssertionError(
            "reliability_kd.self_test failed:\n" + bad.to_string(index=False) +
            "\nA gate that does not reduce to its control makes every Stage H "
            "contrast a two-variable comparison. Do not train on this.")
    return out
