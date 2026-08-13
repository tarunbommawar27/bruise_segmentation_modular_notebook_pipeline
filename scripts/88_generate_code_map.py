#!/usr/bin/env python
"""Emit `docs/CODE_MAP.md` -- where every piece of functionality actually lives.

    python scripts/88_generate_code_map.py

WHY THIS IS GENERATED AND NOT WRITTEN BY HAND
-----------------------------------------------
A hand-written file:line index is wrong the first time anyone edits the file
above the line it points at, and nothing tells you it went wrong -- you follow
the pointer, land in the middle of an unrelated function, and conclude the doc is
untrustworthy. So every entry here is a REGEX that is searched for at build time,
and the line number is whatever the code says today.

It also FAILS LOUDLY. If an anchor no longer matches, the build stops and names
it. That is the point: an anchor that stopped matching means the thing it
described was renamed or deleted, and the map should not be regenerated with a
silent gap where it used to be.

Each entry additionally carries the search that finds it again, so a reader whose
copy has drifted can re-locate it without this script.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "BRUISE_UNIFIED"
DST = ROOT / "docs" / "CODE_MAP.md"

#: (section, [(what, file, regex, why-it-matters)])
#: The regex is matched against whole lines; the FIRST match wins.
MAP: list[tuple[str, str, list[tuple[str, str, str, str]]]] = [
    (
        "Attaching a segmentation head to an encoder that has none",
        "Foundation models output a grid of features, not a mask. These are the "
        "three places a head gets built, and they are deliberately the same head "
        "so cross-stage numbers stay comparable.",
        [
            ("The head itself — 1M-param conv decoder", "bruisekit/finetune_n3.py",
             r"^class ConvDecodeHead",
             "Two 3x3 convs + BN + ReLU, then a 1x1 to one logit, then bilinear "
             "upsample to full resolution. Stage N3, N4 and O all import THIS "
             "class rather than re-typing it."),
            ("Linear probe head (frozen-encoder arms)", "bruisekit/foundation.py",
             r"^class LinearProbeHead",
             "Deliberately weak: a 1x1 conv and nothing else, so a strong decoder "
             "cannot paper over a weak encoder in the frozen comparison."),
            ("Where the head is attached — base contract",
             "bruisekit/dermprobe.py", r"def _build_head",
             "`_ProbeBase` is the architecture-blind contract every probe "
             "subclasses: raw [0,1] pixels in, full-resolution logits out."),
            ("Head attached — SAM / MedSAM", "bruisekit/samprobe.py",
             r"def _build_head",
             "Overrides the linear head with the SAME ConvDecodeHead Stage N3 "
             "used, or the two stages are not comparable."),
        ],
    ),
    (
        "Freezing the encoder, and unfreezing part of it",
        "The difference between 'train only the head' and 'train the last six "
        "blocks'. UNFREEZE_BLOCKS = 6 in both fine-tuning stages, pre-registered.",
        [
            ("Freeze everything (frozen probe)", "bruisekit/dermprobe.py",
             r"def _freeze",
             "Sets requires_grad False on the whole encoder AND pins it to eval() "
             "so BatchNorm/dropout cannot drift — freezing weights alone does not "
             "do that."),
            ("Unfreeze the last N blocks — DINOv2 / DermLIP",
             "bruisekit/finetune_n3.py", r"def unfreeze_last",
             "Last 6 of 12 transformer blocks plus the final LayerNorm. The norm "
             "is included because its statistics are tuned to the blocks below it."),
            ("Unfreeze the last N blocks — SAM / MedSAM",
             "bruisekit/samprobe.py", r"def unfreeze_last",
             "Same 6 blocks, plus the neck. RAISES if it ends up with zero "
             "trainable params — an arm that silently trains nothing scores "
             "low-to-mid and is indistinguishable from a real result."),
            ("How many blocks — the pre-registered constant",
             "bruisekit/samprobe.py", r"^UNFREEZE_BLOCKS = ",
             "Changing this changes the experiment. Do it in the module, "
             "deliberately, and say so in the write-up."),
        ],
    ),
    (
        "Loss functions",
        "Four losses, each a strict extension of the one above it. That nesting "
        "is what makes every distillation contrast one-variable.",
        [
            ("Dice + BCE — the supervised base", "bruisekit/losses.py",
             r"^class DiceBCELoss",
             "Dice is computed PER IMAGE then averaged, not pooled over the "
             "batch — a batch-pooled Dice lets one large bruise dominate the "
             "gradient."),
            ("Supervised loss (+ auxiliary head)", "bruisekit/losses.py",
             r"^class SupervisedLoss", "What every non-distilled arm trains on."),
            ("Distillation loss", "bruisekit/losses.py", r"^class DistillLoss",
             "alpha * DiceBCE(student, GT) + (1-alpha) * BCE(student, calibrated "
             "teacher prob). NOT Hinton KD — the module docstring above it "
             "explains the difference and why it matters for the paper."),
            ("Reliability-gated KD loss (Stage H)",
             "bruisekit/reliability_kd.py", r"def _build_gated_loss_class",
             "Down-weights the teacher per pixel where it is confidently wrong. "
             "Reduces to DistillLoss exactly when the gate is fully open."),
            ("Per-image routed multi-teacher loss (Stage M)",
             "bruisekit/multiteacher.py", r"def _build_routed_loss_class",
             "softmax(beta * per-teacher soft Dice against the label). K=1 "
             "reduces to the Stage H loss bit-for-bit."),
            ("Per-ITA-group routed loss (Stage O)", "bruisekit/itakd.py",
             r"def _build_group_loss_class",
             "Weights fitted once on validation, indexed by the image's skin-tone "
             "group. RAISES if the group vector is missing rather than falling "
             "back to uniform weights."),
        ],
    ),
    (
        "The training loop",
        "One driver for every model in the study. A bespoke loop anywhere would "
        "make that arm's numbers unreadable against the rest.",
        [
            ("The loop itself", "bruisekit/engine.py", r"def train_run",
             "Idempotent and resumable: DONE.json returns immediately, resume.pt "
             "restores model+optimizer+scaler+epoch."),
            ("Backbone/head learning-rate split", "bruisekit/models.py",
             r"def build_param_groups",
             "6e-5 backbone / 6e-4 head, and no weight decay on norms and biases. "
             "Membership is by id(), not by name — a name-prefix rule would put "
             "every YOLO parameter in the wrong group."),
            ("LR schedule — warmup then poly decay", "bruisekit/engine.py",
             r"def lr_multiplier",
             "Applied per OPTIMIZER STEP, not per micro-batch, or accumulation "
             "advances the schedule too fast."),
            ("Batch-size probe", "bruisekit/engine.py", r"def resolve_micro_batch",
             "Measures what actually fits, then accumulates back to the target "
             "effective batch."),
            ("Teacher temperature calibration", "bruisekit/engine.py",
             r"def calibrate_temperature",
             "Fitted by NLL on validation (Guo et al. 2017). Never falls back to "
             "T=1 — an uncalibrated teacher's soft label is the hard label with "
             "extra steps."),
        ],
    ),
    (
        "Choosing the operating point (threshold)",
        "Fitted on validation, applied once to test. Nothing is ever fitted on "
        "test.",
        [
            ("Cache logits once, sweep 481 cuts", "bruisekit/sweep.py",
             r"def cache_logits",
             "481 cuts x 134 images would be 64k forward passes; caching makes "
             "the sweep pure tensor arithmetic."),
            ("The sweep", "bruisekit/sweep.py", r"def sweep_cuts",
             "Reductions are exact int64 on boolean masks. fp16 sums drift by "
             "~1.5e-4, which is a tenth of the signal the tie band ranks on."),
            ("Picking the cut", "bruisekit/sweep.py", r"def select_cut",
             "NOT the argmax. Every cut within one standard error of the peak is "
             "statistically tied; the tie is broken by LOWEST MISS RATE."),
            ("Reading a stored cut", "bruisekit/registry.py", r"def read_cut",
             "The single place that converts between the logit and probability "
             "threshold dialects. sigmoid(cut) == threshold, so reading one as "
             "the other does not raise — it silently moves the boundary."),
        ],
    ),
    (
        "Data, augmentation and the 640 cache",
        None,
        [
            ("Dataset", "bruisekit/data.py", r"^class BruiseDataset",
             "Returns (x[3,H,W] in [0,1], y[1,H,W] in {0,1}, stem). Emits RAW "
             "[0,1] — normalisation happens inside each model's forward."),
            ("Augmentation", "bruisekit/data.py", r"def build_augmentation",
             "Train-time only; validation and test see the deterministic resize."),
            ("Dataloader", "bruisekit/data.py", r"def make_loader",
             "Seeded shuffling and seeded per-worker RNG, so augmentation is "
             "reproducible for a given seed."),
            ("Building the 640 cache", "bruisekit/loaders.py",
             r"def build_cache640",
             "Bilinear for images, NEAREST for masks. Bilinear on a mask gives "
             "fractional boundary pixels, and thresholding those erodes small "
             "bruises — exactly the population the miss metric is about."),
        ],
    ),
    (
        "Scoring and metrics",
        None,
        [
            ("Per-image Dice", "bruisekit/metrics.py", r"def dice_np",
             "Counts pixels. Both-empty scores 1.0, matching the sweep."),
            ("Score a model at a cut", "bruisekit/evaluate.py",
             r"def evaluate_at_cut", "Produces the per-image table everything "
             "downstream reads."),
            ("Normalise any producer's CSV", "bruisekit/report.py",
             r"def normalize",
             "Joins subject and ITA group from the manifest, and derives "
             "complete_miss = (dice == 0). The manifest is the single source of "
             "truth for who each image belongs to."),
            ("Run inference over the registry", "bruisekit/inference.py",
             r"def inference_pass",
             "Delegates to loaders.score_run. There is deliberately no second "
             "inference implementation anywhere in the codebase."),
            ("Best seed per model", "bruisekit/inference.py", r"def resolve_runs",
             "Read from the selection step's own filenames. It is 0 for the "
             "SegFormers and 2 for yolo_sem_direct; scoring a model at another "
             "model's best seed shows per-image gaps up to 0.49 Dice."),
        ],
    ),
    (
        "Analysis — misses, size, fairness",
        None,
        [
            ("Miss taxonomy: blank vs wrong-place", "bruisekit/itakd.py",
             r"def _miss_counts",
             "wrong_place is DERIVED as zero_dice - empty_pred so the three "
             "columns cannot fail to add up. RAISES if empty > zero, which is "
             "arithmetically impossible and means the table is inconsistent."),
            ("Size deciles, cut once globally", "bruisekit/lesionsize.py",
             r"def assign_bins",
             "RAISES if two models disagree about an image's GT area — that means "
             "different mask versions and every cross-model column would be "
             "meaningless."),
            ("Fairness conditioned on lesion size", "bruisekit/lesionsize.py",
             r"def fairness_conditioned",
             "Reports each skin-tone gap marginally AND within the small-lesion "
             "stratum. Where the gap shrinks, it was partly a size effect."),
            ("Cross-host table discovery", "bruisekit/allmodels.py",
             r"def discover",
             "Scans every root that has ever held a per-image CSV. See handbook "
             "10.3 — a hard-coded lineage runs on the laptop and raises on ORC."),
            ("Cohorting — what may be compared", "bruisekit/allmodels.py",
             r"def _signature",
             "Two tables share a cohort iff they scored the same stems with the "
             "same mask areas. Anything else is quarantined, not silently mixed."),
        ],
    ),
    (
        "The pre-registered gates",
        "Each stage decides on VALIDATION whether it is worth training, and "
        "writes the verdict to disk before anything touches test.",
        [
            ("Stage M — multi-teacher oracle gate", "bruisekit/multiteacher.py",
             r"def oracle_gate",
             "Projects the oracle gain through Stage C's MEASURED transfer rate "
             "rather than through an assumption."),
            ("Stage N4 — mask vs caption pretraining",
             "bruisekit/samprobe.py", r"def mask_supervision_gate",
             "The medsam - sam contrast, with the reading fixed in code before "
             "any number existed."),
            ("Stage O — ITA-group gate", "bruisekit/itakd.py",
             r"def ita_group_gate",
             "Adds the identifiability clause the earlier gates lacked: refuses "
             "unless the per-group teacher ranking survives resampling patients."),
            ("Is the routing key even estimable?", "bruisekit/itakd.py",
             r"def identifiability",
             "Bootstraps the per-group argmax. 36-52 % stable across six "
             "candidate teachers — a coin flip, not a ranking."),
        ],
    ),
    (
        "The shims — how a stage changes behaviour without editing the pipeline",
        "Every experiment patches a name rather than editing engine.py. Each shim "
        "falls through to whatever was bound before it when its arm is not active.",
        [
            ("Build a model the pipeline has never heard of",
             "bruisekit/samprobe.py", r"def install_n4_shim",
             "Rebinds build_model in BOTH bruisekit.models and bruisekit.engine — "
             "engine bound it by value at import, so patching one is not enough."),
            ("Efficient architectures", "bruisekit/efficient_models.py",
             r"def install_efficient_shim",
             "Same pattern for the four mobile architectures."),
            ("Return a STACK of teachers", "bruisekit/itakd.py",
             r"def install_teacher_shim",
             "load_teacher returns [B, K, H, W] instead of [B, 1, H, W], so the "
             "whole pool reaches the loss with no edit to the training loop."),
            ("Get the ITA group to the loss", "bruisekit/itakd.py",
             r"def install_group_shim",
             "train_run iterates (x, y, _) and DISCARDS the stem, so the training "
             "loader is wrapped to record each batch's group indices."),
            ("Swap the loss", "bruisekit/itakd.py", r"def install_loss_shim",
             "Install this LAST — the dispatcher must sit on top of Stage H's and "
             "Stage M's."),
            ("Pin the batch to the control's", "bruisekit/multiteacher.py",
             r"def control_batch",
             "Accumulation restores the control's EFFECTIVE batch exactly, so the "
             "arm differs in the teacher signal alone and not also in step count "
             "and LR schedule."),
        ],
    ),
]


def main() -> int:
    out = [
        "# Code map — where each piece of functionality lives",
        "",
        "**Generated** by `scripts/88_generate_code_map.py`. Line numbers are read",
        "from the source at build time, so regenerate after any edit rather than",
        "patching this file. Every entry carries the search that finds it again if",
        "your copy has drifted.",
        "",
        "All paths are relative to `BRUISE_UNIFIED/`.",
        "",
    ]
    missing: list[str] = []
    for title, blurb, entries in MAP:
        out += [f"## {title}", ""]
        if blurb:
            out += [blurb, ""]
        out += ["| What | Where | Find it with |", "|---|---|---|"]
        for what, rel, rx, why in entries:
            p = BUNDLE / rel
            line = None
            if p.exists():
                pat = re.compile(rx)
                for i, txt in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
                    if pat.search(txt):
                        line = i
                        break
            if line is None:
                missing.append(f"{rel}  ::  {rx}")
                loc = "**NOT FOUND**"
            else:
                loc = f"`{rel}:{line}`"
            out.append(f"| **{what}**<br>{why} | {loc} | `grep -n \"{rx}\" "
                       f"{rel}` |")
        out.append("")

    out += [
        "---",
        "",
        "## Three rules that explain most of the code you will read",
        "",
        "1. **Nothing is fitted on test.** Thresholds, best seeds and every gate",
        "   are decided on validation, and the gate verdict is written to disk",
        "   before any test pass runs.",
        "2. **An experiment patches a name; it never edits the pipeline.** The",
        "   shims section above is the whole mechanism. Stage modules are",
        "   deliberately absent from `60_build_unified_bundle.py`'s copy list, so",
        "   an experiment that returns nothing cannot become a dependency of the",
        "   file that produces Stages A–Y.",
        "3. **A function that cannot do its job raises.** It does not warn, and it",
        "   does not return a plausible default. Most of the `raise` statements in",
        "   this codebase mark a place where a silent fallback once produced a",
        "   believable wrong number.",
        "",
        "## Regenerating",
        "",
        "```bash",
        "python scripts/88_generate_code_map.py",
        "```",
        "",
        "It **fails** if an anchor stops matching, which means the thing it",
        "described was renamed or removed. Fix the anchor rather than deleting the",
        "row.",
        "",
    ]

    if missing:
        print("FAIL: these anchors no longer match anything:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        print("\nThe code moved. Update the regex in MAP rather than regenerating "
              "with a gap.", file=sys.stderr)
        return 1

    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text("\n".join(out), encoding="utf-8")
    n = sum(len(e) for _, _, e in MAP)
    print(f"wrote {DST}  ({n} anchors across {len(MAP)} sections, all resolved)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
