# Bruise Segmentation — Project Handbook

Everything this project has done, why it was done that way, what the numbers are,
and how to extend it without breaking the comparisons.

`README.md` tells you how to *run* the bundle. This file tells you what is *in*
it and how to *change* it.

---

## Table of contents

1. [The problem and the headline finding](#1-the-problem-and-the-headline-finding)
2. [Dataset and split](#2-dataset-and-split)
3. [The shared protocol](#3-the-shared-protocol)
4. [Stage A — the five headline models](#4-stage-a--the-five-headline-models)
5. [Stage B — direct baselines](#5-stage-b--direct-baselines)
6. [Stage C — SegFormer-B5 distillation](#6-stage-c--segformer-b5-distillation)
7. [Stage E — mobile baselines](#7-stage-e--mobile-baselines)
7b. [Stage F — DeepLabV3+ → mobile-student distillation](#7b-stage-f--deeplabv3--mobile-student-distillation)
7c. [Stage H — reliability-gated distillation + the B2 teacher axis](#7c-stage-h--reliability-gated-distillation)
8. [Stage D — analysis methodology](#8-stage-d--analysis-methodology)
8b. [Stage G — final significance](#8b-stage-g--final-significance)
9. [Code architecture](#9-code-architecture)
10. [The registry and the three tiers](#10-the-registry-and-the-three-tiers)
11. [How to add a new model](#11-how-to-add-a-new-model)
12. [How to add a new distillation arm](#12-how-to-add-a-new-distillation-arm)
13. [How to add a new analysis](#13-how-to-add-a-new-analysis)
14. [Known gaps and caveats](#14-known-gaps-and-caveats)
15. [Traps we hit, and the guards against them](#15-traps-we-hit-and-the-guards-against-them)
16. [Build and regeneration](#16-build-and-regeneration)
17. [File map](#17-file-map)

---

## 1. The problem and the headline finding

Binary segmentation of bruises in white-light photographs, on a dataset of 1016
images from 143 subjects. The clinical question is not "how well outlined is the
bruise" but "was the bruise found at all", which is why **complete-miss rate**
(`Dice == 0`) carries more weight here than mean Dice.

### The finding that governs how everything else is read

**Every model is at the annotation ceiling.** Human annotators disagree with each
other by more than the models disagree with each other:

| Comparison | mean Dice | median Dice |
|---|---|---|
| human: gbarimah vs majority | 0.873 | 0.926 |
| human: erik vs majority | 0.866 | 0.892 |
| **model: segformer_b2_teacher** | **0.769** | **0.819** |
| **model: segformer_b0_distilled** | **0.768** | **0.817** |
| **model: segformer_b0_direct** | **0.766** | **0.813** |
| human: gbarimah vs erik | 0.755 | 0.809 |
| **model: yolo_sem_distilled** | **0.726** | **0.801** |
| **model: yolo_sem_direct** | **0.702** | **0.806** |
| human: paul vs majority | 0.700 | 0.750 |
| human: paul vs gbarimah | 0.581 | 0.632 |
| human: paul vs erik | 0.581 | 0.616 |

The entire model field sits between `paul_vs_majority` (0.700) and
`gbarimah_vs_erik` (0.755). **A 0.005 Dice gap between two models is not a
result** — it is inside the noise floor of the labels themselves.

> **Two aggregations, don't mix them.** The model numbers above are the
> **val-selected best seed** (what the reported figures are drawn from). §4 and §5
> quote **3-seed mean ± std** for the same models, which is why B2 appears as
> 0.769 here and 0.7651 ± 0.0037 there. Both are correct; state which you mean.
> The best seed is *not* constant across models — 0 for the three SegFormers and
> `yolo_sem_distilled`, **2** for `yolo_sem_direct` (see §15, trap 3).

**Consequences for anything you write:**

- Lead with complete-miss rate, not mean Dice.
- Never claim model X beats model Y on a Dice difference smaller than ~0.05
  without a paired subject-level bootstrap showing the interval excludes zero.
- Always show the annotation ceiling alongside a model ranking.

---

## 2. Dataset and split

| Split | Images | Subjects |
|---|---|---|
| train | 697 | 95 |
| val | 134 | 20 |
| test | 185 | 28 |

**Subject-grouped, never image-grouped.** No subject appears in two splits, and
no image appears twice. Both are re-asserted at build time
(`60_build_unified_bundle.py`) and again at notebook §2, because a manifest is a
text file and text files get edited. A subject leaking across splits would
inflate every number in the study invisibly.

**Physical layout** — `data/train/` holds all 831 train+val images together;
`split` in the manifest decides which is which. Val is a *subset* of the
training-side files, separated by a column, never by directory. Duplicating val
images into their own folder would create a second, silently divergent source of
truth for the 697/134 boundary.

**Metadata carried per image:** `stem`, `subject`, `ITA` (individual typology
angle), `ita_group_index_5`, `skin_tone_category`, `split`, `image_path`,
`mask_path`.

**ITA groups** (fixed order, light → dark, used in every fairness table):
`Light (II-III)`, `Intermediate (III-IV)`, `Tan (IV)`, `Brown (V)`, `Dark (VI)`.

Test-set group sizes: 39 / 38 / 24 / 29 / 55 images.

**The 640 cache.** Models train on 640×640 PNGs derived from the native-resolution
JPEGs — image bilinear, **mask nearest-neighbour**. That distinction is not
cosmetic: bilinear on a mask produces fractional boundary pixels, and
thresholding those back to binary erodes or dilates small bruises, which is
exactly the population the complete-miss metric is about. The cache is derived,
not shipped; it rebuilds deterministically in ~5 minutes.

---

## 3. The shared protocol

Every model in Stages A, B and E trains under the **same recipe**. That is the
entire basis for comparing them; if the recipe varied, differences would be
confounded with tuning effort.

```python
img_size        = 640
epochs          = 100          # with early stopping
patience        = 15
backbone_lr     = 6e-5         # pretrained encoder: conservative
head_lr         = 6e-4         # random head: 10x, catches up
betas           = (0.9, 0.999)
weight_decay    = 0.01
warmup_fraction = 0.01
poly_power      = 1.0          # poly LR decay
gradient_clip   = 1.0
amp             = True
seeds           = (0, 1, 2)
```

**Why the 10× head LR.** The backbone is pretrained and already has good
features; a conservative LR preserves them. The head is randomly initialised and
must catch up. This is the SegFormer paper's recipe (Xie et al. 2021), applied to
*every* architecture here — including YOLO, not because YOLO's paper says so, but
because holding the recipe fixed across architectures is the whole point.

**What the LR split actually splits.** `build_param_groups` assigns groups by
`id(p)` against `model.backbone`, not by name — the architectures name their
parameters completely differently and a name-prefix rule would put every YOLO
parameter in the wrong group. One consequence worth knowing: **Fast-SCNN's
`.backbone` is only `learning_to_downsample`**, three conv blocks. Its global
feature extractor, feature fusion and classifier all land in the head group at
6e-4, and the 6e-5 rate applies to three randomly-initialised convs at the input
stage. The property exists so `build_param_groups` needs no special case, and the
code calls the split "cosmetic" because nothing in Fast-SCNN is pretrained — true
for provenance, but those convs really do learn 10× slower than everything around
them. It applies identically to `fastscnn` and `fastscnn_distilled`, so the Stage
F contrast is unaffected; only Fast-SCNN's absolute number is.

More generally, 6e-5/6e-4 was tuned for AdamW on a pretrained transformer.
Fast-SCNN's paper uses SGD at 0.045 with poly decay, ~100× larger. Holding the
recipe fixed is the point (§11, "do not change the recipe"), but if a Fast-SCNN
run hits the 100-epoch cap with val AP still climbing rather than early-stopping
on patience, that is evidence it was LR-limited and belongs in the limitations,
not a licence to tune one arm.

**Loss:** Dice + BCE (`SupervisedLoss`), or `DistillLoss` when a teacher is
present. `aux_weight = 0.4` for SegFormer; `0.0` for the baselines and mobile
models, which have no auxiliary head.

**Model selection:** best validation **AP** (threshold-free), not best Dice at
some threshold. Selecting on a thresholded metric would entangle model choice
with threshold choice.

**Threshold selection — the operating point.** After training, the logit cut is
swept over 481 values on the **134 validation images**, then applied once to test.

The cut is *not* the argmax. These sweeps are extraordinarily flat — B2's val
Dice moved by 0.009 across thresholds from 0.154 to 0.959. That is not a peak, it
is noise on a plateau, and taking the argmax fits the val set's sampling error.
`select_cut` therefore takes every cut within **one standard error** of the peak
as statistically tied, and breaks the tie by **lowest complete-miss rate**. Cuts
in the band are Dice-equivalent but they are *not* miss-equivalent.

The chosen cut and its diagnostics are written to `operating_point.json`:

```json
{"cut": -0.725, "threshold": 0.3263, "val_dice_at_cut": 0.7717,
 "val_miss_at_cut": 0.0, "val_peak_dice": 0.7717, "peak_cut": -0.725,
 "band_lo_threshold": 0.0247, "band_hi_threshold": 0.8980,
 "band_width_cuts": 235, "n_cuts": 481}
```

**A checkpoint without its `operating_point.json` is unusable** — its test score
is undefined. The registry enforces this.

**Batch size.** SegFormer probes for the largest batch that fits (`per_model`,
landing on 32 for B2, 64 for B0). SMP and mobile models use a **fixed** batch of
16 and skip the probe: the probe escalates from batch 1 in train mode, and
DeepLabV3+'s ASPP image-pool BatchNorm, LR-ASPP's image pool, and StrideFormer's
SE blocks all raise "Expected more than 1 value per channel" at batch 1. A fixed
batch is also cleaner for a baseline comparison.

**The resume contract** (`engine.train_run`):

- `DONE.json` exists → return immediately, touch nothing.
- `resume.pt` exists → restore model + optimizer + scaler + epoch + best AP +
  patience + global step + history, continue.
- Neither → fresh start.

`resume.pt` is written every `drive_sync_every` epochs **even on epochs where val
AP did not improve**: best weights live in `best.pt`, but resuming must continue
from where training actually *was*, or the LR schedule and optimizer moments
silently rewind. It is deleted on completion. Saves are atomic (temp file +
rename) because a truncated checkpoint turns one lost session into a lost run.

**The normalisation contract.** The dataloader emits **raw [0,1] pixels** and each
model applies its own scale internally. This is not a style preference —
SegFormer wants ImageNet normalisation, YOLO wants plain `/255`, and Ultralytics'
BatchNorms carry frozen running statistics for the `/255` distribution. Feeding
YOLO ImageNet-normalised pixels makes it under-fire by 4× and **caps it at Dice
0.479 with no threshold able to recover it**. Pixel scale belongs to the model,
not the loader.

---

## 4. Stage A — the five headline models

| Model | Params | Description |
|---|---|---|
| `segformer_b2_teacher` | 27.35 M | SegFormer MiT-B2, the teacher |
| `segformer_b0_direct` | 3.71 M | SegFormer MiT-B0, supervised only |
| `segformer_b0_distilled` | 3.71 M | B0 distilled from the same-seed B2 |
| `yolo_sem_direct` | 1.63 M | YOLO26n-seg, native Ultralytics |
| `yolo_sem_distilled` | 1.63 M | YOLO26n-seg with offline pseudo-mask KD |

3 seeds each = 15 runs. All 15 checkpoints ship in `checkpoints/final/`.

### Results (3-seed mean ± std, 185 test images)

| Variant | mean Dice | median | miss % |
|---|---|---|---|
| segformer_b2_teacher | 0.7651 ± 0.0037 | 0.8113 | 0.00 ± 0.00 |
| segformer_b0_distilled | 0.7639 ± 0.0047 | 0.8114 | 0.18 ± 0.31 |
| segformer_b0_direct | 0.7600 ± 0.0055 | 0.8113 | 0.54 ± 0.00 |
| yolo_sem_direct (native argmax) | 0.6911 ± 0.0197 | 0.8021 | 7.57 ± 1.08 |
| yolo_sem_distilled (native argmax) | 0.6676 ± 0.0648 | 0.7658 | 7.57 ± 4.80 |

**Distillation works, barely.** B0-distilled gains +0.004 Dice over B0-direct —
statistically indistinguishable — but drops complete misses from 0.54% to 0.18%.
The miss-rate improvement is the real effect; the Dice difference is not.

### SegFormer distillation details

- Teacher: the **same-seed** B2, not the best B2. Pairing seed *k*'s student with
  seed *k*'s teacher means the reported spread includes the teacher's own
  variance, which is part of the pipeline being measured. Using one strong
  teacher for all three students would make the spread artificially narrow.
- Teacher temperature is **calibrated** on validation after training
  (`calibrate_temperature`), written to `calibration.json`. Measured
  T ≈ 1.69–1.76.
- `segformer_alpha = 0.6` — the KD mix.

### YOLO: the two-path story

YOLO is evaluated two ways, and only one is a reporting path:

- **`native_argmax`** — Ultralytics' own semantic head, argmax over classes.
  Parameter-free: there is no threshold to fit, so nothing to overfit. **This is
  the sole reporting path.**
- **`custom255`** — a custom probability path with a swept threshold. Retained in
  the bundle for provenance only. Do not put it in charts.

YOLO trains **natively** (mosaic, EMA, letterbox, its own LR schedule) rather
than through the shared custom loop. An earlier lineage trained YOLO in the
custom loop and is superseded; see §15.

Distilled YOLO uses **offline pseudo-masks**: the same-seed B2 teacher's
native-resolution probability is fused with ground truth as
`class = (alpha*GT + (1-alpha)*teacher_prob >= 0.5)` with `yolo_alpha = 0.4`.
Alpha below 0.5 keeps the fusion non-degenerate. Pseudo-masks are built at native
resolution because Ultralytics letterboxes internally — pre-resizing would apply
the resize twice.

### Speed (full A100, 640 tensor → mask on GPU)

| Model | median ms | FPS | params M | peak activation MB |
|---|---|---|---|---|
| segformer_b2_teacher | 33.67 | 29.7 | 27.35 | 1778.7 |
| segformer_b0_direct | 16.68 | 59.9 | 3.71 | 1204.4 |
| segformer_b0_distilled | 16.64 | 60.1 | 3.71 | 1204.4 |
| yolo_sem_direct | 8.18 | 122.2 | 1.63 | 967.5 |
| yolo_sem_distilled | 8.22 | 121.7 | 1.63 | 967.5 |

Disk read, JPEG decode, resize and host↔GPU copies are deliberately **not** timed
— they are identical for every model and dominated by I/O, so including them
would compress real architectural differences into measurement noise.
`cuda.synchronize()` brackets every call; without it this measures how long it
takes to *queue* the work, reporting every model as equally fast.

### Fairness (Stage A, ITA groups)

| Model | Kruskal p | significant | gap | best group | worst group | max miss gap |
|---|---|---|---|---|---|---|
| segformer_b2_teacher | 0.0105 | **yes** | 0.112 | Intermediate | Tan | 0.000 |
| segformer_b0_direct | 0.639 | no | 0.052 | Light | Dark | 0.000 |
| segformer_b0_distilled | 0.470 | no | 0.050 | Intermediate | Dark | 0.000 |
| yolo_sem_direct (native) | 0.230 | no | 0.104 | Intermediate | Light | 0.154 |
| yolo_sem_distilled (native) | 0.151 | no | 0.090 | Intermediate | Light | 0.051 |

**The counter-intuitive result:** where YOLO shows a fairness gap, the *worst*
group is **Light (II-III)**, not Dark. Light-skin under-detection, not the
expected direction. See §8 for why this is probably a size confound.

---

## 5. Stage B — direct baselines

Same recipe, ImageNet ResNet-50 encoders, fresh 1-class heads.

| Variant | mean Dice | median | miss % |
|---|---|---|---|
| unet_r50 | 0.7526 ± 0.0154 | 0.826 | 3.24 ± 0.54 |
| deeplabv3plus_r50 | 0.7453 ± 0.0182 | 0.814 | 1.80 ± 0.62 |

3 seeds each = 6 runs, all shipped.

**nnU-Net was never run on the canonical split.** No weights, no results. It is
registered as an explicit gap rather than omitted, because "the baseline we did
not run" is exactly the fact a reader needs. Enabling it needs `nnunetv2` and
~8 GPU-hours.

**Fairness:** neither baseline shows a significant ITA effect (unet p=0.222,
deeplab p=0.108).

---

## 6. Stage C — SegFormer-B5 distillation

The most elaborate stage. A B5 teacher was trained, then a grid of KD strategies
distilled B5 (and B2+B5 ensembles) into B0 students, each scored against the
**B2→B0 reference** from Stage A.

### 6.1 The B5 teacher

Trained at 3 seeds; the val-winning seed was promoted to `teachers/`.

| seed | test mean Dice | median | threshold | misses |
|---|---|---|---|---|
| 42 | 0.7527 | 0.8124 | 0.35 | 0 |
| **123 (selected)** | **0.7727** | **0.8273** | **0.50** | **0** |
| 2026 | 0.7676 | 0.8298 | 0.40 | 0 |

Selected on **validation** (`val_mean_dice`: 0.7787 / 0.7881 / 0.7861), never on
test. `segformer_b5_teacher` **is** `segformer_b5__seed123` — the same weights
under two names. `report.load_stage_c` detects the identical per-image vectors
and collapses them, because listing both would show one result twice and invite
reading it as two agreeing runs.

Reference teachers, all with zero complete misses:

| Model | mean Dice | median |
|---|---|---|
| segformer_b5_teacher | 0.7727 | 0.8273 |
| segformer_b2_teacher | 0.7692 | 0.8192 |
| segformer_b0_distilled (B2→B0 reference) | 0.7680 | 0.8167 |

### 6.2 The multi-teacher gate — `val_oracle`

**Before** running any multi-teacher arm, an oracle test asked whether B2 and B5
are complementary enough to be worth combining. Run on the 134 val images:

```json
{"n_val_images": 134, "n_subjects": 20,
 "spearman_dice_b2_b5": 0.819,
 "b5_beats_b2_by_gt_0.10": 18, "b2_beats_b5_by_gt_0.10": 12,
 "b2_mean_dice": 0.7717, "b5_mean_dice": 0.7881,
 "oracle_mean_dice": 0.8140, "oracle_median_dice": 0.8649,
 "oracle_gain_over_best_single": 0.0258,
 "oracle_gain_ci95": [0.0165, 0.0362], "p_gain_positive": 1.0,
 "rel_b_for_adaptive_gate": 0.6343,
 "GATE_run_adaptive": true}
```

Reading: the two teachers correlate at ρ=0.82 but disagree substantially on 30 of
134 images. An oracle that picked the better teacher per image would gain +0.026
Dice over the best single teacher, CI [0.017, 0.036], P(gain>0) = 1.0. **The gate
opened**, and `rel_b = 0.634` became the adaptive arms' mixing weight.

This is the right shape for an experiment: a cheap, decisive pre-test that
determines whether an expensive grid is worth running at all.

### 6.3 Alpha search

KD mix searched on **validation** with 15-epoch short runs, 5 grid points
(Optuna used automatically if installed, grid otherwise):

| Tag | Best alpha | Best val Dice | Trials (alpha → val Dice) |
|---|---|---|---|
| `single_b5_response` | **0.625** | 0.7019 | 0.300→0.7017, 0.462→0.6951, 0.625→0.7019, 0.787→0.6975, 0.950→0.7010 |
| `ensemble_uniform` | **0.950** | 0.7089 | 0.300→0.7016, 0.462→0.7067, 0.625→0.7004, 0.787→0.6952, 0.950→0.7089 |

Note how flat both are — 0.007 across the whole range for single-teacher. This is
the same plateau phenomenon as the threshold sweep, and it means alpha is not a
sensitive knob here.

### 6.4 The arms

All students are SegFormer-B0, seed 42, 697/134 split.

| Arm | mean Dice | alpha | What it does |
|---|---|---|---|
| `p3_adaptive` | **0.7748** | 0.95 | B2+B5 adaptive ensemble, per-image gating at rel_b=0.634 |
| `p2_cwd_b5_to_b0` | 0.7736 | 0.625 | Channel-wise distillation from B5 |
| `expA_b5_to_b0_response` | 0.7683 | 0.625 | Plain response KD from B5 |
| `p3_adaptive_boundary` | 0.7668 | 0.95 | Adaptive + boundary loss (λ=4.0) |
| `p2_bpkd_b5_to_b0` | 0.7651 | 0.625 | Boundary-privileged KD |
| `p2_ensemble_uniform` | 0.7615 | 0.95 | B2+B5 uniform ensemble |
| `p3_adaptive_group` | 0.7586 | 0.95 | Adaptive + ITA-group weighting (λ=0.5) |
| `p3_adaptive_full` | 0.7515 | 0.95 | Adaptive + all auxiliary terms |
| `x_angular_b5_to_b0` | 0.7418 | 0.625 | Angular/relational KD (λ=1.0) |
| `p3_adaptive_hard` | 0.7414 | 0.95 | Adaptive + hard-example focus (γ=2.0) |
| `x_dkd_b5_to_b0` | — | 0.625 | **Never executed** (config only) |

A representative `run_config.json`:

```json
{"run_id": "p3_adaptive", "student": "segformer_b0", "seed": 42,
 "teacher_a": "teachers/segformer_b2_teacher",
 "teacher_b": "teachers/segformer_b5_teacher",
 "ensemble": "adaptive", "kd": "response",
 "group": false, "hard": false, "boundary": false,
 "rel_b": 0.6343, "alpha": 0.95,
 "lambda_cwd": 1.0, "lambda_angular": 1.0, "lambda_boundary": 4.0,
 "lambda_group": 0.5, "gamma_hard": 2.0,
 "micro_batch": 32, "accum": 1, "n_train": 697, "n_val": 134}
```

### 6.5 How arms are scored

Each arm is compared to the **B2→B0 reference** (0.7680) with a **paired
subject-level bootstrap**, and given one of three verdicts against a
non-inferiority margin of one Dice point:

- **WIN** — the 95% interval is entirely above zero.
- **INFERIOR** — entirely below −0.01.
- **NON-INFERIOR** — anything else.

At 28 subjects, **every arm came back NON-INFERIOR**. `p3_adaptive` leads with
Δ=+0.0068, CI [−0.0025, +0.0152], P(better) = 0.93 — suggestive, not significant.

**The honest summary of Stage C: none of the ten KD strategies produced a
statistically distinguishable improvement over plain B2→B0 distillation.** That
is a real result and should be reported as one, not buried.

---

## 7. Stage E — mobile baselines

Four deployment-scale architectures, trained directly. Two further families,
`fastscnn_distilled` and `lraspp_mobilenetv3_distilled`, share those architectures
but take a teacher — see
[§7b](#7b-stage-f--deeplabv3--mobile-student-distillation). Earlier revisions of
this section said "no distillation variants exist for these"; that stopped being
true when Stage F was added, and `EFFICIENT_FAMILIES` now carries six entries in
both `loaders.py` and `registry.py`. (The `registry._scan_stage_e` docstring that
used to say "DIRECT ONLY" was updated when the second arm landed.)

| Model | Params | Init | Source |
|---|---|---|---|
| PP-MobileSeg-Tiny | 1.45 M | StrideFormer-Tiny backbone, ImageNet | OpenMMLab, auto-download |
| TopFormer-Tiny | 1.37 M | ImageNet backbone (66.2% top-1) | Google Drive, manual |
| LR-ASPP MobileNetV3 | 3.22 M | MobileNetV3-Large, ImageNet | torchvision, auto |
| Fast-SCNN | 1.14 M | **none — scratch by design** | n/a |

**Fast-SCNN's lack of weights is not a gap.** At 1.1 M parameters the paper
trains from scratch and no official checkpoint was ever released; random init is
the faithful reproduction. But it *does* mean Fast-SCNN is not initialised like
the others, and that must be stated wherever it is compared.

### 7.1 Vendoring, and why

PP-MobileSeg and TopFormer only have reference implementations inside the
OpenMMLab stack. `mmcv` ships compiled CUDA extensions pinned to narrow torch
versions and routinely fails to build; dragging it in for two 1.4 M-parameter
models would be the heaviest dependency in the project.

But **hand-rewriting them is worse**. The published checkpoints are state dicts
keyed by module path — any deviation in naming makes `load_state_dict` fail or,
worse, half-succeed and leave most weights random while reporting success.

So the reference files are **vendored verbatim** into `bruisekit/vendor/`, with
only their import block rewritten to `bruisekit/mmcv_shim.py`. The shim
reproduces `ConvModule` (including the attribute names `conv`, `bn`, `activate`,
which *are* the state-dict keys), `build_norm_layer`, `build_activation_layer`,
`DropPath`, and inert stubs for the registry plumbing.

**One subtlety that would have been silent:** StrideFormer's SE blocks ask for
`dict(type='Hardsigmoid', slope=0.2, offset=0.5)`. `nn.Hardsigmoid` fixes the
slope at 1/6. Substituting the builtin leaves every state-dict key matching —
this layer has no parameters — while quietly changing the gate on every SE block
in the backbone. The shim implements `HSigmoid` with honoured slope/offset.

Verification that the vendoring worked:

```
arch                 loaded  in_checkpoint  unexpected  verdict
ppmobileseg_tiny        488            488           0  EXACT MATCH
lraspp_mobilenetv3      308            308           0  EXACT MATCH
```

`unexpected = 0` with only the decode head missing is proof the architecture is
the published one. This matters more than parameter count: PP-MobileSeg measures
1.454 M against a published 1.61 M, but that figure is for a 150-class ADE20K
head, and the checkpoint match settles it.

**Backbone-only, always.** Segmentation-pretrained checkpoints exist for both
PP-MobileSeg (ADE20K) and LR-ASPP (COCO-with-VOC-labels) and are deliberately
**not** used — they would give those models segmentation pretraining no other
baseline in the study has.

### 7.2 Results (3 seeds, 185 test images)

> **⚠️ These numbers came from runs that are not in this bundle, and are being
> regenerated (started 2026-07-29).** The bundle ships no `checkpoints/efficient/`
> and no `results/efficient/efficient_test_per_seed.csv` — the file map has always
> said that directory is "populated when you train". So the registry correctly
> reports all Stage E runs as MISSING even though this section quotes results for
> them, and a session with `ALLOW_TRAINING=True` trains them from scratch rather
> than skipping. That is not a bug and not a duplicate of work already in the
> bundle; see §15 trap 13. Treat everything in §7.2/§7.3 as the previous lineage's
> numbers until the current sweep lands, then replace them wholesale.

Seeds 0, 1 and 2 have now all been run, so Stage E is on the same 3-seed footing as
Stages A and B. Each seed is scored at **its own** val-selected cut.

| Model | mean Dice | median | miss % | misses (s0/s1/s2) |
|---|---|---|---|---|
| lraspp_mobilenetv3 | **0.7093 ± 0.0127** | 0.7723 | **0.54 ± 0.54** | 0 / 1 / 2 |
| topformer_tiny | 0.6895 ± 0.0093 | 0.7325 | 0.90 ± 0.82 | 2 / 0 / 3 |
| ppmobileseg_tiny | 0.6617 ± 0.0226 | 0.7161 | 2.52 ± 1.12 | 4 / 3 / 7 |
| fastscnn | 0.6183 ± 0.0306 | 0.6983 | 3.42 ± 1.12 | 7 / 8 / 4 |

Per seed, at the cut each run selected on validation:

| Model | seed | cut | mean Dice | median | IoU | precision | recall | miss % |
|---|---|---|---|---|---|---|---|---|
| lraspp_mobilenetv3 | 0 | — | 0.7231 | 0.7864 | — | — | — | 0.00 |
| lraspp_mobilenetv3 | 1 | −0.625 | 0.7066 | 0.7699 | 0.5769 | 0.8494 | 0.6637 | 0.54 |
| lraspp_mobilenetv3 | 2 | −0.825 | 0.6982 | 0.7607 | 0.5677 | 0.8322 | 0.6548 | 1.08 |
| topformer_tiny | 0 | — | 0.6974 | 0.7364 | — | — | — | 1.08 |
| topformer_tiny | 1 | −0.925 | 0.6918 | 0.7418 | 0.5541 | 0.7421 | 0.7064 | 0.00 |
| topformer_tiny | 2 | −0.250 | 0.6793 | 0.7193 | 0.5399 | 0.7542 | 0.6757 | 1.62 |
| ppmobileseg_tiny | 0 | — | 0.6863 | 0.7304 | — | — | — | 2.16 |
| ppmobileseg_tiny | 1 | −1.025 | 0.6568 | 0.7158 | 0.5251 | 0.7605 | 0.6583 | 1.62 |
| ppmobileseg_tiny | 2 | −1.650 | 0.6420 | 0.7022 | 0.5107 | 0.7980 | 0.6023 | 3.78 |
| fastscnn | 0 | — | 0.6533 | 0.7404 | — | — | — | 3.78 |
| fastscnn | 1 | −2.325 | 0.6053 | 0.6866 | 0.4819 | 0.7601 | 0.5754 | 4.32 |
| fastscnn | 2 | −2.025 | 0.5963 | 0.6679 | 0.4695 | 0.7424 | 0.5650 | 2.16 |

`—` = not recorded in the seed-0 summary that was carried into this handbook; the
values are in that run's `operating_point.json` and per-image CSV. Seed 0's
subject-bootstrap 95% CIs were [0.679, 0.764], [0.668, 0.730], [0.654, 0.724] and
[0.604, 0.701] in the table order above.

**fastscnn's cuts sit far off everyone else's** (−2.0 to −2.3, against ≈ −0.7 for
SegFormer and LR-ASPP). It has to push the operating point a long way negative to
keep the miss rate down, which is what a scratch-initialised model with weak
features looks like at the threshold stage.

Reported validation Dice for the seed-1/2 sweep, with the initialisation each
model used:

| Model | seed | val Dice | init |
|---|---|---|---|
| lraspp_mobilenetv3 | 2 | 0.7277 | MobileNetV3-Large backbone, ImageNet-1k (torchvision) |
| ppmobileseg_tiny | 1 | 0.6985 | StrideFormer-Tiny backbone, ImageNet |
| topformer_tiny | 1 | 0.6929 | TopFormer-Tiny backbone, ImageNet-1k (66.2% top-1) |
| fastscnn | 1 | 0.6859 | random (He) init — no pretrained weights exist |

### 7.2a Two things to check before these numbers are published

**⚠️ Seed 0 is the best seed for all four models.** Every seed-1 and seed-2 test
Dice is below its seed-0 counterpart — eight out of eight. If the seeds were
exchangeable, seed 0 winning all four is a ~1 % event. That is the same shape as
§15 trap 1: it suggests the seed-0 runs may not have been produced under exactly
the same config as seeds 1–2 (recipe, cut-selection code, or scoring path), rather
than that seed 0 is lucky. **Reconcile the seed-0 `run_config.json` and
`operating_point.json` against seeds 1–2 before quoting the 3-seed means.** Until
that is done, treat the 3-seed column as provisional.

**⚠️ A cross-seed aggregate table that does not reconcile.** A model-level summary
produced alongside the seed sweep carries val-best-seed Dice columns but
complete-miss columns that match no single seed at its own cut — topformer 1 miss
(seeds have 2 / 0 / 3), ppmobileseg 4 (4 / 3 / 7), fastscnn 13 (7 / 8 / 4); only
LR-ASPP's row is internally consistent. Dice and misses in that table are coming
from different seeds or different thresholds. Do not use it; regenerate per-seed
and aggregate through `report.normalize()`, which recomputes `complete_miss` from
`dice` for exactly this reason (§8.1).

**SOLVED, 2026-07-30 — and the conclusion above is wrong.** Neither table is
corrupt and no seeds are mixed. The two artefacts count **two different things**,
confirmed on all four models against the per-image CSVs:

| model | `efficient_test_per_seed.csv` | `pred_positive_pixels == 0` | `dice == 0` |
|---|---|---|---|
| topformer_tiny | 0 | **0** | 1 |
| ppmobileseg_tiny | 3 | **3** | 4 |
| fastscnn | 8 | **8** | 13 |
| lraspp_mobilenetv3 | 2 | **2** | 2 |

The per-seed sweep table counts images where the model **predicted nothing at
all**. `report.normalize` counts images where the prediction has **no overlap with
the ground truth** (§8.1). The first is a strict subset of the second: it misses
the case where a model fires confidently on the wrong region, which is still a
complete miss to a clinician and is exactly the failure the metric exists to
catch. LR-ASPP's row looked consistent only because it happens to have no
fire-in-the-wrong-place cases.

**Publish the `dice == 0` column.** Fast-SCNN's complete-miss rate is 13/185 =
7.0 %, not the 4.3 % the per-seed table implies, and every §7.2 miss figure
sourced from that table understates by the same mechanism. The model-level
headline table was right all along; the earlier text in this section, which told
readers to prefer the per-seed counts, had it backwards.

### 7.3 Speed (A100 **MIG 3g.40gb** slice — see caveat)

| Model | median ms | FPS | params M | peak activation MB |
|---|---|---|---|---|
| fastscnn | 3.56 | 281.3 | 1.135 | 965.1 |
| lraspp_mobilenetv3 | 5.01 | 199.8 | 3.218 | 1009.3 |
| topformer_tiny | 6.19 | 161.6 | 1.370 | 1000.9 |
| ppmobileseg_tiny | 10.67 | 93.8 | 1.454 | 999.2 |

**⚠️ These are NOT comparable to the Stage A speed table**, which was measured on
a full A100. A MIG 3g.40gb slice is roughly 3/7 of a card. To compare, re-time
both on the same device.

### 7.4 Reading

**LR-ASPP MobileNetV3 leads the mobile field**, at 0.709 ± 0.013 Dice and 0.54 %
complete misses on 3.22 M params. Its mean Dice sits inside the annotation-ceiling
band (between `paul_vs_majority` 0.700 and `gbarimah_vs_erik` 0.755).

**The zero-miss result did not survive the extra seeds.** On seed 0 LR-ASPP missed
nothing, which earlier drafts of this file reported as matching the B2 teacher and
B0-distilled. Seeds 1 and 2 miss 1 and 2 images, putting the 3-seed rate at 0.54 %
— the same as `segformer_b0_direct`, not the same as the zero-miss models. This is
the single-seed caveat firing exactly as §1 says it should, and it is worth keeping
in view as an example: **a 0-of-185 result is not evidence of a zero rate.** Its
one-sided 95 % bound is ≈ 1.6 %, which covers every number in the Stage E table.

**The ordering is stable, the gaps mostly are not.** LR-ASPP > TopFormer >
PP-MobileSeg > Fast-SCNN holds on all three seeds. But LR-ASPP beats TopFormer by
0.020 Dice, which is under the ~0.05 threshold §1 sets for a claimable difference;
that pair needs a paired subject-level bootstrap before it is stated as a win. The
LR-ASPP → Fast-SCNN gap (0.091) and the miss-rate gap (0.54 % vs 3.42 %) are large
enough to report.

**Precision and recall separate the field more cleanly than Dice.** LR-ASPP runs
at precision ≈ 0.84 with recall ≈ 0.66; TopFormer at precision ≈ 0.75 with recall
≈ 0.69. They land at similar Dice by different routes — LR-ASPP is conservative and
accurate where it fires, TopFormer fires more widely. For a clinical
find-the-bruise task that difference matters more than the 0.02 Dice gap does.

PP-MobileSeg is the *slowest* of the four despite being nearly the smallest — its
strided SEA attention is memory-bound rather than FLOP-bound.

Fast-SCNN is fastest and least accurate, consistent with being the only model
without pretrained initialisation. It also has the widest seed spread (0.031) and
the most extreme thresholds.

**Still open:** see §7.2a — the seed-0 dominance and the unreconciled aggregate
table both need resolving before these numbers go into a paper. The 2026-07-29
sweep resolves both by construction: all fifteen runs come from one config, one
cut-selection path and one scoring path, so seed 0 no longer has a different
provenance from seeds 1–2.

---

## 7b. Stage F — DeepLabV3+ → mobile-student distillation

Two further Stage E families trained with a frozen **DeepLabV3+/ResNet-50**
teacher: `fastscnn_distilled` (added first) and `lraspp_mobilenetv3_distilled`
(added 2026-07-29 — see §7b.6). This is the study's second distillation axis
(Stage A distils B2→B0 within SegFormer; Stage C distils B5→B0), and the first
one that crosses architecture families.

Each arm's contrast is distilled-vs-direct on the same architecture, both at
seeds 0/1/2 under the identical recipe. Nothing else changes between the two
variants of an architecture, so any difference is attributable to the teacher
signal. Across arms, teacher and recipe are also identical, so the
fastscnn-vs-lraspp pair isolates one more variable: what the student's
initialisation (scratch vs ImageNet) does to KD.

### 7b.1 Why DeepLabV3+ is the teacher

Three reasons, in the order they matter:

**Licence.** The SegFormer MiT weights in `pretrained_weights/segformer_mit_b*`
are NVIDIA's, `license: other`, non-commercial. DeepLabV3+/R50 reaches the study
through `segmentation_models_pytorch` (MIT) on a torchvision ResNet-50 (BSD-3), so
a student distilled from it carries no non-commercial link. This does **not** by
itself make the result commercialisable — the ImageNet provenance of the encoder
and, far more importantly, the consent scope of the photographs are separate
questions — but it removes the one unambiguously non-commercial licence.

**Complete misses, not Dice.** DeepLabV3+ and U-Net are indistinguishable on Dice
(0.7584 vs 0.7570 on each one's val-selected seed, with U-Net's median actually
higher) — inside the noise floor §1 warns about. They are *not* indistinguishable
on complete misses, and for a teacher that is the metric that decides. A teacher
that misses a bruise does not hand the student a weak soft target; it hands it a
confidently-empty one on exactly the image the clinical metric cares about. Paired
over all 185 test images:

```
images DeepLab misses that U-Net finds:  0
images U-Net misses that DeepLab finds:  2
both miss:                               5
```

Strict containment, and the ordering holds at every seed (DeepLab 5/5/2 misses,
U-Net 9/7/7).

**Structure.** DeepLabV3+'s ASPP-then-fuse-one-low-level-skip decoder is the same
shape as Fast-SCNN's PPM-then-feature-fusion, so a later feature-level arm (CWD)
has an obvious place to attach. U-Net's four symmetric skips have no counterpart
in the student.

### 7b.2 Why the teacher must be calibrated first

`load_teacher` divides the teacher's logits by a fitted temperature. The Stage B
baselines were trained as endpoints, not as teachers, so they ship no
`calibration.json` and the shim would raise. `ensure_calibration` fits it with the
same `engine.calibrate_temperature` (Guo et al. 2017, L-BFGS on log T over the 134
val images) that the B2 teacher used, so both teachers are prepared identically.

**Distilling from an uncalibrated teacher is the failure mode this guards
against**: a near-binary soft label is the hard label with extra steps, and the
arm would silently measure nothing. The shim raises rather than falling back.

Calibration is written to `WORK_DIR/teacher_calibration/<family>__seed<k>.json`,
never next to the checkpoint: `checkpoints/baselines/` is the verified artefact
that produced the published Stage B table, and adding a file to it would make the
bundle differ from the one those numbers came from.

### 7b.3 How it is wired

`bruisekit/distill_efficient.py`, imported as `DE` in the notebook. It is a
**runtime shim**, not an edit to the training loop:

| Function | What it does |
|---|---|
| `register_student_aliases()` | Registers every arm in `STUDENT_ARCH`: makes each distilled family resolve to its architecture at the call sites that key off a family name (`EFFICIENT_ARCHS`, `PUBLISHED_PARAMS_M`, `_HEAD_IN_CHANNELS`, `weights.SOURCES`). Needed whenever Stage E is *read*, training or not. (`register_student_alias(family, arch)` remains for a single arm.) |
| `ensure_calibration(env, family, seed, CFG, MAN640)` | Fits or reuses the teacher's temperature. Idempotent. |
| `install_teacher_shim(env, CFG)` | Rebinds `engine.load_teacher` so it reads the teacher's architecture from the family spec instead of the hardcoded SegFormer-B2 at `engine.py:147`. Only meaningful when training. |
| `train_arm(env, family, seeds, CFG, MAN640)` | Calibrate-then-train driver, for running the arm on its own without the four direct baselines. |

The shim lives in its own module for the reason §12 gives: `engine.py` is
extracted verbatim from `bruise_colab_baselines.ipynb` at build time (§16), so an
edit there is reverted by the next `60_build_unified_bundle.py`. This is the same
pattern `efficient_models.install_efficient_shim` and
`smp_models.install_build_model_shim` already use. Installing twice does not wrap
the wrapper — `load_teacher._original` is preserved so the fall-through path does
not depend on call order.

**`train_run` itself is untouched.** It already implements distillation when the
spec says `distill=True`; Stage F changed only *which* teacher it loads. The arm
inherits the shared recipe, the LR split, early stopping and the whole resume
contract.

### 7b.4 The settings that matter

```python
efficient_alpha = 0.6        # the KD mix — NOT segformer_alpha
teacher_seed_mode = "same"   # student seed k distils from teacher seed k
```

**`efficient_alpha`, not `segformer_alpha`.** Both are currently 0.6, so reading
the wrong one would look right and silently ignore the other the moment someone
tuned one of them. `train_missing` picks the key off `r.kind`.

**Seed mode `"same"`** is Stage A's convention (§4): the student's reported spread
then includes the teacher's own variance, which is part of the pipeline being
measured. The alternative, `"fixed"`, promotes one val-selected teacher for all
students — a narrower spread that no longer measures teacher variance. Use it only
if you are reporting a single promoted teacher and say so. The teacher-choice
argument survives `"same"` because DeepLab beats U-Net on complete misses at every
seed.

**Fast-SCNN has no auxiliary head**, deliberately. The paper uses one, but
`aux_weight = 0` for the baselines so the loss is identical across architectures;
returning `None` for aux keeps that explicit rather than adding a head whose
gradient is multiplied by zero.

### 7b.5 Status

**The LR-ASPP arm has landed at all three seeds (2026-07-30); results and reading
are in §7b.7.** The Fast-SCNN arm has not been re-run under the current lineage,
so the paragraphs below still describe its previous-lineage numbers.

**Stage E/F is being regenerated as of 2026-07-29** (§7.2's caveat). With the
LR-ASPP arm added the full sweep is `RUN_STAGES="E"`, `ALLOW_TRAINING=True`,
`EFFICIENT_SEEDS=(0,1,2)` — eighteen runs, the six families in
`EFFICIENT_FAMILIES` order, so `fastscnn_distilled` is runs 13–15 and
`lraspp_mobilenetv3_distilled` runs 16–18. The distilled arms are slower per
epoch than the direct ones because a frozen ResNet-50 teacher runs on every step
(~0.7–0.8 GPU-hours per run vs ~0.4–0.5). `train_missing` skips anything with a
`DONE.json`, so re-running the sweep after adding the arm trains only what is new.

**The Fast-SCNN arm's first (previous-lineage) numbers are in
`_work/new_outputs/`**, and they are a null-to-negative result: Δ mean Dice
−0.005 vs direct (CI [−0.046, +0.032], P(better)=0.44), complete misses 7 → 13,
and the skin-tone fairness gap widened from 0.136 to 0.199 with Kruskal-Wallis
p = 0.0017 (the direct model's p = 0.014) — worst group Dark (VI). Treat these as
the previous lineage's numbers until the current sweep lands, but the *shape* of
the result — KD amplifying the miss rate and the fairness gap on a scratch
student — is what motivated the second arm.

When results land, the claim to check is **complete-miss rate and the fairness
gap**, not Dice: Stage A's B2→B0 distillation moved Dice by +0.004 (nothing) and
misses from 0.54 % to 0.18 % (the real effect). Score every contrast with a
paired subject-level bootstrap before stating it.

### 7b.6 The LR-ASPP arm — why it exists

`lraspp_mobilenetv3_distilled`: same DeepLabV3+/R50 teacher, same recipe, same
`efficient_alpha = 0.6`, student swapped from the weakest mobile model to the
strongest.

**The Fast-SCNN arm alone is uninterpretable.** Its null result cannot say
whether cross-architecture KD fails *here*, or whether KD simply cannot rescue a
scratch-initialised student on 697 training images. The two arms differ only in
the student, so the pair turns one anecdote into a two-point design over student
initialisation.

**What can and cannot move.** LR-ASPP direct already sits at the annotation
ceiling on mean Dice (0.746 vs `gbarimah_vs_erik` 0.755 in the previous lineage),
so a Dice win is not the hypothesis and should not be claimed if it appears. The
open headroom is elsewhere: its best seed missed 1 of 185 (the teacher misses
2–5 per seed — note the teacher is *worse* on misses than this student, the
reverse of the Fast-SCNN arm), and its fairness gap (0.082, n.s.) has room to
move in either direction. The interesting outcomes are therefore: (a) KD keeps
Dice at ceiling while tightening misses/fairness → distillation helps even
pretrained students; (b) KD regresses misses/fairness as it did for Fast-SCNN →
the regression is a property of this KD setup, not of scratch init; (c) nothing
moves → NON-INFERIOR, report as such.

**A teacher-worse-than-student caveat, stated up front.** On complete misses the
DeepLab teacher is behind LR-ASPP direct. `efficient_alpha = 0.6` still mixes
40 % hard labels, so the student is not capped by the teacher, but if arm (b)
occurs the first thing to check is whether the teacher's misses are where the
student's new misses appear (paired per-image, same as the DeepLab-vs-U-Net
containment check in §7b.1).

### 7b.7 The LR-ASPP arm — results (2026-07-30)

Three seeds, each scored at its own val-selected cut. The direct counterpart's
rows are the cached Stage E results (§7.2), unchanged by this session.

| | direct | distilled | Δ |
|---|---|---|---|
| 3-seed mean Dice | 0.7093 ± 0.0127 | **0.7263 ± 0.0125** | **+0.0170** |
| 3-seed mean median-Dice | 0.7723 | 0.7865 | +0.0142 |
| complete misses (of 555) | 3 — 0.54 % | 5 — 0.90 % | +2 |

Per seed, distilled:

| seed | cut | mean Dice | median | IoU | precision | recall | misses |
|---|---|---|---|---|---|---|---|
| 0 | −1.275 | 0.7406 | 0.7977 | 0.6139 | 0.8050 | 0.7420 | 1 |
| 1 | −0.900 | 0.7182 | 0.7782 | 0.5899 | 0.8094 | 0.6957 | 2 |
| 2 | −1.775 | 0.7200 | 0.7836 | 0.5912 | 0.8398 | 0.6833 | 2 |

Paired against direct at the same seed: **+0.0175 / +0.0116 / +0.0218**. Val Dice
at the selected seed (2) rose 0.7277 → 0.7348.

**The bootstrap has since run, and it is a WIN (Stage G, 2026-07-30).** At the
val-selected seed, Δ = **+0.0218**, 95 % CI **[+0.0081, +0.0372]** — the interval
excludes zero — two-sided p = 0.0014, **Holm-adjusted p = 0.0042** within the
confirmatory family. The seed-consistency check returns 3/3 positive, with seeds 0
and 2 individually excluding zero. The `mobile_field` omnibus rejects first
(χ² = 16.26, p = 0.0027, Kendall W = 0.145), so the pairwise test is licensed
rather than fished for.

This is the one place in the study where a sub-0.05 Dice difference clears §1's
bar, which permits exactly this: a paired subject-level bootstrap whose interval
excludes zero. It is not a licence to quote other sub-0.05 gaps.

Two qualifications that belong next to the claim. Holm ran over **three**
confirmatory contrasts, not the pre-specified four, because `fastscnn_distilled`
had no current-lineage result to score; at ×4 the adjusted p would be 0.0056, so
the verdict does not depend on it. And significance is not importance — +0.022
Dice remains smaller than the disagreement between two human annotators (§1,
§8b.7).

**What survives the caveat.** Three things are worth more than the point estimate:

- **The sign is the same at every seed**, with magnitudes clustered in 0.012–0.022
  rather than one seed carrying the mean. Three of three is weak evidence on its
  own (a sign test gives p = 0.125) but it is the right kind of weak evidence.
- **Precision and recall both rose** — seed 2 goes 0.8322 → 0.8398 and
  0.6548 → 0.6833. A pure threshold shift trades one against the other. Both
  moving up means the pixel *ranking* improved, which is a claim about the model
  and not about the operating point.
- **The student closed roughly half the gap to its teacher.** DeepLabV3+/R50 is at
  0.7453; direct LR-ASPP 0.7093 (gap 0.036), distilled 0.7263 (gap 0.019), at
  3.22 M parameters against the teacher's ~26.7 M.

**The pre-registered claim still did not land, and the stated risk did.** §7b.6
said the hypothesis was miss rate and fairness, *not* Dice — and Dice is the only
thing that moved in the right direction. On misses the Stage G contrast gives
Δ = +0.54 pp with P(fewer misses) = 0.19 and no containment either way (2 images
the distilled model misses that the direct one finds, 1 the other way). The arm
wins on the endpoint it was not built to win on and leaves its actual endpoint
untouched. Report it that way. Complete misses went 3 → 5 across the three seeds.
That is two images out of 555 and is not itself a result — but its direction
matches the caveat §7b.6 wrote down in advance: the DeepLab teacher misses more
than this student does (12 of 555, vs the student's 3), so the student inheriting
some of the teacher's miss behaviour along with its Dice is the predicted failure
mode, not a surprise. **The containment check §7b.6 specifies is now required**:
are the distilled model's added misses on images the teacher also misses? Until
that runs, the miss movement is unexplained rather than benign.

**The two-arm design did its job.** Fast-SCNN (scratch student): Δ −0.005, misses
7 → 13, fairness gap 0.136 → 0.199 at KW p = 0.0017. LR-ASPP (ImageNet student):
Δ +0.017, misses 3 → 5. The two arms differ only in the student, so **student
initialisation modulates the direction of cross-architecture KD** — it hurts a
scratch student and mildly helps a pretrained one. Neither arm alone could have
said that, which is exactly why the second one exists. Report the pair, never the
LR-ASPP arm by itself.

**Val-selection picked the weaker test seed.** Seed 2 won on validation (0.7348)
but seed 0 is the best on test (0.7406 vs 0.7200). That is the protocol working —
the cut and the seed are chosen on val, never test — but it means the
best-seed-reported number *understates* this arm by 0.021, and the 3-seed mean is
the fairer summary here.

**The cuts moved a long way negative** — −1.275 / −0.900 / −1.775, against direct's
−0.625 / −0.825. Two readings, and they are not exclusive. (a) Expected: the
teacher's logits are divided by a fitted temperature (§7b.2), so the student learns
a flatter logit distribution and needs a lower cut to reach the same operating
point. (b) Worth watching: the spread across seeds is 0.875 of logit against
direct's 0.2, drifting toward the far-negative, unstable-cut behaviour §7.2 reads
as a weak-feature symptom in Fast-SCNN. Since precision and recall both improved,
(a) is the better explanation for the *level*; (b) is still a real deployment
concern, because a shipped model needs one cut, not a per-seed one.

**Two provenance caveats before any of this is published.**

1. **This is a fresh arm scored against cached direct results.** The direct rows
   carry `source: cached` — the registry's RESULTS tier (§10), which is a
   legitimate substitute for a table entry, but the seed-0 provenance question in
   §7.2a is still open. Seeds 1 and 2 are the trustworthy pairs; the seed-0
   contrast (+0.0175) is the one to re-derive first.
2. **The model-level aggregate table gives this arm 3 misses where its own
   per-seed output gives 2.** That table is the known-bad one from §7.2a and it
   reproduced its old mismatches verbatim in the same session. Use the per-seed
   numbers.

**What to run next, in order:** (i) write the three per-image CSVs to
`WORK_DIR/outputs/` so any of this can be checked — nothing about this arm is on
disk in the bundle; (ii) `report.paired_contrast` distilled-vs-direct, subject
level, per seed; (iii) `fairness_analysis` on the distilled arm, which is the
pre-registered primary endpoint and is missing entirely; (iv) the teacher-miss
containment check; (v) regenerate the aggregate table through `report.normalize()`.

---

## 7c. Stage H — reliability-gated distillation

Added 2026-08-02; **all 27 runs trained the same day** (9 families × seeds 0/1/2).
Results are in §7c.11–7c.15. Full scope note: `docs/reliability_gated_kd_scope.md`;
deck: `docs/b2_distillation_deck.pptx`.

> **The headline, stated before the tables.** The gate demonstrably fired, and it
> changed nothing measurable. Reliability-gated KD is **indistinguishable from
> plain response KD** on all five students, on Dice and on complete misses alike.
> What *did* land is one step further out: **B2 as a teacher beats DeepLabV3+**
> (Fast-SCNN +0.034) **and KD beats no KD** (LR-ASPP +0.027, Fast-SCNN +0.036).
> Report the null as a null.

Two changes, separable on purpose, crossed into a design neither could resolve alone:

| | plain response KD | reliability-gated KD |
|---|---|---|
| **DeepLabV3+/R50 teacher** | Stage F — `*_distilled` | — (not run, §7c.6) |
| **SegFormer-B2 teacher** | `*_b2kd` | `*_rgkd` |

### 7c.1 The failure it exists to fix

§7b.7 recorded a pre-registered failure and its predicted cause. LR-ASPP distilled
from DeepLabV3+ gained Dice but went **3 → 5 complete misses**; Fast-SCNN went
**7 → 13**, with the ITA fairness gap widening 0.136 → 0.199 at KW p = 0.0017. The
explanation §7b.6 wrote down in advance: the teacher misses more than the student
(12 of 555 vs 3), so the student inherits the teacher's miss behaviour along with
its ranking.

The mechanism is mechanical. `DistillLoss` regresses the student onto
`sigmoid(teacher_logits / T)` at **every pixel of every image with one fixed weight
`1 − alpha`**. On an image the teacher misses, that soft target is not weak — it is
confidently, uniformly *empty*, applied with the same force it gets on an image the
teacher got right. The student is told with full confidence that there is no bruise
on exactly the images the complete-miss metric exists to catch.

### 7c.2 The mechanism

With `p` the calibrated teacher probability and `y ∈ {0,1}` the label:

```
r         = 1 − |2p − 1| · |p − y|                        per-pixel reliability
dice_T    = 2·Σ(p·y) / (Σp + Σy)                          teacher soft Dice, per image
g         = clip((dice_T − gate_lo)/(gate_hi − gate_lo), 0, 1)   gate_lo=0.10, gate_hi=0.50
w         = g · r
coverage  = mean(w)
alpha_eff = alpha + (1 − alpha)·(1 − coverage)
loss      = alpha_eff · supervised(GT) + (1 − alpha_eff) · Σ(w·BCE_pix)/Σw
```

| teacher is… | confidence | error | `r` | soft term |
|---|---|---|---|---|
| confidently right | ≈1 | ≈0 | ≈1 | full weight |
| **uncertain** | ≈0 | any | **≈1** | **full weight** |
| confidently wrong | ≈1 | ≈1 | ≈0 | suppressed |

**The middle row is the design.** Down-weighting by error alone would delete
exactly the pixels the teacher is unsure about — the dark knowledge distillation
exists to transfer. Multiplying by confidence keeps uncertainty and removes only
assertive error.

**Soft Dice, not thresholded**, so the gate needs no operating point. The cut is
not fitted until after training, and a gate that depended on one would entangle
this arm with threshold choice — the same reason model selection is on AP (§3).

**A ramp, not a step.** Augmentation is re-sampled every epoch, so an image whose
teacher Dice sits near a hard boundary would flip its KD term on and off between
epochs. `dice_T` is computed on the *augmented* batch, not a cached clean score: a
crop that removes most of a bruise genuinely is a view the teacher is less reliable
on.

**`alpha_eff` keeps alpha honest.** Gating removes gradient mass; handing it back
to the supervised term means gating cannot silently lower the effective KD weight
and masquerade as an alpha change. At `coverage = 1` this is `DistillLoss` term for
term; at `coverage = 0` it is `SupervisedLoss`.

### 7c.3 Why every contrast moves one variable

`reliability_kd.self_test()` asserts three properties rather than claiming them:

1. `r ≡ 1, g ≡ 1` ⇒ the gated loss **equals** `losses.DistillLoss` (measured 2.4e-07).
2. A teacher asserting empty on a real bruise ⇒ `g = 0` ⇒ the loss equals
   `losses.SupervisedLoss` exactly.
3. `p = 0.5` ⇒ `r = 1`.

Property 1 is why **alpha is not re-tuned** for the gated arms — they inherit
`segformer_alpha` / `efficient_alpha` / `yolo_alpha` from their controls. A gated
arm with its own alpha would be two changes, and the contrast would be
uninterpretable. Same reasoning as the fixed LR across architectures (§3).

It is also why every gated arm has a **teacher-matched** control.
`fastscnn_distilled` uses DeepLabV3+, so scoring `fastscnn_rgkd` against it would
move teacher and gate at once. The pair decomposes:

```
*_b2kd  vs  *_distilled     what changing the teacher does
*_rgkd  vs  *_b2kd          what the gate does, teacher held fixed
```

B0 already has a plain-B2 control in Stage A, so it needs only the gated arm.

### 7c.4 The twelve families

| family | student | teacher | KD | control |
|---|---|---|---|---|
| `segformer_b0_rgkd` | SegFormer-B0, 3.71 M | B2 | gated | `segformer_b0_distilled` |
| `lraspp_mobilenetv3_rgkd` | LR-ASPP MNv3, 3.22 M | B2 | gated | `lraspp_mobilenetv3_b2kd` |
| `fastscnn_rgkd` | Fast-SCNN, 1.14 M | B2 | gated | `fastscnn_b2kd` |
| `topformer_tiny_rgkd` | TopFormer-Tiny, 1.37 M | B2 | gated | `topformer_tiny_b2kd` |
| `ppmobileseg_tiny_rgkd` | PP-MobileSeg-Tiny, 1.45 M | B2 | gated | `ppmobileseg_tiny_b2kd` |
| `lraspp_mobilenetv3_b2kd` | LR-ASPP MNv3 | B2 | response | `lraspp_mobilenetv3_distilled` |
| `fastscnn_b2kd` | Fast-SCNN | B2 | response | `fastscnn_distilled` |
| `topformer_tiny_b2kd` | TopFormer-Tiny | B2 | response | `topformer_tiny_distilled` |
| `ppmobileseg_tiny_b2kd` | PP-MobileSeg-Tiny | B2 | response | `ppmobileseg_tiny_distilled` |
| `topformer_tiny_distilled` | TopFormer-Tiny | DeepLabV3+ | response | `topformer_tiny` |
| `ppmobileseg_tiny_distilled` | PP-MobileSeg-Tiny | DeepLabV3+ | response | `ppmobileseg_tiny` |

The last two are **Stage F**, not H, and were added here because without them "B2
is the better teacher" would rest on two students instead of four.

Cost at three seeds: Stage H **27 runs ≈ 24 GPU-hours** (nine families: five
gated arms and four teacher-matched controls); Stage E/F 24 runs ≈ 15.
`RGKD_SEEDS` controls the former independently of `EFFICIENT_SEEDS`, and defaults
to matching it — a 3-seed arm against a 1-seed control is not a contrast.

### 7c.5 The YOLO arm — built, not registered, PENDING (§14 gap 2b)

YOLO never sees a loss object; it trains natively under Ultralytics and the teacher
reaches it only through the pseudo-mask baked into the training labels (§4). So the
gate is applied at the fusion, with `alpha` promoted from a scalar to a per-pixel
quantity that rises to 1 — pure ground truth — where the teacher is unreliable:

```
a_pix = alpha + (1 − alpha)·(1 − g·r)
class = (a_pix·gt + (1 − a_pix)·teacher_prob ≥ 0.5)
```

At `g·r = 1` this is Stage A's fusion character for character.
**The YOLO arm is implemented but not registered.** `build_gated_yolo_dataset`
applies the gate to YOLO's offline pseudo-mask fusion and is tested, but
`yolo_sem_rgkd` is absent from `TEACHER_FOR`, `STAGE_H_FAMILIES`, `CONTROL_FOR`
and `CONTRAST_FAMILY_H`. Nothing schedules it, and a Stage E+H session therefore
never needs Ultralytics installed at all.

The cost is stated, not hidden: **every confirmatory Stage H contrast tests the
gate on an ONLINE loss**, and the study says nothing about whether it transfers to
the offline pseudo-mask route. Holm corrects over three contrasts, not four.
Re-enabling is three lines in `reliability_kd` plus the matching `registry` and
`CONTRAST_FAMILY_H` entries; the `FAMILY_SPEC` row, the notebook's YOLO training
branch and its native-argmax scoring branch are all still in place.

`reliability_kd.build_gated_yolo_dataset` is a **deliberate copy** of
`yolo_native.build_yolo_dataset` with those two lines changed — `yolo_native.py` is
a build artefact (§16), and §15 trap 11 is explicit that a tested call sequence gets
copied rather than rewritten.

### 7c.6 Why no gated DeepLabV3+ arm

It is the tidiest fourth cell and it is left empty rather than quietly omitted.
2×2×4 students is 24 arms before seeds, and "does gating fix the inherited-miss
failure" is answerable within one teacher. B2 is chosen because a weak teacher
confounds "gating did not help" with "there was nothing worth transferring"; the
licence argument that put DeepLabV3+ in Stage F (§7b.1) is about deployment and is
unchanged. If gating works, that cell is one line per arm in
`reliability_kd.TEACHER_FOR`.

### 7c.7 How it is wired

`bruisekit/reliability_kd.py`, imported as `RK`. Two runtime rebinds, the same
pattern as `distill_efficient` and `efficient_models`, for the same reason: `engine.py`,
`losses.py` and `yolo_native.py` are extracted verbatim at build time (§16), so an
edit there is reverted by the next `60_build_unified_bundle.py`.

| name | rebound to |
|---|---|
| `engine.DistillLoss` | a dispatcher returning the gated loss for `*_rgkd` arms, the original otherwise |
| `engine.load_teacher` | resolves the arm's teacher from `RK.TEACHER_FOR`; falls through otherwise |

Both dispatch on `RK.ACTIVE_ARM`, set by the `RK.arm(family)` context manager,
because `train_run` tells neither the loss nor the teacher loader which run it is
training — it passes only a seed-bearing path. **Outside `arm()` both shims are
inert**, so installing them cannot change a Stage A/B/C/E/F number. Install
`RK.install_teacher_shim` *last* so it sits in front of Stage F's shim.

`train_run` itself is untouched. It already distils when the spec says
`distill=True`; Stage H changes only which teacher it loads and how the soft term
is weighted.

**B2's temperature is read from the shipped `calibration.json`, never re-fitted.**
`engine.train_run` writes one for any `segformer_b2_teacher` run, so Stage H's
students distil from exactly the temperature Stage A's did (T ≈ 1.760 / 1.767 /
1.792). Two temperatures for one checkpoint would make the two stages' arms
incomparable. DeepLabV3+ has none and is fitted through
`distill_efficient.ensure_calibration`, cached under `WORK_DIR`.

### 7c.8 Multiplicity: a separate confirmatory family

`reliability_kd.CONTRAST_FAMILY_H` — 4 confirmatory, 7 exploratory —
Holm-corrected **within itself**, reported as `FAM_H`, separate from
`significance.CONTRAST_FAMILY` / `FAM`.

Appending these to the existing family would re-penalise a comparison already
conducted and reported (the Stage F LR-ASPP arm at Holm p = 0.0042, k = 3) for
reasons having nothing to do with it. Correction is over the comparisons made to
answer *one* question; this is a different question.

For the same reason `OMNIBUS_SETS["mobile_field"]` was **not** extended with the two
new DeepLabV3+ arms — that omnibus has been run and quoted (χ² = 16.26, p = 0.0027,
W = 0.145), and adding models would silently change a published number. Three new
sets were added instead: `teacher_axis_lraspp`, `teacher_axis_fastscnn`,
`gated_arms`.

### 7c.9 What would count as failure, stated in advance

**The pre-registered primary endpoints are complete-miss rate and the ITA fairness
gap**, not mean Dice — which the annotation ceiling has already capped (§1). §7b.7
is the cautionary case: that arm won on the endpoint it was not built to win on and
left its actual endpoint untouched.

| outcome | how it is reported |
|---|---|
| misses down, fairness gap down or flat | the gate does what it was built to do |
| Dice up, misses unchanged | the same non-result as Stage F; say so in those words |
| `mean_coverage → 1.0` | the gate never fired; the arm **is** its control. Report as "B2 is reliable enough here that gating has nothing to remove" — a finding, not a method |
| `mean_coverage → 0` | the gate ate the arm; it trained as a supervised baseline. Retune `rgkd_gate_lo/hi` and retrain; do **not** reinterpret |
| nothing moves | NON-INFERIOR, or INCONCLUSIVE if the interval is wide (§8b.4) |

**`GATE_H` is read before `HEAD_H`, always.** An arm whose gate never fired and one
whose gate fired constantly are different experiments and look identical in a Dice
table. The notebook prints a loud line for either.

**A known risk, flagged before the first run.** The gate conditions on the
*augmented* batch, which is correct, but the teacher's soft Dice there can sit well
below its 0.769 test mean. The CPU end-to-end test hit exactly this at 128 px
(teacher soft Dice 0.013, coverage 0.0) — an artefact of feeding a 640-trained
teacher a 128 px input, but the same shape as the real failure. **Check
`mean_teacher_soft_dice` on the first real run before trusting anything else in the
stage.**

### 7c.10 Verification already done

- `RK.self_test()` — 5 checks, all pass on CPU in under a second, and it raises
  rather than printing a red row.
- A CPU end-to-end mini-run (6 images, 128 px, 1 epoch) of `fastscnn_rgkd` and
  `fastscnn_b2kd` exercising teacher shim → loss shim → `train_run` → gate stats →
  `cache_logits` → `sweep_cuts` → `select_cut` → `evaluate_at_cut` → registry
  re-scan. §15 trap 11 is precisely a cell in this position shipping broken because
  only a GPU run would have touched it.
- Shim isolation: gated arm → gated loss; control arm and outside `arm()` →
  `losses.DistillLoss`; installing twice does not wrap the wrapper.

### 7c.11 Results — three seeds, 185 test images

Each seed scored at its own val-selected cut. `misses` is the count over all three
seeds (555 image-scorings), per §7.2a: publish the `dice == 0` column.

| arm | teacher | KD | mean Dice | median | misses /555 |
|---|---|---|---|---|---|
| `segformer_b0_rgkd` | B2 | gated | **0.7614 ± 0.0077** | 0.8063 | 1 |
| `lraspp_mobilenetv3_b2kd` | B2 | response | 0.7376 ± 0.0146 | 0.7879 | 1 |
| `lraspp_mobilenetv3_rgkd` | B2 | gated | 0.7209 ± 0.0188 | 0.7771 | 1 |
| `topformer_tiny_b2kd` | B2 | response | 0.6951 ± 0.0021 | 0.7289 | 2 |
| `topformer_tiny_rgkd` | B2 | gated | 0.6883 ± 0.0076 | 0.7319 | 2 |
| `ppmobileseg_tiny_rgkd` | B2 | gated | 0.6813 ± 0.0028 | 0.7331 | 13 |
| `ppmobileseg_tiny_b2kd` | B2 | response | 0.6784 ± 0.0030 | 0.7196 | 10 |
| `fastscnn_b2kd` | B2 | response | 0.6186 ± 0.0547 | 0.6941 | 10 |
| `fastscnn_rgkd` | B2 | gated | 0.6084 ± 0.0301 | 0.6744 | 16 |

`fastscnn_b2kd`'s spread (± 0.055) is by far the widest in the stage and is the one
number here that should never be quoted without it.

### 7c.12 The gate fired — so the null is a real null

Averaged over the 15 gated runs
(`FINAL_RESULT/reliability_gate_diagnostics.csv`):

| diagnostic | value | reading |
|---|---|---|
| mean coverage (`g · r`) | **0.906** | 1.0 would mean the gate never fired |
| mean effective α | 0.638 | against nominal 0.600 |
| image-views fully gated off | 2.8 % | teacher ignored entirely on these |
| teacher near-miss views | 1.9 % | teacher soft Dice ≤ 0.05 — the target population |
| mean pixel reliability | 0.987 | most pixels are confidently correct, as expected |

This is the diagnostic §7c.9 said to read first, and it lands in the useful middle
of both failure modes: the gate was neither inert (`coverage → 1`) nor overwhelming
(`coverage → 0`). Roughly one image-view in thirty-five had its KD term removed
entirely — exactly the population the gate was built for.

**So §7c.13's null is evidence that suppressing the teacher's confident errors does
not change what the student learns on this dataset.** It is not evidence that the
gate did nothing.

### 7c.13 The confirmatory result: gating did not beat response KD

`CONTRAST_FAMILY_H`, Holm over the three confirmatory contrasts. Teacher held fixed
at B2; only the loss differs.

| contrast | Δ Dice | 95 % CI | verdict |
|---|---|---|---|
| `segformer_b0_rgkd` − `segformer_b0_distilled` | −0.0024 | [−0.0156, +0.0110] | INCONCLUSIVE |
| `lraspp_mobilenetv3_rgkd` − `lraspp_mobilenetv3_b2kd` | +0.0033 | [−0.0133, +0.0204] | INCONCLUSIVE |
| `fastscnn_rgkd` − `fastscnn_b2kd` | −0.0087 | [−0.0321, +0.0140] | INCONCLUSIVE |

Every interval spans zero and no Holm-adjusted p falls below 1.0. The two
exploratory students agree (`topformer` −0.0173, `ppmobileseg` +0.0023, both
INCONCLUSIVE).

**The pre-registered endpoint did not move either.** Complete misses as counts
(§8b.5): `segformer_b0_rgkd` 2 vs `segformer_b0_distilled` 0 — *b within a*, i.e.
gating **added** two misses and removed none. `ppmobileseg` 7 vs 4; `topformer`
1 vs 1 (identical); `fastscnn` 6 vs 6 with 4 discordant each way; `lraspp` 2 vs 1.
Nothing here is a gain.

**And the sign is not stable across seeds.** `contrast_by_seed` on the two mobile
gate contrasts returns `sign_consistent = False`, 1 of 3 seeds positive, with
per-seed deltas of −0.049 / −0.005 / +0.003 (LR-ASPP) and −0.050 / −0.008 / +0.028
(Fast-SCNN). A val-selected-seed point estimate for these contrasts is one draw from
a distribution that straddles zero; quoting one would be the §8b.6 error.

### 7c.14 What did land: the teacher, and KD itself

Exploratory, uncorrected, and each interval excludes zero:

| contrast | Δ Dice | 95 % CI | verdict |
|---|---|---|---|
| `fastscnn_b2kd` − `fastscnn_distilled` (**B2 vs DeepLabV3+**) | +0.0343 | [+0.0084, +0.0593] | **WIN** |
| `lraspp_mobilenetv3_rgkd` − `lraspp_mobilenetv3` (**KD vs none**) | +0.0265 | [+0.0065, +0.0517] | **WIN** |
| `fastscnn_rgkd` − `fastscnn` (**KD vs none**) | +0.0364 | [+0.0045, +0.0659] | **WIN** |

The Fast-SCNN teacher swap is the result that justifies Stage H existing: same
student, same recipe, same α, and **only the teacher changes**. It also reverses
Stage F's finding — `fastscnn_distilled` (DeepLabV3+) was null-to-negative; with B2
in its place the same student gains 0.034 Dice. The Fast-SCNN KD-vs-none win also
cuts complete misses by 3.78 pp.

**The fairness result is the most interesting thing in the stage, and it is not
about the gate.** On Fast-SCNN, the ITA gap and its Kruskal–Wallis test
(`FINAL_RESULT/stage_h_fairness.csv`):

| arm | fairness gap | KW p | significant |
|---|---|---|---|
| `fastscnn` (no KD) | 0.160 | 0.021 | yes |
| `fastscnn_distilled` (DeepLabV3+) | 0.141 | 0.0066 | yes |
| **`fastscnn_b2kd` (B2, response)** | **0.064** | **0.358** | **no** |
| `fastscnn_rgkd` (B2, gated) | 0.142 | 0.029 | yes |

A B2 teacher with **plain** response KD removes a significant skin-tone gap that
neither the direct model nor the DeepLabV3+ arm removes — and the gated variant does
not. That is the one place gating looks actively *worse* than its control, on the
endpoint §7c.9 pre-registered. At 28 subjects it is a single comparison and must not
be over-read, but it is the finding to chase next. PP-MobileSeg moves the same way
(0.179 sig → 0.122 sig → 0.109 n.s. gated); LR-ASPP is not significant in any arm.

### 7c.15 Against the `segformer_b0_direct` boundary

A second reading, for a reader who wants one reference line rather than
arm-vs-its-own-control: every B2-taught arm against the strongest model in the study
trained with **no teacher at all** (0.7663 mean Dice, 1 miss). Produced by
`scripts/70_b2_teacher_significance.py` →
`FINAL_RESULT/significance_b2_teacher_vs_b0_direct.csv`, figure
`figures/H1_b2_vs_b0_direct.png`.

| arm | Δ Dice | 95 % CI | p (Holm) | verdict |
|---|---|---|---|---|
| `segformer_b0_distilled` | +0.0017 | [−0.0088, +0.0135] | 1.000 | NON-INFERIOR |
| `segformer_b0_rgkd` | −0.0008 | [−0.0137, +0.0153] | 1.000 | INCONCLUSIVE |
| `lraspp_mobilenetv3_b2kd` | −0.0450 | [−0.0871, −0.0054] | 0.062 | INCONCLUSIVE |
| `lraspp_mobilenetv3_rgkd` | −0.0417 | [−0.0712, −0.0107] | 0.033 | INFERIOR |
| `topformer_tiny_b2kd` | −0.0692 | [−0.0876, −0.0513] | 0.001 | INFERIOR |
| `topformer_tiny_rgkd` | −0.0865 | [−0.1151, −0.0620] | 0.001 | INFERIOR |
| `ppmobileseg_tiny_b2kd` | −0.0905 | [−0.1231, −0.0565] | 0.001 | INFERIOR |
| `ppmobileseg_tiny_rgkd` | −0.0881 | [−0.1250, −0.0504] | 0.001 | INFERIOR |
| `fastscnn_b2kd` | −0.1159 | [−0.1457, −0.0850] | 0.001 | INFERIOR |
| `fastscnn_rgkd` | −0.1246 | [−0.1582, −0.0887] | 0.001 | INFERIOR |

**Only the two B0 arms hold the line.** Distilling B2 into a same-family 3.71 M
student costs nothing measurable; distilling it into a 1–3 M mobile architecture
narrows the gap to that student's own baseline (§7c.14) but does not close the gap to
B0. Both statements are true at once and neither should be quoted alone.

This table is deliberately stricter than §7c.13's: ten arms against one reference is
a *ranking against a boundary*, which is the reading that inflates false positives,
so Holm runs over all ten rather than over a confirmatory subset.

---

## 8. Stage D — analysis methodology

Everything in Stage D is a function of **per-image Dice**, not of model weights.
That is why the whole analysis reproduces on a laptop with no GPU and no `torch`.

### 8.1 The common schema

Five different producers wrote per-image CSVs with different columns.
`report.normalize()` reduces all of them to one schema by **recomputing** the
derived columns from the seven-column core rather than trusting whichever ones
happen to be present:

```
tp = dice * (pred_positive + gt_positive) / 2      # definition of Dice
fp = pred_positive - tp
fn = gt_positive  - tp
complete_miss = dice == 0
pred_gt_ratio = pred_positive / gt_positive
```

Verified exact against the shipped outputs to 4e-12. Recomputing is safer than
reading: if one producer's `complete_miss` ever disagreed with its own `dice`,
taking the stored column would propagate that bug into a cross-model comparison.

Subject and ITA columns are **always** dropped and re-joined from the manifest,
so there is one source of truth for who each image belongs to.

### 8.2 Subject-level cluster bootstrap

The 185 test images come from **28 subjects**, and images of the same bruise are
strongly correlated. Resampling images would treat 185 correlated observations as
independent and produce intervals roughly `sqrt(185/28) ≈ 2.6×` too narrow.

Every CI and contrast resamples **subjects** with replacement, taking all of a
chosen subject's images. Contrasts are **paired** — both models saw the same 185
images, and the same resampled subject list is applied to both on each draw;
resampling independently would discard the pairing and hide real small effects.

`P(A better)` is reported instead of a p-value: at 28 subjects it is the honest
quantity, and it avoids a significant/not-significant dichotomy.

Implementation detail: draws resample **index arrays**, not DataFrames. The
obvious `pd.concat` implementation allocates a new frame 2000 times per model,
which on a 16-model comparison is 32k allocations — enough to kill a Colab
kernel. This was found the hard way.

### 8.3 The figure suite

| ID | Content |
|---|---|
| D1 | Headline accuracy — mean and median Dice side by side |
| D2 | Per-image Dice violins + survival curves |
| D3 | Complete-miss rate + miss-vs-median-Dice scatter |
| D4a | Marginal 95% CIs (subject bootstrap) |
| D4b | Paired contrasts forest plot with P(better) |
| D5 | Per-ITA-group median-Dice heatmap + fairness gap bars |
| D6a | Dice and miss rate by bruise-size quintile |
| D6b | Bruise size distribution by ITA group |
| D7 | Annotation ceiling — models vs human pairs |
| D8 | Accuracy vs latency, marker area ∝ params |

**Median alongside mean, always.** The per-image distribution is strongly
left-skewed; a handful of complete misses drags the mean down several points
while the median barely moves. For the YOLO variants the gap between them *is*
the finding.

### 8.4 The size ↔ fairness confound

Bruise size is the strongest single predictor of whether a model finds a bruise
at all, **and** it is not evenly distributed across ITA groups in this dataset.
Any fairness claim that does not condition on size is measuring both at once.

This is why D6 sits immediately after D5 rather than in an appendix, and why the
apparent Light-skin under-detection in YOLO should not be reported as a skin-tone
effect without the size-stratified analysis.

---

## 8b. Stage G — final significance

Stage D **describes**; Stage G **decides**. They are separate sections backed by
separate modules (`report.py` vs `significance.py`) because a confirmatory
analysis only means anything if its comparison list was fixed before the numbers
were seen — and a contrast list that lives next to the plotting code gets appended
to, while one that lives under `significance.CONTRAST_FAMILY` with a docstring
forbidding it does not.

### 8b.1 Why not an all-pairs tournament

Twenty scored models is 190 ordered pairs. At 28 subjects, with every model inside
the annotation-ceiling band, an all-pairs sweep at α = 0.05 is expected to return
about **ten "significant" differences from noise alone** — and it would be read as
a ranking, which is exactly what §1 forbids. `CONTRAST_FAMILY` is twelve
comparisons, each attached to a question someone actually asked.

Four are **confirmatory** — specified as the point of an experiment before it ran
(both Stage F arms, Stage A's distillation, and the B2→B0 compression claim) — and
Holm–Bonferroni is applied within that set of four. The other eight are
**exploratory**, reported uncorrected and labelled. Folding all twelve into one
correction would either over-penalise the questions the study was designed around
or launder eight post-hoc comparisons into confirmatory ones.

### 8b.2 The omnibus gates the pairwise table

G1 runs a **Friedman test on subject-mean Dice** — 28 rows, not 185. Blocking on
images would reuse the same information ~6.6× per subject and let the test believe
it has 185 independent blocks.

One test over all twenty models is the wrong test: 19 degrees of freedom, and most
of those models are B0 students of one teacher differing by a loss term. So the
omnibus runs on `OMNIBUS_SETS` — four fields a reader would actually ask "are these
different at all?" about. Measured on the shipped cached results:

| set | k | χ² | p | Kendall W | rejects |
|---|---|---|---|---|---|
| headline (Stage A + B) | 7 | 4.52 | 0.607 | 0.027 | **no** |
| segformer_family | 3 | 0.93 | 0.629 | 0.017 | **no** |
| kd_arms (Stage C) | 10 | 20.63 | 0.014 | 0.082 | yes |
| mobile_field | 6 | 18.16 | 0.0027 | 0.130 | yes |
| teacher_axis_lraspp | 4 | 10.16 | 0.017 | 0.121 | yes |
| teacher_axis_fastscnn | 4 | 6.99 | 0.072 | 0.083 | **no** |
| gated_arms (Stage H) | 10 | 73.56 | <0.0001 | 0.292 | yes |

The last four landed 2026-08-02. `gated_arms` rejects harder than anything else in
the study (W = 0.292, an order of magnitude above `headline`'s 0.027) — but read
what that set contains: five gated arms and their five controls, spanning 0.61 to
0.77 Dice. The omnibus is detecting the **student**, not the gate. It licenses
pairwise tests inside Stage H; it says nothing about whether gating works, and
§7c.13 shows it does not. The informative pair is `teacher_axis_lraspp` rejecting
while `teacher_axis_fastscnn` does not.

**The seven headline models are collectively indistinguishable at subject level.**
That is not a null to bury — it is §1's annotation-ceiling finding arriving through
a second, independent route, and it is the single most defensible sentence the
study can make about model choice. Kendall's *W* = 0.027 means subjects essentially
do not agree on how to rank these models.

The Stage C arms do reject, but read that carefully: those ten arms include several
that are plainly weaker (`x_angular` 0.7418, `p3_adaptive_hard` 0.7414 against
`p3_adaptive` 0.7748), so the omnibus is detecting the bad arms, not vindicating
the good ones. It licenses pairwise comparisons *inside* Stage C; it says nothing
about the headline field.

**Quoting a pairwise winner from a non-rejecting set is the easiest way to
over-claim with this data.**

### 8b.3 Three endpoints, one set of draws

`paired_contrast_multi` evaluates mean Dice, median Dice and complete-miss rate on
the **same** resampled subject lists. Three separate bootstraps with three seeds
would give three intervals that cannot be read against each other — a model could
appear to gain Dice and lose misses in draws that never co-occurred. Sharing the
draws makes "Dice up, misses up" a statement about the same 10,000 resampled
worlds, which is precisely the pattern §7b.7 needs to report.

`n_boot` is **10,000** here against Stage D's 2,000: 2,000 is ample for an
interval, but a tail probability quoted to two decimals cannot be resolved below
1/2000, and Stage G's p-values are then Holm-adjusted, where input granularity
propagates straight through.

### 8b.4 INCONCLUSIVE is not NON-INFERIOR

Stage C's rule (§6.5) was WIN / INFERIOR / **anything else → NON-INFERIOR**. That
silently classes an interval running from −0.05 to +0.02 as equivalence, which
reports absence of evidence as evidence of absence. `significance.verdict` adds a
fourth label:

| verdict | condition |
|---|---|
| WIN | `lo > 0` |
| INFERIOR | `hi < −margin` |
| NON-INFERIOR | `lo ≥ −margin` |
| **INCONCLUSIVE** | interval extends below −margin but not entirely |

On the cached results this changes real verdicts: B0-distilled vs B2-teacher is
INCONCLUSIVE (CI [−0.020, +0.018]), not the equivalence Stage C's rule would have
called it. **Stage C's ten arms should be re-scored under this rule** before their
NON-INFERIOR verdicts are quoted again.

### 8b.5 Misses get counts, not just rates

At 0–13 misses of 185, a bootstrapped difference of two rates near the boundary is
unstable and its interval should not be what anyone quotes. `discordance` returns
the raw 2×2 — how many images A misses that B finds, and vice versa — plus a
`containment` verdict. This is the form already used for the DeepLab-vs-U-Net
teacher choice (§7b.1), and it is the evidence itself rather than a summary of it.

It is also where the headline field separates even though the omnibus does not:

| contrast | A misses | B misses | discordant | containment |
|---|---|---|---|---|
| b2_teacher vs unet_r50 | 0 | 7 | 0 / 7 | **b2 strictly within unet** |
| b0_direct vs yolo_sem_direct | 1 | 12 | 0 / 11 | **b0 strictly within yolo** |
| b0_distilled vs b0_direct | 0 | 1 | 0 / 1 | b0_distilled within b0_direct |

Strict containment in every case — no model in the SegFormer line ever misses a
bruise its comparator finds. That is a far stronger statement than any Dice
contrast in the study, and it comes out of counts rather than intervals.

The McNemar column is exact but **ignores subject clustering**, so it is
anti-conservative. It is a sanity check on the counts, not the inferential result;
where it disagrees with the clustered interval, believe the interval.

### 8b.6 Seed consistency

G4 repeats a contrast at every seed both models share and reports
`sign_consistent` / `n_seeds_positive`. A contrast at the val-selected seed is one
draw from the training distribution; three same-signed deltas of similar magnitude
are stronger evidence than one large delta. This is the concrete argument for
pulling **every** seed's `test_per_image.csv` off ORC rather than only the best
seed's — with only the best seed on disk, G4 cannot run at all.

### 8b.7 What Stage G cannot do

Significance answers *is it real*; §1's annotation ceiling answers *does it
matter*. A confirmed 0.02 Dice difference is still smaller than the disagreement
between two human annotators. Both sections must be quoted together, and D7 comes
first.

---

## 9. Code architecture

**Thin notebook, fat library.** Every notebook cell is config, a call into
`bruisekit`, or a rendering of what came back. No cell defines a model, a metric
or a training loop. Reading the notebook tells you *what* the study does; the
modules tell you *how*.

```
bruisekit/
├── data.py            BruiseDataset, make_loader, augmentation. Emits raw [0,1].
├── models.py          SegFormerNet, YoloSemNet, build_model, build_param_groups
├── losses.py          DiceBCELoss, SupervisedLoss, DistillLoss
├── metrics.py         dice_np, iou_np, compute_image_row, summarize, BinnedAP
├── engine.py          train_run (THE training loop), resume, calibration, probe
├── sweep.py           cache_logits, sweep_cuts, select_cut  (threshold fitting)
├── evaluate.py        evaluate_at_cut, fairness_analysis, benchmark_speed
├── postopt.py         TTA, prob sweeps, seed averaging
├── yolo_native.py     native Ultralytics training + argmax inference
├── smp_models.py      SMPNet (U-Net, DeepLabV3+) + build_model shim
├── nnunet_native.py   nnU-Net v2 wrapper (unused — never run)
│
├── paths.py           ★ Env resolution. The ONLY host-aware module.
├── registry.py        ★ The train-or-skip brain. Three tiers.
├── loaders.py         ★ Run → live model → test numbers
├── report.py          ★ per-image CSV → tables and statistics (descriptive)
├── significance.py    ★ Stage G: omnibus, contrast family, multiplicity (§8b)
├── efficient_models.py ★ Stage E: the four mobile architectures
├── distill_efficient.py ★ Stage F: teacher shim, calibration, arm driver (§7b)
├── reliability_kd.py  ★ Stage H: gated loss, loss+teacher shims, gated YOLO (§7c)
├── weights.py         ★ Stage E: download, cache, verify, provenance
├── mmcv_shim.py       ★ minimal mmcv stand-in
├── vendor/            StrideFormer, PP-MobileSeg head, TopFormer — VERBATIM
└── kd/                Stage C distillation suite — vendored unmodified
```

★ = written for this bundle; source of truth is `scripts/unified_lib/`.

### The model interface

Every model, in every stage, satisfies exactly this:

```python
forward_train(x) -> (logits[B,1,H,W], aux_logits | None)
forward(x)       -> logits[B,1,H,W]
.backbone        -> the pretrained part, for the encoder/head LR split
```

`x` is raw [0,1]. One bruise logit at full input resolution. **Nothing downstream
ever branches on a model's name** — loss, sweep, metric and benchmark are all
architecture-blind. That property is what makes adding a model cheap.

### Why one logit and not two classes

All reference implementations emit `num_classes` channels. Building with
`num_classes=1` gives the bruise logit directly, which is what every downstream
consumer expects. Two classes plus softmax is the same function via one more
transformation and one more place to get the sign backwards.

---

## 10. The registry and the three tiers

`registry.py` resolves every run in the study to exactly one tier **before any
compute happens**, and prints the plan.

| Tier | Meaning | Cost |
|---|---|---|
| **WEIGHTS** | usable checkpoint exists | nothing trains |
| **RESULTS** | no checkpoint, but metrics were recorded | nothing trains; labelled `cached` |
| **MISSING** | neither | the only case where training is proposed |

**Why RESULTS is a real tier and not a fudge.** A cached metric cannot make a new
prediction, but it *is* a complete substitute for a number in a table: the same
code, the same weights, the same 185 images produced it. It is labelled
everywhere it surfaces.

**Fail loud, never silently.** A MISSING run stays missing in every table it
would have fed. It is never dropped, never back-filled from a neighbouring seed,
never averaged away.

Per-family checks differ, deliberately:

| Family | "trained" means |
|---|---|
| SegFormer / SMP / efficient | `best.pt` **and** `operating_point.json` |
| YOLO | `ultralytics_runs/train/weights/best.pt` (argmax needs no threshold) |
| nnU-Net | `fold_0/checkpoint_final.pth` |
| KD arm | `DONE.json` **and** `best_model.pt` |

A KD directory with a checkpoint but no `DONE.json` is an interrupted run — its
weights are a snapshot, not a result, and treating it as finished would put an
unconverged model in the comparison table.

**Fresh runs never overwrite shipped checkpoints.** Training writes to
`WORK_DIR/runs/`; the registry prefers shipped checkpoints unless a run_id is in
`FORCE_RETRAIN`.

---

## 11. How to add a new model

The design goal is that this takes four small edits and no changes to the
training loop, the metrics, or the analysis.

### Step 1 — implement the architecture

Add a class to `scripts/unified_lib/efficient_models.py` (or a new module)
satisfying the interface in §9:

```python
class MyNet(_Normalised):
    """One-line description. Params, and what it is pretrained on."""

    def __init__(self, num_classes: int = 1):
        super().__init__()
        self.net = ...                      # your architecture, 1-class head
        self.init_source = "random init"    # overwritten by build_with_pretrained

    @property
    def backbone(self):
        return self.net.encoder             # whatever the pretrained part is

    def forward_train(self, x):
        logits = self.net(self.norm(x))     # self.norm applies ImageNet stats
        # Must come back at INPUT resolution:
        logits = F.interpolate(logits, size=x.shape[2:], mode="bilinear",
                               align_corners=False)
        return logits, None                 # (logits, aux) — aux None if no aux head
```

If it only exists in another framework, **vendor it verbatim** — see §7.1 and
`scripts/vendor_efficient_nets.py`. Do not retype an architecture whose
checkpoint you intend to load.

Register it:

```python
EFFICIENT_ARCHS["mynet"] = MyNet
PUBLISHED_PARAMS_M["mynet"] = (2.5, 19)     # (published params, class count)
_HEAD_IN_CHANNELS["mynet"] = 128            # for the 1-class rescale
```

### Step 2 — declare its weights

In `scripts/unified_lib/weights.py`:

```python
"mynet": WeightSource(
    key="mynet",
    filename="mynet-imagenet-abc123.pth",
    url="https://.../mynet-imagenet-abc123.pth",
    kind="auto",                    # "auto" | "manual" | "none"
    sha256="...",                   # optional but recommended
    init="MyNet backbone, ImageNet-1k",
    note="Backbone only; head trains fresh, as for every other baseline.",
    expect_missing=("decode_head.",),   # prefixes intentionally absent
),
```

Use `kind="none"` if no pretrained weights exist — that is a legitimate state,
not a failure, and the model will be labelled as scratch-initialised everywhere.

### Step 3 — declare the spec

In `scripts/unified_lib/loaders.py`:

```python
FAMILY_SPEC["mynet"] = {"arch": "mynet", "size": None, "distill": False}
EFFICIENT_FAMILIES = (..., "mynet")
```

### Step 4 — register it for scanning

In `scripts/unified_lib/registry.py`:

```python
EFFICIENT_FAMILIES = (..., "mynet")
COST_HOURS["mynet"] = 0.5
```

### Step 5 — verify before spending GPU time

```python
from bruisekit import efficient_models as EM
EM.self_test()                    # shapes, param counts vs published
EM.verify_checkpoint_match(env)   # does the official checkpoint load by key?
```

`self_test` catches structural errors in seconds. A transformer with the wrong
`key_dim` still emits `[B,1,H,W]` and still trains — it is simply not the model
named in your table.

### Step 6 — rebuild and run

```bash
python scripts/60_build_unified_bundle.py
python scripts/61_generate_unified_notebook.py
```

Then `RUN_STAGES = "E"`, `ALLOW_TRAINING = True`. Training, threshold fitting,
scoring, all analysis and all figures come for free — none of them know your
model exists.

### What you must NOT do

- **Do not change the recipe** for your model. If it needs a different LR or more
  epochs to work, that is a finding to report, not a knob to turn — the shared
  recipe is the controlled variable.
- **Do not use segmentation-pretrained weights** unless every other model gets
  them too.
- **Do not normalise in the dataloader.** Pixel scale belongs to the model.
- **Do not fit the threshold on test.** Ever.

---

## 12. How to add a new distillation arm

This section is about **Stage C** arms — SegFormer students under the vendored KD
suite. Two other mechanisms exist and neither goes here:

- A *mobile* student with a *non-SegFormer* teacher: see §7b — add an entry to
  `distill_efficient.TEACHER_FOR` plus a `FAMILY_SPEC` row with `distill=True`.
- A new *loss* on any student: see §7c — add it to `reliability_kd.py` behind the
  `engine.DistillLoss` dispatcher, add the family to `TEACHER_FOR` /
  `STUDENT_ARCH` / `CONTROL_FOR` / `registry.STAGE_H_FAMILIES`, and give it a
  teacher-matched control so the contrast moves one variable. Prove the new loss
  reduces to its control in `self_test()` before training anything.

Stage C is driven by `bruisekit/kd/` (vendored, unmodified). Arms are configured,
not coded, for anything the existing loss menu covers.

An arm is a directory under `checkpoints/distill/distill_out/<arm_name>/`
containing a `run_config.json`:

```json
{"run_id": "my_arm", "student": "segformer_b0", "seed": 42,
 "teacher_a": "teachers/segformer_b2_teacher",
 "teacher_b": "teachers/segformer_b5_teacher",
 "ensemble": "adaptive",      // null | "uniform" | "adaptive"
 "kd": "response",            // "response" | "cwd" | "bpkd" | "angular" | "dkd"
 "group": false,              // ITA-group reweighting
 "hard": false,               // hard-example focusing
 "boundary": false,           // boundary loss
 "rel_b": 0.6343, "alpha": 0.95,
 "lambda_cwd": 1.0, "lambda_angular": 1.0, "lambda_boundary": 4.0,
 "lambda_group": 0.5, "gamma_hard": 2.0}
```

The runner (`distill_segformer.py`) picks up any arm without a `DONE.json` and
trains it; the registry reports it as MISSING until then.

**Before adding a multi-teacher arm**, re-run `val_oracle.py` if the teacher set
changed. The oracle answers whether the teachers are complementary enough to be
worth combining, on validation, cheaply. `GATE_run_adaptive: false` means don't
bother.

**For a genuinely new KD loss**, add it to `kd_core.py` — but note that file is
vendored from the tested ORC suite, so changing it means the shipped Stage C
results and your new code are no longer produced by the same file. Prefer adding
a new module and importing it.

**Scoring** is automatic: `report.paired_contrast` against the B2→B0 reference
with a 0.01 non-inferiority margin, producing WIN / NON-INFERIOR / INFERIOR.

---

## 13. How to add a new analysis

Add a function to `scripts/unified_lib/report.py` taking a normalised per-image
frame:

```python
def my_metric(per_image: pd.DataFrame) -> pd.DataFrame:
    """What it measures, and why that is the right way to measure it."""
    ...
```

Then a cell in `scripts/61_generate_unified_notebook.py`:

```python
md(r"""## D10 · My analysis

Why this figure exists and how to read it. State the confound it controls for.
""")

code(r'''
if ALL:
    MYTABLE = pd.DataFrame([{"model": n, **R.my_metric(d)} for n, d in ALL.items()])
    display(MYTABLE.round(4))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ...
    save(fig, "D10_my_analysis"); plt.show()
''')
```

Add `"MYTABLE"` to the `WANTED` list and `FILENAME` map in the D9 save cell.

**Rules that keep the analysis honest:**

- Any interval or test must resample **subjects**, not images.
- Any cross-model comparison must be **paired**.
- Report median alongside mean for anything Dice-based.
- If your metric could be confounded by bruise size, condition on it or say so.

---

## 14. Known gaps and caveats

### Genuine gaps

1. **nnU-Net** — never run on the canonical 697/134 split. No weights, no
   results. Needs `nnunetv2` and ~8 GPU-hours.
2. **`x_dkd_b5_to_b0`** — a KD arm configured but never executed; the directory
   holds only `run_config.json`.
2b. **`yolo_sem_rgkd` — PENDING, deferred by decision on 2026-08-02.** The
   reliability gate on YOLO's offline pseudo-mask route. Fully implemented
   (`reliability_kd.build_gated_yolo_dataset`), smoke-tested, and **deliberately
   unregistered**: absent from `TEACHER_FOR`, `STAGE_H_FAMILIES`, `CONTROL_FOR`,
   `CONTRAST_FAMILY_H` and `registry.STAGE_H_FAMILIES`, so nothing schedules it.
   Its `loaders.FAMILY_SPEC` row, the notebook's YOLO training branch and its
   native-argmax scoring branch are all still in place and tested.

   *Why it is deferred:* it is the only Stage H arm needing Ultralytics, and the
   Stage E+H sweep was launched on a node where that install was the problem
   (§15 trap 14). Removing it makes the whole sweep dependency-free of
   Ultralytics.

   *What the study gives up until it runs:* **every confirmatory Stage H contrast
   tests the gate on an ONLINE loss only.** Nothing is claimed about whether the
   gate transfers to the offline pseudo-mask route, which is a mechanically
   different application of the same `r` and `g`. Holm corrects over three
   contrasts, not four. State this whenever Stage H results are reported.

   *To run it* — three lines in `reliability_kd`, plus the matching
   `registry.STAGE_H_FAMILIES` / `STAGE_H_KIND` entries, a `CONTRAST_FAMILY_H`
   row, and `COST_HOURS["yolo_sem_rgkd"] = 0.8`. See §7c.5. Cost: 3 runs,
   ~2.4 GPU-hours. Ultralytics must be installed first. Its control,
   `yolo_sem_distilled`, already ships at WEIGHTS tier, so no control run is
   needed.

Gaps 1 and 2 are reported by the registry and appear in `gaps.csv`. Neither is
back-filled.

**Gap 2b is not, and that is the one thing to be careful about.** An *unregistered*
family is invisible to the registry — it is not a MISSING row, it is no row at all,
which is precisely the silence §10 exists to prevent. The trade was made
deliberately (a dependency-free sweep beat a visible gap on the day), but it means
**this handbook entry is the only record that the arm is owed.** Any future
decision to leave an arm unregistered rather than MISSING must be written down
here in the same way, or it will simply be forgotten.

### Limitations that are not gaps

3. **Stage E ships no weights and no per-seed CSVs.** The numbers in §7.2 come
   from runs held outside this bundle, so the registry reports all Stage E runs
   MISSING and a training session regenerates them rather than skipping. Being
   regenerated as of 2026-07-29 (§7.2, §15 trap 13).
   The seed-reconciliation problem this entry used to describe — seed 0 winning
   for all four models, 8/8, a ~1 % coincidence pointing at a config difference
   rather than luck — is resolved by that regeneration, since all fifteen runs
   now share one config, one cut-selection path and one scoring path. Until it
   lands, §7.2's 3-seed means stay provisional.
   Related: a cross-seed aggregate table from the old sweep has Dice and
   complete-miss columns drawn from different seeds; regenerate it through
   `report.normalize()` rather than trusting it.
3b. **Stage F ships no weights and no per-image CSVs.** `fastscnn_distilled` and
   `lraspp_mobilenetv3_distilled` are defined and wired. The Fast-SCNN arm's
   previous-lineage numbers sit in `_work/new_outputs/` (null on Dice, worse on
   misses and fairness — §7b.5). The **LR-ASPP arm has now trained at all three
   seeds** (2026-07-30, §7b.7), but only its summary tables exist — no per-image
   CSV reached disk, so its contrast cannot be bootstrapped, its fairness cannot
   be computed, and none of its numbers can be re-derived here. Do not cite any
   distilled-vs-direct result until the per-image outputs are saved and the
   contrast has been through a paired subject-level bootstrap. See §7b.5–7b.7.
3c. **Stage H is trained; its per-image CSVs are in `FINAL_RESULT/`, its
   checkpoints are not in this bundle.** All 27 runs (9 families × 3 seeds)
   completed 2026-08-02 and every number in §7c.11–7c.15 is a measurement. The
   registry still reports those runs MISSING against a fresh `WORK_DIR`, for the
   same reason Stage E does (§15 trap 13): the weights live in the training
   session's work directory, not in `checkpoints/`.
   **One reconciliation caveat.** `FINAL_RESULT/efficient_test_per_seed.csv` holds
   only **8 rows** — seeds 1 and 2 of the four direct models — because that cell
   writes only what the session it ran in trained, and Stage E was completed across
   more than one session. Stage E's headline table is assembled from the per-image
   CSVs instead, which is why it covers all eight families.
   `stage_h_test_per_seed.csv` **is** complete at 27 rows.
   → *Do not compute a Stage E 3-seed mean from the per-seed file.* This is §7.2a's
   trap in a new costume: an aggregate that looks complete and covers half the grid.
4. **B5 seeds 42 and 2026 have results but no weights** — only the val-selected
   seed was promoted to `teachers/`. Their numbers are exact; they cannot be
   re-inferred.
5. **`yolo_sem_distilled__seed2` stopped at 31 epochs** where its siblings ran
   49–84, and its `best.pt` was never stripped of optimizer state (13.4 MB vs
   3.4 MB). It loads and scores correctly; it is why that model's seed spread is
   wide (std 0.065). Preserved as-is rather than silently re-run.
6. **Speed tables span two devices.** Stage A on a full A100, Stage E on an A100
   MIG 3g.40gb slice. Not comparable; re-time both on one device before
   publishing a latency claim.
7. **TopFormer's weights need a manual download** (Google Drive). Without them it
   trains from scratch and is labelled as such.
8. **28 test subjects** is a small denominator. Most fairness comparisons are
   not significant and should not be reported as if they were.

### Environment caveats

9. **transformers version.** Recent releases renamed SegFormer's internals
   (`encoder.block.N` → `stages.N.blocks`, `attention.self.query` →
   `attention.q_proj`, `mlp.dense1` → `mlp.fc1`, `decode_head.linear_c` →
   `linear_projections`). Stage A checkpoints will not load on a machine whose
   transformers is on the other side of that refactor. Pin to the training-time
   version to re-infer or re-benchmark them.
10. **matplotlib 3.10.9 on some Windows/conda setups** crashes the interpreter
    inside `ax.bar()` with a delay-load DLL failure (`0xc06d007f`). `ax.plot()`
    works in the same process; 3.11.1 is fine. Upgrade if Stage D dies on the
    first figure.

---

## 15. Traps we hit, and the guards against them

Each of these was a real mistake. Each now has an automated check.

**1. A same-named checkpoint set with the wrong lineage.**
`analysis/runs_v2/` in the source project has all fifteen Stage A run_ids, but
with `alpha=0.5`, fixed batch 8, and YOLO trained in the custom loop instead of
natively. Shipping it would produce a bundle that runs, skips, and quietly
reports numbers that disagree with the paper.
→ *Guard:* the build asserts on `alpha == 0.6` and the per-model batch, and on
the native Ultralytics weight path. Names are not evidence; configs are.

**2. A baseline set from the wrong split.**
`EXTRA/smp_baselines.zip` holds U-Net and DeepLabV3+ weights trained on 693/138
at seed 42 — not the canonical 697/134 three-seed set.
→ *Guard:* the build reconciles each run's own `test_per_image.csv` against the
canonical per-seed CSV and refuses anything that does not match.

**3. The best seed is not the same for every model.**
It is 0 for the three SegFormers and `yolo_sem_distilled`, but **2** for
`yolo_sem_direct`. Pairing a model's weights with another seed's results shows
per-image disagreements up to 0.49 Dice and looks exactly like broken inference.
→ *Guard:* `report.best_seeds()` reads the mapping off the selection step's own
filenames, so it cannot drift.

**4. The same model under two names.**
`segformer_b5_teacher` is the val-winning B5 seed (123) promoted into
`teachers/`, matching its per-image Dice bit for bit. Listing both showed one
result twice.
→ *Guard:* `load_stage_c` fingerprints the per-image Dice vector, collapses exact
duplicates, and prints the alias.

**5. A silent ImageNet download.**
`segmentation_models_pytorch` fetches ResNet-50 the moment a model is
constructed — even when a checkpoint is about to overwrite every weight. Pure
latency at best, a hard failure offline at worst.
→ *Guard:* `loaders.build_for_load` passes `encoder_weights=None` on the load
path. Training still uses ImageNet init.

**6. An activation substitution that changes nothing visible.**
`nn.Hardsigmoid` (slope 1/6) for StrideFormer's `HSigmoid(slope=0.2)`. No
parameters, so every state-dict key still matches — while every SE gate in the
backbone computes something different.
→ *Guard:* the shim implements `HSigmoid` with honoured slope/offset, and
`verify_checkpoint_match` proves the load.

**7. Autograd graph retained during evaluation.**
`evaluate_at_cut` runs its forward pass without `no_grad`. Invisible on a 40 GB
A100; on CPU, SegFormer-B2's decode head tries to allocate 1.2 GB in one `cat`
and the interpreter dies.
→ *Guard:* `loaders.score_segformer` wraps the call in `torch.inference_mode()`
and uses a CPU-safe batch. Fixed at the call site, not by editing the verbatim
module.

**8. Bootstrap by DataFrame concatenation.**
2000 `pd.concat` calls per model × 16 models = 32k allocations, enough to kill a
Colab kernel.
→ *Guard:* resample index arrays instead. Same draws, two orders of magnitude
cheaper.

**9. Eager dependency installs on an offline node.**
`_need("ultralytics")` fired whenever Stage A was selected, even when Stage A was
only reading cached CSVs — and a failed pip install halted the notebook.
→ *Guard:* optional installs are gated on whether a model will actually be
*built*, and are non-fatal.
→ **This guard was too weak and the trap recurred on 2026-08-02. See trap 14.**

**10. Demanding CUDA for work that will not happen.**
`train_missing` checked for a GPU before checking whether any MISSING run had a
training path, so a stage whose only gap was nnU-Net raised on a CPU box.
→ *Guard:* filter to trainable kinds first, report the rest, and only then
require CUDA.

**11. Writing a cell instead of following the tested pattern.**
The Stage E threshold cell called `cache_logits` expecting two return values (it
returns three) and passed raw arrays to `select_cut` (which takes the sweep
table). It only executes after a real GPU training run, so no CPU smoke test
caught it.
→ *Lesson:* when a tested cell for the same job exists, copy its call sequence.

**12. Reading a 0-of-185 result as a zero rate.**
LR-ASPP's seed-0 run missed no images, and this file reported it as "zero complete
misses — matching only the B2 teacher and B0-distilled". Seeds 1 and 2 missed 1 and
2 images. Nothing was wrong with the seed-0 number; the error was quoting a count
of zero from one run as if it were a rate.
→ *Lesson:* a zero count on n=185 has a one-sided 95 % bound of ≈ 1.6 %, which
spans the entire Stage E field. Never claim a zero rate from a single seed, and
state the bound when reporting one at all.

**13. Documented results read as shipped artefacts.**
§7.2 has carried a full Stage E results table for some time, which makes it look
as though those models are trained and present. They are not: there is no
`checkpoints/efficient/`, no `results/efficient/`, and `WORK_DIR/runs/` starts
empty. The registry is right and the reader's expectation is wrong — a training
session correctly retrains all fifteen runs, and it is easy to misread that as
the pipeline needlessly redoing finished work.
→ *Guard:* the tier table (§10) is the authority on what exists, not a results
section. If you want to know whether something will train, run §3 of the notebook
alone and read the plan. Two ways a genuinely-finished run still retrains, both
worth checking before assuming a bug: `WORK_DIR` moved since the last session (a
Colab VM's local scratch does not survive), or `best.pt` exists without
`operating_point.json`, which counts as MISSING because a logit-thresholded
model's test score is undefined without its val-fitted cut.

**14. Trap 9 again — a stage-level gate is not a "will it be built" gate.**
Stage H added `if "H" in RUN_STAGES and _BUILDS_MODELS: _need("ultralytics")`.
That reads like trap 9's guard and is not: `RUN_STAGES` says which *stages* were
selected, not whether any run in them will actually be built. A session with
`RUN_STAGES="EH"` had never installed anything before, because neither "A" nor
"B" was present; adding "H" fired a pip install on an offline ORC node, which sat
in retry backoff for minutes on a cell that had always returned instantly. It
presented as "the notebook is hung", and cost a day.

Three things went wrong at once, and each has its own guard now:

→ *Guard A — install where the registry knows.* The optional installs
(`ultralytics`, `segmentation_models_pytorch`) moved out of §1 into
`train_missing` and the Stage E/H wiring cells, which have `reg.to_train()` in
hand. §1 prints `optional deps ... deferred` and installs nothing. **§1 must never
regain a `_need` call for an optional package** — it cannot know, and a gate that
guesses is trap 9 wearing a new coat.

→ *Guard B — optional installs fail fast.* `_need(..., required=False)` now passes
`--retries 1 --timeout 10`. An unreachable optional package costs ~10 s, not
minutes. Required packages keep pip's defaults, where the retry is worth it.

→ *Guard C — `BRUISE_NO_PIP=1`.* Set it on any node you know is offline and every
missing package becomes a one-line named error instead of a stall. **This is the
first thing to reach for when a dependency cell appears to hang**, because it
converts "stuck" into "tell me what to install".

*Lesson beyond the mechanics:* a cell whose runtime changed from 0 s to minutes is
a regression even when it eventually succeeds and even when it is non-fatal. On a
shared cluster nobody waits long enough to find out it was non-fatal.

**15. A patched notebook is not the notebook that is running.**
The 2026-08-02 overlay was extracted correctly — `grep -c` on disk confirmed the
new marker — and the kernel kept executing the old cells anyway. Jupyter reads the
`.ipynb` **once, when the tab is opened**; replacing the file underneath a live tab
changes nothing, and autosave can write the stale in-browser copy back over the
patch. Half a day was spent debugging a cell that had already been fixed.
→ *Guard:* after applying any patch zip, **File → Close and Halt, then reopen** —
restarting the kernel is not enough, because the kernel is not where the cells
live. Verify against a marker string that only the new version contains
(`grep -c BRUISE_NO_PIP bruise_unified.ipynb` → 4 for the Stage H overlay), and
verify it again *from a cell inside the running notebook* if the disk and the
behaviour disagree.

**16. A patch zip whose paths do not match the established convention.**
`bruisekit_distill_patch.zip`, `bruisekit_lraspp_distill_patch.zip` and
`bruisekit_significance_patch.zip` all store paths **relative to the bundle**, with
sources under `_source/`, so they apply with `cd BRUISE_UNIFIED && unzip -o`. The
first Stage H overlay stored `BRUISE_UNIFIED/`-prefixed paths and needed
`unzip -d ..`. It worked, but it broke muscle memory built by three prior patches,
and a plain `unzip -o` with those paths silently produces a nested
`BRUISE_UNIFIED/BRUISE_UNIFIED/` that *looks* applied and is not.
→ *Guard:* `scripts/63_zip_rgkd_overlay.py` writes bundle-relative paths and
prints the extraction rule when it builds. Any future patch zip does the same. The
recipient should never have to be told a `-d` flag.

---

## 16. Build and regeneration

The bundle, the notebook and the vendored architectures are **all build
artefacts**. Never hand-edit them; edit the source and rebuild.

```bash
python scripts/vendor_efficient_nets.py        # refresh vendored architectures
python scripts/60_build_unified_bundle.py      # assemble + verify the folder
python scripts/61_generate_unified_notebook.py # emit the notebook
python scripts/62_zip_unified_bundle.py        # package it
```

A fifth script packages a **patch overlay** — the notebook, the changed modules
and their sources — for dropping onto a bundle that already exists elsewhere:

```bash
python scripts/63_zip_rgkd_overlay.py       # -> BRUISE_RGKD_OVERLAY.zip, ~0.2 MB
```

**Every patch zip stores paths relative to the bundle**, sources under `_source/`,
so it always applies the same way and the recipient never needs a `-d` flag
(§15 trap 16):

```bash
cd /path/to/BRUISE_UNIFIED
unzip -o BRUISE_RGKD_OVERLAY.zip
```

**Then close and reopen the notebook — File → Close and Halt, not a kernel
restart.** Jupyter holds the `.ipynb` in the browser, so a live tab keeps running
the old cells and autosave can overwrite the patch (§15 trap 15). Verify with a
marker string only the new version has, e.g.
`grep -c BRUISE_NO_PIP bruise_unified.ipynb`.

All idempotent. Sources of truth:

| Artefact | Source |
|---|---|
| `bruisekit/{data,models,losses,metrics,engine,sweep,evaluate,postopt,yolo_native,smp_models,nnunet_native}.py` | extracted verbatim from `bruise_colab_baselines.ipynb` |
| `bruisekit/{paths,registry,loaders,report,efficient_models,weights,mmcv_shim,distill_efficient,significance,reliability_kd}.py` | `scripts/unified_lib/` |
| `bruisekit/vendor/*.py` | upstream repos, via `vendor_efficient_nets.py` |
| `bruisekit/kd/*.py` | the ORC distillation suite, copied unmodified |
| `bruise_unified.ipynb` | `scripts/61_generate_unified_notebook.py` |

The build **verifies rather than assumes**: data dedup is hash-checked across two
sources, checkpoint lineage is asserted on config fields, baselines are
reconciled against the canonical per-seed CSV, and split leakage is re-checked.

---

## 17. File map

```
BRUISE_UNIFIED/
├── bruise_unified.ipynb          78 cells, seven stages (A/B/C/E/H/D/G)
├── README.md                     how to run
├── PROJECT_HANDBOOK.md           this file
├── requirements.txt
│
├── bruisekit/                    the library — see §9
│   ├── distill_efficient.py      Stage F: DeepLabV3+ → mobile-student KD shim (§7b)
│   └── reliability_kd.py         Stage H: reliability-gated KD + B2 teacher axis (§7c)
│
├── data/                         1016 native-res images + masks (2.6 GB)
│   ├── train/{images,masks}      831  (697 train + 134 val)
│   └── test/{images,masks}       185
├── manifests/                    train/val/test.csv + kd_train/kd_test.csv
├── splits/  ita_labels/          split definitions, ITA per image/subject
├── interlabeler_agreement_640.csv   human-vs-human Dice (the ceiling)
│
├── pretrained_weights/
│   ├── segformer_mit_b0|b2|b5    HF config + weights (offline)
│   ├── yolo26n-sem.pt            Ultralytics Cityscapes checkpoint
│   └── efficient/                MobileNetV3, StrideFormer, TopFormer backbones
│
├── checkpoints/
│   ├── final/                    15 Stage A runs
│   ├── baselines/                6 Stage B runs
│   ├── distill/
│   │   ├── teachers/             B5, B2, B0-distilled + reference CSVs
│   │   └── distill_out/          10 completed arms + alpha search + oracle
│   └── efficient/                Stage E/F — ABSENT until you train (§15 trap 13)
│
└── results/
    ├── final/                    FINAL_RESULTS, per-seed, best-seed per-image,
    │                             fairness, benchmark_640
    ├── baselines/                BASELINES_RESULTS, per-seed, fairness
    ├── analysis_native/          15 CSVs + 29 figures (the reported analysis)
    └── distill/                  B5 seed sweep, teacher reference CSVs
```

Work directory (created at runtime, never shipped):

```
WORK_DIR/
├── cache640/            640×640 PNG cache, rebuilt deterministically (~1 GB)
├── runs/                new training runs — never overwrites checkpoints/
├── teacher_calibration/ Stage F fitted temperatures, one JSON per teacher seed
└── outputs/             this session's tables and figures
```

**`WORK_DIR` is where every Stage E and Stage F result you produce lives.** If it
is not on persistent storage, the runs die with the session and the next one
retrains from scratch.

---

## Appendix — quick reference

### Before any session on a cluster node — the five-line checklist

Every item below is a trap that has already cost a day. In order:

1. **Applied a patch zip?** `cd BRUISE_UNIFIED && unzip -o <patch>.zip` (never a
   `-d` flag, §15 trap 16), then **File → Close and Halt and reopen the notebook**
   — a kernel restart does not reload it (§15 trap 15).
2. **Offline node?** `export BRUISE_NO_PIP=1` before starting the kernel, or as a
   first cell `import os; os.environ["BRUISE_NO_PIP"] = "1"`. Missing packages
   then name themselves in seconds instead of stalling in pip retries
   (§15 trap 14).
3. **Right interpreter?** `import sys; print(sys.executable)` from a cell. The
   kernel is often not the shell's Python, and a package installed into the wrong
   one is indistinguishable from a package that is missing.
4. **Right node?** `torch.cuda.is_available()`. Training refuses on CPU, but only
   after §1–§4 have run, so check first rather than 10 minutes in.
5. **`WORK_DIR` on persistent storage?** Every new checkpoint, threshold and
   per-image CSV lands there. If scratch is purged mid-sweep, the runs that lost
   their `DONE.json` retrain from scratch.

**A dependency cell that takes minutes is a bug, not slowness.** Interrupt it, do
step 2, and re-run — do not wait it out.

**Run everything from cache, no GPU:** defaults. ~30 seconds.

**Re-run the mobile baselines (all three seeds; finished runs skip via `DONE.json`):**
```python
RUN_STAGES = "ABDE"; ALLOW_TRAINING = True; EFFICIENT_SEEDS = (0, 1, 2)
```

**Run Stage H (reliability-gated KD) plus its controls:**
```python
RUN_STAGES = "ABCDEH"; ALLOW_TRAINING = True
EFFICIENT_SEEDS = (0, 1, 2); RGKD_SEEDS = None
```
51 runs, ~40 GPU-hours: Stage E/F 24, Stage H 27. Needs
`segmentation-models-pytorch` (the DeepLabV3+ teacher); **does not need
Ultralytics**, because no YOLO run is registered (§14 gap 2b). Keep A/B/C
selected — they train nothing and cost seconds, but without them Stage H's
B0 contrast and the annotation ceiling both drop out silently.

**Train only a Stage F distilled arm**, skipping the four direct baselines
(same `train_run`, so the recipe is unchanged; already-finished runs are still
skipped when you go back to `train_missing`):
```python
DE.train_arm(env, "fastscnn_distilled", EFFICIENT_SEEDS, CFG, MAN640)
DE.train_arm(env, "lraspp_mobilenetv3_distilled", EFFICIENT_SEEDS, CFG, MAN640)
```
You still need the direct counterpart (`fastscnn` / `lraspp_mobilenetv3`) at the
same seeds before a contrast means anything. Note `train_arm` trains but does not
threshold-sweep or score — the notebook's Stage E threshold cell only picks up
runs returned by `train_missing`, so after a standalone `train_arm` either re-run
the notebook Stage E cells (the registry will find the runs) or use
`train_missing`, which skips the finished runs and sweeps the rest.

**Re-derive every number from the checkpoints:**
```python
RECOMPUTE_FROM_WEIGHTS = True     # agrees with cache to <2e-4 mean Dice
```

**Retrain one specific run:**
```python
FORCE_RETRAIN = ["topformer_tiny__seed0"]
```

**Check what would happen without doing it:** run §3 alone; the plan prints per
stage, with per-run cost estimates and the total GPU-hours.
