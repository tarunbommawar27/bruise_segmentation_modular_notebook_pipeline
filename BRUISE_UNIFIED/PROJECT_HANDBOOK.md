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
7d. [Stage Y — YOLO26-large](#7d-stage-y--yolo26-large)
7e. [Stage M — multi-teacher routed distillation](#7e-stage-m--multi-teacher-routed-distillation)
7f. [Stage N — foundation encoders **(IN PROGRESS)**](#7f-stage-n--foundation-encoders-vs-imagenet-encoders)
7g. [Stage P — lesion-size-stratified miss containment **(BUILT, NOT YET RUN ON ORC)**](#7g-stage-p--lesion-size-stratified-miss-containment)
7h. [Stage N2 — the grid control, and *dermatology* pretraining **(RUN — gate CLOSED; §7f.8 superseded)**](#7h-stage-n2--the-grid-control-and-dermatology-pretraining)
7i. [Stage N3 — the annotation ceiling on a third axis **(RUN 2026-08-10 — `dinov2_ft` 0.7902 test, 0 misses; does not separate from B2/B0; §7i.7)**](#7i-stage-n3--the-annotation-ceiling-on-a-third-axis)
7j. [Stage N4 — mask vs caption pretraining **(RUN 2026-08-11 — `medsam − sam` = +0.0037, CI spans zero; but MedSAM is the best single teacher on VAL; §7j.3, §7j.4)**](#7j-stage-n4--mask-pretraining-vs-caption-pretraining-run-2026-08-11)
7k. [Stage O — miss taxonomy, distilled-arm fairness, ITA-group routing **(RUN 2026-08-12 — two analyses delivered; the routing gate is CLOSED on both schemes; §7k.5)**](#7k-stage-o--the-miss-taxonomy-distilled-arm-fairness-and-ita-group-routing)
7l. [The re-inference sweep — the pipeline verified against itself **(RUN 2026-08-12)**](#7l-the-re-inference-sweep--the-pipeline-verified-against-itself)
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
18. [Planned next work — for discussion, not yet implemented](#18-planned-next-work--for-discussion-not-yet-implemented)

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
- **Report complete misses in THREE columns, not one** — see below.

### Complete miss is two failures, and §7k splits them

`complete_miss_rate` is `dice == 0`, and that is the **union** of *the model
predicted nothing* (`pred_positive_pixels == 0`) and *the model outlined the wrong
region with zero overlap*. **They differ in 28 of 35 models.** Field-wide,
excluding Fast-SCNN, 87 complete misses split **60 blank / 27 wrong-place** — so
roughly one miss in three is a confident wrong answer, not a silence. A
wrong-place error is the worse clinical failure: an empty mask shows nothing and
invites a second look; a confident outline in the wrong location is an assertion.

`itakd.miss_taxonomy` produces all three, per model, per ITA group and per size
decile, with `wrong_place` **derived** as `zero_dice − empty_pred` so they cannot
fail to add up. Lead with zero Dice — it is the union and therefore the
conservative number — and quote the split beside it. Full treatment in §7k.1;
this supersedes §7.2a's two-column framing without contradicting it.

### And two numbers that change how every fairness table reads (§7k.2, §7k.3)

**Lesion size is confounded with skin tone in this test set.** 59 % of
Light (II-III) photographs contain a small bruise against 33 % of Dark (VI) —
~1.8×. Part of every unconditioned "worse on light skin" figure is a size effect
wearing a skin-tone label. Report the gap marginally *and* within the small
stratum.

**Small bruises are not uniformly harder, and capacity does not fix them.**
`lraspp_mobilenetv3` (3.22 M) has the field's best smallest-decile recall at
0.863, above `segformer_b5_teacher` (85 M) at 0.828. Recall is flat-to-noisy
across size for most architectures; several are worst on the *largest* decile.
YOLO is the exception and the reason it carries 12 misses.

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

### The current lineage — `FINAL_RESULT/RESULT_AUGUST_08/`

**As of 2026-08-05 that directory is the single source for every reported number**,
Stages A through Y, all scored in one session through one `report.normalize()`
path. `headline_all.csv` carries all 24 models; `headline_stage_*.csv` are its
per-stage slices; `per_image_*.csv` are the tables every contrast is recomputed
from. Complete misses everywhere in it are `dice == 0`, the definition §7.2a
settles on and the one to publish.

Those are **val-selected best seed**, not the 3-seed means above. Both are correct
and they are not interchangeable — §1's warning about mixing the two aggregations
applies to this pair specifically. Stage A on the current lineage:

| Variant | mean Dice | median | IQR | misses |
|---|---|---|---|---|
| segformer_b2_teacher | 0.7692 | 0.8192 | 0.154 | 0 (0.00 %) |
| segformer_b0_distilled | 0.7680 | 0.8167 | 0.168 | 0 (0.00 %) |
| segformer_b0_direct | 0.7663 | 0.8129 | 0.179 | 1 (0.54 %) |
| yolo_sem_distilled | 0.7261 | 0.8012 | 0.240 | 5 (2.70 %) |
| yolo_sem_direct | 0.7021 | 0.8061 | 0.261 | 12 (6.49 %) |

The IQR column is the one that separates them and mean Dice does not: the two
YOLO arms carry roughly 1.5× the SegFormers' spread (0.240–0.261 against
0.154–0.179) at a median within 0.02 of theirs.
They are not worse at outlining bruises, they are less reliable about finding
them — which is the same story the miss column tells, in a metric that does not
throw away 179 of the 185 images to say it.

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

### Speed (full A100 via Colab, 640 tensor → mask on GPU)

> ## ⚠️ THREE OF THESE FIVE ROWS ARE WRONG. Read §7.3a.
>
> The shipped `benchmark_640.csv` rows were measured in a process that had
> already run the whole notebook. Re-measured in a **fresh kernel** — same
> machine, same weights, same library versions, same TF32 flags — all three
> SegFormers come out **~1.72× faster**. The two YOLO rows are correct to 1.3 %.
>
> The corrected column below is the one to quote. The superseded column is kept
> only so that a number found in an old figure or slide can be recognised.

| Model | ~~superseded~~ | **corrected (fresh process)** | params M |
|---|---|---|---|
| segformer_b2_teacher | ~~33.67 ms — 29.7 FPS~~ | **19.74 ms — 50.7 FPS** | 27.35 |
| segformer_b0_direct | ~~16.68 — 59.9~~ | **9.45 — 105.8** | 3.71 |
| segformer_b0_distilled | ~~16.64 — 60.1~~ | **9.58 — 104.4** | 3.71 |
| yolo_sem_direct | 8.18 — 122.2 | 8.02 — 124.6 | 1.63 |
| yolo_sem_distilled | 8.22 — 121.7 | 8.07 — 123.9 | 1.63 |

**What this changes.** YOLO's speed advantage over SegFormer-B0 was 2× on the old
numbers and is **18 %** on the corrected ones (8.02 vs 9.45 ms). Any argument
resting on YOLO being dramatically faster is built on the artefact. Accuracy is
unaffected.

The `peak activation MB` column has been dropped: it was never activation memory
(§7.3b). Absolute milliseconds remain a property of the machine — §7.3 is the
single-machine table to use for cross-model comparison.

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

| Variant | mean Dice | median | miss % (blank preds) |
|---|---|---|---|
| unet_r50 | 0.7526 ± 0.0154 | 0.826 | 3.24 ± 0.54 |
| deeplabv3plus_r50 | 0.7453 ± 0.0182 | 0.814 | 1.80 ± 0.62 |

3 seeds each = 6 runs, all shipped.

> **⚠️ The miss column above is the wrong definition, and it understates.** Those
> percentages come from `smp_baselines_test_per_seed.csv`, which counts images
> where the model predicted **nothing at all** (`pred_positive_pixels == 0`).
> §7.2a settled that this project publishes `dice == 0` instead — no overlap with
> the ground truth, which also catches a model firing confidently on the wrong
> region. That is still a missed injury to a clinician. On the current lineage:
>
> | Variant | mean Dice | median | **misses (`dice == 0`)** |
> |---|---|---|---|
> | deeplabv3plus_r50 | 0.7584 | 0.8183 | **5 (2.70 %)** |
> | unet_r50 | 0.7570 | 0.8329 | **7 (3.78 %)** |
>
> DeepLabV3+ contains U-Net's misses strictly (§7b.1) on both definitions, so the
> teacher argument is unaffected. What changes is the absolute rate: quote 2.70 %
> and 3.78 %, not 1.80 % and 3.24 %.

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

### 7.2 Results (185 test images)

> **✅ LANDED 2026-08-05. The regeneration that began 2026-07-29 is complete and
> `FINAL_RESULT/RESULT_AUGUST_08/` supersedes everything below.** All six Stage
> E/F families plus the four Stage H arms come from one config, one cut-selection
> path and one scoring path, so the seed-0 provenance worry in §7.2a is resolved
> by construction. Complete misses are `dice == 0` throughout.
>
> **Current lineage, val-selected best seed** (`headline_stage_e.csv`):
>
> | Model | mean Dice | median | IQR | misses |
> |---|---|---|---|---|
> | lraspp_mobilenetv3_distilled | **0.7200** | 0.7836 | 0.228 | 3 (1.62 %) |
> | lraspp_mobilenetv3 | 0.6982 | 0.7607 | 0.240 | 2 (1.08 %) |
> | topformer_tiny | 0.6918 | 0.7418 | 0.219 | **1 (0.54 %)** |
> | ppmobileseg_tiny_distilled | 0.6891 | 0.7551 | 0.226 | 4 (2.16 %) |
> | topformer_tiny_distilled | 0.6862 | 0.7503 | 0.258 | **1 (0.54 %)** |
> | ppmobileseg_tiny | 0.6568 | 0.7158 | 0.259 | 4 (2.16 %) |
> | fastscnn_distilled | 0.6161 | 0.6847 | 0.341 | 8 (4.32 %) |
> | fastscnn | 0.6053 | 0.6866 | 0.344 | 13 (7.03 %) |
>
> **Three things changed against the previous-lineage table below.**
>
> 1. **LR-ASPP's lead narrowed and its miss rate doubled.** 0.7093 → 0.6982 mean,
>    0.54 % → 1.08 % misses. The val-selected seed is now seed 2, the weakest of
>    its three. That is cut-selection working as specified — selection is on val,
>    and val did not pick the seed test would have — not a regression.
> 2. **TopFormer now has the lowest miss rate of the four**, 1 of 185, against
>    LR-ASPP's 2 and its own previous 0.90 %. On Dice it is still second. §7.4's
>    reading that "precision and recall separate the field more cleanly than Dice"
>    now extends to misses: the Dice ranking and the miss ranking disagree, and
>    §1 says the miss ranking is the one that decides.
> 3. **Distillation helps every mobile student except on misses.** All four
>    `*_distilled` arms gain Dice over their direct baseline (+0.022 LR-ASPP,
>    +0.032 PP-MobileSeg, +0.011 Fast-SCNN) — and all four gains are inside the
>    ±0.05 noise floor §1 sets. Fast-SCNN's miss count 13 → 8 is the one movement
>    large enough to survive it.
>
> Everything from here to §7.3 is the **previous lineage**, kept because figures
> and slides quoting 0.7093 exist and a reader has to be able to recognise them.

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

### 7.3 Speed — read 7.3a before quoting any number here

**THE TABLE TO QUOTE.** ORC **A100 MIG 3g.40gb**, fp32, fresh process, seed 0,
2026-08-05, with Stage Y and both SegFormers folded in. One machine, one session,
one recipe, one precision — every row is comparable to every other row, and to
nothing outside this table.

| Model | params M | median ms | FPS | median Dice | misses |
|---|---|---|---|---|---|
| lraspp_mobilenetv3 | 3.22 | 4.98 | **200.9** | 0.761 | 2 |
| yolo_sem_direct | 1.63 | 5.70 | 175.3 | 0.806 | 12 |
| topformer_tiny | 1.37 | 6.42 | 155.8 | 0.742 | 1 |
| segformer_b0_direct | 3.71 | 8.99 | 111.3 | 0.813 | 1 |
| deeplabv3plus_r50 | 26.68 | 10.47 | 95.5 | 0.818 | 5 |
| ppmobileseg_tiny | 1.45 | 11.16 | 89.6 | 0.716 | 4 |
| **yolo_sem_l_direct** | **17.86** | **11.17** | **89.5** | **0.810** | **13** |
| unet_r50 | 32.52 | 11.83 | 84.5 | 0.833 | 7 |
| segformer_b2_teacher | 27.35 | 29.41 | 34.0 | 0.819 | 0 |

Within ±2–3 % of the per-stage fp32 CSVs it supersedes
(`inference/benchmark_640_cuda_fp32_orc-mig.csv`: LR-ASPP 198.2, YOLO-N 171.8,
TopFormer 153.7, DeepLab 95.8, PP-MobileSeg 87.6, U-Net 84.6) — that spread is
the run-to-run drift these models show anyway, not a discrepancy.

**Two readings this table supports and one it does not.**

- **Parameter count does not predict latency here.** PP-MobileSeg at 1.45 M is
  slower than DeepLabV3+ at 26.7 M, and YOLO-L at 17.9 M matches PP-MobileSeg to
  0.1 %. §7.3b explains why: at batch 1 this benchmark is dispatch-bound.
- **The Pareto front is LR-ASPP → SegFormer-B0 → SegFormer-B2**, and everything
  else is dominated. U-Net's +0.014 median Dice over DeepLabV3+ costs 12 % of the
  throughput and 7 misses against 5.
- **It does not support "YOLO is the fast one".** YOLO-N is 13 % *slower* than
  LR-ASPP (5.70 vs 4.98 ms) and carries 12 complete misses against LR-ASPP's 2.
  The corrected Colab table (§7.3a) already killed the 2×-over-SegFormer claim;
  this table takes the remaining one, by putting YOLO-N second on speed and last
  on misses at the same time.

The `peak activation MB` column has been dropped: it never measured activations
(§7.3b).

**Do not merge this with the Stage A table in §4.** That one is a Colab full
A100, and three of its five rows are wrong (§7.3a).

### 7.3a ⚠️ The shipped `benchmark_640.csv` is inflated in three of five rows

Discovered 2026-08-04. Same notebook, same cell, same A100-SXM4-40GB, same
weights, same `torch 2.11.0+cu128`, same `transformers 5.13.1`, same TF32 flags.
The one difference: a **fresh kernel**, running only setup and then the speed
cell, instead of running the whole notebook first.

| model | shipped | fresh kernel | |
|---|---|---|---|
| segformer_b2_teacher | 33.649 ms — 29.7 FPS | **19.738 — 50.7** | **1.70× off** |
| segformer_b0_direct | 16.553 — 60.4 | **9.452 — 105.8** | **1.75× off** |
| segformer_b0_distilled | 16.762 — 59.7 | **9.578 — 104.4** | **1.75× off** |
| yolo_sem_direct | 8.130 — 123.0 | 8.024 — 124.6 | 1.01× |
| yolo_sem_distilled | 8.084 — 123.7 | 8.072 — 123.9 | 1.00× |

**All three `transformers` models inflated ~1.72×. Both convolutional models
untouched, to within 1.3%.**

**Why "it reproduced" was not evidence.** 60 FPS was measured in July and again in
August and agreed to **0.8%**. That read as reproducibility and was trusted on
that basis for weeks. It was not: *both runs executed the full notebook first*, so
both inherited the same process state. A number can be perfectly repeatable and
still be an artifact — repeatability only tests what you varied, and the process
state was never varied until it was tested directly. This is trap 14 (§15).

**What the number is actually a property of.** Not the model: *(model, host,
library versions, and what the process did beforehand)*, and only the first
travels with the bundle. Four independent measurements of SegFormer-B0 in clean
processes agree — ORC MIG 9.294, Colab speed-harness 9.870, Colab fresh kernel
9.452, the 51-run registry sweep 8.740 — all inside the ±5–6 % run-to-run drift
these models show anyway. **The 16.55 ms row is the sole outlier**, and the
"cross-machine SegFormer anomaly" chased through earlier revisions of §14 caveat 6
traces back to it and to nothing else.

**Why only the transformers.** At batch 1 with `cuda.synchronize()` on both sides
this benchmark is dominated by kernel dispatch, not FLOPs. SegFormer's cost sits
in GEMM and attention; the CNNs' sits in cuDNN convolutions. The artifact hits the
first and not the second, which points at a GEMM/attention fast path being lost.
**Not bisected to a specific call** — a lead, not a finding.

**The consequence that matters.** Corrected, YOLO is 8.02 ms against
SegFormer-B0's 9.45 — an **18 % gap, not the 2× the old figure showed**. Any claim
resting on YOLO being dramatically faster than SegFormer-B0 is built on the
artifact. Accuracy is untouched; this was only ever a timing error.

**The rule.** Time in a fresh process, on one machine, and record both.
`inference.speed_table` now writes `fresh_process` and `process_prior_peak_MB` on
every row and warns before starting when the process is dirty. A row without those
columns predates the check and cannot be verified.

### 7.3b What this benchmark actually measures

**Latency tracks module count, not parameter count.**

| model | params | leaf modules | median ms | ms per module |
|---|---|---|---|---|
| fastscnn | 1.14 M | 124 | 4.88 | 0.039 |
| lraspp_mobilenetv3 | 3.22 M | 171 | 6.75 | 0.040 |
| topformer_tiny | 1.37 M | 168 | 8.71 | 0.052 |
| ppmobileseg_tiny | 1.45 M | 271 | 14.81 | 0.055 |

(Colab A100, one session, so the four are mutually comparable.)

PP-MobileSeg has **2.2× the modules of Fast-SCNN at nearly the same parameter
count**, and is 3× slower. LR-ASPP has ~3× PP-MobileSeg's parameters and is 2.2×
faster. On ORC, PP-MobileSeg (1.45 M) is slower than DeepLabV3+/R50 (26.7 M).
Parameter count does not predict latency in this regime.

The two pure CNNs land on an identical 0.039–0.040 ms per module; the two
attention hybrids cost ~35 % more per module, because each attention block expands
into several kernels that are not separate `nn.Module`s. PP-MobileSeg's census —
98 `Conv2d`, 79 `BatchNorm2d`, 16 `SqueezeAxialPositionalEmbedding` — shows both
effects: 16 axial-attention blocks, and 79 BatchNorms that PyTorch eager mode does
**not** fold into the preceding convolution, so each is an extra launch.

PP-MobileSeg was designed for mobile ARM/NPU deployment, where the cost model is
FLOPs and the graph is fused ahead of time. This benchmark is eager-mode dispatch
on an A100 at batch 1 — a regime it was never optimised for, and one where "small"
and "fast" come apart entirely.

**Two things owed before any efficiency claim** (§18.5 items 1–2): a `torch.compile`
run, which may collapse those launches and would mean the current table measures
our inference setup rather than the architecture; and a batch-size sweep, since at
batch 8–16 the GPU becomes the bottleneck and the ranking may invert. The
`peak activation MB` column of earlier revisions is gone because
`reset_peak_memory_stats` was called *after* the 185 test images were staged, so
every row silently carried 909 MB of images — which is why all four mobile nets
read ~1000 MB regardless of size. `peak_incremental_MB` replaces it.

### 7.3c Precision: `autocast` makes these models slower

Confirmed on two machines across 21 model-pairs with **zero exceptions**: at batch
1, fp16 autocast is 1.27–1.44× **slower** than fp32 for every model except the two
big ResNets (DeepLabV3+ 1.39× faster, U-Net 1.16× faster). Autocast adds cast
kernels to a benchmark that is already dispatch-bound, and only the two
compute-bound models win enough from tensor cores to pay for them.

`speed_table(precision="fp16")` exists to *attribute* a gap, never to report one.
`check_single_machine` raises on a table that mixes precisions, for the same reason
it raises on one that mixes devices.

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

## 7d. Stage Y — YOLO26-large

**The question.** `yolo26n` is the fastest model in the study (8.02 ms fresh-process,
1.63 M params) and has the **worst complete-miss rate of the reporting models** —
7.57 % ± 1.08 % across three seeds by native argmax, against 0.0–0.5 % for the
SegFormers. §8 is explicit that miss containment, not Dice, is the endpoint this
study is judged on. So the question the nano arms cannot answer is whether that
miss rate is a property of *YOLO* or a property of *1.6 M parameters*.

Stage Y isolates the variable: **the identical native Ultralytics recipe on the
large backbone**. Same mosaic, EMA, letterbox, LR schedule, `close_mosaic`, seed
handling — `yolo26l-sem.pt` instead of `yolo26n-sem.pt`, nothing else.

**Scored by native argmax only.** The custom `/255` path exists for Stage A
because that stage needed a SegFormer-comparable geometry; it is not re-derived
here. Argmax is parameter-free, so a Stage Y run needs `best.pt` and nothing
else — no val-fitted cut, no `operating_point.json`, nothing that can be missing.

**Its own stage letter.** Stage A is quoted as "the five headline models" here and
in the paper. A table that silently grows rows is indistinguishable from a bug —
the same reasoning that gave Stage H its letter (§7c).

**One arm.** `STAGE_Y_FAMILIES = ("yolo_sem_l_direct",)`. A distilled large arm has
a `FAMILY_SPEC` and a cost estimate but is **deliberately unregistered**, exactly
as `yolo_sem_rgkd` is (§7c), so `RUN_STAGES = "…Y"` cannot quietly cost an extra
~4 GPU-hours for an arm nobody asked for.

**Seeds.** `YOLO_L_SEEDS` defaults to `(0,)` — the only stage that defaults to one
seed. At ~3.5 h a run against a mobile arm's ~0.5 h, the three-seed habit that
costs 1.5 GPU-hours in Stage E costs 21 here. **A one-seed miss count is not a
rate** (§15 trap 12): read it as a direction, and buy the other two seeds before
it goes in a table.

**The bar it has to clear.** yolo26n's own miss count moves 16 → 14 → 12 across
seeds 0/1/2 with nothing changed but the seed — 7.57 % ± 1.08 %, mean Dice
0.6911 ± 0.0197. A yolo26l difference smaller than that is seed noise, not
capacity.

**Selection rule.** Best seed by **val** mean Dice, then that seed's **test**
numbers reported — the convention in
`results/final/best_seed_val_selected/`, where yolo26n picked seed 2 on val
0.7311 and reported test 0.7021 rather than its best test seed. Selecting on test
with a ~2-point per-seed spread would inflate the reported figure by roughly the
size of the effect being claimed.

**What it needs.** `pretrained_weights/yolo26l-sem.pt`, ~50 MB, not shipped. §2
reports it as **WARN** when `Y` is not in `RUN_STAGES` and **FAIL** when it is —
a session not running Stage Y should not be blocked by a file it will never open.
`yolo_native.pretrained_for(family, paths)` resolves the backbone from the family
name, because passing nano weights to a Stage Y run raises nothing: it trains,
produces a `best.pt`, and lands in a table as the large arm, with the only symptom
a 28 M-parameter row reporting 1.6 M.

### 7d.1 Results — landed 2026-08-05, three seeds

`YOLO_L_SEEDS` defaulted to `(0,)` and all three were bought anyway. That turned
out to matter more than usual:

| seed | val mean Dice | val misses | test mean Dice | test median | test misses |
|---|---|---|---|---|---|
| 0 | 0.7302 | 4 (2.99 %) | 0.7377 | 0.8319 | 13 (7.03 %) |
| **1 (selected)** | **0.7359** | **5 (3.73 %)** | **0.7055** | **0.8096** | **13 (7.03 %)** |
| 2 | 0.5707 | 20 (14.93 %) | 0.5904 | 0.7256 | 25 (13.51 %) |

Selected on **val mean Dice**, then that seed's test numbers reported — the
convention in `results/final/best_seed_val_selected/`. Seed 0 has the better test
Dice and was not selected, which is the rule working, not failing.

**The answer to the question Stage Y was built to ask.**

| | params | test mean Dice | median | misses |
|---|---|---|---|---|
| yolo_sem_direct (nano) | 1.63 M | 0.7021 | 0.8061 | 12 (6.49 %) |
| yolo_sem_l_direct (large) | 17.86 M | 0.7055 | 0.8096 | **13 (7.03 %)** |

**11× the parameters buys +0.003 Dice and one more complete miss.** The 6.5 %
miss rate is a property of the YOLO recipe or its semantic head, **not of 1.6 M
parameters** — which is exactly the alternative the nano arms could not rule out
and the reason this stage exists. It is a clean negative result on a
pre-registered question, and it should be reported as the finding it is rather
than as a footnote to Stage A.

It also settles the deployment argument: at 11.17 ms YOLO-L is *slower* than
PP-MobileSeg and 2.2× slower than YOLO-N, for no accuracy and one extra miss.
There is no operating point at which the large arm is the right choice.

**⚠️ Seed 2 collapsed, and it is not seed noise.** 0.5904 test mean Dice and 25
misses against 0.7377/13 and 0.7055/13 — a 0.15 Dice gap where the nano arm's
whole three-seed spread is 0.0197. Val saw it too (0.5707, 20 misses), so
val-selection caught it and the reported number is unaffected. **Do not quote a
three-seed mean for Stage Y**: averaging a failed run into a rate reports a
training-stability problem as a model property. Quote the selected seed, state
that one of three runs failed to converge, and put it in the limitations. Whether
the failure is a seed-dependent divergence or something in the large-backbone
recipe is **not diagnosed** — a lead, not a finding.

**Cost, now calibrated.** ~3.5 h per run was the extrapolation; three seeds
landed. `COST_HOURS` can stop being a guess for this row.

---

## 7e. Stage M — multi-teacher routed distillation

> **STATUS 2026-08-06 — COMPLETE, and the answer is a null.** The gate opened
> (§7e.3a), six runs trained, and both arms came back non-inferior on Dice while
> *regressing* complete misses and the skin-tone gap (§7e.8). The load-bearing
> caveat is §7e.9: at β = 8 the router stayed ~88 % of the way to uniform, so what
> was trained is a fixed-weight ensemble and **the routing hypothesis is still
> untested**. One further run (β = ∞, seed 0, ~2.6 GPU-hours) closes it.
>
> §7e.3a and §7e.8–7e.10 are measurements. Everything else is a design decision or
> a pre-registration, and the section says which is which.

**Where it lives, and why not here.** `bruisekit/multiteacher.py` and
`bruise_stage_m.ipynb`, shipped by `69b_zip_stage_m.py`. The
module is **deliberately absent from `60_build_unified_bundle.py`'s copy list** —
it has no source in `scripts/unified_lib/`, so a rebuild neither regenerates nor
deletes it. Stage M is an experiment with a real chance of returning nothing, and
putting an untested loss, an untested teacher shim and a third
`resolve_micro_batch` patch into the notebook that produces Stages A–Y would make
a mistake in any of them a mistake in all of them. Graduating it is two lines when
it earns them.

### 7e.1 The question

Every distillation stage in this study fixes one teacher for the whole training
set — Stage A B2→B0, Stage C B5→B0, Stage F DeepLabV3+→mobile. The teachers are
not uniformly better or worse than each other. On the 185 test images four
different teachers win the five ITA groups:

| group | best teacher | its Dice | B0 student |
|---|---|---|---|
| Light (II-III) | segformer_b5_teacher | 0.766 | 0.757 |
| Intermediate (III-IV) | segformer_b2_teacher | 0.833 | 0.794 |
| Tan (IV) | unet_r50 | 0.766 | 0.730 |
| Brown (V) | unet_r50 | 0.813 | 0.795 |
| Dark (VI) | deeplabv3plus_r50 | 0.791 | 0.755 |

So: route each training image to whichever teacher is best **on that image**.

### 7e.2 Why the routing is feasible, which is the non-obvious part

Stage C's adaptive arm routed on teacher-vs-teacher *agreement* (§6.2's `rel_b`),
because it was framed as an inference-time ensemble and the label is not
available then. Stage M routes at **training time only**, where the ground-truth
mask is already in the batch — `engine.train_run` hands it to the loss on every
step. The router scores each teacher against the actual label, per image.

**There is no generalisation gap in the router, because the router never runs at
test time.** Only the student does. That makes the oracle a *reachable* ceiling
on teacher quality rather than an aspirational one, which is the one respect in
which this differs from `kd/val_oracle.py`.

What no oracle can settle is how much better teacher quality becomes better
**student** quality.

### 7e.3 The gate, and why it is stricter than Stage C's

Stage C is this project's only measurement of that transfer:

```
oracle gain over the best single teacher    +0.0258
realised student gain (p3_adaptive)         +0.0068     ->  ~26 %
```

§6.2's gate opened on the oracle number, and §6.5 records what followed: ten arms,
every one NON-INFERIOR. So Stage M's gate projects through the measured rate and
opens only if the **projected student gain** clears the same one-Dice-point margin
those arms were judged against:

```
open  iff  oracle-gain CI clears zero
      AND  oracle_gain x 0.264  >  0.010        <- the clause Stage C did not have
```

A separate **miss-endpoint** gate is reported alongside and is deliberately not
ANDed in: §1 says miss containment is the endpoint this study is judged on, so a
pool that finds images no single teacher finds is interesting even when Dice says
nothing. The printed verdict names which clause opened, and §11 of the notebook
refuses to let a miss-only opening be written up as a Dice win.

`oracle_gate` also reports **drop-one marginals**, which is what decides pool
membership: a teacher whose marginal is ~0 contributes nothing the rest of the
pool does not already cover, however good its mean Dice looks.

### 7e.3a The gate result — measured, 2026-08-05

Run on the 134 validation images, `seed=0`, 5000 subject-cluster bootstrap
resamples. **This is the only measurement in §7e.**

The pool was first run with four teachers, then U-Net was dropped on the evidence
of its own drop-one marginal. Both are recorded because the pair is the argument:

| | 4 teachers | **3 teachers (adopted)** |
|---|---|---|
| oracle val Dice | 0.8419 | 0.8363 |
| oracle gain over best single | +0.0537 [+0.0327, +0.0657] | **+0.0482 [+0.0276, +0.0609]** |
| projected student gain | +0.0142 | **+0.0127** (margin +0.0100) |
| verdict | RUN | **RUN** |

Per teacher, three-teacher pool:

| teacher | val Dice | wins | drop-one marginal | misses |
|---|---|---|---|---|
| segformer_b2_teacher | 0.7717 | 27.6 % | +0.0121 | 1 |
| segformer_b5_teacher | 0.7881 | 44.8 % | +0.0138 | 3 |
| deeplabv3plus_r50 | 0.7838 | 27.6 % | **+0.0224** | 2 |
| ORACLE (per-image best) | **0.8363** | | | 1 |

**Why U-Net was dropped, and what it revealed.** In the four-teacher pool U-Net
had a drop-one marginal of +0.0055 against +0.011–+0.013 for the others, and won
fewest images (14.9 %). Removing it cost **exactly its marginal** — oracle gain
0.0537 → 0.0482, a loss of 0.0055 to four decimal places — which is a small but
real validation of the marginal as a diagnostic. More interestingly it **doubled
DeepLabV3+'s marginal**, +0.0113 → +0.0224. U-Net and DeepLabV3+ are both
ResNet-50 SMP models: they were covering for each other, so each looked half as
useful as it was. The three-teacher pool is three architecture families with no
redundant member, at 25 % less teacher-forward cost per step.

**The oracle gain is ~2× Stage C's**, +0.0482 against +0.0258, which is the whole
reason this stage is not simply a repeat of §6.2.

**⚠️ The miss clause reads PASS and is illusory. Do not report it.** The gate
compares the pool against the *Dice*-best single teacher, which is B5 with 3
misses, so 1-vs-3 opens the clause. But **B2 alone has 1**, the same as the pool —
and since a pool-oracle miss means every teacher missed that image, B2's single
miss *is* the pool's. The pool recovers nothing. That one image is the
`Unclassified` ITA row (n=1) that no teacher finds. **Dice is the pre-registered
endpoint**; the notebook says so in three places and §13 of it re-states the
warning next to the contrast table.

**The cushion is thin and rests on one number.** +0.0127 against a +0.0100 margin
is 1.27×, and the 0.264 transfer rate is a *single* measurement, from Stage C, on
a different student and a different KD strategy. At a transfer rate of 0.208 the
gate would have closed. State the projection as an assumption in any writeup, not
as a prediction.

### 7e.4 One knob, and it nests the arms that already exist

Routing weights are `softmax(beta * dice_k)` over the K teachers, `dice_k` being
teacher k's **soft** Dice against the label — soft, because the operating point is
not fitted until after training and a router depending on one would entangle this
arm with threshold choice (the same reason `reliability_kd.image_gate` uses soft
Dice).

| beta | what it is |
|---|---|
| `0` | uniform mean of all teachers == Stage C's `p2_ensemble_uniform` |
| `8` | soft routing — the default arm |
| `inf` | hard argmax — the oracle router |
| `K = 1` | **bit-identical to the existing single-teacher KD** |

The fused probability then goes through Stage H's gate unchanged, so Stage M
inherits `ReliabilityGatedDistillLoss` rather than inventing a second gate.
`multiteacher.self_test()` **asserts** both reductions rather than the docstring
claiming them — "reduces to" is exactly the claim that goes unchecked and turns a
contrast into a confound. It runs in §1 of the notebook and at zip time.

### 7e.5 The arms, and the batch pin

| arm | student | control | contrast |
|---|---|---|---|
| `segformer_b0_mtkd` | SegFormer-B0 | `segformer_b0_distilled` | the B2→B0 reference Stage C scored every arm against |
| `lraspp_mobilenetv3_mtkd` | LR-ASPP | `lraspp_mobilenetv3_b2kd` | the strongest mobile student, teacher-matched |

**The batch pin is not housekeeping.** `segformer_b0_distilled` probed to
micro-batch **64**, accum 1 — its `config.json` says so. The same probe with four
teachers resident lands far lower, because the probe measures what fits and what
fits changed. Without a pin the arm would differ from its control in the teacher
signal **and** in batch size **and** in optimizer-step count — three variables,
one of which is precisely what §3's fixed recipe exists to prevent, and the same
failure mode as the old pipeline's B0-vs-B2 batch bug (§15 trap 1).

So `install_batch_shim` reads the batch off the control's own `config.json` and
pins it, and teacher forwards are **chunked** so that pinning it is affordable.
Chunking changes no number: these are eval-mode forwards with frozen BatchNorms,
so a sub-batch and a full batch give identical outputs. Mobile arms need no pin —
`efficient_micro_batch` is a fixed 16 for every efficient family, so
`lraspp_mobilenetv3_mtkd` matches `lraspp_mobilenetv3_b2kd` by construction.

**Pinning to 64 outright does not fit, so the pin is on the EFFECTIVE batch.**
The first training attempt OOM'd: three teachers resident (~138 M frozen
parameters) alongside the student's own forward at micro-batch 64 misses a 40 GB
MIG slice by about 200 MB. `control_batch(max_micro=16)` returns **16 × 4**, which
lands on the control's effective batch of 64 exactly — the divisor is searched, never
rounded, because an approximate effective batch moves the optimizer-step count and
the LR schedule with it.

**The one recipe difference that survives, and it belongs in the limitations.**
SegFormer's decode head contains a BatchNorm, and BatchNorm normalises over the
*micro*-batch. At 16 × 4 it sees groups of 16 where the control saw 64. Effective
batch, step count and schedule are identical; the BN group size is not. That is a
real difference, much smaller than a changed step count, and it should be stated
rather than buried.

### 7e.6 What it does not do

**It does not route by skin tone**, and the fairness framing that motivated it
does not survive the August 08 tables. Every teacher's complete misses are on
**Light (II-III)**; on 55 dark-skin test images not one of the four teachers
misses anything. There is no dark-skin failure to route around — the failures are
on light skin, which §8 identifies as a bruise-size confound. A per-ITA-group
router would be fitting five weights on three-to-five validation subjects each to
fix something that is not there. `stratified_oracle` reports the per-group and
per-size breakdown with `n_subjects` as a **column** so that stays visible.

### 7e.7 The prior, written down before the runs

Three arms of this shape had already returned nulls: Stage C's `p3_adaptive`
(+0.0068, CI crossing zero), Stage C's `p3_adaptive_group` (ITA-weighted KD, 7th
of 10), and Stage H's reliability gate (27 runs, the gate fired, nothing moved).
A fourth null was the expected outcome. What made it worth the ~14 GPU-hours
*given that the gate opened* was that the pool has no redundant member and that
the routing signal is exact rather than estimated. It returned the fourth null.

---

### 7e.8 RESULT — trained 2026-08-06, six runs, and the verdict

**VERDICT: NULL on Dice, REGRESSION on complete misses and on the fairness gap.
And the arm did not test the hypothesis it was built to test** — see 7e.9, which
is the finding that matters.

Two arms × three seeds, paired subject-level bootstrap, 10 000 resamples, β = 8,
three-teacher pool. Contrasts were *intended* to be seed-matched and are so for
LR-ASPP only — see §7e.10 before quoting the SegFormer interval.

| arm | control | mean Δ Dice | per seed | sign |
|---|---|---|---|---|
| `segformer_b0_mtkd` | `segformer_b0_distilled` | −0.0035 | −0.0038, +0.0007, −0.0075 | mixed |
| `lraspp_mobilenetv3_mtkd` | `lraspp_mobilenetv3_b2kd` | **−0.0129** | −0.0157, −0.0123, −0.0108 | **all negative** |

Every seed of both arms returns `INCONCLUSIVE` against the one-Dice-point margin.
LR-ASPP seed 0 has CI [−0.0336, −0.0004], which excludes zero with
`p_a_better = 0.022`; it reads INCONCLUSIVE only because the harm is *smaller*
than the non-inferiority margin. Three seeds negative at a consistent magnitude
is a real if small regression, not noise.

**Complete misses moved the wrong way, and that is the endpoint (§1).**

| model | misses, seeds 0/1/2 |
|---|---|
| `segformer_b0_distilled` (control) | **0 / 0 / 0** |
| `segformer_b0_mtkd` | 2 / 2 / 3 |
| `lraspp_mobilenetv3_b2kd` (control) | 0 / 0 / 1 |
| `lraspp_mobilenetv3_mtkd` | 1 / 2 / 1 |

The SegFormer control is a zero-miss model at all three seeds and the routed arm
is not, at any. Same shape as Stage F's Fast-SCNN arm (§7b.5): KD amplifying the
miss rate while Dice barely moves.

**The fairness gap widened**, which is the direct test of the premise this stage
was motivated by:

| model | ITA gap, seeds 0/1/2 | worst group |
|---|---|---|
| `segformer_b0_distilled` (control) | 0.043 / 0.043 / 0.043 | Dark (VI) |
| `segformer_b0_mtkd` | 0.045 / 0.049 / 0.055 | Dark (VI) |
| `lraspp_mobilenetv3_b2kd` (control) | 0.067 / 0.022 / 0.060 | Tan / Dark / Dark |
| `lraspp_mobilenetv3_mtkd` | 0.072 / 0.076 / 0.089 | Dark (VI) |

No Kruskal–Wallis test is significant for any arm or control. **Multi-teacher
fusion made the skin-tone Dice gap slightly wider, not narrower.** Report that as
the finding; it is a clean negative on a pre-registered fairness hypothesis.

### 7e.9 Why this result does not refute the routing hypothesis

**The router barely routed.** From `stage_m_router_diagnostics.csv`, consistent
across all six runs:

| quantity | value |
|---|---|
| mean routing entropy | 0.95 – 0.98 |
| uniform, `log 3` | 1.099 |
| **fraction of maximum** | **≈ 88 %** |
| mean weights | B2 0.27, B5 0.48, DeepLabV3+ 0.26 |

At β = 8 the teachers' per-image soft Dice values are too close together to
separate: a spread of ~0.05 gives `exp(8 × 0.05) ≈ 1.5`, a mild tilt rather than a
choice. **What was trained is a near-uniform weighted teacher ensemble, not
per-image routing** — which is approximately Stage C's `p2_ensemble_uniform` with
a gate, and *that* arm was also non-inferior. The gate's premise was the oracle,
β → ∞, per-image argmax; that premise was never exercised.

So the honest statement of what Stage M has shown so far is: **a fixed-weight
three-teacher ensemble does not beat single-teacher KD, and slightly harms miss
containment and group equity.** Whether per-image routing does is still open.

**The one run that would close it:** `BETA = float("inf")`, seed 0,
`segformer_b0_mtkd` only — ~2.6 GPU-hours, hard argmax, the exact quantity the
oracle measured. Still null → a complete finding, that even oracle-quality
per-image teacher selection does not transfer to the student, which is the
question the whole Stage C → Stage M line has been circling. Positive → β was the
bug and a sweep is warranted. Until that run exists, "routing does not help" and
"β = 8 was not routing" are not distinguishable, and the writeup must not claim
the first.

### 7e.10 A defect in the contrast, and what it does not change

In `report.load_per_image`, all three seed queries for `segformer_b0_distilled`
resolve to the **same** cached results-tier CSV — the val-selected best seed. Its
three control rows are identical to six decimal places (0.816681 median, 0.768011
mean, 0 misses, and 0.768011 is exactly `headline_all.csv`'s value). So the
SegFormer contrast is three arm seeds against **one** control, not the
seed-matched design §7e.8 reports.

The LR-ASPP controls are correct — they were read from the run directories under
`EXTRA_RUNS` by the notebook's fallback path, and their three seeds differ
(0.795 / 0.791 / 0.777). **Force that path for both arms before publishing any
Stage M contrast.** The conclusion is unlikely to move, since the arm loses to
the *best* control seed and would lose to the mean of three by at least as much,
but the design claim in the writeup has to be true.

### 7e.11 Where Stage M's numbers live

`STAGE_M_RESULTS/` at the bundle root, never `results/` or `FINAL_RESULT/`, and
runs in `STAGE_M_RESULTS/runs/` rather than the shared run directory — so a
stage that returned a null leaves no trace in the directories the published
numbers come from. `stage_m_contrasts.csv` is the answer,
`stage_m_test_per_seed.csv` the table, `stage_m_router_diagnostics.csv` the
reason.

---

## 7f. Stage N — foundation encoders vs ImageNet encoders

> **STATUS 2026-08-06 — GATE MEASURED (§7f.8). The gate opened and the
> attribution arm overturned the premise.** MedSigLIP beats ImageNet ResNet-50 by
> +0.368, but **loses to DINOv2 — which has no medical data — by −0.166,
> CI [−0.201, −0.131].** Medical pretraining is not the cause of the gain and is
> measurably worse than generic self-supervised pretraining at this task.
>
> §7f.8 is a measurement. §7f.9 is the confound that has to be closed before it is
> quoted, and it is cheap. The first gate run (§7f.7a) is void and separate.
>
> This is the stage that follows Stage M's null (§7e.8). It is deliberately
> **not** another distillation stage — see §7f.2, which is the load-bearing part
> of this section.

**Where it lives, and why not here.** `bruisekit/foundation.py` and
`bruise_foundation.ipynb`, shipped by `70b_zip_foundation.py`, generated by
`70_generate_foundation_notebook.py`. The module is **deliberately absent from
`60_build_unified_bundle.py`'s copy list**, for exactly the reason
`multiteacher.py` is (§7e): it has no source in `scripts/unified_lib/`, so a
rebuild neither regenerates nor deletes it, and an untested architecture wrapper
plus a third `resolve_micro_batch` patch stay out of the notebook that produces
Stages A–Y. Graduating it is two lines when it earns them.

### 7f.1 The question

**Does the pretraining corpus matter?** Same decoder, same recipe, same split,
same seeds — only the encoder's pretraining changes. Every encoder in this study
so far is ImageNet-supervised (MiT, ResNet-50, MobileNetV3, StrideFormer,
TopFormer) or scratch (Fast-SCNN). Medical vision-language encoders now exist
that were trained on clinical images including dermatology. Nobody has asked
whether that helps here.

That is the whole scope. It is a **Stage B-style baseline question**, and it
stands on its own whichever way it comes out.

### 7f.2 Why it is a baseline stage and not a distillation stage

The proposal this was cut down from chained a foundation teacher into
reliability-gated + multi-teacher + ITA-fairness-aware KD. Three objections,
every one of them from this project's own results rather than from principle:

**1. There is no teacher deficit to fix.** On the current lineage
`segformer_b2_teacher`, `segformer_b5_teacher` and `segformer_b0_distilled` each
have **zero** complete misses on the 185 test images (§4, §6.1). The
pre-registered endpoint is already saturated at the top of the field. A better
teacher cannot improve on zero.

**2. The bottleneck is transfer, not teachers.** §7e.8 is the measurement: the
oracle had **+0.048** of real, reproducible headroom and the student captured
**none** of it, because the fused target scored 0.723 — *below every individual
teacher in the pool* (0.772 / 0.788 / 0.784). Adding a fifth, larger teacher to a
fusion that already degrades its inputs changes the wrong variable.

**3. The fairness target is not there.** Twenty of the twenty-one ITA tests in
this study are non-significant. The one that is (`segformer_b2_teacher`,
p = 0.0105) has its worst group at **Tan (IV)**, not Dark; where YOLO shows a gap
the worst group is **Light (II-III)**, which §8 attributes to a lesion-size
confound. And the mechanism has already been run: Stage C's `p3_adaptive_group`
— ITA-group-weighted KD, λ = 0.5 — scored **0.7586** against the 0.7680 reference,
one of the worst arms in that grid (§6.4).

Stacking three mechanisms that have each independently returned null or negative
means an outcome nobody can attribute: a failure cannot be localised and a success
cannot be credited. So distillation is out of scope **by construction**. If the
baseline shows something, *one* KD arm against a teacher-matched control is the
next decision, and it is a separate one.

### 7f.3 The arms

| key | encoder | params | pretraining | licence |
|---|---|---|---|---|
| `medsiglip` | SigLIP vision tower, 448 px, **patch 14** (earlier revisions said 16 — see §7f.9) | ~400 M | medical image-text pairs incl. dermatology | **Health AI Developer Foundations — use-restricted, gated** |
| `dinov2` | ViT-B/14 | ~86 M | LVD-142M natural images, self-supervised | Apache-2.0 |
| `resnet50` | ResNet-50 | 25.6 M | ImageNet-1k supervised | BSD-3-Clause |

Each appears twice: a `*_probe` arm (encoder frozen, **linear** head) and a
`*_seg` arm (last 6 blocks unfrozen for the ViTs, last 2 stages for ResNet-50,
conv decoder).

**`resnet50` is the control the gate is scored against**, and it is the right one:
same encoder family as `unet_r50` and `deeplabv3plus_r50`, so a win over it is
directly interpretable against Stage B rather than against a number from a
different lineage.

**`dinov2` is the attribution arm, and it is the most important addition.** It is
a modern self-supervised ViT with **zero** medical data. If it matches MedSigLIP,
then whatever the probe measures is *modern ViT pretraining*, not *medical
pretraining*, and the headline claim has to change. The original proposal had no
arm that could tell those two apart — which is the single most likely way this
stage would have produced a confidently wrong sentence in a paper.

### 7f.4 The gate, and why a frozen probe

Fixed in `foundation.py` before any number was produced:

```
open  iff  the (medsiglip_probe - resnet50_probe) val-Dice CI clears zero
```

Two things are reported **alongside** and deliberately **not** ANDed in — the same
policy §7e.3 uses, and for the same reason:

- **Attribution.** `medsiglip - dinov2`, plus `medical_share_of_gain`. The gate
  can open and the *claim* still be wrong; there is no honest way to fold that
  into one boolean, so it prints separately and warns below 50 %.
- **Misses.** The endpoint §1 says decides.

**Why a frozen probe rather than just fine-tuning.** A fine-tuned 400 M encoder on
697 images from 95 subjects will fit the training *subjects* whatever its
pretraining was, so a good fine-tuned score is weak evidence that the pretraining
helped. A frozen encoder with a linear head can only report what is **already in
the features**. It also costs ~1 GPU-hour against ~20, which is the entire logic
of gating (§6.2, then §7e.3).

**The head must be linear, and that is not a stylistic choice.** The moment the
head can learn spatial structure it stops measuring the encoder and starts
measuring the head, and a strong decoder papers over a weak encoder well enough to
make every arm tie — which reads as *"no difference between pretraining corpora"*
when it actually means *"the experiment could not see one"*.

### 7f.5 The prior, written down before the runs

| model | params | mean Dice |
|---|---|---|
| segformer_b0_direct | 3.71 M | 0.7663 |
| segformer_b2_teacher | 27.35 M | 0.7692 |
| segformer_b5_teacher | ~85 M | 0.7727 |

**23× the parameters bought +0.006 Dice**, and §1's annotation ceiling puts
human-vs-human agreement between 0.581 and 0.873. A 400 M encoder landing near
0.775 is the *expected* outcome and would tell us nothing new about Dice.

So the interesting outcomes are, in order: **the gate closing** (cheap, clean,
publishable null); **misses moving**; or **DINOv2 matching MedSigLIP**, which
would reframe the foundation-model question for this task entirely. §13 of the
notebook therefore leads with **complete misses and IQR** — the two metrics that
separated the field in Stage A while mean Dice did not (§4).

A closed gate is written up as: *"a 400 M encoder pretrained on medical image-text
pairs including dermatology does not produce better frozen features for bruise
segmentation than a 26 M ImageNet ResNet-50."* That is a result. Stop there.

### 7f.6 Two limitations that are owed regardless of the outcome

**1. Resolution.** The ViT arms see a **448**-pixel image where every other model
in this study sees 640. That is MedSigLIP's native resolution, and the input is
resized to it rather than having the position embeddings interpolated to 640 —
because a probe of pretrained features should show those features at the
resolution they were pretrained at. It costs small-bruise detail, which is exactly
the population the complete-miss metric is about. This is a property of the
encoder, not of a choice made here, and it must be stated wherever these arms are
compared.

**2. Batch.** A 400 M ViT with gradients does not fit past micro-batch 4 alongside
a 40 GB MIG slice. These arms run `4 × 4` accumulation, so the **effective** batch
(16), the optimizer-step count and the LR schedule are identical to the SMP and
mobile baselines. The decoder uses **GroupNorm**, which is per-sample — so unlike
Stage M's BatchNorm note (§7e.5) this costs nothing statistically. State it anyway.

**And the licence.** §7b.1 chose DeepLabV3+ over SegFormer as the Stage F teacher
specifically to escape NVIDIA's non-commercial MiT licence. MedSigLIP is
use-restricted and gated; DINOv2 is Apache-2.0. **If the two tie, the licence
decides**, and `report_sources` puts the licence in the results table so that
cannot be forgotten.

### 7f.7a ⚠️ The first gate run (2026-08-06) is INVALID — do not quote it

The gate ran and printed `VERDICT: OPEN` on `medsiglip − resnet50 = +0.3663`,
CI [+0.3200, +0.4276], with `medical_share_of_gain = 85.9 %`. **Discard it.**

`_load_vision_encoder` chose its class by trying `SiglipVisionModel`, then
`Dinov2Model`, then `AutoModel`, and returning the first that did not raise.
`SiglipVisionModel.from_pretrained` **does not raise on a DINOv2 directory**: it
reads `hidden_size` / `patch_size` / `image_size` out of the config, builds a
SigLIP body of those dimensions, matches **none** of the checkpoint's parameter
names, emits a warning, and returns a randomly initialised model. The log line
`dinov2: loaded via SiglipVisionModel -> SiglipVisionTransformer` is the tell, and
so is the parameter count: **93.65 M against DINOv2 ViT-B/14's published 86.6 M.**

So the DINOv2 arm was a random ViT. Its 0.177 Dice and 0.114 val AP are noise
floors, and **the entire attribution number was MedSigLIP measured against
noise** — the one arm added specifically to stop this stage making a confidently
wrong claim was itself the broken one.

**Two guards were added, and both are cheap:**

1. **Dispatch on `config.json`'s `model_type`**, never trial-and-error
   (`_MODEL_TYPE_TO_CLASS`). An unrecognised type raises rather than falling back
   to `AutoModel`.
2. **Verify the load and the size.** `from_pretrained(..., output_loading_info=True)`
   and any missing key raises; then the built encoder's parameter count is checked
   against a published figure recorded per source (`expected_params_M`), tolerance
   5 %. Either guard alone would have caught this; the second is the one that also
   catches a right-config/wrong-class body.

This is the same lesson as §7.3a and trap 1: **a number that is produced without
error is not a number that was produced correctly**, and the check has to be on
the mechanism rather than on the plausibility of the output. `weights.py`'s
docstring already said so for the mobile backbones — the HuggingFace path simply
never inherited it.

**What is still unknown.** The headline contrast may well survive: MedSigLIP's own
load looks right (428.6 M against SoViT-400m's published ~428 M, hidden 1152). But
it came through the same unverified path, so it is *unconfirmed*, not *confirmed*.
Re-run the gate on the patched module before anything is written down.

### 7f.7b What the re-run has to explain even if it holds

The absolute probe numbers were low across the board — 0.492 / 0.177 / 0.125 —
against 0.77 for trained models. Low is *expected* for a frozen encoder under a
single 1×1 convolution, and the gate is a paired difference so a common offset
cancels. But **ResNet-50 at 0.125 with val AP 0.069 deserves a second look**: an
ImageNet trunk should carry more than that about a coloured blob, and its
stride-32 20×20 grid at 640 is coarser than MedSigLIP's 32×32 at 448. If the
re-run leaves the control that low, the honest reading is that the probe is partly
measuring **feature-grid resolution**, not only pretraining — and that is a
confound to state, not to argue away.

### 7f.8 THE GATE RESULT — measured 2026-08-06, patched loader

> # ⚠️ SUPERSEDED 2026-08-09. THE +0.5342 IN THIS SECTION IS ~74 % ARTEFACT.
>
> Stage N2 (§7h) re-ran all three arms at a matched grid and with the ResNet
> probed at `layer3` instead of `layer4`. `dinov2 − resnet50` is **+0.1383**, not
> +0.5342.
>
> **The cause is not the grid.** §7h.2a measured the grid effect directly, on one
> encoder at four input sizes, and it is **−0.101 Dice per doubling** — a finer
> grid is *worse*, and the interval excludes zero. §7f.9's suspicion was
> well-founded as a suspicion and wrong as a diagnosis.
>
> **The cause is the probe layer.** At an identical 20 × 20 grid, ResNet-50's
> `layer3` scores **0.5568** against `layer4`'s **0.1242** — the same weights,
> +0.4326 Dice. The 0.1225 below is `layer4`, and `layer4` scores *below the raw
> pixel floor* (0.1417) because post-ReLU ImageNet-class features carry almost no
> linearly-readable bruise location. Nothing in this run was broken; the reading
> of it was.
>
> **What survives:** modern self-supervised ViT features do beat ImageNet
> ResNet-50 features here — by +0.138, not +0.534. **What does not:** the word
> *dramatically*, and any argument resting on ResNet-50 scoring near zero.
>
> The MedSigLIP-vs-DINOv2 sign below survives too, and §7h.7 reproduces it at a
> matched grid (0.467 vs 0.663). Quote §7h.7's table, not this one.

134 validation images, 20 subjects, encoder frozen, linear head, 10 000
subject-cluster bootstrap resamples. **This is the only measurement in §7f.**

| arm | encoder params | pretraining | val AP | mean Dice | median | misses |
|---|---|---|---|---|---|---|
| **dinov2_probe** | 86.6 M | LVD-142M natural images, self-supervised | — | **0.6567** | 0.6882 | 2 |
| medsiglip_probe | 428 M | medical image-text, incl. dermatology | — | 0.4907 | 0.4873 | 2 |
| resnet50_probe | 23.5 M | ImageNet-1k supervised | — | 0.1225 | 0.1023 | 5 |

| contrast | Δ Dice | CI 95 % |
|---|---|---|
| medsiglip − resnet50 | +0.3682 | [+0.3220, +0.4293] |
| dinov2 − resnet50 | **+0.5342** | [+0.4860, +0.5869] |
| **medsiglip − dinov2** | **−0.1660** | **[−0.2010, −0.1306]** |

**The gate opened on its pre-registered rule and the attribution arm overturned
the premise.** `medical_share_of_gain = −45 %`: the medical corpus does not
explain the gain, it *subtracts* from it. A 5× smaller, Apache-2.0 encoder with
zero medical images beats the 428 M dermatology-pretrained one, and the interval
is nowhere near zero.

**What this licenses saying, and what it does not.**

- ✅ *"Frozen features from modern self-supervised ViTs are dramatically better
  than ImageNet-supervised ResNet-50 features for this task."*
- ✅ *"Medical-domain pretraining gave no advantage over generic self-supervised
  pretraining here, and performed worse."*
- ❌ *"Foundation models help because they were trained on medical images."* This
  is the claim the stage was built to test, and it did not survive. It is
  precisely the sentence that would have been written if the DINOv2 arm had been
  omitted, or left broken as it was in §7f.7a.

**Consequence for §10.** DINOv2 is now the headline arm: better features, 5×
smaller, Apache-2.0 against MedSigLIP's use-restricted licence. MedSigLIP is still
worth training — the probe ranks features, not fine-tuned models, and the whole
point of §7f.4 is that those are different questions — but it is no longer the
protagonist.

### 7f.9 ⚠️ The confound that has to be closed before §7f.8 is quoted

> # ✅ CLOSED 2026-08-09 by §7h — and the diagnosis below is WRONG.
>
> The suspicion was right to block §7f.8. The mechanism named in it was not.
>
> **The grid explains none of the gap.** One encoder, four input sizes, only the
> grid moving: **−0.101 Dice per doubling**, CI [−0.128, −0.074] (§7h.2a). Finer
> is *worse*. Bruises are large connected regions and a coarse grid is a
> smoothness prior. A zero-pretraining pixel floor is flat across the same span.
>
> **The layer explains ~74 % of it.** At an identical 20 × 20 grid, `layer3`
> scores 0.5568 against `layer4`'s 0.1242 (§7h.2b). The `layer3` control proposed
> at the end of this section was the right experiment for the wrong reason — it
> would have worked, and the explanation written here for *why* it would work is
> not the one that turned out to be true.
>
> **The second point in this section — that linear probing is DINOv2's home
> benchmark — is untouched and got worse.** §7h.9 item 2 upgrades it: DINOv2 is
> also the only arm with a *patch-level* pretraining objective, against three
> CLIP-family arms trained on a pooled vector. That is the live threat now.
>
> **One number below is also wrong:** MedSigLIP is SoViT-400m/**14**, not patch
> 16, so its grid at 448 was **32 × 32**, not 28 × 28. The rank-correlation claim
> is unaffected — 37 / 32 / 20 is still monotone with the score order.

**The probe's score ordering is the same as its feature-grid ordering, exactly.**

| arm | input | patch/stride | grid | mean Dice |
|---|---|---|---|---|
| dinov2 | 518 | 14 | **37 × 37** | 0.6567 |
| medsiglip | 448 | 14 *(corrected)* | **32 × 32** | 0.4907 |
| resnet50 | 640 | 32 | **20 × 20** | 0.1225 |

A linear probe is a 1×1 convolution on that grid, bilinearly upsampled to 640. Its
achievable Dice is bounded by how coarsely it can draw a boundary — so a finer
grid is worth Dice *independently of what the encoder knows*, which is the one
thing this stage is trying to isolate. Rank correlation of 1.0 between grid size
and score is not proof of a confound, but it is exactly what a confound looks
like, and it must be ruled out rather than argued away.

**A second, narrower version of the same problem: the protocol favours DINOv2 by
construction.** Frozen-feature linear probing is the benchmark DINOv2 was designed
and tuned against — it is the headline evaluation in its own paper. A supervised
ResNet-50's post-ReLU `layer4` features were never optimised to be linearly
separable. Some of the +0.534 is real; some is the evaluation protocol matching
one contestant's training objective.

**The control that settles the first one, and it is cheap.** Re-run
`resnet50_probe` on `layer3` (stride 16 → **40 × 40** at 640) instead of `layer4`.
That gives the ImageNet arm a *finer* grid than DINOv2 has. Then:

- if ResNet-50 stays near 0.12 → the gap is representational, §7f.8 stands as
  written;
- if it climbs toward DINOv2 → the probe was substantially measuring resolution
  and **every number in §7f.8 is a resolution ranking wearing a pretraining
  label**.

~20 minutes of GPU time, one config change, and it is the difference between a
publishable claim and a retracted one. **Do not write §7f.8 up until it has run.**

> **BUILT 2026-08-09 as §7h (Stage N2), and generalised.** The `layer3` control
> above is one point; Stage N2 runs the same encoder at four input sizes (grids
> 20/28/37/40) plus a zero-pretraining pixel floor at the same grids, so the
> output is a calibration *curve* rather than a single comparison — and it adds
> the pair that needs no curve at all, `layer4 @ 640` against `layer3 @ 640`,
> where the same weights see the same input and only the stride differs. Stage N2
> also fixes the second problem this section does not raise: **MedSigLIP is a
> general-medical encoder, not a dermatology one**, so §7f.8 never tested the
> claim it states. See §7h.

---

### 7f.7 Where Stage N's numbers will live

`FOUNDATION_RESULTS/` at the bundle root, never `results/`, `FINAL_RESULT/` or
`_work/runs/` — same isolation as Stage M, and for the same reason. `gate.json`
is the decision, `probe_val_summary.csv` the evidence for it,
`foundation_test_per_seed.csv` the table, `foundation_contrasts.csv` the answer.

Encoder weights go in `pretrained_weights/foundation/` and are **not** in the
overlay zip: MedSigLIP is gated and redistributing it would route around the
licence acceptance. The notebook runs with `HF_HUB_OFFLINE=1` and §2a **stops with
the exact download commands** if an encoder is absent — it never falls back to
random init, which on 697 images is a large and completely invisible handicap
(the same policy `weights.py` applies to the mobile backbones).

---

## 7g. Stage P — lesion-size-stratified miss containment

> **STATUS 2026-08-07 — RUN AND CONFIRMED. §7g.6 is the measurement.**
> **6 of 18 contrast × endpoint cells clear zero.** The small-lesion endpoint
> separates models that mean Dice cannot (§8b: Friedman p = 0.607), and the one
> post-hoc hypothesis in the family **reversed** when it was tested properly.
>
> **Confirmed across two independent result trees.** The laptop dry run read
> `<bundle>/FINAL_RESULT/RESULT_AUGUST_08/`; the ORC run read
> `/scratch/tbommawa/bruise_work/outputs/`. Same 18 models, same 74-image
> stratum, **every number identical**. That is a stronger provenance check than
> either run alone, and it retires the concern in §7g.9 that the two trees had
> drifted.
>
> **No GPU was involved in either run.** Stage P reads per-image CSVs, like
> Stage D — it was unaffected by the GPU cluster being down (§7g.9).

**Where it lives, and why not here.** `bruisekit/lesionsize.py` and
`bruise_lesion_size.ipynb`, shipped by `71b_zip_lesion_size.py`, generated by
`71_generate_lesion_size_notebook.py`. The module is **deliberately absent from
`60_build_unified_bundle.py`'s copy list**, for exactly the reason `foundation.py`
(§7f) and `multiteacher.py` (§7e) are: a stratum-and-contrast pre-registration
should not silently become part of the file that produces Stages A–Y. A rebuild
neither regenerates nor deletes it. Graduating it is two lines.

### 7g.1 The question

§8b measured that the headline field does not separate on Dice — Friedman
p = 0.607, Kendall W = 0.027 across the seven headline models — and §1 explains
why: every model is inside the annotation-ceiling band. Both Stage A confirmatory
contrasts return NON-INFERIOR or INCONCLUSIVE.

But the descriptive pass on 2026-08-07 found that **complete misses are not spread
over the test set**:

| GT-area decile | misses, summed over 39 models |
|---|---|
| D1 (median 4071 px, 1.0% of frame) | **58** |
| D2 | 8 |
| D3 | 21 |
| D4 | 19 |
| D5–D10 | 13 combined |

**89% of every complete miss in the study is in D1–D4; 49% is in D1 alone.** And
the spread across models is far larger there than on Dice: mean Dice ranges 0.605
→ 0.775 across the field (0.169), while bottom-decile recall ranges 0.341 → 0.844
(0.504) — **three times the separation.**

So the question is: **is that separation real, or is the small-lesion endpoint
underpowered at 185 images from 28 subjects?** Both answers are useful, which is
what makes it worth running first.

### 7g.2 Why it runs before every queued GPU job

Stage N's `layer3` control (§7f.9), an ALS→white-light distillation stage, and a
Fenwick merge are all justified by the same sentence: *"models miss small bruises,
so let us fix that."* That premise currently rests on single-seed counts with no
intervals.

If the differences are not resolvable at n = 28 subjects, the premise is not
established, and the correct next move is **more data rather than more mechanism**
— which reorders the entire queue (§18). Settling it costs minutes on a laptop.

This is the same gating logic §6.2 and §7e.3 use, applied to an analysis rather
than to training: run the cheap thing that can invalidate the expensive thing.

### 7g.3 The pre-registration

Module constants in `lesionsize.py`, not choices in the notebook — the same
policy `significance.CONTRAST_FAMILY` uses, and for the same reason. A stratum
chosen after seeing which stratum separates the models is not a test, it is a
search.

```
N_BINS            = 10                          deciles, ~18-19 images each
PRIMARY_STRATUM   = ("D1","D2","D3","D4")       74 images, 18 subjects
PRIMARY_ENDPOINTS = zero_dice_rate, mean_recall, median_dice
CONTRASTS         = 6 pairs -- 4 confirmatory, 2 exploratory
```

**D1 alone is not the primary stratum.** 19 images from 7 subjects cannot support
a claim; it is reported separately in §7b of the notebook so its fragility is
visible rather than assumed. Median CI width there is 0.218 against 0.05–0.09 in
the primary stratum.

**Two of the six contrasts are labelled `exploratory` because they were generated
by looking at the descriptive table**, including the observation that a 3.22 M
mobile arm showed the study's highest bottom-decile recall. The label travels
into the output CSV. §7g.6 shows exactly why that mattered.

### 7g.4 Two miss definitions, kept separate

```
zero_dice    dice == 0             the published endpoint (§1)
empty_pred   pred_positive == 0    what the per-seed tables count
wrong_place  = zero_dice - empty_pred
```

`wrong_place` is a model outputting a substantial region **entirely in the wrong
place** — 0 Dice while predicting thousands of pixels. On the current lineage
`fastscnn` has 13 zero-Dice images against 8 empty predictions and `fastscnn_rgkd`
has 6 against 1: **five of `fastscnn_rgkd`'s six failures are confident errors.**
That is a worse clinical failure than predicting nothing, it is invisible if the
two are collapsed, and it needs no bootstrap to be worth reporting.

This is the measurement behind the existing note in §14 that the per-seed table
and the normalize path count different things. They are both right; they are
counting different failures.

### 7g.5 Rates, never counts; and which subjects get resampled

A cluster-bootstrap draw does not contain a fixed number of images — resampling
subjects with replacement changes the row count on every draw. **Bootstrapping a
count would measure how many rows the draw happened to contain.** Every quantity
resampled here is a rate or a mean.

Only subjects with **at least one image in the stratum** are resampled. Including
the rest adds draws contributing zero rows, which inflates the statistic's
variance for reasons that have nothing to do with the data. In the primary
stratum that is 18 of the 28 test subjects — a number that must be stated
wherever these intervals are quoted, because it is not 28.

Contrasts are **paired**: the same resampled subject list is applied to both
models on every draw. At 74 images, discarding the pairing would hide every
effect there is.

### 7g.6 THE RESULT — 2026-08-07, 10 000 resamples, confirmed on two trees

74 images, **18 subjects** (not 28 — only subjects with an image in the stratum
are resampled, §7g.5). **6 of 18 contrast × endpoint cells clear zero.**

| contrast | endpoint | Δ | CI 95 % | p | p Holm | kind |
|---|---|---|---|---|---|---|
| **b0_direct − yolo_sem_direct** | **zero-Dice rate** | **−0.1216** | [−0.2642, −0.0349] | 0.0014 | **0.0168 ✓** | confirmatory |
| b0_direct − yolo_sem_direct | mean recall | +0.1278 | [+0.0230, +0.2460] | 0.0146 | 0.161 ✗ | confirmatory |
| b0_direct − unet_r50 | zero-Dice rate | −0.0676 | [−0.1452, −0.0141] | 0.0222 | 0.222 ✗ | confirmatory |
| b5_teacher − b2_teacher | mean recall | +0.0510 | [+0.0212, +0.0761] | 0.0026 | — | exploratory |
| lraspp_b2kd − b5_teacher | mean recall | −0.1152 | [−0.2098, −0.0258] | 0.0128 | — | exploratory |
| lraspp_b2kd − b5_teacher | median Dice | −0.1144 | [−0.1842, −0.0090] | 0.0328 | — | exploratory |

### ⚠️ After multiplicity correction, ONE cell survives

The family is 4 confirmatory pairs × 3 endpoints = **12 cells**. Testing twelve
things at α = 0.05 and reporting whichever cleared is how a study manufactures
findings — about 0.6 are expected to clear from noise alone. **Holm–Bonferroni is
applied within the confirmatory set**, the same policy §8b.1 uses. Added
2026-08-07 after the first run reported uncorrected intervals only.

**Only `b0_direct − yolo_sem_direct` on zero-Dice rate survives** (p_holm 0.0168).

**`b0_direct − unet_r50` does NOT survive** (p 0.0222 → **p_holm 0.222**). That
was the most interesting cell in the family — the accurate-baseline dissociation
where U-Net holds the stratum's second-best median Dice (0.836) and six times the
misses — and it is **not** a confirmatory finding. It may be reported as a
descriptive observation with the Holm-adjusted p attached, and it is a good
candidate for a pre-registered replication on more data. It is not a claim.

The two exploratory rows stay uncorrected **and labelled**; folding them into the
same correction would either over-penalise the four questions this stage was
designed around or launder two post-hoc comparisons into confirmatory ones. One
of them has already reversed once.

The four **zero-miss** models — b5, b2, b0_distilled and (at one miss) b0_direct —
carry **0, 0, 0 and 1** complete misses inside the 74-image stratum. Their whole
separation is in *recall*, not in miss count, which is why `mean_recall` is the
endpoint that moves and `zero_dice_rate` is flat at the top of the table.

**Four things this says.**

1. **The endpoint has power, and it separates models Dice cannot.**
   `segformer_b0_direct` beats `unet_r50` on small-lesion miss containment
   (−0.068, CI clears) — while `unet_r50` has the **best median Dice in the entire
   study** (0.833). The Dice ranking and the miss ranking disagree, and that
   disagreement is statistically supported. This is the finding §1 has been
   asserting qualitatively since the beginning.

2. **Stage A's two confirmatory contrasts stay null even here.** Neither
   `b0_distilled − b0_direct` nor `b2_teacher − b0_direct` clears zero on any
   endpoint. Distillation and 7.4× teacher capacity do not buy small-lesion miss
   containment either — so §6's NON-INFERIOR verdict was not a power artefact of
   averaging over easy images. **`segformer_b0_direct` remains the pick** (§10).

3. **The accuracy-vs-speed gap is a small-lesion gap.** `b0_direct` beats
   `yolo_sem_direct` on both endpoints, and §8b's whole-set version of that
   contrast (Δ +0.064) is now localised: it is happening in D1–D4.

4. **The one post-hoc hypothesis REVERSED.** The descriptive table appeared to
   show `lraspp_mobilenetv3_b2kd` beating `segformer_b5_teacher` on bottom-decile
   recall (0.844 vs 0.828). Tested properly on D1–D4 it is **wrong by 0.115 in the
   other direction**, CI [−0.210, −0.026], P(lraspp better) = 0.001. A
   single-decile, 19-image, single-seed observation inverted under a 74-image
   paired bootstrap. This is exactly what the `exploratory` label exists to catch,
   and it is the strongest argument in this section for keeping the
   pre-registration in the module.

**Minimum detectable effect** (CI half-width, median over the six contrasts):

| endpoint | median | worst |
|---|---|---|
| zero-Dice rate | **0.0235** (≈ 1.7 of 74 images) | 0.1146 (≈ 8.5 images) |
| mean recall | 0.0696 | 0.1115 |
| median Dice | 0.0524 | 0.0876 |

So at 18 subjects the test resolves a ~2-image difference in miss count for the
easier pairs and needs ~8 images for the hardest. Any claim about a one-image
difference — which is what separates most of the zero-miss models — is below the
floor and must not be made.

### 7g.6a ⚠️ Degenerate contrasts, and the reporting bug the run exposed

`b5_teacher − b2_teacher` on `zero_dice_rate` returned **Δ = 0.0000, CI
[0.0000, 0.0000]**. Both models have **zero** misses in the stratum, so every one
of the 10 000 resamples produced exactly the same difference: zero. There is no
variation for the bootstrap to propagate and the interval collapses to a point.

As first shipped, that row reported `min_detectable = 0.0000` and
`P(a better) = 0.000`. **Both readings are wrong in opposite directions:**

- `min_detectable = 0` reads as *"this test can detect any effect whatsoever"* —
  the exact inverse of the truth, which is that the test saw no variation and
  learned nothing about its own sensitivity.
- `P(a better) = 0.000` reads as *"B5 is never better than B2"* when the two are
  **identical**, because `(draws < 0).mean()` is 0.0 when every draw is 0.0.

Fixed 2026-08-07: `paired_stratum` now flags `degenerate` when the resampled
differences have zero range, sets `p_a_better` and `min_detectable` to **NaN**,
forces `clears_zero = False` (a tie is not a win), and `min_detectable()` excludes
degenerate rows from the median and the max while reporting `n_degenerate`
separately. **No verdict changed** — 6 of 18 before and after; only two cells'
presentation did.

This is the same class of error as §7.3a and §7f.7a: a number that appeared
without an error message, was arithmetically correct, and meant the opposite of
what it looked like. It is recorded here because a `min_detectable` of 0 in a
power table is exactly the kind of figure that gets quoted.

### 7g.7 What this does NOT license

- ❌ *"Model X misses fewer small bruises than model Y"* for any pair not in
  `CONTRASTS`. The family is six pairs; the descriptive tables cover 18 models and
  are descriptive only.
- ❌ Anything from D1 alone. 19 images, 7 subjects, median CI width 0.218.
- ❌ Reading the two `exploratory` rows as confirmatory. One of them already
  reversed.
- ❌ Treating `zero_dice` and `empty_pred` as the same number anywhere.

### 7g.8 Where Stage P's numbers live

`LESION_SIZE_RESULTS/` at the bundle root — never `results/`, `FINAL_RESULT/` or
`_work/runs/`, the same isolation Stages M and N use.

| file | what it is |
|---|---|
| `lesion_size_contrasts.csv` | **the answer** — the pre-registered family |
| `lesion_size_power.csv` | minimum detectable effect per endpoint |
| `lesion_size_headline.csv` | one row per model: whole set, D1–D4, D1 |
| `lesion_size_by_decile.csv` | one row per (model, decile) |
| `lesion_size_marginal_ci.csv` | per-model CIs in the primary stratum |
| `lesion_size_contrasts_d1.csv` | the same on D1 alone — fragile, 19 images |
| `lesion_size_bins.csv` | the decile assignment, so the cut is reproducible |

### 7g.10 Fairness conditioned on lesion size — §8.4's arithmetic, finally done

§8.4 has stated since Stage D that any fairness claim not conditioned on bruise
size is measuring two things at once. **Nothing in the study had done the
arithmetic.** Stage P had already cut the deciles, so it cost nothing.

**Part 1 — the confound is real, and it is large.**

| ITA group | images | subjects | median GT px | **share in D1–D4** |
|---|---|---|---|---|
| Light (II-III) | 39 | 9 | 8 085 | **0.59** |
| Intermediate (III-IV) | 38 | 17 | 12 118 | 0.34 |
| Tan (IV) | 24 | 15 | 13 550 | 0.33 |
| Brown (V) | 29 | 12 | 10 961 | 0.41 |
| Dark (VI) | 55 | 15 | 13 751 | 0.33 |

**Light-skin images are 1.8× more likely to be small-lesion images than
dark-skin ones** (0.59 vs 0.33), and their median bruise is 40% smaller. Since
size is the strongest predictor of failure in this study (§7g.1), **every
unconditioned per-group number in §8.3's D5 heatmap is confounded, and this table
is the size of it.** That is no longer a caveat; it is a measurement.

**Part 2 — conditioning does NOT shrink the gap. In four of five models it grows.**

Best-minus-worst group gap in small-lesion recall, marginal vs conditioned:

| model | all | **small (D1–D4)** | large | |
|---|---|---|---|---|
| segformer_b0_direct | 0.078 | **0.142** | 0.104 | grows |
| segformer_b0_distilled | 0.075 | **0.167** | 0.102 | grows |
| segformer_b2_teacher | 0.109 | 0.089 | 0.128 | shrinks |
| **yolo_sem_direct** | 0.220 | **0.341** | 0.196 | grows |
| **unet_r50** | 0.216 | **0.312** | 0.142 | grows |

**§8.4's hypothesis is not confirmed.** It anticipated that the apparent
skin-tone gaps were partly a size artefact and would shrink once size was held
fixed. They do the opposite: conditioning on size makes the gaps *larger*, in four
of five models. Whatever is happening in the per-group numbers, the size confound
is not the explanation for it.

**Part 3 — the direction is not the one anyone assumed.**

Worst and best group in the small-lesion stratum, per model:

| model | worst group | best group |
|---|---|---|
| segformer_b0_direct | Brown (V) 0.700 *(3 subj)* | Intermediate 0.842 |
| segformer_b0_distilled | Tan (IV) 0.684 *(4 subj)* | Intermediate 0.851 |
| segformer_b2_teacher | Brown (V) 0.690 *(3 subj)* | Intermediate 0.779 |
| **yolo_sem_direct** | **Light (II-III) 0.462** *(7 subj)* | **Dark (VI) 0.804** |
| **unet_r50** | **Light (II-III) 0.536** *(7 subj)* | **Dark (VI) 0.849** |

**Dark (VI) is never the worst group in any model, and is the *best* group in
two.** This is consistent with the study's existing position (20 of 21 ITA tests
non-significant; the one that is has its worst group at Tan, §7f.2) and it should
be stated plainly rather than left implicit: **on this dataset the per-group
differences do not run light-to-dark.**

**The one fairness signal that is both measurable and reproducible runs the other
way.** `yolo_sem_direct` and `unet_r50` — the two models with the most complete
misses — have their worst small-lesion recall at **Light (II-III)**, with gaps of
0.34 and 0.31, on **7 subjects and 23 images**, which is above the CI threshold.
§8.4 previously said the YOLO Light-skin gap *"should not be reported as a
skin-tone effect without the size-stratified analysis."* **That analysis has now
run, and the gap survives it and grows.** That does not make it a confirmed
effect — see the limits below — but it does retire the size explanation, and it
is the only directionally consistent per-group signal in the study.

**Part 4 — the limits, which are severe.**

- **10 of 25 small-stratum cells carry no interval at all** (60% coverage). Brown
  has **3** subjects and Tan **4**; `MIN_SUBJECTS_FOR_CI = 5` refuses to
  bootstrap them, because a cluster bootstrap over three clusters is arithmetic,
  not evidence. For the three SegFormers the worst group is *always* one of those
  two — so their gaps are anchored on cells nobody can measure.
- **The gap is max-minus-min over five noisy estimates and is biased upward.** It
  is descriptive. A gap is not a test, and none of these has been tested.
- Nothing here is corrected for multiplicity across 5 models × 5 groups.

**What this licenses, and what it does not.**

- ✅ Reporting Part 1 as a measured property of the dataset: **every
  unconditioned per-group number in §8.3's D5 heatmap is confounded by lesion
  size, by this much.** That is now a measurement, not a caveat.
- ✅ Retiring the size explanation for the YOLO/U-Net Light-skin gap (§8.4).
- ✅ Stating that the differences do not run light-to-dark on this dataset.
- ❌ Any conditioned per-group *claim*, in any direction. Nothing here was tested.
- ❌ Anything at all about Brown (V) or Tan (IV): 3 and 4 subjects.

**The fix is subjects, not analysis.** Brown and Tan need roughly double their
current subject count before the cell they anchor can be filled in. That is a
concrete data-collection requirement, and it is the most useful thing this
section produces.

Written to `lesion_size_ita_confound.csv` and
`lesion_size_fairness_conditioned.csv`.

### 7g.9 ⏸ What the ORC outage does and does not block (2026-08-07)

The GPU cluster has not spun up. That is not a general stop, and it is worth
being precise about which work it actually holds, because the CPU-only half of
this project is large and is where the next result is.

**NOT blocked — runnable on a laptop today:**

| work | why it runs offline |
|---|---|
| **Stage P itself** (§7g) | per-image CSVs only; the dry run in §7g.6 already happened |
| Stage D tables, Stage G significance | §8 — functions of per-image Dice, no weights |
| The size-conditioned fairness re-analysis (§8.4) | same inputs; the one fairness item worth doing (§18) |
| Fenwick label prep, size distribution, inter-labeler agreement | mask arithmetic, no model |
| Everything in `PROJECT_STORY.md` | prose against numbers already on disk |

**Blocked until the cluster returns:**

| work | why it needs a GPU |
|---|---|
| Stage N `resnet50_probe_hires` (§7f.9) | a forward pass through ResNet-50 |
| Stage N seg arms (§7f) | training |
| ALS → white-light distillation | training, and the paired-visibility audit needs the images |
| Any Fenwick retrain | training |

**So the plan while the cluster is down**, in order:

1. **Confirm Stage P offline.** The notebook runs on the laptop against the local
   `RESULT_AUGUST_08/`. If §2's printed lineage path matches, the dry run *is*
   the run and §7g.6 can be promoted from dry-run to confirmed without ORC. The
   only thing an ORC run adds is a check that the two trees have not drifted.
2. **Do the size-conditioned fairness re-analysis** (§8.4, §18). Free, uses
   Stage P's decile assignment, and it either kills the fairness question
   honestly or finds a real effect under the size confound. This is the one
   fairness item this project should still spend time on.
3. **Prepare Fenwick without training anything** — pull masks, measure the bruise
   size distribution, compute pairwise labeler agreement on the 128-image
   white-light three-way intersection. If Fenwick has no small lesions it cannot
   help the D1–D4 stratum, and that is a five-minute answer worth having before
   the cluster returns.
4. ~~**Queue Stage N2 (§7h) as the first GPU job.**~~ **DONE 2026-08-09.** Both
   experiments ran. The gate closed (`dermlip − dinov2 = −0.0913`, INFERIOR), the
   grid confound turned out not to exist, and §7f.8's headline lost 74 % of its
   magnitude to a probe-layer artefact. §7h.7–§7h.9. **The follow-on item is
   documentary, not computational:** §7f.8's number appears in any figure, slide
   or draft written between 2026-08-06 and 2026-08-09 and every one of them needs
   the corrected +0.138.
5. ~~**Queue Stage N3 (§7i) as the next GPU job.**~~ **RUN 2026-08-10 — §7i.7.**
   `dinov2_ft` 0.7902 test, 0/185 misses, the study's highest Dice — and
   inconclusive against B2 and B0, with the pre-registered 0.79 trigger cleared
   by 0.0002. **The follow-on is two more seeds (~5 GPU-hours).** One seed cannot
   distinguish "the ceiling is 0.79 rather than 0.78" from a lucky draw, and that
   distinction is the whole content of the result. Everything else about Stage N3
   is already settled: the probe ranking transferred (−0.0913 → −0.0859), so
   §7h.9's open question is closed and dermatology pretraining has no remaining
   escape route.

The one thing not to do is wait. Items 1–3 are the whole of the next result and
none of them touches a GPU. Item 5 is now a 5-GPU-hour confirmation rather than a
question, and **item 3 is what it hands off to**: Stage N3 did not raise the
ceiling meaningfully, so §7i.6's chain holds — the remaining lever is **label
quality**, and item 3 is the measurement that tells us whether Fenwick can supply
it. The Fenwick labeler CV (`bruisekit/fenwick_cv.py`, built 2026-08-10) is the
first half of that measurement.

---

## 7h. Stage N2 — the grid control, and *dermatology* pretraining

> **STATUS 2026-08-09 — RUN AND MEASURED. §7h.7 is the result, §7h.8 is what it
> does to §7f.8, and §7h.9 is what it does not license.**
>
> **The gate CLOSED, and hard.** `dermlip − dinov2 = −0.0913`,
> CI [−0.1208, −0.0544], p = 0.0001, verdict **INFERIOR**. A dermatology-pretrained
> ViT-B/16 produced *worse* frozen features than an identically-sized
> self-supervised natural-image encoder, at a matched grid.
>
> **The grid confound does not exist, and runs the other way.** −0.101 Dice per
> doubling of grid, CI [−0.128, −0.074], pretraining held fixed.
>
> **But the real confound was never the grid — it was the ResNet LAYER, and it
> accounts for ~74 % of §7f.8's headline.** Same weights, same 20 × 20 grid,
> `layer3` scores 0.557 against `layer4`'s 0.124. §7f.8 must be rewritten before
> it is quoted anywhere; §7h.8 has the corrected number.

**Where it lives, and why not in the bundle build.** `bruisekit/dermprobe.py` and
`bruise_derm_probe.ipynb`, shipped by `77b_zip_derm_probe.py`, generated by
`77_generate_derm_probe_notebook.py`. Deliberately absent from
`60_build_unified_bundle.py`'s copy list for the reason `multiteacher.py` and
`foundation.py` are (§7e, §7f): it has no source in `scripts/unified_lib/`, so a
rebuild neither regenerates nor deletes it, and a fourth `resolve_micro_batch`
patch stays out of the notebook that produces Stages A–Y. Results go to
`DERM_PROBE_RESULTS/`, never `FOUNDATION_RESULTS/` — Stage N's numbers are next
door and are not overwritten.

### 7h.1 The two things wrong with §7f.8

**1. The grid confound (§7f.9), unclosed.** The three Stage N arms scored in
exactly the order of their feature-grid size. A linear probe is a 1×1 convolution
*on that grid*, so a finer grid is worth Dice independently of what the encoder
knows.

**2. MedSigLIP is not a dermatology model.** It is a general-medical
vision-language encoder — radiology, histopathology, ophthalmology, with
dermatology as one slice. Our images are consumer-camera photographs of skin.
*"Medical pretraining does not help on bruises"* was never tested; what was
tested is *"this particular general-hospital encoder does not help"*, which is a
much weaker sentence than the one §7f.8 writes.

### 7h.2 Experiment 1 — grid calibration

One encoder (ImageNet ResNet-50), one stage (`layer3`, stride 16), **four input
sizes**. Identical weights, identical depth, identical recipe — only the grid
moves, so whatever Dice moves *is* the grid.

| arm | stage | input | grid |
|---|---|---|---|
| `rn50_l3_g20` | layer3 | 320 | 20 × 20 |
| `rn50_l3_g28` | layer3 | 448 | 28 × 28 |
| `rn50_l3_g37` | layer3 | 592 | 37 × 37 |
| `rn50_l3_g40` | layer3 | 640 | 40 × 40 |
| `rn50_l4_g20` | layer4 | 640 | 20 × 20 — **reproduces Stage N's arm** |

This generalises §7f.9's proposed `layer3` control from one point to a curve, and
adds the pair that needs no curve at all: **`rn50_l4_g20` vs `rn50_l3_g40` is the
same weights on the same 640 input with only the stride changed.** Whatever
separates those two is the grid, full stop.

Run alongside a **pixel floor** — a 1×1 convolution on raw RGB average-pooled to
the same four grids, no encoder, no pretraining. It bounds what a grid buys with
zero learned features, and it is what makes the ResNet slope interpretable rather
than merely suggestive: if the floor tracks the slope, the grid effect is
*geometry*, not features.

Nine arms, ~40 GPU-minutes. `grid_calibration()` fits Dice against `log2(grid)`
with a subject-cluster bootstrap and reports the slope per doubling, then the
20→37 span — the span §7f.8's `dinov2 − resnet50 = +0.534` was measured across.

#### 7h.2a THE CALIBRATION RESULT — measured 2026-08-09

134 validation images, 20 subjects, seed 0, frozen encoder, linear head.

| arm | grid | val AP | val Dice | misses |
|---|---|---|---|---|
| `rn50_l3_g20` | 20 × 20 | 0.634 | **0.5568** | 2 |
| `rn50_l3_g28` | 28 × 28 | 0.589 | 0.5264 | 3 |
| `rn50_l3_g37` | 37 × 37 | 0.514 | 0.4758 | 1 |
| `rn50_l3_g40` | 40 × 40 | 0.483 | 0.4457 | 1 |
| `rn50_l4_g20` | 20 × 20 | 0.068 | **0.1242** | 5 |
| `pixel_g20` | 20 × 20 | 0.088 | 0.1417 | 15 |
| `pixel_g28` | 28 × 28 | 0.086 | 0.1385 | 14 |
| `pixel_g37` | 37 × 37 | 0.086 | 0.1388 | 15 |
| `pixel_g40` | 40 × 40 | 0.085 | 0.1473 | 22 |

Fitted on subject-level means, 10 000 subject-cluster resamples:

| family | slope, Dice per **doubling** of grid | CI 95 % |
|---|---|---|
| ResNet-50 `layer3` | **−0.1005** | [−0.1278, −0.0737] |
| pixel floor | −0.0036 | [−0.0260, +0.0161] |

> The arm table above is the **image-level** mean (`calibration_val_summary.csv`);
> the slope is fitted on **subject-level** means, because that is the bootstrap
> unit. The two differ by up to 0.012 on the same arm — subjects contribute
> unequal image counts. Neither is wrong; do not mix them in one sentence.

**§7f.9's confound does not exist, and the sign is the opposite of the one
feared.** A finer feature grid is worth *less* Dice, not more, and the interval
excludes zero. The mechanism is not mysterious: bruises are large connected
regions, and a coarse grid is a smoothness prior on a constant-per-cell mask. The
pixel floor is flat across the same span, so with no features at all the grid does
essentially nothing — which is what makes the ResNet slope attributable to how the
grid interacts with real features rather than to geometry.

#### 7h.2b THE FINDING THE STAGE WAS NOT LOOKING FOR — it was the layer

The pair that holds the grid fixed and changes only depth:

| arm | grid | val Dice |
|---|---|---|
| `rn50_l3_g20` — layer3, stride 16 @ 320 | 20 × 20 | **0.5568** |
| `rn50_l4_g20` — layer4, stride 32 @ 640 | 20 × 20 | **0.1242** |

**Same network, same weights, same 20 × 20 grid, +0.4326 Dice.** (The
same-input version, `l4 @ 640` against `l3 @ 640`, is +0.3215 and is the number
`grid_calibration` reports as `stride_only_delta`; the equal-grid pair above is
the cleaner statement and the one to quote.)

`rn50_l4_g20` scores **0.1242 against the raw-pixel floor's 0.1417** — an ImageNet
ResNet-50's final-stage features, probed linearly, carry *less* usable
bruise-location signal than raw downsampled RGB. That is not a bug: post-ReLU
`layer4` is the most ImageNet-class-semantic representation in the network, and
"where is the discoloured region" is not an ImageNet class.

Stage N's `resnet50_probe` used `layer4` and scored **0.1225**. It reproduces here
to 0.002, so nothing in either run is broken. The number was right and the
*reading* of it was wrong.

### 7h.3 Experiment 2 — the corpus arms, all at 28 × 28

The nuisance variable is **pinned by construction**. `corpus_gate` raises if any
arm is not at 28 × 28, because the stage's only claim is a comparison at a fixed
grid.

> **⚠️ The stated reason for choosing 28 was wrong, and the choice was still
> fine.** Earlier revisions of this section and of §7f.3 said 28 is MedSigLIP's
> native 448/16, making it the one arm needing no resample. **MedSigLIP is
> SoViT-400m/14, patch 14, not 16.** The run proves it: `medsiglip_g28` reports
> `input = 392`, i.e. 28 × 14. Its native grid at 448 is **32 × 32**, so §7f.9's
> confound table understated it. Consequences: (a) *every* ViT arm here was
> resampled, none was native — state that in the limitations; (b) §7f.9's
> rank-correlation claim is unaffected, since 37 / 32 / 20 is still monotone with
> the score order; (c) the 28 × 28 target itself is arbitrary but applied
> identically to all six arms, which is all the design requires.

| arm | encoder | corpus | licence |
|---|---|---|---|
| `dermlip_g28` | ViT-B/16 | **Derm1M** — 1.03 M dermatology image-text pairs over 403 k images | CC-BY-NC-ND-4.0 |
| `dermlip_panderm_g28` | ViT-B/16 | **PanDerm** — >2 M skin images self-supervised, then Derm1M | CC-BY-NC-ND-4.0 |
| `biomedclip_g28` | ViT-B/16 | PMC-15M — biomedical *figure* captions | MIT |
| `dinov2_g28` | ViT-B/14 | LVD-142M natural images, self-supervised | Apache-2.0 |
| `medsiglip_g28` | SoViT-400m | general medical image-text | HAI-DEF, restricted |
| `rn50_g28` | ResNet-50 layer3 | ImageNet-1k supervised | BSD-3 |

**Three arms are ViT-B/16 at 224 native, same loader, same resample.** So
`dermlip − biomedclip` varies the corpus and nothing else — skin photographs
against journal figures at identical capacity, patch size and grid. That is a
cleaner contrast than anything Stage N could form, where every arm differed in
size, patch, native resolution and vendor at once.

The three Stage N controls are **re-run here** rather than quoted from
`FOUNDATION_RESULTS/`, because those numbers were taken at three different grids.
Comparing a new 28 × 28 arm against a stored 37 × 37 one would reproduce the exact
confound the stage exists to close.

**`dermlip_panderm_g28` did not run** — see §7h.7. Five arms carry the result.

**Where the numbers live.** `DERM_PROBE_RESULTS/` at the bundle root:
`grid_calibration.json` (§7h.2a–b), `calibration_val_summary.csv`,
`corpus_val_summary.csv`, `gate.json` (§7h.7), `derm_probe_misses.csv`,
`corpus_failed_arms.json`, `val_per_image__<arm>.csv` for every contrast, and
`runs/<arm>__seed0/` with each arm's `operating_point.json` and `val_sweep.csv`.
Nothing was written to `results/`, `FINAL_RESULT/`, `FOUNDATION_RESULTS/` or
`_work/runs/`.

### 7h.4 The gate

Fixed in `dermprobe.py` before any number was produced:

```
open  iff  the (dermlip_g28 − dinov2_g28) val-Dice CI clears zero
```

`dermlip_g28` is the treatment **by name**; `dermlip_panderm_g28` is secondary
and cannot be promoted afterwards. That substitution is how a two-arm experiment
becomes a one-arm experiment with two chances, and §15 trap 3 already refuses the
equivalent move on seeds.

Reported alongside, never ANDed in: `vs_medical` (the contrast Stage N thought it
was running), `vs_biomed` (the architecture-matched one), `vs_imagenet` (ties back
to Stage B), and misses.

**A closed gate is the deliverable, and it is a stronger claim than §7f.8's:**

> *Frozen features from a ViT-B/16 pretrained on 1.03 M dermatology image-text
> pairs do not outperform the same-capacity self-supervised natural-image encoder
> on bruise segmentation, at a matched 28 × 28 feature grid.*

That names a real dermatology corpus, holds architecture and grid fixed, and
cannot be explained by resolution. §7f.8's version has none of those three
properties.

### 7h.5 `google/derm-foundation` — an explicit gap, not an omission

By corpus it is the best-targeted model in existence for this task: BiT-M
ResNet-101×3 trained on teledermatology photographs from consumer cameras. **It
cannot be probed.** It ships as a TensorFlow/Keras SavedModel whose only exported
signature returns a single **6144-dimensional embedding per 448 × 448 image** —
no token grid, no intermediate feature map, nothing for a 1×1 convolution to sit
on. A probe of it would predict a constant mask, and its Dice would measure the
dataset's mean bruise area.

It is registered in `SOURCES` with `spatial=False` so it appears in the
provenance table, the same treatment §5 gives nnU-Net and for the same reason:
*"the model we could not run"* is exactly the fact a reader needs.
`dermfoundation_tiled_note()` describes the one honest workaround — a sliding
448 window on a K×K lattice, ~53 k forwards of a 380 M CNN for an **8 × 8** grid.
**§7h.2a now makes that decisively not worth doing, and for the opposite reason to
the one first given:** the original argument was that an 8 × 8 grid is too coarse,
and the calibration says a coarser grid is if anything an *advantage*. The real
disqualifier is §7h.2b — the probe's outcome is dominated by *which representation*
you read, and a pooled embedding head is the most global one available. Cite the
architecture, do not run it.

**MedImageInsight is absent for a different reason.** It is a DaViT on broad
medical imaging — the same question MedSigLIP already answers, through a
different vendor's code. A replication, not a new contrast.

### 7h.6 Two limitations owed regardless of the outcome

**1. Position embeddings are resampled.** §7f.6 took the opposite decision — run
each encoder at its native resolution, never interpolate — and that decision is
precisely what produced the confound. You cannot have both *every encoder at its
native grid* and *every encoder at the same grid*. This stage picks the second,
because the question is about the corpus and the grid is the nuisance variable.
Resampling is applied **identically to every ViT arm** — and, per §7h.3's
correction, that really is *every* arm: MedSigLIP's patch is 14, so even it was
moved off its native 32 × 32.

**2. The protocol still favours DINOv2** — and §7h.9 item 2 upgrades this from a
footnote to the stage's main threat. Frozen-feature linear probing is the headline
evaluation in DINOv2's own paper, *and* DINOv2 is the only arm here whose
pretraining objective operates on patches rather than on a pooled vector. A
matched grid does not fix either, and nothing in this stage claims to.

**And the licence.** *Both* dermatology encoders are non-commercial
(CC-BY-NC-ND-4.0 and CC-BY-NC-4.0), against DINOv2's Apache-2.0 and BiomedCLIP's
MIT. §7b.1 chose DeepLabV3+ over SegFormer specifically to escape a
non-commercial licence. **If a derm arm ties DINOv2, the licence decides** — and on
the measured result it does not tie, so the licence never has to arbitrate:
the Apache-2.0 encoder is also the better one.

### 7h.7 THE GATE RESULT — measured 2026-08-09

134 validation images, 20 subjects, seed 0, every encoder **frozen**, linear head,
**every arm at 28 × 28**, 10 000 subject-cluster bootstrap resamples.

| arm | corpus | encoder | input | val AP | val Dice | median | IQR | misses |
|---|---|---|---|---|---|---|---|---|
| `dinov2_g28` | natural images, self-supervised | 86.6 M | 392 | **0.7481** | **0.6635** | 0.7178 | 0.282 | 3 |
| `biomedclip_g28` | biomedical figures, image-text | 86.6 M | 448 | 0.7268 | 0.6218 | 0.6441 | 0.278 | 2 |
| `dermlip_g28` | **dermatology, image-text** | 86.6 M | 448 | 0.7184 | 0.5721 | 0.6203 | **0.341** | 3 |
| `rn50_g28` | ImageNet-1k, supervised | **8.5 M** | 448 | 0.5878 | 0.5252 | 0.5430 | 0.311 | 3 |
| `medsiglip_g28` | general medical, image-text | **428.6 M** | 392 | 0.5538 | 0.4670 | 0.4641 | 0.329 | 3 |

| contrast | Δ Dice | CI 95 % | p | verdict |
|---|---|---|---|---|
| **`dermlip − dinov2` (PRIMARY)** | **−0.0913** | **[−0.1208, −0.0544]** | 0.0001 | **INFERIOR** |
| `dermlip − medsiglip` | +0.1052 | [+0.0685, +0.1430] | 0.0001 | WIN |
| `dermlip − biomedclip` | −0.0496 | [−0.0882, −0.0059] | 0.0218 | INCONCLUSIVE |
| `dermlip − rn50` | +0.0469 | [+0.0055, +0.0988] | 0.0270 | WIN |

**`GATE_run_seg_arms = false`.** Dermatology pretraining is not merely *no better*
than generic self-supervised pretraining on this task — it is **measurably worse**,
with the interval nowhere near zero. ~20 GPU-hours saved and the null is
publishable as it stands.

**Three checks that make the ordering trustworthy.**

1. **Threshold-free AP gives the identical ranking.** Nothing here is an
   operating-point artefact.
2. **Two of the three ViT-B/16 arms are matched to the patch** — `dermlip` and
   `biomedclip` differ in corpus and nothing else.
3. **The 8.5 M ImageNet ResNet trunk beats the 428 M medical encoder by 0.058.**
   A 50× parameter difference in the wrong direction, consistent with §7f.5's
   scaling prior and with Stage N's own finding that capacity is not the variable.

**Misses say nothing here** — 2–3 of 134 on every arm, Δ = 0.0, CI spanning zero.
Expected for frozen probes and not a result in either direction.

**`dermlip_panderm_g28` did not run, and the guard is why.**
`redlessone/DermLIP_PanDerm-base-w-PubMed-256` builds a stock open_clip ViT-B/16
from its own config, but **152 of that tower's parameters are absent from the
checkpoint** (`class_embedding`, `positional_embedding`, `conv1.weight`, `proj`) —
its PanDerm backbone uses different key names. The loader raised rather than train
a randomly-initialised encoder. This is exactly the §7f.7a failure, caught this
time. The failure is recorded in `corpus_failed_arms.json`; the arm was secondary,
so the primary contrast is unaffected. **Report it as an arm that did not run.**

### 7h.8 What this does to §7f.8 — the number changes by 74 %

§7f.8 reports `dinov2 − resnet50 = +0.5342` and attributes all of it to
pretraining. Its ResNet arm was probed at **`layer4`** and at a 20 × 20 grid.

| | Δ Dice |
|---|---|
| published (§7f.8): `dinov2` (37 × 37) − `resnet50` (layer4, 20 × 20) | **+0.5342** |
| matched grid, ResNet at `layer3`: `dinov2_g28 − rn50_g28` | **+0.1383** |
| **share of the published gap that was the layer choice** | **≈ 74 %** |

The `stage_n_reinterpretation` block in `gate.json` answers only the
**pre-registered** question — how much the *grid* explains — and its answer is
`grid_share = −0.15`, i.e. **the grid explains none of it and slightly the wrong
way**. The layer does. That comparison was in the design (§7h.2's arm table lists
`rn50_l4_g20` as "reproduces Stage N's arm"), but its being the *dominant* effect
was not anticipated, and it is not automated in `corpus_gate`. Compute it from the
two tables above.

**What §7f.8 may still say:** modern self-supervised ViT features beat ImageNet
ResNet-50 features on this task, by **+0.138** at a matched grid and a sensible
probe layer. **What it may no longer say:** *dramatically* better, +0.534, or
anything resting on ResNet-50 scoring near the pixel floor — that was `layer4`,
not ImageNet.

### 7h.9 What this does NOT license — read before writing any of it up

**1. "The more medical the pretraining, the worse" is over-reading.** The
descriptive ordering says it, the mechanism does not support it, and inside the
image-text family the ordering is not monotone in medical specificity anyway
(biomedical figures 0.622 > dermatology 0.572 > general medical 0.467). Claim only
what the pre-registered contrast tested: **dermatology pretraining did not beat
generic self-supervised pretraining under this protocol.**

**2. The winning arm has a training objective this protocol rewards.** DINOv2 is
the only arm here with a **patch-level** pretraining objective — its patch tokens
were explicitly optimised to be informative. `dermlip`, `biomedclip` and
`medsiglip` are all CLIP/SigLIP-style: one pooled vector per image against a
caption, with no term that makes individual patch tokens locally discriminative.
A *dense* linear probe is close to DINOv2's training objective and far from
theirs. This is a strictly stronger version of §7f.9's second point, it survives
the grid fix completely untouched, and **it is the single biggest threat to this
stage's conclusion.** State it in the limitations without being asked.

**3. `dermlip − biomedclip` is not a clean loss.** The verdict is INCONCLUSIVE,
and the *AP* gap (−0.009) is a fifth of the *Dice* gap (−0.050) — so most of
DermLIP's Dice deficit against BiomedCLIP is threshold calibration, not feature
quality. Its selected cut is **−1.75**, far off every other arm's ≈ −0.6 to −0.9,
and its IQR is the widest in the pool at 0.341. Report "dermatology ≈ biomedical
figures", never "worse".

**4. A frozen probe ranks features; it does not predict fine-tuned models.**
§7f.4 said so when the method was chosen and it is still true. The counter-argument
worth having in writing: the fine-tuned field in this study spans **~0.02** Dice
(B0 0.766 / B2 0.769 / B5 0.773 / U-Net 0.757 / DeepLab 0.758) against the probe's
**~0.20**, because the annotation ceiling (§1) eats the signal. So the probe is the
*less realistic* measurement and simultaneously the *only one with enough dynamic
range to separate encoders at all*. That is the honest defence of the gate, and it
is a better one than "the probe is a good proxy", which it is not.

**If the fine-tuned answer is wanted anyway** — a legitimate ask — the cheap
version is **two arms, one seed**: `dermlip_seg` against `dinov2_seg`, last six
blocks unfrozen, ~4–5 GPU-hours rather than 20. Pre-register the reading first:
*if both land inside 0.75–0.78, the probe ranking did not transfer AND neither
measurement separates the encoders, which is itself the answer.*

---

## 7i. Stage N3 — the annotation ceiling on a third axis

> **STATUS 2026-08-10 — RUN. See §7i.7 for the result and §7i.7a for the void
> gate file.** `dinov2_ft` 0.7902 on test with 0 complete misses — the highest
> Dice in the study — but inconclusive against B2 and B0, and the pre-registered
> 0.79 trigger cleared by 0.0002. The frozen-probe ranking transferred almost
> exactly (−0.0913 → −0.0859), which answers §7h.9.
>
> **§7i.1–7i.6 below are the pre-registration, written before the numbers
> existed and left unedited.** Read them first; that is the point of them.

### 7i.1 The question, and why the existing ceiling evidence is not enough

§1's annotation ceiling is the load-bearing claim of this study: model choice is
exhausted, and what remains is label quality. The evidence for it is strong but
it varies **only two things** — capacity and architecture:

| | Dice | | Dice |
|---|---|---|---|
| segformer_b0 (3.71 M) | 0.7663 | unet_r50 | 0.7570 |
| segformer_b2 (27.35 M) | 0.7692 | deeplabv3plus_r50 | 0.7580 |
| segformer_b5 (~85 M) | 0.7727 | Friedman, seven models | **p = 0.61** |

23× the parameters bought **+0.006 Dice**, and the headline omnibus does not
reject (§8b). But every arm in that table is **ImageNet-supervised or scratch**.
A sceptic's reading is available and it is not silly: *these models all land in
the same place because they are all the same kind of model.*

Stage N2 supplies the missing control. Its frozen probe measured DINOv2's
features at **0.6635** against ResNet-50's **0.5252** — a 0.14 gap in linear
decodability at a matched 28×28 grid (§7h.7). Whatever else is true, DINOv2 is
the one encoder in this study *known* to carry different information.

So: **fine-tune that encoder and see where it lands.** If a demonstrably
different feature space also arrives at ~0.77, the ceiling is binding across
capacity, architecture **and** pretraining paradigm, and the sceptic's reading is
closed off. That is a claim about where the next six months go, which is why it
is worth 4–5 GPU-hours.

It also settles §7h.9's open question for free: **did the frozen-probe ranking
transfer to fine-tuning?**

### 7i.2 Design — two arms, one seed, everything else held fixed

| arm | corpus | why it is here |
|---|---|---|
| `dinov2_ft` | natural images, self-supervised | the paradigm control — zero medical data, patch-level objective |
| `dermlip_ft` | dermatology image-text | did §7h.7's −0.0913 survive once patches can adapt? |

Three design decisions carry the validity of the result:

**1. Both arms at a 40×40 grid, decoding to 160×160.** DINOv2 is patch-14
(40 × 14 = 560 px); DermLIP is patch-16 (40 × 16 = **640 px**, the pipeline's own
`img_size`, so that arm is never resampled). The decode head reaches stride 4
relative to the encoder grid, which is **the same output stride SegFormer
produces at 640**. Had the head been left at the encoder grid and bilinear-
upsampled 16×, a low Dice would be a boundary-resolution artefact and would be
*indistinguishable from the ceiling result this stage is testing for*. §7h.2a
already showed this study's intuitions about grid effects are unreliable; the fix
is to remove the variable, not to reason about it.

**2. Training goes through `engine.train_run` unmodified.** Same optimiser, LR
split, warmup, early stopping, augmentation and resume contract that produced
segformer_b0's 0.7663 — `kd_core.DEFAULTS` verbatim. A bespoke loop would make
*"did it land in the band?"* unanswerable, because any difference could always be
the recipe rather than the encoder.

**3. The head is a real decoder, not a linear probe.** §7f.4's linear constraint
exists so a strong head cannot paper over a weak encoder. Here the encoder
adapts, so the opposite constraint applies: the head must be a *fair* decoder.
`ConvDecodeHead` is two conv stages, ~1.1 M parameters against an 86 M encoder —
strong enough to be fair, small enough not to be the story.

### 7i.3 The pre-registration

Fixed in `bruisekit/finetune_n3.py` before any number is produced:

```
CEILING_BAND    = (0.75, 0.78)     the span the seven headline models occupy
UNFREEZE_BLOCKS = 6                last six transformer blocks + final norm
TARGET_GRID     = 40               both arms
SEEDS           = (0,)             screening run
```

| outcome | reading |
|---|---|
| **both arms in 0.75–0.78** | **CEILING CONFIRMED** on a third axis. The frozen-probe ranking did not transfer. Encoders are exhausted as a lever; the bottleneck is label quality. |
| **either arm > 0.79** | **Ceiling NOT binding.** Better features do buy Dice here, §7h's null needs reinterpreting, and encoders are live again. |
| **either arm < 0.73** | **INCONCLUSIVE.** One seed cannot separate a weak encoder from a collapsed run — Stage Y's seed 2 did exactly this. Inspect the loss curve and re-run that arm on another seed before reading anything into it. |

**Misses and IQR are reported alongside Dice, not after.** Dice is saturated;
complete misses are where this study's models actually separate (§4, and the
Friedman result above). DINOv2 landing at 0.77 *while moving misses* is a real
finding that a Dice-only readout would discard. `ceiling_gate` raises rather than
print a verdict without the miss column, and it counts misses as `dice == 0` —
the published definition, not the per-seed empty-prediction count (§4 trap).

CIs are **subject-clustered** bootstrap over 10 000 resamples, the same policy as
Stages D, G and N2. `ceiling_gate` raises if the frame has no `subject` column
rather than silently reporting an image-clustered interval that is too narrow.

### 7i.4 The one failure that would fake a confirmed ceiling

An arm that silently unfreezes **nothing** trains like a frozen probe, scores
low-to-mid, and looks exactly like the result this stage hopes to see. It would
confirm the hypothesis for entirely the wrong reason, and — unlike §7f.7a's
silent random-init — there is no parameter count that gives it away.

Three independent guards, because one is not enough for a failure that
*confirms* rather than breaks:

1. `find_blocks` handles the three tower layouts in this study's checkpoints
   (`encoder.encoder.layer`, `trunk.blocks`, `transformer.resblocks`) and
   **raises** on anything else rather than returning empty.
2. `unfreeze_last` raises if the trainable parameter count comes back zero.
3. Notebook cell 5 builds each arm and asserts a non-zero trainable fraction
   **before any training starts**. It should read ≈ 45 % for six of twelve blocks.

### 7i.5 Where it lives, and why not in the bundle

`bruisekit/finetune_n3.py` and `bruise_stage_n3.ipynb`, shipped by
`78b_zip_stage_n3.py`, generated by `78_generate_stage_n3_notebook.py`. The
module is **deliberately absent from `60_build_unified_bundle.py`'s copy list**,
the same policy as `foundation.py`, `dermprobe.py`, `multiteacher.py` and
`lesionsize.py` (§7e, §7f): an experiment that may return nothing must not become
a dependency of the file that produces Stages A–Y. The zip script fails to build
if that has changed, because the overlay would then ship a stale duplicate.

It **imports** `dermprobe.py` for encoder loading — including that module's guard
against the §7f.7a failure — and deliberately does not ship a second copy of it.

At runtime everything goes to `STAGE_N3_RESULTS/`; training writes to
`STAGE_N3_RESULTS/runs`, **not** `env.runs`, so `Registry` cannot pick these arms
up and inject them into a table they are not part of.

### 7i.6 What a confirmed ceiling will and will not license

**What it licenses.** *A pretraining paradigm known to carry different
information lands in the same 0.75–0.78 band as everything else. Encoder choice
is exhausted as a lever on this task.*

**What it does not.** It does not say encoders never matter — it says they do not
matter *here, at this label noise level*. And it is one seed: an arm inside the
band is **consistent with** the ceiling, not a tight interval around it.

**And one correction that must not be skipped.** The tempting next sentence is
*"so add more data."* That is wrong, and the error is worth stating in the paper
rather than discovering in review:

> **More images labelled to the same standard do not raise an annotation
> ceiling.** The ceiling is set by label *noise*, not by data volume. Another
> 2 500 single-annotator images adds more data at the same noise level and the
> asymptote does not move.

What moves it is **multiple independent annotations per image**, and the Fenwick
set is the only data in this project with that structure: three annotators
(`hliu36`, `mzehra2`, `nmousta5`) and **128 white-light images all three
annotated independently** — the strict intersection, provenance-verified against
project `cmnmn9zfm19ea07z17f9s9g90`. That supports two things nothing else here
can do:

1. **Measure the ceiling directly on our own data** — inter-rater Dice between
   the three, rather than quoting 0.581–0.873 from the literature.
2. **Reduce label noise** — train against consensus (union / majority /
   intersection) instead of one person's opinion. A consensus label is genuinely
   less noisy than a single annotation, and *that* moves the ceiling.

So the honest chain, and the one to write up, is: **ceiling confirmed on a third
axis → the remaining lever is label quality → Fenwick is the only dataset with
the structure to pull it.** Not "ceiling confirmed → add images."

### 7i.7 THE RESULT — run 2026-08-10, one seed

> **The ceiling did not close, and it did not break either.** `dinov2_ft` scored
> **0.7902** on test — the highest single number in this study — and **misses
> nothing**, 0 of 185. But it does not statistically separate from SegFormer-B2
> or B0, and the pre-registered "> 0.79 ⇒ ceiling not binding" trigger is cleared
> by **0.0002**. Treating that as a broken ceiling would be reading noise.

| arm | split | Dice | 95 % CI (subject-clustered) | median | IQR | misses | verdict |
|---|---|---|---|---|---|---|---|
| `dinov2_ft` | val (134) | 0.7824 | [0.7411, 0.8265] | 0.8386 | 0.2033 | 2 | NEAR_BAND |
| `dinov2_ft` | **test (185)** | **0.7902** | [0.7511, 0.8239] | 0.8305 | 0.1758 | **0** | ABOVE_BAND |
| `dermlip_ft` | val (134) | 0.7235 | [0.6791, 0.7804] | 0.7708 | 0.2001 | 2 | below band |
| `dermlip_ft` | **test (185)** | 0.7043 | [0.6702, 0.7385] | 0.7584 | 0.2326 | **0** | below band |

Cut fitted on val (`dinov2_ft` +1.475, `dermlip_ft` −1.400) and applied unchanged
to test — the §7f.4 rule, so test was never used to tune anything.

**Paired, subject-clustered, 10 000 draws, against the models it has to beat:**

| contrast | Δ Dice | 95 % CI | reading |
|---|---|---|---|
| `dinov2_ft` − `segformer_b5` seed123 (0.7727) | +0.0175 | [−0.0104, +0.0542] | **inconclusive** |
| `dinov2_ft` − `segformer_b2_teacher` (0.7692) | +0.0209 | [−0.0023, +0.0454] | **inconclusive** |
| `dinov2_ft` − `segformer_b0_direct` (0.7663) | +0.0238 | [−0.0055, +0.0514] | **inconclusive** |
| `dinov2_ft` − `segformer_b5` seed42 (0.7527) | +0.0374 | [+0.0099, +0.0650] | clears zero |
| `dinov2_ft` − `dermlip_ft` | +0.0859 | [+0.0425, +0.1281] | clears zero |

The only headline model it beats outright is B5's **worst** seed. Against every
model this study actually reports, the interval crosses zero. At 86.6 M
parameters `dinov2_ft` is capacity-matched to B5 (~85 M), so B5 is the honest
comparator and B0 is not — B0 gets within 0.024 with **4 % of the parameters**.

**Misses: no separation, because there is nothing left to separate.**
`dinov2_ft` 0/185, and so are `segformer_b2_teacher` and `segformer_b0_distilled`.
B0-direct misses 1, B5-seed123 misses 0. The endpoint §4 says decides has
saturated at the top of the table, which is itself worth saying: on this test set
the good models no longer miss injuries, and the remaining spread is boundary
agreement.

**§7h.9's open question is answered, and cleanly.** Did the frozen-probe ranking
transfer to fine-tuning?

```
frozen probe (§7h.7)   dermlip − dinov2 = −0.0913
fine-tuned (test)      dermlip − dinov2 = −0.0859
```

**Yes — almost exactly.** Six unfrozen blocks and a real decoder moved that gap
by 0.005. A cheap frozen probe predicted the expensive fine-tuned ranking, which
is a genuinely useful methodological result: encoder screening for this task does
not need fine-tuning. It also closes the last escape route for dermatology
pretraining — DermLIP does not merely lose frozen, it loses after adaptation.

**What this licenses.** *A demonstrably different feature space, fine-tuned,
lands at the top of the same band rather than above it. The ceiling is not
broken; it may be very slightly higher than 0.78.* Encoders are not fully
exhausted as a lever, but the lever is short: the best available alternative
paradigm buys ~0.02 Dice against B2, does not clear the bar, and costs 3× the
parameters.

**What it does not license.** Not "DINOv2 is better." One seed, and a CI that
spans 0.75–0.82 contains every headline model. Not a change to §1's conclusion.
And *not* the pre-registered "encoders are live again" reading, despite the
literal trigger firing — the design fixed 0.79 as a bright line, the observation
landed 0.0002 past it, and honouring the letter over the interval would be
exactly the kind of threshold-gaming pre-registration exists to prevent. Recorded
as **inconclusive-leaning-confirmed**, and the way to settle it is two more seeds
(~5 GPU-hours), not argument.

#### 7i.7a ⚠️ The shipped `ceiling_gate.json` from this run is void

`STAGE_N3_RESULTS/ceiling_gate.json` and `ceiling_gate_test.json` both read
*"MIXED — arms disagree or sit just outside the band"* with `verdicts: {}`. **The
gate never ran.** It was handed a table dict that did not match `ARMS` by key, so
every arm was skipped by the `if arm not in val_tables: continue` line and the
function fell through to its MIXED default — a considered-looking verdict for a
gate that looked at nothing, on the stage with the study's highest Dice.

Compounding it, the per-image CSVs are written **without a `subject` column**, so
even a correctly-keyed dict would have raised. The numbers in §7i.7 were
recomputed by joining `manifests/{val,test}.csv` on `stem` first.

`ceiling_gate` now **raises** on an empty match instead of returning a fall-through
reading. Two general lessons, both already this project's policy and both
re-learned here: a summariser must never have a plausible default for "no input",
and §7f.7a's rule — *a result you cannot see is worse than one you do not have* —
applies to verdicts as much as to checkpoints.

---

## 7j. Stage N4 — mask pretraining vs caption pretraining (RUN 2026-08-11)

### 7j.1 The question the first three foundation stages could not answer

Three medical encoders had lost to DINOv2 by the time this stage was written:

| | contrast | |
|---|---|---|
| frozen probe (§7f.8) | `medsiglip − dinov2` | **−0.1660** |
| frozen probe (§7h.7) | `dermlip − dinov2` | **−0.0913** |
| fine-tuned (§7i.7) | `dermlip − dinov2` | **−0.0859** |

Every one of those medical arms is CLIP/SigLIP-style: **one pooled vector per
image against one sentence.** A caption describes a picture globally and never
says which pixels are the lesion, so that objective is *rewarded* for discarding
spatial detail. DINOv2's objective is patch-level. §7h.9 item 2 named this as the
likeliest mechanism behind the whole ranking.

If it is the mechanism, then *"medical pretraining does not help"* is the wrong
conclusion from those three rows. The right one is narrower: **"medical *caption*
pretraining does not help."** N, N2 and N3 all moved corpus and objective
together and cannot separate them.

### 7j.2 Design — the corpus isolated

SAM and MedSAM are both pretrained with **dense mask supervision**, the objective
this task actually wants. They differ in exactly one thing:

| arm | corpus |
|---|---|
| `sam_ft` | SA-1B — 11 M natural images, ~1.1 B masks |
| `medsam_ft` | ~1.5 M medical image–mask pairs, ~10 modalities |

So `medsam − sam` holds objective, architecture (ViT-B/16), capacity and recipe
fixed and moves **only the corpus**. Two arms, one seed, ~4–5 GPU-h.

Both arms keep the image encoder only and discard the prompt encoder and mask
decoder, because this pipeline is automatic and has no prompt to give. **"MedSAM
with a ground-truth box" would answer a different question** — it hands the model
the answer's location, which is the hard half of this task. Write it up as
*MedSAM's features*, never *MedSAM*.

Three things SAM does that no other encoder in this study does, all handled in
`samprobe.py` and all capable of faking a result if mishandled: a 2-D `[1,64,64,C]`
position embedding added to the patch grid (resampled 64→40 for 640 px), no
CLS/register prefix to strip, and a `[B,C,H,W]` output rather than a token
sequence. `_features` **checks** the layout rather than reshaping, because a
tower returning tokens would otherwise be reshaped into a plausible-looking grid
of garbage.

### 7j.3 THE RESULT — the corpus buys nothing

**Primary contrast, pre-registered:**

```
medsam_ft − sam_ft  =  +0.0037   CI95 [−0.0109, +0.0151]   contains zero
```

A 1.5 M-pair medical mask corpus buys **nothing** over natural images once the
objective is held fixed. Fourth axis, same answer.

| arm | val Dice | test mean | test median | test misses |
|---|---|---|---|---|
| `medsam_ft` | **0.7957** | 0.7672 | 0.8260 | **1** |
| `sam_ft` | 0.7921 | 0.7551 | 0.8133 | 3 |

Neither `medsam − dinov2` (+0.0133) nor `sam − dinov2` (+0.0096) clears zero
either. The one contrast that does is exploratory: `medsam − dermlip` = **+0.0723**
[+0.0325, +0.1061] — medical *masks* beat medical *captions* decisively, which is
§7h.9 item 2 confirmed directly.

### 7j.3a Two readings that must not be merged

`medsam_ft` cleared **0.79 on validation**, which is §7i.3's pre-registered
"ceiling not binding" trigger. That reading is **independent of the corpus
verdict** and belongs in the write-up separately. It is also, on its own, not
enough to break the ceiling: one seed, and the test median lands inside the same
band as everything else.

### 7j.4 The result that changed a recommendation

On **test mean Dice** MedSAM (0.7672) sits below `segformer_b5_teacher` (0.7727)
and `dinov2_ft` (0.7902), which reads as *"not a useful teacher."*

On **validation** — the only split a teacher may be selected on — MedSAM is the
**strongest single model in the project at 0.7957**, above `sam_ft` 0.7921,
`segformer_b5_teacher` 0.7881, `deeplabv3plus_r50` 0.7838, `dinov2_ft` 0.7824 and
`segformer_b2_teacher` 0.7717.

It also has the field's **highest precision (0.873)**, the **best small-lesion
row** (D1 mean Dice 0.792, recall 0.841 — above `dinov2_ft`), and the **smallest
skin-tone gap of any candidate teacher (0.044** against DeepLabV3+'s 0.104).

That reversal is why it entered Stage O's teacher pool. **Judging a teacher on
test Dice and selecting one on val Dice give opposite answers here**, and only the
second is admissible.

### 7j.5 Where it lives

`bruisekit/samprobe.py`, `bruise_stage_n4.ipynb`, `run_stage_n4.py`, generators
`scripts/79*`. Results in `STAGE_N4_RESULTS/tables/`; the `runs/` tree stayed on
the GPU box and is **empty in the laptop bundle** (§10.3). `load_pool_member` in
`itakd.py` is the supported way to reuse the checkpoint as a teacher — it reads
the cut from `test_summaries.json` and **never re-fits it**, because a re-fit
would make Stage O's copy a different operating point from the one Stage N4
reported.

### 7j.6 What this does not license

- Not *"MedSAM does not work."* We removed its prompt.
- Not a seed-robust result. One seed.
- Not a distribution-matched test. MedSAM's corpus is mostly radiology and
  dermoscopy; ours is clinical photography across skin tones. That is the same
  gap that put MedSigLIP last at 0.4670, and part of the loss here.
- **MedSAM2 is untested.** It is a video model with a memory bank, not a version
  bump — a different question needing its own stage.

---

## 7k. Stage O — the miss taxonomy, distilled-arm fairness, and ITA-group routing

Three deliverables in one module because they share one set of per-image tables,
and because the first two are what decide whether the third is worth GPU time.

### 7k.1 "Complete miss" was hiding two different failures

The study publishes one miss number, `complete_miss_rate`, and it is `dice == 0`.
That is the **union** of two clinically different failures:

| | definition | what it means |
|---|---|---|
| **empty prediction** | `pred_positive_pixels == 0` | the model found nothing; the clinician sees no outline and takes a second look |
| **wrong place** | `dice == 0` and `pred > 0` | the model outlined a region with zero overlap — a confident assertion, and the worse failure |

**They differ in 28 of the 35 models** in the current lineage. Field-wide,
excluding Fast-SCNN: **87 complete misses = 60 blank + 27 wrong-place**, so
roughly one miss in three is the model pointing somewhere wrong.

Two rows make the case that one column was not enough:

| arm | zero Dice | blank | wrong place |
|---|---|---|---|
| `p2_cwd_b5_to_b0` | 4 | 1 | **3** |
| `ppmobileseg_tiny_b2kd` | 4 | **3** | 1 |

Same total, mirror-image composition. And the best distillation arm in the study,
`p3_adaptive` (0.7748), misses two photographs and **both are wrong-place** — it
never returns a blank.

`wrong_place` is **derived** as `zero_dice − empty_pred`, never counted
separately, so the three columns cannot print numbers that fail to add up. An
empty prediction always scores Dice 0, so `empty ≤ zero` is arithmetic;
`_miss_counts` **raises** if a table violates it, because that means the Dice
column and the pixel counts came from different evaluations and everything below
is void.

This **supersedes §7.2a's two-definition framing.** That section correctly
identified that the per-seed table counts empty predictions while
`report.normalize` counts `dice == 0`. It is still right; it is now the
two-column version of a three-column answer. Publish all three, and lead with
zero Dice because it is the union and therefore conservative.

### 7k.2 Lesion size is confounded with skin tone, and by how much

`fairness__size_by_ita.csv`, a property of the data and of no model:

| ITA group | n | share in the small-lesion stratum |
|---|---|---|
| Light (II-III) | 39 | **0.59** |
| Brown (V) | 29 | 0.41 |
| Intermediate (III-IV) | 38 | 0.34 |
| Tan (IV) | 24 | 0.33 |
| Dark (VI) | 55 | 0.33 |

Photographs of light skin are **~1.8× more likely to contain a small bruise.**
Pooled across all 35 models, 105 of 115 zero-Dice events fall in deciles D1–D4.
So a material part of every unconditioned *"worse on light skin"* number in this
study is a **lesion-size effect wearing a skin-tone label** — which §8.4 warned
about and nothing had quantified for the distillation arms.

`lesionsize.fairness_conditioned` now reports each gap twice, marginally and
within the small stratum, and labels each model HOLDS or SHRINKS.
`topformer_tiny_distilled` is a clean SHRINKS (0.175 → 0.051);
`fastscnn_distilled` HOLDS (0.182 → 0.260).

### 7k.2a The gap the previous fairness export left

`FINAL_RESULT/RESULT_AUGUST_08/fairness_stats.csv` covers **24 models and
excludes every Stage C distillation arm** — including `p3_adaptive`, the study's
best. The export was built from unprefixed filenames and those arms are written
`per_image_distill_*`. So the one question the deck is asked about the best
distilled model had no table behind it. `STAGE_O_RESULTS/` closes that with
seven tables over 35 models.

### 7k.3 Small-lesion recall — and the result nobody predicted

Recall, not Dice: on a bruise covering ~1 % of the frame the clinical question is
whether the model found it, and a few boundary pixels move Dice far more than
they move recall on a target that small.

| model | all 185 | smallest decile (D1) |
|---|---|---|
| **`lraspp_mobilenetv3` (3.22 M)** | 0.765 | **0.863** |
| `segformer_b5_teacher` (85 M) | 0.716 | 0.828 |
| `segformer_b2_teacher` | 0.728 | 0.822 |
| `p3_adaptive` | 0.742 | 0.797 |
| `unet_r50` | 0.746 | 0.664 |
| `yolo_sem_direct` | 0.671 | **0.474** |

**A 3.22 M mobile model beats an 85 M teacher on the bruises that are hardest to
see.** Capacity is not what small lesions need — which is §7b.7's post-hoc
observation, now confirmed on the full field.

And the assumption behind the whole size stratification is **mostly false**:
recall does not fall off with size for most architectures. Several are worst on
the **largest** decile. YOLO is the exception — 0.47 at D1 rising to 0.82 at D8 —
and that single decile is where its 12 complete misses come from.

### 7k.4 ITA-group-routed distillation — built, gated SHUT

Stage M routed per **image** on each teacher's soft Dice against the label, and
is explicit that it does not route by skin tone (§7e.6). Stage O routes per **ITA
group** on weights fitted once on validation:

```
w_g = softmax(beta * mean_val_dice_k_within_group_g)
```

Three consequences, each answering an objection Stage M invites:

1. **No label at routing time.** The weights come from validation before training
   starts; the only per-image input is its ITA group, a manifest column. Unlike
   Stage M's router, this one is not restricted to training.
2. **K weights per group, not per image.** Five ITA groups over 20 validation
   subjects cannot support 5×K free parameters. `SCHEME` defaults to
   `light_vs_rest`, a two-group collapse.
3. **An identifiability clause the earlier gates lacked** — §7k.5.

**Pool: `segformer_b5_teacher`, `deeplabv3plus_r50`, `medsam_ft`.**
`segformer_b2_teacher` is dropped — it wins no group on validation, has the
lowest drop-one marginal in Stage M's own gate (0.0121 vs DeepLabV3+'s 0.0224),
and is the same MiT family as B5, so it is simultaneously the least useful and
the most correlated member. `unet_r50` is dropped because it appears as a
per-group winner only in the **test** table quoted in `multiteacher.py`'s
docstring and was never in Stage M's actual pool; selecting a per-group teacher
on test is leakage.

`Very Light (I-II)` maps into the Light group **on purpose**: it is 12 train
images and **0 validation images**, so under the five-group scheme those 12 would
have no fitted weight at all. `group_weights` **raises** on a group with no
validation images rather than falling back to uniform — a uniform fallback is
Stage C's `p2_ensemble_uniform` wearing this stage's name.

### 7k.5 THE GATE — closed on both schemes, and the reason is the finding

```
open iff  weighting-gain CI clears zero
     AND  projected student gain > 0.01 Dice
     AND  >=1 group's argmax is identifiable at p >= 0.75
```

The third clause is new and load-bearing. Stage C's gate opened on +0.0258 of
oracle gain and its arm delivered +0.0068; Stage M's opened on +0.0482 and
returned six inconclusive contrasts. **Both were measuring headroom that
genuinely existed. Neither asked whether the routing KEY was estimable.**

| clause | `light_vs_rest` | `five` |
|---|---|---|
| weighting gain over best single | **−0.0056** [−0.0174, −0.0016] | −0.0050 [−0.0166, −0.0012] |
| projected student gain vs +0.01 | −0.0015 ✗ | −0.0013 ✗ |
| routing key identifiable | **0 of 2** ✗ | 1 of 5 ✗ |
| pool contains misses best single lacks | 2 vs 2 ✗ | 2 vs 2 ✗ |

**Identifiability, five-group scheme** — how often the group's argmax teacher
survives resampling the 20 validation subjects:

| group | n patients | best | margin | P(stable) |
|---|---|---|---|---|
| Tan (IV) | 11 | B5 | +0.0134 | **0.84** |
| Dark (VI) | 8 | B5 | +0.0082 | 0.72 |
| Light (II-III) | 7 | MedSAM | +0.0153 | 0.69 |
| Brown (V) | 4 | B5 | +0.0128 | 0.63 |
| Intermediate (III-IV) | 12 | DeepLabV3+ | +0.0038 | **0.40** |

Four of five groups are coin flips. **An arm routed on that argmax is fitting
sampling noise**, and would have produced a fourth null indistinguishable from
the first three.

### 7k.5a The blend loses on fairness too, which was the fallback justification

| | overall val Dice | best−worst skin-tone gap |
|---|---|---|
| `medsam_ft` alone | **0.7957** | **0.044** |
| `segformer_b5_teacher` | 0.7881 | 0.087 |
| `deeplabv3plus_r50` | 0.7838 | 0.104 |
| group-weighted blend | 0.7901 | 0.061 |

The blend is fairer than the two weak members and **worse than the best single
teacher on both axes at once** — −0.006 Dice *and* +0.018 gap. There is no trade
to make. A weighted average is pulled toward the middle: it rescues you from the
worst member and prevents you reaching the best.

Honest limit: subject-clustered bootstrap on those gap differences gives
`blend − medsam` = +0.018 [−0.026, +0.047]. **No gap difference here is
significant** at 20 subjects. The strict reading is "no detectable fairness
difference", which still does not rescue the blend — it would have to *win* on
fairness to justify losing on Dice, and it does not lead even on the point
estimate.

### 7k.6 What the closed gate licenses

**The per-image oracle gain is +0.0450.** Picking the best teacher per
*photograph* is worth real Dice — the teachers do complement each other.
**Grouping those photographs by skin tone captures none of it.** So whatever
makes one teacher better on a given image, it is not the patient's skin tone.

That is the measured answer to *"why not just group by skin tone"* — a question
Stage C's `p3_adaptive_group` (0.7586, 11th of 15 arms) and Stage M (§7e.6) each
settled by **assertion**. It is a result about the study design, and it is
stronger than the arm would have been.

**The practical lever is teacher CHOICE, not teacher blending.** 0.044 against
0.104 is a bigger fairness move than any blend produced, and it costs nothing.

### 7k.7 Where it lives, and the failure that would fake a result

`bruisekit/itakd.py`, `bruise_stage_o.ipynb`, `run_stage_o.py`, generators
`scripts/83*`. Results in `STAGE_O_RESULTS/tables/` — including the gate as
plain text as well as JSON, because a closed gate is a result and someone has to
read it without a script.

**`engine.train_run` iterates `for step, (x, y, _)` and discards the stem**, so
the loss has no idea which images it is looking at. Rather than edit the shared
training loop, `install_group_shim` wraps the *training* loader so each batch
records its stems' group indices in a module global immediately before the batch
is yielded. Loader, teacher forward and loss all run synchronously in the main
process within one iteration, so the global is always the current batch's.

That coupling is **guarded, not trusted**: `GroupRoutedDistillLoss.forward`
raises if the group vector is absent or its length does not match the batch.
Without that check a stale global would silently mean uniform weights, the arm
would quietly become `p2_ensemble_uniform` with a gate, and it would report an
entirely plausible number for a different experiment. `self_test` asserts both
raises, plus the two identities that keep the contrast one-variable: K=1 reduces
to Stage H's gated loss exactly, and a uniform weight matrix reproduces Stage C's
uniform ensemble.

**Read `group_loss_stats.json`'s `images_per_group` before any Dice number.** An
arm whose loss never saw one of the groups did not run the experiment the gate
authorised.

---

## 7l. The re-inference sweep — the pipeline verified against itself

Run 2026-08-12 on ORC. 24 models re-scored from their checkpoints through
`inference.run`, each at its own val-fitted cut, and compared against the table
its original run wrote.

| | |
|---|---|
| bit-for-bit identical | **6 of 12** with a shipped table |
| largest disagreement on any single photograph | **0.0068** |
| complete-miss count changed | 1 model, by 1 (`yolo_sem_distilled`) |

The handbook's standing claim — that every reporting model agrees to better than
2e-4 mean Dice against the original A100 run — is now **checked rather than
repeated**. Results in `inference/inference_reconciliation.csv`.

**The caveat that limits what this proves.** `inference.resolve_runs` reads best
seeds from `report.best_seeds`, which covers **five families**; every other
family falls back to **seed 0**. So 19 of the 24 rows are a different seed from
the published lineage, and their per-image scores differ by up to **0.81 Dice** —
which is the documented signature of a seed mismatch (§7l is the third time this
trap has been recorded), not of a broken pipeline. **This sweep confirms the
pipeline, not those specific published rows.** A full reconciliation needs
`best_seeds` extended to every family.

**No second inference implementation exists, deliberately.** A ~600-line module
with its own loader, cut resolution and reconcile was written during this work
and **deleted**: `inference.inference_pass`'s docstring says *"There is
deliberately no second inference implementation in this file,"* and a
re-inference whose only job is to confirm published numbers must not be computed
by a different path than produced them. `scripts/84b` enforces it — the build
fails if the notebook it packages contains `def score_`, `evaluate_at_cut(` or
`load_state_dict(`.

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

### 10.2 The checkpoint search path — `env.run_roots`

The study's weights are not in one directory and never were. `env.run_roots`
searches, in priority order:

| # | root | holds |
|---|---|---|
| 1 | `WORK_DIR/runs` | anything this session trains |
| 2 | `EXTRA_RUNS` | set in §0 — e.g. `/scratch/$USER/bruise_work/runs` |
| 3 | `<bundle>/checkpoints/final` | Stage A, 5 models × 3 seeds |
| 4 | `<bundle>/checkpoints/baselines` | U-Net, DeepLabV3+ |
| 5 | `<bundle>/checkpoints/efficient` | Stage E/F |
| 6 | `<bundle>/checkpoints/rgkd` | Stage H |
| 7 | `<bundle>/checkpoints/yolo_l` | Stage Y |
| 8 | `<bundle>/checkpoints/distill/teachers` | B2, B5, B0-distilled |

**First hit wins**, so your own training always beats a shipped copy. Before
2026-08-04 only root 1 and *one* of 3–8 were searched, and the fallback between
them was silent — see §15 trap 18 for what that cost.

**Three on-disk layouts**, all understood by `registry.find_checkpoint`:

| layout | weights | threshold |
|---|---|---|
| `standard` | `<family>__seed<N>/best.pt` | `operating_point.json` — a **logit** |
| `yolo` | `…/ultralytics_runs/train/weights/best.pt` | none needed (argmax) |
| `teacher` | `<family>/best_model.pt` — **no seed** | `threshold.json` — a **probability** |

`registry.read_cut()` is the single place that converts between the two threshold
dialects. They are not interchangeable: `sigmoid(cut) == threshold`, so reading
one as the other does not raise — it silently moves the decision boundary.

**SegFormer-B5 was invisible until this existed**, for three independent reasons
at once: `best_model.pt` not `best.pt`, `threshold.json` not
`operating_point.json`, and no seed in the directory name — its
`seed_selection.csv` picked seed **123**, so a registry asking for `__seed0` could
never have matched. It is now Stage T, `WEIGHTS` tier, and usable as a teacher.
That matters for §18.2: B2 is the only model in this study with a statistically
detectable skin-tone disparity, and every Stage H arm distils from it.

**Provenance is recorded, not inferred.** Every `Run` carries `source_root`,
`layout` and `threshold_file`, and §3 of the notebook prints them per family. If a
family you trained shows a shipped root, your training is not being used — and you
learn that from the plan rather than from a crash.

**Stage T is never trained here.** The teacher store is an input; a MISSING teacher
is a statement about the bundle, not a job about to start. `train_missing` acts on
stages A/B/E/H/Y only.

### 10.3 The RESULTS search path — where per-image CSVs live, per host

§10.2 covers *weights*. This covers *tables*, and it is a separate problem with a
separate failure mode. **The two hosts do not have the same result trees, and
neither is a superset of the other.**

| tree | laptop bundle | ORC |
|---|---|---|
| `FINAL_RESULT/RESULT_AUGUST_08/` — 40 per-image CSVs, the current lineage | ✅ | ❌ **never synced** |
| `LESION_SIZE_RESULTS/`, `STAGE_M_RESULTS/`, `ALL_MODELS_RESULTS/` | ✅ | ❌ |
| `STAGE_N4_RESULTS/tables/` | ✅ | ✅ |
| `STAGE_N4_RESULTS/runs/` — the checkpoints | ❌ **empty** | ✅ |
| `results/analysis_native/` | ✅ | ✅ (5 models) |
| `_work/outputs/` or `/scratch/$USER/bruise_work/outputs/` | ❌ | ✅ — **the ORC default** |
| `/scratch/$USER/bruise_work/runs/` | ❌ | ✅ — set as `EXTRA_RUNS` |

**The ORC roots, verbatim.** Paste these; do not retype them.

```python
EXTRA_ROOTS = [
    "/scratch/tbommawa/bruise_work",
    "/scratch/tbommawa/bruise_work/outputs",
    "/scratch/tbommawa/BRUISE_UNIFIED",
]
EXTRA_RUNS = "/scratch/tbommawa/bruise_work/runs"      # checkpoints, §10.2
```

**The rule: never hard-code a lineage directory in a new notebook.** Scan, and
report what was found. Two modules already do this and a third must not be
written:

| function | what it does |
|---|---|
| `allmodels.search_roots(env, extra)` | every plausible directory that exists on *this* host |
| `allmodels.discover` + `load_all` | scans them all, **merges across roots**, groups tables into cohorts by (stem set, GT-area vector), labels the largest `REFERENCE` |
| `itakd.load_tables(env)` | the above, filtered to the reference cohort, with the `distill_` prefix resolved |
| `lesionsize.find_lineages(env, extra)` | discovery for the single-directory case |

`lesionsize.load_lineage` picks **one** directory. That is fine on the laptop and
wrong on ORC, where the tables are spread across `results/`, `STAGE_M_RESULTS/`,
`STAGE_N4_RESULTS/` and `outputs/` and the best single directory holds five
models. Use `itakd.load_tables` or `allmodels.run` when you need more than one
tree.

> **2026-08-12 — the failure this section exists to prevent.** Stage O's notebook
> shipped with `LINEAGE = "FINAL_RESULT/RESULT_AUGUST_08"`. It ran on the laptop
> and raised `FileNotFoundError` on ORC for a directory that was never going to
> exist there. The error text compounded it by printing
> `FINAL_RESULT/FINAL_RESULT/RESULT_AUGUST_08` — `lineage_dir` tries five
> candidates and, on failure, reports only the *conventional* one rather than the
> list it actually tried, so the message looks like a path bug and is not.
> Both are fixed: `itakd` scans, and this table exists so the next stage does not
> have to rediscover it.

**Corollary for splitting work across hosts.** The analysis stages that read
per-image CSVs belong on the **laptop**; the stages that load checkpoints belong
on **ORC**. Stage O is built that way on purpose — `run_stage_o.py --only
miss,fairness` is laptop work and `--only matrix,gate,train` is ORC work. Do not
try to run either half on the wrong machine and then debug the paths.

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

Add `"MYTABLE"` to the `WANTED` list and `FILENAME` map in the D10 save cell.

**Rules that keep the analysis honest:**

- Any interval or test must resample **subjects**, not images.
- Any cross-model comparison must be **paired**.
- Report median alongside mean for anything Dice-based.
- If your metric could be confounded by bruise size, condition on it or say so.

---

## 14. Known gaps and caveats

### Genuine gaps

0. **`report.best_seeds` covers five families, not all of them.** It returns the
   val-selected best seed for the three SegFormers and both YOLO arms; every
   other family falls back to **seed 0** in `inference.resolve_runs`. That is why
   §7l's re-inference sweep confirms the pipeline but not the published rows for
   the mobile family and the baselines — 19 of its 24 rows are a different seed
   from the reported lineage, differing by up to 0.81 Dice per image. Extending
   `best_seeds` to every family is the fix and is a table lookup, not a rerun.
0b. **MedSAM2 untested** (§7j.6). A video model with a memory bank, not a
   version bump; it needs its own stage rather than a swap into Stage N4.
0c. **The ITA-grouped multi-teacher arm is built and never trained** (§7k.4).
   Deliberate — its gate is closed on both schemes and the closure is the
   result. `run_stage_o.py --only train` refuses without `--force-train`. Do not
   run it without writing down why.
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
6. **Speed tables span three machines, and the device is not what separates
   them.** Stage A was measured on a full A100 (Colab), Stage E on an A100 MIG
   3g.40gb slice. This entry used to say "a MIG slice is roughly 3/7 of a card,
   so they are not comparable" — **that premise is wrong**, and the way it is
   wrong matters more than the fact. Measured 2026-08-03 on the same five
   architectures across three machines:

   | model | ORC MIG 3g.40gb (42 SM) | mlidl A100-PCIE-40GB (108 SM) | mlidl ÷ ORC |
   |---|---|---|---|
   | `fastscnn` | 3.68 ms | 12.52 ms | 3.40× |
   | `lraspp_mobilenetv3` | 5.07 ms | 17.19 ms | 3.39× |
   | `topformer_tiny` | 6.26 ms | 20.67 ms | 3.30× |
   | `ppmobileseg_tiny` | 10.85 ms | 34.94 ms | 3.22× |
   | `segformer_b0` | 8.97 ms | 21.49 ms | 2.40× |

   The card with **2.6× the SMs is 3.3× slower**, near-uniformly, across five
   architectures with completely different FLOP profiles — including four that
   never touch `transformers`. A near-constant multiplier across unrelated
   architectures is the signature of **fixed per-call overhead**, not of compute.
   Which is what the recipe measures: batch of 1, `cuda.synchronize()` on both
   sides, so each timed call is a few ms of work bracketed by launch and sync
   cost. Fast-SCNN is 1.14 M parameters; 12.5 ms for it on an A100 is not a
   compute number. MIG partitions SMs and bandwidth and does nothing to dispatch,
   which is exactly why a slice can beat a full card.

   **The leading explanation is the GPU clock governor.** `mlidl` GPU 1 reports
   `persistence_mode: Disabled`, an applications clock of **765 MHz** against a
   max of 1410, and idles at 210 MHz in a 250 W envelope. A bursty batch-1
   workload never presents sustained demand, so the clock never ramps; ten warmup
   iterations do not change that. A cluster node running MIG normally has
   persistence on and clocks pinned. Unconfirmed until clocks are sampled *under
   load* — see §18.5 item 1.

   **What is portable, and what is not.** Absolute latency is not. Within-machine
   ratios are — normalised to Fast-SCNN, the four mobile nets hold position to
   within 5 % across machines:

   | | lraspp | topformer | ppmobileseg | segformer_b0 |
   |---|---|---|---|---|
   | ORC | 1.38 | 1.70 | 2.95 | **2.44** |
   | mlidl | 1.37 | 1.65 | 2.79 | **1.72** |

   **⚠️ UPDATE 2026-08-04 — the SegFormer column above is contaminated, and the
   explanation this entry used to give for it was wrong.** Earlier revisions
   attributed SegFormer's ~1.4× residual to the `transformers` key refactor
   (caveat 9). It is not that. **The SegFormer rows in the Stage A table were
   measured in a dirty process and are inflated ~1.72 ×** (§7.3a). Once
   re-measured in a fresh kernel, four independent clean measurements of
   SegFormer-B0 agree to within run-to-run drift — 9.294 (ORC), 9.870 (Colab
   harness), 9.452 (Colab fresh), 8.740 (the 51-run sweep). There is no
   cross-machine SegFormer anomaly. There was one contaminated row.

   Also note this entry's ORC SegFormer figure (8.97 ms) has **no counterpart in
   `benchmark_stage_e.csv`**, which contains only the four mobile nets — so that
   row came from a different harness than the four beside it. Two of the three
   things this caveat originally blamed on hardware were measurement artefacts.

   **What survives.** The launch-bound finding for the four mobile nets stands:
   they were measured with one harness across machines, their ratios hold to 5 %,
   and the near-constant `mlidl ÷ ORC` multiplier is still the signature of fixed
   per-call overhead rather than compute. The clock-governor hypothesis for
   `mlidl` is still unconfirmed (§18.5 item 3).

   **Consequences.** Never publish a cross-machine absolute latency. Publish one
   machine's full table (§7.3 is now that table), or publish ratios against a
   named reference model. Every latency row needs `persistence_mode`, the locked
   SM clock, **and `fresh_process`** recorded beside it — `bruisekit/gpustate.py`
   and the `speed_table` columns added 2026-08-04 supply all three.
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

**17. Reproducibility mistaken for correctness.**
SegFormer-B0 was published at 16.55 ms / 60 FPS. It was measured in July and again
in August, three weeks apart, and agreed to **0.8 %**. That agreement was treated
as evidence the number was right, and it was quoted for weeks. It was wrong by
1.75× — both runs had executed the full notebook before reaching the benchmark
cell, so both inherited the same contaminated process state (§7.3a). **Repeating a
measurement only tests the things you varied between the repeats.** The variable
that mattered had never been varied.
→ *Guard:* `speed_table` records `fresh_process` and `process_prior_peak_MB` on
every row and prints a warning before the work starts when the process has already
touched the GPU. More generally: when a number is surprising, the question is not
"does it reproduce" but "what have I *not* varied".

**18. A search path that finds a different checkpoint and says nothing.**
`registry.py` looked for weights in `env.runs`, then fell back to the bundle's
shipped `checkpoints/<stage>/` — silently, with no warning, still returning a
`WEIGHTS`-tier run. On ORC the real training lived in `/scratch/$USER/bruise_work/runs`,
which nobody had pointed the notebook at, so every SegFormer loaded an
old-lineage shipped copy instead. The only symptom was a `RuntimeError` three
steps later, from a `transformers` layout mismatch that was itself a consequence
rather than the cause. Two ORC runs were wasted on the wrong diagnosis. **A model
you have on disk and cannot see is worse than one you do not have:** the second is
reported as a gap, the first is substituted for the one you asked for.
→ *Guard:* `env.run_roots` is an ordered search path over every known location
(§10.2), every `Run` carries `source_root` and `layout`, and §3 of the notebook
prints a provenance table. If a family you trained shows a shipped root, you see
it in the plan rather than in a crash.

**19. An oracle that opens a gate the student cannot walk through.**
Stage C's `val_oracle` measured +0.0258 of oracle gain from combining B2 and B5,
CI [0.0165, 0.0362], P(gain > 0) = 1.0. The gate opened, ten KD arms were trained,
and the best of them realised +0.0068 — every arm NON-INFERIOR (§6.5). The gate
was not wrong about the complementarity; it was answering a question nobody had
asked it. **An oracle bounds the TEACHER signal. The endpoint is the STUDENT.**
Nothing in that pre-test estimated the transfer between the two, so it could not
have failed to open on any pool with complementary teachers — which, at that
point, was the only thing it had been asked to check.
→ *Guard:* `multiteacher.oracle_gate` projects the oracle gain through the
transfer rate Stage C actually measured (0.264) and requires the **projection**,
not the oracle, to clear the non-inferiority margin. It prints the rate and its
provenance so a reader can reject the projection rather than inherit it. More
generally: a pre-test whose passing condition does not include the quantity you
will be judged on is a ritual, not a gate.

**20. A four-teacher arm that is also a smaller-batch arm.**
Nearly shipped in Stage M. `segformer_b0_distilled` probed to micro-batch 64 with
one teacher resident; the same `per_model` probe with four teachers resident
lands far lower, because the probe measures what fits and what fits had changed.
The arm would have differed from its control in the teacher signal, the batch
size, and the optimizer-step count — and only the first would have been in the
table. This is trap 1's B0-vs-B2 batch confound arriving by a new route: not a
config someone edited, but a config the *probe* computed differently because the
memory around it moved.
→ *Guard:* `multiteacher.install_batch_shim` pins each arm to the batch its
control's own `config.json` records, and teacher forwards are chunked so that
pinning it is affordable. Anything that changes VRAM pressure changes an
auto-tuned batch, and an auto-tuned batch is a hyperparameter that varies without
appearing in any diff.

**21. One architecture, two wrapper classes, two state-dict spellings.**
`segformer_b5_teacher` would not load into `models.SegFormerNet`: 1172 keys
against 1174. It was trained by the Stage C `kd/` lineage, whose wrapper names its
submodule `model` rather than `net` and registers no `mean`/`std` buffers, because
it normalises in the **dataloader** (`A.Normalize`) instead of in `forward`.
`_reconcile_segformer_layout` correctly refused it — the rename it knows is the
encoder/stages one, and this is a different difference stacked on top of it.
**The dangerous part is not the crash, it is what a careless fix would have
done.** Swapping the prefix makes the checkpoint load cleanly whether or not the
two wrappers agree about pixel scale. Here they do — both use ImageNet mean/std,
just at different points — but had they differed, every B5 probability would have
been silently mis-scaled and the multi-teacher gate would have reported the
resulting garbage as complementarity. Same shape as trap 4: a normalisation
mismatch does not raise, it just changes the answer.
→ *Guard:* `multiteacher.load_teacher_model` composes the prefix swap with the
shipped reconciler rather than reimplementing it, supplies only `mean`/`std` from
`__init__` and raises on any other missing key, and its docstring records the
normalisation check. Verified against B5's own shipped `test_per_image.csv`: max
per-image Dice difference 4e-4. **When a checkpoint needs a remap, re-score it
against a number it already produced** — key-set agreement proves the shapes
line up, not that the weights mean the same thing.

**22. An OOM that was not this session's OOM.**
Stage M's gate — a batch-8 eval pass that needs about 5 GB — failed with CUDA out
of memory on a 39.25 GiB slice. The message contains the answer, two lines below
the part everyone reads:

```
GPU 0 has a total capacity of 39.25 GiB of which 759.31 MiB is free.
Process 4085555 has 33.09 GiB memory in use.
Including non-PyTorch memory, this process has 5.29 GiB memory in use.
```

Two different processes. The current kernel held 5.29 GiB and was fine; a
**previous Jupyter kernel**, still alive after an earlier failed run, held 33.09
GiB and had done so for hours. The same 33.09 GiB figure appears in the earlier
genuine OOM, which is what makes it identifiable in hindsight. Roughly an hour
went into shrinking batch sizes against a constraint that had nothing to do with
the batch size.
→ *Guard:* read the `Process <pid>` line before touching any configuration —
if that PID is not `os.getpid()`, the fix is `os.kill`, not a smaller batch. On a
cluster with no terminal, `nvidia-smi --query-compute-apps=pid,used_memory
--format=csv` from a notebook cell lists them; kill everything that is not the
current kernel, then **restart** (the surviving context is fragmented). A Jupyter
kernel holds its GPU allocation until it is shut down, not until its notebook tab
is closed.

**23. An arm that never exercised its own mechanism.**
Stage M trained six runs at `beta = 8` and returned a null. The router
diagnostics say why: mean routing entropy 0.95–0.98 against a uniform `log 3` of
1.099 — **88 % of the way to uniform**, with mean weights 0.27 / 0.48 / 0.26. The
teachers' per-image soft Dice values sit within ~0.05 of each other, so
`exp(8 × 0.05) ≈ 1.5` produces a tilt, not a choice. The gate had been opened on
the *oracle* (per-image argmax, `beta → ∞`); what actually trained was a
fixed-weight ensemble. Reporting that null as "per-image routing does not help"
would have been false — it is "a near-uniform ensemble does not help", which
Stage C had already shown.
→ *Guard:* every arm with a knob must log **what the knob did**, not only what
the arm scored. `RoutedDistillLoss.stats()` records the routing entropy and the
mean weight per teacher, `dump_router_stats` writes them beside the run, and §15
of the notebook prints the interpretation (uniform / collapsed / mixing) rather
than the raw number. A mechanism that is not measured cannot be distinguished
from a mechanism that did not fire, and both produce plausible tables.

**24. Three seeds of a control that are the same seed.**
Stage M's SegFormer contrast asked `report.load_per_image` for
`segformer_b0_distilled__seed{0,1,2}` and got the **same cached results-tier CSV**
three times — the val-selected best seed. The symptom was visible and easy to
miss: three control rows identical to six decimal places (0.768011 mean, which is
exactly `headline_all.csv`'s value). The seed-matched paired design in §7e.5 was
therefore not what ran. The LR-ASPP arm was unaffected only because its controls
are not in the bundle, so a fallback read them per-seed from `EXTRA_RUNS` — the
correct path, reached by accident.
→ *Guard:* when a contrast is per-seed, assert the control tables **differ**
between seeds before trusting the result. Identical floats across seeds are not a
coincidence, they are one file. The RESULTS tier is keyed by family, not by
(family, seed); asking it for a seed is a question it cannot refuse and cannot
answer.

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

Further scripts package **patch overlays** — a notebook, the modules it needs and
their sources — for dropping onto a bundle that already exists elsewhere:

```bash
python scripts/63_zip_rgkd_overlay.py           # -> BRUISE_RGKD_OVERLAY.zip, ~0.2 MB
python scripts/69_generate_stage_m_notebook.py  # Stage M (§7e)
python scripts/69b_zip_stage_m.py               # -> BRUISE_STAGE_M.zip
python scripts/70_generate_foundation_notebook.py   # Stage N (§7f)
python scripts/70b_zip_foundation.py                # -> BRUISE_FOUNDATION.zip, ~0.05 MB
python scripts/71_generate_lesion_size_notebook.py  # Stage P (§7g)
python scripts/71b_zip_lesion_size.py               # -> BRUISE_LESION_SIZE.zip, ~0.05 MB
```

The three experiment overlays **refuse to build** if their module has since been
added to `60_build`'s copy list or acquired a twin in `scripts/unified_lib/` —
otherwise the overlay would ship a stale duplicate of a file that now has a
source of truth. Both also run their module's `self_test()` and compile every
notebook cell before writing the archive, because an archive containing a
mislabelled arm or a cell that will not parse is not shippable whatever else is
right.

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
| `bruise_stage_m.ipynb` | `scripts/69_generate_stage_m_notebook.py` |
| `bruise_foundation.ipynb` | `scripts/70_generate_foundation_notebook.py` |
| `bruise_lesion_size.ipynb` | `scripts/71_generate_lesion_size_notebook.py` |
| `bruisekit/{multiteacher,foundation,lesionsize}.py` | **authored in place, no source in `unified_lib/`** — see §7e, §7f, §7g |

The build **verifies rather than assumes**: data dedup is hash-checked across two
sources, checkpoint lineage is asserted on config fields, baselines are
reconciled against the canonical per-seed CSV, and split leakage is re-checked.

---

## 17. File map

```
BRUISE_UNIFIED/
├── bruise_unified.ipynb          84 cells, eight stages (A/B/C/E/H/Y/D/G)
├── bruise_stage_m.ipynb          Stage M — 32 cells, GATE FIRST (§7e). Standalone:
│                                 nothing in the pipeline imports it, and §0-§7
│                                 train nothing.
├── bruise_foundation.ipynb       Stage N — 34 cells, GATE FIRST (§7f). Standalone.
│                                 §0-§7 is the ~1 GPU-hour gate; §9 guards §10+.
├── bruise_lesion_size.ipynb      Stage P — 22 cells (§7g). Standalone, CPU ONLY:
│                                 no torch, no checkpoints, minutes on a laptop.
│                                 §1 is the only cell you edit.
├── README.md                     how to run
├── PROJECT_HANDBOOK.md           this file
├── PROJECT_STORY.md              the same project in plain language, for
│                                 non-technical readers: what every architecture
│                                 and loss was for, what each technique returned,
│                                 and what it meant. 15 chapters. Where the two
│                                 disagree THIS file wins — it tracks the code.
├── requirements.txt
│
├── FINAL_RESULT/
│   ├── RESULT_AUGUST_08/         ★ THE CURRENT LINEAGE. Every reported number,
│   │                             Stages A-Y, one session, one scoring path.
│   │                             headline_all.csv + per_image_*.csv + inference/
│   └── STAGE_M_RESULTS/          Stage M (§7e). NOT part of the lineage above —
│                                 a null, kept isolated on purpose.
├── FOUNDATION_RESULTS/           Stage N (§7f). Written at run time; same
│                                 isolation. gate.json is the decision.
├── LESION_SIZE_RESULTS/          Stage P (§7g). Written at run time; same
│                                 isolation. lesion_size_contrasts.csv is the
│                                 answer, lesion_size_power.csv is the fallback
│                                 deliverable when nothing clears zero.
│
├── bruisekit/                    the library — see §9
│   ├── distill_efficient.py      Stage F: DeepLabV3+ → mobile-student KD shim (§7b)
│   ├── reliability_kd.py         Stage H: reliability-gated KD + B2 teacher axis (§7c)
│   ├── multiteacher.py           Stage M — NOT A BUILD OUTPUT (§7e). Absent from
│   │                             60_build's copy list on purpose; a rebuild
│   │                             neither regenerates nor deletes it.
│   ├── foundation.py             Stage N — NOT A BUILD OUTPUT (§7f), same reason.
│   │                             FoundationSegNet / ResNet50ProbeNet, probe_gate.
│   ├── lesionsize.py             Stage P — NOT A BUILD OUTPUT (§7g), same reason.
│   │                             Global decile cut, stratum cluster bootstrap,
│   │                             min-detectable-effect. Pure pandas, no torch.
│   ├── dermprobe.py              Stage N2 — _ProbeBase, the architecture-blind
│   │                             arm contract every later probe subclasses (§7h)
│   ├── finetune_n3.py            Stage N3 — unfrozen probe + ConvDecodeHead,
│   │                             which N4 and O import so the head is the SAME
│   │                             object across stages (§7i)
│   ├── samprobe.py               Stage N4 — SAM/MedSAM encoders retargeted
│   │                             1024/64 → 640/40. resample_pos_embed_2d,
│   │                             find_sam_blocks, mask_supervision_gate (§7j)
│   ├── itakd.py                  Stage O — miss_taxonomy, distilled_fairness,
│   │                             ita_group_gate + identifiability, the group
│   │                             shim and GroupRoutedDistillLoss (§7k)
│   ├── allmodels.py              cross-root discovery + cohorting. The module
│   │                             that makes an analysis runnable on BOTH hosts
│   │                             (§10.3); merges across roots, quarantines
│   │                             tables that cannot share a decile cut
│   ├── fenwick_cv.py             matched-core labeler cross-validation — the
│   │                             only data with >1 annotation per image, so the
│   │                             only direct measurement of the ceiling (§1)
│   └── gpustate.py               clocks, persistence, SM clock under load (§7.3a)
│
├── bruise_stage_n3.ipynb         Stage N3 (§7i)      + run_stage_n4.py
├── bruise_stage_n4.ipynb         Stage N4 (§7j)      + run_stage_o.py
├── bruise_stage_o.ipynb          Stage O  (§7k)      + run_all_models.py
├── bruise_all_models.ipynb       every model, every endpoint, cohorted
├── bruise_derm_probe.ipynb       Stage N2 (§7h)
├── bruise_fenwick_cv.ipynb       labeler CV
├── bruise_inference_all.ipynb    §7l. Calls bruisekit.inference and adds NO
│                                 scoring code; scripts/84b fails the build if
│                                 it ever does.
├── pyproject.toml                pip install -e .  (v2.0)
├── tests/                        pytest over every module's self_test() — 12
│                                 checks, no GPU, no weights, no network
│
├── STAGE_N3_RESULTS/             §7i. NOTE: ceiling_gate.json from that run is
│                                 VOID — see §7i.7a.
├── STAGE_N4_RESULTS/tables/      §7j. runs/ is EMPTY on the laptop (§10.3).
├── STAGE_O_RESULTS/tables/       §7k. ita_group_gate__*.txt is the verdict in
│                                 plain text as well as JSON.
├── ALL_MODELS_RESULTS/           the one-row-per-model table, with cohort flags
├── FENWICK_RESULTS/tables/       labeler CV; cross_dice is the ranking column
└── inference/                    §7l re-inference + the fp32 speed sweep.
                                  Speed CSVs are named for device AND precision
                                  AND machine — three earlier benchmarks all
                                  wrote to one filename (§7.3a).
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
│   ├── yolo26l-sem.pt            Stage Y — NOT SHIPPED, ~50 MB, fetch it (§7d)
│   ├── efficient/                MobileNetV3, StrideFormer, TopFormer backbones
│   └── foundation/               Stage N — NOT SHIPPED and cannot be (§7f.7).
│                                 medsiglip-448/ is licence-GATED (~1.6 GB);
│                                 dinov2-base/ is open (~350 MB). §2a of the
│                                 notebook stops with the commands if absent.
│
├── checkpoints/
│   ├── final/                    15 Stage A runs
│   ├── baselines/                6 Stage B runs
│   ├── distill/
│   │   ├── teachers/             B5, B2, B0-distilled + reference CSVs
│   │   └── distill_out/          10 completed arms + alpha search + oracle
│   ├── efficient/                Stage E/F — ABSENT until you train (§15 trap 13)
│   └── yolo_l/                   Stage Y — ABSENT until you train (§7d)
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

## 18. Planned next work — for discussion, not yet implemented

Nothing in this section is built. It exists so the four items below are written
down in one place before the next meeting, in the same form as §14: what is
being proposed, what already exists that it would reuse, and what has to be
decided before anyone spends GPU time. **No code has been written for any of
it. Do not read a family name here as a registry entry** — none of these are in
`FAMILY_SPEC`, `TEACHER_FOR` or `STAGE_H_FAMILIES`, and §14's warning about
unregistered arms being invisible to the registry applies to all of them.

### 18.1 Item 1 — inference for the three models — **IMPLEMENTED 2026-08-03**

**Status 2026-08-05: built, run on GPU on two machines, and extended.** The
original scope is kept below because it is also the spec. Four things were added
after the first GPU runs exposed them:

| added | why |
|---|---|
| `fresh_process`, `process_prior_peak_MB` | §7.3a — a benchmark in a dirty process was 1.75× off, reproducibly |
| `precision=` + `INFERENCE_PRECISION` | §7.3c — attributes a gap; `check_single_machine` refuses to mix fp16 and fp32 rows |
| `machine_tag=` + `MACHINE_TAG` | three incomparable tables once shared one filename and had to be told apart from memory |
| per-model failure isolation | one unloadable checkpoint used to discard every model already timed; now `FAIL`s, is listed at the end, and the sweep completes |

`peak_activation_MB` is retained (the shipped CSV calls its memory column that)
but is **not** activation memory — `reset_peak_memory_stats` runs after the 185
test images are staged, so every row carries 909 MB of images.
`peak_incremental_MB` is the honest column.

`bruisekit/inference.py` (source of truth `scripts/unified_lib/inference.py`,
shipped by `copy_authored_modules` in `scripts/60_build_unified_bundle.py`).

**The notebook is the primary entry point** — §D9, driven by two §0 flags, so the
block stays where every other stage lives rather than becoming a script you have
to remember exists:

```python
RUN_INFERENCE_BLOCK = True      # False by default: D9 costs nothing and says so
INFERENCE_MODELS    = None      # None = the three SegFormers; any registry family works
```

`NEED_CACHE` picks the flag up, so the 640 cache builds itself. The cell is four
lines — config, one call into `bruisekit`, two `display`s — which is the same
contract as every other cell in the notebook.

The CLI is the same code for a headless node:

```bash
python -m bruisekit.inference                                   # the three SegFormers
python -m bruisekit.inference --models fastscnn --no-inference  # speed only, any family
```

| API | What it does |
|---|---|
| `DEFAULT_MODELS` | the SegFormer trio — see the resolved open question below |
| `resolve_runs(env, reg, models, seed=None)` | family → `Run` at the val-selected best seed via `report.best_seeds`; non-WEIGHTS families are printed and skipped, never back-filled |
| `inference_pass(...)` | delegates to `loaders.score_run` + `report.normalize`. **No second inference implementation exists in this file** |
| `speed_table(...)` | delegates to `evaluate.benchmark_speed` on CUDA; `_benchmark_cpu` / `_benchmark_yolo_cuda` otherwise |
| `reconcile(...)` | fresh vs shipped per-image, per model |
| `check_single_machine(df)` | raises if a table mixes `device_name`. Called before any write |

Written to `env.out/inference/`: `per_image_<family>.csv`,
`inference_headline.csv`, `inference_reconciliation.csv`, and
`benchmark_640_{cuda,cpu}.csv` — device-tagged in the filename so a CPU
smoke-test cannot silently overwrite the table it is not comparable to.

Schema is `results/final/benchmark_640.csv`'s nine columns plus `kind`, `run_id`,
`device`, `device_name`, `repeats`, `warmup`, `seed`. Those last six were the
context you needed to interpret the shipped five rows and were not written down.

**What the smoke test showed.** All three re-inferred on CPU, against tables
produced on an A100:

| model (seed 0) | fresh | shipped | Δ mean | max per-image Δ | Δ misses |
|---|---|---|---|---|---|
| `segformer_b2_teacher` | 0.769247 | 0.769240 | +7e-6 | 2.5e-3 | 0 |
| `segformer_b0_distilled` | 0.767989 | 0.768011 | −2.2e-5 | 7.6e-3 | 0 |
| `segformer_b0_direct` | 0.766278 | 0.766318 | −4.0e-5 | 6.8e-3 | 0 |

Every mean agrees to better than 2e-4 and no complete-miss count moves. That
reproduces this bundle's README agreement table exactly, from a different code
path, which is the point of `reconcile`. Roughly 2 minutes per SegFormer on CPU.

**The GPU run happened on 2026-08-03, and it produced a finding rather than a
table.** The inference pass reconciled exactly — fresh vs shipped
`mean_dice_delta = 0.0`, `max_abs_dice_delta = 1.1e-16` for
`lraspp_mobilenetv3`, `lraspp_mobilenetv3_distilled` and `fastscnn` on ORC — so
(a) is settled and the block is verified end-to-end on CUDA. The **speed** half
is not: the same five architectures timed on three machines disagree by up to
3.4× in a direction the hardware cannot explain, and the cause looks like GPU
clock state rather than the device. Full data and reasoning in §14 caveat 6.

**What is still owed:** a *trustworthy* speed table — one machine, persistence
mode and locked clocks recorded, covering all families rather than the subset
timed so far. The four constraints below are enforced in code and were exercised
by the 2026-08-03 run; the fifth constraint that run discovered — record the
clock state — is not yet in code. See §18.5.

**Resolved open question — *which* three models.** `DEFAULT_MODELS` is the
SegFormer trio (`segformer_b2_teacher`, `segformer_b0_direct`,
`segformer_b0_distilled`): the analysis notebook's own `SEGFORMER_MODELS` dict
and the exact three rows of `track_a_evaluation/track_a_comparison.csv`.
`--models` takes any registry family, so the mobile and gated arms — which have
no published timings at all — need no new code, only GPU time.

---

**The ask.** Reproduce what `bruise_colab_final_analysis.ipynb` does for
inference, for the three models.

**What "inference" means there.** Two different things live under that word in
that notebook, and the plan needs both named separately:

| | What it is | Where it happens today |
|---|---|---|
| (a) the **inference pass** | one best-seed forward over the 185-image test set → per-image rows | `bruise_colab_final_analysis.ipynb` §6 "Which seed, and one inference pass"; `evaluate_at_cut` (`bruisekit/evaluate.py:13`), `yolo_native.predict_native_argmax` (`bruisekit/yolo_native.py:117`) |
| (b) the **speed benchmark** | median/p95 ms and FPS at 640 | **not computed in the analysis notebook** — §F3 only *reads* `results/final/benchmark_640.csv`. It is produced by `benchmark_speed` (`bruisekit/evaluate.py:110`), called from `bruise_colab_final.ipynb` |

Most of the value of "calculate inference" is (b), because (a) is already
cached for the headline models. Do not let the two be conflated in the meeting.

**Reuse, do not rewrite.** `bruisekit/loaders.py:266 score_run` already
dispatches over `segformer` / `smp` / `efficient` / `yolo` and returns the
per-image frame; `bruisekit/evaluate.py:110 benchmark_speed` already returns
`{median_ms, mean_ms, p95_ms, fps, n_timed}`. The work is a driver cell plus a
CSV, not new inference code.

**Open question, settled above — *which* three models.** There were two
defensible readings and they produce different work:

- the **SegFormer trio** — `segformer_b2_teacher`, `segformer_b0_direct`,
  `segformer_b0_distilled`. This is the notebook's own `SEGFORMER_MODELS` dict
  and the exact 3 rows of `track_a_evaluation/track_a_comparison.csv`
  (`raw_fwd_fps`, `mask_out_fps`, `e2e_fps`). If this is the ask, the numbers
  largely already exist and the job is to re-derive them on one machine.
- **three of the newer mobile/gated arms**, which have no published timings at
  all. `FINAL_RESULT/benchmark_stage_e.csv` covers four Stage E models, and
  nothing covers the `_rgkd` / `_b2kd` arms.

**Constraints that must carry into whatever is produced.** These are not
optional caveats, they are the reason the existing table is trustworthy:

1. **Timings from different machines are not comparable.** Stage A's numbers are
   full-A100; Stage E's are an A100 MIG 3g.40gb slice (§7.3). A new table must
   be single-machine, or it is not a table.
2. **SegFormer rows time forward + threshold; YOLO rows time raw forward only**
   (`path == "yolo_native_raw_forward"`). Keep the `path` column.
3. **`cuda.synchronize()` on both sides**, and disk read / decode / resize / H2D
   / D2H stay out of the timed region (`bruisekit/evaluate.py:110-122`).
4. **YOLO is `/255`, never ImageNet norm** — the memoised trap; ImageNet norm
   silently caps YOLO at 0.479 Dice. Native-argmax is the sole reporting path.

### 18.2 Item 2 — multi-teacher reliability-gated distillation

**The ask.** Extend Stage H from one teacher to two — SegFormer-B2 **and a large
YOLO** — into the four smaller students, and run both a per-pixel and a
per-image variant of the gate (`reliability gated group` vs plain
`reliability gated`).

This is three separable pieces of work. They are listed in dependency order and
should be costed separately, because piece (b) is small and pieces (a) and (c)
are not.

**(a) A YOLO *teacher*. This does not exist and is the largest unknown.**
YOLO appears in this repo only as a **student** (`YoloSemNet`,
`bruisekit/models.py:73`). There is no large YOLO checkpoint, no teacher
wrapper, no calibration, and no `FAMILY_SPEC` teacher row. Before this is
scoped, decide: which YOLO variant/size, trained on what, and by whom. It also
reintroduces the Ultralytics dependency that §14 gap 2b deliberately removed
from the sweep — that trade gets re-opened, and it should be re-opened
knowingly.

**(b) Teacher fusion — this already exists, in the wrong stage.**
`bruisekit/kd/kd_core.py:366 fuse_teachers` takes exactly two teachers and
supports `uniform` (mean) and `adaptive` (confidence-weighted, tempered by a
scalar `rel_b`). `bruisekit/kd/val_oracle.py` derives `rel_b` from the
validation split and gates whether the adaptive arm is allowed to run at all —
it returned `rel_b = 0.6343` for B2-vs-B5 (§6.2). **Reuse both.** The catch is
that they live in `bruisekit/kd/`, a vendored standalone CLI package that
`reliability_kd.py` does not import; the two KD systems share no code today.
Bridging them, or lifting `fuse_teachers` into `reliability_kd`, is the real
task. Note also that `fuse_teachers` is GT-free by construction while the
reliability gate `r` uses GT — combining them needs a stated ordering
(fuse-then-gate vs gate-each-then-fuse), and those are different methods.

**Non-negotiable:** if the teacher set changes, `val_oracle.py` is re-run before
any arm is scheduled (§12). A multi-teacher arm whose fusion weight was fitted
for a different teacher pair is not interpretable.

**(c) Per-pixel vs per-image as separable variants. Not expressible today.**
The gate is currently one thing: `w = g * r` at
`bruisekit/reliability_kd.py:516`, hard-multiplied, with only `gate_lo` /
`gate_hi` exposed. `g` can be neutered by abusing the ramp; **there is no way at
all to turn `r` off**, so a pixel-off / image-on ablation cannot be run. Making
the two independently switchable is new plumbing in
`ReliabilityGatedDistillLoss` (`reliability_kd.py:440-548`) and a new config
key, and it is the prerequisite for the whole "group vs normal" comparison. The
per-branch diagnostics already exist (`_sum_reliability` vs `_sum_gate`,
`:471-501`) and should be reported per variant so a null can be distinguished
from a gate that never fired.

**Scale, for the cost conversation.** 2 teacher configurations × 2 gate variants
× 4 students × 3 seeds = **48 runs**. At the `COST_HOURS` rates for the existing
gated arms (0.8–1.0 h) that is **roughly 40–48 GPU-hours**, before the controls
that any new contrast needs and before the YOLO teacher's own training. Stage H
as it stands is 27 runs. This roughly triples it.

**And the multiplicity problem is the real cost.** §8b.1 and §7c.8 exist because
an all-pairs sweep at this scale manufactures significance from noise; the
current design survives review by having **three** pre-registered confirmatory
contrasts and Holm-correcting over them. A 48-run grid needs its confirmatory
list written down **before the runs**, not after. Given the headline omnibus does
not reject (Friedman p = 0.61) and every arm sits inside the annotation-ceiling
band, the honest prior is that most of these 48 runs will land INCONCLUSIVE on
Dice. **Decide in advance what the primary endpoint is** — complete-miss
containment and the fairness gap have shown more separation than mean Dice, and
picking that after seeing the Dice results is the one thing that would sink it.

### 18.3 Item 3 — teacher → white-light student distillation

**Flagged as under-specified: this cannot be scoped until it is defined.**

A full search of the repo — code, docs, notebooks, manifests, CSVs — for
`white light` / `ALS` / `alternate light` / `modality` / `multispectral` /
`wavelength` and the rest returns **one substantive hit**: §1's sentence saying
the dataset *is* white-light photographs. The manifests carry
`stem, subject, ita_group_index_5, skin_tone_category, ITA, split, image_path,
mask_path` — no illumination or modality column. Every image in the dataset is
already white light. So "teacher → white-light student" has no referent yet.

The most likely intent is **cross-modal**: a teacher trained on alternate-light
(ALS) or otherwise non-white-light bruise imagery distilled into a student that
runs on ordinary white-light phone photographs. If that is the intent, say so
explicitly, and note that it starts from zero on the data side:

- new imagery in the second modality, with masks;
- a **modality column** in the manifest schema, and split logic that becomes
  subject × modality grouped — the current subject-grouped guarantee does not
  prevent the same subject appearing in both modalities across splits, which
  would leak invisibly and inflate every number (§2, §15);
- a decision on whether teacher and student see *paired* images of the same
  bruise or merely the same distribution. Paired is a different and much
  stronger method than unpaired; they should not be discussed as one thing.

**Nothing else in §18 depends on this**, so it should not block items 1 and 2.

### 18.4 The four questions to answer in the meeting

1. ~~Which three models does item 1 mean?~~ ~~Scheduling the GPU run.~~
   **Settled and done.** Run on two machines 2026-08-04/05; it found the §7.3a
   artefact in the process. §7.3 is the resulting single-machine table.
2. ~~**Where does the large YOLO teacher come from?**~~ **Partly settled**:
   `yolo26l-sem.pt` is fetched and Stage Y (§7d) trains a *direct* large arm on
   the identical native recipe. That answers "does capacity fix the miss rate",
   not "is it a usable teacher" — using it as a KD teacher is still §18.2 work,
   and still means accepting Ultralytics back into the sweep.
3. **What is the pre-registered primary endpoint** for the 48-run grid, and how
   many confirmatory contrasts? Written down before the runs. (§18.2)
4. **What does "white light" mean here** — what is the other modality, and does
   the data exist? (§18.3)

### 18.5 Open queue as of 2026-08-07

Item 0 is what is running now; 1–3 are the live speed thread; 4–8 are older debts
that predate it and are cheap; 9+ is registered or new work.

**RUN THIS FIRST — it gates items 0b and 9+ (§7g)**

0a. ~~**Stage P — lesion-size-stratified miss containment.**~~ **RUN AND CONFIRMED
   2026-08-07 — §7g.6.** Laptop and ORC agree number for number across two
   independent result trees. **6 of 18 cells clear zero:** small-lesion miss
   containment separates models that mean Dice cannot, `b0_direct` beats
   `unet_r50` on miss containment despite `unet_r50` holding the study's best
   median Dice, and the post-hoc mobile-beats-B5 claim **reversed**. Both Stage A
   confirmatory contrasts stay null even here, so `segformer_b0_direct` remains
   the pick (§10). One degenerate contrast exposed a reporting bug in the power
   column, now fixed (§7g.6a); no verdict changed.
   - **After Holm** (added 2026-08-07): **1 of 12** confirmatory cells survives —
     `b0_direct − yolo_sem_direct` on zero-Dice rate, p_holm 0.0168. The
     `unet_r50` dissociation does **not** survive (p_holm 0.222) and is
     descriptive only. Testing 12 cells at 5% is why the correction exists.
   - **Consequence for the queue:** the premise behind items 0b, ALS→WL and a
     Fenwick merge is **established** — but on one surviving contrast, not six.
     The strongest argument for all three is now the power result (§7g.6): the
     stratum resolves ~2 images for easy pairs and ~8 for hard ones, so most of
     the field cannot be separated at this n.

0c. ~~**Size-conditioned fairness (§8.4).**~~ **RUN 2026-08-07 — §7g.10.** The
   confound is real and large (`share_small` 0.33→0.59 across ITA groups) — so
   every unconditioned per-group number in §8.3's D5 heatmap is confounded, and
   §7g.10 Part 1 is the size of it. But conditioning **does not shrink the gaps;
   in 4 of 5 models they grow.** Dark (VI) is never the worst group and is the
   best in two. The only measurable, reproducible signal is `yolo_sem_direct` and
   `unet_r50` worst at **Light (II-III)** — which survives size conditioning and
   therefore retires §8.4's size explanation for it.
   - **Blocked on data, not analysis:** 10 of 25 small-stratum cells have no
     interval. Brown has **3** subjects, Tan **4**. Both need roughly double
     before any conditioned per-group claim is possible. **This is the single
     most concrete data-collection requirement the study has produced** and it
     should go to whoever scopes the next collection.
   - **Why first:** item 0b, an ALS→white-light stage and a Fenwick merge are all
     justified by *"models miss small bruises, so let us fix that."* Until §7g has
     intervals, that premise rests on single-seed counts.
   - **Dry-run says it has power:** 6 of 18 contrast × endpoint cells clear zero.
     `b0_direct` beats `unet_r50` on small-lesion miss containment even though
     `unet_r50` has the study's best median Dice — the Dice ranking and the miss
     ranking genuinely disagree (§7g.6).
   - **And it already caught one error:** the post-hoc "a 3.22 M mobile arm beats
     B5 on small-lesion recall" observation **reversed** under a proper paired
     bootstrap, Δ −0.115, CI [−0.210, −0.026].
   - **Minimum detectable effect** is 0.024 on the miss rate (≈1.7 of 74 images)
     for the easier pairs, 0.115 (≈8.5 images) for the hardest. Do not make
     one-image claims.

**IN PROGRESS — the current thread (§7f)**

0. ~~**Stage N gate.**~~ **RUN 2026-08-06 — measured, §7f.8.** The gate opened and
   the attribution arm overturned the premise: DINOv2 (no medical data, 86 M,
   Apache-2.0) **beat** MedSigLIP (428 M, dermatology-pretrained) by +0.166,
   CI [+0.131, +0.201]. Medical pretraining is not the cause of the gain.

0b. ⚠️ **NEXT — the resolution control (§7f.9). ~20 min GPU, one config change.
   Nothing from §7f.8 may be written up or presented until this has run.**
   The probe's score ordering is *identical* to its feature-grid ordering —
   dinov2 37×37 → 0.657, medsiglip 28×28 → 0.491, resnet50 20×20 → 0.123, rank
   correlation 1.0. A linear probe on a finer grid draws better boundaries
   regardless of what the encoder knows, so the ranking may be measuring
   resolution wearing a pretraining label.
   - **The control:** re-run `resnet50_probe` reading `layer3` (stride 16 →
     **40×40** at 640) instead of `layer4`, giving the ImageNet arm a *finer*
     grid than DINOv2 has. Ship as `resnet50_probe_hires`; leave the existing
     arm in place so both are on record.
   - **Reads:** ResNet stays ≈0.12 → the gap is representational, §7f.8 stands.
     ResNet climbs toward DINOv2 → §7f.8 is a resolution ranking and must be
     withdrawn.
   - Note the *second*, narrower confound it does **not** close: frozen linear
     probing is the benchmark DINOv2 was designed and tuned against. State that
     as a limitation regardless of how the control lands.

0c. **Then §10** — 3 arms × 3 seeds, ~20 GPU-h. Gated on 0b, and 0b may change
   which arm is the protagonist: on current evidence DINOv2 is better, 5× smaller
   and Apache-2.0 against MedSigLIP's use-restricted licence.

**RESOLVED since the 2026-08-05 revision of this list**

- ~~"Stage M is built but not run."~~ Done, and it is a **null** (§7e.8): six
  runs, all six contrasts inconclusive, complete misses worse on 6 of 6, and the
  router stayed ~88 % of the way to uniform so what was trained is a fixed-weight
  ensemble. **The routing hypothesis is still untested** — §7e.9's β = ∞ run
  (~2.6 GPU-h) is what closes it, and it is item 13 below.

**RESOLVED since the 2026-08-03 revision of this list**

- ~~"Pick a reference machine and re-time everything on it."~~ Done. §7.3 is that
  table: ORC MIG, fp32, fresh process, eight architectures, one session.
- ~~"Every latency row needs `persistence_mode` and the locked SM clock."~~ Done —
  `bruisekit/gpustate.py` supplies both, plus the SM clock sampled *under load*,
  and `speed_table` adds `fresh_process` / `process_prior_peak_MB`.
- ~~"SegFormer is a cross-machine `transformers` anomaly."~~ **Withdrawn.** It was
  a contaminated row, not a machine effect (§7.3a, §14 caveat 6 update).

**The speed thread (live, cheap, no training)**

1. **Bisect the process-state artefact.** §7.3a is reproducible on demand but the
   mechanism is unproven: running the notebook's training/scoring sections before
   the benchmark costs the transformers ~1.72× and the CNNs nothing. Cheapest
   route: fresh kernel, run §6–§12 one section at a time, re-timing SegFormer-B0
   after each. Until it is found, the guard catches the symptom but nothing
   prevents whatever causes it from affecting a *training* run too — which nobody
   has checked.
2. **`torch.compile` the mobile nets, and sweep batch size.** §7.3b shows latency
   tracks module count, not parameters — PP-MobileSeg (1.45 M, 271 modules) is
   slower than DeepLabV3+/R50 (26.7 M). Both facts may be artefacts of eager-mode
   batch-1 dispatch. If fusion collapses the launches, the current table measures
   our inference setup rather than the architectures, and **no efficiency claim in
   the paper is safe until this is known**. A batched-throughput column belongs
   beside the batch-1 one; **add, do not replace**.
3. **Sample GPU clocks under load on `mlidl`.** One terminal
   `nvidia-smi -i 1 --query-gpu=timestamp,clocks.sm,utilization.gpu,power.draw
   --format=csv -l 1 > ~/clocks.log`, the benchmark in another. `gpustate.ClockSampler`
   now does this in-process. Still the only untested half of §14 caveat 6's
   surviving claim — though it now affects only that machine's rows, not the
   study's conclusions.

**Debts owed on work already reported (no training, blocks publication)**

4. ⚠️ **Resolve a contradiction in this handbook before quoting either side.**
   §7b.7 says the LR-ASPP distilled bootstrap ran and returned a WIN (Δ +0.0218,
   CI [+0.0081, +0.0372], Holm p = 0.0042). §14 limitation 3b says no per-image
   CSV for that arm ever reached disk, so the contrast *cannot* be bootstrapped.
   Both cannot be current. Whichever is stale, fix it here.
5. **Write the Stage F per-image CSVs to disk** (§7b.7 item i, §14 3b). Nothing
   about the LR-ASPP arm is re-derivable in this bundle without them.
6. **Fairness analysis on `lraspp_mobilenetv3_distilled`** (§7b.7 item iii).
   This was the arm's *pre-registered primary endpoint* and it is missing
   entirely, while the arm won on Dice — the endpoint §7b.6 said not to claim.
7. **The teacher-miss containment check** (§7b.7 item iv). Are the distilled
   model's added misses on images the DeepLab teacher also misses? Until it
   runs, the 3 → 5 miss movement is unexplained rather than benign.
8. **Regenerate the Stage E aggregate table through `report.normalize()`**
   (§7.2a, §7b.7 item v) — and never compute a Stage E 3-seed mean from
   `efficient_test_per_seed.csv`, which covers 8 rows of the grid (§14 3c).

**Registered gaps (training required)**

9. `yolo_sem_rgkd` — 3 runs, ~2.4 GPU-h, needs Ultralytics and registration in
   five places (§14 gap 2b). Until it runs, every Stage H confirmatory contrast
   tests the gate on an **online loss only** and Holm corrects over three
   contrasts, not four. That sentence must appear wherever Stage H is reported.
10. `nnU-Net` on the canonical 697/134 split — ~8 GPU-h, needs `nnunetv2`
   (§14 gap 1).
11. `x_dkd_b5_to_b0` — configured, never executed (§14 gap 2).

**New work — gated on decisions, and two of these are now gated on evidence**

12. ⚠️ **Multi-teacher reliability-gated KD (§18.2): 48 runs, ~40–48 GPU-h.
    Do not schedule this.** It was scoped before Stage M ran. Stage M is the
    direct empirical test of its core premise and returned a null (§7e.8) —
    combining teachers produced a fused target *worse than any single teacher*.
    Crossing that mechanism with Stage H's gate, which also fired and changed
    nothing, spends 48 runs on the product of two independently measured nulls.
    Its blockers stand regardless: (a) a large YOLO teacher that does not exist,
    (b) bridging `kd_core.fuse_teachers` into `reliability_kd`, (c) making `g`
    and `r` independently switchable — today `r` cannot be turned off at all, so
    the per-pixel/per-image ablation is not expressible. **Revisit only if §7e.9's
    β = ∞ run shows genuine routing works.**
13. **Stage M β = ∞ (§7e.9): one arm, seed 0, ~2.6 GPU-h.** The cheapest open
    item in the queue and the one that decides whether §7e is reported as *"routed
    distillation does not help"* or the weaker and more honest *"a fixed-weight
    teacher ensemble does not help, and routing was never actually tested."* Those
    are different claims and only one of them is currently supported.
14. Teacher → white-light student (§18.3): still has no referent in the data. Do
    not scope until §18.4 question 4 is answered.

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
4. **Right interpreter?** `import sys; print(sys.executable)` from a cell. The
   kernel is often not the shell's Python, and a package installed into the wrong
   one is indistinguishable from a package that is missing.
5. **Right node?** `torch.cuda.is_available()`. Training refuses on CPU, but only
   after §1–§4 have run, so check first rather than 10 minutes in.
6. **`WORK_DIR` on persistent storage?** Every new checkpoint, threshold and
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
