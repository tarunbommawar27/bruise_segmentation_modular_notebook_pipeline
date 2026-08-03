# Bruise Segmentation — Modular Notebook Pipeline

Binary segmentation of bruises in white-light photographs, built as a **thin
notebook over a fat library**. One notebook runs the whole study; every model,
metric and statistical test lives in tested modules beside it.

The pipeline covers seven stages — five headline models, two CNN baselines, a
SegFormer-B5 distillation grid, four mobile-scale architectures, two
cross-architecture distillation axes, a descriptive analysis suite and a
confirmatory significance layer.

> **No data, no weights in this repository.** The study uses 1016 clinical
> photographs of 143 human subjects. Neither the images, the masks, nor the 27
> trained checkpoints are published here. See
> [Data availability](#data-availability) below.

---

## The finding that governs everything else

**Every model is at the annotation ceiling.** Human annotators disagree with each
other by more than the models disagree with each other:

| Comparison | mean Dice |
|---|---|
| human: gbarimah vs majority | 0.873 |
| human: erik vs majority | 0.866 |
| **model: segformer_b2_teacher** | **0.769** |
| **model: segformer_b0_direct** | **0.766** |
| human: gbarimah vs erik | 0.755 |
| **model: lraspp_mobilenetv3_b2kd** | **0.721** |
| human: paul vs majority | 0.700 |
| human: paul vs gbarimah | 0.581 |

The entire model field sits between `paul_vs_majority` (0.700) and
`gbarimah_vs_erik` (0.755). **A 0.005 Dice gap between two models is not a
result** — it is inside the noise floor of the labels themselves.

Three consequences run through the whole codebase:

1. Lead with **complete-miss rate** (`dice == 0`), not mean Dice. The clinical
   question is *was the bruise found at all*, not how neatly it was outlined.
2. Never claim model X beats model Y on a sub-0.05 Dice difference without a
   paired subject-level bootstrap whose interval excludes zero.
3. Always show the annotation ceiling next to a model ranking.

---

## Stages

| Stage | What it covers | Runs |
|---|---|---|
| **A** | SegFormer-B2 teacher, B0 direct, B0 distilled, YOLO26n direct + distilled | 15 |
| **B** | U-Net and DeepLabV3+ (ResNet-50) baselines; nnU-Net registered as a gap | 6 |
| **C** | SegFormer-B5 teacher and 10 knowledge-distillation arms | 13 |
| **E** | Mobile baselines: PP-MobileSeg-Tiny, TopFormer-Tiny, LR-ASPP MobileNetV3, Fast-SCNN | 12 |
| **F** | DeepLabV3+ → mobile-student distillation | 12 |
| **H** | **Reliability-gated distillation** + SegFormer-B2 as teacher for every small student | 27 |
| **D** | Descriptive analysis: distributions, bootstrap intervals, fairness, size confound, annotation ceiling | — |
| **G** | Confirmatory significance: omnibus first, then a pre-specified contrast family with multiplicity control | — |

---

## Stage H — reliability-gated distillation

The contribution this repository was extended for, and an honest null.

### The failure it targets

Stage F recorded a pre-registered failure: distilling DeepLabV3+ into LR-ASPP
gained Dice but moved complete misses **3 → 5**, and into Fast-SCNN **7 → 13**
with the skin-tone fairness gap widening 0.136 → 0.199.

The mechanism is not subtle. Standard response KD regresses the student onto
`sigmoid(teacher_logits / T)` at **every pixel of every image with one fixed
weight**. On an image the teacher completely misses, that soft target is not
weak — it is confidently, uniformly *empty*, applied with exactly the force it
gets on an image the teacher got right. The student is told, with full
confidence, that there is no bruise on precisely the images the clinical metric
exists to catch.

### The mechanism

With `p` the calibrated teacher probability and `y ∈ {0,1}` the label:

```
per-pixel reliability   r  = 1 − |2p − 1| · |p − y|
per-image gate          g  = clip((Dice_T − 0.10) / 0.40, 0, 1)     Dice_T = soft Dice
weight                  w  = g · r
coverage                   = mean(w)
effective alpha         α' = α + (1 − α)(1 − coverage)
loss                       = α' · supervised(GT) + (1 − α') · Σ(w · BCE) / Σw
```

| teacher is… | confidence | error | `r` | soft term |
|---|---|---|---|---|
| confidently right | ≈1 | ≈0 | ≈1 | full weight |
| **uncertain** | ≈0 | any | **≈1** | **full weight** |
| confidently wrong | ≈1 | ≈1 | ≈0 | suppressed |

**The middle row is the design.** Down-weighting by error alone would delete
exactly the pixels the teacher is unsure about — the dark knowledge distillation
exists to transfer. Multiplying by confidence keeps uncertainty intact and
removes only assertive error.

There is **one teacher**, not two. The "second opinion" is the ground truth,
which is available at training time and only at training time. The gate is
therefore a training mechanism; the deployed student is an ordinary model and
carries no gate.

### Why it is a one-variable contrast

`reliability_kd.self_test()` asserts three properties rather than claiming them:

1. With `r ≡ 1, g ≡ 1` the loss **equals** `losses.DistillLoss` (measured 2.4e-07).
2. A teacher asserting empty on a real bruise ⇒ `g = 0` ⇒ the loss equals
   `losses.SupervisedLoss` exactly.
3. `p = 0.5` ⇒ `r = 1`.

Property 1 is why **α is not re-tuned** for gated arms — they inherit their
control's α — and why `α'` hands the gated-away weight back to the supervised
term instead of quietly lowering the effective KD weight.

### Results

The gate fired: **coverage 0.906**, effective α 0.638 against a nominal 0.600,
**2.8 %** of image-views fully gated off, 1.9 % teacher near-misses. Neither
degenerate case (gate inert, or gate eating the arm) occurred.

And it changed nothing measurable:

| contrast (teacher and α held fixed) | Δ Dice | 95 % CI | verdict |
|---|---|---|---|
| `segformer_b0_rgkd` − `segformer_b0_distilled` | −0.0024 | [−0.0156, +0.0110] | INCONCLUSIVE |
| `lraspp_mobilenetv3_rgkd` − `lraspp_mobilenetv3_b2kd` | +0.0033 | [−0.0133, +0.0204] | INCONCLUSIVE |
| `fastscnn_rgkd` − `fastscnn_b2kd` | −0.0087 | [−0.0321, +0.0140] | INCONCLUSIVE |

Complete misses did not move either, and the two mobile gate contrasts are
sign-inconsistent across seeds (1 of 3 positive). **Reported as the null it is.**

What *did* land, each interval excluding zero:

| contrast | Δ Dice | 95 % CI |
|---|---|---|
| **B2 vs DeepLabV3+ teacher** (Fast-SCNN, KD method fixed) | **+0.0343** | [+0.0084, +0.0593] |
| KD vs no KD (LR-ASPP) | +0.0265 | [+0.0065, +0.0517] |
| KD vs no KD (Fast-SCNN) | +0.0364 | [+0.0045, +0.0659] |

The teacher swap is the result that justifies the stage: same student, same
recipe, same α, and only the teacher changes — it reverses Stage F's null.

The most interesting result is not about the gate at all. On Fast-SCNN the ITA
fairness gap goes **0.160 (p = 0.021, significant)** with no KD → **0.064
(p = 0.358, not significant)** with plain B2 response KD → **0.142 (p = 0.029,
significant again)** with gating. A B2 teacher removes a skin-tone gap that
neither the direct model nor the DeepLabV3+ arm removes, and gating undoes it.
That is the pre-registered primary endpoint. At 28 subjects it is a single
comparison — flag it, do not over-read it.

**Not run:** `yolo_sem_rgkd`. The gate for YOLO's offline pseudo-mask route is
implemented and tested (`build_gated_yolo_dataset`) but deliberately
unregistered, so every confirmatory Stage H result tests the gate on an *online*
loss only. Re-enabling is three lines; see `PROJECT_HANDBOOK.md` §14 gap 2b.

---

## Method notes that are easy to get wrong

**The loss is not Hinton KD.** No `T²` term, and the student's logits are not
temperature-scaled. `T` is a *calibration* temperature fitted by NLL on
validation (Guo et al. 2017, L-BFGS on `log T`), so the teacher is used as a
better-calibrated estimate of P(bruise | pixel). Cite Menon et al. (2021), not
Hinton et al. (2015).

**Normalisation belongs to the model, not the loader.** The dataloader emits raw
`[0,1]` pixels. SegFormer applies ImageNet statistics; YOLO applies plain `/255`,
because Ultralytics' BatchNorms carry frozen running statistics for that
distribution. Feeding YOLO ImageNet-normalised pixels caps it at 0.479 Dice with
**no threshold able to recover it**.

**The threshold is not the argmax.** The logit cut is swept over 481 values on
the 134 validation images. These sweeps are extraordinarily flat — B2's val Dice
moved 0.009 across thresholds from 0.154 to 0.959 — so the argmax fits the
validation set's sampling error. Every cut within **one standard error** of the
peak is treated as tied and the tie is broken by **lowest complete-miss rate**.
A checkpoint without its `operating_point.json` is unusable; the registry
enforces this.

**Two definitions of "complete miss" exist in the codebase.**
`metrics.summarize` counts `pred_positive == 0`; `report.normalize` counts
`dice == 0`. The first is a strict subset — it misses the case where a model
fires confidently on the wrong region, which is still a complete miss to a
clinician. **All reported tables use `dice == 0`.**

**Significance resamples subjects, not images.** 185 test images come from 28
subjects and images of the same bruise are strongly correlated; resampling
images would give intervals ≈ √(185/28) ≈ 2.6× too narrow. Contrasts are paired
on the same resampled subject list, evaluated at 10 000 draws, on three
endpoints from one set of draws, with Holm–Bonferroni inside a comparison list
fixed before the models were trained. `INCONCLUSIVE` is kept separate from
`NON-INFERIOR`: calling a wide interval "equivalent" reports absence of evidence
as evidence of absence.

---

## Repository layout

```
BRUISE_UNIFIED/
├── bruise_unified.ipynb        the notebook — 78 cells, seven stages
├── PROJECT_HANDBOOK.md         the reference: every decision and why
├── requirements.txt
└── bruisekit/                  the library
    ├── data.py                 dataset, loaders, augmentation (emits raw [0,1])
    ├── models.py               SegFormerNet, YoloSemNet, build_param_groups
    ├── losses.py               DiceBCE, SupervisedLoss, DistillLoss
    ├── metrics.py              per-image Dice/IoU, threshold-free AP
    ├── engine.py               the training loop, resume contract, calibration
    ├── sweep.py                threshold sweep and cut selection
    ├── evaluate.py             scoring, fairness, speed benchmark
    ├── paths.py                the only host-aware module
    ├── registry.py             the train-or-skip brain (three tiers)
    ├── loaders.py              run → live model → test numbers
    ├── report.py               per-image CSV → tables (descriptive)
    ├── significance.py         omnibus, contrast family, multiplicity
    ├── efficient_models.py     the four mobile architectures
    ├── distill_efficient.py    Stage F: DeepLabV3+ → mobile KD shim
    ├── reliability_kd.py       Stage H: the gated loss and its shims
    ├── weights.py              backbone download, verify, provenance
    ├── mmcv_shim.py            minimal mmcv stand-in
    ├── vendor/                 StrideFormer, PP-MobileSeg, TopFormer — verbatim
    └── kd/                     Stage C distillation suite — vendored unmodified

scripts/
├── unified_lib/                source of truth for the ★ modules above
├── 60_build_unified_bundle.py  assemble + verify the bundle
├── 61_generate_unified_notebook.py   emit the notebook
├── 62_zip_unified_bundle.py    package the full bundle
├── 63_zip_rgkd_overlay.py      package a code-only patch overlay
├── 70_b2_teacher_significance.py     contrasts against the B0-direct boundary
├── 74_generate_b2_decks.py     the result decks (mean and median endpoints)
├── 75_generate_meeting_cheatsheet.py formulas and shapes, .tex + .pdf
└── vendor_efficient_nets.py    refresh the vendored architectures

docs/                           LaTeX sources and scope notes
```

**The notebook and `bruisekit/*.py` are build artefacts.** Edit
`scripts/unified_lib/` and re-run the generators; a direct edit is reverted by
the next build. This is why the generator scripts ship alongside the output.

---

## Design principles

**Thin notebook, fat library.** Every notebook cell is config, a call into
`bruisekit`, or a rendering of what came back. No cell defines a model, a metric
or a training loop.

**The registry decides before any compute happens.** Every run resolves to one
of three tiers and the plan is printed before a single gradient is computed:

| Tier | Meaning | Cost |
|---|---|---|
| **WEIGHTS** | a usable checkpoint exists | nothing trains |
| **RESULTS** | no checkpoint, but metrics were recorded | nothing trains; labelled `cached` |
| **MISSING** | neither | the only case where training is proposed |

**Fail loud, never silently.** A MISSING run stays missing in every table it
would have fed. It is never dropped, never back-filled from a neighbouring seed,
never averaged away. A contrast whose models are absent is **named** as skipped.

**Runtime shims, not edits to tested code.** `engine.py`, `losses.py` and
`yolo_native.py` are extracted verbatim from the notebook they were developed
in, so an edit there is reverted by the next build. New teachers and new losses
are installed by rebinding module globals at runtime, and every shim is inert
outside its own context — so adding Stage H cannot change a Stage A number.

**One shared recipe.** Identical LR split, schedule, loss, batch policy and
seeds across every architecture. If a model needs different hyperparameters to
work, that is a finding to report, not a knob to turn.

---

## Running it

```bash
pip install -r BRUISE_UNIFIED/requirements.txt
jupyter lab BRUISE_UNIFIED/bruise_unified.ipynb
```

Everything is controlled from the first cell:

```python
RUN_STAGES     = "ABCDEH"   # any subset
ALLOW_TRAINING = False      # True only if you intend to spend GPU-hours
RECOMPUTE_FROM_WEIGHTS = False
EFFICIENT_SEEDS = (0, 1, 2)
RGKD_SEEDS      = None      # None = same as EFFICIENT_SEEDS
```

With the defaults and the data bundle present, *Run All* reproduces every table
and figure from cached per-image results in about 30 seconds on a laptop with no
GPU. `ALLOW_TRAINING = True` trains the 51 Stage E/F/H runs, ≈40 GPU-hours,
resumable — anything with a `DONE.json` is skipped.

Verify the gate before spending GPU time:

```python
from bruisekit import reliability_kd as RK
RK.self_test()      # 5 checks, CPU, under a second; raises on failure
```

**On an offline cluster node,** export `BRUISE_NO_PIP=1` before starting the
kernel. Optional heavyweight dependencies (`ultralytics`,
`segmentation-models-pytorch`) are installed only at the point a run that needs
them is about to be built, and this turns a missing package into a named error
rather than minutes of pip retry backoff.

---

## Data availability

The dataset is **1016 white-light photographs of bruises on 143 human subjects**,
with expert annotations. It is not in this repository and will not be:

- no images or masks
- no manifests, split definitions or per-subject skin-tone (ITA) labels
- no trained checkpoints or pretrained backbones
- no per-image result tables

`.gitignore` excludes these by explicit path as well as by pattern, so a future
directory rename cannot silently start publishing patient photographs.

The one derived table that *is* included is
`BRUISE_UNIFIED/interlabeler_agreement_640.csv` — per-image Dice between human
annotators, the annotation ceiling of §1. It carries no images and no identifiers
beyond pseudonymous image stems.

**Consequence: this repository is not runnable end-to-end as published.** It is
the complete method, not a reproducible artefact. Access to the data is a
separate question governed by the consent scope of the photographs, not by a
licence file.

---

## Known gaps

Reported rather than omitted, because "the baseline we did not run" is exactly
what a reader needs:

1. **nnU-Net** — never run on the canonical 697/134 split. Registered as a gap;
   needs `nnunetv2` and ≈8 GPU-hours.
2. **`x_dkd_b5_to_b0`** — a Stage C arm configured but never executed.
3. **`yolo_sem_rgkd`** — implemented and tested, deliberately unregistered
   (above). Note that an *unregistered* family is invisible to the registry, so
   `PROJECT_HANDBOOK.md` §14 is the only record it is owed.
4. **Speed tables span two devices** — Stage A on a full A100, Stage E on an
   A100 MIG 3g.40gb slice. Not comparable; re-time on one device before
   publishing a latency claim.
5. **28 test subjects** is a small denominator. Most fairness comparisons are not
   significant and must not be reported as if they were.

`PROJECT_HANDBOOK.md` §15 documents sixteen traps hit during development and the
automated guard added for each — including two dependency-install regressions,
a patched notebook that kept running its old cells, and a `transformers` version
refactor that silently renames every SegFormer state-dict key.

---

## Documentation

| File | Contents |
|---|---|
| `BRUISE_UNIFIED/PROJECT_HANDBOOK.md` | the reference — every stage, decision, result and trap |
| `docs/reliability_gated_kd_scope.md` | Stage H scope note, pre-registered before the runs, outcome appended after |
| `docs/bruise_meeting_cheatsheet.tex` | losses, formulas, tensor shapes, hyperparameters — one place |
| `docs/b5_distillation_scope.md` | Stage C scope |

Build the PDFs and decks with `scripts/74_*` and `scripts/75_*`; the LaTeX
sources are tracked, the built binaries are not.
