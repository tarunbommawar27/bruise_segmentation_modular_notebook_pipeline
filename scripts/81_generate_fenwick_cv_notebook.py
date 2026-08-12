#!/usr/bin/env python
"""Emit `bruise_fenwick_cv.ipynb` -- the Fenwick labeler cross-validation.

Same generator discipline as 70/71/77/78: the notebook is an output, never hand
edited, and ships with zero executed cells.

The notebook does the parts that belong in one kernel -- cutting the matched
core, the preflight, the batch probe, and reading the result -- and hands the
two-GPU training to `scripts/80_fenwick_cv_run.py`, because CUDA_VISIBLE_DEVICES
is per-process and one kernel cannot drive two cards.

Everything is written under `FENWICK_CV_RESULTS/`. Nothing touches `results/`,
`FINAL_RESULT/` or `_work/runs/`.
"""
from __future__ import annotations

import json
from pathlib import Path

DST = Path(__file__).resolve().parent.parent / "BRUISE_UNIFIED" / "bruise_fenwick_cv.ipynb"

CELLS: list[tuple[str, str]] = []


def md(src: str) -> None:
    CELLS.append(("markdown", src.strip("\n")))


def code(src: str) -> None:
    CELLS.append(("code", src.strip("\n")))


# ─────────────────────────────────────────────────────────────────────────────
md("""
# Fenwick — which labeler's annotations train the best model?

Four people annotated the Fenwick corpus. Three of them — `hliu36`, `mzehra2`,
`nmousta5` — also annotated the shared 128-image white-light test set, so those
three are the only ones whose models can be compared on common ground. `eporti5`
is excluded by the data, not by preference.

**One model per labeler, 5-fold subject-grouped CV, single seed.**

## The two confounds this design removes before training starts

| confound | what it would do | the fix |
|---|---|---|
| **volume** | pools are 1356 / 2268 / 611 — a win could be 3.7× the data | per subject, every arm gets `n_s = min` over labelers → **360 images each, identical per-subject histogram** |
| **subjects** | different people, different bruises, different difficulty | restrict to the **48 subjects all three labelled** |

What is left varying across the three arms is **who drew the mask**, and nothing
else. That is what makes a Dice gap attributable to annotation.

## The leakage the on-disk check does not catch

`tables/verify_on_disk.csv` confirms no test *image* is in any pool. But 12–13 of
the 15 test **subjects** are, and images of one bruise on one subject are
correlated — the reason this study clusters by subject everywhere else. Training
on subject 182 and testing on another photograph of subject 182 measures
memorisation. `build_core` drops every test subject from every pool first; it
costs ~17 % of the images and it is not optional.

## Apples to apples with NIJ

Same architecture (`segformer_b0_direct`, the 3.71 M benchmark), same recipe
(`kd_core.DEFAULTS` + the study's `aux_weight=0.4`), same 640 px cache from the
same `loaders.build_cache640`, same operating-point rule (one standard error,
ties broken on misses), same reporting (`complete_miss` is `dice == 0`).
Training goes through `engine.train_run` **unmodified**.

> **The absolute Dice values here are not comparable to 0.7663.** Each fold
> trains on 288 images against NIJ's 697. Only the *between-labeler* differences
> measured here are meaningful, and quoting one of these numbers beside the NIJ
> headline would be wrong.
""")

# ── §1 ───────────────────────────────────────────────────────────────────────
md("""
## §1 — Paths and scope

`FENWICK` is the dataset root you copied off ORC. `BUNDLE` supplies the
pretrained SegFormer weights and the `bruisekit` package; the images come from
`FENWICK`. Nothing is written back to either.
""")

code('''
from pathlib import Path

FENWICK = Path("/data/FENWICK_LABELER_DATASET")   # <- the copied dataset root
BUNDLE  = Path.cwd()                              # this notebook lives in the bundle
OUT     = FENWICK.parent / "FENWICK_CV_RESULTS"
WORK    = OUT / "_work"                           # the 640 cache lands here (~1 GB)

RUNS, TABLES = OUT / "runs", OUT / "tables"
for d in (OUT, RUNS, TABLES):
    d.mkdir(parents=True, exist_ok=True)

assert (FENWICK / "by_labeler").is_dir(), f"no by_labeler/ under {FENWICK}"
assert (BUNDLE / "bruisekit").is_dir(), f"no bruisekit/ under {BUNDLE}"
assert (BUNDLE / "pretrained_weights" / "segformer_mit_b0").is_dir(), \\
    "pretrained_weights/segformer_mit_b0 is missing -- SegFormer cannot be built offline"

print(f"fenwick {FENWICK}\\nbundle  {BUNDLE}\\nout     {OUT}")
''')

# ── §2 ───────────────────────────────────────────────────────────────────────
md("""
## §2 — Import and self-test

`self_test()` is structural: no data, no GPU, no network. It guards the two
failures that would silently produce a plausible ranking — a fold assignment
that puts one subject on both sides of a split, and a subject id parsed from a
filename that does not carry one.
""")

code('''
import json, sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(BUNDLE))

from bruisekit import fenwick_cv as F

assert F.self_test(), "fenwick_cv self-test failed -- do not train on this"
print(f"labelers {F.TOP3}   folds {F.N_FOLDS}   seed {F.SEED}   family {F.FAMILY}")
''')

# ── §3 ───────────────────────────────────────────────────────────────────────
md("""
## §3 — Cut the matched core

Read the report carefully — it is the audit trail for every claim this study will
make. `core_images` **must** be identical across the three rows, and so must
`test_masks`; `build_core` raises if either is not, rather than letting an
unmatched design reach the GPU.

`core_shared_with_all` is how many of each arm's 360 images all three labelers
drew. Higher is better (the arms overlap more), but it is descriptive: the
matching guarantee comes from the subject and volume columns, not from this one.
""")

code('''
core = F.build_core(FENWICK)
F.write_core(core, OUT)

print(core.report.to_string(index=False))
print(f"\\n{core.n_per_arm} images per arm over {len(core.shared_subjects)} shared subjects")
print(f"{len(core.dropped_subjects)} test subjects dropped from every pool: "
      f"{', '.join(core.dropped_subjects)}")
if core.test_images_dropped:
    print(f"{len(core.test_images_dropped)} test image(s) dropped for an incomplete "
          f"mask set: {list(core.test_images_dropped)}")

fold_sizes = core.frame.pivot_table(index="fold", columns="labeler",
                                    values="stem", aggfunc="count")
display(fold_sizes)
print("val subjects per fold:",
      core.frame[core.frame.labeler == F.TOP3[0]].groupby("fold").subject.nunique().tolist())
''')

# ── §4 ───────────────────────────────────────────────────────────────────────
md("""
## §4 — Preflight: no subject on both sides, no test subject in any pool

Three assertions, each guarding a way this study could return a confident wrong
answer. Run them before spending GPU-hours, not after.
""")

code('''
env = F.make_env(BUNDLE, FENWICK, WORK)
print("device:", env.device)

test_subjects = set(core.test.subject)
for lab in F.TOP3:
    pool = core.frame[core.frame.labeler == lab]

    leaked = test_subjects & set(pool.subject)
    assert not leaked, f"{lab}: test subjects in the training pool: {sorted(leaked)}"

    for fold in range(F.N_FOLDS):
        val = set(pool[pool.fold == fold].subject)
        trn = set(pool[pool.fold != fold].subject)
        assert not (val & trn), f"{lab} fold {fold}: subject on both sides"

# The arms must see the same subjects in the same folds, or fold-to-fold
# comparisons across arms are not paired.
ref = core.frame[core.frame.labeler == F.TOP3[0]].groupby("subject").fold.first()
for lab in F.TOP3[1:]:
    other = core.frame[core.frame.labeler == lab].groupby("subject").fold.first()
    assert ref.equals(other), f"{lab}: fold assignment differs from {F.TOP3[0]}"

print("preflight OK -- no leakage, folds are subject-disjoint and shared across arms")
''')

# ── §5 ───────────────────────────────────────────────────────────────────────
md("""
## §5 — Build the 640 cache, and see what the batch finder picks

The cache is the same deterministic resize the NIJ study uses — bilinear for
images, nearest for masks — so training and evaluation see bit-identical pixels.
About 1500 image/mask pairs, a minute or two, ~1 GB.

`probe_batch` runs the *same* `engine.resolve_micro_batch` call `train_run`
makes, on a deepcopy, so what it prints is the batch you will actually get. In
`matched` mode it reports `micro × accum = 8`: the probe finds the largest
micro-batch this card holds under 75 % VRAM and accumulation restores the
effective batch of 8. That is what lets the two GPUs train the same recipe.
""")

code('''
cfg = F.default_cfg()
cached = F.build_cache(env, core)
print({k: len(v) for k, v in cached.items()})

print("\\nbatch finder:", F.probe_batch(env, cfg))
print({k: cfg[k] for k in ("img_size", "epochs", "patience", "effective_batch",
                           "max_probe_batch", "batch_mode", "backbone_lr",
                           "head_lr", "aux_weight", "amp")})
''')

# ── §6 ───────────────────────────────────────────────────────────────────────
md("""
## §6 — Train, one process per GPU

`CUDA_VISIBLE_DEVICES` is per-process and cannot be changed after torch
initialises, so a single kernel would serialise the arms. Run these in two
shells. Both read the core this notebook just wrote, so the two GPUs cannot end
up training against two different cuts of the data.

Resumable: a fold with `DONE.json` is skipped, an interrupted fold restarts from
`resume.pt`. Re-running either shell is safe.

The arms are volume-matched, so the 2+1 split leaves GPU 0 with twice the work.
If you would rather finish sooner, give each GPU all three labelers and half the
folds — the commented variant below.
""")

code('''
print(f"""
# shell 1 -- GPU 0, two labelers
CUDA_VISIBLE_DEVICES=0 python {BUNDLE.parent}/scripts/80_fenwick_cv_run.py \\\\
    --fenwick {FENWICK} --out {OUT} --labelers hliu36 nmousta5

# shell 2 -- GPU 1, the third
CUDA_VISIBLE_DEVICES=1 python {BUNDLE.parent}/scripts/80_fenwick_cv_run.py \\\\
    --fenwick {FENWICK} --out {OUT} --labelers mzehra2

# balanced alternative: both GPUs run all three arms, split by fold
# CUDA_VISIBLE_DEVICES=0 ... --folds 0 1 2
# CUDA_VISIBLE_DEVICES=1 ... --folds 3 4
""")
''')

md("""
Watch progress from here without touching the training processes:
""")

code('''
done = sorted(p.parent.name for p in RUNS.glob("*/DONE.json"))
print(f"{len(done)} / {len(F.TOP3) * F.N_FOLDS} folds finished")
for r in done:
    print("  ", r)
''')

# ── §7 ───────────────────────────────────────────────────────────────────────
md("""
## §7 — Score: the labeler × labeler matrix

Run **after both shells finish** — the matrix needs every arm present.

Each fold model gets its operating point fitted on **its own validation fold**,
then that one cut is applied unchanged to all three mask sets. A model is never
tuned against the annotator it is being judged by.

You can run this here, or as `--score` in one shell if the GPU is busier than the
kernel.
""")

code('''
per_image = pd.concat(
    [F.score_labeler(env, cfg, cached, lab, RUNS, TABLES) for lab in F.TOP3],
    ignore_index=True)
per_image.to_csv(TABLES / "per_image_all.csv", index=False)
print(f"{len(per_image)} per-image rows")
''')

# ── §8 ───────────────────────────────────────────────────────────────────────
md("""
## §8 — Read it

**The matrix is the result, not the diagonal.** Rows are who the model trained
on, columns are whose masks it was scored against.

* A high **diagonal** with a low off-diagonal means that model learned its own
  annotator's idiosyncrasies — style, not bruise.
* A high **column** means everyone's model agrees with that labeler, which is the
  strongest single sign that the labeler is drawing the actual injury.
* `cross_dice` — mean against the *other two* — is the column the ranking sorts
  on, for exactly that reason.

Read the miss rate beside the Dice, never after it. Dice on this task is
saturated and the models separate on complete misses (`dice == 0`).
""")

code('''
print("CROSS-LABELER TEST DICE   (rows = trained on, cols = scored against)")
display(F.cross_matrix(per_image).round(4))

print("COMPLETE-MISS RATE  (dice == 0)")
display(F.cross_matrix(per_image, "complete_miss").round(3))

table = F.labeler_table(per_image)
table.to_csv(TABLES / "labeler_table.csv", index=False)
for m in ("dice", "complete_miss"):
    F.cross_matrix(per_image, m).to_csv(TABLES / f"cross_{m}.csv")
display(table)
''')

md("""
### The verdict, and the margin

`paired_contrast` pairs by image and fold — both arms saw the same test images
from the same split — and bootstraps over **subjects**, so the interval is not
narrowed by treating correlated images as independent.

`print_verdict` applies the study's 0.01 Dice margin. An interval inside ±0.01 is
reported as **equivalent**, not as a win for whoever came top. That rule is why
this project has published nulls, and it applies here too: with 48 subjects and
one seed, "no labeler separates" is a real and likely outcome — and it is a
finding, not a failed experiment. It would say the three annotators are
interchangeable for training purposes, which is exactly what you want to know
before commissioning more annotation.
""")

code('''
contrasts = [F.paired_contrast(per_image, a, b, evl)
             for evl in F.TOP3
             for i, a in enumerate(F.TOP3) for b in F.TOP3[i + 1:]]
(TABLES / "contrasts.json").write_text(json.dumps(contrasts, indent=2))

F.print_verdict(table, contrasts)
display(pd.DataFrame(contrasts))
''')

# ── §9 ───────────────────────────────────────────────────────────────────────
md("""
## §9 — What this licenses, and what it does not

**Licensed.** A statement about which annotator's masks train the most
transferable model *on the Fenwick white-light corpus, at 360 images per arm,
one seed*.

**Not licensed.**

* Comparison against the NIJ headline numbers. Different corpus, different split,
  40 % of the training images.
* A claim about annotation *accuracy*. This measures learnability and
  cross-annotator agreement. A labeler could be consistently, learnably wrong,
  and this design would rank them first. Nothing here reads a clinical
  ground truth, because none exists in the dataset.
* Anything about `eporti5`, who has no test masks.
* A seed-level claim. One seed. Stage Y seed 2 collapsed on a much larger
  training set, so treat any single arm that lands far below the others as
  possibly a bad run until a second seed says otherwise.

**What the off-diagonal is worth on its own.** Whatever the ranking says, the
three mask sets scored against each other are a direct measurement of
inter-annotator disagreement on our own data — the annotation ceiling that Stage
N3 argues is the binding constraint. That number is worth reporting even if no
labeler separates.
""")

# ─────────────────────────────────────────────────────────────────────────────
nb = {
    "cells": [
        {"cell_type": t, "metadata": {},
         "source": s.splitlines(keepends=True),
         **({"outputs": [], "execution_count": None} if t == "code" else {})}
        for t, s in CELLS
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

DST.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"wrote {DST}  ({len(CELLS)} cells, "
      f"{sum(1 for t, _ in CELLS if t == 'code')} code)")
