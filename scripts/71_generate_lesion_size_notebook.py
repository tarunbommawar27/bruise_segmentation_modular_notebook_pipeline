#!/usr/bin/env python
"""Emit `BRUISE_UNIFIED/bruise_lesion_size.ipynb` -- Stage P, runs top to bottom.

WHAT THIS STAGE ASKS, IN ONE SENTENCE
---------------------------------------
89% of all complete misses in this study fall in the smallest four GT-area
deciles. Are the differences between models INSIDE that stratum real, or is the
endpoint underpowered at 185 images from 28 subjects?

WHY IT RUNS BEFORE THE QUEUED GPU WORK
----------------------------------------
Stage N's layer3 control, ALS->WL distillation and a Fenwick merge are all
justified by the same sentence: "models miss small bruises, so let us fix that."
If that difference is not resolvable at n = 28 subjects then the premise is not
established, and the correct next move is more data rather than more mechanism.
This notebook costs minutes on a laptop and settles which of those it is.

NO GPU, NO TORCH, NO CHECKPOINTS
----------------------------------
Every number is a function of per-image Dice and per-image GT area, both already
on disk in FINAL_RESULT/<lineage>/per_image_*.csv. Same property as Stage D.

EVERY SETTING, AND WHERE IT CAME FROM
---------------------------------------
  LINEAGE = "RESULT_AUGUST_08"  the current lineage -- every reported number for
                                Stages A-Y. FINAL_RESULT/ at the top level is the
                                older export; mixing them compares models scored
                                on different runs.
  N_BINS  = 10                  deciles give 18-19 images per bin at n=185. Any
                                finer and a bin is a handful of subjects.
  PRIMARY = D1-D4               74 images carrying 89% of the misses -- the
                                largest stratum still meaningfully "small". D1
                                alone (19 images) is reported but is NOT primary.
  N_BOOT  = 10000               matches significance.py. The Monte-Carlo error at
                                that count is far below the interval width, which
                                at 28 subjects is the thing that dominates.
  SEED    = 0                   the study's convention.
  RESULTS LESION_SIZE_RESULTS/  at the bundle root. Never results/, FINAL_RESULT/
                                or _work/runs/ -- same isolation as Stages M/N.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DST = ROOT / "BRUISE_UNIFIED" / "bruise_lesion_size.ipynb"

CELLS: list[tuple[str, str]] = []


def md(src: str) -> None:
    CELLS.append(("markdown", src.strip("\n")))


def code(src: str) -> None:
    CELLS.append(("code", src.strip("\n")))


# ── §0 ───────────────────────────────────────────────────────────────────────
md("""
# Stage P — lesion-size-stratified miss containment, and whether it has power

**One question:** complete misses are not spread over the test set — 89% of them
fall in the smallest four GT-area deciles, 49% in the smallest one. Are the
differences *between models* inside that stratum real, or can 185 images from 28
subjects not resolve them either?

**No GPU. No torch. No checkpoints.** Everything is a function of per-image Dice
and per-image GT area, both already on disk.

---

### Why this runs before any queued GPU work

Stage N's `layer3` control, ALS→white-light distillation and a Fenwick merge are
all justified by the same sentence: *"models miss small bruises, so let us fix
that."* If that difference is not resolvable at n = 28 subjects, the premise is
not established and the correct next move is **more data**, not more mechanism.

### The two outcomes, both of which are results

| outcome | what it means | what to do |
|---|---|---|
| **CI clears zero** | models that tie on mean Dice separate on small-lesion miss containment | report it; it names the lever, and the queued work is justified |
| **CI spans zero** | the endpoint is underpowered at this sample size | report the **minimum detectable effect** — that converts "we found nothing" into "an effect below X is invisible here", which is a statement about the experiment, not the models |

### How to run

Top to bottom. §1 is the only cell you edit. Takes a few minutes at
`N_BOOT = 10000`; drop it to 2000 for a fast look.
""")

# ── §1 CONFIG ────────────────────────────────────────────────────────────────
md("""
## §1 — Configuration

**The only cell you edit.** Every switch below is stated with its default and
what flipping it costs.
""")

code('''
# ─── paths ───────────────────────────────────────────────────────────────────
# BUNDLE_ROOT: None auto-detects (a directory counts only if it has BOTH
# bruisekit/ and manifests/train.csv). Set it explicitly if autodetect picks the
# wrong copy -- on ORC there is usually only one, but a stale /home copy has
# fooled this before.
BUNDLE_ROOT = None
# BUNDLE_ROOT = "/scratch/tbommawa/BRUISE_UNIFIED"     # ORC
# BUNDLE_ROOT = r"C:\\BRUISE_SEGMENTATION_PROJECT\\BRUISE_UNIFIED"   # laptop

# WORK_DIR: where THIS host writes outputs. This matters and is easy to get
# wrong. On ORC the work tree is a scratch directory OUTSIDE the bundle, and if
# you do not name it here `setup()` silently defaults to <bundle>/_work -- which
# is a different, probably empty, tree. On the laptop <bundle>/_work is correct.
WORK_DIR = None
# WORK_DIR = "/scratch/tbommawa/bruise_work"           # ORC

# LINEAGE: which per-image export to read.
#   "auto"   <- RECOMMENDED. Scans the bundle AND the work dir and picks the
#               directory holding the most of the models we want. Prints every
#               candidate first so a wrong pick is visible, not silent.
#   "<name>" <- tried under FINAL_RESULT/, the bundle root and the work dir.
#   "/abs/path" <- used exactly as given, no search.
#
# The two layouts this project actually uses are NOT the same, which is why this
# defaults to discovery rather than to a path:
#   laptop bundle : <bundle>/FINAL_RESULT/RESULT_AUGUST_08/
#   ORC run       : <work>/outputs/     e.g. /scratch/tbommawa/bruise_work/outputs
# Never mix two lineages in one table -- they are different runs of the same
# models. §4 prints exactly which directory answered.
LINEAGE = "auto"

# EXTRA_SEARCH: additional directories to scan when LINEAGE="auto". Add anything
# non-standard here rather than editing the module.
EXTRA_SEARCH = ()
# EXTRA_SEARCH = ("/scratch/tbommawa/bruise_work/outputs",
#                 "/scratch/tbommawa/bruise_work/runs")

# MODELS: None = lesionsize.DEFAULT_MODELS (18 models, aliases already resolved).
# Pass a tuple to restrict. Absent models are reported and skipped, never
# silently dropped.
MODELS = None

# ─── statistics ──────────────────────────────────────────────────────────────
N_BOOT = 10000      # matches significance.py. 2000 is enough for a quick look.
SEED   = 0          # the study's convention.
ALPHA  = 0.05       # 95% intervals.

# ─── switches ────────────────────────────────────────────────────────────────
RUN_DESCRIPTIVE = True   # §4-§5  the per-model tables. Seconds. Always leave on.
RUN_BOOTSTRAP   = True   # §6     marginal CIs per model in the primary stratum.
RUN_CONTRASTS   = True   # §7     THE ANSWER -- the pre-registered contrast list.
RUN_POWER       = True   # §8     minimum detectable effect. Leave ON: this is
                         #        the deliverable when §7 finds nothing.
RUN_D1_ONLY     = True   # §7b    repeat the contrasts on D1 alone (19 images).
                         #        Reported as fragile by construction, never as
                         #        a primary claim.
RUN_FAIRNESS    = True   # §8b    the size-conditioned ITA analysis §8.4 has
                         #        been asking for since Stage D.

# Which models to run the fairness conditioning on. All 18 is a wall of text;
# these are the ones a fairness claim would actually be made about.
FAIRNESS_MODELS = ("segformer_b0_direct", "segformer_b0_distilled",
                   "segformer_b2_teacher", "yolo_sem_direct", "unet_r50")
# mean_recall, not median_dice: §1's endpoint is finding the bruise, and recall
# is the per-group quantity that moved in the Stage P contrasts.
FAIRNESS_ENDPOINT = "mean_recall"

WRITE_RESULTS = True     # write CSVs to LESION_SIZE_RESULTS/ at the bundle root.
                         # False = print only, touch nothing on disk.

print("config loaded -- nothing has been read yet")
''')

# ── §2 setup ─────────────────────────────────────────────────────────────────
md("""
## §2 — Environment

`device="cpu"` on purpose: this stage never builds a tensor, and grabbing a GPU
it will not use would block a real training job on a shared node.
""")

code('''
import sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Make the bundle importable when the notebook is opened from elsewhere.
_here = Path.cwd()
for _c in (_here, *_here.parents):
    if (_c / "bruisekit").is_dir() and (_c / "manifests" / "train.csv").exists():
        if str(_c) not in sys.path:
            sys.path.insert(0, str(_c))
        break

from bruisekit import paths, lesionsize as LS

env = paths.setup(root=BUNDLE_ROOT, work=WORK_DIR, device="cpu")
print(env.describe())

# WHERE ARE THE PER-IMAGE CSVs ON THIS HOST? Printed before anything is read,
# because the laptop bundle and an ORC run put them in different trees and a
# wrong guess should be visible here rather than fatal three cells later.
print("\\n" + "-" * 78)
_found = LS.find_lineages(env, EXTRA_SEARCH)
print("-" * 78)

print(f"\\nresults dir     : {env.root / LS.RESULTS_DIRNAME}"
      f"{'' if WRITE_RESULTS else '   (WRITE_RESULTS=False -- nothing written)'}")
print(f"manifest        : {env.root / 'manifests' / 'test.csv'}")
if _found.empty:
    print("\\n*** No per-image CSVs found. Set EXTRA_SEARCH in §1 to the directory")
    print("*** holding them (on ORC that is usually <work>/outputs), or set")
    print("*** LINEAGE to an absolute path. §4 will fail until this is fixed.")

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 400)
pd.set_option("display.float_format", lambda v: f"{v:.4f}")
''')

# ── §3 pre-registration ──────────────────────────────────────────────────────
md("""
## §3 — The pre-registration, printed before any number is read

Stratum, endpoints and the contrast list are **module constants** in
`bruisekit/lesionsize.py`, not choices made in this notebook. A stratum picked
after seeing which stratum separates the models is not a test, it is a search.

Two of the six contrasts are labelled **exploratory** because they were suggested
by looking at the descriptive table on 2026-08-07. That label is the difference
between a hypothesis and a fishing trip, and it travels into the output CSV.
""")

code('''
print(f"primary stratum   : {LS.PRIMARY_STRATUM}  (deciles of GT area, cut globally)")
print(f"primary endpoints : {LS.PRIMARY_ENDPOINTS}")
print(f"bins              : {LS.N_BINS} deciles")
print(f"\\npre-registered contrasts ({len(LS.CONTRASTS)}):")
for a, b, kind, q in LS.CONTRASTS:
    print(f"\\n  [{kind:<13}] {a}  vs  {b}")
    print(f"      {q}")
print("\\n" + "=" * 78)
print("TWO MISS DEFINITIONS ARE KEPT SEPARATE THROUGHOUT:")
print("  zero_dice   dice == 0            <- the published endpoint")
print("  empty_pred  pred_positive == 0   <- what the per-seed tables count")
print("  wrong_place = zero_dice - empty_pred: output pixels, ALL of them wrong.")
print("=" * 78)
''')

# ── §4 load + bin ────────────────────────────────────────────────────────────
md("""
## §4 — Load the lineage and cut the deciles

**Check the `lineage :` line this prints.** It is the directory that actually
answered, and on ORC that is normally `<work>/outputs`, not
`FINAL_RESULT/RESULT_AUGUST_08` — that tree only exists in the shipped laptop
bundle. If the wrong one was picked, set `LINEAGE` in §1 to the path §2 listed.

Deciles are cut **once** on the global GT-area vector, never per model — if each
model got its own bin edges the columns would not be comparable.

The loader **raises** if two models disagree about an image's GT area. They score
the same 185 masks, so a disagreement means different test sets or different mask
versions, and every cross-model number below would be silently meaningless.
""")

code('''
tables = LS.load_lineage(env, LINEAGE, MODELS, EXTRA_SEARCH)
key = LS.assign_bins(tables, verbose=True)

PRIMARY = key.is_primary.to_numpy()
SMALLEST = key.is_smallest.to_numpy()

print(f"\\nprimary stratum : {int(PRIMARY.sum())} images, "
      f"{key[PRIMARY].subject.nunique()} subjects")
print(f"smallest decile : {int(SMALLEST.sum())} images, "
      f"{key[SMALLEST].subject.nunique()} subjects")
''')

# ── §5 descriptive ───────────────────────────────────────────────────────────
md("""
## §5 — Descriptive tables

No inference yet. `head` is one row per model; `long` is one row per
(model, decile) and is the full per-model breakdown.
""")

code('''
if RUN_DESCRIPTIVE:
    head = LS.headline(tables, key)
    long = LS.by_bin(tables, key)

    cols = ["model",
            "all_mean_dice", "all_median_dice",
            "all_zero_dice_n", "all_empty_pred_n", "all_wrong_place_n",
            "D1_D4_n", "D1_D4_zero_dice_n", "D1_D4_zero_dice_rate",
            "D1_D4_mean_recall", "D1_D4_median_dice",
            "D1_zero_dice_n", "D1_mean_recall"]
    print("PER-MODEL HEADLINE  (sorted by misses in the primary stratum)")
    print("=" * 150)
    print(head[cols].to_string(index=False))

    print("\\n\\nZERO-DICE COUNT BY DECILE  (D1 smallest)")
    print("=" * 150)
    piv = long.pivot(index="model", columns="bin", values="zero_dice_n")
    piv = piv.reindex(columns=[f"D{i+1}" for i in range(LS.N_BINS)]).reindex(head.model)
    piv["TOTAL"] = piv.sum(axis=1)
    print(piv.astype(int).to_string())

    print("\\n\\nMEAN RECALL BY DECILE")
    print("=" * 150)
    print(long.pivot(index="model", columns="bin", values="mean_recall")
          .reindex(columns=[f"D{i+1}" for i in range(LS.N_BINS)])
          .reindex(head.model).to_string())
else:
    head = long = None
    print("RUN_DESCRIPTIVE = False -- skipped")
''')

# ── §6 marginal CIs ──────────────────────────────────────────────────────────
md("""
## §6 — Marginal confidence intervals in the primary stratum

Subject-level cluster bootstrap: **subjects** are resampled with replacement and
all of a chosen subject's stratum images come along. Resampling images would
treat correlated observations as independent and produce intervals far too
narrow.

Only subjects with at least one image *in the stratum* are resampled — including
the others adds draws contributing zero rows, which inflates the variance for
reasons that have nothing to do with the data.

**Rates, never counts.** A bootstrap draw does not contain a fixed number of
images, so a resampled count would measure the draw size.
""")

code('''
if RUN_BOOTSTRAP:
    rows = []
    for name, df in tables.items():
        for ep in LS.PRIMARY_ENDPOINTS:
            r = LS.bootstrap_stratum(df, key, PRIMARY, ep, N_BOOT, SEED, ALPHA)
            rows.append({"model": name, **r})
    marginal = pd.DataFrame(rows)

    for ep in LS.PRIMARY_ENDPOINTS:
        s = marginal[marginal.endpoint == ep].copy()
        s = s.sort_values("point", ascending=(ep == "zero_dice_rate"))
        print(f"\\n{ep}  --  primary stratum {LS.PRIMARY_STRATUM}, "
              f"{int(s.n_images.iloc[0])} images, {int(s.n_subjects.iloc[0])} subjects")
        print("=" * 92)
        for _, r in s.iterrows():
            print(f"  {r.model:<32} {r.point:>7.4f}   [{r.lo:>7.4f}, {r.hi:>7.4f}]"
                  f"   width {r.hi - r.lo:.4f}")
else:
    marginal = None
    print("RUN_BOOTSTRAP = False -- skipped")
''')

# ── §7 contrasts ─────────────────────────────────────────────────────────────
md("""
## §7 — THE ANSWER: the pre-registered paired contrasts

Paired — the same resampled subject list is applied to both models on every draw,
because both scored the same images. In a stratum of 74 images, discarding the
pairing would hide every effect there is.

**`survives_holm` is the only column a confirmatory claim may be made from.**
This family is 4 confirmatory pairs × 3 endpoints = **12 cells**; testing twelve
things at 5% and reporting whichever cleared is how a study manufactures
findings. Holm–Bonferroni is applied **within the confirmatory set only** — the
same policy §8b uses. The two exploratory pairs stay uncorrected and labelled,
because folding them in would launder post-hoc comparisons into confirmatory ones.

`clears_zero` is the **uncorrected** interval and is kept beside it, so the
difference between the two is visible rather than quietly resolved.
`min_detectable` is the CI half-width: **any true difference smaller than that
cannot clear zero at this sample size, no matter what it is.**
""")

code('''
if RUN_CONTRASTS:
    contrasts = LS.contrast_table(tables, key, PRIMARY, LS.PRIMARY_ENDPOINTS,
                                  None, N_BOOT, SEED)
    for ep in LS.PRIMARY_ENDPOINTS:
        s = contrasts[contrasts.endpoint == ep]
        if s.empty:
            continue
        print(f"\\n{ep}   (stratum {LS.PRIMARY_STRATUM}, "
              f"lower is better = {bool(s.lower_is_better.iloc[0])})")
        print("=" * 124)
        for _, r in s.iterrows():
            if r.degenerate:
                flag = "DEGENERATE -- identical on every draw, no evidence"
            elif r.kind == "confirmatory":
                flag = ("**SURVIVES HOLM**" if r.survives_holm is True
                        else "clears zero but NOT after Holm" if r.clears_zero
                        else "spans zero")
            else:
                flag = ("clears zero (EXPLORATORY, uncorrected)"
                        if r.clears_zero else "spans zero")
            print(f"  [{r.kind:<13}] {r.a} - {r.b}")
            print(f"      delta {r.delta:>+8.4f}   CI [{r.lo:>+8.4f}, {r.hi:>+8.4f}]"
                  f"   p {r.p_two_sided:.4f}"
                  + (f"   p_holm {r.p_holm:.4f}" if r.kind == "confirmatory" else "")
                  + f"   {flag}")
            mde = ("n/a (degenerate)" if r.degenerate else f"{r.min_detectable:.4f}")
            print(f"      minimum detectable effect at this n: {mde}")

    n_clear = int(contrasts.clears_zero.sum())
    n_holm = int((contrasts.survives_holm == True).sum())  # noqa: E712 -- nullable
    print("\\n" + "=" * 124)
    print(f"VERDICT: {n_clear} of {len(contrasts)} cells clear zero UNCORRECTED.")
    print(f"         {n_holm} confirmatory cells survive Holm -- "
          "these are the only ones a claim may be made from.")
    if n_holm == 0 and n_clear:
        print("  -> Nothing survives correction. The uncorrected wins are what you")
        print("     expect from testing 12 cells at 5%. Report §8, not §7.")
    if n_clear == 0:
        print("  -> The small-lesion endpoint does NOT separate these models at n=28")
        print("     subjects. Read Sec 8: the deliverable is the minimum detectable")
        print("     effect, and the next move is MORE DATA, not more mechanism.")
    else:
        print("  -> Small-lesion miss containment separates models that tie on mean")
        print("     Dice. Report it, and it names which lever to pull next.")
    print("=" * 124)
else:
    contrasts = None
    print("RUN_CONTRASTS = False -- skipped")
''')

md("""
### §7b — The same contrasts on D1 alone

**19 images.** Fragile by construction and included only so the fragility is
visible rather than assumed. Never quote these as a primary claim.
""")

code('''
if RUN_CONTRASTS and RUN_D1_ONLY:
    d1 = LS.contrast_table(tables, key, SMALLEST, LS.PRIMARY_ENDPOINTS,
                           None, N_BOOT, SEED, verbose=False)
    if not d1.empty:
        print(f"D1 only -- {int(SMALLEST.sum())} images, "
              f"{key[SMALLEST].subject.nunique()} subjects")
        print("=" * 124)
        print(d1[["a", "b", "endpoint", "kind", "delta", "lo", "hi",
                  "clears_zero", "min_detectable"]].to_string(index=False))
        print(f"\\nCI widths here are ~{(d1.hi - d1.lo).median():.3f} on average. "
              "Compare with Sec 7 before reading anything into these.")
else:
    d1 = None
    print("skipped")
''')

# ── §8 power ─────────────────────────────────────────────────────────────────
md("""
## §8 — THE POWER ANSWER

If §7 found nothing, **this is the deliverable.** It converts *"we found no
difference"* into *"a difference smaller than X is invisible at 28 subjects"* —
a statement about the experiment rather than about the models, and the one that
tells you how much more data would be needed.
""")

code('''
if RUN_POWER:
    power = LS.min_detectable(tables, key, PRIMARY, LS.PRIMARY_ENDPOINTS,
                              N_BOOT, SEED)
    print("MINIMUM DETECTABLE EFFECT -- primary stratum "
          f"{LS.PRIMARY_STRATUM}, {int(PRIMARY.sum())} images, "
          f"{key[PRIMARY].subject.nunique()} subjects")
    print("=" * 104)
    print(power.to_string(index=False))
    print("""
HOW TO READ THIS
  median_min_detectable  the typical smallest effect a contrast could resolve.
  worst_min_detectable   the widest -- the hardest comparison in the family.
  any_clears_zero        whether ANY contrast reached significance at this endpoint.

For zero_dice_rate, multiply by the stratum size to read it as images: an effect
of 0.05 over 74 images is ~4 images. If the minimum detectable effect is larger
than the difference you care about clinically, this test set cannot answer the
question and no amount of re-analysis will change that.""")
else:
    power = None
    print("RUN_POWER = False -- skipped")
''')

# ── §8b fairness ─────────────────────────────────────────────────────────────
md("""
## §8b — Fairness, conditioned on lesion size

§8.4 has stated the problem since Stage D and **nothing in the study had done the
arithmetic**: bruise size is the strongest single predictor of whether a model
finds a bruise at all, *and* size is not evenly distributed across ITA groups. So
every unconditioned per-group number is measuring both at once, and cannot tell
*"worse on darker skin"* from *"this group happens to have smaller bruises"*.

Stage P already cut the size deciles, so conditioning costs nothing.

**Read the two outputs in order.** The first says whether the confound is even
present in this test set. The second is only interpretable given the first.

Cells with fewer than 5 subjects get **no interval** — with 28 subjects over five
groups and then split by size, several cells fall to 1–3 subjects, and a cluster
bootstrap over two clusters is arithmetic, not evidence.
""")

code('''
if RUN_FAIRNESS:
    conf = LS.size_by_ita(tables, key)

    fair = LS.fairness_conditioned(tables, key, FAIRNESS_MODELS,
                                   FAIRNESS_ENDPOINT, N_BOOT, SEED)
    print("""
HOW TO READ THE GAPS ABOVE
  gap SHRINKS from 'all' to 'small'  -> the marginal fairness gap was partly a
                                        SIZE effect wearing a skin-tone label.
                                        That is what §8.4 warned about.
  gap HOLDS                          -> the gap is not explained by size and is
                                        a real per-group difference worth
                                        reporting -- subject to the n above.

The gap is max-minus-min over five noisy estimates, so it is biased UPWARD and is
DESCRIPTIVE. Do not read a gap as a test.""")
else:
    conf = fair = None
    print("RUN_FAIRNESS = False -- skipped")
''')

# ── §9 write ─────────────────────────────────────────────────────────────────
md("""
## §9 — Write results

Everything goes to `LESION_SIZE_RESULTS/` at the bundle root — never `results/`,
`FINAL_RESULT/` or `_work/runs/`. Same isolation as Stages M and N, so an
analysis that turns out to be wrong leaves no trace in the directories the
published numbers come from.

| file | what it is |
|---|---|
| `lesion_size_headline.csv` | one row per model: whole set, D1–D4, D1 |
| `lesion_size_by_decile.csv` | one row per (model, decile) — the full breakdown |
| `lesion_size_marginal_ci.csv` | per-model CIs in the primary stratum |
| `lesion_size_contrasts.csv` | **the answer** — pre-registered paired contrasts |
| `lesion_size_contrasts_d1.csv` | the same on D1 alone (fragile, 19 images) |
| `lesion_size_power.csv` | minimum detectable effect per endpoint |
| `lesion_size_bins.csv` | the decile assignment, so the cut is reproducible |
| `lesion_size_ita_confound.csv` | is size confounded with ITA group? (§8.4) |
| `lesion_size_fairness_conditioned.csv` | per-group performance, marginal vs size-conditioned |
""")

code('''
if WRITE_RESULTS:
    print(f"writing to {env.root / LS.RESULTS_DIRNAME}/")
    LS.save(env, "lesion_size_bins", key)
    if head is not None:
        LS.save(env, "lesion_size_headline", head)
        LS.save(env, "lesion_size_by_decile", long)
    if marginal is not None:
        LS.save(env, "lesion_size_marginal_ci", marginal)
    if contrasts is not None:
        LS.save(env, "lesion_size_contrasts", contrasts)
    if d1 is not None and not d1.empty:
        LS.save(env, "lesion_size_contrasts_d1", d1)
    if power is not None:
        LS.save(env, "lesion_size_power", power)
    if conf is not None:
        LS.save(env, "lesion_size_ita_confound", conf)
        LS.save(env, "lesion_size_fairness_conditioned", fair)
    print("\\ndone.")
else:
    print("WRITE_RESULTS = False -- nothing written")
''')

# ── §10 ──────────────────────────────────────────────────────────────────────
md("""
## §10 — What to paste back, and what each outcome means

Paste **§7's verdict block** and **§8's table**. Those two are the whole result.

### If §7 clears zero on `zero_dice_rate`

Models that are statistically indistinguishable on mean Dice **separate on
small-lesion miss containment.** That is a genuine finding and it is the headline
this study has been missing — mean Dice is at the annotation ceiling, this is not.
It also names the lever, and the queued GPU work becomes justified.

### If §7 spans zero everywhere

The premise behind the whole queue is not established at this sample size. Then:

- **Do not** run Stage N's seg arms, ALS→WL distillation, or a Fenwick merge
  expecting to *measure* a small-lesion improvement — the instrument cannot see it.
- **Do** report §8 as a power result. "An effect below X images is invisible at 28
  subjects" is publishable and is the honest case for collecting more data.
- The Fenwick merge moves to the **top** of the queue, because its value is then
  sample size rather than label quality.

### Either way, one thing is already true

`wrong_place` (§5) is a failure mode the per-seed tables cannot see: a model that
outputs a substantial region entirely in the wrong place scores 0 Dice while
predicting thousands of pixels. On the current lineage `fastscnn_rgkd` has 6
zero-Dice images and 1 empty prediction. That distinction does not need a
bootstrap to be worth reporting.
""")


def main() -> int:
    nb = {
        "cells": [
            {"cell_type": t, "metadata": {},
             **({"source": s.splitlines(keepends=True)} if t == "markdown"
                else {"source": s.splitlines(keepends=True),
                      "execution_count": None, "outputs": []})}
            for t, s in CELLS
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text(json.dumps(nb, indent=1), encoding="utf-8")

    n_code = sum(1 for t, _ in CELLS if t == "code")
    for i, (t, s) in enumerate(CELLS):
        if t == "code":
            compile(s, f"cell{i}", "exec")
    print(f"wrote {DST}")
    print(f"  {len(CELLS)} cells, {n_code} code cells, all compile")
    return 0


if __name__ == "__main__":
    sys.exit(main())
