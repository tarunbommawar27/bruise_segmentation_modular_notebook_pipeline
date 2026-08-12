#!/usr/bin/env python
"""Emit `bruise_all_models.ipynb` -- one table for every model ever scored.

Same generator discipline as 70/71/77/78/79: the notebook is an OUTPUT, never
hand edited, and ships with zero executed cells.

Everything goes to `ALL_MODELS_RESULTS/`. Nothing is written to `results/`,
`FINAL_RESULT/`, `LESION_SIZE_RESULTS/` or any stage's own directory -- every
per-image CSV is READ and none is rewritten.
"""
from __future__ import annotations

import json
from pathlib import Path

DST = Path(__file__).resolve().parent.parent / "BRUISE_UNIFIED" / "bruise_all_models.ipynb"

CELLS: list[tuple[str, str]] = []


def md(src: str) -> None:
    CELLS.append(("markdown", src.strip("\n")))


def code(src: str) -> None:
    CELLS.append(("code", src.strip("\n")))


# ─────────────────────────────────────────────────────────────────────────────
md("""
# Every model, every endpoint, one table

`LESION_SIZE_RESULTS/` covers 18 arms. `STAGE_N4_RESULTS/` covers four more. The
**Stage C distillation grid**, the **Stage H reliability-gated arms** and the
**Stage M multi-teacher arms** have per-image CSVs on disk and have never been
through a size or miss analysis at all.

That matters. Those families were called **null on Dice** — and Dice is the
endpoint this study has argued all along is saturated (Friedman p = 0.61 across
the seven headline models). *"We looked and it was null"* and *"we did not look"*
are different claims, and right now only one of them is true.

This notebook computes, for **every** model with a per-image table anywhere:

| | |
|---|---|
| **mean / median Dice** | with subject-clustered CI and IQR |
| **zero-Dice count and rate** | the published complete-miss definition |
| **wrong-place misses** | zero Dice *while predicting something* — the per-seed tables cannot see these |
| **small-lesion recall** | `D1–D4` (four smallest GT-area deciles) and `D1` alone |
| **fairness** | per-ITA-group Dice, recall and miss rate; gap; Kruskal–Wallis |
| **size-conditioned fairness** | does the gap shrink inside the small stratum? |

## The one thing to understand before reading the output

The input is deliberately heterogeneous — CSVs from several lineages, splits and
mask versions. `lesionsize.assign_bins` *raises* when two tables disagree about
an image's GT area, which is right for its job but would kill this sweep on the
first stale file.

So this module **detects cohorts** instead: tables are grouped by (stem set,
GT-area vector), the largest coherent group becomes the reference cohort and is
binned once, and everything else is binned separately and flagged
`comparable = False`.

**Only rows with `comparable = True` may be compared against each other.** The
rest are kept, labelled, and quarantined — not silently mixed and not silently
dropped.

## Writes only to `ALL_MODELS_RESULTS/`
""")

code('''
import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from pathlib import Path

BUNDLE = None      # None = auto-detect
WORK   = None      # None = <bundle>/_work

# Extra trees to scan. ON ORC THE RUNS DO NOT LIVE IN THE BUNDLE -- outputs land
# under the work directory on scratch, and FINAL_RESULT/ may never have been
# synced there at all. Add every root that has ever held a per-image CSV; the
# discovery log will tell you which ones actually contained anything.
EXTRA_ROOTS = [
    "/scratch/tbommawa/bruise_work",
    "/scratch/tbommawa/bruise_work/outputs",
    "/scratch/tbommawa/BRUISE_UNIFIED",
]

N_BOOT = 10000     # subject-clustered resamples. 2000 is plenty for a first look.

print(f"extra roots: {EXTRA_ROOTS}")
''')

# ── §1 ───────────────────────────────────────────────────────────────────────
md("""
## §1 — Environment and self-test

The self-test is structural: no disk, no manifests, no network. It checks the
three things that would silently corrupt the table — that filename cleaning does
not merge two different models, that the cohort signature separates different
mask versions, and that `wrong_place` really is `zero_dice` minus `empty_pred`.
""")

code('''
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
pd.set_option("display.width", 260)
pd.set_option("display.max_columns", 120)
pd.set_option("display.max_rows", 200)

from bruisekit import allmodels as AM
from bruisekit import paths as P

env = P.setup(root=BUNDLE, work=WORK)
print(env.describe())

RESULTS = AM.results_dir(env)
print(f"\\nresults : {RESULTS}")
print("          every per-image CSV is READ; none is rewritten")

print("\\n-- self test --")
assert AM.self_test(), "allmodels self-test failed"
''')

# ── §2 ───────────────────────────────────────────────────────────────────────
md("""
## §2 — Discovery: what is on this machine, and where

Run this first and read it. If a family you expect is missing, the fix is almost
always another entry in `EXTRA_ROOTS` — not a code change.
""")

code('''
roots = AM.search_roots(env, EXTRA_ROOTS)
print(f"{len(roots)} root(s) exist and will be scanned:\\n")
for r in roots:
    n = sum(len(list(r.glob(g))) for g in AM.GLOBS)
    if n:
        print(f"  {n:>4} file(s)   {r}")

found = AM.discover(env, EXTRA_ROOTS, verbose=True)
''')

# ── §3 ───────────────────────────────────────────────────────────────────────
md("""
## §3 — Load, cohort, summarise

`run()` does discovery, loading, alias collapse, cohort detection, binning and
the statistics in one call. Aliases are collapsed by comparing the **Dice
vector**, not the filename: byte-identical scores are the same run whatever the
file is called, and the shortest name wins.
""")

code('''
out = AM.run(env, extra_roots=EXTRA_ROOTS, n_boot=N_BOOT, seed=0, verbose=True)

summary   = out["summary"]
by_decile = out["by_decile"]
by_group  = out["by_group"]

print(f"\\n{len(summary)} model(s), {int(summary.comparable.sum())} comparable")
''')

# ── §4 ───────────────────────────────────────────────────────────────────────
md("""
## §4 — The table

Sorted by **complete misses first**, then Dice. That ordering is deliberate:
sorting by Dice would put the table in the order of the endpoint that does not
discriminate.
""")

code('''
AM.print_summary(out, top=80)
''')

# ── §5 ───────────────────────────────────────────────────────────────────────
md("""
## §5 — The three views worth reading separately

**Small-lesion recall** is the endpoint a clinician cares about and the one the
headline Dice table hides. **`wrong_place`** is the failure where a model outputs
pixels and every one is wrong — not the same thing as predicting nothing.
""")

code('''
comp = summary[summary.comparable].copy()

print("=" * 110)
print("SMALL LESIONS -- D1-D4, the four smallest GT-area deciles")
print("=" * 110)
cols = ["model", "D1_D4_n", "D1_D4_zero_dice_n", "D1_D4_mean_recall",
        "D1_D4_median_dice", "D1_mean_recall", "D1_zero_dice_n"]
print(comp.sort_values("D1_D4_mean_recall", ascending=False)
          [[c for c in cols if c in comp.columns]].to_string(index=False))

print("\\n" + "=" * 110)
print("MISS CONTAINMENT -- and how many misses were WRONG-PLACE, not empty")
print("=" * 110)
cols = ["model", "all_zero_dice_n", "all_zero_dice_rate",
        "zero_dice_rate_ci_lo", "zero_dice_rate_ci_hi",
        "all_empty_pred_n", "all_wrong_place_n"]
print(comp.sort_values("all_zero_dice_n")
          [[c for c in cols if c in comp.columns]].to_string(index=False))
''')

code('''
print("=" * 118)
print("FAIRNESS -- descriptive gap, inferential test, and whether size explains it")
print("=" * 118)
cols = ["model", "fairness_gap_median_dice", "fairness_best_group",
        "fairness_worst_group", "fairness_miss_rate_gap", "kruskal_p",
        "kruskal_significant", "recall_gap_all", "recall_gap_small",
        "gap_shrinks_with_size"]
f = comp[[c for c in cols if c in comp.columns]].sort_values("kruskal_p")
print(f.to_string(index=False))

print("\\n  THE GAP IS DESCRIPTIVE, THE TEST IS INFERENTIAL, AND THEY ROUTINELY")
print("  DISAGREE HERE. With 28 subjects over five groups, a visually large gap")
print("  usually is not significant. The gap is a max-minus-min over five noisy")
print("  estimates and is biased UPWARD; do not read it as a test.")
print()
print("  MULTIPLICITY: this is one Kruskal test per model. Testing ~40 models at")
print("  alpha = 0.05 expects ~2 to clear from noise alone. Apply Holm across the")
print("  models before claiming any single one, and say which family you corrected")
print("  within -- handbook 8b applies the same policy for the same reason.")
print()
print("  gap_shrinks_with_size = True means the marginal gap was partly a lesion-")
print("  size effect wearing a skin-tone label (handbook 8.4).")

if "kruskal_p" in comp.columns:
    from bruisekit.significance import holm
    sig = comp.dropna(subset=["kruskal_p"]).copy()
    sig["kruskal_p_holm"] = holm(sig.kruskal_p.values)
    n_raw = int((sig.kruskal_p < 0.05).sum())
    n_holm = int((sig.kruskal_p_holm < 0.05).sum())
    print(f"\\n  {n_raw} model(s) significant uncorrected; "
          f"{n_holm} survive(s) Holm across {len(sig)} models.")
    if n_holm:
        print(sig[sig.kruskal_p_holm < 0.05]
              [["model", "kruskal_p", "kruskal_p_holm", "fairness_worst_group"]]
              .to_string(index=False))
''')

# ── §6 ───────────────────────────────────────────────────────────────────────
md("""
## §6 — Save

`ALL_MODELS_SUMMARY.csv` is the single table — one row per model, every endpoint.
The long-form tables are there for anything the wide one cannot hold, and
`discovery_log.csv` records every file found, what it became, and every file
skipped **with the reason**. Nothing is silently dropped.
""")

code('''
for p in AM.save(env, out):
    print(f"  written -> {p}")

print(f"\\nThe single table is {RESULTS / 'ALL_MODELS_SUMMARY.csv'}")
print(f"  {len(summary)} rows x {summary.shape[1]} columns")
print("\\nTo bring it back:")
print(f"    zip -r all_models.zip {RESULTS.name}")
''')

# ── §7 ───────────────────────────────────────────────────────────────────────
md("""
## §7 — What was skipped, and why

Read this before concluding a model is missing. The usual causes, in order of
frequency:

1. **stems match no manifest split** — the table is from a different test set
   (the Fenwick matched core is 127 images, not 185). Correctly excluded: it
   cannot share a decile cut with the main lineage.
2. **collapsed as an alias** — an identical Dice vector to another file. The
   model is present under its shorter name.
3. **not a reporting path** — the retired YOLO `custom255` preprocessing, kept in
   the bundle for provenance but never a reporting path.
""")

code('''
log = out["discovery"]
print(log.status.value_counts().to_string())

for st in ("skipped", "alias"):
    sub = log[log.status == st]
    if not len(sub):
        continue
    print(f"\\n--- {st} ({len(sub)}) ---")
    cols = ["model", "reason"] if st == "skipped" else ["model", "alias_of"]
    print(sub[cols].drop_duplicates().head(40).to_string(index=False))
''')

# ── §8 ───────────────────────────────────────────────────────────────────────
md("""
## §8 — How to read this, and what it does not license

**Read the miss columns before the Dice column.** 23× the parameters bought
+0.006 Dice on this task and the headline omnibus does not reject (p = 0.61).
Every remaining signal in this project lives in complete misses, small-lesion
recall, and fairness.

**A Dice tie with a miss-rate difference is a result**, not an absence of one.

### What this cannot tell you

- **These are not paired tests.** The table is descriptive: one row per model,
  with marginal CIs. A difference between two rows whose CIs overlap is *not*
  therefore null — a paired subject-clustered bootstrap on the same images is
  far more sensitive than comparing two marginal intervals. Use
  `lesionsize.paired_stratum` for any claim about a specific pair.
- **Nothing here is multiplicity-corrected except where §5 says so.** ~40 models
  × several endpoints is a large family. Pick the contrast first, then test it.
- **Every number is at whatever operating point its source run fitted.** This
  notebook re-fits nothing. A model whose CSV was written at a badly-chosen cut
  looks bad here, correctly — but that is a property of the run, not of the model.
- **`comparable = False` rows may not be compared to the others.** Different
  split or different mask version. They are kept so you can see they exist.
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
