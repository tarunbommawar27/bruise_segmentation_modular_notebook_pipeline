#!/usr/bin/env python
"""Emit `bruise_inference_all.ipynb` -- the shipped inference pass, over every model.

THIS NOTEBOOK ADDS NO INFERENCE CODE. It calls `bruisekit.inference.run`, which is
the same function §18.1 of the handbook documents and the same one
`bruise_unified.ipynb`'s inference block uses. The only thing that changes is the
`models` argument: the shipped default is three families, and here it is every
family the registry can resolve on this machine.

    inference.inference_pass  ->  loaders.score_run  ->  score_segformer
                                                     ->  score_yolo_native
    inference.reconcile       ->  fresh vs the shipped per-image table

`inference.inference_pass`'s own docstring says it: "There is deliberately no
second inference implementation in this file." A Stage-R module with its own
loader and its own cut resolution was written and DELETED for exactly that reason
-- it would have been a second path that could drift from the published one, and
a re-inference whose only job is to confirm the published numbers must not be
computed differently from them.

WHAT THIS NOTEBOOK IS FOR
--------------------------
On 2026-08-12 the laptop and ORC reported different miss counts for `fastscnn`,
`fastscnn_distilled` and `ppmobileseg_tiny` -- the same model names over different
recorded runs, with nothing in either table saying so. Re-scoring from checkpoints
and reconciling against the shipped tables is how that is settled.

`inference.reconcile` already produces exactly that comparison, including
`max_abs_dice_delta`, which is the column that matters: a mean can agree while
individual images disagree wildly, and that is the signature of a seed mismatch
rather than of float noise.

RUN IT ON A GPU
----------------
On CUDA this is seconds per model. On CPU it is 1-2 minutes per model, because
SegFormer-B2's decode head is doing 185 full-resolution forwards without an
accelerator -- that is the whole reason the first attempt felt slow. §1 asserts on
the device rather than letting a CPU kernel quietly start a two-hour job.
"""
from __future__ import annotations

import json
from pathlib import Path

DST = (Path(__file__).resolve().parent.parent / "BRUISE_UNIFIED"
       / "bruise_inference_all.ipynb")

CELLS: list[tuple[str, str]] = []


def md(src: str) -> None:
    CELLS.append(("markdown", src.strip("\n")))


def code(src: str) -> None:
    CELLS.append(("code", src.strip("\n")))


# ─────────────────────────────────────────────────────────────────────────────
md("""
# Inference over every model — the shipped pass, wider model list

**This notebook adds no inference code.** It calls `bruisekit.inference.run`, the
same function the unified notebook's inference block uses and the one §18.1 of the
handbook documents. The only thing that changes is the `models` argument.

```
inference.inference_pass  ->  loaders.score_run  ->  score_segformer
                                                 ->  score_yolo_native
inference.reconcile       ->  fresh vs the shipped per-image table
```

`inference_pass`'s docstring says why there is nothing else here: *"There is
deliberately no second inference implementation in this file."* A re-inference
whose whole job is to confirm the published numbers must not be computed by a
different path than produced them.

## Why you want this

On **2026-08-12** the laptop and ORC reported different miss counts for the same
three arms — `fastscnn` (13 vs 7), `fastscnn_distilled` (8 vs 13),
`ppmobileseg_tiny` (4 vs 7). Same model names, different recorded runs, and
nothing in either table saying so.

Re-scoring from checkpoints and reconciling against the shipped tables settles it.
`inference.reconcile` gives you `max_abs_dice_delta` per model — the column that
matters, because a **mean can agree while individual images disagree wildly**, and
that is the signature of a seed mismatch rather than float noise.

## Two things that are already handled and you should not re-do

**The best seed is not the same for every model.** `resolve_runs` reads it from
the selection step's own filenames: 0 for the three SegFormers and
`yolo_sem_distilled`, **2** for `yolo_sem_direct`. Scoring a model at another
model's best seed shows per-image disagreements up to 0.49 Dice and looks exactly
like a broken inference path.

**Each model is scored at its own val-fitted cut**, read from
`operating_point.json` / `threshold.json` by `registry.read_cut`. Nothing here
re-fits an operating point, and YOLO is scored by native argmax, which has no
threshold to fit.

## Run this on a GPU

Seconds per model on CUDA; **1–2 minutes per model on CPU**. §2 warns you rather
than letting a CPU kernel quietly start a two-hour job.
""")

code('''
import os
import sys

assert "torch" not in sys.modules, (
    "torch is already imported -- PYTORCH_CUDA_ALLOC_CONF is read when the CUDA "
    "allocator initialises and setting it now has NO EFFECT. Restart the kernel "
    "and run this cell first.")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
print("allocator configured; torch not yet imported")
''')

# ── §1 ───────────────────────────────────────────────────────────────────────
md("""
## §1 — Configuration

`EXTRA_RUNS` is the setting that decides how much this can see. Without it the
registry finds only the shipped checkpoints, and the laptop bundle has **no
`checkpoints/efficient` and no `checkpoints/rgkd`** — the entire mobile family,
which is the family that disagreed. Handbook §10.3.
""")

code('''
BUNDLE     = None      # None = auto-detect
WORK       = None      # None = <bundle>/_work

# THE SETTING THAT MATTERS. On ORC this is where the mobile family lives.
EXTRA_RUNS = "/scratch/tbommawa/bruise_work/runs"

# Every family worth re-scoring. inference.DEFAULT_MODELS is only three; this is
# the full reporting set. Families with no WEIGHTS-tier run on this machine are
# REPORTED and skipped by resolve_runs, never back-filled from another seed.
MODELS = (
    "segformer_b5_teacher", "segformer_b2_teacher",
    "segformer_b0_direct", "segformer_b0_distilled", "segformer_b0_rgkd",
    "unet_r50", "deeplabv3plus_r50",
    "yolo_sem_direct", "yolo_sem_distilled",
    "lraspp_mobilenetv3", "lraspp_mobilenetv3_b2kd",
    "lraspp_mobilenetv3_distilled", "lraspp_mobilenetv3_rgkd",
    "topformer_tiny", "topformer_tiny_b2kd",
    "topformer_tiny_distilled", "topformer_tiny_rgkd",
    "ppmobileseg_tiny", "ppmobileseg_tiny_b2kd",
    "ppmobileseg_tiny_distilled", "ppmobileseg_tiny_rgkd",
    "fastscnn", "fastscnn_b2kd", "fastscnn_distilled", "fastscnn_rgkd",
)

SEED     = None        # None = each model's val-selected best seed. DO NOT pin
                       # this to 0 without reading resolve_runs' docstring.
DO_SPEED = False       # the speed table is a separate, machine-specific claim
                       # (handbook 18.5) -- leave it off unless that is the job
OUT_DIR  = None        # None = <env.out>/inference
''')

# ── §2 ───────────────────────────────────────────────────────────────────────
md("""
## §2 — Environment

The device check is a warning, not a block — CPU scoring is numerically identical,
just slow. If you are on CPU, cut `MODELS` down to the handful you actually need
to settle.
""")

code('''
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 80)
pd.set_option("display.max_rows", 200)

from bruisekit import inference as INF
from bruisekit import loaders as L
from bruisekit import paths as P
from bruisekit.registry import Registry

env = P.setup(root=BUNDLE, work=WORK, extra_runs=EXTRA_RUNS)
print(env.describe())

on_gpu = str(env.device).startswith("cuda")
if not on_gpu:
    print(f"\\n  !! device is {env.device}. Expect ~1-2 MINUTES PER MODEL, so "
          f"{len(MODELS)} models is\\n     roughly {len(MODELS) * 1.5:.0f} minutes. "
          f"Numerically identical to GPU -- just slow.\\n"
          f"     Cut MODELS down, or run this on the box with the GPU.")
else:
    print(f"\\n  seconds per model on {env.device}")
''')

# ── §3 ───────────────────────────────────────────────────────────────────────
md("""
## §3 — Manifests and the 640 cache

Models are scored on **exactly the tensors they were trained on**, not a fresh
resize, so a re-scored number stays comparable with the shipped one. YOLO is the
exception and is scored from native-resolution images, because Ultralytics
letterboxes internally and feeding it pre-resized images would apply the resize
twice.
""")

code('''
MAN = {s: pd.read_csv(env.manifests / f"{s}.csv") for s in ("train", "val", "test")}
for s, d in MAN.items():
    print(f"  {s:<6} {len(d):>4} images  {d.subject.nunique():>3} subjects")

for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
    overlap = set(MAN[a].subject) & set(MAN[b].subject)
    assert not overlap, f"{a}/{b} share subjects: {sorted(overlap)[:5]}"
print("  no subject appears in two splits")

MAN640 = L.build_cache640(env, MAN)
CFG = dict(INF.DEFAULT_CFG)
CFG["eval_batch"] = 8 if on_gpu else 2
print(f"\\ncfg: {CFG}")
''')

# ── §4 ───────────────────────────────────────────────────────────────────────
md("""
## §4 — Which runs this machine can actually resolve

`resolve_runs` maps each family to the `Run` this session should use, at that
model's **val-selected best seed**. A family with no WEIGHTS-tier run is printed
and skipped — never back-filled from a neighbouring seed.

Read this before §5. If the mobile family is missing here, `EXTRA_RUNS` is not
pointing anywhere useful and the sweep will not answer the question you ran it
for.
""")

code('''
REG = Registry(env).scan()
RUNS = INF.resolve_runs(env, REG, MODELS, SEED)

print(f"\\n{len(RUNS)}/{len(MODELS)} families resolved")
print(pd.DataFrame([{"model": f, "run_id": r.run_id, "kind": r.kind,
                     "seed": r.seed, "source_root": str(getattr(r, "source_root", "")),
                     "layout": getattr(r, "layout", "")}
                    for f, r in RUNS.items()]).to_string(index=False))
''')

# ── §5 ───────────────────────────────────────────────────────────────────────
md("""
## §5 — The inference pass

One forward over the 185 test images per model, at its own val-fitted cut. This is
`inference.run` with `do_speed` off — the same call the unified notebook makes.

Per-image CSVs land in `<out>/per_image_<family>.csv`, the headline in
`inference_headline.csv`, and the comparison in `inference_reconciliation.csv`.
""")

code('''
t0 = time.time()
OUT = INF.run(env, REG, CFG, MAN, MAN640, models=tuple(RUNS),
              do_inference=True, do_reconcile=True, do_speed=DO_SPEED,
              seed=SEED, out_dir=OUT_DIR)
print(f"\\ntotal {time.time() - t0:.0f}s")

TABLES = OUT.get("per_image", {})
''')

# ── §6 ───────────────────────────────────────────────────────────────────────
md("""
## §6 — Fresh vs shipped

The column that matters is **`max_abs_dice_delta`**, not `mean_dice_delta`. A mean
can agree while individual images disagree wildly, and that is the signature of a
**seed mismatch** rather than of float noise.

| what you see | reading |
|---|---|
| `max_abs_dice_delta` ≲ 2e-4 | float ordering. The shipped number is confirmed. |
| large `max` with `mean_delta` ≈ 0 | different seed or different run under the same name |
| `miss_delta` ≠ 0 | the complete-miss count changed — this is the 2026-08-12 case |
| `no shipped table` | nothing published under that name on this machine |

The handbook's claim is that every reporting model agrees to better than 2e-4 mean
Dice against the original A100 run. This cell is how that gets **checked** rather
than repeated.
""")

code('''
REC = OUT.get("reconcile")
if REC is None or REC.empty:
    print("no reconciliation -- nothing was scored")
else:
    cols = ["model", "run_id", "status", "n", "mean_dice_fresh",
            "mean_dice_shipped", "mean_dice_delta", "max_abs_dice_delta",
            "miss_delta"]
    print(REC[[c for c in cols if c in REC.columns]].to_string(index=False))

    bad = REC[(REC.max_abs_dice_delta.fillna(0) > 2e-4) | (REC.miss_delta.fillna(0) != 0)]
    print(f"\\n{len(REC) - len(bad)}/{len(REC)} model(s) reproduce to better than 2e-4.")
    if len(bad):
        print(f"\\n  {len(bad)} DISAGREE -- these are the ones to investigate:")
        print(bad[["model", "run_id", "max_abs_dice_delta", "miss_delta"]]
              .to_string(index=False))
        print("\\n  Check in this order: (1) is the shipped table from a different\\n"
              "  run of the same name (compare source_root in 4)? (2) did the seed\\n"
              "  resolve as expected? (3) only then suspect the checkpoint.")
''')

# ── §7 ───────────────────────────────────────────────────────────────────────
md("""
## §7 — The miss taxonomy on the fresh tables

The output directory uses the project's `per_image_<model>.csv` naming, so it is a
drop-in lineage. Running Stage O's taxonomy over it gives zero Dice split into
empty prediction and wrong place, computed from tables that all came from one
pass.
""")

code('''
from bruisekit import itakd

TAX = itakd.miss_taxonomy(TABLES) if TABLES else None
if TAX is not None:
    itakd.print_miss_taxonomy(TAX)
    dst = (OUT_DIR or (env.out / "inference")) / "miss_taxonomy_reinferred.csv"
    TAX.to_csv(dst, index=False)
    print(f"\\n  wrote {dst}")
''')

# ── §8 ───────────────────────────────────────────────────────────────────────
md("""
## §8 — What this licenses

**Licensed.** Any model whose `max_abs_dice_delta` is under 2e-4 has a published
number confirmed against a fresh forward pass from its checkpoint, through the
same code path that produced it.

**Licensed.** A disagreement is now located rather than argued about: the table
names the model, the run_id and the size of the difference.

**Not licensed — a disagreement is not automatically a bad published number.**
Check the seed and the `source_root` first. The 2026-08-12 case was two machines
holding different runs under the same name; the tables were each internally
correct.

**Not licensed — this is not a re-selection.** Every model is scored at the cut
its own run fitted on validation, and `SEED = None` uses the val-selected best
seed. Nothing here re-picks seeds or re-fits operating points, and nothing may be
fitted on test.

**Not licensed — coverage is not completeness.** §4 tells you what resolved on
*this* machine. The Stage C `distill_out/` arms (`p3_adaptive`, `p2_cwd_b5_to_b0`,
…) are registered as `arm::<name>` with no seed, so `resolve_runs`' `family__seedN`
lookup does not reach them; they need their own resolution and are not in `MODELS`.
A complete 35-model reconciliation also needs every checkpoint and the published
lineage on one machine — a sync job, not a code change.
""")


def main() -> int:
    nb = {
        "cells": [
            {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
            | ({"outputs": [], "execution_count": None} if kind == "code" else {})
            for kind, src in CELLS
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    DST.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    n_code = sum(1 for k, _ in CELLS if k == "code")
    print(f"wrote {DST}  ({len(CELLS)} cells, {n_code} code)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
