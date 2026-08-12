#!/usr/bin/env python
"""Emit `bruise_speed_harness.ipynb` -- the fp32/fp16 speed comparison, as cells.

WHY A NOTEBOOK AND NOT JUST THE CLI
-------------------------------------
Every run in this study happens in Jupyter, on ORC or on Colab. A `python -m`
invocation is the wrong shape for that: it cannot be resumed cell by cell, its
output scrolls past instead of staying next to the code, and when a model name is
wrong you find out after the benchmark instead of before it. Shipping a library
change without the cells that drive it makes the recipient rebuild the driver by
hand every time, which is exactly the repetition this file removes.

Same convention as 61_generate_unified_notebook.py: raw nbformat 4.5 JSON, one
`id` per cell, no nbformat dependency.

WHAT THE NOTEBOOK DOES
-----------------------
One question, answered by measurement: how much of the 1.89x gap between
`benchmark_speed` (16.55 ms for segformer_b0) and the 51-run sweep (8.74 ms) is
PRECISION rather than hardware? Both numbers came off a Colab A100 on the same
day, so no machine explains them.

Cell order is deliberate. The registry check runs BEFORE either benchmark,
because `--models` takes registry family names and an unknown name is skipped
rather than raised -- the failure mode is a long wait ending in an empty table.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "BRUISE_UNIFIED" / "bruise_speed_harness.ipynb"

cells: list[dict] = []


def _cid() -> str:
    """nbformat 4.5+ requires a cell id; derive it from the index so it is stable."""
    return f"cell{len(cells):03d}"


def md(text: str) -> None:
    cells.append({"cell_type": "markdown", "id": _cid(), "metadata": {},
                  "source": text.strip("\n").splitlines(keepends=True)})


def code(text: str) -> None:
    cells.append({"cell_type": "code", "id": _cid(), "execution_count": None,
                  "metadata": {}, "outputs": [],
                  "source": text.strip("\n").splitlines(keepends=True)})


# ── 0 ────────────────────────────────────────────────────────────────────────
md("""
# Speed harness — is it the machine, or is it the precision?

On **one** Colab A100, **one** session, **one** architecture, measured 2026-08-04:

| harness | segformer_b0 |
|---|---|
| `evaluate.benchmark_speed` | 16.55 ms — 60.4 FPS |
| the 51-run sweep | 8.74 ms — 114.4 FPS |

1.89× apart with no machine to blame. `benchmark_speed` contains **no autocast
anywhere in it**, so the published table is pure fp32, while every other forward
in this codebase runs under `torch.amp.autocast`. On an A100 that is roughly 2×
for a transformer and much less for a depthwise-conv mobile net — which is the
shape of the gap.

That was untestable until now, because there was no fp16 path to compare against
on one machine. This notebook is that comparison. Run it top to bottom.

**Separately, and already settled:** `bruise_colab_final.ipynb` reproduced its own
July table to under 1% (16.55 vs 16.68 ms; the B2 teacher 33.649 vs 33.668). The
shipped `benchmark_640.csv` was never the problem. What was the problem is
comparing two rows without checking how each was measured.

Requires `BRUISE_SPEED_HARNESS_OVERLAY.zip` to have been extracted into this
bundle. §1 checks that before anything else.
""")

# ── 1 ────────────────────────────────────────────────────────────────────────
md("## 1 · Environment, patch check, and what the GPU is actually doing")

code("""
import os, sys, importlib
from pathlib import Path

# Run from the bundle root. Adjust BUNDLE if this notebook lives elsewhere.
BUNDLE = Path.cwd()
if not (BUNDLE / "bruisekit").is_dir():
    for cand in (BUNDLE.parent, *Path.cwd().glob("*/BRUISE_UNIFIED"), Path("/scratch")):
        hits = list(Path(cand).glob("**/bruisekit/inference.py")) if cand.exists() else []
        if hits:
            BUNDLE = hits[0].parent.parent
            break
os.chdir(BUNDLE); sys.path.insert(0, str(BUNDLE))
print("bundle :", BUNDLE)

# THE PATCH CHECK, before anything expensive. gpustate.py only exists if the
# overlay was extracted; without it the precision= argument is not there either
# and every number below would silently be the old fp32-only path.
try:
    from bruisekit import gpustate as G
    importlib.reload(G)
except ModuleNotFoundError:
    raise SystemExit(
        "bruisekit/gpustate.py is missing -- the overlay was not extracted here.\\n"
        "    cd " + str(BUNDLE) + " && unzip -o BRUISE_SPEED_HARNESS_OVERLAY.zip")

import inspect
from bruisekit import inference as INF
importlib.reload(INF)
if "precision" not in inspect.signature(INF.speed_table).parameters:
    raise SystemExit("bruisekit/inference.py is the OLD copy -- re-extract the overlay.")
print("patch  : APPLIED (gpustate present, speed_table takes precision=)")

# What the GPU is allowed to do. A latency without this is a number about a
# machine on a particular afternoon, not about a model.
print()
print(G.describe(G.probe()))
""")

# ── 2 ────────────────────────────────────────────────────────────────────────
md("""
## 2 · Point at the real runs directory — **this is the one that bites**

`registry.py` looks for each checkpoint in `env.runs` (= `WORK/runs`) **first**,
and silently falls back to the bundle's shipped `checkpoints/` when it finds
nothing there. The fallback is not an error and prints no warning — you get a
`WEIGHTS`-tier run pointing at an **old-lineage checkpoint** and never learn that
your real one was not consulted.

That is exactly what happened on ORC on 2026-08-04: `WORK` defaulted to
`BRUISE_UNIFIED/_work`, the real runs are in `/scratch/$USER/bruise_work/runs`,
and all three SegFormers loaded the shipped copies — which were saved under a
different `transformers` module layout and died with a missing-key
`RuntimeError`. The crash was the *second* symptom. The first was silent.

So `WORK` is set explicitly here, and the next cell **counts what it found** and
refuses to continue if the answer is zero.
""")

code("""
import os
from pathlib import Path

# The scratch dir holding runs/, cache640/ and outputs/. NOT the bundle.
# ORC: /scratch/<user>/bruise_work   Colab: /content/bruise_work
WORK = None                       # <- set explicitly, or leave None to auto-detect

if WORK is None:
    user = os.environ.get("USER", os.environ.get("LOGNAME", ""))
    for cand in (Path(f"/scratch/{user}/bruise_work"),
                 Path("/content/bruise_work"),
                 BUNDLE / "_work"):
        if (cand / "runs").is_dir() and any((cand / "runs").iterdir()):
            WORK = cand
            break
    else:
        WORK = BUNDLE / "_work"

print("WORK   :", WORK)
print("runs   :", WORK / "runs")
""")

md("""
## 3 · What the registry actually found, and where each checkpoint came from

`source` is the column to read. `runs` means your real training output;
`shipped` means the bundle's bundled copy was used instead — which is a silent
fallback, not a choice you made.
""")

code("""
import pandas as pd
from bruisekit.paths import setup
from bruisekit.registry import Registry, WEIGHTS

env = setup(work=WORK)
print(env.describe())

n_runs = len(list(env.runs.iterdir())) if env.runs.is_dir() else 0
print(f"\\n{n_runs} entries in {env.runs}")
if n_runs == 0:
    raise SystemExit(
        f"{env.runs} is empty or missing -- every checkpoint would silently come "
        f"from the bundle's shipped copies. Set WORK in the cell above to the "
        f"scratch dir that holds runs/ (on ORC: /scratch/$USER/bruise_work).")

reg = Registry(env, efficient_seeds=(0, 1, 2)).scan()

def _source(run):
    \"\"\"Did this checkpoint come from YOUR runs dir, or the bundle's shipped copy?\"\"\"
    if run.weights is None:
        return "-"
    try:
        run.weights.relative_to(env.runs)
        return "runs"
    except ValueError:
        return "shipped"

rows = {}
for r in reg.runs.values():
    d = rows.setdefault(r.family, {"family": r.family, "stage": r.stage,
                                   "kind": r.kind, "seeds_with_weights": [],
                                   "source": "-"})
    if r.tier == WEIGHTS:
        d["seeds_with_weights"].append(r.seed)
        if r.seed == 0:
            d["source"] = _source(r)

tbl = pd.DataFrame(sorted(rows.values(), key=lambda d: (d["stage"], d["family"])))
tbl["seeds_with_weights"] = tbl["seeds_with_weights"].apply(lambda s: sorted(x for x in s if x is not None))
tbl["timeable"] = tbl["seeds_with_weights"].apply(lambda s: 0 in s)

print()
print(tbl.to_string(index=False))

TIMEABLE = sorted(tbl.loc[tbl["timeable"], "family"])
print(f"\\n{len(TIMEABLE)} families have a seed-0 checkpoint and can be timed:")
print("   ", TIMEABLE)

shipped = sorted(tbl.loc[tbl["source"] == "shipped", "family"])
if shipped:
    print(f"\\n  WARNING  {len(shipped)} family(ies) fell back to the bundle's SHIPPED "
          f"checkpoint because {env.runs} has no run for them:")
    print("           ", shipped)
    print("           These may be an older lineage than what you trained. If a "
          "SegFormer is in that list, check WORK above before trusting its timing.")
""")

# ── 3 ────────────────────────────────────────────────────────────────────────
md("""
## 4 · Pick the models and name the machine

`MACHINE_TAG` goes into the output filename, so two machines' tables can never
collide or be mistaken for one another. Use something you will recognise in three
weeks: `orc-mig`, `colab-a100`, `mlidl-a100`.
""")

code("""
# Everything timeable, or replace with an explicit list. Names must come from
# TIMEABLE above -- anything else is silently skipped.
MODELS = tuple(TIMEABLE)

MACHINE_TAG = "orc-mig"      # <- CHANGE ME per machine

# The published recipe. Do not edit these to "speed things up": a row measured
# with different values is not comparable to the five in benchmark_640.csv.
REPEATS, WARMUP = 3, 10

print(f"{len(MODELS)} models on '{MACHINE_TAG}':")
for m in MODELS:
    print("   ", m)
print(f"\\nrecipe: {REPEATS} repeats x 185 images = {REPEATS * 185} timed calls per model, "
      f"{WARMUP} warmup, seed 0, batch 1, double cuda.synchronize()")
""")

# ── 4 ────────────────────────────────────────────────────────────────────────
md("""
## 5 · Build the 640 cache

Deterministic resize of `data/`, about a minute, reused by both runs. Already
built in most bundles — this is a no-op then.
""")

code("""
from bruisekit import loaders as L

man = {s: pd.read_csv(env.manifests / f"{s}.csv") for s in ("train", "val", "test")}
man640 = L.build_cache640(env, man)
""")

# ── 5 ────────────────────────────────────────────────────────────────────────
md("""
## 6 · fp32 — the published recipe

This is the path that produced the five shipped rows. `benchmark_speed` is called
directly, not a copy of it, so this number cannot drift from the published one.

The `UNDER LOAD` warning fires if the SM clock stayed below 90% of the card's max
during the timing loop — the check handbook §18.5 item 1 has been owed since the
private server's 765 MHz was read at idle and never confirmed under load.
""")

code("""
SP32 = INF.run(env, reg, INF.DEFAULT_CFG, man, man640, MODELS,
               do_inference=False, do_speed=True, do_reconcile=False,
               repeats=REPEATS, warmup=WARMUP,
               precision="fp32", machine_tag=MACHINE_TAG)["speed"]

SP32[["model", "median_ms", "fps", "p95_ms", "params_M", "peak_incremental_MB"]].round(3)
""")

# ── 6 ────────────────────────────────────────────────────────────────────────
md("""
## 7 · fp16 — the same loop, one line different

`_benchmark_logit_amp` mirrors `benchmark_speed` exactly — per-image batches,
double synchronisation, same repeats and warmup, threshold inside the timed
region — and differs only by wrapping the forward in `torch.amp.autocast`.

It is a separate function rather than a flag on the published one, for the reason
`_benchmark_cpu` already gives in that file: `benchmark_speed` is written out
verbatim by the notebook's `%%writefile` cell, so a branch inside it would put the
published number one default-argument change away from silently becoming a
different number.

**These rows are not publishable as speed results.** They exist to attribute a gap.
""")

code("""
SP16 = INF.run(env, reg, INF.DEFAULT_CFG, man, man640, MODELS,
               do_inference=False, do_speed=True, do_reconcile=False,
               repeats=REPEATS, warmup=WARMUP,
               precision="fp16", machine_tag=MACHINE_TAG)["speed"]

SP16[["model", "median_ms", "fps", "p95_ms", "params_M", "peak_incremental_MB"]].round(3)
""")

# ── 7 ────────────────────────────────────────────────────────────────────────
md("""
## 8 · The verdict

If `fp32 / fp16` lands near **1.9× for segformer_b0** and materially lower for the
mobile nets, precision was the confound and the "SegFormer is a `transformers`
version anomaly" story in handbook §7.3 / §14 caveat 6 is unnecessary.

If both precisions come out near 16.5 ms, precision is **not** the answer and the
sweep differed some other way — post that result and we look again at warmup,
batch shape, and what the sweep actually timed.
""")

code("""
cmp = (SP32[["model", "median_ms", "fps", "params_M"]]
       .merge(SP16[["model", "median_ms", "fps"]], on="model",
              suffixes=("_fp32", "_fp16")))
cmp["speedup_fp32_over_fp16"] = (cmp["median_ms_fp32"] / cmp["median_ms_fp16"]).round(3)

# Ratios normalised to the fastest model are the only thing portable across
# machines -- absolute latency is not. Two FULL A100s disagree by 1.29x on the
# same model (Colab 16.64 ms vs mlidl 21.49 ms for segformer_b0).
base = cmp["median_ms_fp32"].min()
cmp["ratio_vs_fastest_fp32"] = (cmp["median_ms_fp32"] / base).round(3)

cmp = cmp.sort_values("median_ms_fp32").reset_index(drop=True)
print(cmp.to_string(index=False))

print()
seg = cmp[cmp["model"].str.startswith("segformer")]
if not seg.empty:
    s = seg.iloc[0]
    print(f"segformer fp32/fp16 = {s['speedup_fp32_over_fp16']:.2f}x  "
          f"({s['median_ms_fp32']:.2f} ms -> {s['median_ms_fp16']:.2f} ms)")
    print("  the sweep's 8.74 ms is explained by precision"
          if abs(s["median_ms_fp16"] - 8.74) < 1.5 else
          "  fp16 does NOT land on the sweep's 8.74 ms -- precision is not the whole story")

out = env.out / "inference" / f"speed_precision_comparison_{MACHINE_TAG}.csv"
cmp.to_csv(out, index=False)
print(f"\\nwrote {out}")
""")

# ── 8 ────────────────────────────────────────────────────────────────────────
md("""
## 9 · What to send back

Three files from `_work/outputs/inference/`:

    benchmark_640_cuda_fp32_<tag>.csv
    benchmark_640_cuda_fp16_<tag>.csv
    speed_precision_comparison_<tag>.csv

Run the same three cells on every machine you care about — change `MACHINE_TAG`,
nothing else. The filenames carry device and precision, so nothing collides.

**Two guards will stop you rather than warn you**, both raising inside
`check_single_machine`: a table that mixes devices, and a table that mixes
precisions. Each produces a file that reads like a hardware result and is not one.

**Still open:** the 51-run sweep script is in neither the bundle nor the repo
(`peak_incremental_mb`, `training_variant` match zero files). Until it turns up,
"the sweep used fp16" is a hypothesis — but §7 tests it without needing the script
at all.

**Do not rewrite handbook §7.3 or §14 caveat 6 from one machine's output.** Caveat
6's ORC SegFormer row (8.97 ms) has no counterpart in `benchmark_stage_e.csv`,
which contains only the four mobile nets — so that row came from a different
harness than the four beside it. Fixing it needs the same recipe on both machines,
which is what these cells produce.
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "colab": {"provenance": [], "toc_visible": True},
    },
    "nbformat": 4, "nbformat_minor": 5,
}

OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
n_code = sum(1 for c in cells if c["cell_type"] == "code")
print(f"wrote {OUT}\n  {len(cells)} cells ({n_code} code, {len(cells) - n_code} markdown)")
