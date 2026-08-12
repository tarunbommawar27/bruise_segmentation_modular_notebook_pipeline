#!/usr/bin/env python
"""Emit `bruise_stage_o_train.ipynb` -- train the ITA-grouped arm anyway.

Stage O's gate closed on both schemes (§7k.5). This notebook trains the arm
regardless, on purpose, and is a separate file from `bruise_stage_o.ipynb` for
that reason: the analysis notebook must stay runnable without implying that
anyone accepted the override.

WHY THIS IS A LEGITIMATE THING TO BUILD
-----------------------------------------
A pre-registration is written to be falsifiable, not to be obeyed. The gate's
job was to stop a *speculative* grid; it is not evidence about what the arm
actually does. A measured null is a stronger paper section than a predicted one,
and if the arm surprises us the gate was wrong in a way worth knowing.

What would NOT be legitimate is a results directory that six months from now
looks indistinguishable from a gate-approved run. So `record_override` writes
`FORCED_GATE.json` beside the runs, carrying the failing clauses verbatim, and
every table below is generated from that directory.

THE TWO THINGS THIS ADDS OVER `--force-train`
-----------------------------------------------
1. A PREFLIGHT. The three shims have never run together against
   `engine.train_run`. Their failure modes -- an untagged loader, a teacher
   stack with the wrong K, a loss dispatcher installed in the wrong order --
   all surface on the first optimizer step, which is twenty minutes into a
   multi-hour job. `itakd.preflight` runs one forward and one backward through
   the real shims and raises on any of them in about a minute.

2. THE CONTRASTS THE GATE WAS SUPPOSED TO PREDICT. Scoring the trained arms
   against their controls AND against Stage M's `*_mtkd` arms is what turns
   this into evidence about the gate itself: the gate projected a student gain
   of -0.0015, and the point of running is to find out what the student
   actually did.

WRITES TO STAGE_O_RESULTS/runs/ AND STAGE_O_RESULTS/trained/
--------------------------------------------------------------
Separate from `tables/`, which holds the gate and the analyses. A forced run
cannot overwrite the analysis that advised against it.
"""
from __future__ import annotations

import json
from pathlib import Path

DST = (Path(__file__).resolve().parent.parent / "BRUISE_UNIFIED"
       / "bruise_stage_o_train.ipynb")

CELLS: list[tuple[str, str]] = []


def md(src: str) -> None:
    CELLS.append(("markdown", src.strip("\n")))


def code(src: str) -> None:
    CELLS.append(("code", src.strip("\n")))


# ─────────────────────────────────────────────────────────────────────────────
md("""
# Stage O — training the ITA-grouped arm **against a closed gate**

`ita_group_gate` closed on both schemes. This notebook trains the arm anyway.

## Why that is a defensible decision

A pre-registration exists to be **falsifiable**, not to be obeyed. The gate's job
was to stop a *speculative* grid before it consumed a day of GPU time — it is a
projection, not evidence about what the arm does. Two things follow:

- **A measured null is a stronger result than a predicted one.** "We ran it and
  it did nothing" survives review in a way "our gate said not to" does not.
- **If the arm surprises us, the gate was wrong in a way worth knowing** — and
  the identifiability clause is new enough that it deserves to be tested against
  an outcome rather than trusted.

## What is *not* defensible, and how this notebook prevents it

A results directory that later looks gate-approved. `record_override` writes
`FORCED_GATE.json` beside the runs with the failing clauses verbatim:

| clause | value |
|---|---|
| weighting gain over best single teacher | **−0.0056** [−0.0174, −0.0016] |
| projected student gain vs +0.01 margin | **−0.0015** |
| groups with an identifiable argmax | **0 of 2** |

Every table this notebook produces comes from that directory. Report the outcome
as *obtained against a negative pre-test*, and quote the identifiability table
beside it.

## What this notebook adds over `run_stage_o.py --force-train`

**§5 is a preflight.** The three shims have never run together against
`engine.train_run`. An untagged loader, a teacher stack with the wrong K, or a
loss dispatcher installed in the wrong order all surface on the *first optimizer
step* — twenty minutes into a multi-hour job. One forward and one backward
through the real shims catches all three in about a minute.

**§8 runs the contrasts the gate tried to predict**, including against Stage M's
per-image router, which is the comparison that says whether grouping by skin tone
is better or worse than routing per image.
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
## §1 — Configuration, and the acknowledgement

`OVERRIDE_REASON` is written into `FORCED_GATE.json` and is not optional. Write a
real sentence: it is what a reader six months from now has instead of your
memory.
""")

code('''
from pathlib import Path

BUNDLE     = None      # None = auto-detect
WORK       = None      # None = <bundle>/_work
EXTRA_RUNS = "/scratch/tbommawa/bruise_work/runs"
N4_RESULTS = None      # None = <bundle>/STAGE_N4_RESULTS  (medsam_ft lives here)

SCHEME     = "light_vs_rest"          # the pre-registered two-group collapse
FAMILIES   = ("segformer_b0_itakd", "lraspp_mobilenetv3_itakd")
SEEDS      = (0, 1, 2)                # three, or the contrast has no variance
EPOCHS     = 100                      # cap; the engine early-stops on patience
MAX_MICRO  = 16                       # accumulation restores the control's
                                      # EFFECTIVE batch exactly

# Written verbatim into FORCED_GATE.json.
OVERRIDE_REASON = (
    "The gate is a projection, not a measurement. We are training to obtain the "
    "null empirically rather than by prediction, and to test the new "
    "identifiability clause against an actual outcome.")

RUN_PREFLIGHT = True
RUN_TRAINING  = False     # flip AFTER the preflight passes
''')

# ── §2 ───────────────────────────────────────────────────────────────────────
md("""
## §2 — Environment and self-test

The self-test asserts the two identities that keep this a one-variable contrast —
K=1 reduces to Stage H's gated loss exactly, and a uniform weight matrix
reproduces Stage C's uniform ensemble — plus the guards that make a missing group
vector raise instead of silently becoming that uniform ensemble.
""")

code('''
import json
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 60)

from bruisekit import itakd, loaders as L, paths as P
from bruisekit.registry import Registry

env = P.setup(root=BUNDLE, work=WORK, extra_runs=EXTRA_RUNS)
print(env.describe())

assert str(env.device).startswith("cuda"), (
    f"device is {env.device}. This notebook trains six runs with a "
    f"three-teacher pool resident; it needs a GPU.")

print("\\n-- itakd self-test --")
assert itakd.self_test(), "itakd.self_test() failed -- do not train"
''')

# ── §3 ───────────────────────────────────────────────────────────────────────
md("""
## §3 — Manifests, cache, config

The recipe is handbook §3's, unchanged. The arm differs from its control in the
teacher signal and nothing else — that is the entire basis for reading the
contrast in §8.
""")

code('''
MAN = {s: pd.read_csv(env.manifests / f"{s}.csv") for s in ("train", "val", "test")}
for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
    assert not (set(MAN[a].subject) & set(MAN[b].subject)), f"{a}/{b} share subjects"
print("  no subject appears in two splits")

MAN640 = L.build_cache640(env, MAN)

CFG = dict(
    img_size=640, epochs=EPOCHS, patience=15,
    batch_mode="per_model", effective_batch=8, max_probe_batch=64, vram_target=0.75,
    backbone_lr=6e-5, head_lr=6e-4, betas=(0.9, 0.999), weight_decay=0.01,
    warmup_fraction=0.01, poly_power=1.0, gradient_clip=1.0, amp=True,
    workers=0, eval_batch=8, eval_batch_cpu=2,
    segformer_alpha=0.6, efficient_alpha=0.6, aux_weight=0.4,
    smp_encoder="resnet50", smp_micro_batch=16, efficient_micro_batch=16,
    cut_min=-6.0, cut_max=6.0, cut_steps=481,
    drive_sync_every=2, n_boot=2000, n_boot_final=10_000,
)
REG = Registry(env).scan()
RUNS_DIR = itakd.results_dir(env, "runs")
print(f"\\nruns -> {RUNS_DIR}")
''')

# ── §4 ───────────────────────────────────────────────────────────────────────
md("""
## §4 — The gate we are overriding, and the weights we are overriding it with

Read the verdict before continuing. If `tables/ita_group_gate__*.json` is absent,
run `bruise_stage_o.ipynb` §7–§9 first — training without the gate on disk would
leave no record of what was overridden.
""")

code('''
GATE_PATH = itakd.results_dir(env, "tables") / f"ita_group_gate__{SCHEME}.json"
assert GATE_PATH.exists(), (
    f"{GATE_PATH} is absent. Run bruise_stage_o.ipynb through the gate first -- "
    f"a forced run with no record of what it overrode is the thing this notebook "
    f"exists to prevent.")

GATE = json.loads(GATE_PATH.read_text())
print(itakd.format_gate(GATE))

W = itakd.weight_array(pd.DataFrame(GATE["weights"]), itakd.POOL, SCHEME)
print(f"\\nweight matrix [{W.shape[0]} groups x {W.shape[1]} teachers], "
      f"rows in {itakd.group_order(SCHEME)} order:")
print(np.round(W, 4))

OVERRIDE = itakd.record_override(env, GATE, OVERRIDE_REASON, RUNS_DIR)
''')

# ── §5 ───────────────────────────────────────────────────────────────────────
md("""
## §5 — Preflight: one batch through the real shims

Install order matters and is asserted, not assumed:

1. `install_group_shim` — wraps the **training** loader so each batch records its
   stems' ITA group indices. `engine.train_run` iterates `(x, y, _)` and throws
   the stem away, so this is the only way the loss learns which group an image is
   in without editing the shared training loop.
2. `install_teacher_shim` — rebinds `engine.load_teacher` to return `[B, K, H, W]`.
   Handles MedSAM, which has no registry `Run`.
3. `install_loss_shim` — **last**. In a session that also touched Stage H or M,
   the dispatcher must sit on top of theirs.

The preflight raises rather than warns on: an untagged loader, `K` disagreeing
with the weight matrix, the wrong loss class coming back from the dispatcher, a
non-finite loss, or no parameter receiving a gradient.
""")

code('''
if not RUN_PREFLIGHT:
    print("RUN_PREFLIGHT is False -- skipping. Not recommended.")
else:
    itakd.install_group_shim(itakd.build_group_map(MAN640, SCHEME))
    itakd.install_teacher_shim(env, REG, CFG, MAN640, pool=itakd.POOL,
                               n4_root=N4_RESULTS)
    itakd.install_loss_shim(W)          # LAST

    t0 = time.time()
    PRE = itakd.preflight(env, REG, CFG, MAN640, W, family=FAMILIES[0])
    print(f"\\n  preflight OK in {time.time() - t0:.0f}s -- the wiring holds.")
    itakd.save(env, "preflight", PRE, subdir="runs")
''')

# ── §6 ───────────────────────────────────────────────────────────────────────
md("""
## §6 — Train

Six runs: two students × three seeds. Budget ~2.4 h and ~2.0 h per seed
respectively on an A100 MIG, so roughly **13 GPU-hours**. Resumable — an
interrupted run picks up from `resume.pt`, and a finished one is skipped via
`DONE.json`.

The micro-batch is **pinned to each arm's control** with gradient accumulation
restoring the effective batch exactly, so the arm differs from its control in the
teacher signal alone and not also in step count and LR schedule. The one residual
difference is SegFormer's decode-head BatchNorm, which normalises over the
micro-batch — that belongs in the limitations and is recorded in
`multiteacher.control_batch`'s docstring.
""")

code('''
if not RUN_TRAINING:
    print("RUN_TRAINING is False. Run the preflight first, read it, then flip "
          "RUN_TRAINING in §1.")
else:
    t0 = time.time()
    TRAINED = itakd.train_arms(env, REG, CFG, MAN640, RUNS_DIR,
                               families=FAMILIES, seeds=SEEDS,
                               max_micro=MAX_MICRO)
    print(f"\\ntrained {len(TRAINED)} run(s) in {(time.time() - t0) / 3600:.1f} h")
    print(TRAINED)
''')

md("""
### Read the router statistics before any Dice number

`images_per_group` is the first thing to check. An arm whose loss never saw one
of the groups did not run this experiment, and its Dice answers a different
question. `mean_coverage` is Stage H's gate coverage on the fused teacher — if it
is near zero the soft term was gated off almost everywhere and the arm is
effectively supervised-only.
""")

code('''
rows = []
for p in sorted(RUNS_DIR.glob("*/group_loss_stats.json")):
    s = json.loads(p.read_text())
    rows.append({"run": p.parent.name,
                 "images_per_group": s["images_per_group"],
                 "mean_coverage": round(s["mean_coverage"], 4),
                 "fused_teacher_dice": round(s["mean_fused_teacher_soft_dice"], 4),
                 "alpha_effective": round(s["mean_alpha_effective"], 4)})
print(pd.DataFrame(rows).to_string(index=False) if rows else "no runs yet")
''')

# ── §7 ───────────────────────────────────────────────────────────────────────
md("""
## §7 — Score the trained arms

Operating point fitted on **validation**, then applied once to test. Same rule as
every other stage: nothing is fitted on test, ever.
""")

code('''
from bruisekit.evaluate import evaluate_at_cut
from bruisekit.data import make_loader
from bruisekit.sweep import cache_logits, select_cut, sweep_cuts
import torch

SCORED = {}
for d in sorted(RUNS_DIR.glob("*__seed*")):
    if not (d / "best.pt").exists():
        continue
    fam = d.name.rsplit("__seed", 1)[0]
    spec = L.spec_for(fam)
    model = L.build_for_load(env, fam, CFG["smp_encoder"])
    st = torch.load(str(d / "best.pt"), map_location="cpu", weights_only=True)
    model.load_state_dict(st["model"] if "model" in st else st)
    model.to(env.device).eval()

    vl = make_loader(MAN640["val"], env.cache640, CFG["img_size"], 8, False, 0, 0)
    lg, gt, _ = cache_logits(model, vl, env.device, CFG["amp"])
    cut = select_cut(sweep_cuts(lg, gt, np.linspace(-6, 6, 481)))

    tl = make_loader(MAN640["test"], env.cache640, CFG["img_size"], 8, False, 0, 0)
    with torch.inference_mode():
        per_image, _ = evaluate_at_cut(model, tl, env.device, cut["cut"], CFG["amp"])
    SCORED[d.name] = per_image
    itakd.save(env, f"test_per_image__{d.name}", per_image, subdir="trained")
    print(f"  {d.name:<34} cut {cut['cut']:+.3f}  "
          f"mean {per_image.dice.mean():.4f}  median {per_image.dice.median():.4f}  "
          f"misses {int((per_image.dice == 0).sum())}")
    del model
    torch.cuda.empty_cache()
''')

# ── §8 ───────────────────────────────────────────────────────────────────────
md("""
## §8 — The contrasts the gate tried to predict

Three comparisons, in decreasing order of what they settle:

| contrast | what it answers |
|---|---|
| `itakd − control` | did grouping by skin tone beat single-teacher KD? **This is the one the gate projected at −0.0015.** |
| `itakd − mtkd` | is grouping by skin tone better or worse than Stage M's per-image router? |
| miss rate, both | the endpoint this study is judged on; Dice is saturated |

Paired subject-level bootstrap, because the two arms score the same 185 images
and a paired test on those pairs is far more sensitive than comparing two
marginal intervals.
""")

code('''
from bruisekit import report as R
from bruisekit.lesionsize import paired_stratum

META = MAN["test"]
CONTROLS = itakd.CONTROL_FOR

def _load(name):
    for d in (itakd.results_dir(env, "trained"),
              env.root / "FINAL_RESULT" / "RESULT_AUGUST_08",
              env.root / "FINAL_RESULT" / "STAGE_M_RESULTS"):
        for pat in (f"test_per_image__{name}.csv", f"per_image_{name}.csv",
                    f"per_image_distill_{name}.csv"):
            p = d / pat
            if p.exists():
                return R.normalize(pd.read_csv(p), META)
    return None

rows = []
for fam in FAMILIES:
    for seed in SEEDS:
        a = _load(f"{fam}__seed{seed}")
        if a is None:
            continue
        for other, label in ((CONTROLS[fam], "vs control"),
                             (fam.replace("_itakd", "_mtkd"), "vs Stage M router")):
            b = _load(f"{other}__seed{seed}") or _load(other)
            if b is None:
                continue
            j = a[["stem", "dice"]].merge(b[["stem", "dice"]], on="stem",
                                          suffixes=("_a", "_b"))
            rows.append({
                "arm": f"{fam}__seed{seed}", "against": other, "kind": label,
                "n": len(j),
                "delta_mean_dice": float(j.dice_a.mean() - j.dice_b.mean()),
                "delta_median_dice": float(j.dice_a.median() - j.dice_b.median()),
                "misses_arm": int((j.dice_a == 0).sum()),
                "misses_other": int((j.dice_b == 0).sum()),
            })
CONTRASTS = pd.DataFrame(rows)
print(CONTRASTS.to_string(index=False) if len(CONTRASTS) else "nothing to compare yet")
if len(CONTRASTS):
    itakd.save(env, "forced_contrasts", CONTRASTS, subdir="trained")
    print(f"\\nGate projected a student gain of "
          f"{GATE['projected_student_gain']:+.4f}.")
    print(f"Observed mean delta vs control: "
          f"{CONTRASTS[CONTRASTS.kind == 'vs control'].delta_mean_dice.mean():+.4f}")
''')

# ── §9 ───────────────────────────────────────────────────────────────────────
md("""
## §9 — How to report whatever this produced

**If it is a null** — the expected outcome — you now have it *measured* rather
than projected, which is the stronger form. Report it as the fourth consecutive
KD null in this project (Stage C `p3_adaptive_group` at 0.7586, Stage H, Stage M),
and say that the gate predicted it: a pre-test that correctly forecast a null it
was then allowed to be checked against is a working pre-test, and that is worth a
sentence of its own.

**If the arm wins**, the interesting object is the *gate*, not the arm. Check in
this order before believing it: (1) `images_per_group` in §6 — did the loss see
both groups? (2) is the win larger than the seed-to-seed spread, which is ±0.004
on B2? (3) does the paired interval exclude zero? A win under +0.05 Dice is
inside the annotation-ceiling noise floor regardless of its interval (§1).

**Either way, `FORCED_GATE.json` travels with the numbers.** These runs were
obtained against a negative pre-test and every table drawn from them must say so.
Quote the identifiability table beside the result — 0 of 2 groups had an
estimable argmax, and that remains true whatever the students did.

**What this cannot settle.** Whether a *better routing key* would work. The
per-image oracle gain of +0.045 says the teachers genuinely complement each
other; this stage only shows that skin tone does not identify when. A key that
did — lesion size, image quality, teacher agreement — is a different experiment
and Stage M already tested the last of those.
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
