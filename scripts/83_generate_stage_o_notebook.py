#!/usr/bin/env python
"""Emit `bruise_stage_o.ipynb` -- the miss taxonomy, the distilled-arm fairness
re-analysis, and ITA-group-routed gated multi-teacher distillation.

Same generator discipline as 69/70/71/77/78/79: the notebook is an OUTPUT, never
hand edited, and ships with zero executed cells so no key, path or partial result
can travel inside it.

Everything the notebook writes goes to `STAGE_O_RESULTS/`. It never touches
`results/`, `FINAL_RESULT/`, `LESION_SIZE_RESULTS/`, `STAGE_M_RESULTS/`,
`STAGE_N4_RESULTS/` or `_work/runs/`. Those trees are READ -- for the per-image
CSVs, for MedSAM's checkpoint and its val-fitted cut, and for the reference rows
-- and never rewritten.

WHY THE THREE TODO ITEMS SHARE ONE NOTEBOOK
--------------------------------------------
They share one set of per-image tables and one results directory, and -- more to
the point -- items 1 and 3 are the evidence that decides whether item 2 is worth
the GPU time. The miss taxonomy is what shows that `fastscnn_rgkd`'s gate
converted empty predictions into misplacements rather than fixing them, and the
per-group fairness tables are what show that the teachers' misses live on Light
(II-III). Both of those bear directly on whether an ITA-grouped router has
anything to route.

Splitting them into three notebooks would mean three copies of the lineage load
and three chances for the model list to drift.

THE FIRST HALF NEEDS NO GPU. Cells through §6 run on a laptop off the per-image
CSVs in the bundle, and they close TODO 1 and TODO 3 on their own. §7 onward needs
the teacher checkpoints, and §10 needs CUDA.
"""
from __future__ import annotations

import json
from pathlib import Path

DST = Path(__file__).resolve().parent.parent / "BRUISE_UNIFIED" / "bruise_stage_o.ipynb"

CELLS: list[tuple[str, str]] = []


def md(src: str) -> None:
    CELLS.append(("markdown", src.strip("\n")))


def code(src: str) -> None:
    CELLS.append(("code", src.strip("\n")))


# ─────────────────────────────────────────────────────────────────────────────
md("""
# Stage O — the miss taxonomy, distilled-arm fairness, and ITA-grouped multi-teacher KD

Three open items, one notebook, because the first two decide whether the third is
worth running.

| | item | needs |
|---|---|---|
| **§4** | **TODO 1** — complete miss rate **and** zero Dice for the distillation arms | per-image CSVs only |
| **§5** | **TODO 3** — skin tone × lesion size for the best distilled models | per-image CSVs only |
| **§7–§11** | **TODO 2** — gated multi-teacher distillation grouped by ITA | teacher checkpoints, then CUDA |

## §4 — the thing the study has been reporting as one number

`complete_miss_rate` is `dice == 0`. That is the **union** of two clinically
different failures:

| | definition | what the clinician sees |
|---|---|---|
| **empty prediction** | `pred_positive_pixels == 0` | nothing — which invites a second look |
| **wrong place** | `dice == 0` and `pred > 0` | a confident outline in the wrong location |

They differ in **28 of the 40** per-image tables in `RESULT_AUGUST_08`.
`fastscnn_rgkd` is the case that matters: reliability gating took its empty
predictions from 6 to 1 while its zero-Dice count moved only 8 → 6, because the
gate converted blank outputs into confident misplacements. A one-column table
calls that "roughly unchanged". It is not.

## §5 — the table the deck was missing

`FINAL_RESULT/RESULT_AUGUST_08/fairness_stats.csv` covers 24 models and excludes
**every Stage C distillation arm** — including `p3_adaptive`, the best arm in the
study at 0.7748. So the one question the deck gets asked about the best distilled
model had no table behind it. Nothing here is retrained.

## §7–§11 — why this is not Stage M again

Stage M routed **per image** on each teacher's soft Dice against the label, and
is explicit that it does *not* route by skin tone. It came back a null: six
contrasts, all inconclusive, miss rate slightly worse.

Stage O routes **per ITA group** on weights fitted once on validation:

$$w_g = \\mathrm{softmax}(\\beta \\cdot \\overline{\\mathrm{dice}}_{k,g})$$

- **No label is consulted at routing time.** The only per-image input is its ITA
  group, a manifest column — so unlike Stage M's router this one is not
  restricted to training.
- **K weights per group, not per image.** Two groups over 20 validation subjects
  is estimable; five is not, which is the next point.
- **The gate has an identifiability clause Stage M did not have.** With six
  candidate teachers the per-group argmax on 134 val images is bootstrap-stable
  only 36–52 % of the time, with margins of 0.001–0.008 Dice. An arm routed on
  an argmax that unstable is fitting sampling noise and will produce a fourth
  null. `ita_group_gate` refuses unless one group's argmax survives resampling
  at p ≥ 0.75.

**Everything is written to `STAGE_O_RESULTS/`** and nowhere else.
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

`POOL`, `SCHEMES`, `BETA`, `GATE_LO/HI`, `MIN_IDENTIFIABILITY` and `MARGIN` are
pre-registered in `bruisekit/itakd.py`. Changing them changes the experiment — do
it in the module, deliberately, and say so in the write-up.
""")

code('''
from pathlib import Path

BUNDLE      = None      # None = auto-detect
WORK        = None      # None = <bundle>/_work
EXTRA_RUNS  = "/scratch/tbommawa/bruise_work/runs"

# Trees to scan for per-image CSVs. NEVER a single hard-coded lineage: the laptop
# keeps them in FINAL_RESULT/RESULT_AUGUST_08 and ORC HAS NO FINAL_RESULT AT ALL,
# so a fixed path runs on one host and raises FileNotFoundError on the other.
# Handbook 10.3 has the per-host table. Same list bruise_all_models.ipynb uses.
EXTRA_ROOTS = [
    "/scratch/tbommawa/bruise_work",
    "/scratch/tbommawa/bruise_work/outputs",
    "/scratch/tbommawa/BRUISE_UNIFIED",
]

N4_RESULTS  = None      # None = <bundle>/STAGE_N4_RESULTS  (medsam_ft lives here)

SCHEME      = "light_vs_rest"   # pre-registered. "five" is reported alongside.
ALSO_FIVE   = True              # run the gate on both, so "underpowered" is shown
                                # rather than asserted

FAMILIES    = ("segformer_b0_itakd", "lraspp_mobilenetv3_itakd")
SEEDS       = (0, 1, 2)
EPOCHS      = 100       # cap; the engine stops early on patience
MAX_MICRO   = 16        # student micro-batch cap; accumulation restores the
                        # control's EFFECTIVE batch exactly, so step count and LR
                        # schedule are unchanged

N_BOOT      = 10_000    # fairness / contrast bootstrap
REPS        = 4_000     # identifiability bootstrap

RUN_TRAINING = False    # flip to True only after reading the gate in §9
''')

# ── §2 ───────────────────────────────────────────────────────────────────────
md("""
## §2 — Environment and the module self-test

`self_test()` needs no weights, no GPU and no network. It asserts the three
identities the stage rests on: `zero_dice == empty + wrong_place`, `K = 1` reduces
to Stage H's gated loss exactly, and a uniform weight matrix reproduces Stage C's
`p2_ensemble_uniform`. It also asserts that the loss **raises** when the batch's
ITA groups were not recorded — see §9 for why that guard is the important one.
""")

code('''
import numpy as np
import pandas as pd

from bruisekit import itakd, lesionsize as LS, paths

env = paths.setup(root=BUNDLE, work=WORK, extra_runs=EXTRA_RUNS)
print(f"root    : {env.root}")
print(f"device  : {env.device}")
print(f"results : {itakd.results_dir(env)}")

print("\\n-- itakd self-test --")
assert itakd.self_test(), "itakd.self_test() failed -- do not run anything below"
''')

# ── §3 ───────────────────────────────────────────────────────────────────────
md("""
## §3 — Discovery: what per-image tables are on *this* machine

`load_tables` **scans**; it does not take a lineage path. That is deliberate and it
is handbook §10.3:

| | laptop | ORC |
|---|---|---|
| `FINAL_RESULT/RESULT_AUGUST_08/` (40 CSVs) | ✅ | ❌ **never synced** |
| `<work>/outputs/` | ❌ | ✅ the ORC default |
| `STAGE_N4_RESULTS/runs/` (checkpoints) | ❌ empty | ✅ |

A hard-coded lineage runs on one host and raises `FileNotFoundError` on the other.
This scans every root in `EXTRA_ROOTS` plus the bundle and work dirs, **merges
across them**, and groups the results into cohorts by (stem set, GT-area vector) —
only the largest coherent group is returned, because everything below compares
models to each other and a decile cut computed over two mask versions belongs to
neither.

Read the discovery log it prints. **If a model you expect is missing, the fix is
almost always another entry in `EXTRA_ROOTS`, not a code change.**

`itakd.all_models()` is `lesionsize.DEFAULT_MODELS` plus the ten Stage C
distillation arms and the seven `_distilled` / `_rgkd` variants the 24-model
fairness export left out. On a host that has only some of them, the missing ones
are **named** and the rest proceed.
""")

code('''
MODELS = itakd.all_models()
print(f"requesting {len(MODELS)} models\\n")

TABLES, KEY, FOUND = itakd.load_tables(env, extra_roots=EXTRA_ROOTS, models=MODELS)
''')

# ── §4 ───────────────────────────────────────────────────────────────────────
md("""
## §4 — TODO 1: zero Dice, decomposed

`wrong_place` is **derived** as `zero_dice − empty_pred`, never counted
separately, so the three columns cannot print numbers that fail to add up. An
image with no predicted pixels always scores Dice 0, so `empty ≤ zero` is an
arithmetic fact — `_miss_counts` raises if a table violates it, because that means
the Dice column and the pixel counts came from different evaluations and every
number below would be void.
""")

code('''
TAX = itakd.miss_taxonomy(TABLES)
itakd.print_miss_taxonomy(TAX)

TAX_ITA  = itakd.miss_taxonomy_by(TABLES, "skin_tone_category")
TAX_SIZE = itakd.miss_taxonomy_by(TABLES, "size", KEY)

for name, obj in (("miss_taxonomy", TAX),
                  ("miss_taxonomy_by_ita", TAX_ITA),
                  ("miss_taxonomy_by_size", TAX_SIZE)):
    print(f"  wrote {itakd.save(env, name, obj)}")
''')

md("""
### The two rows to read before anything else

`fastscnn_rgkd` against `fastscnn_b2kd` is the whole argument for reporting three
columns: same student, same teacher, the gate is the only difference, and it moves
the *composition* of the misses far more than their count.

The KD arms with **zero empty predictions but non-zero wrong-place** are the other
group worth naming. An arm that never returns a blank mask looks flawless on an
"empty prediction" metric and still misses bruises entirely.
""")

code('''
FOCUS = ["fastscnn", "fastscnn_b2kd", "fastscnn_rgkd", "fastscnn_distilled",
         "p3_adaptive", "p2_cwd_b5_to_b0", "segformer_b0_distilled",
         "segformer_b0_rgkd", "yolo_sem_direct"]
cols = ["model", "n", "zero_dice_n", "empty_pred_n", "wrong_place_n", "mean_dice"]
print(TAX[TAX.model.isin(FOCUS)][cols].to_string(index=False))

only_wrong = TAX[(TAX.empty_pred_n == 0) & (TAX.wrong_place_n > 0)]
print(f"\\n{len(only_wrong)} arms never return a blank mask and still miss "
      f"entirely:\\n  {', '.join(only_wrong.model)}")
''')

# ── §5 ───────────────────────────────────────────────────────────────────────
md("""
## §5 — TODO 3: skin tone × lesion size for the distilled arms

Every frame here comes out of `lesionsize`, not out of a second implementation.
The only thing Stage O adds is the **model list**.

`fairness_conditioned` is the one to read carefully. It reports each model's
best-minus-worst group gap marginally and again **within the small-lesion
stratum**. If the gap shrinks once size is roughly held fixed, the marginal gap
was partly a size effect wearing a skin-tone label — which is exactly what §8.4 of
the handbook warns about and what nothing in the study had checked for these arms.

Cells with fewer than 5 subjects get `NaN` intervals rather than a number nobody
should read. Several will.
""")

code('''
BEST = itakd.best_distilled(TAX, TABLES, n=6)
print(f"leading distilled arms (mean Dice, misses as tiebreak):\\n  {BEST}\\n")

FAIR = itakd.distilled_fairness(TABLES, KEY, models=None, n_boot=N_BOOT)
for name, obj in FAIR.items():
    print(f"  wrote {itakd.save(env, f'fairness__{name}', obj)}")
print(f"  wrote {itakd.save(env, 'leading_distilled_arms', {'models': BEST})}")
''')

md("""
### The leading arms, side by side

Read the miss columns before the Dice columns. §1 of the handbook puts every
model in this table inside the annotation-ceiling band, so a Dice gap under ~0.05
between any two rows is not a result.
""")

code('''
h = FAIR["headline"]
show = [c for c in ("model", "all_mean_dice", "all_median_dice", "all_zero_dice_n",
                    "all_empty_pred_n", "D1_D4_mean_recall", "D1_D4_zero_dice_n",
                    "D1_mean_recall") if c in h.columns]
print(h[h.model.isin(BEST)][show].to_string(index=False))

print("\\n-- misses by ITA group, leading arms --")
m = FAIR["miss_by_ita"]
print(m[m.model.isin(BEST)][["model", "stratum", "n", "n_subjects",
                             "zero_dice_n", "empty_pred_n", "wrong_place_n"]]
      .to_string(index=False))
''')

# ── §6 ───────────────────────────────────────────────────────────────────────
md("""
## §6 — Everything above is done. Everything below needs checkpoints.

§4 and §5 close TODO 1 and TODO 3 from the per-image CSVs alone. If that is all
you came for, stop here — `STAGE_O_RESULTS/tables/` has eleven CSVs and the
JSON.

**These two halves belong on different machines** (handbook §10.3). §4–§5 read
per-image CSVs, which live on the **laptop**. §7–§11 load checkpoints, which live
on **ORC**. Running the wrong half on the wrong host is a path error, not a bug —
if §3 reported far fewer models than you expected, you are on ORC and should skip
to §7.

From §7 on, the notebook needs:

- the pooled teachers' checkpoints (`EXTRA_RUNS` on ORC), and
- `STAGE_N4_RESULTS/runs/medsam_ft__seed0/best.pt` **plus**
  `STAGE_N4_RESULTS/tables/val_summaries.json` for its val-fitted cut.

MedSAM's cut is **read**, never re-fitted: re-fitting it here would make Stage O's
copy a different operating point from the one Stage N4 reported, and the two
stages' numbers would stop being comparable for a reason nobody would look for.
""")

# ── §7 ───────────────────────────────────────────────────────────────────────
md("""
## §7 — Scoring the pool on validation

One forward pass per teacher over the 134 validation images, cached to CSV so the
gate can be re-read and argued about without paying for it again. Delete
`STAGE_O_RESULTS/tables/val_pool_matrix.csv` to force a recompute.

**What these absolute numbers mean.** Every teacher's cut was fitted on these same
134 images, so every val Dice below is mildly optimistic. The bias is shared, so
it cancels in the *gain* — which is what the gate reads — and does not cancel in
the per-teacher columns, which are context and never a result.

The pool is three teachers, not Stage M's four. `segformer_b2_teacher` is dropped:
it wins no group on validation, it had the lowest drop-one marginal in Stage M's
own gate (0.0121 against DeepLabV3+'s 0.0224), and it is the same MiT family as B5
— simultaneously the least useful and the most correlated member.
""")

code('''
from bruisekit import loaders as L
from bruisekit.registry import Registry

MAN = {s: pd.read_csv(env.manifests / f"{s}.csv") for s in ("train", "val", "test")}
for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
    overlap = set(MAN[a].subject) & set(MAN[b].subject)
    assert not overlap, f"{a}/{b} share subjects: {sorted(overlap)[:5]}"
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
CACHE = itakd.results_dir(env, "tables") / "val_pool_matrix.csv"

MATRIX = itakd.val_group_matrix(env, REG, CFG, MAN640, MAN["val"],
                                pool=itakd.POOL, n4_root=N4_RESULTS, cache=CACHE)
print(f"\\nmatrix: {len(MATRIX)} val images x {len(itakd.POOL)} teachers")
print(MATRIX[list(itakd.POOL)].mean().round(4).to_string())
''')

# ── §8 ───────────────────────────────────────────────────────────────────────
md("""
## §8 — Per-group Dice, and whether the ranking is real

Before the gate: the descriptive table the whole design rests on. Read the
`spread` column against §1's ~0.05 annotation-ceiling noise floor, and read
`n_subjects` beside every cell.
""")

code('''
for scheme in ("five", "light_vs_rest"):
    ident = itakd.identifiability(MATRIX, itakd.POOL, scheme, reps=REPS)
    print(f"\\n-- scheme={scheme!r} --")
    print(ident[["group", "n_images", "n_subjects", "best_teacher", "best_dice",
                 "runner_up", "margin", "spread", "p_argmax_stable",
                 "identifiable"]].to_string(index=False))
''')

# ── §9 ───────────────────────────────────────────────────────────────────────
md("""
## §9 — The gate. Pre-registered, validation only, written before anything trains.

```
open  iff  the weighting-gain CI clears zero
      AND  projected student gain > 0.01 Dice
      AND  at least one group's argmax is identifiable at p >= 0.75
```

The third clause is the new one and the reason this stage exists in this shape.
Stage M's gate opened on a real oracle gain of +0.048 and every contrast came back
inconclusive; Stage C's opened on +0.026 and delivered +0.007. Both gates were
measuring headroom that genuinely existed. Neither asked whether the **routing
key** was estimable.

Note also that the gate scores the **group-weighted** ensemble, not the per-image
oracle. Stage M's gate used the oracle, which no group weighting can reach — the
oracle is printed here as an upper bound so the distance between "what routing
could give" and "what grouping can give" is visible instead of implied.

**A closed gate is a result.** On the five-group scheme it is the expected one,
and it is the measured answer to *"why not just group by skin tone"* — a question
Stages C and M each answered by assertion.
""")

code('''
GATES = {}
schemes = [SCHEME] + (["five"] if ALSO_FIVE and SCHEME != "five" else [])
for scheme in schemes:
    res = itakd.ita_group_gate(MATRIX, pool=itakd.POOL, scheme=scheme, reps=REPS)
    GATES[scheme] = res
    print(itakd.format_gate(res))
    print()
    for p in itakd.save_gate(env, res):
        print(f"  wrote {p}")
    print()
''')

# ── §10 ──────────────────────────────────────────────────────────────────────
md("""
## §10 — Training, if the gate opened

Three shims, all on existing patch points, installed in this order:

1. **`install_group_shim`** wraps the *training* loader so each batch records its
   stems' ITA group indices. `engine.train_run` iterates `(x, y, _)` and throws
   the stem away, so this is how the loss learns which group each image is in
   without editing the shared training loop.
2. **`install_teacher_shim`** rebinds `engine.load_teacher` to return
   `[B, K, H, W]` — the whole pool — so the routing happens where the label
   already is. It handles MedSAM, which has no registry `Run`.
3. **`install_loss_shim`** rebinds `engine.DistillLoss`. Install this **last**: in
   a session that also trains Stage H or M, the dispatcher must sit on top of
   theirs.

**The coupling in step 1 is real and it is guarded rather than trusted.** The
loader, the teacher forward and the loss run synchronously in the main process
within one iteration, so the recorded group vector is always the current batch's.
`GroupRoutedDistillLoss.forward` **raises** if it is absent or its length does not
match the batch — because a silent fallback to uniform weights would turn this arm
into `p2_ensemble_uniform` with a gate, and it would report a plausible number for
a different experiment.

The micro-batch is pinned to each arm's control with accumulation making up the
difference, so the arm differs from its control in the teacher signal alone and
not also in effective batch, step count and LR schedule.
""")

code('''
if not RUN_TRAINING:
    print("RUN_TRAINING is False. Read the gate in §9 first, then flip it in §1.")
elif not GATES[SCHEME]["GATE_any"]:
    print(itakd.format_gate(GATES[SCHEME]))
    print("\\nThe gate is CLOSED. That is the result -- report it. Override only "
          "deliberately, and write down why.")
else:
    W = itakd.weight_array(pd.DataFrame(GATES[SCHEME]["weights"]),
                           itakd.POOL, SCHEME)
    print(f"weights [{W.shape[0]} groups x {W.shape[1]} teachers], rows in "
          f"{itakd.group_order(SCHEME)} order:\\n{np.round(W, 3)}\\n")

    itakd.install_group_shim(itakd.build_group_map(MAN640, SCHEME))
    itakd.install_teacher_shim(env, REG, CFG, MAN640, pool=itakd.POOL,
                               n4_root=N4_RESULTS)
    itakd.install_loss_shim(W)          # LAST

    RUNS = itakd.train_arms(env, REG, CFG, MAN640, itakd.results_dir(env, "runs"),
                            families=FAMILIES, seeds=SEEDS, max_micro=MAX_MICRO)
    print(f"\\ntrained: {RUNS}")
''')

md("""
### Read `group_loss_stats.json` before any Dice number

`images_per_group` is the first thing to check. An arm whose loss never saw one of
the groups did not run the experiment the gate authorised, and its Dice is
answering a different question.
""")

code('''
import json

for d in sorted(itakd.results_dir(env, "runs").glob("*/group_loss_stats.json")):
    s = json.loads(d.read_text())
    print(f"{d.parent.name}")
    print(f"   images/group {s['images_per_group']}   "
          f"coverage {s['mean_coverage']:.3f}   "
          f"fused teacher soft Dice {s['mean_fused_teacher_soft_dice']:.4f}")
    print(f"   weights {np.round(np.array(s['weights']), 3).tolist()}")
''')

# ── §11 ──────────────────────────────────────────────────────────────────────
md("""
## §11 — What this stage does and does not license

**Licensed by §4 (TODO 1).** "Complete miss" as published is `dice == 0` and is
the union of empty predictions and wrong-place errors. All three columns are now
on disk for every arm in the lineage, per ITA group and per size decile. Quote
three columns; lead with zero Dice because it is the union and therefore the
conservative number.

**Licensed by §5 (TODO 3).** Per-group and per-size behaviour for the Stage C
distillation arms, which the 24-model fairness export omitted. Read
`fairness__fairness_recall.csv`'s `all` vs `D1_D4` gaps together: where the gap
shrinks, the marginal skin-tone gap was partly a lesion-size effect.

**Not licensed.** None of §4 or §5 is a significance test. They are descriptive
tables over one seed's worth of per-image CSVs, and a best-minus-worst gap over
five noisy cells is biased upward by construction. For a test, use
`lesionsize.contrast_table` with a named, pre-registered contrast.

**On §7–§11.** If the gate closed, the reportable finding is the identifiability
table, not a failed method: *the per-group teacher ranking is not estimable on 134
validation images, so any arm routed on it is fitting sampling noise.* That is a
result about the study design, and it is the measured answer to a question Stages
C and M each settled by assertion.

If the gate opened and the arm still came back null, that is the **fourth**
consecutive KD null in this project (Stage C `p3_adaptive_group`, Stage H, Stage
M). Four nulls with one shared explanation — every model is at the annotation
ceiling — is a stronger paper section than any of them is alone.

**MedSAM is in this pool as its FEATURES, never as MedSAM.** `samprobe` keeps the
image encoder and discards the prompt encoder and mask decoder, because this
pipeline is automatic and has no prompt to give. Write it up that way.
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
