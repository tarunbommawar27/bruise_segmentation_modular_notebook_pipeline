#!/usr/bin/env python
"""Emit `bruise_stage_n4.ipynb` -- Stage N4, mask-supervised foundation encoders.

Same generator discipline as 70/71/77/78: the notebook is an OUTPUT, never hand
edited, and ships with zero executed cells so no key, path or partial result can
travel inside it.

Everything the notebook writes goes to `STAGE_N4_RESULTS/`. It never touches
`results/`, `FINAL_RESULT/`, `FOUNDATION_RESULTS/`, `DERM_PROBE_RESULTS/`,
`STAGE_N3_RESULTS/` or `_work/runs/`, so a failed experiment leaves no trace in
the directories the study's published numbers come from. Stage N3's tables are
READ, for the reference rows and the cross-stage contrasts, and never rewritten.

ONE THING THIS NOTEBOOK DOES DIFFERENTLY FROM 78's, ON PURPOSE
---------------------------------------------------------------
`bruise_stage_n3.ipynb` calls `foundation.fit_operating_point(..., verbose=True)`,
and that function takes no `verbose` argument -- the call raises TypeError. The
Stage N3 run evidently got past it by hand. It is not fixed here (78 is that
notebook's source of truth and rewriting a shipped stage's cell from this script
would be worse), but it is NOT reproduced: §7 below calls it with the signature
it actually has.
"""
from __future__ import annotations

import json
from pathlib import Path

DST = Path(__file__).resolve().parent.parent / "BRUISE_UNIFIED" / "bruise_stage_n4.ipynb"

CELLS: list[tuple[str, str]] = []


def md(src: str) -> None:
    CELLS.append(("markdown", src.strip("\n")))


def code(src: str) -> None:
    CELLS.append(("code", src.strip("\n")))


# ─────────────────────────────────────────────────────────────────────────────
md("""
# Stage N4 — does **mask**-supervised pretraining beat **caption**-supervised pretraining?

Three medical encoders have now lost to DINOv2 in this project:

| | contrast | |
|---|---|---|
| frozen probe (§7h.7) | `dermlip − dinov2` | **−0.0913** [−0.1208, −0.0544] |
| fine-tuned (§7i.7) | `dermlip − dinov2` | **−0.0859** [−0.0425, +0.1281] |
| frozen probe (§7f.8) | `medsiglip − dinov2` | **−0.1660** [−0.2010, −0.1306] |

But every one of those medical arms is CLIP/SigLIP-style: **one pooled vector per
image, trained against one sentence**. A caption describes a picture globally and
never says which pixels are the lesion, so that objective is *rewarded* for
discarding spatial detail. DINOv2's objective is patch-level. §7h.9 item 2 already
names this as the likeliest mechanism behind the entire ranking.

If that is the mechanism, then *"medical pretraining does not help"* is the wrong
conclusion from those three rows. The right one is narrower: **"medical *caption*
pretraining does not help."**

## SAM and MedSAM separate those two readings

Both were pretrained with **dense mask supervision** — the objective this task
actually wants. They differ from each other in exactly one thing, the corpus:

| arm | corpus | licence |
|---|---|---|
| `sam_ft` | SA-1B — 11 M natural images, ~1.1 B masks | Apache-2.0 |
| `medsam_ft` | ~1.5 M medical image–mask pairs, ~10 modalities | Apache-2.0 |

So **`medsam − sam` holds the objective, architecture, capacity and recipe fixed
and moves only the corpus.** Nothing in Stages N, N2 or N3 managed that — there,
corpus and objective moved together.

## This is *not* a test of MedSAM as a segmentation model

SAM and MedSAM are **promptable**: give them a box and they segment what is
inside it. This stage keeps only the image encoder and throws the prompt encoder
and mask decoder away, because our pipeline is fully automatic and has no prompt
to give. *"MedSAM with a ground-truth box"* would score high and answer a
different question — it hands the model the answer's location, which is the hard
half of this task.

**The claim shape is "MedSAM's *features*", never "MedSAM".** Write it up that way.

## Pre-registered, in `bruisekit/samprobe.py`, before any number exists

| outcome | reading |
|---|---|
| `medsam − sam` clears zero **positive** | The medical **corpus** buys something once the objective is right. N/N2/N3 measured the objective, not the corpus. A foundation teacher becomes worth costing out. |
| `medsam − sam` **contains zero** | **Null**, and a strong one — the medical-pretraining question closed on a fourth axis. *Read the small-lesion table before calling it a null.* |
| `medsam − sam` clears zero **negative** | MedSAM is **worse**, same direction as DermLIP and MedSigLIP, now with the objective held fixed. |
| either arm **< 0.73** | **Inconclusive** — one seed cannot separate a weak encoder from a collapsed run (Stage Y seed 2). |

**Read the miss column before the Dice column.** Dice is saturated here (Friedman
p = 0.61 across the seven headline models); complete misses on small bruises are
the endpoint that has ever moved and the one a clinician cares about.
`mask_supervision_gate` refuses to print a verdict without them.

## Writes only to `STAGE_N4_RESULTS/`
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

`UNFREEZE_BLOCKS`, `TARGET_GRID`, `USE_NECK` and `SEEDS` are pre-registered in
`samprobe.py`. Changing them changes the experiment — do it deliberately, in the
module, and say so in the write-up.
""")

code('''
from pathlib import Path

BUNDLE     = None      # None = auto-detect
WORK       = None      # None = <bundle>/_work
EXTRA_RUNS = "/scratch/tbommawa/bruise_work/runs"

ARMS_TO_RUN = ("sam_ft", "medsam_ft")
SEEDS       = (0,)     # pre-registered: one seed, this is a screening run
EPOCHS      = 100      # engine stops early on patience; this is the cap

# Fixed, never the VRAM probe. Six unfrozen SAM blocks at 640 with WINDOWED
# attention is a different memory profile from anything the probe was calibrated
# on, and a probe that guesses high dies mid-epoch after an hour.
N4_MICRO_BATCH = 2

# The gate is VAL-only by design (§7f.4). The TEST pass in §9 is REPORTING, and
# the notebook's cell order enforces that: the verdict in §8 is already written
# to disk before anything looks at test. Fairness and small-lesion tables need
# test, because that is the split the rest of the study's fairness numbers use.
SCORE_TEST = True

# Stage N3's tables, read for the reference rows and the cross-stage contrasts.
# Never rewritten.
N3_DIRNAME = "STAGE_N3_RESULTS"

print(f"arms  : {ARMS_TO_RUN}")
print(f"seeds : {SEEDS}")
''')

# ── §2 ───────────────────────────────────────────────────────────────────────
md("""
## §2 — Environment and self-test

`samprobe.self_test()` is structural: no weights, no GPU, no network. It checks
the things that would silently fake a result — that the 2-D position-embedding
resample refuses a token-sequence tensor, that `find_sam_blocks` **raises**
rather than unfreezing nothing, and that the gate refuses a dict keyed by
`run_id` (which is how `STAGE_N3_RESULTS/ceiling_gate.json` ended up holding a
considered-looking verdict for a gate that never ran — §7i.7a).
""")

code('''
import json
import warnings

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore", category=UserWarning)
pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 80)

from bruisekit import loaders as L
from bruisekit import paths as P
from bruisekit import report as RP
from bruisekit import samprobe as N4

env = P.setup(root=BUNDLE, work=WORK, extra_runs=EXTRA_RUNS)
print(env.describe())

RESULTS = N4.results_dir(env)
RUNS    = RESULTS / "runs"
TABLES  = RESULTS / "tables"
for d in (RESULTS, RUNS, TABLES):
    d.mkdir(parents=True, exist_ok=True)
print(f"\\nresults : {RESULTS}")
print("          nothing is written to results/, FINAL_RESULT/,")
print("          FOUNDATION_RESULTS/, DERM_PROBE_RESULTS/, STAGE_N3_RESULTS/")
print("          or _work/runs/")

print("\\n-- self test --")
assert N4.self_test(), "samprobe self-test failed -- do not train on this"
''')

# ── §3 ───────────────────────────────────────────────────────────────────────
md("""
## §3 — Are both encoders on disk?

An absent encoder stops the notebook **here**, with copyable download commands,
rather than forty minutes into a run or — worse — training from random weights.
That failure has already cost this project one entire stage (§7f.7a).
""")

code('''
SRC = N4.report_sources(env)
display(SRC)

missing = SRC.loc[~SRC.present, "encoder"].tolist()
if missing:
    print(f"\\nMISSING (and required): {missing}\\n")
    print(N4.download_instructions(env))
    raise SystemExit("download the encoders above, then re-run this cell")

for _, r in SRC.iterrows():
    print(f"  {r.encoder:<8} present  {r.size_MB:>7.1f} MB  {r['path']}")
''')

# ── §4 ───────────────────────────────────────────────────────────────────────
md("""
## §4 — Manifests, the training config, and a preflight

The config is `kd_core.DEFAULTS` — the recipe that produced segformer_b0's 0.7663
and Stage N3's two arms, verbatim. Using it unmodified is what makes
`medsam − sam` readable: a bespoke recipe would leave any difference attributable
to training rather than to the corpus.
""")

code('''
from bruisekit.kd import kd_core

man = {s: pd.read_csv(env.manifests / f"{s}.csv") for s in ("train", "val", "test")}
META = {"val": man["val"], "test": man["test"]}

print("640 cache")
man640 = L.build_cache640(env, man)

CFG = {
    **kd_core.DEFAULTS,
    "epochs": EPOCHS,
    "alpha": 0.5,              # unused: these arms are supervised, not distilled
    "aux_weight": 0.4,         # unused: ViT arms return aux=None
    "drive_sync_every": 5,
    "eval_batch": 8,
    "micro_batch": N4_MICRO_BATCH,
    "max_probe_batch": N4_MICRO_BATCH,   # pins resolve_micro_batch, see §1
}

REQUIRED = ("img_size", "amp", "workers", "backbone_lr", "head_lr", "weight_decay",
            "betas", "epochs", "warmup_fraction", "alpha", "aux_weight",
            "drive_sync_every", "patience")
missing = [k for k in REQUIRED if k not in CFG]
assert not missing, f"CFG is missing keys engine.train_run reads: {missing}"

# SAM is patch-16 and this stage runs at grid 40, so 40 x 16 = 640 -- the
# pipeline's own img_size. Neither arm is ever resampled at the image level, and
# an img_size that drifted from that product would silently reintroduce the
# resolution confound §7i.2 removed.
assert CFG["img_size"] == N4.TARGET_GRID * 16, (
    f"img_size {CFG['img_size']} != TARGET_GRID*16 = {N4.TARGET_GRID * 16}. "
    f"Stage N4 assumes the encoder sees the pipeline's native 640.")

print(f"\\nrecipe: backbone_lr={CFG['backbone_lr']}  head_lr={CFG['head_lr']}  "
      f"epochs={CFG['epochs']}  patience={CFG['patience']}  img={CFG['img_size']}")
print(f"train {len(man640['train'])} / val {len(man640['val'])} images")
''')

# ── §5 ───────────────────────────────────────────────────────────────────────
md("""
## §5 — Build each arm once and inspect it, **before** training anything

This is the guard cell. Two failures would produce a plausible number and no
symptom:

1. **An arm that unfreezes nothing** trains like a frozen probe and scores
   low-to-mid — indistinguishable from a real result. Check
   `encoder_trainable_fraction` reads ≈ **0.48**, not `0.0`.
2. **A position embedding that did not resample** would leave the arm running at
   the wrong grid. `_features` raises on a grid mismatch rather than reshaping
   into a plausible-looking garbage map, and `pos_embed_native_grid` →
   `pos_embed_resampled_to` below should read **64 → 40**.

Expect roughly **43 M trainable encoder parameters + 0.95 M decoder**, which puts
these arms between `segformer_b2` (27.4 M) and `segformer_b5` (~85 M) and level
with Stage N3's 43.6 M.
""")

code('''
N4.install_n4_shim(env)

built = {}
for arm in ARMS_TO_RUN:
    print(f"\\n{'-' * 72}\\n{arm}  ({N4.ARMS[arm]['corpus']})\\n{'-' * 72}")
    m = N4.build_arm(env, arm, verbose=True)
    info = m.n4_info

    assert info["encoder_trainable_params"] > 0, f"{arm}: nothing unfrozen"
    assert info["encoder_trainable_fraction"] > 0.1, (
        f"{arm}: only {100*info['encoder_trainable_fraction']:.1f} % trainable -- "
        f"this arm is effectively a frozen probe and would fake a result")
    assert info["pos_embed_resampled_to"] == N4.TARGET_GRID

    n_head = sum(p.numel() for p in m.decode_head.parameters())
    n_all  = sum(p.numel() for p in m.parameters())
    n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"    decoder {n_head:,}   total {n_all:,}   trainable {n_train:,}")

    with torch.no_grad():
        y = m(torch.zeros(1, 3, CFG["img_size"], CFG["img_size"]))
    print(f"    forward: {tuple(y.shape)}")
    assert y.shape[-2:] == (CFG["img_size"], CFG["img_size"]), "logits must be full-res"

    built[arm] = {**info, "decoder_params": n_head, "total_params": n_all,
                  "trainable_params": n_train}
    del m
    if str(env.device).startswith("cuda"):
        torch.cuda.empty_cache()

INFO = pd.DataFrame(built).T
INFO.to_csv(TABLES / "arm_build_info.csv")
display(INFO)
''')

# ── §6 ───────────────────────────────────────────────────────────────────────
md("""
## §6 — Train

Through `engine.train_run` unmodified, so the arms inherit the shared recipe, LR
split, early stopping and the whole resume contract (`DONE.json` to skip,
`resume.pt` every few epochs). Interrupting and re-running this cell resumes.

**~2–2.5 GPU-hours per arm**, ~4–5 total. SAM's windowed attention at 640 is
somewhat heavier than DINOv2's dense attention at 560, so budget the upper end.
""")

code('''
run_ids = N4.train_arms(env, CFG, man640, RUNS,
                        arms=ARMS_TO_RUN, seeds=SEEDS, verbose=True)
print(f"\\nruns: {run_ids}")
''')

# ── §7 ───────────────────────────────────────────────────────────────────────
md("""
## §7 — Score on **validation** at each arm's own fitted operating point

Threshold is fitted on val and applied to val, matching how every other arm in
this study is scored. `select_cut` takes every cut within one standard error of
the peak as tied and breaks the tie on complete-miss rate — these sweeps are flat
plateaus and the bare argmax fits the val set's sampling error.

Tables are normalised through `report.normalize` with the manifest joined, so
`subject` and `skin_tone_category` are attached and `complete_miss` is recomputed
as `dice == 0` in exactly one place.
""")

code('''
from bruisekit.foundation import fit_operating_point, score_split

val_tables, summaries, CUTS = {}, {}, {}

for arm in ARMS_TO_RUN:
    for seed in SEEDS:
        run_id  = f"{arm}__seed{seed}"
        run_dir = RUNS / run_id

        model = N4.load_trained(env, arm, run_dir).to(env.device)

        # NOTE: fit_operating_point takes NO `verbose` argument. bruise_stage_n3's
        # §7 passes one and raises TypeError; do not copy that call.
        op  = fit_operating_point(model, env, CFG, man640, run_dir)
        cut = float(op["cut"] if isinstance(op, dict) else op)
        CUTS[arm] = cut

        df = score_split(model, env, CFG, man640, "val", cut, META["val"])
        df["arm"], df["seed"], df["cut"], df["split"] = arm, seed, cut, "val"
        val_tables[arm] = df
        summaries[run_id] = {
            "split": "val", "cut": cut, "n": int(len(df)),
            "mean_dice": float(df.dice.mean()),
            "median_dice": float(df.dice.median()),
            "misses": int((df.dice == 0).sum()),
            "mean_recall": float(df.recall.mean()),
        }
        df.to_csv(TABLES / f"val_per_image__{run_id}.csv", index=False)
        print(f"  {run_id}: cut={cut:+.3f}  dice={df.dice.mean():.4f}  "
              f"median={df.dice.median():.4f}  misses={(df.dice == 0).sum()}")

        del model
        if str(env.device).startswith("cuda"):
            torch.cuda.empty_cache()

json.dump(summaries, open(TABLES / "val_summaries.json", "w"), indent=2, default=str)
''')

# ── §8 ───────────────────────────────────────────────────────────────────────
md("""
## §8 — Load Stage N3's arms as reference rows

`dinov2_ft` and `dermlip_ft` are **read, never re-scored**. They were trained at
the same grid, with the same head and the same recipe, so their val tables drop
straight into the contrast list.

If Stage N3's tables are absent the cross-stage contrasts are simply skipped and
the primary `medsam − sam` contrast still runs — that one needs nothing but this
stage's own two arms.
""")

code('''
N3_DIR = env.root / N3_DIRNAME

def load_n3(split: str) -> dict:
    """Stage N3's per-image tables for `split`, normalised and keyed by ARM name.

    Globs both the directory root and tables/ because the two Stage N3 runs wrote
    to different places. Keyed by arm, NOT run_id -- the gate raises on a run_id
    key, deliberately (§7i.7a).
    """
    out = {}
    for d in (N3_DIR, N3_DIR / "tables"):
        if not d.exists():
            continue
        for p in sorted(d.glob(f"{split}_per_image__*.csv")):
            arm = p.stem.split("__")[1]
            if arm in out:
                continue
            out[arm] = RP.normalize(pd.read_csv(p), META[split])
    return out

n3_val = load_n3("val")
if n3_val:
    for a, t in n3_val.items():
        print(f"  N3 {a:<12} val  n={len(t):<4} dice={t.dice.mean():.4f}  "
              f"misses={(t.dice == 0).sum()}")
else:
    print(f"  no Stage N3 val tables under {N3_DIR} -- cross-stage contrasts "
          f"will be skipped")

GATE_TABLES = {**val_tables, **n3_val}
print(f"\\ninto the gate: {sorted(GATE_TABLES)}")
''')

# ── §9 ───────────────────────────────────────────────────────────────────────
md("""
## §9 — The gate

Applies the pre-registered reading. Nothing here is a judgement call made after
seeing the numbers: the bands, the collapse guard, the contrast list and the
Holm family were all fixed in `samprobe.py` before the first run.

**The verdict is written to disk in this cell, before §10 touches test.**
""")

code('''
gate = N4.mask_supervision_gate(GATE_TABLES, n_boot=10000, seed=0)
N4.print_gate(gate)

for p in N4.save_gate(env, gate):
    print(f"  written -> {p}")
''')

# ── §10 ──────────────────────────────────────────────────────────────────────
md("""
## §10 — Score on **test**, at the cut already fitted on val

The cut is **read, never re-fitted**. Re-sweeping on test would pick a better
threshold and report a meaningless number, because the threshold would have been
fitted on the data it is scored on.

This is a **reporting** pass, not a decision pass — §9's verdict is already on
disk. Test is needed because it is the split every other fairness and
lesion-size number in this study is computed on (185 images, 28 subjects, five
ITA groups).
""")

code('''
test_tables = {}

if SCORE_TEST:
    for arm in ARMS_TO_RUN:
        for seed in SEEDS:
            run_id  = f"{arm}__seed{seed}"
            run_dir = RUNS / run_id
            cut = CUTS[arm]

            model = N4.load_trained(env, arm, run_dir).to(env.device)
            df = score_split(model, env, CFG, man640, "test", cut, META["test"])
            df["arm"], df["seed"], df["cut"], df["split"] = arm, seed, cut, "test"
            test_tables[arm] = df

            summaries[run_id + "__test"] = {
                "split": "test", "cut": cut, "n": int(len(df)),
                "mean_dice": float(df.dice.mean()),
                "median_dice": float(df.dice.median()),
                "misses": int((df.dice == 0).sum()),
                "mean_recall": float(df.recall.mean()),
            }
            df.to_csv(TABLES / f"test_per_image__{run_id}.csv", index=False)
            print(f"  {run_id}: cut={cut:+.3f} (from val)  "
                  f"dice={df.dice.mean():.4f}  median={df.dice.median():.4f}  "
                  f"misses={(df.dice == 0).sum()}")

            del model
            if str(env.device).startswith("cuda"):
                torch.cuda.empty_cache()

    json.dump(summaries, open(TABLES / "test_summaries.json", "w"),
              indent=2, default=str)
else:
    print("SCORE_TEST is False -- skipping. §11 needs test tables and will skip too.")
''')

# ── §11 ──────────────────────────────────────────────────────────────────────
md("""
## §11 — Fairness and small-lesion recall

**This is the section to read first when the Dice table ties**, which on this
project's record is the likeliest outcome.

Three things happen here, in this order and for this reason:

1. **The confound is measured before any fairness number is quoted.** §8.4: lesion
   size is confounded with ITA group in this test set. `size_by_ita` reports what
   share of each group's images are small — if that varies across groups, every
   *unconditioned* fairness number in the study is confounded by exactly that much.
2. **Small-lesion performance**, on the `D1–D4` stratum (the four smallest GT-area
   deciles) and on `D1` alone. Note `wrong_place_n` = zero-Dice minus
   empty-prediction: the model output pixels and every one of them was wrong. The
   per-seed tables cannot see it and it is the failure a clinician cares about most.
3. **Per-ITA-group recall, marginally *and* within the small stratum.** Compare a
   model's best-minus-worst gap in `all` against the same gap in `D1_D4`: if it
   shrinks, the marginal gap was partly a size effect wearing a skin-tone label.

The contrast table is Holm-corrected **within the confirmatory family only**. The
`medsam − dermlip` row is exploratory and stays uncorrected and labelled — folding
it in would launder a post-hoc comparison into a confirmatory one.

Stage N3's arms are included so every row is binned by the **same deciles**.
`assign_bins` raises if two tables disagree about an image's GT area, which is
what catches tables taken from different test sets.
""")

code('''
if test_tables:
    n3_test = load_n3("test")
    FAIR_TABLES = {**test_tables, **n3_test}
    print(f"fairness over: {sorted(FAIR_TABLES)}\\n")

    fair = N4.fairness_report(FAIR_TABLES, n_boot=10000, seed=0, verbose=True)
    print()
    N4.print_fairness(fair)

    for p in N4.save_fairness(env, fair):
        print(f"  written -> {p}")
else:
    print("no test tables -- set SCORE_TEST = True in §1 and re-run §10")
''')

# ── §12 ──────────────────────────────────────────────────────────────────────
md("""
## §12 — What this licenses, and what it does not

**If `medsam − sam` is null**, the claim you may make is:

> *With the pretraining objective held fixed (both arms mask-supervised), the
> architecture matched (ViT-B/16), the grid matched (40×40 at 640 px) and the
> recipe identical, a 1.5 M-pair medical mask corpus buys nothing over natural
> images. Taken with Stage N2 (captions, frozen) and Stage N3 (captions,
> fine-tuned), medical pretraining has now failed to help on four independent
> axes.*

**If `sam − dinov2` is null too**, that is the stronger and more interesting
result: it says the *objective* did not matter either, and §7h.9 item 2's
explanation for the whole N/N2/N3 ranking is wrong. Say so.

### What it does not license

- **Not "MedSAM does not work."** We removed its prompt. This measures its
  features under automatic, prompt-free segmentation, which is not what it was
  built or evaluated for.
- **Not a licence claim.** Both arms are recorded as Apache-2.0 from their
  releases, not from a licence review. Confirm on the model cards before any
  commercialisation claim — this project has already been bitten by a card whose
  header and body disagreed.
- **Not a seed-robust result.** One seed. An arm inside the band is *consistent
  with* the ceiling, not a tight interval around it.
- **Not a distribution-matched test.** MedSAM's corpus is mostly radiology and
  dermoscopy; ours is clinical photography across skin tones. A loss here is
  partly a distribution gap and should be reported as such — the same gap that
  put MedSigLIP last at 0.4670.

### Where this points

If the small-lesion table shows **a miss-rate win under a Dice null**, that is the
result and it should lead the write-up. Dice is saturated at this label-noise
level; misses on small bruises are the only endpoint in this study that has ever
moved, and they are the one that matters clinically.

If everything is null, the chain from §7i.9 holds unchanged and gets one more
link: *encoders are exhausted → the lever is label quality → Fenwick is the only
dataset here with the structure to pull it.*
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
