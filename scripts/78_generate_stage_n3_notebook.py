#!/usr/bin/env python
"""Emit `bruise_stage_n3.ipynb` -- Stage N3, the ceiling test on a third axis.

Same generator discipline as 70/71/77: the notebook is an output, never hand
edited, and ships with zero executed cells so no key, path or partial result can
travel inside it.

Everything the notebook writes goes to `STAGE_N3_RESULTS/`. It never touches
`results/`, `FINAL_RESULT/`, `FOUNDATION_RESULTS/`, `DERM_PROBE_RESULTS/` or
`_work/runs/`, so a failed experiment leaves no trace in the directories the
study's published numbers come from.
"""
from __future__ import annotations

import json
from pathlib import Path

DST = Path(__file__).resolve().parent.parent / "BRUISE_UNIFIED" / "bruise_stage_n3.ipynb"

CELLS: list[tuple[str, str]] = []


def md(src: str) -> None:
    CELLS.append(("markdown", src.strip("\n")))


def code(src: str) -> None:
    CELLS.append(("code", src.strip("\n")))


# ─────────────────────────────────────────────────────────────────────────────
md("""
# Stage N3 — does a different *pretraining paradigm* also land at the ceiling?

Every existing piece of ceiling evidence varies **capacity** or **architecture**:

| | Dice | | Dice |
|---|---|---|---|
| segformer_b0 (3.71 M) | 0.7663 | unet_r50 | 0.7570 |
| segformer_b2 (27.35 M) | 0.7692 | deeplabv3+_r50 | 0.7580 |
| segformer_b5 (~85 M) | 0.7727 | Friedman across the seven | p = 0.61 |

23× the parameters bought **+0.006 Dice**. What has never been varied is the
*pretraining paradigm* — every arm above is ImageNet-supervised or scratch.

Stage N2 showed DINOv2's features are genuinely different: **0.6635** frozen-probe
Dice against ResNet-50's **0.5252**. So if that same encoder, fine-tuned, *also*
lands at ~0.77, the ceiling is binding across capacity, architecture **and**
pretraining — and the bottleneck is the labels, not the model.

## Pre-registered before any number is produced

Fixed in `bruisekit/finetune_n3.py`, not in this notebook:

| outcome | reading |
|---|---|
| both arms in **0.75–0.78** | **Ceiling confirmed** on a third axis. Probe ranking did not transfer. Encoders exhausted — go to label quality. |
| either arm **> 0.79** | Ceiling **not** binding. Stage N2's null needs reinterpreting; encoders live again. |
| either arm **< 0.73** | **Inconclusive.** One seed cannot separate a weak encoder from a collapsed run (Stage Y seed 2). Check the loss curve first. |

**Read the miss column before the Dice column.** Dice is saturated; complete
misses are where this study's models actually separate. `ceiling_gate` refuses to
print a verdict without them.

## Two arms, one seed, ~4–5 GPU-hours

- `dinov2_ft` — natural images, self-supervised. The paradigm control.
- `dermlip_ft` — dermatology image-text. Answers §7h.9's open question: did the
  frozen-probe ranking survive fine-tuning?

Both at a **40×40 grid** so output resolution is not a variable, decoding to
160×160 — the same stride SegFormer produces at 640.

## Writes only to `STAGE_N3_RESULTS/`
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

`UNFREEZE_BLOCKS` and `SEEDS` are the pre-registered values. Changing them
changes the experiment — do it deliberately, and say so in the write-up.
""")

code('''
from pathlib import Path

BUNDLE     = None      # None = auto-detect
WORK       = None      # None = <bundle>/_work
EXTRA_RUNS = "/scratch/tbommawa/bruise_work/runs"

ARMS_TO_RUN     = ("dinov2_ft", "dermlip_ft")
SEEDS           = (0,)          # pre-registered: one seed, this is a screening run
UNFREEZE_BLOCKS = 6             # pre-registered
EPOCHS          = 100           # engine stops early on patience; this is the cap

# Fixed, never the VRAM probe. Six unfrozen ViT blocks at 640 is a different
# memory profile from anything the probe was calibrated on, and a probe that
# guesses high here dies mid-epoch after an hour.
N3_MICRO_BATCH = 2

SCORE_TEST = False     # the gate is VAL-only by design; every look at test costs

print(f"arms   : {ARMS_TO_RUN}")
print(f"seeds  : {SEEDS}   unfreeze: last {UNFREEZE_BLOCKS} blocks")
''')

# ── §2 ───────────────────────────────────────────────────────────────────────
md("""
## §2 — Environment, self-test, dependencies

`finetune_n3.self_test()` is structural: it needs no weights, no GPU and no
network, and it checks the two things that would silently fake a result — that
`find_blocks` raises rather than unfreezing nothing, and that
`build_param_groups` really does drop frozen parameters.
""")

code('''
import json
import warnings

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore", category=UserWarning)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)

from bruisekit import dermprobe as DP
from bruisekit import finetune_n3 as N3
from bruisekit import loaders as L
from bruisekit import paths as P

env = P.setup(root=BUNDLE, work=WORK, extra_runs=EXTRA_RUNS)
print(env.describe())

RESULTS = env.root / "STAGE_N3_RESULTS"
RUNS    = RESULTS / "runs"
TABLES  = RESULTS / "tables"
for d in (RESULTS, RUNS, TABLES):
    d.mkdir(parents=True, exist_ok=True)
print(f"\\nresults : {RESULTS}")
print("          nothing is written to results/, FINAL_RESULT/,")
print("          FOUNDATION_RESULTS/, DERM_PROBE_RESULTS/ or _work/runs/")

print("\\n-- dependency: open_clip --")
OPEN_CLIP_OK, msg = DP.ensure_open_clip(install=True)
print(msg)

print("\\n-- self test --")
assert N3.self_test(), "finetune_n3 self-test failed -- do not train on this"
''')

# ── §3 ───────────────────────────────────────────────────────────────────────
md("""
## §3 — Are the encoders present?

Both arms load from a local directory. An absent encoder stops the notebook here
with download instructions rather than forty minutes into a run.
""")

code('''
SRC = DP.report_sources(env)
need = [N3.ARMS[a]["source"] for a in ARMS_TO_RUN]
have = SRC.set_index("encoder")

missing = [s for s in need if not bool(have.loc[s, "present"])]
if missing:
    print(f"MISSING (and required): {missing}\\n")
    print(DP.download_instructions(env))
    raise SystemExit("download the encoders above, then re-run this cell")

for s in need:
    print(f"  {s:<12} present   {DP.SOURCES[s].init}")
''')

# ── §4 ───────────────────────────────────────────────────────────────────────
md("""
## §4 — Manifests, the training config, and a preflight

The config is `kd_core.DEFAULTS` — the recipe that produced segformer_b0's
0.7663, verbatim. Using it unmodified is what makes "did it land in the band?"
answerable: a bespoke recipe would leave the difference attributable to training
rather than to the encoder.

The preflight asserts every key `engine.train_run` reads. A missing one fails in
seconds here instead of after a model is built.
""")

code('''
from bruisekit.kd import kd_core

man = {s: pd.read_csv(env.manifests / f"{s}.csv") for s in ("train", "val", "test")}
print("640 cache")
man640 = L.build_cache640(env, man)

CFG = {
    **kd_core.DEFAULTS,
    "epochs": EPOCHS,
    "alpha": 0.5,              # unused: these arms are supervised, not distilled
    "aux_weight": 0.4,         # unused: ViT arms return aux=None
    "drive_sync_every": 5,
    "eval_batch": 8,
    "micro_batch": N3_MICRO_BATCH,
    "max_probe_batch": N3_MICRO_BATCH,   # pins resolve_micro_batch, see §1
}

REQUIRED = ("img_size", "amp", "workers", "backbone_lr", "head_lr", "weight_decay",
            "betas", "epochs", "warmup_fraction", "alpha", "aux_weight",
            "drive_sync_every", "patience")
missing = [k for k in REQUIRED if k not in CFG]
assert not missing, f"CFG is missing keys engine.train_run reads: {missing}"

print(f"\\nrecipe: backbone_lr={CFG['backbone_lr']}  head_lr={CFG['head_lr']}  "
      f"epochs={CFG['epochs']}  patience={CFG['patience']}  img={CFG['img_size']}")
print(f"train {len(man640['train'])} / val {len(man640['val'])} images")
''')

# ── §5 ───────────────────────────────────────────────────────────────────────
md("""
## §5 — Build each arm once and inspect it, before training anything

This is the cell that catches the failure that would fake a confirmed ceiling: an
arm that silently unfreezes **nothing** trains like a frozen probe and scores in
a way that looks exactly like the result we are hoping for.

Check `encoder trainable` is a real fraction — around 45 % for six of twelve
blocks — and not `0.0`.
""")

code('''
N3.install_n3_shim(env)

built = {}
for arm in ARMS_TO_RUN:
    print(f"\\n{'-' * 66}\\n{arm}  ({N3.ARMS[arm]['corpus']})\\n{'-' * 66}")
    m = N3.build_arm(env, arm, verbose=True)
    info = m.n3_info
    assert info["encoder_trainable_params"] > 0, f"{arm}: nothing unfrozen"
    built[arm] = info

    n_head = sum(p.numel() for p in m.decode_head.parameters())
    n_all = sum(p.numel() for p in m.parameters())
    print(f"    decoder {n_head:,}  total {n_all:,}")

    with torch.no_grad():
        y = m(torch.zeros(1, 3, CFG["img_size"], CFG["img_size"]))
    print(f"    forward: {tuple(y.shape)}")
    assert y.shape[-2:] == (CFG["img_size"], CFG["img_size"]), "logits must be full-res"
    del m
    if str(env.device).startswith("cuda"):
        torch.cuda.empty_cache()

pd.DataFrame(built).T.to_csv(TABLES / "arm_build_info.csv")
display(pd.DataFrame(built).T)
''')

# ── §6 ───────────────────────────────────────────────────────────────────────
md("""
## §6 — Train

Through `engine.train_run` unmodified, so the arms inherit the shared recipe, LR
split, early stopping and the whole resume contract (`DONE.json` to skip,
`resume.pt` every few epochs). Interrupting and re-running this cell resumes.

~2–2.5 GPU-hours per arm.
""")

code('''
run_ids = N3.train_arms(env, CFG, man640, RUNS,
                        arms=ARMS_TO_RUN, seeds=SEEDS, verbose=True)
print(f"\\nruns: {run_ids}")
''')

# ── §7 ───────────────────────────────────────────────────────────────────────
md("""
## §7 — Score on validation at each arm's own fitted operating point

Threshold is fitted **on val** and applied to val, matching how every other arm in
this study is scored. The gate is val-only by design (§7f.4): a decision taken on
test is a decision taken on the data the paper reports.
""")

code('''
from bruisekit.engine import make_loader
from bruisekit.evaluate import evaluate_at_cut
from bruisekit.foundation import fit_operating_point

val_tables, summaries = {}, {}
for arm in ARMS_TO_RUN:
    for seed in SEEDS:
        run_id = f"{arm}__seed{seed}"
        run_dir = RUNS / run_id
        model = N3.build_arm(env, arm, verbose=False).to(env.device)
        state = torch.load(str(run_dir / "best.pt"), map_location=env.device,
                           weights_only=False)
        model.load_state_dict(state["model"] if "model" in state else state)
        model.eval()

        cut = fit_operating_point(model, env, CFG, man640, run_dir, verbose=True)
        cut = float(cut["cut"] if isinstance(cut, dict) else cut)

        loader = make_loader(man640["val"], env.cache640, CFG["img_size"],
                             CFG["eval_batch"], training=False,
                             workers=CFG["workers"], seed=seed)
        df, summary = evaluate_at_cut(model, loader, env.device, cut, CFG["amp"])

        df["arm"], df["seed"], df["cut"] = arm, seed, cut
        val_tables[arm] = df
        summaries[run_id] = {**summary, "cut": cut}
        df.to_csv(TABLES / f"val_per_image__{run_id}.csv", index=False)
        print(f"  {run_id}: cut={cut:+.3f}  dice={df.dice.mean():.4f}  "
              f"misses={(df.dice == 0).sum()}")

        del model
        if str(env.device).startswith("cuda"):
            torch.cuda.empty_cache()

json.dump(summaries, open(TABLES / "val_summaries.json", "w"), indent=2, default=str)
''')

# ── §8 ───────────────────────────────────────────────────────────────────────
md("""
## §8 — The gate

Applies the pre-registered reading. Nothing here is a judgement call made after
seeing the numbers — the bands, the collapse guard and the transfer check were
all fixed in `finetune_n3.py` before the first run.
""")

code('''
gate = N3.ceiling_gate(val_tables, n_boot=10000, seed=0)
N3.print_gate(gate)

gate["table"].to_csv(TABLES / "ceiling_gate.csv", index=False)
json.dump({k: v for k, v in gate.items() if k != "table"},
          open(TABLES / "ceiling_gate.json", "w"), indent=2, default=str)
print(f"\\nwritten -> {TABLES}")
''')

# ── §9 ───────────────────────────────────────────────────────────────────────
md("""
## §9 — What this licenses, and what it does not

**If the ceiling is confirmed**, the claim you may make is:

> *A pretraining paradigm known to carry different information (Stage N2: 0.6635
> vs 0.5252 frozen) lands in the same 0.75–0.78 band as everything else. Encoder
> choice is exhausted as a lever on this task; the bottleneck is label quality.*

**What it does not license.** It does not say encoders never matter — it says they
do not matter *here, at this label noise level*. And it is one seed: an arm inside
the band is consistent with the ceiling, it does not prove a tight interval around
it.

**Where this points next.** More images labelled to the same standard do **not**
raise an annotation ceiling — the ceiling is set by label *noise*, not by data
volume. What moves it is multiple independent annotations per image, which is
exactly what the Fenwick set has: three annotators, and 128 images all three
labelled independently. That supports two things nothing else in this project can
do — measuring the ceiling directly as inter-rater Dice on our own data, and
training against consensus labels instead of one person's opinion.

So the honest chain is: *ceiling confirmed → the lever is label quality → Fenwick
is the only dataset with the structure to pull it.*
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
