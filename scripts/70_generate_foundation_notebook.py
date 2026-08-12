#!/usr/bin/env python
"""Emit `BRUISE_UNIFIED/bruise_foundation.ipynb` -- Stage N, runs top to bottom.

WHAT THIS STAGE ASKS, IN ONE SENTENCE
---------------------------------------
Does a medical foundation encoder beat an ImageNet encoder on bruise
segmentation, holding decoder, recipe, split and seeds fixed?

WHY IT IS A BASELINE STAGE AND NOT A DISTILLATION STAGE
---------------------------------------------------------
The proposal it was cut down from chained a foundation teacher into
reliability-gated + multi-teacher + ITA-fairness-aware KD. Three of those four
components have already returned null or negative in this project (Stage H's
gate fired and changed nothing; Stage C's ten arms were all non-inferior; Stage
M's six runs were inconclusive with misses worse on 6 of 6; Stage C's
`p3_adaptive_group` -- ITA-weighted KD -- scored 0.7586 against a 0.7680
reference). Stacking them means an outcome nobody can attribute. So this stage
asks the one question that stands on its own, and distillation is a later,
separate decision.

THE THREE THINGS THAT MAKE THE GATE WORTH RUNNING
---------------------------------------------------
  1. IT IS CHEAP.        ~1 GPU-hour against ~20 for the seg arms.
  2. IT CANNOT BE FAKED. A frozen encoder + linear head can only report what is
                         already in the features. A fine-tuned 400 M encoder on
                         95 training subjects will score well whatever its
                         pretraining was.
  3. IT HAS A CONTROL.   DINOv2 -- modern self-supervised ViT, zero medical data.
                         If it matches MedSigLIP then the win is "ViT", not
                         "medical", and the paper claim changes. The original
                         proposal had no arm that could detect this.

EVERY SETTING BELOW, AND WHERE IT CAME FROM
---------------------------------------------
  FOUNDATION_MICRO_BATCH = 4   a 400 M ViT with gradients at 448 does not fit
                               past 4 alongside a 40 GB MIG slice. 4 x 4 holds
                               the EFFECTIVE batch at 16, matching the SMP and
                               mobile baselines' step count and LR schedule.
                               GroupNorm decoder, so unlike Stage M's BatchNorm
                               note this costs nothing statistically.
  PROBE_EPOCHS = 15            the same short-run budget Stage C's alpha search
                               used. A linear head on frozen features converges
                               long before this; patience 5 stops it earlier.
  SEEDS = (0, 1, 2)            the study's convention. The probe runs ONE seed --
                               it is a gate, not a reported number.
  CUTS = 481                   identical to every other operating point in the
                               study, so `select_cut`'s tie-band rule behaves the
                               same way here as it does in Stage A.
  RESULTS_DIR                  FOUNDATION_RESULTS/ at the bundle root. Nothing is
                               written to results/, FINAL_RESULT/ or _work/runs/,
                               so an experiment that fails leaves no trace in the
                               directories the published numbers come from.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DST = ROOT / "BRUISE_UNIFIED" / "bruise_foundation.ipynb"

CELLS: list[tuple[str, str]] = []


def md(src: str) -> None:
    CELLS.append(("markdown", src.strip("\n")))


def code(src: str) -> None:
    CELLS.append(("code", src.strip("\n")))


# ─────────────────────────────────────────────────────────────────────────────
md("""
# Stage N — do medical foundation encoders beat ImageNet encoders?

**One question:** does the *pretraining corpus* matter for bruise segmentation?
Same decoder, same recipe, same split, same seeds — only the encoder's
pretraining changes.

This notebook is **not** a distillation stage, and that is deliberate. See §8.

---

### How it is meant to be run

| § | what | cost |
|---|---|---|
| §0–§4 | setup, weights check, cache, recipe | minutes |
| §5–§6 | train **3 frozen probes**, score on val | **~1 GPU-hour** |
| **§7** | **THE GATE** — a printed verdict | seconds |
| §8–§9 | how to read it, and the guard | — |
| §10–§13 | the seg arms, 3 seeds, test once, contrasts | ~20 GPU-hours |

**Stop at §7 if the gate closes.** A closed gate is a publishable result — *"a
400 M dermatology-pretrained encoder does not beat ImageNet features on this
task"* — and it saves the twenty hours.

### Where everything is written

`FOUNDATION_RESULTS/` at the bundle root. Nothing touches `results/`,
`FINAL_RESULT/` or `_work/runs/`.
""")

# ── §0 ───────────────────────────────────────────────────────────────────────
md("""
## §0 — Memory allocator

`PYTORCH_CUDA_ALLOC_CONF` is read once, when the CUDA allocator initialises. Set
after `import torch` it silently does nothing — so this cell asserts torch has
not been imported yet. A 400 M ViT plus activations fragments the arena enough
for this to matter.
""")

code("""
import os
import sys

assert "torch" not in sys.modules, (
    "torch is already imported -- PYTORCH_CUDA_ALLOC_CONF is read when the CUDA "
    "allocator initialises and setting it now has NO EFFECT. Restart the kernel "
    "and run this cell first.")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Nothing in this stage fetches anything at run time: every encoder loads with
# local_files_only=True. Making that explicit turns a mis-set path into an
# immediate error rather than a 40-minute stall on an offline compute node.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
print("allocator configured; torch not yet imported")
""")

# ── §1 ───────────────────────────────────────────────────────────────────────
md("""
## §1 — Configuration

**`EXTRA_RUNS` is the one that bites.** The registry searches `env.runs` first,
then `extra_runs`, then the bundle's shipped checkpoints — and falls through
*silently*. Point it at the real scratch tree or you may score a stale
checkpoint and never know.
""")

code('''
from pathlib import Path

# ── where things are ─────────────────────────────────────────────────────────
BUNDLE      = None          # None = auto-detect. Set explicitly if it guesses wrong.
WORK        = None          # None = <bundle>/_work
EXTRA_RUNS  = "/scratch/tbommawa/bruise_work/runs"   # the OTHER machine's runs

# ── what to run ──────────────────────────────────────────────────────────────
RUN_PROBES   = True         # §5-§6: the gate. Cheap. Always run this.
RUN_SEG      = True         # §10+: only proceeds if the gate opened (§9 enforces)
RUN_BIOMEDPARSE = False     # §2b: optional zero-shot, needs a manual install

SEEDS        = (0, 1, 2)    # seg arms
PROBE_SEED   = 0            # the gate is a decision, not a reported number
PROBE_EPOCHS = 15
PROBE_PATIENCE = 5

# ── the one recipe deviation, see §4 ─────────────────────────────────────────
FOUNDATION_MICRO_BATCH        = 4    # 400M ViT WITH gradients
FOUNDATION_PROBE_MICRO_BATCH  = 8    # frozen: no encoder grads, fits more
FOUNDATION_EFFECTIVE_BATCH    = 16   # == the SMP / mobile baselines

N_BOOT = 10000
print("configured")
''')

# ── §2 ───────────────────────────────────────────────────────────────────────
md("""
## §2 — Environment, and the self-test that runs before any GPU time

`foundation.self_test()` checks the two things that would silently invalidate
every number downstream:

1. **The interface** — every arm returns `(logits[B,1,640,640], None)` and
   `forward` matches `forward_train`.
2. **The freeze** — a frozen probe has *zero* trainable encoder parameters and a
   partially unfrozen arm has more than zero. Getting this backwards is
   invisible: the run completes, the loss falls, and the arm is mislabelled in
   every table.
""")

code('''
import json
import warnings

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore", category=UserWarning)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)

from bruisekit import paths as P
from bruisekit import foundation as FN

env = P.setup(root=BUNDLE, work=WORK, extra_runs=EXTRA_RUNS)
print(env.describe())

RESULTS = env.root / "FOUNDATION_RESULTS"
RUNS    = RESULTS / "runs"
RUNS.mkdir(parents=True, exist_ok=True)
print(f"\\nresults    : {RESULTS}")
print("             nothing is written to results/, FINAL_RESULT/ or _work/runs/")

print("\\n── self test ──")
assert FN.self_test(), "foundation.self_test FAILED -- do not spend GPU time"
''')

# ── §2a ──────────────────────────────────────────────────────────────────────
md("""
## §2a — Are the encoders on disk?

Nothing is downloaded at run time. If an encoder is missing this cell prints the
exact commands and **stops** — it does not fall back to random init, because on
697 images that is a large and completely invisible handicap.

`licence` is in the table on purpose. Handbook §7b.1 chose DeepLabV3+ over
SegFormer as the Stage F teacher specifically to escape NVIDIA's non-commercial
MiT licence; re-introducing a restricted encoder without recording it would
silently undo that.
""")

code('''
SRC = FN.report_sources(env)
display(SRC[["encoder", "repo", "kind", "present", "licence"]])

missing = SRC[~SRC.present].encoder.tolist()
if missing:
    print(f"\\nMISSING: {missing}\\n")
    print(FN.download_instructions(env))
    raise SystemExit("download the encoders above, then re-run this cell")

print("\\nall encoders present\\n")
for _, r in SRC.iterrows():
    print(f"  {r.encoder:<12} {r.init}")
''')

# ── §2b ──────────────────────────────────────────────────────────────────────
md("""
## §2b — *optional* — BiomedParse zero-shot

A **different, cheaper question** than the gate: not *"are these features
better"* but *"does a biomedical foundation model see a bruise at all when you
ask it to"*. No training, no threshold to fit — like native-argmax YOLO, the
absence of a fitted parameter is exactly why the number would be trustworthy.

It needs BiomedParse's own repo and dependencies. The cell **skips cleanly** if
they are absent; nothing below depends on it.
""")

code('''
BP = None
if RUN_BIOMEDPARSE:
    man_test_native = pd.read_csv(env.manifests / "test.csv")
    BP, note = FN.biomedparse_zeroshot(env, man_test_native, prompt="bruise")
    print(note)
    if BP is not None:
        print(f"\\n  mean Dice {BP.dice.mean():.4f}   median {BP.dice.median():.4f}"
              f"   dice==0 {(BP.dice == 0).sum()} / {len(BP)}")
        BP.to_csv(RESULTS / "biomedparse_zeroshot.csv", index=False)
        print("\\n  Whatever this number is, it is a finding: a zero-shot biomedical")
        print("  foundation model's Dice on bruises has not been reported anywhere.")
else:
    print("skipped (RUN_BIOMEDPARSE = False)")
''')

# ── §3 ───────────────────────────────────────────────────────────────────────
md("""
## §3 — Data, the split guard, and the 640 cache

The subject-grouped split is re-asserted here even though the build asserts it
too. A manifest is a text file and text files get edited; a subject leaking
across splits would inflate every number in this notebook invisibly.
""")

code('''
MAN = {s: pd.read_csv(env.manifests / f"{s}.csv") for s in ("train", "val", "test")}
for s, d in MAN.items():
    print(f"  {s:<6} {len(d):>4} images  {d.subject.nunique():>3} subjects")

subs = {s: set(d.subject) for s, d in MAN.items()}
for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
    shared = subs[a] & subs[b]
    assert not shared, f"SUBJECT LEAK {a}/{b}: {sorted(shared)[:5]}"
stems = pd.concat(MAN.values()).stem
assert stems.is_unique, "an image appears in more than one split"
print("\\n  subject-grouped split verified; no image appears twice")

from bruisekit import loaders as L
man640 = L.build_cache640(env, MAN)
META = MAN["test"]                     # subject + ITA, for report.normalize
META_VAL = MAN["val"]
''')

# ── §4 ───────────────────────────────────────────────────────────────────────
md("""
## §4 — The shared recipe, and the one deviation

Everything here is handbook §3, unchanged: 6e-5 backbone / 6e-4 head, AdamW,
poly decay, warmup 1 %, grad clip 1.0, model selection on **threshold-free
validation AP**.

**The deviation, stated up front.** A 400 M ViT with gradients at 448×448 does
not fit past micro-batch 4 alongside a 40 GB MIG slice. These arms train at
`4 × 4` accumulation, so the *effective* batch (16), the optimizer-step count and
the LR schedule are **identical** to the SMP and mobile baselines. The decoder
uses **GroupNorm**, which is per-sample — so unlike Stage M's BatchNorm note,
this costs nothing statistically. Put it in the limitations anyway.
""")

code('''
CFG = {
    # ── handbook §3, unchanged ───────────────────────────────────────────────
    "img_size": 640, "epochs": 100, "patience": 15,
    "backbone_lr": 6e-5, "head_lr": 6e-4,
    "betas": (0.9, 0.999), "weight_decay": 0.01,
    "warmup_fraction": 0.01, "poly_power": 1.0, "gradient_clip": 1.0,
    "amp": True, "aux_weight": 0.0,       # no auxiliary head on these decoders
    "alpha": 0.6,                          # unused: nothing here distils
    "workers": 4, "drive_sync_every": 5, "eval_batch": 8,

    # ── the engine's probe is bypassed for these arms; see §4 ────────────────
    "batch_mode": "matched", "effective_batch": 16, "max_probe_batch": 64,
    "vram_target": 0.75,
    "foundation_micro_batch": FOUNDATION_MICRO_BATCH,
    "foundation_probe_micro_batch": FOUNDATION_PROBE_MICRO_BATCH,
    "foundation_effective_batch": FOUNDATION_EFFECTIVE_BATCH,
}

CFG_PROBE = {**CFG, "epochs": PROBE_EPOCHS, "patience": PROBE_PATIENCE}

FN.install_foundation_shim(env)
PATHS = env.paths_for_models()
print("recipe installed; build_model now knows:", sorted(FN.FOUNDATION_ARCHS))
print(f"\\nprobe arms : {FN.PROBE_ARMS}")
print(f"seg arms   : {FN.SEG_ARMS}")
print(f"\\nGATE: {FN.GATE_TREATMENT} vs {FN.GATE_CONTROL}")
print(f"      attribution arm: {FN.GATE_ATTRIBUTION}  (reported, NOT ANDed in)")
''')

# ── §5 ───────────────────────────────────────────────────────────────────────
md("""
## §5 — Train the three frozen probes

Encoder **frozen**, head is a single 1×1 convolution. That restriction is the
whole design: the moment the head can learn spatial structure it stops measuring
the encoder and starts measuring the head, and a strong decoder papers over a
weak encoder well enough to make every arm tie — which reads as *"no difference
between pretraining corpora"* when it actually means *"the experiment could not
see one"*.

One seed. This is a decision procedure, not a reported number.

`train_run` is idempotent: `DONE.json` → skip, `resume.pt` → continue. Re-run the
cell after a dropped connection.
""")

code('''
from bruisekit.engine import train_run

probe_results = []
if RUN_PROBES:
    for arch in FN.PROBE_ARMS:
        run_id = f"{arch}__seed{PROBE_SEED}"
        print(f"\\n── {run_id} " + "─" * (58 - len(run_id)))
        spec = {"arch": arch, "size": None, "distill": False,
                **FN.FOUNDATION_ARCHS[arch]}
        r = train_run(run_id, spec, PROBE_SEED, CFG_PROBE, PATHS,
                      man640, env.cache640, RUNS, env.device)
        probe_results.append(r)
        print(f"  {r['status']}  best_val_ap={r.get('best_val_ap', float('nan')):.4f}")
    display(pd.DataFrame(probe_results))
else:
    print("skipped (RUN_PROBES = False)")
''')

# ── §6 ───────────────────────────────────────────────────────────────────────
md("""
## §6 — Score the probes on validation

Two numbers per arm, and they answer slightly different questions:

- **val AP** — threshold-free. No operating point anywhere in it, so it cannot be
  distorted by cut selection. This is the honest headline.
- **val Dice at each arm's own val-fitted cut** — comparable to every other
  number in the study, and the input to the paired bootstrap.

**The optimism, stated:** scoring on the same 134 images the cut was fitted on is
optimistic. It is *identically* optimistic for all three arms and the gate is a
**paired difference**, so it cancels. The val AP column is the check that it did.
""")

code('''
val_tables, val_rows = {}, []
if RUN_PROBES:
    from bruisekit.data import make_loader
    from bruisekit.engine import eval_ap

    amp = CFG["amp"] and str(env.device).startswith("cuda")
    val_loader = make_loader(man640["val"], env.cache640, CFG["img_size"],
                             CFG["eval_batch"], False, CFG["workers"], 0)

    for arch in FN.PROBE_ARMS:
        run_dir = RUNS / f"{arch}__seed{PROBE_SEED}"
        model = FN.build_foundation(env, arch, verbose=False)
        model.load_state_dict(torch.load(str(run_dir / "best.pt"),
                                         map_location="cpu", weights_only=True))
        model.to(env.device).eval()

        ap = eval_ap(model, val_loader, env.device, amp)
        op = FN.fit_operating_point(model, env, CFG, man640, run_dir)
        tab = FN.score_split(model, env, CFG, man640, "val", op["cut"], META_VAL)
        val_tables[arch] = tab

        n_par = sum(p.numel() for p in model.parameters())
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        val_rows.append({"arm": arch, "val_AP": ap, "cut": op["cut"],
                         "val_mean_dice": tab.dice.mean(),
                         "val_median_dice": tab.dice.median(),
                         "misses": int(tab.complete_miss.sum()),
                         "params_M": n_par / 1e6, "trainable_M": n_tr / 1e6})
        tab.to_csv(RESULTS / f"val_per_image__{arch}.csv", index=False)
        del model
        torch.cuda.empty_cache()

    VAL = pd.DataFrame(val_rows).sort_values("val_AP", ascending=False)
    display(VAL)
    VAL.to_csv(RESULTS / "probe_val_summary.csv", index=False)
else:
    print("skipped -- reloading tables from disk")
    for arch in FN.PROBE_ARMS:
        f = RESULTS / f"val_per_image__{arch}.csv"
        if f.exists():
            val_tables[arch] = pd.read_csv(f)
    print(f"  loaded {sorted(val_tables)}")
''')

# ── §7 ───────────────────────────────────────────────────────────────────────
md("""
## §7 — THE GATE

Fixed in `bruisekit/foundation.py` **before any number was produced**, so it
cannot be chosen after the fact:

```
open  iff  the (medsiglip_probe − resnet50_probe) val-Dice CI clears zero
```

Reported alongside and deliberately **not** ANDed in:

- **Attribution** — `medsiglip − dinov2`. DINOv2 is a modern self-supervised ViT
  with *zero* medical data. If it matches MedSigLIP, the gate can open and the
  *claim* still has to change: the win would be "modern ViT pretraining", not
  "dermatology pretraining". There is no honest way to fold that into one
  boolean.
- **Misses** — the endpoint this study is judged on.
""")

code('''
GATE = FN.probe_gate(val_tables, n_boot=N_BOOT)
FN.print_gate(GATE)
(RESULTS / "gate.json").write_text(json.dumps(GATE, indent=1))
print(f"\\nwritten -> {RESULTS / 'gate.json'}")
''')

# ── §8 ───────────────────────────────────────────────────────────────────────
md("""
## §8 — How to read the gate

**If it CLOSED** — that is a result, not a failure. Write it up:

> *A 400 M encoder pretrained on medical image-text pairs including dermatology
> does not produce better frozen features for bruise segmentation than a 26 M
> ImageNet ResNet-50, on 134 held-out validation images across 20 subjects.*

Stop here. You have saved ~20 GPU-hours and the null is publishable as-is.

**If it OPENED but `medical_share_of_gain` is below ~50 %** — the gain is generic
ViT pretraining, not medical data. Continue if you like, but the paper says
*"modern self-supervised vision encoders help"*, and DINOv2 is Apache-2.0 while
MedSigLIP carries Health-AI-Developer-Foundations use restrictions. On that
reading DINOv2 is the better engineering choice regardless.

**If it OPENED and the gain is attributable** — run §10.

---

### What §10 can and cannot show, before you spend the hours

Remember what the scaling curve in this project already says:

| model | params | mean Dice |
|---|---|---|
| segformer_b0_direct | 3.71 M | 0.7663 |
| segformer_b2_teacher | 27.35 M | 0.7692 |
| segformer_b5_teacher | ~85 M | 0.7727 |

**23× the parameters bought +0.006 Dice**, and human annotators disagree with
each other by more than that (§1: `gbarimah_vs_erik` 0.755, `paul_vs_erik`
0.581). A 400 M encoder landing at ~0.775 tells you nothing new about Dice.

So **§13 leads with complete misses and the IQR**, not mean Dice — the two
metrics that actually separated the field in Stage A while mean Dice did not.
""")

# ── §9 ───────────────────────────────────────────────────────────────────────
md("""
## §9 — The guard

Re-reads the JSON §7 wrote. `RUN_SEG = True` on its own authorises nothing —
this is the same shape as Stage M's §9, and it exists because the expensive half
of a gated experiment must not be runnable by flipping one flag at the top of a
notebook.
""")

code('''
gate_file = RESULTS / "gate.json"
if not gate_file.exists():
    raise RuntimeError("no gate.json -- run §7 before anything below it")
G = json.loads(gate_file.read_text())

PROCEED = bool(RUN_SEG and G["GATE_run_method"])
if not G["GATE_run_method"]:
    print("GATE CLOSED. §10+ will not run.")
    print(f"  delta {G['delta_dice']:+.4f}  CI {G['delta_dice_ci95']}")
    print("\\n  This is the result. Report it and stop -- see §8.")
    print("  To override deliberately (and say so in the writeup):")
    print("      PROCEED = True")
elif not RUN_SEG:
    print("gate OPEN but RUN_SEG = False -- §10+ skipped by request")
else:
    print(f"GATE OPEN  (delta {G['delta_dice']:+.4f}, CI {G['delta_dice_ci95']})")
    if "attribution" in G:
        s = G["attribution"]["medical_share_of_gain"]
        if np.isfinite(s) and s < 0.5:
            print(f"  ATTRIBUTION WARNING: only {s:.0%} of the gain is medical-specific.")
            print("  Proceed if you want, but do NOT claim dermatology pretraining.")
    print("\\n  §10 trains 3 arms x 3 seeds. ~20 GPU-hours. Resumable.")
print(f"\\nPROCEED = {PROCEED}")
''')

# ── §10 ──────────────────────────────────────────────────────────────────────
md("""
## §10 — Train the segmentation arms

Three arms × three seeds. Each unfreezes only its **last few blocks** (6 for the
ViTs, 2 stages for ResNet-50) — "the few layers" a 697-image, 95-subject training
set can support. Fully unfreezing 400 M parameters here fits the *subjects*, and
the frozen probe in §6 is what already told us whether the features were any good.

`resnet50_seg` is trained too, and it is not redundant: it is the arm that makes
the ViT numbers interpretable, under an identical decoder and recipe.
""")

code('''
seg_results = []
if PROCEED:
    for arch in FN.SEG_ARMS:
        for seed in SEEDS:
            run_id = f"{arch}__seed{seed}"
            print(f"\\n── {run_id} " + "─" * (58 - len(run_id)))
            spec = {"arch": arch, "size": None, "distill": False,
                    **FN.FOUNDATION_ARCHS[arch]}
            r = train_run(run_id, spec, seed, CFG, PATHS,
                          man640, env.cache640, RUNS, env.device)
            seg_results.append({"arm": arch, "seed": seed, **r})
    display(pd.DataFrame(seg_results))
else:
    print("skipped -- the gate closed or RUN_SEG is False")
''')

# ── §11 ──────────────────────────────────────────────────────────────────────
md("""
## §11 — Operating point on validation, then score test **once**

The cut is swept over 481 values on the 134 val images and applied once to test.
Not the argmax: `select_cut` takes every cut within one standard error of the
peak as statistically tied and breaks the tie on **complete-miss rate**, because
these sweeps are flat plateaus and the argmax fits the val set's sampling error.

Complete misses are `dice == 0` throughout — handbook §7.2a. The other definition
(`pred_positive_pixels == 0`) misses the case where a model fires confidently on
the wrong region, which is still a missed injury to a clinician.
""")

code('''
test_tables, test_rows = {}, []
if PROCEED:
    for arch in FN.SEG_ARMS:
        for seed in SEEDS:
            run_dir = RUNS / f"{arch}__seed{seed}"
            if not (run_dir / "best.pt").exists():
                print(f"  SKIP {arch}__seed{seed}: no best.pt")
                continue
            model = FN.build_foundation(env, arch, verbose=False)
            model.load_state_dict(torch.load(str(run_dir / "best.pt"),
                                             map_location="cpu", weights_only=True))
            model.to(env.device).eval()

            op = FN.fit_operating_point(model, env, CFG, man640, run_dir)
            tab = FN.score_split(model, env, CFG, man640, "test", op["cut"], META)
            tab.to_csv(run_dir / "test_per_image.csv", index=False)
            test_tables[f"{arch}__seed{seed}"] = tab

            q1, q3 = tab.dice.quantile([0.25, 0.75])
            test_rows.append({
                "arm": arch, "seed": seed, "cut": op["cut"], "n_images": len(tab),
                "mean_dice": tab.dice.mean(), "median_dice": tab.dice.median(),
                "iqr_dice": q3 - q1,
                "mean_iou": tab.iou.mean(),
                "mean_precision": tab.precision.mean(),
                "mean_recall": tab.recall.mean(),
                "misses": int(tab.complete_miss.sum()),
                "miss_rate": float(tab.complete_miss.mean()),
            })
            del model
            torch.cuda.empty_cache()

    TEST = pd.DataFrame(test_rows)
    display(TEST)
    TEST.to_csv(RESULTS / "foundation_test_per_seed.csv", index=False)
else:
    print("skipped")
''')

# ── §12 ──────────────────────────────────────────────────────────────────────
md("""
## §12 — Against the Stage B controls

The controls come from `FINAL_RESULT/RESULT_AUGUST_08/` — the current lineage,
every number scored in one session through one `report.normalize()` path.

`unet_r50` and `deeplabv3plus_r50` are the *right* comparison: same ImageNet
ResNet-50 encoder family, same recipe, same split. `segformer_b0_direct` is
included because it is the accuracy-per-parameter benchmark the whole study keeps
coming back to — 3.71 M parameters at 0.7663.

Verdicts are `significance.verdict` against the study's 0.01 Dice margin.
**INCONCLUSIVE** is a real outcome here and is not the same as NON-INFERIOR: an
interval of [−0.05, +0.02] is an underpowered comparison, not equivalence.
""")

code('''
from bruisekit.significance import paired_contrast_multi

CONTROL_DIR = env.root / "FINAL_RESULT" / "RESULT_AUGUST_08"
CONTROLS = ("unet_r50", "deeplabv3plus_r50", "segformer_b0_direct")

controls = {}
for c in CONTROLS:
    f = CONTROL_DIR / f"per_image_{c}.csv"
    if f.exists():
        controls[c] = pd.read_csv(f)
    else:
        print(f"  MISSING control (named, not dropped): {f}")

CON = pd.DataFrame()
if PROCEED and test_tables and controls:
    rows = []
    for arch in FN.SEG_ARMS:
        # Best seed selected on VALIDATION, never on test -- the val AP is in the
        # run's DONE.json, which is what model selection already used.
        best, best_ap = None, -np.inf
        for seed in SEEDS:
            d = RUNS / f"{arch}__seed{seed}" / "DONE.json"
            if d.exists():
                ap = json.loads(d.read_text()).get("best_val_ap", -np.inf)
                if ap > best_ap:
                    best, best_ap = seed, ap
        key = f"{arch}__seed{best}"
        if best is None or key not in test_tables:
            continue
        for cname, ctab in controls.items():
            r = paired_contrast_multi(test_tables[key], ctab, arch, cname,
                                      n_boot=N_BOOT)
            rows.append({"arm": arch, "val_selected_seed": best, **r})
    CON = pd.DataFrame(rows)
    display(CON[["arm", "b", "delta_dice", "lo", "hi", "p_two_sided", "verdict",
                 "delta_miss_rate", "p_a_fewer_misses"]])
    CON.to_csv(RESULTS / "foundation_contrasts.csv", index=False)
else:
    print("skipped")
''')

# ── §13 ──────────────────────────────────────────────────────────────────────
md("""
## §13 — Misses, spread, and fairness

**Lead with these, not with mean Dice.** In Stage A the IQR and the miss column
separated the field (YOLO carried ~1.5× the SegFormers' spread at a median within
0.02) while mean Dice did not.

Fairness is reported for completeness and with a warning attached: 20 of the 21
ITA tests in this study are non-significant, and the one that is has its worst
group at Tan (IV). A significant result here would be genuinely new — an
insignificant one is the expected outcome, not a disappointment.
""")

code('''
MISS = FAIR = pd.DataFrame()
if PROCEED and test_tables:
    from bruisekit.evaluate import fairness_analysis

    mrows, frows = [], []
    for name, tab in sorted(test_tables.items()):
        arch, seed = name.rsplit("__seed", 1)
        q1, q3 = tab.dice.quantile([0.25, 0.75])
        mrows.append({"model": arch, "seed": int(seed), "n": len(tab),
                      "misses_dice0": int(tab.complete_miss.sum()),
                      "miss_pct": 100 * float(tab.complete_miss.mean()),
                      "median_dice": tab.dice.median(), "iqr_dice": q3 - q1})
        frows.append({**fairness_analysis(tab, META, arch)["stats"], "seed": int(seed)})

    for cname, ctab in controls.items():
        q1, q3 = ctab.dice.quantile([0.25, 0.75])
        mrows.append({"model": f"{cname} (control)", "seed": -1, "n": len(ctab),
                      "misses_dice0": int(ctab.complete_miss.sum()),
                      "miss_pct": 100 * float(ctab.complete_miss.mean()),
                      "median_dice": ctab.dice.median(), "iqr_dice": q3 - q1})

    MISS = pd.DataFrame(mrows).sort_values(["model", "seed"])
    FAIR = pd.DataFrame(frows)
    display(MISS)
    display(FAIR[["model", "seed", "kruskal_p", "significant", "fairness_gap",
                  "best_group", "worst_group", "max_miss_rate_gap"]])
    MISS.to_csv(RESULTS / "foundation_misses.csv", index=False)
    FAIR.to_csv(RESULTS / "foundation_fairness.csv", index=False)

    if FAIR.significant.any():
        print("\\n  A significant ITA effect appeared. Check the WORST group before")
        print("  reading it as a dark-skin gap -- in this study the significant one")
        print("  had its worst group at Tan (IV), and YOLO's at Light (II-III).")
    else:
        print("\\n  No significant ITA effect -- consistent with the other 20 tests")
        print("  in this study. Report it as a finding, not as a null result.")
else:
    print("skipped")
''')

# ── §14 ──────────────────────────────────────────────────────────────────────
md("""
## §14 — What to write down

Fill these in from the cells above. Each is a sentence that stands on its own
whichever way the number went.

1. **The gate.** *"Frozen features from a 400 M medical encoder scored X against
   an ImageNet ResNet-50's Y on 134 validation images (Δ, CI)."*
2. **The attribution.** *"Of that gain, Z % survived comparison against DINOv2, a
   modern self-supervised ViT with no medical data."* — this is the sentence a
   reviewer will look for and the original plan had no arm that could produce it.
3. **The baseline.** *"Fine-tuning the last six blocks reached X on 185 test
   images against DeepLabV3+'s 0.7584 and U-Net's 0.7570."*
4. **Misses.** The number that decides. B2, B5 and B0-distilled are all at **0**.
5. **The limitation you owe.** The ViT arms see a 448-pixel image where SegFormer
   sees 640. That costs small-bruise detail — the exact population the
   complete-miss metric is about — and it is a property of the encoder's native
   resolution, not of a choice made here.
6. **The licence.** MedSigLIP is under Health AI Developer Foundations terms, not
   a standard open licence. DINOv2 is Apache-2.0. If the two tie, that decides it.
""")

code('''
print(f"── everything Stage N wrote, in {RESULTS} ──")
for f in sorted(RESULTS.rglob("*")):
    if f.is_file():
        print(f"  {f.relative_to(RESULTS)}")

print("\\n── the two lines to take to a meeting ──")
if (RESULTS / "gate.json").exists():
    G = json.loads((RESULTS / "gate.json").read_text())
    lo, hi = G["delta_dice_ci95"]
    print(f"  GATE  {G['treatment']} - {G['control']} = {G['delta_dice']:+.4f} "
          f"CI [{lo:+.4f}, {hi:+.4f}]  -> "
          f"{'OPEN' if G['GATE_run_method'] else 'CLOSED'}")
    if "attribution" in G:
        s = G["attribution"]["medical_share_of_gain"]
        print(f"  ATTRIBUTION  medical-specific share of the gain: {s:.1%}"
              if np.isfinite(s) else "  ATTRIBUTION  undefined (gain ~ 0)")
if len(CON):
    best = CON.sort_values("delta_dice", ascending=False).iloc[0]
    print(f"  BEST ARM  {best['arm']} vs {best['b']}: {best['delta_dice']:+.4f} "
          f"[{best['lo']:+.4f}, {best['hi']:+.4f}]  {best['verdict']}")
''')


def main() -> int:
    nb = {
        "cells": [
            {"cell_type": kind, "metadata": {},
             "source": src.splitlines(keepends=True),
             **({"execution_count": None, "outputs": []} if kind == "code" else {})}
            for kind, src in CELLS
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

    n_code = 0
    for i, (kind, src) in enumerate(CELLS):
        if kind == "code":
            compile(src, f"cell{i}", "exec")      # a cell that will not parse is not shippable
            n_code += 1
    back = json.loads(DST.read_text(encoding="utf-8"))
    assert len(back["cells"]) == len(CELLS)

    print(f"wrote {DST}")
    print(f"  {len(CELLS)} cells ({n_code} code, {len(CELLS) - n_code} markdown)")
    print(f"  {DST.stat().st_size / 1024:.1f} KB")
    print("  every code cell compiles; JSON round-trip ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
