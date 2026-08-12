#!/usr/bin/env python
"""Emit `BRUISE_UNIFIED/bruise_derm_probe.ipynb` -- Stage N2, runs top to bottom.

WHAT THIS STAGE ASKS, AND WHY STAGE N COULD NOT ANSWER IT
-----------------------------------------------------------
Stage N (handbook 7f) reported `medsiglip - dinov2 = -0.166` and read it as
"medical pretraining does not help". Two separate problems with quoting that:

  THE GRID CONFOUND (7f.9). The three arms scored in exactly the order of their
  feature-grid size: 37x37 -> 0.657, 28x28 -> 0.491, 20x20 -> 0.123. A linear
  probe is a 1x1 convolution on that grid, so a finer grid is worth Dice
  independently of what the encoder knows. Handbook 7f.9 says in as many words:
  "Do not write 7f.8 up until it has run."

  THE WRONG MODEL. MedSigLIP is a GENERAL-MEDICAL encoder -- radiology,
  histopathology, ophthalmology, with dermatology as one slice. Our images are
  consumer-camera photographs of skin. The claim "medical pretraining does not
  help on bruises" was never tested by it.

So this notebook is two experiments, in order:

  EXPERIMENT 1  GRID CALIBRATION. One encoder (ImageNet ResNet-50), one stage
                (layer3), four input sizes -> grids 20/28/37/40. Identical
                weights, identical depth. Whatever Dice moves IS the grid.
                Alongside: a pixel floor -- 1x1 conv on raw RGB pooled to the same
                grids, zero pretraining -- bounding what a grid buys with no
                features at all. 9 arms, ~40 GPU-minutes.

  EXPERIMENT 2  THE CORPUS ARMS, ALL AT 28x28. DermLIP (Derm1M: 1.03 M
                dermatology image-text pairs) and DermLIP-PanDerm (2 M skin
                images, self-supervised) against DINOv2, MedSigLIP, BiomedCLIP
                and ResNet-50 -- every one of them at the same grid, so the
                confound cannot recur. 6 arms, ~1.5 GPU-hours.

WHY NO SEG ARMS AND NO DISTILLATION
-------------------------------------
`foundation.py`'s docstring gives three reasons distillation is out of scope and
all three still hold (zero teacher deficit, transfer is the bottleneck, no
fairness target in the data). This stage inherits that rather than reopening it.
If the gate opens, the seg arms are Stage N's 10 onward with a new encoder key --
a separate decision, taken after.

EVERY SETTING BELOW, AND WHERE IT CAME FROM
---------------------------------------------
  TARGET_GRID = 28        MedSigLIP's native 448/16. Chosen so the largest and
                          most expensive encoder is the one arm needing NO
                          position-embedding resample, and every other arm moves
                          to meet it.
  PROBE_EPOCHS = 15       the short-run budget Stage C's alpha search used. A
                          single 1x1 conv on frozen features converges long
                          before this; patience 5 stops it earlier.
  PROBE_SEED = 0          one seed. This is a decision procedure, not a reported
                          number -- same as Stage N's gate.
  CUTS = 481              identical to every operating point in the study, so
                          `select_cut`'s tie-band rule behaves the same here.
  SCORE_TEST = False      the probes are val-only BY DESIGN. Handbook 7f.4: a
                          gate scored on test is a decision taken on the data the
                          paper reports. The flag exists; the default does not.
  RESULTS_DIR             DERM_PROBE_RESULTS/ at the bundle root. Nothing is
                          written to results/, FINAL_RESULT/, FOUNDATION_RESULTS/
                          or _work/runs/, so an experiment that fails leaves no
                          trace in the directories the published numbers come from
                          -- and Stage N's own results stay untouched next door.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DST = ROOT / "BRUISE_UNIFIED" / "bruise_derm_probe.ipynb"

CELLS: list[tuple[str, str]] = []


def md(src: str) -> None:
    CELLS.append(("markdown", src.strip("\n")))


def code(src: str) -> None:
    CELLS.append(("code", src.strip("\n")))


# ─────────────────────────────────────────────────────────────────────────────
md("""
# Stage N2 — close the grid confound, then test *dermatology* pretraining properly

Stage N asked *"does medical pretraining help?"* and answered with
**MedSigLIP**, a general-hospital encoder — chest X-ray, histopathology,
ophthalmology, dermatology as one slice. Our images are consumer-camera
photographs of skin. That question was never actually asked.

And its answer is not yet quotable anyway. Handbook §7f.9: the three arms scored
in **exactly** the order of their feature-grid size, and a linear probe is a 1×1
convolution *on that grid*.

| arm | grid | mean Dice |
|---|---|---|
| dinov2 | 37 × 37 | 0.6567 |
| medsiglip | 28 × 28 | 0.4907 |
| resnet50 | 20 × 20 | 0.1225 |

So this notebook runs two experiments, and the first one decides how the second
is read.

---

### How it is meant to be run

| § | what | cost |
|---|---|---|
| §0–§4 | setup, dependency + weights check, cache, recipe | minutes |
| §5–§6 | **Experiment 1** — 9 calibration probes, scored on val | ~40 GPU-min |
| **§7** | **THE GRID CURVE** — how much Dice the grid is worth | seconds |
| §8 | what that does to handbook §7f.8 | — |
| §9–§10 | **Experiment 2** — 6 corpus probes, all at 28×28 | ~1.5 GPU-hr |
| **§11** | **THE GATE** — a printed verdict | seconds |
| §12–§15 | how to read it, misses, what to write down | — |

**Neither half is expensive and both are decisive.** Total ≈ 2–3 GPU-hours
against Stage N's ~20 for seg arms — this whole notebook is a gate.

### Where everything is written

`DERM_PROBE_RESULTS/` at the bundle root. Nothing touches `results/`,
`FINAL_RESULT/`, `FOUNDATION_RESULTS/` or `_work/runs/` — Stage N's numbers are
next door and stay exactly as they are.

### One model you will ask about, and why it is not here

**`google/derm-foundation`** is by corpus the best-targeted model in existence
for this task — BiT-M ResNet-101×3 on teledermatology photographs. It ships as a
TensorFlow SavedModel whose only exported signature returns **one 6144-d vector
per image**. No token grid, nothing for a 1×1 convolution to sit on. It is
registered in `SOURCES` as an explicit gap and §2b prints the full reasoning.
""")

# ── §0 ───────────────────────────────────────────────────────────────────────
md("""
## §0 — Memory allocator

`PYTORCH_CUDA_ALLOC_CONF` is read once, when the CUDA allocator initialises. Set
after `import torch` it silently does nothing — so this cell asserts torch has
not been imported yet.
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
# Nothing here fetches at run time: every encoder loads from a local directory.
# Making that explicit turns a mis-set path into an immediate error instead of a
# forty-minute stall on an offline compute node.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
print("allocator configured; torch not yet imported")
""")

# ── §1 ───────────────────────────────────────────────────────────────────────
md("""
## §1 — Configuration

`EXTRA_RUNS` is the one that bites: the registry searches `env.runs`, then
`extra_runs`, then the bundle's shipped checkpoints, and falls through
*silently*. Nothing in this notebook loads a trained checkpoint from the
registry, but `paths.setup` still wants it pointed somewhere real.
""")

code('''
from pathlib import Path

# ── where things are ─────────────────────────────────────────────────────────
BUNDLE      = None          # None = auto-detect. Set explicitly if it guesses wrong.
WORK        = None          # None = <bundle>/_work
EXTRA_RUNS  = "/scratch/tbommawa/bruise_work/runs"

# ── what to run ──────────────────────────────────────────────────────────────
RUN_CALIBRATION = True      # §5-§7   Experiment 1. Cheap. Run this first, always.
RUN_CORPUS      = True      # §9-§11  Experiment 2. Needs the downloaded encoders.

# The probes are VAL-ONLY by design (handbook 7f.4: a gate scored on test is a
# decision taken on the data the paper reports). Turning this on scores the six
# corpus arms once on the 185 test images at each arm's own val-fitted cut. It is
# not needed for the gate and it is not free -- every look at test costs.
SCORE_TEST      = False

PROBE_SEED      = 0         # a decision procedure, not a reported number
PROBE_EPOCHS    = 15
PROBE_PATIENCE  = 5

# ── batch: fixed, never the VRAM probe. See §4. ──────────────────────────────
DERM_PROBE_MICRO_BATCH = 8
DERM_EFFECTIVE_BATCH   = 16

N_BOOT = 10000
print("configured")
''')

# ── §2 ───────────────────────────────────────────────────────────────────────
md("""
## §2 — Environment, dependency check, and the self-test

Three things happen before any GPU time, and each of them is a failure this
project has already paid for once:

1. **`open_clip` importable?** Three of the six corpus arms are unreachable
   without it. `ensure_open_clip(install=True)` fixes it in place if it is
   missing — but **only with `--no-deps`**, and it verifies in a fresh subprocess
   that `torch` and `torchvision` did not move.

   That guard is the whole point. `open_clip_torch` declares torch and
   torchvision as dependencies, and a plain `pip install` can drop different
   wheels into `~/.local/…/site-packages`, which precedes the system prefix on
   `sys.path` — so a fix for three probe arms silently re-versions torch for
   **every stage in this bundle**, and the symptom surfaces somewhere else
   entirely. If the versions change, it raises with the uninstall command instead
   of continuing.
2. **`dermprobe.self_test()`** — the resample, the interface, **the grid**, and
   the freeze. The grid check is the one specific to this stage: an arm that
   quietly runs at a different grid than it declares is worse than an arm that
   crashes, because this stage's *only* claim is a comparison at a fixed grid.
3. The parameter-count guard, which lives inside `build_arm` and fires per arm.
   On 2026-08-06 a DINOv2 directory opened through the wrong class produced a
   93.6 M body against a published 86.6 M, trained happily, and invalidated Stage
   N's entire attribution arm (§7f.7a). 8 % would have caught it.
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

from bruisekit import paths as P
from bruisekit import dermprobe as DP

env = P.setup(root=BUNDLE, work=WORK, extra_runs=EXTRA_RUNS)
print(env.describe())

RESULTS = env.root / "DERM_PROBE_RESULTS"
RUNS    = RESULTS / "runs"
RUNS.mkdir(parents=True, exist_ok=True)
print(f"\\nresults    : {RESULTS}")
print("             nothing is written to results/, FINAL_RESULT/,")
print("             FOUNDATION_RESULTS/ or _work/runs/")

print("\\n── dependency: open_clip ──")
# install=True installs ONLY with --no-deps and verifies in a fresh subprocess
# that torch and torchvision did not move. It raises rather than continue if they
# did: a ~/.local torch shadows the system one for EVERY notebook in this bundle.
# Set install=False to check without touching anything.
OPEN_CLIP_OK, msg = DP.ensure_open_clip(install=True)
print(msg)

print("\\n── self test ──")
assert DP.self_test(), "dermprobe.self_test FAILED -- do not spend GPU time"
''')

# ── §2a ──────────────────────────────────────────────────────────────────────
md("""
## §2a — Are the encoders on disk?

Nothing is downloaded at run time. If an encoder is missing this cell prints the
exact commands — it does **not** fall back to random init, because on 697 images
that is a large and completely invisible handicap.

**Both dermatology encoders are non-commercial** (CC-BY-NC-ND-4.0 and
CC-BY-NC-4.0). Handbook §7b.1 chose DeepLabV3+ over SegFormer as the Stage F
teacher specifically to escape a non-commercial licence; re-introducing one
without recording it would quietly undo that. The licence is in the table for
that reason, and **if a derm arm and DINOv2 tie, the licence decides** — DINOv2
is Apache-2.0.

Stage N2 shares `pretrained_weights/foundation/` with Stage N, so if you have
already run Stage N then `dinov2-base` and `medsiglip-448` are already there.
""")

code('''
SRC = DP.report_sources(env)
display(SRC[["encoder", "corpus", "loader", "kind", "probeable", "present", "licence"]])

# Only PROBEABLE encoders can block the run. derm_foundation is expected to be
# absent and is not a failure -- it is a documented gap (§2b).
need = SRC[SRC.probeable & ~SRC.present].encoder.tolist()
if need:
    print(f"\\nMISSING (and required): {need}\\n")
    print(DP.download_instructions(env))
    raise SystemExit("download the encoders above, then re-run this cell")

print("\\nall probeable encoders present\\n")
for _, r in SRC[SRC.probeable].iterrows():
    print(f"  {r.encoder:<18} {DP.SOURCES[r.encoder].init}")
''')

# ── §2b ──────────────────────────────────────────────────────────────────────
md("""
## §2b — Why `google/derm-foundation` is not an arm

The single most likely question about this notebook, answered from the module so
the reason travels with the code rather than living in a chat log.
""")

code('''
print(DP.dermfoundation_tiled_note())
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
META      = MAN["test"]                # subject + ITA, for report.normalize
META_VAL  = MAN["val"]
''')

# ── §4 ───────────────────────────────────────────────────────────────────────
md("""
## §4 — The recipe, and the two deviations

Handbook §3, unchanged: 6e-5 backbone / 6e-4 head, AdamW, poly decay, warmup 1 %,
grad clip 1.0, model selection on **threshold-free validation AP**.

**Deviation 1 — batch.** Every arm here is frozen with a single 1×1 convolution
trainable, so the engine's VRAM probe would spend minutes escalating from batch 1
to rediscover a known answer. `micro × accum` is pinned to 16 so the *effective*
batch, the optimizer-step count and the LR schedule match the SMP and mobile
baselines exactly.

**Deviation 2 — position embeddings are resampled.** Stage N took the opposite
decision (§7f.6: run every encoder at its native resolution, never interpolate) —
and that decision is precisely what produced the confound in §7f.9, because the
native resolutions differ. You cannot have both *"every encoder at its native
grid"* and *"every encoder at the same grid"*. This stage picks the second: the
question is about the **corpus** and the grid is the nuisance variable.
Resampling is applied **identically to every ViT arm**, so it is a property of
the whole experiment rather than a difference between arms. It goes in the
limitations either way.
""")

code('''
CFG = {
    # ── handbook 3, unchanged ────────────────────────────────────────────────
    "img_size": 640, "epochs": PROBE_EPOCHS, "patience": PROBE_PATIENCE,
    "backbone_lr": 6e-5, "head_lr": 6e-4,
    "betas": (0.9, 0.999), "weight_decay": 0.01,
    "warmup_fraction": 0.01, "poly_power": 1.0, "gradient_clip": 1.0,
    "amp": True, "aux_weight": 0.0,        # no auxiliary head on a linear probe
    "alpha": 0.6,                           # unused: nothing here distils
    "workers": 4, "drive_sync_every": 5, "eval_batch": 8,

    # ── the engine's VRAM probe is bypassed for these arms; see 4 ────────────
    "batch_mode": "matched", "effective_batch": 16, "max_probe_batch": 64,
    "vram_target": 0.75,
    "derm_probe_micro_batch": DERM_PROBE_MICRO_BATCH,
    "derm_effective_batch": DERM_EFFECTIVE_BATCH,
}

DP.install_derm_shim(env)
PATHS = env.paths_for_models()

print(f"target grid for every corpus arm : {DP.TARGET_GRID}x{DP.TARGET_GRID}")
print(f"\\ncalibration arms ({len(DP.CALIBRATION_ARMS)}):")
for a in DP.CALIBRATION_ARMS:
    print(f"    {a:<16} grid {DP.arm_grid(a)}x{DP.arm_grid(a)}")
print(f"\\ncorpus arms ({len(DP.CORPUS_ARMS)}):")
for a in DP.CORPUS_ARMS:
    print(f"    {a:<24} grid {DP.arm_grid(a)}x{DP.arm_grid(a)}")
print(f"\\nPRE-REGISTERED GATE: {DP.GATE_TREATMENT} vs {DP.GATE_GENERIC}")
print(f"  reported alongside: vs {DP.GATE_MEDICAL}, vs {DP.GATE_BIOMED}, "
      f"vs {DP.GATE_IMAGENET}")
print(f"  secondary (never promoted): {DP.GATE_SECONDARY}")
''')

# ── §5 ───────────────────────────────────────────────────────────────────────
md("""
## §5 — EXPERIMENT 1: train the 9 calibration probes

**Five ResNet arms.** Identical ImageNet ResNet-50 weights, identical depth
(`layer3`), four input sizes → grids 20 / 28 / 37 / 40. Nothing changes but the
grid, so whatever Dice moves *is* the grid. The fifth, `rn50_l4_g20`, reproduces
Stage N's own ResNet arm — same input, one stage deeper, stride 32 instead of 16
— which makes the `l4_g20` vs `l3_g40` pair the single cleanest statement of the
confound available: same weights, same 640 input, **only the stride differs**.

**Four pixel arms.** A 1×1 convolution on raw RGB average-pooled to the same four
grids. No encoder, no pretraining, three channels. Whatever this reaches is
reachable with *zero* learned features — it is the floor that makes the ResNet
slope interpretable instead of merely suggestive.

`train_run` is idempotent: `DONE.json` → skip, `resume.pt` → continue. Re-run the
cell after a dropped connection.
""")

code('''
from bruisekit.engine import train_run

cal_results = []
if RUN_CALIBRATION:
    for arch in DP.CALIBRATION_ARMS:
        run_id = f"{arch}__seed{PROBE_SEED}"
        print(f"\\n── {run_id}  (grid {DP.arm_grid(arch)}) " + "─" * 34)
        spec = {"arch": arch, "size": None, "distill": False}
        r = train_run(run_id, spec, PROBE_SEED, CFG, PATHS,
                      man640, env.cache640, RUNS, env.device)
        cal_results.append({"arm": arch, "grid": DP.arm_grid(arch), **r})
        print(f"  {r['status']}  best_val_ap={r.get('best_val_ap', float('nan')):.4f}")
    display(pd.DataFrame(cal_results))
else:
    print("skipped (RUN_CALIBRATION = False)")
''')

# ── §6 ───────────────────────────────────────────────────────────────────────
md("""
## §6 — Score the calibration probes on validation

Two numbers per arm, answering slightly different questions:

- **val AP** — threshold-free. No operating point in it at all, so it cannot be
  distorted by cut selection.
- **val Dice at each arm's own val-fitted cut** — comparable to every other
  number in the study, and the input to the bootstrap in §7.

**The optimism, stated:** scoring on the same 134 images the cut was fitted on is
optimistic. It is *identically* optimistic for every arm and §7 fits a slope
across arms, so a common offset cancels. The val AP column is the check that it
did.
""")

code('''
cal_tables, cal_rows = {}, []
if RUN_CALIBRATION:
    from bruisekit import foundation as FN     # fit_operating_point / score_split
    from bruisekit.data import make_loader
    from bruisekit.engine import eval_ap

    amp = CFG["amp"] and str(env.device).startswith("cuda")
    val_loader = make_loader(man640["val"], env.cache640, CFG["img_size"],
                             CFG["eval_batch"], False, CFG["workers"], 0)

    for arch in DP.CALIBRATION_ARMS:
        run_dir = RUNS / f"{arch}__seed{PROBE_SEED}"
        if not (run_dir / "best.pt").exists():
            print(f"  SKIP {arch}: no best.pt")
            continue
        model = DP.build_arm(env, arch, verbose=False)
        model.load_state_dict(torch.load(str(run_dir / "best.pt"),
                                         map_location="cpu", weights_only=True))
        model.to(env.device).eval()

        ap  = eval_ap(model, val_loader, env.device, amp)
        op  = FN.fit_operating_point(model, env, CFG, man640, run_dir)
        tab = FN.score_split(model, env, CFG, man640, "val", op["cut"], META_VAL)
        cal_tables[arch] = tab

        cal_rows.append({"arm": arch, "grid": DP.arm_grid(arch),
                         "input": model.enc_size, "embed": model.embed_dim,
                         "val_AP": ap, "cut": op["cut"],
                         "val_mean_dice": tab.dice.mean(),
                         "val_median_dice": tab.dice.median(),
                         "misses": int(tab.complete_miss.sum())})
        tab.to_csv(RESULTS / f"val_per_image__{arch}.csv", index=False)
        del model
        if str(env.device).startswith("cuda"):
            torch.cuda.empty_cache()

    # An empty table is a state to REPORT, not to sort -- `sort_values` on a frame
    # with no columns raises KeyError and buries the real cause, which is always
    # that §5 did not train anything.
    if cal_rows:
        CAL = pd.DataFrame(cal_rows).sort_values(["arm"])
        display(CAL)
        CAL.to_csv(RESULTS / "calibration_val_summary.csv", index=False)
    else:
        CAL = pd.DataFrame()
        print("\\n  NOTHING TO SCORE -- no calibration arm has a best.pt. Run §5.")
else:
    print("skipped -- reloading tables from disk")
    for arch in DP.CALIBRATION_ARMS:
        f = RESULTS / f"val_per_image__{arch}.csv"
        if f.exists():
            cal_tables[arch] = pd.read_csv(f)
    print(f"  loaded {sorted(cal_tables)}")
''')

# ── §7 ───────────────────────────────────────────────────────────────────────
md("""
## §7 — THE GRID CURVE

Dice against `log2(grid)`, fitted separately for the two families, bootstrapped
over **subjects** — images from one subject are not independent and an
image-level resample would give intervals too narrow by roughly √(images per
subject).

The number this exists to produce is **how much of handbook §7f.8's
`dinov2 − resnet50 = +0.534` was the grid**, measured on an encoder whose
pretraining never changed.
""")

code('''
CALIB = {}
if cal_tables:
    CALIB = DP.grid_calibration(cal_tables, n_boot=N_BOOT)
    DP.print_calibration(CALIB)
    (RESULTS / "grid_calibration.json").write_text(json.dumps(CALIB, indent=1))
    print(f"\\nwritten -> {RESULTS / 'grid_calibration.json'}")
else:
    print("no calibration tables -- run §5-§6 first")
''')

# ── §8 ───────────────────────────────────────────────────────────────────────
md("""
## §8 — How to read the grid curve

**If the ResNet slope is flat** (interval covering zero) — the grid buys nothing
on this task, §7f.9's worry is retired, and §7f.8 stands as written. Say so
explicitly; a closed confound is a result and it is the cheap outcome.

**If the ResNet slope is steep** — then the ordering in §7f.8 was substantially a
resolution ranking wearing a pretraining label, and **§7f.8 must be rewritten
before it is quoted anywhere**. It does not become wrong, it becomes
uninterpretable, which is different and worse to publish.

**Read the pixel row next to it.** If the pixel floor moves nearly as much as the
ResNet arms do, then most of the grid effect is *geometry* — how finely a
constant-per-cell mask can trace a boundary — and none of it is about features at
all. If the pixel floor is flat while ResNet climbs, the grid is helping the
*encoder* express what it knows, which is a different and more interesting
statement.

**The one pair to quote either way** is `rn50_l4_g20` vs `rn50_l3_g40`: same
weights, same 640 input, only the stride changes. Nothing about pretraining,
corpus, capacity or resolution-of-the-input differs. Whatever separates them is
the grid, full stop.

---

Note what §9 onward does **not** need from this. The corpus arms are all at 28×28
by construction, so no correction is applied to them and none is needed. §7
exists to re-read the *old* numbers honestly, not to patch the new ones.
""")

# ── §9 ───────────────────────────────────────────────────────────────────────
md("""
## §9 — EXPERIMENT 2: train the 6 corpus probes, all at 28 × 28

| arm | encoder | corpus | licence |
|---|---|---|---|
| `dermlip_g28` | ViT-B/16 | **Derm1M** — 1.03 M dermatology image-text pairs, 403 k images | CC-BY-NC-ND-4.0 |
| `dermlip_panderm_g28` | ViT-B/16 | **PanDerm** — 2 M skin images self-supervised, then Derm1M | CC-BY-NC-4.0 |
| `biomedclip_g28` | ViT-B/16 | PMC-15M — 15 M biomedical *figure* captions | MIT |
| `dinov2_g28` | ViT-B/14 | LVD-142M natural images, self-supervised | Apache-2.0 |
| `medsiglip_g28` | SoViT-400m | general medical image-text | HAI-DEF (restricted) |
| `rn50_g28` | ResNet-50 | ImageNet-1k supervised | BSD-3 |

**Three of these are ViT-B/16 at 224 native, loaded by the same loader and
resampled the same way.** So `dermlip − biomedclip` varies the *corpus* and
nothing else: skin photographs against journal figures, at identical capacity,
identical patch size, identical grid. That is a cleaner contrast than anything
Stage N could form, where every arm differed in size, patch, native resolution
and vendor at once.

MedSigLIP is 5× the parameters of everything else here. If capacity mattered on
this task it would show up as that arm winning — and Stage N already says it does
not.
""")

md("""
**One arm is allowed to fail, and exactly one kind of failure is tolerated.**
`dermlip_panderm_g28`'s vision tower is PanDerm, and its model card says to
install the Derm1M package before loading — so plain `open_clip` may not be able
to construct it. It is *secondary*, so the cell records the failure loudly and
continues.

Any arm in the **primary contrast** (`dermlip_g28`, `dinov2_g28`) failing
**re-raises**. A gate whose treatment or control quietly went missing is not a
gate, and silently dropping one is how §7f.7a's random-weights arm survived to
produce a number.
""")

code('''
# Re-checked HERE rather than trusting §2's OPEN_CLIP_OK. If the package was
# installed after §2 ran, that variable is a stale False and every open_clip arm
# would be skipped for a reason that is no longer true.
OPEN_CLIP_OK, _msg = DP.check_open_clip()
print(_msg)

# Re-initialised on every run of this cell, which is what clears a stale failure
# from an earlier attempt. §10 skips any arm listed here, so a leftover entry
# would keep an arm out of the table long after the cause was fixed.
corpus_results, corpus_failed = [], {}
CRITICAL = {DP.GATE_TREATMENT, DP.GATE_GENERIC}

if RUN_CORPUS:
    for arch in DP.CORPUS_ARMS:
        if DP.CORPUS_ARCHS[arch]["kind"] == "openclip" and not OPEN_CLIP_OK:
            corpus_failed[arch] = "open_clip not importable (see §2)"
            print(f"  SKIP {arch}: open_clip not importable")
            if arch in CRITICAL:
                raise RuntimeError(
                    f"{arch} is the pre-registered "
                    f"{'treatment' if arch == DP.GATE_TREATMENT else 'control'} arm "
                    f"of the gate and cannot be built. Re-run §2 with "
                    f"install=True, or install by hand:\\n"
                    f"    {sys.executable} -m pip install --user --no-deps "
                    f"open_clip_torch ftfy regex timm safetensors wcwidth\\n"
                    f"  Do not skip -- a gate missing its treatment is not a gate.")
            continue
        run_id = f"{arch}__seed{PROBE_SEED}"
        print(f"\\n── {run_id} " + "─" * (56 - len(run_id)))
        spec = {"arch": arch, "size": None, "distill": False}
        try:
            r = train_run(run_id, spec, PROBE_SEED, CFG, PATHS,
                          man640, env.cache640, RUNS, env.device)
        except Exception as e:
            corpus_failed[arch] = f"{type(e).__name__}: {e}"
            print(f"\\n  !! {arch} FAILED TO BUILD OR TRAIN")
            print(f"     {type(e).__name__}: {e}")
            if arch in CRITICAL:
                raise
            print(f"     {arch} is SECONDARY -- recorded and skipped. The gate's")
            print(f"     primary contrast is unaffected. Do NOT substitute another")
            print(f"     checkpoint for it.")
            continue
        corpus_results.append({"arm": arch, **r})
        print(f"  {r['status']}  best_val_ap={r.get('best_val_ap', float('nan')):.4f}")

    display(pd.DataFrame(corpus_results))
    if corpus_failed:
        print("\\n  ARMS THAT DID NOT RUN -- report these, do not omit them:")
        for a, why in corpus_failed.items():
            print(f"    {a:<24} {why[:110]}")
        (RESULTS / "corpus_failed_arms.json").write_text(json.dumps(corpus_failed, indent=1))
else:
    print("skipped (RUN_CORPUS = False)")
''')

# ── §10 ──────────────────────────────────────────────────────────────────────
md("""
## §10 — Score the corpus probes on validation

Same protocol as §6, and `build_arm` re-asserts the parameter count and the grid
on every reload. A model whose weights did not land, or whose body was built by
the wrong class, fails here rather than forty minutes later as a
plausible-looking Dice number.
""")

code('''
corpus_tables, corpus_rows = {}, []
if RUN_CORPUS:
    from bruisekit import foundation as FN
    from bruisekit.data import make_loader
    from bruisekit.engine import eval_ap

    amp = CFG["amp"] and str(env.device).startswith("cuda")
    val_loader = make_loader(man640["val"], env.cache640, CFG["img_size"],
                             CFG["eval_batch"], False, CFG["workers"], 0)

    for arch in DP.CORPUS_ARMS:
        run_dir = RUNS / f"{arch}__seed{PROBE_SEED}"
        if arch in corpus_failed:
            # Named, not silently dropped: an arm that vanishes from the table with
            # no line of output is how a missing arm goes unnoticed.
            print(f"  SKIP {arch}: failed in §9 -- {corpus_failed[arch][:90]}")
            continue
        if not (run_dir / "best.pt").exists():
            print(f"  SKIP {arch}: no best.pt")
            continue
        model = DP.build_arm(env, arch, verbose=True)
        model.load_state_dict(torch.load(str(run_dir / "best.pt"),
                                         map_location="cpu", weights_only=True))
        model.to(env.device).eval()

        ap  = eval_ap(model, val_loader, env.device, amp)
        op  = FN.fit_operating_point(model, env, CFG, man640, run_dir)
        tab = FN.score_split(model, env, CFG, man640, "val", op["cut"], META_VAL)
        corpus_tables[arch] = tab

        enc = DP.CORPUS_ARCHS[arch].get("encoder")
        q1, q3 = tab.dice.quantile([0.25, 0.75])
        corpus_rows.append({
            "arm": arch, "corpus": DP.SOURCES[enc].corpus if enc else "ImageNet-1k",
            "grid": DP.arm_grid(arch), "input": model.enc_size,
            "encoder_M": sum(p.numel() for p in model.backbone.parameters()) / 1e6,
            "val_AP": ap, "cut": op["cut"],
            "val_mean_dice": tab.dice.mean(),
            "val_median_dice": tab.dice.median(),
            "val_iqr_dice": q3 - q1,
            "misses": int(tab.complete_miss.sum()),
            "licence": DP.SOURCES[enc].licence.split("--")[0].strip() if enc else "BSD-3",
        })
        tab.to_csv(RESULTS / f"val_per_image__{arch}.csv", index=False)
        del model
        if str(env.device).startswith("cuda"):
            torch.cuda.empty_cache()

    # Reaching here with zero rows means §9 trained nothing -- almost always
    # because it was never re-run after a fix, so every arm is still carrying a
    # stale failure from an earlier attempt.
    if corpus_rows:
        CORPUS = pd.DataFrame(corpus_rows).sort_values("val_AP", ascending=False)
        display(CORPUS)
        CORPUS.to_csv(RESULTS / "corpus_val_summary.csv", index=False)
    else:
        CORPUS = pd.DataFrame()
        print("\\n  NOTHING TO SCORE -- no corpus arm has a trained checkpoint.")
        print("  Run §9 first. If §9 already failed, fix the cause and re-run it:")
        print("  it resets `corpus_failed` at the top, so a stale failure from an")
        print("  earlier attempt keeps every arm skipped until §9 runs again.")
        for a, why in corpus_failed.items():
            print(f"     {a:<24} {why[:100]}")
else:
    print("skipped -- reloading tables from disk")
    for arch in DP.CORPUS_ARMS:
        f = RESULTS / f"val_per_image__{arch}.csv"
        if f.exists():
            corpus_tables[arch] = pd.read_csv(f)
    print(f"  loaded {sorted(corpus_tables)}")
''')

# ── §11 ──────────────────────────────────────────────────────────────────────
md("""
## §11 — THE GATE

Fixed in `bruisekit/dermprobe.py` **before any number was produced**:

```
open  iff  the (dermlip_g28 − dinov2_g28) val-Dice CI clears zero
```

`dermlip_g28` is the treatment **by name**. The second dermatology arm is
secondary and cannot be promoted into the primary slot afterwards — that
substitution is how a two-arm experiment quietly becomes a one-arm experiment
with two chances, and this study already refuses the equivalent move on seeds
(§15, trap 3).

Reported alongside and deliberately **not** ANDed in:

- **vs_medical** — `dermlip − medsiglip`. *Is a skin corpus better than a
  general-hospital corpus?* This is the contrast Stage N thought it was running.
- **vs_biomed** — `dermlip − biomedclip`. Architecture-matched to the patch. The
  cleanest contrast in the pool.
- **vs_imagenet** — ties the whole stage back to Stage B.
- **misses** — the endpoint §1 says decides.

`corpus_gate` **raises** if any corpus arm is not at 28 × 28. The entire premise
is a comparison at a fixed grid; if that silently stopped being true the gate
would reproduce the confound it exists to close, wearing a new label.
""")

code('''
GATE = {}
if corpus_tables:
    GATE = DP.corpus_gate(corpus_tables, n_boot=N_BOOT, calibration=CALIB or None)
    DP.print_gate(GATE)
    (RESULTS / "gate.json").write_text(json.dumps(GATE, indent=1))
    print(f"\\nwritten -> {RESULTS / 'gate.json'}")
else:
    print("no corpus tables -- run §9-§10 first")
''')

# ── §12 ──────────────────────────────────────────────────────────────────────
md("""
## §12 — How to read the gate

**If it CLOSED** — that is the result, and it is a *much* stronger claim than
§7f.8 could make:

> *Frozen features from a ViT-B/16 pretrained on 1.03 M dermatology image-text
> pairs do not outperform the same-capacity self-supervised natural-image encoder
> on bruise segmentation, at a matched 28 × 28 feature grid, on 134 held-out
> validation images across 20 subjects.*

That names a **real dermatology corpus** rather than a general-medical one, holds
architecture and grid fixed, and cannot be explained by feature-grid resolution.
Stage N's version of this sentence had none of those three properties. Stop here
and write it.

**If it OPENED** — dermatology pretraining is worth something on this task. Then,
before anything else:

1. Check **vs_biomed**. If DermLIP also beats BiomedCLIP, the win is *skin*, not
   *biomedical*. If it does not, the win is "any domain-shifted image-text
   corpus" and the claim is weaker.
2. Check the **licence** column. Both derm encoders are non-commercial; DINOv2 is
   Apache-2.0. A +0.02 Dice win that costs commercial viability is a trade to
   state, not a result to celebrate.
3. Only then consider seg arms — Stage N §10 onward with a new encoder key. That
   is ~20 GPU-hours and a separate decision.

**If the two dermatology arms disagree** — that is informative, not noise.
DermLIP is language-supervised on skin; DermLIP-PanDerm is self-supervised on
skin first. A split between them says the *training objective* matters more than
the corpus, and the primary/secondary split declared in §4 is what stops that
becoming a post-hoc pick.

---

### What to expect, honestly

Handbook §7f.5's scaling prior has not changed: **23× the parameters bought
+0.006 Dice** across the SegFormer family, and human annotators disagree with
each other by 0.581 to 0.873. Nothing in this notebook is going to move mean Dice
in a way that matters clinically. What it *can* do is settle two claims cheaply
and correctly — which of §7f.8's numbers were real, and whether "medical
foundation model" was ever the right question.
""")

# ── §13 ──────────────────────────────────────────────────────────────────────
md("""
## §13 — Misses and spread

**Lead with these, not with mean Dice.** In Stage A the IQR and the miss column
separated the field while mean Dice did not, and §1 says the miss ranking is the
one that decides.

Complete misses are `dice == 0` throughout — handbook §7.2a. The other definition
(`pred_positive_pixels == 0`) misses the case where a model fires confidently on
the wrong region, which is still a missed injury to a clinician.
""")

code('''
MISS = pd.DataFrame()
rows = []
for name, tab in sorted({**cal_tables, **corpus_tables}.items()):
    q1, q3 = tab.dice.quantile([0.25, 0.75])
    rows.append({"arm": name, "grid": DP.arm_grid(name), "n": len(tab),
                 "misses_dice0": int(tab.complete_miss.sum()),
                 "miss_pct": 100 * float(tab.complete_miss.mean()),
                 "mean_dice": tab.dice.mean(),
                 "median_dice": tab.dice.median(), "iqr_dice": q3 - q1})
if rows:
    MISS = pd.DataFrame(rows).sort_values("mean_dice", ascending=False)
    display(MISS)
    MISS.to_csv(RESULTS / "derm_probe_misses.csv", index=False)

    print("\\n  These are FROZEN LINEAR PROBES on 134 validation images, not")
    print("  trained models on 185 test images. Do NOT put these miss counts in")
    print("  the same table as Stage A's -- different data, different split,")
    print("  different question. They rank the arms against each other and")
    print("  nothing else.")
else:
    print("nothing scored yet")
''')

# ── §14 ──────────────────────────────────────────────────────────────────────
md("""
## §14 — *optional* — score the corpus arms once on test

Off by default and it should usually stay off. Handbook §7f.4: a gate scored on
test is a decision taken on the data the paper reports, and every downstream
number inherits that. The gate in §11 is complete without this.

Turn it on only if the gate has **already** been decided and you want the miss
endpoint on the real test set for a write-up. The cut is read from each arm's
`operating_point.json` — fitted on val, applied once, never re-swept.
""")

code('''
TEST = pd.DataFrame()
if SCORE_TEST and corpus_tables:
    from bruisekit import foundation as FN

    trows = []
    for arch in DP.CORPUS_ARMS:
        run_dir = RUNS / f"{arch}__seed{PROBE_SEED}"
        op_file = run_dir / "operating_point.json"
        if not (run_dir / "best.pt").exists() or not op_file.exists():
            continue
        cut = float(json.loads(op_file.read_text())["cut"])
        model = DP.build_arm(env, arch, verbose=False)
        model.load_state_dict(torch.load(str(run_dir / "best.pt"),
                                         map_location="cpu", weights_only=True))
        model.to(env.device).eval()
        tab = FN.score_split(model, env, CFG, man640, "test", cut, META)
        tab.to_csv(run_dir / "test_per_image.csv", index=False)
        q1, q3 = tab.dice.quantile([0.25, 0.75])
        trows.append({"arm": arch, "cut": cut, "n": len(tab),
                      "mean_dice": tab.dice.mean(),
                      "median_dice": tab.dice.median(), "iqr_dice": q3 - q1,
                      "mean_precision": tab.precision.mean(),
                      "mean_recall": tab.recall.mean(),
                      "misses": int(tab.complete_miss.sum())})
        del model
        if str(env.device).startswith("cuda"):
            torch.cuda.empty_cache()
    if trows:
        TEST = pd.DataFrame(trows).sort_values("mean_dice", ascending=False)
        display(TEST)
        TEST.to_csv(RESULTS / "corpus_test_summary.csv", index=False)
    else:
        print("  no corpus arm has both best.pt and operating_point.json")
else:
    print("skipped (SCORE_TEST = False) -- the gate is val-only by design")
''')

# ── §15 ──────────────────────────────────────────────────────────────────────
md("""
## §15 — What to write down

Fill these in from the cells above. Each is a sentence that stands on its own
whichever way the number went.

1. **The confound.** *"On a fixed ImageNet ResNet-50, moving the feature grid
   from 20×20 to 37×37 changed val Dice by X (CI …), with pretraining, depth and
   weights held constant."* Then: *"Handbook §7f.8 reports dinov2 − resnet50 =
   +0.534 across exactly that grid span."* Those two sentences next to each other
   are the finding, and they need no editorialising.
2. **The stride pair.** `rn50_l4_g20` vs `rn50_l3_g40` — same weights, same 640
   input, only the stride differs. The cleanest single statement of the confound.
3. **The floor.** What the pixel probe reached at each grid. If it tracks the
   ResNet slope, the grid effect is geometry rather than features.
4. **The real question.** *"A ViT-B/16 pretrained on 1.03 M dermatology
   image-text pairs scored X against an identically-sized self-supervised
   natural-image encoder's Y, at a matched 28×28 grid (Δ, CI)."* This is the
   sentence Stage N could not produce.
5. **The matched contrast.** DermLIP vs BiomedCLIP — same architecture, same
   patch, same loader, same grid, different corpus. Skin photographs against
   journal figures.
6. **Misses.** The endpoint §1 says decides. Report the arm ranking, and say
   plainly that these are frozen probes on val and not comparable to Stage A.
7. **The model you could not run.** `google/derm-foundation` — the best-targeted
   encoder in existence for this task, excluded because Google published it as an
   embedding-only SavedModel with no token grid. That is a checkable architectural
   fact, not a compute-budget excuse, and a reviewer will ask.
8. **The licences.** Both dermatology encoders are non-commercial
   (CC-BY-NC-ND-4.0, CC-BY-NC-4.0). DINOv2 is Apache-2.0, BiomedCLIP is MIT. If a
   derm arm ties DINOv2, the licence decides.
9. **The limitation you owe.** Position embeddings were resampled to a common
   grid — the opposite of Stage N's native-resolution policy, done deliberately
   and applied identically to every arm. And frozen-feature linear probing is the
   benchmark DINOv2 was designed against (§7f.9), which the matched grid does
   *not* fix.
""")

code('''
print(f"── everything Stage N2 wrote, in {RESULTS} ──")
for f in sorted(RESULTS.rglob("*")):
    if f.is_file() and f.suffix in (".csv", ".json"):
        print(f"  {f.relative_to(RESULTS)}")

print("\\n── the three lines to take to a meeting ──")
if (RESULTS / "grid_calibration.json").exists():
    C = json.loads((RESULTS / "grid_calibration.json").read_text())
    rf = C.get("families", {}).get("resnet", {})
    if "slope_dice_per_doubling" in rf:
        lo, hi = rf["slope_ci95"]
        print(f"  GRID    {rf['slope_dice_per_doubling']:+.4f} Dice per doubling "
              f"of grid, CI [{lo:+.4f}, {hi:+.4f}], pretraining held FIXED")
    if "stride_only_delta" in C:
        print(f"  STRIDE  same weights, same 640 input, layer4->layer3: "
              f"{C['stride_only_delta']:+.4f}")
if (RESULTS / "gate.json").exists():
    G = json.loads((RESULTS / "gate.json").read_text())
    lo, hi = G["delta_dice_ci95"]
    print(f"  GATE    {G['treatment']} - {G['control']} = {G['delta_dice']:+.4f} "
          f"CI [{lo:+.4f}, {hi:+.4f}]  -> "
          f"{'OPEN' if G['GATE_run_seg_arms'] else 'CLOSED'}")
    if "stage_n_reinterpretation" in G:
        r = G["stage_n_reinterpretation"]
        print(f"  7f.8    {r['grid_share']:.0%} of the published +0.5342 is grid, "
              f"{r['residual_attributable_to_pretraining']:+.4f} residual")
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
            compile(src, f"cell{i}", "exec")   # a cell that will not parse is not shippable
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
