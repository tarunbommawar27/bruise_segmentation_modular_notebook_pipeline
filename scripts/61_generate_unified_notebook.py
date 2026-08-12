#!/usr/bin/env python
"""Emit BRUISE_UNIFIED/bruise_unified.ipynb -- the single four-stage notebook.

WHY GENERATE RATHER THAN HAND-EDIT
-----------------------------------
A notebook is a JSON blob with source split into line-lists; editing one by hand
is how cells drift out of order and how a fixed bug reappears three cells later
in a copy. The notebook is a build artefact here, exactly like the bundle: this
script is the source, and re-running it produces the same notebook.

DESIGN: THIN NOTEBOOK, FAT LIBRARY
-----------------------------------
Every cell is config, a call into `bruisekit`, or a rendering of what came back.
No cell defines a model, a metric or a training loop -- those live in tested
modules that ship next to it. The consequence that matters: reading the notebook
top to bottom tells you WHAT the study does, and the modules tell you HOW, and
neither is diluted with the other.
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "BRUISE_UNIFIED" / "bruise_unified.ipynb"

cells: list[dict] = []


def _cid() -> str:
    """Stable per-position cell id.

    nbformat 4.5+ requires one. Deriving it from the cell's index rather than
    generating a random uuid keeps the notebook byte-reproducible: regenerating
    after an unrelated edit produces a clean diff instead of 45 changed ids.
    """
    return f"cell-{len(cells):02d}"


def md(text: str) -> None:
    cells.append({"cell_type": "markdown", "id": _cid(), "metadata": {},
                  "source": text.strip("\n").splitlines(keepends=True)})


def code(text: str) -> None:
    cells.append({"cell_type": "code", "id": _cid(), "execution_count": None,
                  "metadata": {}, "outputs": [],
                  "source": text.strip("\n").splitlines(keepends=True)})


# ═════════════════════════════════════════════════════════════════════════════
md(r"""
# Bruise segmentation — unified pipeline

One notebook for the whole study. It **trains nothing that is already trained**:
every run is resolved against the shipped checkpoints first, then against the
shipped results, and only trains as a last resort — and even then only if you
explicitly allow it.

| Stage | What it covers |
|---|---|
| **A** | SegFormer-B2 teacher, B0 direct, B0 distilled, YOLO26n direct + distilled — 3 seeds each |
| **B** | U-Net and DeepLabV3+ direct baselines (3 seeds), nnU-Net |
| **C** | SegFormer-B5 teacher and the knowledge-distillation arms |
| **E** | Mobile baselines (PP-MobileSeg-Tiny, TopFormer-Tiny, LR-ASPP MobileNetV3, Fast-SCNN) and the Stage F DeepLabV3+ distilled arms — 3 seeds each |
| **H** | **Reliability-gated distillation**, plus SegFormer-B2 as the teacher for every small student — the gated arms and their teacher-matched controls |
| **D** | Headline tables, per-image distributions, bootstrap intervals, fairness, size confound, annotation ceiling |
| **G** | Final significance — omnibus first, then a pre-specified contrast family with multiplicity control |

**Runs on Colab and on local Jupyter without edits.** Set `BUNDLE_ROOT` in §0 if
auto-detection does not find the bundle; nothing else is host-specific.

**Default behaviour is cheap.** `ALLOW_TRAINING = False` and
`RECOMPUTE_FROM_WEIGHTS = False`, so *Run All* takes a couple of minutes on a
laptop with no GPU and reproduces every table and figure from the shipped
per-image results. Flip either flag to do real work.
""")

# ── §0 configuration ─────────────────────────────────────────────────────────
md(r"""
## §0 · Configuration

The only cell you may need to touch. Everything below reads from here.
""")

code(r'''
# ── where the bundle is ──────────────────────────────────────────────────────
# None = auto-detect (checks $BRUISE_BUNDLE, Colab Drive locations, then this
# notebook's own folder and its parents). Set a path to be explicit, e.g.
#   BUNDLE_ROOT = "/content/drive/MyDrive/BRUISE_UNIFIED"
#   BUNDLE_ROOT = r"C:\BRUISE_SEGMENTATION_PROJECT\BRUISE_UNIFIED"
BUNDLE_ROOT = None
WORK_DIR    = None      # scratch for the 640 cache and any new runs. None = sensible default.
DEVICE      = None      # None = auto (cuda -> mps -> cpu). "cpu" forces cached-results mode.

# ── extra places to LOOK for checkpoints ─────────────────────────────────────
# The study's weights are not in one directory and never were. `env.run_roots`
# searches, in this order:
#
#   1. WORK_DIR/runs                        anything this session trains
#   2. EXTRA_RUNS                           <- you set this
#   3. <bundle>/checkpoints/final           Stage A, 5 models x 3 seeds
#   4. <bundle>/checkpoints/baselines       U-Net, DeepLabV3+
#   5. <bundle>/checkpoints/efficient       Stage E/F mobile arms
#   6. <bundle>/checkpoints/rgkd            Stage H arms
#   7. <bundle>/checkpoints/yolo_l          Stage Y
#   8. <bundle>/checkpoints/distill/teachers  B2, B5, B0-distilled
#
# First hit wins, so your own training always beats a shipped copy, and every
# Run records WHICH root answered (`source_root`) and in which on-disk layout
# (`layout`). Before this, only 1 and one of 3-8 were searched, the fallback
# between them was silent, and on 2026-08-04 that loaded old-lineage SegFormer
# checkpoints because the real runs sat in a scratch dir nobody had named.
#
# Point it at the machine's scratch tree. A single string or a list both work:
#   EXTRA_RUNS = "/scratch/tbommawa/bruise_work/runs"
#   EXTRA_RUNS = ["/scratch/tbommawa/bruise_work/runs", "/content/bruise_work/runs"]
EXTRA_RUNS = None

# ── what this session is allowed to do ───────────────────────────────────────
# Any subset, e.g. "ED" for just the mobile baselines + analysis.
# "Y" adds YOLO26-large (Stage Y) and is OFF by default: it is the most expensive
# single arm in the study (~3.5 h) and needs a pretrained file the bundle does not
# ship. Add it deliberately -- "ABCDEHY" -- not by leaving it on.
RUN_STAGES  = "ABCDEH"

# False: a run with no checkpoint AND no cached result is reported as a gap and
#        skipped. True: it is trained. Left False so that an accidental Run All
#        on a fresh bundle costs seconds rather than a GPU day.
ALLOW_TRAINING = False

# False: use the shipped per-image CSVs (fast, exact, no GPU needed).
# True:  re-run inference from the checkpoints. Verified on all seven reporting
#        models: every mean Dice agrees with the cached table to better than 2e-4
#        (CPU vs the original A100). The residual is float ordering, not
#        disagreement -- no ranking, miss count or verdict changes. Budget ~33 min
#        on CPU, a couple of minutes on a GPU.
RECOMPUTE_FROM_WEIGHTS = False

# run_ids to retrain even though a checkpoint exists. Output goes to the work dir,
# never over the shipped checkpoints.
FORCE_RETRAIN = []

# Which seeds Stage E should have. Stages A-C are always scanned at the seeds
# their shipped checkpoints were trained with; this controls only the NEW mobile
# baselines, where the seed count is purely a decision about GPU time.
# Eight families now (four direct + four DeepLabV3+ distilled arms):
#   (0,)      8 runs,  ~5.4 GPU-hours  -- one seed, enough for a first comparison
#   (0, 1, 2) 24 runs, ~16.2 GPU-hours -- a spread you can quote a std for
#
# Use three. A one-seed Stage E is what produced the caveat in handbook 7.2a, and
# a distilled-vs-direct claim from a single seed is not a claim (handbook 15,
# trap 12: a 0-of-185 count is not a rate).
EFFICIENT_SEEDS = (0, 1, 2)

# Which seeds Stage H should have. None = the same as Stage E, which is the right
# default: every Stage H arm is read against a Stage E/F control AT THE SAME SEED,
# and a 3-seed arm against a 1-seed control is not a contrast.
# Ten families (six reliability-gated + four B2 plain-KD controls):
#   (0,)      10 runs, ~9.0 GPU-hours
#   (0, 1, 2) 30 runs, ~27 GPU-hours
# Set it to (0,) only for a first look at whether the gate fires at all -- and
# read GATE_H, not the Dice table, if you do.
RGKD_SEEDS = None

# ── the inference block (D9) ─────────────────────────────────────────────────
# False: D9 prints what it would do and costs nothing. True: run one test-set
#        inference pass AND time the models at 640, writing both to the work dir.
#
# Separate from RECOMPUTE_FROM_WEIGHTS on purpose. That flag re-derives the
# tables the notebook already reports, for the models it already reports. This
# one answers a different question -- "what does an arbitrary set of models score
# and how fast is it?" -- and is the only path that produces a SPEED table at all.
# D8 reads a benchmark CSV; nothing in this notebook ever wrote one.
RUN_INFERENCE_BLOCK = False

# Which models D9 covers. None = the three SegFormers (inference.DEFAULT_MODELS):
# the analysis notebook's own SEGFORMER_MODELS dict and the exact three rows of
# track_a_comparison.csv. Any registry family name works, so the four mobile
# baselines and the Stage F/H arms -- none of which have ever been timed -- need
# no new code, only GPU time:
#   INFERENCE_MODELS = ["fastscnn", "lraspp_mobilenetv3",
#                       "topformer_tiny", "ppmobileseg_tiny"]
# A family with no checkpoint on this host prints SKIP and is left out.
INFERENCE_MODELS = None

# "fp32" is the published recipe and the only publishable setting: `benchmark_speed`
# contains no autocast, so every shipped row is full precision. "fp16" runs the
# mirrored autocast path, which exists to ATTRIBUTE a gap, not to report one --
# measured on two machines across 21 model-pairs, autocast made every model except
# the two big ResNets SLOWER at batch 1 (1.27-1.44x), because it adds cast kernels
# to a benchmark that is already dispatch-bound. A table may not mix the two;
# check_single_machine raises.
INFERENCE_PRECISION = "fp32"

# Short name for this host, e.g. "orc-mig", "colab-a100". Goes into the output
# filename so two machines' tables cannot collide or be mistaken for one another.
# None = device-tagged only, which was how three incomparable tables came to share
# a filename and have to be told apart from memory.
MACHINE_TAG = None

# ── Stage Y · YOLO26-large ───────────────────────────────────────────────────
# Which seeds Stage Y should have. ONE by default, unlike every other stage.
# A yolo26l run is ~3.5 h against a mobile arm's ~0.5 h, so the three-seed habit
# that costs 1.5 GPU-hours in Stage E costs 21 here. One seed answers the question
# the stage is asked -- "does capacity fix yolo26n's 6.5% complete-miss rate?" --
# and three are worth buying only once that answer is yes.
#
# Needs `pretrained_weights/yolo26l-sem.pt`, which most bundles do not ship (~50 MB).
# §2 reports it as a WARN rather than a preflight failure: a session that is not
# running Stage Y should not be blocked by a file it will never open.
YOLO_L_SEEDS = (0,)

# ── recipe (identical to the recipe the shipped checkpoints were trained with) ─
CFG = dict(
    img_size        = 640,
    seeds           = (0, 1, 2),
    epochs          = 100,
    patience        = 15,
    batch_mode      = "per_model",
    effective_batch = 8,
    max_probe_batch = 64,
    vram_target     = 0.75,
    backbone_lr     = 6e-5,
    head_lr         = 6e-4,
    betas           = (0.9, 0.999),
    weight_decay    = 0.01,
    warmup_fraction = 0.01,
    poly_power      = 1.0,
    gradient_clip   = 1.0,
    amp             = True,
    workers         = 0,          # 0 is safe everywhere; raise on Linux for speed
    eval_batch      = 8,          # GPU
    eval_batch_cpu  = 2,          # CPU re-inference; batch cannot affect per-image scores
    segformer_alpha = 0.6,        # SegFormer KD mix
    aux_weight      = 0.4,
    yolo_alpha      = 0.4,        # YOLO offline pseudo-mask KD
    yolo_batch      = -1,         # Ultralytics auto-batch
    yolo_optimizer  = "auto",
    yolo_lrf        = 0.01,
    yolo_warmup_epochs = 3,
    yolo_weight_decay  = 0.0005,
    yolo_close_mosaic  = 10,
    pseudo_threshold   = 0.50,
    smp_encoder     = "resnet50",
    smp_micro_batch = 16,
    efficient_micro_batch = 16,   # Stage E: fixed batch, same reason as smp_micro_batch
    efficient_alpha = 0.6,        # Stage F: KD mix for the mobile student. If you
                                  # lower efficient_micro_batch for VRAM, it applies
                                  # to the DIRECT baseline too -- otherwise the
                                  # distilled-vs-direct contrast is confounded by
                                  # batch size, which is what the fixed recipe
                                  # (handbook 3) exists to prevent.
    # Stage H -- the reliability gate. NOT an alpha: the gated arms deliberately
    # inherit their control's alpha (segformer_alpha / efficient_alpha /
    # yolo_alpha) so that `*_rgkd` vs its control moves ONE variable. These two
    # are the only new numbers the stage introduces.
    #
    # The gate fades the KD term out as the teacher's own soft Dice on the image
    # falls: full weight at >= rgkd_gate_hi, none at <= rgkd_gate_lo, linear
    # between. 0.10 is chosen to sit just above the complete-miss population
    # (dice == 0) without also gating merely-mediocre teacher outputs, which are
    # still informative; 0.50 is roughly the 5th percentile of the B2 teacher's
    # per-image test Dice, so in the ordinary case the gate is inert and the arm
    # is its own control.
    rgkd_gate_lo = 0.10,
    rgkd_gate_hi = 0.50,
    cut_min = -6.0, cut_max = 6.0, cut_steps = 481,
    drive_sync_every = 2,
    n_boot = 2000,                # subject-level bootstrap resamples (Stage D)
    n_boot_final = 10_000,        # Stage G. 2000 is plenty for an interval, but a
                                  # tail probability quoted to two decimals needs
                                  # more resolution than 1/2000 -- and Stage G's
                                  # p-values get Holm-adjusted, where the input
                                  # granularity propagates straight through.
)

print(f"stages={RUN_STAGES}  allow_training={ALLOW_TRAINING}  "
      f"recompute={RECOMPUTE_FROM_WEIGHTS}  force={FORCE_RETRAIN or 'none'}")
''')

# ── §1 environment ───────────────────────────────────────────────────────────
md(r"""
## §1 · Environment

Installs only what is actually missing, mounts Drive when on Colab, and resolves
every path once. This is the **only** host-aware cell in the notebook — nothing
below it mentions Colab, Drive or `/content`.

**On an offline compute node**, export `BRUISE_NO_PIP=1` before starting the
kernel. The two optional heavyweights (`ultralytics`, `segmentation-models-pytorch`)
are then reported as unavailable in one line instead of costing minutes in pip's
retry backoff. Both are also **deferred**: they install at the point a run that
needs them is about to be built, not because a stage was selected.
""")

code(r'''
import importlib.util, os, subprocess, sys

def _need(mod, pip=None, required=True):
    """Install `pip` only if `mod` cannot be imported. Keeps re-runs instant.

    `required=False` downgrades a failed install to a warning. That matters on an
    offline compute node: a package that this session will never call is not a
    reason to halt the notebook, and raising there would block a run that only
    needs to read cached CSVs.

    TWO GUARDS AGAINST THE OFFLINE-NODE HANG
    -----------------------------------------
    An ORC compute node usually has no route to PyPI, and pip's default is five
    retries with exponential backoff -- so a single unreachable package costs
    MINUTES of a cell that looks like it is doing nothing. Both guards target that:

      BRUISE_NO_PIP=1   skip the attempt entirely and say so. Set it on any node
                        you know is offline; a missing optional package is then a
                        one-line message instead of a stall.
      --retries/--timeout  an OPTIONAL package fails fast (~10s) rather than
                        backing off. A required one keeps pip's defaults, because
                        there the retry is worth the wait.
    """
    if importlib.util.find_spec(mod) is not None:
        return True
    pkg = pip or mod
    if os.environ.get("BRUISE_NO_PIP"):
        msg = f"{pkg} is not installed and BRUISE_NO_PIP is set -- not attempting it."
        if required:
            raise RuntimeError(msg + " Install it into this environment first.")
        print(f"  {msg}")
        print(f"           Anything needing it will be reported as unavailable, not silently skipped.")
        return False
    print(f"installing {pkg} ...")
    cmd = [sys.executable, "-m", "pip", "install", "-q", pkg]
    if not required:
        cmd += ["--retries", "1", "--timeout", "10"]
    try:
        subprocess.run(cmd, check=True)
        return True
    except Exception as e:
        if required:
            raise RuntimeError(f"{pkg} is required and could not be installed: {e}") from e
        print(f"  WARNING: could not install {pkg} ({e}).")
        print(f"           Anything needing it will be reported as unavailable, not silently skipped.")
        return False

# ── HPC / ORC preset ─────────────────────────────────────────────────────────
# Compute nodes are usually offline and often carry a system CUDA/cuDNN that is
# older than the one torch bundles. Both are handled here, and both are harmless
# on Colab or a laptop:
#   * HF offline   -- every encoder loads from pretrained_weights/, so a blocked
#                     huggingface.co must not turn into a 60s timeout per model.
#   * LD_LIBRARY_PATH -- dropping the system cuda/cudnn entries lets torch use its
#                     own bundled cuDNN instead of failing to load the older one.
import os
if not os.environ.get("BRUISE_ALLOW_HF_ONLINE"):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["LD_LIBRARY_PATH"] = ":".join(
    p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":")
    if p and "cuda" not in p.lower() and "cudnn" not in p.lower())

_need("torch")
_need("transformers")
_need("albumentations")
_need("cv2", "opencv-python-headless")
_need("scipy"); _need("pandas"); _need("matplotlib"); _need("yaml", "pyyaml")

# ── the two optional heavyweights, installed only when something needs them ──
#   ultralytics                   native YOLO training and argmax inference
#   segmentation_models_pytorch   U-Net, DeepLabV3+, and the DeepLabV3+ TEACHER
#                                 the Stage F/H distilled arms load on every step
#
# Both are needed only to BUILD a model. Reading a stage's cached per-image CSVs
# is pure pandas and needs neither.
#
# THE CALLS DELIBERATELY DO NOT LIVE HERE. Handbook 15 trap 9 is precisely this
# install firing on an offline node for a package the session will never call,
# and the guard it prescribes is "gated on whether a model will actually be
# BUILT". This cell knows only which STAGES were selected, which is much coarser:
# `RUN_STAGES="EH"` selects two stages that may or may not need Ultralytics
# depending on whether yolo_sem_rgkd is already trained. So the calls sit in
# `train_missing` and in the Stage E/H wiring cells, which have the registry's
# MISSING list in hand and can decide exactly.
#
# Stage E's four direct architectures need nothing extra either way: the two
# vendored nets run against bruisekit's own mmcv shim and LR-ASPP comes from
# torchvision.
_BUILDS_MODELS = ALLOW_TRAINING or RECOMPUTE_FROM_WEIGHTS

_PIP_NAME = {"segmentation_models_pytorch": "segmentation-models-pytorch"}

def _need_optional(*mods):
    """Install an optional heavyweight, at the point something is about to use it."""
    return {m: _need(m, _PIP_NAME.get(m, m), required=False) for m in mods}

if RECOMPUTE_FROM_WEIGHTS:
    # This path re-infers EVERY selected model from its checkpoint, so there is
    # nothing left to defer -- both are certain to be called.
    if "A" in RUN_STAGES:
        _need_optional("ultralytics")
    if "B" in RUN_STAGES:
        _need_optional("segmentation_models_pytorch")
elif _BUILDS_MODELS:
    print("optional deps (ultralytics, segmentation-models-pytorch) deferred --\n"
          "  installed only if a run that actually needs them is MISSING")
print("dependencies ready")
''')

code(r'''
from pathlib import Path
import json, sys, time, warnings
import numpy as np, pandas as pd

# The notebook may be opened from anywhere; make sure the bundle's own package
# wins over any same-named install, then import it.
_here = Path(BUNDLE_ROOT).expanduser() if BUNDLE_ROOT else Path.cwd()
for _c in (_here, *_here.parents):
    if (_c / "bruisekit" / "paths.py").exists():
        sys.path.insert(0, str(_c)); break

from bruisekit.paths import setup
from bruisekit.registry import Registry, WEIGHTS, RESULTS, MISSING, summarize_gaps
from bruisekit import loaders as L
from bruisekit import report as R

env = setup(root=BUNDLE_ROOT, work=WORK_DIR, device=DEVICE, extra_runs=EXTRA_RUNS)
print(env.describe())

# Print the search path, marking which roots actually exist. This is the line
# that would have caught the 2026-08-04 wrong-checkpoint substitution in ten
# seconds instead of two wasted ORC runs: if the root holding your training is
# not on this list, nothing below is reading it.
print("\ncheckpoint search path (first hit wins):")
for _i, _r in enumerate(env.run_roots, 1):
    _n = len(list(_r.iterdir())) if _r.is_dir() else 0
    print(f"  {_i}. {'OK ' if _r.is_dir() else '-- '} {_n:>3} entries  {_r}")
if not any(r.is_dir() and any(r.iterdir()) for r in env.run_roots):
    print("  WARNING  every search root is empty or absent -- nothing can load.")

if str(env.device) == "cpu" and (ALLOW_TRAINING or RECOMPUTE_FROM_WEIGHTS):
    warnings.warn("No GPU: training is impractical and re-inference will be slow. "
                  "Consider ALLOW_TRAINING = RECOMPUTE_FROM_WEIGHTS = False.")
''')

# ── §2 preflight ─────────────────────────────────────────────────────────────
md(r"""
## §2 · Preflight

Verifies the bundle before anything reads from it. Split sizes, **subject- and
image-level leakage**, that every manifest row points at a file that exists, and
that the pretrained backbones are present.

Leakage is re-checked here rather than trusted from the build, because a manifest
is a text file and text files get edited. A subject appearing in both train and
test would inflate every number in the notebook and would otherwise be invisible.
""")

code(r'''
MAN = {s: pd.read_csv(env.manifests / f"{s}.csv") for s in ("train", "val", "test")}
EXPECT = {"train": 697, "val": 134, "test": 185}

ok = True
for s, df in MAN.items():
    good = len(df) == EXPECT[s]
    ok &= good
    print(f"{'PASS' if good else 'FAIL'}  {s:>5}: {len(df):>3} images, "
          f"{df.subject.nunique():>3} subjects (expected {EXPECT[s]})")

for a, b in [("train","val"), ("train","test"), ("val","test")]:
    subj = set(MAN[a].subject) & set(MAN[b].subject)
    stem = set(MAN[a].stem)    & set(MAN[b].stem)
    ok &= not subj and not stem
    print(f"{'PASS' if not (subj or stem) else 'FAIL'}  no leakage {a}/{b}"
          + (f"  -- {len(subj)} subjects, {len(stem)} images SHARED" if (subj or stem) else ""))

missing = [p for s, df in MAN.items() for p in df.image_path if not (env.data / p).exists()]
ok &= not missing
print(f"{'PASS' if not missing else 'FAIL'}  all {sum(len(d) for d in MAN.values())} images present"
      + (f" -- {len(missing)} MISSING, e.g. {missing[0]}" if missing else ""))

for w in ("segformer_mit_b0", "segformer_mit_b2", "yolo26n-sem.pt"):
    e = (env.weights / w).exists(); ok &= e
    print(f"{'PASS' if e else 'FAIL'}  pretrained: {w}")

# Stage Y's backbone is a WARN, not a FAIL. It is a ~50 MB file that only a
# Stage Y session opens, and blocking every other session on it would make the
# preflight lie about what this run actually needs.
_yl = env.weights / "yolo26l-sem.pt"
if _yl.exists():
    print(f"PASS  pretrained: {_yl.name} (Stage Y available)")
elif "Y" in RUN_STAGES:
    ok = False
    print(f"FAIL  pretrained: {_yl.name} MISSING and 'Y' is in RUN_STAGES\n"
          f"      from ultralytics import YOLO; YOLO('yolo26l-sem.pt')  "
          f"-> move it to {env.weights}")
else:
    print(f"WARN  pretrained: {_yl.name} absent -- Stage Y unavailable "
          f"(not requested, so not a failure)")

print("\n" + ("PREFLIGHT OK" if ok else "PREFLIGHT FAILED -- do not trust anything below"))
assert ok, "preflight failed"
''')

# ── §3 registry ──────────────────────────────────────────────────────────────
md(r"""
## §3 · Checkpoint registry — the train-or-skip plan

Every run in the study is resolved to exactly one of three tiers **before any
compute happens**:

- **WEIGHTS** — a usable checkpoint exists. Nothing trains.
- **RESULTS** — no checkpoint, but this run's metrics were recorded. Reported from
  cache, always labelled as such. This tier is why the whole analysis reproduces
  on a laptop.
- **MISSING** — neither. The only case where training is even proposed.

A MISSING run stays missing in every table below. It is never back-filled from a
neighbouring seed and never quietly averaged away.
""")

code(r'''
reg = Registry(env, allow_training=ALLOW_TRAINING, force=FORCE_RETRAIN,
               efficient_seeds=EFFICIENT_SEEDS, rgkd_seeds=RGKD_SEEDS,
               yolo_l_seeds=YOLO_L_SEEDS).scan()
plan = reg.report()
''')

code(r'''
# WHERE DID EACH CHECKPOINT COME FROM?
#
# The plan above says WHAT resolved. This says FROM WHERE -- the question that
# went unanswered for two ORC runs because the registry searched two directories
# and fell back between them without a word. Read the `root` column: if a family
# you trained yourself shows a shipped root, your training is not being used.
from bruisekit.registry import WEIGHTS as _W

_prov = pd.DataFrame([
    {"run_id": r.run_id, "stage": r.stage, "family": r.family, "kind": r.kind,
     "layout": r.layout, "root": r.source_root.name if r.source_root else "-",
     "threshold": r.threshold_file.name if r.threshold_file else "-"}
    for r in reg.runs.values() if r.tier == _W
])
if len(_prov):
    print(f"{len(_prov)} usable checkpoints, by source:")
    print(_prov.groupby(["root", "layout", "stage"]).size().to_string())
    display(_prov.sort_values(["stage", "run_id"]).reset_index(drop=True))
else:
    print("no WEIGHTS-tier runs found in any search root -- check EXTRA_RUNS in §0")

# The teachers, called out separately because the fairness work turns on which
# one you distil from. B2 is the ONLY model in this study with a statistically
# detectable skin-tone disparity (Kruskal p=0.011, gap 0.112); B5 has gap 0.070
# at p=0.220 and beats B2 on every ITA group -- +0.057 on Tan (IV), +0.027 on
# Dark (VI). Every Stage H arm currently distils from B2.
_t = [r for r in reg.runs.values() if r.stage == "T"]
if _t:
    print("\nteacher store:")
    for r in sorted(_t, key=lambda x: x.family):
        print(f"  {r.family:<26} {r.tier:<8} "
              f"{'cut available' if r.threshold_file else 'no threshold (fine for KD)'}")
''')

code(r'''
# The gaps, as a table you can paste into a limitations section.
gaps = summarize_gaps(reg)
gaps if len(gaps) else "No gaps: every run in the study has weights or cached results."
''')

# ── §4 cache ─────────────────────────────────────────────────────────────────
md(r"""
## §4 · The 640 cache

The SegFormer and baseline dataloaders read 640×640 PNGs. Building them once
means training and evaluation see **bit-identical pixels**, and every epoch reads
a small PNG instead of decoding a multi-megapixel JPEG.

Masks are resized **nearest-neighbour**, images bilinear. That is not
interchangeable: bilinear on a mask produces fractional boundary pixels, and
thresholding those back to binary erodes or dilates small bruises — exactly the
population the complete-miss metric is about.

Skipped entirely unless something actually needs to run inference.
""")

code(r'''
NEED_CACHE = ALLOW_TRAINING or RECOMPUTE_FROM_WEIGHTS or RUN_INFERENCE_BLOCK
if NEED_CACHE:
    t0 = time.time()
    MAN640 = L.build_cache640(env, MAN)
    print(f"ready in {time.time()-t0:.0f}s")
else:
    MAN640 = None
    print("skipped -- reporting from cached per-image results "
          "(set RECOMPUTE_FROM_WEIGHTS = True to rebuild and re-infer)")
''')

# ── Stage A ──────────────────────────────────────────────────────────────────
md(r"""
---
# Stage A · The five headline models

SegFormer-B2 (teacher), SegFormer-B0 (direct), SegFormer-B0 (distilled from B2),
YOLO26n (direct), YOLO26n (distilled) — three seeds each.

**YOLO is reported on the native-argmax path only.** The custom `/255` path is
kept in the bundle for provenance but is not a reporting path; argmax is
parameter-free, so there is no operating point to fit and nothing to overfit.
""")

code(r'''
def train_missing(stage, reg, cfg):
    """Train every MISSING run in a stage. No-op when ALLOW_TRAINING is False.

    Fresh runs are written to env.runs, never over the shipped checkpoints, so a
    retrain can always be compared against — or discarded in favour of — the run
    that produced the published numbers.
    """
    todo = reg.to_train(stage)
    if not todo:
        print(f"Stage {stage}: nothing to train."); return []
    if not ALLOW_TRAINING:
        print(f"Stage {stage}: {len(todo)} run(s) MISSING, skipped (ALLOW_TRAINING=False):")
        for r in todo:
            print(f"    {r.run_id:<34} {r.note}")
        return []

    # Separate "missing" from "trainable". nnU-Net is MISSING and stays MISSING:
    # it has no training path here, so it must not make this function demand a GPU
    # or look like work that is about to happen. Reported, then set aside.
    TRAINABLE = {"segformer", "smp", "efficient", "yolo"}
    skipped = [r for r in todo if r.kind not in TRAINABLE]
    todo = [r for r in todo if r.kind in TRAINABLE]
    for r in skipped:
        print(f"Stage {stage}: {r.run_id} has no training path here -- left MISSING ({r.note})")
    if not todo:
        return []

    if not str(env.device).startswith("cuda"):
        raise RuntimeError(
            f"Stage {stage} would train {len(todo)} run(s) but no CUDA device is visible. "
            f"Set ALLOW_TRAINING=False to report from cache, or start a GPU session.")

    # The optional heavyweights, installed HERE and not in §1 -- this is the first
    # point at which the registry has said what is genuinely about to be built
    # (handbook 15 trap 9). An SMP dependency arises two ways: the student IS an
    # SMP model (kind "smp"), or its TEACHER is one, which is how every Stage F
    # and every DeepLabV3+-teacher Stage H arm loads its soft labels.
    kinds = {r.kind for r in todo}
    teachers = {L.FAMILY_SPEC.get(r.family, {}).get("teacher") for r in todo}
    if "yolo" in kinds:
        _need_optional("ultralytics")
    if "smp" in kinds or any(t and t.endswith("_r50") for t in teachers):
        _need_optional("segmentation_models_pytorch")

    from bruisekit.engine import train_run
    from bruisekit import reliability_kd as RK
    import bruisekit.yolo_native as yn
    done = []
    for r in todo:
        print(f"\n{'='*70}\n{r.run_id}\n{'='*70}")
        t0 = time.time()
        # ONE context manager governs both Stage H shims. Inside it, engine's
        # DistillLoss becomes the gated loss (for `*_rgkd` arms only) and
        # load_teacher resolves this arm's teacher from reliability_kd.TEACHER_FOR.
        # Outside it -- i.e. for every Stage A/B/E/F run -- both shims fall through
        # to what was bound before them, so nothing that already worked changes.
        # train_run tells neither the loss nor the teacher loader which run it is
        # training, so this is where that fact has to be supplied.
        with RK.arm(r.family if RK.is_stage_h(r.family) else None):
            if r.kind in ("segformer", "smp", "efficient"):
                spec = L.spec_for(r.family)
                # Each stage owns its KD mix. Reading segformer_alpha for a Stage F
                # run would look right (both are 0.6) and silently ignore
                # efficient_alpha the moment someone tuned one of them. A Stage H
                # gated arm deliberately takes its CONTROL's alpha -- same key, same
                # value -- so the gate is the only thing that differs.
                alpha_key = "efficient_alpha" if r.kind == "efficient" else "segformer_alpha"
                run_cfg = {**cfg, "alpha": cfg[alpha_key] if spec["distill"] else None}
                res = train_run(r.run_id, spec, r.seed, run_cfg, env.paths_for_models(),
                                MAN640, env.cache640, env.runs, env.device)
                gs = RK.dump_stats(env.runs / r.run_id)
                print(f"  -> {res.get('status','trained')} in {(time.time()-t0)/60:.1f} min")
                if gs:
                    print(f"     gate: coverage {gs['mean_coverage']:.4f}, alpha_eff "
                          f"{gs['mean_alpha_effective']:.4f}, "
                          f"{gs['images_fully_gated_off']} of {gs['images_seen']} "
                          f"image-views fully gated off")
            elif r.kind == "yolo":
                rd = env.runs / r.run_id; rd.mkdir(parents=True, exist_ok=True)
                data_dir = rd / "yolo_data"
                if not (rd / "DATASET_DONE.json").exists():
                    # YOLO never sees a loss object -- it trains natively under
                    # Ultralytics, so the teacher reaches it only through the
                    # pseudo-mask baked into the training labels. Three variants:
                    #   direct    plain GT
                    #   distilled scalar-alpha fusion with the same-seed B2 teacher
                    #   rgkd      the same fusion with alpha raised to 1 per pixel
                    #             wherever the teacher is unreliable
                    gated = r.family.endswith("_rgkd")
                    distilled = gated or r.family.endswith("distilled")
                    tfn = L.make_teacher_prob_fn(env, r.seed, cfg) if distilled else None
                    if gated:
                        st = RK.build_gated_yolo_dataset(
                            MAN["train"], env.data, data_dir, "train",
                            alpha=cfg["yolo_alpha"], teacher_prob_fn=tfn,
                            imgsz=cfg["img_size"],
                            gate_lo=cfg["rgkd_gate_lo"], gate_hi=cfg["rgkd_gate_hi"],
                            stats_out=rd / "reliability_gate.json")
                        print(f"     gate: mean image gate {st['mean_image_gate']:.4f}, "
                              f"{st['images_fully_gated_off']} of {st['images_fused']} "
                              f"training images fully gated off")
                    else:
                        yn.build_yolo_dataset(MAN["train"], env.data, data_dir, "train",
                                              alpha=cfg["yolo_alpha"] if distilled else None,
                                              teacher_prob_fn=tfn, imgsz=cfg["img_size"])
                    yn.build_yolo_dataset(MAN["val"], env.data, data_dir, "val",
                                          imgsz=cfg["img_size"])
                    (rd / "DATASET_DONE.json").write_text("{}")
                yn.write_data_yaml(data_dir, rd)
                # Size-aware, via the family name. Passing the nano weights to a
                # Stage Y run is SILENT -- it builds, trains, produces a best.pt
                # and lands in a table as the large arm, with the only symptom
                # being a 28 M-parameter row reporting 1.6 M. yn.pretrained_for
                # resolves it once and raises with the download command if the
                # file is absent.
                yn.train_native(yn.pretrained_for(r.family, env.paths_for_models()),
                                rd / "data.yaml", rd, cfg, r.seed)
                print(f"  -> trained in {(time.time()-t0)/60:.1f} min")
            else:
                print(f"  !! unhandled kind={r.kind}; left MISSING"); continue
        done.append(r.run_id)
    reg.scan()
    return done

trained_A = train_missing("A", reg, CFG) if "A" in RUN_STAGES else []
''')

code(r'''
# Assemble Stage A per-image tables at the val-selected best seed.
#
# The best seed is NOT the same for every model -- it is 0 for the three
# SegFormers and yolo_sem_distilled, but 2 for yolo_sem_direct. Pairing a model's
# weights with another seed's results shows per-image disagreements up to 0.49
# Dice and looks exactly like a broken inference path. R.best_seeds reads the
# mapping off the selection step's own filenames so it cannot drift.
META = MAN["test"]
BEST = R.best_seeds(env)
print("val-selected best seed:", BEST)

if "A" in RUN_STAGES:
    if RECOMPUTE_FROM_WEIGHTS:
        A = {}
        for model, seed in BEST.items():
            run = reg.get(f"{model}__seed{seed}")
            if run is None or run.tier != WEIGHTS:
                print(f"  {model}: no checkpoint, falling back to cached"); continue
            t0 = time.time()
            A[model] = R.normalize(L.score_run(env, run, CFG, MAN640, MAN["test"]), META)
            print(f"  {model:<24} re-inferred from {run.run_id} in {time.time()-t0:.0f}s")
        for model, df in R.load_stage_a(env, reg, META).items():
            A.setdefault(model, df)
    else:
        A = R.load_stage_a(env, reg, META)
    print(f"\nStage A: {len(A)} models loaded "
          f"({'re-inferred' if RECOMPUTE_FROM_WEIGHTS else 'cached'})")
else:
    A = {}
''')

code(r'''
HEAD_A = R.headline(A) if A else pd.DataFrame()
HEAD_A
''')

# ── Stage B ──────────────────────────────────────────────────────────────────
md(r"""
---
# Stage B · Direct baselines

U-Net and DeepLabV3+ with ImageNet ResNet-50 encoders, trained on the **same
recipe** as the SegFormers — same loss, same LR split, same poly schedule, same
val-swept threshold applied once to test. Holding the recipe fixed is the whole
point of a fair baseline.

The best seed is chosen on **validation**, never on test. Picking the seed that
happens to score best on the 185 test images would leak the test set into model
selection.

**nnU-Net was never run on the canonical 697/134 split.** It is registered as a
gap rather than omitted, because "the baseline we did not run" is exactly the
fact a reader needs.
""")

code(r'''
trained_B = train_missing("B", reg, CFG) if "B" in RUN_STAGES else []
''')

code(r'''
if "B" in RUN_STAGES:
    if RECOMPUTE_FROM_WEIGHTS:
        B = {}
        sel = env.results / "baselines" / "baselines_bestseed_val_selected.csv"
        for _, row in pd.read_csv(sel).iterrows():
            run = reg.get(f"{row.model}__seed{int(row.seed)}")
            if run is None or run.tier != WEIGHTS:
                continue
            t0 = time.time()
            B[row.model] = R.normalize(L.score_run(env, run, CFG, MAN640, MAN["test"]), META)
            print(f"  {row.model:<24} re-inferred from {run.run_id} in {time.time()-t0:.0f}s")
        for m, df in R.load_stage_b(env, reg, META).items():
            B.setdefault(m, df)
    else:
        B = R.load_stage_b(env, reg, META)
    print(f"\nStage B: {len(B)} baselines loaded")
else:
    B = {}

HEAD_B = R.headline(B) if B else pd.DataFrame()
HEAD_B
''')

# ── Stage C ──────────────────────────────────────────────────────────────────
md(r"""
---
# Stage C · SegFormer-B5 distillation

The B5 teacher and every completed KD arm, scored on the same 185 test images
against the **B2→B0 reference** — the distilled student from Stage A. An arm is
only included if it wrote a `DONE.json`; a directory with a checkpoint but no
completion marker is an interrupted run whose weights are a snapshot, not a
result.
""")

code(r'''
if "C" in RUN_STAGES:
    C = R.load_stage_c(env, reg, META)
    HEAD_C = R.headline(C, label="arm")
    print(f"Stage C: {len(C)} distinct arms/teachers with per-image results")

    # Some entries are the same weights under two names -- the promoted B5 teacher
    # IS the val-winning B5 seed. Collapsed, and reported so nothing is hidden.
    ALIASES = R.alias_report(C)
    for _, a in ALIASES.iterrows():
        print(f"  alias  {a.alias}  ==  {a.kept}  (identical per-image Dice; counted once)")

    for r in reg.to_train("C"):
        print(f"  GAP    {r.run_id:<32} {r.note}")
else:
    C, HEAD_C = {}, pd.DataFrame()
HEAD_C
''')

code(r'''
# Every arm against the B2->B0 reference, with a paired subject-level bootstrap.
#
# WIN / NON-INFERIOR / INFERIOR rather than a bare p-value: with 28 subjects the
# question that can actually be answered is "is this at least as good", not "is
# this significantly different". The non-inferiority margin is one Dice point.
REF_NAME = "segformer_b0_distilled"      # the B2->B0 student from Stage A
MARGIN = 0.01                             # non-inferiority margin, one Dice point

SCORE_C = pd.DataFrame()
if REF_NAME in C:
    ref = C[REF_NAME]
    rows_c = []
    for name, df in C.items():
        if name == REF_NAME:
            continue
        c = R.paired_contrast(df, ref, name, REF_NAME, n_boot=CFG["n_boot"])
        # Ordered so the strongest claim is tested first: an interval entirely
        # above zero is a win; one entirely below -MARGIN is a real regression;
        # anything else is indistinguishable at this sample size and is labelled
        # NON-INFERIOR rather than being rounded up to a win.
        c["verdict"] = ("WIN" if c["lo"] > 0 else
                        "INFERIOR" if c["hi"] < -MARGIN else "NON-INFERIOR")
        rows_c.append(c)
    if rows_c:
        SCORE_C = (pd.DataFrame(rows_c).sort_values("delta", ascending=False)
                   .reset_index(drop=True))
    print(f"{len(SCORE_C)} arms scored against {REF_NAME} "
          f"(subjects = {SCORE_C.n_subjects.iloc[0] if len(SCORE_C) else 0})")
else:
    print(f"reference {REF_NAME!r} not among the Stage C tables: {sorted(C)}")
SCORE_C
''')

# ── Stage E ──────────────────────────────────────────────────────────────────
md(r"""
---
# Stage E · Mobile baselines, and Stage F · the distilled students

Four lightweight architectures at the scale you would actually deploy on a phone,
plus two distilled variants:

| Model | Params | Pretrained init | Trained |
|---|---|---|---|
| **PP-MobileSeg-Tiny** | 1.45 M | StrideFormer-Tiny backbone, ImageNet | direct |
| **TopFormer-Tiny** | 1.37 M | ImageNet backbone — **manual download** | direct |
| **LR-ASPP MobileNetV3** | 3.22 M | MobileNetV3-Large backbone, ImageNet | direct |
| **Fast-SCNN** | 1.14 M | none — trains from scratch **by design** | direct |
| **Fast-SCNN (distilled)** | 1.14 M | none — same scratch init | **KD from DeepLabV3+/R50** |
| **LR-ASPP (distilled)** | 3.22 M | same ImageNet init as direct | **KD from DeepLabV3+/R50** |
| **TopFormer (distilled)** | 1.37 M | same ImageNet init as direct | **KD from DeepLabV3+/R50** |
| **PP-MobileSeg (distilled)** | 1.45 M | same ImageNet init as direct | **KD from DeepLabV3+/R50** |

The last two arms were added with Stage H. Without them, "SegFormer-B2 is the
better teacher" — a Stage H claim — would rest on two students rather than four.
Same teacher, same recipe, same `efficient_alpha`; only the student moves.

## Why there are distilled arms, and why their teacher is DeepLabV3+

**Why Fast-SCNN is a student.** It is the only model in the study with no
pretrained initialisation, and the weakest of the four (0.618 Dice, 3.42 % misses
in the numbers carried into the handbook). KD substituting for absent pretraining
is the mechanism most likely to produce a gap larger than the ~0.05 Dice noise
floor — i.e. the one arm here with a realistic chance of a result rather than
another NON-INFERIOR.

**Why LR-ASPP is the second student.** The Fast-SCNN arm came back null on Dice
(Δ = −0.005, CI crossing 0) with complete misses **worse** (7 → 13) and a
significant skin-tone fairness regression — but that arm cannot distinguish "KD
does not help here" from "KD cannot rescue a scratch-init student on 697 images".
LR-ASPP is the opposite pole: the strongest mobile student, ImageNet-initialised,
already at the annotation ceiling on Dice. Same teacher, same recipe, same seeds —
the pair isolates student initialisation as the variable. The claims to check are
**complete-miss rate and the fairness gap**, not mean Dice, which the ceiling has
already capped.

**Why not SegFormer as the teacher.** The MiT weights in `pretrained_weights/`
carry NVIDIA's `license: other`, which is non-commercial. Distilling from them
puts a non-commercial link in the student's provenance. DeepLabV3+/R50 arrives
through `segmentation_models_pytorch` (MIT) on a torchvision ResNet-50 (BSD-3).
That removes the one unambiguously non-commercial licence — it does **not** by
itself make the result commercialisable, since the ImageNet provenance of the
encoder and, far more importantly, the consent scope of the photographs are
separate questions.

**Why not U-Net.** They are indistinguishable on Dice (0.7584 vs 0.7570 on the
val-selected seed of each, U-Net's median actually higher) — inside the noise
floor. They are not indistinguishable on complete misses, and for a *teacher*
that is the metric that matters: a teacher that misses a bruise hands the student
a confidently **empty** soft target, not merely a weak one. Paired over all 185
test images:

```
images DeepLab misses that U-Net finds:  0
images U-Net misses that DeepLab finds:  2
both miss:                               5
```

Strict containment, and the ordering holds on every seed (DeepLab 5/5/2 misses,
U-Net 9/7/7).

**The teacher is calibrated first.** The Stage B baselines were trained as
endpoints, not teachers, so they carry no `calibration.json`. It is fitted on the
134 val images with the same `engine.calibrate_temperature` (Guo et al. 2017) the
B2 teacher used, and written under the work dir — never into `checkpoints/`,
which is the verified artefact that produced the published Stage B table.

**Two of the four architectures are vendored verbatim** from their reference
implementations (OpenMMLab's StrideFormer, hustvl's TopFormer) behind a small
mmcv shim, so the published checkpoints load by key rather than by hope. The
check below reports how many of each official checkpoint's tensors found an
exact name-and-shape match.

**Initialisation is part of the experiment.** Every other baseline in this study
starts from an ImageNet-pretrained encoder with a fresh head. A model that ends
up training from scratch is labelled as such everywhere it appears, because on
697 training images the difference is large and would otherwise be invisible.
""")

code(r'''
from bruisekit import efficient_models as EM
from bruisekit import weights as W

# Downloads into pretrained_weights/efficient/, resumably: a dropped connection
# costs the bytes in flight, not the whole file. Cached files are checksum-verified
# rather than re-fetched.
PROV = W.provision_all(env) if "E" in RUN_STAGES else pd.DataFrame()
PROV[["model", "status", "init"]] if len(PROV) else "Stage E not selected"
''')

code(r'''
# Structural verification, before any GPU time is spent.
#
#   self_test           builds all four, checks they emit [B,1,640,640] from raw
#                       [0,1], and diffs parameter counts against the published
#                       figures rescaled to a 1-class head.
#   verify_checkpoint   the decisive one: loads each official checkpoint and counts
#                       tensors matching by BOTH name and shape. `unexpected = 0`
#                       with only the decode head missing means the vendored
#                       architecture IS the published one.
if "E" in RUN_STAGES:
    SELFTEST_E = EM.self_test()
    print()
    CKPT_E = EM.verify_checkpoint_match(env)
else:
    SELFTEST_E = CKPT_E = pd.DataFrame()
''')

code(r'''
# Stage F wiring. Two steps, deliberately gated differently.
#
#   register_student_aliases makes each distilled family (fastscnn_distilled,
#                            lraspp_mobilenetv3_distilled) resolve to its
#                            ARCHITECTURE at the call sites that key off a family
#                            name. Needed whenever Stage E is read, training or
#                            not, because the best-seed table below looks the
#                            family up in weights.SOURCES.
#
#   install_teacher_shim     redirects engine.load_teacher to DeepLabV3+/R50 and
#                            makes it read the teacher's architecture from the
#                            family spec instead of the hardcoded SegFormer-B2 at
#                            engine.py:147. Only meaningful when training.
#
# The shim lives in bruisekit/distill_efficient.py rather than in engine.py: engine
# is extracted verbatim from bruise_colab_baselines.ipynb at build time (handbook
# 16), so an edit there is reverted by the next 60_build_unified_bundle.py. This
# is the same pattern efficient_models.install_efficient_shim already uses.
from bruisekit import distill_efficient as DE

if "E" in RUN_STAGES:
    DE.register_student_aliases()

if "E" in RUN_STAGES and ALLOW_TRAINING:
    # The DeepLabV3+ teacher is an SMP model, so calibrating it needs
    # segmentation_models_pytorch -- which §1 deliberately did not install, since
    # a Stage E session whose distilled arms are all trained never touches it.
    # This is the first line that certainly does.
    if reg.to_train("E"):
        _need_optional("segmentation_models_pytorch")
    DE.install_teacher_shim(env, CFG)
    # Fit T on the 134 val images, once per teacher seed this session will ask for.
    # Up front, so a missing temperature fails in seconds rather than after a model
    # is built. Idempotent -- a second run reads the cached file.
    for _s in EFFICIENT_SEEDS:
        DE.ensure_calibration(env, "deeplabv3plus_r50", _s, CFG, MAN640)
''')

code(r'''
# Route the efficient architectures into the shared training loop.
#
# `env` is passed deliberately: train_run builds its model through build_model and
# starts optimising immediately, so this is the ONLY point at which pretrained
# backbones can be attached. Without env every Stage E model would train from
# random init -- silently, and with a large penalty on 697 images.
#
# Everything else about training is unchanged from Stages A and B: the same loss,
# LR split, poly schedule, early stopping, and the same resume contract
# (DONE.json to skip, resume.pt every few epochs to continue). The distilled arm
# takes the same path -- train_run already implements distillation when the spec
# says distill=True; the only thing Stage F changed is WHICH teacher it loads.
if "E" in RUN_STAGES and (ALLOW_TRAINING or RECOMPUTE_FROM_WEIGHTS):
    EM.install_efficient_shim(env, verbose=True)
    print("build_model now routes:", ", ".join(EM.EFFICIENT_ARCHS))

trained_E = train_missing("E", reg, CFG) if "E" in RUN_STAGES else []
''')

code(r'''
# Fit the threshold on VALIDATION, then score test once at that cut.
#
# Only runs for freshly trained models: a shipped checkpoint already carries its
# operating_point.json, and re-fitting would be re-deriving a number we were given.
if trained_E:
    import torch
    from bruisekit.data import make_loader
    from bruisekit.sweep import cache_logits, sweep_cuts, select_cut
    from bruisekit.evaluate import evaluate_at_cut
    import json as _json

    # Same three steps the Stage A/B threshold cells use, in the same order:
    #   cache_logits -> (logits, gts, stems)   one forward pass, logits kept
    #   sweep_cuts   -> a table of per-cut Dice, SE and miss rate
    #   select_cut   -> the tie band's lowest-miss cut, NOT the argmax
    # The band matters: these sweeps are flat enough that argmax fits val noise.
    CUTS = np.linspace(CFG["cut_min"], CFG["cut_max"], CFG["cut_steps"])
    e_rows_test = []
    for run_id in trained_E:
        run = reg.get(run_id)
        rd = env.runs / run_id
        model = EM.build_with_pretrained(env, run.family, 1, verbose=False)
        model.load_state_dict(torch.load(str(rd / "best.pt"), map_location=env.device,
                                         weights_only=True))
        model.to(env.device).eval()

        logits, gts, _ = cache_logits(
            model, make_loader(MAN640["val"], env.cache640, CFG["img_size"],
                               CFG["eval_batch"], False, CFG["workers"], 0),
            env.device, CFG["amp"])
        grid = sweep_cuts(logits, gts, CUTS)
        sel = select_cut(grid)
        grid.to_csv(rd / "threshold_sweep.csv", index=False)
        (rd / "operating_point.json").write_text(_json.dumps(sel, indent=2))

        tpi, summ = evaluate_at_cut(
            model, make_loader(MAN640["test"], env.cache640, CFG["img_size"],
                               CFG["eval_batch"], False, CFG["workers"], 0),
            env.device, sel["cut"], CFG["amp"])
        tpi.to_csv(rd / "test_per_image.csv", index=False)
        e_rows_test.append({"run_id": run_id, "model": run.family, "seed": run.seed,
                            "cut": sel["cut"], **summ})
        print(f"  {run_id:<28} cut {sel['cut']:+.3f} -> test Dice {summ['mean_dice']:.4f} "
              f"(miss {summ['complete_miss_rate']*100:.2f}%)")
        del model, logits, gts
        if str(env.device).startswith("cuda"):
            torch.cuda.empty_cache()

    # Written where the registry's RESULTS tier looks for it, so a later session
    # finds these numbers even without the checkpoints.
    E_TEST = pd.DataFrame(e_rows_test)
    (env.results / "efficient").mkdir(parents=True, exist_ok=True)
    E_TEST.to_csv(env.results / "efficient" / "efficient_test_per_seed.csv", index=False)
    reg.scan()
    display(E_TEST)
''')

code(r'''
# Best seed per family, selected on validation where an operating point records it,
# otherwise the seed with the most complete evidence. Mirrors Stage B.
if "E" in RUN_STAGES:
    E_TABLES, e_rows = {}, []
    from bruisekit.registry import EFFICIENT_FAMILIES as _EFAM
    for family in _EFAM:
        best, best_val = None, -1.0
        for seed in EFFICIENT_SEEDS:
            run = reg.get(f"{family}__seed{seed}")
            if run is None or run.per_image is None:
                continue
            op = (run.weights.parent / "operating_point.json") if run.weights else None
            val = json.loads(op.read_text()).get("val_dice_at_cut", -1.0) if (op and op.exists()) else -1.0
            if val > best_val:
                best, best_val = run, val
        if best is not None:
            E_TABLES[family] = R.normalize(pd.read_csv(best.per_image), META)
            e_rows.append({"model": family, "seed": best.seed,
                           "val_dice": round(best_val, 4) if best_val >= 0 else None,
                           "init": W.SOURCES[family].init})
    HEAD_E = R.headline(E_TABLES) if E_TABLES else pd.DataFrame()
    if e_rows:
        display(pd.DataFrame(e_rows))
    if not E_TABLES:
        _todo_e = reg.to_train("E")
        print(f"No Stage E results yet. Set ALLOW_TRAINING = True to train "
              f"{len(_todo_e)} run(s) at seeds {EFFICIENT_SEEDS} "
              f"(~{sum(r.cost_hours for r in _todo_e):.1f} GPU-hours).")
else:
    E_TABLES, HEAD_E = {}, pd.DataFrame()
HEAD_E
''')

# ── Stage H ──────────────────────────────────────────────────────────────────
md(r"""
---
# Stage H · Reliability-gated distillation, and the SegFormer-B2 teacher axis

Two changes, deliberately separable, crossed into a design that neither one alone
could resolve.

| | plain response KD | **reliability-gated KD** |
|---|---|---|
| **DeepLabV3+/R50 teacher** | Stage F — `*_distilled` | — (not run; see below) |
| **SegFormer-B2 teacher** | `*_b2kd` | `*_rgkd` |

**Nine families × 3 seeds = 27 runs, ≈24 GPU-hours.** Five gated arms
(SegFormer-B0, LR-ASPP, Fast-SCNN, TopFormer, PP-MobileSeg) and four
teacher-matched controls. The YOLO gated arm is built but not registered — see
below.

## The failure this stage exists to fix

Stage F recorded a specific, *pre-registered* failure. The LR-ASPP student
distilled from DeepLabV3+ gained Dice but went from **3 complete misses to 5**,
and Fast-SCNN did the same thing harder — **7 → 13 misses**, with the skin-tone
fairness gap widening from 0.136 to 0.199 (Kruskal–Wallis p = 0.0017). §7b.6 had
written down in advance why: the teacher misses more than the student does, so the
student inherits some of the teacher's miss behaviour along with its ranking.

The mechanism is not subtle. `DistillLoss` regresses the student onto
`sigmoid(teacher_logits / T)` at **every pixel of every image with one fixed
weight**. On an image the teacher completely misses, that soft target is not weak
— it is confidently, uniformly *empty*, applied with exactly the force it gets on
an image the teacher got right. The student is being told, with full confidence,
that there is no bruise on precisely the images the clinical metric exists to
catch.

## What reliability gating does

**Per pixel**, with `p` the calibrated teacher probability and `y` the label:

```
r = 1 − |2p − 1| · |p − y|
```

| teacher is… | confidence | error | `r` | soft term |
|---|---|---|---|---|
| confidently right | ≈1 | ≈0 | ≈1 | full weight |
| **uncertain** | ≈0 | any | **≈1** | **full weight** |
| confidently wrong | ≈1 | ≈1 | ≈0 | suppressed |

The middle row is the whole design. Down-weighting by error *alone* would delete
exactly the pixels the teacher is unsure about — which is the dark knowledge
distillation exists to transfer. Multiplying by confidence keeps uncertainty
intact and removes only assertive error.

**Per image**, the gate is the teacher's own soft Dice against the label:

```
g = clip((dice_T − gate_lo) / (gate_hi − gate_lo), 0, 1)      lo = 0.10, hi = 0.50
```

An image the teacher misses gets `g = 0` and trains on hard labels alone — which
is what the *direct* baseline gives it, and the direct baseline does not have the
miss. Soft Dice, so no threshold is involved: the operating point is not fitted
until after training, and a gate that depended on one would entangle this arm with
threshold choice.

## Why this is one variable and not two

The gated loss **reduces exactly to `losses.DistillLoss`** when `r ≡ 1` and
`g ≡ 1` — same alpha, same supervised term, same BCE against the same calibrated
teacher probability. `RK.self_test()` asserts it below rather than leaving it as a
claim, because a gate that does not reduce to its control makes every contrast in
this stage a two-variable comparison.

Two consequences follow, and both are deliberate:

- **alpha is not re-tuned.** Each `*_rgkd` arm inherits its control's
  `segformer_alpha` / `efficient_alpha` / `yolo_alpha`. A gated arm with its own
  alpha would be two changes at once — the same reason §3 holds the LR fixed
  across architectures.
- **The freed weight goes back to the supervised term.**
  `alpha_eff = alpha + (1−alpha)(1−coverage)`, so gating cannot silently lower the
  effective KD weight and masquerade as an alpha change. At `coverage = 1` this is
  `DistillLoss` term for term; at `coverage = 0` it is `SupervisedLoss`.

## Why every arm has a teacher-matched control

`fastscnn_distilled` uses DeepLabV3+, so scoring `fastscnn_rgkd` against it would
move the teacher **and** the gate. `fastscnn_b2kd` is plain `DistillLoss` from B2
— identical to `fastscnn_distilled` except for the teacher — so the two contrasts
decompose cleanly:

```
*_b2kd  vs  *_distilled      what changing the teacher does
*_rgkd  vs  *_b2kd           what the gate does, teacher held fixed
```

B0 already has a plain-B2 control in Stage A (`segformer_b0_distilled`), so it
needs only the gated arm. TopFormer and PP-MobileSeg get the DeepLabV3+ arms
Stage F never defined, which takes the teacher axis from 2×2 students to 2×4.

**Why no gated DeepLabV3+ arm.** It would be the tidiest fourth cell and it is
left empty on purpose rather than quietly omitted: 2×2×4 is 24 arms before seeds,
and "does gating fix the inherited-miss failure" is answerable within one teacher.
If it works, that cell is the obvious next run — `TEACHER_FOR` is one line per arm.

## The YOLO arm is implemented but **not registered**

YOLO never sees a loss object; the teacher reaches it only through the pseudo-mask
baked into the training labels (§4). `reliability_kd.build_gated_yolo_dataset`
implements the gate there, promoting `alpha` from a scalar to a per-pixel quantity
that rises to 1 — pure ground truth — exactly where the teacher is unreliable:

```
a_pix = alpha + (1 − alpha)·(1 − g·r)
class = (a_pix·gt + (1 − a_pix)·teacher_prob ≥ 0.5)
```

At `g·r = 1` this is Stage A's fusion character for character. It is tested — but
`yolo_sem_rgkd` is **absent from every arm table**, so nothing schedules it and a
Stage E+H session never needs Ultralytics installed.

**State the cost when you report.** Every confirmatory contrast below now tests
the gate on an **online loss** only; the study says nothing about whether it
transfers to the offline pseudo-mask route. Holm corrects over **three**
contrasts, not four. Re-enabling is three lines — see `reliability_kd`'s module
docstring.

## What to read, and in what order

**`GATE_H` before `HEAD_H`.** An arm whose gate never fired is a relabelled
control; an arm whose gate fired on half the images is a different experiment.
Dice cannot tell those apart.

**Complete misses before Dice.** Stage F's arms won on the endpoint they were not
built to win on and left their actual endpoint untouched (§7b.7). This stage's
pre-registered endpoint is **miss rate and the fairness gap**. If Dice moves and
misses do not, say so.
""")

code(r'''
from bruisekit import reliability_kd as RK

# Prove the three properties every Stage H contrast depends on. Seconds, CPU only,
# and it raises rather than printing a red row -- a silently-broken gate produces
# entirely plausible numbers.
if "H" in RUN_STAGES:
    SELFTEST_H = RK.self_test()
else:
    SELFTEST_H = pd.DataFrame()
''')

code(r'''
# Stage H wiring. Three steps, gated differently for the same reason Stage F's are.
#
#   register_student_aliases  makes each mobile arm resolve to its ARCHITECTURE at
#                             the call sites keyed by family name. Needed whenever
#                             Stage H is READ, training or not.
#   install_loss_shim         rebinds engine.DistillLoss to a dispatcher that
#                             returns the gated loss for `*_rgkd` arms and the
#                             untouched original for everything else.
#   install_teacher_shim      rebinds engine.load_teacher to resolve this arm's
#                             teacher from RK.TEACHER_FOR. Installed LAST so it
#                             sits in front of Stage F's shim and falls through to
#                             it for non-Stage-H arms.
#
# Both shims dispatch on RK.ACTIVE_ARM, which train_missing sets with RK.arm().
# Outside that context they are inert, so installing them cannot change a Stage
# A/B/E/F number.
if "H" in RUN_STAGES:
    RK.register_student_aliases()

if "H" in RUN_STAGES and ALLOW_TRAINING:
    RK.install_loss_shim(CFG["rgkd_gate_lo"], CFG["rgkd_gate_hi"])
    RK.install_teacher_shim(env, CFG)

    # Resolve every temperature this session will ask for, UP FRONT. B2 ships one
    # next to each checkpoint (engine.train_run writes it), so Stage H's students
    # distil from exactly the temperature Stage A's did -- two temperatures for one
    # checkpoint would make the two stages' arms incomparable. DeepLabV3+ has none
    # and is fitted on the 134 val images, cached under the work dir.
    _H_SEEDS = EFFICIENT_SEEDS if RGKD_SEEDS is None else RGKD_SEEDS
    _needed = sorted({RK.TEACHER_FOR[r.family] for r in reg.to_train("H")}
                     | {"deeplabv3plus_r50"
                        for r in reg.to_train("E")
                        if r.family in RK.TEACHER_FOR})
    # Only a DeepLabV3+ teacher pulls in SMP. Every gated arm distils from B2,
    # which builds from the local HF config in pretrained_weights/ and needs
    # nothing installed -- so a pure Stage H session stays offline-clean.
    if any(t.endswith("_r50") for t in _needed):
        _need_optional("segmentation_models_pytorch")
    for _tf in _needed:
        for _s in _H_SEEDS:
            RK.teacher_temperature(env, _tf, _s, CFG, MAN640)
''')

code(r'''
# Stage H trains through the SAME train_missing as every other stage -- the shared
# recipe, LR split, early stopping and resume contract are all inherited, and the
# only thing that differs is what RK.arm() puts in front of the loss and the
# teacher loader.
if "H" in RUN_STAGES and ALLOW_TRAINING and not str(env.device).startswith("cuda"):
    print("Stage H needs a GPU: every arm runs a frozen teacher on every step.")

trained_H = train_missing("H", reg, CFG) if "H" in RUN_STAGES else []
''')

code(r'''
# Fit the threshold on VALIDATION, then score test once at that cut.
#
# Deliberately a copy of the Stage E threshold cell's call sequence rather than a
# fresh one (handbook 15, trap 11 -- the Stage E cell was itself written that way
# and shipped with two signature errors that only a GPU run could catch). The one
# addition is the YOLO branch: yolo_sem_rgkd is scored by native argmax, which is
# parameter-free and therefore has no threshold to sweep.
if trained_H:
    import torch
    from bruisekit.data import make_loader
    from bruisekit.sweep import cache_logits, sweep_cuts, select_cut
    from bruisekit.evaluate import evaluate_at_cut
    from bruisekit import efficient_models as EM
    import bruisekit.yolo_native as yn
    import json as _json

    CUTS = np.linspace(CFG["cut_min"], CFG["cut_max"], CFG["cut_steps"])
    h_rows_test = []
    for run_id in trained_H:
        run = reg.get(run_id)
        rd = env.runs / run_id

        if run.kind == "yolo":
            best_pt = rd / "ultralytics_runs" / "train" / "weights" / "best.pt"
            tpi, summ = yn.predict_native_argmax(best_pt, MAN["test"], env.data,
                                                 CFG["img_size"])
            tpi.to_csv(rd / "test_per_image_native_argmax.csv", index=False)
            h_rows_test.append({"run_id": run_id, "model": run.family, "seed": run.seed,
                                "cut": None, "path": "native_argmax", **summ})
            print(f"  {run_id:<30} native argmax -> test Dice {summ['mean_dice']:.4f} "
                  f"(miss {summ['complete_miss_rate']*100:.2f}%)")
            continue

        if run.kind == "segformer":
            spec = L.spec_for(run.family)
            from bruisekit.models import build_model as _bm
            model = _bm(spec["arch"], spec["size"], env.paths_for_models())
        else:
            model = EM.build_with_pretrained(env, run.family, 1, verbose=False)
        model.load_state_dict(torch.load(str(rd / "best.pt"), map_location=env.device,
                                         weights_only=True))
        model.to(env.device).eval()

        logits, gts, _ = cache_logits(
            model, make_loader(MAN640["val"], env.cache640, CFG["img_size"],
                               CFG["eval_batch"], False, CFG["workers"], 0),
            env.device, CFG["amp"])
        grid = sweep_cuts(logits, gts, CUTS)
        sel = select_cut(grid)
        grid.to_csv(rd / "threshold_sweep.csv", index=False)
        (rd / "operating_point.json").write_text(_json.dumps(sel, indent=2))

        tpi, summ = evaluate_at_cut(
            model, make_loader(MAN640["test"], env.cache640, CFG["img_size"],
                               CFG["eval_batch"], False, CFG["workers"], 0),
            env.device, sel["cut"], CFG["amp"])
        tpi.to_csv(rd / "test_per_image.csv", index=False)
        h_rows_test.append({"run_id": run_id, "model": run.family, "seed": run.seed,
                            "cut": sel["cut"], "path": "logit_cut", **summ})
        print(f"  {run_id:<30} cut {sel['cut']:+.3f} -> test Dice {summ['mean_dice']:.4f} "
              f"(miss {summ['complete_miss_rate']*100:.2f}%)")
        del model, logits, gts
        if str(env.device).startswith("cuda"):
            torch.cuda.empty_cache()

    # Written where the registry's RESULTS tier looks for it, so a later session
    # finds these numbers even without the checkpoints.
    H_TEST = pd.DataFrame(h_rows_test)
    (env.results / "rgkd").mkdir(parents=True, exist_ok=True)
    H_TEST.to_csv(env.results / "rgkd" / "rgkd_test_per_seed.csv", index=False)
    reg.scan()
    display(H_TEST)
''')

md(r"""
## H1 · What the gate actually did

**Read this before any Dice number in this stage.** An arm whose gate never fired
is a relabelled control, and an arm whose gate fired on half its images is a
different experiment — the accuracy table cannot distinguish those two, and this
one can.

| column | reading |
|---|---|
| `mean_coverage` | mean of `g·r` over all pixels. **1.0 means the gate was inert.** |
| `mean_alpha_effective` | the alpha the run actually trained at. Compare to `alpha_nominal`. |
| `frac_images_fully_gated_off` | image-views where the teacher was ignored entirely |
| `mean_teacher_soft_dice` | how good the teacher was on the augmented batches |

`images_seen` counts image *views*, not images: each of the 697 training images is
seen once per epoch under fresh augmentation, and the gate is recomputed each time
because a crop that removes most of a bruise genuinely is a view the teacher is
less reliable on.

**Two failure directions, and they look identical in an accuracy table.**

- `mean_coverage → 1.0` — the gate never fired. The arm *is* its control. That is
  a finding ("the B2 teacher is reliable enough here that gating has nothing to
  remove"), and a more useful one than a 0.003 Dice difference — but it must not
  be reported as a distinct method.
- `mean_coverage → 0` — the gate ate the arm, `alpha_eff → 1`, and the run trained
  as a supervised baseline. Check `mean_teacher_soft_dice` against B2's test Dice
  (0.769 mean, 0.819 median): if it is far lower, the teacher is scoring badly on
  *augmented* batches and `rgkd_gate_lo/hi` are calibrated for the wrong
  distribution. Lower them and retrain — do not reinterpret the result.

The cell prints a loud line for either case. **This is the first number to look
at, before any Dice in the stage.**
""")

code(r'''
if "H" in RUN_STAGES:
    _H_SEEDS = EFFICIENT_SEEDS if RGKD_SEEDS is None else RGKD_SEEDS
    GATE_H = RK.gate_report(env, seeds=_H_SEEDS)
    if len(GATE_H):
        display(GATE_H[["family", "seed", "alpha_nominal", "mean_alpha_effective",
                        "mean_coverage", "mean_pixel_reliability", "mean_image_gate",
                        "mean_teacher_soft_dice", "frac_images_fully_gated_off",
                        "images_seen"]].round(4))
        # Both failure directions, because they look identical in a Dice table and
        # mean opposite things.
        for _, _r in GATE_H.iterrows():
            _tag = f"{_r.family}__seed{int(_r.seed)}"
            if _r.mean_coverage > 0.999:
                print(f"  !! {_tag}: coverage {_r.mean_coverage:.4f} -- the gate "
                      f"NEVER FIRED. This arm IS its control under another name. "
                      f"Report that as the finding; do not report it as a method.")
            elif _r.mean_coverage < 0.10:
                print(f"  !! {_tag}: coverage {_r.mean_coverage:.4f} -- the gate ate "
                      f"the arm (alpha_eff {_r.mean_alpha_effective:.3f} vs nominal "
                      f"{_r.alpha_nominal}). This trained as a supervised baseline, "
                      f"not as distillation. Check mean_teacher_soft_dice: if the "
                      f"teacher is scoring far below its test Dice on augmented "
                      f"batches, rgkd_gate_lo/hi are set for the wrong distribution.")
    else:
        print("No gate diagnostics on disk yet -- they are written by a training run.")
else:
    GATE_H = pd.DataFrame()
''')

code(r'''
# Best seed per Stage H family, selected on VALIDATION where an operating point
# records it. Mirrors Stage B and Stage E exactly; YOLO has no operating point, so
# it falls back to the seed with the most complete evidence, as Stage A does.
if "H" in RUN_STAGES:
    _H_SEEDS = EFFICIENT_SEEDS if RGKD_SEEDS is None else RGKD_SEEDS
    H_TABLES, h_rows = {}, []
    for family in RK.STAGE_H_FAMILIES:
        best, best_val = None, -1.0
        for seed in _H_SEEDS:
            run = reg.get(f"{family}__seed{seed}")
            if run is None or run.per_image is None:
                continue
            op = (run.weights.parent / "operating_point.json") if run.weights else None
            val = json.loads(op.read_text()).get("val_dice_at_cut", -1.0) \
                if (op and op.exists()) else -1.0
            if val > best_val:
                best, best_val = run, val
        if best is not None:
            H_TABLES[family] = R.normalize(pd.read_csv(best.per_image), META)
            h_rows.append({"model": family, "seed": best.seed,
                           "val_dice": round(best_val, 4) if best_val >= 0 else None,
                           "teacher": RK.TEACHER_FOR[family],
                           "gated": RK.is_gated(family),
                           "control": RK.CONTROL_FOR.get(family)})
    HEAD_H = R.headline(H_TABLES) if H_TABLES else pd.DataFrame()
    if h_rows:
        display(pd.DataFrame(h_rows))
    if not H_TABLES:
        _todo_h = reg.to_train("H")
        print(f"No Stage H results yet. Set ALLOW_TRAINING = True to train "
              f"{len(_todo_h)} run(s) at seeds {_H_SEEDS} "
              f"(~{sum(r.cost_hours for r in _todo_h):.1f} GPU-hours).")
else:
    H_TABLES, HEAD_H = {}, pd.DataFrame()
HEAD_H
''')

md(r"""
## H2 · Each arm against its teacher-matched control

One row per contrast, each one moving exactly one thing. `delta` is `a − b`, so a
positive number favours the arm on the left.

The **full** Stage H family — with its own Holm correction over its own
confirmatory set — is scored in Stage G under `FAM_H`. This table is the local
readout, at Stage D's `n_boot`, so the stage can be read on its own; the
confirmatory numbers to quote are Stage G's.

Multiplicity is corrected **within `CONTRAST_FAMILY_H`, separately from
`CONTRAST_FAMILY`**. Appending these to the existing confirmatory set would
re-penalise a contrast that has already been conducted and reported (the Stage F
LR-ASPP arm at Holm p = 0.0042, k = 3). Correction is over the comparisons made to
answer *one* question; this stage asks a different one.
""")

code(r'''
# Every pair whose control is on disk. Missing controls are NAMED, never dropped:
# a shorter table that looks complete is the failure mode this guards against.
SCORE_H = pd.DataFrame()
if "H" in RUN_STAGES and H_TABLES:
    _pool = {**H_TABLES, **A, **B, **E_TABLES}
    rows_h, skipped_h = [], []
    for _arm_name, _ctrl in RK.CONTROL_FOR.items():
        if _arm_name not in _pool or _ctrl not in _pool:
            missing = [n for n in (_arm_name, _ctrl) if n not in _pool]
            skipped_h.append(f"{_arm_name} vs {_ctrl} -- missing {', '.join(missing)}")
            continue
        c = R.paired_contrast(_pool[_arm_name], _pool[_ctrl], _arm_name, _ctrl,
                              n_boot=CFG["n_boot"])
        c["moves"] = ("the gate, teacher held fixed" if RK.is_gated(_arm_name)
                      else "the teacher, KD method held fixed")
        rows_h.append(c)
    if rows_h:
        SCORE_H = pd.DataFrame(rows_h).sort_values("delta", ascending=False).reset_index(drop=True)
        display(SCORE_H.round(4))
    if skipped_h:
        print(f"\n{len(skipped_h)} contrast(s) skipped:")
        for s in skipped_h:
            print("   ", s)
''')

code(r'''
# The pre-registered endpoint: complete misses, as COUNTS, and whether the arm's
# misses are contained in its control's. At 0-13 misses of 185 a bootstrapped rate
# difference near the boundary is unstable and is not the thing to quote (handbook
# 8b.5) -- and the specific question 7b.7 left open is whether a distilled arm's
# NEW misses land on images its teacher also misses.
DISC_H = pd.DataFrame()
if "H" in RUN_STAGES and H_TABLES:
    from bruisekit import significance as SG
    _pool = {**H_TABLES, **A, **B, **E_TABLES}
    rows_dh = []
    for _arm_name, _ctrl in RK.CONTROL_FOR.items():
        if _arm_name in _pool and _ctrl in _pool:
            rows_dh.append(SG.discordance(_pool[_arm_name], _pool[_ctrl], _arm_name, _ctrl))
    # The teacher itself, against every arm that distilled from it: this is the
    # containment check 7b.7 says is required before a miss movement can be called
    # benign rather than merely unexplained.
    for _arm_name, _teacher in RK.TEACHER_FOR.items():
        if _arm_name in _pool and _teacher in _pool:
            d = SG.discordance(_pool[_arm_name], _pool[_teacher], _arm_name, _teacher)
            d["note"] = "arm vs its own teacher"
            rows_dh.append(d)
    if rows_dh:
        DISC_H = pd.DataFrame(rows_dh)
        display(DISC_H[["a", "b", "a_total", "b_total", "a_misses_b_finds",
                        "b_misses_a_finds", "both_miss", "containment",
                        "mcnemar_exact_p"]].round(4))
''')

code(r'''
# Fairness is the stage's OTHER pre-registered endpoint, and Stage F shipped
# without it -- 7b.7's list of what to run next has it at (iii). Computed here so
# it cannot be forgotten again.
FAIR_H = pd.DataFrame()
if "H" in RUN_STAGES and H_TABLES:
    _pool = {**H_TABLES, **E_TABLES}
    rows_fh = []
    for _name, _df in _pool.items():
        g = R.fairness_gap(_df)
        rows_fh.append({"model": _name, "gated": RK.is_gated(_name),
                        "teacher": RK.TEACHER_FOR.get(_name), **g})
    FAIR_H = pd.DataFrame(rows_fh).sort_values("model").reset_index(drop=True)
    display(FAIR_H.round(4))
''')

md(r"""
## H3 · How to read Stage H

**`GATE_H` first, always.** `mean_coverage = 1.0` means the gate never fired and
the arm is its control under another name. Report that as the result if it happens
— "the B2 teacher is reliable enough on this dataset that gating has nothing to
remove" is a finding, and a more useful one than a 0.003 Dice difference.

**The pre-registered endpoint is misses and fairness, not Dice.** Stage F's arms
won on the endpoint they were not built to win on and left their actual endpoint
untouched (§7b.7). If that happens again, say so in those words.

**Neither `*_rgkd` nor `*_b2kd` is interpretable alone.** `*_rgkd` vs `*_b2kd`
isolates the gate; `*_b2kd` vs `*_distilled` isolates the teacher. Quoting a
`*_rgkd` arm against `*_distilled` moves both and is not a result about either.

**The annotation ceiling still applies (§D7).** Every model in this stage sits
inside the band between `paul_vs_majority` (0.700) and `gbarimah_vs_erik` (0.755).
A confirmed 0.02 Dice gain is real and is still smaller than the disagreement
between two human annotators.

**A 0-of-185 miss count is not a zero rate** (§15, trap 12). Its one-sided 95 %
bound is ≈1.6 %, which spans this entire field.
""")

# ── Stage Y · YOLO26-large ───────────────────────────────────────────────────
md(r"""
---
# Stage Y · YOLO26-large, native Ultralytics, native argmax

**The question.** `yolo26n` is the fastest model in the study (8.02 ms, 1.63 M
params, fresh-process) and has the **worst complete-miss rate of the reporting
models** — 6.5 % native-argmax at its best seed, against 0.0–0.5 % for the
SegFormers. §D3 is explicit that miss containment, not Dice, is the endpoint this
study is judged on. So the obvious question the nano arms cannot answer is
whether that miss rate is a property of *YOLO* or a property of *1.6 M
parameters*.

Stage Y answers it the only way that isolates the variable: **the identical
native recipe on the large backbone**. Same mosaic, same EMA, same letterbox,
same LR schedule, same `close_mosaic`, same seed handling — `yolo26l-sem.pt`
instead of `yolo26n-sem.pt`, and nothing else.

**Native argmax only.** The custom `/255` path exists for Stage A because that
stage needed a SegFormer-comparable geometry; it is not re-derived here. Argmax
is parameter-free, so a Stage Y run needs `best.pt` and nothing else — no
val-fitted cut, no `operating_point.json`, nothing that can be missing.

**Its own stage letter.** Stage A is quoted as "the five headline models" in the
handbook and in the paper. A table that silently grows two rows is
indistinguishable from a bug, so Stage Y is scanned, planned, costed and reported
separately — exactly the reasoning that gave Stage H its own letter.

**One arm, one seed.** Stage Y registers `yolo_sem_l_direct` and nothing else.
A distilled large arm (`yolo_sem_l_distilled`) has a spec and a cost estimate but
is deliberately unregistered — it answers a different question, and leaving it on
would make `RUN_STAGES = "...Y"` cost ~4 GPU-hours nobody asked for. One line in
`registry.STAGE_Y_FAMILIES` enables it when that question is being asked.

`YOLO_L_SEEDS = (0,)`. At ~3.5 h a run, the three-seed
habit that costs 1.5 GPU-hours in Stage E costs 21 here. One seed answers "does
capacity fix the miss rate?"; three are worth buying once the answer is yes and
you want to quote a spread. **A one-seed result is not a rate** (§15, trap 12) —
read it as a direction, and if it points the right way, buy the other two seeds
before it goes in a table.

**What it needs.** `pretrained_weights/yolo26l-sem.pt`, ~50 MB, which the bundle
does not ship. §2 reports it as a WARN, not a preflight failure — a session not
running Stage Y should not be blocked by a file it will never open. Fetch once:

```python
from ultralytics import YOLO; YOLO("yolo26l-sem.pt")   # then move it into pretrained_weights/
```

Turn the stage on with `RUN_STAGES = "ABCDEHY"` and `ALLOW_TRAINING = True`.
""")

code(r'''
trained_Y = train_missing("Y", reg, CFG) if "Y" in RUN_STAGES else []
''')

code(r'''
# Score every Stage Y run that HAS WEIGHTS -- not only the ones trained just now.
#
# Gating on `trained_Y` (as Stages E and H do) has a trap: `train_missing`
# returns only what it trained in THIS session, so a run finished in an earlier
# session is never scored. You end up with a best.pt and no CSV, permanently,
# and re-running does not help because there is nothing left to train. Reading
# the registry instead makes this cell idempotent -- run it as often as you like.
#
# No threshold sweep, no val pass, no operating point: argmax has no parameter to
# fit. That is why this is short where Stage H is forty lines.
from bruisekit.registry import WEIGHTS as _W

_y_runs = sorted([r for r in reg.runs.values() if r.stage == "Y" and r.tier == _W],
                 key=lambda r: r.run_id)
if _y_runs:
    import bruisekit.yolo_native as yn

    y_rows = []
    for run in _y_runs:
        run_id, best_pt = run.run_id, run.weights
        # .../<run_id>/ultralytics_runs/train/weights/best.pt -> .../<run_id>/
        rd = best_pt.parents[3]
        pi_path = rd / "test_per_image_native_argmax.csv"
        if pi_path.exists():
            # Already scored. Re-reading is exact and free; re-predicting 185
            # images is neither, and would also let a re-run silently overwrite
            # the table a published number came from.
            tpi = pd.read_csv(pi_path)
            summ = {"mean_dice": float(tpi["dice"].mean()),
                    "median_dice": float(tpi["dice"].median()),
                    "complete_miss_rate": float((tpi["dice"] == 0).mean())}
            print(f"  {run_id:<30} cached")
        else:
            tpi, summ = yn.predict_native_argmax(best_pt, MAN["test"], env.data,
                                                 CFG["img_size"])
            tpi.to_csv(pi_path, index=False)
        y_rows.append({"run_id": run_id, "model": run.family, "seed": run.seed,
                       "path": "native_argmax", **summ})
        print(f"  {run_id:<30} test Dice {summ['mean_dice']:.4f}  "
              f"miss {summ['complete_miss_rate']*100:.2f}%")
        print(f"     per-image -> {pi_path}")

    Y_TEST = pd.DataFrame(y_rows)
    out_dir = env.results / "yolo_l"; out_dir.mkdir(parents=True, exist_ok=True)
    _summary = out_dir / "yolo_l_test_per_seed.csv"
    Y_TEST.to_csv(_summary, index=False)
    print(f"\n  summary   -> {_summary}")
    display(Y_TEST.round(4))

    # The comparison the stage exists for, stated as a contrast rather than left
    # for the reader to eyeball out of two tables in different sections.
    nano = reg.get("yolo_sem_direct__seed2")          # its val-selected best seed
    nano_pi = nano.per_image if nano is not None else None
    if nano_pi is not None and Path(nano_pi).exists():
        n = pd.read_csv(nano_pi)
        n_miss = float((n["dice"] == 0).mean())
        for r in y_rows:
            print(f"\n  MISS RATE  yolo26n {n_miss*100:.2f}%  ->  "
                  f"{r['model']} {r['complete_miss_rate']*100:.2f}%")
            print("  A single seed cannot support a rate claim (15, trap 12). If this "
                  "moved the right way, buy seeds 1 and 2 before quoting it.")
else:
    # Distinguish "not asked for" from "asked for and nothing landed" -- they need
    # different fixes and printing one message for both is how a failed run gets
    # mistaken for a skipped one.
    _expected = env.runs / f"yolo_sem_l_direct__seed{YOLO_L_SEEDS[0]}"
    _bp = _expected / "ultralytics_runs" / "train" / "weights" / "best.pt"
    if "Y" not in RUN_STAGES:
        print("Stage Y not requested -- add 'Y' to RUN_STAGES in section 0.")
    elif _bp.exists():
        print(f"best.pt EXISTS at {_bp} but the registry did not resolve it.")
        print("     Check that WORK_DIR points at the tree containing runs/ -- "
              "the search path is printed in section 1.")
    else:
        print(f"Stage Y has no weights yet. Expected:\n     {_bp}")
        print(f"     exists: run dir {_expected.is_dir()}")
        print("     needs pretrained_weights/yolo26l-sem.pt (~50 MB, not shipped)")
        print("     cost ~3.5 h on an A100; set ALLOW_TRAINING = True to train it")
''')

# ── Stage D ──────────────────────────────────────────────────────────────────
md(r"""
---
# Stage D · Analysis

Everything below is a function of **per-image Dice**, not of model weights — which
is why it reproduces with no GPU. Where a number came from a checkpoint it is
labelled; where it came from cache it is labelled too.
""")

code(r'''
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": ":",
})
PALETTE = ["#3b6ea5", "#4c9f70", "#c26a3d", "#8d6cab", "#b5495b",
           "#7d8b99", "#c9a227", "#4f7d7a"]

FIGDIR = env.out / "figures"; FIGDIR.mkdir(parents=True, exist_ok=True)

def save(fig, name):
    """Write a figure next to this session's tables and report where it went."""
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"{name}.{ext}")
    print(f"  saved figures/{name}.png")

# Stage H joins the pool under its own family names. `setdefault` order matters
# nowhere here -- no Stage H family collides with a Stage A/B/E name, by
# construction: a gated arm is a new FAMILY on an existing architecture, exactly as
# segformer_b0_distilled is a second family on b0.
ALL = {**A, **B, **E_TABLES, **H_TABLES}
print(f"{len(ALL)} models available for analysis: {', '.join(ALL)}")
''')

md(r"""
## D1 · Headline accuracy

Median Dice is shown alongside the mean because the per-image distribution is
strongly left-skewed: a handful of complete misses drags the mean down several
points while the median barely moves. For the YOLO variants, the gap between the
two *is* the finding.
""")

code(r'''
if ALL:
    HEAD = R.headline(ALL)
    HEAD["source"] = ["re-inferred" if RECOMPUTE_FROM_WEIGHTS else "cached"] * len(HEAD)
    display(HEAD.round(4))

    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(HEAD)); w = 0.38
    ax.bar(x - w/2, HEAD.mean_dice, w, label="mean Dice", color=PALETTE[0])
    ax.bar(x + w/2, HEAD.median_dice, w, label="median Dice", color=PALETTE[1])
    for i, r in HEAD.iterrows():
        ax.text(i - w/2, r.mean_dice + .008, f"{r.mean_dice:.3f}", ha="center", fontsize=8)
        ax.text(i + w/2, r.median_dice + .008, f"{r.median_dice:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(HEAD.model, rotation=20, ha="right")
    ax.set_ylabel("Dice"); ax.set_ylim(0, 1.0)
    ax.set_title(f"Headline accuracy on the held-out test set (images = {HEAD.n_images.iloc[0]})")
    ax.legend(frameon=False)
    save(fig, "D1_headline"); plt.show()
''')

md(r"""
## D2 · Per-image Dice distributions

The violin shows the whole distribution; the survival curve answers the question
a clinician actually asks — *what fraction of bruises does this model segment at
least this well?*
""")

code(r'''
if ALL:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    names = list(ALL)
    parts = axes[0].violinplot([ALL[n].dice.values for n in names],
                               showmedians=True, widths=0.85)
    for i, b in enumerate(parts["bodies"]):
        b.set_facecolor(PALETTE[i % len(PALETTE)]); b.set_alpha(0.65)
    axes[0].set_xticks(range(1, len(names) + 1))
    axes[0].set_xticklabels(names, rotation=20, ha="right")
    axes[0].set_ylabel("per-image Dice"); axes[0].set_ylim(-0.03, 1.03)
    axes[0].set_title("Per-image Dice distribution")

    grid = np.linspace(0, 1, 201)
    for i, n in enumerate(names):
        d = ALL[n].dice.values
        axes[1].plot(grid, [(d >= g).mean() for g in grid],
                     color=PALETTE[i % len(PALETTE)], lw=2, label=n)
    axes[1].set_xlabel("Dice threshold t"); axes[1].set_ylabel("fraction of images with Dice >= t")
    axes[1].set_title("Survival curve"); axes[1].legend(frameon=False, fontsize=8)
    save(fig, "D2_dice_distributions"); plt.show()
''')

md(r"""
## D3 · Complete-miss rate

A complete miss is `Dice == 0` — the model returned no overlap at all. This is the
metric that separates these models: their mean Dice sits within a couple of points
of each other, while their miss rates differ by an order of magnitude. A missed
bruise is a different kind of failure from a poorly-outlined one.
""")

code(r'''
if ALL:
    miss = pd.DataFrame([{"model": n, "miss_rate": d.complete_miss.mean(),
                          "miss_count": int(d.complete_miss.sum()),
                          "median_dice": d.dice.median()} for n, d in ALL.items()]
                        ).sort_values("miss_rate")
    display(miss.round(4))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    axes[0].barh(miss.model, miss.miss_rate * 100,
                 color=[PALETTE[i % len(PALETTE)] for i in range(len(miss))])
    for i, r in enumerate(miss.itertuples()):
        axes[0].text(r.miss_rate * 100 + .1, i, f"{r.miss_rate*100:.2f}%  ({r.miss_count})",
                     va="center", fontsize=8)
    axes[0].set_xlabel("complete-miss rate (%)"); axes[0].set_title("Complete misses")
    axes[0].set_xlim(0, max(miss.miss_rate * 100) * 1.35 + 1)

    for i, r in enumerate(miss.itertuples()):
        axes[1].scatter(r.median_dice, r.miss_rate * 100, s=90,
                        color=PALETTE[i % len(PALETTE)], zorder=3)
        axes[1].annotate(r.model, (r.median_dice, r.miss_rate * 100),
                         textcoords="offset points", xytext=(6, 5), fontsize=8)
    axes[1].set_xlabel("median Dice"); axes[1].set_ylabel("complete-miss rate (%)")
    axes[1].set_title("Median Dice hides the miss rate")
    save(fig, "D3_complete_miss"); plt.show()
''')

md(r"""
## D4 · Confidence intervals and paired contrasts

Both use a **subject-level cluster bootstrap**. The 185 images come from 28
subjects and images of the same bruise are strongly correlated; resampling images
would treat 185 correlated observations as independent and produce intervals
roughly 2.6× too narrow.

Contrasts are **paired** — every model saw the same 185 images, and the same
resampled subject list is applied to both models on each draw. `P(A better)` is
reported instead of a p-value: at 28 subjects it is the honest quantity, and it
does not invite a significant/not-significant dichotomy.
""")

code(r'''
if ALL:
    CIS = pd.DataFrame([{"model": n, **R.bootstrap_ci(d, "mean_dice", CFG["n_boot"])}
                        for n, d in ALL.items()]).sort_values("point", ascending=False)
    display(CIS.round(4))

    fig, ax = plt.subplots(figsize=(8.5, 0.5 * len(CIS) + 2))
    y = np.arange(len(CIS))
    ax.hlines(y, CIS.lo, CIS.hi, color=PALETTE[0], lw=3, alpha=.65)
    ax.scatter(CIS.point, y, color=PALETTE[0], zorder=3, s=55)
    ax.set_yticks(y); ax.set_yticklabels(CIS.model); ax.invert_yaxis()
    ax.set_xlabel("mean Dice (95% subject-level bootstrap CI)")
    ax.set_title(f"Marginal intervals — subjects = {CIS.n_subjects.iloc[0]}")
    save(fig, "D4a_marginal_ci"); plt.show()
''')

code(r'''
CONTRASTS = [("segformer_b0_distilled", "segformer_b0_direct"),
             ("segformer_b2_teacher",   "segformer_b0_direct"),
             ("segformer_b0_distilled", "yolo_sem_distilled"),
             ("yolo_sem_distilled",     "yolo_sem_direct"),
             ("segformer_b0_direct",    "unet_r50"),
             ("segformer_b0_direct",    "deeplabv3plus_r50"),
             # Stage F. Per arm, the first pair is the arm's own question; the
             # second asks how much of the DeepLab teacher's skill survived the
             # parameter cut (29x for Fast-SCNN, 8x for LR-ASPP), which is the
             # number a deployment decision actually needs. The two arms share a
             # teacher and recipe, so contrasting them isolates student init.
             ("fastscnn_distilled",     "fastscnn"),
             ("fastscnn_distilled",     "deeplabv3plus_r50"),
             ("lraspp_mobilenetv3_distilled", "lraspp_mobilenetv3"),
             ("lraspp_mobilenetv3_distilled", "deeplabv3plus_r50")]

rows = [R.paired_contrast(ALL[a], ALL[b], a, b, n_boot=CFG["n_boot"])
        for a, b in CONTRASTS if a in ALL and b in ALL]
if rows:
    CON = pd.DataFrame(rows)
    display(CON.round(4))

    fig, ax = plt.subplots(figsize=(9, 0.55 * len(CON) + 2))
    y = np.arange(len(CON))
    ax.axvline(0, color="0.35", lw=1, ls="--")
    ax.hlines(y, CON.lo, CON.hi, color=PALETTE[2], lw=3, alpha=.7)
    ax.scatter(CON.delta, y, color=PALETTE[2], s=55, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.a}\n  vs {r.b}" for r in CON.itertuples()], fontsize=8)
    ax.invert_yaxis()
    for i, r in enumerate(CON.itertuples()):
        ax.text(ax.get_xlim()[1], i, f"  P(better)={r.p_a_better:.2f}",
                va="center", fontsize=8)
    ax.set_xlabel("difference in mean Dice (paired subject bootstrap)")
    ax.set_title("Paired contrasts — intervals crossing 0 are indistinguishable")
    save(fig, "D4b_paired_contrasts"); plt.show()
''')

md(r"""
## D5 · Fairness across skin tone (ITA)

Groups are ordered light → dark and held in that order for every model, so the
tables read side by side.

The **gap** is descriptive and the **Kruskal–Wallis test** is inferential, and
they routinely disagree here: with 28 subjects across five groups a visually
large gap is often not significant. Both are reported so neither can be quoted
alone.
""")

code(r'''
if ALL:
    per_group = {n: R.fairness_by_group(d) for n, d in ALL.items()}
    FAIR = pd.DataFrame([{"model": n, **R.fairness_gap(d)} for n, d in ALL.items()])
    display(FAIR.round(4))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4))
    mat = pd.DataFrame({n: g.set_index("skin_tone_category").median_dice
                        for n, g in per_group.items()}).reindex(R.GROUP_ORDER)
    im = axes[0].imshow(mat.T.values, cmap="YlGnBu", aspect="auto", vmin=0.5, vmax=1.0)
    axes[0].set_xticks(range(len(mat.index)))
    axes[0].set_xticklabels(mat.index, rotation=25, ha="right", fontsize=8)
    axes[0].set_yticks(range(len(mat.columns))); axes[0].set_yticklabels(mat.columns, fontsize=8)
    for i in range(mat.shape[1]):
        for j in range(mat.shape[0]):
            v = mat.values[j, i]
            if pd.notna(v):
                axes[0].text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                             color="white" if v > 0.82 else "black")
    axes[0].grid(False); axes[0].set_title("Median Dice by ITA group")
    fig.colorbar(im, ax=axes[0], shrink=.85)

    axes[1].barh(FAIR.model, FAIR.fairness_gap,
                 color=[PALETTE[i % len(PALETTE)] for i in range(len(FAIR))])
    for i, r in enumerate(FAIR.itertuples()):
        axes[1].text(r.fairness_gap + .002, i,
                     f"{r.fairness_gap:.3f}  {'(sig.)' if r.significant else '(n.s.)'}",
                     va="center", fontsize=8)
    axes[1].set_xlabel("best-minus-worst group median Dice")
    axes[1].set_title("Fairness gap"); axes[1].set_xlim(0, FAIR.fairness_gap.max() * 1.5)
    save(fig, "D5_fairness"); plt.show()
''')

md(r"""
## D6 · The size ↔ fairness confound

Bruise size is the strongest single predictor of whether a model finds a bruise at
all, **and** it is not evenly distributed across skin-tone groups in this dataset.
Any fairness claim that does not condition on size is measuring both at once —
which is why this section sits immediately after D5 rather than in an appendix.
""")

code(r'''
if ALL:
    ref_name = "segformer_b2_teacher" if "segformer_b2_teacher" in ALL else list(ALL)[0]
    SIZE = R.size_quintiles(ALL[ref_name])
    display(SIZE.round(4))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for i, n in enumerate(ALL):
        q = R.size_quintiles(ALL[n])
        axes[0].plot(q.size_bin.astype(str), q.median_dice, "o-",
                     color=PALETTE[i % len(PALETTE)], label=n, lw=2)
        axes[1].plot(q.size_bin.astype(str), q.miss_rate * 100, "o-",
                     color=PALETTE[i % len(PALETTE)], label=n, lw=2)
    axes[0].set_xlabel("bruise-size quintile (Q1 smallest)"); axes[0].set_ylabel("median Dice")
    axes[0].set_title("Accuracy collapses on small bruises")
    axes[1].set_xlabel("bruise-size quintile (Q1 smallest)")
    axes[1].set_ylabel("complete-miss rate (%)")
    axes[1].set_title("Misses concentrate on small bruises")
    axes[1].legend(frameon=False, fontsize=8)
    save(fig, "D6a_size"); plt.show()

    sz = ALL[ref_name][["gt_positive_pixels", "skin_tone_category"]].dropna()
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    data = [sz[sz.skin_tone_category == g].gt_positive_pixels.values for g in R.GROUP_ORDER]
    data = [d for d in data if len(d)]
    labels = [g for g in R.GROUP_ORDER if len(sz[sz.skin_tone_category == g])]
    # Tick labels are set afterwards rather than passed in: boxplot's keyword for
    # them was `labels` up to matplotlib 3.8, `tick_labels` from 3.9, and `labels`
    # was removed outright in 3.11. Setting the ticks explicitly works on all of them.
    bp = ax.boxplot(data, patch_artist=True, showfliers=False)
    for i, b in enumerate(bp["boxes"]):
        b.set_facecolor(PALETTE[i % len(PALETTE)]); b.set_alpha(.65)
    ax.set_yscale("log"); ax.set_ylabel("GT bruise area (pixels, log scale)")
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title("Bruise size is not evenly distributed across ITA groups")
    save(fig, "D6b_size_by_ita"); plt.show()
''')

md(r"""
## D7 · The annotation ceiling

The context every other table needs. The models are separated from each other by
a few Dice points; **the human annotators disagree with each other by more than
that.** A table of model scores without this comparison invites reading a 0.005
gap as meaningful when it sits well inside the noise floor of the labels
themselves.
""")

code(r'''
CEIL = R.annotation_ceiling(env, ALL) if ALL else pd.DataFrame()
if len(CEIL):
    display(CEIL.round(4))
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(CEIL) + 2))
    colors = ["#b5495b" if c.startswith("human") else "#3b6ea5" for c in CEIL.comparison]
    ax.barh(CEIL.comparison, CEIL.median_dice, color=colors, alpha=.85)
    ax.invert_yaxis(); ax.set_xlabel("median Dice"); ax.set_xlim(0, 1)
    for i, r in enumerate(CEIL.itertuples()):
        ax.text(r.median_dice + .008, i, f"{r.median_dice:.3f}", va="center", fontsize=8)
    ax.set_title("Model accuracy against human–human agreement (red = human pairs)")
    save(fig, "D7_annotation_ceiling"); plt.show()
else:
    print("interlabeler_agreement_640.csv not found -- ceiling not computed "
          "(no substitute is invented)")
''')

md(r"""
## D8 · Speed and cost

### ⚠️ The shipped `benchmark_640.csv` is wrong in three of its five rows

Measured 2026-08-04. Same notebook, same cell, same A100-SXM4-40GB, same weights,
same `torch 2.11.0+cu128`, same `transformers 5.13.1`, same TF32 flags. The one
difference: a **fresh kernel**, running §1–§5 and then jumping straight to the
speed cell, instead of running the whole notebook first.

| model | shipped | fresh kernel | |
|---|---|---|---|
| segformer_b2_teacher | 33.649 ms — 29.7 FPS | **19.738 — 50.7** | **1.70× off** |
| segformer_b0_direct | 16.553 — 60.4 | **9.452 — 105.8** | **1.75× off** |
| segformer_b0_distilled | 16.762 — 59.7 | **9.578 — 104.4** | **1.75× off** |
| yolo_sem_direct | 8.130 — 123.0 | 8.024 — 124.6 | 1.01× |
| yolo_sem_distilled | 8.084 — 123.7 | 8.072 — 123.9 | 1.00× |

**All three `transformers` models inflated ~1.72×. Both convolutional models
untouched, to within 1.3%.**

### Why "it reproduced" was not evidence

60 FPS was measured in July and again in August and agreed to **0.8%**. That read
as reproducibility and was trusted on that basis for weeks. It was not: **both
runs executed the full notebook first**, so both inherited the same process
state. A number can be perfectly repeatable and still be an artifact —
repeatability only tests what you varied, and the process state was never varied
until it was tested directly.

### What the number is actually a property of

Not the model. It is a property of *(model, host, library versions, and what the
process did beforehand)* — and only the first of those travels with the bundle.
Four independent measurements of SegFormer-B0 in clean processes agree:

    ORC A100 MIG, fp32          9.294 ms
    Colab speed-harness         9.870
    Colab fresh kernel          9.452
    the 51-run registry sweep   8.740

…all inside the ±5–6% run-to-run drift these models show anyway. The 16.55 ms row
is the sole outlier, and every "cross-machine SegFormer anomaly" chased in the
handbook (§7.3, §14 caveat 6) traces back to it.

### Why only the transformers

At batch 1 with `cuda.synchronize()` on both sides, this benchmark is dominated
by kernel dispatch, not FLOPs — PP-MobileSeg (1.45 M) is *slower* here than
DeepLabV3+/R50 (26.7 M). SegFormer's cost sits in GEMM and attention; the CNNs'
sits in cuDNN convolutions. The artifact hits the first and not the second, which
points at a GEMM/attention fast path being lost. **Not yet bisected to a specific
call** — so that is a lead, not a finding, and is written here as one.

The related result, confirmed on two machines across 21 model-pairs with zero
exceptions: **`autocast` makes every one of these models slower at batch 1**
(1.27–1.44×), except the two big ResNets. More casts to dispatch, no compute won
back.

### The rule this leaves

Time in a fresh process, on one machine, and record both. `speed_table` now
carries `fresh_process` and `process_prior_peak_MB` on every row and warns when
asked to benchmark inside a process that has already touched the GPU. A row
without those columns predates the check and cannot be verified.
""")

code(r'''
bench_path = env.results / "final" / "benchmark_640.csv"
if bench_path.exists():
    BENCH = pd.read_csv(bench_path)

    # Flag the inflated rows rather than displaying them plainly. Hardcoded
    # because these are measurements, not derivations -- and because a reader who
    # displays this table must not be able to see the old number without also
    # seeing that it is superseded. Once the CSV is regenerated in a fresh
    # process these annotations stop matching; delete them then, do not update.
    FRESH_KERNEL_2026_08_04 = {          # median_ms, Colab A100-SXM4-40GB
        "segformer_b2_teacher":   19.738,
        "segformer_b0_direct":     9.452,
        "segformer_b0_distilled":  9.578,
        "yolo_sem_direct":         8.024,
        "yolo_sem_distilled":      8.072,
    }
    B = BENCH.copy()
    B["fresh_ms"] = B["model"].map(FRESH_KERNEL_2026_08_04)
    B["inflation"] = (B["median_ms"] / B["fresh_ms"]).round(3)
    B["status"] = np.where(B["inflation"] > 1.1, "SUPERSEDED -- artifact", "ok")
    display(B[["model", "params_M", "median_ms", "fps", "fresh_ms",
               "inflation", "status"]].round(3))

    n_bad = int((B["inflation"] > 1.1).sum())
    if n_bad:
        print(f"\n  {n_bad} of {len(B)} rows are inflated by a stale-process artifact "
              f"(see the section above). Do NOT quote their median_ms or fps.")
        print("  Regenerate with D9 in a FRESH kernel: Restart, then run "
              "§0-§4 and D9 only.")
    if ALL:
        # Plotted from fresh_ms, NOT median_ms. The old figure placed SegFormer-B0
        # at 60 FPS and YOLO at 123, which reads as a 2x speed advantage for YOLO
        # and was used that way. Corrected, it is 9.45 vs 8.02 ms -- an 18% gap,
        # not 2x. That is the substantive consequence of this artifact, so the
        # figure must not be drawable from the superseded column.
        fig, ax = plt.subplots(figsize=(8.5, 4.6))
        for i, r in enumerate(B.itertuples()):
            d = ALL.get(r.model)
            x = r.fresh_ms if pd.notna(r.fresh_ms) else None
            if d is None or x is None:
                continue
            ax.scatter(x, d.dice.mean(), s=40 + r.params_M * 12,
                       color=PALETTE[i % len(PALETTE)], alpha=.85, zorder=3)
            ax.annotate(f"{r.model}\n{r.params_M:.1f}M", (x, d.dice.mean()),
                        textcoords="offset points", xytext=(8, -4), fontsize=8)
        ax.set_xlabel("median latency per image (ms) — fresh process, Colab A100")
        ax.set_ylabel("mean Dice")
        ax.set_title("Accuracy vs latency (marker area ~ parameter count)")
        save(fig, "D8_speed_vs_accuracy"); plt.show()
else:
    print("benchmark_640.csv not found")
''')

# ── D9 the inference block ───────────────────────────────────────────────────
md(r"""
## D9 · The inference block — run it, don't read it

D8 **reads** `benchmark_640.csv`. Nothing in this notebook ever **wrote** one:
the five shipped rows were produced once, elsewhere, and no model added since —
not the four mobile baselines, not a single Stage F or Stage H arm — has a timing
at all.

This cell is the missing half. It runs both things the word "inference" covers,
which are not the same thing and are reported separately:

- **the inference pass** — one forward over the 185 test images at each model's
  val-selected best seed, at the operating point the run already carries. Exact
  on CPU. Reconciled against the shipped table, so a fresh number that disagrees
  says so instead of quietly replacing one.
- **the speed benchmark** — median/mean/p95 ms and FPS at 640, on the published
  recipe: 3 repeats, 10 warmup, seed 0, per-image batches, double
  `cuda.synchronize()`, images staged through this same 640 cache.

Set `RUN_INFERENCE_BLOCK = True` in §0. `INFERENCE_MODELS` takes any registry
family, so timing the mobile and gated arms needs no new code.

**Four things it will not let you do.** A speed table that mixes devices raises
rather than writes — the shipped Stage A rows are full-A100 and the Stage E rows
are an A100 MIG slice, so those two are *already* not one table. It also raises
on a table that mixes **precisions**, because a fp16 row and a fp32 row of the
same model differ by ~1.4× and that reads exactly like a hardware difference. A
YOLO row is labelled `yolo_native_raw_forward` and a SegFormer row `segformer`,
because the first times a forward and the second times a forward plus a
threshold. And a CPU run is tagged `device == "cpu"`, timed over a 16-image
subset, and written to a differently-named file, so it can never be mistaken for
a publishable number.

### ⚠️ Restart the kernel before you believe a speed row

**This is the one operational rule D8 exists to teach.** A `speed_table` run in a
process that has already trained or scored anything reported SegFormer-B0 at
1.75× its true latency — reproducibly, twice, three weeks apart. Every row now
carries `fresh_process` and `process_prior_peak_MB`, and the cell prints a loud
warning when they are wrong, but the check can only tell you the row is suspect;
it cannot repair it.

For a publishable speed table: **Restart kernel → run §0–§4 → run D9.** Nothing
else. The inference pass (accuracy) is unaffected and can be run any time — it is
only the *timing* that is contaminated by process state.
""")

code(r'''
if RUN_INFERENCE_BLOCK:
    import torch as _torch
    from bruisekit import inference as INF

    # Say it before the work starts, not after: on a GPU this is the difference
    # between a publishable row and a 1.75x artifact, and after 20 minutes of
    # benchmarking nobody re-reads the preamble.
    if str(env.device).startswith("cuda"):
        _prior = _torch.cuda.max_memory_allocated(env.device) / 1e6
        if _prior > 0:
            print(f"!! NOT A FRESH PROCESS -- {_prior:.0f} MB already peaked on this GPU.")
            print("!! The ACCURACY half below is fine. The SPEED half is not "
                  "publishable from this kernel.")
            print("!! Restart, then run only §0-§4 and D9.\n")
        else:
            print("fresh process -- speed rows from this kernel are publishable\n")

    _models = tuple(INFERENCE_MODELS) if INFERENCE_MODELS else INF.DEFAULT_MODELS
    INFER = INF.run(env, reg, CFG, MAN, MAN640, _models,
                    precision=INFERENCE_PRECISION,
                    machine_tag=MACHINE_TAG)

    # The reconciliation is the one to read first. A large max_abs_dice_delta
    # beside a tiny mean_dice_delta is the signature of a SEED MISMATCH, not of
    # float noise -- the best seed is 0 for the three SegFormers and for
    # yolo_sem_distilled but 2 for yolo_sem_direct, and pairing a model's weights
    # with the wrong seed's table shows per-image gaps up to 0.49 Dice.
    if "reconcile" in INFER:
        display(INFER["reconcile"].round(6))

    if "speed" in INFER and len(INFER["speed"]):
        SPEED = INFER["speed"]
        display(SPEED[["model", "params_M", "median_ms", "fps", "p95_ms",
                       "peak_incremental_MB", "precision", "fresh_process"]].round(3))

        if not bool(SPEED["fresh_process"].iloc[0]):
            print("\n  ^ fresh_process is False on every row above. Treat these as "
                  "diagnostics, not results.")
        else:
            print(f"\n  All rows fresh_process=True on {SPEED['device_name'].iloc[0]}. "
                  f"Quotable, for THIS machine only.")
            print("  To replace the superseded D8 rows, copy this table over "
                  "results/final/benchmark_640.csv and delete D8's "
                  "FRESH_KERNEL_2026_08_04 annotations.")
else:
    print("D9 skipped -- set RUN_INFERENCE_BLOCK = True in section 0 to run it.")
    print("     inference pass: a couple of minutes on a GPU, ~2 min per SegFormer on CPU")
    print("     speed table:    seconds, but ONLY meaningful on a GPU, and only")
    print("                     from a freshly restarted kernel (see D8)")
''')

# ── Stage G ──────────────────────────────────────────────────────────────────
md(r"""
---
# Stage G · Final significance

Stage D describes. Stage G **decides**, and it is deliberately a separate section
with a separate module (`bruisekit/significance.py`) because a confirmatory
analysis only means something if its comparison list was fixed before the numbers
were seen. That list is `SG.CONTRAST_FAMILY`, twelve comparisons, each attached to
a question someone actually asked.

**Why not compare everything to everything.** Twenty scored models is 190 ordered
pairs. At 28 subjects, with every model inside the annotation-ceiling band, an
all-pairs sweep at α=0.05 yields about ten "significant" results from noise alone
— and presents them as a ranking, which is the exact reading §D7 forbids.

**The order matters.** G1 (omnibus) gates G2 (pairwise). If a field does not
reject collectively, a difference found inside it is being read out of noise.
""")

md(r"""
## G1 · Omnibus first

Friedman across models on **subject-mean** Dice — 28 rows, not 185. Images within
a subject are strongly correlated, so blocking on images would reuse the same
information ~6.6× and let the test believe it has 185 independent blocks.

One test over all twenty models would be the wrong test: it carries 19 degrees of
freedom and most of those models are B0 students of the same teacher differing by
one KD loss term. Each set below is a field a reader would ask "are these
different at all?" about.

Kendall's *W* is the effect size — 0 means subjects do not agree on how to rank
the models at all, 1 means every subject ranks them identically.
""")

code(r'''
from bruisekit import significance as SG

# Stage C arms join the pool under their own names; where a name already exists
# from Stage A (the aliases in trap 4) the Stage A table wins, and they are the
# same weights anyway.
SG_TABLES = dict(ALL)
for _k, _v in (C or {}).items():
    SG_TABLES.setdefault(_k, _v)

print(f"{len(SG_TABLES)} models in the significance pool")

OMNI = SG.run_omnibus_sets(SG_TABLES)
display(OMNI.round(4))

for r in OMNI.itertuples():
    if r.rejects_at_05 is None:
        print(f"  {r.set:18s} -- {r.note}")
    elif r.rejects_at_05:
        print(f"  {r.set:18s} REJECTS (p={r.p:.4f}, W={r.kendall_w:.3f}) "
              f"-- pairwise comparisons inside this set are licensed")
    else:
        print(f"  {r.set:18s} does NOT reject (p={r.p:.4f}, W={r.kendall_w:.3f}) "
              f"-- these {r.n_models} models are collectively indistinguishable")
''')

md(r"""
## G2 · The pre-specified contrast family

Every endpoint comes from the **same** resampled subject lists, so "Dice up,
misses up" is a statement about the same 10,000 resampled worlds rather than
three unrelated bootstraps.

Verdicts use Stage C's 0.01 Dice margin, with one refinement: an interval running
from −0.05 to +0.02 is **INCONCLUSIVE**, not NON-INFERIOR. Stage C's rule called
everything that was not a WIN or an INFERIOR non-inferior, which reports absence
of evidence as evidence of absence.

Holm–Bonferroni is applied **within the four confirmatory contrasts only**. The
eight exploratory ones are uncorrected and labelled; folding them into the
correction would either over-penalise the questions the study was designed around
or launder post-hoc comparisons into confirmatory ones.
""")

code(r'''
FAM, FAM_SKIPPED = SG.run_family(SG_TABLES, n_boot=CFG["n_boot_final"])

if FAM_SKIPPED:
    print(f"{len(FAM_SKIPPED)} contrast(s) skipped -- named, never silently dropped:")
    for s in FAM_SKIPPED:
        print("   ", s)
    print()

if len(FAM):
    display(FAM[["a", "b", "kind", "delta_dice", "lo", "hi", "verdict",
                 "p_a_better", "p_two_sided", "p_holm", "delta_miss_rate"]].round(4))
    print()
    for _, r in FAM.iterrows():
        print(" -", SG.interpret(r))
''')

md(r"""
## G2b · Stage H's own confirmatory family

Scored as a **separate** Holm family from G2, and that separation is the point.
Appending Stage H's contrasts to `CONTRAST_FAMILY` would re-penalise a comparison
that has already been conducted and reported — the Stage F LR-ASPP arm sits at
adjusted p = 0.0042 over k = 3, and it would move for reasons that have nothing to
do with it. Multiplicity control is over the comparisons made to answer *one*
question; `CONTRAST_FAMILY_H` asks a different one and corrects within itself.

**Three** confirmatory contrasts, each one the reason an arm was built: gating on
B0, on the strongest mobile student, and on the scratch one. The fourth — gating
on YOLO's offline pseudo-mask route — is not in the family because that arm is not
registered, so **every confirmatory result here is about an online loss**.
Everything else is exploratory and uncorrected.
""")

code(r'''
FAM_H, FAM_H_SKIPPED = SG.run_family(SG_TABLES, family=RK.CONTRAST_FAMILY_H,
                                     n_boot=CFG["n_boot_final"]) \
    if "H" in RUN_STAGES else (pd.DataFrame(), [])

if FAM_H_SKIPPED:
    print(f"{len(FAM_H_SKIPPED)} Stage H contrast(s) skipped -- named, never dropped:")
    for s in FAM_H_SKIPPED:
        print("   ", s)
    print()

if len(FAM_H):
    display(FAM_H[["a", "b", "kind", "delta_dice", "lo", "hi", "verdict",
                   "p_a_better", "p_two_sided", "p_holm", "delta_miss_rate",
                   "p_a_fewer_misses"]].round(4))
    print()
    for _, r in FAM_H.iterrows():
        print(" -", SG.interpret(r))
''')

code(r'''
if len(FAM):
    fig, ax = plt.subplots(figsize=(9.5, 0.62 * len(FAM) + 2))
    y = np.arange(len(FAM))
    ax.axvline(0, color="0.35", lw=1, ls="--")
    ax.axvspan(-SG.MARGIN, 0, color="0.85", alpha=.5, zorder=0)
    colors = [PALETTE[0] if k == "confirmatory" else PALETTE[5] for k in FAM.kind]
    for i, (r, c) in enumerate(zip(FAM.itertuples(), colors)):
        ax.hlines(i, r.lo, r.hi, color=c, lw=3, alpha=.75)
        ax.scatter(r.delta_dice, i, color=c, s=55, zorder=3,
                   marker="D" if r.kind == "confirmatory" else "o")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.a}\n  vs {r.b}" for r in FAM.itertuples()], fontsize=8)
    ax.invert_yaxis()
    for i, r in enumerate(FAM.itertuples()):
        ax.text(ax.get_xlim()[1], i, f"  {r.verdict}", va="center", fontsize=8)
    ax.set_xlabel("difference in mean Dice (paired subject bootstrap)")
    ax.set_title("Pre-specified contrasts — diamonds confirmatory, circles exploratory\n"
                 "shaded band is the 0.01 non-inferiority margin")
    save(fig, "G2_contrast_family"); plt.show()
''')

md(r"""
## G3 · Complete misses, as counts

At 0–13 misses out of 185 a bootstrapped difference of two rates near the boundary
is unstable, and its interval should not be the thing anyone quotes. The 2×2 below
is the evidence itself: how many images A misses that B finds, and vice versa.
`containment` says whether one model's misses are a strict subset of the other's —
which is the question a teacher choice or a deployment decision actually turns on.

The McNemar column is exact but **ignores subject clustering**, so it is
anti-conservative. It is a sanity check on the counts, not the inferential result.
Where it disagrees with G2's clustered interval, believe G2.
""")

code(r'''
rows_d = [SG.discordance(SG_TABLES[a], SG_TABLES[b], a, b)
          for a, b, _kind, _q in SG.CONTRAST_FAMILY
          if a in SG_TABLES and b in SG_TABLES]
DISC = pd.DataFrame(rows_d)
if len(DISC):
    display(DISC[["a", "b", "a_total", "b_total", "a_misses_b_finds",
                  "b_misses_a_finds", "both_miss", "containment",
                  "mcnemar_exact_p"]].round(4))
''')

md(r"""
## G4 · Does the contrast hold at every seed?

A contrast at the val-selected seed is one draw from the training distribution.
Three same-signed deltas of similar size are stronger evidence than one large
delta, and cost nothing once every seed's `test_per_image.csv` exists — which is
the argument for pulling all of them off the cluster, not only the best seed's.

Read `sign_consistent` and `n_seeds_positive`. The per-seed intervals are
individually wide and are not meant to be read one at a time.
""")

code(r'''
# Per-seed tables straight off the registry, for the two Stage F arms whose
# distilled-vs-direct contrast is the point of the experiment.
def _by_seed(family):
    out = {}
    for seed in sorted(set(EFFICIENT_SEEDS) | set(RGKD_SEEDS or ())):
        run = reg.get(f"{family}__seed{seed}")
        if run is not None and run.per_image is not None:
            out[seed] = R.normalize(pd.read_csv(run.per_image), META)
    return out

# The Stage F arms, then every Stage H gated arm against its teacher-matched
# control. Same argument in both cases: three same-signed deltas of similar
# magnitude are stronger evidence than one large delta at the val-selected seed.
_SEED_PAIRS = [("lraspp_mobilenetv3_distilled", "lraspp_mobilenetv3"),
               ("fastscnn_distilled", "fastscnn")]
if "H" in RUN_STAGES:
    _SEED_PAIRS += [(a, b) for a, b in RK.CONTROL_FOR.items() if RK.is_gated(a)]

SEEDC = pd.DataFrame()
seed_rows = []
for _arm, _direct in _SEED_PAIRS:
    a_s, b_s = _by_seed(_arm), _by_seed(_direct)
    shared = sorted(set(a_s) & set(b_s))
    if not shared:
        print(f"{_arm} vs {_direct}: no shared seeds on disk -- skipped")
        continue
    t = SG.contrast_by_seed(a_s, b_s, _arm, _direct, n_boot=CFG["n_boot"])
    t.insert(0, "contrast", f"{_arm} vs {_direct}")
    seed_rows.append(t)
if seed_rows:
    SEEDC = pd.concat(seed_rows, ignore_index=True)
    display(SEEDC.round(4))
''')

md(r"""
## G5 · How to read Stage G

**G1 gates G2.** A set that does not reject means those models are collectively
indistinguishable on these 28 subjects. Quoting a pairwise winner from inside a
non-rejecting set is the single easiest way to over-claim with this data.

**A NON-INFERIOR verdict is a result.** "Indistinguishable from the 27 M teacher at
3.7 M parameters" is a stronger and truer claim than any superiority test this
sample size can pass.

**INCONCLUSIVE is not NON-INFERIOR.** It means the study cannot answer that
question at 28 subjects. Report it that way rather than converting it to
equivalence.

**Read the miss columns next to the Dice columns.** They come from the same draws
on purpose. A contrast that gains Dice and loses misses is the pattern §7b.6
predicted for a student whose teacher misses more than it does, and it should be
reported as both, never as whichever half is more favourable.

**Nothing here defeats the annotation ceiling (§D7).** A confirmed difference of
0.02 Dice is still smaller than the disagreement between two human annotators.
Significance answers "is it real"; D7 answers "does it matter".
""")

md(r"""
## D10 · Save everything
""")

code(r'''
# Names rather than values: a table is only written if its cell actually ran, so
# a partial session (RUN_STAGES="AD", say) saves what it produced instead of
# raising a NameError on the first thing it skipped.
WANTED = ["HEAD", "HEAD_A", "HEAD_B", "HEAD_C", "HEAD_E", "HEAD_H",
          "SCORE_C", "SCORE_H", "CIS", "CON",
          "FAIR", "SIZE", "CEIL", "plan", "gaps",
          "PROV", "SELFTEST_E", "CKPT_E", "SELFTEST_H", "GATE_H",
          "DISC_H", "FAIR_H", "H_TEST",
          "OMNI", "FAM", "FAM_H", "DISC", "SEEDC"]
FILENAME = {"HEAD": "headline_all", "HEAD_A": "headline_stage_a",
            "HEAD_B": "headline_stage_b", "HEAD_C": "headline_stage_c",
            "SCORE_C": "stage_c_vs_reference", "CIS": "marginal_ci",
            "CON": "paired_contrasts", "FAIR": "fairness_stats",
            "SIZE": "size_quintiles", "CEIL": "annotation_ceiling",
            "plan": "registry_plan", "gaps": "gaps",
            "HEAD_E": "headline_stage_e", "PROV": "efficient_weight_provenance",
            "SELFTEST_E": "efficient_self_test", "CKPT_E": "efficient_checkpoint_match",
            "OMNI": "significance_omnibus", "FAM": "significance_contrast_family",
            "DISC": "significance_miss_discordance",
            "SEEDC": "significance_seed_consistency",
            "HEAD_H": "headline_stage_h", "SCORE_H": "stage_h_vs_control",
            "SELFTEST_H": "reliability_gate_self_test",
            "GATE_H": "reliability_gate_diagnostics",
            "DISC_H": "stage_h_miss_discordance", "FAIR_H": "stage_h_fairness",
            "H_TEST": "stage_h_test_per_seed",
            "FAM_H": "significance_contrast_family_stage_h"}

saved = []
for var in WANTED:
    obj = globals().get(var)
    if isinstance(obj, pd.DataFrame) and len(obj):
        saved.append(R.save_all(env, FILENAME[var], obj))

for name, df in ALL.items():
    saved.append(R.save_all(env, f"per_image_{name}", df))
for name, df in C.items():
    saved.append(R.save_all(env, f"per_image_distill_{name}", df))

print(f"{len(saved)} tables + {len(list(FIGDIR.glob('*.png')))} figures -> {env.out}")
for p in sorted(saved):
    print("  ", p.relative_to(env.out))
''')

md(r"""
---
### How to read this notebook

**Where the numbers came from.** §3 prints the tier for every run and D1 carries a
`source` column. `cached` means the number was read from a CSV written by the same
code, from the same weights, on the same 185 test images; `re-inferred` means it
was recomputed here. The two were checked against each other on all seven
reporting models and every mean Dice agreed to better than `2e-4` — the residual
is CPU/GPU floating-point ordering, not disagreement. See the README for the
full table.

**What is genuinely missing.** The `gaps` table in §3 is the complete list. At the
time of writing it is two entries: nnU-Net (never run on the canonical 697/134
split) and one distillation arm that was configured but never executed. Neither
is back-filled or approximated anywhere in this notebook.

**Read D7 before quoting D1.** The models are separated by a few Dice points and
the human annotators disagree with each other by more than that. The ranking in
D1 is real but it is not a ranking of clinical usefulness — D3's complete-miss
rate separates these models by an order of magnitude and is the number that
carries the practical difference.

**Read D6 before quoting D5.** Bruise size predicts detection far better than skin
tone does, and the two are confounded in this dataset.

**To do real work:** set `ALLOW_TRAINING = True` to fill a gap, or
`RECOMPUTE_FROM_WEIGHTS = True` to regenerate every per-image table from the
checkpoints. Both need a GPU to be practical.
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
print(f"wrote {OUT}\n  {len(cells)} cells ({n_code} code, {len(cells)-n_code} markdown)")
