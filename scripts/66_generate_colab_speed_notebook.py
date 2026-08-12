#!/usr/bin/env python
"""Emit `bruise_colab_speed_harness.ipynb` -- the ORC speed harness, on Colab paths.

THE POINT
----------
One prediction, testable in about ten minutes: run the SAME `benchmark_speed` on
Colab that ORC just ran, and SegFormer-B0 should come back at ~16.6 ms / ~60 FPS
while ORC gave 9.29 ms / 107.6 FPS. If it does, the method is confirmed
reproducible WITHIN a machine and the host is confirmed as the variable. If Colab
returns ORC's number, the whole dispatch-bound account is wrong.

The sharper test is in the same notebook: on ORC, fp16 made SegFormer SLOWER
(9.29 -> 11.87 ms). If Colab agrees, that is two machines agreeing that autocast
costs more than it buys at batch 1. If Colab shows fp16 faster, the account fails.

WHY A SEPARATE NOTEBOOK FROM bruise_speed_harness.ipynb
--------------------------------------------------------
Colab and ORC do not share a filesystem layout, and pretending they do is what
produced the silent wrong-checkpoint bug on ORC. The ORC notebook assumes a
BRUISE_UNIFIED bundle with `bruisekit/`, `manifests/`, `data/` and a scratch dir
beside it. Colab has none of that: the dataset arrives as `bruise_colab_final.zip`
and unpacks to `/content/bruise_final` with splits at the TOP level, not under
`data/`, and the library has to be shipped in separately.

So this notebook does the assembly explicitly and prints what it found at each
step, rather than auto-detecting and hoping.

PATHS ARE TAKEN FROM bruise_colab_final.ipynb, NOT INVENTED
-------------------------------------------------------------
    drive_dir  /content/drive/MyDrive/bruise_segmentation_gpu
    zip_name   bruise_colab_final.zip        (native-res, ~2.7 GB)
    work_dir   /content/bruise_final         (local SSD, wiped on disconnect)

plus the `bruise_work` zip the user uploaded to the same Drive folder, which
carries the ORC `runs/` tree, and `BRUISEKIT_LIB.zip` from scripts/67.

THE data/ SHIM
---------------
`build_cache640` reads `env.data / r.image_path`, i.e. `<root>/data/test/images/X.jpg`.
The Colab zip puts those at `<root>/test/images/X.jpg` -- `bruise_colab_final.ipynb`
reads them as `WORK / r.image_path`. Rather than edit either manifest, §6 makes
`<root>/data/<split>` a symlink to `<root>/<split>`. Per-split symlinks, not a
self-referential `data -> .`, because the latter makes `find_root` and any
`rglob` walk an infinite tree.

If the uploaded `bruise_work` already carries a complete `cache640/`, none of
that is touched: `build_cache640` skips every file that exists and so never
resolves `env.data` at all.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "BRUISE_UNIFIED" / "bruise_colab_speed_harness.ipynb"

cells: list[dict] = []


def _cid() -> str:
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
# Colab speed harness — testing one prediction against ORC

ORC has already run this. The numbers below are what it produced, and this
notebook exists to see whether Colab disagrees in the way the dispatch-bound
account says it must.

**Prediction 1 — the machine is the variable.**

| | ORC (measured) | Colab (predicted) |
|---|---|---|
| segformer_b0, fp32 | 9.29 ms — 107.6 FPS | **~16.6 ms — ~60 FPS** |

60 FPS is what `bruise_colab_final.ipynb` measured twice on Colab, three weeks
apart, to within 0.8% (16.68 → 16.55 ms). Same `benchmark_speed`, called
directly. If §11 lands there, the method is reproducible *within* a machine and
the host is the thing that changes across machines.

**Prediction 2 — the sharper one, and the one that can actually fail.**

On ORC, fp16 made SegFormer **slower**: 9.29 → 11.87 ms, and every small model
lost ~1.35×. Only the two big ResNets gained. If Colab agrees, that is two
independent machines saying autocast costs more than it buys at batch 1 — because
what is being measured is kernel dispatch, not compute. **If Colab shows fp16
faster, the account is wrong** and we start over.

Weights do not affect latency — ORC timed four checkpoints per architecture and
they agreed to 0.08–0.4% — so `segformer_b0_direct` here is a valid comparison
against ORC's `segformer_b0_rgkd`.

---

**Before running, three things must be in
`/content/drive/MyDrive/bruise_segmentation_gpu/`:**

| file | what for |
|---|---|
| `bruise_colab_final.zip` | manifests + images + `pretrained_weights/` (already there) |
| the `bruise_work` zip you uploaded | the ORC `runs/` tree — the checkpoints |
| `BRUISEKIT_LIB.zip` | the library (build with `scripts/67_zip_bruisekit_lib.py`) |

Runtime → Change runtime type → **A100 GPU**.
""")

# ── 1 ────────────────────────────────────────────────────────────────────────
md("## §1 · Configuration — the same paths `bruise_colab_final.ipynb` uses")

code("""
from pathlib import Path

CFG = dict(
    # Verbatim from bruise_colab_final.ipynb §1.
    drive_dir = "/content/drive/MyDrive/bruise_segmentation_gpu",
    zip_name  = "bruise_colab_final.zip",
    work_dir  = "/content/bruise_final",     # local SSD, wiped on disconnect

    # The ORC scratch tree you uploaded. None = find any *bruise_work*.zip in drive_dir.
    work_zip  = None,
    scratch   = "/content/bruise_work",      # where runs/ + cache640/ + outputs/ land

    lib_zip   = "BRUISEKIT_LIB.zip",

    img_size  = 640,
    workers   = 2,
)

MACHINE_TAG = "colab-a100"        # goes in the output filenames

# The published recipe. Changing these makes new rows incomparable with the
# five in benchmark_640.csv and with everything ORC just produced.
REPEATS, WARMUP = 3, 10

BUNDLE  = Path(CFG["work_dir"])
SCRATCH = Path(CFG["scratch"])
DRIVE   = Path(CFG["drive_dir"])
print("bundle root :", BUNDLE)
print("scratch     :", SCRATCH)
""")

# ── 2 ────────────────────────────────────────────────────────────────────────
md("## §2 · Drive, GPU, dependencies")

code("""
import os, sys, time
from google.colab import drive
drive.mount("/content/drive")

import torch
if not torch.cuda.is_available():
    raise RuntimeError("No GPU. Runtime -> Change runtime type -> A100. "
                       "A CPU timing is not comparable to anything ORC produced.")
print("GPU:", torch.cuda.get_device_name(0))

assert DRIVE.is_dir(), f"{DRIVE} not found -- is the Drive folder named differently?"
print("\\nDrive folder contains:")
for p in sorted(DRIVE.iterdir())[:25]:
    kind = "dir " if p.is_dir() else f"{p.stat().st_size/1e9:5.2f} GB"
    print(f"  {kind}  {p.name}")
""")

code("""
# One line on purpose: a backslash continuation inside a %magic is a needless
# thing to be uncertain about at the top of a notebook someone else will run.
%pip install -q "transformers>=4.40,<6" "ultralytics>=8.4,<9" "albumentations>=2.0,<3" "scipy>=1.11" "pandas>=2.0" "matplotlib>=3.7" "pyyaml" "segmentation-models-pytorch"

import transformers
print("transformers", transformers.__version__)
""")

# ── 3 ────────────────────────────────────────────────────────────────────────
md("""
## §3 · Unpack the dataset zip

Gives `manifests/`, the native-resolution splits, and `pretrained_weights/` —
the last of which is required to build a SegFormer skeleton offline.
""")

code("""
import zipfile

ZIP_SRC = DRIVE / CFG["zip_name"]
if not ZIP_SRC.exists():
    raise FileNotFoundError(f"{ZIP_SRC} not found. It is the same zip "
                            f"bruise_colab_final.ipynb uses.")

if not (BUNDLE / "manifests" / "train.csv").exists():
    BUNDLE.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with zipfile.ZipFile(ZIP_SRC) as zf:
        zf.extractall(BUNDLE)
    print(f"unzipped dataset in {time.time()-t0:.0f}s")
else:
    print("dataset already unpacked")

import pandas as pd
MAN = {s: pd.read_csv(BUNDLE / "manifests" / f"{s}.csv") for s in ("train", "val", "test")}
assert (len(MAN["train"]), len(MAN["val"]), len(MAN["test"])) == (697, 134, 185), \\
    f"unexpected split sizes: {[len(MAN[s]) for s in ('train','val','test')]}"
print("splits 697/134/185 OK")
print("pretrained_weights:", sorted(p.name for p in (BUNDLE / "pretrained_weights").iterdir())
      if (BUNDLE / "pretrained_weights").is_dir() else "MISSING")
""")

# ── 4 ────────────────────────────────────────────────────────────────────────
md("""
## §4 · Unpack the `bruise_work` tree — the checkpoints

This is the ORC scratch dir. What matters is `runs/`: the registry looks there
**first** and silently falls back to shipped copies if it is empty, which is
exactly the bug that wasted two ORC runs. §7 counts what landed and refuses to
continue if the answer is zero.
""")

code("""
cands = ([DRIVE / CFG["work_zip"]] if CFG["work_zip"]
         else sorted(DRIVE.glob("*bruise_work*.zip")))
cands = [p for p in cands if p.exists()]
if not cands:
    raise FileNotFoundError(
        f"No *bruise_work*.zip in {DRIVE}. Set CFG['work_zip'] to its exact name.")
WORK_ZIP = cands[0]
print(f"using {WORK_ZIP.name}  ({WORK_ZIP.stat().st_size/1e9:.2f} GB)")

SCRATCH.mkdir(parents=True, exist_ok=True)
if not (SCRATCH / "runs").is_dir():
    t0 = time.time()
    with zipfile.ZipFile(WORK_ZIP) as zf:
        names = zf.namelist()
        zf.extractall(SCRATCH)
    print(f"unzipped {len(names)} entries in {time.time()-t0:.0f}s")
else:
    print("already unpacked")

# The zip may or may not have a top-level bruise_work/ folder. Normalise, so
# SCRATCH/runs is the runs tree either way rather than SCRATCH/bruise_work/runs.
if not (SCRATCH / "runs").is_dir():
    inner = [p for p in SCRATCH.iterdir() if p.is_dir() and (p / "runs").is_dir()]
    if inner:
        print(f"  runs/ was nested under {inner[0].name}/ -- using that as scratch")
        SCRATCH = inner[0]

runs_dir = SCRATCH / "runs"
if not runs_dir.is_dir():
    raise SystemExit(f"no runs/ inside {WORK_ZIP.name}. Contents: "
                     f"{sorted(p.name for p in SCRATCH.iterdir())[:20]}")

entries = sorted(p.name for p in runs_dir.iterdir() if p.is_dir())
print(f"\\n{len(entries)} run directories in {runs_dir}")
for e in entries[:40]:
    print("   ", e)
if len(entries) > 40:
    print(f"    ... and {len(entries)-40} more")

print("\\ncache640 shipped in the work zip:",
      (SCRATCH / "cache640" / "test" / "images").is_dir())
""")

# ── 5 ────────────────────────────────────────────────────────────────────────
md("""
## §5 · Install the library

`paths.setup()` only accepts a root that contains **both** `bruisekit/` and
`manifests/train.csv`. The dataset zip supplies the second; this supplies the first.
""")

code("""
LIB_ZIP = DRIVE / CFG["lib_zip"]
if not (BUNDLE / "bruisekit" / "inference.py").exists():
    if not LIB_ZIP.exists():
        raise FileNotFoundError(
            f"{LIB_ZIP} not found. Build it locally with:\\n"
            f"    python scripts/67_zip_bruisekit_lib.py\\n"
            f"and upload BRUISEKIT_LIB.zip to {DRIVE}")
    with zipfile.ZipFile(LIB_ZIP) as zf:
        zf.extractall(BUNDLE)
    print("library installed")
else:
    print("library already present")

sys.path.insert(0, str(BUNDLE))
os.chdir(BUNDLE)

import importlib, inspect
from bruisekit import gpustate as G, inference as INF
importlib.reload(G); importlib.reload(INF)
if "precision" not in inspect.signature(INF.speed_table).parameters:
    raise SystemExit("bruisekit is the OLD copy -- rebuild BRUISEKIT_LIB.zip from the "
                     "patched bundle (it must contain gpustate.py).")
print("patch  : APPLIED")
print()
print(G.describe(G.probe()))
""")

# ── 6 ────────────────────────────────────────────────────────────────────────
md("""
## §6 · `data/` shim

The Colab zip puts the image folders at the top level; `build_cache640` resolves
them as `env.data / image_path`, i.e. under `data/`. This bridges the two.

The folder names come from the manifests rather than a hardcoded list — `val`
images live under `train/`, so there is no `val/` directory and assuming one
would report a phantom failure.

Symlink first, move if the filesystem refuses (Colab allows symlinks; some do
not). Then it **resolves a real path from the test manifest and raises here** if
that fails — three cells before `build_cache640` would have failed anyway, with a
message that says what to fix.
""")

code("""
import shutil

data_dir = BUNDLE / "data"
data_dir.mkdir(exist_ok=True)

# Which top-level folders do the manifests actually reference?
prefixes = sorted({Path(p).parts[0] for df in MAN.values() for p in df["image_path"]})
print("manifests reference:", prefixes)

for name in prefixes:
    dst, src = data_dir / name, BUNDLE / name
    if dst.exists():
        continue
    if not src.is_dir():
        print(f"  {name}: neither data/{name} nor {name}/ exists")
        continue
    try:
        dst.symlink_to(src, target_is_directory=True)
        print(f"  {name}: symlinked")
    except OSError:
        shutil.move(str(src), str(dst))
        print(f"  {name}: moved into data/ (symlinks not permitted here)")

# Resolve one real path end to end. Cheap, and it fails HERE with a clear cause.
probe = data_dir / MAN["test"]["image_path"].iloc[0]
cache_ready = (SCRATCH / "cache640" / "test" / "images").is_dir()
print(f"\\nprobe: {probe}")
print(f"  resolves: {probe.exists()}   |  cache640 already present: {cache_ready}")
if not probe.exists() and not cache_ready:
    raise SystemExit(
        f"{probe} does not exist and there is no prebuilt cache640, so §9 cannot "
        f"build one.\\nTop level of {BUNDLE}: "
        f"{sorted(p.name for p in BUNDLE.iterdir())}\\n"
        f"Fix: check that {CFG['zip_name']} unpacked fully in §3.")
""")

# ── 7 ────────────────────────────────────────────────────────────────────────
md("""
## §7 · Registry — and **where each checkpoint came from**

`source` is the column to read. `runs` means the ORC tree you uploaded;
`shipped` means a bundled fallback was used instead, which is a silent
substitution and not a choice you made.
""")

code("""
from bruisekit.paths import setup
from bruisekit.registry import Registry, WEIGHTS

env = setup(root=BUNDLE, work=SCRATCH, mount=False)
print(env.describe())

n_runs = len(list(env.runs.iterdir())) if env.runs.is_dir() else 0
print(f"\\n{n_runs} entries in {env.runs}")
if n_runs == 0:
    raise SystemExit(f"{env.runs} is empty -- every checkpoint would silently come "
                     f"from a shipped copy. Check §4.")

reg = Registry(env, efficient_seeds=(0, 1, 2)).scan()

def _source(run):
    if run.weights is None:
        return "-"
    try:
        run.weights.relative_to(env.runs); return "runs"
    except ValueError:
        return "shipped"

rows = {}
for r in reg.runs.values():
    d = rows.setdefault(r.family, {"family": r.family, "stage": r.stage,
                                   "kind": r.kind, "seeds": [], "source": "-"})
    if r.tier == WEIGHTS:
        d["seeds"].append(r.seed)
        if r.seed == 0:
            d["source"] = _source(r)

tbl = pd.DataFrame(sorted(rows.values(), key=lambda d: (d["stage"], d["family"])))
tbl["seeds"] = tbl["seeds"].apply(lambda s: sorted(x for x in s if x is not None))
tbl["timeable"] = tbl["seeds"].apply(lambda s: 0 in s)
print()
print(tbl.to_string(index=False))

TIMEABLE = sorted(tbl.loc[tbl["timeable"], "family"])
print(f"\\n{len(TIMEABLE)} timeable families:")
print("   ", TIMEABLE)

shipped = sorted(tbl.loc[tbl["source"] == "shipped", "family"])
if shipped:
    print(f"\\n  WARNING  fell back to shipped checkpoints for: {shipped}")
""")

# ── 8 ────────────────────────────────────────────────────────────────────────
md("""
## §8 · Choose the models

Leave `MODELS` as everything timeable for the full picture. If you only want the
prediction tested, set it to whichever `segformer_b0_*` §7 listed — that is the
one row both predictions are about.
""")

code("""
MODELS = tuple(TIMEABLE)

seg = [m for m in TIMEABLE if m.startswith("segformer_b0")]
if not seg:
    print("  WARNING  no segformer_b0_* is timeable -- NEITHER prediction can be "
          "tested. Check §4 and §7 before spending GPU time.")
else:
    print("SegFormer rows available for the prediction:", seg)

print(f"\\n{len(MODELS)} models, {REPEATS} repeats x 185 images = "
      f"{REPEATS*185} timed calls each, {WARMUP} warmup, seed 0, batch 1")
""")

# ── 9 ────────────────────────────────────────────────────────────────────────
md("## §9 · 640 cache")

code("""
from bruisekit import loaders as L
man640 = L.build_cache640(env, MAN)
""")

# ── 10 ───────────────────────────────────────────────────────────────────────
md("""
## §10 · fp32 — the published recipe

`benchmark_speed` is called directly, not a copy, so this is the same function
that produced 16.68 ms in July and 16.55 ms in August on this machine.
""")

code("""
SP32 = INF.run(env, reg, INF.DEFAULT_CFG, MAN, man640, MODELS,
               do_inference=False, do_speed=True, do_reconcile=False,
               repeats=REPEATS, warmup=WARMUP,
               precision="fp32", machine_tag=MACHINE_TAG)["speed"]

SP32[["model", "median_ms", "fps", "p95_ms", "params_M", "peak_incremental_MB"]].round(3)
""")

# ── 11 ───────────────────────────────────────────────────────────────────────
md("## §11 · fp16 — the same loop, one line different")

code("""
SP16 = INF.run(env, reg, INF.DEFAULT_CFG, MAN, man640, MODELS,
               do_inference=False, do_speed=True, do_reconcile=False,
               repeats=REPEATS, warmup=WARMUP,
               precision="fp16", machine_tag=MACHINE_TAG)["speed"]

SP16[["model", "median_ms", "fps", "p95_ms", "params_M", "peak_incremental_MB"]].round(3)
""")

# ── 12 ───────────────────────────────────────────────────────────────────────
md("""
## §12 · The verdict — both predictions, scored

ORC's numbers are embedded below exactly as it produced them, so the comparison
is printed rather than eyeballed across two windows.
""")

code("""
# ORC A100 MIG 3g.40gb, median_ms, measured 2026-08-04. Both precisions, same recipe.
ORC = {
 "deeplabv3plus_r50":            (10.441,  7.554),
 "fastscnn":                     ( 3.612,  4.888),
 "fastscnn_b2kd":                ( 3.611,  4.846),
 "fastscnn_distilled":           ( 3.608,  4.836),
 "fastscnn_rgkd":                ( 3.624,  4.831),
 "lraspp_mobilenetv3":           ( 5.045,  6.857),
 "lraspp_mobilenetv3_b2kd":      ( 5.031,  6.866),
 "lraspp_mobilenetv3_distilled": ( 5.022,  6.869),
 "lraspp_mobilenetv3_rgkd":      ( 5.048,  6.832),
 "ppmobileseg_tiny":             (11.415, 14.468),
 "ppmobileseg_tiny_b2kd":        (11.500, 14.458),
 "ppmobileseg_tiny_distilled":   (11.402, 14.450),
 "ppmobileseg_tiny_rgkd":        (11.377, 14.456),
 "segformer_b0_rgkd":            ( 9.294, 11.865),
 "topformer_tiny":               ( 6.507,  8.409),
 "topformer_tiny_b2kd":          ( 6.511,  8.353),
 "topformer_tiny_distilled":     ( 6.506,  8.337),
 "topformer_tiny_rgkd":          ( 6.506,  8.346),
 "unet_r50":                     (11.817, 10.232),
 "yolo_sem_direct":              ( 5.821,  7.545),
 "yolo_sem_distilled":           ( 5.889,  6.933),
}

cmp = (SP32[["model", "median_ms", "fps", "params_M"]]
       .merge(SP16[["model", "median_ms"]], on="model", suffixes=("_fp32", "_fp16")))
cmp["fp32_over_fp16"] = (cmp["median_ms_fp32"] / cmp["median_ms_fp16"]).round(3)
cmp["orc_fp32"] = cmp["model"].map(lambda m: ORC.get(m, (float("nan"),))[0])
cmp["orc_fp16"] = cmp["model"].map(lambda m: ORC.get(m, (0, float("nan")))[1])
cmp["colab_over_orc_fp32"] = (cmp["median_ms_fp32"] / cmp["orc_fp32"]).round(3)
cmp = cmp.sort_values("median_ms_fp32").reset_index(drop=True)
print(cmp.to_string(index=False))

print("\\n" + "=" * 72)
seg32 = cmp[cmp["model"].str.startswith("segformer_b0")]
if seg32.empty:
    print("PREDICTION 1  UNTESTED -- no segformer_b0_* was timed.")
else:
    ms, fps = seg32.iloc[0]["median_ms_fp32"], seg32.iloc[0]["fps"]
    print(f"PREDICTION 1  segformer_b0 fp32 on Colab: {ms:.2f} ms / {fps:.1f} FPS")
    print(f"              predicted ~16.6 ms / ~60 FPS   (ORC gave 9.29 ms / 107.6)")
    print("              ->", "CONFIRMED" if 15.5 <= ms <= 18.0 else
          ("matches ORC, NOT Colab's own history -- the account is wrong"
           if ms < 11.0 else "neither -- something else is going on"))

slower = cmp[(cmp["params_M"] < 5) & (cmp["fp32_over_fp16"] < 1.0)]
faster = cmp[(cmp["params_M"] > 20) & (cmp["fp32_over_fp16"] > 1.0)]
print(f"\\nPREDICTION 2  fp16 slower for {len(slower)}/{(cmp['params_M'] < 5).sum()} "
      f"small models, faster for {len(faster)}/{(cmp['params_M'] > 20).sum()} big ones")
print("              ->", "CONFIRMED -- two machines agree autocast costs more than it "
      "buys at batch 1" if len(slower) >= 3 else
      "NOT confirmed -- fp16 helped the small models here but hurt them on ORC")
print("=" * 72)

out = env.out / "inference" / f"speed_comparison_{MACHINE_TAG}_vs_orc.csv"
cmp.to_csv(out, index=False)
print(f"\\nwrote {out}")
""")

# ── 13 ───────────────────────────────────────────────────────────────────────
md("""
## §13 · Send back

Three files from `/content/bruise_work/outputs/inference/`:

    benchmark_640_cuda_fp32_colab-a100.csv
    benchmark_640_cuda_fp16_colab-a100.csv
    speed_comparison_colab-a100_vs_orc.csv

`/content` is wiped on disconnect — copy them to Drive first:

```python
import shutil
dst = DRIVE / "speed_harness_results"; dst.mkdir(exist_ok=True)
for p in (env.out / "inference").glob("*.csv"):
    shutil.copy2(p, dst / p.name)
print("copied to", dst)
```

**What each outcome means**

*Prediction 1 confirmed (~16.6 ms).* The method is reproducible within a machine
and not across machines. Latency is a property of *(model, host)*, not of the
model, and no single FPS can serve as a correctness anchor. Use the inference
pass and Dice for that instead — `INF.run(..., do_inference=True)` reconciles
fresh per-image scores against the shipped table, and *that* number is portable.

*Prediction 1 fails and Colab returns ~9 ms.* Then Colab's own July and August
`benchmark_640.csv` runs were measuring something this notebook is not, and the
next thing to check is what differs between this call path and
`bruise_colab_final.ipynb` §D8 — same function, so the difference would have to
be in what is staged or which checkpoint is loaded.

*Prediction 2 fails.* The dispatch-bound explanation does not survive, and the
ORC fp16 result needs a different account before any of it goes in the handbook.

Either way: **do not rewrite handbook §7.3 or §14 caveat 6 from this run.** Those
sections currently mix harnesses and machines in one column; fixing them needs the
same recipe on both machines, which is what these two CSVs finally provide.
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "colab": {"provenance": [], "toc_visible": True},
        "accelerator": "GPU",
    },
    "nbformat": 4, "nbformat_minor": 5,
}

OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
n_code = sum(1 for c in cells if c["cell_type"] == "code")
print(f"wrote {OUT}\n  {len(cells)} cells ({n_code} code, {len(cells) - n_code} markdown)")
