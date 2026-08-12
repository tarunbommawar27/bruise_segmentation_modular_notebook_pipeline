#!/usr/bin/env python
"""Zip the Stage Y + inference-block overlay for `bruise_unified.ipynb`.

TWO CHANGES, ONE ARCHIVE
--------------------------
1. THE INFERENCE BLOCK IS NOW IN THE UNIFIED NOTEBOOK, with the reason written
   next to it. D8 used to read `benchmark_640.csv` and display it; it now flags
   three of its five rows as superseded, because they are:

       segformer_b2_teacher    33.649 ms -> 19.738   1.70x off
       segformer_b0_direct     16.553    ->  9.452   1.75x off
       segformer_b0_distilled  16.762    ->  9.578   1.75x off
       yolo_sem_direct          8.130    ->  8.024   1.01x
       yolo_sem_distilled       8.084    ->  8.072   1.00x

   Same notebook, same cell, same A100, same torch/transformers/TF32 flags. The
   only difference is a FRESH KERNEL. All three `transformers` models inflated
   ~1.72x; both convolutional models untouched.

   The number had been measured twice three weeks apart and agreed to 0.8%, which
   read as reproducibility. It was not: both runs executed the whole notebook
   first, so both inherited the same process state. Repeatability only tests what
   you varied.

   D9 now runs the benchmark rather than only describing it, takes `precision`
   and `machine_tag`, and prints a loud warning BEFORE the work starts if the
   process has already touched the GPU. `speed_table` records `fresh_process` and
   `process_prior_peak_MB` on every row, so a suspect row is identifiable after
   the fact instead of indistinguishable from a good one.

2. STAGE Y -- YOLO26-LARGE, native Ultralytics, native argmax. `yolo26n` is the
   fastest model in the study and has the worst complete-miss rate of the
   reporting models (6.5% against 0.0-0.5%). Miss containment is the endpoint
   this study is judged on, so "is that YOLO, or is it 1.6 M parameters?" is the
   question the nano arms cannot answer. Stage Y is the identical recipe on the
   large backbone -- same mosaic, EMA, letterbox, LR schedule -- and nothing else.

   Its own stage letter, for the reason Stage H got one: Stage A is quoted as
   "the five headline models" and a table that silently grows two rows is
   indistinguishable from a bug.

WHY yolo_native.py IS PATCHED IN bruise_colab_final.ipynb AND NOT unified_lib
-------------------------------------------------------------------------------
`60_build_unified_bundle.py` copies most modules from `scripts/unified_lib/`, but
`yolo_native.py` is not one of them: it is extracted from the
`%%writefile bruisekit/yolo_native.py` cell of `bruise_colab_final.ipynb`. Editing
only the bundle copy would have worked until the next build silently reverted it.
Both are shipped here and `main()` refuses to build if they have diverged.

LAYOUT: the same convention as 63/64 -- paths relative to the bundle, sources
under `_source/`. Upload into BRUISE_UNIFIED and extract in place, no `-d`.
"""
from __future__ import annotations

import json
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DST = ROOT / "BRUISE_PATHS_YOLOL_V2.zip"

MEMBERS: list[tuple[str, str, str]] = [
    # ── runnable ──────────────────────────────────────────────────────────────
    ("BRUISE_UNIFIED/bruise_unified.ipynb", "bruise_unified.ipynb",
     "83 cells. D8 flags the superseded rows, D9 runs the benchmark, Stage Y is new."),
    ("BRUISE_UNIFIED/bruisekit/inference.py", "bruisekit/inference.py",
     "fresh_process + process_prior_peak_MB columns and the pre-flight warning; "
     "precision=, the amp mirrors, per-model failure isolation."),
    ("BRUISE_UNIFIED/bruisekit/registry.py", "bruisekit/registry.py",
     "find_checkpoint() over all roots x 3 layouts, read_cut() for both threshold "
     "dialects, Run provenance, _scan_teachers (B5 finally visible), Stage Y."),
    ("BRUISE_UNIFIED/bruisekit/loaders.py", "bruisekit/loaders.py",
     "FAMILY_SPEC rows for the two Stage Y arms; SegFormer layout reconciliation."),
    ("BRUISE_UNIFIED/bruisekit/paths.py", "bruisekit/paths.py",
     "Env.run_roots -- the checkpoint SEARCH PATH; extra_runs=; 'yolo_l'."),
    ("BRUISE_UNIFIED/bruisekit/yolo_native.py", "bruisekit/yolo_native.py",
     "pretrained_for() + _is_large() -- size-aware backbone selection."),
    ("BRUISE_UNIFIED/bruisekit/gpustate.py", "bruisekit/gpustate.py",
     "clock/persistence probe and the under-load sampler (from the v2 overlay)."),

    # ── documentation ─────────────────────────────────────────────────────────
    ("BRUISE_UNIFIED/PROJECT_HANDBOOK.md", "PROJECT_HANDBOOK.md",
     "NEW 7.3a/7.3b/7.3c (the speed artefact), NEW 7d (Stage Y), NEW 10.2 (the "
     "search path), traps 17-18, 4/14-caveat-6/18.1/18.4/18.5 corrected."),

    # ── sources (handbook 16) ─────────────────────────────────────────────────
    ("scripts/unified_lib/inference.py", "_source/scripts/unified_lib/inference.py",
     "source of truth."),
    ("scripts/unified_lib/registry.py", "_source/scripts/unified_lib/registry.py",
     "source of truth."),
    ("scripts/unified_lib/loaders.py", "_source/scripts/unified_lib/loaders.py",
     "source of truth."),
    ("scripts/unified_lib/paths.py", "_source/scripts/unified_lib/paths.py",
     "source of truth."),
    ("scripts/unified_lib/gpustate.py", "_source/scripts/unified_lib/gpustate.py",
     "source of truth."),
    ("bruise_colab_final.ipynb", "_source/bruise_colab_final.ipynb",
     "REQUIRED -- carries the %%writefile cell that yolo_native.py is built from. "
     "Without it the next 60_build_unified_bundle.py reverts pretrained_for()."),
    ("scripts/60_build_unified_bundle.py", "_source/scripts/60_build_unified_bundle.py",
     "gpustate.py in the authored-module copy list."),
    ("scripts/61_generate_unified_notebook.py",
     "_source/scripts/61_generate_unified_notebook.py", "emits the notebook."),
    ("scripts/68_zip_inference_yolol_overlay.py",
     "_source/scripts/68_zip_inference_yolol_overlay.py", "this script."),
]

README = """\
# Checkpoint search path + Stage Y + the inference block

## Part 0 — every checkpoint is now reachable from one config

The study's weights live in at least four places with **three different on-disk
layouts**, and the registry used to search exactly two directories, falling back
between them silently. `env.run_roots` is now an ordered search path:

    1. WORK_DIR/runs                          this session's training
    2. EXTRA_RUNS                             <- set in §0, e.g. /scratch/$USER/bruise_work/runs
    3. checkpoints/final                      Stage A, 5 models x 3 seeds
    4. checkpoints/baselines                  U-Net, DeepLabV3+
    5. checkpoints/efficient                  Stage E/F mobile arms
    6. checkpoints/rgkd                       Stage H arms
    7. checkpoints/yolo_l                     Stage Y
    8. checkpoints/distill/teachers           B2, B5, B0-distilled

First hit wins, so your own training always beats a shipped copy. Set it once:

```python
EXTRA_RUNS = "/scratch/tbommawa/bruise_work/runs"     # or a list
```

**The three layouts, now all understood by one resolver:**

| layout | weights | threshold |
|---|---|---|
| `standard` | `<family>__seed<N>/best.pt` | `operating_point.json` — a **logit** |
| `yolo` | `.../ultralytics_runs/train/weights/best.pt` | none needed (argmax) |
| `teacher` | `<family>/best_model.pt` — **no seed** | `threshold.json` — a **probability** |

`read_cut()` is the single place that converts between the two threshold
dialects. They are not interchangeable: sigmoid(cut) == threshold, so reading one
as the other does not raise, it silently moves the decision boundary.

**SegFormer-B5 is now visible for the first time.** It was invisible for three
independent reasons at once — `best_model.pt` not `best.pt`, `threshold.json` not
`operating_point.json`, and no seed in the directory name (its
`seed_selection.csv` picked seed **123**, so asking for `__seed0` could never
match). That matters for the fairness work: B2 is the only model in this study
with a statistically detectable skin-tone disparity (Kruskal p=0.011, gap 0.112),
B5 has gap 0.070 at p=0.220 and beats B2 on **every** ITA group, and every Stage H
arm currently distils from B2.

**Every `Run` now records provenance** — `source_root`, `layout`,
`threshold_file` — and §3 prints a table of it. If a family you trained shows a
shipped root, your training is not being used, and you find out in the plan
rather than in a crash three cells later.

---

# Stage Y + the inference block

    cd /path/to/BRUISE_UNIFIED
    unzip -o BRUISE_PATHS_YOLOL_V2.zip

No `-d`. **Restart the kernel afterwards** — `bruisekit` is already imported in a
running session and `importlib.reload` will not reliably pick up replaced files.

---

## Part 1 — the speed numbers were wrong, and D8 now says so

| model | shipped | fresh kernel | |
|---|---|---|---|
| segformer_b2_teacher | 33.649 ms — 29.7 FPS | **19.738 — 50.7** | **1.70x off** |
| segformer_b0_direct | 16.553 — 60.4 | **9.452 — 105.8** | **1.75x off** |
| segformer_b0_distilled | 16.762 — 59.7 | **9.578 — 104.4** | **1.75x off** |
| yolo_sem_direct | 8.130 — 123.0 | 8.024 — 124.6 | 1.01x |
| yolo_sem_distilled | 8.084 — 123.7 | 8.072 — 123.9 | 1.00x |

Same notebook, same cell, same A100-SXM4-40GB, same torch 2.11.0+cu128, same
transformers 5.13.1, same TF32 flags. **The only difference is a fresh kernel.**

**Why 60 FPS looked trustworthy.** It was measured in July and again in August and
agreed to 0.8%. Both runs executed the whole notebook first, so both inherited the
same process state. A number can be perfectly repeatable and still be an artifact —
repeatability only tests what you varied.

**Why only the transformers.** At batch 1 with double `cuda.synchronize()` this
benchmark is dispatch-bound, not FLOP-bound — PP-MobileSeg (1.45 M) is *slower*
here than DeepLabV3+/R50 (26.7 M). SegFormer's cost is GEMM and attention; the
CNNs' is cuDNN convolutions. The artifact hits the first and not the second.
**Not bisected to a specific call** — written up as a lead, not a finding.

**The consequence that matters.** Corrected, YOLO is 8.02 ms against
SegFormer-B0's 9.45 — an 18% gap, not the 2x the old figure showed. D8's scatter
plot is now drawn from the corrected column so the superseded one cannot be
plotted by accident.

**What is now enforced.** Every speed row carries `fresh_process` and
`process_prior_peak_MB`. D9 prints a warning *before* the work starts if the
process has already touched the GPU. `check_single_machine` already refused to mix
devices; it now also refuses to mix precisions.

### To produce a publishable speed table

    Restart kernel -> run §0-§4 -> run D9.   Nothing else.

```python
RUN_INFERENCE_BLOCK = True
INFERENCE_MODELS    = None          # or any registry family names
INFERENCE_PRECISION = "fp32"        # the published recipe; fp16 attributes, never reports
MACHINE_TAG         = "colab-a100"  # or "orc-mig"
```

The **accuracy** half is unaffected by process state and can be run any time.
Only the timing is contaminated.

---

## Part 2 — Stage Y: YOLO26-large

`yolo26n` is the fastest model in the study and has the **worst complete-miss
rate of the reporting models** — 6.5% native-argmax against 0.0–0.5% for the
SegFormers. §D3 says miss containment, not Dice, is the endpoint. So: is 6.5% a
property of YOLO, or of 1.6 M parameters?

Stage Y is the identical native recipe on the large backbone — same mosaic, EMA,
letterbox, LR schedule, `close_mosaic`, seed handling. Scored by **native argmax
only**; argmax is parameter-free, so a run needs `best.pt` and nothing else.

### Turn it on

```python
RUN_STAGES     = "ABCDEHY"
ALLOW_TRAINING = True
YOLO_L_SEEDS   = (0,)      # one seed -- ~3.5 h
```

**One arm only.** Stage Y registers `yolo_sem_l_direct` and nothing else. A
distilled large arm has a spec and a cost estimate but is deliberately
unregistered, so `"...Y"` cannot quietly cost an extra ~4 GPU-hours. Enable it by
adding it to `registry.STAGE_Y_FAMILIES` when that question is being asked.

### First, fetch the backbone (~50 MB, not shipped)

```python
from ultralytics import YOLO; YOLO("yolo26l-sem.pt")
# then move the downloaded file into pretrained_weights/
```

§2 reports it as **WARN** when absent and `Y` is not in `RUN_STAGES`, and as
**FAIL** when it is — a session not running Stage Y is not blocked by a file it
will never open.

### Read the result as a direction, not a rate

`YOLO_L_SEEDS` defaults to one seed because a yolo26l run is ~3.5 h against a
mobile arm's ~0.5 h. **A one-seed miss count is not a rate** (handbook §15, trap
12). If seed 0 moves the miss rate the right way, buy seeds 1 and 2 before the
number goes in a table. The cell prints this next to the comparison.

### The silent failure this avoids

`yn.pretrained_for(family, paths)` picks the backbone from the family name.
Passing nano weights to a Stage Y run raises nothing — it builds, trains,
produces a `best.pt` and lands in a table as the large arm, with the only symptom
a 28 M-parameter row reporting 1.6 M. One function, one call site.

---

## Both halves ship their sources

`bruisekit/*.py` are build outputs of `scripts/unified_lib/`, and
**`yolo_native.py` is not** — it is extracted from the `%%writefile` cell of
`bruise_colab_final.ipynb`, which is why that notebook is in `_source/`. Editing
only the bundle copy works until the next `60_build_unified_bundle.py` silently
reverts it. This script refuses to build if any pair has diverged.
"""


def _yolo_native_from_notebook(nb_path: Path) -> str:
    """The yolo_native.py body as bruise_colab_final.ipynb will write it."""
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    bodies = [
        "".join(c["source"]).split("\n", 1)[1]
        for c in nb["cells"]
        if "".join(c["source"]).startswith("%%writefile bruisekit/yolo_native.py")
    ]
    if len(bodies) != 1:
        raise SystemExit(f"expected exactly 1 yolo_native writefile cell, found {len(bodies)}")
    return bodies[0]


def main() -> int:
    missing = [s for s, _, _ in MEMBERS if not (ROOT / s).exists()]
    if missing:
        raise SystemExit("not built yet -- run 61_generate_unified_notebook.py and "
                         "60_build_unified_bundle.py first. Missing:\n  "
                         + "\n  ".join(missing))

    # Bundle copies are build outputs. Shipping them out of sync with their
    # sources means the next rebuild silently reverts this overlay, and the
    # failure surfaces weeks later as "the fix stopped working".
    for bundled, authored in (("BRUISE_UNIFIED/bruisekit/inference.py",
                               "scripts/unified_lib/inference.py"),
                              ("BRUISE_UNIFIED/bruisekit/registry.py",
                               "scripts/unified_lib/registry.py"),
                              ("BRUISE_UNIFIED/bruisekit/loaders.py",
                               "scripts/unified_lib/loaders.py"),
                              ("BRUISE_UNIFIED/bruisekit/paths.py",
                               "scripts/unified_lib/paths.py"),
                              ("BRUISE_UNIFIED/bruisekit/gpustate.py",
                               "scripts/unified_lib/gpustate.py")):
        if (ROOT / bundled).read_bytes() != (ROOT / authored).read_bytes():
            raise SystemExit(f"{bundled} and {authored} differ -- sync them first.")

    nb_body = _yolo_native_from_notebook(ROOT / "bruise_colab_final.ipynb")
    bundle_body = (ROOT / "BRUISE_UNIFIED/bruisekit/yolo_native.py").read_text(encoding="utf-8")
    if nb_body.strip() != bundle_body.strip():
        raise SystemExit(
            "bruisekit/yolo_native.py does not match the %%writefile cell in "
            "bruise_colab_final.ipynb. That cell is its source of truth (60_build_"
            "unified_bundle.py extracts it), so shipping them out of sync means the "
            "next build reverts pretrained_for().")

    t0 = time.time()
    total = sum((ROOT / s).stat().st_size for s, _, _ in MEMBERS)
    print(f"zipping {len(MEMBERS)} files, {total / 1e6:.2f} MB -> {DST.name}")
    print("archive paths are relative to BRUISE_UNIFIED/ -- extract INSIDE the "
          "bundle, no -d\n")

    with zipfile.ZipFile(DST, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.writestr("README_PATHS_STAGE_Y.md", README)
        for src, arc, why in MEMBERS:
            zf.write(ROOT / src, arc)
            print(f"  {arc:<50} {why}")

    with zipfile.ZipFile(DST) as zf:
        bad = zf.testzip()
        n = len(zf.namelist())
    if bad:
        raise SystemExit(f"archive is corrupt at {bad}")

    print(f"\nDONE  {DST}\n      {DST.stat().st_size / 1e6:.2f} MB, {n} entries, "
          f"integrity check passed ({time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
