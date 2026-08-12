#!/usr/bin/env python
"""Zip the speed-harness overlay: precision as a column, and GPU clock state on every row.

WHAT THIS FIXES
----------------
Three speed tables were published from three machines and compared as though the
machine were the only thing that differed between them. It was not.

Measured 2026-08-04 on ONE Colab A100, one session, one architecture:

    evaluate.benchmark_speed   segformer_b0   16.55 ms
    the 51-run sweep           segformer_b0    8.74 ms

1.89x apart with no machine to blame. `benchmark_speed` contains no autocast
anywhere in it, so the published table is pure fp32, while every other forward in
the codebase runs under `torch.amp.autocast`. On an A100 that is roughly 2x for a
transformer and much less for a depthwise-conv mobile net -- which is the shape of
the gap. It stayed a hypothesis because there was no fp16 path to compare against
on one machine. This overlay is that path.

The same run also settled the older worry: `bruise_colab_final.ipynb` reproduced
its own July table to under 1% (16.55 vs 16.68 ms; the B2 teacher 33.649 vs
33.668). The shipped `benchmark_640.csv` was never suspect. What is suspect is
comparing any two rows without first checking how each was measured.

WHAT IS IN THE LIBRARY HALF
----------------------------
  bruisekit/gpustate.py    NEW. persistence mode, applications vs max clock, power
                           limit, throttle reasons -- and `ClockSampler`, which polls
                           the SM clock on a background thread WHILE the timing loop
                           runs. Handbook 18.5 item 1 has been owed this since the
                           765 MHz reading on the private server was found at idle
                           and never confirmed under load. Every value degrades to
                           None; a missing clock reading must never kill a sweep.

  bruisekit/inference.py   `speed_table(precision=...)`, `_benchmark_logit_amp` and
                           `_benchmark_yolo_amp` (mirrors of the published loops
                           differing in one line), the GPU-state columns,
                           `peak_incremental_MB`, and `--precision` / `--machine-tag`.

`_benchmark_logit_amp` is a SEPARATE function rather than a flag on
`evaluate.benchmark_speed`, for the reason `_benchmark_cpu` already gives in that
file: benchmark_speed produces the five shipped rows and is written out verbatim
by the notebook's %%writefile cell, so a branch inside it would put the published
number one default-argument change away from silently becoming a different number.

TWO CORRECTNESS FIXES THAT CAME OUT OF THE SAME READ
------------------------------------------------------
1. `check_single_machine` now rejects mixed PRECISION as well as mixed devices.
   It already refused to merge a laptop row into a GPU table; a fp16 row next to
   a fp32 row is the same class of error and was the one that actually bit.

2. `peak_activation_MB` is not activation memory and never was.
   `reset_peak_memory_stats` seeds the peak at whatever is already resident, and
   all 185 test images stage before the loop -- 185x3x640x640x4 B = 909 MB carried
   by every row. That is why all four Stage E mobile nets report ~1000 MB
   regardless of size. The column is kept (it is what the shipped CSV calls its
   memory column) and `peak_incremental_MB` is added beside it.

LAYOUT: THE SAME CONVENTION AS 63_zip_rgkd_overlay.py
-------------------------------------------------------
Paths are relative to the bundle, sources under `_source/`. Upload into
BRUISE_UNIFIED and extract in place, no `-d`.

Both halves ship, for the reason handbook 16 gives: bruisekit/*.py are build
outputs of scripts/unified_lib/, so an overlay shipping only the outputs would
leave the next `60_build_unified_bundle.py` free to silently revert all of it.
`60_build_unified_bundle.py` itself is included because gpustate.py is a NEW
module and had to be added to its copy list -- without that edit a clean rebuild
produces a bundle whose inference.py imports a module that is not in it.
"""
from __future__ import annotations

import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "BRUISE_UNIFIED"
DST = ROOT / "BRUISE_SPEED_HARNESS_V2.zip"

# (source path relative to ROOT, path INSIDE the archive, why it is here).
MEMBERS: list[tuple[str, str, str]] = [
    # ── runnable: the notebook ────────────────────────────────────────────────
    # Every run in this study happens in Jupyter. An overlay that ships only a
    # `python -m` entry point makes the recipient rebuild the driver by hand, so
    # the cells ship with the library change that needs them.
    ("BRUISE_UNIFIED/bruise_speed_harness.ipynb", "bruise_speed_harness.ipynb",
     "16 cells: patch check, registry check BEFORE any GPU time, fp32, fp16, verdict."),

    # ── runnable: the library ─────────────────────────────────────────────────
    ("BRUISE_UNIFIED/bruisekit/gpustate.py", "bruisekit/gpustate.py",
     "NEW -- persistence mode, app/max clocks, ClockSampler (under load), describe()."),
    ("BRUISE_UNIFIED/bruisekit/inference.py", "bruisekit/inference.py",
     "precision= on speed_table, the two amp mirrors, GPU-state columns, "
     "peak_incremental_MB, --precision/--machine-tag, precision guard, "
     "per-model failure isolation."),
    ("BRUISE_UNIFIED/bruisekit/loaders.py", "bruisekit/loaders.py",
     "SegFormer transformers-layout reconciliation at load time, both directions, "
     "each verified against the model's own key set before being accepted."),

    # ── sources (handbook 16) ─────────────────────────────────────────────────
    ("scripts/unified_lib/gpustate.py",
     "_source/scripts/unified_lib/gpustate.py", "source of truth for the new module."),
    ("scripts/unified_lib/inference.py",
     "_source/scripts/unified_lib/inference.py", "source of truth."),
    ("scripts/unified_lib/loaders.py",
     "_source/scripts/unified_lib/loaders.py", "source of truth."),
    ("scripts/60_build_unified_bundle.py",
     "_source/scripts/60_build_unified_bundle.py",
     "gpustate.py added to the authored-module copy list -- REQUIRED, or a clean "
     "rebuild ships an inference.py whose import fails."),
    ("scripts/65_generate_speed_harness_notebook.py",
     "_source/scripts/65_generate_speed_harness_notebook.py", "emits the notebook."),
    ("scripts/64_zip_speed_harness_overlay.py",
     "_source/scripts/64_zip_speed_harness_overlay.py", "this script."),
]

README = """\
# Speed harness v2 — the runs-directory fix, plus precision as a column

    cd /scratch/$USER/BRUISE_UNIFIED
    unzip -o BRUISE_SPEED_HARNESS_V2.zip

No `-d`. **Restart the kernel afterwards** — `bruisekit` is already imported in a
running session and `importlib.reload` will not reliably pick up files replaced
underneath it. Three Python files and one notebook change; no data, no
checkpoints, no results move.

Then open **`bruise_speed_harness.ipynb`** and run it top to bottom.

## READ THIS FIRST — why v1's run died

`registry.py` looks for each checkpoint in `env.runs` (= `WORK/runs`) **first**,
then silently falls back to the bundle's shipped `checkpoints/`. The fallback
prints nothing.

On ORC, `WORK` defaulted to `BRUISE_UNIFIED/_work`, but the real runs are in
`/scratch/$USER/bruise_work/runs`. So all three SegFormers loaded the **shipped**
copies — a different lineage, saved under a different `transformers` module
layout — and died with a missing-key `RuntimeError`.

**The crash was the second symptom. The first was silent**, and the mobile nets
that *did* time may have come from the wrong place too.

Three things now prevent that:

| | |
|---|---|
| §2 sets `WORK` explicitly | auto-detects `/scratch/$USER/bruise_work`, falls back only as a last resort |
| §3 counts `env.runs` and **raises if empty** | no more silent fallback to shipped checkpoints |
| §3 prints a `source` column per family | `runs` = yours, `shipped` = the bundle's, with a WARNING listing every fallback |

`loaders.py` also reconciles the SegFormer layout rename at load time now (both
directions, each checked against the model's own key set before being accepted),
so a genuine version skew is survivable rather than fatal. A real architecture
mismatch still raises — only the known `encoder.block` ↔ `stages.N.blocks` rename
is papered over, and it says so when it does.

## Why the precision half exists

On one Colab A100, one session, one architecture, measured 2026-08-04:

| harness | segformer_b0 |
|---|---|
| `evaluate.benchmark_speed` | 16.55 ms |
| the 51-run sweep | 8.74 ms |

1.89x apart with no machine to blame. `benchmark_speed` has **no autocast in it** —
the published table is fp32, while every other forward in this codebase runs under
`torch.amp.autocast`. That is the leading explanation and it was untestable,
because there was no fp16 path to compare against on one machine.

Now there is. Run both and read the answer off the two files.

Separately: `bruise_colab_final.ipynb` reproduced its own July table to under 1%
(16.55 vs 16.68 ms). The shipped `benchmark_640.csv` was never the problem.

## Verify it applied

§1 of the notebook does this for you and stops if anything is missing. By hand:

    python -c "from bruisekit import gpustate; print(gpustate.describe(gpustate.probe()))"
    python -m bruisekit.inference --help | grep -q -- --precision && echo APPLIED

On a healthy node the first prints persistence, clocks and power limit. On the
private server it should print two WARNING lines — that is the finding, not a
failure. Inside a MIG instance the applications clock reads `[N/A]`; that is
expected and does not invalidate the row.

## What the ORC fp32/fp16 run already showed

From the v1 run, before the SegFormer rows were fixed — **autocast made the small
models slower, not faster**:

| model | fp32 | fp16 | |
|---|---|---|---|
| fastscnn | 3.57 ms | 4.83 ms | 1.35x SLOWER |
| lraspp_mobilenetv3 | 5.02 | 6.77 | 1.35x slower |
| topformer_tiny | 6.22 | 8.44 | 1.36x slower |
| ppmobileseg_tiny | 10.76 | 14.49 | 1.35x slower |
| yolo_sem_direct | 5.80 | 6.98 | 1.20x slower |
| deeplabv3plus_r50 | 10.44 | 7.52 | 1.39x faster |
| unet_r50 | 11.82 | 10.17 | 1.16x faster |

At batch 1 the per-op cast overhead dominates for a small net; only the two big
ResNets are compute-bound enough to win from tensor cores. Which means the
original guess -- "the sweep was fp16, so it looked 1.9x faster" -- has the sign
backwards for the mobile nets, and the SegFormer row is the only one that can
settle it. That row is what this version unblocks.

Worth noting: the sweep's mobile numbers (4.73 / 6.70 / 8.16 / 13.85) sit within
1-5% of ORC's **fp16** column and ~30% off its fp32 column. Suggestive, but those
are different machines -- run this notebook on Colab too before drawing the line.

## Run it — open `bruise_speed_harness.ipynb`

Eighteen cells, top to bottom. Set `WORK` in §2 (or let it auto-detect) and
`MACHINE_TAG` in §4. Nothing else to edit.

§3 lists every family with a seed-0 checkpoint **and where that checkpoint came
from**, before any GPU time is spent. `MODELS` takes registry family names, and a
name the registry does not know is skipped rather than raised — so the failure
mode is a long benchmark ending in an empty table. Check that list, then run §6
and §7.

A model that cannot be loaded is now reported as `FAIL`, listed again at the end,
and **left out** — one bad checkpoint no longer discards the models already timed.
The table says explicitly that it is not a complete sweep.

Three files land in `<WORK>/outputs/inference/`:

    benchmark_640_cuda_fp32_<tag>.csv
    benchmark_640_cuda_fp16_<tag>.csv
    speed_precision_comparison_<tag>.csv

Same notebook on every machine — change `MACHINE_TAG`, nothing else. The
filenames carry device and precision, so nothing collides.

The CLI is still there if you want it headless:

```bash
python -m bruisekit.inference --no-inference --machine-tag orc-mig --precision fp32 \\
  --models fastscnn_rgkd lraspp_mobilenetv3_rgkd topformer_tiny_rgkd segformer_b0_rgkd
```

## New columns

| column | |
|---|---|
| `precision` | `fp32` (published) or `fp16`. **Check this before comparing any two rows.** |
| `peak_incremental_MB` | peak minus the 909 MB of staged test images. `peak_activation_MB` never subtracted them — that is why all four Stage E mobile nets read ~1000 MB regardless of size. |
| `persistence_mode`, `clock_applications_mhz`, `clock_max_graphics_mhz`, `clock_headroom` | static GPU config. 765/1410 = 0.54 is the private-server reading that started this. |
| `sm_clock_under_load_{median,max,min}_mhz` | what the clock actually **did** during the timing loop. Handbook §18.5 item 1 has been owed this. |

## Two guards that now refuse rather than warn

`check_single_machine` raises on a table mixing devices (it already did) **or
mixing precisions** (new). Both produce a file that reads like a hardware result
and is not one.

## What this does NOT settle

The 51-run sweep script is not in the bundle or the repo — `peak_incremental_mb`
and `training_variant` match zero files. Until it turns up, "the sweep used fp16"
is the leading hypothesis, not a finding. The fp32/fp16 pair above tests it
without needing the script at all: if fp16 lands near 8.7 ms and fp32 near 16.5,
the question is closed.

Handbook §7.3 and §14 caveat 6 still describe the SegFormer row as a
`transformers`-version anomaly. Note that `benchmark_stage_e.csv` contains only
the four mobile nets — there is **no ORC SegFormer row from that harness** — yet
caveat 6 lists one at 8.97 ms, within 3% of the sweep's 8.74. Those sections
should not be rewritten until the four CSVs above exist.
"""


def main() -> int:
    missing = [s for s, _, _ in MEMBERS if not (ROOT / s).exists()]
    if missing:
        raise SystemExit("not built yet -- copy the modules into scripts/unified_lib/ and "
                         "run 60_build_unified_bundle.py first. Missing:\n  "
                         + "\n  ".join(missing))

    # The bundle copy and the source of truth must be byte-identical, or the next
    # rebuild reverts the overlay and the failure is silent and weeks later.
    for bundled, authored in (("BRUISE_UNIFIED/bruisekit/inference.py",
                               "scripts/unified_lib/inference.py"),
                              ("BRUISE_UNIFIED/bruisekit/loaders.py",
                               "scripts/unified_lib/loaders.py"),
                              ("BRUISE_UNIFIED/bruisekit/gpustate.py",
                               "scripts/unified_lib/gpustate.py")):
        if (ROOT / bundled).read_bytes() != (ROOT / authored).read_bytes():
            raise SystemExit(
                f"{bundled} and {authored} differ. The bundle copy is a build output of "
                f"the authored file (handbook 16); shipping them out of sync means the "
                f"next 60_build_unified_bundle.py silently reverts this overlay.")

    t0 = time.time()
    total = sum((ROOT / s).stat().st_size for s, _, _ in MEMBERS)
    print(f"zipping {len(MEMBERS)} files, {total / 1e6:.2f} MB -> {DST.name}")
    print("archive paths are relative to BRUISE_UNIFIED/ -- extract INSIDE the "
          "bundle, no -d\n")

    with zipfile.ZipFile(DST, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.writestr("README_SPEED_HARNESS.md", README)
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
