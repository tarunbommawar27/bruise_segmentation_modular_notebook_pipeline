#!/usr/bin/env python
"""Zip the Stage H overlay: the notebook, the library, the sources and the docs.

WHY AN OVERLAY AND NOT THE WHOLE BUNDLE
----------------------------------------
`62_zip_unified_bundle.py` packages BRUISE_UNIFIED in full -- 6.45 GB, most of it
1016 photographs and 27 checkpoints that Stage H does not change a byte of.
Re-shipping them to deliver a new loss function is minutes of I/O for no
information.

This archive is everything Stage H touched and nothing else.

LAYOUT: THE SAME CONVENTION AS THE EARLIER PATCH ZIPS
------------------------------------------------------
`bruisekit_distill_patch.zip`, `bruisekit_lraspp_distill_patch.zip` and
`bruisekit_significance_patch.zip` all store paths RELATIVE TO the bundle, with
the build sources under `_source/`. So the recipient uploads the zip into
BRUISE_UNIFIED and extracts in place:

    cd /path/to/BRUISE_UNIFIED
    unzip -o BRUISE_RGKD_OVERLAY.zip

An earlier revision of this script wrote `BRUISE_UNIFIED/...`-prefixed paths,
which needed `unzip -d ..` and scattered `docs/` and `scripts/` into the bundle's
PARENT. It worked, but it silently broke the muscle memory built by three
previous patches -- and a wrong `unzip -o` with those paths produces a nested
`BRUISE_UNIFIED/BRUISE_UNIFIED/` that looks like it applied and did not. Matching
the established convention removes both hazards.

Both halves are included on purpose:

  bruisekit/, *.ipynb, *.md   the RUNNABLE artefacts. This is what executes.
  _source/scripts/            the SOURCES they are built from. Handbook 16: the
                              notebook and bruisekit/*.py are build outputs, and
                              an overlay shipping only the outputs would leave the
                              recipient unable to regenerate them -- the next
                              `60_build_unified_bundle.py` would silently revert
                              every change in this zip.

Run `62_zip_unified_bundle.py` instead when the recipient has no bundle at all.
"""
from __future__ import annotations

import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "BRUISE_UNIFIED"
DST = ROOT / "BRUISE_RGKD_OVERLAY.zip"

# (source path relative to ROOT, path INSIDE the archive, why it is here).
# The archive path is relative to BRUISE_UNIFIED -- see the module docstring.
MEMBERS: list[tuple[str, str, str]] = [
    # ── runnable: the notebook ────────────────────────────────────────────────
    ("BRUISE_UNIFIED/bruise_unified.ipynb", "bruise_unified.ipynb",
     "78 cells, seven stages (A/B/C/E/H/D/G). Stage H and G2b are new."),

    # ── runnable: the library ─────────────────────────────────────────────────
    ("BRUISE_UNIFIED/bruisekit/reliability_kd.py", "bruisekit/reliability_kd.py",
     "NEW -- the gated loss, both runtime shims, the gated YOLO pseudo-mask "
     "builder, CONTRAST_FAMILY_H, self_test."),
    ("BRUISE_UNIFIED/bruisekit/loaders.py", "bruisekit/loaders.py",
     "12 new FAMILY_SPEC rows; EFFICIENT_FAMILIES extended to 16."),
    ("BRUISE_UNIFIED/bruisekit/registry.py", "bruisekit/registry.py",
     "_scan_stage_h, STAGE_H_FAMILIES, rgkd_seeds, Stage H/F cost estimates."),
    ("BRUISE_UNIFIED/bruisekit/distill_efficient.py", "bruisekit/distill_efficient.py",
     "TopFormer and PP-MobileSeg DeepLabV3+ arms; B2 in TEACHER_STAGE_DIR."),
    ("BRUISE_UNIFIED/bruisekit/significance.py", "bruisekit/significance.py",
     "three new omnibus sets; the existing ones are untouched."),

    # ── documentation ─────────────────────────────────────────────────────────
    ("BRUISE_UNIFIED/PROJECT_HANDBOOK.md", "PROJECT_HANDBOOK.md",
     "new 7c, gap 3c, and the 9/12/16/17 cross-references."),
    ("docs/reliability_gated_kd_scope.md", "_source/docs/reliability_gated_kd_scope.md",
     "NEW -- the scope note: what was decided, why, and what counts as failure."),

    # ── sources (handbook 16) ─────────────────────────────────────────────────
    ("scripts/unified_lib/reliability_kd.py",
     "_source/scripts/unified_lib/reliability_kd.py", "source of truth for the module."),
    ("scripts/unified_lib/loaders.py",
     "_source/scripts/unified_lib/loaders.py", "source of truth."),
    ("scripts/unified_lib/registry.py",
     "_source/scripts/unified_lib/registry.py", "source of truth."),
    ("scripts/unified_lib/distill_efficient.py",
     "_source/scripts/unified_lib/distill_efficient.py", "source of truth."),
    ("scripts/unified_lib/significance.py",
     "_source/scripts/unified_lib/significance.py", "source of truth."),
    ("scripts/61_generate_unified_notebook.py",
     "_source/scripts/61_generate_unified_notebook.py", "emits the notebook."),
    ("scripts/60_build_unified_bundle.py",
     "_source/scripts/60_build_unified_bundle.py",
     "copies reliability_kd.py into bruisekit/."),
    ("scripts/63_zip_rgkd_overlay.py",
     "_source/scripts/63_zip_rgkd_overlay.py", "this script."),
]

README = """\
# Stage H overlay — reliability-gated distillation + the SegFormer-B2 teacher axis

Upload into `BRUISE_UNIFIED` and extract in place — same as the earlier
`bruisekit_*_patch.zip` files:

    cd /path/to/BRUISE_UNIFIED
    unzip -o BRUISE_RGKD_OVERLAY.zip

No `-d`. It overwrites in place and adds no data, no checkpoints and no results —
nothing in the bundle's 6.4 GB of photographs or weights changes.

## What is in it

| path (relative to `BRUISE_UNIFIED/`) | |
|---|---|
| `bruise_unified.ipynb` | the notebook, now with Stage H and G2b |
| `bruisekit/reliability_kd.py` | **new** — the whole stage |
| `bruisekit/{loaders,registry,distill_efficient,significance}.py` | families, scanning, omnibus sets |
| `PROJECT_HANDBOOK.md` | new §7c |
| `_source/docs/reliability_gated_kd_scope.md` | **new** — the scope note |
| `_source/scripts/…` | the sources the runnable files are built from |

The `_source/` half matters: the notebook and `bruisekit/*.py` are build artefacts
(handbook §16), so without the sources the next `60_build_unified_bundle.py` would
silently revert everything here.

## Verify it applied

    grep -c BRUISE_NO_PIP bruise_unified.ipynb     # 4 = this version, 0 = old one

## Verify before spending GPU time

```python
from bruisekit import reliability_kd as RK
RK.self_test()          # 5 checks, CPU, under a second — raises on failure
```

The decisive one is the first: with a uniformly reliable teacher the gated loss
**equals** `losses.DistillLoss` (measured 2.4e-07). That is what makes every
`*_rgkd` vs `*_b2kd` contrast a one-variable comparison.

## Run it

```python
RUN_STAGES     = "EH"        # Stage H's controls live in E; H alone is not enough
ALLOW_TRAINING = True
EFFICIENT_SEEDS = (0, 1, 2)
RGKD_SEEDS      = None       # None = same as EFFICIENT_SEEDS. Keep it that way:
                             # a 3-seed arm against a 1-seed control is not a contrast.
```

Cost at three seeds: Stage H 30 runs ≈ 27 GPU-hours, Stage E/F 24 runs ≈ 15.
Everything with a `DONE.json` is skipped, so this is resumable across sessions.

One arm at a time, without its controls:

```python
RK.train_arm(env, "lraspp_mobilenetv3_rgkd", (0, 1, 2), CFG, MAN640)
```

`train_arm` trains but does not threshold-sweep or score — re-run the notebook's
Stage H cells afterwards, or use `train_missing`, which does both.

## Read `GATE_H` before `HEAD_H`

Two failure directions look identical in a Dice table:

- `mean_coverage → 1.0` — the gate never fired; the arm **is** its control. Report
  that as the finding, not as a method.
- `mean_coverage → 0` — the gate ate the arm; it trained as a supervised baseline.
  Check `mean_teacher_soft_dice` against B2's 0.769 test mean, retune
  `rgkd_gate_lo`/`rgkd_gate_hi`, retrain. Do not reinterpret.

The notebook prints a loud line for either.

## The pre-registered endpoint is misses and fairness, not Dice

Mean Dice is capped by the annotation ceiling (handbook §1). Stage F's arms won on
the endpoint they were not built to win on and left their actual endpoint
untouched (§7b.7); if that happens again, say so in those words.

## Nothing here has been trained

All 30 Stage H runs and all 24 Stage E/F runs are MISSING, and the registry says
so. Every number in §7c is a design decision, not a measurement.

Full detail: `docs/reliability_gated_kd_scope.md` and handbook §7c.
"""


def main() -> int:
    missing = [s for s, _, _ in MEMBERS if not (ROOT / s).exists()]
    if missing:
        raise SystemExit("not built yet -- run 61_generate_unified_notebook.py and "
                         "60_build_unified_bundle.py first. Missing:\n  "
                         + "\n  ".join(missing))

    t0 = time.time()
    total = sum((ROOT / s).stat().st_size for s, _, _ in MEMBERS)
    print(f"zipping {len(MEMBERS)} files, {total / 1e6:.2f} MB -> {DST.name}")
    print("archive paths are relative to BRUISE_UNIFIED/ -- extract INSIDE the "
          "bundle, no -d\n")

    with zipfile.ZipFile(DST, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.writestr("README_STAGE_H.md", README)
        for src, arc, why in MEMBERS:
            zf.write(ROOT / src, arc)
            print(f"  {arc:<52} {why}")

    # A zip that cannot be opened is worse than no zip; verify before reporting.
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
