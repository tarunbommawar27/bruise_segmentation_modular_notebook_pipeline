#!/usr/bin/env python
"""Zip `bruisekit/` alone, for uploading to Drive so Colab can import it.

WHY THIS EXISTS SEPARATELY FROM THE OVERLAY ZIPS
--------------------------------------------------
`BRUISE_SPEED_HARNESS_V2.zip` is a PATCH: it carries the three files that
changed and is extracted over a bundle that already exists. Colab has no bundle
-- the dataset arrives as `bruise_colab_final.zip` and unpacks to
`/content/bruise_final` with no `bruisekit/` in it at all. Extracting a patch
there would produce a directory with two modules in it and nothing else.

So this ships the whole package, and only the package: no data, no checkpoints,
no pretrained weights (the dataset zip already carries `pretrained_weights/`),
no results. A few hundred KB against the dataset zip's 2.7 GB.

`paths.setup()` accepts a root only if it contains BOTH `bruisekit/` and
`manifests/train.csv`. The dataset zip provides the second; this provides the
first. Extract it INTO the unpacked dataset dir and that directory becomes a
valid bundle root.

    cd /content/bruise_final && unzip -o BRUISEKIT_LIB.zip

`bruise_colab_speed_harness.ipynb` §5 does this for you.

WHAT IS DELIBERATELY INCLUDED
-------------------------------
The whole package plus `vendor/`, rather than the import closure of the speed
path. The closure is 17 files today and would silently become wrong the first
time someone adds an import -- and the saving is measured in kilobytes against a
2.7 GB dataset. `kd/` is excluded because it is the training half: several
thousand lines that the benchmark never touches, and shipping it would mean a
version skew there could break an import that has nothing to do with timing.
"""
from __future__ import annotations

import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "BRUISE_UNIFIED" / "bruisekit"
DST = ROOT / "BRUISEKIT_LIB.zip"

# Modules whose presence proves this was built from a patched bundle. Without
# gpustate.py the notebook's §5 check fires and the upload is wasted.
REQUIRED = ("gpustate.py", "inference.py", "loaders.py", "paths.py", "registry.py")


def main() -> int:
    if not SRC.is_dir():
        raise SystemExit(f"{SRC} not found -- run 60_build_unified_bundle.py first.")

    missing = [n for n in REQUIRED if not (SRC / n).exists()]
    if missing:
        raise SystemExit(f"{SRC} is missing {missing}. Apply the speed-harness patch "
                         f"to the bundle before building the lib zip.")

    if "precision" not in (SRC / "inference.py").read_text(encoding="utf-8"):
        raise SystemExit("bruisekit/inference.py has no `precision` argument -- this is "
                         "the pre-patch copy. Rebuild the bundle, then re-run this.")

    files = [p for p in sorted(SRC.rglob("*.py"))
             if "__pycache__" not in p.parts and "kd" not in p.relative_to(SRC).parts]

    t0 = time.time()
    total = sum(p.stat().st_size for p in files)
    print(f"zipping {len(files)} modules, {total / 1e6:.2f} MB -> {DST.name}")

    with zipfile.ZipFile(DST, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for p in files:
            zf.write(p, str(Path("bruisekit") / p.relative_to(SRC)).replace("\\", "/"))

    with zipfile.ZipFile(DST) as zf:
        bad = zf.testzip()
        names = zf.namelist()
    if bad:
        raise SystemExit(f"archive is corrupt at {bad}")
    for n in REQUIRED:
        if f"bruisekit/{n}" not in names:
            raise SystemExit(f"bruisekit/{n} did not make it into the archive")

    n_vendor = sum(1 for n in names if "/vendor/" in n)
    print(f"  {len(names)} entries ({n_vendor} vendored nets)")
    print(f"\nDONE  {DST}\n      {DST.stat().st_size / 1e6:.2f} MB, integrity check passed "
          f"({time.time() - t0:.1f}s)")
    print(f"\nUpload to: /content/drive/MyDrive/bruise_segmentation_gpu/{DST.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
