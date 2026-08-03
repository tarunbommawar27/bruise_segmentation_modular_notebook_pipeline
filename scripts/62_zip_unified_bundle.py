#!/usr/bin/env python
"""Zip BRUISE_UNIFIED/ for distribution.

WHAT IS EXCLUDED AND WHY
-------------------------
_work/       Scratch. Holds the 640 cache and this machine's session outputs --
             both are derived, both rebuild on first run, and together they are
             ~800 MB of bytes that would go stale the moment anyone re-ran a cell.
__pycache__/ Bytecode for a specific Python version. Shipping it invites an
             ImportError on a different interpreter for no benefit.

COMPRESSION
-----------
Most of the payload is JPEG images and .pt tensors, which are already compressed;
deflating them again costs minutes and saves almost nothing. Those are STORED,
and only the genuinely compressible files (code, CSV, JSON, YAML, Markdown) are
deflated. That keeps the archive close to the folder's size while still shrinking
the parts that shrink.
"""
from __future__ import annotations

import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "BRUISE_UNIFIED"
DST = ROOT / "BRUISE_UNIFIED.zip"

EXCLUDE_DIRS = {"_work", "__pycache__", ".ipynb_checkpoints", ".git"}
# Extensions that are already compressed -- storing them is faster and just as small.
STORE_EXT = {".jpg", ".jpeg", ".png", ".pt", ".pth", ".safetensors", ".bin",
             ".zip", ".pdf", ".xlsx", ".gz"}


def main() -> int:
    if not SRC.exists():
        raise SystemExit(f"{SRC} not found -- run 60_build_unified_bundle.py first")

    files = [p for p in SRC.rglob("*")
             if p.is_file() and not (EXCLUDE_DIRS & set(p.relative_to(SRC).parts))]
    total = sum(p.stat().st_size for p in files)
    print(f"zipping {len(files)} files, {total / 1e9:.2f} GB -> {DST.name}")

    t0 = time.time()
    done = 0
    with zipfile.ZipFile(DST, "w", allowZip64=True) as zf:
        for p in files:
            rel = Path("BRUISE_UNIFIED") / p.relative_to(SRC)
            method = (zipfile.ZIP_STORED if p.suffix.lower() in STORE_EXT
                      else zipfile.ZIP_DEFLATED)
            zf.write(p, rel.as_posix(), compress_type=method)
            done += p.stat().st_size
            if done % (500 * 1024 * 1024) < p.stat().st_size:
                print(f"  {done / 1e9:.1f} / {total / 1e9:.1f} GB "
                      f"({time.time() - t0:.0f}s)", flush=True)

    size = DST.stat().st_size
    print(f"\nDONE  {DST}\n      {size / 1e9:.2f} GB in {time.time() - t0:.0f}s")

    # A zip that cannot be opened is worse than no zip; verify before reporting success.
    with zipfile.ZipFile(DST) as zf:
        bad = zf.testzip()
        n = len(zf.namelist())
    if bad:
        raise SystemExit(f"archive is corrupt at {bad}")
    print(f"      integrity check passed, {n} entries readable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
