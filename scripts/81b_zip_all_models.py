#!/usr/bin/env python
"""Zip the all-models analysis -- one module, one notebook, one runner.

WHAT IT ADDS, AND WHAT IT DOES NOT TOUCH
------------------------------------------
    bruisekit/allmodels.py    NEW FILE  discovery, cohorting, every endpoint
    bruise_all_models.ipynb   NEW FILE  run top to bottom
    run_all_models.py         NEW FILE  the same thing from a shell

Pure analysis. It trains nothing, runs no inference, and re-fits no threshold --
every per-image CSV it touches is READ. No shipped module, no existing notebook
and no result file is overwritten, so applying this cannot change a published
number.

WHAT IT DEPENDS ON, AND WHY THOSE ARE NOT SHIPPED HERE
-------------------------------------------------------
    bruisekit/report.py        normalize()  -- in the bundle already
    bruisekit/significance.py  holm()       -- in the bundle already
    bruisekit/paths.py         setup()      -- in the bundle already

All three are build outputs of 60_build_unified_bundle.py, so they are present in
any bundle. `lesionsize.py` is deliberately NOT a dependency: its `assign_bins`
raises when two tables disagree about an image's GT area, which is correct for
one coherent lineage and fatal for a sweep over fifty heterogeneous files.
allmodels detects cohorts instead. See its docstring.

WHY allmodels.py IS NOT A BUILD OUTPUT
---------------------------------------
Same policy as foundation.py, dermprobe.py, finetune_n3.py, samprobe.py,
multiteacher.py and lesionsize.py: analysis and experiment modules are authored
in bruisekit/ but kept out of 60_build's copy list. This script FAILS if it has
been added to that list, because then the overlay would ship a stale duplicate.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DST = ROOT / "BRUISE_ALL_MODELS.zip"

MEMBERS: list[tuple[str, str, str]] = [
    ("BRUISE_UNIFIED/bruisekit/allmodels.py", "bruisekit/allmodels.py",
     "discover, load_all, assign_bins (per cohort), summarize, by_decile, "
     "by_group, discovery_log, run, save, self_test."),
    ("BRUISE_UNIFIED/bruise_all_models.ipynb", "bruise_all_models.ipynb",
     "18 cells. 2 is discovery, 3 runs it, 5 is the three views, 7 says what "
     "was skipped and why."),
    ("BRUISE_UNIFIED/run_all_models.py", "run_all_models.py",
     "the same analysis from a shell. --discover-only lists what is on disk."),
    ("scripts/81_generate_all_models_notebook.py",
     "_source/scripts/81_generate_all_models_notebook.py",
     "emits the notebook -- its source of truth."),
    ("scripts/81b_zip_all_models.py", "_source/scripts/81b_zip_all_models.py",
     "this script."),
]

PREREQS: list[str] = [
    "BRUISE_UNIFIED/bruisekit/report.py",
    "BRUISE_UNIFIED/bruisekit/significance.py",
    "BRUISE_UNIFIED/bruisekit/paths.py",
]

README = """\
BRUISE ALL MODELS -- every model, every endpoint, one CSV
==========================================================

WHAT THIS IS
------------
LESION_SIZE_RESULTS/ covers 18 arms. STAGE_N4_RESULTS/ covers four more. The
Stage C distillation grid, the Stage H reliability-gated arms and the Stage M
multi-teacher arms have per-image CSVs on disk and have NEVER been through a size
or miss analysis at all.

That matters, because those families were called NULL ON DICE -- and Dice is the
endpoint this study has argued all along is saturated (Friedman p = 0.61 across
the seven headline models). "We looked and it was null" and "we did not look" are
different claims, and only one of them is currently true.

This computes, for every model with a per-image table anywhere:

    mean / median Dice        with a subject-clustered CI and the IQR
    zero-Dice count and rate  the PUBLISHED complete-miss definition (dice == 0)
    wrong-place misses        zero Dice WHILE PREDICTING SOMETHING -- the per-seed
                              tables cannot see these, and it is the failure a
                              clinician would care about most
    small-lesion recall       D1-D4 (four smallest GT-area deciles) and D1 alone
    fairness                  per-ITA-group Dice, recall, miss rate; gap; Kruskal
    conditioned fairness      does the gap shrink inside the small stratum?

It trains nothing, runs no inference, and re-fits no threshold.

HOW TO APPLY
------------
1. Unzip into the bundle root, over the top:

       unzip -o BRUISE_ALL_MODELS.zip -d /path/to/BRUISE_UNIFIED

   Three new files. It overwrites nothing you have results in.

2. Either:

       python run_all_models.py --discover-only     # what is on this machine
       python run_all_models.py                     # the whole thing

   or open bruise_all_models.ipynb and run top to bottom.

ON ORC, SET THE ROOTS FIRST -- THIS IS THE ONE THING THAT GOES WRONG
---------------------------------------------------------------------
The runs are NOT all in the bundle. Outputs land under the work directory on
scratch, and FINAL_RESULT/ may never have been synced there at all. A hard-coded
tree is exactly what made lesionsize raise FileNotFoundError on ORC for a
directory that was never going to exist.

So pass every plausible tree:

    python run_all_models.py --extra-roots \\
        /scratch/tbommawa/bruise_work \\
        /scratch/tbommawa/bruise_work/outputs \\
        /scratch/tbommawa/BRUISE_UNIFIED

An over-broad list costs a few seconds of globbing. discovery_log.csv records
which roots actually held anything, so run --discover-only first and read it.

THE COHORT MECHANISM -- READ THIS BEFORE READING THE TABLE
------------------------------------------------------------
The input is deliberately heterogeneous: CSVs from several lineages, splits and
mask versions. lesionsize.assign_bins RAISES when two tables disagree about an
image's GT area -- correct for binning one coherent lineage, fatal for a sweep
that must survive a stale file.

So this detects COHORTS instead. Tables are grouped by (stem set, GT-area
vector); the largest coherent group becomes the reference cohort and is binned
once; every other group is binned separately and flagged comparable = False.

    ONLY ROWS WITH comparable = True MAY BE COMPARED AGAINST EACH OTHER.

The rest are kept, labelled and quarantined -- not silently mixed, and not
silently dropped either.

DEDUPLICATION
-------------
The same model appears under several filenames (per_image_x.csv,
x_best_seed0_test_per_image.csv, reference_x_test_per_image.csv). They are
collapsed by comparing the DICE VECTOR, not the name: byte-identical scores are
the same run whatever the file is called. The shortest name wins, and the rest
are recorded in discovery_log.csv as aliases so a reader can see the collapse
happened rather than wondering where a file went.

On the reference bundle this collapsed 103 files into 46 distinct models, 42 of
them comparable.

WHERE RESULTS GO
----------------
    ALL_MODELS_RESULTS/
      ALL_MODELS_SUMMARY.csv    <- THE SINGLE TABLE, one row per model
      all_models_by_decile.csv     long form: (model, decile)
      all_models_by_group.csv      long form: (model, stratum, ITA group)
      discovery_log.csv            every file found, what it became, and why
      size_bins.csv                the decile assignment, per cohort
      run_info.json

Nothing is written to results/, FINAL_RESULT/, LESION_SIZE_RESULTS/ or any
stage's own directory.

    zip -r all_models.zip ALL_MODELS_RESULTS

HOW TO READ IT
--------------
READ THE MISS COLUMNS BEFORE THE DICE COLUMN. 23x the parameters bought +0.006
Dice on this task and the headline omnibus does not reject (p = 0.61). Every
remaining signal lives in complete misses, small-lesion recall and fairness.

A DICE TIE WITH A MISS-RATE DIFFERENCE IS A RESULT, not an absence of one.

WHAT IT CANNOT TELL YOU
------------------------
- THESE ARE NOT PAIRED TESTS. The table is descriptive: one row per model with
  MARGINAL CIs. Two rows whose intervals overlap are NOT therefore equal -- a
  paired subject-clustered bootstrap on the same images is far more sensitive.
  Use lesionsize.paired_stratum for any claim about a specific pair.
- NOTHING IS MULTIPLICITY-CORRECTED except the Kruskal column, where the notebook
  applies Holm across models and prints both counts. ~40 models at alpha = 0.05
  expects ~2 to clear from noise alone. Pick the contrast first, then test it.
- EVERY NUMBER IS AT WHATEVER OPERATING POINT ITS SOURCE RUN FITTED. This re-fits
  nothing. A model whose CSV was written at a badly-chosen cut looks bad here,
  correctly -- but that is a property of the run, not of the model.
- THE FAIRNESS GAP IS DESCRIPTIVE AND THE KRUSKAL TEST IS INFERENTIAL, and they
  routinely disagree here. The gap is a max-minus-min over five noisy estimates
  and is biased UPWARD. Do not read a gap as a test.
"""


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def main() -> int:
    for rel in PREREQS:
        if not (ROOT / rel).exists():
            print(f"FAIL: {rel} is absent -- allmodels.py imports it.",
                  file=sys.stderr)
            return 1

    build = ROOT / "scripts" / "60_build_unified_bundle.py"
    if build.exists() and "allmodels" in build.read_text(encoding="utf-8"):
        print("FAIL: 60_build_unified_bundle.py now copies allmodels.py. It has "
              "been graduated into the bundle, so this overlay is obsolete -- "
              "shipping it would create a second copy.", file=sys.stderr)
        return 1

    missing = [src for src, _, _ in MEMBERS if not (ROOT / src).exists()]
    if missing:
        print(f"FAIL: missing {missing}", file=sys.stderr)
        return 1

    nb = json.loads((ROOT / "BRUISE_UNIFIED" / "bruise_all_models.ipynb")
                    .read_text(encoding="utf-8"))
    if any(c.get("outputs") for c in nb["cells"]):
        print("FAIL: the notebook has executed outputs. Regenerate it with "
              "scripts/81_generate_all_models_notebook.py.", file=sys.stderr)
        return 1

    # No weights, no GPU, no network -- there is no excuse for skipping this.
    sys.path.insert(0, str(ROOT / "BRUISE_UNIFIED"))
    from bruisekit import allmodels
    print("-- allmodels self-test --")
    if not allmodels.self_test(verbose=False):
        print("FAIL: allmodels.self_test() did not pass. Not shipping this.",
              file=sys.stderr)
        return 1
    print("   ALL PASS\n")

    manifest = {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "members": []}
    DST.unlink(missing_ok=True)
    with zipfile.ZipFile(DST, "w", zipfile.ZIP_DEFLATED) as z:
        for src, arc, note in MEMBERS:
            p = ROOT / src
            z.write(p, arc)
            manifest["members"].append(
                {"arcname": arc, "bytes": p.stat().st_size, "sha256_16": sha(p),
                 "note": note})
            print(f"  + {arc:<48} {p.stat().st_size / 1024:7.1f} KB  {sha(p)}")
        z.writestr("README.txt", README)
        z.writestr("MANIFEST.json", json.dumps(manifest, indent=2))

    print(f"\nwrote {DST}  ({DST.stat().st_size / 1024:.1f} KB)")
    print("upload this one file")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
