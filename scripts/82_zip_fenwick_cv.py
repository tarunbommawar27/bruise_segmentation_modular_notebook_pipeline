#!/usr/bin/env python
"""Zip everything the Fenwick labeler CV needs on the GPU server -- except the data.

WHAT IS IN HERE                                                          ~30 MB
    bruisekit/                       the package, source only
    pretrained_weights/segformer_mit_b0/   config + safetensors + torch bin
    bruise_fenwick_cv.ipynb          build the core, preflight, read the result
    scripts/80_fenwick_cv_run.py     the two-GPU launcher
    scripts/81_generate_..._.py      the notebook's source of truth

WHAT IS NOT, AND WHY
    FENWICK_LABELER_DATASET (1.4 GB). It is already on ORC; move it ORC ->
    server directly rather than round-tripping through a laptop. It is also
    patient imagery, and a copy inside a code archive is a copy nobody is
    tracking.

    tf_model.h5 (14.5 MB). TensorFlow weights for a PyTorch-only pipeline.

    Any run, table or checkpoint. The server produces those.

WHY THE WHOLE PACKAGE AND NOT SIX MODULES
------------------------------------------
`fenwick_cv` imports paths, loaders, engine, models, data, evaluate, sweep,
report, foundation and kd.kd_core, and those import further. Hand-picking the
closure is how you get a server that fails on the fourth module at 2 a.m.; the
package is 2.4 MB of source.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "BRUISE_UNIFIED"
DST = ROOT / "BRUISE_FENWICK_CV.zip"

#: Weight files that are actually loaded. `tf_model.h5` and `.cache/` are not.
B0_FILES = ("config.json", "model.safetensors", "pytorch_model.bin",
            "preprocessor_config.json")

SINGLES: list[tuple[str, str, str]] = [
    ("BRUISE_UNIFIED/bruise_fenwick_cv.ipynb", "bruise_fenwick_cv.ipynb",
     "22 cells. Cuts the core, preflights, probes the batch, reads the matrix."),
    ("scripts/80_fenwick_cv_run.py", "scripts/80_fenwick_cv_run.py",
     "the launcher -- one process per GPU."),
    ("scripts/81_generate_fenwick_cv_notebook.py",
     "scripts/81_generate_fenwick_cv_notebook.py",
     "emits the notebook -- its source of truth."),
]

README = """\
FENWICK LABELER CV -- which annotator's masks train the best model?
====================================================================

WHAT YOU NEED ON THE SERVER
----------------------------
1. This archive, unzipped into a working directory:

       unzip BRUISE_FENWICK_CV.zip -d ~/fenwick

   giving ~/fenwick/BRUISE_UNIFIED/{bruisekit,pretrained_weights} and
   ~/fenwick/scripts/.

2. THE DATASET, which is not in here (1.4 GB, and it is patient imagery).
   Pull it straight from ORC rather than via a laptop:

       scp -r tbommawa@hopper.orc.gmu.edu:/scratch/tbommawa/BRUISE_UNIFIED/FENWICK_LABELER_DATASET ~/data/

   Expect: by_labeler/{eporti5,hliu36,mzehra2,nmousta5}/{images,masks},
   test_set/{images,masks/<labeler>}, tables/.

3. A python env with torch (CUDA), pandas, numpy, opencv-python, albumentations,
   transformers, safetensors, scikit-learn, tqdm. No network is needed at run
   time -- SegFormer is built from the local weights folder in this archive.

RUN IT
------
    cd ~/fenwick

    # shell 1 -- GPU 0, two labelers, sequential
    CUDA_VISIBLE_DEVICES=0 python scripts/80_fenwick_cv_run.py \\
        --fenwick ~/data/FENWICK_LABELER_DATASET --labelers hliu36 nmousta5

    # shell 2 -- GPU 1, the third
    CUDA_VISIBLE_DEVICES=1 python scripts/80_fenwick_cv_run.py \\
        --fenwick ~/data/FENWICK_LABELER_DATASET --labelers mzehra2

    # ONCE, after both finish -- the matrix needs every arm present
    python scripts/80_fenwick_cv_run.py \\
        --fenwick ~/data/FENWICK_LABELER_DATASET --score

The first process to start writes the matched core; the others reuse it, so the
two GPUs cannot train against two different cuts of the data. Resumable: a fold
with DONE.json is skipped and an interrupted fold restarts from resume.pt, so
re-running either shell is always safe.

The arms are volume-matched, so 2+1 leaves GPU 0 with twice the work. To finish
sooner, give each GPU all three labelers and split the folds instead:
`--folds 0 1 2` on one, `--folds 3 4` on the other.

Prefer the notebook for the setup and the read-out: it prints the matching
report and runs three preflight assertions before anything reaches a GPU.

WHERE OUTPUT GOES
-----------------
    FENWICK_CV_RESULTS/          (beside the dataset, or wherever --out says)
      core_design.json           the exact cut of the data, reproducible
      cv_core.csv                360 imgs x 3 arms, with fold and subject
      matching_report.csv        the audit trail -- read this first
      _work/cache640/            the 640 px cache (~1 GB, derived, deletable)
      runs/<labeler>__fold<k>/   best.pt, resume.pt, DONE.json, operating_point.json
      tables/                    per-image CSVs, cross_dice.csv, labeler_table.csv,
                                 contrasts.json

THE DESIGN, IN ONE PARAGRAPH
-----------------------------
The raw pools are 1356 / 2268 / 611 images, so a raw comparison measures data
volume. Instead: drop every TEST SUBJECT from every pool (12-13 of the 15 test
subjects were in all three -- the on-disk check only looks at images), keep the
48 subjects all three labelled, and per subject give every arm n_s = the minimum
any of them drew. That is 360 images per arm with an identical per-subject
histogram. What still differs between the arms is who drew the mask.

Then 5 subject-grouped folds (72 images each, the SAME subjects for every arm,
so fold-to-fold comparisons are paired), single seed, segformer_b0 on the NIJ
recipe through engine.train_run unmodified.

HOW TO READ IT
--------------
The result is the 3x3 matrix, not the diagonal. Rows are who the model trained
on, columns are whose masks it was scored against, and the cut applied to all
three columns was fitted on that fold's own validation split -- a model is never
tuned against the annotator judging it.

  high diagonal, low off-diagonal  -> that model learned its annotator's style
  high COLUMN                      -> everyone's model agrees with that labeler,
                                      the strongest sign they drew the injury

`cross_dice` (mean against the OTHER two) is what the ranking sorts on. Read the
complete-miss rate beside the Dice, never after it.

WHAT IT CANNOT CLAIM
---------------------
- Nothing comparable to the 0.7663 NIJ headline. 288 training images against 697.
- Not annotation ACCURACY. This measures learnability and cross-annotator
  agreement; a consistently, learnably wrong labeler would rank first. No
  clinical ground truth exists in this dataset.
- Nothing about eporti5, who has no test masks.
- One seed. Treat an arm far below the others as possibly a bad run until a
  second seed says otherwise -- Stage Y seed 2 collapsed on far more data.

The 0.01 Dice margin applies. "No labeler separates" is a real outcome and is
reported as one, not as a win for whoever came top.
"""


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def main() -> int:
    missing = [s for s, _, _ in SINGLES if not (ROOT / s).exists()]
    if missing:
        print(f"FAIL: missing {missing}", file=sys.stderr)
        return 1

    b0 = BUNDLE / "pretrained_weights" / "segformer_mit_b0"
    absent = [f for f in B0_FILES if not (b0 / f).exists()]
    if absent:
        print(f"FAIL: {b0} is missing {absent}. SegFormer cannot be built offline "
              f"without it and the server has no network fallback.", file=sys.stderr)
        return 1

    # A notebook with saved outputs can carry paths, partial results, or a key.
    nb = json.loads((BUNDLE / "bruise_fenwick_cv.ipynb").read_text(encoding="utf-8"))
    if any(c.get("outputs") for c in nb["cells"]):
        print("FAIL: the notebook has executed outputs. Regenerate it with "
              "scripts/81_generate_fenwick_cv_notebook.py.", file=sys.stderr)
        return 1

    # The module is an experiment and must not have been graduated into the build:
    # then this archive would ship a second copy with no rule about which wins.
    build = ROOT / "scripts" / "60_build_unified_bundle.py"
    if build.exists() and "fenwick_cv" in build.read_text(encoding="utf-8"):
        print("FAIL: 60_build_unified_bundle.py now copies fenwick_cv.py, so this "
              "overlay is obsolete.", file=sys.stderr)
        return 1

    manifest = {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "members": []}
    DST.unlink(missing_ok=True)
    n_pkg = 0
    with zipfile.ZipFile(DST, "w", zipfile.ZIP_DEFLATED) as z:
        # The package, source only -- no __pycache__, no stray notebooks.
        for p in sorted((BUNDLE / "bruisekit").rglob("*.py")):
            if "__pycache__" in p.parts:
                continue
            z.write(p, str(Path("BRUISE_UNIFIED") / p.relative_to(BUNDLE)).replace("\\", "/"))
            n_pkg += 1
        print(f"  + BRUISE_UNIFIED/bruisekit/                     {n_pkg} modules")

        for f in B0_FILES:
            p = b0 / f
            z.write(p, f"BRUISE_UNIFIED/pretrained_weights/segformer_mit_b0/{f}")
            print(f"  + segformer_mit_b0/{f:<28} {p.stat().st_size / 1e6:6.1f} MB")

        for src, arc, note in SINGLES:
            p = ROOT / src
            z.write(p, arc)
            manifest["members"].append(
                {"arcname": arc, "bytes": p.stat().st_size, "sha256_16": sha(p),
                 "note": note})
            print(f"  + {arc:<44} {p.stat().st_size / 1024:7.1f} KB  {sha(p)}")

        manifest["bruisekit_modules"] = n_pkg
        manifest["excluded"] = {
            "FENWICK_LABELER_DATASET": "1.4 GB of patient imagery -- move ORC -> server directly",
            "tf_model.h5": "TensorFlow weights, unused by this PyTorch pipeline",
        }
        z.writestr("README.txt", README)
        z.writestr("MANIFEST.json", json.dumps(manifest, indent=2))

    print(f"\nwrote {DST}  ({DST.stat().st_size / 1e6:.1f} MB)")
    print("upload this one file; pull the dataset separately (see README.txt)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
