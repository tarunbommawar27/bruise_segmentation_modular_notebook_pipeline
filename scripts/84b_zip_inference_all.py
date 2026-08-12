#!/usr/bin/env python
"""Zip the all-models inference notebook. One notebook, no module.

WHAT IT ADDS
-------------
    bruise_inference_all.ipynb   NEW FILE  run top to bottom on a GPU box

THAT IS THE WHOLE ARCHIVE, and the absence of a module is the point.

The notebook calls `bruisekit.inference.run`, which is already in the bundle --
it is a 60_build copy-list module and the same function `bruise_unified.ipynb`'s
inference block uses. Nothing here re-implements scoring, cut resolution, seed
selection or reconciliation.

    inference.inference_pass -> loaders.score_run -> score_segformer
                                                  -> score_yolo_native
    inference.reconcile      -> fresh vs the shipped per-image table

A Stage-R module with its own loader and its own cut resolution was written and
DELETED during this work. `inference.inference_pass`'s docstring is explicit --
"There is deliberately no second inference implementation in this file" -- and a
re-inference whose only job is to confirm the published numbers must not be
computed by a different path than produced them. If a future change needs
behaviour this notebook cannot express, change `inference.py`, do not fork it.

WHY THE NOTEBOOK EXISTS AT ALL, GIVEN THE CODE ALREADY DID
------------------------------------------------------------
`inference.DEFAULT_MODELS` is three families. This notebook passes twenty-five.
That is the entire delta.

The reason to want it: on 2026-08-12 the laptop and ORC reported different miss
counts for `fastscnn` (13 vs 7), `fastscnn_distilled` (8 vs 13) and
`ppmobileseg_tiny` (4 vs 7) -- the same model names over different recorded runs,
with nothing in either table saying so. Re-scoring from checkpoints and reading
`inference.reconcile`'s `max_abs_dice_delta` settles it.

NO CHECKPOINTS, NO RESULTS, NO WEIGHTS IN THE ZIP
---------------------------------------------------
The archive is one notebook. It writes to `<env.out>/inference/`.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DST = ROOT / "BRUISE_INFERENCE_ALL.zip"

MEMBERS: list[tuple[str, str, str]] = [
    ("BRUISE_UNIFIED/bruise_inference_all.ipynb", "bruise_inference_all.ipynb",
     "17 cells, 8 code. Cell 9 shows what resolved on this machine -- read it "
     "before cell 11 starts scoring."),
    ("scripts/84_generate_inference_all_notebook.py",
     "_source/scripts/84_generate_inference_all_notebook.py",
     "emits the notebook -- its source of truth."),
    ("scripts/84b_zip_inference_all.py", "_source/scripts/84b_zip_inference_all.py",
     "this script."),
]

#: Already in the bundle via 60_build. Named here so a missing one fails loudly at
#: build time rather than at cell 5 on the GPU box.
PREREQS: list[str] = [
    "BRUISE_UNIFIED/bruisekit/inference.py",
    "BRUISE_UNIFIED/bruisekit/loaders.py",
    "BRUISE_UNIFIED/bruisekit/registry.py",
    "BRUISE_UNIFIED/bruisekit/report.py",
]

#: Cell 15 calls it for the miss taxonomy. Overlay-shipped, so it can legitimately
#: be absent -- the notebook is still useful without that one cell.
OPTIONAL: list[tuple[str, str]] = [
    ("BRUISE_UNIFIED/bruisekit/itakd.py", "BRUISE_STAGE_O.zip"),
]

README = """\
BRUISE -- INFERENCE OVER EVERY MODEL
=====================================

ONE NOTEBOOK. NO MODULE. That is deliberate.

It calls `bruisekit.inference.run`, which is already in your bundle and is the
same function the unified notebook's inference block uses:

    inference.inference_pass -> loaders.score_run -> score_segformer
                                                  -> score_yolo_native
    inference.reconcile      -> fresh vs the shipped per-image table

`inference.DEFAULT_MODELS` is three families; this notebook passes twenty-five.
That is the entire difference. There is no second inference implementation, no
second cut resolution and no second seed rule, because a re-inference whose job is
to confirm the published numbers must not be computed differently from them.

WHY YOU WANT IT
---------------
On 2026-08-12 the laptop and ORC reported different miss counts for the same three
arms:

    model                laptop        ORC
    fastscnn             13 / 8 / 5    7 / 4 / 3      (zero Dice / empty / wrong place)
    fastscnn_distilled    8 / 6 / 2   13 / 8 / 5
    ppmobileseg_tiny      4 / 3 / 1    7 / 6 / 1

Nothing was corrupt. The two machines held DIFFERENT RUNS under the same names and
nothing in either table said so. Re-scoring from checkpoints and reading
`inference.reconcile` settles which table belongs to which run.

READ max_abs_dice_delta, NOT mean_dice_delta. A mean can agree while individual
images disagree wildly, and that is the signature of a SEED MISMATCH rather than
of float noise. The handbook's claim is that every reporting model agrees to
better than 2e-4 mean Dice against the original A100 run; cell 13 is how that gets
checked rather than repeated.

HOW TO APPLY
------------
1. Unzip into the bundle root:

       unzip -o BRUISE_INFERENCE_ALL.zip -d /path/to/BRUISE_UNIFIED

   It adds one notebook. It overwrites nothing.

2. RUN IT ON THE GPU BOX. Seconds per model on CUDA; 1-2 MINUTES PER MODEL on CPU,
   because SegFormer-B2's decode head is doing 185 full-resolution forwards with
   no accelerator. Cell 7 warns you and estimates the wall clock. CPU results are
   numerically identical -- just slow. If you must use CPU, cut MODELS down to the
   handful you actually need to settle.

3. SET EXTRA_RUNS IN CELL 3. This is the setting that decides how much the
   notebook can see:

       EXTRA_RUNS = "/scratch/tbommawa/bruise_work/runs"

   The laptop bundle has NO checkpoints/efficient and NO checkpoints/rgkd, so the
   entire mobile family -- fastscnn, ppmobileseg, topformer, lraspp and their KD
   variants -- resolves only on ORC. That is the family that disagreed, so running
   this on the laptop answers nothing. Handbook 10.3.

   On the laptop 7 of 25 families resolve; on ORC with EXTRA_RUNS set, most of the
   rest do.

4. READ CELL 9 BEFORE CELL 11 STARTS SCORING. It prints every family that resolved
   with its run_id, seed, source_root and layout, and it prints every family that
   did NOT with the reason. A family with no WEIGHTS-tier run is skipped, never
   back-filled from a neighbouring seed.

TWO THINGS ALREADY HANDLED -- DO NOT RE-DO THEM
------------------------------------------------
THE BEST SEED IS NOT THE SAME FOR EVERY MODEL. `resolve_runs` reads it from the
selection step's own filenames: 0 for the three SegFormers and yolo_sem_distilled,
2 for yolo_sem_direct. Scoring a model at another model's best seed shows
per-image disagreements up to 0.49 Dice and looks exactly like a broken inference
path. Leave SEED = None.

EACH MODEL IS SCORED AT ITS OWN VAL-FITTED CUT, read from operating_point.json or
threshold.json by registry.read_cut, which is also the single place that converts
between the logit and probability dialects. Nothing here re-fits an operating
point. YOLO is scored by NATIVE ARGMAX -- no threshold exists and none is missing,
which is why it is the sole reporting path.

WHERE RESULTS GO
----------------
    <env.out>/inference/
      per_image_<family>.csv            one row per test image, per model
      inference_headline.csv            report.headline over the fresh tables
      inference_reconciliation.csv      fresh vs shipped -- THE OUTPUT THAT MATTERS
      miss_taxonomy_reinferred.csv      zero Dice split into empty / wrong place

Nothing is written to results/, FINAL_RESULT/, STAGE_O_RESULTS/ or any run
directory. RESULT_AUGUST_08/ is the reported lineage and this writes nowhere near
it -- the output is a SECOND OPINION, produced for comparison.

WHAT IT DOES NOT LICENSE
-------------------------
- A disagreement is NOT automatically a bad published number. Check in order:
  (1) did the seed resolve as expected, (2) is the shipped table from a different
  run of the same name (compare source_root), (3) only then suspect the
  checkpoint. The 2026-08-12 case was (2), and both tables were internally correct.
- NOT a re-selection. Nothing re-picks best seeds, re-fits operating points, or
  fits anything on test.
- NOT completeness. Cell 9 tells you what resolved HERE. The Stage C distill_out
  arms (p3_adaptive, p2_cwd_b5_to_b0, ...) are registered as `arm::<name>` with no
  seed, so resolve_runs' `family__seedN` lookup does not reach them; they are not
  in MODELS. A full 35-model reconciliation also needs every checkpoint AND the
  published lineage on one machine -- a sync job, not a code change.
- The speed table is OFF by default (DO_SPEED = False). It is a separate,
  machine-specific claim and the published one is fp32/no-autocast; mixing a new
  row into it without matching precision and machine is handbook 18.5's trap.
"""


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def main() -> int:
    for rel in PREREQS:
        if not (ROOT / rel).exists():
            print(f"FAIL: {rel} is absent from the bundle. The notebook calls it "
                  f"and it is a 60_build copy-list module -- rebuild the bundle.",
                  file=sys.stderr)
            return 1

    for rel, overlay in OPTIONAL:
        if not (ROOT / rel).exists():
            print(f"  note: {rel} absent ({overlay}); the notebook's miss-taxonomy "
                  f"cell will fail, the rest is unaffected.")

    missing = [src for src, _, _ in MEMBERS if not (ROOT / src).exists()]
    if missing:
        print(f"FAIL: missing {missing}", file=sys.stderr)
        return 1

    nb_path = ROOT / "BRUISE_UNIFIED" / "bruise_inference_all.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    if any(c.get("outputs") for c in nb["cells"]):
        print("FAIL: the notebook has executed outputs. Regenerate it with "
              "scripts/84_generate_inference_all_notebook.py.", file=sys.stderr)
        return 1

    # The notebook's whole premise is that it adds no inference code. Check it.
    src = "\n".join("".join(c["source"]) for c in nb["cells"]
                    if c["cell_type"] == "code")
    for banned in ("def score_", "evaluate_at_cut(", "load_state_dict("):
        if banned in src:
            print(f"FAIL: the notebook contains {banned!r}. It is supposed to call "
                  f"bruisekit.inference, not re-implement scoring. Move the change "
                  f"into inference.py.", file=sys.stderr)
            return 1
    if "INF.run(" not in src:
        print("FAIL: the notebook never calls inference.run(). That is the only "
              "thing it is for.", file=sys.stderr)
        return 1
    print("  [ok] notebook calls inference.run() and adds no scoring code")

    manifest = {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "members": []}
    DST.unlink(missing_ok=True)
    with zipfile.ZipFile(DST, "w", zipfile.ZIP_DEFLATED) as z:
        for src_rel, arc, note in MEMBERS:
            p = ROOT / src_rel
            z.write(p, arc)
            manifest["members"].append(
                {"arcname": arc, "bytes": p.stat().st_size, "sha256_16": sha(p),
                 "note": note})
            print(f"  + {arc:<52} {p.stat().st_size / 1024:7.1f} KB  {sha(p)}")
        z.writestr("README.txt", README)
        z.writestr("MANIFEST.json", json.dumps(manifest, indent=2))

    print(f"\nwrote {DST}  ({DST.stat().st_size / 1024:.1f} KB)")
    print("upload this one file")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
