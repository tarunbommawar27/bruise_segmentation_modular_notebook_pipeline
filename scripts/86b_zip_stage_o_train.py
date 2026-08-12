#!/usr/bin/env python
"""Patch kit: train the ITA-grouped arm against a closed gate.

    python scripts/86b_zip_stage_o_train.py   ->  BRUISE_STAGE_O_TRAIN.zip

WHAT IT ADDS
-------------
    bruisekit/itakd.py            UPDATED  adds preflight() and record_override()
    bruise_stage_o_train.ipynb    NEW      the forced training run

This is a PATCH ON TOP OF BRUISE_STAGE_O.zip, not a replacement for it. It ships
the same `itakd.py` with two functions added, so applying it over an existing
Stage O install upgrades the module in place and adds one notebook. Apply Stage O
first or the analysis notebook and the result tables will be missing.

WHY itakd.py IS SHIPPED AGAIN RATHER THAN AS A DIFF
-----------------------------------------------------
A .patch against a file that may have been hand-edited on the far end fails in
the least useful way possible -- halfway. The module is 2,000 lines and 90 KB;
shipping the whole thing costs nothing and cannot half-apply. The zip records
the sha256 prefix of both copies so a receiver can tell whether theirs was
already current.

THE OVERRIDE IS THE POINT, AND IT IS RECORDED
-----------------------------------------------
`ita_group_gate` closed on both schemes. This kit exists to train the arm anyway,
which is a legitimate decision -- a pre-registration is written to be
falsifiable, and a measured null beats a projected one. What it must not do is
produce a results directory that later looks gate-approved, so
`record_override()` writes FORCED_GATE.json beside the runs with the failing
clauses verbatim, and the notebook refuses to start if no gate file exists to
override.

THE PREFLIGHT IS WHY THIS IS NOT JUST `--force-train`
-------------------------------------------------------
Stage O's three shims have never run together against engine.train_run. An
untagged loader, a teacher stack whose K disagrees with the fitted weights, or a
loss dispatcher installed in the wrong order all surface on the FIRST optimizer
step -- twenty minutes into a 13 GPU-hour job. `preflight()` runs one forward and
one backward through the real shims and raises on all three in about a minute.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DST = ROOT / "BRUISE_STAGE_O_TRAIN.zip"

MEMBERS: list[tuple[str, str, str]] = [
    ("BRUISE_UNIFIED/bruise_stage_o_train.ipynb", "bruise_stage_o_train.ipynb",
     "19 cells. 11 is the preflight -- run it and read it before flipping "
     "RUN_TRAINING in cell 3."),
    ("BRUISE_UNIFIED/bruisekit/itakd.py", "bruisekit/itakd.py",
     "the Stage O module, now with preflight() and record_override(). "
     "OVERWRITES the copy from BRUISE_STAGE_O.zip -- same file, two functions "
     "added."),
    ("scripts/86_generate_stage_o_train_notebook.py",
     "_source/scripts/86_generate_stage_o_train_notebook.py",
     "emits the notebook -- its source of truth."),
    ("scripts/86b_zip_stage_o_train.py",
     "_source/scripts/86b_zip_stage_o_train.py", "this script."),
]

#: Must already be installed. This kit patches Stage O; it does not contain it.
PREREQS: list[tuple[str, str]] = [
    ("BRUISE_UNIFIED/bruisekit/lesionsize.py", "BRUISE_LESION_SIZE.zip"),
    ("BRUISE_UNIFIED/bruisekit/multiteacher.py", "BRUISE_STAGE_M.zip"),
    ("BRUISE_UNIFIED/bruisekit/samprobe.py", "BRUISE_STAGE_N4.zip"),
    ("BRUISE_UNIFIED/bruise_stage_o.ipynb", "BRUISE_STAGE_O.zip"),
]

README = """\
BRUISE STAGE O -- TRAINING PATCH KIT
=====================================
Train the ITA-group-routed multi-teacher arm AGAINST A CLOSED GATE.

APPLY BRUISE_STAGE_O.zip FIRST. This is a patch on top of it, not a replacement.

    unzip -o BRUISE_STAGE_O.zip       -d /path/to/BRUISE_UNIFIED
    unzip -o BRUISE_STAGE_O_TRAIN.zip -d /path/to/BRUISE_UNIFIED

It adds one notebook and OVERWRITES bruisekit/itakd.py with the same file plus
two functions: preflight() and record_override(). Nothing else changes.

WHAT YOU ARE OVERRIDING
------------------------
`ita_group_gate` closed on BOTH schemes:

    weighting gain over best single teacher   -0.0056  [-0.0174, -0.0016]
    projected student gain vs +0.01 margin    -0.0015
    groups with an identifiable argmax        0 of 2
    pool contains misses best single lacks    2 vs 2

The third clause is the interesting one. Across six candidate teachers the
per-ITA-group best teacher survives resampling the 20 validation patients only
36-52 % of the time. An arm routed on that argmax is fitting sampling noise.

WHY RUNNING ANYWAY IS DEFENSIBLE
---------------------------------
A pre-registration is written to be FALSIFIABLE, not to be obeyed. The gate's job
was to stop a speculative grid; it is a projection, not evidence about what the
arm does. A measured null is a stronger paper section than a predicted one, and
the identifiability clause is new enough to deserve testing against an outcome.

WHAT WOULD NOT BE DEFENSIBLE, AND WHAT PREVENTS IT
----------------------------------------------------
A results directory that six months from now looks gate-approved.
record_override() writes STAGE_O_RESULTS/runs/FORCED_GATE.json carrying the
failing clauses verbatim, plus the reason you type into OVERRIDE_REASON in cell
3. That field is not optional -- it is what a reader has instead of your memory.

The notebook REFUSES to start if tables/ita_group_gate__<scheme>.json is absent.
A forced run with no record of what it overrode is the thing this kit exists to
prevent.

RUN THE PREFLIGHT FIRST -- CELL 11
------------------------------------
Stage O's three shims have never run together against engine.train_run:

    install_group_shim     wraps the TRAINING loader so each batch records its
                           stems' ITA group indices. engine.train_run iterates
                           (x, y, _) and throws the stem away, so this is the
                           only way the loss learns which group an image is in
                           without editing the shared training loop.
    install_teacher_shim   rebinds engine.load_teacher to return [B, K, H, W].
                           Handles MedSAM, which has no registry Run.
    install_loss_shim      LAST. In a session that also touched Stage H or M the
                           dispatcher must sit on top of theirs.

Their failure modes -- untagged loader, K disagreeing with the fitted weights,
wrong loss class from the dispatcher, non-finite loss, no gradient -- ALL surface
on the first optimizer step, which is twenty minutes into a 13 GPU-hour job.
preflight() runs one forward and one backward through the real shims and RAISES
on every one of them in about a minute.

It also checks the thing no exception would catch: that the group index actually
CHANGES the loss. If it does not, the arm is Stage C's p2_ensemble_uniform
wearing this stage's name and would report a plausible number for a different
experiment.

COST
----
Six runs: two students x three seeds, ~2.4 h and ~2.0 h per seed on an A100 MIG.
Roughly 13 GPU-hours. Resumable -- an interrupted run picks up from resume.pt, a
finished one is skipped via DONE.json.

WHERE IT WRITES
---------------
    STAGE_O_RESULTS/runs/       best.pt, config.json, group_loss_stats.json,
                                FORCED_GATE.json, preflight.json
    STAGE_O_RESULTS/trained/    test_per_image__*.csv, forced_contrasts.csv

Deliberately separate from STAGE_O_RESULTS/tables/, which holds the gate and the
analyses. A forced run cannot overwrite the analysis that advised against it.

READ group_loss_stats.json BEFORE ANY DICE NUMBER. images_per_group is the first
thing to check: an arm whose loss never saw one of the groups did not run this
experiment, and its Dice answers a different question.

HOW TO REPORT THE OUTCOME
--------------------------
IF IT IS A NULL -- the expected outcome -- you now have it MEASURED rather than
projected, which is the stronger form. Report it as the fourth consecutive KD
null here (Stage C p3_adaptive_group at 0.7586, Stage H, Stage M) and say the
gate predicted it. A pre-test that correctly forecast a null it was then checked
against is a working pre-test and deserves its own sentence.

IF THE ARM WINS, the interesting object is the GATE, not the arm. Check in order:
(1) did the loss see both groups, (2) is the win larger than the seed-to-seed
spread of +-0.004, (3) does the paired interval exclude zero. A win under +0.05
Dice is inside the annotation-ceiling noise floor regardless (handbook 1).

EITHER WAY, FORCED_GATE.json travels with the numbers, and the identifiability
table is quoted beside the result. 0 of 2 groups had an estimable argmax, and
that stays true whatever the students did.

WHAT THIS CANNOT SETTLE
------------------------
Whether a BETTER ROUTING KEY would work. The per-image oracle gain of +0.045 says
the teachers genuinely complement each other; this stage only shows that skin
tone does not identify when. A key that did -- lesion size, image quality,
teacher agreement -- is a different experiment, and Stage M already tested the
last of those.
"""


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def main() -> int:
    for rel, overlay in PREREQS:
        if not (ROOT / rel).exists():
            print(f"FAIL: {rel} is absent. This kit patches Stage O rather than "
                  f"containing it; apply {overlay} first.", file=sys.stderr)
            return 1

    missing = [s for s, _, _ in MEMBERS if not (ROOT / s).exists()]
    if missing:
        print(f"FAIL: missing {missing}", file=sys.stderr)
        return 1

    nb_path = ROOT / "BRUISE_UNIFIED" / "bruise_stage_o_train.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    if any(c.get("outputs") for c in nb["cells"]):
        print("FAIL: the notebook has executed outputs. Regenerate with "
              "scripts/86_generate_stage_o_train_notebook.py.", file=sys.stderr)
        return 1

    src = "\n".join("".join(c["source"]) for c in nb["cells"]
                    if c["cell_type"] == "code")
    # The two guards this kit is FOR. If either is missing the notebook is a
    # plain --force-train with extra steps, and the override goes unrecorded.
    for needed, why in (("itakd.record_override(",
                         "the override would go unrecorded"),
                        ("itakd.preflight(",
                         "a 13-hour job would start unverified")):
        if needed not in src:
            print(f"FAIL: the notebook never calls {needed} -- {why}.",
                  file=sys.stderr)
            return 1
    if "GATE_PATH.exists()" not in src:
        print("FAIL: the notebook does not assert that a gate file exists. A "
              "forced run with no record of what it overrode is what this kit "
              "exists to prevent.", file=sys.stderr)
        return 1
    print("  [ok] notebook records the override, preflights, and requires a gate")

    sys.path.insert(0, str(ROOT / "BRUISE_UNIFIED"))
    from bruisekit import itakd
    for fn in ("preflight", "record_override"):
        if not callable(getattr(itakd, fn, None)):
            print(f"FAIL: itakd.{fn}() is missing from the module being shipped.",
                  file=sys.stderr)
            return 1
    print("-- itakd self-test --")
    if not itakd.self_test(verbose=True):
        print("FAIL: itakd.self_test() did not pass.", file=sys.stderr)
        return 1
    print()

    manifest = {"built": time.strftime("%Y-%m-%d %H:%M:%S"),
                "patches": "BRUISE_STAGE_O.zip", "members": []}
    DST.unlink(missing_ok=True)
    with zipfile.ZipFile(DST, "w", zipfile.ZIP_DEFLATED) as z:
        for s, arc, note in MEMBERS:
            p = ROOT / s
            z.write(p, arc)
            manifest["members"].append(
                {"arcname": arc, "bytes": p.stat().st_size, "sha256_16": sha(p),
                 "note": note})
            print(f"  + {arc:<52} {p.stat().st_size / 1024:7.1f} KB  {sha(p)}")
        z.writestr("README.txt", README)
        z.writestr("MANIFEST.json", json.dumps(manifest, indent=2))

    print(f"\nwrote {DST}  ({DST.stat().st_size / 1024:.1f} KB)")
    print("apply AFTER BRUISE_STAGE_O.zip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
