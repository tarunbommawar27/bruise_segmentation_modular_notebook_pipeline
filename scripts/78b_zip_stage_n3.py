#!/usr/bin/env python
"""Zip Stage N3 -- one notebook, one module, one upload.

WHAT IT ADDS, AND WHAT IT DOES NOT TOUCH
------------------------------------------
    bruisekit/finetune_n3.py    NEW FILE  arms, partial unfreeze, the ceiling gate
    bruise_stage_n3.ipynb       NEW FILE  run top to bottom

No shipped module, no existing notebook and no result file is overwritten, so
applying this cannot change a published number.

`bruisekit/dermprobe.py` is IMPORTED by `finetune_n3.py` and is deliberately NOT
in this archive: Stage N2's overlay ships it, and shipping a second copy is how
two files with no rule about which wins get created. This script FAILS if it is
absent from the bundle -- apply BRUISE_DERM_PROBE.zip first.

At runtime the notebook writes only to `STAGE_N3_RESULTS/` -- never to
`results/`, `FINAL_RESULT/`, `FOUNDATION_RESULTS/`, `DERM_PROBE_RESULTS/` or
`_work/runs/`. An experiment that fails leaves no trace in the directories the
study's numbers come from, and the reporting stages cannot pick these arms up:
`Registry` scans `env.runs` by name and Stage N3 trains into its own directory.

WHY finetune_n3.py IS NOT A BUILD OUTPUT
-----------------------------------------
Same policy as foundation.py, dermprobe.py, multiteacher.py and lesionsize.py:
authored in bruisekit/ but absent from 60_build_unified_bundle.py's copy list. It
is an experiment that may return nothing, and it must not become a dependency of
the file that produces Stages A through Y. This script FAILS if it has been added
to that list, because then the overlay would ship a stale duplicate.

THE WEIGHTS ARE NOT IN THE ZIP, AND ONE OF THEM CANNOT BE
-----------------------------------------------------------
DermLIP is non-commercial (CC-BY-NC-*), which is exactly the kind of term that
should not be laundered through a private archive. The archive carries the
INSTRUCTIONS -- `dermprobe.download_instructions()` -- and the notebook stops with
them printed if an encoder is absent, rather than training from random init.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DST = ROOT / "BRUISE_STAGE_N3.zip"

MEMBERS: list[tuple[str, str, str]] = [
    ("BRUISE_UNIFIED/bruise_stage_n3.ipynb", "bruise_stage_n3.ipynb",
     "19 cells. 6 trains (~2-2.5 GPU-h per arm); 8 is the gate."),
    ("BRUISE_UNIFIED/bruisekit/finetune_n3.py", "bruisekit/finetune_n3.py",
     "ConvDecodeHead, find_blocks, FineTunedHFViT / FineTunedOpenClip, "
     "train_arms, ceiling_gate, the shim, self_test."),
    ("scripts/78_generate_stage_n3_notebook.py",
     "_source/scripts/78_generate_stage_n3_notebook.py",
     "emits the notebook -- its source of truth."),
    ("scripts/78b_zip_stage_n3.py", "_source/scripts/78b_zip_stage_n3.py",
     "this script."),
]

README = """\
BRUISE STAGE N3 -- the ceiling test on a third axis
====================================================

WHAT THIS IS
------------
Every existing piece of annotation-ceiling evidence varies capacity or
architecture. This stage varies the PRETRAINING PARADIGM, which Stage N2 showed
carries genuinely different information (DINOv2 0.6635 frozen-probe Dice against
ResNet-50's 0.5252).

If a demonstrably different feature space ALSO lands in 0.75-0.78 once fine-tuned,
the ceiling is binding across capacity, architecture and pretraining -- and the
bottleneck is label quality, not the model.

Two arms, one seed, ~4-5 GPU-hours total.

HOW TO APPLY
------------
1. Unzip into the bundle root, over the top:

       unzip -o BRUISE_STAGE_N3.zip -d /path/to/BRUISE_UNIFIED

   It adds exactly two files. It overwrites nothing you have results in.

2. PREREQUISITE: bruisekit/dermprobe.py must already be present (Stage N2's
   overlay, BRUISE_DERM_PROBE.zip). finetune_n3.py imports it for encoder
   loading -- including the guard against the SigLIP-loads-DINOv2-silently
   failure that invalidated an entire earlier run.

3. Open bruise_stage_n3.ipynb and run top to bottom.

WHERE RESULTS GO
----------------
    STAGE_N3_RESULTS/
      runs/     one directory per (arm, seed): best.pt, DONE.json, resume.pt
      tables/   per-image val CSVs, arm_build_info.csv, ceiling_gate.{csv,json}

Nothing is written to results/, FINAL_RESULT/, FOUNDATION_RESULTS/,
DERM_PROBE_RESULTS/ or _work/runs/. The reporting stages cannot see these arms.

PRE-REGISTERED READING -- fixed in code before any number exists
----------------------------------------------------------------
    both arms in 0.75-0.78  -> CEILING CONFIRMED on a third axis. The frozen-probe
                               ranking did not transfer. Encoders are exhausted as
                               a lever; go to label quality.
    either arm > 0.79       -> ceiling NOT binding. Stage N2's null needs
                               reinterpreting and encoders are live again.
    either arm < 0.73       -> INCONCLUSIVE. One seed cannot separate a weak
                               encoder from a collapsed run. Check the loss curve.

READ THE MISS COLUMN BEFORE THE DICE COLUMN. Dice is saturated; complete misses
are where this study's models separate. ceiling_gate refuses to print a verdict
without them.

THE ONE FAILURE THAT WOULD FAKE A RESULT
-----------------------------------------
An arm that silently unfreezes NOTHING trains like a frozen probe and scores in a
way indistinguishable from a confirmed ceiling. find_blocks() raises rather than
guessing, unfreeze_last() raises if trainable params come back zero, and cell 5
asserts a non-zero trainable fraction before any training starts. Check that
cell's output says roughly 45 percent, not 0.0.

WHAT YOU NEED THAT IS NOT IN HERE
----------------------------------
The encoder weights. DINOv2 is Apache-2.0; DermLIP is non-commercial, so neither
is redistributed here. Cell 3 stops with download instructions if either is
absent, rather than training from random init.

WHAT THIS DOES NOT LICENSE
---------------------------
A confirmed ceiling does not say encoders never matter. It says they do not
matter HERE, AT THIS LABEL NOISE LEVEL. And it is one seed: an arm inside the
band is consistent with the ceiling, not a tight interval around it.

More images labelled to the same standard do NOT raise an annotation ceiling --
the ceiling is set by label noise, not data volume. What moves it is multiple
independent annotations per image.
"""


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def main() -> int:
    # dermprobe must be present (imported) but must NOT be shipped here.
    dermprobe = ROOT / "BRUISE_UNIFIED" / "bruisekit" / "dermprobe.py"
    if not dermprobe.exists():
        print(f"FAIL: {dermprobe} is absent. finetune_n3.py imports it; apply "
              f"BRUISE_DERM_PROBE.zip first.", file=sys.stderr)
        return 1

    # If finetune_n3 has been graduated into the build, this overlay would ship a
    # stale duplicate with no rule about which wins.
    build = ROOT / "scripts" / "60_build_unified_bundle.py"
    if build.exists() and "finetune_n3" in build.read_text(encoding="utf-8"):
        print("FAIL: 60_build_unified_bundle.py now copies finetune_n3.py. It has "
              "been graduated into the bundle, so this overlay is obsolete -- "
              "shipping it would create a second copy.", file=sys.stderr)
        return 1

    missing = [src for src, _, _ in MEMBERS if not (ROOT / src).exists()]
    if missing:
        print(f"FAIL: missing {missing}", file=sys.stderr)
        return 1

    # A notebook with saved outputs can carry paths, partial results, or a key.
    nb = json.loads((ROOT / "BRUISE_UNIFIED" / "bruise_stage_n3.ipynb")
                    .read_text(encoding="utf-8"))
    if any(c.get("outputs") for c in nb["cells"]):
        print("FAIL: the notebook has executed outputs. Regenerate it with "
              "scripts/78_generate_stage_n3_notebook.py.", file=sys.stderr)
        return 1

    manifest = {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "members": []}
    DST.unlink(missing_ok=True)
    with zipfile.ZipFile(DST, "w", zipfile.ZIP_DEFLATED) as z:
        for src, arc, note in MEMBERS:
            p = ROOT / src
            z.write(p, arc)
            manifest["members"].append(
                {"arcname": arc, "bytes": p.stat().st_size, "sha256_16": sha(p),
                 "note": note})
            print(f"  + {arc:<44} {p.stat().st_size / 1024:7.1f} KB  {sha(p)}")
        z.writestr("README.txt", README)
        z.writestr("MANIFEST.json", json.dumps(manifest, indent=2))

    print(f"\nwrote {DST}  ({DST.stat().st_size / 1024:.1f} KB)")
    print("upload this one file")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
