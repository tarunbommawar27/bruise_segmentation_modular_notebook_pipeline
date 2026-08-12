#!/usr/bin/env python
"""Zip Stage N4 -- one notebook, one module, one upload.

WHAT IT ADDS, AND WHAT IT DOES NOT TOUCH
------------------------------------------
    bruisekit/samprobe.py       NEW FILE  SAM/MedSAM arms, the gate, fairness
    bruise_stage_n4.ipynb       NEW FILE  run top to bottom

No shipped module, no existing notebook and no result file is overwritten, so
applying this cannot change a published number.

TWO MODULES ARE IMPORTED AND DELIBERATELY NOT SHIPPED HERE
-----------------------------------------------------------
    bruisekit/dermprobe.py    _ProbeBase, the architecture-blind arm contract
    bruisekit/finetune_n3.py  ConvDecodeHead, DECODER_WIDTH, the bootstrap

Both arrive with Stage N2's and Stage N3's overlays. Shipping second copies is how
two files with no rule about which wins get created. This script FAILS if either
is absent from the bundle -- apply BRUISE_DERM_PROBE.zip and BRUISE_STAGE_N3.zip
first.

The decoder in particular MUST be the imported one. Stage N4's numbers are only
readable against Stage N3's if the head is the same object; a re-typed copy that
drifts by one channel turns every cross-stage comparison into a confound nobody
would think to check.

WHY samprobe.py IS NOT A BUILD OUTPUT
--------------------------------------
Same policy as foundation.py, dermprobe.py, finetune_n3.py, multiteacher.py and
lesionsize.py: authored in bruisekit/ but absent from 60_build_unified_bundle.py's
copy list. It is an experiment that may return nothing, and it must not become a
dependency of the file that produces Stages A through Y. This script FAILS if it
has been added to that list, because then the overlay would ship a stale duplicate.

THE WEIGHTS ARE NOT IN THE ZIP
-------------------------------
Both encoders are ~375 MB and both are open, but redistributing a checkpoint
through a private archive routes around the model card -- which is where the
licence, the citation and the training-set description live. The archive carries
the INSTRUCTIONS (`samprobe.download_instructions()`), and the notebook stops with
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
DST = ROOT / "BRUISE_STAGE_N4.zip"

MEMBERS: list[tuple[str, str, str]] = [
    ("BRUISE_UNIFIED/bruise_stage_n4.ipynb", "bruise_stage_n4.ipynb",
     "25 cells. 5 is the guard, 6 trains (~2-2.5 GPU-h per arm), 9 is the gate, "
     "11 is fairness and small-lesion recall."),
    ("BRUISE_UNIFIED/bruisekit/samprobe.py", "bruisekit/samprobe.py",
     "SamSource registry, load_sam_encoder, resample_pos_embed_2d, "
     "find_sam_blocks, SamViTProbe, train_arms, mask_supervision_gate, "
     "fairness_report, self_test."),
    ("BRUISE_UNIFIED/run_stage_n4.py", "run_stage_n4.py",
     "the same pipeline from a shell -- no Jupyter. Seven stages, each "
     "skippable and each resumable. Use this on a batch node."),
    ("scripts/79_generate_stage_n4_notebook.py",
     "_source/scripts/79_generate_stage_n4_notebook.py",
     "emits the notebook -- its source of truth."),
    ("scripts/79b_zip_stage_n4.py", "_source/scripts/79b_zip_stage_n4.py",
     "this script."),
]

#: Imported by samprobe.py, shipped by an earlier overlay, and required to be
#: present in the bundle before this archive is built.
PREREQS: list[tuple[str, str]] = [
    ("BRUISE_UNIFIED/bruisekit/dermprobe.py", "BRUISE_DERM_PROBE.zip"),
    ("BRUISE_UNIFIED/bruisekit/finetune_n3.py", "BRUISE_STAGE_N3.zip"),
    ("BRUISE_UNIFIED/bruisekit/lesionsize.py", "BRUISE_LESION_SIZE.zip"),
]

README = """\
BRUISE STAGE N4 -- mask-supervised foundation encoders (SAM / MedSAM)
=====================================================================

WHAT THIS IS
------------
Three medical encoders have now lost to DINOv2 in this project:

    frozen probe (7h.7)    dermlip - dinov2   = -0.0913
    fine-tuned   (7i.7)    dermlip - dinov2   = -0.0859
    frozen probe (7f.8)    medsiglip - dinov2 = -0.1660

But all three are CLIP/SigLIP-style: one pooled vector per image against one
sentence. A caption never says which pixels are the lesion, so that objective is
REWARDED for discarding spatial detail. DINOv2's objective is patch-level.
Handbook 7h.9 item 2 already names this as the likely mechanism behind the whole
ranking -- and if it is, then "medical pretraining does not help" is the wrong
conclusion. The right one is narrower: "medical CAPTION pretraining does not
help."

SAM and MedSAM separate those two readings. Both were pretrained with DENSE MASK
supervision. They differ from each other in exactly one thing, the corpus:

    sam_ft      SA-1B, 11 M natural images, ~1.1 B masks      Apache-2.0
    medsam_ft   ~1.5 M medical image-mask pairs, ~10 modalities

So medsam - sam holds objective, architecture, capacity and recipe fixed and
moves only the corpus. Nothing in Stages N, N2 or N3 managed that.

Two arms, one seed, ~4-5 GPU-hours total.

THIS IS NOT A TEST OF MedSAM AS A SEGMENTATION MODEL
-----------------------------------------------------
SAM and MedSAM are PROMPTABLE. This stage keeps only the image encoder and
discards the prompt encoder and mask decoder, because our pipeline is fully
automatic and has no prompt to give. "MedSAM with a ground-truth box" would score
high and answer a different question -- it hands the model the answer's location,
which is the hard half of this task.

Write it up as "MedSAM's FEATURES", never "MedSAM".

HOW TO APPLY
------------
1. Unzip into the bundle root, over the top:

       unzip -o BRUISE_STAGE_N4.zip -d /path/to/BRUISE_UNIFIED

   It adds exactly two files. It overwrites nothing you have results in.

2. PREREQUISITES, all shipped by earlier overlays:

       bruisekit/dermprobe.py     BRUISE_DERM_PROBE.zip   (_ProbeBase)
       bruisekit/finetune_n3.py   BRUISE_STAGE_N3.zip     (ConvDecodeHead)
       bruisekit/lesionsize.py    BRUISE_LESION_SIZE.zip  (size + fairness)

   samprobe.py IMPORTS all three. The decoder in particular must be the imported
   one: Stage N4's numbers are only readable against Stage N3's if the head is
   the same object.

3. Download both encoders (see below), then open bruise_stage_n4.ipynb and run
   top to bottom.

THE ENCODERS -- NOT IN THIS ZIP
--------------------------------
    mkdir -p <bundle>/pretrained_weights/sam

    hf download facebook/sam-vit-base \\
         --local-dir <bundle>/pretrained_weights/sam/sam-vit-base

    hf download wanglab/medsam-vit-base \\
         --local-dir <bundle>/pretrained_weights/sam/medsam-vit-base

~375 MB each. Cell 3 stops with these instructions if either is absent, rather
than training from random init.

VERIFY THE REPO IDS before starting a long job. They are recorded from the
published releases, not from a live check.

The HuggingFace layout is the only one this module loads (config.json with
model_type 'sam'). The original .pth releases --
dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth and
github.com/bowang-lab/MedSAM -- need conversion first.

WHERE RESULTS GO
----------------
    STAGE_N4_RESULTS/
      runs/     one directory per (arm, seed): best.pt, DONE.json, resume.pt
      tables/   per-image val and test CSVs, arm_build_info.csv,
                mask_supervision_gate.{csv,json}, fairness__*.csv

Nothing is written to results/, FINAL_RESULT/, FOUNDATION_RESULTS/,
DERM_PROBE_RESULTS/, STAGE_N3_RESULTS/ or _work/runs/. Stage N3's tables are READ
for the reference rows and cross-stage contrasts, never rewritten. The reporting
stages cannot see these arms: Registry scans env.runs by name and Stage N4 trains
into its own directory.

PRE-REGISTERED READING -- fixed in code before any number exists
----------------------------------------------------------------
    medsam - sam clears zero POSITIVE
        -> the medical CORPUS buys something once the objective is right.
           N/N2/N3 measured the objective, not the corpus, and their conclusion
           needs narrowing. A foundation teacher becomes worth costing out.
    medsam - sam CONTAINS zero
        -> NULL, and a strong one. Objective fixed, capacity matched, recipe
           identical: a 1.5 M-pair medical mask corpus buys nothing over natural
           images. Fourth axis, same answer.
    medsam - sam clears zero NEGATIVE
        -> MedSAM is WORSE, the same direction DermLIP and MedSigLIP went.
           Report it plainly.
    either arm < 0.73
        -> INCONCLUSIVE. One seed cannot separate a weak encoder from a collapsed
           run (Stage Y seed 2 did exactly this). Check the loss curve first.

READ THE MISS COLUMN BEFORE THE DICE COLUMN. Dice is saturated (Friedman p = 0.61
across the seven headline models); complete misses on small bruises are the
endpoint that has ever moved and the one a clinician cares about.
mask_supervision_gate REFUSES to print a verdict without them.

A DICE NULL WITH A MISS-RATE WIN IS A RESULT, and cell 11 is where you would see
it. That cell is not an appendix.

THE FAILURES THAT WOULD FAKE A RESULT
--------------------------------------
1. AN ARM THAT UNFREEZES NOTHING trains like a frozen probe, scores low-to-mid,
   and is indistinguishable from a real ceiling result. find_sam_blocks() RAISES
   rather than returning empty, unfreeze_last() RAISES if trainable params come
   back zero, and cell 5 asserts a real trainable fraction before any training
   starts. Check it reads about 0.48, not 0.0.

2. A POSITION EMBEDDING THAT DID NOT RESAMPLE. SAM stores a 2-D [1,64,64,C]
   embedding for a 1024 px input and ADDS it to the patch grid; at 640 that is
   40x40 and the addition is a shape error. resample_pos_embed_2d handles it and
   REFUSES a token-sequence tensor, and _features raises on a grid mismatch
   rather than reshaping into a plausible-looking garbage map. Cell 5 should
   print 64 -> 40.

3. A LOAD THAT DID NOT LAND. load_sam_encoder chooses the class by model_type
   (never AutoModel), checks missing_keys, and checks the parameter count against
   89.7 M BEFORE the resample -- which legitimately removes ~1.9 M. This is the
   2026-08-06 failure (7f.7a), where SiglipVisionModel silently loaded a DINOv2
   directory as a random ViT and invalidated an entire run plus the attribution
   number computed from it.

WHAT THIS DOES NOT LICENSE
---------------------------
- Not "MedSAM does not work." We removed its prompt.
- Not a licence claim. Both are recorded as Apache-2.0 from their releases, not
  from a licence review. Confirm on the model cards.
- Not a seed-robust result. One seed.
- Not a distribution-matched test. MedSAM's corpus is mostly radiology and
  dermoscopy; ours is clinical photography across skin tones. That is the same
  gap that put MedSigLIP LAST at 0.4670, and a loss here is partly it.
"""


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def main() -> int:
    for rel, overlay in PREREQS:
        p = ROOT / rel
        if not p.exists():
            print(f"FAIL: {p} is absent. samprobe.py imports it; apply {overlay} "
                  f"first.", file=sys.stderr)
            return 1

    # If samprobe has been graduated into the build, this overlay would ship a
    # stale duplicate with no rule about which wins.
    build = ROOT / "scripts" / "60_build_unified_bundle.py"
    if build.exists() and "samprobe" in build.read_text(encoding="utf-8"):
        print("FAIL: 60_build_unified_bundle.py now copies samprobe.py. It has "
              "been graduated into the bundle, so this overlay is obsolete -- "
              "shipping it would create a second copy.", file=sys.stderr)
        return 1

    missing = [src for src, _, _ in MEMBERS if not (ROOT / src).exists()]
    if missing:
        print(f"FAIL: missing {missing}", file=sys.stderr)
        return 1

    # A notebook with saved outputs can carry paths, partial results, or a key.
    nb = json.loads((ROOT / "BRUISE_UNIFIED" / "bruise_stage_n4.ipynb")
                    .read_text(encoding="utf-8"))
    if any(c.get("outputs") for c in nb["cells"]):
        print("FAIL: the notebook has executed outputs. Regenerate it with "
              "scripts/79_generate_stage_n4_notebook.py.", file=sys.stderr)
        return 1

    # The module must pass its own structural checks before it is shipped. This
    # needs no weights, no GPU and no network, so there is no excuse for skipping
    # it -- and it is the check that catches a find_sam_blocks that stopped
    # raising.
    sys.path.insert(0, str(ROOT / "BRUISE_UNIFIED"))
    try:
        from bruisekit import samprobe
    except Exception as exc:                                  # pragma: no cover
        print(f"FAIL: cannot import bruisekit.samprobe: {exc}", file=sys.stderr)
        return 1
    print("-- samprobe self-test --")
    if not samprobe.self_test(verbose=True):
        print("FAIL: samprobe.self_test() did not pass. Not shipping this.",
              file=sys.stderr)
        return 1
    print()

    manifest = {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "members": []}
    DST.unlink(missing_ok=True)
    with zipfile.ZipFile(DST, "w", zipfile.ZIP_DEFLATED) as z:
        for src, arc, note in MEMBERS:
            p = ROOT / src
            z.write(p, arc)
            manifest["members"].append(
                {"arcname": arc, "bytes": p.stat().st_size, "sha256_16": sha(p),
                 "note": note})
            print(f"  + {arc:<46} {p.stat().st_size / 1024:7.1f} KB  {sha(p)}")
        z.writestr("README.txt", README)
        z.writestr("MANIFEST.json", json.dumps(manifest, indent=2))

    print(f"\nwrote {DST}  ({DST.stat().st_size / 1024:.1f} KB)")
    print("upload this one file")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
