#!/usr/bin/env python
"""Zip Stage N -- one notebook, one module, one upload.

WHAT IT ADDS, AND WHAT IT DOES NOT TOUCH
------------------------------------------
    bruisekit/foundation.py        NEW FILE  encoders, the gate, the shim
    bruise_foundation.ipynb        NEW FILE  run top to bottom

No shipped module, no existing notebook and no result file is overwritten, so
applying this cannot change a published number. At runtime the notebook writes
only to `FOUNDATION_RESULTS/` -- never to `results/`, `FINAL_RESULT/` or
`_work/runs/` -- so an experiment that fails leaves no trace in the directories
the study's numbers come from.

WHY foundation.py IS NOT A BUILD OUTPUT
-----------------------------------------
Every other module in `bruisekit/` is copied from `scripts/unified_lib/` by
`60_build_unified_bundle.py`. This one is deliberately absent from that list, for
the same reason `multiteacher.py` is: Stage N is an experiment that may return
nothing, and an untested architecture wrapper plus a third `resolve_micro_batch`
patch do not belong in the file that produces Stages A through Y.
`copy_authored_modules` copies a fixed list and does not clear the directory, so
a rebuild neither regenerates nor deletes it. Graduating it is two lines. This
script FAILS if that has already happened, because then the overlay ships a stale
duplicate.

THE WEIGHTS ARE NOT IN THE ZIP, AND CANNOT BE
-----------------------------------------------
MedSigLIP is licence-gated: a human has to accept the Health AI Developer
Foundations terms before the files can be fetched, and redistributing them inside
an archive would route around that. DINOv2 (Apache-2.0) is ~350 MB and ResNet-50
comes from torchvision's cache. So the archive carries the *instructions* --
`foundation.download_instructions()` -- and the notebook stops with them printed
if an encoder is absent, rather than silently training from random init.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DST = ROOT / "BRUISE_FOUNDATION.zip"

MEMBERS: list[tuple[str, str, str]] = [
    ("BRUISE_UNIFIED/bruise_foundation.ipynb", "bruise_foundation.ipynb",
     "34 cells. §0-§7 is the gate (~1 GPU-hour), §9 guards, §10+ trains."),
    ("BRUISE_UNIFIED/bruisekit/foundation.py", "bruisekit/foundation.py",
     "FoundationSegNet / ResNet50ProbeNet, probe_gate, the shim, self_test."),
    ("BRUISE_UNIFIED/PROJECT_HANDBOOK.md", "PROJECT_HANDBOOK.md",
     "NEW §7f (Stage N); §18.5 queue re-dated with Stage M's null recorded and "
     "§18.2 marked do-not-schedule; §16/§17 updated."),
    ("scripts/70_generate_foundation_notebook.py",
     "_source/scripts/70_generate_foundation_notebook.py",
     "emits the notebook -- its source of truth."),
    ("scripts/70b_zip_foundation.py", "_source/scripts/70b_zip_foundation.py",
     "this script."),
]

README = """\
# Stage N — do medical foundation encoders beat ImageNet encoders?

    cd /scratch/tbommawa/BRUISE_UNIFIED
    unzip -o BRUISE_FOUNDATION.zip

No `-d`. **Restart the kernel afterwards.** Then open `bruise_foundation.ipynb`.

Adds two new files. Overwrites no module, no notebook and no result, so
`bruise_unified.ipynb` and `bruise_stage_m.ipynb` behave exactly as before.

---

## STEP 1 — download the encoders FIRST (they are not in the zip)

Do this on your laptop or an **ORC login node** — compute nodes are usually
offline, and the notebook runs with `HF_HUB_OFFLINE=1` on purpose.

Everything goes under `pretrained_weights/foundation/`.

### 1. MedSigLIP — **gated, needs a licence acceptance**

```bash
# a) open this in a browser, signed in, and accept the terms.
#    Without this, every download below returns 401.
#    https://huggingface.co/google/medsiglip-448

# b) then:
huggingface-cli login          # paste a token with 'read' scope
huggingface-cli download google/medsiglip-448 \\
    --local-dir pretrained_weights/foundation/medsiglip-448
```
~1.6 GB (vision + text towers; only the **vision** tower is ever loaded — this
stage uses no text prompt anywhere).

### 2. DINOv2 — open, no login, **the most important arm**

```bash
huggingface-cli download facebook/dinov2-base \\
    --local-dir pretrained_weights/foundation/dinov2-base
```
~350 MB. This is the attribution control: a modern self-supervised ViT with
**zero** medical data. If it matches MedSigLIP, the win is "modern ViT", not
"dermatology", and the paper claim changes. Do not skip it.

### 3. ResNet-50 — torchvision

```bash
python -c "import torchvision; torchvision.models.resnet50(weights='IMAGENET1K_V2')"
```
~100 MB into `~/.cache/torch/hub/checkpoints/`. Run it on the login node so the
compute node finds it cached.

### 4. *optional* — BiomedParse, for the zero-shot cell only

```bash
huggingface-cli download microsoft/BiomedParse \\
    --local-dir pretrained_weights/foundation/BiomedParse
```
Skip unless you want §2b. The notebook detects its absence and continues.

Then `rsync` the whole `pretrained_weights/foundation/` directory to the bundle
on ORC. §2a checks all of this and **stops with the exact commands** if anything
is missing — it never falls back to random init.

---

## STEP 2 — run to §7 and stop

| § | what | cost |
|---|---|---|
| §0–§4 | setup, weights check, cache, recipe | minutes |
| §5–§6 | train 3 **frozen** probes, score on val | **~1 GPU-hour** |
| **§7** | **THE GATE** — printed verdict | seconds |
| §10–§13 | the seg arms, 3 seeds, test once, contrasts | ~20 GPU-hours |

**If the gate closes, stop.** That is a publishable result, not a failure:

> *A 400 M encoder pretrained on medical image-text pairs including dermatology
> does not produce better frozen features for bruise segmentation than a 26 M
> ImageNet ResNet-50.*

§9 enforces this. `RUN_SEG = True` on its own authorises nothing.

---

## The gate, in one line

Fixed in `foundation.py` before any number was produced:

```
open  iff  the (medsiglip_probe − resnet50_probe) val-Dice CI clears zero
```

Two things are reported **alongside** and deliberately **not** ANDed in:

- **Attribution** — `medsiglip − dinov2`. The gate can open and the *claim* still
  has to change.
- **Misses** — the endpoint this study is judged on.

---

## Why this is a baseline stage and not a distillation stage

The plan this was cut down from chained a foundation teacher into
reliability-gated + multi-teacher + ITA-fairness-aware KD. Three objections, all
from this project's own results:

1. **No teacher deficit exists.** `segformer_b2_teacher`, `segformer_b5_teacher`
   and `segformer_b0_distilled` each have **0** complete misses on the 185 test
   images. You cannot improve on zero.
2. **The bottleneck is transfer, not teachers.** Stage M measured +0.048 of
   oracle gain and the student captured none of it — its fused target scored
   0.723, *below every individual teacher*. A bigger teacher does not fix a
   fusion that degrades its inputs.
3. **The fairness target is not there.** 20 of 21 ITA tests in this study are
   non-significant; the one that is has its worst group at **Tan (IV)**, and
   YOLO's is at **Light (II-III)** (a lesion-size confound, §D6). Stage C already
   ran ITA-weighted KD — `p3_adaptive_group`, 0.7586 against a 0.7680 reference,
   one of the worst arms in the grid.

If the baseline shows something, **one** KD arm against a teacher-matched control
is the next decision. It is a separate one.

---

## What to expect, honestly

| model | params | mean Dice |
|---|---|---|
| segformer_b0_direct | 3.71 M | 0.7663 |
| segformer_b2_teacher | 27.35 M | 0.7692 |
| segformer_b5_teacher | ~85 M | 0.7727 |

**23× the parameters bought +0.006 Dice**, and human annotators disagree with
each other by more than that. A 400 M encoder landing near 0.775 is the *expected*
outcome and tells you nothing new about Dice — which is why §13 leads with
**complete misses and IQR**, the two metrics that actually separated the field in
Stage A.

The interesting outcomes are therefore: the gate **closing** (cheap, clean null),
or **misses** moving, or DINOv2 **matching** MedSigLIP (which reframes the whole
foundation-model question for this task).

---

## Two limitations to write down before you start

1. **Resolution.** The ViT arms see a **448**-pixel image where SegFormer sees
   640, because that is MedSigLIP's native resolution and interpolating position
   embeddings would perturb the very features being probed. It costs small-bruise
   detail — the exact population the complete-miss metric is about.
2. **Batch.** A 400 M ViT with gradients does not fit past micro-batch 4 on a
   40 GB MIG slice. These arms run `4 × 4` accumulation, so the **effective**
   batch (16), step count and LR schedule match the SMP and mobile baselines. The
   decoder is **GroupNorm**, which is per-sample — so unlike Stage M's BatchNorm
   note, this costs nothing statistically. State it anyway.

---

## Licence, and why it is in the results table

MedSigLIP ships under **Health AI Developer Foundations** terms — a custom
use-restricted licence, not OSS. DINOv2 is **Apache-2.0**. Handbook §7b.1 picked
DeepLabV3+ over SegFormer as the Stage F teacher specifically to escape NVIDIA's
non-commercial MiT licence; re-introducing a restricted encoder without recording
it would quietly undo that. **If MedSigLIP and DINOv2 tie, the licence decides.**
"""


def main() -> int:
    missing = [s for s, _, _ in MEMBERS if not (ROOT / s).exists()]
    if missing:
        raise SystemExit("not built -- run 70_generate_foundation_notebook.py first. "
                         "Missing:\n  " + "\n  ".join(missing))

    build = (ROOT / "scripts" / "60_build_unified_bundle.py").read_text(encoding="utf-8")
    if "foundation.py" in build:
        raise SystemExit(
            "60_build_unified_bundle.py now copies foundation.py, so it has a "
            "source of truth in scripts/unified_lib/ and this overlay is stale.")
    if (ROOT / "scripts" / "unified_lib" / "foundation.py").exists():
        raise SystemExit(
            "scripts/unified_lib/foundation.py exists -- two copies, no rule "
            "about which wins. Pick one before shipping.")

    # The interface and the freeze, asserted at BUILD time as well as at run time.
    # An archive shipping a mislabelled arm is an archive whose every downstream
    # number is confounded by the thing it claims to be measuring.
    r = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, r'%s'); "
         "from bruisekit import foundation as FN; "
         "raise SystemExit(0 if FN.self_test(verbose=False) else 1)"
         % str(ROOT / "BRUISE_UNIFIED")],
        capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"foundation.self_test failed -- refusing to ship.\n"
                         f"{r.stdout}\n{r.stderr}")
    print("foundation.self_test passed (interface + freeze semantics hold)")

    nb = json.loads((ROOT / "BRUISE_UNIFIED" / "bruise_foundation.ipynb")
                    .read_text(encoding="utf-8"))
    n_code = 0
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            compile("".join(c["source"]), f"cell{i}", "exec")
            n_code += 1
    print(f"notebook parses: {len(nb['cells'])} cells, {n_code} code cells compile")

    t0 = time.time()
    total = sum((ROOT / s).stat().st_size for s, _, _ in MEMBERS)
    print(f"\nzipping {len(MEMBERS)} files, {total / 1e6:.2f} MB -> {DST.name}")
    print("archive paths are relative to BRUISE_UNIFIED/ -- extract INSIDE the "
          "bundle, no -d\n")

    with zipfile.ZipFile(DST, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.writestr("README_FOUNDATION.md", README)
        for src, arc, why in MEMBERS:
            zf.write(ROOT / src, arc)
            print(f"  {arc:<48} {why}")

    with zipfile.ZipFile(DST) as zf:
        bad = zf.testzip()
        names = zf.namelist()
    if bad:
        raise SystemExit(f"archive is corrupt at {bad}")
    for required in ("bruisekit/foundation.py", "bruise_foundation.ipynb"):
        if required not in names:
            raise SystemExit(f"{required} did not make it into the archive")

    print(f"\nDONE  {DST}\n      {DST.stat().st_size / 1e6:.2f} MB, {len(names)} "
          f"entries, integrity check passed ({time.time() - t0:.1f}s)")
    print("\n  1. download the encoders (README_FOUNDATION.md, STEP 1)")
    print("  2. cd /scratch/tbommawa/BRUISE_UNIFIED && unzip -o BRUISE_FOUNDATION.zip")
    print("  3. restart the kernel, open bruise_foundation.ipynb, run to §7 and STOP")
    return 0


if __name__ == "__main__":
    sys.exit(main())
