#!/usr/bin/env python
"""Zip Stage N2 -- one notebook, one module, one upload.

WHAT IT ADDS, AND WHAT IT DOES NOT TOUCH
------------------------------------------
    bruisekit/dermprobe.py         NEW FILE  arms, grid calibration, the gate
    bruise_derm_probe.ipynb        NEW FILE  run top to bottom
    PROJECT_HANDBOOK.md            UPDATED   new 7h; 7f.9 and 18.5 cross-referenced

No shipped module, no existing notebook and no result file is overwritten, so
applying this cannot change a published number. `bruisekit/foundation.py` is
IMPORTED by `dermprobe.py` and is deliberately NOT in this archive: Stage N ships
it and shipping a second copy is how two files with no rule about which wins get
created. The zip refuses to build if it is absent from the bundle.

At runtime the notebook writes only to `DERM_PROBE_RESULTS/` -- never to
`results/`, `FINAL_RESULT/`, `FOUNDATION_RESULTS/` or `_work/runs/` -- so Stage
N's numbers survive next door untouched and an experiment that fails leaves no
trace in the directories the study's numbers come from.

WHY dermprobe.py IS NOT A BUILD OUTPUT
----------------------------------------
Every other module in `bruisekit/` is copied from `scripts/unified_lib/` by
`60_build_unified_bundle.py`. This one is deliberately absent from that list, for
the same reason `multiteacher.py` and `foundation.py` are (handbook 7e, 7f): it
is an experiment that may return nothing, and a fourth `resolve_micro_batch`
patch does not belong in the file that produces Stages A through Y.
`copy_authored_modules` copies a fixed list and does not clear the directory, so
a rebuild neither regenerates nor deletes it. Graduating it is two lines. This
script FAILS if that has already happened, because then the overlay ships a stale
duplicate.

THE WEIGHTS ARE NOT IN THE ZIP, AND TWO OF THEM CANNOT BE
-----------------------------------------------------------
MedSigLIP is licence-gated and redistributing it would route around the
acceptance. Both dermatology encoders are non-commercial (CC-BY-NC-ND-4.0 and
CC-BY-NC-4.0), which is exactly the kind of term that should not be laundered
through a private archive. So the archive carries the INSTRUCTIONS --
`dermprobe.download_instructions()` -- and the notebook stops with them printed
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
DST = ROOT / "BRUISE_DERM_PROBE.zip"

MEMBERS: list[tuple[str, str, str]] = [
    ("BRUISE_UNIFIED/bruise_derm_probe.ipynb", "bruise_derm_probe.ipynb",
     "36 cells. 5-7 is the grid control (~40 min), 9-11 the corpus gate (~1.5 h)."),
    ("BRUISE_UNIFIED/bruisekit/dermprobe.py", "bruisekit/dermprobe.py",
     "OpenClipProbe / HFViTProbe / ResNetStageProbe / PixelProbe, "
     "grid_calibration, corpus_gate, the shim, self_test."),
    ("BRUISE_UNIFIED/PROJECT_HANDBOOK.md", "PROJECT_HANDBOOK.md",
     "NEW 7h (Stage N2); 7f.9 marked superseded-and-generalised; 18.5 item 4 "
     "re-pointed; TOC updated."),
    ("scripts/77_generate_derm_probe_notebook.py",
     "_source/scripts/77_generate_derm_probe_notebook.py",
     "emits the notebook -- its source of truth."),
    ("scripts/77b_zip_derm_probe.py", "_source/scripts/77b_zip_derm_probe.py",
     "this script."),
]

README = """\
# Stage N2 — close the grid confound, then test *dermatology* pretraining properly

    cd /scratch/tbommawa/BRUISE_UNIFIED
    unzip -o BRUISE_DERM_PROBE.zip

No `-d`. **Restart the kernel afterwards.** Then open `bruise_derm_probe.ipynb`.

Adds two new files and updates the handbook. Overwrites no module, no other
notebook and no result, so `bruise_unified.ipynb`, `bruise_foundation.ipynb` and
`bruise_stage_m.ipynb` behave exactly as before. **Requires Stage N's overlay to
have been applied** — `dermprobe.py` imports `bruisekit/foundation.py`.

---

## Why this exists

Handbook §7f.8 reported `medsiglip − dinov2 = −0.166` and read it as *"medical
pretraining does not help"*. Two separate problems with quoting that:

**1. The grid confound (§7f.9), still open.** The three arms scored in exactly
the order of their feature-grid size:

| arm | grid | mean Dice |
|---|---|---|
| dinov2 | 37 × 37 | 0.6567 |
| medsiglip | 28 × 28 | 0.4907 |
| resnet50 | 20 × 20 | 0.1225 |

A linear probe is a 1×1 convolution *on that grid*, so a finer grid is worth Dice
independently of what the encoder knows. §7f.9 says it plainly: **"Do not write
§7f.8 up until it has run."**

**2. MedSigLIP is not a dermatology model.** It is a general-medical encoder —
radiology, histopathology, ophthalmology, dermatology as one slice. Our images
are consumer-camera photographs of skin. The claim §7f.8 states was never tested.

---

## STEP 1 — install one package and download three encoders

On your laptop or an **ORC login node** — compute nodes are usually offline and
the notebook runs with `HF_HUB_OFFLINE=1` on purpose.

```bash
pip install open_clip_torch          # three of the six corpus arms need it
```

Everything else goes under `pretrained_weights/foundation/` — **the same
directory Stage N uses**, so if Stage N has run then steps 4–5 are no-ops.

### 1. DermLIP — **the primary arm**, open, no login

```bash
hf download redlessone/DermLIP_ViT-B-16 \\
    --local-dir pretrained_weights/foundation/DermLIP_ViT-B-16
```
~600 MB. ViT-B/16 CLIP trained on **Derm1M**: 1.03 M dermatology image-text pairs
over 403 k unique clinical and dermoscopic skin images, 390 conditions. This is
the closest published corpus to our data. Licence **CC-BY-NC-ND-4.0 —
non-commercial**.

### 2. DermLIP-PanDerm — secondary derm arm, open, no login

```bash
hf download redlessone/DermLIP_PanDerm-base-w-PubMed-256 \\
    --local-dir pretrained_weights/foundation/DermLIP_PanDerm-base-w-PubMed-256
```
~600 MB. PanDerm-base ViT-B/16 — self-supervised on >2 M skin-disease images
across four modalities — then CLIP-aligned on Derm1M. Licence
**CC-BY-NC-ND-4.0** (the card's header says `cc-by-4.0` and its body says
`cc-by-nc-nd-4.0`; the restrictive reading is what the code records).

**If this 404s, or `open_clip` cannot build it, drop it.** Its vision tower is
PanDerm and the model card says to install the Derm1M package first, so plain
open_clip may not construct it. §9 catches that, records it in
`corpus_failed_arms.json`, and continues — this arm is *secondary*. The gate's
primary arm is DermLIP. Do **not** substitute a different checkpoint for it; the
primary/secondary split is pre-registered.

### 3. BiomedCLIP — the non-derm biomedical control, MIT

```bash
hf download microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 \\
    --local-dir pretrained_weights/foundation/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
```
~800 MB. **Architecture-matched to both derm arms** — ViT-B/16, 224 native, same
loader, same resample. So `dermlip − biomedclip` varies the *corpus* and nothing
else: skin photographs against journal figures.

### 4. DINOv2 — the generic control *(already present if Stage N ran)*

```bash
hf download facebook/dinov2-base \\
    --local-dir pretrained_weights/foundation/dinov2-base
```

### 5. MedSigLIP — gated *(already present if Stage N ran)*

Accept the terms at <https://huggingface.co/google/medsiglip-448> while signed
in, then `hf auth login` and download to
`pretrained_weights/foundation/medsiglip-448`.

### 6. ResNet-50 — torchvision

```bash
python -c "import torchvision; torchvision.models.resnet50(weights='IMAGENET1K_V2')"
```

Then `rsync` the whole `pretrained_weights/foundation/` directory to the bundle
on ORC. §2 checks `open_clip`, §2a checks the encoders, and both **stop with the
exact commands** if anything is missing. Neither ever falls back to random init.

---

## STEP 2 — run §5–§7 and read the grid curve before anything else

| § | what | cost |
|---|---|---|
| §0–§4 | setup, dependency + weights check, cache, recipe | minutes |
| §5–§6 | **Experiment 1** — 9 calibration probes | ~40 GPU-min |
| **§7** | **THE GRID CURVE** | seconds |
| §9–§10 | **Experiment 2** — 6 corpus probes, all at 28 × 28 | ~1.5 GPU-hr |
| **§11** | **THE GATE** | seconds |

Total ≈ 2–3 GPU-hours. There are no seg arms and no distillation here — this
whole notebook is a gate.

---

## Experiment 1 — how much Dice is the grid worth?

One encoder (ImageNet ResNet-50), one stage (`layer3`), **four input sizes**:

| arm | input | grid |
|---|---|---|
| `rn50_l3_g20` | 320 | 20 × 20 |
| `rn50_l3_g28` | 448 | 28 × 28 |
| `rn50_l3_g37` | 592 | 37 × 37 |
| `rn50_l3_g40` | 640 | 40 × 40 |
| `rn50_l4_g20` | 640, stride 32 | 20 × 20 — **reproduces Stage N's arm** |

Identical weights, identical depth. Whatever Dice moves **is** the grid. This
generalises §7f.9's single `layer3` control into a curve, and adds the pair that
needs no curve at all: **`rn50_l4_g20` vs `rn50_l3_g40` — same weights, same 640
input, only the stride differs.**

Plus a **pixel floor**: a 1×1 conv on raw RGB average-pooled to the same four
grids. No encoder, no pretraining. It bounds what a grid buys with zero learned
features. If the floor tracks the ResNet slope, the grid effect is *geometry*,
not features.

**What it decides.** §7f.8 reports `dinov2 − resnet50 = +0.5342` and reads all of
it as pretraining. Those two arms differed by 37 × 37 vs 20 × 20. §7 prints how
much of that gap the grid alone explains.

---

## Experiment 2 — the corpus arms, all at 28 × 28

The nuisance variable is pinned **by construction**; `corpus_gate` raises if any
arm is not at 28 × 28.

| arm | encoder | corpus | licence |
|---|---|---|---|
| `dermlip_g28` | ViT-B/16 | Derm1M — 1.03 M dermatology image-text pairs | CC-BY-NC-ND-4.0 |
| `dermlip_panderm_g28` | ViT-B/16 | PanDerm — 2 M skin images SSL, then Derm1M | CC-BY-NC-4.0 |
| `biomedclip_g28` | ViT-B/16 | PMC-15M biomedical figures | MIT |
| `dinov2_g28` | ViT-B/14 | LVD-142M natural images, SSL | Apache-2.0 |
| `medsiglip_g28` | SoViT-400m | general medical image-text | HAI-DEF, restricted |
| `rn50_g28` | ResNet-50 | ImageNet-1k supervised | BSD-3 |

Stage N's three controls are **re-run here**, not quoted from
`FOUNDATION_RESULTS/` — those numbers were taken at three different grids, and
comparing a new 28 × 28 arm against a stored 37 × 37 one would reproduce the exact
confound this stage closes.

---

## The gate, in one line

Fixed in `dermprobe.py` before any number was produced:

```
open  iff  the (dermlip_g28 − dinov2_g28) val-Dice CI clears zero
```

`dermlip_g28` is the treatment **by name**. `dermlip_panderm_g28` is secondary
and cannot be promoted afterwards — that substitution turns a two-arm experiment
into a one-arm experiment with two chances, and §15 trap 3 already refuses the
equivalent move on seeds.

Reported alongside and **not** ANDed in: `vs_medical` (the contrast Stage N
thought it was running), `vs_biomed` (architecture-matched — the cleanest in the
pool), `vs_imagenet` (ties back to Stage B), and misses.

**A closed gate is the deliverable**, and it is a stronger claim than §7f.8's:

> *Frozen features from a ViT-B/16 pretrained on 1.03 M dermatology image-text
> pairs do not outperform the same-capacity self-supervised natural-image encoder
> on bruise segmentation, at a matched 28 × 28 feature grid.*

That names a real dermatology corpus, holds architecture and grid fixed, and
cannot be explained by resolution. §7f.8's version has none of those properties.

---

## `google/derm-foundation` — why it is not here

By corpus it is the best-targeted model in existence for this task: BiT-M
ResNet-101×3 on teledermatology photographs from consumer cameras.

**It cannot be probed.** It ships as a TensorFlow/Keras SavedModel whose only
exported signature returns a single **6144-dimensional embedding per image** — no
token grid, no intermediate feature map, nothing for a 1×1 convolution to sit on.
A probe of it would predict a constant mask and its Dice would measure the
dataset's mean bruise area.

It is registered in `SOURCES` with `spatial=False` so it shows up in the
provenance table as an explicit gap — the same treatment §5 gives nnU-Net.
Notebook §2b prints the full reasoning and the one honest workaround (a sliding
448 window on an 8 × 8 lattice: ~53 k forwards of a 380 M CNN, for a grid
*coarser* than the 20 × 20 that scored 0.12 in Stage N). Experiment 1 is what
makes that disqualifying on its own. **Cite the architecture; do not run it.**

**MedImageInsight is absent for a different reason:** a DaViT on broad medical
imaging is the same question MedSigLIP already answers, through a different
vendor's code. A replication, not a new contrast.

---

## Two limitations to write down before you start

1. **Position embeddings are resampled to a common grid.** §7f.6 took the
   opposite decision — native resolution, never interpolate — and that decision
   is precisely what produced the confound, because the native resolutions
   differ. You cannot have both. This stage pins the grid because the question is
   about the corpus. Applied **identically to every ViT arm**, so it is a property
   of the experiment rather than a difference between arms.
2. **The protocol still favours DINOv2.** Frozen-feature linear probing is the
   headline evaluation in DINOv2's own paper; a supervised ResNet-50's post-ReLU
   features were never optimised to be linearly separable. A matched grid does
   not fix that and nothing here claims to.

**And the licence.** *Both* dermatology encoders are non-commercial. §7b.1 chose
DeepLabV3+ over SegFormer as the Stage F teacher specifically to escape a
non-commercial licence. **If a derm arm ties DINOv2, the licence decides**, and
`report_sources` puts it in the results table so that cannot be forgotten.

---

## Where results go

`DERM_PROBE_RESULTS/` at the bundle root:

    grid_calibration.json          the answer to 7f.9
    calibration_val_summary.csv    the 9 calibration arms
    corpus_val_summary.csv         the 6 corpus arms
    gate.json                      the decision
    derm_probe_misses.csv          the endpoint 1 says decides
    val_per_image__<arm>.csv       the tables every contrast recomputes from
    runs/<arm>__seed0/             checkpoints, operating points, val sweeps

Nothing touches `results/`, `FINAL_RESULT/`, `FOUNDATION_RESULTS/` or
`_work/runs/`.
"""


def main() -> int:
    missing = [s for s, _, _ in MEMBERS if not (ROOT / s).exists()]
    if missing:
        raise SystemExit("not built -- run 77_generate_derm_probe_notebook.py first. "
                         "Missing:\n  " + "\n  ".join(missing))

    # dermprobe.py IMPORTS foundation.py. Shipping a copy of it here would create
    # two files with no rule about which wins; requiring it instead makes the
    # dependency on Stage N's overlay explicit and checkable.
    if not (ROOT / "BRUISE_UNIFIED" / "bruisekit" / "foundation.py").exists():
        raise SystemExit(
            "bruisekit/foundation.py is absent. dermprobe.py imports it, so Stage "
            "N's overlay (BRUISE_FOUNDATION.zip) must be applied first. This "
            "archive deliberately does NOT carry a second copy.")

    build = (ROOT / "scripts" / "60_build_unified_bundle.py").read_text(encoding="utf-8")
    if "dermprobe.py" in build:
        raise SystemExit(
            "60_build_unified_bundle.py now copies dermprobe.py, so it has a "
            "source of truth in scripts/unified_lib/ and this overlay is stale.")
    if (ROOT / "scripts" / "unified_lib" / "dermprobe.py").exists():
        raise SystemExit(
            "scripts/unified_lib/dermprobe.py exists -- two copies, no rule about "
            "which wins. Pick one before shipping.")

    # The resample, the interface, THE GRID and the freeze, asserted at BUILD time
    # as well as at run time. An archive shipping an arm that runs at a different
    # grid than it declares is an archive whose only claim -- a comparison at a
    # fixed grid -- is false.
    r = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, r'%s'); "
         "from bruisekit import dermprobe as DP; "
         "raise SystemExit(0 if DP.self_test(verbose=False) else 1)"
         % str(ROOT / "BRUISE_UNIFIED")],
        capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"dermprobe.self_test failed -- refusing to ship.\n"
                         f"{r.stdout}\n{r.stderr}")
    print("dermprobe.self_test passed (resample + interface + grid + freeze hold)")

    # Every corpus arm must declare the target grid, checked here and not only
    # inside self_test, because this is the property the README makes a claim about.
    r2 = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, r'%s'); "
         "from bruisekit import dermprobe as DP; "
         "g = {a: DP.arm_grid(a) for a in DP.CORPUS_ARMS}; "
         "bad = {k: v for k, v in g.items() if v != DP.TARGET_GRID}; "
         "print(g); raise SystemExit(1 if bad else 0)"
         % str(ROOT / "BRUISE_UNIFIED")],
        capture_output=True, text=True)
    if r2.returncode != 0:
        raise SystemExit(f"corpus arms are not grid-matched:\n{r2.stdout}\n{r2.stderr}")
    print(f"corpus arms grid-matched: {r2.stdout.strip()}")

    nb = json.loads((ROOT / "BRUISE_UNIFIED" / "bruise_derm_probe.ipynb")
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
        zf.writestr("README_DERM_PROBE.md", README)
        for src, arc, why in MEMBERS:
            zf.write(ROOT / src, arc)
            print(f"  {arc:<50} {why}")

    with zipfile.ZipFile(DST) as zf:
        bad = zf.testzip()
        names = zf.namelist()
    if bad:
        raise SystemExit(f"archive is corrupt at {bad}")
    for required in ("bruisekit/dermprobe.py", "bruise_derm_probe.ipynb"):
        if required not in names:
            raise SystemExit(f"{required} did not make it into the archive")

    print(f"\nDONE  {DST}\n      {DST.stat().st_size / 1e6:.2f} MB, {len(names)} "
          f"entries, integrity check passed ({time.time() - t0:.1f}s)")
    print("\n  1. pip install open_clip_torch, then download 3 encoders "
          "(README_DERM_PROBE.md, STEP 1)")
    print("  2. cd /scratch/tbommawa/BRUISE_UNIFIED && unzip -o BRUISE_DERM_PROBE.zip")
    print("  3. restart the kernel, open bruise_derm_probe.ipynb")
    print("  4. run to §7 and READ THE GRID CURVE before running §9 onward")
    return 0


if __name__ == "__main__":
    sys.exit(main())
