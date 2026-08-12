#!/usr/bin/env python
"""Build a SELF-CONTAINED zip of BRUISE_UNIFIED for a private server.

    python scripts/80_build_server_package.py                 # n4 profile
    python scripts/80_build_server_package.py --profile full
    python scripts/80_build_server_package.py --dry-run       # size it first

Unlike the overlay zips (67/69b/70b/71b/77b/78b/79b), which add two files to an
existing bundle, this produces a tree that can be dropped onto a bare machine
and run. Code, data, manifests, encoder weights, a CLI runner and the commands.

PROFILES -- and why the default is not "everything"
----------------------------------------------------
    n4    (default, ~3.3 GB)  everything Stage N4 needs and nothing else
    full  (~6.5 GB)           adds checkpoints/ and the SegFormer/YOLO encoders,
                              i.e. enough to re-run the reporting stages too

"Everything" would be 11 GB, and most of that is bytes the target does not need:

    DERM_PROBE_RESULTS.zip  2.7 GB  a zip OF RESULTS. Never an input to anything.
    checkpoints/            2.1 GB  trained weights for Stages A-Y. Stage N4
                                    trains its own; it reads none of these.
    _work/cache640          774 MB  a DETERMINISTIC resize of data/. It rebuilds
                                    in about a minute, and shipping it creates a
                                    second copy of the pixels that can drift from
                                    the first (paths.py says the same).
    segformer_mit_b*        650 MB  encoder inits for models Stage N4 does not build.

WHAT IS TRIMMED FROM THE ENCODER DIRECTORIES, AND WHY IT IS SAFE
------------------------------------------------------------------
`facebook/sam-vit-base` ships the SAME weights three times: `model.safetensors`,
`pytorch_model.bin` and `tf_model.h5` -- 1.1 GB for 375 MB of parameters.
`from_pretrained` reads exactly one, preferring safetensors. So one weights file
is kept per encoder (safetensors if present, else the .bin) and the duplicates are
dropped. config.json, preprocessor_config.json and the model card travel with it,
because the card is where the licence and the training-set description live and
this project has already been bitten by a card it had not read.

This is a REDUCTION, not a conversion. Nothing is repacked, re-serialised or
renamed, so the file that lands is byte-identical to the one downloaded.

COMPRESSION
-----------
Images, safetensors and .bin are already compressed; DEFLATE on them costs
minutes and saves ~1 %. Those are STORED and everything else is DEFLATED, which
makes this roughly disk-copy speed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "BRUISE_UNIFIED"

#: Already-compressed payloads. STORED, not DEFLATED.
STORED_SUFFIXES = {".jpg", ".jpeg", ".png", ".safetensors", ".bin", ".pt", ".pth",
                   ".h5", ".zip", ".gz", ".xlsx", ".pptx", ".npz"}

#: Directories every profile carries. Code, labels, splits, and the data.
COMMON_DIRS: list[tuple[str, str]] = [
    ("bruisekit", "the library -- every module, including samprobe.py"),
    ("manifests", "train/val/test splits by stem, with subject and ITA group"),
    ("splits", "the split definitions the manifests were cut from"),
    ("ita_labels", "per-image ITA and skin-tone category"),
    ("data", "the images and masks: 831 train+val, 185 test"),
    ("STAGE_N3_RESULTS", "READ by Stage N4 for its reference rows and the "
                         "cross-stage contrasts. Never rewritten."),
]

FULL_ONLY_DIRS: list[tuple[str, str]] = [
    ("checkpoints", "trained weights for Stages A-Y"),
    ("results", "the shipped result tables"),
    ("FINAL_RESULT", "the current reporting lineage"),
    ("FENWICK_RESULTS", "the labeler cross-validation tables"),
    ("LESION_SIZE_RESULTS", "size-stratified tables"),
]

#: Loose files at the bundle root.
COMMON_FILES: list[tuple[str, str]] = [
    ("run_stage_n4.py", "THE RUNNER. Stage N4 end to end from a shell."),
    ("bruise_stage_n4.ipynb", "the same pipeline as a notebook, if you prefer it"),
    ("PROJECT_HANDBOOK.md", "the study. Sections 7f-7i are this stage's context."),
    ("interlabeler_agreement_640.csv", "human-vs-human Dice -- the ceiling"),
]

#: Encoder directories under pretrained_weights/, per profile.
N4_ENCODERS = ["sam"]
FULL_ENCODERS = ["sam", "segformer_mit_b0", "segformer_mit_b2", "segformer_mit_b5",
                 "efficient"]

#: Inside an encoder directory: keep the first weights file found, in this order,
#: and drop the rest. They are duplicates of each other.
WEIGHT_PREFERENCE = ["model.safetensors", "pytorch_model.bin", "diffusion_pytorch_model.safetensors"]
KEEP_ALWAYS = {"config.json", "preprocessor_config.json", "README.md",
               "tokenizer_config.json", "special_tokens_map.json"}


def is_weight_file(name: str) -> bool:
    return name.endswith((".safetensors", ".bin", ".h5", ".msgpack", ".ckpt"))


def select_encoder_files(d: Path) -> tuple[list[Path], list[Path]]:
    """(kept, dropped) for one HuggingFace encoder directory.

    `.cache/huggingface/` is excluded outright: it is the downloader's bookkeeping
    (lock files, blob metadata, CACHEDIR.TAG), it is machine-specific, and
    shipping it to a different host is how a stale lock makes a load look broken.
    """
    files = [p for p in sorted(d.rglob("*"))
             if p.is_file() and ".cache" not in p.parts]
    weights = [p for p in files if is_weight_file(p.name)]
    chosen = None
    for want in WEIGHT_PREFERENCE:
        for p in weights:
            if p.name == want:
                chosen = p
                break
        if chosen:
            break
    if chosen is None and weights:
        chosen = max(weights, key=lambda p: p.stat().st_size)

    kept, dropped = [], []
    for p in files:
        if is_weight_file(p.name):
            (kept if p == chosen else dropped).append(p)
        elif p.name in KEEP_ALWAYS or p.suffix in {".json", ".txt", ".md"}:
            kept.append(p)
        else:
            dropped.append(p)
    return kept, dropped


def gather(profile: str) -> tuple[list[tuple[Path, str]], list[Path]]:
    """(members, dropped). `members` is (source path, arcname)."""
    members: list[tuple[Path, str]] = []
    dropped: list[Path] = []
    dirs = COMMON_DIRS + (FULL_ONLY_DIRS if profile == "full" else [])

    for name, _ in dirs:
        d = BUNDLE / name
        if not d.exists():
            print(f"  note: {name}/ absent, skipping")
            continue
        for p in sorted(d.rglob("*")):
            if not p.is_file():
                continue
            if "__pycache__" in p.parts or p.suffix == ".pyc":
                continue
            members.append((p, str(Path("BRUISE_UNIFIED") / p.relative_to(BUNDLE)).replace("\\", "/")))

    for name, _ in COMMON_FILES:
        p = BUNDLE / name
        if p.exists():
            members.append((p, f"BRUISE_UNIFIED/{name}"))
        else:
            print(f"  note: {name} absent, skipping")

    encoders = FULL_ENCODERS if profile == "full" else N4_ENCODERS
    for enc in encoders:
        base = BUNDLE / "pretrained_weights" / enc
        if not base.exists():
            print(f"  note: pretrained_weights/{enc}/ absent, skipping")
            continue
        subdirs = [d for d in sorted(base.iterdir()) if d.is_dir()] or [base]
        for d in subdirs:
            kept, drop = select_encoder_files(d)
            dropped += drop
            for p in kept:
                members.append((p, str(Path("BRUISE_UNIFIED") / p.relative_to(BUNDLE)).replace("\\", "/")))
        for p in sorted(base.glob("*")):
            if p.is_file() and ".cache" not in p.parts:
                members.append((p, str(Path("BRUISE_UNIFIED") / p.relative_to(BUNDLE)).replace("\\", "/")))

    # Deduplicate, keeping first occurrence.
    seen, out = set(), []
    for p, arc in members:
        if arc not in seen:
            seen.add(arc)
            out.append((p, arc))
    return out, dropped


RUN_SH = """\
#!/usr/bin/env bash
# Stage N4 -- SAM vs MedSAM. Run from the directory this file is in.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# Put the 640 cache and outputs on fast local disk. It is a deterministic resize
# of data/ and rebuilds in about a minute, so it does not need to be backed up.
WORK="${WORK:-$HERE/_work}"

echo "bundle : $HERE"
echo "work   : $WORK"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || {
  echo "no GPU visible -- training needs one" >&2; exit 2; }

python run_stage_n4.py --root "$HERE" --work "$WORK" "$@"
"""

README = """\
BRUISE_UNIFIED -- Stage N4 server package
==========================================

SAM vs MedSAM: does MASK-supervised pretraining beat CAPTION-supervised
pretraining? Self-contained. Code, data, encoder weights, runner, commands.

QUICK START  (server home is /home/tbommawa)
---------------------------------------------

    cd /home/tbommawa
    unzip -q BRUISE_UNIFIED_SERVER.zip
    cd BRUISE_UNIFIED

    # 1. sanity, no GPU needed, ~1 minute
    python run_stage_n4.py --only check,build

    # 2. the whole thing, ~4-5 GPU-hours
    ./run.sh

`run.sh` is a thin wrapper; it passes any extra flags straight through, so
`./run.sh --only gate` works.

WHAT run_stage_n4.py DOES
--------------------------
Seven stages, each skippable, each RESUMABLE. Re-running the same command after
an interruption picks up where it stopped -- engine.train_run honours DONE.json
and resume.pt, and the scoring stages re-read best.pt.

    check     encoders present, structural self-test, config preflight
    build     construct each arm and ASSERT it is really unfrozen   <- the guard
    train     engine.train_run, unmodified               ~2-2.5 GPU-h per arm
    val       fit the operating point on val, score val
    gate      the pre-registered verdict          <- written BEFORE test is read
    test      score test at the cut ALREADY fitted on val
    fairness  size-stratified and ITA-conditioned tables

    python run_stage_n4.py --only check,build          # verify the install
    python run_stage_n4.py --only train --arms sam_ft  # one arm
    python run_stage_n4.py --skip train                # score existing runs
    python run_stage_n4.py --only gate,test,fairness   # re-report

`--only test` REFUSES to run if the gate has not been written. The verdict is
val-only by design (handbook 7f.4): a decision taken on test is a decision taken
on the data the paper reports, and making that an exit code rather than a
convention is the point.

REQUIREMENTS
------------
    python >= 3.10
    torch (CUDA build), transformers, pandas, numpy, scipy, pillow, tqdm

    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install transformers pandas numpy scipy pillow tqdm

One GPU. ~16 GB VRAM is comfortable at the fixed micro-batch of 2; below that
lower it with --micro-batch 1. That value is FIXED and never the VRAM probe --
six unfrozen SAM blocks at 640 with windowed attention is a memory profile the
probe was not calibrated on, and a probe that guesses high dies mid-epoch after
an hour.

Disk: the package unzips to about 3.4 GB, plus ~800 MB for the 640 cache the
first run builds, plus ~1 GB for the two checkpoints.

WHAT IS IN HERE
---------------
    bruisekit/               the library, samprobe.py included
    data/                    831 train+val images and masks, 185 test
    manifests/               the splits, with subject and ITA group per image
    splits/, ita_labels/     what the manifests were cut from
    pretrained_weights/sam/  sam-vit-base and medsam-vit-base
    STAGE_N3_RESULTS/        READ for the reference rows. Never rewritten.
    run_stage_n4.py          the runner
    bruise_stage_n4.ipynb    the same pipeline as a notebook
    PROJECT_HANDBOOK.md      the study; sections 7f-7i are this stage's context

The 640 cache is NOT shipped. It is a deterministic resize of data/ (bilinear for
images, nearest for masks) that rebuilds in about a minute, and a second copy of
the pixels is a second thing that can drift.

The encoder directories are TRIMMED, not converted: facebook/sam-vit-base ships
the same weights three times (safetensors, .bin, .h5) and from_pretrained reads
one. The file that lands is byte-identical to the one downloaded.

WHERE RESULTS GO
----------------
    STAGE_N4_RESULTS/
      runs/     sam_ft__seed0/, medsam_ft__seed0/  best.pt, DONE.json, resume.pt
      tables/   arm_build_info.csv
                val_per_image__*.csv, test_per_image__*.csv
                val_summaries.json, test_summaries.json, operating_points.json
                mask_supervision_gate.{csv,json}
                fairness__*.csv, fairness__gaps.json

Nothing is written to results/, FINAL_RESULT/, FOUNDATION_RESULTS/,
DERM_PROBE_RESULTS/, STAGE_N3_RESULTS/ or _work/runs/. Training goes to
STAGE_N4_RESULTS/runs, not env.runs, so the Registry cannot pull these arms into
a table they are not part of.

To bring results home, that one directory is all you need:

    cd /home/tbommawa/BRUISE_UNIFIED
    zip -r n4_results.zip STAGE_N4_RESULTS/tables STAGE_N4_RESULTS/runs/*/DONE.json

PRE-REGISTERED READING -- fixed in code before any number exists
----------------------------------------------------------------
    medsam - sam clears zero POSITIVE
        -> the medical CORPUS buys something once the objective is right.
           Stages N/N2/N3 measured the objective, not the corpus.
    medsam - sam CONTAINS zero
        -> NULL, and a strong one. Fourth axis, same answer.
    medsam - sam clears zero NEGATIVE
        -> MedSAM is WORSE, the direction DermLIP and MedSigLIP already went.
    either arm < 0.73
        -> INCONCLUSIVE. One seed cannot separate a weak encoder from a collapsed
           run (Stage Y seed 2 did exactly this). Check the loss curve first.

READ THE MISS COLUMN BEFORE THE DICE COLUMN. Dice is saturated (Friedman p = 0.61
across the seven headline models); complete misses on small bruises are the
endpoint that has ever moved and the one a clinician cares about. The gate REFUSES
to print a verdict without them, and a Dice null with a miss-rate win IS a result.

THE THREE FAILURES THAT WOULD FAKE A RESULT
--------------------------------------------
1. AN ARM THAT UNFREEZES NOTHING trains like a frozen probe and is
   indistinguishable from a real ceiling result. The build stage asserts a real
   trainable fraction before training starts. It should print 49.4 %, not 0.0.

2. A POSITION EMBEDDING THAT DID NOT RESAMPLE. SAM stores a 2-D [1,64,64,C]
   embedding for 1024 px input; at 640 that is 40x40. The build stage prints
   "pos_embed resampled 64 -> 40".

3. A LOAD THAT DID NOT LAND. load_sam_encoder picks the class by model_type
   (never AutoModel), checks missing_keys, and checks the parameter count against
   89.7 M before the resample. This is the 2026-08-06 failure, where
   SiglipVisionModel silently loaded a DINOv2 directory as a random ViT and
   invalidated an entire run.

Expected build output, both arms:

    grid 40x40 at 640px (patch 16), features from neck (256-d)
    pos_embed resampled 64 -> 40
    unfrozen 6/12 blocks at .layers, neck=neck (unfrozen=True)
    encoder trainable 43,361,024 / 87,753,984 (49.4 %)
    decoder 951,169   trainable total 44,312,193

Those numbers were verified on the real weights before this package was built. If
yours differ, stop and find out why before spending GPU-hours.

WHAT THIS DOES NOT LICENSE
---------------------------
- Not "MedSAM does not work." We keep only its IMAGE ENCODER and discard the
  prompt encoder and mask decoder, because this pipeline is automatic and has no
  prompt to give. The claim shape is "MedSAM's FEATURES", never "MedSAM".
- Not a licence claim. Both encoders are recorded as Apache-2.0 from their
  releases, not from a licence review. Confirm on the model cards.
- Not a seed-robust result. One seed.
- Not a distribution-matched test. MedSAM's corpus is mostly radiology and
  dermoscopy; ours is clinical photography across skin tones. That is the same
  gap that put MedSigLIP LAST at 0.4670.
"""


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=("n4", "full"), default="n4")
    ap.add_argument("--out", default=None, help="output zip path")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be packed, and how big, without writing")
    args = ap.parse_args()

    dst = Path(args.out) if args.out else \
        ROOT / f"BRUISE_UNIFIED_SERVER{'' if args.profile == 'n4' else '_FULL'}.zip"

    if not BUNDLE.exists():
        print(f"FAIL: {BUNDLE} does not exist", file=sys.stderr)
        return 1

    # The runner is the whole point of this package; refuse to ship without it.
    for required in ("run_stage_n4.py", "bruisekit/samprobe.py"):
        if not (BUNDLE / required).exists():
            print(f"FAIL: {required} is absent. Build the Stage N4 overlay first "
                  f"(scripts/79_generate_stage_n4_notebook.py).", file=sys.stderr)
            return 1

    # The module must pass its own structural checks before it is shipped. No
    # weights, no GPU, no network -- there is no excuse for skipping it.
    sys.path.insert(0, str(BUNDLE))
    from bruisekit import samprobe
    print("-- samprobe self-test --")
    if not samprobe.self_test(verbose=False):
        print("FAIL: samprobe.self_test() did not pass. Not shipping this.",
              file=sys.stderr)
        return 1
    print("   ALL PASS\n")

    print(f"gathering [{args.profile}] ...")
    members, dropped = gather(args.profile)
    total = sum(p.stat().st_size for p, _ in members)
    saved = sum(p.stat().st_size for p in dropped)

    by_top: dict[str, list[int]] = {}
    for p, arc in members:
        top = arc.split("/")[1] if arc.count("/") >= 1 else arc
        b = by_top.setdefault(top, [0, 0])
        b[0] += 1
        b[1] += p.stat().st_size
    print(f"\n{'component':<28} {'files':>7} {'size':>10}")
    print("-" * 48)
    for k in sorted(by_top, key=lambda k: -by_top[k][1]):
        n, b = by_top[k]
        print(f"{k:<28} {n:>7} {b / 1e9:>9.2f}G" if b > 1e9
              else f"{k:<28} {n:>7} {b / 1e6:>9.1f}M")
    print("-" * 48)
    print(f"{'TOTAL':<28} {len(members):>7} {total / 1e9:>9.2f}G")
    if dropped:
        big = sorted((p for p in dropped if p.stat().st_size > 1e6),
                     key=lambda p: -p.stat().st_size)
        print(f"\ndropped {len(dropped)} file(s), {saved / 1e9:.2f}G saved")
        if big:
            print("  duplicate weights (from_pretrained reads one of these):")
            for p in big:
                print(f"    - {p.relative_to(BUNDLE)}  "
                      f"({p.stat().st_size / 1e6:.0f}M)")
        rest = len(dropped) - len(big)
        if rest:
            print(f"  + {rest} small file(s): downloader bookkeeping and "
                  f"non-PyTorch formats")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    print(f"\nwriting {dst} ...")
    t0 = time.time()
    dst.unlink(missing_ok=True)
    done_bytes = 0
    with zipfile.ZipFile(dst, "w", allowZip64=True) as z:
        for i, (p, arc) in enumerate(members, 1):
            mode = (zipfile.ZIP_STORED if p.suffix.lower() in STORED_SUFFIXES
                    else zipfile.ZIP_DEFLATED)
            z.write(p, arc, compress_type=mode)
            done_bytes += p.stat().st_size
            if i % 250 == 0 or i == len(members):
                pct = 100 * done_bytes / max(total, 1)
                print(f"  {i:>6}/{len(members)}  {pct:5.1f}%  "
                      f"{done_bytes / 1e9:.2f}G", end="\r", flush=True)
        print()
        z.writestr("BRUISE_UNIFIED/README_SERVER.txt", README)
        run_sh = zipfile.ZipInfo("BRUISE_UNIFIED/run.sh")
        run_sh.external_attr = 0o755 << 16          # executable on unzip
        run_sh.date_time = time.localtime()[:6]
        z.writestr(run_sh, RUN_SH)
        runner = zipfile.ZipInfo("BRUISE_UNIFIED/run_stage_n4.py")
        z.writestr("BRUISE_UNIFIED/MANIFEST.json", json.dumps({
            "built": time.strftime("%Y-%m-%d %H:%M:%S"),
            "profile": args.profile,
            "n_files": len(members),
            "uncompressed_bytes": total,
            "dropped_duplicate_weights": [str(p.relative_to(BUNDLE)) for p in dropped],
            "entry_point": "run_stage_n4.py",
            "results_dir": "STAGE_N4_RESULTS",
        }, indent=2))

    size = dst.stat().st_size
    print(f"\nwrote {dst}")
    print(f"  {size / 1e9:.2f} GB  ({len(members)} files, "
          f"{(time.time() - t0) / 60:.1f} min)")
    print(f"  sha256[:16] {sha(dst)}")
    print(f"\nupload it, then on the server:")
    print(f"    cd /home/tbommawa")
    print(f"    unzip -q {dst.name}")
    print(f"    cd BRUISE_UNIFIED")
    print(f"    python run_stage_n4.py --only check,build")
    print(f"    ./run.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
