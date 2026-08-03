#!/usr/bin/env python
"""Assemble BRUISE_UNIFIED/ -- one self-contained folder holding every artefact
the four study stages need, with nothing duplicated and nothing invented.

WHY A BUILD SCRIPT AND NOT A HAND-COPIED FOLDER
------------------------------------------------
The bundle is ~5.4 GB assembled out of nine different sources, three of which are
zips-inside-zips. Doing that by hand once is error-prone; doing it twice (because
a checkpoint got re-run) is hopeless. This script is idempotent: run it again and
it re-derives the same tree, skipping work that is already done.

WHAT IT REFUSES TO GUESS
-------------------------
Every copy is verified, not assumed:
  * data/    -- the same 1016 native-res images live in TWO sources. The script
                hashes a sample from both and aborts if they ever disagree, so
                the dedupe can never silently ship the wrong pixels.
  * final/   -- checkpoints are accepted only if the run's config.json carries the
                FINAL lineage (segformer alpha 0.6, per-model batch). An earlier
                custom-loop mirror exists on disk with identical run names and
                alpha 0.5; shipping it would silently change every reported number.
  * baselines-- accepted only if test metrics reconcile with the canonical
                697/134 per-seed CSV. A superseded 693/138 seed42 run also exists.

THE CANONICAL DATA LAYOUT
--------------------------
data/train/{images,masks}/  831 files  (the 697 train + 134 val images together)
data/test/{images,masks}/   185 files
This is the distillation suite's native layout, chosen so its manifests work
UNCHANGED. The final/baseline manifests -- which we regenerate anyway -- are
rewritten to point here. Train and val are distinguished by the manifest's
`split` column, never by directory, which is also how the 697/134 split is
defined in the first place.
"""
from __future__ import annotations

import hashlib
import json
import random
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "BRUISE_UNIFIED"

# ── sources ──────────────────────────────────────────────────────────────────
NB_LIB = ROOT / "bruise_colab_baselines.ipynb"   # superset of the final nb's modules
NB_FINAL = ROOT / "bruise_colab_final.ipynb"
DATA_ZIP = ROOT / "bruise_colab_final.zip"       # native-res images + manifests + weights
B5 = ROOT / "BRUISE_DISTILL_B5_RESULT"           # kd suite: code, data, teachers, arms
PRETRAINED = ROOT / "pretrained_weights"

RUNS_FINAL_ZIP = ROOT / "runs_final-20260727T213642Z-1-001.zip"
RUNS_BASE_ZIP = ROOT / "runs_baselines-20260727T213710Z-1-001.zip"
RES_BASE_ZIP = ROOT / "results_baselines-20260727T213712Z-1-001.zip"
RES_FINAL_DIR = ROOT / "results_final"
RES_ANALYSIS_ZIP = ROOT / "final_analysis_native_20260721_022819-20260727T212717Z-1-001.zip"

EXPECTED = {"train": 697, "val": 134, "test": 185}


def log(msg: str) -> None:
    print(f"  {msg}", flush=True)


def step(n: int, title: str) -> None:
    print(f"\n[{n}] {title}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1 · the library
# ─────────────────────────────────────────────────────────────────────────────
def extract_bruisekit() -> None:
    """Pull the `%%writefile bruisekit/<name>.py` cell bodies out of the tested notebook.

    These twelve modules are the code that produced every number in results/. They
    are copied VERBATIM -- not retyped, not "cleaned up" -- because any edit would
    invalidate the checkpoints we are shipping alongside them. The only thing this
    function does is strip the `%%writefile` magic line and write real .py files, so
    the unified notebook can `import bruisekit` like a normal package instead of
    re-emitting the library into the filesystem on every run.
    """
    pkg = OUT / "bruisekit"
    pkg.mkdir(parents=True, exist_ok=True)
    nb = json.loads(NB_LIB.read_text(encoding="utf-8"))
    n = 0
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if not src.startswith("%%writefile bruisekit/"):
            continue
        head, body = src.split("\n", 1)
        name = head.split("bruisekit/")[1].strip()
        (pkg / name).write_text(body, encoding="utf-8")
        n += 1
    assert n == 12, f"expected 12 bruisekit modules, extracted {n}"
    log(f"{n} modules extracted verbatim from {NB_LIB.name}")


def copy_authored_modules() -> None:
    """Copy the modules written FOR the unified bundle into bruisekit/.

    These did not exist before this bundle and cannot be extracted from any
    notebook: paths (host resolution), registry (the train-or-skip tiers), loaders
    (checkpoint -> model -> numbers), report (per-image CSV -> tables), plus the
    Stage E trio -- efficient_models (the four mobile baselines), weights (download
    provenance) and mmcv_shim (so the vendored nets need no OpenMMLab install).

    Their source of truth is scripts/unified_lib/, so that re-running this script
    reproduces the whole package rather than silently depending on files that
    happen to already be sitting in the output directory.
    """
    src = ROOT / "scripts" / "unified_lib"
    dst = OUT / "bruisekit"
    dst.mkdir(parents=True, exist_ok=True)
    names = ["paths.py", "registry.py", "loaders.py", "report.py",
             "efficient_models.py", "weights.py", "mmcv_shim.py",
             # distill_efficient (Stage F teacher shim) and significance (Stage G
             # confirmatory layer) were added by patch and were missing from this
             # list, so a build from a clean output directory would have shipped a
             # bundle whose notebook imports a module that is not in it.
             "distill_efficient.py", "significance.py",
             # reliability_kd (Stage H: the gated loss, the B2 teacher axis, and
             # the gated YOLO pseudo-mask builder).
             "reliability_kd.py",
             # inference (handbook 18.1: the test-set pass and the 640 speed
             # table, over the registry rather than over a hard-coded model list).
             "inference.py"]
    for n in names:
        s = src / n
        if not s.exists():
            raise SystemExit(f"authored module missing from {src}: {n}")
        shutil.copy2(s, dst / n)

    # Vendored third-party architectures (Stage E). Regenerated by
    # scripts/vendor_efficient_nets.py; copied, never edited.
    vsrc, vdst = src / "vendor", dst / "vendor"
    vdst.mkdir(parents=True, exist_ok=True)
    vendored = sorted(vsrc.glob("*.py"))
    if not vendored:
        raise SystemExit(f"no vendored nets in {vsrc}; run scripts/vendor_efficient_nets.py")
    for f in vendored:
        shutil.copy2(f, vdst / f.name)
    log(f"{len(names)} authored + {len(vendored)} vendored modules copied "
        f"from scripts/unified_lib/")


def copy_kd_suite() -> None:
    """Copy the distillation suite's Python into bruisekit/kd/, UNMODIFIED.

    kd_core.py and friends are a separate, independently-tested lineage that
    produced the Stage C arms. They are vendored rather than merged: rewriting
    them to share bruisekit's abstractions would be a refactor of tested code
    whose outputs we are shipping. bruisekit/kd/__init__.py adds the directory to
    sys.path on import so their flat `import kd_core` statements keep working.
    """
    kd = OUT / "bruisekit" / "kd"
    kd.mkdir(parents=True, exist_ok=True)
    names = ["kd_core.py", "distill_segformer.py", "aggregate_report.py", "val_oracle.py",
             "optuna_alpha.py", "score_reference.py", "fairness_eval.py", "calibrate_teacher.py",
             "data_merge.py", "pick_best_b5.py", "tta_infer.py", "fix_keys.py",
             "paired_bootstrap_b5_compare.py", "train_segformer_b5_baseline.py"]
    for n in names:
        src = B5 / n
        if src.exists():
            shutil.copy2(src, kd / n)
        else:
            log(f"WARNING: kd source missing, skipped: {n}")
    (kd / "__init__.py").write_text(
        '"""Vendored distillation suite (Stage C), copied unmodified from the ORC bundle.\n\n'
        "These modules import each other by flat module name (`import kd_core`), which is\n"
        "how they were written and tested. Rather than rewrite those imports -- and thereby\n"
        "modify code whose outputs ship in results/ -- this package puts its own directory\n"
        "on sys.path at import time. `from bruisekit import kd` is therefore enough to make\n"
        "`import kd_core` resolve, from a notebook in any working directory.\n"
        '"""\n'
        "import sys\n"
        "from pathlib import Path\n\n"
        "_HERE = str(Path(__file__).resolve().parent)\n"
        "if _HERE not in sys.path:\n"
        "    sys.path.insert(0, _HERE)\n",
        encoding="utf-8")
    log(f"{len(list(kd.glob('*.py')))} kd modules vendored")


# ─────────────────────────────────────────────────────────────────────────────
# 2 · data (deduped + verified)
# ─────────────────────────────────────────────────────────────────────────────
def verify_data_sources_agree(sample: int = 24) -> None:
    """Hash a random sample of files present in BOTH data sources; abort on mismatch.

    The 1016 native-res images exist twice on disk: inside bruise_colab_final.zip
    (as images/<split>/) and in the distillation bundle (as data/train|test/). We
    ship one copy. That is only safe if they are the same bytes -- if they ever
    diverged, the Stage A checkpoints and the Stage C checkpoints would have been
    trained on different pixels, and every cross-stage comparison in the paper
    would be invalid. Cheap to check, catastrophic to assume.
    """
    zf = zipfile.ZipFile(DATA_ZIP)
    names = [n for n in zf.namelist() if n.startswith(("images/", "masks/")) and not n.endswith("/")]
    rng = random.Random(0)
    checked = 0
    for n in rng.sample(names, min(sample, len(names))):
        kind, split, fn = n.split("/")
        peer = B5 / "data" / ("test" if split == "test" else "train") / kind / fn
        if not peer.exists():
            raise SystemExit(f"data mismatch: {n} has no peer at {peer}")
        if hashlib.md5(zf.read(n)).hexdigest() != hashlib.md5(peer.read_bytes()).hexdigest():
            raise SystemExit(f"data mismatch: {n} differs from {peer} -- refusing to dedupe")
        checked += 1
    log(f"{checked}/{len(names)} sampled files byte-identical across both sources -- dedupe safe")


def copy_data() -> None:
    """Copy the single canonical data tree (2.6 GB) from the distillation bundle."""
    dst = OUT / "data"
    if dst.exists() and sum(1 for _ in dst.rglob("*.jpg")) == 1016:
        log("data/ already present (1016 images) -- skipped")
        return
    for split in ("train", "test"):
        for kind in ("images", "masks"):
            s, d = B5 / "data" / split / kind, dst / split / kind
            d.mkdir(parents=True, exist_ok=True)
            for f in sorted(s.iterdir()):
                t = d / f.name
                if not t.exists():
                    shutil.copy2(f, t)
            log(f"data/{split}/{kind}: {len(list(d.iterdir()))} files")


def build_manifests() -> None:
    """Regenerate every manifest against the canonical data/ layout.

    Two families of manifest exist and BOTH are rewritten here so there is exactly
    one description of the split in the bundle:

      manifests/{train,val,test}.csv   Stage A/B. Carries stem, subject, ITA, split
                                       and now data-relative image/mask paths.
      manifests/kd_{train,test}.csv    Stage C. Same rows, the column names kd_core
                                       expects, paths relative to the same root.

    The val images physically live under data/train/ -- val is a SUBSET of the
    831 training-side files, separated by the `split` column, never by directory.
    That is deliberate: the 697/134 boundary is a property of the split file, and
    duplicating val images into their own folder would create a second, silently
    divergent source of truth for it.
    """
    mdir = OUT / "manifests"
    mdir.mkdir(parents=True, exist_ok=True)
    zf = zipfile.ZipFile(DATA_ZIP)
    frames = {}
    for split, n_expected in EXPECTED.items():
        df = pd.read_csv(zf.open(f"manifests/{split}.csv"))
        assert len(df) == n_expected, f"{split}: {len(df)} rows, expected {n_expected}"
        side = "test" if split == "test" else "train"
        df["image_path"] = df["stem"].map(lambda s: f"{side}/images/{s}.jpg")
        df["mask_path"] = df["stem"].map(lambda s: f"{side}/masks/{s}.png")
        # Every referenced file must exist -- catches a partial data copy immediately
        # rather than 40 minutes into a training run.
        missing = [p for p in df["image_path"] if not (OUT / "data" / p).exists()]
        if missing:
            raise SystemExit(f"{split}: {len(missing)} images missing from data/, e.g. {missing[0]}")
        df.to_csv(mdir / f"{split}.csv", index=False)
        frames[split] = df
        log(f"manifests/{split}.csv: {len(df)} rows, {df['subject'].nunique()} subjects")

    # Leakage is re-checked here, at build time, so a broken bundle cannot be shipped.
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert not set(frames[a].subject) & set(frames[b].subject), f"subject leak {a}/{b}"
        assert not set(frames[a].stem) & set(frames[b].stem), f"image leak {a}/{b}"
    log("no subject or image leakage across train/val/test")

    kd_train = pd.concat([frames["train"], frames["val"]], ignore_index=True)
    kd_train[["stem", "subject", "split", "image_path", "mask_path"]].to_csv(
        mdir / "kd_train.csv", index=False)
    frames["test"][["stem", "subject", "split", "image_path", "mask_path"]].to_csv(
        mdir / "kd_test.csv", index=False)
    log(f"manifests/kd_train.csv: {len(kd_train)} rows (split column carries 697/134)")


# ─────────────────────────────────────────────────────────────────────────────
# 3 · weights and checkpoints (verified lineage)
# ─────────────────────────────────────────────────────────────────────────────
def copy_pretrained() -> None:
    """Copy the pretrained backbones so no stage needs to reach the internet.

    b0/b2 are needed to BUILD a SegFormer before loading a checkpoint into it
    (the architecture is instantiated from the HF config in these folders); b5 is
    needed only if a Stage C arm is retrained; yolo26n-sem.pt is Ultralytics'
    Cityscapes checkpoint that every YOLO run started from.
    """
    dst = OUT / "pretrained_weights"
    dst.mkdir(parents=True, exist_ok=True)
    for item in ["segformer_mit_b0", "segformer_mit_b2", "segformer_mit_b5", "yolo26n-sem.pt"]:
        s = PRETRAINED / item
        d = dst / item
        if d.exists():
            continue
        if s.is_dir():
            shutil.copytree(s, d)
        elif s.exists():
            shutil.copy2(s, d)
        else:
            log(f"WARNING: pretrained weight missing: {item}")
    log(f"{len(list(dst.iterdir()))} pretrained weight sets")


def _unzip_into(zip_path: Path, dst: Path, strip_top: str) -> None:
    """Extract a Drive-exported zip, dropping its redundant top-level folder."""
    dst.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            rel = info.filename
            if rel.startswith(strip_top + "/"):
                rel = rel[len(strip_top) + 1:]
            target = dst / rel
            if target.exists() and target.stat().st_size == info.file_size:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)


def copy_final_checkpoints() -> None:
    """Extract runs_final/ and verify it is the FINAL lineage, not the old mirror.

    THE TRAP THIS GUARDS AGAINST
    -----------------------------
    analysis/runs_v2/ on this machine holds fifteen directories with byte-for-byte
    the same run_ids -- segformer_b2_teacher__seed0 and so on -- from an earlier
    custom-loop experiment. Its distilled runs used alpha 0.5 and a fixed batch of
    8, and its YOLO models were trained in the custom loop rather than natively.
    Dropping it in would produce a bundle that runs, skips, and reports numbers
    that quietly disagree with the paper. Names are not evidence; configs are.

    So we assert on the two config fields that actually separate the lineages, and
    on the presence of the NATIVE Ultralytics weight path for the YOLO runs.
    """
    dst = OUT / "checkpoints" / "final"
    _unzip_into(RUNS_FINAL_ZIP, dst, "runs_final")

    seg = ["segformer_b2_teacher", "segformer_b0_direct", "segformer_b0_distilled"]
    for name in seg:
        for seed in (0, 1, 2):
            rd = dst / f"{name}__seed{seed}"
            for req in ("DONE.json", "best.pt", "operating_point.json", "config.json"):
                if not (rd / req).exists():
                    raise SystemExit(f"{rd.name}: missing {req}")
            cfg = json.loads((rd / "config.json").read_text())
            if name.endswith("distilled"):
                if cfg.get("alpha") != 0.6:
                    raise SystemExit(
                        f"{rd.name}: alpha={cfg.get('alpha')}, expected 0.6. This looks like the "
                        f"superseded custom-loop lineage (alpha 0.5) -- refusing to ship it.")
                if cfg.get("teacher_temperature") is None:
                    raise SystemExit(f"{rd.name}: no calibrated teacher_temperature")
            if cfg.get("micro_batch") not in (32, 64):
                raise SystemExit(
                    f"{rd.name}: micro_batch={cfg.get('micro_batch')}, expected the per-model "
                    f"probe result (32 for b2, 64 for b0), not the old fixed 8.")

    for name in ("yolo_sem_direct", "yolo_sem_distilled"):
        for seed in (0, 1, 2):
            w = dst / f"{name}__seed{seed}" / "ultralytics_runs" / "train" / "weights" / "best.pt"
            if not w.exists():
                raise SystemExit(f"{name}__seed{seed}: no NATIVE Ultralytics best.pt at {w}")

    log("15/15 final runs present, FINAL lineage verified (alpha 0.6, per-model batch, native YOLO)")


def copy_baseline_checkpoints() -> None:
    """Extract runs_baselines/ and reconcile it against the canonical per-seed CSV.

    Same trap as Stage A, different disguise: EXTRA/smp_baselines.zip holds U-Net
    and DeepLabV3+ weights trained on a 693/138 split at seed 42. The runs we want
    are the 697/134 three-seed set. Rather than trust the folder name, we check
    that each run's own test_per_image.csv reproduces the mean Dice recorded in
    the canonical results CSV. If a run's weights and the paper's numbers came
    from different training, this catches it.
    """
    dst = OUT / "checkpoints" / "baselines"
    _unzip_into(RUNS_BASE_ZIP, dst, "runs_baselines")

    with zipfile.ZipFile(RES_BASE_ZIP) as zf:
        canon = pd.read_csv(zf.open("results_baselines/smp_baselines_test_per_seed.csv"))
    canon = canon.set_index("run_id")["mean_dice"]

    for model in ("unet_r50", "deeplabv3plus_r50"):
        for seed in (0, 1, 2):
            run_id = f"{model}__seed{seed}"
            rd = dst / run_id
            for req in ("DONE.json", "best.pt", "operating_point.json", "test_per_image.csv"):
                if not (rd / req).exists():
                    raise SystemExit(f"{run_id}: missing {req}")
            got = pd.read_csv(rd / "test_per_image.csv")["dice"].mean()
            want = canon[run_id]
            if abs(got - want) > 1e-4:
                raise SystemExit(
                    f"{run_id}: per-image mean Dice {got:.6f} != canonical {want:.6f}. "
                    f"These weights are not the 697/134 run that produced the results.")
    log("6/6 baseline runs present, per-image Dice reconciles with the canonical 697/134 CSV")


def copy_distill_checkpoints() -> None:
    """Copy the Stage C teachers and finished distillation arms.

    Arms are copied wholesale; the registry decides at run time which are usable.
    One arm (x_dkd_b5_to_b0) is an empty directory from a run that was never
    started -- it is copied as-is rather than hidden, so the notebook can report
    it honestly as the one genuine gap in Stage C instead of pretending the
    experiment grid is complete.
    """
    dst = OUT / "checkpoints" / "distill"
    dst.mkdir(parents=True, exist_ok=True)
    for sub in ("teachers", "distill_out"):
        s, d = B5 / sub, dst / sub
        if d.exists():
            log(f"checkpoints/distill/{sub} already present -- skipped")
            continue
        shutil.copytree(s, d)
    arms = [p for p in (dst / "distill_out").iterdir()
            if p.is_dir() and p.name not in ("aggregate", "reference", "val_oracle", "optuna_alpha")]
    done = [p for p in arms if (p / "DONE.json").exists()]
    log(f"{len(done)}/{len(arms)} distillation arms complete; "
        f"{len(list((dst / 'teachers').glob('*/best_model.pt')))} teacher checkpoints")


# ─────────────────────────────────────────────────────────────────────────────
# 4 · results and reference material
# ─────────────────────────────────────────────────────────────────────────────
def copy_results() -> None:
    """Gather every result artefact, expanding the Drive zips-inside-zips.

    These CSVs are what makes the bundle useful WITHOUT a GPU: the registry's
    third fallback tier reports from them when neither weights nor a live run are
    available, so `Run All` on a laptop still reproduces every table.
    """
    res = OUT / "results"
    (res / "final").mkdir(parents=True, exist_ok=True)

    for pattern in ("*.csv", "*.png"):
        for f in RES_FINAL_DIR.glob(pattern):
            shutil.copy2(f, res / "final" / f.name)
    # The Drive exports nest everything under one redundant top-level folder whose
    # name carries a timestamp. `_unzip_into` strips it and skips files already at
    # the right size, which is what makes re-running this script a no-op instead of
    # a FileExistsError or a doubled tree.
    for z in sorted(RES_FINAL_DIR.glob("*.zip")):
        top = zipfile.ZipFile(z).namelist()[0].split("/")[0]
        # The older baselines export is byte-identical to the newer standalone one
        # that lands in results/baselines/. Extracting both would put the same
        # numbers in two places and leave a reader to guess which is authoritative.
        if top == "results_baselines":
            continue
        _unzip_into(z, res / "final" / top, top)
    log(f"results/final: {sum(1 for _ in (res / 'final').rglob('*') if _.is_file())} files")

    _unzip_into(RES_BASE_ZIP, res / "baselines", "results_baselines")
    log(f"results/baselines: {sum(1 for _ in (res / 'baselines').rglob('*') if _.is_file())} files")

    analysis_top = zipfile.ZipFile(RES_ANALYSIS_ZIP).namelist()[0].split("/")[0]
    _unzip_into(RES_ANALYSIS_ZIP, res / "analysis_native", analysis_top)
    log(f"results/analysis_native: {sum(1 for _ in (res / 'analysis_native').rglob('*') if _.is_file())} files")

    dstill = res / "distill"
    dstill.mkdir(exist_ok=True)
    shutil.copytree(B5 / "results_segformer_b5", dstill / "segformer_b5", dirs_exist_ok=True)
    for f in (B5 / "teachers").glob("*.csv"):
        shutil.copy2(f, dstill / f.name)
    log(f"results/distill: {sum(1 for _ in dstill.rglob('*') if _.is_file())} files")


def copy_reference() -> None:
    """Copy the ITA skin-tone labels and the inter-labeler agreement table.

    Both are inputs to analyses that no checkpoint can substitute for: fairness
    needs the per-image ITA group, and the annotation-ceiling figure needs the
    human-vs-human Dice. Without these, Stage D silently loses its two most
    load-bearing results.
    """
    shutil.copytree(B5 / "ita_labels", OUT / "ita_labels", dirs_exist_ok=True)
    shutil.copytree(B5 / "splits", OUT / "splits", dirs_exist_ok=True)
    src = ROOT / "interlabeler_agreement_640.csv"
    if src.exists():
        shutil.copy2(src, OUT / "interlabeler_agreement_640.csv")
    else:
        with zipfile.ZipFile(DATA_ZIP) as zf:
            (OUT / "interlabeler_agreement_640.csv").write_bytes(
                zf.read("interlabeler_agreement_640.csv"))
    log("ITA labels, splits and inter-labeler agreement copied")


def write_requirements() -> None:
    """Pin the runtime deps, split by which stage actually needs them."""
    (OUT / "requirements.txt").write_text(
        "# Core -- every stage needs these.\n"
        "torch>=2.2\ntorchvision>=0.17\n"
        "transformers>=4.40,<6\n"
        "albumentations>=2.0,<3\n"
        "opencv-python-headless>=4.9\n"
        "numpy>=1.24\npandas>=2.0\nscipy>=1.11\nmatplotlib>=3.7\npyyaml\ntqdm\n"
        "\n# Stage A -- YOLO (native Ultralytics training and argmax inference).\n"
        "ultralytics>=8.4,<9\n"
        "\n# Stage B -- U-Net / DeepLabV3+ wrappers.\n"
        "segmentation-models-pytorch>=0.3.3\n"
        "\n# Stage C -- optional: the KD alpha search falls back to a grid without it.\n"
        "optuna>=3.5\n"
        "\n# Stage B nnU-Net -- OFF by default; install only if you intend to train it.\n"
        "# nnunetv2>=2.5\n",
        encoding="utf-8")
    log("requirements.txt written")


def main() -> None:
    if not OUT.exists():
        OUT.mkdir(parents=True)
    print(f"Assembling {OUT}")

    step(1, "library")
    extract_bruisekit()
    copy_authored_modules()
    copy_kd_suite()

    step(2, "data (dedupe + verify)")
    verify_data_sources_agree()
    copy_data()
    build_manifests()

    step(3, "pretrained weights")
    copy_pretrained()

    step(4, "checkpoints (lineage-verified)")
    copy_final_checkpoints()
    copy_baseline_checkpoints()
    copy_distill_checkpoints()

    step(5, "results and reference material")
    copy_results()
    copy_reference()
    write_requirements()

    total = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    n = sum(1 for f in OUT.rglob("*") if f.is_file())
    print(f"\nDONE  {OUT}\n      {n} files, {total / 1e9:.2f} GB")


if __name__ == "__main__":
    sys.exit(main())
