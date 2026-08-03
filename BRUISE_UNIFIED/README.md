# BRUISE_UNIFIED

Everything the bruise-segmentation study needs, in one folder, driven by one
notebook: **`bruise_unified.ipynb`**.

The notebook **trains nothing that is already trained**. Every run in the study is
resolved against the shipped checkpoints first, then against the shipped results,
and only trains as a last resort — and only if you explicitly allow it.

---

## Quick start

### Local Jupyter

```bash
pip install -r requirements.txt
jupyter lab bruise_unified.ipynb      # then Run All
```

### Google Colab

Upload or mount the folder, open the notebook, **Run All**. If auto-detection
does not find the bundle, set one variable in §0:

```python
BUNDLE_ROOT = "/content/drive/MyDrive/BRUISE_UNIFIED"
```

That is the only host-specific setting. Nothing else in the notebook or the
library mentions Colab, Drive or `/content`.

### What a default Run All does

`ALLOW_TRAINING = False` and `RECOMPUTE_FROM_WEIGHTS = False`, so it takes a
couple of minutes **on a laptop with no GPU** and reproduces every table and
figure from the shipped per-image results. Stage D needs no GPU and, in fact, no
`torch` at all.

---

## What is in here

```
bruise_unified.ipynb        the notebook — 51 cells, five stages (A/B/C/E/D)
bruisekit/                  the library (thin notebook, fat library)
  data.py models.py losses.py metrics.py engine.py sweep.py
  evaluate.py postopt.py yolo_native.py         core, verbatim from the tested notebooks
  smp_models.py nnunet_native.py                Stage B architectures
  paths.py                                      host/environment resolution
  registry.py                                   the train-or-skip brain
  loaders.py                                    checkpoint -> model -> test numbers
  report.py                                     Stage D: per-image CSVs -> tables
  efficient_models.py                           Stage E: the four mobile baselines
  weights.py                                    Stage E: download + provenance
  mmcv_shim.py                                  minimal mmcv stand-in (no OpenMMLab needed)
  vendor/                                       StrideFormer / TopFormer, verbatim
  kd/                                           Stage C suite, vendored unmodified
data/                       1016 native-resolution images + masks (2.6 GB)
  train/{images,masks}      831  (697 train + 134 val — the split column decides)
  test/{images,masks}       185
manifests/                  train/val/test.csv + kd_*.csv, all pointing at data/
pretrained_weights/         SegFormer MiT b0/b2/b5, yolo26n-sem.pt
  efficient/                StrideFormer + MobileNetV3 backbones (Stage E)
checkpoints/
  final/                    Stage A — 15 runs (5 models × 3 seeds)
  baselines/                Stage B — 6 runs (U-Net, DeepLabV3+ × 3 seeds)
  distill/                  Stage C — 4 teachers + 10 completed KD arms
  efficient/                Stage E — empty until you train (ALLOW_TRAINING)
results/                    every shipped CSV, JSON and figure
ita_labels/  splits/  interlabeler_agreement_640.csv
```

Total ≈ 5.6 GB.

---

## The three tiers

§3 of the notebook prints a plan before any compute happens. Each run resolves to
exactly one tier:

| Tier | Meaning | Cost |
|---|---|---|
| **WEIGHTS** | a usable checkpoint exists | nothing trains |
| **RESULTS** | no checkpoint, but the metrics were recorded | nothing trains; labelled `cached` |
| **MISSING** | neither | the only case where training is proposed |

A MISSING run **stays** missing in every table. It is never back-filled from a
neighbouring seed and never quietly averaged away.

Current state: **34 runs covered, 2 genuine gaps, 12 Stage E runs awaiting training.**

| Stage | WEIGHTS | RESULTS | MISSING |
|---|---|---|---|
| A · final (SegFormer + native YOLO) | 15 | 0 | 0 |
| B · baselines | 6 | 0 | 1 |
| C · B5 distillation | 13 | 3 | 1 |
| E · mobile baselines (direct only) | 0 | 0 | 12 |

Stage E ships **architectures and pretrained weights but no trained runs** — set
`ALLOW_TRAINING = True` to train them (~1.5 GPU-hours for all four × 3 seeds).

---

## Stage E · Mobile baselines (direct training only)

| Model | Params | Pretrained init | Auto-download |
|---|---|---|---|
| PP-MobileSeg-Tiny | 1.45 M | StrideFormer-Tiny backbone, ImageNet | ✅ openmmlab |
| TopFormer-Tiny | 1.37 M | ImageNet backbone | ⚠️ manual (Google Drive) |
| LR-ASPP MobileNetV3 | 3.22 M | MobileNetV3-Large, ImageNet | ✅ download.pytorch.org |
| Fast-SCNN | 1.14 M | none — scratch **by design** | n/a |

**Direct only.** None of these has a distilled counterpart, so a missing
`..._distilled` run is not a gap — that variant does not exist in this study.

**Two architectures are vendored verbatim** from their reference implementations
(OpenMMLab's StrideFormer, hustvl's TopFormer) behind `bruisekit/mmcv_shim.py`,
which reproduces the handful of mmcv pieces they need without the OpenMMLab
install. Vendoring rather than reimplementing is what makes the published
checkpoints load *by key*:

```
arch                 loaded  in_checkpoint  unexpected  verdict
ppmobileseg_tiny        488            488           0  EXACT MATCH
lraspp_mobilenetv3      308            308           0  EXACT MATCH
```

Regenerate the vendored files with `python scripts/vendor_efficient_nets.py`;
only their import block differs from upstream, and the script aborts rather than
emit a file whose dependencies it has not checked.

**Fast-SCNN's lack of weights is not a gap.** At 1.1 M parameters the paper
trains from scratch and no official checkpoint was ever released, so random init
is the faithful reproduction.

**TopFormer needs one manual step.** The authors publish only via Google Drive
and Baidu, neither fetchable non-interactively. Drop
`topformer-T-224-66.2.pth` into `pretrained_weights/efficient/` to enable
pretrained init; without it the model trains from scratch and is **labelled as
such everywhere it appears**, because on 697 images that difference is large and
would otherwise be invisible.

### Weight downloads are resumable

`bruisekit/weights.py` fetches with an HTTP Range header into a `.part` file, so
a dropped Colab connection costs the bytes in flight rather than the whole file.
Cached files are checksum-verified, not blindly reused; a mismatch raises rather
than training from a file of unknown origin.

---

## Known gaps — read this before quoting completeness

1. **nnU-Net was never run on the canonical 697/134 split.** No weights, no
   results. It is registered as a gap rather than omitted, because "the baseline
   we did not run" is exactly the fact a reader needs. Enabling it needs
   `nnunetv2` installed and roughly 8 GPU-hours.
2. **`x_dkd_b5_to_b0`** is a distillation arm that was configured but never
   executed — the directory holds only its `run_config.json`.

Both appear in the `gaps` table in §3 and in `_work/outputs/gaps.csv`.

Two further limitations that are **not** gaps but do constrain what the bundle
can do:

- **B5 seeds 42 and 2026 have results but no weights.** Only the val-selected
  seed was promoted to `checkpoints/distill/teachers/`. Their numbers are exact;
  they simply cannot be re-inferred.
- **`yolo_sem_distilled__seed2` stopped at 31 epochs** where its siblings ran
  49–84, and its `best.pt` was never stripped of optimizer state (13.4 MB vs
  3.4 MB). It loads and scores correctly; it is why that model's seed spread is
  wide (std 0.065). Preserved as-is rather than silently re-run.

---

## Two flags do all the real work

```python
ALLOW_TRAINING = True          # fill a MISSING run.  Needs CUDA.
RECOMPUTE_FROM_WEIGHTS = True  # regenerate every per-image table from checkpoints.
```

Fresh runs are written to the work directory, **never over the shipped
checkpoints**, so a retrain can always be compared against — or discarded in
favour of — the run that produced the published numbers.

### Every run resumes

`train_run` writes `resume.pt` (model + optimizer + scaler + epoch + best score +
patience + history) every `drive_sync_every` epochs and deletes it on completion.
A disconnect costs at most that many epochs, for **direct and distillation
training alike**. Verified by killing a run mid-flight:

```
--- killed at epoch 2 ---            run dir: best.pt, config.json, history.csv, resume.pt
--- re-run ---                       [resume] fastscnn__resumetest from epoch 3 (best_ap=0.0380)
--- completed ---                    DONE.json written, resume.pt removed
```

A finished run is skipped on sight via `DONE.json`, so re-running a cell is free.

`FORCE_RETRAIN = ["run_id", ...]` retrains specific runs even when a checkpoint
exists. `RUN_STAGES = "AD"` runs a subset.

### Do cached and re-inferred numbers agree?

Yes. All seven reporting models were re-inferred from their checkpoints on **CPU**
and compared against the shipped tables, which were produced on an A100:

| model (val-selected seed) | cached mean Dice | re-inferred | Δ mean | max per-image Δ |
|---|---|---|---|---|
| `segformer_b2_teacher` (0) | 0.769240 | 0.769247 | 7e-6 | 2.6e-3 |
| `segformer_b0_distilled` (0) | 0.768011 | 0.767989 | 2.2e-5 | 7.6e-3 |
| `segformer_b0_direct` (0) | 0.766318 | 0.766278 | 4.0e-5 | 6.8e-3 |
| `deeplabv3plus_r50` (0) | 0.758377 | 0.758349 | 2.8e-5 | 4.4e-3 |
| `unet_r50` (1) | 0.757013 | 0.757059 | 4.6e-5 | 6.9e-3 |
| `yolo_sem_distilled` (0) | 0.726070 | 0.726248 | 1.8e-4 | 4.4e-3 |
| `yolo_sem_direct` (**2**) | 0.702126 | 0.702065 | 6.1e-5 | 6.7e-3 |

Every mean agrees to better than `2e-4`. The residual is floating-point ordering
between CPU and GPU, not disagreement — no ranking, no miss count and no
significance verdict changes.

Full CPU re-inference of all seven takes about 33 minutes; on a GPU it is a
couple of minutes.

---

## Traps this bundle is built to avoid

These are real mistakes that were caught during assembly. The build script
(`scripts/60_build_unified_bundle.py`) asserts against each one, so a future
rebuild cannot reintroduce them.

- **A same-named checkpoint set with the wrong lineage.** `analysis/runs_v2/` in
  the source project has all fifteen Stage A run_ids, but with `alpha=0.5`, a
  fixed batch of 8, and YOLO trained in the custom loop instead of natively.
  Shipping it would produce a bundle that runs, skips, and quietly reports
  numbers that disagree with the paper. The build asserts on `alpha == 0.6` and
  the per-model batch, not on folder names.
- **A baseline set from the wrong split.** `EXTRA/smp_baselines.zip` holds U-Net
  and DeepLabV3+ weights trained on 693/138 at seed 42. The build reconciles each
  run's own `test_per_image.csv` against the canonical 697/134 per-seed CSV and
  refuses anything that does not match.
- **The best seed is not the same for every model.** It is 0 for the three
  SegFormers and `yolo_sem_distilled`, but **2** for `yolo_sem_direct`. Pairing a
  model's weights with another seed's results shows per-image disagreements up to
  0.49 Dice and looks exactly like broken inference. `report.best_seeds()` reads
  the mapping off the selection step's own filenames.
- **The same model under two names.** `segformer_b5_teacher` is not an
  independent model — it is the val-winning B5 seed (123) promoted into
  `teachers/`, matching its per-image Dice bit for bit. Listing both would show
  one result twice. `load_stage_c` collapses exact duplicates and prints the
  alias.
- **A silent ImageNet download.** `segmentation_models_pytorch` fetches ResNet-50
  the moment a model is constructed. When loading a checkpoint every one of those
  weights is about to be overwritten, so `loaders.build_for_load` passes
  `encoder_weights=None` and the bundle stays fully offline.

---

## How to read the results

- **Read D7 before quoting D1.** The models are separated from each other by a
  few Dice points; the human annotators disagree with each other by more than
  that. The D1 ranking is real but sits inside the noise floor of the labels.
- **D3, not D1, carries the practical difference.** Mean Dice spans 0.702–0.769
  across all seven models; the complete-miss rate spans 0% to 6.5% — an order of
  magnitude. A missed bruise is a different kind of failure from a poorly
  outlined one.
- **Read D6 before quoting D5.** Bruise size predicts detection far better than
  skin tone does, and the two are confounded in this dataset.
- **Intervals are subject-level.** The 185 images come from 28 subjects. Every CI
  and contrast resamples subjects, not images; resampling images would make the
  intervals about 2.6× too narrow.
- **YOLO is reported on the native-argmax path only.** The custom `/255` path is
  retained for provenance but is not a reporting path.

---

## Rebuilding

The bundle and the notebook are both build artefacts:

```bash
python scripts/vendor_efficient_nets.py        # refresh the vendored architectures
python scripts/60_build_unified_bundle.py      # assemble + verify the folder
python scripts/61_generate_unified_notebook.py # emit the notebook
python scripts/62_zip_unified_bundle.py        # package it
```

Both are idempotent.

## Environment note

On one Windows/conda environment used during development, `matplotlib 3.10.9`
crashed the interpreter inside `ax.bar()` with a delay-load DLL failure
(`0xc06d007f`). It is an environment defect, not a notebook one — `ax.plot()`
worked in the same process, and `matplotlib 3.11.1` in a sibling environment ran
every figure. If you hit a hard interpreter exit on the first Stage D figure,
upgrade matplotlib.
