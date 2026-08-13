# Code map — where each piece of functionality lives

**Generated** by `scripts/88_generate_code_map.py`. Line numbers are read
from the source at build time, so regenerate after any edit rather than
patching this file. Every entry carries the search that finds it again if
your copy has drifted.

All paths are relative to `BRUISE_UNIFIED/`.

## Attaching a segmentation head to an encoder that has none

Foundation models output a grid of features, not a mask. These are the three places a head gets built, and they are deliberately the same head so cross-stage numbers stay comparable.

| What | Where | Find it with |
|---|---|---|
| **The head itself — 1M-param conv decoder**<br>Two 3x3 convs + BN + ReLU, then a 1x1 to one logit, then bilinear upsample to full resolution. Stage N3, N4 and O all import THIS class rather than re-typing it. | `bruisekit/finetune_n3.py:117` | `grep -n "^class ConvDecodeHead" bruisekit/finetune_n3.py` |
| **Linear probe head (frozen-encoder arms)**<br>Deliberately weak: a 1x1 conv and nothing else, so a strong decoder cannot paper over a weak encoder in the frozen comparison. | `bruisekit/foundation.py:282` | `grep -n "^class LinearProbeHead" bruisekit/foundation.py` |
| **Where the head is attached — base contract**<br>`_ProbeBase` is the architecture-blind contract every probe subclasses: raw [0,1] pixels in, full-resolution logits out. | `bruisekit/dermprobe.py:679` | `grep -n "def _build_head" bruisekit/dermprobe.py` |
| **Head attached — SAM / MedSAM**<br>Overrides the linear head with the SAME ConvDecodeHead Stage N3 used, or the two stages are not comparable. | `bruisekit/samprobe.py:639` | `grep -n "def _build_head" bruisekit/samprobe.py` |

## Freezing the encoder, and unfreezing part of it

The difference between 'train only the head' and 'train the last six blocks'. UNFREEZE_BLOCKS = 6 in both fine-tuning stages, pre-registered.

| What | Where | Find it with |
|---|---|---|
| **Freeze everything (frozen probe)**<br>Sets requires_grad False on the whole encoder AND pins it to eval() so BatchNorm/dropout cannot drift — freezing weights alone does not do that. | `bruisekit/dermprobe.py:707` | `grep -n "def _freeze" bruisekit/dermprobe.py` |
| **Unfreeze the last N blocks — DINOv2 / DermLIP**<br>Last 6 of 12 transformer blocks plus the final LayerNorm. The norm is included because its statistics are tuned to the blocks below it. | `bruisekit/finetune_n3.py:222` | `grep -n "def unfreeze_last" bruisekit/finetune_n3.py` |
| **Unfreeze the last N blocks — SAM / MedSAM**<br>Same 6 blocks, plus the neck. RAISES if it ends up with zero trainable params — an arm that silently trains nothing scores low-to-mid and is indistinguishable from a real result. | `bruisekit/samprobe.py:647` | `grep -n "def unfreeze_last" bruisekit/samprobe.py` |
| **How many blocks — the pre-registered constant**<br>Changing this changes the experiment. Do it in the module, deliberately, and say so in the write-up. | `bruisekit/samprobe.py:122` | `grep -n "^UNFREEZE_BLOCKS = " bruisekit/samprobe.py` |

## Loss functions

Four losses, each a strict extension of the one above it. That nesting is what makes every distillation contrast one-variable.

| What | Where | Find it with |
|---|---|---|
| **Dice + BCE — the supervised base**<br>Dice is computed PER IMAGE then averaged, not pooled over the batch — a batch-pooled Dice lets one large bruise dominate the gradient. | `bruisekit/losses.py:43` | `grep -n "^class DiceBCELoss" bruisekit/losses.py` |
| **Supervised loss (+ auxiliary head)**<br>What every non-distilled arm trains on. | `bruisekit/losses.py:66` | `grep -n "^class SupervisedLoss" bruisekit/losses.py` |
| **Distillation loss**<br>alpha * DiceBCE(student, GT) + (1-alpha) * BCE(student, calibrated teacher prob). NOT Hinton KD — the module docstring above it explains the difference and why it matters for the paper. | `bruisekit/losses.py:87` | `grep -n "^class DistillLoss" bruisekit/losses.py` |
| **Reliability-gated KD loss (Stage H)**<br>Down-weights the teacher per pixel where it is confidently wrong. Reduces to DistillLoss exactly when the gate is fully open. | `bruisekit/reliability_kd.py:428` | `grep -n "def _build_gated_loss_class" bruisekit/reliability_kd.py` |
| **Per-image routed multi-teacher loss (Stage M)**<br>softmax(beta * per-teacher soft Dice against the label). K=1 reduces to the Stage H loss bit-for-bit. | `bruisekit/multiteacher.py:1044` | `grep -n "def _build_routed_loss_class" bruisekit/multiteacher.py` |
| **Per-ITA-group routed loss (Stage O)**<br>Weights fitted once on validation, indexed by the image's skin-tone group. RAISES if the group vector is missing rather than falling back to uniform weights. | `bruisekit/itakd.py:1394` | `grep -n "def _build_group_loss_class" bruisekit/itakd.py` |

## The training loop

One driver for every model in the study. A bespoke loop anywhere would make that arm's numbers unreadable against the rest.

| What | Where | Find it with |
|---|---|---|
| **The loop itself**<br>Idempotent and resumable: DONE.json returns immediately, resume.pt restores model+optimizer+scaler+epoch. | `bruisekit/engine.py:257` | `grep -n "def train_run" bruisekit/engine.py` |
| **Backbone/head learning-rate split**<br>6e-5 backbone / 6e-4 head, and no weight decay on norms and biases. Membership is by id(), not by name — a name-prefix rule would put every YOLO parameter in the wrong group. | `bruisekit/models.py:132` | `grep -n "def build_param_groups" bruisekit/models.py` |
| **LR schedule — warmup then poly decay**<br>Applied per OPTIMIZER STEP, not per micro-batch, or accumulation advances the schedule too fast. | `bruisekit/engine.py:41` | `grep -n "def lr_multiplier" bruisekit/engine.py` |
| **Batch-size probe**<br>Measures what actually fits, then accumulates back to the target effective batch. | `bruisekit/engine.py:49` | `grep -n "def resolve_micro_batch" bruisekit/engine.py` |
| **Teacher temperature calibration**<br>Fitted by NLL on validation (Guo et al. 2017). Never falls back to T=1 — an uncalibrated teacher's soft label is the hard label with extra steps. | `bruisekit/engine.py:167` | `grep -n "def calibrate_temperature" bruisekit/engine.py` |

## Choosing the operating point (threshold)

Fitted on validation, applied once to test. Nothing is ever fitted on test.

| What | Where | Find it with |
|---|---|---|
| **Cache logits once, sweep 481 cuts**<br>481 cuts x 134 images would be 64k forward passes; caching makes the sweep pure tensor arithmetic. | `bruisekit/sweep.py:10` | `grep -n "def cache_logits" bruisekit/sweep.py` |
| **The sweep**<br>Reductions are exact int64 on boolean masks. fp16 sums drift by ~1.5e-4, which is a tenth of the signal the tie band ranks on. | `bruisekit/sweep.py:33` | `grep -n "def sweep_cuts" bruisekit/sweep.py` |
| **Picking the cut**<br>NOT the argmax. Every cut within one standard error of the peak is statistically tied; the tie is broken by LOWEST MISS RATE. | `bruisekit/sweep.py:74` | `grep -n "def select_cut" bruisekit/sweep.py` |
| **Reading a stored cut**<br>The single place that converts between the logit and probability threshold dialects. sigmoid(cut) == threshold, so reading one as the other does not raise — it silently moves the boundary. | `bruisekit/registry.py:251` | `grep -n "def read_cut" bruisekit/registry.py` |

## Data, augmentation and the 640 cache

| What | Where | Find it with |
|---|---|---|
| **Dataset**<br>Returns (x[3,H,W] in [0,1], y[1,H,W] in {0,1}, stem). Emits RAW [0,1] — normalisation happens inside each model's forward. | `bruisekit/data.py:55` | `grep -n "^class BruiseDataset" bruisekit/data.py` |
| **Augmentation**<br>Train-time only; validation and test see the deterministic resize. | `bruisekit/data.py:31` | `grep -n "def build_augmentation" bruisekit/data.py` |
| **Dataloader**<br>Seeded shuffling and seeded per-worker RNG, so augmentation is reproducible for a given seed. | `bruisekit/data.py:95` | `grep -n "def make_loader" bruisekit/data.py` |
| **Building the 640 cache**<br>Bilinear for images, NEAREST for masks. Bilinear on a mask gives fractional boundary pixels, and thresholding those erodes small bruises — exactly the population the miss metric is about. | `bruisekit/loaders.py:483` | `grep -n "def build_cache640" bruisekit/loaders.py` |

## Scoring and metrics

| What | Where | Find it with |
|---|---|---|
| **Per-image Dice**<br>Counts pixels. Both-empty scores 1.0, matching the sweep. | `bruisekit/metrics.py:9` | `grep -n "def dice_np" bruisekit/metrics.py` |
| **Score a model at a cut**<br>Produces the per-image table everything downstream reads. | `bruisekit/evaluate.py:13` | `grep -n "def evaluate_at_cut" bruisekit/evaluate.py` |
| **Normalise any producer's CSV**<br>Joins subject and ITA group from the manifest, and derives complete_miss = (dice == 0). The manifest is the single source of truth for who each image belongs to. | `bruisekit/report.py:68` | `grep -n "def normalize" bruisekit/report.py` |
| **Run inference over the registry**<br>Delegates to loaders.score_run. There is deliberately no second inference implementation anywhere in the codebase. | `bruisekit/inference.py:151` | `grep -n "def inference_pass" bruisekit/inference.py` |
| **Best seed per model**<br>Read from the selection step's own filenames. It is 0 for the SegFormers and 2 for yolo_sem_direct; scoring a model at another model's best seed shows per-image gaps up to 0.49 Dice. | `bruisekit/inference.py:114` | `grep -n "def resolve_runs" bruisekit/inference.py` |

## Analysis — misses, size, fairness

| What | Where | Find it with |
|---|---|---|
| **Miss taxonomy: blank vs wrong-place**<br>wrong_place is DERIVED as zero_dice - empty_pred so the three columns cannot fail to add up. RAISES if empty > zero, which is arithmetically impossible and means the table is inconsistent. | `bruisekit/itakd.py:166` | `grep -n "def _miss_counts" bruisekit/itakd.py` |
| **Size deciles, cut once globally**<br>RAISES if two models disagree about an image's GT area — that means different mask versions and every cross-model column would be meaningless. | `bruisekit/lesionsize.py:324` | `grep -n "def assign_bins" bruisekit/lesionsize.py` |
| **Fairness conditioned on lesion size**<br>Reports each skin-tone gap marginally AND within the small-lesion stratum. Where the gap shrinks, it was partly a size effect. | `bruisekit/lesionsize.py:710` | `grep -n "def fairness_conditioned" bruisekit/lesionsize.py` |
| **Cross-host table discovery**<br>Scans every root that has ever held a per-image CSV. See handbook 10.3 — a hard-coded lineage runs on the laptop and raises on ORC. | `bruisekit/allmodels.py:189` | `grep -n "def discover" bruisekit/allmodels.py` |
| **Cohorting — what may be compared**<br>Two tables share a cohort iff they scored the same stems with the same mask areas. Anything else is quarantined, not silently mixed. | `bruisekit/allmodels.py:223` | `grep -n "def _signature" bruisekit/allmodels.py` |

## The pre-registered gates

Each stage decides on VALIDATION whether it is worth training, and writes the verdict to disk before anything touches test.

| What | Where | Find it with |
|---|---|---|
| **Stage M — multi-teacher oracle gate**<br>Projects the oracle gain through Stage C's MEASURED transfer rate rather than through an assumption. | `bruisekit/multiteacher.py:502` | `grep -n "def oracle_gate" bruisekit/multiteacher.py` |
| **Stage N4 — mask vs caption pretraining**<br>The medsam - sam contrast, with the reading fixed in code before any number existed. | `bruisekit/samprobe.py:916` | `grep -n "def mask_supervision_gate" bruisekit/samprobe.py` |
| **Stage O — ITA-group gate**<br>Adds the identifiability clause the earlier gates lacked: refuses unless the per-group teacher ranking survives resampling patients. | `bruisekit/itakd.py:992` | `grep -n "def ita_group_gate" bruisekit/itakd.py` |
| **Is the routing key even estimable?**<br>Bootstraps the per-group argmax. 36-52 % stable across six candidate teachers — a coin flip, not a ranking. | `bruisekit/itakd.py:941` | `grep -n "def identifiability" bruisekit/itakd.py` |

## The shims — how a stage changes behaviour without editing the pipeline

Every experiment patches a name rather than editing engine.py. Each shim falls through to whatever was bound before it when its arm is not active.

| What | Where | Find it with |
|---|---|---|
| **Build a model the pipeline has never heard of**<br>Rebinds build_model in BOTH bruisekit.models and bruisekit.engine — engine bound it by value at import, so patching one is not enough. | `bruisekit/samprobe.py:771` | `grep -n "def install_n4_shim" bruisekit/samprobe.py` |
| **Efficient architectures**<br>Same pattern for the four mobile architectures. | `bruisekit/efficient_models.py:492` | `grep -n "def install_efficient_shim" bruisekit/efficient_models.py` |
| **Return a STACK of teachers**<br>load_teacher returns [B, K, H, W] instead of [B, 1, H, W], so the whole pool reaches the loss with no edit to the training loop. | `bruisekit/itakd.py:1318` | `grep -n "def install_teacher_shim" bruisekit/itakd.py` |
| **Get the ITA group to the loss**<br>train_run iterates (x, y, _) and DISCARDS the stem, so the training loader is wrapped to record each batch's group indices. | `bruisekit/itakd.py:1282` | `grep -n "def install_group_shim" bruisekit/itakd.py` |
| **Swap the loss**<br>Install this LAST — the dispatcher must sit on top of Stage H's and Stage M's. | `bruisekit/itakd.py:1536` | `grep -n "def install_loss_shim" bruisekit/itakd.py` |
| **Pin the batch to the control's**<br>Accumulation restores the control's EFFECTIVE batch exactly, so the arm differs in the teacher signal alone and not also in step count and LR schedule. | `bruisekit/multiteacher.py:891` | `grep -n "def control_batch" bruisekit/multiteacher.py` |

---

## Three rules that explain most of the code you will read

1. **Nothing is fitted on test.** Thresholds, best seeds and every gate
   are decided on validation, and the gate verdict is written to disk
   before any test pass runs.
2. **An experiment patches a name; it never edits the pipeline.** The
   shims section above is the whole mechanism. Stage modules are
   deliberately absent from `60_build_unified_bundle.py`'s copy list, so
   an experiment that returns nothing cannot become a dependency of the
   file that produces Stages A–Y.
3. **A function that cannot do its job raises.** It does not warn, and it
   does not return a plausible default. Most of the `raise` statements in
   this codebase mark a place where a silent fallback once produced a
   believable wrong number.

## Regenerating

```bash
python scripts/88_generate_code_map.py
```

It **fails** if an anchor stops matching, which means the thing it
described was renamed or removed. Fix the anchor rather than deleting the
row.
