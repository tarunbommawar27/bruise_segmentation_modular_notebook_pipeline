#!/usr/bin/env python
"""Emit `BRUISE_UNIFIED/bruise_stage_m.ipynb` -- one notebook, runs top to bottom.

WHAT CHANGED FROM THE FIRST VERSION, AND WHY EACH ONE IS HERE
---------------------------------------------------------------
Every setting below is one we established by running it, not a guess:

  EXTRA_RUNS                confirmed resolving -- all three b2kd seeds found
  POOL = 3 teachers         U-Net's drop-one marginal was +0.0055 against
                            +0.011..+0.022 for the rest. Dropping it cost exactly
                            0.0055 of oracle gain and DOUBLED DeepLab's marginal,
                            because U-Net and DeepLab are both ResNet-50 and were
                            covering for each other.
  MAX_MICRO_BATCH = 16      the OOM. 3 teachers resident + the student at
                            micro-batch 64 misses a 40 GB MIG slice by ~200 MB.
                            16 x 4 keeps the effective batch, the step count and
                            the LR schedule identical to the control.
  TEACHER_CHUNK = 4         teacher forwards sub-batched; changes no number.
  PYTORCH_CUDA_ALLOC_CONF   set in the FIRST cell, before torch is imported --
                            after that it has no effect. ~1 GB of fragmentation.
  RESULTS_DIR               STAGE_M_RESULTS/ at the bundle root. Nothing is
                            written to results/ or FINAL_RESULT/, and the runs go
                            to STAGE_M_RESULTS/runs/ rather than _work/runs/, so
                            an experiment that fails leaves no trace in the
                            directories the published numbers come from.
  Dice-only pre-registration
                            the miss clause PASSES but is illusory: B2 alone has
                            1 val miss, the same as the whole pool, and that miss
                            is the one unlabelled image nobody finds. It reads
                            PASS only because it is compared against B5 (3
                            misses). Said in the notebook where it will be read.

The gate still runs first and the guard still reads the JSON it wrote. That is
not ceremony -- it is the one thing separating this from Stage C, which opened on
an oracle number and spent ten arms discovering the student inherited a quarter
of it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DST = ROOT / "BRUISE_UNIFIED" / "bruise_stage_m.ipynb"

CELLS: list[tuple[str, str]] = []


def md(src: str) -> None:
    CELLS.append(("markdown", src.strip("\n")))


def code(src: str) -> None:
    CELLS.append(("code", src.strip("\n")))


# ─────────────────────────────────────────────────────────────────────────────
md(r"""
# Stage M — multi-teacher routed distillation

**Run every cell in order.** Sections 0–7 are the gate: validation only, no
training, a few minutes. Sections 8–15 train. The guard in §9 reads the JSON the
gate wrote, so if the gate closes, nothing trains.

Everything below is configured from runs we have already done — the teacher pool,
the batch sizes, the paths. You should not have to change anything.

### What it does

Every distillation stage in this study fixes one teacher for the whole training
set. The teachers are not uniformly better or worse than each other, so Stage M
hands each training image to whichever teacher is best **on that image** and
distils from the routed signal.

The routing is feasible because it happens at **training time only**, where the
ground-truth mask is already in the batch. The router never runs at test time —
only the student does — so there is no generalisation gap in the router.

### Two things to be honest about, before you see the numbers

**Report Dice, not misses.** The gate's miss clause passes, but it is an
artefact: SegFormer-B2 alone has 1 complete miss on validation, exactly as many
as the whole pool, and that miss is a single unlabelled image that nobody finds.
The clause reads PASS only because it is compared against B5, which has 3. The
pool recovers nothing. Pre-register Dice.

**This is not a fairness method.** It began as one, but every teacher's complete
misses are on **light** skin — on 55 dark-skin test images not one teacher misses
anything. There is no dark-skin failure to route around. Call it multi-teacher
distillation, and report the fairness finding separately: *the models do not have
the dark-skin gap people assume.*

### Where results go

`STAGE_M_RESULTS/` at the bundle root — tables at the top level, checkpoints in
`STAGE_M_RESULTS/runs/`. Nothing touches `results/`, `FINAL_RESULT/` or
`_work/runs/`.
""")

# ─────────────────────────────────────────────────────────────────────────────
md(r"""
## §0 — Memory allocator

**Must be the first cell executed, before anything imports torch.**
`PYTORCH_CUDA_ALLOC_CONF` is read when the CUDA allocator initialises; setting it
later has no effect at all. The first OOM had ~1 GB sitting in reserved-but-
unallocated fragments, and this is what reclaims it.

If you have already imported torch in this kernel, restart it now.
""")

code(r"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
assert "torch" not in sys.modules, (
    "torch is already imported, so the allocator setting above did nothing. "
    "Restart the kernel and run this cell first.")
print("allocator: expandable_segments:True  (set before torch import)")
""")

# ─────────────────────────────────────────────────────────────────────────────
md(r"""
## §1 — Configuration

Every value here is one we established by running it. The comments say which.
""")

code(r"""
# ── where things are ─────────────────────────────────────────────────────────
BUNDLE_ROOT = None       # None = auto-detect from this notebook's location
WORK_DIR    = None       # None = <bundle>/_work
EXTRA_RUNS  = "/scratch/tbommawa/bruise_work/runs"   # verified: finds all b2kd seeds
RESULTS_DIR = None       # None = <bundle>/STAGE_M_RESULTS

# ── the teacher pool ─────────────────────────────────────────────────────────
# U-Net dropped: its drop-one marginal was +0.0055 against +0.011..+0.022 for the
# others, it won fewest images (14.9%), and removing it cost exactly its marginal
# while DOUBLING DeepLab's -- the two ResNet-50 models were covering for each
# other. Three complementary architectures, no redundant member.
POOL = ("segformer_b2_teacher", "segformer_b5_teacher", "deeplabv3plus_r50")
STUDENT_REF = "segformer_b0_direct"     # scored alongside for the headroom line
SEED = 0                                # the seed the GATE reads

# ── the method ───────────────────────────────────────────────────────────────
RUN_METHOD = True        # §9's guard still re-checks the gate; this alone authorises nothing
ARMS  = ("segformer_b0_mtkd", "lraspp_mobilenetv3_mtkd")
SEEDS = (0, 1, 2)
BETA  = 8.0              # 0 = uniform ensemble, inf = hard argmax
GATE_LO, GATE_HI = 0.10, 0.50     # inherited from Stage H, deliberately unchanged

# ── memory ───────────────────────────────────────────────────────────────────
# Both of these exist because of the first OOM and neither changes a number.
MAX_MICRO_BATCH = 16     # -> 16 x 4 accumulation; effective batch stays 64
TEACHER_CHUNK   = 4      # teacher forwards sub-batched; frozen BN, identical output

# ── the gate ─────────────────────────────────────────────────────────────────
BOOT_REPS = 5000
FORCE_RESCORE = False    # the cached matrix may hold MORE teachers than POOL; the
                         # gate selects the POOL columns, so a re-score is not needed
""")

# ─────────────────────────────────────────────────────────────────────────────
md(r"""
## §2 — Environment and the two identities

`self_test` asserts that the routed loss reduces **exactly** to the losses that
already exist, in the two limits the contrast depends on:

- `K = 1` → the Stage H reliability-gated loss, bit-for-bit
- `beta = 0` → the uniform teacher ensemble

If either fails, `*_mtkd` vs its control is confounded by the loss itself and
nothing below means what it says — so it raises rather than warns.
""")

code(r"""
import json, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

import bruisekit.paths as P
from bruisekit import loaders as L, registry as REG, report as R
from bruisekit import multiteacher as MT

env = P.setup(BUNDLE_ROOT, work=WORK_DIR, extra_runs=EXTRA_RUNS)
print(env.describe())

STAGE_M = Path(RESULTS_DIR) if RESULTS_DIR else env.root / "STAGE_M_RESULTS"
STAGE_M_RUNS = STAGE_M / "runs"
STAGE_M_RUNS.mkdir(parents=True, exist_ok=True)
print(f"\nstage M results -> {STAGE_M}")
print(f"stage M runs    -> {STAGE_M_RUNS}")
print("  (nothing is written to results/, FINAL_RESULT/ or _work/runs/)")

print("\ncheckpoint search path:")
for i, r in enumerate(env.run_roots, 1):
    print(f"  {i}. {r}{'' if r.exists() else '   (absent)'}")

print()
if not MT.self_test():
    raise RuntimeError(
        "multiteacher.self_test failed: the routed loss no longer reduces to the "
        "single-teacher losses. Every *_mtkd contrast below would be confounded "
        "by the loss itself. Fix before continuing.")
""")

# ─────────────────────────────────────────────────────────────────────────────
md(r"""
## §3 — Data and the shared recipe

Identical to the unified notebook's. An arm that quietly changed the recipe would
not be comparable to the control it is scored against.
""")

code(r"""
MAN = {s: pd.read_csv(env.manifests / f"{s}.csv") for s in ("train", "val", "test")}
META = MAN["test"]
for s, d in MAN.items():
    print(f"  {s:<6} {len(d):>4} images  {d.subject.nunique():>3} subjects")

for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
    overlap = set(MAN[a].subject) & set(MAN[b].subject)
    if overlap:
        raise RuntimeError(f"{a}/{b} share subjects: {sorted(overlap)[:5]}")
print("  no subject appears in two splits")

MAN640 = L.build_cache640(env, MAN)

CFG = dict(
    img_size=640, epochs=100, patience=15,
    batch_mode="per_model", effective_batch=8, max_probe_batch=64, vram_target=0.75,
    backbone_lr=6e-5, head_lr=6e-4, betas=(0.9, 0.999), weight_decay=0.01,
    warmup_fraction=0.01, poly_power=1.0, gradient_clip=1.0, amp=True,
    workers=0, eval_batch=8, eval_batch_cpu=2,
    segformer_alpha=0.6, efficient_alpha=0.6, aux_weight=0.4,
    smp_encoder="resnet50", smp_micro_batch=16, efficient_micro_batch=16,
    cut_min=-6.0, cut_max=6.0, cut_steps=481,
    drive_sync_every=2, n_boot=2000, n_boot_final=10_000,
)
""")

# ─────────────────────────────────────────────────────────────────────────────
md(r"""
## §4 — Resolve the pool

Fails here in seconds if a checkpoint is unreachable, rather than three hours into
a run. Read the `source_root` column: teachers resolving to `checkpoints/` is
**correct** — those are the published weights. The controls must resolve to
`EXTRA_RUNS`, and §9 checks that separately.
""")

code(r"""
reg = REG.Registry(env, allow_training=False).scan()
MT.register_specs()

runs = MT.resolve_teachers(env, reg, POOL, SEED)
student_run = reg.get(f"{STUDENT_REF}__seed{SEED}")
if student_run is None or student_run.weights is None:
    warnings.warn(f"{STUDENT_REF}__seed{SEED} not found; the headroom line is omitted")
    student_run = None

rows = [{"family": r.family, "seed": r.seed,
         "role": "teacher" if r in runs else "student",
         "layout": getattr(r, "layout", "?"),
         "source_root": str(getattr(r, "source_root", Path(r.weights).parent.parent)),
         "weights": Path(r.weights).name}
        for r in runs + ([student_run] if student_run else [])]
display(pd.DataFrame(rows))

# The controls live in EXTRA_RUNS, not in the bundle. Without them the arms train
# for hours and §13 has nothing to compare against -- silently, as a skipped row.
missing_controls = [MT.CONTROL_FOR[a] for a in ARMS
                    if not any(r.family == MT.CONTROL_FOR[a] for r in reg.usable())]
if missing_controls:
    raise FileNotFoundError(
        f"controls not found: {missing_controls}. These live in EXTRA_RUNS "
        f"({EXTRA_RUNS}); check the path exists and re-run §2. Training without a "
        f"control produces a number with nothing to compare it to.")
print(f"  controls present for all {len(ARMS)} arm(s)")
""")

# ─────────────────────────────────────────────────────────────────────────────
md(r"""
## §5 — Score the pool on validation

One forward pass per teacher over the 134 val images, each at **its own**
val-fitted cut. Cached, so re-reading the gate is free.

Each teacher's cut was fitted *on* these images, so every teacher's val Dice is
mildly optimistic. That bias is shared by all of them and by the oracle, so it
cancels in the **gain** — the only quantity the gate reads.
""")

code(r"""
cache = STAGE_M / f"val_dice_matrix__seed{SEED}.csv"
if FORCE_RESCORE and cache.exists():
    cache.unlink()

t0 = time.time()
MATRIX = MT.val_dice_matrix(env, reg, CFG, MAN640, MAN["val"], pool=POOL, seed=SEED,
                            student=STUDENT_REF if student_run else None, cache=cache)

# gt_positive_pixels is a property of the LABEL and comes from the mask cache, not
# from a re-score, when an older cached matrix does not carry it.
if "gt_positive_pixels" not in MATRIX.columns:
    MATRIX = MT.backfill_gt_pixels(MATRIX, env, "val")
    MATRIX.to_csv(cache, index=False)

print(f"\n  {len(MATRIX)} val images in {time.time() - t0:.0f}s")
display(MATRIX.head())
""")

# ─────────────────────────────────────────────────────────────────────────────
md(r"""
## §6 — THE GATE

Three clauses, evaluated and printed separately:

1. **oracle-gain CI clears zero** — is there complementarity at all
2. **projected student gain clears the margin** — *the clause Stage C did not
   have.* Stage C's gate opened on +0.0258 of oracle gain; the student realised
   +0.0068, a transfer rate of 0.264. This projects through that measured rate and
   requires the projection, not the oracle, to clear one Dice point.
3. **miss endpoint** — reported, and **not** to be believed (see the header).

`drop-one marginal` decides pool membership: a teacher whose marginal is ~0 adds
nothing the rest of the pool does not already cover.
""")

code(r"""
GATE = MT.oracle_gate(MATRIX, pool=POOL,
                      student=STUDENT_REF if student_run else None,
                      reps=BOOT_REPS, seed=0)
print(MT.format_gate(GATE))

(STAGE_M / f"gate__seed{SEED}.json").write_text(json.dumps(GATE, indent=2))
print(f"\n  written -> {STAGE_M / f'gate__seed{SEED}.json'}")
print("\n  REMINDER: the miss clause is illusory -- B2 alone matches the pool's "
      "miss count.\n  Pre-register DICE.")
""")

# ─────────────────────────────────────────────────────────────────────────────
md(r"""
## §7 — Where the headroom is, and where it is not

`n_subjects` is a column, not a footnote: five ITA groups over 20 validation
subjects is three to five subjects per cell, and a gain computed from four
subjects is not a measurement.

**Read the skin-tone table expecting nothing** — that is the point of printing it.
""")

code(r"""
print("── by bruise size (quintiles of gt_positive_pixels) ──")
SIZE_STRAT = MT.stratified_oracle(MATRIX, pool=POOL, by="size")
display(SIZE_STRAT)

print("\n── by ITA skin-tone group ──")
ITA_STRAT = MT.stratified_oracle(MATRIX, pool=POOL, by="skin_tone_category")
display(ITA_STRAT)

SIZE_STRAT.to_csv(STAGE_M / f"stratified_size__seed{SEED}.csv", index=False)
ITA_STRAT.to_csv(STAGE_M / f"stratified_ita__seed{SEED}.csv", index=False)

wins = MATRIX[list(POOL)].to_numpy(float).argmax(axis=1)
print("\n── win counts ──")
for k, c in enumerate(POOL):
    print(f"  {c:<26} {int((wins == k).sum()):>4} / {len(MATRIX)}")
""")

# ─────────────────────────────────────────────────────────────────────────────
md(r"""
---

## §8 — Read the verdict before continuing

Everything above cost no training. If §6 says **DO NOT RUN**, stop here: a
four-teacher pool with no headroom the student could inherit, established in one
validation pass instead of a grid, is a reportable result.

If it says **RUN**, keep going. Roughly 14 GPU-hours for 2 arms × 3 seeds.
`train_run` resumes from `DONE.json` / `resume.pt`, so a dropped connection costs
only the epochs since the last sync.

---
""")

md(r"""
## §9 — The guard

Three conditions: the gate ran, it opened, and the controls are reachable. A flag
is not authorisation to spend GPU-hours.
""")

code(r"""
gate_file = STAGE_M / f"gate__seed{SEED}.json"
if not RUN_METHOD:
    raise RuntimeError("RUN_METHOD is False -- nothing below has run.")
if not gate_file.exists():
    raise RuntimeError(f"no gate at {gate_file}: run §5-§6 first. The method is "
                       f"authorised by the pre-test, not by a flag.")

_g = json.loads(gate_file.read_text())
if not _g["GATE_any"]:
    raise RuntimeError(
        "The gate did NOT open:\n"
        f"  oracle gain {_g['oracle_gain_over_best_single']:+.4f} "
        f"CI {_g['oracle_gain_ci95']}\n"
        f"  projected student gain {_g['projected_student_gain']:+.4f} "
        f"vs margin {_g['margin']:+.4f}\n"
        "Overriding this is a decision to spend GPU hours the pre-test says are "
        "wasted. If you mean to, do it deliberately and say so in the writeup.")

DICE_HYPOTHESIS = bool(_g["GATE_run_method"])
print("gate opened.")
print(f"  teachers            {_g['teachers']}")
print(f"  oracle gain         {_g['oracle_gain_over_best_single']:+.4f} "
      f"CI {[round(x, 4) for x in _g['oracle_gain_ci95']]}")
print(f"  projected student   {_g['projected_student_gain']:+.4f} "
      f"vs margin {_g['margin']:+.4f}")
print(f"  arms                {ARMS} x seeds {SEEDS}, beta = {BETA}")
print(f"  estimated           "
      f"{sum(MT.COST_HOURS.get(a, 2.5) for a in ARMS) * len(SEEDS):.1f} GPU-hours")
print("\n  Dice is the pre-registered endpoint. The miss clause is illusory.")
""")

# ─────────────────────────────────────────────────────────────────────────────
md(r"""
## §10 — Install the shims

Four patches, all on seams Stage F and Stage H already use, all preserving
`_original` so installing twice does not wrap the wrapper. Stage M's go **last**,
so every one of them falls through for any family that is not a Stage M arm.

| patch | what it does |
|---|---|
| `efficient_models.install_efficient_shim` | builds the mobile student **with** its ImageNet backbone |
| `MT.install_teacher_shim` | `load_teacher` returns a `[B, K, H, W]` stack |
| `MT.install_loss_shim` | `DistillLoss` routes over that stack |
| `MT.install_batch_shim` | pins each arm to its control's **effective** batch |

**The batch pin, and the one difference it leaves.** The control trained at
micro-batch 64 with one teacher resident; three teachers do not fit alongside it
on a 40 GB slice. `MAX_MICRO_BATCH = 16` gives 16 × 4 accumulation — same
effective batch, same optimizer-step count, same LR schedule. What remains is that
SegFormer's decode-head BatchNorm now normalises over 16 images instead of 64.
That is real, it is small, and it belongs in the limitations.
""")

code(r"""
import bruisekit.efficient_models as EM
import bruisekit.engine as ENG

EM.install_efficient_shim(env)
MT.register_specs()

# Fail on a missing or unfittable temperature now, not three hours in.
TEMPS = MT.warm_calibration(env, reg, CFG, MAN640, pool=POOL, seeds=SEEDS)
print()
for k, v in TEMPS.items():
    print(f"  {k:<34} T = {v:.4f}")

PINNED = {}
for a in ARMS:
    if MT.STUDENT_ARCH[a] == "segformer":
        PINNED[a] = MT.control_batch(env, reg, a, SEEDS[0], max_micro=MAX_MICRO_BATCH)
        eff = PINNED[a][0] * PINNED[a][1]
        print(f"\n  {a}: control {MT.CONTROL_FOR[a]} -> pinned {PINNED[a][0]} x "
              f"{PINNED[a][1]} (effective {eff}, unchanged)")
    else:
        print(f"\n  {a}: efficient_micro_batch=16 for every efficient family, so it "
              f"already matches {MT.CONTROL_FOR[a]}")

print()
MT.install_teacher_shim(env, reg, CFG, pool=POOL, seed_mode="same",
                        teacher_chunk=TEACHER_CHUNK)
MT.install_loss_shim(beta=BETA, gate_lo=GATE_LO, gate_hi=GATE_HI)
if PINNED:
    MT.install_batch_shim(env, PINNED)
""")

# ─────────────────────────────────────────────────────────────────────────────
md(r"""
## §11 — Train

Runs go to `STAGE_M_RESULTS/runs/`, not `_work/runs/`. `MT.arm()` scopes the
active arm as a context manager, so an exception cannot leak it and apply routed
KD to the next family trained.

**Watch the router entropy.** Uniform for 3 teachers is `log 3 = 1.10`; collapsed
onto one teacher is 0. An arm whose router stayed uniform is a teacher ensemble
and should be reported as one; an arm whose router collapsed is single-teacher KD
with the teacher chosen per image. That number decides what the method is called.
""")

code(r"""
from bruisekit.engine import train_run

TRAINED = []
for family in ARMS:
    spec = L.spec_for(family)
    alpha_key = ("efficient_alpha" if MT.STUDENT_ARCH[family] != "segformer"
                 else "segformer_alpha")
    run_cfg = {**CFG, "alpha": CFG[alpha_key]}
    for seed in SEEDS:
        run_id = f"{family}__seed{seed}"
        t0 = time.time()
        print(f"\n=== {run_id} ===", flush=True)
        with MT.arm(family):
            res = train_run(run_id, spec, seed, run_cfg, env.paths_for_models(),
                            MAN640, env.cache640, STAGE_M_RUNS, env.device)
            rs = MT.dump_router_stats(STAGE_M_RUNS / run_id)
        print(f"  -> {res.get('status', 'trained')} in {(time.time() - t0) / 60:.1f} min")
        if rs:
            w = ", ".join(f"{c.split('_')[0]}={v:.3f}"
                          for c, v in zip(POOL, rs["mean_routing_weight_per_teacher"]))
            print(f"     router weights: {w}")
            print(f"     entropy {rs['mean_routing_entropy']:.3f} "
                  f"(uniform={np.log(len(POOL)):.3f}, collapsed=0)  "
                  f"fused teacher Dice {rs['mean_fused_teacher_soft_dice']:.4f}")
        TRAINED.append(run_id)

print(f"\n{len(TRAINED)} runs in {STAGE_M_RUNS}")
""")

# ─────────────────────────────────────────────────────────────────────────────
md(r"""
## §12 — Operating point on val, then score test once

The same three steps every other stage uses: cache the val logits, sweep 481
cuts, take `select_cut`'s **tie band's lowest-miss cut** — not the argmax, which
fits val noise on a plateau this flat.
""")

code(r"""
import torch
from bruisekit.data import make_loader
from bruisekit.evaluate import evaluate_at_cut
from bruisekit.sweep import cache_logits, select_cut, sweep_cuts

CUTS = np.linspace(CFG["cut_min"], CFG["cut_max"], CFG["cut_steps"])
m_rows = []
for run_id in TRAINED:
    rd = STAGE_M_RUNS / run_id
    family = run_id.rsplit("__seed", 1)[0]
    seed = int(run_id.rsplit("seed", 1)[1])

    if MT.STUDENT_ARCH[family] == "segformer":
        model = ENG.build_model("segformer", "b0", env.paths_for_models())
    else:
        model = EM.build_with_pretrained(env, family, 1, verbose=False)
    model.load_state_dict(torch.load(str(rd / "best.pt"), map_location=env.device,
                                     weights_only=True))
    model.to(env.device).eval()

    logits, gts, _ = cache_logits(
        model, make_loader(MAN640["val"], env.cache640, CFG["img_size"],
                           CFG["eval_batch"], False, CFG["workers"], 0),
        env.device, CFG["amp"])
    grid = sweep_cuts(logits, gts, CUTS)
    sel = select_cut(grid)
    grid.to_csv(rd / "threshold_sweep.csv", index=False)
    (rd / "operating_point.json").write_text(json.dumps(sel, indent=2))

    tpi, summ = evaluate_at_cut(
        model, make_loader(MAN640["test"], env.cache640, CFG["img_size"],
                           CFG["eval_batch"], False, CFG["workers"], 0),
        env.device, sel["cut"], CFG["amp"])
    tpi.to_csv(rd / "test_per_image.csv", index=False)
    m_rows.append({"run_id": run_id, "model": family, "seed": seed,
                   "cut": sel["cut"], **summ})
    print(f"  {run_id:<34} cut {sel['cut']:+.3f} -> test Dice {summ['mean_dice']:.4f} "
          f"(miss {summ['complete_miss_rate'] * 100:.2f}%)")
    del model, logits, gts
    if str(env.device).startswith("cuda"):
        torch.cuda.empty_cache()

M_TEST = pd.DataFrame(m_rows)
M_TEST.to_csv(STAGE_M / "stage_m_test_per_seed.csv", index=False)
display(M_TEST)
""")

# ─────────────────────────────────────────────────────────────────────────────
md(r"""
## §13 — The contrasts

**Seed-matched**: arm seed *k* against its control seed *k*, so the pair differs
in one thing only — how the teacher signal is formed. Paired subject-level
bootstrap on three endpoints at once.

Read `verdict` against the one-Dice-point margin. `delta_miss_rate` is reported
for completeness and is **not** the pre-registered endpoint.
""")

code(r"""
from bruisekit import significance as SIG

def stage_m_table(run_id):
    p = STAGE_M_RUNS / run_id / "test_per_image.csv"
    return R.normalize(pd.read_csv(p), META) if p.exists() else None

CONTRASTS = []
for family in ARMS:
    control = MT.CONTROL_FOR[family]
    for seed in SEEDS:
        a = stage_m_table(f"{family}__seed{seed}")
        b = R.load_per_image(env, reg, f"{control}__seed{seed}", META)
        if a is None or b is None:
            print(f"  SKIP {family} seed {seed}: "
                  f"missing {'arm' if a is None else 'control'} table")
            continue
        row = SIG.paired_contrast_multi(a, b, f"{family}__seed{seed}",
                                        f"{control}__seed{seed}",
                                        n_boot=CFG["n_boot_final"])
        row["arm"], row["control"], row["seed"] = family, control, seed
        CONTRASTS.append(row)

CON = pd.DataFrame(CONTRASTS)
if len(CON):
    display(CON[["arm", "control", "seed", "delta_dice", "lo", "hi", "verdict",
                 "p_a_better", "delta_miss_rate", "miss_lo", "miss_hi"]])
    CON.to_csv(STAGE_M / "stage_m_contrasts.csv", index=False)

    print("\n── across seeds, per arm ──")
    for family, g in CON.groupby("arm"):
        d = g["delta_dice"]
        signs = "all positive" if (d > 0).all() else (
                "all negative" if (d < 0).all() else "MIXED SIGN across seeds")
        print(f"  {family:<28} mean delta {d.mean():+.4f}  "
              f"(seeds: {', '.join(f'{v:+.4f}' for v in d)})  -> {signs}")
    print("\n  Three seeds of the same sign is suggestive, not significant. A "
          "delta smaller\n  than ~0.05 is inside the annotation noise floor "
          "(handbook §1) whatever its sign.")
else:
    print("  no contrasts computed -- check the SKIP lines above")
""")

# ─────────────────────────────────────────────────────────────────────────────
md(r"""
## §14 — Misses and fairness

`report.normalize` recomputes `complete_miss` from `dice`, the `dice == 0`
definition this project publishes.

24 models were tested for a skin-tone effect in the August sweep with no
multiplicity control; at Bonferroni α = 0.002 not one of the five "significant"
results survives. Read `significant=True` as a flag to investigate, not a finding.
""")

code(r"""
from bruisekit.evaluate import fairness_analysis

miss_rows, fair_rows = [], []
for family in ARMS:
    for seed in SEEDS:
        pi = stage_m_table(f"{family}__seed{seed}")
        if pi is None:
            continue
        miss_rows.append({"model": family, "seed": seed, "n": len(pi),
                          "misses_dice0": int((pi.dice == 0).sum()),
                          "miss_pct": float((pi.dice == 0).mean() * 100),
                          "median_dice": float(pi.dice.median()),
                          "mean_dice": float(pi.dice.mean())})
        fair_rows.append({**fairness_analysis(pi, META, family)["stats"], "seed": seed})

for family in {MT.CONTROL_FOR[a] for a in ARMS}:
    for seed in SEEDS:
        pi = R.load_per_image(env, reg, f"{family}__seed{seed}", META)
        if pi is None:
            continue
        miss_rows.append({"model": f"{family} (control)", "seed": seed, "n": len(pi),
                          "misses_dice0": int((pi.dice == 0).sum()),
                          "miss_pct": float((pi.dice == 0).mean() * 100),
                          "median_dice": float(pi.dice.median()),
                          "mean_dice": float(pi.dice.mean())})
        fair_rows.append({**fairness_analysis(pi, META, f"{family} (control)")["stats"],
                          "seed": seed})

MISS = pd.DataFrame(miss_rows).sort_values(["model", "seed"])
FAIR = pd.DataFrame(fair_rows)
display(MISS)
display(FAIR[["model", "seed", "fairness_gap", "best_group", "worst_group",
              "max_miss_rate_gap", "kruskal_p", "significant"]])
MISS.to_csv(STAGE_M / "stage_m_misses.csv", index=False)
FAIR.to_csv(STAGE_M / "stage_m_fairness.csv", index=False)
""")

# ─────────────────────────────────────────────────────────────────────────────
md(r"""
## §15 — What the router actually did, and what was written

An arm whose router stayed uniform and an arm whose router collapsed are
**different experiments**, and Dice alone cannot tell them apart.
""")

code(r"""
diag = []
for run_id in TRAINED:
    p = STAGE_M_RUNS / run_id / "router_stats.json"
    if not p.exists():
        continue
    s = json.loads(p.read_text())
    row = {"run_id": run_id, "beta": s["beta"],
           "entropy": s["mean_routing_entropy"],
           "entropy_uniform": float(np.log(len(POOL))),
           "coverage": s["mean_coverage"], "alpha_eff": s["mean_alpha_effective"],
           "fused_teacher_dice": s["mean_fused_teacher_soft_dice"]}
    row.update({f"w_{c}": v for c, v in
                zip(POOL, s["mean_routing_weight_per_teacher"])})
    diag.append(row)

if diag:
    DIAG = pd.DataFrame(diag)
    display(DIAG)
    DIAG.to_csv(STAGE_M / "stage_m_router_diagnostics.csv", index=False)
    e, u = DIAG["entropy"].mean(), float(np.log(len(POOL)))
    if e > 0.95 * u:
        print(f"\n  Router stayed ~UNIFORM ({e:.3f} vs {u:.3f}). This arm is a "
              f"teacher ENSEMBLE;\n  report it as one, not as routing.")
    elif e < 0.15 * u:
        print(f"\n  Router COLLAPSED ({e:.3f} vs {u:.3f}). This is single-teacher "
              f"KD with the\n  teacher chosen per image -- say which teacher won.")
    else:
        print(f"\n  Router is genuinely mixing ({e:.3f} of a possible {u:.3f}).")
else:
    print("  no router_stats.json -- every run was skipped (DONE.json present)")

print(f"\n── everything Stage M wrote, in {STAGE_M} ──")
for f in sorted(STAGE_M.rglob("*")):
    if f.is_file():
        print(f"  {f.relative_to(STAGE_M)}")
""")


def main() -> int:
    nb = {
        "cells": [
            {"cell_type": kind, "metadata": {},
             "source": src.splitlines(keepends=True),
             **({"execution_count": None, "outputs": []} if kind == "code" else {})}
            for kind, src in CELLS
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text(json.dumps(nb, indent=1), encoding="utf-8")

    n_code = sum(1 for k, _ in CELLS if k == "code")
    for i, (kind, src) in enumerate(CELLS):
        if kind == "code":
            compile(src, f"cell{i}", "exec")          # a cell that will not parse
    back = json.loads(DST.read_text(encoding="utf-8"))
    assert len(back["cells"]) == len(CELLS)

    print(f"wrote {DST}")
    print(f"  {len(CELLS)} cells ({n_code} code, {len(CELLS) - n_code} markdown)")
    print(f"  {DST.stat().st_size / 1024:.1f} KB")
    print("  every code cell compiles; JSON round-trip ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
