"""The train-or-skip brain: decide, per run, what this session can actually do.

THE PROBLEM
------------
The bundle ships 27 trained runs across three stages, but not all 28 that the
experiment grid describes, and one stage (nnU-Net) has neither weights nor
results. A notebook that just calls `train()` and relies on each trainer's own
DONE.json would work -- but you would not find out that something was about to
train for eight hours until it had already started, and you would not find out
that nnU-Net was missing entirely until its cell raised.

So the decision is hoisted out of the trainers and made once, up front, for every
run, and printed as a plan before a single gradient is computed.

THE THREE TIERS
----------------
Each run resolves to exactly one of:

  WEIGHTS   a usable checkpoint exists -> load it; re-run inference if you want
            fresh per-image numbers, or reuse the cached ones. Nothing trains.
  RESULTS   no checkpoint, but this run's metrics were recorded -> report from the
            CSV, clearly labelled as cached. Nothing trains. This tier is what
            lets the whole analysis reproduce on a laptop with no GPU.
  MISSING   neither -> the only case where training is even proposed.

WHY "RESULTS" IS A REAL TIER AND NOT A FUDGE
---------------------------------------------
A cached metric is not a substitute for a model -- you cannot make a new
prediction with it. It IS a complete substitute for a number in a table, because
it is the same number the model would produce: these CSVs were written by the
same code, from the same weights, on the same 185 test images. The tier is
labelled everywhere it surfaces so a reader always knows which is which, and
`Registry.report()` prints the counts per tier so nothing hides.

FAIL LOUD, NEVER SILENTLY
--------------------------
The one behaviour this module refuses is quiet substitution. If a run is MISSING
and training is disabled, it stays MISSING in the table and in every downstream
table it would have fed. It is never dropped, never back-filled from a
neighbouring seed, and never averaged away.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

WEIGHTS, RESULTS, MISSING = "WEIGHTS", "RESULTS", "MISSING"

# Rough single-run training cost on an A100, used only to make the plan's
# "you are about to spend N hours" line honest. Measured from the epochs_trained
# recorded in the shipped DONE.json files, rounded up.
COST_HOURS = {
    "segformer_b2_teacher": 1.1, "segformer_b0_direct": 0.7, "segformer_b0_distilled": 0.9,
    "yolo_sem_direct": 0.6, "yolo_sem_distilled": 0.8,
    "unet_r50": 0.8, "deeplabv3plus_r50": 0.8, "nnunet": 8.0,
    "distill_arm": 1.5, "segformer_b5": 3.0,
    # Stage E -- all four are 1-3 M parameters, so an epoch is cheap; the wall
    # clock is dominated by data loading rather than by the model.
    "ppmobileseg_tiny": 0.5, "topformer_tiny": 0.5,
    "lraspp_mobilenetv3": 0.5, "fastscnn": 0.4,
    # Stage F -- same students, plus a frozen ResNet-50 teacher forward on every
    # step. The teacher is ~10x the student, so the step cost is the teacher's.
    "fastscnn_distilled": 0.7, "lraspp_mobilenetv3_distilled": 0.8,
    "topformer_tiny_distilled": 0.8, "ppmobileseg_tiny_distilled": 0.9,
    # Stage H -- a frozen SegFormer-B2 (27.4 M) forward on every step, so the step
    # cost is the teacher's again and is a little above the DeepLabV3+ arms'. The
    # gated arms cost the same as the plain ones: the gate is four elementwise ops
    # on a tensor the loss already has, under no_grad.
    "segformer_b0_rgkd": 0.9,
    "fastscnn_rgkd": 0.8, "lraspp_mobilenetv3_rgkd": 0.9,
    "topformer_tiny_rgkd": 0.9, "ppmobileseg_tiny_rgkd": 1.0,
    "fastscnn_b2kd": 0.8, "lraspp_mobilenetv3_b2kd": 0.9,
    "topformer_tiny_b2kd": 0.9, "ppmobileseg_tiny_b2kd": 1.0,
    # Stage Y -- YOLO26-LARGE. Scaled from the nano arms by parameter count
    # (~1.6 M -> ~28 M, so ~17x the FLOPs per step) and then damped, because
    # native Ultralytics training is substantially dataloader-bound at 697
    # images: mosaic, letterbox and augmentation run on the CPU regardless of
    # how big the network is. Treat as an estimate, not a measurement -- unlike
    # every other row here, no yolo26l run has finished yet to calibrate it.
    "yolo_sem_l_direct": 3.5, "yolo_sem_l_distilled": 4.0,
}

# Stage E families, in the order they should appear in tables. The distilled arms
# are scanned alongside the direct ones so that each is reported MISSING until it
# is trained, rather than being absent from the plan entirely -- the registry's
# whole point (§10) is that a run you have not done is a row, not a silence.
EFFICIENT_FAMILIES = ("ppmobileseg_tiny", "topformer_tiny",
                      "lraspp_mobilenetv3", "fastscnn",
                      "fastscnn_distilled", "lraspp_mobilenetv3_distilled",
                      "topformer_tiny_distilled", "ppmobileseg_tiny_distilled")

# Stage H families, in table order: the gated arms first, then the teacher-matched
# controls they must be read against. Scanned as their own stage rather than
# folded into E so that Stage E's counts, plan lines and tables keep meaning what
# they meant before this stage existed -- an existing table that silently grows
# twelve rows is indistinguishable from a bug.
#
# `segformer_b0_rgkd` is Stage H even though its student is a Stage A model: the
# stage letter names the EXPERIMENT, not the architecture.
#
# NO YOLO ARM IS REGISTERED. `yolo_sem_rgkd` is implemented and tested but left
# out of every table (see reliability_kd's module docstring for the three lines
# that re-enable it, and for what the study gives up meanwhile). One practical
# consequence worth knowing: with Stage A's YOLO runs all at WEIGHTS tier and no
# YOLO run in Stage H, `train_missing` never sees kind "yolo", so a session that
# trains E and H never needs Ultralytics installed at all.
STAGE_H_FAMILIES = (
    "segformer_b0_rgkd",
    "lraspp_mobilenetv3_rgkd", "fastscnn_rgkd",
    "topformer_tiny_rgkd", "ppmobileseg_tiny_rgkd",
    "lraspp_mobilenetv3_b2kd", "fastscnn_b2kd",
    "topformer_tiny_b2kd", "ppmobileseg_tiny_b2kd",
)

# Which loader each Stage H family needs. Everything not named here is "efficient".
# The "yolo" branch in _scan_stage_h is retained for the same reason as the guard
# in reliability_kd.train_arm: re-enabling the arm should not also require
# re-deriving how it is scanned.
STAGE_H_KIND = {"segformer_b0_rgkd": "segformer"}

# ── Stage Y · YOLO26-large ───────────────────────────────────────────────────
# The same native-Ultralytics recipe as Stage A's YOLO arms, on the LARGE
# backbone instead of nano. Its own stage letter for the reason Stage H got one:
# Stage A is "the five headline models", it is quoted with that count in the
# handbook and in the paper, and a table that silently grows two rows is
# indistinguishable from a bug.
#
# WHY LARGE AT ALL. yolo26n is 1.63 M parameters and is the fastest model in the
# study, but it also has the worst complete-miss rate of the reporting models
# (6.5% native-argmax at its best seed against 0.0-0.5% for the SegFormers).
# Miss containment, not Dice, is the endpoint this study is judged on (§D3), so
# "the same architecture with enough capacity to stop missing" is the obvious
# question the nano arms cannot answer.
#
# SCORED BY NATIVE ARGMAX ONLY. The custom /255 path exists for the nano arms
# because Stage A needed a SegFormer-comparable geometry; it is not re-derived
# here. Native argmax is parameter-free -- there is no threshold to fit on val,
# so a Stage Y run needs `best.pt` and nothing else, exactly like Stage A's YOLO.
# ONE ARM. `yolo_sem_l_distilled` has a FAMILY_SPEC and a cost estimate but is
# deliberately NOT registered, exactly as `yolo_sem_rgkd` is not -- so nothing
# schedules it and `RUN_STAGES = "...Y"` cannot quietly cost an extra ~4 GPU-hours
# for an arm nobody asked for. Stage Y asks one question ("does capacity fix
# yolo26n's miss rate?") and one direct arm answers it; a distilled arm answers a
# different question and should be turned on when that question is being asked:
#
#     STAGE_Y_FAMILIES = ("yolo_sem_l_direct", "yolo_sem_l_distilled")
STAGE_Y_FAMILIES = ("yolo_sem_l_direct",)

# ── Stage T · the teacher store ──────────────────────────────────────────────
# Seedless, val-selected teacher checkpoints under checkpoints/distill/teachers/.
# Never trained by this notebook -- they are inputs, and a MISSING one is a
# statement about the bundle rather than a job about to start.
TEACHER_FAMILIES = ("segformer_b2_teacher", "segformer_b5_teacher",
                    "segformer_b0_distilled")


# ── the three on-disk layouts ────────────────────────────────────────────────
# Checkpoints in this study were written by three different pipelines over about
# a year, and they do not agree on filenames, on where the threshold lives, or on
# whether a run directory carries a seed at all. Each layout is described once,
# here, so that adding a fourth means adding a row rather than editing every
# _scan_stage_* method.
#
#   STANDARD   <root>/<family>__seed<N>/best.pt        + operating_point.json
#              Stages A, B, E, F, H, and anything trained by this notebook.
#              `operating_point.json` carries `cut`, a LOGIT threshold.
#
#   YOLO       <root>/<family>__seed<N>/ultralytics_runs/train/weights/best.pt
#              No threshold file and none needed -- native argmax is
#              parameter-free, so weights alone make the run usable.
#
#   TEACHER    <root>/<family>/best_model.pt           + threshold.json
#              The B2/B5/B0-distilled store. NO SEED IN THE PATH, and the
#              threshold is a PROBABILITY, not a logit. B5's own
#              seed_selection.csv records its selected seed as 123 -- so a
#              registry that only ever asks for seeds 0/1/2 could not see it
#              even if the filenames matched, which is why it has been invisible.
_LAYOUTS = ("standard", "yolo", "teacher")


def _probe_layout(run_dir: Path, layout: str):
    """Return (weights, threshold_file) for `layout` under `run_dir`, or None.

    A run counts only when BOTH parts a given layout needs are present. A
    checkpoint without its threshold is not a usable model for a logit-thresholded
    family -- its test score is undefined without the cut fitted on validation --
    and half-answering here is what produces a WEIGHTS-tier run that fails at load.
    """
    if layout == "standard":
        w, t = run_dir / "best.pt", run_dir / "operating_point.json"
        return (w, t) if w.exists() and t.exists() else None
    if layout == "yolo":
        w = run_dir / "ultralytics_runs" / "train" / "weights" / "best.pt"
        return (w, None) if w.exists() else None      # argmax needs no threshold
    if layout == "teacher":
        w = run_dir / "best_model.pt"
        if not w.exists():
            return None
        # The threshold is OPTIONAL here, unlike the standard layout, and the two
        # dialects coexist inside this one store: B5 ships threshold.json (a
        # probability), B2 ships operating_point.json (a logit), B0-distilled
        # ships neither. Absent is still usable -- a teacher is consumed as SOFT
        # PROBABILITIES by the KD loss and never thresholded, so demanding a cut
        # would report the fairest teacher in the study as unusable for the one
        # job it is actually for. `load_model` raises if a cut is later needed
        # and none was found.
        for cand in ("threshold.json", "operating_point.json"):
            if (run_dir / cand).exists():
                return w, run_dir / cand
        return w, None
    raise ValueError(f"unknown layout {layout!r}")


def find_checkpoint(roots, family: str, seed=None, layouts=_LAYOUTS):
    """Search every root x every layout for one family's checkpoint.

    Returns (weights, threshold_file, root, layout, dir) or None.

    Roots are tried in order and the first hit wins, so `env.runs` beating
    `checkpoints/final/` is a property of the search path rather than of a
    conditional somewhere. Within a root the layouts are cheap `exists()` calls.

    A seeded directory is preferred over a seedless one when a seed is given: the
    teacher store's `segformer_b2_teacher/` and Stage A's
    `segformer_b2_teacher__seed0/` are DIFFERENT checkpoints of the same family,
    and quietly serving one where the other was asked for is exactly the class of
    substitution this function exists to make visible.
    """
    names = []
    if seed is not None:
        names.append(f"{family}__seed{seed}")
    names.append(family)
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for name in names:
            rd = root / name
            if not rd.is_dir():
                continue
            for layout in layouts:
                hit = _probe_layout(rd, layout)
                if hit:
                    return hit[0], hit[1], root, layout, rd
    return None


def read_cut(threshold_file) -> float | None:
    """The logit cut, from either threshold-file dialect.

    `operating_point.json` stores `cut` (a logit). `threshold.json` stores
    `threshold` (a probability). They are the same operating point expressed two
    ways -- sigmoid(cut) == threshold -- and mixing them up silently shifts the
    decision boundary rather than raising, which on this data moves the
    complete-miss rate by several points.
    """
    if threshold_file is None:
        return None
    d = json.loads(Path(threshold_file).read_text())
    if "cut" in d:
        return float(d["cut"])
    if "threshold" in d:
        import math
        p = min(max(float(d["threshold"]), 1e-6), 1 - 1e-6)
        return math.log(p / (1.0 - p))
    raise ValueError(f"{threshold_file} has neither 'cut' nor 'threshold'")


@dataclass
class Run:
    """One trainable unit and everything known about it.

    Attributes
    ----------
    run_id : unique key, e.g. "segformer_b0_distilled__seed1".
    stage : "A" | "B" | "C" -- which section of the study it belongs to.
    family : model name without the seed, used for cost lookup and grouping.
    seed : training seed, or None for single-run experiments.
    kind : "segformer" | "yolo" | "smp" | "nnunet" | "kd" -- selects the loader.
    weights : path to the checkpoint, if one exists.
    per_image : path to a per-image test CSV, if one exists.
    cached : row of summary metrics recovered from a results CSV, if any.
    tier : WEIGHTS | RESULTS | MISSING.
    note : short human explanation, shown in the plan table.
    """

    run_id: str
    stage: str
    family: str
    seed: int | None
    kind: str
    weights: Path | None = None
    per_image: Path | None = None
    cached: dict | None = None
    # PROVENANCE. Which of the search roots answered, which on-disk layout it
    # used, and where the threshold came from. Recorded because the failure this
    # study actually hit was not "checkpoint missing" -- it was "a different
    # checkpoint answered and nothing said so".
    source_root: Path | None = None
    layout: str | None = None
    threshold_file: Path | None = None
    tier: str = MISSING
    note: str = ""

    @property
    def cost_hours(self) -> float:
        return COST_HOURS.get(self.family, COST_HOURS.get(self.kind, 1.0))


def _first_existing(*paths: Path) -> Path | None:
    """Return the first path that exists, or None. Order encodes precedence."""
    for p in paths:
        if p is not None and Path(p).exists():
            return Path(p)
    return None


class Registry:
    """Scans the bundle once and answers "what can this session do?".

    Usage
    -----
        reg = Registry(env)
        reg.scan()
        reg.report()                  # the plan table
        reg.to_train()                # [] when everything is covered
        reg.get("segformer_b0_direct__seed0")

    The scan is pure filesystem inspection -- no torch, no CUDA, no model
    construction -- so it is fast and works identically on a CPU-only laptop.
    """

    def __init__(self, env, allow_training: bool = False, force: tuple = (),
                 efficient_seeds: tuple = (0, 1, 2), rgkd_seeds: tuple | None = None,
                 yolo_l_seeds: tuple | None = None):
        """
        Parameters
        ----------
        env : the `Env` from `bruisekit.paths.setup()`.
        allow_training : when False (the default) a MISSING run is reported and
            skipped rather than trained. Left False deliberately: an accidental
            Run All on a fresh bundle should cost seconds, not a GPU day.
        force : run_ids to retrain even if a checkpoint exists. Their fresh output
            goes to `env.runs`, never over the shipped checkpoints.
        efficient_seeds : which seeds Stage E is expected to have.

            This applies to Stage E ONLY, and that asymmetry is deliberate.
            Stages A-C are scanned at the seeds their shipped checkpoints were
            actually trained with (0, 1, 2); narrowing that would hide runs that
            exist, which is the opposite of this class's job. Stage E has no
            shipped runs at all, so its expected grid is a decision you make when
            you decide how much GPU time to spend -- one seed for a first look,
            three when you want a spread to report.
        rgkd_seeds : which seeds Stage H is expected to have. None means "the same
            as Stage E", which is the right default: a Stage H arm is only ever
            read against a Stage E or Stage F control at the same seed, and a
            contrast between a 3-seed arm and a 1-seed control is not a contrast.
            Given separately so a first look at the gate can be bought for a third
            of the GPU time without also shrinking Stage E.
        """
        self.env = env
        self.allow_training = allow_training
        self.force = set(force)
        self.efficient_seeds = tuple(efficient_seeds)
        self.rgkd_seeds = tuple(efficient_seeds if rgkd_seeds is None else rgkd_seeds)
        # Stage Y defaults to ONE seed, not three -- the only default in this
        # class that does. A yolo26l run is ~3.5 h against a mobile arm's ~0.5 h,
        # so the three-seed habit that costs 1.5 GPU-hours in Stage E costs 21
        # here. One seed answers the question Stage Y is asked ("does capacity
        # fix yolo26n's miss rate?"); three are needed only once the answer is
        # yes and you want to quote a spread for it.
        self.yolo_l_seeds = tuple((0,) if yolo_l_seeds is None else yolo_l_seeds)
        self.runs: dict[str, Run] = {}

    # ── scanning ─────────────────────────────────────────────────────────────
    def scan(self) -> "Registry":
        """Populate the registry. Idempotent; call again after training."""
        self.runs = {}
        self._scan_stage_a()
        self._scan_stage_b()
        self._scan_stage_c()
        self._scan_stage_e()
        self._scan_stage_h()
        self._scan_stage_y()
        self._scan_teachers()
        for r in self.runs.values():
            if r.run_id in self.force:
                r.tier = MISSING
                r.note = "forced retrain"
        return self

    def _add(self, r: Run) -> None:
        self.runs[r.run_id] = r

    @staticmethod
    def _attach(r: Run, hit) -> Path | None:
        """Record which root and layout answered, and return the weights path.

        Every scan goes through here so provenance cannot be forgotten in one
        branch and remembered in the others -- which is how a search path stops
        being an improvement and becomes a second way to substitute silently.
        """
        if hit is None:
            return None
        w, tf, root, layout, _rd = hit
        r.source_root, r.layout, r.threshold_file = root, layout, tf
        return w

    def _resolve(self, r: Run, weights: Path | None,
                 per_image: Path | None, cached: dict | None) -> Run:
        """Apply the three-tier rule to one run and attach the evidence."""
        r.weights, r.per_image, r.cached = weights, per_image, cached
        if weights is not None:
            r.tier = WEIGHTS
            r.note = "checkpoint"
        elif per_image is not None or cached is not None:
            r.tier = RESULTS
            r.note = "cached metrics" + ("" if per_image is not None else " (summary only)")
        else:
            r.tier = MISSING
            r.note = f"would train ~{r.cost_hours:.1f} h"
        self._add(r)
        return r

    def _scan_stage_a(self) -> None:
        """Stage A: the five headline models, three seeds each.

        SegFormer and YOLO are checked differently on purpose. A SegFormer run is
        usable only if it has BOTH weights and an operating_point.json, because
        its threshold was fitted on val and scoring it at any other cut would not
        reproduce the reported number. YOLO's native-argmax path has no threshold
        at all -- argmax is parameter-free -- so its weights alone are sufficient.
        """
        seg_seed = self._load_csv(self.env.results / "final" / "segformer_test_per_seed.csv")
        yolo_seed = self._load_csv(self.env.results / "final" / "yolo_test_per_seed.csv")
        best = self._load_csv(
            self.env.results / "final" / "best_seed_val_selected" / "best_seed_val_selected_results.csv")

        for family in ("segformer_b2_teacher", "segformer_b0_direct", "segformer_b0_distilled"):
            for seed in (0, 1, 2):
                rid = f"{family}__seed{seed}"
                r = Run(rid, "A", family, seed, "segformer")
                # Searches EVERY root, not just runs/ and checkpoints/final/.
                # Restricted to the seeded standard layout on purpose: the
                # teacher store holds a seedless `segformer_b2_teacher/`, and
                # serving that where `__seed1` was asked for would substitute a
                # different checkpoint of the same family without saying so.
                hit = find_checkpoint(self.env.run_roots, family, seed=seed,
                                      layouts=("standard",))
                w = self._attach(r, hit)
                self._resolve(r, w, self._best_seed_csv(family, best),
                              self._row(seg_seed, rid))

        for family in ("yolo_sem_direct", "yolo_sem_distilled"):
            for seed in (0, 1, 2):
                rid = f"{family}__seed{seed}"
                r = Run(rid, "A", family, seed, "yolo")
                hit = find_checkpoint(self.env.run_roots, family, seed=seed,
                                      layouts=("yolo",))
                w = self._attach(r, hit)
                pi = _first_existing(
                    self.env.ckpt / "final" / rid / "test_per_image_native_argmax.csv")
                cached = self._row(yolo_seed, rid, where={"path": "native_argmax"})
                self._resolve(r, w, pi or self._best_seed_csv(family, best), cached)

    def _scan_stage_b(self) -> None:
        """Stage B: the direct baselines.

        nnU-Net is registered even though nothing for it exists. Leaving it out
        would make the plan table look complete when the experiment grid is not,
        and "the baseline we never ran" is exactly the fact a reader most needs.
        """
        per_seed = self._load_csv(self.env.results / "baselines" / "smp_baselines_test_per_seed.csv")
        for family in ("unet_r50", "deeplabv3plus_r50"):
            for seed in (0, 1, 2):
                rid = f"{family}__seed{seed}"
                r = Run(rid, "B", family, seed, "smp")
                hit = find_checkpoint(self.env.run_roots, family, seed=seed,
                                      layouts=("standard",))
                w = self._attach(r, hit)
                pi = _first_existing(self.env.runs / rid / "test_per_image.csv",
                                     self.env.ckpt / "baselines" / rid / "test_per_image.csv")
                self._resolve(r, w, pi, self._row(per_seed, rid))

        nn = self.env.ckpt / "baselines" / "nnunet"
        w = _first_existing(*(nn.rglob("checkpoint_final.pth")))
        r = Run("nnunet_fold0", "B", "nnunet", None, "nnunet")
        self._resolve(r, w, None, None)
        if r.tier == MISSING:
            r.note = "never run -- no weights, no results (~8 h)"

    def _scan_stage_c(self) -> None:
        """Stage C: distillation teachers, arms, and the B5 seed sweep.

        An arm counts as usable only if DONE.json is present. A directory holding
        a best_model.pt but no DONE.json is a run that was interrupted mid-flight;
        its weights are a snapshot, not a result, and treating it as finished
        would put an unconverged model into the comparison table.
        """
        tdir = self.env.ckpt / "distill" / "teachers"
        for name in ("segformer_b5_teacher", "segformer_b2_teacher", "segformer_b0_distilled"):
            d = tdir / name
            r = Run(f"teacher::{name}", "C", name, None, "kd")
            r.source_root, r.layout = tdir, "teacher"
            self._resolve(r, _first_existing(d / "best_model.pt"),
                          _first_existing(d / "test_per_image.csv"), None)

        adir = self.env.ckpt / "distill" / "distill_out"
        skip = {"aggregate", "reference", "val_oracle", "optuna_alpha"}
        if adir.exists():
            for d in sorted(p for p in adir.iterdir() if p.is_dir() and p.name not in skip):
                done = (d / "DONE.json").exists()
                w = _first_existing(d / "best_model.pt") if done else None
                pi = _first_existing(d / "test_per_image.csv") if done else None
                r = Run(f"arm::{d.name}", "C", "distill_arm", None, "kd")
                r.source_root, r.layout = adir, "distill_out"
                self._resolve(r, w, pi, None)
                if not done:
                    r.tier = MISSING
                    contents = {p.name for p in d.iterdir()}
                    if not contents:
                        r.note = "never started -- empty directory"
                    elif contents <= {"run_config.json"}:
                        r.note = "configured but never ran -- run_config.json only"
                    else:
                        r.note = f"interrupted -- no DONE.json ({len(contents)} partial files)"

        rdir = self.env.results / "distill" / "segformer_b5" / "runs"
        if rdir.exists():
            for d in sorted(p for p in rdir.iterdir() if p.is_dir()):
                r = Run(f"b5::{d.name}", "C", "segformer_b5", None, "kd")
                r.source_root, r.layout = rdir, "b5_seed_sweep"
                self._resolve(r, _first_existing(d / "best_model.pt"),
                              _first_existing(d / "test_per_image.csv"), None)

    def _scan_stage_e(self) -> None:
        """Stage E: the mobile-grade families in EFFICIENT_FAMILIES, three seeds each.

        Four direct baselines plus the Stage F distilled arms (handbook §7b),
        which train and score through the identical path and differ only in spec.

        Structurally identical to the SMP baselines -- a checkpoint counts only
        with BOTH `best.pt` and `operating_point.json`, because a logit-thresholded
        model's test score is undefined without its val-fitted cut.
        """
        per_seed = self._load_csv(self.env.results / "efficient" / "efficient_test_per_seed.csv")
        for family in EFFICIENT_FAMILIES:
            for seed in self.efficient_seeds:
                rid = f"{family}__seed{seed}"
                r = Run(rid, "E", family, seed, "efficient")
                hit = find_checkpoint(self.env.run_roots, family, seed=seed,
                                      layouts=("standard",))
                w = self._attach(r, hit)
                pi = _first_existing(self.env.runs / rid / "test_per_image.csv",
                                     self.env.ckpt / "efficient" / rid / "test_per_image.csv")
                self._resolve(r, w, pi, self._row(per_seed, rid))

    def _scan_teachers(self) -> None:
        """Stage T: the teacher store -- B2, B5 and the distilled B0 reference.

        These are SEEDLESS by design. Each is a single val-selected checkpoint,
        not a 3-seed family: B5's `seed_selection.csv` picked seed 123 out of
        {42, 123, ...}, which is why asking for `__seed0` finds nothing and why
        this study has never once been able to use B5 as a teacher despite having
        had it on disk the whole time.

        Registered as stage "T" and never trained here -- `train_missing` only
        acts on stages A/B/E/H/Y, so a teacher reported MISSING is a statement
        about the bundle, not a job that is about to start.

        `segformer_b5_teacher` is the one to notice. B2 is the ONLY model in this
        study with a statistically detectable skin-tone disparity (Kruskal
        p=0.011, gap 0.112); B5 has gap 0.070 at p=0.220 and beats B2 on every
        ITA group, by +0.057 on Tan (IV) and +0.027 on Dark (VI). Every Stage H
        arm currently distills from B2.
        """
        for family in TEACHER_FAMILIES:
            hit = find_checkpoint(self.env.run_roots, family, seed=None,
                                  layouts=("teacher", "standard"))
            r = Run(family, "T", family, None, "segformer")
            if hit:
                w, tf, root, layout, rd = hit
                r.source_root, r.layout, r.threshold_file = root, layout, tf
                self._resolve(r, w, _first_existing(rd / "test_per_image.csv"), None)
            else:
                self._resolve(r, None, None, None)

    def _scan_stage_y(self) -> None:
        """Stage Y: YOLO26-large, native Ultralytics, native-argmax scoring.

        Structurally the simplest scan in the class, and deliberately so. A YOLO
        run is usable on its weights alone -- argmax is parameter-free, so unlike
        every logit-thresholded family there is no `operating_point.json` that
        could be missing and no val-fitted cut whose absence would make the test
        score undefined.

        Nothing is shipped: the stage is new, so every run starts MISSING and the
        plan reports it with its cost rather than omitting it.
        """
        per_seed = self._load_csv(self.env.results / "yolo_l" / "yolo_l_test_per_seed.csv")
        native = ("ultralytics_runs", "train", "weights", "best.pt")
        for family in STAGE_Y_FAMILIES:
            for seed in self.yolo_l_seeds:
                rid = f"{family}__seed{seed}"
                fresh, shipped = self.env.runs / rid, self.env.ckpt / "yolo_l" / rid
                r = Run(rid, "Y", family, seed, "yolo")
                hit = find_checkpoint(self.env.run_roots, family, seed=seed,
                                      layouts=("yolo",))
                w = self._attach(r, hit)
                pi = _first_existing(fresh / "test_per_image_native_argmax.csv",
                                     shipped / "test_per_image_native_argmax.csv")
                self._resolve(r, w, pi,
                              self._row(per_seed, rid, where={"path": "native_argmax"}))

    def _scan_stage_h(self) -> None:
        """Stage H: reliability-gated KD and the SegFormer-B2 teacher axis.

        Three kinds in one stage, and each keeps the check its family already had
        rather than acquiring a new one:

          efficient / segformer   `best.pt` AND `operating_point.json` -- a
                                  logit-thresholded model's test score is undefined
                                  without the cut it fitted on validation.
          yolo                    the native Ultralytics weight path only. The
                                  argmax head is parameter-free, so there is no
                                  operating point to be missing.

        Nothing is ever shipped for this stage -- it is new -- so every run starts
        MISSING and the plan says so with its cost. That is the honest state, not
        an omission.
        """
        per_seed = self._load_csv(self.env.results / "rgkd" / "rgkd_test_per_seed.csv")
        native = ("ultralytics_runs", "train", "weights", "best.pt")
        for family in STAGE_H_FAMILIES:
            kind = STAGE_H_KIND.get(family, "efficient")
            for seed in self.rgkd_seeds:
                rid = f"{family}__seed{seed}"
                fresh = self.env.runs / rid
                shipped = self.env.ckpt / "rgkd" / rid
                r = Run(rid, "H", family, seed, kind)
                if kind == "yolo":
                    hit = find_checkpoint(self.env.run_roots, family, seed=seed,
                                          layouts=("yolo",))
                    pi = _first_existing(fresh / "test_per_image_native_argmax.csv",
                                         shipped / "test_per_image_native_argmax.csv")
                else:
                    hit = find_checkpoint(self.env.run_roots, family, seed=seed,
                                          layouts=("standard",))
                    pi = _first_existing(fresh / "test_per_image.csv",
                                         shipped / "test_per_image.csv")
                w = self._attach(r, hit)
                self._resolve(r, w, pi, self._row(per_seed, rid))

    # ── small CSV helpers ────────────────────────────────────────────────────
    @staticmethod
    def _load_csv(p: Path) -> pd.DataFrame | None:
        return pd.read_csv(p) if p.exists() else None

    @staticmethod
    def _row(df: pd.DataFrame | None, run_id: str, where: dict | None = None) -> dict | None:
        """Recover one run's cached summary row, or None."""
        if df is None or "run_id" not in df.columns:
            return None
        m = df[df.run_id == run_id]
        for k, v in (where or {}).items():
            if k in m.columns:
                m = m[m[k] == v]
        return m.iloc[0].to_dict() if len(m) else None

    def _best_seed_csv(self, family: str, best: pd.DataFrame | None) -> Path | None:
        """Find the shipped best-seed per-image CSV for a family, if this IS that seed.

        Only the val-selected seed has a per-image CSV in results/final. Returning
        it for every seed would silently give three seeds the same numbers, so the
        caller gets it only when the seed matches.
        """
        d = self.env.results / "final" / "best_seed_val_selected"
        if not d.exists():
            return None
        hits = sorted(d.glob(f"{family}_best_seed*_test_per_image.csv"))
        return hits[0] if hits else None

    # ── the plan ─────────────────────────────────────────────────────────────
    def frame(self) -> pd.DataFrame:
        """The whole registry as a DataFrame, ordered stage then run_id."""
        rows = [{"stage": r.stage, "run_id": r.run_id, "family": r.family, "seed": r.seed,
                 "tier": r.tier, "note": r.note,
                 "weights": str(r.weights.relative_to(self.env.root)) if r.weights
                            and self.env.root in r.weights.parents else (str(r.weights) if r.weights else ""),
                 "per_image": bool(r.per_image), "cached": bool(r.cached)}
                for r in self.runs.values()]
        return pd.DataFrame(rows).sort_values(["stage", "run_id"]).reset_index(drop=True)

    def to_train(self, stage: str | None = None) -> list[Run]:
        """Runs that would train. Empty means this session costs no GPU time."""
        return [r for r in self.runs.values()
                if r.tier == MISSING and (stage is None or r.stage == stage)]

    def usable(self, stage: str | None = None) -> list[Run]:
        """Runs that can contribute a number, from weights or from cache."""
        return [r for r in self.runs.values()
                if r.tier != MISSING and (stage is None or r.stage == stage)]

    def get(self, run_id: str) -> Run | None:
        return self.runs.get(run_id)

    def report(self, verbose: bool = True) -> pd.DataFrame:
        """Print the plan: per-stage tier counts, then every run that will not run.

        The summary comes first because it answers the only question that matters
        before you press Run All -- "is this going to cost me a GPU day?" -- and
        the exceptions come second because a list of things that worked is noise.
        """
        df = self.frame()
        names = {"A": "A · final (SegFormer + native YOLO)",
                 "B": "B · baselines (U-Net, DeepLabV3+, nnU-Net)",
                 "C": "C · B5 distillation",
                 "E": "E/F · mobile baselines + DeepLabV3+ distilled arms",
                 "H": "H · reliability-gated KD + the SegFormer-B2 teacher axis"}
        print("=" * 78)
        print("CHECKPOINT REGISTRY".center(78))
        print("=" * 78)
        for stage in ("A", "B", "C", "E", "H"):
            sub = df[df.stage == stage]
            if not len(sub):
                continue
            c = sub.tier.value_counts()
            print(f"\n{names[stage]}   ({len(sub)} runs)")
            print(f"  WEIGHTS {c.get(WEIGHTS, 0):>3}   loaded from checkpoint, nothing trains")
            print(f"  RESULTS {c.get(RESULTS, 0):>3}   reported from cached metrics")
            print(f"  MISSING {c.get(MISSING, 0):>3}   no checkpoint and no cached result")

        gaps = self.to_train()
        print("\n" + "-" * 78)
        if not gaps:
            print("Every run is covered. This session will not train anything.")
        else:
            hours = sum(r.cost_hours for r in gaps)
            verb = "WILL TRAIN" if self.allow_training else "SKIPPED (allow_training=False)"
            print(f"{len(gaps)} run(s) have no checkpoint and no cached result -- {verb}")
            print(f"estimated cost if trained: ~{hours:.1f} GPU-hours on an A100\n")
            for r in sorted(gaps, key=lambda x: (x.stage, x.run_id)):
                print(f"  [{r.stage}] {r.run_id:<34} {r.note}")
            if not self.allow_training:
                print("\n  These stay MISSING in every table below -- they are never "
                      "back-filled\n  from another seed. Set ALLOW_TRAINING = True to fill them.")
        print("-" * 78)
        if verbose:
            return df
        return df


def summarize_gaps(reg: "Registry") -> pd.DataFrame:
    """One row per genuine gap, for pasting into a report's limitations section."""
    return pd.DataFrame([{"stage": r.stage, "run_id": r.run_id, "reason": r.note,
                          "est_gpu_hours": r.cost_hours} for r in reg.to_train()])
