"""Stage M -- multi-teacher routed distillation, and the oracle that gates it.

THIS MODULE IS EXPERIMENTAL AND DELIBERATELY NOT PART OF THE BUILD
===================================================================
Every other module in `bruisekit/` is copied from `scripts/unified_lib/` by
`60_build_unified_bundle.py`. This one is not in that list on purpose. A rebuild
will neither regenerate nor delete it, and nothing in the modular pipeline
imports it -- `bruise_multiteacher.ipynb` is its only caller. If Stage M ever
produces a result worth reporting, moving this file into `scripts/unified_lib/`
and adding its name to `copy_authored_modules` is the graduation step. Until
then an experiment that fails costs the pipeline nothing.

It touches the shipped code in exactly two places, both of them existing patch
points that Stage F and Stage H already use:

    engine.load_teacher   -> returns a STACK of teacher probabilities
    engine.DistillLoss    -> routes over that stack, per image

Both preserve `_original` and fall through for every family that is not a Stage
M arm, so installing this shim leaves Stages A/C/F/H bit-for-bit unchanged.


THE IDEA
--------
Stage A distils B2 -> B0. Stage C distils B5 (and B2+B5 ensembles) -> B0. Stage
F distils DeepLabV3+ -> mobile. Every one of them fixes the teacher for the whole
training set. But the teachers are not uniformly better or worse than each other
-- on the 185 test images the best teacher differs by skin-tone group:

    group                    best teacher   its Dice   B0 student
    Light (II-III)           B5             0.766      0.757
    Intermediate (III-IV)    B2             0.833      0.794
    Tan (IV)                 U-Net          0.766      0.730
    Brown (V)                U-Net          0.813      0.795
    Dark (VI)                DeepLabV3+     0.791      0.755

Four different teachers win the five groups. So: hand each training image to
whichever teacher is best on THAT image, and distil from the routed signal.


WHY THE ROUTING IS FEASIBLE, WHICH IS THE PART THAT SURPRISES PEOPLE
---------------------------------------------------------------------
Stage C's adaptive arm had to route on teacher-vs-teacher agreement, because it
was framed as an inference-time ensemble and the label is not available then.
Stage M routes at TRAINING time only, where the ground-truth mask is right there
in the batch -- `train_run` already hands it to the loss. The router can
therefore score each teacher against the actual label, per image, and weight
accordingly. There is no generalisation gap in the router because the router
never runs at test time; only the student does.

That makes the oracle a REACHABLE ceiling on teacher quality rather than an
aspirational one, which is the single respect in which this differs from
`kd/val_oracle.py`. What remains uncertain -- and what no oracle can settle -- is
how much better teacher quality becomes better STUDENT quality. Stage C is the
only measurement of that transfer rate this project has:

    oracle gain over the best single teacher   +0.0258
    realised student gain (p3_adaptive)        +0.0068     -> ~26%

`oracle_gate` projects through that rate and refuses to open on a gain the
student demonstrably cannot inherit. That is the whole point of running the gate
before the grid.


ONE KNOB, AND IT NESTS THE ARMS THAT ALREADY EXIST
----------------------------------------------------
Routing weights are `softmax(beta * dice_k)` over the K teachers, where `dice_k`
is teacher k's soft Dice against the label on that image:

    beta = 0     uniform mean of all teachers   == Stage C's p2_ensemble_uniform
    beta -> inf  hard argmax, the oracle router
    K = 1        exactly the existing single-teacher KD, bit-for-bit

The K=1 identity is asserted in `self_test()` rather than claimed, because it is
what makes `*_mtkd` vs its control a one-variable contrast. Soft Dice, not
thresholded Dice: the operating point is not fitted until after training, and a
router that depended on one would entangle this arm with threshold choice -- the
same reason `reliability_kd.image_gate` uses soft Dice.


WHAT THIS DOES NOT DO
----------------------
It does not route by skin tone. The August 08 tables say the teachers' complete
misses are concentrated entirely on Light (II-III) -- on 55 dark-skin images not
one of the four teachers misses anything -- so a skin-tone router would be
tuning five weights on ~4 subjects per group to fix a failure that is not there.
`stratified_oracle` reports the per-group and per-size-quintile breakdown with n
printed next to every cell, so that stays visible instead of being assumed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# the pool
# ─────────────────────────────────────────────────────────────────────────────
TEACHER_POOL: tuple[str, ...] = (
    "segformer_b2_teacher",
    "segformer_b5_teacher",
    "deeplabv3plus_r50",
    "unet_r50",
)

# `segformer_b5_teacher` is scanned by `registry._scan_teachers` but has no
# `loaders.FAMILY_SPEC` row, so `build_for_load` raises on it. Added here rather
# than there for the reason in the module docstring: loaders.py is a build
# output and an edit to it is reverted by the next bundle build.
EXTRA_SPECS = {
    "segformer_b5_teacher": {"arch": "segformer", "size": "b5", "distill": False},
}

# Stage M arms. Students chosen to give the pair Stage F wishes it had: the
# reference SegFormer student (so the contrast is against a control that already
# exists at three seeds) and the strongest mobile student (so a positive result
# is deployable). Both have a teacher-matched control already trained.
STAGE_M_FAMILIES: tuple[str, ...] = (
    "segformer_b0_mtkd",
    "lraspp_mobilenetv3_mtkd",
)

STUDENT_ARCH = {
    "segformer_b0_mtkd": "segformer",
    "lraspp_mobilenetv3_mtkd": "lraspp_mobilenetv3",
}

# The one-variable contrast for each arm. `*_mtkd` differs from its control ONLY
# in how the teacher signal is formed; alpha, recipe, seeds and student are
# identical. segformer_b0_distilled is the B2->B0 reference Stage C scored every
# arm against, so Stage M is scored on the same ruler.
CONTROL_FOR = {
    "segformer_b0_mtkd": "segformer_b0_distilled",
    "lraspp_mobilenetv3_mtkd": "lraspp_mobilenetv3_b2kd",
}

# Wall-clock, extrapolated from Stage F's distilled arms (~0.75 h with ONE frozen
# ResNet-50 teacher) scaled by the pool's forward cost, B5 dominating. Not
# calibrated by a finished run -- no Stage M run exists yet.
COST_HOURS = {
    "segformer_b0_mtkd": 2.6,
    "lraspp_mobilenetv3_mtkd": 2.2,
}

# Stage C, measured: +0.0258 of oracle teacher gain bought +0.0068 of student
# gain. The gate projects through this rather than through an assumption, and
# prints it, because it is the load-bearing number and a reader should be able to
# reject it.
STAGE_C_ORACLE_GAIN = 0.0258
STAGE_C_STUDENT_GAIN = 0.0068
TRANSFER_RATE = STAGE_C_STUDENT_GAIN / STAGE_C_ORACLE_GAIN     # 0.264

# The non-inferiority margin Stage C scored its arms against (one Dice point).
# Reused deliberately: an arm that cannot clear the bar its predecessor was
# judged on should not consume GPU hours to find that out.
MARGIN = 0.01

ACTIVE_ARM: str | None = None


def is_stage_m(family: str | None) -> bool:
    return family in STAGE_M_FAMILIES


def register_specs() -> list[str]:
    """Teach `loaders` about B5 and the Stage M students. Idempotent.

    Same mechanism and the same reason as `distill_efficient.register_student_aliases`:
    a distilled variant is a new FAMILY on an existing ARCHITECTURE, and the dicts
    that map one to the other are keyed at their definition site.
    """
    from bruisekit import loaders as L

    added = []
    for family, spec in EXTRA_SPECS.items():
        if family not in L.FAMILY_SPEC:
            L.FAMILY_SPEC[family] = dict(spec)
            added.append(family)

    for family, arch in STUDENT_ARCH.items():
        if family not in L.FAMILY_SPEC:
            L.FAMILY_SPEC[family] = {
                "arch": "segformer" if arch == "segformer" else arch,
                "size": "b0" if arch == "segformer" else None,
                "distill": True, "teacher": "POOL", "kd": "routed",
            }
            added.append(family)
        if arch != "segformer" and family not in L.EFFICIENT_FAMILIES:
            L.EFFICIENT_FAMILIES = L.EFFICIENT_FAMILIES + (family,)

    # The mobile student additionally needs the efficient-model plumbing to know
    # which architecture to build and which published parameter count to quote.
    from bruisekit import distill_efficient as DE
    for family, arch in STUDENT_ARCH.items():
        if arch != "segformer":
            DE.register_student_alias(family, arch)
    return added


# ─────────────────────────────────────────────────────────────────────────────
# scoring a teacher on VALIDATION -- the path loaders.py does not have
# ─────────────────────────────────────────────────────────────────────────────
def load_teacher_model(env, run, cfg: dict, verbose: bool = True):
    """`(model, cut)` for any pooled teacher, including the Stage C `kd/` lineage.

    `loaders.load_model` handles the three checkpoint LAYOUTS and the known
    encoder/stages rename, and it is enough for B2, DeepLabV3+ and U-Net. It is
    not enough for B5, which was trained by a DIFFERENT WRAPPER CLASS:

        models.SegFormerNet                       kd/train_segformer_b5_baseline.py
        ------------------------------------      -----------------------------------
        submodule named `net`                     submodule named `model`
        registers `mean`/`std` buffers            no buffers
        normalises [0,1] -> ImageNet in forward   dataloader normalises (A.Normalize)

    so its state dict is spelled `model.segformer.…` where `SegFormerNet` wants
    `net.segformer.…`, and it carries no `mean`/`std`. That is 1172 keys against
    1174 and `_reconcile_segformer_layout` correctly refuses it — the rename it
    knows about is the encoder/stages one, not this.

    **The normalisation is equivalent, which is the part that had to be checked
    rather than assumed.** Both wrappers use the same ImageNet mean/std; they
    differ only in WHERE it is applied. Loading B5's weights into `SegFormerNet`
    and feeding it the standard [0,1] loader output reproduces exactly the tensor
    B5 was trained on. Had the two used different statistics this remap would have
    loaded cleanly and silently mis-scaled every B5 probability — the same class
    of failure as feeding YOLO ImageNet-normalised pixels (§3, §15 trap 4).

    `mean` and `std` are the only keys allowed to be missing, because they are
    constants that `SegFormerNet.__init__` has already set correctly. Any other
    missing or unexpected key raises: a teacher loaded with random weights
    anywhere in it still produces plausible probabilities, and the gate would
    report them as complementarity.
    """
    import torch
    from bruisekit import loaders as L
    from bruisekit.registry import read_cut

    if run.weights is None:
        raise ValueError(f"{run.run_id} has no checkpoint (tier={run.tier})")

    try:
        return L.load_model(env, run, cfg.get("smp_encoder", "resnet50"))
    except (RuntimeError, ValueError) as first_error:
        state = torch.load(str(run.weights), map_location="cpu", weights_only=True)
        if not any(k.startswith("model.") for k in state):
            raise                                  # not the wrapper mismatch
        model = L.build_for_load(env, run.family, cfg.get("smp_encoder", "resnet50"))
        own = model.state_dict()
        remapped = {("net." + k[len("model."):] if k.startswith("model.") else k): v
                    for k, v in state.items()}

        # Supply the two buffers the kd/ wrapper never had, from the freshly built
        # model where __init__ has already set them to the ImageNet constants. They
        # go in BEFORE the layout reconciliation because that function matches key
        # SETS exactly, and a checkpoint two keys short can never match.
        supplied = [k for k in ("mean", "std") if k not in remapped and k in own]
        for k in supplied:
            remapped[k] = own[k]

        # The prefix swap alone is enough only when the installed transformers
        # wants the same inner layout the checkpoint was saved with. When it does
        # not, the encoder/stages rename has to be composed on top -- so hand the
        # prefix-swapped dict to the shipped reconciler rather than reimplementing
        # a second copy of that rename here.
        direction = "prefix only"
        try:
            model.load_state_dict(remapped)
        except RuntimeError as e:
            if "size mismatch" in str(e) or not L._looks_like_segformer_layout_clash(e):
                raise ValueError(
                    f"{run.run_id}: `model.` -> `net.` did not reconcile the "
                    f"checkpoint, and the residual difference is not the known "
                    f"encoder/stages rename.\n  {e}\n"
                    f"  original error: {first_error}") from e
            remapped, direction = L._reconcile_segformer_layout(model, remapped)
            model.load_state_dict(remapped)

        model.to(env.device).eval()
        if verbose:
            print(f"  NOTE  {run.run_id}: remapped `model.` -> `net.` (Stage C kd/ "
                  f"wrapper) + {direction}; {supplied or 'no'} buffer(s) taken from "
                  f"__init__. Normalisation is equivalent -- see load_teacher_model.")

        tf = getattr(run, "threshold_file", None)
        if tf is None:
            raise FileNotFoundError(f"{run.run_id}: no threshold file beside {run.weights}")
        cut = read_cut(tf)
        if cut is None:
            raise ValueError(f"{run.run_id}: {tf} yielded no usable cut")
        return model, cut


def score_on_split(env, run, cfg: dict, man640: dict, split: str = "val") -> pd.DataFrame:
    """Per-image table for one run on any split, at its own val-fitted cut.

    `loaders.score_segformer` hardcodes `cache640_manifests["test"]`, which is
    correct for reporting and useless for a gate: the whole point of deciding on
    validation is not to touch test. This is that function with the split as an
    argument and nothing else changed -- same `load_model`, same
    `evaluate_at_cut`, same inference-mode and batch guards.

    A NOTE ON WHAT VAL DICE MEANS HERE. Each teacher's cut was fitted ON these
    134 images, so every teacher's val Dice is mildly optimistic. That bias is
    shared by all of them and by the oracle, so it cancels in the GAIN, which is
    the only quantity the gate reads. It does not cancel in the absolute
    per-teacher numbers, which is why those are printed as context and never as a
    result.
    """
    import gc

    import torch
    from bruisekit.data import make_loader
    from bruisekit.evaluate import evaluate_at_cut

    if split not in man640:
        raise KeyError(f"no {split!r} manifest; have {sorted(man640)}")

    model, cut = load_teacher_model(env, run, cfg)
    on_cuda = str(env.device).startswith("cuda")
    amp = bool(cfg.get("amp", True)) and on_cuda
    batch = cfg.get("eval_batch", 8) if on_cuda else cfg.get("eval_batch_cpu", 2)
    loader = make_loader(man640[split], env.cache640, cfg["img_size"], batch,
                         False, cfg.get("workers", 0), 0)
    with torch.inference_mode():
        per_image, _ = evaluate_at_cut(model, loader, env.device, cut, amp)

    del model, loader
    gc.collect()
    if on_cuda:
        torch.cuda.empty_cache()
    per_image.attrs["cut"] = cut
    return per_image


def resolve_teachers(env, reg, pool=TEACHER_POOL, seed: int = 0) -> list:
    """The `Run` for each pooled teacher, or a readable failure naming the misses.

    B5 is promoted rather than seeded -- `segformer_b5__seed123` won on validation
    and ships as `teachers/segformer_b5_teacher` with no seed in the path -- so
    asking the registry for `__seed{k}` can never match it. `find_checkpoint`
    already understands that layout; this just stops the caller from having to.
    """
    runs, missing = [], []
    for family in pool:
        r = None
        for candidate_seed in (seed, None):
            r = reg.get(f"{family}__seed{candidate_seed}") if candidate_seed is not None \
                else reg.get(family)
            if r is not None and getattr(r, "weights", None) is not None:
                break
            r = None
        if r is None:
            for run in reg.usable():
                if run.family == family and getattr(run, "weights", None) is not None:
                    r = run
                    break
        if r is None:
            missing.append(family)
        else:
            runs.append(r)
    if missing:
        raise FileNotFoundError(
            "Stage M needs every pooled teacher's checkpoint and none of these "
            f"resolved: {missing}.\n"
            "  Set EXTRA_RUNS to the scratch that holds them (handbook 10.2) and "
            "re-scan. The gate is a val pass over trained weights -- it cannot be "
            "run from results CSVs, which only cover test.")
    return runs


def backfill_gt_pixels(matrix: pd.DataFrame, env, split: str = "val",
                       verbose: bool = True) -> pd.DataFrame:
    """Add `gt_positive_pixels` to a matrix that was cached without it.

    Counts positive pixels straight from the 640 mask cache -- 134 PNG reads, no
    model forwards -- so a matrix cached by an earlier version does not have to be
    re-scored to get the size stratification back. The count is a property of the
    label alone, so this is the same number `evaluate_at_cut` would have written.

    Returns the frame unchanged if the column is already present.
    """
    if "gt_positive_pixels" in matrix.columns:
        return matrix

    import cv2

    mdir = env.cache640 / split / "masks"
    if not mdir.is_dir():
        raise FileNotFoundError(
            f"no 640 mask cache at {mdir}; run L.build_cache640 (notebook §2) first")

    counts = []
    for stem in matrix["stem"]:
        p = mdir / f"{stem}.png"
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(f"unreadable mask: {p}")
        counts.append(int((m > 0).sum()))

    out = matrix.copy()
    out["gt_positive_pixels"] = counts
    if verbose:
        print(f"  back-filled gt_positive_pixels for {len(out)} images from "
              f"{mdir} (no re-scoring)")
    return out


def val_dice_matrix(env, reg, cfg: dict, man640: dict, meta: pd.DataFrame,
                    pool=TEACHER_POOL, seed: int = 0, student: str | None = None,
                    cache: Path | None = None, verbose: bool = True) -> pd.DataFrame:
    """`[stem, subject, <one Dice column per teacher>, (student)]` on validation.

    Cached to CSV because it costs one forward pass per teacher over 134 images
    and the gate is meant to be re-read, re-plotted and argued about without
    paying for it again. Delete the file to force a recompute.
    """
    if cache is not None and Path(cache).exists():
        if verbose:
            print(f"  [cached] {cache}")
        return pd.read_csv(cache)

    register_specs()
    runs = resolve_teachers(env, reg, pool, seed)
    if student is not None:
        s = reg.get(f"{student}__seed{seed}")
        if s is None or getattr(s, "weights", None) is None:
            raise FileNotFoundError(f"student {student}__seed{seed} has no checkpoint")
        runs = runs + [s]

    out = None
    for run in runs:
        if verbose:
            print(f"  scoring {run.run_id} on val ...", flush=True)
        scored = score_on_split(env, run, cfg, man640, "val")
        pi = scored[["stem", "dice"]].rename(columns={"dice": run.family})
        if out is None:
            # gt_positive_pixels is a property of the LABEL, not of the model, so
            # it is identical in every teacher's table and taken from the first.
            # It is carried here because the manifest does not have it and
            # `stratified_oracle(by="size")` does -- dropping it was what made the
            # size stratification, the one likely to carry signal, unavailable.
            out = pi.merge(scored[["stem", "gt_positive_pixels"]], on="stem",
                           validate="one_to_one")
        else:
            out = out.merge(pi, on="stem", validate="one_to_one")

    keep = [c for c in ("stem", "subject", "skin_tone_category",
                        "ita_group_index_5") if c in meta.columns]
    out = out.merge(meta[keep], on="stem", how="left", validate="one_to_one")
    if out["subject"].isna().any():
        raise ValueError(f"{int(out['subject'].isna().sum())} val stems have no manifest row")

    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(cache, index=False)
        if verbose:
            print(f"  wrote {cache}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# the gate
# ─────────────────────────────────────────────────────────────────────────────
def _cluster_boot(fn, subjects: np.ndarray, reps: int, seed: int) -> np.ndarray:
    """Subject-cluster bootstrap of `fn(index_array)`.

    Subjects, not images: this dataset has 1016 images from 143 subjects and
    several images per subject are near-duplicates of one pose. Resampling images
    would treat those as independent evidence and shrink every interval.
    """
    uniq = np.unique(subjects)
    idxby = {s: np.where(subjects == s)[0] for s in uniq}
    rng = np.random.default_rng(seed)
    draws = np.empty(reps)
    for r in range(reps):
        pick = rng.choice(uniq, uniq.size, replace=True)
        draws[r] = fn(np.concatenate([idxby[s] for s in pick]))
    return draws


def routing_weights(dice: np.ndarray, beta: float) -> np.ndarray:
    """`softmax(beta * dice_k)` over teachers, row-wise. `[N, K] -> [N, K]`.

    beta = 0 is the uniform ensemble; beta -> inf is hard argmax. Written to be
    numerically safe at large beta (subtract the row max) because the arms
    deliberately include a very large beta as the hard-routing limit.
    """
    d = np.asarray(dice, dtype=np.float64)
    if not np.isfinite(beta):
        w = np.zeros_like(d)
        w[np.arange(len(d)), d.argmax(axis=1)] = 1.0
        return w
    z = beta * d
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def oracle_gate(matrix: pd.DataFrame, pool=TEACHER_POOL, student: str | None = None,
                reps: int = 5000, seed: int = 0, margin: float = MARGIN,
                transfer_rate: float = TRANSFER_RATE) -> dict:
    """Does a K-teacher router have enough headroom to be worth training?

    Reports, all on validation and all with a subject-cluster bootstrap:

      - each teacher's val Dice, and the oracle's
      - ORACLE GAIN over the best single teacher, with CI and P(gain > 0)
      - DROP-ONE marginals: how much of that gain each teacher is solely
        responsible for. A teacher with a zero marginal is a teacher the pool
        does not need, and four teachers that each look useful pairwise can still
        have three redundant marginals.
      - oracle gain over the STUDENT, when a student column is present -- the
        quantity KD could in principle transfer
      - PROJECTED STUDENT GAIN = oracle gain x Stage C's measured transfer rate

    THE GATE RULE, fixed before the numbers are looked at:

        open  iff  the oracle-gain CI clears zero
              AND  projected student gain > margin (one Dice point)

    The second clause is the one that matters and the one Stage C did not have.
    Its gate opened on +0.0258 of oracle gain and the arm delivered +0.0068 --
    non-inferior, indistinguishable, ten arms of GPU time. Requiring the
    PROJECTION to clear the margin, rather than the oracle itself, is what stops
    this from being the eleventh.

    A separate miss-endpoint gate is reported alongside and is NOT ANDed in:
    complete misses are the endpoint this study is judged on, so a pool that
    contains every teacher's misses is interesting even when Dice says nothing.
    Read both; the printed verdict says which one opened.
    """
    cols = [c for c in pool if c in matrix.columns]
    if len(cols) < 2:
        raise ValueError(f"need >=2 teacher columns, found {cols}")

    D = matrix[cols].to_numpy(float)                 # [N, K]
    subjects = matrix["subject"].to_numpy()
    orc = D.max(axis=1)
    all_ix = np.arange(len(D))

    def gain_full(ix):
        return orc[ix].mean() - D[ix].mean(axis=0).max()

    point = float(gain_full(all_ix))
    draws = _cluster_boot(gain_full, subjects, reps, seed)
    lo, hi = (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))

    per_teacher = {c: float(D[:, k].mean()) for k, c in enumerate(cols)}
    best_single = max(per_teacher, key=per_teacher.get)

    # Drop-one marginals. Recomputed against the SAME best-single baseline the
    # full pool is scored on, so the numbers are differences in oracle value, not
    # differences in two separately-optimised baselines.
    marginals = {}
    for k, c in enumerate(cols):
        rest = [j for j in range(len(cols)) if j != k]
        orc_rest = D[:, rest].max(axis=1)
        g_rest = float(orc_rest.mean() - D[:, rest].mean(axis=0).max())
        marginals[c] = point - g_rest

    # How often each teacher actually wins an image. A teacher that never wins
    # contributes nothing no matter how good its mean is.
    win = D.argmax(axis=1)
    win_share = {c: float((win == k).mean()) for k, c in enumerate(cols)}

    res = {
        "n_val_images": int(len(D)), "n_subjects": int(np.unique(subjects).size),
        "teachers": cols,
        "per_teacher_val_dice": per_teacher,
        "best_single_teacher": best_single,
        "best_single_val_dice": per_teacher[best_single],
        "oracle_val_dice": float(orc.mean()),
        "oracle_val_median": float(np.median(orc)),
        "oracle_gain_over_best_single": point,
        "oracle_gain_ci95": [lo, hi],
        "p_gain_positive": float((draws > 0).mean()),
        "drop_one_marginal": marginals,
        "win_share": win_share,
        "transfer_rate_used": float(transfer_rate),
        "transfer_rate_provenance": (
            f"Stage C: student +{STAGE_C_STUDENT_GAIN} realised from oracle "
            f"+{STAGE_C_ORACLE_GAIN}"),
        "projected_student_gain": float(point * transfer_rate),
        "projected_student_gain_ci95": [lo * transfer_rate, hi * transfer_rate],
        "margin": float(margin),
        "stage_c_oracle_gain_for_reference": STAGE_C_ORACLE_GAIN,
    }

    if student is not None and student in matrix.columns:
        s = matrix[student].to_numpy(float)

        def gain_vs_student(ix):
            return orc[ix].mean() - s[ix].mean()

        gs = float(gain_vs_student(all_ix))
        ds = _cluster_boot(gain_vs_student, subjects, reps, seed)
        res["student"] = student
        res["student_val_dice"] = float(s.mean())
        res["oracle_gain_over_student"] = gs
        res["oracle_gain_over_student_ci95"] = [float(np.percentile(ds, 2.5)),
                                                float(np.percentile(ds, 97.5))]

    # ── the miss endpoint ────────────────────────────────────────────────────
    miss = (D == 0)
    pool_miss = miss.all(axis=1)              # no teacher in the pool finds it
    res["misses_per_teacher"] = {c: int(miss[:, k].sum()) for k, c in enumerate(cols)}
    res["misses_best_single"] = int(miss[:, cols.index(best_single)].sum())
    res["misses_pool_oracle"] = int(pool_miss.sum())
    res["miss_gate_open"] = bool(res["misses_pool_oracle"] < res["misses_best_single"])

    res["GATE_dice_ci_clears_zero"] = bool(lo > 0)
    res["GATE_projection_clears_margin"] = bool(res["projected_student_gain"] > margin)
    res["GATE_run_method"] = bool(res["GATE_dice_ci_clears_zero"]
                                  and res["GATE_projection_clears_margin"])
    res["GATE_any"] = bool(res["GATE_run_method"] or res["miss_gate_open"])
    return res


def stratified_oracle(matrix: pd.DataFrame, pool=TEACHER_POOL, by: str = "skin_tone_category",
                      q: int = 5) -> pd.DataFrame:
    """Per-stratum oracle gain, with n and n_subjects printed beside every cell.

    `by="size"` bins on `gt_positive_pixels` quintiles instead of a manifest
    column. Size, because the August 08 fairness tables put every teacher's
    complete misses on Light (II-III) and none on Dark (VI), which is the
    signature of a bruise-size confound rather than a skin-tone effect -- so the
    size stratification is the one likely to carry signal and the skin-tone one is
    here to show that it does not.

    n_subjects is a column and not a footnote on purpose. Five ITA groups over 20
    validation subjects is three to five subjects a cell, and a per-cell gain
    computed from four subjects is not a measurement. Anyone reading this table
    should be able to see that without being told.
    """
    cols = [c for c in pool if c in matrix.columns]
    df = matrix.copy()
    if by == "size":
        if "gt_positive_pixels" not in df.columns:
            raise KeyError("size stratification needs gt_positive_pixels in the matrix")
        df["_stratum"] = pd.qcut(df["gt_positive_pixels"], q,
                                 labels=[f"Q{i + 1}" for i in range(q)], duplicates="drop")
    else:
        df["_stratum"] = df[by]

    rows = []
    for name, g in df.groupby("_stratum", observed=True):
        D = g[cols].to_numpy(float)
        orc = D.max(axis=1)
        means = D.mean(axis=0)
        best = int(means.argmax())
        rows.append({
            "stratum": str(name), "n_images": len(g),
            "n_subjects": int(g["subject"].nunique()),
            "best_teacher": cols[best], "best_teacher_dice": float(means[best]),
            "oracle_dice": float(orc.mean()),
            "oracle_gain": float(orc.mean() - means[best]),
            "misses_best_teacher": int((D[:, best] == 0).sum()),
            "misses_pool": int((D == 0).all(axis=1).sum()),
        })
    return pd.DataFrame(rows)


def format_gate(res: dict) -> str:
    """The gate as a block of text a human decides on. Verdict first, always."""
    L = []
    a = L.append
    a("=" * 74)
    a("STAGE M -- MULTI-TEACHER ORACLE GATE  (validation only; test untouched)")
    a("=" * 74)
    a(f"  {res['n_val_images']} val images, {res['n_subjects']} subjects, "
      f"{len(res['teachers'])} teachers")
    a("")
    a("  teacher                    val Dice   wins    drop-one marginal   misses")
    for c in res["teachers"]:
        a(f"    {c:<24} {res['per_teacher_val_dice'][c]:>7.4f}  "
          f"{res['win_share'][c] * 100:>5.1f}%   {res['drop_one_marginal'][c]:>+16.4f}"
          f"   {res['misses_per_teacher'][c]:>5d}")
    a(f"    {'ORACLE (per-image best)':<24} {res['oracle_val_dice']:>7.4f}"
      f"{'':>25}   {res['misses_pool_oracle']:>5d}")
    a("")
    a(f"  oracle gain over {res['best_single_teacher']}: "
      f"{res['oracle_gain_over_best_single']:+.4f}  "
      f"CI95 [{res['oracle_gain_ci95'][0]:+.4f}, {res['oracle_gain_ci95'][1]:+.4f}]  "
      f"P(>0) = {res['p_gain_positive']:.3f}")
    if "oracle_gain_over_student" in res:
        a(f"  oracle gain over {res['student']}: "
          f"{res['oracle_gain_over_student']:+.4f}  "
          f"CI95 [{res['oracle_gain_over_student_ci95'][0]:+.4f}, "
          f"{res['oracle_gain_over_student_ci95'][1]:+.4f}]")
    a("")
    a(f"  x transfer rate {res['transfer_rate_used']:.3f}  "
      f"({res['transfer_rate_provenance']})")
    a(f"  PROJECTED STUDENT GAIN {res['projected_student_gain']:+.4f}  "
      f"vs margin {res['margin']:+.4f}")
    a("")
    a(f"  [{'PASS' if res['GATE_dice_ci_clears_zero'] else 'fail'}] "
      f"oracle-gain CI clears zero")
    a(f"  [{'PASS' if res['GATE_projection_clears_margin'] else 'fail'}] "
      f"projected student gain clears the margin")
    a(f"  [{'PASS' if res['miss_gate_open'] else 'fail'}] "
      f"pool contains misses the best single teacher does not "
      f"({res['misses_pool_oracle']} vs {res['misses_best_single']})")
    a("")
    a("-" * 74)
    if res["GATE_run_method"]:
        a("  VERDICT: RUN Stage M. Both Dice clauses opened.")
    elif res["miss_gate_open"]:
        a("  VERDICT: RUN Stage M ON THE MISS ENDPOINT ONLY.")
        a("           The Dice projection does not clear the margin, so do not")
        a("           pre-register a Dice hypothesis. Report misses; report Dice")
        a("           as non-inferiority. This is the honest version of the arm.")
    else:
        a("  VERDICT: DO NOT RUN. The pool has no headroom the student could")
        a("           inherit on either endpoint. Report the gate itself -- a")
        a("           negative pre-test is a result, and it cost one val pass")
        a("           instead of a grid.")
    a("-" * 74)
    return "\n".join(L)


# ─────────────────────────────────────────────────────────────────────────────
# the method -- two shims, both on existing patch points
# ─────────────────────────────────────────────────────────────────────────────
def set_active_arm(family: str | None) -> None:
    global ACTIVE_ARM
    ACTIVE_ARM = family


class arm:
    """Context manager scoping `ACTIVE_ARM`, so a raised exception cannot leak it.

    Same contract as `reliability_kd.arm`. Both shims dispatch on this global, and
    an arm left set after a failed run would silently apply routed KD to the next
    family trained -- a corruption that produces plausible numbers.
    """

    def __init__(self, family: str | None):
        self.family = family
        self.previous = None

    def __enter__(self):
        self.previous = ACTIVE_ARM
        set_active_arm(self.family)
        return self

    def __exit__(self, *exc):
        set_active_arm(self.previous)
        return False


def teacher_temperature(env, run, cfg: dict, man640: dict | None = None,
                        verbose: bool = True) -> float:
    """The calibrated temperature for a pooled teacher, whatever its layout.

    `reliability_kd.teacher_temperature` resolves the checkpoint through
    `TEACHER_STAGE_DIR`, which has three entries and knows only the
    `<family>__seed<N>/best.pt` layout. Two of the four pooled teachers do not fit
    that: B5 is promoted rather than seeded and lives at
    `checkpoints/distill/teachers/segformer_b5_teacher/best_model.pt`. This takes
    the resolved `Run` instead, so the layout question is already answered by the
    registry, and looks in four places in this order:

      1. `calibration.json` beside the weights -- what `engine.train_run` writes
         at the end of a B2 run, so a Stage M student sees exactly the temperature
         Stage A's students saw.
      2. `temperature.json` beside the weights -- the distillation teacher store's
         spelling. B5's is 2.1999, B2's promoted copy 1.7604.
      3. The WORK_DIR cache `distill_efficient` writes, so a fit done once in a
         session is not repeated.
      4. A fresh fit on the 134 val images.

    Never falls back to T = 1. An uncalibrated teacher's soft label is the hard
    label with extra steps, and the arm would silently measure nothing -- the same
    guard, and the same reason, as Stage F's.
    """
    from bruisekit import distill_efficient as DE

    family, seed = run.family, (run.seed if run.seed is not None else 0)
    beside = Path(run.weights).parent
    for name in ("calibration.json", "temperature.json"):
        p = beside / name
        if p.exists():
            t = float(json.loads(p.read_text())["temperature"])
            if verbose:
                print(f"  [cal] {family}__seed{seed}: T={t:.4f} ({name}, shipped)")
            return t

    cached = DE.calibration_path(env, family, seed)
    if cached.exists():
        t = float(json.loads(cached.read_text())["temperature"])
        if verbose:
            print(f"  [cal] {family}__seed{seed}: T={t:.4f} (cached)")
        return t

    if man640 is None:
        raise RuntimeError(
            f"{family}__seed{seed} ships no calibration.json or temperature.json "
            f"and none was fitted earlier in this session, so its temperature is "
            f"unknown at the point the teacher is loaded. Call\n"
            f"    MT.warm_calibration(env, reg, CFG, MAN640)\n"
            f"once before training -- the setup cell does this for the whole pool "
            f"precisely so this fails in seconds rather than mid-run.")

    import torch
    from bruisekit.data import make_loader
    from bruisekit.engine import calibrate_temperature

    model, _cut = load_teacher_model(env, run, cfg, verbose=verbose)
    model.to(env.device).eval()
    loader = make_loader(man640["val"], env.cache640, cfg["img_size"],
                         cfg.get("eval_batch", 8), False, cfg.get("workers", 0), 0)
    cal = calibrate_temperature(model, loader, env.device, cfg["amp"])
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps({**cal, "teacher": f"{family}__seed{seed}"}, indent=2))
    if verbose:
        print(f"  [cal] {family}__seed{seed}: T={cal['temperature']:.4f}  "
              f"NLL {cal['nll_before']:.5f} -> {cal['nll_after']:.5f}  "
              f"plausible={cal['plausible']}")
    del model
    if str(env.device).startswith("cuda"):
        torch.cuda.empty_cache()
    return float(cal["temperature"])


def warm_calibration(env, reg, cfg: dict, man640: dict, pool=TEACHER_POOL,
                     seeds=(0, 1, 2), verbose: bool = True) -> dict:
    """Resolve and calibrate every (teacher, seed) the session will ask for.

    Called once in setup so that a missing checkpoint or an unfittable temperature
    fails in seconds, rather than three hours into a run when `load_teacher` is
    first invoked. Idempotent.
    """
    out = {}
    for seed in seeds:
        for run in resolve_teachers(env, reg, pool, seed):
            key = f"{run.family}__seed{run.seed if run.seed is not None else 0}"
            if key in out:
                continue
            out[key] = teacher_temperature(env, run, cfg, man640, verbose=verbose)
    return out


def install_batch_shim(env, pinned: dict, verbose: bool = True):
    """Pin each Stage M arm's micro-batch to the batch its CONTROL trained at.

    THE CONFOUND THIS EXISTS TO PREVENT. `segformer_b0_distilled` was probed under
    `batch_mode="per_model"` with ONE B2 teacher resident and landed at micro-batch
    64, accum 1 (its `config.json` says so). The same probe with four teachers
    resident lands far lower, because the probe measures what fits, and what fits
    changed. The arm would then differ from its control in the teacher signal AND
    in batch size AND in optimizer-step count -- three variables, one of which is
    the very thing handbook 3 fixes the recipe to avoid.

    So the batch is read off the control's `config.json` and pinned, and the
    teacher forwards are chunked (`teacher_chunk` in `install_teacher_shim`) so
    that pinning it is actually affordable. Teacher memory and student batch are
    then independent, which is the only arrangement in which a four-teacher arm
    can be recipe-matched to a one-teacher control at all.

    Mobile arms need no entry: `efficient_micro_batch` is a fixed 16 for every
    efficient family, direct or distilled, so `lraspp_mobilenetv3_mtkd` already
    matches `lraspp_mobilenetv3_b2kd` by construction.

    `pinned` maps family -> (micro_batch, accum_steps). Build it with
    `control_batch()` rather than by hand.
    """
    import bruisekit.engine as _be

    previous = getattr(_be.resolve_micro_batch, "_original", _be.resolve_micro_batch)

    def resolve_micro_batch(model, cfg, device, teacher=None):
        if is_stage_m(ACTIVE_ARM) and ACTIVE_ARM in pinned:
            micro, accum = pinned[ACTIVE_ARM]
            if verbose:
                print(f"  [batch] {ACTIVE_ARM}: pinned to {micro} x {accum} "
                      f"(its control's, not a fresh probe)")
            return int(micro), int(accum)
        return previous(model, cfg, device, teacher)

    resolve_micro_batch._original = previous
    _be.resolve_micro_batch = resolve_micro_batch
    if verbose:
        print(f"engine.resolve_micro_batch patched -> pinned for {sorted(pinned)}; "
              f"falls through otherwise.")
    return resolve_micro_batch


def control_batch(env, reg, family: str, seed: int = 0,
                  max_micro: int | None = None) -> tuple[int, int]:
    """`(micro_batch, accum_steps)` matching the control's EFFECTIVE batch.

    Read from the control run's own `config.json`, not assumed: the probe's answer
    depends on the GPU it ran on, and hardcoding 64 here would silently stop being
    true the first time a control is retrained on a different card.

    `max_micro` caps the micro-batch and makes up the difference with gradient
    accumulation, so that `micro * accum` still equals the control's effective
    batch. THIS IS THE OOM FIX AND IT IS NOT COSMETIC. `segformer_b0_distilled`
    trained at micro-batch 64 with ONE teacher resident. Three teachers resident
    is ~138 M more frozen parameters, and on a 40 GB MIG slice the student's own
    forward at 64 no longer fits -- it missed by about 200 MB.

    Shrinking the micro-batch WITHOUT accumulation would change the effective
    batch, the number of optimizer steps and therefore the LR schedule: three
    differences from the control instead of one, which is exactly the confound
    handbook §3's fixed recipe exists to prevent (and §15 trap 20). Accumulating
    back to the same effective batch keeps all three identical.

    THE ONE DIFFERENCE THAT REMAINS, and it should be stated rather than buried:
    SegFormer's decode head contains a BatchNorm, and BatchNorm normalises over
    the MICRO-batch, not the effective one. At 16 x 4 it sees groups of 16 where
    the control saw 64. That is a real difference, far smaller than a changed step
    count, and it belongs in the limitations.
    """
    control = CONTROL_FOR[family]
    for run_id in (f"{control}__seed{seed}", f"{control}__seed0"):
        r = reg.get(run_id)
        if r is None or getattr(r, "weights", None) is None:
            continue
        cfgp = Path(r.weights).parent / "config.json"
        if not cfgp.exists():
            continue
        c = json.loads(cfgp.read_text())
        if not c.get("micro_batch"):
            continue
        micro = int(c["micro_batch"])
        accum = int(c.get("accum_steps", 1))
        effective = micro * accum
        if max_micro is None or micro <= max_micro:
            return micro, accum
        # Largest divisor of the effective batch that fits the cap, so
        # micro * accum lands EXACTLY on it -- never approximately, or the
        # step count moves and the schedule with it.
        m = min(max_micro, effective)
        while effective % m != 0:
            m -= 1
        return max(1, m), effective // max(1, m)
    raise FileNotFoundError(
        f"no config.json found for {family}'s control ({control}); cannot pin the "
        f"batch to it. Without the pin the arm differs from its control in batch "
        f"size as well as in the teacher signal -- fix the checkpoint path (see "
        f"EXTRA_RUNS) rather than skipping this.")


def install_teacher_shim(env, reg, cfg: dict, pool=TEACHER_POOL, seed_mode: str = "same",
                         fixed_seed: int = 0, teacher_chunk: int = 8,
                         verbose: bool = True):
    """Rebind `engine.load_teacher` to return a STACK of teacher probabilities.

    `train_run` does `tprob = teacher(x)` and passes the result straight into the
    loss. Returning `[B, K, H, W]` instead of `[B, 1, H, W]` therefore carries the
    whole pool to the loss with no edit to the training loop, and the routing --
    which needs the label -- happens where the label already is.

    Outside a Stage M `arm()` this calls whatever was bound before it, so
    installing it last in a session that also trains Stage F or Stage H leaves
    those arms on their own teachers.

    MEMORY. Four frozen teachers is ~170 M parameters resident (~0.7 GB fp32) plus
    one forward at a time; the stack itself is B x K x 640 x 640 x 4 B, which is
    105 MB at B = 16. Teachers run under `no_grad` inside the caller's autocast,
    so no backward graph is built through any of them -- the same reason the
    original `load_teacher` does.
    """
    if seed_mode not in ("same", "fixed"):
        raise ValueError(f"seed_mode must be 'same' or 'fixed', got {seed_mode!r}")

    import torch
    import bruisekit.engine as _be
    from bruisekit import loaders as L

    register_specs()
    previous = getattr(_be.load_teacher, "_original", _be.load_teacher)
    cache: dict = {}

    def _one(run, device, amp: bool):
        key = run.run_id
        if key in cache:
            return cache[key]
        temperature = teacher_temperature(env, run, cfg, None, verbose=verbose)
        # Same loader as the gate uses: B5 needs the `model.` -> `net.` remap, and
        # a teacher silently half-initialised here would corrupt every student.
        model, _cut = load_teacher_model(env, run, cfg, verbose=verbose)
        model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        cache[key] = (run.family, model, temperature)
        return cache[key]

    def load_teacher(dir_from_train_run, paths: dict, device, amp: bool):
        if not is_stage_m(ACTIVE_ARM):
            return previous(dir_from_train_run, paths, device, amp)

        name = Path(dir_from_train_run).name
        try:
            student_seed = int(name.rsplit("seed", 1)[1])
        except (IndexError, ValueError):
            return previous(dir_from_train_run, paths, device, amp)
        seed = student_seed if seed_mode == "same" else fixed_seed

        # The registry, not a path template: B5 is promoted rather than seeded and
        # its weights are `best_model.pt` under `teachers/`, so a
        # `<family>__seed<N>/best.pt` guess can never find it. `resolve_teachers`
        # already understands all three layouts and raises a readable error naming
        # whichever teacher is missing.
        members = [_one(r, device, amp) for r in resolve_teachers(env, reg, pool, seed)]

        def teacher_fn(x):
            # Chunked over the batch, per teacher. The student trains at its
            # control's batch (64 for the SegFormer arm); four teachers -- one of
            # them B5 -- forwarding 64 images at 640x640 at once does not fit, and
            # shrinking the student's batch instead would confound the contrast.
            # Chunking changes no number: these are eval-mode forwards with frozen
            # BatchNorms, so a sub-batch and a full batch give identical outputs.
            probs = []
            for _f, model, temperature in members:
                parts = []
                for i in range(0, x.shape[0], teacher_chunk):
                    with torch.no_grad(), torch.amp.autocast("cuda", enabled=amp):
                        z = model(x[i:i + teacher_chunk])
                    parts.append(torch.sigmoid(z.float() / temperature))
                probs.append(torch.cat(parts, dim=0) if len(parts) > 1 else parts[0])
            return torch.cat(probs, dim=1)             # [B, K, H, W]

        teacher_fn.pool = [f for f, _m, _t in members]
        teacher_fn.temperature = {f: t for f, _m, t in members}
        teacher_fn.source = "+".join(teacher_fn.pool)
        if verbose:
            print(f"  [pool] {ACTIVE_ARM} <- {len(members)} teachers: "
                  f"{', '.join(teacher_fn.pool)}")
        return teacher_fn

    load_teacher._original = previous
    _be.load_teacher = load_teacher
    if verbose:
        print(f"engine.load_teacher patched -> Stage M pool of {len(pool)}; "
              f"falls through otherwise.")
    return load_teacher


def _build_routed_loss_class():
    """Defined lazily so importing this module never requires torch.

    Same reason `reliability_kd` and `loaders` do it: reading a Stage M table on a
    laptop must not need a deep-learning stack installed.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from bruisekit.losses import SupervisedLoss
    from bruisekit.reliability_kd import image_gate, reliability

    class RoutedDistillLoss(nn.Module):
        """Per-image routing over K teachers, then the Stage H gated soft term.

        `teacher_prob` arrives as `[B, K, H, W]` from the pool shim. Each teacher's
        soft Dice against the label gives the routing weights
        `softmax(beta * dice_k)`; the fused probability is the weighted sum. From
        there this is exactly `ReliabilityGatedDistillLoss` applied to the fused
        map, so Stage M inherits Stage H's gate rather than inventing a second one.

        THE TWO IDENTITIES THAT MAKE THE CONTRAST ONE-VARIABLE, both asserted in
        `self_test`:
          - K = 1 reduces to the Stage H gated loss exactly.
          - beta = 0 is the uniform teacher ensemble, i.e. Stage C's
            `p2_ensemble_uniform` with a gate.

        Routing weights are computed under `no_grad`. They are a property of the
        teachers and the label, not of the student: letting autograd see them would
        let the student reduce its loss by changing which teacher it is compared
        against, which is not a thing it should be able to optimise.
        """

        def __init__(self, alpha: float = 0.6, aux_weight: float = 0.4,
                     beta: float = 8.0, gate_lo: float = 0.10, gate_hi: float = 0.50,
                     eps: float = 1e-6):
            super().__init__()
            self.alpha = float(alpha)
            self.beta = float(beta)
            self.gate_lo = float(gate_lo)
            self.gate_hi = float(gate_hi)
            self.eps = float(eps)
            self.sup = SupervisedLoss(aux_weight)
            self.reset_stats()

        def reset_stats(self) -> None:
            self._n_batches = 0
            self._n_images = 0
            self._sum_w = None
            self._sum_entropy = 0.0
            self._sum_coverage = 0.0
            self._sum_fused_dice = 0.0
            self._sum_alpha_eff = 0.0

        def stats(self) -> dict:
            """What the router actually did. An arm whose router collapsed onto one
            teacher and an arm whose router stayed uniform are different
            experiments, and Dice alone cannot tell them apart."""
            b = max(1, self._n_batches)
            i = max(1, self._n_images)
            w = (self._sum_w / i).tolist() if self._sum_w is not None else []
            return {
                "loss": "RoutedDistillLoss",
                "alpha_nominal": self.alpha, "beta": self.beta,
                "gate_lo": self.gate_lo, "gate_hi": self.gate_hi,
                "batches_seen": self._n_batches, "images_seen": self._n_images,
                "mean_routing_weight_per_teacher": w,
                "mean_routing_entropy": self._sum_entropy / i,
                "mean_coverage": self._sum_coverage / b,
                "mean_fused_teacher_soft_dice": self._sum_fused_dice / i,
                "mean_alpha_effective": self._sum_alpha_eff / b,
            }

        def forward(self, logits, aux_logits, target, teacher_prob):
            hard = self.sup(logits, aux_logits, target)

            with torch.no_grad():
                t = teacher_prob.detach().float()               # [B, K, H, W]
                y = target.detach().float()                     # [B, 1, H, W]
                if t.dim() != 4:
                    raise ValueError(f"expected [B,K,H,W] teacher stack, got {tuple(t.shape)}")
                K = t.shape[1]

                # Per-teacher soft Dice against the label -> routing weights.
                inter = (t * y).sum(dim=(2, 3))                 # [B, K]
                denom = t.sum(dim=(2, 3)) + y.sum(dim=(2, 3))   # [B, K]
                dice_k = (2.0 * inter) / (denom + self.eps)
                w = torch.softmax(self.beta * dice_k, dim=1)    # [B, K]
                fused = (w[:, :, None, None] * t).sum(dim=1, keepdim=True)   # [B,1,H,W]

                g, dice_f = image_gate(fused, y, self.gate_lo, self.gate_hi)
                r = reliability(fused, y)
                gw = g * r
                coverage = gw.mean()

            bce = F.binary_cross_entropy_with_logits(logits, fused, reduction="none")
            soft = (gw * bce).sum() / gw.sum().clamp_min(1e-6)

            # The gated-away weight goes back to the supervised term so the total
            # loss scale -- and the gradient magnitude the LR schedule was tuned
            # against -- does not move with the gate. Stage H's convention.
            alpha_eff = self.alpha + (1.0 - self.alpha) * (1.0 - coverage)
            loss = alpha_eff * hard + (1.0 - alpha_eff) * soft

            with torch.no_grad():
                ent = -(w.clamp_min(1e-9).log() * w).sum(dim=1)
                n = int(w.shape[0])
                sw = w.sum(dim=0).cpu()
                self._sum_w = sw if self._sum_w is None else self._sum_w + sw
                self._n_batches += 1
                self._n_images += n
                self._sum_entropy += float(ent.sum())
                self._sum_coverage += float(coverage)
                self._sum_fused_dice += float(dice_f.reshape(n, -1)[:, 0].sum())
                self._sum_alpha_eff += float(alpha_eff)
                del K

            return loss

    return RoutedDistillLoss


_ROUTED_CLASS = None


def routed_loss_class():
    global _ROUTED_CLASS
    if _ROUTED_CLASS is None:
        _ROUTED_CLASS = _build_routed_loss_class()
    return _ROUTED_CLASS


def install_loss_shim(beta: float = 8.0, gate_lo: float = 0.10, gate_hi: float = 0.50,
                      verbose: bool = True):
    """Rebind `engine.DistillLoss` to a dispatcher keyed on `ACTIVE_ARM`.

    `train_run` resolves `DistillLoss` as a module global at call time, and the
    global it resolves is the one bound in `bruisekit.engine` -- so that is the
    name that has to move. Outside a Stage M `arm()` the dispatcher returns
    whatever was bound before, which in a session that also trains Stage H is
    `reliability_kd`'s dispatcher. Install this one LAST.

    `LIVE` holds the criterion of the arm currently training so its `stats()` can
    be dumped after `train_run` returns -- `train_run` does not hand the criterion
    back, and a router whose behaviour was never recorded cannot be interpreted
    later.
    """
    import bruisekit.engine as _be

    previous = getattr(_be.DistillLoss, "_original", _be.DistillLoss)
    cls = routed_loss_class()

    def DistillLoss(alpha=0.5, aux_weight=0.4):
        if not is_stage_m(ACTIVE_ARM):
            return previous(alpha, aux_weight)
        crit = cls(alpha=alpha, aux_weight=aux_weight, beta=beta,
                   gate_lo=gate_lo, gate_hi=gate_hi)
        install_loss_shim.LIVE = crit
        if verbose:
            print(f"  [loss] {ACTIVE_ARM}: RoutedDistillLoss(alpha={alpha}, beta={beta})")
        return crit

    DistillLoss._original = previous
    _be.DistillLoss = DistillLoss
    if verbose:
        print(f"engine.DistillLoss patched -> routed KD for {len(STAGE_M_FAMILIES)} arm(s); "
              f"falls through otherwise.")
    return DistillLoss


install_loss_shim.LIVE = None


def dump_router_stats(run_dir: Path) -> dict | None:
    """Write the live criterion's `stats()` next to the run. None if nothing ran."""
    crit = install_loss_shim.LIVE
    if crit is None:
        return None
    s = crit.stats()
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    (Path(run_dir) / "router_stats.json").write_text(json.dumps(s, indent=2))
    return s


# ─────────────────────────────────────────────────────────────────────────────
def self_test(verbose: bool = True) -> bool:
    """Assert the two identities the one-variable contrast rests on.

    Claims that a new loss "reduces to" an old one are exactly the claims that go
    unchecked and turn a contrast into a confound, so these are asserted, not
    documented. Runs on CPU in under a second and needs no checkpoints.
    """
    import torch

    from bruisekit.reliability_kd import gated_loss_class

    ok = True
    torch.manual_seed(0)
    B, H, W = 3, 16, 16
    logits = torch.randn(B, 1, H, W)
    target = (torch.rand(B, 1, H, W) > 0.6).float()
    t1 = torch.rand(B, 1, H, W)

    # 1. K = 1 reduces to the Stage H gated loss, exactly.
    routed = routed_loss_class()(alpha=0.6, aux_weight=0.0, beta=8.0)
    gated = gated_loss_class()(alpha=0.6, aux_weight=0.0)
    a = float(routed(logits, None, target, t1))
    b = float(gated(logits, None, target, t1))
    same = abs(a - b) < 1e-5
    ok &= same
    if verbose:
        print(f"  [{'ok' if same else 'FAIL'}] K=1 == ReliabilityGatedDistillLoss "
              f"({a:.6f} vs {b:.6f})")

    # 2. beta = 0 is the uniform ensemble: routing the stack must equal gating the
    #    plain mean of the same teachers.
    t2 = torch.rand(B, 3, H, W)
    routed0 = routed_loss_class()(alpha=0.6, aux_weight=0.0, beta=0.0)
    a = float(routed0(logits, None, target, t2))
    b = float(gated(logits, None, target, t2.mean(dim=1, keepdim=True)))
    same = abs(a - b) < 1e-5
    ok &= same
    if verbose:
        print(f"  [{'ok' if same else 'FAIL'}] beta=0 == uniform ensemble + gate "
              f"({a:.6f} vs {b:.6f})")

    # 3. Large beta routes to the teacher with the best Dice against the label.
    #    Built so teacher 1 is the label and the others are noise -- if the router
    #    does not pick it, nothing downstream means what it says.
    t3 = torch.rand(B, 3, H, W) * 0.1
    t3[:, 1] = target[:, 0]
    w = routing_weights(
        np.stack([[float(((2 * (t3[i, k] * target[i, 0]).sum())
                          / (t3[i, k].sum() + target[i, 0].sum() + 1e-6)))
                   for k in range(3)] for i in range(B)]), beta=1e3)
    picked = bool((w.argmax(axis=1) == 1).all())
    ok &= picked
    if verbose:
        print(f"  [{'ok' if picked else 'FAIL'}] large beta routes to the best teacher")

    # 4. routing_weights is a proper distribution at every beta, including inf.
    d = np.random.default_rng(0).random((7, 4))
    for beta in (0.0, 1.0, 8.0, 1e3, np.inf):
        w = routing_weights(d, beta)
        good = np.allclose(w.sum(axis=1), 1.0) and bool((w >= 0).all())
        ok &= good
        if not good and verbose:
            print(f"  [FAIL] routing_weights not a distribution at beta={beta}")
    if verbose and ok:
        print("  [ok] routing_weights sums to 1 at beta in {0, 1, 8, 1e3, inf}")

    if verbose:
        print(f"\nself_test: {'PASS' if ok else 'FAIL'}")
    return bool(ok)
