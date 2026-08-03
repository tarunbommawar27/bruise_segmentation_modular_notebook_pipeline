"""Stage F -- distillation into the mobile students, with a permissive teacher.

WHY THIS MODULE EXISTS
-----------------------
`engine.train_run` already implements distillation: `spec["distill"]` makes it
build a `DistillLoss` and call a frozen teacher on every step. But it hardcodes
*which* teacher, at engine.py:288:

    teacher = load_teacher(runs_dir / f"segformer_b2_teacher__seed{seed}", ...)

and `engine.load_teacher` hardcodes the teacher's *architecture*, at
engine.py:147:

    model = _bm("segformer", "b2", paths)

Both are fine for Stage A, where the only teacher is B2. They are the only two
things standing between this project and distilling from anything else.

`engine.py` is a build artefact -- §16 of the handbook extracts it verbatim from
`bruise_colab_baselines.ipynb`, so an edit there is reverted by the next
`60_build_unified_bundle.py`. The handbook's own instruction for this situation
(§12, "prefer adding a new module and importing it") is what this file does: it
patches the two names at runtime, exactly as `efficient_models.install_efficient_shim`
and `smp_models.install_build_model_shim` already do for `build_model`.

Nothing in the training loop, the threshold sweep, the metrics or the analysis
learns that a new teacher exists.

WHY DEEPLABV3+ AND NOT SEGFORMER
---------------------------------
The SegFormer MiT weights in `pretrained_weights/segformer_mit_b*` are NVIDIA's
(`license: other`, non-commercial). DeepLabV3+/ResNet-50 reaches the study through
`segmentation_models_pytorch` (MIT) on a torchvision ResNet-50 (BSD-3), so a
student distilled from it has no non-commercial link in its chain. This does not
by itself make the result commercialisable -- the ImageNet provenance of the
encoder and, far more importantly, the consent scope of the bruise photographs
are separate questions -- but it removes the one unambiguously non-commercial
licence.

WHY DEEPLABV3+ AND NOT U-NET
-----------------------------
They are indistinguishable on Dice (0.7584 vs 0.7570 on the val-selected seed of
each, with U-Net's median actually higher) -- inside the noise floor §1 warns
about. They are not indistinguishable on complete misses, and for a *teacher*
that is the metric that matters: a teacher that misses a bruise does not hand the
student a weak soft target, it hands it a confidently-empty one on exactly the
image the clinical metric cares about.

Paired over all 185 test images:

    images DeepLab misses that U-Net finds:  0
    images U-Net misses that DeepLab finds:  2
    both miss:                               5

Strict containment, and the ordering holds on every seed (DeepLab 5/5/2 misses,
U-Net 9/7/7). DeepLabV3+'s ASPP-then-fuse-one-low-level-skip decoder is also
structurally the same shape as Fast-SCNN's PPM-then-feature-fusion, so a later
feature-level arm (CWD) has an obvious place to attach; U-Net's four symmetric
skip levels have no counterpart in the student.

WHY THE TEACHER MUST BE CALIBRATED FIRST
-----------------------------------------
`load_teacher` reads `calibration.json` and divides the teacher's logits by that
temperature. The Stage B baselines were never calibrated -- they were trained as
endpoints, not as teachers -- so that file does not exist and the shim would
raise. `ensure_calibration` fits it with the same `engine.calibrate_temperature`
(Guo et al. 2017, L-BFGS on log T over the 134 val images) that the B2 teacher
used, so the two teachers are prepared identically.

It writes into `env.work`, never into `checkpoints/`: the shipped baselines are
inputs to this experiment and adding a file to them would make the bundle differ
from the one that produced the published Stage B numbers.
"""
from __future__ import annotations

import json
from pathlib import Path

# Student family -> the family it distils from. One entry per arm.
#
# The two arms share one teacher on purpose: fastscnn is the only scratch-init
# student and lraspp_mobilenetv3 the strongest pretrained one, so with recipe and
# teacher held fixed the pair isolates what student initialisation does to KD --
# the fastscnn arm's null result (delta -0.005, CI crossing 0, misses 7 -> 13)
# cannot say on its own whether KD failed or the scratch student did.
TEACHER_FOR = {
    "fastscnn_distilled": "deeplabv3plus_r50",
    "lraspp_mobilenetv3_distilled": "deeplabv3plus_r50",
    # Added with Stage H (see reliability_kd.py). The DeepLabV3+ teacher never
    # had an arm on these two students, which left "B2 is the better teacher" -- a
    # Stage H claim -- resting on two students instead of four. Same teacher, same
    # recipe, same efficient_alpha; only the student moves.
    "topformer_tiny_distilled": "deeplabv3plus_r50",
    "ppmobileseg_tiny_distilled": "deeplabv3plus_r50",
}

# Distilled family -> the architecture it runs on. One entry per arm; consumed by
# `register_student_aliases`, which the notebook calls whenever Stage E is read.
STUDENT_ARCH = {
    "fastscnn_distilled": "fastscnn",
    "lraspp_mobilenetv3_distilled": "lraspp_mobilenetv3",
    "topformer_tiny_distilled": "topformer_tiny",
    "ppmobileseg_tiny_distilled": "ppmobileseg_tiny",
}

# Where a teacher family's shipped checkpoints live, relative to env.ckpt.
TEACHER_STAGE_DIR = {
    "deeplabv3plus_r50": "baselines",
    "unet_r50": "baselines",
    # Stage A's teacher. Listed so `ensure_calibration` can be pointed at it if a
    # caller ever needs to re-fit it -- but note it already SHIPS a
    # calibration.json (engine.train_run writes one for any segformer_b2_teacher
    # run), and `reliability_kd.teacher_temperature` reads that in preference to
    # fitting a second one. Two temperatures for one checkpoint would make Stage A
    # and Stage H distil from measurably different teachers.
    "segformer_b2_teacher": "final",
}


# ─────────────────────────────────────────────────────────────────────────────
# locating and preparing a teacher
# ─────────────────────────────────────────────────────────────────────────────
def teacher_dir(env, family: str, seed: int) -> Path:
    """The shipped checkpoint directory for one (teacher family, seed)."""
    sub = TEACHER_STAGE_DIR.get(family)
    if sub is None:
        raise KeyError(f"no shipped location known for teacher family {family!r}; "
                       f"known: {sorted(TEACHER_STAGE_DIR)}")
    d = env.ckpt / sub / f"{family}__seed{seed}"
    if not (d / "best.pt").exists():
        raise FileNotFoundError(
            f"teacher {family}__seed{seed} has no best.pt at {d}. Stage B ships "
            f"seeds 0, 1 and 2 for both baselines; check the bundle.")
    return d


def calibration_path(env, family: str, seed: int) -> Path:
    """Where this module keeps a teacher's fitted temperature.

    Under env.work, not next to the checkpoint: `checkpoints/` is the shipped,
    verified artefact that produced the published Stage B table, and this
    experiment must not mutate it.
    """
    return env.work / "teacher_calibration" / f"{family}__seed{seed}.json"


def ensure_calibration(env, family: str, seed: int, cfg: dict, man640: dict,
                       force: bool = False, verbose: bool = True) -> dict:
    """Fit (or reuse) the teacher's temperature on the 134 validation images.

    Returns the calibration dict. Idempotent: a second call reads the file.
    """
    dest = calibration_path(env, family, seed)
    if dest.exists() and not force:
        cal = json.loads(dest.read_text())
        if verbose:
            print(f"  [cal] {family}__seed{seed}: T={cal['temperature']:.4f} (cached)")
        return cal

    import torch
    from bruisekit import loaders as L
    from bruisekit.data import make_loader
    from bruisekit.engine import calibrate_temperature

    d = teacher_dir(env, family, seed)
    # build_for_load passes encoder_weights=None -- see handbook §15 trap 5: SMP
    # otherwise downloads ImageNet ResNet-50 only to overwrite every weight below.
    model = L.build_for_load(env, family, cfg.get("smp_encoder", "resnet50"))
    model.load_state_dict(torch.load(str(d / "best.pt"), map_location=env.device,
                                     weights_only=True))
    model.to(env.device).eval()

    val_loader = make_loader(man640["val"], env.cache640, cfg["img_size"],
                             cfg.get("eval_batch", 8), False, cfg.get("workers", 0), 0)
    cal = calibrate_temperature(model, val_loader, env.device, cfg["amp"])

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({**cal, "teacher": f"{family}__seed{seed}"}, indent=2))
    if verbose:
        print(f"  [cal] {family}__seed{seed}: T={cal['temperature']:.4f}  "
              f"NLL {cal['nll_before']:.5f} -> {cal['nll_after']:.5f}  "
              f"plausible={cal['plausible']}")

    del model
    if str(env.device).startswith("cuda"):
        torch.cuda.empty_cache()
    return cal


# ─────────────────────────────────────────────────────────────────────────────
# the shim
# ─────────────────────────────────────────────────────────────────────────────
def install_teacher_shim(env, cfg: dict, teacher_family: str | None = None,
                         teacher_seed_mode: str = "same",
                         fixed_seed: int = 0, verbose: bool = True):
    """Make `engine.load_teacher` architecture-blind and redirect it to our teacher.

    `train_run` builds the teacher path itself and only then calls `load_teacher`,
    so the path arrives already spelled `.../segformer_b2_teacher__seed{k}`. The
    shim therefore takes the SEED off that path -- the one piece of information it
    carries that we need -- and resolves the rest from `TEACHER_FOR`. Reassigning
    the name in `bruisekit.engine` (not only in its defining module) is required
    for the same reason the `build_model` shims do it: `train_run` resolves
    `load_teacher` as a module global at call time, and that global is the one
    bound in `bruisekit.engine`.

    teacher_family
        Which family to distil from. None resolves it from `TEACHER_FOR`, which is
        unambiguous while there is one arm; add the argument explicitly the moment
        a second arm with a different teacher exists.

    teacher_seed_mode
        "same"  -- student seed k distils from teacher seed k. Stage A's
                   convention (§4): the student's reported spread then includes
                   the teacher's own variance, which is part of the pipeline being
                   measured. DeepLab beats U-Net on complete misses at every seed
                   (5/5/2 vs 9/7/7), so the teacher-choice argument survives it.
        "fixed" -- every student distils from `fixed_seed`. Stage C's convention:
                   one val-selected teacher promoted for all arms. Narrower spread,
                   and it no longer measures teacher variance. Use it only if you
                   are reporting a single promoted teacher and say so.
    """
    if teacher_seed_mode not in ("same", "fixed"):
        raise ValueError(f"teacher_seed_mode must be 'same' or 'fixed', got {teacher_seed_mode!r}")

    if teacher_family is None:
        families = sorted(set(TEACHER_FOR.values()))
        if len(families) != 1:
            raise ValueError(
                f"TEACHER_FOR names {len(families)} teachers {families}; pass "
                f"teacher_family= explicitly so the shim is not guessing.")
        teacher_family = families[0]

    import torch
    import bruisekit.engine as _be
    from bruisekit import loaders as L

    # Installing twice must not wrap the wrapper: re-entering would chain a second
    # redirect onto the first and make the fall-through path depend on call order.
    previous = getattr(_be.load_teacher, "_original", _be.load_teacher)

    def load_teacher(dir_from_train_run: Path, paths: dict, device, amp: bool):
        name = Path(dir_from_train_run).name
        # "<family>__seed<k>" -- the seed is the only part we keep.
        try:
            student_seed = int(name.rsplit("seed", 1)[1])
        except (IndexError, ValueError):
            return previous(dir_from_train_run, paths, device, amp)

        family = teacher_family
        seed = student_seed if teacher_seed_mode == "same" else fixed_seed
        d = teacher_dir(env, family, seed)

        cal_file = calibration_path(env, family, seed)
        if not cal_file.exists():
            raise FileNotFoundError(
                f"teacher {family}__seed{seed} has no fitted temperature at {cal_file}. "
                f"Call ensure_calibration(env, '{family}', {seed}, CFG, MAN640) before "
                f"training -- distilling from an uncalibrated teacher would regress the "
                f"student onto a near-binary soft label, which is just the hard label "
                f"with extra steps (see engine.calibrate_temperature).")
        temperature = json.loads(cal_file.read_text())["temperature"]

        model = L.build_for_load(env, family, cfg.get("smp_encoder", "resnet50"))
        model.load_state_dict(torch.load(str(d / "best.pt"), map_location=device,
                                         weights_only=True))
        model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)

        def teacher_fn(x):
            # no_grad for the same reason as the original: the teacher is frozen,
            # and a backward graph through it is the largest avoidable allocation
            # in the distillation setup.
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=amp):
                logits = model(x)
            return torch.sigmoid(logits.float() / temperature)

        teacher_fn.temperature = temperature
        teacher_fn.source = f"{family}__seed{seed}"
        if verbose:
            print(f"  [teacher] {family}__seed{seed}  T={temperature:.4f}  "
                  f"(student seed {student_seed}, mode={teacher_seed_mode})")
        return teacher_fn

    load_teacher._original = previous
    _be.load_teacher = load_teacher
    if verbose:
        print(f"engine.load_teacher patched -> teacher {teacher_family} "
              f"(seed mode: {teacher_seed_mode}), architecture read from the family "
              f"spec rather than hardcoded to SegFormer-B2.")
    return load_teacher


def register_student_aliases():
    """Register every distilled arm in STUDENT_ARCH. Idempotent.

    The one call site the notebook needs, however many arms exist. Returns the
    families registered, in STUDENT_ARCH order.
    """
    return [register_student_alias(family, arch)
            for family, arch in STUDENT_ARCH.items()]


def register_student_alias(family: str = "fastscnn_distilled", arch: str = "fastscnn"):
    """Let the Stage E call sites treat a distilled family like its architecture.

    `install_efficient_shim`, `build_with_pretrained` and the threshold cell all
    key off a name that is a FAMILY at the call site and an ARCH inside
    `efficient_models`. For the direct baselines those two coincide. A distilled
    variant is a new family on the same architecture -- exactly as
    `segformer_b0_distilled` is a second family on the b0 architecture -- so the
    alias is registered rather than the architecture duplicated.

    Registered here, not in `efficient_models.EFFICIENT_ARCHS` itself, so that
    dict keeps meaning "the four architectures" and `self_test` keeps checking
    each architecture once instead of twice.
    """
    from bruisekit import efficient_models as EM
    from bruisekit import weights as W

    EM.EFFICIENT_ARCHS.setdefault(family, EM.EFFICIENT_ARCHS[arch])
    EM.PUBLISHED_PARAMS_M.setdefault(family, EM.PUBLISHED_PARAMS_M[arch])
    EM._HEAD_IN_CHANNELS.setdefault(family, EM._HEAD_IN_CHANNELS[arch])
    # The alias inherits the architecture's init label (kind="none"/scratch for
    # fastscnn, ImageNet backbone for lraspp_mobilenetv3) -- which stays true of
    # the distilled arm: distillation supplies a training signal, not an
    # initialisation.
    W.SOURCES.setdefault(family, W.SOURCES[arch])
    return family


# ─────────────────────────────────────────────────────────────────────────────
# the driver
# ─────────────────────────────────────────────────────────────────────────────
def train_arm(env, family: str, seeds, cfg: dict, man640: dict,
              teacher_seed_mode: str = "same", fixed_seed: int = 0,
              verbose: bool = True) -> list:
    """Calibrate the teacher, then train the arm at each seed. Returns run_ids.

    Uses `engine.train_run` unmodified, so the arm inherits the shared recipe, the
    LR split, early stopping and the whole resume contract (DONE.json to skip,
    resume.pt every few epochs). A separate driver rather than the notebook's
    `train_missing` only because the KD mix has to be `efficient_alpha`, not
    `segformer_alpha` -- see handbook §15 trap 11 on writing a new cell where a
    tested call sequence exists.
    """
    import torch
    from bruisekit import loaders as L
    from bruisekit.engine import train_run

    if not str(env.device).startswith("cuda"):
        raise RuntimeError(
            f"train_arm would train {len(tuple(seeds))} run(s) but device is "
            f"{env.device}. Distillation runs a frozen teacher on every step; this "
            f"needs a GPU session.")

    teacher_family = TEACHER_FOR[family]
    install_teacher_shim(env, cfg, teacher_family=teacher_family,
                         teacher_seed_mode=teacher_seed_mode, fixed_seed=fixed_seed,
                         verbose=verbose)

    # Calibrate every teacher seed this run will actually ask for, up front, so a
    # missing temperature fails in seconds rather than after a model is built.
    needed = sorted({s for s in seeds} if teacher_seed_mode == "same" else {fixed_seed})
    for s in needed:
        ensure_calibration(env, teacher_family, s, cfg, man640, verbose=verbose)

    spec = L.spec_for(family)
    run_cfg = {**cfg, "alpha": cfg.get("efficient_alpha", 0.6)}

    done = []
    for seed in seeds:
        run_id = f"{family}__seed{seed}"
        if verbose:
            print(f"\n{'='*70}\n{run_id}  (teacher: {teacher_family}, alpha={run_cfg['alpha']})\n{'='*70}")
        res = train_run(run_id, spec, seed, run_cfg, env.paths_for_models(),
                        man640, env.cache640, env.runs, env.device)
        if verbose:
            print(f"  -> {res.get('status', 'trained')}")
        done.append(run_id)
        if str(env.device).startswith("cuda"):
            torch.cuda.empty_cache()
    return done
