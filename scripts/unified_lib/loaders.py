"""Turn a registry entry into a live model, and a live model into test numbers.

WHAT THIS ADDS OVER bruisekit.models.build_model
-------------------------------------------------
`build_model` constructs an architecture. Getting from a `Run` to a scored model
needs four more things that differ per family and are easy to get subtly wrong:

  1. WHICH architecture to build from a run_id (b0 vs b2 is not in the filename
     in any parseable way -- it is in the family name, and only by convention).
  2. NOT fetching pretrained weights when we are about to overwrite them anyway.
  3. The OPERATING POINT -- the val-fitted logit cut, without which a SegFormer's
     test score is not the reported score.
  4. The right INFERENCE PATH -- SegFormer thresholds a logit; native YOLO takes
     an argmax and has no threshold at all.

Doing this in the notebook would mean four near-identical blocks with a subtle
difference in each. Doing it here means one tested path per family.

THE OFFLINE RULE
-----------------
`segmentation_models_pytorch` defaults to `encoder_weights="imagenet"`, which
issues an HTTP request to fetch ResNet-50 the moment the model is constructed.
When we are loading a trained checkpoint, every one of those weights is about to
be overwritten by `load_state_dict` -- so the download is pure latency at best,
and a hard failure on an offline or firewalled machine at worst. `build_for_load`
therefore passes `encoder_weights=None`. Training is different and still uses
ImageNet init; only the LOAD path skips it.

WHY THE CUT IS READ FROM DISK AND NEVER RE-FITTED
--------------------------------------------------
Each SegFormer-family run ships an `operating_point.json` holding the threshold
chosen on the 134 validation images. Re-sweeping it on test would pick a better
cut and report a better number -- and that number would be meaningless, because
the threshold would have been fitted on the data it is scored on. The cut is an
input here, never an output.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

# torch is imported INSIDE the functions that need it, not here. Importing this
# module is a prerequisite for the notebook's Stage A/B cells, but a reader who
# only wants Stage D's tables should not need a deep-learning stack installed to
# get past the import block. Everything in this module that touches a tensor
# imports torch at the top of its own body.

# Which architecture each Stage A/B family is. The run_id alone cannot tell you:
# "segformer_b0_distilled" and "segformer_b0_direct" differ only in training, and
# "unet_r50" says nothing about the encoder to a builder that takes (arch, size).
FAMILY_SPEC = {
    "segformer_b2_teacher": {"arch": "segformer", "size": "b2", "distill": False},
    "segformer_b0_direct": {"arch": "segformer", "size": "b0", "distill": False},
    "segformer_b0_distilled": {"arch": "segformer", "size": "b0", "distill": True},
    "unet_r50": {"arch": "unet", "size": None, "distill": False},
    "deeplabv3plus_r50": {"arch": "deeplabv3plus", "size": None, "distill": False},
    # Stage E -- mobile-grade baselines. Direct training only: `distill` is False
    # by construction, not by default.
    "ppmobileseg_tiny": {"arch": "ppmobileseg_tiny", "size": None, "distill": False},
    "topformer_tiny": {"arch": "topformer_tiny", "size": None, "distill": False},
    "lraspp_mobilenetv3": {"arch": "lraspp_mobilenetv3", "size": None, "distill": False},
    "fastscnn": {"arch": "fastscnn", "size": None, "distill": False},
    # Stage F -- distilled mobile students. Each is a second FAMILY on an existing
    # ARCHITECTURE, exactly as segformer_b0_distilled is a second family on b0.
    # `teacher` is read by distill_efficient.install_teacher_shim; engine.train_run
    # ignores it and passes it through into the run's config.json, which is where
    # a reader needs it. See distill_efficient.py for why DeepLabV3+, and why the
    # scratch-init (fastscnn) and pretrained (lraspp) students share one teacher.
    "fastscnn_distilled": {"arch": "fastscnn", "size": None, "distill": True,
                           "teacher": "deeplabv3plus_r50"},
    "lraspp_mobilenetv3_distilled": {"arch": "lraspp_mobilenetv3", "size": None,
                                     "distill": True,
                                     "teacher": "deeplabv3plus_r50"},
    # The two students Stage F never gave a DeepLabV3+ arm. Defined here so the
    # teacher axis is 2x4 rather than 2x2 -- without them, "B2 beats DeepLab as a
    # teacher" would rest on two students. Registered, therefore reported MISSING
    # until trained; whether they are trained is a seed-budget decision.
    "topformer_tiny_distilled": {"arch": "topformer_tiny", "size": None,
                                 "distill": True, "teacher": "deeplabv3plus_r50"},
    "ppmobileseg_tiny_distilled": {"arch": "ppmobileseg_tiny", "size": None,
                                   "distill": True, "teacher": "deeplabv3plus_r50"},

    # ── Stage H -- reliability-gated KD, and the SegFormer-B2 teacher axis ─────
    # `kd: "gated"` is documentation, not dispatch: engine.train_run passes the
    # whole spec through into the run's config.json and ignores keys it does not
    # know, and the gate is selected by reliability_kd.arm() at the call site. A
    # reader opening a run directory should not have to consult a second table to
    # find out which loss produced it.
    #
    # Each `*_rgkd` arm carries the SAME alpha as the control named in
    # reliability_kd.CONTROL_FOR, so the pair differs in one thing only.
    "segformer_b0_rgkd": {"arch": "segformer", "size": "b0", "distill": True,
                          "teacher": "segformer_b2_teacher", "kd": "gated"},
    # Defined but NOT REGISTERED as an arm: absent from reliability_kd's
    # TEACHER_FOR / STAGE_H_FAMILIES / CONTROL_FOR, so nothing schedules it and a
    # Stage E+H session never needs Ultralytics. The spec is kept because
    # re-enabling the arm is three lines there and should not also mean
    # re-deriving its architecture here. See reliability_kd's module docstring for
    # what the study gives up meanwhile.
    "yolo_sem_rgkd": {"arch": "yolo", "size": None, "distill": True,
                      "teacher": "segformer_b2_teacher", "kd": "gated"},
    # Stage Y -- YOLO26-large. `size` is carried rather than left None (as the
    # nano arms leave it) so that anything reading the spec can tell the two
    # backbones apart without pattern-matching the family name.
    "yolo_sem_l_direct": {"arch": "yolo", "size": "l", "distill": False},
    "yolo_sem_l_distilled": {"arch": "yolo", "size": "l", "distill": True,
                             "teacher": "segformer_b2_teacher"},
    "fastscnn_rgkd": {"arch": "fastscnn", "size": None, "distill": True,
                      "teacher": "segformer_b2_teacher", "kd": "gated"},
    "lraspp_mobilenetv3_rgkd": {"arch": "lraspp_mobilenetv3", "size": None,
                                "distill": True,
                                "teacher": "segformer_b2_teacher", "kd": "gated"},
    "topformer_tiny_rgkd": {"arch": "topformer_tiny", "size": None, "distill": True,
                            "teacher": "segformer_b2_teacher", "kd": "gated"},
    "ppmobileseg_tiny_rgkd": {"arch": "ppmobileseg_tiny", "size": None, "distill": True,
                              "teacher": "segformer_b2_teacher", "kd": "gated"},
    # Plain response KD from B2 -- the gated arms' teacher-matched controls.
    "fastscnn_b2kd": {"arch": "fastscnn", "size": None, "distill": True,
                      "teacher": "segformer_b2_teacher", "kd": "response"},
    "lraspp_mobilenetv3_b2kd": {"arch": "lraspp_mobilenetv3", "size": None,
                                "distill": True,
                                "teacher": "segformer_b2_teacher", "kd": "response"},
    "topformer_tiny_b2kd": {"arch": "topformer_tiny", "size": None, "distill": True,
                            "teacher": "segformer_b2_teacher", "kd": "response"},
    "ppmobileseg_tiny_b2kd": {"arch": "ppmobileseg_tiny", "size": None, "distill": True,
                              "teacher": "segformer_b2_teacher", "kd": "response"},
}

# Families whose architecture lives in bruisekit.efficient_models. Every mobile
# arm, direct or distilled, whatever its teacher -- this list drives
# `build_for_load`'s routing, which cares only about where the architecture is
# defined, not about how the run was trained.
EFFICIENT_FAMILIES = ("ppmobileseg_tiny", "topformer_tiny",
                      "lraspp_mobilenetv3", "fastscnn",
                      "fastscnn_distilled", "lraspp_mobilenetv3_distilled",
                      "topformer_tiny_distilled", "ppmobileseg_tiny_distilled",
                      "fastscnn_rgkd", "lraspp_mobilenetv3_rgkd",
                      "topformer_tiny_rgkd", "ppmobileseg_tiny_rgkd",
                      "fastscnn_b2kd", "lraspp_mobilenetv3_b2kd",
                      "topformer_tiny_b2kd", "ppmobileseg_tiny_b2kd")


def spec_for(family: str) -> dict:
    """The (arch, size, distill) spec for a Stage A/B family name."""
    if family not in FAMILY_SPEC:
        raise KeyError(f"no architecture spec for family {family!r}; "
                       f"known: {sorted(FAMILY_SPEC)}")
    return dict(FAMILY_SPEC[family])


def build_for_load(env, family: str, smp_encoder: str = "resnet50"):
    """Construct an untrained model of the right shape, WITHOUT downloading anything.

    SegFormer is built from the local HF config folder in pretrained_weights/, so
    it is offline already. SMP is forced to `encoder_weights=None` -- see the
    module docstring. The returned model is on CPU; move it yourself.
    """
    spec = spec_for(family)
    if spec["arch"] == "segformer":
        from bruisekit.models import build_model
        return build_model(spec["arch"], spec["size"], env.paths_for_models())
    if family in EFFICIENT_FAMILIES:
        # Untrained on purpose: a checkpoint is about to overwrite every weight,
        # so fetching the ImageNet backbone first would be wasted I/O and would
        # make an offline load depend on the network. Training takes the other
        # path -- efficient_models.install_efficient_shim(env) -- which DOES load
        # the pretrained backbone.
        from bruisekit.efficient_models import build_efficient
        return build_efficient(spec["arch"], num_classes=1)
    from bruisekit.smp_models import SMPNet
    return SMPNet(spec["arch"], encoder=smp_encoder, encoder_weights=None)


# ── SegFormer module-layout reconciliation ───────────────────────────────────
# HuggingFace refactored SegformerForSemanticSegmentation's internal module names,
# so a checkpoint saved under one `transformers` version raises a missing/unexpected
# key RuntimeError against a skeleton built by another. This is the same mapping
# `scripts/26_migrate_legacy_segformer_checkpoints.py` derived and verified against
# two independent checkpoints; it is applied here at LOAD time rather than by
# rewriting the .pt, because which direction is needed depends on the machine and
# a bundle is copied between machines with different versions installed.
#
# "OLD" is the encoder.block.N.M layout, "NEW" is stages.N.blocks.M.
_SEGFORMER_OLD_TO_NEW: list[tuple[str, str]] = [
    (r"encoder\.patch_embeddings\.(\d+)\.", r"stages.\1.patch_embeddings."),
    (r"encoder\.layer_norm\.(\d+)\.", r"stages.\1.layer_norm."),
    (r"encoder\.block\.(\d+)\.(\d+)\.layer_norm_1\.", r"stages.\1.blocks.\2.layernorm_before."),
    (r"encoder\.block\.(\d+)\.(\d+)\.layer_norm_2\.", r"stages.\1.blocks.\2.layernorm_after."),
    (r"encoder\.block\.(\d+)\.(\d+)\.attention\.self\.query\.",
     r"stages.\1.blocks.\2.attention.q_proj."),
    (r"encoder\.block\.(\d+)\.(\d+)\.attention\.self\.key\.",
     r"stages.\1.blocks.\2.attention.k_proj."),
    (r"encoder\.block\.(\d+)\.(\d+)\.attention\.self\.value\.",
     r"stages.\1.blocks.\2.attention.v_proj."),
    (r"encoder\.block\.(\d+)\.(\d+)\.attention\.self\.sr\.",
     r"stages.\1.blocks.\2.attention.sequence_reduction.sequence_reduction."),
    (r"encoder\.block\.(\d+)\.(\d+)\.attention\.self\.layer_norm\.",
     r"stages.\1.blocks.\2.attention.sequence_reduction.layer_norm."),
    (r"encoder\.block\.(\d+)\.(\d+)\.attention\.output\.dense\.",
     r"stages.\1.blocks.\2.attention.o_proj."),
    (r"encoder\.block\.(\d+)\.(\d+)\.mlp\.dense1\.", r"stages.\1.blocks.\2.mlp.fc1."),
    (r"encoder\.block\.(\d+)\.(\d+)\.mlp\.dense2\.", r"stages.\1.blocks.\2.mlp.fc2."),
    # The depthwise conv keeps its own name but still moves under stages.N.blocks.M.
    # Omitting this rule leaves those keys old-style, which fails the exact key-set
    # check and blocks an otherwise valid remap.
    (r"encoder\.block\.(\d+)\.(\d+)\.mlp\.dwconv\.dwconv\.",
     r"stages.\1.blocks.\2.mlp.dwconv.dwconv."),
    (r"decode_head\.linear_c\.(\d+)\.", r"decode_head.linear_projections.\1."),
]

# The inverse. Written out rather than derived, because inverting a regex with
# capture groups programmatically is a good way to produce a mapping that is
# almost right -- and "almost right" here means weights loaded into the wrong
# tensor, which scores plausibly and is wrong.
_SEGFORMER_NEW_TO_OLD: list[tuple[str, str]] = [
    (r"stages\.(\d+)\.patch_embeddings\.", r"encoder.patch_embeddings.\1."),
    (r"stages\.(\d+)\.layer_norm\.", r"encoder.layer_norm.\1."),
    (r"stages\.(\d+)\.blocks\.(\d+)\.layernorm_before\.", r"encoder.block.\1.\2.layer_norm_1."),
    (r"stages\.(\d+)\.blocks\.(\d+)\.layernorm_after\.", r"encoder.block.\1.\2.layer_norm_2."),
    (r"stages\.(\d+)\.blocks\.(\d+)\.attention\.q_proj\.",
     r"encoder.block.\1.\2.attention.self.query."),
    (r"stages\.(\d+)\.blocks\.(\d+)\.attention\.k_proj\.",
     r"encoder.block.\1.\2.attention.self.key."),
    (r"stages\.(\d+)\.blocks\.(\d+)\.attention\.v_proj\.",
     r"encoder.block.\1.\2.attention.self.value."),
    (r"stages\.(\d+)\.blocks\.(\d+)\.attention\.sequence_reduction\.sequence_reduction\.",
     r"encoder.block.\1.\2.attention.self.sr."),
    (r"stages\.(\d+)\.blocks\.(\d+)\.attention\.sequence_reduction\.layer_norm\.",
     r"encoder.block.\1.\2.attention.self.layer_norm."),
    (r"stages\.(\d+)\.blocks\.(\d+)\.attention\.o_proj\.",
     r"encoder.block.\1.\2.attention.output.dense."),
    (r"stages\.(\d+)\.blocks\.(\d+)\.mlp\.fc1\.", r"encoder.block.\1.\2.mlp.dense1."),
    (r"stages\.(\d+)\.blocks\.(\d+)\.mlp\.fc2\.", r"encoder.block.\1.\2.mlp.dense2."),
    (r"stages\.(\d+)\.blocks\.(\d+)\.mlp\.dwconv\.dwconv\.",
     r"encoder.block.\1.\2.mlp.dwconv.dwconv."),
    (r"decode_head\.linear_projections\.(\d+)\.", r"decode_head.linear_c.\1."),
]


def _looks_like_segformer_layout_clash(err: Exception) -> bool:
    """Is this RuntimeError the transformers-layout rename, or a real mismatch?

    Only the rename is safe to paper over. A genuine architecture mismatch -- a
    different backbone size, a different head -- must still raise, because
    remapping it would either fail again or, worse, half-succeed.
    """
    s = str(err)
    return ("Missing key(s)" in s or "Unexpected key(s)" in s) and (
        "segformer" in s or "decode_head" in s)


def _remap(state: dict, rules: list[tuple[str, str]]) -> dict:
    out = {}
    for k, v in state.items():
        nk = k
        for pattern, repl in rules:
            cand = re.sub(pattern, repl, nk)
            if cand != nk:
                nk = cand
                break
        out[nk] = v
    return out


def _reconcile_segformer_layout(model, state: dict) -> tuple[dict, str]:
    """Return (state_dict that loads strictly, which direction was applied).

    Both directions are tried and each is CHECKED against the model's own key set
    before being returned -- a remap that merely reduces the number of missing
    keys is not a fix, and accepting one would load a checkpoint into a skeleton
    it does not match. Raises the original problem as a ValueError if neither
    direction produces an exact key-set match.
    """
    want = set(model.state_dict().keys())
    for rules, name in ((_SEGFORMER_OLD_TO_NEW, "OLD->NEW"),
                        (_SEGFORMER_NEW_TO_OLD, "NEW->OLD")):
        cand = _remap(state, rules)
        if set(cand.keys()) == want:
            return cand, name
    have = set(state.keys())
    raise ValueError(
        "SegFormer checkpoint does not match the model built by the installed "
        "transformers, and neither layout remapping reconciles them.\n"
        f"  model wants {len(want)} keys, checkpoint has {len(have)}\n"
        f"  example missing: {sorted(want - have)[:2]}\n"
        f"  example unexpected: {sorted(have - want)[:2]}\n"
        "  This is not the known encoder/stages rename -- check the transformers "
        "version and that the checkpoint is the model you think it is.")


def load_model(env, run, smp_encoder: str = "resnet50"):
    """Load a `Run`'s checkpoint into a model, ready for eval. Returns (model, cut).

    Parameters
    ----------
    run : a `bruisekit.registry.Run` in the WEIGHTS tier with kind "segformer" or "smp".

    Returns
    -------
    (model, cut) -- `cut` is the val-fitted logit threshold from the run's
    operating_point.json. Never re-derived here.

    `weights_only=True` is passed to `torch.load` deliberately: these checkpoints
    are plain state dicts, and refusing to unpickle arbitrary objects means a
    corrupted or swapped file fails as a load error instead of executing.
    """
    import torch
    if run.weights is None:
        raise ValueError(f"{run.run_id} has no checkpoint (tier={run.tier})")
    model = build_for_load(env, run.family, smp_encoder)
    state = torch.load(str(run.weights), map_location="cpu", weights_only=True)
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        if "size mismatch" in str(e) or not _looks_like_segformer_layout_clash(e):
            raise
        state, direction = _reconcile_segformer_layout(model, state)
        print(f"  NOTE  {run.run_id}: remapped {direction} SegFormer keys to match the "
              f"installed transformers. The checkpoint is fine; the module layout "
              f"differs between the version that saved it and the one here.")
        model.load_state_dict(state)
    model.to(env.device).eval()

    # The cut comes from whichever threshold file the registry actually found.
    # Two dialects exist and they are NOT interchangeable: `operating_point.json`
    # stores `cut`, a logit; the teacher store's `threshold.json` stores
    # `threshold`, a probability. sigmoid(cut) == threshold, so reading one as
    # the other does not raise -- it silently moves the decision boundary, which
    # on this data shifts the complete-miss rate by several points. `read_cut`
    # is the single place that knows the difference.
    from bruisekit.registry import read_cut

    tf = getattr(run, "threshold_file", None)
    if tf is None:                                   # pre-search-path Run objects
        tf = run.weights.parent / "operating_point.json"
        tf = tf if tf.exists() else None
    if tf is None:
        raise FileNotFoundError(
            f"{run.run_id}: checkpoint present but no threshold file "
            f"(operating_point.json or threshold.json) beside it at "
            f"{run.weights.parent}. Its test score is undefined without the "
            f"val-fitted cut.")
    cut = read_cut(tf)
    if cut is None:
        raise ValueError(f"{run.run_id}: {tf} yielded no usable cut")
    return model, cut


def score_segformer(env, run, cfg, cache640_manifests, smp_encoder="resnet50") -> pd.DataFrame:
    """Run a SegFormer/SMP checkpoint over the 185 test images at its val-fitted cut.

    Returns the per-image table. Requires the 640 cache to have been built --
    these models are scored on exactly the tensors they were trained on, not on a
    fresh resize, so that a re-scored number is comparable to the shipped one.

    TWO MEMORY GUARDS, NEITHER OF WHICH CHANGES A NUMBER
    -----------------------------------------------------
    1. `torch.inference_mode()`. `evaluate_at_cut` runs its forward pass without
       a no-grad context, so every intermediate activation is retained for a
       backward pass that never comes. On a 40 GB A100 that is invisible; on CPU,
       SegFormer-B2's decode head tries to allocate 1.2 GB in a single `cat` and
       the interpreter dies. Wrapping the call is numerically identical to not
       wrapping it -- inference mode only suppresses graph recording -- and it is
       applied HERE rather than by editing evaluate.py, which is shipped verbatim
       because its outputs are the published ones.
    2. A CPU-safe batch size. The training-era default of 8 assumes GPU memory.
       Batch size cannot affect per-image scores -- each image is thresholded and
       scored independently -- so shrinking it on CPU costs only wall-clock.
    """
    import gc

    import torch
    from bruisekit.data import make_loader
    from bruisekit.evaluate import evaluate_at_cut

    model, cut = load_model(env, run, smp_encoder)
    on_cuda = str(env.device).startswith("cuda")
    amp = bool(cfg.get("amp", True)) and on_cuda
    batch = cfg.get("eval_batch", 8) if on_cuda else cfg.get("eval_batch_cpu", 2)
    loader = make_loader(cache640_manifests["test"], env.cache640, cfg["img_size"],
                         batch, False, cfg.get("workers", 0), 0)
    with torch.inference_mode():
        per_image, _ = evaluate_at_cut(model, loader, env.device, cut, amp)

    # Models are scored in a loop; without an explicit drop the previous model's
    # parameters stay reachable through the local frame until the next collection,
    # which on CPU is the difference between fitting and not.
    del model, loader
    gc.collect()
    if on_cuda:
        torch.cuda.empty_cache()
    return per_image


def score_yolo_native(env, run, cfg, test_manifest: pd.DataFrame) -> pd.DataFrame:
    """Run a native YOLO checkpoint over the 185 test images via argmax.

    No threshold argument exists and none is missing: the native path takes
    `argmax` over the semantic head, which is parameter-free. That is precisely
    why this is the sole reporting path for YOLO -- there is no operating point
    to fit, so there is nothing to overfit.

    Predicts from the NATIVE-resolution images (env.data), not the 640 cache,
    because Ultralytics does its own letterboxing internally and feeding it
    pre-resized images would apply the resize twice.
    """
    import bruisekit.yolo_native as yn
    per_image, _ = yn.predict_native_argmax(run.weights, test_manifest,
                                            env.data, cfg["img_size"])
    return per_image


def score_run(env, run, cfg, cache640_manifests, test_manifest, smp_encoder="resnet50"):
    """Score any WEIGHTS-tier run, dispatching on its kind.

    "efficient" shares the SegFormer path deliberately: those four models emit a
    single logit at input resolution and carry a val-fitted cut, which is exactly
    the contract `score_segformer` implements. Giving them their own near-identical
    branch would be four more lines that can drift out of sync.
    """
    if run.kind in ("segformer", "smp", "efficient"):
        return score_segformer(env, run, cfg, cache640_manifests, smp_encoder)
    if run.kind == "yolo":
        return score_yolo_native(env, run, cfg, test_manifest)
    raise ValueError(f"no scoring path for kind={run.kind!r} ({run.run_id})")


def make_teacher_prob_fn(env, seed: int, cfg: dict):
    """Native-resolution teacher probability, for building distilled YOLO pseudo-masks.

    Returns a callable `(image_path, (H, W)) -> prob[H, W]` backed by the
    SAME-SEED calibrated B2 teacher.

    WHY THE SAME SEED, NOT THE BEST ONE
    ------------------------------------
    Using one strong teacher for all three student seeds would make the students'
    spread across seeds narrower than the pipeline's true variance, because every
    student would inherit the identical teacher. Pairing seed k's student with
    seed k's teacher means the reported spread includes the teacher's own
    variance -- which is part of what is being measured.

    WHY NATIVE RESOLUTION
    ----------------------
    Ultralytics letterboxes internally. The pseudo-mask must therefore be built at
    the image's own resolution and handed to YOLO unresized, or the resize is
    applied twice and the mask no longer registers with the image.
    """
    import torch
    import cv2
    import numpy as np
    from bruisekit.engine import load_teacher

    run_dir = env.ckpt / "final" / f"segformer_b2_teacher__seed{seed}"
    if not (run_dir / "best.pt").exists():
        run_dir = env.runs / f"segformer_b2_teacher__seed{seed}"
    if not (run_dir / "best.pt").exists():
        raise FileNotFoundError(
            f"distilled YOLO needs the seed-{seed} B2 teacher, and no checkpoint was "
            f"found for segformer_b2_teacher__seed{seed}. Train it first.")

    amp = bool(cfg.get("amp", True)) and str(env.device).startswith("cuda")
    teacher = load_teacher(run_dir, env.paths_for_models(), env.device, amp)
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    size = cfg["img_size"]

    def prob_fn(img_path, native_hw):
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        r = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        x = torch.from_numpy(((r - mean) / std).transpose(2, 0, 1)).unsqueeze(0).float().to(env.device)
        # load_teacher's wrapper already applies the calibrated temperature and sigmoid.
        prob = teacher(x)[0, 0].float().cpu().numpy()
        return np.clip(cv2.resize(prob, (native_hw[1], native_hw[0]),
                                  interpolation=cv2.INTER_LINEAR), 0, 1)

    return prob_fn


def build_cache640(env, manifests: dict, force: bool = False) -> dict:
    """Materialise the 640x640 PNG cache and return manifests that point at it.

    The cache is a deterministic resize of data/ -- bilinear for images, nearest
    for masks -- which is exactly what the training dataloader would compute on
    the fly. Precomputing it means every epoch reads a 640 PNG instead of
    decoding and resizing a multi-megapixel JPEG, and (more importantly) that
    training and evaluation see bit-identical pixels.

    Nearest-neighbour for masks is not interchangeable with bilinear: bilinear
    would produce fractional values at every boundary pixel, and thresholding
    those back to binary silently erodes or dilates small bruises -- the exact
    population the complete-miss metric is about.

    Returns
    -------
    {"train"|"val"|"test": DataFrame} with image_path/mask_path relative to
    `env.cache640`, ready for `make_loader(df, env.cache640, ...)`.
    """
    import cv2
    import numpy as np

    out = {}
    todo = 0
    for split, df in manifests.items():
        idir = env.cache640 / split / "images"
        mdir = env.cache640 / split / "masks"
        idir.mkdir(parents=True, exist_ok=True)
        mdir.mkdir(parents=True, exist_ok=True)
        for _, r in df.iterrows():
            ip, mp = idir / f"{r.stem}.png", mdir / f"{r.stem}.png"
            if force or not ip.exists():
                im = cv2.imread(str(env.data / r.image_path), cv2.IMREAD_COLOR)
                if im is None:
                    raise FileNotFoundError(f"unreadable image: {env.data / r.image_path}")
                cv2.imwrite(str(ip), cv2.resize(im, (640, 640), interpolation=cv2.INTER_LINEAR))
                todo += 1
            if force or not mp.exists():
                m = cv2.imread(str(env.data / r.mask_path), cv2.IMREAD_GRAYSCALE)
                if m is None:
                    raise FileNotFoundError(f"unreadable mask: {env.data / r.mask_path}")
                if m.ndim == 3:
                    m = m[..., 0]
                b = (m > 0).astype(np.uint8)
                cv2.imwrite(str(mp), cv2.resize(b, (640, 640), interpolation=cv2.INTER_NEAREST) * 255)
        d = df.copy()
        d["image_path"] = d["stem"].map(lambda s, sp=split: f"{sp}/images/{s}.png")
        d["mask_path"] = d["stem"].map(lambda s, sp=split: f"{sp}/masks/{s}.png")
        out[split] = d
    if todo:
        print(f"  built {todo} cache files -> {env.cache640}")
    else:
        print(f"  640 cache already complete -> {env.cache640}")
    return out
