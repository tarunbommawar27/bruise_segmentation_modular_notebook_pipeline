"""Stage N3 — does a different PRETRAINING PARADIGM also land at the ceiling?

THE QUESTION
-------------
Every existing piece of ceiling evidence varies capacity or architecture:

    segformer_b0  3.71 M -> 0.7663      unet_r50      -> 0.7570
    segformer_b2 27.35 M -> 0.7692      deeplabv3+_r50-> 0.7580
    segformer_b5 ~85   M -> 0.7727      Friedman p = 0.61 across the seven

23x the parameters bought +0.006 Dice. What has NOT been varied is the
pretraining paradigm: every arm above is ImageNet-supervised or scratch. Stage N2
established that DINOv2's self-supervised features are genuinely different --
0.6635 frozen-probe Dice against ResNet-50's 0.5252, a 0.14 gap in linear
decodability -- so DINOv2 is the one encoder in the study known to carry
different information.

If a demonstrably different feature space ALSO lands at ~0.77 once fine-tuned,
the ceiling is binding across capacity, architecture and pretraining, and the
bottleneck is the labels rather than the model. That is the claim this stage
exists to license, and it is a claim about where to spend the next six months.

It settles Stage N2's open question for free: DID THE PROBE RANKING TRANSFER?

WHAT IS PRE-REGISTERED (fixed here before any number is produced)
-----------------------------------------------------------------
    CEILING_BAND      = (0.75, 0.78)      the span the seven headline models occupy
    UNFREEZE_BLOCKS   = 6                 last six transformer blocks + final norm
    TARGET_GRID       = 40                both arms, so resolution is not a variable
    SEEDS             = (0,)              screening run

    dinov2_ft in band AND dermlip_ft in band
        -> CEILING CONFIRMED on a third axis; probe ranking did NOT transfer;
           encoders are exhausted as a lever. Go to label quality.
    either arm > 0.79
        -> ceiling NOT binding; Stage N2's null needs reinterpreting and
           encoders become live again.
    either arm < 0.73
        -> DO NOT read as "worse encoder" from one seed. Check for a collapsed
           run first (Stage Y seed 2 did exactly this) before concluding anything.

REPORT MISSES AND IQR ALONGSIDE DICE, NOT AFTER
------------------------------------------------
Dice is saturated; complete misses are where this study's models actually
separate (handbook 4, and the Friedman result above). DINOv2 landing at 0.77
while MOVING MISSES is a real finding that a Dice-only readout would throw away.
`ceiling_gate` therefore refuses to print a verdict without the miss columns.

WHY THIS SUBCLASSES dermprobe RATHER THAN REIMPLEMENTING
---------------------------------------------------------
Encoder loading, token->grid reshaping, the CLS/register-prefix rule and the
pixel-scale buffers were all debugged in Stage N2, including the SigLIP-loads-
DINOv2-silently failure (7f.7a) that invalidated a whole run. Re-typing that is
how you get a second silent-init bug. Exactly two things change here, and they
are the two the experiment is about: the head becomes a real decoder, and the
last N blocks unfreeze.

WHY THIS MODULE IS NOT IN THE UNIFIED BUNDLE
---------------------------------------------
Same policy as foundation.py, dermprobe.py, lesionsize.py and multiteacher.py:
experiment modules are authored in bruisekit/ but kept out of 60_build's copy
list, and ship as a standalone notebook + overlay zip. Nothing in the reporting
pipeline imports this.
"""
from __future__ import annotations

from pathlib import Path as _Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import dermprobe as _dp

# ── pre-registered constants ─────────────────────────────────────────────────

CEILING_BAND = (0.75, 0.78)
CEILING_HIGH = 0.79
CEILING_LOW = 0.73
UNFREEZE_BLOCKS = 6
TARGET_GRID = 40
DECODER_WIDTH = 256

#: The fine-tuned field this stage is asking DINOv2 to join. Handbook 7f.5.
REFERENCE_DICE = {
    "segformer_b0_direct": 0.7663,
    "segformer_b2_teacher": 0.7692,
    "segformer_b5_teacher": 0.7727,
    "unet_r50": 0.7570,
    "deeplabv3plus_r50": 0.7580,
}

#: Both arms sit at TARGET_GRID so the comparison is not confounded by output
#: resolution. DINOv2 is patch-14 (40*14 = 560); DermLIP is patch-16 (40*16 =
#: 640, which is the pipeline's own img_size, so that arm is never resampled).
ARMS = {
    "dinov2_ft": {
        "source": "dinov2",
        "corpus": "natural images, self-supervised",
        "note": "the paradigm control -- zero medical data, patch-level objective",
    },
    "dermlip_ft": {
        "source": "dermlip",
        "corpus": "dermatology image-text",
        "note": "did the frozen-probe ranking transfer once patches can adapt?",
    },
}

GATE_PRIMARY = "dinov2_ft"
GATE_TRANSFER = "dermlip_ft"


# ── the decoder ──────────────────────────────────────────────────────────────

class ConvDecodeHead(nn.Module):
    """[B,C,g,g] -> [B,1,4g,4g]. Two learned 2x stages, then a 1x1 classifier.

    Deliberately reaches stride 4 relative to the encoder grid, which at
    TARGET_GRID=40 is 160x160 -- the SAME output stride SegFormer's decode head
    produces at 640 input. If this head were left at the encoder grid and simply
    bilinear-upsampled 16x, a poor Dice would be a boundary-resolution artefact
    and would be indistinguishable from the ceiling result this stage is testing
    for. Matching the stride removes that confound.

    It is deliberately SMALL (two conv stages, ~1.5 M params against the 86 M
    encoder). Stage N2's head had to be linear so it could not paper over a weak
    encoder; here the encoder adapts, so the head must be strong enough to be a
    fair decoder and weak enough not to be the story.
    """

    def __init__(self, embed_dim: int, width: int = DECODER_WIDTH,
                 num_classes: int = 1):
        super().__init__()
        self.proj = nn.Conv2d(embed_dim, width, kernel_size=1, bias=False)
        self.stage1 = self._stage(width, width)
        self.stage2 = self._stage(width, width // 2)
        self.classifier = nn.Conv2d(width // 2, num_classes, kernel_size=1)

    @staticmethod
    def _stage(cin: int, cout: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = self.proj(feat)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.stage1(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.stage2(x)
        return self.classifier(x)


# ── partial unfreezing ───────────────────────────────────────────────────────

def find_blocks(encoder: nn.Module) -> tuple[nn.ModuleList, str]:
    """Locate the transformer block list. Raises rather than guessing.

    The three layouts this stage can meet, all present in Stage N2's checkpoints:

        HF DINOv2         encoder.encoder.layer
        open_clip + timm  visual.trunk.blocks
        open_clip native  visual.transformer.resblocks

    A silent failure here is the whole experiment: if no blocks are found and we
    quietly train nothing, the arm reports a frozen probe's score with a
    fine-tuned label on it, and it looks exactly like the ceiling result we are
    hoping to see. That is the one confusion this stage cannot afford, so this
    raises instead.
    """
    candidates = (
        ("encoder.encoder.layer", lambda m: m.encoder.layer),
        ("trunk.blocks", lambda m: m.trunk.blocks),
        ("blocks", lambda m: m.blocks),
        ("transformer.resblocks", lambda m: m.transformer.resblocks),
        ("encoder.layer", lambda m: m.encoder.layer),
        ("layers", lambda m: m.layers),
    )
    for name, get in candidates:
        try:
            blocks = get(encoder)
        except AttributeError:
            continue
        if isinstance(blocks, (nn.ModuleList, nn.Sequential)) and len(blocks) > 0:
            return blocks, name
    raise RuntimeError(
        f"Could not find the transformer block list on {type(encoder).__name__}. "
        f"Tried {[c[0] for c in candidates]}. Refusing to continue: an arm that "
        f"silently unfreezes nothing scores like a frozen probe and reads like a "
        f"confirmed ceiling, which is the one wrong answer this stage must not "
        f"produce. Add the correct path here.")


def find_final_norm(encoder: nn.Module):
    """The norm after the last block, if the tower has a separate one.

    It sits directly on top of the unfrozen blocks and its statistics are tuned
    to the frozen ones. Leaving it frozen while the blocks beneath it move is a
    mismatch that costs Dice for no reason.
    """
    for name in ("layernorm", "norm", "ln_post", "final_layer_norm"):
        mod = getattr(encoder, name, None)
        if isinstance(mod, nn.Module):
            return mod, name
    return None, ""


# ── the arms ─────────────────────────────────────────────────────────────────

class _FineTuneMixin:
    """Turns a Stage N2 frozen probe into a Stage N3 partially-fine-tuned arm."""

    def _build_head(self, num_classes: int = 1):
        # Overrides _ProbeBase's LinearProbeHead. See ConvDecodeHead's docstring
        # for why a linear head is right for N2 and wrong here.
        self.decode_head = ConvDecodeHead(self.embed_dim, DECODER_WIDTH, num_classes)

    def unfreeze_last(self, n_blocks: int = UNFREEZE_BLOCKS) -> dict:
        """Unfreeze the last `n_blocks` blocks plus the final norm.

        Returns a summary that the notebook prints and DONE.json records, because
        "how much was actually trainable" is the first thing anyone re-reading
        this result will want and the last thing anyone remembers to log.
        """
        blocks, where = find_blocks(self.encoder)
        if n_blocks > len(blocks):
            raise ValueError(
                f"asked to unfreeze {n_blocks} blocks but the tower has "
                f"{len(blocks)}. Pre-registration says {UNFREEZE_BLOCKS}; changing "
                f"it changes the experiment, so change it deliberately in the "
                f"config rather than clamping silently here.")

        for p in self.encoder.parameters():
            p.requires_grad_(False)
        for blk in blocks[len(blocks) - n_blocks:]:
            for p in blk.parameters():
                p.requires_grad_(True)

        norm, norm_name = find_final_norm(self.encoder)
        if norm is not None:
            for p in norm.parameters():
                p.requires_grad_(True)

        # `frozen` drives _ProbeBase.forward_train's no_grad path AND its train()
        # override that pins the encoder to eval(). Both must stop now.
        self.frozen = False

        trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.encoder.parameters())
        if trainable == 0:
            raise RuntimeError(
                "unfreeze_last left every encoder parameter frozen. See "
                "find_blocks' docstring: this arm would silently be a probe.")
        return {
            "blocks_found_at": where,
            "blocks_total": len(blocks),
            "blocks_unfrozen": n_blocks,
            "final_norm": norm_name or "(none found)",
            "encoder_trainable_params": trainable,
            "encoder_total_params": total,
            "encoder_trainable_fraction": round(trainable / max(total, 1), 4),
        }


class FineTunedHFViT(_FineTuneMixin, _dp.HFViTProbe):
    """DINOv2 and other HuggingFace vision towers."""


class FineTunedOpenClip(_FineTuneMixin, _dp.OpenClipProbe):
    """DermLIP, BiomedCLIP and other open_clip towers."""


def dermprobe_arch(source: str) -> str:
    """The `dermprobe.ALL_ARCHS` key that loads encoder `source`.

    Stage N2 has TWO namespaces and they are not interchangeable: `SOURCES` is
    keyed by encoder (`dinov2`), `ALL_ARCHS` by arm (`dinov2_g28`, the grid in the
    name). `ARMS` above names the ENCODER, because Stage N3 picks its own grid and
    the g28 in N2's arm names is N2's choice, not a property of the weights.

    Resolved by searching for the entry whose `encoder` matches rather than by
    formatting `f"{source}_g28"`: if N2 ever re-grids its corpus arms, a formatted
    name breaks loudly here at load time, while this keeps working and stays
    correct.
    """
    hits = [name for name, spec in _dp.CORPUS_ARCHS.items()
            if spec.get("encoder") == source]
    if not hits:
        raise KeyError(
            f"no dermprobe arm loads encoder {source!r}. Encoders with a corpus "
            f"arm: {sorted({s['encoder'] for s in _dp.CORPUS_ARCHS.values() if 'encoder' in s})}")
    if len(hits) > 1:                                          # pragma: no cover
        raise KeyError(
            f"{source!r} is loaded by {sorted(hits)} -- ambiguous. Name the arm "
            f"explicitly in ARMS rather than letting this pick one.")
    return hits[0]


def build_arm(env, arm: str, num_classes: int = 1, verbose: bool = True):
    """Build one Stage N3 arm at TARGET_GRID with the last blocks unfrozen.

    Reuses dermprobe.build_arm to obtain a correctly-loaded encoder -- including
    its guard against the 7f.7a silent-init failure -- then re-wraps that encoder
    in the fine-tuning classes above.
    """
    if arm not in ARMS:
        raise KeyError(f"unknown Stage N3 arm {arm!r}; have {sorted(ARMS)}")
    source = ARMS[arm]["source"]

    # Built at THIS stage's grid, not Stage N2's. `_dp.build_arm` reads
    # `dermprobe.TARGET_GRID` (28) at call time, and OpenClipProbe._retarget
    # RESAMPLES the position embedding in place -- so letting it build at 28 and
    # re-wrapping at 40 below would resample twice (14 -> 28 -> 40) and the arm
    # would carry a degradation that has nothing to do with the experiment.
    # Restored in `finally` because N2's gate asserts against its own constant.
    dp_grid = _dp.TARGET_GRID
    _dp.TARGET_GRID = TARGET_GRID
    try:
        probe = _dp.build_arm(env, dermprobe_arch(source),
                              num_classes=num_classes, verbose=verbose)
    finally:
        _dp.TARGET_GRID = dp_grid

    encoder = probe.encoder
    mean = probe.mean.view(-1).tolist()
    std = probe.std.view(-1).tolist()

    if isinstance(probe, _dp.OpenClipProbe):
        model = FineTunedOpenClip(encoder, mean, std, grid=TARGET_GRID,
                                  patch=probe.patch, num_classes=num_classes)
    elif isinstance(probe, _dp.HFViTProbe):
        model = FineTunedHFViT(encoder, mean, std, grid=TARGET_GRID,
                               num_classes=num_classes)
    else:
        raise TypeError(
            f"{arm}: dermprobe returned {type(probe).__name__}, which Stage N3 has "
            f"no fine-tuning wrapper for. Only ViT towers are in scope.")

    info = model.unfreeze_last(UNFREEZE_BLOCKS)
    model.init_source = getattr(probe, "init_source", "")
    model.n3_info = {"arm": arm, "grid": TARGET_GRID, "enc_size": model.enc_size,
                     "patch": model.patch, **info}
    if verbose:
        print(f"  {arm}: grid {model.grid}x{model.grid} at {model.enc_size}px "
              f"(patch {model.patch})")
        print(f"    unfrozen {info['blocks_unfrozen']}/{info['blocks_total']} blocks "
              f"at {info['blocks_found_at']}, norm={info['final_norm']}")
        print(f"    encoder trainable {info['encoder_trainable_params']:,} / "
              f"{info['encoder_total_params']:,} "
              f"({100 * info['encoder_trainable_fraction']:.1f} %)")
    return model


def install_n3_shim(env, verbose: bool = True) -> None:
    """Teach `engine.train_run`'s `build_model` and `loaders.spec_for` about the
    Stage N3 arms.

    A shim rather than an edit, for the same reason Stage N2 uses one: engine.py
    and models.py are the shared pipeline, and an experiment must not be able to
    change what the reporting stages build.

    Reassigns the name in BOTH `bruisekit.models` and `bruisekit.engine` because
    engine bound `build_model` by value at import time (engine.py's top-level
    `from bruisekit.models import build_model`), so patching models alone leaves
    train_run calling the original -- the same note install_derm_shim and
    install_foundation_shim carry. Any arch this shim does not recognise falls
    through untouched, so the N2/N3 overlays can share one kernel.

    The family table is `loaders.FAMILY_SPEC`, which is what `loaders.spec_for`
    reads and what `train_arms` looks the arms up in.
    """
    import bruisekit.engine as _be
    import bruisekit.models as _bm

    from . import loaders

    if getattr(_bm, "_n3_shim", False):
        if verbose:
            print("Stage N3 shim already installed")
        return

    original = _bm.build_model

    def build_model(arch: str, size: str, paths: dict):
        if arch == "n3":
            return build_arm(env, size, verbose=False)
        return original(arch, size, paths)

    _bm.build_model = build_model
    _be.build_model = build_model
    _bm._n3_shim = True

    for arm in ARMS:
        loaders.FAMILY_SPEC[arm] = {"arch": "n3", "size": arm, "distill": False}

    if verbose:
        print(f"Stage N3 shim installed: {', '.join(ARMS)}")


# ── training driver ──────────────────────────────────────────────────────────

def train_arms(env, cfg: dict, man640: dict, runs_dir, arms=tuple(ARMS),
               seeds=(0,), verbose: bool = True) -> list:
    """Train each Stage N3 arm through `engine.train_run`, unmodified.

    Going through the shared driver is the entire point: the arms inherit the
    same optimiser, LR split, warmup, early stopping, augmentation and resume
    contract that produced segformer_b0's 0.7663. A bespoke training loop here
    would make "did it land in the band?" unanswerable, because a difference
    could always be the recipe rather than the encoder.

    `runs_dir` is a Stage N3 directory, NOT env.runs. The reporting stages scan
    env.runs by name, and an experiment must not be able to inject arms into a
    table it is not part of.
    """
    import torch
    from bruisekit import loaders as L
    from bruisekit.engine import train_run

    if not str(env.device).startswith("cuda"):
        raise RuntimeError(
            f"train_arms would fine-tune {len(tuple(arms)) * len(tuple(seeds))} "
            f"run(s) but device is {env.device}. Unfreezing six ViT blocks needs "
            f"a GPU session; on CPU this would not finish.")

    install_n3_shim(env, verbose=verbose)

    runs_dir = _Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    done = []
    for arm in arms:
        spec = L.spec_for(arm)
        for seed in seeds:
            run_id = f"{arm}__seed{seed}"
            if verbose:
                print(f"\n{'=' * 70}\n{run_id}  ({ARMS[arm]['corpus']})\n{'=' * 70}")
            res = train_run(run_id, spec, seed, cfg, env.paths_for_models(),
                            man640, env.cache640, runs_dir, env.device)
            if verbose:
                print(f"  -> {res.get('status', 'trained')}")
            done.append(run_id)
            if str(env.device).startswith("cuda"):
                torch.cuda.empty_cache()
    return done


# ── the gate ─────────────────────────────────────────────────────────────────

def _bootstrap_ci(values: np.ndarray, groups: np.ndarray, n_boot: int,
                  rng: np.random.Generator) -> tuple[float, float]:
    """Subject-clustered bootstrap CI on a mean.

    Resamples SUBJECTS, not images: images from one subject are correlated, and
    resampling them independently would report an interval far narrower than the
    data supports. Same policy as Stages D, G and N2.
    """
    uniq = np.unique(groups)
    idx_by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        take = np.concatenate([idx_by_group[g] for g in picked])
        boots[b] = values[take].mean()
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def ceiling_gate(val_tables: dict[str, pd.DataFrame], n_boot: int = 10000,
                 seed: int = 0, band: tuple[float, float] = CEILING_BAND) -> dict:
    """Apply the pre-registered reading at the top of this module.

    `val_tables[arm]` must be a per-image frame with columns `dice`, `subject`
    and `dice == 0` recoverable -- i.e. the frame evaluate.evaluate_at_cut
    already returns, PLUS a subject column, which that function does not attach.
    Join the manifest yourself. Misses are counted as `dice == 0`, which is the
    definition the handbook publishes (NOT the per-seed empty-prediction count;
    the two differ and mixing them is a known trap).

    RAISES ON AN EMPTY MATCH, and that check is load-bearing. Handed a dict with
    no recognised arm -- an empty dict, or one keyed by run_id or filename rather
    than arm name -- every arm was skipped and the function returned
    `verdicts={}` with the MIXED fall-through reading: "arms disagree or sit just
    outside the band". That is a considered-looking verdict for a gate that never
    ran, and on 2026-08-10 it was written to `ceiling_gate.json` for a stage whose
    primary arm had in fact scored the highest Dice in the study.
    """
    rng = np.random.default_rng(seed)
    lo_band, hi_band = band
    rows, verdicts = [], {}

    matched = [a for a in ARMS if a in val_tables]
    if not matched:
        raise KeyError(
            f"none of the Stage N3 arms {sorted(ARMS)} is in the table dict "
            f"(got keys {sorted(val_tables)}). Key by ARM NAME, not run_id or "
            f"filename -- otherwise this returns a MIXED reading for a gate that "
            f"never looked at anything.")

    for arm in ARMS:
        if arm not in val_tables:
            continue
        t = val_tables[arm]
        for col in ("dice", "subject"):
            if col not in t.columns:
                raise KeyError(
                    f"{arm}: val table needs a {col!r} column. Without subject the "
                    f"CI would be image-clustered and too narrow to trust.")
        dice = t["dice"].to_numpy(dtype=float)
        groups = t["subject"].to_numpy()
        mean = float(dice.mean())
        lo, hi = _bootstrap_ci(dice, groups, n_boot, rng)

        if mean > CEILING_HIGH:
            v = "ABOVE_BAND"
        elif mean < CEILING_LOW:
            v = "BELOW_BAND_CHECK_FOR_COLLAPSE"
        elif lo_band <= mean <= hi_band:
            v = "IN_BAND"
        else:
            v = "NEAR_BAND"
        verdicts[arm] = v

        rows.append({
            "arm": arm,
            "corpus": ARMS[arm]["corpus"],
            "dice": round(mean, 4),
            "ci_lo": round(lo, 4),
            "ci_hi": round(hi, 4),
            "median": round(float(np.median(dice)), 4),
            "iqr": round(float(np.percentile(dice, 75) - np.percentile(dice, 25)), 4),
            "misses": int((dice == 0).sum()),
            "n": int(len(dice)),
            "verdict": v,
        })

    table = pd.DataFrame(rows)
    both_in = (verdicts.get(GATE_PRIMARY) == "IN_BAND"
               and verdicts.get(GATE_TRANSFER) == "IN_BAND")
    any_above = any(v == "ABOVE_BAND" for v in verdicts.values())
    any_collapse = any(v == "BELOW_BAND_CHECK_FOR_COLLAPSE" for v in verdicts.values())

    if any_collapse:
        reading = ("INCONCLUSIVE -- an arm is below the band. One seed cannot tell "
                   "a weak encoder from a collapsed run. Inspect the loss curve "
                   "and re-run that arm on another seed BEFORE reading anything "
                   "into it.")
        ceiling = None
    elif any_above:
        reading = ("CEILING NOT BINDING -- an arm cleared 0.79. Better features do "
                   "buy Dice on this task, Stage N2's null needs reinterpreting, "
                   "and encoders are live again.")
        ceiling = False
    elif both_in:
        reading = ("CEILING CONFIRMED on a third axis. A pretraining paradigm known "
                   "to carry different information (Stage N2: 0.6635 vs 0.5252 "
                   "frozen) lands in the same 0.75-0.78 band as everything else. "
                   "The frozen-probe ranking did NOT transfer. Encoders are "
                   "exhausted as a lever; the bottleneck is label quality.")
        ceiling = True
    else:
        reading = ("MIXED -- arms disagree or sit just outside the band. Report the "
                   "table, claim nothing about the ceiling.")
        ceiling = None

    transfer = None
    if {GATE_PRIMARY, GATE_TRANSFER} <= set(verdicts):
        d = float(table.set_index("arm").loc[GATE_TRANSFER, "dice"]
                  - table.set_index("arm").loc[GATE_PRIMARY, "dice"])
        transfer = {
            "delta_dice": round(d, 4),
            "probe_delta": -0.0913,
            "note": ("Stage N2's frozen probe had dermlip - dinov2 = -0.0913. If "
                     "this delta is near zero the probe ranking did not survive "
                     "fine-tuning, which is itself the answer to 7h.9 point 4."),
        }

    return {"table": table, "verdicts": verdicts, "ceiling_confirmed": ceiling,
            "reading": reading, "transfer": transfer, "band": list(band),
            "reference_dice": REFERENCE_DICE, "n_boot": n_boot, "seed": seed}


def print_gate(gate: dict) -> None:
    t = gate["table"]
    if t.empty:
        print("no arms present")
        return
    if "misses" not in t.columns:
        raise RuntimeError("refusing to print a verdict without the miss column")

    print("=" * 78)
    print(f"STAGE N3 -- ceiling band {gate['band'][0]:.2f}-{gate['band'][1]:.2f}, "
          f"{gate['n_boot']} subject-clustered resamples")
    print("=" * 78)
    print(t.to_string(index=False))

    print("\nthe field this is being compared against:")
    for k, v in gate["reference_dice"].items():
        print(f"  {k:<24} {v:.4f}")

    if gate["transfer"]:
        tr = gate["transfer"]
        print(f"\ntransfer check: dermlip - dinov2 fine-tuned = {tr['delta_dice']:+.4f}"
              f"   (frozen probe was {tr['probe_delta']:+.4f})")

    print(f"\nMISSES are the endpoint that separates models here -- Dice is "
          f"saturated.\nRead the miss column before the dice column.")
    print(f"\nREADING: {gate['reading']}")
    print("=" * 78)


# ── self-test ────────────────────────────────────────────────────────────────

def self_test(verbose: bool = True) -> bool:
    """Structural checks that need no weights, no GPU and no network."""
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name}{'  ' + detail if detail else ''}")

    # 1. decoder reaches stride 4 relative to the encoder grid
    head = ConvDecodeHead(768, DECODER_WIDTH, 1)
    out = head(torch.zeros(2, 768, TARGET_GRID, TARGET_GRID))
    check("decoder 40x40 -> 160x160", tuple(out.shape) == (2, 1, 160, 160),
          f"got {tuple(out.shape)}")
    n_head = sum(p.numel() for p in head.parameters())
    check("decoder stays small (<5 M)", n_head < 5e6, f"{n_head:,} params")

    # 2. block discovery finds each layout, and raises on none
    class HFLike(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Module()
            self.encoder.layer = nn.ModuleList([nn.Linear(4, 4) for _ in range(12)])
            self.layernorm = nn.LayerNorm(4)

    class TimmLike(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = nn.Module()
            self.trunk.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(12)])

    class ClipLike(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = nn.Module()
            self.transformer.resblocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(12)])
            self.ln_post = nn.LayerNorm(4)

    for cls, want in ((HFLike, "encoder.encoder.layer"), (TimmLike, "trunk.blocks"),
                      (ClipLike, "transformer.resblocks")):
        blocks, where = find_blocks(cls())
        check(f"find_blocks({cls.__name__})", len(blocks) == 12 and where == want, where)

    try:
        find_blocks(nn.Linear(4, 4))
        check("find_blocks raises on unknown tower", False)
    except RuntimeError:
        check("find_blocks raises on unknown tower", True)

    # 3. unfreezing touches exactly the last N blocks, and never zero
    class Tower(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Module()
            self.encoder.layer = nn.ModuleList([nn.Linear(8, 8) for _ in range(12)])
            self.layernorm = nn.LayerNorm(8)

    class Fake(_FineTuneMixin, nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Tower()
            self.embed_dim = 8
            self.frozen = True

    f = Fake()
    info = f.unfreeze_last(6)
    blocks = f.encoder.encoder.layer
    frozen_head = all(not p.requires_grad for b in blocks[:6] for p in b.parameters())
    warm_tail = all(p.requires_grad for b in blocks[6:] for p in b.parameters())
    check("first 6 blocks stay frozen", frozen_head)
    check("last 6 blocks unfrozen", warm_tail)
    check("final norm unfrozen", all(p.requires_grad for p in f.encoder.layernorm.parameters()))
    check("frozen flag cleared", f.frozen is False)
    check("reports trainable params", info["encoder_trainable_params"] > 0,
          f"{info['encoder_trainable_params']:,}")

    try:
        Fake().unfreeze_last(99)
        check("over-unfreeze raises", False)
    except ValueError:
        check("over-unfreeze raises", True)

    # 4. build_param_groups must skip frozen params -- the whole partial-unfreeze
    #    design rests on this, so assert it against the real function.
    from .models import build_param_groups

    class WithBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Linear(8, 8)
            self.decode_head = nn.Linear(8, 1)
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    groups = build_param_groups(WithBackbone(), 1e-4, 1e-3, 0.01)
    n_grouped = sum(len(g["params"]) for g in groups)
    check("build_param_groups drops frozen params", n_grouped == 2, f"{n_grouped} tensors")

    # 5. the gate reads each pre-registered outcome correctly
    def frame(vals):
        return pd.DataFrame({"dice": vals,
                             "subject": [f"s{i % 10}" for i in range(len(vals))]})

    in_band = [0.77] * 100
    high = [0.83] * 100
    low = [0.60] * 100

    g = ceiling_gate({"dinov2_ft": frame(in_band), "dermlip_ft": frame(in_band)}, n_boot=200)
    check("both in band -> ceiling confirmed", g["ceiling_confirmed"] is True)
    g = ceiling_gate({"dinov2_ft": frame(high), "dermlip_ft": frame(in_band)}, n_boot=200)
    check("one above -> ceiling not binding", g["ceiling_confirmed"] is False)
    g = ceiling_gate({"dinov2_ft": frame(low), "dermlip_ft": frame(in_band)}, n_boot=200)
    check("one below -> inconclusive, not 'worse'", g["ceiling_confirmed"] is None
          and "collapse" in g["reading"].lower())

    zeros = [0.0] * 7 + [0.8] * 93
    g = ceiling_gate({"dinov2_ft": frame(zeros), "dermlip_ft": frame(in_band)}, n_boot=200)
    check("misses counted as dice == 0",
          int(g["table"].set_index("arm").loc["dinov2_ft", "misses"]) == 7)

    try:
        ceiling_gate({"dinov2_ft": pd.DataFrame({"dice": [0.7]})}, n_boot=10)
        check("gate demands a subject column", False)
    except KeyError:
        check("gate demands a subject column", True)

    if verbose:
        print(f"\nStage N3 self-test: {'PASS' if ok else 'FAIL'}")
    return bool(ok)
