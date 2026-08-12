"""Stage P -- lesion-size-stratified miss containment, and whether it has power.

THE QUESTION, IN ONE SENTENCE
------------------------------
Complete misses in this study are not spread over the test set: 89% of them fall
in the smallest four GT-area deciles and 49% in the smallest one. This stage asks
whether the differences BETWEEN models inside that small-lesion stratum are real,
or whether 185 images from 28 subjects cannot resolve them either.

WHY IT IS AN ANALYSIS STAGE AND NOT A TRAINING STAGE
-----------------------------------------------------
Nothing here trains, loads a checkpoint, or imports torch. Every number is a
function of per-image Dice and per-image GT area, both of which are already on
disk in `FINAL_RESULT/<lineage>/per_image_*.csv`. That is the same property that
makes Stage D reproduce on a laptop, and it is why this runs in minutes.

It is worth running BEFORE any of the queued GPU work (Stage N's layer3 control,
ALS->WL distillation, a Fenwick merge) because all three are justified by the
same sentence -- "models miss small bruises, so let us fix that". If that
difference is not resolvable at n = 28 subjects, the justification is not
established and the correct next move is more data, not more mechanism.

THE TWO OUTCOMES, BOTH OF WHICH ARE USEFUL
--------------------------------------------
  CI clears zero    the small-lesion separation is real. Report it: models that
                    are indistinguishable on mean Dice separate on small-lesion
                    miss containment. That is a finding, and it names the lever.
  CI spans zero     the endpoint is underpowered at this sample size. Then the
                    honest deliverable is the MINIMUM DETECTABLE EFFECT (see
                    `min_detectable`), which converts "we found nothing" into
                    "an effect smaller than X is invisible here" -- a statement
                    about the experiment rather than about the models.

STRATUM AND ENDPOINTS ARE PRE-REGISTERED IN THIS MODULE
---------------------------------------------------------
`PRIMARY_STRATUM`, `PRIMARY_ENDPOINTS` and `CONTRASTS` are module constants for
the same reason `significance.CONTRAST_FAMILY` is: a stratum chosen after seeing
which stratum separates the models is not a test, it is a search. Deciles are cut
ONCE on the global GT-area vector, never per model, or the columns are not
comparable across models.

TWO MISS DEFINITIONS, KEPT SEPARATE
-------------------------------------
    zero_dice   dice == 0             the published endpoint (handbook Sec 1)
    empty_pred  pred_positive == 0    what the per-seed tables count
They are NOT the same number. On the current lineage `fastscnn_rgkd` has 6
zero-Dice images and 1 empty prediction: five of its six failures output a
substantial region in the wrong place, which is a worse clinical failure than
outputting nothing and is invisible if the two are collapsed.

RATES, NOT COUNTS, GO INTO THE BOOTSTRAP
------------------------------------------
A cluster bootstrap draw does not contain a fixed number of images -- resampling
subjects with replacement changes the row count on every draw. Bootstrapping a
COUNT would therefore measure how many rows the draw happened to contain.
Everything resampled here is a rate or a mean.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import report

# ─────────────────────────────────────────────────────────────────────────────
# 1. the pre-registration -- fixed before the numbers, do not edit to fit them
# ─────────────────────────────────────────────────────────────────────────────

#: Deciles of GT area, cut once on the global vector. D1 is the smallest.
N_BINS = 10

#: The stratum every primary claim is made in. D1-D4 carries 89% of all
#: complete misses in the study on the RESULT_AUGUST_08 lineage, and at 74 of 185
#: images it is the largest stratum that is still meaningfully "small". D1 alone
#: (19 images) is reported alongside but is NOT the primary -- 19 images from a
#: handful of subjects cannot support a claim.
PRIMARY_STRATUM = ("D1", "D2", "D3", "D4")

#: Reported in this order. `zero_dice_rate` is the endpoint Sec 1 says decides.
PRIMARY_ENDPOINTS = ("zero_dice_rate", "mean_recall", "median_dice")

#: Fixed contrast list. `kind` is load-bearing: `exploratory` rows were suggested
#: by looking at the descriptive table on 2026-08-07 and are therefore post-hoc.
#: Labelling them is the difference between a hypothesis and a fishing trip.
CONTRASTS: list[tuple[str, str, str, str]] = [
    ("segformer_b0_distilled", "segformer_b0_direct", "confirmatory",
     "Stage A's distillation question, restricted to small lesions. The whole-set "
     "version is NON-INFERIOR (delta +0.0017, CI [-0.0088, +0.0135]); if KD buys "
     "anything it should be here or nowhere."),
    ("segformer_b2_teacher", "segformer_b0_direct", "confirmatory",
     "Does the 7.4x larger teacher find small bruises the student misses? The "
     "whole-set contrast is INCONCLUSIVE with identical miss rates."),
    ("segformer_b0_direct", "yolo_sem_direct", "confirmatory",
     "Accuracy tier vs speed tier. The whole-set version is the one exploratory "
     "contrast in the study that already WINS (delta +0.064); this asks whether "
     "the win is entirely a small-lesion effect."),
    ("segformer_b0_direct", "unet_r50", "confirmatory",
     "vs the strongest Stage B baseline. unet_r50 has the study's best median "
     "Dice (0.833) and 7 complete misses, all of them empty predictions."),
    ("segformer_b5_teacher", "segformer_b2_teacher", "exploratory",
     "Does more teacher capacity help at the small end? B5 has the best D1 recall "
     "in the field but the worst large-lesion Dice."),
    ("lraspp_mobilenetv3_b2kd", "segformer_b5_teacher", "exploratory",
     "POST-HOC, suggested by the 2026-08-07 descriptive table: a 3.22M mobile arm "
     "showed the highest bottom-decile recall in the study (0.844 vs 0.828). This "
     "is a hypothesis generated FROM the data and is reported as such."),
]

#: Every model whose per-image CSV is read when `models=None`. Ordering is by
#: whole-set zero-Dice count so the tables read top-down from best to worst.
DEFAULT_MODELS: tuple[str, ...] = (
    "segformer_b5_teacher", "segformer_b2_teacher", "segformer_b0_distilled",
    "segformer_b0_direct", "segformer_b0_rgkd",
    "unet_r50", "deeplabv3plus_r50",
    "yolo_sem_direct", "yolo_sem_distilled",
    "lraspp_mobilenetv3", "lraspp_mobilenetv3_b2kd", "lraspp_mobilenetv3_distilled",
    "topformer_tiny", "topformer_tiny_b2kd",
    "ppmobileseg_tiny", "ppmobileseg_tiny_b2kd",
    "fastscnn", "fastscnn_b2kd",
)

#: Where the per-image CSVs come from. "auto" DISCOVERS them -- see
#: `find_lineages`. Do not hard-code a tree here.
#:
#: WHY THIS IS NOT A FIXED PATH
#: -----------------------------
#: `FINAL_RESULT/RESULT_AUGUST_08/` is where the shipped laptop bundle keeps the
#: current lineage. It is NOT where a run on ORC puts anything. There, outputs
#: land under the WORK directory -- `/scratch/$USER/bruise_work/outputs` when
#: `setup(work=...)` was given, or `<bundle>/_work/outputs` when it was not --
#: and the bundle's `FINAL_RESULT/` may not have been synced at all.
#:
#: Defaulting to the laptop path made this module raise FileNotFoundError on ORC
#: for a directory that was never going to exist there. Discovery is the fix:
#: scan every plausible root, report what was found with counts, and let the
#: caller override. A wrong guess must be visible, not fatal.
DEFAULT_LINEAGE = "auto"

#: Directories that have held per-image CSVs in this project, relative to the
#: bundle root or the work dir. Order is preference when several qualify.
_LINEAGE_HINTS: tuple[str, ...] = (
    "FINAL_RESULT/RESULT_AUGUST_08",   # shipped laptop bundle, current lineage
    "FINAL_RESULT",                    # older top-level export
    "outputs",                         # <work>/outputs  <- the ORC default
    "results/analysis_native",
    "results/final",
    "results/distill",
    "results/baselines",
)

#: Written here and nowhere else. Never `results/`, `FINAL_RESULT/` or
#: `_work/runs/` -- the same isolation Stage M and Stage N use, so an analysis
#: that turns out to be wrong leaves no trace in the directories the published
#: numbers come from.
RESULTS_DIRNAME = "LESION_SIZE_RESULTS"


# ─────────────────────────────────────────────────────────────────────────────
# 2. loading
# ─────────────────────────────────────────────────────────────────────────────
def _search_roots(env, extra: "tuple | list | None" = None) -> list[Path]:
    """Every directory worth scanning for `per_image_*.csv`, deduplicated.

    Covers BOTH layouts this project actually uses:
      laptop bundle : <root>/FINAL_RESULT/RESULT_AUGUST_08/
      ORC run       : <work>/outputs/   where work is /scratch/$USER/bruise_work
                      when setup(work=...) was given, else <root>/_work/
    """
    root, work = Path(env.root), Path(env.work)
    bases = [root, work, root / "_work"]
    if extra:
        bases = [Path(e).expanduser() for e in extra] + bases

    out: list[Path] = []
    for b in bases:
        out.append(b)
        for h in _LINEAGE_HINTS:
            out.append(b / h)
        # Any direct child of FINAL_RESULT/ (RESULT_AUGUST_08, STAGE_M_RESULTS,
        # and whatever the next lineage is called) without naming it here.
        fr = b / "FINAL_RESULT"
        if fr.is_dir():
            out.extend(sorted(p for p in fr.iterdir() if p.is_dir()))

    seen, uniq = set(), []
    for p in out:
        rp = p.resolve() if p.exists() else p
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return uniq


def find_lineages(env, extra: "tuple | list | None" = None,
                  verbose: bool = True) -> pd.DataFrame:
    """Scan for directories holding per-image CSVs. Reports, never guesses silently.

    Returns one row per candidate with the file count and how many of
    `DEFAULT_MODELS` it can answer, sorted best first. `n_default_models` is the
    ranking key rather than raw file count: a directory with 60 CSVs from a
    single stage is worth less here than one with 18 covering the whole field.
    """
    rows = []
    for d in _search_roots(env, extra):
        if not d.is_dir():
            continue
        hits = sorted(d.glob("per_image_*.csv"))
        if not hits:
            continue
        stems = {p.stem[len("per_image_"):] for p in hits}
        stems |= {s[len("distill_"):] for s in stems if s.startswith("distill_")}
        rows.append({"path": str(d),
                     "n_per_image_csv": len(hits),
                     "n_default_models": len(stems & set(DEFAULT_MODELS))})
    df = (pd.DataFrame(rows)
          .sort_values(["n_default_models", "n_per_image_csv"], ascending=False)
          .reset_index(drop=True)) if rows else pd.DataFrame(
              columns=["path", "n_per_image_csv", "n_default_models"])
    if verbose:
        if df.empty:
            print("NO per_image_*.csv FOUND under any of:")
            for d in _search_roots(env, extra):
                print(f"   {d}")
        else:
            print(f"found {len(df)} candidate lineage director"
                  f"{'y' if len(df) == 1 else 'ies'} (best first):")
            for _, r in df.iterrows():
                print(f"   {r.n_default_models:>2}/{len(DEFAULT_MODELS)} models, "
                      f"{r.n_per_image_csv:>3} csv   {r.path}")
    return df


def lineage_dir(env, lineage: str = DEFAULT_LINEAGE,
                extra: "tuple | list | None" = None) -> Path:
    """Resolve `lineage` to a directory holding per-image CSVs.

    Accepts, in order of precedence:
      an absolute path  -- used as-is, no search;
      "auto"            -- discovered by `find_lineages`, best candidate wins;
      a relative name   -- tried under the bundle root, FINAL_RESULT/, and the
                           work dir, so both layouts resolve the same name.
    """
    if lineage and lineage != "auto":
        p = Path(lineage).expanduser()
        if p.is_absolute():
            return p
        root, work = Path(env.root), Path(env.work)
        for cand in (root / "FINAL_RESULT" / lineage, root / lineage,
                     work / lineage, work / "FINAL_RESULT" / lineage,
                     root / "_work" / lineage):
            if cand.is_dir():
                return cand
        return root / "FINAL_RESULT" / lineage      # report the conventional one

    found = find_lineages(env, extra, verbose=False)
    if found.empty:
        raise FileNotFoundError(
            "LINEAGE='auto' found no per_image_*.csv anywhere. Searched:\n  "
            + "\n  ".join(str(d) for d in _search_roots(env, extra))
            + "\n\nFix: set LINEAGE to an absolute path, or pass EXTRA_SEARCH "
              "with the directory your per-image CSVs are actually in "
              "(on ORC that is usually <work>/outputs).")
    return Path(found.path.iloc[0])


def load_meta(env) -> pd.DataFrame:
    """The test manifest -- the single source of truth for subject and ITA."""
    return pd.read_csv(Path(env.root) / "manifests" / "test.csv")


def load_lineage(env, lineage: str = DEFAULT_LINEAGE,
                 models: "tuple | list | None" = None,
                 extra: "tuple | list | None" = None,
                 verbose: bool = True) -> dict[str, pd.DataFrame]:
    """Read every requested model's per-image CSV, normalized to the common schema.

    Names are resolved with a `distill_` prefix fallback: the Stage C export wrote
    `per_image_distill_segformer_b2_teacher.csv` for the same run Stage A exported
    as `per_image_segformer_b2_teacher.csv`. Preferring the unprefixed file means
    the two aliases cannot both enter one table and be counted twice.
    """
    d = lineage_dir(env, lineage, extra)
    if not d.exists():
        # Never fail with a bare "not found" for a path the caller may never have
        # had. Show what DOES exist on this host -- the laptop and ORC layouts
        # differ, and that difference is exactly what produces this error.
        found = find_lineages(env, extra, verbose=False)
        hint = ("\n\nper-image CSVs WERE found here -- set LINEAGE to one of:\n  "
                + "\n  ".join(f"{r.path}   ({r.n_default_models} models, "
                              f"{r.n_per_image_csv} csv)"
                              for _, r in found.iterrows())
                if not found.empty else
                "\n\nNothing found under:\n  "
                + "\n  ".join(str(p) for p in _search_roots(env, extra)))
        raise FileNotFoundError(
            f"lineage directory not found: {d}{hint}\n\n"
            "LINEAGE='auto' does this discovery for you.")
    meta = load_meta(env)
    want = tuple(models) if models is not None else DEFAULT_MODELS

    out: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for m in want:
        for cand in (d / f"per_image_{m}.csv", d / f"per_image_distill_{m}.csv"):
            if cand.exists():
                out[m] = report.normalize(pd.read_csv(cand), meta)
                break
        else:
            missing.append(m)
    if verbose:
        print(f"lineage : {d}")
        print(f"loaded  : {len(out)} models"
              + (f"   MISSING {len(missing)}: {', '.join(missing)}" if missing else ""))
    if not out:
        raise FileNotFoundError(f"no per_image_*.csv matched any requested model in {d}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. binning -- cut ONCE, globally
# ─────────────────────────────────────────────────────────────────────────────
def assign_bins(tables: dict[str, pd.DataFrame], n_bins: int = N_BINS,
                verbose: bool = True) -> pd.DataFrame:
    """Return the per-image key frame: stem, subject, gt area, decile, stratum flag.

    RAISES if two models disagree about an image's GT area. They are scoring the
    same 185 masks, so any disagreement means the tables came from different test
    sets or different mask versions -- and every cross-model column below would be
    silently meaningless. This has to be an exception, not a warning.
    """
    ref_name, ref = next(iter(tables.items()))
    ref = ref.sort_values("stem").reset_index(drop=True)
    key = ref[["stem", "subject", "gt_positive_pixels"]].copy()

    for name, df in tables.items():
        d = df.sort_values("stem").reset_index(drop=True)
        if len(d) != len(key) or not (d.stem.to_numpy() == key.stem.to_numpy()).all():
            raise ValueError(
                f"'{name}' does not cover the same images as '{ref_name}' "
                f"({len(d)} vs {len(key)} rows) -- these tables are not comparable.")
        if not np.array_equal(d.gt_positive_pixels.to_numpy(),
                              key.gt_positive_pixels.to_numpy()):
            n = int((d.gt_positive_pixels.to_numpy()
                     != key.gt_positive_pixels.to_numpy()).sum())
            raise ValueError(
                f"'{name}' disagrees with '{ref_name}' about GT area on {n} images. "
                "Different mask version or different test set -- refusing to bin.")

    labels = [f"D{i + 1}" for i in range(n_bins)]
    key["bin"] = pd.qcut(key.gt_positive_pixels, n_bins, labels=labels)
    key["bin"] = key["bin"].astype(str)
    key["is_primary"] = key["bin"].isin(PRIMARY_STRATUM)
    key["is_smallest"] = key["bin"].eq(PRIMARY_STRATUM[0])

    if verbose:
        frame = 640 * 640
        print(f"\nbins    : {n_bins} deciles cut once on the global GT-area vector")
        g = key.groupby("bin", observed=True).gt_positive_pixels
        for b in labels:
            if b in g.groups:
                v = g.get_group(b)
                star = " <- primary stratum" if b in PRIMARY_STRATUM else ""
                print(f"   {b:<4} n={len(v):<3} median {v.median():>8.0f} px "
                      f"({100 * v.median() / frame:5.2f}% of frame){star}")
        pr = key[key.is_primary]
        print(f"\nprimary : {PRIMARY_STRATUM} -> {len(pr)} images, "
              f"{pr.subject.nunique()} subjects")
    return key


def stratum_mask(key: pd.DataFrame, bins: "tuple | list | str") -> np.ndarray:
    """Boolean mask over the key frame's row order for a set of decile labels."""
    b = (bins,) if isinstance(bins, str) else tuple(bins)
    return key["bin"].isin(b).to_numpy()


def _aligned(df: pd.DataFrame, key: pd.DataFrame) -> pd.DataFrame:
    """One model's table in the key frame's row order. Every array here shares it."""
    return df.sort_values("stem").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 4. descriptive tables
# ─────────────────────────────────────────────────────────────────────────────
def _stats(d: pd.DataFrame, m: np.ndarray) -> dict:
    dice = d.dice.to_numpy(float)[m]
    return {
        "n": int(m.sum()),
        "median_dice": float(np.median(dice)) if m.any() else np.nan,
        "mean_dice": float(dice.mean()) if m.any() else np.nan,
        "mean_recall": float(d.recall.to_numpy(float)[m].mean()) if m.any() else np.nan,
        "mean_precision": float(d.precision.to_numpy(float)[m].mean()) if m.any() else np.nan,
        "zero_dice_n": int((dice == 0).sum()),
        "zero_dice_rate": float((dice == 0).mean()) if m.any() else np.nan,
        "empty_pred_n": int((d.pred_positive_pixels.to_numpy()[m] == 0).sum()),
    }


def headline(tables: dict[str, pd.DataFrame], key: pd.DataFrame) -> pd.DataFrame:
    """One row per model: whole set, the primary stratum, and the smallest decile."""
    allm = np.ones(len(key), bool)
    prim = key.is_primary.to_numpy()
    small = key.is_smallest.to_numpy()
    rows = []
    for name, df in tables.items():
        d = _aligned(df, key)
        r = {"model": name}
        for tag, m in (("all", allm), ("D1_D4", prim), ("D1", small)):
            for k, v in _stats(d, m).items():
                r[f"{tag}_{k}"] = v
        # wrong_place: output pixels, all of them wrong. Not the same failure as
        # predicting nothing, and the per-seed tables cannot see it.
        r["all_wrong_place_n"] = r["all_zero_dice_n"] - r["all_empty_pred_n"]
        r["D1_D4_wrong_place_n"] = r["D1_D4_zero_dice_n"] - r["D1_D4_empty_pred_n"]
        rows.append(r)
    return (pd.DataFrame(rows)
            .sort_values(["D1_D4_zero_dice_n", "all_zero_dice_n"])
            .reset_index(drop=True))


def by_bin(tables: dict[str, pd.DataFrame], key: pd.DataFrame) -> pd.DataFrame:
    """Long table: one row per (model, decile). The full per-model breakdown."""
    rows = []
    for name, df in tables.items():
        d = _aligned(df, key)
        for b in sorted(key["bin"].unique(), key=lambda s: int(s[1:])):
            m = (key["bin"] == b).to_numpy()
            rows.append({"model": name, "bin": b,
                         "median_gt_px": float(key.gt_positive_pixels[m].median()),
                         **_stats(d, m)})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 5. subject-cluster bootstrap, restricted to a stratum
# ─────────────────────────────────────────────────────────────────────────────
def _endpoint_values(d: pd.DataFrame, name: str) -> np.ndarray:
    """Per-image values whose MEAN (or median) is the endpoint."""
    if name == "zero_dice_rate":
        return (d.dice.to_numpy(float) == 0).astype(float)
    if name == "empty_pred_rate":
        return (d.pred_positive_pixels.to_numpy() == 0).astype(float)
    if name == "mean_recall":
        return d.recall.to_numpy(float)
    if name in ("mean_dice", "median_dice"):
        return d.dice.to_numpy(float)
    raise ValueError(f"unknown endpoint: {name}")


def _agg(name: str):
    return np.median if name == "median_dice" else np.mean


def _subject_groups(key: pd.DataFrame, m: np.ndarray) -> list[np.ndarray]:
    """Row indices per subject, restricted to the stratum.

    Only subjects with at least one image IN THE STRATUM are resampled. Including
    the others would add draws contributing zero rows, which inflates the variance
    of the statistic by an amount that has nothing to do with the data.
    """
    sub = key.subject.to_numpy()
    idx = np.flatnonzero(m)
    out: dict[str, list[int]] = {}
    for i in idx:
        out.setdefault(sub[i], []).append(i)
    return [np.asarray(v, dtype=np.intp) for v in out.values()]


def bootstrap_stratum(df: pd.DataFrame, key: pd.DataFrame, m: np.ndarray,
                      endpoint: str = "zero_dice_rate",
                      n_boot: int = 10000, seed: int = 0,
                      alpha: float = 0.05) -> dict:
    """Cluster-bootstrap CI for one model's endpoint inside one stratum."""
    d = _aligned(df, key)
    vals = _endpoint_values(d, endpoint)
    agg = _agg(endpoint)
    groups = _subject_groups(key, m)
    if not groups:
        raise ValueError("stratum is empty")
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(groups), size=len(groups))
        draws[i] = agg(vals[np.concatenate([groups[j] for j in pick])])
    return {"endpoint": endpoint, "n_images": int(m.sum()),
            "n_subjects": len(groups), "point": float(agg(vals[m])),
            "lo": float(np.quantile(draws, alpha / 2)),
            "hi": float(np.quantile(draws, 1 - alpha / 2)), "n_boot": n_boot}


def paired_stratum(a: pd.DataFrame, b: pd.DataFrame, key: pd.DataFrame,
                   m: np.ndarray, name_a: str = "a", name_b: str = "b",
                   endpoint: str = "zero_dice_rate",
                   n_boot: int = 10000, seed: int = 0,
                   alpha: float = 0.05) -> dict:
    """Paired cluster-bootstrap of (a - b) inside one stratum.

    Paired: the SAME resampled subject list is applied to both models on every
    draw, because both scored the same images. Resampling them independently
    would discard the pairing and hide real but small effects -- which, in a
    stratum of 74 images, is every effect there is.
    """
    da, db = _aligned(a, key), _aligned(b, key)
    va, vb = _endpoint_values(da, endpoint), _endpoint_values(db, endpoint)
    agg = _agg(endpoint)
    groups = _subject_groups(key, m)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(groups), size=len(groups))
        rows = np.concatenate([groups[j] for j in pick])
        draws[i] = agg(va[rows]) - agg(vb[rows])
    lo = float(np.quantile(draws, alpha / 2))
    hi = float(np.quantile(draws, 1 - alpha / 2))
    delta = float(agg(va[m]) - agg(vb[m]))
    # For a miss RATE, lower is better; for recall/Dice, higher is. Reporting a
    # bare "P(a better)" without saying which direction "better" is has bitten
    # this project before, so the direction is part of the record.
    lower_is_better = endpoint.endswith("_rate")

    # DEGENERATE CONTRAST -- every resample gives EXACTLY the same difference.
    # This happens when both models have zero misses in the stratum: there is no
    # variation for the bootstrap to propagate, so the interval collapses to a
    # point. Reporting that as `min_detectable = 0` would read as "this test can
    # detect any effect whatsoever", which is the exact opposite of the truth --
    # the test saw no variation at all and learned nothing about sensitivity.
    # `p_a_better` is equally misleading: (draws < 0).mean() is 0.0 when every
    # draw is 0.0, which reads as "a is never better" when they are identical.
    # Both are therefore NaN, and the row is flagged so the power summary can
    # exclude it rather than have its median dragged toward zero.
    degenerate = bool(np.ptp(draws) == 0.0)
    p_better = (float("nan") if degenerate else
                float((draws < 0).mean() if lower_is_better else (draws > 0).mean()))
    # Reuse significance.boot_p_two_sided so a Stage P p-value means exactly what
    # a Stage G p-value means. A degenerate contrast has no evidence against
    # H0: delta = 0 -- it IS delta = 0 on every draw -- so p = 1.
    from .significance import boot_p_two_sided
    p_two = 1.0 if degenerate else boot_p_two_sided(draws)
    return {
        "a": name_a, "b": name_b, "endpoint": endpoint,
        "n_images": int(m.sum()), "n_subjects": len(groups),
        "delta": delta, "lo": lo, "hi": hi,
        "lower_is_better": lower_is_better,
        "p_a_better": p_better,
        "p_two_sided": p_two,
        # A degenerate contrast never "clears zero" -- both models are identical
        # on every draw, which is a tie, not a win.
        "clears_zero": bool(not degenerate and (lo > 0 or hi < 0)),
        "degenerate": degenerate,
        # The half-width IS the minimum detectable effect for this test at this
        # sample size: any true difference smaller than it cannot clear zero.
        "min_detectable": float("nan") if degenerate else float((hi - lo) / 2.0),
        "n_boot": n_boot,
    }


def contrast_table(tables: dict[str, pd.DataFrame], key: pd.DataFrame,
                   m: np.ndarray, endpoints: "tuple | list" = PRIMARY_ENDPOINTS,
                   contrasts: "list | None" = None,
                   n_boot: int = 10000, seed: int = 0,
                   verbose: bool = True) -> pd.DataFrame:
    """Run the pre-registered contrast list at every endpoint. Skips absent models."""
    rows = []
    for a, b, kind, question in (contrasts or CONTRASTS):
        if a not in tables or b not in tables:
            if verbose:
                print(f"  SKIP {a} vs {b} -- not loaded")
            continue
        for ep in endpoints:
            r = paired_stratum(tables[a], tables[b], key, m, a, b, ep, n_boot, seed)
            r["kind"] = kind
            r["question"] = question
            rows.append(r)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)

    # MULTIPLICITY -- the same policy §8b applies, for the same reason.
    #
    # This family is 4 confirmatory pairs x 3 endpoints = 12 cells. Testing
    # twelve things at alpha = 0.05 and reporting whichever cleared is how a
    # study manufactures findings: ~0.6 of them are expected to clear from noise
    # alone. Holm-Bonferroni is applied WITHIN the confirmatory set only.
    #
    # The two exploratory pairs stay uncorrected and are LABELLED, rather than
    # being folded into the same correction. Folding them in would either
    # over-penalise the questions this stage was designed around, or launder two
    # post-hoc comparisons into confirmatory ones. One of them has already
    # reversed once (§7g.6), which is exactly why it is not in the family.
    from .significance import holm
    out["p_holm"] = np.nan
    conf = out.kind == "confirmatory"
    if conf.any():
        out.loc[conf, "p_holm"] = holm(out.loc[conf, "p_two_sided"])
    # `survives_holm` is the ONLY column a confirmatory claim may be made from.
    # `clears_zero` is the uncorrected interval and is kept so the difference
    # between the two is visible rather than quietly resolved.
    # Nullable boolean, not plain bool: an exploratory row has no Holm verdict at
    # all, and writing False there would read as "tested and failed" rather than
    # "not in the family". pandas refuses NA in a bool column, which is the right
    # instinct -- so the column is `boolean` from the start.
    out["survives_holm"] = pd.array(out.p_holm < 0.05, dtype="boolean")
    out.loc[~conf, "survives_holm"] = pd.NA

    cols = ["a", "b", "endpoint", "kind", "delta", "lo", "hi", "clears_zero",
            "p_two_sided", "p_holm", "survives_holm",
            "p_a_better", "lower_is_better", "min_detectable", "degenerate",
            "n_images", "n_subjects", "n_boot", "question"]
    return out[cols]


def min_detectable(tables: dict[str, pd.DataFrame], key: pd.DataFrame,
                   m: np.ndarray, endpoints: "tuple | list" = PRIMARY_ENDPOINTS,
                   n_boot: int = 10000, seed: int = 0) -> pd.DataFrame:
    """THE POWER ANSWER: the smallest effect this stratum could ever detect.

    Averaged over the pre-registered contrasts. If every CI in `contrast_table`
    spans zero, this is the number that turns "no difference found" into the
    honest statement: an effect smaller than X is invisible at 28 subjects, so
    the experiment cannot answer the question and more data is the only fix.
    """
    ct = contrast_table(tables, key, m, endpoints, None, n_boot, seed, verbose=False)
    if ct.empty:
        return ct
    # Degenerate rows carry NaN and are excluded from the median and the max --
    # a contrast where both models are identical on every resample says nothing
    # about how small an effect this stratum could resolve, and counting it as a
    # zero would understate the floor. They are still counted, separately, so the
    # reader can see how much of the family was uninformative.
    return (ct.groupby("endpoint")
            .agg(median_min_detectable=("min_detectable", "median"),
                 worst_min_detectable=("min_detectable", "max"),
                 n_contrasts=("min_detectable", "size"),
                 n_degenerate=("degenerate", "sum"),
                 any_clears_zero=("clears_zero", "any"))
            .reset_index())


# ─────────────────────────────────────────────────────────────────────────────
# 5b. fairness, CONDITIONED ON LESION SIZE
# ─────────────────────────────────────────────────────────────────────────────
# §8.4 states the problem and nothing in the study had yet done the arithmetic:
# bruise size is the strongest single predictor of whether a model finds a bruise
# at all, AND size is not evenly distributed across ITA groups. So a per-group
# comparison that does not condition on size is measuring both at once, and the
# published fairness numbers cannot distinguish "this model is worse on darker
# skin" from "this group happens to have smaller bruises".
#
# Stage P already cut the size deciles, so conditioning is free. Two things are
# produced, and the SECOND is only interpretable given the first:
#   size_by_ita              is the confound actually present in this dataset?
#   fairness_conditioned     does the per-group gap survive holding size fixed?

#: A cell with fewer subjects than this is reported but never bootstrapped. With
#: 28 test subjects spread over 5 ITA groups and then split by size, several
#: cells fall to 1-3 subjects; a cluster bootstrap over 2 clusters produces an
#: interval that is arithmetic, not evidence.
MIN_SUBJECTS_FOR_CI = 5

GROUP_COL = "skin_tone_category"


def size_by_ita(tables: dict[str, pd.DataFrame], key: pd.DataFrame,
                verbose: bool = True) -> pd.DataFrame:
    """Is lesion size confounded with ITA group in this test set? (§8.4)

    A property of the DATA, not of any model -- computed from the first table's
    manifest join, since every table carries the same manifest columns.

    `share_small` is the number that matters: the fraction of each group's images
    that land in the primary (small-lesion) stratum. If that varies across
    groups, every unconditioned fairness number in the study is confounded, and
    by exactly this much.
    """
    ref = _aligned(next(iter(tables.values())), key)
    d = key.copy()
    d[GROUP_COL] = ref[GROUP_COL].to_numpy()
    rows = []
    from .report import GROUP_ORDER
    for g in GROUP_ORDER:
        m = (d[GROUP_COL] == g).to_numpy()
        if not m.any():
            continue
        rows.append({
            GROUP_COL: g,
            "n_images": int(m.sum()),
            "n_subjects": int(d.subject[m].nunique()),
            "median_gt_px": float(d.gt_positive_pixels[m].median()),
            "share_small": float(d.is_primary[m].mean()),
            "n_small": int(d.is_primary[m].sum()),
        })
    out = pd.DataFrame(rows)
    if verbose and not out.empty:
        print("LESION SIZE BY ITA GROUP -- the confound itself (§8.4)")
        print("=" * 88)
        print(out.to_string(index=False))
        spread = out.share_small.max() - out.share_small.min()
        print(f"\nshare of images in the small stratum ranges "
              f"{out.share_small.min():.2f} to {out.share_small.max():.2f} "
              f"(spread {spread:.2f}).")
        print("  spread near 0 -> size is balanced across groups and an "
              "unconditioned\n                   fairness number is already clean.")
        print("  spread large  -> every unconditioned fairness number in this "
              "study is\n                   confounded, and this is the size of it.")
    return out


def fairness_conditioned(tables: dict[str, pd.DataFrame], key: pd.DataFrame,
                         models: "tuple | list | None" = None,
                         endpoint: str = "mean_recall",
                         n_boot: int = 10000, seed: int = 0,
                         verbose: bool = True) -> pd.DataFrame:
    """Per-ITA-group performance MARGINALLY and WITHIN the small-lesion stratum.

    One row per (model, group, stratum). `stratum` is "all", "D1_D4" (small) or
    "D5_D10" (large). Cells with fewer than `MIN_SUBJECTS_FOR_CI` subjects get
    NaN intervals rather than a number nobody should read.

    HOW TO READ IT
    ---------------
    Compare a model's best-minus-worst group gap in "all" against the same gap in
    "D1_D4". If the gap shrinks substantially once size is held roughly fixed,
    the marginal gap was partly a size effect wearing a skin-tone label -- which
    is precisely what §8.4 warns about and what nothing in the study had checked.

    The gap is a max-minus-min over five noisy estimates and is therefore biased
    UPWARD; it is descriptive. Do not read a gap as a test.
    """
    from .report import GROUP_ORDER
    ref = _aligned(next(iter(tables.values())), key)
    grp = ref[GROUP_COL].to_numpy()
    strata = {"all": np.ones(len(key), bool),
              "D1_D4": key.is_primary.to_numpy(),
              "D5_D10": ~key.is_primary.to_numpy()}

    want = tuple(models) if models is not None else tuple(tables)
    rows = []
    for name in want:
        if name not in tables:
            continue
        d = _aligned(tables[name], key)
        vals = _endpoint_values(d, endpoint)
        agg = _agg(endpoint)
        for sname, smask in strata.items():
            for g in GROUP_ORDER:
                m = smask & (grp == g)
                if not m.any():
                    continue
                nsub = int(key.subject[m].nunique())
                r = {"model": name, "stratum": sname, GROUP_COL: g,
                     "endpoint": endpoint, "n_images": int(m.sum()),
                     "n_subjects": nsub, "point": float(agg(vals[m])),
                     "median_gt_px": float(key.gt_positive_pixels[m].median()),
                     "lo": np.nan, "hi": np.nan}
                if nsub >= MIN_SUBJECTS_FOR_CI:
                    b = bootstrap_stratum(tables[name], key, m, endpoint,
                                          n_boot, seed)
                    r["lo"], r["hi"] = b["lo"], b["hi"]
                rows.append(r)
    out = pd.DataFrame(rows)
    if verbose and not out.empty:
        print(f"\nPER-GROUP {endpoint.upper()}, MARGINAL vs SIZE-CONDITIONED")
        print("=" * 104)
        for name in out.model.unique():
            s = out[out.model == name]
            gaps = {}
            for sname in ("all", "D1_D4", "D5_D10"):
                t = s[s.stratum == sname]
                if len(t) >= 2:
                    gaps[sname] = t.point.max() - t.point.min()
            g_all = gaps.get("all", np.nan)
            g_sm = gaps.get("D1_D4", np.nan)
            arrow = ("SHRINKS -- marginal gap was partly size"
                     if g_sm < g_all - 0.01 else
                     "HOLDS -- not explained by size"
                     if g_sm > g_all + 0.01 else "unchanged")
            print(f"\n  {name}")
            print(f"     best-worst group gap:  all {g_all:.3f}   "
                  f"small {g_sm:.3f}   large {gaps.get('D5_D10', float('nan')):.3f}"
                  f"   -> {arrow}")
            t = s[s.stratum == "D1_D4"].sort_values("point")
            for _, r in t.iterrows():
                ci = ("      [n<%d, no CI]" % MIN_SUBJECTS_FOR_CI
                      if np.isnan(r.lo) else f"   [{r.lo:.3f}, {r.hi:.3f}]")
                print(f"       small lesions  {r[GROUP_COL]:<24} "
                      f"{r.point:.3f}{ci}   n={int(r.n_images)}, "
                      f"{int(r.n_subjects)} subj")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 6. writing
# ─────────────────────────────────────────────────────────────────────────────
def results_dir(env) -> Path:
    """`LESION_SIZE_RESULTS/` at the bundle root. Created on demand."""
    p = Path(env.root) / RESULTS_DIRNAME
    p.mkdir(parents=True, exist_ok=True)
    return p


def save(env, name: str, obj: pd.DataFrame) -> Path:
    p = results_dir(env) / f"{name}.csv"
    obj.to_csv(p, index=False)
    print(f"  wrote {p.relative_to(Path(env.root))}  ({len(obj)} rows)")
    return p


# ─────────────────────────────────────────────────────────────────────────────
# 7. self-test -- asserted at build time by 71b_zip_lesion_size.py
# ─────────────────────────────────────────────────────────────────────────────
def self_test(verbose: bool = True) -> bool:
    """Synthetic data with a KNOWN answer. Catches the errors that matter here.

    Checks, in order:
      1. binning is global, not per model;
      2. a GT-area disagreement between two tables RAISES rather than warns;
      3. the paired bootstrap recovers a planted difference with the right sign;
      4. `zero_dice_rate` is scored lower-is-better;
      5. min_detectable equals the CI half-width.
    """
    rng = np.random.default_rng(0)
    n, n_sub = 200, 20
    stems = [f"S{i:03d}" for i in range(n)]
    subj = [f"P{i % n_sub:02d}" for i in range(n)]
    gt = rng.integers(1000, 40000, n).astype(float)

    def mk(dice):
        return pd.DataFrame({
            "stem": stems, "dice": dice, "iou": dice / 2,
            "precision": np.clip(dice + 0.05, 0, 1), "recall": np.clip(dice, 0, 1),
            "pred_positive_pixels": (gt * 1.05).astype(int),
            "gt_positive_pixels": gt.astype(int),
            "subject": subj, "complete_miss": dice == 0,
            "tp_pixels": 0.0, "fp_pixels": 0.0, "fn_pixels": 0.0,
            "pred_gt_ratio": 1.05,
        })

    base = np.clip(rng.normal(0.75, 0.10, n), 0, 1)
    a = base.copy()
    b = np.clip(base - 0.08, 0, 1)          # b is planted WORSE by 0.08 Dice
    small = gt < np.quantile(gt, 0.4)
    b[small & (rng.random(n) < 0.25)] = 0.0  # and misses only small lesions
    tables = {"A": mk(a), "B": mk(b)}

    ok = True

    def chk(cond, msg):
        nonlocal ok
        ok = ok and bool(cond)
        if verbose:
            print(f"  {'PASS' if cond else 'FAIL'}  {msg}")

    key = assign_bins(tables, verbose=False)
    chk(len(key) == n, "key frame covers every image")
    chk(key.is_primary.sum() == (key["bin"].isin(PRIMARY_STRATUM)).sum(),
        "primary stratum flag matches the decile labels")
    chk(set(key["bin"].unique()) == {f"D{i + 1}" for i in range(N_BINS)},
        "all ten deciles are populated")

    bad = mk(a)
    bad.loc[0, "gt_positive_pixels"] = 999999
    try:
        assign_bins({"A": mk(a), "BAD": bad}, verbose=False)
        chk(False, "GT-area disagreement raises")
    except ValueError:
        chk(True, "GT-area disagreement raises")

    m = key.is_primary.to_numpy()
    r = paired_stratum(tables["A"], tables["B"], key, m, "A", "B",
                       "mean_dice", n_boot=600, seed=0)
    chk(r["delta"] > 0, "paired bootstrap recovers the planted sign (A better)")
    chk(abs(r["min_detectable"] - (r["hi"] - r["lo"]) / 2) < 1e-12,
        "min_detectable is the CI half-width")

    rz = paired_stratum(tables["A"], tables["B"], key, m, "A", "B",
                        "zero_dice_rate", n_boot=600, seed=0)
    chk(rz["lower_is_better"], "zero_dice_rate is scored lower-is-better")
    chk(rz["delta"] < 0 and rz["p_a_better"] > 0.5,
        "A has the lower miss rate and is credited for it")

    h = headline(tables, key)
    chk(set(h.model) == {"A", "B"} and "D1_D4_zero_dice_n" in h.columns,
        "headline table has both models and the stratum columns")
    lb = by_bin(tables, key)
    chk(len(lb) == 2 * N_BINS, "long table is one row per (model, decile)")

    # A contrast where both arms are identical on every draw must not be
    # credited: the interval collapses to a point and there is no evidence.
    same = paired_stratum(tables["A"], tables["A"], key, m, "A", "A",
                          "zero_dice_rate", n_boot=200, seed=0)
    chk(same["degenerate"] and not same["clears_zero"]
        and np.isnan(same["min_detectable"]) and same["p_two_sided"] == 1.0,
        "a self-contrast is flagged degenerate, never 'clears zero'")

    # Holm must be applied WITHIN the confirmatory set and leave exploratory NaN.
    ct = contrast_table(tables, key, m,
                        ("mean_dice",),
                        [("A", "B", "confirmatory", "q1"),
                         ("B", "A", "confirmatory", "q2"),
                         ("A", "B", "exploratory", "q3")],
                        n_boot=400, seed=0, verbose=False)
    chk(ct.loc[ct.kind == "confirmatory", "p_holm"].notna().all(),
        "confirmatory rows carry a Holm-adjusted p")
    chk(ct.loc[ct.kind == "exploratory", "p_holm"].isna().all(),
        "exploratory rows are left uncorrected and labelled")
    chk((ct.loc[ct.kind == "confirmatory", "p_holm"]
         >= ct.loc[ct.kind == "confirmatory", "p_two_sided"]).all(),
        "Holm never makes a p-value smaller")

    # Fairness conditioning: every (model, group, stratum) cell present, and a
    # low-subject cell must carry no interval rather than a fake one.
    tables["A"] = tables["A"].assign(
        skin_tone_category=np.where(gt < np.median(gt), "Light (II-III)", "Dark (VI)"))
    fc = fairness_conditioned({"A": tables["A"]}, key, endpoint="mean_recall",
                              n_boot=200, seed=0, verbose=False)
    chk(set(fc.stratum) == {"all", "D1_D4", "D5_D10"},
        "fairness table covers marginal and both size strata")
    chk(((fc.n_subjects >= MIN_SUBJECTS_FOR_CI) | fc.lo.isna()).all(),
        "low-subject fairness cells carry no confidence interval")

    if verbose:
        print(f"\nself_test: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if self_test() else 1)
