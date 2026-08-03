"""Stage D: turn per-image CSVs into the tables and figures, whatever their source.

WHY THIS EXISTS SEPARATELY FROM evaluate.py
--------------------------------------------
`bruisekit.evaluate` scores a MODEL: it needs weights, a dataloader and a GPU.
This module scores a RESULT: it needs a CSV with one row per test image. That
distinction is the whole reason the bundle is useful without a GPU -- every
headline number, fairness table, bootstrap interval and annotation-ceiling
comparison in the study is a function of per-image Dice, not of model weights.

Splitting them also removes a real hazard. When both a checkpoint and a cached
CSV exist, re-running inference and reading the CSV must agree; if the analysis
lived inside the evaluator, "which one did this table come from?" would depend on
control flow three cells away. Here it is an explicit argument.

THE COMMON SCHEMA
------------------
Five different producers wrote the per-image CSVs in this bundle, and they do not
agree on columns:

    Stage A best-seed   stem dice iou precision recall pred_positive gt_positive
    Stage A analysis    ... plus complete_miss, tp/fp/fn, pred_gt_ratio, subject, ITA
    Stage B runs        the seven-column core
    Stage C arms        the core plus subject
    Stage C teachers    the core plus subject

`normalize()` reduces all five to one schema by recomputing the derived columns
from the core rather than trusting whichever ones happen to be present. The
formulas are exact, not approximations -- verified against the shipped analysis
outputs to within 4e-12:

    tp = dice * (pred_positive + gt_positive) / 2      (definition of Dice)
    fp = pred_positive - tp
    fn = gt_positive  - tp
    complete_miss = dice == 0
    pred_gt_ratio = pred_positive / gt_positive

Recomputing is safer than reading. If a producer's `complete_miss` ever disagreed
with its own `dice`, taking the stored column would silently propagate one
producer's bug into a cross-model comparison; deriving everything from `dice`
means every model in a table was measured the same way by construction.

SUBJECT-LEVEL BOOTSTRAP, NOT IMAGE-LEVEL
-----------------------------------------
The 185 test images come from 28 subjects, and images of the same bruise are
strongly correlated. Resampling images would treat 185 correlated observations as
185 independent ones and produce confidence intervals roughly sqrt(185/28) ~ 2.6x
too narrow. `bootstrap_ci` therefore resamples SUBJECTS with replacement and
takes all of a chosen subject's images, which is the cluster bootstrap.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CORE = ["stem", "dice", "iou", "precision", "recall",
        "pred_positive_pixels", "gt_positive_pixels"]

GROUP_ORDER = ["Light (II-III)", "Intermediate (III-IV)", "Tan (IV)",
               "Brown (V)", "Dark (VI)"]


# ─────────────────────────────────────────────────────────────────────────────
# loading
# ─────────────────────────────────────────────────────────────────────────────
def normalize(df: pd.DataFrame, meta: pd.DataFrame | None = None) -> pd.DataFrame:
    """Reduce any producer's per-image CSV to the common schema.

    Parameters
    ----------
    df : a per-image table containing at least the seven CORE columns.
    meta : the test manifest, used to attach subject and ITA group. Any subject or
        skin-tone columns already in `df` are DROPPED first, so the join is the
        single source of truth for who each image belongs to. Two producers wrote
        `subject` themselves and one did not; taking the manifest's copy every
        time removes the possibility of a table mixing the two conventions.

    Returns
    -------
    A frame with the core columns plus complete_miss, tp/fp/fn_pixels,
    pred_gt_ratio, and (when `meta` is given) subject, skin_tone_category and
    ita_group_index_5.
    """
    missing = [c for c in CORE if c not in df.columns]
    if missing:
        raise ValueError(f"per-image table is missing required columns: {missing}")

    out = df[CORE].copy()
    out["tp_pixels"] = out.dice * (out.pred_positive_pixels + out.gt_positive_pixels) / 2.0
    out["fp_pixels"] = out.pred_positive_pixels - out.tp_pixels
    out["fn_pixels"] = out.gt_positive_pixels - out.tp_pixels
    out["complete_miss"] = out.dice == 0
    # gt_positive_pixels is never 0 in this test set (every image has a bruise),
    # but guard anyway so a future split cannot turn a table into infinities.
    out["pred_gt_ratio"] = np.where(out.gt_positive_pixels > 0,
                                    out.pred_positive_pixels / out.gt_positive_pixels.replace(0, np.nan),
                                    np.nan)
    if meta is not None:
        keep = [c for c in ("stem", "subject", "skin_tone_category", "ita_group_index_5", "ITA")
                if c in meta.columns]
        out = out.merge(meta[keep], on="stem", how="left", validate="one_to_one")
        if out.subject.isna().any():
            n = int(out.subject.isna().sum())
            raise ValueError(f"{n} test images have no manifest row -- stems do not match")
    return out


def load_per_image(env, reg, run_id: str, meta: pd.DataFrame | None = None) -> pd.DataFrame | None:
    """Load one run's per-image table via the registry, normalized. None if absent.

    Reads the cached CSV; it never runs inference. Use `bruisekit.evaluate` when
    you want numbers computed fresh from weights.
    """
    run = reg.get(run_id)
    if run is None or run.per_image is None:
        return None
    return normalize(pd.read_csv(run.per_image), meta)


def best_seeds(env) -> dict[str, int]:
    """Which seed each Stage A model's shipped per-image table came from.

    THIS IS NOT A CONSTANT ACROSS MODELS. The val-selected best seed is 0 for the
    three SegFormers and for yolo_sem_distilled, but 2 for yolo_sem_direct. That
    asymmetry is a live hazard: scoring yolo_sem_direct__seed0 and diffing it
    against the shipped yolo_sem_direct table compares two different models and
    shows per-image disagreements up to 0.49 Dice, which reads exactly like a
    broken inference path. It is not -- it is the wrong seed.

    So any fresh-vs-cached reconciliation must go through this mapping. Read from
    the filenames written by the selection step, so it cannot drift from them.
    """
    d = env.results / "final" / "best_seed_val_selected"
    out: dict[str, int] = {}
    if not d.exists():
        return out
    for p in d.glob("*_best_seed*_test_per_image.csv"):
        model, _, rest = p.name.partition("_best_seed")
        out[model] = int(rest.split("_")[0])
    return out


def best_run_id(env, model: str) -> str | None:
    """The run_id whose weights produced the shipped per-image table for `model`."""
    seeds = best_seeds(env)
    return f"{model}__seed{seeds[model]}" if model in seeds else None


def load_stage_a(env, reg, meta: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The five headline models at their val-selected best seed.

    Prefers `results/analysis_native/per_image_<model>.csv` because those are the
    tables the reported figures were drawn from. Falls back to the registry's
    best-seed CSVs. YOLO is taken on the NATIVE-ARGMAX path only: the custom /255
    path is retained in the bundle for provenance but is not a reporting path,
    and mixing the two in one table would compare models measured differently.
    """
    models = ["segformer_b2_teacher", "segformer_b0_direct", "segformer_b0_distilled",
              "yolo_sem_direct", "yolo_sem_distilled"]
    out = {}
    adir = env.results / "analysis_native"
    for m in models:
        p = adir / f"per_image_{m}.csv"
        if p.exists():
            out[m] = normalize(pd.read_csv(p), meta)
            continue
        d = env.results / "final" / "best_seed_val_selected"
        hits = sorted(d.glob(f"{m}_best_seed*_test_per_image.csv"))
        if hits:
            out[m] = normalize(pd.read_csv(hits[0]), meta)
    return out


def load_stage_b(env, reg, meta: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The baselines at their val-selected best seed.

    The best seed is read from `baselines_bestseed_val_selected.csv` -- selected on
    VALIDATION, never on test. Picking the seed that happens to score best on the
    185 test images would leak the test set into model selection and inflate every
    baseline number by roughly the seed spread.
    """
    sel = env.results / "baselines" / "baselines_bestseed_val_selected.csv"
    out = {}
    if sel.exists():
        for _, r in pd.read_csv(sel).iterrows():
            run_id = f"{r.model}__seed{int(r.seed)}"
            d = load_per_image(env, reg, run_id, meta)
            if d is not None:
                out[r.model] = d
    return out


def load_stage_c(env, reg, meta: pd.DataFrame, collapse_aliases: bool = True
                 ) -> dict[str, pd.DataFrame]:
    """Every finished distillation arm plus the teachers, keyed by short name.

    COLLAPSING ALIASES
    -------------------
    Some Stage C entries are the SAME model under two names. `segformer_b5_teacher`
    is not an independently trained model: it is whichever B5 seed won on
    validation, promoted into the teachers/ folder -- currently seed 123, whose
    per-image Dice it matches exactly, to the last bit. Listing both in a
    comparison table shows one result twice and invites a reader to treat it as
    two agreeing runs, which would be circular.

    So identical per-image vectors are detected and collapsed to one entry. The
    surviving name is the more descriptive one (`segformer_b5_teacher` over
    `segformer_b5__seed123`), and the discarded names are attached as
    `.attrs["aliases"]` so nothing is hidden -- callers can print them.

    Set `collapse_aliases=False` to get the raw one-entry-per-directory view.
    """
    out = {}
    for run in reg.usable("C"):
        if run.per_image is None:
            continue
        name = run.run_id.split("::", 1)[1]
        d = load_per_image(env, reg, run.run_id, meta)
        if d is not None:
            out[name] = d
    if not collapse_aliases:
        return out

    # Fingerprint on the exact per-image Dice vector, ordered by stem. Two runs
    # matching to full float precision across 185 images are the same weights;
    # genuinely distinct training runs differ in the third decimal at worst.
    seen: dict[tuple, str] = {}
    collapsed: dict[str, pd.DataFrame] = {}
    for name in sorted(out, key=lambda n: ("__seed" in n, n)):
        df = out[name]
        key = tuple(df.sort_values("stem").dice.round(12))
        if key in seen:
            keeper = seen[key]
            collapsed[keeper].attrs.setdefault("aliases", []).append(name)
        else:
            seen[key] = name
            collapsed[name] = df
    return collapsed


def alias_report(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Which names were collapsed into which, for anything `load_stage_c` merged."""
    rows = [{"kept": k, "alias": a}
            for k, df in tables.items() for a in df.attrs.get("aliases", [])]
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# statistics
# ─────────────────────────────────────────────────────────────────────────────
def summarize(per_image: pd.DataFrame) -> dict:
    """The headline row for one model: central tendency, spread, and miss rate.

    Median Dice is reported alongside the mean because the per-image distribution
    is strongly left-skewed -- a handful of complete misses drag the mean down by
    several points while the median barely moves. Reporting only one of them
    tells half the story, and the difference between them IS the story for the
    YOLO variants.
    """
    d = per_image.dice
    return {
        "n_images": int(len(per_image)),
        "mean_dice": float(d.mean()),
        "median_dice": float(d.median()),
        "iqr_dice": float(d.quantile(0.75) - d.quantile(0.25)),
        "mean_iou": float(per_image.iou.mean()),
        "mean_precision": float(per_image.precision.mean()),
        "mean_recall": float(per_image.recall.mean()),
        "complete_miss_count": int(per_image.complete_miss.sum()),
        "complete_miss_rate": float(per_image.complete_miss.mean()),
    }


def headline(tables: dict[str, pd.DataFrame], label: str = "model") -> pd.DataFrame:
    """One summary row per model, ready to print or write to CSV."""
    rows = [{label: name, **summarize(df)} for name, df in tables.items()]
    return pd.DataFrame(rows).sort_values("mean_dice", ascending=False).reset_index(drop=True)


def bootstrap_ci(per_image: pd.DataFrame, statistic="mean_dice",
                 n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Subject-level cluster bootstrap CI for a per-image statistic.

    Parameters
    ----------
    per_image : must carry a `subject` column -- pass `meta` to `normalize` first.
    statistic : "mean_dice", "median_dice" or "miss_rate".
    n_boot : resamples. 2000 is enough for a stable 95% interval; the estimate's
        Monte-Carlo error at that count is well under the interval's width.

    Returns
    -------
    dict with point, lo, hi, and n_subjects.

    Resamples SUBJECTS, not images -- see the module docstring. With 28 subjects
    the interval is genuinely wide, and that width is the honest one.
    """
    if "subject" not in per_image.columns:
        raise ValueError("bootstrap_ci needs a subject column; call normalize(df, meta) first")
    values = {"mean_dice": per_image.dice.to_numpy(float),
              "median_dice": per_image.dice.to_numpy(float),
              "miss_rate": per_image.complete_miss.to_numpy(float)}[statistic]
    agg = np.median if statistic == "median_dice" else np.mean

    # Group row indices per subject ONCE, then resample indices rather than frames.
    # The obvious implementation -- pd.concat of the chosen subjects' sub-frames on
    # every draw -- allocates a new DataFrame 2000 times per model, which on a
    # 16-model comparison is 32k allocations and enough memory churn to kill a
    # Colab kernel. Index arrays make the same draws two orders of magnitude cheaper.
    idx = per_image.groupby("subject").indices
    groups = [np.asarray(v, dtype=np.intp) for v in idx.values()]
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(groups), size=len(groups))
        draws[i] = agg(values[np.concatenate([groups[j] for j in pick])])
    return {"statistic": statistic, "point": float(agg(values)),
            "lo": float(np.quantile(draws, alpha / 2)),
            "hi": float(np.quantile(draws, 1 - alpha / 2)),
            "n_subjects": int(len(groups)), "n_boot": n_boot}


def paired_contrast(a: pd.DataFrame, b: pd.DataFrame, name_a: str, name_b: str,
                    n_boot: int = 2000, seed: int = 0) -> dict:
    """Paired subject-level bootstrap of mean-Dice difference (a - b).

    Paired because both models were scored on the SAME 185 images: resampling the
    two independently would discard that pairing and inflate the difference's
    variance, hiding real but small effects. The same resampled subject list is
    applied to both models on every draw.

    Returns the difference, its interval, and P(delta > 0) -- the probability the
    first model is better, which is more directly readable than a p-value and does
    not invite a significance/no-difference dichotomy at 28 subjects.
    """
    m = a.merge(b, on="stem", suffixes=("_a", "_b"))
    if "subject_a" in m.columns:
        m["subject"] = m["subject_a"]
    if len(m) != len(a):
        raise ValueError(f"{name_a} and {name_b} do not cover the same images "
                         f"({len(a)} vs {len(b)}, {len(m)} shared) -- cannot pair")
    # Same index-resampling trick as bootstrap_ci; see the note there.
    da = m.dice_a.to_numpy(float)
    db = m.dice_b.to_numpy(float)
    groups = [np.asarray(v, dtype=np.intp) for v in m.groupby("subject").indices.values()]
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(groups), size=len(groups))
        sel = np.concatenate([groups[j] for j in pick])
        draws[i] = da[sel].mean() - db[sel].mean()
    return {"a": name_a, "b": name_b,
            "delta": float(da.mean() - db.mean()),
            "lo": float(np.quantile(draws, 0.025)), "hi": float(np.quantile(draws, 0.975)),
            "p_a_better": float((draws > 0).mean()), "n_subjects": int(len(groups))}


def fairness_by_group(per_image: pd.DataFrame) -> pd.DataFrame:
    """Per-ITA-group median Dice, recall and miss rate for one model.

    Groups are returned in the fixed light-to-dark order rather than by size or by
    score, so the same row order holds across every model and the tables can be
    read side by side.
    """
    rows = []
    for g in GROUP_ORDER:
        sub = per_image[per_image.skin_tone_category == g]
        if not len(sub):
            continue
        rows.append({"skin_tone_category": g, "n_images": len(sub),
                     "n_subjects": sub.subject.nunique(),
                     "median_dice": float(sub.dice.median()),
                     "mean_recall": float(sub.recall.mean()),
                     "mean_precision": float(sub.precision.mean()),
                     "miss_rate": float(sub.complete_miss.mean())})
    return pd.DataFrame(rows)


def fairness_gap(per_image: pd.DataFrame) -> dict:
    """Best-minus-worst group median Dice, with a Kruskal-Wallis test across groups.

    The gap is descriptive and the test is inferential, and they routinely
    disagree here: with 28 subjects spread over five groups, a visually large gap
    is usually not significant. Both are returned so neither can be quoted alone.
    """
    from scipy.stats import kruskal
    per_group = fairness_by_group(per_image)
    if len(per_group) < 2:
        return {}
    samples = [per_image[per_image.skin_tone_category == g].dice.values
               for g in per_group.skin_tone_category]
    H, p = kruskal(*samples)
    best = per_group.loc[per_group.median_dice.idxmax()]
    worst = per_group.loc[per_group.median_dice.idxmin()]
    return {"fairness_gap": float(best.median_dice - worst.median_dice),
            "best_group": best.skin_tone_category, "worst_group": worst.skin_tone_category,
            "max_miss_rate_gap": float(per_group.miss_rate.max() - per_group.miss_rate.min()),
            "kruskal_H": float(H), "kruskal_p": float(p), "significant": bool(p < 0.05)}


def annotation_ceiling(env, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compare each model's Dice against human-vs-human agreement on the same images.

    This is the context every other table in the study needs. The models are
    separated from each other by a few Dice points; the annotators disagree with
    each other by more than that. A table of model scores without this row invites
    the reader to treat a 0.005 gap as meaningful when it sits well inside the
    noise floor of the labels themselves.

    Returns an empty frame if interlabeler_agreement_640.csv is absent, rather
    than fabricating a ceiling.
    """
    p = env.root / "interlabeler_agreement_640.csv"
    if not p.exists():
        return pd.DataFrame()
    il = pd.read_csv(p)
    rows = []
    for col in il.columns:
        if col == "stem" or not pd.api.types.is_numeric_dtype(il[col]):
            continue
        rows.append({"comparison": f"human: {col}", "n": int(il[col].notna().sum()),
                     "mean_dice": float(il[col].mean()), "median_dice": float(il[col].median())})
    for name, df in tables.items():
        rows.append({"comparison": f"model: {name}", "n": len(df),
                     "mean_dice": float(df.dice.mean()), "median_dice": float(df.dice.median())})
    return pd.DataFrame(rows).sort_values("median_dice", ascending=False).reset_index(drop=True)


def size_quintiles(per_image: pd.DataFrame, q: int = 5) -> pd.DataFrame:
    """Dice, recall and miss rate by GT-area quintile.

    Bruise size is the strongest single predictor of whether a model finds a
    bruise at all, and it is confounded with skin tone in this dataset. Any
    fairness claim that does not condition on size is measuring both at once.
    """
    d = per_image.copy()
    d["size_bin"] = pd.qcut(d.gt_positive_pixels, q,
                            labels=[f"Q{i + 1}" for i in range(q)])
    g = d.groupby("size_bin", observed=True)
    return pd.DataFrame({
        "n_images": g.size(),
        "median_gt_pixels": g.gt_positive_pixels.median(),
        "median_dice": g.dice.median(),
        "mean_recall": g.recall.mean(),
        "mean_precision": g.precision.mean(),
        "miss_rate": g.complete_miss.mean(),
    }).reset_index()


def save_all(env, name: str, obj) -> Path:
    """Write a DataFrame to `env.out/<name>.csv` and return the path."""
    env.out.mkdir(parents=True, exist_ok=True)
    p = env.out / f"{name}.csv"
    (obj if isinstance(obj, pd.DataFrame) else pd.DataFrame(obj)).to_csv(p, index=False)
    return p
