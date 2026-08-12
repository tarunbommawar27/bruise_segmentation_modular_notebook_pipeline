"""One table for every model this project has ever scored.

WHAT THIS IS FOR
----------------
The study's endpoints are scattered. `LESION_SIZE_RESULTS/` covers 18 arms.
`STAGE_N4_RESULTS/` covers four more. The Stage C distillation grid, the Stage H
reliability-gated arms and the Stage M multi-teacher arms have per-image CSVs on
disk and have never been through a size or miss analysis at all -- which matters,
because those families were called NULL on Dice, and Dice is the endpoint this
study has argued all along is saturated (Friedman p = 0.61 across the seven
headline models). "We looked and it was null" and "we did not look" are
different claims and only one of them is currently true.

So: discover every per-image table wherever it lives, normalise it, and emit ONE
row per model carrying mean and median Dice, zero-Dice count and rate, complete-
miss rate, small-lesion recall, and the ITA fairness block -- marginal AND
conditioned on lesion size.

THE PROBLEM THIS MODULE EXISTS TO SOLVE, AND WHY IT IS NOT lesionsize
----------------------------------------------------------------------
`lesionsize.assign_bins` RAISES when two tables disagree about an image's GT
area, and that is correct for its job: it bins one coherent lineage, and a
disagreement there means the tables came from different test sets, so every
cross-model column would be silently meaningless.

Here the input is deliberately heterogeneous -- 50-odd CSVs from several
lineages, splits and mask versions -- so a hard raise would mean the whole sweep
dies on the first stale file. Instead this module DETECTS COHORTS: tables are
grouped by (stem set, GT-area vector), the largest coherent group becomes the
reference cohort and is binned once, and every other group is binned separately
and flagged `comparable = False` with its cohort id. Nothing is silently mixed,
and nothing is silently dropped either -- `discovery` records every file found,
what it was matched to, and every file skipped WITH THE REASON.

DEDUPLICATION
-------------
The same model appears under several filenames (`per_image_x.csv`,
`x_best_seed0_test_per_image.csv`, `reference_x_test_per_image.csv`). They are
collapsed by comparing the actual Dice VECTOR, not the name: byte-identical
scores are the same run regardless of what the file is called. The shortest name
wins and the rest are recorded as aliases, so a reader can see that the collapse
happened rather than wondering where a file went.

WHAT IT DOES NOT DO
-------------------
It does not run inference and it does not re-fit any threshold. Every number
here is computed from a per-image CSV that some other stage already produced, at
that stage's own operating point. A model whose CSV was written at a badly-fitted
cut will look bad here, and correctly so -- but that is a property of the run, not
of this analysis.

Written to `ALL_MODELS_RESULTS/`. Never to `results/`, `FINAL_RESULT/`,
`LESION_SIZE_RESULTS/` or any stage's own directory.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIRNAME = "ALL_MODELS_RESULTS"

#: Decile scheme, matched to `lesionsize` so rows from the two are comparable.
N_BINS = 10
PRIMARY_STRATUM = ("D1", "D2", "D3", "D4")     # the small-lesion stratum
SMALLEST = "D1"

#: Every directory that has ever held a per-image CSV in this project, relative
#: to the bundle root or to the work dir. Order is preference, not exclusion:
#: all of them are scanned.
#:
#: WHY THIS IS A SCAN AND NOT A FIXED PATH. `FINAL_RESULT/` is where the shipped
#: laptop bundle keeps the current lineage and it does not exist on ORC at all;
#: there, outputs land under the WORK directory. A hard-coded tree made
#: `lesionsize` raise FileNotFoundError on ORC for a directory that was never
#: going to be there. Discovery is the fix: scan everything plausible, report
#: what was found with counts, and let the caller override.
SEARCH_HINTS: tuple[str, ...] = (
    "results/analysis_native",
    "results/final",
    "results/final/best_seed_val_selected",
    "results/distill",
    "results/baselines",
    "results",
    "FINAL_RESULT/RESULT_AUGUST_08",
    "FINAL_RESULT",
    "STAGE_M_RESULTS",
    "STAGE_N3_RESULTS",
    "STAGE_N3_RESULTS/tables",
    "STAGE_N4_RESULTS",
    "STAGE_N4_RESULTS/tables",
    "FOUNDATION_RESULTS",
    "DERM_PROBE_RESULTS",
    "LESION_SIZE_RESULTS",
    "outputs",
    "new_outputs",
)

#: Filename patterns that are per-image tables.
GLOBS = ("per_image*.csv", "*_per_image*.csv", "*per_image.csv")

#: Files that look like per-image tables and are not reporting paths.
#: `custom255` is the retired YOLO preprocessing path: it is kept in the bundle
#: for provenance but is NOT a reporting path (handbook 3, and the memo that the
#: native-argmax path is the sole one), and mixing it in would put two
#: differently-measured versions of the same model in one table.
EXCLUDE_SUBSTRINGS = ("custom255",)

GROUP_ORDER = ["Light (II-III)", "Intermediate (III-IV)", "Tan (IV)",
               "Brown (V)", "Dark (VI)"]
GROUP_COL = "skin_tone_category"
MIN_SUBJECTS_FOR_CI = 5

CORE = ["stem", "dice", "iou", "precision", "recall",
        "pred_positive_pixels", "gt_positive_pixels"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. discovery
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Found:
    """One per-image CSV on disk, and what became of it."""
    path: Path
    model: str
    n_rows: int
    split: str = ""
    cohort: str = ""
    status: str = "ok"            # ok | skipped | alias
    reason: str = ""
    alias_of: str = ""


def _clean_name(p: Path) -> str:
    """A model name from a filename, with the bookkeeping suffixes stripped.

    Handles the four conventions this project has used, in order of how much they
    need removing. Deliberately conservative: anything it cannot confidently
    strip is left alone, because a name collision would silently merge two models
    and that is far worse than an ugly name.
    """
    n = p.stem
    for pre in ("per_image__", "per_image_", "test_per_image__",
                "val_per_image__", "reference_"):
        if n.startswith(pre):
            n = n[len(pre):]
    for suf in ("_test_per_image", "_val_per_image", "_per_image"):
        if n.endswith(suf):
            n = n[: -len(suf)]
    # `<model>_best_seed<N>` -> `<model>`; the seed is recorded separately by
    # whoever selected it, and keeping it here would split one model into three.
    if "_best_seed" in n:
        n = n.split("_best_seed")[0]
    # `<arm>__seed<N>` -> keep the seed: Stage N3/N4 genuinely run several seeds
    # of the same arm and they are different runs, not aliases.
    return n or p.stem


def search_roots(env, extra=None) -> list[Path]:
    """Every directory worth scanning, deduplicated and existing."""
    roots: list[Path] = []
    bases = [Path(env.root)]
    work = getattr(env, "work", None)
    if work:
        bases.append(Path(work))
    for e in (extra or ()):
        bases.append(Path(e))
    for b in bases:
        if not b.exists():
            continue
        roots.append(b)
        for hint in SEARCH_HINTS:
            d = b / hint
            if d.exists() and d.is_dir():
                roots.append(d)
    seen, out = set(), []
    for r in roots:
        k = str(r.resolve()).lower()
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def discover(env, extra_roots=None, verbose: bool = True) -> list[Found]:
    """Find every per-image CSV under every plausible root. Reads nothing yet."""
    hits: dict[str, Found] = {}
    for root in search_roots(env, extra_roots):
        for pat in GLOBS:
            for p in sorted(root.glob(pat)):
                if not p.is_file():
                    continue
                key = str(p.resolve()).lower()
                if key in hits:
                    continue
                if any(x in p.name for x in EXCLUDE_SUBSTRINGS):
                    hits[key] = Found(p, _clean_name(p), 0, status="skipped",
                                      reason="not a reporting path (see "
                                             "EXCLUDE_SUBSTRINGS)")
                    continue
                hits[key] = Found(p, _clean_name(p), 0)
    out = list(hits.values())
    if verbose:
        ok = sum(1 for f in out if f.status == "ok")
        print(f"discovery: {len(out)} candidate file(s) under "
              f"{len(search_roots(env, extra_roots))} root(s); {ok} to read")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 2. loading, normalising, cohorting
# ─────────────────────────────────────────────────────────────────────────────
def _normalize(df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """`report.normalize`, with the manifest joined. Raises on a stem mismatch."""
    from .report import normalize
    return normalize(df, meta)


def _signature(t: pd.DataFrame) -> str:
    """A cohort id from the images scored and their GT areas.

    Two tables share a cohort iff they scored the SAME stems with the SAME mask
    areas. That is exactly the condition under which a shared decile cut is
    meaningful, so making it the grouping key means a cross-cohort comparison
    cannot happen by accident.
    """
    d = t.sort_values("stem")
    h = hashlib.sha256()
    h.update(",".join(d.stem.astype(str)).encode())
    h.update(np.asarray(d.gt_positive_pixels, dtype=np.int64).tobytes())
    return h.hexdigest()[:10]


def load_all(env, found: list[Found], manifests: dict[str, pd.DataFrame],
             verbose: bool = True) -> tuple[dict[str, pd.DataFrame], list[Found]]:
    """Read every candidate, match it to a split, normalise, dedupe, cohort.

    `manifests` maps split name -> manifest frame. A table is matched to the
    split whose stems it is a subset of; a table matching none is SKIPPED with
    that recorded, rather than being force-joined to the 185-image manifest and
    silently producing NaN subjects.
    """
    stems = {k: set(v.stem) for k, v in manifests.items()}
    tables: dict[str, pd.DataFrame] = {}
    by_dice: dict[str, str] = {}

    for f in found:
        if f.status == "skipped":
            continue
        try:
            raw = pd.read_csv(f.path)
        except Exception as exc:                              # pragma: no cover
            f.status, f.reason = "skipped", f"unreadable: {exc}"
            continue
        missing = [c for c in CORE if c not in raw.columns]
        if missing:
            f.status = "skipped"
            f.reason = f"not a per-image table (missing {missing[:3]})"
            continue

        f.n_rows = len(raw)
        got = set(raw.stem.astype(str))
        split = next((s for s, ss in stems.items() if got and got <= ss), "")
        if not split:
            f.status = "skipped"
            f.reason = (f"{len(got)} stems match no manifest split "
                        f"({', '.join(f'{k}={len(v)}' for k, v in stems.items())})")
            continue
        f.split = split

        try:
            t = _normalize(raw, manifests[split])
        except Exception as exc:
            f.status, f.reason = "skipped", f"normalize failed: {exc}"
            continue

        # Collapse byte-identical runs. The name is not trusted; the scores are.
        key = hashlib.sha256(
            np.asarray(t.sort_values("stem").dice, dtype=np.float64).tobytes()
        ).hexdigest()[:16] + f"|{split}"
        if key in by_dice:
            keep = by_dice[key]
            f.status, f.alias_of = "alias", keep
            f.reason = "identical Dice vector to " + keep
            continue

        name = f.model
        if name in tables:                        # same name, different scores
            name = f"{name}#{_signature(t)[:4]}"
            f.reason = f"name collision; renamed to {name}"
        f.model = name
        by_dice[key] = name
        f.cohort = _signature(t)
        tables[name] = t

    # Cohort labels: the largest group on the largest split is the reference.
    counts: dict[tuple[str, str], int] = {}
    for f in found:
        if f.status == "ok":
            counts[(f.split, f.cohort)] = counts.get((f.split, f.cohort), 0) + 1
    ref = max(counts, key=lambda k: (counts[k], k[0] == "test")) if counts else None
    for f in found:
        if f.status == "ok":
            f.cohort = ("REFERENCE" if (f.split, f.cohort) == ref
                        else f"{f.split}:{f.cohort}")

    if verbose:
        ok = [f for f in found if f.status == "ok"]
        print(f"loaded {len(ok)} distinct model table(s)")
        for (split, coh), n in sorted(counts.items(), key=lambda x: -x[1]):
            tag = "  <- reference cohort" if (split, coh) == ref else ""
            print(f"    {split:<6} cohort {coh}  {n:>3} model(s){tag}")
        skipped = [f for f in found if f.status == "skipped"]
        aliases = [f for f in found if f.status == "alias"]
        if aliases:
            print(f"  collapsed {len(aliases)} alias file(s) "
                  f"(identical Dice vectors)")
        if skipped:
            print(f"  skipped {len(skipped)} file(s):")
            for f in skipped[:10]:
                print(f"      {f.path.name}: {f.reason}")
            if len(skipped) > 10:
                print(f"      ... and {len(skipped) - 10} more "
                      f"(see discovery_log.csv)")
    return tables, found


# ─────────────────────────────────────────────────────────────────────────────
# 3. binning -- per cohort, never across
# ─────────────────────────────────────────────────────────────────────────────
def assign_bins(tables: dict[str, pd.DataFrame], found: list[Found],
                n_bins: int = N_BINS) -> dict[str, pd.DataFrame]:
    """One decile key per cohort, cut on that cohort's own GT-area vector.

    Cutting per cohort rather than globally is the whole point: a decile boundary
    computed over two different mask versions belongs to neither of them.
    """
    cohort_of = {f.model: f.cohort for f in found if f.status == "ok"}
    keys: dict[str, pd.DataFrame] = {}
    labels = [f"D{i + 1}" for i in range(n_bins)]
    for coh in sorted(set(cohort_of.values())):
        members = [m for m, c in cohort_of.items() if c == coh and m in tables]
        if not members:
            continue
        ref = tables[members[0]].sort_values("stem").reset_index(drop=True)
        key = ref[["stem", "subject", "gt_positive_pixels", GROUP_COL]].copy() \
            if GROUP_COL in ref.columns else \
            ref[["stem", "subject", "gt_positive_pixels"]].copy()
        n_unique = key.gt_positive_pixels.nunique()
        bins = min(n_bins, max(2, n_unique))
        key["bin"] = pd.qcut(key.gt_positive_pixels, bins,
                             labels=labels[:bins], duplicates="drop").astype(str)
        key["is_primary"] = key["bin"].isin(PRIMARY_STRATUM)
        key["is_smallest"] = key["bin"].eq(SMALLEST)
        keys[coh] = key
    return keys


# ─────────────────────────────────────────────────────────────────────────────
# 4. the statistics
# ─────────────────────────────────────────────────────────────────────────────
def _boot_ci(values: np.ndarray, subjects: np.ndarray, stat, n_boot: int,
             seed: int) -> tuple[float, float]:
    """Subject-clustered bootstrap CI. Resamples SUBJECTS, never images.

    Images from one subject are correlated -- several photographs of one bruise --
    so resampling them independently reports an interval far narrower than the
    data supports. Same policy as Stages D, G, N2, N3 and N4.
    """
    if len(values) == 0:
        return (np.nan, np.nan)
    uniq = np.unique(subjects)
    if len(uniq) < 2:
        return (np.nan, np.nan)
    idx = {g: np.flatnonzero(subjects == g) for g in uniq}
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx[g] for g in pick])
        draws[i] = stat(values[rows])
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _block(d: pd.DataFrame, mask: np.ndarray, tag: str) -> dict:
    """Every endpoint, restricted to one stratum. Prefixed with `tag`."""
    if not mask.any():
        return {f"{tag}_n": 0}
    dice = d.dice.to_numpy(float)[mask]
    pred = d.pred_positive_pixels.to_numpy()[mask]
    zero = dice == 0
    empty = pred == 0
    return {
        f"{tag}_n": int(mask.sum()),
        f"{tag}_mean_dice": float(dice.mean()),
        f"{tag}_median_dice": float(np.median(dice)),
        f"{tag}_iqr_dice": float(np.percentile(dice, 75) - np.percentile(dice, 25)),
        f"{tag}_mean_recall": float(d.recall.to_numpy(float)[mask].mean()),
        f"{tag}_mean_precision": float(d.precision.to_numpy(float)[mask].mean()),
        f"{tag}_zero_dice_n": int(zero.sum()),
        f"{tag}_zero_dice_rate": float(zero.mean()),
        f"{tag}_empty_pred_n": int(empty.sum()),
        # Output pixels, all of them wrong. NOT the same failure as predicting
        # nothing, and the per-seed tables cannot see it. This is the number a
        # clinician would care about most.
        f"{tag}_wrong_place_n": int((zero & ~empty).sum()),
    }


def summarize(tables: dict[str, pd.DataFrame], found: list[Found],
              keys: dict[str, pd.DataFrame], n_boot: int = 10000,
              seed: int = 0, verbose: bool = True) -> pd.DataFrame:
    """THE TABLE. One row per model, every endpoint, marginal and stratified."""
    from scipy.stats import kruskal

    cohort_of = {f.model: f.cohort for f in found if f.status == "ok"}
    split_of = {f.model: f.split for f in found if f.status == "ok"}
    path_of = {f.model: str(f.path) for f in found if f.status == "ok"}
    rows = []

    for i, (name, t) in enumerate(sorted(tables.items()), 1):
        coh = cohort_of.get(name, "")
        key = keys.get(coh)
        d = t.sort_values("stem").reset_index(drop=True)
        n = len(d)
        r: dict = {
            "model": name,
            "split": split_of.get(name, ""),
            "cohort": coh,
            "comparable": coh == "REFERENCE",
            "n_images": n,
            "n_subjects": int(d.subject.nunique()),
            "source_file": path_of.get(name, ""),
        }

        allm = np.ones(n, bool)
        r.update(_block(d, allm, "all"))

        if key is not None and len(key) == n:
            r.update(_block(d, key.is_primary.to_numpy(), "D1_D4"))
            r.update(_block(d, key.is_smallest.to_numpy(), "D1"))
            r.update(_block(d, ~key.is_primary.to_numpy(), "D5_D10"))

        # CIs on the two endpoints anyone will quote.
        subj = d.subject.to_numpy()
        dice = d.dice.to_numpy(float)
        lo, hi = _boot_ci(dice, subj, np.mean, n_boot, seed)
        r["mean_dice_ci_lo"], r["mean_dice_ci_hi"] = lo, hi
        lo, hi = _boot_ci((dice == 0).astype(float), subj, np.mean, n_boot, seed)
        r["zero_dice_rate_ci_lo"], r["zero_dice_rate_ci_hi"] = lo, hi

        # ── fairness, marginal ───────────────────────────────────────────────
        if GROUP_COL in d.columns and d[GROUP_COL].notna().any():
            per, samples = [], []
            for g in GROUP_ORDER:
                m = (d[GROUP_COL] == g).to_numpy()
                if not m.any():
                    continue
                gd = dice[m]
                per.append((g, float(np.median(gd)),
                            float(d.recall.to_numpy(float)[m].mean()),
                            float((gd == 0).mean())))
                samples.append(gd)
                tagg = g.split(" ")[0]
                r[f"median_dice__{tagg}"] = float(np.median(gd))
                r[f"mean_recall__{tagg}"] = float(d.recall.to_numpy(float)[m].mean())
                r[f"miss_rate__{tagg}"] = float((gd == 0).mean())
            if len(per) >= 2:
                meds = [p[1] for p in per]
                misses = [p[3] for p in per]
                best = per[int(np.argmax(meds))]
                worst = per[int(np.argmin(meds))]
                # The gap is a max-minus-min over five noisy estimates and is
                # therefore biased UPWARD. It is descriptive. The Kruskal test is
                # the inferential one and they routinely disagree here, so both
                # are carried and neither can be quoted alone.
                r["fairness_gap_median_dice"] = float(max(meds) - min(meds))
                r["fairness_best_group"] = best[0]
                r["fairness_worst_group"] = worst[0]
                r["fairness_miss_rate_gap"] = float(max(misses) - min(misses))
                try:
                    H, p = kruskal(*samples)
                    r["kruskal_H"], r["kruskal_p"] = float(H), float(p)
                    r["kruskal_significant"] = bool(p < 0.05)
                except Exception:                             # pragma: no cover
                    pass

            # ── fairness, conditioned on lesion size ─────────────────────────
            # 8.4: lesion size is confounded with ITA group in this test set, so
            # an unconditioned gap measures both at once. If the gap SHRINKS
            # inside the small stratum, the marginal gap was partly a size
            # effect wearing a skin-tone label.
            if key is not None and len(key) == n:
                rec = d.recall.to_numpy(float)
                small = key.is_primary.to_numpy()
                for tag, m0 in (("all", allm), ("small", small)):
                    vals = [rec[m0 & (d[GROUP_COL] == g).to_numpy()].mean()
                            for g in GROUP_ORDER
                            if (m0 & (d[GROUP_COL] == g).to_numpy()).any()]
                    if len(vals) >= 2:
                        r[f"recall_gap_{tag}"] = float(max(vals) - min(vals))
                ga, gs = r.get("recall_gap_all"), r.get("recall_gap_small")
                if ga is not None and gs is not None:
                    r["gap_shrinks_with_size"] = bool(gs < ga - 0.01)

        rows.append(r)
        if verbose and (i % 10 == 0 or i == len(tables)):
            print(f"  {i}/{len(tables)} models", end="\r", flush=True)

    if verbose:
        print()
    out = pd.DataFrame(rows)
    # Miss containment first, then Dice. Handbook 4: misses are where the models
    # in this study actually separate; ordering by Dice would put the table in
    # the order of the endpoint that does not discriminate.
    sort_cols = [c for c in ("comparable", "all_zero_dice_n", "all_mean_dice")
                 if c in out.columns]
    asc = [False, True, False][: len(sort_cols)]
    return out.sort_values(sort_cols, ascending=asc).reset_index(drop=True)


def by_decile(tables, found, keys) -> pd.DataFrame:
    """Long form: one row per (model, decile)."""
    cohort_of = {f.model: f.cohort for f in found if f.status == "ok"}
    rows = []
    for name, t in sorted(tables.items()):
        key = keys.get(cohort_of.get(name, ""))
        if key is None or len(key) != len(t):
            continue
        d = t.sort_values("stem").reset_index(drop=True)
        for b in sorted(key["bin"].unique(), key=lambda s: int(s[1:])):
            m = (key["bin"] == b).to_numpy()
            rows.append({"model": name, "bin": b,
                         "median_gt_px": float(key.gt_positive_pixels[m].median()),
                         **_block(d, m, "s")})
    return pd.DataFrame(rows)


def by_group(tables, found, keys) -> pd.DataFrame:
    """Long form: one row per (model, stratum, ITA group)."""
    cohort_of = {f.model: f.cohort for f in found if f.status == "ok"}
    rows = []
    for name, t in sorted(tables.items()):
        d = t.sort_values("stem").reset_index(drop=True)
        if GROUP_COL not in d.columns:
            continue
        key = keys.get(cohort_of.get(name, ""))
        strata = {"all": np.ones(len(d), bool)}
        if key is not None and len(key) == len(d):
            strata["D1_D4"] = key.is_primary.to_numpy()
            strata["D5_D10"] = ~key.is_primary.to_numpy()
        for sname, smask in strata.items():
            for g in GROUP_ORDER:
                m = smask & (d[GROUP_COL] == g).to_numpy()
                if not m.any():
                    continue
                rows.append({"model": name, "stratum": sname, GROUP_COL: g,
                             "n_subjects": int(d.subject[m].nunique()),
                             **_block(d, m, "g")})
    return pd.DataFrame(rows)


def discovery_log(found: list[Found]) -> pd.DataFrame:
    """Every file considered, what it became, and why. Nothing silently dropped."""
    return pd.DataFrame([{
        "path": str(f.path), "model": f.model, "status": f.status,
        "split": f.split, "cohort": f.cohort, "n_rows": f.n_rows,
        "alias_of": f.alias_of, "reason": f.reason,
    } for f in found]).sort_values(["status", "model"]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 5. the one call that does everything
# ─────────────────────────────────────────────────────────────────────────────
def run(env, manifests: dict[str, pd.DataFrame] | None = None,
        extra_roots=None, n_boot: int = 10000, seed: int = 0,
        verbose: bool = True) -> dict:
    """Discover, load, bin, summarise. Returns every frame; writes nothing."""
    if manifests is None:
        manifests = {s: pd.read_csv(Path(env.manifests) / f"{s}.csv")
                     for s in ("test", "val", "train")}
    found = discover(env, extra_roots, verbose=verbose)
    tables, found = load_all(env, found, manifests, verbose=verbose)
    if not tables:
        raise RuntimeError(
            "no per-image table matched any manifest split. Either the roots are "
            "wrong -- pass extra_roots=['/scratch/<user>/bruise_work'] -- or the "
            "CSVs use stems this bundle's manifests do not contain. "
            "discovery_log() says which, per file.")
    keys = assign_bins(tables, found)
    if verbose:
        print("\nsummarising")
    summary = summarize(tables, found, keys, n_boot=n_boot, seed=seed,
                        verbose=verbose)
    return {"summary": summary,
            "by_decile": by_decile(tables, found, keys),
            "by_group": by_group(tables, found, keys),
            "discovery": discovery_log(found),
            "bins": pd.concat([k.assign(cohort=c) for c, k in keys.items()],
                              ignore_index=True) if keys else pd.DataFrame(),
            "tables": tables, "found": found, "keys": keys}


def results_dir(env) -> Path:
    p = Path(env.root) / RESULTS_DIRNAME
    p.mkdir(parents=True, exist_ok=True)
    return p


def save(env, out: dict) -> list[Path]:
    """Write the single summary CSV plus its supporting long-form tables."""
    d = results_dir(env)
    written = []
    names = {"summary": "ALL_MODELS_SUMMARY", "by_decile": "all_models_by_decile",
             "by_group": "all_models_by_group", "discovery": "discovery_log",
             "bins": "size_bins"}
    for k, fname in names.items():
        obj = out.get(k)
        if isinstance(obj, pd.DataFrame) and not obj.empty:
            p = d / f"{fname}.csv"
            obj.to_csv(p, index=False)
            written.append(p)
    (d / "run_info.json").write_text(json.dumps({
        "n_models": int(len(out["summary"])),
        "n_comparable": int(out["summary"].comparable.sum())
        if "comparable" in out["summary"] else None,
        "n_files_seen": int(len(out["found"])),
        "cohorts": sorted({f.cohort for f in out["found"] if f.status == "ok"}),
        "note": "ALL_MODELS_SUMMARY.csv is the single table. `comparable=True` "
                "rows share one decile cut and one test set and may be compared "
                "directly; other rows may not.",
    }, indent=2), encoding="utf-8")
    written.append(d / "run_info.json")
    return written


def print_summary(out: dict, top: int = 60) -> None:
    """The table, in the order it should be read: misses first, Dice last."""
    s = out["summary"]
    comp = s[s.comparable] if "comparable" in s.columns else s
    print("=" * 118)
    print(f"ALL MODELS -- {len(s)} model(s), {len(comp)} in the reference cohort")
    print("=" * 118)
    cols = [c for c in ("model", "n_images", "all_zero_dice_n", "all_zero_dice_rate",
                        "all_wrong_place_n", "all_mean_dice", "all_median_dice",
                        "D1_D4_zero_dice_n", "D1_D4_mean_recall", "D1_mean_recall",
                        "kruskal_p", "fairness_worst_group") if c in comp.columns]
    print(comp[cols].head(top).to_string(index=False))
    print("\nREAD THE MISS COLUMNS FIRST. Dice is saturated on this task "
          "(Friedman p = 0.61\nacross the seven headline models); complete misses "
          "and small-lesion recall are\nwhere the models in this study actually "
          "separate.")
    print("  all_zero_dice_n   complete misses, dice == 0 (the PUBLISHED "
          "definition)")
    print("  all_wrong_place_n of those, the ones that predicted SOMETHING and "
          "got it all wrong")
    print("  D1_D4_*           the four smallest GT-area deciles")
    other = s[~s.comparable] if "comparable" in s.columns else s.iloc[:0]
    if len(other):
        print(f"\n{len(other)} model(s) are NOT in the reference cohort and may "
              f"NOT be compared\nagainst the rows above -- different split or "
              f"different mask version:")
        for _, r in other.iterrows():
            print(f"    {r.model:<44} {r.cohort:<22} n={r.n_images}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. self-test
# ─────────────────────────────────────────────────────────────────────────────
def self_test(verbose: bool = True) -> bool:
    """Structural checks on synthetic frames. No disk, no manifests, no network."""
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
                  f"{'  ' + detail if detail else ''}")

    for raw, want in (("per_image_segformer_b0.csv", "segformer_b0"),
                      ("segformer_b0_best_seed0_test_per_image.csv", "segformer_b0"),
                      ("reference_b2_teacher_test_per_image.csv", "b2_teacher"),
                      ("test_per_image__sam_ft__seed0.csv", "sam_ft__seed0")):
        got = _clean_name(Path(raw))
        check(f"name: {raw}", got == want, f"got {got!r}")

    n = 40
    rng = np.random.default_rng(0)
    gt = rng.integers(100, 50000, n)
    base = pd.DataFrame({
        "stem": [f"s{i}" for i in range(n)],
        "subject": [f"p{i % 8}" for i in range(n)],
        "dice": np.clip(rng.normal(0.75, 0.2, n), 0, 1),
        "recall": rng.random(n), "precision": rng.random(n),
        "pred_positive_pixels": gt, "gt_positive_pixels": gt,
        GROUP_COL: [GROUP_ORDER[i % 5] for i in range(n)],
    })
    base.loc[:2, "dice"] = 0.0
    base.loc[0, "pred_positive_pixels"] = 0        # empty prediction
    b = _block(base, np.ones(n, bool), "all")
    check("zero_dice_n counts dice == 0", b["all_zero_dice_n"] == 3,
          str(b["all_zero_dice_n"]))
    check("empty_pred_n is separate from zero_dice", b["all_empty_pred_n"] == 1,
          str(b["all_empty_pred_n"]))
    check("wrong_place = zero minus empty", b["all_wrong_place_n"] == 2,
          str(b["all_wrong_place_n"]))

    t2 = base.copy()
    t2["gt_positive_pixels"] = t2.gt_positive_pixels + 1
    check("cohort signature separates different GT areas",
          _signature(base) != _signature(t2))
    check("cohort signature is stable for identical tables",
          _signature(base) == _signature(base.sample(frac=1, random_state=1)))

    found = [Found(Path("a.csv"), "m1", n, "test", "REFERENCE"),
             Found(Path("b.csv"), "m2", n, "test", "REFERENCE")]
    keys = assign_bins({"m1": base, "m2": base}, found)
    check("bins built for the cohort", "REFERENCE" in keys)
    k = keys["REFERENCE"]
    check("small stratum is D1-D4",
          set(k[k.is_primary]["bin"]) <= set(PRIMARY_STRATUM),
          str(sorted(set(k[k.is_primary]['bin']))))

    s = summarize({"m1": base, "m2": base}, found, keys, n_boot=200)
    check("summary has one row per model", len(s) == 2, str(len(s)))
    for c in ("all_zero_dice_n", "all_mean_dice", "all_median_dice",
              "D1_D4_mean_recall", "D1_mean_recall", "mean_dice_ci_lo",
              "fairness_gap_median_dice", "kruskal_p"):
        check(f"summary carries {c}", c in s.columns)

    if verbose:
        print(f"\n  {'ALL PASS' if ok else 'FAILURES ABOVE'}")
    return ok
