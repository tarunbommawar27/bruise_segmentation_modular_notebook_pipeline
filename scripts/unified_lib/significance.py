"""Stage G: the confirmatory statistics layer.

WHY THIS EXISTS SEPARATELY FROM report.py
------------------------------------------
`report.bootstrap_ci` and `report.paired_contrast` answer "what is this one
number, and how uncertain is it". They are descriptive, they are called ad hoc
throughout Stage D, and that is the right shape for exploration.

This module answers a different question: **across the whole study, which
differences survive?** That is a confirmatory question, and it needs three things
the descriptive layer deliberately does not provide:

1. a **pre-specified** list of comparisons, written down before the numbers are
   looked at, so the family size is fixed and not chosen after seeing which pairs
   looked promising;
2. **multiplicity control** within that family;
3. an **omnibus test first**, so no pairwise claim is made when the models are
   collectively indistinguishable.

Keeping it in its own module is what makes (1) meaningful. A contrast list that
lives next to the plotting code gets appended to; one that lives in a module with
this docstring above it does not.

WHY NOT AN ALL-PAIRS TOURNAMENT
--------------------------------
The study has ~20 scored models, which is 190 ordered pairs. At 28 subjects, with
every model inside the annotation-ceiling band (human annotators disagree with
each other by more than the models disagree with each other), an all-pairs sweep
at alpha=0.05 is expected to produce ~10 "significant" differences from noise
alone. It would also be reported as a ranking, which is exactly the reading the
ceiling result forbids. `CONTRAST_FAMILY` is 12 comparisons, each attached to a
question someone actually asked.

CONFIRMATORY vs EXPLORATORY
----------------------------
Four contrasts are confirmatory: they were specified as the point of an
experiment before it ran (the two Stage F arms, Stage A's distillation, and the
B2->B0 compression claim). Holm-Bonferroni is applied *within that set of four*.
The remaining eight are exploratory, reported uncorrected and labelled as such.
Mixing them into one correction would either over-penalise the four questions the
study was designed to answer, or launder eight post-hoc comparisons into
confirmatory ones.

EVERY ENDPOINT FROM THE SAME DRAWS
-----------------------------------
`paired_contrast_multi` evaluates mean Dice, median Dice and complete-miss rate
on the *same* resampled subject lists. Running three separate bootstraps with
three different seeds would give three intervals that cannot be read against each
other -- a model could appear to gain Dice and lose misses in draws that never
co-occurred. Sharing the draws makes "Dice up, misses up" a statement about the
same 10,000 resampled worlds.

WHY MISSES ALSO GET A DISCORDANCE TABLE
----------------------------------------
Complete-miss counts here are small (0 to 13 of 185). A bootstrapped difference
of two rates near the boundary is unstable and its interval is not trustworthy at
these counts. `discordance` reports the raw 2x2 -- how many images A misses that B
finds, and vice versa -- which is the evidence itself rather than a summary of it,
and is the form already used for the DeepLab-vs-U-Net teacher choice (handbook
7b.1). Report both; believe the counts.

NON-INFERIORITY, NOT SUPERIORITY
---------------------------------
At the ceiling, "indistinguishable from the 27M teacher at 3.7M parameters" is
both the stronger claim and the true one. `verdict` uses the 0.01 Dice margin
Stage C established, with one refinement: Stage C called everything that was not
a WIN or an INFERIOR a NON-INFERIOR, which silently includes intervals running
from -0.05 to +0.02. Those are not evidence of equivalence, they are evidence of
nothing. They are labelled INCONCLUSIVE here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MARGIN = 0.01          # Dice points; the Stage C non-inferiority margin
N_BOOT_FINAL = 10_000  # 2000 is fine for an interval, coarse for a tail probability

CONFIRMATORY = "confirmatory"
EXPLORATORY = "exploratory"

# ─────────────────────────────────────────────────────────────────────────────
# the pre-specified family
# ─────────────────────────────────────────────────────────────────────────────
# (a, b, kind, question).  Sign convention: delta = a - b, so `a` is always the
# model whose case is being made.  Edit this list ONLY before a run, never after
# seeing the output -- that is the entire point of it being here.
CONTRAST_FAMILY: list[tuple[str, str, str, str]] = [
    # ── confirmatory: each one is the reason an experiment was run ────────────
    ("lraspp_mobilenetv3_distilled", "lraspp_mobilenetv3", CONFIRMATORY,
     "Stage F: does DeepLabV3+ KD help a pretrained mobile student?"),
    ("fastscnn_distilled", "fastscnn", CONFIRMATORY,
     "Stage F: does the same KD help a scratch-init student?"),
    ("segformer_b0_distilled", "segformer_b0_direct", CONFIRMATORY,
     "Stage A: does B2->B0 distillation help?"),
    ("segformer_b0_distilled", "segformer_b2_teacher", CONFIRMATORY,
     "Stage A: what does 7.4x compression cost? (non-inferiority)"),

    # ── exploratory: reported uncorrected, labelled ───────────────────────────
    ("segformer_b2_teacher", "unet_r50", EXPLORATORY,
     "transformer vs the CNN baseline"),
    ("unet_r50", "deeplabv3plus_r50", EXPLORATORY,
     "the two Stage B baselines against each other"),
    ("segformer_b0_direct", "yolo_sem_direct", EXPLORATORY,
     "accuracy tier vs speed tier"),
    ("lraspp_mobilenetv3", "topformer_tiny", EXPLORATORY,
     "the mobile top two -- gap is 0.02, under the claimable threshold"),
    ("lraspp_mobilenetv3", "fastscnn", EXPLORATORY,
     "mobile spread: best vs worst"),
    ("segformer_b0_direct", "lraspp_mobilenetv3", EXPLORATORY,
     "3.71M vs 3.22M -- is the mobile gap about size at all?"),
    ("lraspp_mobilenetv3_distilled", "deeplabv3plus_r50", EXPLORATORY,
     "Stage F: how much of the teacher survived an 8x parameter cut?"),
    ("fastscnn_distilled", "deeplabv3plus_r50", EXPLORATORY,
     "Stage F: the same, at 29x"),
]


# ─────────────────────────────────────────────────────────────────────────────
# paired bootstrap, three endpoints, one set of draws
# ─────────────────────────────────────────────────────────────────────────────
def _pair(a: pd.DataFrame, b: pd.DataFrame, name_a: str, name_b: str) -> pd.DataFrame:
    """Merge two per-image tables on `stem`, refusing anything but an exact cover.

    Same guard as `report.paired_contrast`: two models that were scored on
    different image sets can still be merged into a plausible-looking frame, and
    the resulting contrast would be silently meaningless.
    """
    m = a.merge(b, on="stem", suffixes=("_a", "_b"))
    if len(m) != len(a) or len(m) != len(b):
        raise ValueError(
            f"{name_a} and {name_b} do not cover the same images "
            f"({len(a)} vs {len(b)}, {len(m)} shared) -- cannot pair")
    if "subject_a" in m.columns:
        m["subject"] = m["subject_a"]
    if "subject" not in m.columns:
        raise ValueError("no subject column; call report.normalize(df, meta) first")
    return m


def paired_contrast_multi(a: pd.DataFrame, b: pd.DataFrame, name_a: str, name_b: str,
                          n_boot: int = N_BOOT_FINAL, seed: int = 0,
                          margin: float = MARGIN) -> dict:
    """Paired subject-level bootstrap of (a - b) on three endpoints at once.

    Returns mean-Dice, median-Dice and complete-miss-rate differences, each with a
    95% interval, plus a two-sided bootstrap p-value on mean Dice and the
    equivalence verdict against `margin`.

    All three endpoints are evaluated on the SAME resampled subject lists -- see
    the module docstring. The resampling itself is the cluster bootstrap: draw
    subjects with replacement, take all of a chosen subject's images.

    Sign conventions differ by endpoint and are named accordingly: for Dice,
    higher is better, so `p_a_better = P(delta > 0)`. For misses, LOWER is
    better, so the reported probability is `p_a_fewer_misses = P(delta < 0)`.
    """
    m = _pair(a, b, name_a, name_b)
    da, db = m.dice_a.to_numpy(float), m.dice_b.to_numpy(float)
    ma = m.complete_miss_a.to_numpy(float)
    mb = m.complete_miss_b.to_numpy(float)

    # Index arrays, not frames -- concatenating DataFrames per draw is what killed
    # a Colab kernel once (handbook 15, trap 8).
    groups = [np.asarray(v, dtype=np.intp) for v in m.groupby("subject").indices.values()]
    rng = np.random.default_rng(seed)
    d_mean = np.empty(n_boot)
    d_med = np.empty(n_boot)
    d_miss = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(groups), size=len(groups))
        sel = np.concatenate([groups[j] for j in pick])
        d_mean[i] = da[sel].mean() - db[sel].mean()
        d_med[i] = np.median(da[sel]) - np.median(db[sel])
        d_miss[i] = ma[sel].mean() - mb[sel].mean()

    delta = float(da.mean() - db.mean())
    lo, hi = (float(np.quantile(d_mean, .025)), float(np.quantile(d_mean, .975)))
    return {
        "a": name_a, "b": name_b, "n_subjects": len(groups), "n_images": len(m),
        "delta_dice": delta, "lo": lo, "hi": hi,
        "p_a_better": float((d_mean > 0).mean()),
        "p_two_sided": boot_p_two_sided(d_mean),
        "verdict": verdict(delta, lo, hi, margin),
        "delta_median": float(np.median(da) - np.median(db)),
        "median_lo": float(np.quantile(d_med, .025)),
        "median_hi": float(np.quantile(d_med, .975)),
        "delta_miss_rate": float(ma.mean() - mb.mean()),
        "miss_lo": float(np.quantile(d_miss, .025)),
        "miss_hi": float(np.quantile(d_miss, .975)),
        "p_a_fewer_misses": float((d_miss < 0).mean()),
        "n_boot": n_boot,
    }


def boot_p_two_sided(draws: np.ndarray) -> float:
    """Two-sided bootstrap p-value for H0: delta = 0.

    Twice the smaller tail, floored at 1/n_boot because a bootstrap cannot
    resolve a probability below its own resolution and reporting p = 0 would
    claim it can.
    """
    n = len(draws)
    p = 2.0 * min((draws <= 0).mean(), (draws >= 0).mean())
    return float(min(1.0, max(p, 1.0 / n)))


def verdict(delta: float, lo: float, hi: float, margin: float = MARGIN) -> str:
    """WIN / NON-INFERIOR / INFERIOR / INCONCLUSIVE against a Dice margin.

    INCONCLUSIVE is the addition over Stage C's three-way rule. An interval of
    [-0.05, +0.02] is not equivalence -- it is an underpowered comparison, and
    calling it NON-INFERIOR reports absence of evidence as evidence of absence.
    """
    if lo > 0:
        return "WIN"
    if hi < -margin:
        return "INFERIOR"
    if lo >= -margin:
        return "NON-INFERIOR"
    return "INCONCLUSIVE"


# ─────────────────────────────────────────────────────────────────────────────
# multiplicity
# ─────────────────────────────────────────────────────────────────────────────
def holm(pvals) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values, order preserved.

    Holm rather than Bonferroni because it is uniformly more powerful at no cost
    in assumptions, and rather than Benjamini-Hochberg because a family of four
    confirmatory tests is small enough that controlling the family-wise error is
    the appropriate and more conservative target.
    """
    p = np.asarray(list(pvals), dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    adj = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (n - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


# ─────────────────────────────────────────────────────────────────────────────
# omnibus
# ─────────────────────────────────────────────────────────────────────────────
def subject_matrix(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-subject mean Dice, one column per model, subjects as rows.

    Collapsing to subjects before any omnibus test is not a convenience -- images
    within a subject are strongly correlated, so treating them as blocks would
    reuse the same information 6.6 times on average and shrink the test's
    effective sample from 28 to something the test still believes is 185.
    """
    bad = [n for n, df in tables.items() if "subject" not in df.columns]
    if bad:
        raise ValueError(
            f"no subject column on: {', '.join(bad)}. `report.normalize` DROPS "
            "subject and re-joins it from the manifest, so it must be called as "
            "normalize(df, META) -- calling it with one argument silently returns "
            "a frame that no clustered statistic in this module can use.")
    cols = {name: df.groupby("subject").dice.mean() for name, df in tables.items()}
    mat = pd.DataFrame(cols)
    return mat.dropna(axis=0, how="any")


# Omnibus sets. ONE test over every model in the bundle is the wrong test: with
# ~20 models it carries 19 degrees of freedom, and most of those models are B0
# students of the same teacher differing by a KD loss term. The chi-square is
# diluted by near-duplicates and the result says little about the comparison
# anyone cares about. Each set below is a field that a reader would actually ask
# "are these different at all?" about.
OMNIBUS_SETS: dict[str, tuple[str, ...]] = {
    "headline": ("segformer_b2_teacher", "segformer_b0_direct", "segformer_b0_distilled",
                 "yolo_sem_direct", "yolo_sem_distilled", "unet_r50", "deeplabv3plus_r50"),
    "segformer_family": ("segformer_b2_teacher", "segformer_b0_direct",
                         "segformer_b0_distilled"),
    "mobile_field": ("lraspp_mobilenetv3", "topformer_tiny", "ppmobileseg_tiny",
                     "fastscnn", "fastscnn_distilled", "lraspp_mobilenetv3_distilled"),
    "kd_arms": ("p3_adaptive", "p2_cwd_b5_to_b0", "expA_b5_to_b0_response",
                "p3_adaptive_boundary", "p2_bpkd_b5_to_b0", "p2_ensemble_uniform",
                "p3_adaptive_group", "p3_adaptive_full", "x_angular_b5_to_b0",
                "p3_adaptive_hard"),
    # Stage H. Two sets, deliberately, because they answer different questions and
    # a single "all the mobile arms" omnibus would answer neither.
    #
    #   teacher_axis  one student's four training regimes -- no KD, DeepLabV3+ KD,
    #                 B2 KD, gated B2 KD. Rejecting means the training signal
    #                 matters at all for that student; not rejecting means it does
    #                 not, and no pairwise gate result inside it is licensed.
    #   gated_arms    every gated arm against every teacher-matched control. This
    #                 is the set that gates the CONTRAST_FAMILY_H pairwise table.
    "teacher_axis_lraspp": ("lraspp_mobilenetv3", "lraspp_mobilenetv3_distilled",
                            "lraspp_mobilenetv3_b2kd", "lraspp_mobilenetv3_rgkd"),
    "teacher_axis_fastscnn": ("fastscnn", "fastscnn_distilled",
                              "fastscnn_b2kd", "fastscnn_rgkd"),
    #                 No YOLO pair: that arm is implemented but not registered
    #                 (see reliability_kd), so every member here tests the gate on
    #                 an ONLINE loss.
    "gated_arms": ("segformer_b0_distilled", "segformer_b0_rgkd",
                   "lraspp_mobilenetv3_b2kd", "lraspp_mobilenetv3_rgkd",
                   "fastscnn_b2kd", "fastscnn_rgkd",
                   "topformer_tiny_b2kd", "topformer_tiny_rgkd",
                   "ppmobileseg_tiny_b2kd", "ppmobileseg_tiny_rgkd"),
}


def run_omnibus_sets(tables: dict[str, pd.DataFrame],
                     sets: dict[str, tuple[str, ...]] | None = None) -> pd.DataFrame:
    """Friedman on each pre-specified set, one row per set.

    Read this table before the pairwise one. A set that does not reject means the
    models in it are collectively indistinguishable on these 28 subjects, and any
    pairwise difference inside it is being read out of noise.
    """
    sets = OMNIBUS_SETS if sets is None else sets
    rows = []
    for name, members in sets.items():
        present = [m for m in members if m in tables]
        if len(present) < 3:
            rows.append({"set": name, "n_models": len(present), "n_subjects": None,
                         "chi2": None, "p": None, "kendall_w": None,
                         "rejects_at_05": None,
                         "note": f"needs 3+ models, have {len(present)}"})
            continue
        o = friedman_omnibus({m: tables[m] for m in present})
        rows.append({"set": name, "n_models": o["n_models"], "n_subjects": o["n_subjects"],
                     "chi2": o["chi2"], "p": o["p"], "kendall_w": o["kendall_w"],
                     "rejects_at_05": o["rejects_at_05"],
                     "note": "" if len(present) == len(members)
                             else f"missing {len(members) - len(present)} of {len(members)}"})
    return pd.DataFrame(rows)


def friedman_omnibus(tables: dict[str, pd.DataFrame]) -> dict:
    """Friedman test across models on subject-mean Dice, with Kendall's W.

    Ranks within each subject, so it asks "do the models order themselves
    consistently across subjects" without assuming normality or equal variances,
    both of which fail on Dice.

    **Run this before reading any pairwise result.** If it does not reject, the
    models are collectively indistinguishable on this test set and the pairwise
    table is a description of noise -- which, given the annotation ceiling, is a
    live possibility and a publishable finding in its own right.

    Kendall's W is the effect size: 0 = no agreement between subjects on how to
    rank the models, 1 = every subject ranks them identically.
    """
    from scipy.stats import friedmanchisquare

    mat = subject_matrix(tables)
    n, k = mat.shape
    if k < 3:
        raise ValueError(f"Friedman needs 3+ models, got {k}")
    stat, p = friedmanchisquare(*[mat[c].to_numpy() for c in mat.columns])
    return {"n_subjects": int(n), "n_models": int(k),
            "chi2": float(stat), "p": float(p),
            "kendall_w": float(stat / (n * (k - 1))),
            "rejects_at_05": bool(p < 0.05),
            "models": list(mat.columns)}


# ─────────────────────────────────────────────────────────────────────────────
# complete misses, as counts
# ─────────────────────────────────────────────────────────────────────────────
def discordance(a: pd.DataFrame, b: pd.DataFrame, name_a: str, name_b: str) -> dict:
    """The 2x2 of complete misses, plus an exact McNemar on the discordant cells.

    At 0-13 misses out of 185 this is more informative than any rate difference:
    it says whether one model's misses are a subset of the other's, which is the
    question a teacher choice or a deployment decision actually turns on.

    The McNemar p is **exact binomial on the discordant pairs and ignores subject
    clustering**, so it is anti-conservative here. It is reported as a sanity
    check on the counts, not as the inferential result -- the clustered bootstrap
    interval in `paired_contrast_multi` is that. When they disagree, believe the
    bootstrap.
    """
    from scipy.stats import binomtest

    m = _pair(a, b, name_a, name_b)
    am = m.complete_miss_a.to_numpy(bool)
    bm = m.complete_miss_b.to_numpy(bool)
    a_only = int((am & ~bm).sum())
    b_only = int((bm & ~am).sum())
    both = int((am & bm).sum())
    n_disc = a_only + b_only
    p = float(binomtest(a_only, n_disc, 0.5).pvalue) if n_disc else 1.0
    return {"a": name_a, "b": name_b,
            "a_misses_b_finds": a_only, "b_misses_a_finds": b_only,
            "both_miss": both, "neither_miss": int(len(m) - a_only - b_only - both),
            "a_total": int(am.sum()), "b_total": int(bm.sum()),
            "n_discordant": n_disc, "mcnemar_exact_p": p,
            "containment": ("a within b" if a_only == 0 and b_only > 0 else
                            "b within a" if b_only == 0 and a_only > 0 else
                            "none" if n_disc else "identical")}


# ─────────────────────────────────────────────────────────────────────────────
# robustness: does the contrast hold at every seed?
# ─────────────────────────────────────────────────────────────────────────────
def contrast_by_seed(by_seed_a: dict[int, pd.DataFrame], by_seed_b: dict[int, pd.DataFrame],
                     name_a: str, name_b: str, n_boot: int = 2000,
                     seed: int = 0) -> pd.DataFrame:
    """Repeat one contrast at every seed both models share.

    A point estimate from the val-selected seed is one draw from the training
    distribution. Three same-signed deltas of similar magnitude are much stronger
    evidence than one large delta, and cost nothing to check once the per-seed
    tables exist -- which is the whole argument for pulling every seed's
    `test_per_image.csv` off the cluster rather than only the best one's.

    A `sign_consistent` column and the count of agreeing seeds are the output to
    read; the per-seed intervals will individually be wide.
    """
    shared = sorted(set(by_seed_a) & set(by_seed_b))
    rows = []
    for s in shared:
        r = paired_contrast_multi(by_seed_a[s], by_seed_b[s], name_a, name_b,
                                  n_boot=n_boot, seed=seed)
        rows.append({"seed": s, "delta_dice": r["delta_dice"], "lo": r["lo"],
                     "hi": r["hi"], "p_a_better": r["p_a_better"],
                     "delta_miss_rate": r["delta_miss_rate"]})
    out = pd.DataFrame(rows)
    if len(out):
        pos = int((out.delta_dice > 0).sum())
        out["sign_consistent"] = (pos == len(out)) or (pos == 0)
        out["n_seeds_positive"] = pos
    return out


# ─────────────────────────────────────────────────────────────────────────────
# the driver
# ─────────────────────────────────────────────────────────────────────────────
def run_family(tables: dict[str, pd.DataFrame],
               family: list[tuple[str, str, str, str]] | None = None,
               n_boot: int = N_BOOT_FINAL, seed: int = 0,
               margin: float = MARGIN) -> tuple[pd.DataFrame, list[str]]:
    """Score the whole pre-specified family; Holm-adjust within the confirmatory set.

    Returns `(table, skipped)`. A contrast whose models are not both present is
    **skipped and named**, never dropped silently -- a missing Stage F arm must
    show up as a missing row, not as a shorter table that looks complete.
    """
    family = CONTRAST_FAMILY if family is None else family
    rows, skipped = [], []
    for a, b, kind, question in family:
        if a not in tables or b not in tables:
            missing = [n for n in (a, b) if n not in tables]
            skipped.append(f"{a} vs {b} -- missing {', '.join(missing)}")
            continue
        r = paired_contrast_multi(tables[a], tables[b], a, b,
                                  n_boot=n_boot, seed=seed, margin=margin)
        r["kind"] = kind
        r["question"] = question
        rows.append(r)

    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = np.nan
        conf = out.kind == CONFIRMATORY
        if conf.any():
            out.loc[conf, "p_holm"] = holm(out.loc[conf, "p_two_sided"])
        out["significant_holm"] = out.p_holm < 0.05
        cols = ["a", "b", "kind", "delta_dice", "lo", "hi", "verdict",
                "p_a_better", "p_two_sided", "p_holm", "significant_holm",
                "delta_median", "delta_miss_rate", "miss_lo", "miss_hi",
                "p_a_fewer_misses", "n_subjects", "n_images", "n_boot", "question"]
        out = out[[c for c in cols if c in out.columns]]
    return out, skipped


def interpret(row) -> str:
    """One plain sentence per contrast, so the table cannot be mis-summarised.

    Written from the interval, never from the point estimate: the failure mode
    this guards against is a reader taking delta = +0.017 as "better" when the
    interval runs from -0.005 to +0.039.
    """
    d, lo, hi = row["delta_dice"], row["lo"], row["hi"]
    dirn = "higher" if d > 0 else "lower"
    base = (f"{row['a']} is {abs(d):.4f} Dice {dirn} than {row['b']} "
            f"(95% CI [{lo:+.4f}, {hi:+.4f}], P better = {row['p_a_better']:.2f})")
    tail = {"WIN": " -- interval excludes zero.",
            "INFERIOR": f" -- interval entirely below the {MARGIN} margin.",
            "NON-INFERIOR": f" -- within the {MARGIN} margin; equivalent, not better.",
            "INCONCLUSIVE": " -- interval too wide to conclude anything."}[row["verdict"]]
    mr = row.get("delta_miss_rate")
    miss = ""
    if mr is not None and not pd.isna(mr):
        miss = (f" Complete misses {'up' if mr > 0 else 'down' if mr < 0 else 'unchanged'}"
                f" by {abs(mr) * 100:.2f} pp.")
    return base + tail + miss
