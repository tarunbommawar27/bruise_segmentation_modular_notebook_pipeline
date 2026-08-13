"""Stage O -- the miss taxonomy, the distilled-arm fairness re-analysis, and
ITA-group-routed gated multi-teacher distillation.

Three deliverables in one module because they share one set of per-image tables
and one results directory, and because parts 1 and 2 are what tell you whether
part 3 is worth the GPU time.

    PART 1  miss_taxonomy          zero Dice vs empty prediction vs wrong place
    PART 2  distilled_fairness     skin tone x lesion size for the KD arms
    PART 3  the ITA-grouped arm    gate, group weights, shims, training

PART 1 -- WHY "COMPLETE MISS" NEEDED SPLITTING
-----------------------------------------------
The study publishes ONE miss number, `complete_miss_rate`, and it is `dice == 0`.
That is the union of two clinically different failures:

    empty prediction   pred_positive_pixels == 0   the model found nothing
    wrong place        dice == 0 and pred > 0      the model outlined the wrong
                                                   region with zero overlap

They differ in 28 of the 40 per-image tables in RESULT_AUGUST_08, and the split
changes how an arm reads. `fastscnn_rgkd` is the clearest case: reliability
gating took its empty predictions from 6 to 1 while its zero-Dice count barely
moved from 8 to 6, because the gate converted blank outputs into confident
misplacements. A one-column table calls that "roughly unchanged". It is not.

A wrong-place error is worse in a clinical read than an empty one. An empty
prediction shows the clinician nothing and invites a second look; a confident
outline in the wrong place is an assertion. Report all three columns, always,
and lead with zero Dice because it is the union and therefore the conservative
number.

PART 2 -- THE GAP THIS FILLS
-----------------------------
`FINAL_RESULT/RESULT_AUGUST_08/fairness_stats.csv` covers 24 models and excludes
every Stage C distillation arm -- including `p3_adaptive`, the best arm in the
study at 0.7748. So the one question the deck is asked about the best distilled
model ("how does it do across skin tones?") had no table behind it.

Nothing here is retrained. The per-image CSVs already carry `subject`,
`skin_tone_category`, `ita_group_index_5` and `ITA`, so this is a join and an
aggregation over tables that already exist, run through `lesionsize`'s machinery
rather than a second implementation of it.

PART 3 -- WHAT MAKES THIS DIFFERENT FROM STAGE M, AND WHY IT IS NOT STAGE M AGAIN
---------------------------------------------------------------------------------
Stage M routed per IMAGE on each teacher's soft Dice against the label. It is
explicit that it does NOT route by skin tone (`multiteacher.py`, "WHAT THIS DOES
NOT DO"), and it came back a null: all six contrasts inconclusive, miss rate
slightly worse.

Stage O routes per ITA GROUP on weights fitted ONCE on validation:

    w_g = softmax(beta * mean_val_dice_k_within_group_g)          [K weights]

Three consequences, and each is the answer to an objection Stage M invites:

  1. NO LABEL IS CONSULTED AT ROUTING TIME. The weights come from validation
     before training starts. The only per-image input is its ITA group, which is
     a manifest column and available at test time too -- so unlike Stage M's
     router this one is not restricted to training.
  2. K weights per group instead of K per image. Five groups over 20 validation
     subjects cannot support 5 x K free parameters; two groups can support 2 x K.
     That is why `SCHEME` defaults to the collapse and not to the five groups.
  3. THE GATE HAS AN IDENTIFIABILITY CLAUSE Stage M did not have, and it is the
     load-bearing addition. With six candidate teachers the per-group argmax on
     134 validation images is bootstrap-stable only 36-52 % of the time -- worse
     than a coin flip in two of the five groups, with margins of 0.001-0.008
     Dice. A gate that opens on an argmax that unstable is fitting noise and
     will produce a fourth null. `ita_group_gate` refuses unless at least one
     group's argmax survives resampling at P >= 0.75.

THE ONE GROUP WHERE THE DATA SAYS THIS COULD WORK
--------------------------------------------------
Across six candidate teachers on the 134 validation images, per-group Dice spans:

    Light (II-III)          0.723 -> 0.785      spread 0.062     <- the signal
    Intermediate (III-IV)   0.802 -> 0.833      spread 0.031
    Dark (VI)               0.769 -> 0.793      spread 0.024
    Tan (IV)                0.780 -> 0.813      spread 0.033
    Brown (V)               0.727 -> 0.805      spread 0.078  (7 images, 4 subj)

Only Light (II-III) has a spread above the annotation-ceiling noise floor on a
usable number of subjects, and Light (II-III) is where every teacher's complete
misses are concentrated -- on the 55 dark-skin test images not one of the four
Stage M teachers missed anything. So the pre-registered scheme is the two-group
collapse `{Very Light, Light} vs everything else`, which targets the one group
where teacher choice demonstrably matters and spends its degrees of freedom
there instead of spreading them over five cells of four subjects.

`Very Light (I-II)` is in the Light group on purpose: it is 12 TRAIN images and
0 validation images, so under the five-group scheme those 12 images would have no
fitted weight at all. `group_weights` RAISES on an unmapped group rather than
falling back to uniform, because a uniform fallback is Stage C's
`p2_ensemble_uniform` wearing this stage's name.

WHY THE POOL IS THREE TEACHERS AND NOT STAGE M's FOUR
------------------------------------------------------
    segformer_b5_teacher   MiT transformer          wins Dark, Tan, Brown on val
    deeplabv3plus_r50      CNN + ASPP               wins Intermediate, and is
                                                    second on Light
    medsam_ft              ViT, mask-pretrained     best overall val Dice
                                                    (0.7957), ties Light

`segformer_b2_teacher` is dropped: it wins no group on validation, it has the
lowest drop-one marginal in Stage M's own gate (0.0121 against DeepLabV3+'s
0.0224), and it is the same MiT family as B5 -- so it is simultaneously the least
useful and the most correlated member. Diversity is the mechanism a pool has; a
redundant member only adds forward cost.

`unet_r50` is dropped for the reason Stage M's own numbers give: it appears as a
per-group winner only in the TEST table quoted in `multiteacher.py`'s docstring,
and it was never in Stage M's actual pool. Selecting a per-group teacher on test
is leakage, and this module never reads test before the gate is on disk.

MedSAM enters as its FEATURES, never as MedSAM: `samprobe` keeps the image
encoder and discards the prompt encoder and mask decoder, because this pipeline
is automatic and has no prompt to give. Write it up that way.

WHERE THE GROUP REACHES THE LOSS, AND THE FAILURE THAT WOULD FAKE A RESULT
---------------------------------------------------------------------------
`engine.train_run` iterates `for step, (x, y, _) in enumerate(train_loader)` --
it DISCARDS the stem, so the loss has no idea which images it is looking at.
Rather than edit the shared training loop, `install_group_shim` wraps the
training loader so each batch records its stems' group indices in a module
global immediately before the batch is yielded. The loader, the teacher forward
and the loss all run synchronously in the main process within one iteration, so
the global is always the current batch's.

That is a real coupling and it is guarded rather than trusted:
`GroupRoutedDistillLoss.forward` RAISES if the recorded group vector is absent or
its length does not match the batch. Without that check a stale or missing global
would silently mean uniform weights, the arm would quietly become
`p2_ensemble_uniform` with a gate, and it would report a plausible number that
answers a different question.

WRITES ONLY TO STAGE_O_RESULTS/
--------------------------------
Never `results/`, `FINAL_RESULT/`, `LESION_SIZE_RESULTS/`, `STAGE_N4_RESULTS/`
or `_work/runs/`. Stage M's, Stage N3's and Stage N4's tables are READ for
reference rows and never rewritten, and Stage O trains into its own runs
directory so the reporting stages -- which scan `env.runs` by name -- cannot see
these arms.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIRNAME = "STAGE_O_RESULTS"


# ═════════════════════════════════════════════════════════════════════════════
# PART 1 -- the miss taxonomy
# ═════════════════════════════════════════════════════════════════════════════
#: The three columns, in the order they should always be printed. `zero_dice` is
#: first because it is the union and the number the study has always published;
#: the other two are its decomposition and must sum to it.
MISS_KINDS = ("zero_dice", "empty_pred", "wrong_place")


def _miss_counts(df: pd.DataFrame, m: np.ndarray | None = None) -> dict:
    """The three miss counts over an optional row mask, plus the denominator.

    `wrong_place` is DERIVED as `zero_dice - empty_pred` rather than counted
    independently, so the identity `empty + wrong = zero` holds by construction
    and a table can never print three numbers that do not add up.
    """
    dice = df["dice"].to_numpy(float)
    pred = df["pred_positive_pixels"].to_numpy(float)
    if m is None:
        m = np.ones(len(df), bool)
    n = int(m.sum())
    zero = int(((dice == 0) & m).sum())
    empty = int(((pred == 0) & m).sum())

    # An image with no predicted pixels ALWAYS scores Dice 0, so empty is a
    # subset of zero and the difference cannot be negative. If it is, the table
    # is internally inconsistent -- a Dice column computed against a different
    # mask version from the pixel counts -- and every number below it is void.
    if empty > zero:
        raise ValueError(
            f"{empty} images have no predicted pixels but only {zero} score "
            f"Dice 0. An empty prediction cannot score above zero, so this "
            f"table's dice column and its pixel counts come from different "
            f"evaluations. Refusing to report a miss taxonomy from it.")
    return {
        "n": n,
        "zero_dice_n": zero,
        "empty_pred_n": empty,
        "wrong_place_n": zero - empty,
        "zero_dice_rate": zero / n if n else np.nan,
        "empty_pred_rate": empty / n if n else np.nan,
        "wrong_place_rate": (zero - empty) / n if n else np.nan,
    }


def miss_taxonomy(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per model: zero Dice split into empty prediction and wrong place.

    This is TODO item 1. `tables` is what `lesionsize.load_lineage` returns, so
    it covers whatever model set the caller asked for -- including the ten Stage C
    distillation arms that `headline_all.csv` reports with a single miss column.

    Sorted by zero-Dice count descending, then by wrong-place count, so the arms
    whose misses are misplacements rather than blanks surface at the top.
    """
    rows = []
    for name, df in tables.items():
        rows.append({"model": name, **_miss_counts(df),
                     "mean_dice": float(df["dice"].mean()),
                     "median_dice": float(df["dice"].median())})
    out = pd.DataFrame(rows)
    return out.sort_values(["zero_dice_n", "wrong_place_n", "model"],
                           ascending=[False, False, True]).reset_index(drop=True)


def miss_taxonomy_by(tables: dict[str, pd.DataFrame], by: str = "skin_tone_category",
                     key: pd.DataFrame | None = None) -> pd.DataFrame:
    """The same three columns per (model, stratum). Long form, one row per cell.

    `by="skin_tone_category"` strata come from the manifest join `report.normalize`
    already performed. `by="size"` needs `key` -- `lesionsize.assign_bins`' frame --
    and uses its `bin` column, so Stage O's size strata are the SAME deciles
    `LESION_SIZE_RESULTS` cut rather than a second, silently different binning.

    `n_subjects` is a column and not a footnote, for the reason
    `multiteacher.stratified_oracle` gives: a miss count from four subjects is not
    a rate, and the reader should be able to see that without being told.
    """
    rows = []
    for name, df in tables.items():
        d = df.sort_values("stem").reset_index(drop=True)
        if by == "size":
            if key is None:
                raise ValueError(
                    "by='size' needs the key frame from lesionsize.assign_bins, so "
                    "Stage O's deciles are the same cut LESION_SIZE_RESULTS used. "
                    "Pass key=.")
            k = key.sort_values("stem").reset_index(drop=True)
            if not (k["stem"].to_numpy() == d["stem"].to_numpy()).all():
                raise ValueError(f"key frame and '{name}' cover different images")
            strata = k["bin"].to_numpy()
        else:
            if by not in d.columns:
                raise KeyError(f"'{name}' has no {by!r} column; "
                               f"was it loaded through report.normalize?")
            strata = d[by].to_numpy()

        for s in pd.unique(strata):
            m = strata == s
            rows.append({"model": name, "stratum": str(s),
                         "n_subjects": int(d.loc[m, "subject"].nunique()),
                         **_miss_counts(d, m)})
    return pd.DataFrame(rows)


def print_miss_taxonomy(tax: pd.DataFrame, top: int | None = None) -> None:
    """The headline table as text, with the two readings spelled out underneath."""
    t = tax if top is None else tax.head(top)
    print("=" * 78)
    print("MISS TAXONOMY -- zero Dice decomposed  (TODO 1)")
    print("=" * 78)
    print(f"  {'model':<34}{'n':>5}{'zeroD':>7}{'empty':>7}{'wrong':>7}{'zeroD%':>9}")
    for _, r in t.iterrows():
        flag = "  <-- misses are misplacements" if (
            r.wrong_place_n > r.empty_pred_n and r.zero_dice_n > 0) else ""
        print(f"  {r.model:<34}{r.n:>5}{r.zero_dice_n:>7}{r.empty_pred_n:>7}"
              f"{r.wrong_place_n:>7}{100 * r.zero_dice_rate:>8.1f}%{flag}")
    print()
    print("  zero Dice   = empty prediction + wrong place, by construction.")
    print("  empty       the model found nothing. Shows the clinician nothing and")
    print("              invites a second look.")
    print("  wrong place the model outlined a region with ZERO overlap. A confident")
    print("              assertion in the wrong location -- the worse failure.")
    print()
    print("  The study publishes only the union. Quote all three.")


# ═════════════════════════════════════════════════════════════════════════════
# PART 2 -- skin tone x lesion size for the distillation arms
# ═════════════════════════════════════════════════════════════════════════════
#: The arms `fairness_stats.csv` omits. Every one has a per-image CSV in
#: RESULT_AUGUST_08 under a `per_image_distill_` prefix, which is why the
#: 24-model fairness export -- built from the unprefixed names -- missed them.
STAGE_C_ARMS: tuple[str, ...] = (
    "p3_adaptive",
    "p3_adaptive_boundary",
    "p3_adaptive_group",
    "p3_adaptive_full",
    "p3_adaptive_hard",
    "p2_cwd_b5_to_b0",
    "p2_bpkd_b5_to_b0",
    "p2_ensemble_uniform",
    "expA_b5_to_b0_response",
    "x_angular_b5_to_b0",
)

#: The distilled/gated arms present in the lineage but absent from
#: `lesionsize.DEFAULT_MODELS`, so the Stage O tables cover every KD arm the
#: study has rather than the subset an earlier export happened to list.
EXTRA_KD_ARMS: tuple[str, ...] = (
    "fastscnn_distilled", "fastscnn_rgkd",
    "lraspp_mobilenetv3_rgkd",
    "ppmobileseg_tiny_distilled", "ppmobileseg_tiny_rgkd",
    "topformer_tiny_distilled", "topformer_tiny_rgkd",
)


#: Roots that have held per-image CSVs on the machines this project runs on.
#:
#: WHY THIS LIST EXISTS AND WHY IT IS NOT A SINGLE PATH. `FINAL_RESULT/` is where
#: the shipped laptop bundle keeps the current lineage and **it does not exist on
#: ORC at all** -- there, outputs land under the work directory on scratch. A
#: hard-coded `LINEAGE="FINAL_RESULT/RESULT_AUGUST_08"` therefore runs on the
#: laptop and raises FileNotFoundError on ORC, for a directory that was never
#: going to be there. That is exactly what happened on 2026-08-12.
#:
#: This is the same list `bruise_all_models.ipynb` uses, kept in sync deliberately.
#: See handbook §10.3 for the per-host table.
EXTRA_ROOTS: tuple[str, ...] = (
    "/scratch/tbommawa/bruise_work",
    "/scratch/tbommawa/bruise_work/outputs",
    "/scratch/tbommawa/BRUISE_UNIFIED",
)


def load_tables(env, extra_roots=EXTRA_ROOTS, models: tuple | list | None = None,
                verbose: bool = True):
    """Every per-image table on THIS machine, from every root. `(tables, key, found)`.

    HOST-PORTABLE BY CONSTRUCTION, which is the whole reason this exists rather
    than a call to `lesionsize.load_lineage` with a fixed lineage string. It scans
    `allmodels.SEARCH_HINTS` under the bundle root, the work dir and every entry in
    `extra_roots`, so the laptop's `FINAL_RESULT/RESULT_AUGUST_08` and ORC's
    `<work>/outputs` both resolve without the caller naming either.

    It also merges ACROSS roots. `lesionsize.load_lineage` picks one directory, and
    on ORC the best single directory holds 5 models where Stage O wants 35 -- the
    tables are spread over `results/`, `STAGE_M_RESULTS/`, `STAGE_N4_RESULTS/` and
    `outputs/`. `allmodels.discover` was written for precisely that and is reused
    here rather than reimplemented.

    COHORTS, and why only one is returned. Tables from different splits or
    different mask versions cannot share a decile cut, and `lesionsize.assign_bins`
    correctly RAISES when two tables disagree about an image's GT area. Rather than
    let one stale file kill the sweep, `allmodels` groups tables by (stem set,
    GT-area vector) and labels the largest coherent group `REFERENCE`. Only that
    group is returned, because everything downstream here compares models to each
    other. The rest are visible in the returned `found` list with their reason.
    """
    import pandas as pd

    from . import allmodels as AM

    man = {s: pd.read_csv(env.manifests / f"{s}.csv")
           for s in ("train", "val", "test")}
    found = AM.discover(env, extra_roots, verbose=verbose)
    tables, found = AM.load_all(env, found, man, verbose=verbose)
    keys = AM.assign_bins(tables, found)

    cohort_of = {f.model: f.cohort for f in found if f.status == "ok"}
    ref = [m for m, c in cohort_of.items() if c == "REFERENCE"]
    if not ref:
        raise FileNotFoundError(
            "no reference cohort was formed -- no two per-image tables scored the "
            "same images with the same mask areas.\n"
            f"  roots scanned: {[str(r) for r in AM.search_roots(env, extra_roots)]}\n"
            "  Add the directory your CSVs are actually in to EXTRA_ROOTS. On ORC "
            "that is usually <work>/outputs; see handbook §10.3.")

    # `allmodels` keeps the export's `distill_` prefix in the model name;
    # `lesionsize.load_lineage` strips it. Resolve BOTH spellings and return the
    # unprefixed one, so a table generated on the laptop through lesionsize and one
    # generated on ORC through here carry the same model names -- otherwise the two
    # hosts produce CSVs that cannot be diffed and nothing says why.
    alias = {}
    for m in ref:
        alias[m] = m
        if m.startswith("distill_"):
            alias.setdefault(m[len("distill_"):], m)

    if models is not None:
        want = tuple(models)
        keep = [(m, alias[m]) for m in want if m in alias]
        absent = [m for m in want if m not in alias]
        if verbose and absent:
            print(f"  NOT ON THIS MACHINE ({len(absent)}): {', '.join(absent)}")
        if verbose:
            print(f"  present in the reference cohort: {len(keep)} of {len(want)}")
        if not keep:
            raise FileNotFoundError(
                f"none of the {len(want)} requested models is in the reference "
                f"cohort on this machine. Present: {sorted(ref)[:12]} ...\n"
                f"  roots scanned: "
                f"{[str(r) for r in AM.search_roots(env, extra_roots)]}\n"
                f"  Add the directory your CSVs are in to EXTRA_ROOTS "
                f"(handbook §10.3).")
    else:
        keep = [(m if not m.startswith("distill_") else m[len("distill_"):], m)
                for m in ref]

    out = {name: tables[src] for name, src in keep}
    key = keys["REFERENCE"]
    if verbose:
        print(f"\ntables : {len(out)} models in the reference cohort")
        print(f"binned : {len(key)} images, {key.subject.nunique()} subjects")
    return out, key, found


def all_models() -> tuple[str, ...]:
    """`lesionsize.DEFAULT_MODELS` plus every KD arm it leaves out. Deduplicated.

    Order is preserved from DEFAULT_MODELS so the shared rows of a Stage O table
    and a LESION_SIZE_RESULTS table line up when read side by side.
    """
    from . import lesionsize as LS

    seen, out = set(), []
    for m in tuple(LS.DEFAULT_MODELS) + STAGE_C_ARMS + EXTRA_KD_ARMS:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return tuple(out)


def best_distilled(tax: pd.DataFrame, tables: dict[str, pd.DataFrame],
                   n: int = 5) -> list[str]:
    """The KD arms to lead the fairness report with: best mean Dice, ties broken
    by zero-Dice count.

    "Best performing distilled models (reported in the latest slide deck)" is the
    TODO's phrasing, and the deck's ranking is mean Dice. Ties are broken on
    misses rather than on median, because §1 of the handbook says a Dice gap
    under ~0.05 is inside the label noise floor and the miss column is the one
    that has ever separated arms.
    """
    kd = {m for m in tables if m in set(STAGE_C_ARMS) | set(EXTRA_KD_ARMS)
          or any(s in m for s in ("_distilled", "_b2kd", "_rgkd", "_mtkd", "_itakd"))}
    t = tax[tax.model.isin(kd)].sort_values(
        ["mean_dice", "zero_dice_n"], ascending=[False, True])
    return t.model.head(n).tolist()


def distilled_fairness(tables: dict[str, pd.DataFrame], key: pd.DataFrame,
                       models: tuple | list | None = None,
                       n_boot: int = 10000, seed: int = 0,
                       verbose: bool = True) -> dict[str, pd.DataFrame]:
    """TODO item 3 -- skin tone and lesion size for the distilled arms.

    Every frame here comes out of `lesionsize`, not out of a second
    implementation. The only thing Stage O adds is the model list: the same
    functions over the arms the 24-model fairness export left out, plus the miss
    taxonomy cross-tabbed both ways so the wrong-place column is available per
    group and per size decile.

    Returns the frames keyed by the filename stem they are saved under.
    """
    from . import lesionsize as LS

    sub = ({m: t for m, t in tables.items() if m in set(models)}
           if models is not None else tables)
    if not sub:
        raise ValueError(f"none of {models} is in the loaded tables")

    out = {
        "headline":            LS.headline(sub, key),
        "by_bin":              LS.by_bin(sub, key),
        "size_by_ita":         LS.size_by_ita(sub, key, verbose=verbose),
        "fairness_recall":     LS.fairness_conditioned(
            sub, key, endpoint="mean_recall", n_boot=n_boot, seed=seed,
            verbose=verbose),
        "fairness_zero_dice":  LS.fairness_conditioned(
            sub, key, endpoint="zero_dice_rate", n_boot=n_boot, seed=seed,
            verbose=False),
        "miss_by_ita":         miss_taxonomy_by(sub, "skin_tone_category"),
        "miss_by_size":        miss_taxonomy_by(sub, "size", key),
    }
    if verbose:
        print(f"\ndistilled_fairness: {len(sub)} models, "
              f"{len(key)} images, {key.subject.nunique()} subjects")
    return out


# ═════════════════════════════════════════════════════════════════════════════
# PART 3 -- ITA-group-routed gated multi-teacher KD
# ═════════════════════════════════════════════════════════════════════════════
#: Pre-registered pool. See the module docstring for why B2 and U-Net are out.
#: `medsam_ft` is a Stage N4 arm and is loaded through `samprobe`, not through
#: the registry -- `resolve_pool` handles both kinds.
POOL: tuple[str, ...] = (
    "segformer_b5_teacher",
    "deeplabv3plus_r50",
    "medsam_ft",
)

#: Stage N4 arms in the pool, which need `samprobe.load_trained` and a freshly
#: fitted temperature rather than `loaders.load_model` and a shipped one.
N4_POOL_ARMS: frozenset[str] = frozenset({"sam_ft", "medsam_ft"})

#: The pre-registered group scheme. `light_vs_rest` is the default for the reason
#: in the module docstring: it is the only collapse whose per-group Dice spread
#: exceeds the annotation-ceiling noise floor on a usable subject count.
#:
#: `five` is provided so the underpowered version can be REPORTED rather than
#: merely asserted to be underpowered -- run the gate on it, show the
#: identifiability column, and let the reader see 0.36-0.52.
SCHEMES: dict[str, dict[str, str]] = {
    "light_vs_rest": {
        "Very Light (I-II)":      "Light",
        "Light (II-III)":         "Light",
        "Intermediate (III-IV)":  "Other",
        "Tan (IV)":               "Other",
        "Brown (V)":              "Other",
        "Dark (VI)":              "Other",
        "Unclassified":           "Other",
    },
    "five": {
        "Very Light (I-II)":      "Light (II-III)",     # 12 train, 0 val images
        "Light (II-III)":         "Light (II-III)",
        "Intermediate (III-IV)":  "Intermediate (III-IV)",
        "Tan (IV)":               "Tan (IV)",
        "Brown (V)":              "Brown (V)",
        "Dark (VI)":              "Dark (VI)",
        "Unclassified":           "Intermediate (III-IV)",
    },
}
DEFAULT_SCHEME = "light_vs_rest"

#: `softmax(BETA * group_mean_dice)`. 8.0 is Stage M's pre-registered value,
#: reused deliberately: the two stages then differ in the ROUTING KEY alone --
#: per image vs per group -- and not also in how sharply the weights concentrate.
BETA = 8.0

#: Stage H's gate bounds, inherited rather than re-tuned, for the same reason.
GATE_LO, GATE_HI = 0.10, 0.50

#: Stage C, measured: +0.0258 of oracle teacher gain bought +0.0068 of student
#: gain. Same provenance and same use as Stage M's -- the gate projects through a
#: measured transfer rate rather than an assumption, and prints it so a reader
#: can reject it.
TRANSFER_RATE = 0.0068 / 0.0258
MARGIN = 0.01

#: The identifiability bar. A per-group argmax that survives subject resampling
#: less than three times in four is not a teacher ranking, it is a draw, and a
#: pool routed on it is Stage C's uniform ensemble with extra steps.
MIN_IDENTIFIABILITY = 0.75

STAGE_O_FAMILIES: tuple[str, ...] = (
    "segformer_b0_itakd",
    "lraspp_mobilenetv3_itakd",
)

STUDENT_ARCH = {
    "segformer_b0_itakd": "segformer",
    "lraspp_mobilenetv3_itakd": "lraspp_mobilenetv3",
}

#: One-variable contrasts. Same controls Stage M used, so Stage O, Stage M and
#: Stage A are all scored on one ruler and `itakd - mtkd` is readable directly.
CONTROL_FOR = {
    "segformer_b0_itakd": "segformer_b0_distilled",
    "lraspp_mobilenetv3_itakd": "lraspp_mobilenetv3_b2kd",
}

#: Stage M's measured wall clock, adjusted for a three-teacher pool with MedSAM
#: replacing B2 and U-Net. Not calibrated by a finished Stage O run.
COST_HOURS = {"segformer_b0_itakd": 2.4, "lraspp_mobilenetv3_itakd": 2.0}

ACTIVE_ARM: str | None = None

#: Set by the loader wrapper, read by the loss. See the module docstring's
#: "WHERE THE GROUP REACHES THE LOSS".
CURRENT_GROUPS: "np.ndarray | None" = None


def is_stage_o(family: str | None) -> bool:
    return family in STAGE_O_FAMILIES


def group_order(scheme: str = DEFAULT_SCHEME) -> tuple[str, ...]:
    """The scheme's group labels in a fixed order, so a weight row index means
    the same thing in the gate, the JSON and the loss."""
    if scheme not in SCHEMES:
        raise KeyError(f"unknown scheme {scheme!r}; have {sorted(SCHEMES)}")
    seen, out = set(), []
    for v in SCHEMES[scheme].values():
        if v not in seen:
            seen.add(v)
            out.append(v)
    return tuple(out)


def collapse(categories, scheme: str = DEFAULT_SCHEME) -> np.ndarray:
    """Map `skin_tone_category` values onto the scheme's groups.

    RAISES on a category the scheme does not name. The alternative -- mapping the
    unknown to a default group -- is how 12 `Very Light (I-II)` training images
    would silently be routed with weights fitted on a group they are not in.
    """
    table = SCHEMES[scheme] if scheme in SCHEMES else None
    if table is None:
        raise KeyError(f"unknown scheme {scheme!r}; have {sorted(SCHEMES)}")
    vals = np.asarray(pd.Series(categories).astype(str))
    unknown = sorted(set(vals) - set(table))
    if unknown:
        raise KeyError(
            f"scheme {scheme!r} does not map {unknown}. Add them to SCHEMES "
            f"deliberately -- a default group would route those images with "
            f"weights fitted on a group they are not in.")
    return np.array([table[v] for v in vals], dtype=object)


def group_index(categories, scheme: str = DEFAULT_SCHEME) -> np.ndarray:
    """`collapse` as integer indices into `group_order(scheme)`."""
    order = {g: i for i, g in enumerate(group_order(scheme))}
    return np.array([order[g] for g in collapse(categories, scheme)], dtype=np.int64)


# ── registering the arms ─────────────────────────────────────────────────────
def register_specs() -> list[str]:
    """Teach `loaders` about B5, the N4 pool arms and the Stage O students.

    Idempotent. Same mechanism and reason as `multiteacher.register_specs`: a
    distilled variant is a new FAMILY on an existing ARCHITECTURE, and loaders.py
    is a build output whose edits the next bundle build would revert.
    """
    from . import loaders as L
    from . import multiteacher as MT

    added = MT.register_specs()                 # B5 + the Stage M students

    # Teach multiteacher's CONTROL_FOR about Stage O's arms. `control_batch` is
    # multiteacher's helper and resolves the control by looking THAT dict up, so
    # a Stage O family it has never heard of raises KeyError -- which is exactly
    # what happened on the first ORC run, inside train_arms, after the preflight
    # had already passed. Registered here rather than duplicating control_batch,
    # because that function carries the accumulation arithmetic that keeps the
    # arm's EFFECTIVE batch identical to its control's, and a second copy of that
    # is a second thing to get wrong.
    for family, control in CONTROL_FOR.items():
        MT.CONTROL_FOR.setdefault(family, control)

    for family, arch in STUDENT_ARCH.items():
        if family not in L.FAMILY_SPEC:
            L.FAMILY_SPEC[family] = {
                "arch": "segformer" if arch == "segformer" else arch,
                "size": "b0" if arch == "segformer" else None,
                "distill": True, "teacher": "POOL", "kd": "group_routed",
            }
            added.append(family)
        if arch != "segformer" and family not in L.EFFICIENT_FAMILIES:
            L.EFFICIENT_FAMILIES = L.EFFICIENT_FAMILIES + (family,)

    from . import distill_efficient as DE
    for family, arch in STUDENT_ARCH.items():
        if arch != "segformer":
            DE.register_student_alias(family, arch)
    return added


# ── the pool: two kinds of member, one interface ─────────────────────────────
def n4_run_dir(env, arm: str, results_root: Path | None = None) -> Path:
    """Where Stage N4 put `<arm>__seed0/best.pt`.

    Stage N4 trains into `STAGE_N4_RESULTS/runs/`, not `env.runs`, precisely so
    the reporting stages cannot see its arms -- which means the registry cannot
    find them either and this has to be a path.
    """
    root = Path(results_root) if results_root is not None \
        else Path(env.root) / "STAGE_N4_RESULTS"
    return root / "runs" / f"{arm}__seed0"


def load_pool_member(env, reg, cfg: dict, name: str, seed: int = 0,
                     man640: dict | None = None, n4_root: Path | None = None,
                     verbose: bool = True):
    """`(family, model, temperature)` for one pool member, whichever kind it is.

    Registry teachers go through `multiteacher.load_teacher_model`, which already
    handles the three checkpoint layouts AND B5's `model.` -> `net.` wrapper
    remap; reimplementing that here is how a second, subtly different B5 loader
    gets created.

    Stage N4 arms go through `samprobe.load_trained` and have their temperature
    FITTED on validation, because nothing shipped one for them. Never falls back
    to T = 1: an uncalibrated teacher's soft label is the hard label with extra
    steps, and the arm would measure nothing.
    """
    import torch

    from . import multiteacher as MT

    if name in N4_POOL_ARMS:
        from . import samprobe as SP

        run_dir = n4_run_dir(env, name, n4_root)
        if not (run_dir / "best.pt").exists():
            raise FileNotFoundError(
                f"{name} has no checkpoint at {run_dir / 'best.pt'}.\n"
                f"  Stage N4 trained into STAGE_N4_RESULTS/runs/ on the GPU box; "
                f"only its tables were synced to the laptop bundle. Either run "
                f"Stage O where those runs live, or drop {name} from POOL and say "
                f"so in the write-up -- do not substitute a different teacher "
                f"silently.")
        model = SP.load_trained(env, name, run_dir).to(env.device).eval()
        for p in model.parameters():
            p.requires_grad_(False)

        if man640 is None:
            raise RuntimeError(
                f"{name} ships no calibration and man640 was not given, so its "
                f"temperature cannot be fitted. Pass man640=.")
        from .data import make_loader
        from .engine import calibrate_temperature
        loader = make_loader(man640["val"], env.cache640, cfg["img_size"],
                             cfg.get("eval_batch", 4), False, 0, 0)
        cal = calibrate_temperature(model, loader, env.device, cfg.get("amp", True))
        t = float(cal["temperature"])
        if not np.isfinite(t) or t <= 0:
            raise ValueError(f"{name}: fitted temperature {t} is unusable")
        if verbose:
            print(f"  [cal] {name}: T={t:.4f} (fitted on 134 val images)")
        return name, model, t

    run = MT.resolve_teachers(env, reg, (name,), seed)[0]
    t = MT.teacher_temperature(env, run, cfg, man640, verbose=verbose)
    model, _cut = MT.load_teacher_model(env, run, cfg, verbose=verbose)
    model.to(env.device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    del torch
    return run.family, model, t


def val_group_matrix(env, reg, cfg: dict, man640: dict, meta: pd.DataFrame,
                     pool=POOL, seed: int = 0, student: str | None = None,
                     n4_root: Path | None = None, cache: Path | None = None,
                     verbose: bool = True) -> pd.DataFrame:
    """`[stem, subject, skin_tone_category, gt_positive_pixels, <Dice per teacher>]`.

    One forward pass per pool member over the 134 validation images. Cached to
    CSV because the gate is meant to be re-read and argued about without paying
    for it again; delete the file to force a recompute.

    Registry members are scored through `multiteacher.score_on_split` -- the same
    function Stage M's gate used, at each teacher's own val-fitted cut. Stage N4
    arms have no `Run` and are scored here at the cut Stage N4 fitted for them,
    read from its tables.

    A NOTE ON WHAT THESE NUMBERS MEAN, inherited from `score_on_split`: every
    teacher's cut was fitted on these same 134 images, so every absolute val Dice
    below is mildly optimistic. The bias is shared, so it cancels in the GAIN --
    which is what the gate reads -- and does not cancel in the per-teacher
    columns, which are context and never a result.
    """
    if cache is not None and Path(cache).exists():
        if verbose:
            print(f"  [cached] {cache}")
        return pd.read_csv(cache)

    from . import multiteacher as MT

    register_specs()
    out = None
    for name in pool:
        if verbose:
            print(f"  scoring {name} on val ...", flush=True)
        if name in N4_POOL_ARMS:
            pi = _score_n4_arm(env, cfg, man640, name, n4_root, verbose=verbose)
        else:
            run = MT.resolve_teachers(env, reg, (name,), seed)[0]
            pi = MT.score_on_split(env, run, cfg, man640, "val")
        col = pi[["stem", "dice"]].rename(columns={"dice": name})
        if out is None:
            out = col.merge(pi[["stem", "gt_positive_pixels"]], on="stem",
                            validate="one_to_one")
        else:
            out = out.merge(col, on="stem", validate="one_to_one")

    if student is not None:
        s = reg.get(f"{student}__seed{seed}")
        if s is None or getattr(s, "weights", None) is None:
            raise FileNotFoundError(f"student {student}__seed{seed} has no checkpoint")
        pi = MT.score_on_split(env, s, cfg, man640, "val")
        out = out.merge(pi[["stem", "dice"]].rename(columns={"dice": student}),
                        on="stem", validate="one_to_one")

    keep = [c for c in ("stem", "subject", "skin_tone_category", "ita_group_index_5")
            if c in meta.columns]
    out = out.merge(meta[keep], on="stem", how="left", validate="one_to_one")
    if out["subject"].isna().any():
        raise ValueError(
            f"{int(out['subject'].isna().sum())} val stems have no manifest row")

    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(cache, index=False)
        if verbose:
            print(f"  wrote {cache}")
    return out


def _score_n4_arm(env, cfg: dict, man640: dict, arm: str,
                  n4_root: Path | None = None, split: str = "val",
                  verbose: bool = True) -> pd.DataFrame:
    """Per-image table for a Stage N4 arm at the cut Stage N4 fitted for it.

    The cut is READ from `STAGE_N4_RESULTS/tables/<split>_summaries.json`, never
    re-fitted: re-fitting it here would make Stage O's copy of MedSAM a
    different operating point from the one Stage N4 reported, and the two stages'
    numbers would stop being comparable for a reason nobody would look for.
    """
    import gc

    import torch

    from . import samprobe as SP
    from .data import make_loader
    from .evaluate import evaluate_at_cut

    root = Path(n4_root) if n4_root is not None else Path(env.root) / "STAGE_N4_RESULTS"
    summ = root / "tables" / f"{split}_summaries.json"
    if not summ.exists():
        raise FileNotFoundError(
            f"{summ} is absent, so {arm}'s val-fitted cut cannot be read. Stage O "
            f"will not re-fit it -- that would silently change the operating point "
            f"Stage N4 reported.")
    key = f"{arm}__seed0" if split == "val" else f"{arm}__seed0__{split}"
    js = json.loads(summ.read_text())
    if key not in js:
        raise KeyError(f"{summ} has no {key!r}; have {sorted(js)}")
    cut = float(js[key]["cut"])

    model = SP.load_trained(env, arm, n4_run_dir(env, arm, n4_root)).to(env.device)
    model.eval()
    on_cuda = str(env.device).startswith("cuda")
    amp = bool(cfg.get("amp", True)) and on_cuda
    batch = cfg.get("eval_batch", 4) if on_cuda else 2
    loader = make_loader(man640[split], env.cache640, cfg["img_size"], batch,
                         False, cfg.get("workers", 0), 0)
    with torch.inference_mode():
        per_image, _ = evaluate_at_cut(model, loader, env.device, cut, amp)

    del model, loader
    gc.collect()
    if on_cuda:
        torch.cuda.empty_cache()
    if verbose:
        print(f"    {arm} at Stage N4's cut {cut:+.3f}: "
              f"mean Dice {per_image['dice'].mean():.4f}")
    per_image.attrs["cut"] = cut
    return per_image


# ── group weights ────────────────────────────────────────────────────────────
def group_weights(matrix: pd.DataFrame, pool=POOL, scheme: str = DEFAULT_SCHEME,
                  beta: float = BETA) -> pd.DataFrame:
    """`softmax(beta * group_mean_val_dice)` -- one weight row per group.

    Long form: `[group, teacher, mean_val_dice, weight, n_images, n_subjects]`.
    Weights within a group sum to 1, which `self_test` asserts.

    RAISES if any group in the scheme has no validation images. That is not a
    hypothetical: under `scheme="five"` the training set's 12
    `Very Light (I-II)` images have no validation counterpart, and a group with
    no data would otherwise get uniform weights -- turning that slice of the arm
    into `p2_ensemble_uniform` while the table still said "group-routed".
    """
    cols = [c for c in pool if c in matrix.columns]
    if len(cols) < 2:
        raise ValueError(f"need >=2 teacher columns, found {cols}; matrix has "
                         f"{list(matrix.columns)}")
    df = matrix.copy()
    df["_g"] = collapse(df["skin_tone_category"], scheme)

    order = group_order(scheme)
    missing = [g for g in order if not (df["_g"] == g).any()]
    if missing:
        raise ValueError(
            f"scheme {scheme!r} has group(s) {missing} with no validation images, "
            f"so no weights can be fitted for them. Training images in those "
            f"groups would fall back to uniform weights and the arm would "
            f"silently become a uniform ensemble on that slice. Use "
            f"scheme='light_vs_rest', which every val image populates.")

    rows = []
    for g in order:
        m = (df["_g"] == g).to_numpy()
        means = np.array([df.loc[m, c].mean() for c in cols], float)
        z = beta * means
        w = np.exp(z - z.max())
        w = w / w.sum()
        for c, mu, wi in zip(cols, means, w):
            rows.append({"group": g, "teacher": c, "mean_val_dice": float(mu),
                         "weight": float(wi), "n_images": int(m.sum()),
                         "n_subjects": int(df.loc[m, "subject"].nunique())})
    return pd.DataFrame(rows)


def weight_array(weights: pd.DataFrame, pool=POOL,
                 scheme: str = DEFAULT_SCHEME) -> np.ndarray:
    """The long weight frame as `[n_groups, K]`, rows in `group_order` order and
    columns in `pool` order. The loss indexes this by group, so the two orders
    are the contract between the gate's table and the training run."""
    order = group_order(scheme)
    cols = [c for c in pool if c in set(weights.teacher)]
    W = np.zeros((len(order), len(cols)), float)
    piv = weights.pivot(index="group", columns="teacher", values="weight")
    for i, g in enumerate(order):
        if g not in piv.index:
            raise KeyError(f"weight frame has no row for group {g!r}")
        W[i] = [float(piv.loc[g, c]) for c in cols]
    bad = np.abs(W.sum(axis=1) - 1.0) > 1e-6
    if bad.any():
        raise ValueError(f"weight rows {np.where(bad)[0].tolist()} do not sum to 1")
    return W


# ── the gate ─────────────────────────────────────────────────────────────────
def _cluster_boot(fn, subjects: np.ndarray, reps: int, seed: int) -> np.ndarray:
    """Subject-cluster bootstrap of `fn(index_array)`. Same definition and same
    reason as `multiteacher._cluster_boot`: 1016 images from 143 subjects, several
    of them near-duplicates of one pose, so resampling images would treat those as
    independent evidence and shrink every interval."""
    uniq = np.unique(subjects)
    idxby = {s: np.where(subjects == s)[0] for s in uniq}
    rng = np.random.default_rng(seed)
    draws = np.empty(reps)
    for r in range(reps):
        pick = rng.choice(uniq, uniq.size, replace=True)
        draws[r] = fn(np.concatenate([idxby[s] for s in pick]))
    return draws


def identifiability(matrix: pd.DataFrame, pool=POOL, scheme: str = DEFAULT_SCHEME,
                    reps: int = 4000, seed: int = 0) -> pd.DataFrame:
    """Per group: does the best teacher survive resampling the subjects?

    THE CLAUSE STAGE M DID NOT HAVE, and the reason this stage exists in this
    shape. For each group, resample its subjects with replacement and record how
    often the argmax teacher is the same one the full sample chose.

    `p_argmax_stable` near 1/K is a draw dressed as a ranking. On the five-group
    scheme over six candidate teachers this column reads 0.36-0.52 -- so an arm
    routed on that argmax is fitting sampling noise, and it would produce a
    fourth null indistinguishable from the first three.

    `margin` is the full-sample gap between the best and second-best teacher in
    the group. Read it against §1's ~0.05 annotation-ceiling noise floor: a
    margin an order of magnitude below that is not a teacher difference, whatever
    its bootstrap says.
    """
    cols = [c for c in pool if c in matrix.columns]
    df = matrix.copy()
    df["_g"] = collapse(df["skin_tone_category"], scheme)

    rows = []
    for g in group_order(scheme):
        sub = df[df["_g"] == g]
        if sub.empty:
            continue
        D = sub[cols].to_numpy(float)
        means = D.mean(axis=0)
        best = int(means.argmax())
        second = int(np.argsort(means)[-2]) if len(cols) > 1 else best

        def wins(ix, _D=D, _best=best):
            return float(_D[ix].mean(axis=0).argmax() == _best)

        draws = _cluster_boot(wins, sub["subject"].to_numpy(), reps, seed)
        rows.append({
            "group": g, "n_images": len(sub),
            "n_subjects": int(sub["subject"].nunique()),
            "best_teacher": cols[best],
            "best_dice": float(means[best]),
            "runner_up": cols[second],
            "runner_up_dice": float(means[second]),
            "margin": float(means[best] - means[second]),
            "spread": float(means.max() - means.min()),
            "p_argmax_stable": float(draws.mean()),
            "identifiable": bool(draws.mean() >= MIN_IDENTIFIABILITY),
        })
    return pd.DataFrame(rows)


def ita_group_gate(matrix: pd.DataFrame, pool=POOL, scheme: str = DEFAULT_SCHEME,
                   beta: float = BETA, student: str | None = None,
                   reps: int = 4000, seed: int = 0, margin: float = MARGIN,
                   transfer_rate: float = TRANSFER_RATE,
                   min_identifiability: float = MIN_IDENTIFIABILITY) -> dict:
    """Is a per-ITA-group teacher weighting worth training? Validation only.

    Reports, all with a subject-cluster bootstrap:

      - each teacher's val Dice, and the group-weighted ensemble's
      - GROUP-WEIGHTING GAIN over the best single teacher, with CI and P(>0).
        This is the quantity a student could inherit -- NOT the per-image oracle,
        which Stage M used and which no group weighting can reach.
      - the per-image oracle as an upper bound, for reference only, so the
        distance between "what routing could give" and "what grouping can give"
        is visible instead of implied.
      - IDENTIFIABILITY per group (see `identifiability`).
      - projected student gain = weighting gain x Stage C's measured transfer rate.

    THE GATE RULE, fixed here before any number is looked at:

        open  iff  the weighting-gain CI clears zero
              AND  projected student gain > margin (one Dice point)
              AND  at least one group's argmax is identifiable at p >= 0.75

    The third clause is new. Stage M's gate opened on a real oracle gain of
    +0.048 and every contrast came back inconclusive; Stage C's opened on +0.026
    and delivered +0.007. Both gates were measuring headroom that existed. What
    neither asked was whether the ROUTING KEY was estimable, and on five ITA
    groups over 20 validation subjects it is not.

    The miss endpoint is reported alongside and deliberately NOT ANDed in: this
    study is judged on complete misses, so a pool that contains misses its best
    single member does not is interesting even when Dice says nothing.
    """
    cols = [c for c in pool if c in matrix.columns]
    if len(cols) < 2:
        raise ValueError(f"need >=2 teacher columns, found {cols}")

    df = matrix.copy()
    df["_g"] = collapse(df["skin_tone_category"], scheme)
    order = group_order(scheme)

    weights = group_weights(matrix, pool, scheme, beta)
    W = weight_array(weights, pool, scheme)
    gi = np.array([order.index(g) for g in df["_g"]], dtype=int)

    D = df[cols].to_numpy(float)                       # [N, K]
    subjects = df["subject"].to_numpy()

    # The group-weighted ensemble's per-image Dice, as the student would see it:
    # a convex combination of the teachers' Dice under that image's group weights.
    # This is a linear surrogate -- the true fused-probability Dice is not a
    # convex combination of the members' Dice -- and it is used for the GATE only,
    # where being conservative is the point. Stated rather than buried.
    fused = (D * W[gi]).sum(axis=1)
    orc = D.max(axis=1)
    all_ix = np.arange(len(D))

    def gain(ix):
        return fused[ix].mean() - D[ix].mean(axis=0).max()

    point = float(gain(all_ix))
    draws = _cluster_boot(gain, subjects, reps, seed)
    lo, hi = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))

    per_teacher = {c: float(D[:, k].mean()) for k, c in enumerate(cols)}
    best_single = max(per_teacher, key=per_teacher.get)
    ident = identifiability(matrix, pool, scheme, reps, seed)

    res: dict = {
        "scheme": scheme, "groups": list(order), "beta": float(beta),
        "n_val_images": int(len(df)), "n_subjects": int(df["subject"].nunique()),
        "teachers": cols,
        "per_teacher_val_dice": per_teacher,
        "best_single_teacher": best_single,
        "best_single_val_dice": per_teacher[best_single],
        "group_weighted_val_dice": float(fused.mean()),
        "weighting_gain_over_best_single": point,
        "weighting_gain_ci95": [lo, hi],
        "p_gain_positive": float((draws > 0).mean()),
        "per_image_oracle_val_dice": float(orc.mean()),
        "per_image_oracle_gain": float(orc.mean() - per_teacher[best_single]),
        "transfer_rate_used": float(transfer_rate),
        "transfer_rate_provenance": "Stage C: student +0.0068 realised from oracle +0.0258",
        "projected_student_gain": float(point * transfer_rate),
        "projected_student_gain_ci95": [lo * transfer_rate, hi * transfer_rate],
        "margin": float(margin),
        "min_identifiability": float(min_identifiability),
        "identifiability": ident.to_dict("records"),
        "n_identifiable_groups": int(ident["identifiable"].sum()),
        "weights": weights.to_dict("records"),
    }

    if student is not None and student in df.columns:
        s = df[student].to_numpy(float)
        res["student"] = student
        res["student_val_dice"] = float(s.mean())
        res["weighting_gain_over_student"] = float(fused.mean() - s.mean())

    miss = (D == 0)
    res["misses_per_teacher"] = {c: int(miss[:, k].sum()) for k, c in enumerate(cols)}
    res["misses_best_single"] = int(miss[:, cols.index(best_single)].sum())
    res["misses_pool_oracle"] = int(miss.all(axis=1).sum())
    res["miss_gate_open"] = bool(res["misses_pool_oracle"] < res["misses_best_single"])

    res["GATE_gain_ci_clears_zero"] = bool(lo > 0)
    res["GATE_projection_clears_margin"] = bool(res["projected_student_gain"] > margin)
    res["GATE_key_identifiable"] = bool(res["n_identifiable_groups"] >= 1)
    res["GATE_run_method"] = bool(res["GATE_gain_ci_clears_zero"]
                                 and res["GATE_projection_clears_margin"]
                                 and res["GATE_key_identifiable"])
    res["GATE_any"] = bool(res["GATE_run_method"] or res["miss_gate_open"])
    return res


def format_gate(res: dict) -> str:
    """The gate as text a human decides on. Verdict last, evidence first."""
    L = []
    a = L.append
    a("=" * 78)
    a("STAGE O -- ITA-GROUP WEIGHTING GATE   (validation only; test untouched)")
    a("=" * 78)
    a(f"  {res['n_val_images']} val images, {res['n_subjects']} subjects, "
      f"{len(res['teachers'])} teachers, scheme={res['scheme']!r} "
      f"({len(res['groups'])} groups)")
    a("")
    a("  teacher                    val Dice   misses")
    for c in res["teachers"]:
        a(f"    {c:<24} {res['per_teacher_val_dice'][c]:>7.4f}   "
          f"{res['misses_per_teacher'][c]:>5d}")
    a(f"    {'GROUP-WEIGHTED':<24} {res['group_weighted_val_dice']:>7.4f}")
    a(f"    {'per-image oracle (bound)':<24} {res['per_image_oracle_val_dice']:>7.4f}"
      f"   {res['misses_pool_oracle']:>5d}")
    a("")
    a("  WEIGHTS  (softmax(beta x group mean val Dice))")
    for w in res["weights"]:
        a(f"    {w['group']:<14} {w['teacher']:<24} {w['weight']:>6.3f}"
          f"   (dice {w['mean_val_dice']:.4f}, n={w['n_images']}, "
          f"{w['n_subjects']} subj)")
    a("")
    a("  IDENTIFIABILITY  -- is the per-group argmax estimable at all?")
    a(f"    {'group':<14}{'n':>4}{'subj':>5}  {'best':<22}{'margin':>8}"
      f"{'P(stable)':>11}")
    for r in res["identifiability"]:
        mark = "ok " if r["identifiable"] else "NO "
        a(f"  {mark} {r['group']:<14}{r['n_images']:>4}{r['n_subjects']:>5}  "
          f"{r['best_teacher']:<22}{r['margin']:>+8.4f}"
          f"{r['p_argmax_stable']:>11.2f}")
    a("")
    a(f"  group-weighting gain over {res['best_single_teacher']}: "
      f"{res['weighting_gain_over_best_single']:+.4f}  "
      f"CI95 [{res['weighting_gain_ci95'][0]:+.4f}, "
      f"{res['weighting_gain_ci95'][1]:+.4f}]  "
      f"P(>0) = {res['p_gain_positive']:.3f}")
    a(f"  per-image oracle gain (UPPER BOUND, unreachable by grouping): "
      f"{res['per_image_oracle_gain']:+.4f}")
    a(f"  x transfer rate {res['transfer_rate_used']:.3f} "
      f"({res['transfer_rate_provenance']})")
    a(f"  PROJECTED STUDENT GAIN {res['projected_student_gain']:+.4f}  "
      f"vs margin {res['margin']:+.4f}")
    a("")
    a(f"  [{'PASS' if res['GATE_gain_ci_clears_zero'] else 'fail'}] "
      f"weighting-gain CI clears zero")
    a(f"  [{'PASS' if res['GATE_projection_clears_margin'] else 'fail'}] "
      f"projected student gain clears the margin")
    a(f"  [{'PASS' if res['GATE_key_identifiable'] else 'fail'}] "
      f"routing key is identifiable in >=1 group "
      f"({res['n_identifiable_groups']}/{len(res['groups'])} at "
      f"p>={res['min_identifiability']:.2f})")
    a(f"  [{'PASS' if res['miss_gate_open'] else 'fail'}] "
      f"pool contains misses the best single teacher does not "
      f"({res['misses_pool_oracle']} vs {res['misses_best_single']})")
    a("")
    a("-" * 78)
    if res["GATE_run_method"]:
        a("  VERDICT: RUN Stage O. All three Dice clauses opened.")
    elif res["miss_gate_open"]:
        a("  VERDICT: RUN Stage O ON THE MISS ENDPOINT ONLY. Do not pre-register")
        a("           a Dice hypothesis; report Dice as non-inferiority.")
    elif not res["GATE_key_identifiable"]:
        a("  VERDICT: DO NOT RUN -- and the reason is the interesting one. The")
        a("           per-group teacher ranking is not estimable on 134 val")
        a("           images, so ANY arm routed on it is fitting sampling noise.")
        a("           This is a result about the DESIGN, not about the method:")
        a("           report the identifiability table. It is the answer to")
        a("           'why not just group by skin tone', which Stages C and M")
        a("           each asserted without measuring.")
    else:
        a("  VERDICT: DO NOT RUN. No headroom a student could inherit on either")
        a("           endpoint. A negative pre-test is a result, and it cost one")
        a("           val pass instead of a grid.")
    a("-" * 78)
    return "\n".join(L)


# ── the shims ────────────────────────────────────────────────────────────────
def set_active_arm(family: str | None) -> None:
    global ACTIVE_ARM
    ACTIVE_ARM = family


class arm:
    """Context manager scoping `ACTIVE_ARM` so a raised exception cannot leak it.

    Same contract as `reliability_kd.arm` and `multiteacher.arm`. All three shim
    families dispatch on their own global, and an arm left set after a failed run
    would silently apply group-routed KD to whatever family trained next -- a
    corruption that produces entirely plausible numbers.
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
        global CURRENT_GROUPS
        CURRENT_GROUPS = None
        return False


class _GroupTaggingLoader:
    """Wraps a training loader so each batch records its group indices.

    `engine.train_run` iterates `(x, y, _)` and throws the stem away, so this is
    how the loss learns which ITA group each image belongs to without editing the
    shared training loop. `__iter__` runs in the MAIN process -- DataLoader
    workers only collate -- and the loader, the teacher forward and the loss all
    execute synchronously inside one iteration, so the global this sets is always
    the batch the loss is about to see.

    Everything else is delegated, including `dataset` and `batch_size`, so
    anything that introspects the loader sees the real one.
    """

    def __init__(self, loader, stem_to_index: dict[str, int], strict: bool = True):
        self._loader = loader
        self._map = stem_to_index
        self._strict = strict

    def __iter__(self):
        global CURRENT_GROUPS
        for batch in self._loader:
            x, y, stems = batch
            missing = [s for s in stems if s not in self._map]
            if missing and self._strict:
                raise KeyError(
                    f"{len(missing)} training stems have no ITA group "
                    f"(first: {missing[0]!r}). The group map was built from a "
                    f"different manifest than the loader is reading, and a "
                    f"default group would route these images with weights "
                    f"fitted on a group they are not in.")
            CURRENT_GROUPS = np.array([self._map[s] for s in stems], dtype=np.int64)
            yield x, y, stems
        CURRENT_GROUPS = None

    def __len__(self):
        return len(self._loader)

    def __getattr__(self, name):
        return getattr(self._loader, name)


def build_group_map(man640: dict, scheme: str = DEFAULT_SCHEME,
                    splits=("train", "val")) -> dict[str, int]:
    """`stem -> group index`, from the manifests the loader is actually reading.

    Built from `man640` rather than from `manifests/*.csv` on disk so the map and
    the loader can never disagree about which images are in the split.
    """
    out: dict[str, int] = {}
    for split in splits:
        if split not in man640:
            continue
        df = man640[split]
        if "skin_tone_category" not in df.columns:
            raise KeyError(
                f"the {split} manifest has no skin_tone_category column, so no "
                f"group can be assigned. Stage O cannot run against this manifest.")
        idx = group_index(df["skin_tone_category"], scheme)
        out.update(dict(zip(df["stem"].astype(str), idx.tolist())))
    return out


def install_group_shim(group_map: dict[str, int], verbose: bool = True):
    """Rebind `engine.make_loader` so Stage O's TRAINING loader tags its batches.

    Patches the name in `bruisekit.engine`, not only in `bruisekit.data`:
    engine.py does `from bruisekit.data import make_loader` at import time and so
    holds the function by value -- patching data alone leaves `train_run` calling
    the original. Same failure mode `samprobe.install_n4_shim` documents for
    `build_model`.

    Only training loaders are wrapped, and only inside a Stage O `arm()`. The
    validation loader is left alone because `train_run`'s val pass computes AP
    with no teacher and no loss, so it has no use for a group.
    """
    import bruisekit.data as _bd
    import bruisekit.engine as _be

    previous = getattr(_be.make_loader, "_original", _be.make_loader)

    def make_loader(df, root, img_size, batch_size, training, workers, seed=0):
        loader = previous(df, root, img_size, batch_size, training, workers, seed)
        if training and is_stage_o(ACTIVE_ARM):
            if verbose:
                print(f"  [group] {ACTIVE_ARM}: training loader tagged with ITA "
                      f"group indices ({len(group_map)} stems mapped)")
            return _GroupTaggingLoader(loader, group_map)
        return loader

    make_loader._original = previous
    _be.make_loader = make_loader
    _bd.make_loader_stage_o = make_loader          # discoverable, not authoritative
    if verbose:
        print("engine.make_loader patched -> group-tagging for Stage O training "
              "loaders; falls through otherwise.")
    return make_loader


def install_teacher_shim(env, reg, cfg: dict, man640: dict, pool=POOL,
                         seed_mode: str = "same", fixed_seed: int = 0,
                         teacher_chunk: int = 4, n4_root: Path | None = None,
                         verbose: bool = True):
    """Rebind `engine.load_teacher` to return a STACK of pool probabilities.

    Structurally Stage M's shim -- `train_run` does `tprob = teacher(x)` and hands
    the result to the loss, so returning `[B, K, H, W]` carries the whole pool
    with no edit to the loop -- with one difference: `load_pool_member` handles
    Stage N4 arms as well as registry runs, because MedSAM has no `Run`.

    Outside a Stage O `arm()` this calls whatever was bound before, so a session
    that also trains Stage F, H or M leaves those arms on their own teachers.
    Install this AFTER those.

    `teacher_chunk` defaults to 4 rather than Stage M's 8: MedSAM is a ViT-B at
    640 with windowed attention and a 40x40 grid, a heavier forward than any
    member of Stage M's pool.
    """
    import torch

    import bruisekit.engine as _be

    if seed_mode not in ("same", "fixed"):
        raise ValueError(f"seed_mode must be 'same' or 'fixed', got {seed_mode!r}")

    register_specs()
    previous = getattr(_be.load_teacher, "_original", _be.load_teacher)
    cache: dict = {}

    def _one(name: str, seed: int):
        key = f"{name}__seed{seed}"
        if key not in cache:
            cache[key] = load_pool_member(env, reg, cfg, name, seed, man640,
                                          n4_root, verbose=verbose)
        return cache[key]

    def load_teacher(dir_from_train_run, paths: dict, device, amp: bool):
        if not is_stage_o(ACTIVE_ARM):
            return previous(dir_from_train_run, paths, device, amp)

        name = Path(dir_from_train_run).name
        try:
            student_seed = int(name.rsplit("seed", 1)[1])
        except (IndexError, ValueError):
            return previous(dir_from_train_run, paths, device, amp)
        seed = student_seed if seed_mode == "same" else fixed_seed
        members = [_one(n, seed) for n in pool]

        def teacher_fn(x):
            probs = []
            for _f, model, temperature in members:
                parts = []
                for i in range(0, x.shape[0], teacher_chunk):
                    with torch.no_grad(), torch.amp.autocast("cuda", enabled=amp):
                        z = model(x[i:i + teacher_chunk])
                    parts.append(torch.sigmoid(z.float() / temperature))
                probs.append(torch.cat(parts, dim=0) if len(parts) > 1 else parts[0])
            return torch.cat(probs, dim=1)                 # [B, K, H, W]

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
        print(f"engine.load_teacher patched -> Stage O pool of {len(pool)}; "
              f"falls through otherwise.")
    return load_teacher


def _build_group_loss_class():
    """Defined lazily so importing this module never needs torch -- reading a
    Stage O table on a laptop must not require a deep-learning stack."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from .losses import SupervisedLoss
    from .reliability_kd import image_gate, reliability

    class GroupRoutedDistillLoss(nn.Module):
        """Per-ITA-group teacher weighting, then Stage H's gated soft term.

        `teacher_prob` arrives `[B, K, H, W]` from the pool shim. Each image's
        ITA group indexes a FIXED weight row fitted on validation; the fused
        probability is that convex combination. From there this is exactly
        `ReliabilityGatedDistillLoss` on the fused map, so Stage O inherits Stage
        H's gate instead of inventing a second one -- and `itakd - mtkd` differs
        in the routing key alone.

        THE IDENTITIES THAT KEEP THE CONTRAST ONE-VARIABLE, both in `self_test`:
          - K = 1 reduces to the Stage H gated loss exactly.
          - a uniform weight matrix reproduces Stage C's `p2_ensemble_uniform`
            with a gate, i.e. Stage M at beta = 0.

        No gradient flows through the weights. They are constants fitted before
        training, not parameters: letting autograd see them would let the student
        reduce its loss by choosing which teacher it is compared against.
        """

        def __init__(self, weights: np.ndarray, alpha: float = 0.6,
                     aux_weight: float = 0.4, gate_lo: float = GATE_LO,
                     gate_hi: float = GATE_HI, eps: float = 1e-6):
            super().__init__()
            W = np.asarray(weights, dtype=np.float64)
            if W.ndim != 2:
                raise ValueError(f"weights must be [n_groups, K], got {W.shape}")
            if not np.allclose(W.sum(axis=1), 1.0, atol=1e-6):
                raise ValueError("every weight row must sum to 1")
            self.register_buffer("W", torch.tensor(W, dtype=torch.float32))
            self.alpha = float(alpha)
            self.gate_lo = float(gate_lo)
            self.gate_hi = float(gate_hi)
            self.eps = float(eps)
            self.sup = SupervisedLoss(aux_weight)
            self.reset_stats()

        def reset_stats(self) -> None:
            self._n_batches = 0
            self._n_images = 0
            self._group_counts = np.zeros(int(self.W.shape[0]), dtype=np.int64)
            self._sum_coverage = 0.0
            self._sum_fused_dice = 0.0
            self._sum_alpha_eff = 0.0

        def stats(self) -> dict:
            """What the weighting actually did. An arm that saw one group and an
            arm that saw a balanced mix are different experiments, and Dice alone
            cannot tell them apart."""
            b = max(1, self._n_batches)
            i = max(1, self._n_images)
            return {
                "loss": "GroupRoutedDistillLoss",
                "alpha_nominal": self.alpha,
                "gate_lo": self.gate_lo, "gate_hi": self.gate_hi,
                "weights": self.W.detach().cpu().numpy().tolist(),
                "batches_seen": self._n_batches, "images_seen": self._n_images,
                "images_per_group": self._group_counts.tolist(),
                "mean_coverage": self._sum_coverage / b,
                "mean_fused_teacher_soft_dice": self._sum_fused_dice / i,
                "mean_alpha_effective": self._sum_alpha_eff / b,
            }

        def forward(self, logits, aux_logits, target, teacher_prob):
            hard = self.sup(logits, aux_logits, target)

            with torch.no_grad():
                t = teacher_prob.detach().float()          # [B, K, H, W]
                y = target.detach().float()                # [B, 1, H, W]
                if t.dim() != 4:
                    raise ValueError(
                        f"expected a [B,K,H,W] teacher stack, got {tuple(t.shape)}")
                B, K = t.shape[0], t.shape[1]
                if K != int(self.W.shape[1]):
                    raise ValueError(
                        f"teacher stack has K={K} but the weight matrix was built "
                        f"for K={int(self.W.shape[1])}. The pool and the fitted "
                        f"weights disagree -- refit the weights for this pool "
                        f"rather than truncating either.")

                g = CURRENT_GROUPS
                if g is None or len(g) != B:
                    raise RuntimeError(
                        f"no ITA group vector for this batch "
                        f"(got {None if g is None else len(g)} for B={B}). "
                        f"install_group_shim() was not installed, or it was "
                        f"installed before the arm() context, so the loader is not "
                        f"tagging batches. Falling back to uniform weights would "
                        f"silently turn this arm into p2_ensemble_uniform and it "
                        f"would report a plausible number for a different "
                        f"experiment.")
                gi = torch.as_tensor(np.asarray(g), dtype=torch.long,
                                     device=t.device)
                w = self.W.to(t.device)[gi]                # [B, K]
                fused = (w[:, :, None, None] * t).sum(dim=1, keepdim=True)

                gate, dice_f = image_gate(fused, y, self.gate_lo, self.gate_hi)
                gw = gate * reliability(fused, y)
                coverage = gw.mean()

            bce = F.binary_cross_entropy_with_logits(logits, fused, reduction="none")
            soft = (gw * bce).sum() / gw.sum().clamp_min(1e-6)

            # Gated-away weight returns to the supervised term so the total loss
            # scale -- and the gradient magnitude the LR schedule was tuned
            # against -- does not move with the gate. Stage H's convention.
            alpha_eff = self.alpha + (1.0 - self.alpha) * (1.0 - coverage)
            loss = alpha_eff * hard + (1.0 - alpha_eff) * soft

            with torch.no_grad():
                self._n_batches += 1
                self._n_images += int(B)
                np.add.at(self._group_counts, np.asarray(g), 1)
                self._sum_coverage += float(coverage)
                self._sum_fused_dice += float(dice_f.reshape(int(B), -1)[:, 0].sum())
                self._sum_alpha_eff += float(alpha_eff)

            return loss

    return GroupRoutedDistillLoss


_GROUP_CLASS = None


def group_loss_class():
    global _GROUP_CLASS
    if _GROUP_CLASS is None:
        _GROUP_CLASS = _build_group_loss_class()
    return _GROUP_CLASS


def install_loss_shim(weights: np.ndarray, gate_lo: float = GATE_LO,
                      gate_hi: float = GATE_HI, verbose: bool = True):
    """Rebind `engine.DistillLoss` to a dispatcher keyed on `ACTIVE_ARM`.

    `train_run` resolves `DistillLoss` as a module global at call time, and the
    global it resolves is the one bound in `bruisekit.engine` -- so that is the
    name that moves. Outside a Stage O `arm()` the dispatcher returns whatever was
    bound before, which in a session that also trains Stage H or M is that stage's
    dispatcher. Install this one LAST.

    `LIVE` holds the current criterion so its `stats()` can be dumped after
    `train_run` returns: `train_run` does not hand the criterion back, and a
    weighting whose behaviour was never recorded cannot be interpreted later.
    """
    import bruisekit.engine as _be

    previous = getattr(_be.DistillLoss, "_original", _be.DistillLoss)
    cls = group_loss_class()
    W = np.asarray(weights, float)

    def DistillLoss(alpha=0.5, aux_weight=0.4):
        if not is_stage_o(ACTIVE_ARM):
            return previous(alpha, aux_weight)
        crit = cls(W, alpha=alpha, aux_weight=aux_weight,
                   gate_lo=gate_lo, gate_hi=gate_hi)
        install_loss_shim.LIVE = crit
        if verbose:
            print(f"  [loss] {ACTIVE_ARM}: GroupRoutedDistillLoss(alpha={alpha}, "
                  f"{W.shape[0]} groups x {W.shape[1]} teachers)")
        return crit

    DistillLoss._original = previous
    _be.DistillLoss = DistillLoss
    if verbose:
        print(f"engine.DistillLoss patched -> group-routed KD for "
              f"{len(STAGE_O_FAMILIES)} arm(s); falls through otherwise.")
    return DistillLoss


install_loss_shim.LIVE = None


def dump_loss_stats(run_dir: Path) -> dict | None:
    """Write the live criterion's `stats()` beside the run. None if nothing ran."""
    crit = install_loss_shim.LIVE
    if crit is None:
        return None
    s = crit.stats()
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    (Path(run_dir) / "group_loss_stats.json").write_text(json.dumps(s, indent=2))
    return s


# ── training ─────────────────────────────────────────────────────────────────
def record_override(env, gate: dict, reason: str, runs_dir, verbose: bool = True) -> Path:
    """Stamp a forced run with the gate it overrode, and why.

    THE POINT. `ita_group_gate` closed on both schemes. Training anyway is a
    legitimate decision -- a measured null is a stronger paper section than a
    predicted one, and the pre-registration was written to be falsifiable, not to
    be obeyed. What is NOT legitimate is a results directory that looks
    indistinguishable from a gate-approved run six months later.

    So the override is written beside the runs, carrying the failing clauses
    verbatim. Any table built from this directory can be traced back to the fact
    that the gate said no first, which is the honest way to report it: "we ran it
    anyway and here is what happened" reads very differently from "we ran it".
    """
    out = Path(runs_dir)
    out.mkdir(parents=True, exist_ok=True)
    rec = {
        "forced": True,
        "reason": reason,
        "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gate_scheme": gate.get("scheme"),
        "gate_verdict": "CLOSED" if not gate.get("GATE_any") else "OPEN",
        "failing_clauses": [k for k in ("GATE_gain_ci_clears_zero",
                                        "GATE_projection_clears_margin",
                                        "GATE_key_identifiable")
                            if not gate.get(k, True)],
        "weighting_gain_over_best_single": gate.get("weighting_gain_over_best_single"),
        "weighting_gain_ci95": gate.get("weighting_gain_ci95"),
        "projected_student_gain": gate.get("projected_student_gain"),
        "n_identifiable_groups": gate.get("n_identifiable_groups"),
        "per_image_oracle_gain": gate.get("per_image_oracle_gain"),
        "how_to_report": (
            "The gate closed BEFORE this ran. Report the result as a measured "
            "null (or a measured win) obtained against a negative pre-test, and "
            "quote the identifiability table alongside it. Do not present these "
            "runs as gate-approved."),
    }
    p = out / "FORCED_GATE.json"
    p.write_text(json.dumps(rec, indent=2))
    if verbose:
        print(f"  [override] gate was {rec['gate_verdict']}; failing clauses "
              f"{rec['failing_clauses']}")
        print(f"  [override] recorded -> {p}")
    return p


def preflight(env, reg, cfg: dict, man640: dict, weights: np.ndarray,
              family: str = "segformer_b0_itakd", seed: int = 0,
              verbose: bool = True) -> dict:
    """Run ONE training batch through the real shims and prove the wiring.

    WHY THIS EXISTS. `train_arms` is a multi-hour job whose three shims have
    never run together against `engine.train_run`. The failures they can produce
    are the expensive kind: a loader that is not tagged, a teacher stack with the
    wrong K, a batch pin that does not fit. Each surfaces on the first optimizer
    step, and each would otherwise surface after model build, cache warm and
    teacher load -- twenty minutes in, or worse, silently.

    So: build the student, install all three shims, pull ONE batch, run the
    teacher forward, compute the loss, and check the four things that matter:

      1. the loader tagged the batch          (CURRENT_GROUPS is set, right length)
      2. the teacher returned a [B, K, H, W]  stack, K == the weight matrix
      3. the loss is finite and has a grad path
      4. the group vector actually reached the loss and both groups can appear

    Costs one forward and one backward. Returns a dict; RAISES on anything that
    would make the run meaningless rather than warning about it.
    """
    global CURRENT_GROUPS

    import torch

    from . import loaders as L
    from .engine import build_model as _bm  # noqa: F401  (shim target may be patched)

    register_specs()
    W = np.asarray(weights, float)
    spec = L.spec_for(family)

    out: dict = {"family": family, "seed": seed}
    with arm(family):
        # The loader must be the tagging wrapper, and it must tag.
        import bruisekit.engine as _be
        loader = _be.make_loader(man640["train"], env.cache640, cfg["img_size"],
                                 2, True, 0, seed)
        if not isinstance(loader, _GroupTaggingLoader):
            raise RuntimeError(
                "install_group_shim did not wrap the training loader. It was "
                "either not installed, or installed outside the arm() context, "
                "or engine.make_loader was rebound afterwards by another shim. "
                "Without the wrapper the loss cannot see ITA groups and would "
                "raise on the first batch.")

        x, y, stems = next(iter(loader))
        g = CURRENT_GROUPS
        if g is None or len(g) != x.shape[0]:
            raise RuntimeError(
                f"the loader did not record group indices for its own batch "
                f"(got {None if g is None else len(g)} for B={x.shape[0]})")
        out["batch_groups"] = np.asarray(g).tolist()
        out["batch_stems"] = list(stems)

        x = x.to(env.device)
        y = y.to(env.device)

        teacher = _be.load_teacher(Path(f"x/{family}__seed{seed}"),
                                   env.paths_for_models(), env.device,
                                   bool(cfg.get("amp", True)))
        with torch.no_grad():
            tprob = teacher(x)
        out["teacher_stack"] = tuple(tprob.shape)
        out["pool"] = list(getattr(teacher, "pool", []))
        out["temperatures"] = dict(getattr(teacher, "temperature", {}))
        if tprob.shape[1] != W.shape[1]:
            raise RuntimeError(
                f"teacher stack has K={tprob.shape[1]} but the fitted weights are "
                f"for K={W.shape[1]}. The pool and the weights disagree -- refit "
                f"the weights for this pool.")

        model = _bm(spec["arch"], spec["size"], env.paths_for_models()).to(env.device)

        # Build the criterion EXACTLY as engine.train_run will -- `cfg["alpha"]`,
        # resolved from the per-arch key first. The earlier version read
        # `segformer_alpha` directly and so passed while train_run raised
        # `KeyError: 'alpha'` on the very next call. A preflight that takes a
        # shortcut around the code it is checking is not checking it.
        alpha_key = ("efficient_alpha" if STUDENT_ARCH.get(family) != "segformer"
                     else "segformer_alpha")
        if alpha_key not in cfg:
            raise KeyError(
                f"cfg has no {alpha_key!r}. engine.train_run reads cfg['alpha'] "
                f"and train_arms resolves it from this key.")
        run_cfg = {**cfg, "alpha": cfg[alpha_key]}
        out["alpha"] = run_cfg["alpha"]
        out["alpha_key"] = alpha_key
        crit = _be.DistillLoss(run_cfg["alpha"], run_cfg["aux_weight"])
        if type(crit).__name__ != "GroupRoutedDistillLoss":
            raise RuntimeError(
                f"engine.DistillLoss returned {type(crit).__name__}, not the "
                f"group-routed loss. install_loss_shim was not installed LAST -- "
                f"another stage's dispatcher is on top of it.")

        logits, aux = model.forward_train(x)
        loss = crit(logits, aux, y, tprob)
        loss.backward()
        out["loss"] = float(loss)
        if not np.isfinite(out["loss"]):
            raise RuntimeError(f"preflight loss is {out['loss']}")
        n_grad = sum(1 for p in model.parameters()
                     if p.grad is not None and torch.isfinite(p.grad).all())
        out["params_with_finite_grad"] = n_grad
        if n_grad == 0:
            raise RuntimeError("no parameter received a finite gradient")

        # The group index must CHANGE the loss, or the arm is a uniform ensemble
        # wearing this stage's name. Compare against every image forced to group 0.
        keep = CURRENT_GROUPS
        try:
            CURRENT_GROUPS = np.zeros(x.shape[0], dtype=np.int64)
            with torch.no_grad():
                flat = float(crit(logits.detach(), None if aux is None
                                  else aux.detach(), y, tprob))
        finally:
            CURRENT_GROUPS = keep
        out["loss_all_group0"] = flat
        out["group_index_changes_loss"] = bool(
            len(set(out["batch_groups"])) > 1 and abs(flat - out["loss"]) > 1e-9)

        stats = crit.stats()
        out["images_per_group_seen"] = stats["images_per_group"]

    # 5. The batch pin must resolve for every family train_arms will pin.
    #
    # THIS CHECK EXISTS BECAUSE THE PREFLIGHT MISSED IT ONCE. The first ORC run
    # passed every check above and then died in train_arms on
    # `MT.CONTROL_FOR[family]` -- a KeyError, twenty seconds in, because Stage O's
    # controls live in itakd.CONTROL_FOR and `control_batch` reads multiteacher's.
    # A preflight that exercises the loss but not the setup around it is only
    # half a preflight, so this walks the same path train_arms will.
    from . import multiteacher as MT
    pins = {}
    for fam in STAGE_O_FAMILIES:
        if STUDENT_ARCH.get(fam) != "segformer":
            continue                            # efficient arms use a fixed 16
        try:
            pins[fam] = MT.control_batch(env, reg, fam, seed, 16)
        except KeyError as exc:
            raise RuntimeError(
                f"control_batch cannot resolve {fam}'s control: {exc}. "
                f"register_specs() should have added it to "
                f"multiteacher.CONTROL_FOR -- call itakd.register_specs() "
                f"before training, or install the shims, which call it.") from exc
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"{fam}'s control has no config.json to read a batch size from: "
                f"{exc}\n  Without the pin the arm differs from its control in "
                f"batch size AND step count AND LR schedule, not just the teacher "
                f"signal, and the contrast stops being one-variable. Fix the "
                f"checkpoint path (EXTRA_RUNS) rather than skipping the pin."
            ) from exc
    out["batch_pins"] = {k: list(v) for k, v in pins.items()}

    if verbose:
        print("  PREFLIGHT")
        print(f"    loader tagged batch      groups {out['batch_groups']}")
        print(f"    teacher stack            {out['teacher_stack']}  "
              f"pool {out['pool']}")
        print(f"    temperatures             "
              f"{ {k: round(v, 4) for k, v in out['temperatures'].items()} }")
        print(f"    alpha                    {out['alpha']} "
              f"(from cfg[{out['alpha_key']!r}])")
        print(f"    loss                     {out['loss']:.6f}  "
              f"({out['params_with_finite_grad']} tensors with finite grad)")
        if len(set(out["batch_groups"])) > 1:
            print(f"    group index matters      {out['group_index_changes_loss']} "
                  f"(all-group-0 loss {out['loss_all_group0']:.6f})")
        else:
            print(f"    group index matters      not testable on this batch "
                  f"(all {out['batch_groups'][0]}); it is a 2-group scheme and "
                  f"this batch drew one group")
        for fam, (micro, accum) in out["batch_pins"].items():
            print(f"    batch pin {fam:<24} {micro} x {accum} "
                  f"= {micro * accum} effective")
    return out


def train_arms(env, reg, cfg: dict, man640: dict, runs_dir,
               families=STAGE_O_FAMILIES, seeds=(0, 1, 2),
               max_micro: int | None = 16, verbose: bool = True) -> list[str]:
    """Train each Stage O arm through `engine.train_run`, unmodified.

    Going through the shared driver is the point: the arms inherit the optimiser,
    the 6e-5/6e-4 LR split, warmup, early stopping, augmentation and the resume
    contract that produced every other number in the study. A bespoke loop would
    make `itakd - control` unreadable, because a difference could always be the
    recipe.

    The micro-batch is PINNED to each arm's control via
    `multiteacher.control_batch`, with gradient accumulation making up the
    difference, so the arm differs from its control in the teacher signal alone
    and not also in effective batch, optimizer-step count and LR schedule. See
    that function for the one residual difference (SegFormer's decode-head
    BatchNorm normalises over the micro-batch) which belongs in the limitations.

    `runs_dir` is a Stage O directory, NOT `env.runs`: the reporting stages scan
    `env.runs` by name and an experiment must not be able to inject arms into a
    table it is not part of.

    ALPHA IS SET PER ARM HERE, not taken from the caller's cfg. `engine.train_run`
    reads `cfg["alpha"]`, and the notebooks' CFG carries `segformer_alpha` and
    `efficient_alpha` separately -- the unified notebook resolves between them at
    the call site and this does the same, keyed on `STUDENT_ARCH`. Relying on the
    caller to have done it is how the shell runner and the notebook diverge, and
    it is why the first ORC run raised `KeyError: 'alpha'` after the pool had
    already loaded and the batch pin had already resolved.
    """
    from . import loaders as L
    from . import multiteacher as MT
    from .engine import train_run

    if not str(env.device).startswith("cuda"):
        raise RuntimeError(
            f"train_arms would train {len(tuple(families)) * len(tuple(seeds))} "
            f"run(s) with a three-teacher pool but device is {env.device}. This "
            f"needs a GPU session.")

    # `engine.build_model` knows segformer / smp / yolo and nothing else, so a
    # non-SegFormer student dies with `ValueError: unknown arch` -- which is what
    # the first ORC run hit, after the SegFormer arm had already trained for an
    # hour. Installed here rather than left to the caller for the same reason
    # alpha is resolved here: a shell run and a notebook run must not be able to
    # disagree about what got installed.
    if any(STUDENT_ARCH[f] != "segformer" for f in families):
        from . import efficient_models as EM
        EM.install_efficient_shim(env, verbose=verbose)

    register_specs()
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    pinned = {}
    for family in families:
        if STUDENT_ARCH[family] == "segformer":
            pinned[family] = MT.control_batch(env, reg, family, 0, max_micro)
    if pinned:
        _install_batch_shim(pinned, verbose=verbose)

    done = []
    for family in families:
        spec = L.spec_for(family)
        # The unified notebook's rule, applied here so a shell run and a notebook
        # run cannot disagree about the KD mix.
        alpha_key = ("efficient_alpha" if STUDENT_ARCH[family] != "segformer"
                     else "segformer_alpha")
        if alpha_key not in cfg:
            raise KeyError(
                f"cfg has no {alpha_key!r}; engine.train_run needs cfg['alpha'] "
                f"and this is where it comes from. Use the CFG block from "
                f"bruise_stage_o_train.ipynb §3.")
        run_cfg = {**cfg, "alpha": cfg[alpha_key]}

        for seed in seeds:
            run_id = f"{family}__seed{seed}"
            if verbose:
                print(f"\n{'=' * 74}\n{run_id}   control={CONTROL_FOR[family]}   "
                      f"alpha={run_cfg['alpha']} ({alpha_key})\n{'=' * 74}")
            with arm(family):
                res = train_run(run_id, spec, seed, run_cfg, env.paths_for_models(),
                                man640, env.cache640, runs_dir, env.device)
                stats = dump_loss_stats(runs_dir / run_id)
            if verbose:
                print(f"  -> {res.get('status', 'trained')}")
                if stats:
                    print(f"     images per group: {stats['images_per_group']}, "
                          f"mean coverage {stats['mean_coverage']:.3f}")
            done.append(run_id)
    return done


def _install_batch_shim(pinned: dict, verbose: bool = True):
    """Pin each Stage O arm's micro-batch to its control's. See
    `multiteacher.install_batch_shim` for the confound this prevents; this is that
    function keyed on Stage O's global instead of Stage M's."""
    import bruisekit.engine as _be

    previous = getattr(_be.resolve_micro_batch, "_original", _be.resolve_micro_batch)

    def resolve_micro_batch(model, cfg, device, teacher=None):
        if is_stage_o(ACTIVE_ARM) and ACTIVE_ARM in pinned:
            micro, accum = pinned[ACTIVE_ARM]
            if verbose:
                print(f"  [batch] {ACTIVE_ARM}: pinned to {micro} x {accum} "
                      f"(its control's, not a fresh probe)")
            return int(micro), int(accum)
        return previous(model, cfg, device, teacher)

    resolve_micro_batch._original = previous
    _be.resolve_micro_batch = resolve_micro_batch
    if verbose:
        print(f"engine.resolve_micro_batch patched -> pinned for {sorted(pinned)}.")
    return resolve_micro_batch


# ═════════════════════════════════════════════════════════════════════════════
# results io
# ═════════════════════════════════════════════════════════════════════════════
def results_dir(env, sub: str | None = None) -> Path:
    """`<bundle>/STAGE_O_RESULTS[/sub]`, created. Never `results/`,
    `FINAL_RESULT/`, `LESION_SIZE_RESULTS/` or `_work/runs/`."""
    d = Path(env.root) / RESULTS_DIRNAME
    if sub:
        d = d / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


def save(env, name: str, obj, subdir: str = "tables") -> Path:
    """Write a DataFrame as CSV or anything else as JSON. Returns the path."""
    d = results_dir(env, subdir)
    if isinstance(obj, pd.DataFrame):
        p = d / f"{name}.csv"
        obj.to_csv(p, index=False)
    else:
        p = d / f"{name}.json"
        p.write_text(json.dumps(obj, indent=2, default=str))
    return p


def save_gate(env, res: dict, subdir: str = "tables") -> list[Path]:
    """The gate as JSON plus its two tables as CSV, so the verdict is on disk in a
    form a spreadsheet can open as well as one a script can read."""
    out = [save(env, f"ita_group_gate__{res['scheme']}", res, subdir)]
    out.append(save(env, f"ita_group_weights__{res['scheme']}",
                    pd.DataFrame(res["weights"]), subdir))
    out.append(save(env, f"ita_group_identifiability__{res['scheme']}",
                    pd.DataFrame(res["identifiability"]), subdir))
    (results_dir(env, subdir) / f"ita_group_gate__{res['scheme']}.txt").write_text(
        format_gate(res), encoding="utf-8")
    return out


# ═════════════════════════════════════════════════════════════════════════════
# self test -- no weights, no GPU, no network
# ═════════════════════════════════════════════════════════════════════════════
def self_test(verbose: bool = True) -> bool:
    """Structural checks. Everything here runs on a laptop in under a second, so
    there is no excuse for shipping without it."""
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        if verbose:
            print(f"  [{'ok' if cond else 'FAIL'}] {name}"
                  + (f"  {detail}" if detail else ""))

    # 1. the miss identity, and the guard that catches an inconsistent table
    df = pd.DataFrame({
        "stem": list("abcdef"),
        "subject": ["s1", "s1", "s2", "s2", "s3", "s3"],
        "dice": [0.0, 0.0, 0.0, 0.8, 0.5, 0.9],
        "pred_positive_pixels": [0, 10, 0, 100, 50, 80],
        "recall": [0.0, 0.0, 0.0, 0.8, 0.5, 0.9],
        "precision": [0.0, 0.0, 0.0, 0.8, 0.5, 0.9],
        "skin_tone_category": ["Light (II-III)"] * 3 + ["Dark (VI)"] * 3,
    })
    c = _miss_counts(df)
    check("zero_dice = empty + wrong_place",
          c["zero_dice_n"] == c["empty_pred_n"] + c["wrong_place_n"],
          f"{c['zero_dice_n']} = {c['empty_pred_n']} + {c['wrong_place_n']}")
    check("wrong_place counts a non-empty zero-Dice prediction",
          c["wrong_place_n"] == 1, f"got {c['wrong_place_n']}")

    # Two images scoring above zero while predicting nothing: 4 empty against 3
    # zero-Dice, which is arithmetically impossible and must not be reported.
    bad = df.copy()
    bad.loc[[3, 4], "pred_positive_pixels"] = 0
    try:
        _miss_counts(bad)
        check("_miss_counts raises on empty > zero", False)
    except ValueError:
        check("_miss_counts raises on empty > zero", True)

    tax = miss_taxonomy({"m": df})
    check("miss_taxonomy returns one row per model", len(tax) == 1)
    by = miss_taxonomy_by({"m": df}, "skin_tone_category")
    check("miss_taxonomy_by splits both groups", len(by) == 2,
          f"strata {sorted(by.stratum)}")

    # 2. the group scheme
    check("light_vs_rest maps Very Light into Light",
          collapse(["Very Light (I-II)"], "light_vs_rest")[0] == "Light")
    check("light_vs_rest has exactly 2 groups",
          len(group_order("light_vs_rest")) == 2, str(group_order("light_vs_rest")))
    try:
        collapse(["Neon (XI)"], "light_vs_rest")
        check("collapse raises on an unmapped category", False)
    except KeyError:
        check("collapse raises on an unmapped category", True)
    gi = group_index(["Light (II-III)", "Dark (VI)"], "light_vs_rest")
    check("group_index returns positions in group_order", gi.tolist() == [0, 1],
          str(gi.tolist()))

    # 3. weights
    mat = pd.DataFrame({
        "stem": [f"i{k}" for k in range(8)],
        "subject": ["s1", "s1", "s2", "s2", "s3", "s3", "s4", "s4"],
        "skin_tone_category": ["Light (II-III)"] * 4 + ["Dark (VI)"] * 4,
        "A": [0.90, 0.88, 0.86, 0.92, 0.40, 0.42, 0.38, 0.44],
        "B": [0.40, 0.44, 0.42, 0.38, 0.90, 0.86, 0.92, 0.88],
    })
    w = group_weights(mat, ("A", "B"), "light_vs_rest", beta=BETA)
    check("group_weights rows sum to 1 per group",
          np.allclose(w.groupby("group").weight.sum().to_numpy(), 1.0))
    W = weight_array(w, ("A", "B"), "light_vs_rest")
    check("weight_array is [n_groups, K]", W.shape == (2, 2), str(W.shape))
    check("Light prefers A, Dark prefers B",
          W[0, 0] > W[0, 1] and W[1, 1] > W[1, 0],
          f"Light {W[0].round(3).tolist()}, Dark {W[1].round(3).tolist()}")

    try:
        group_weights(mat, ("A", "B"), "five")
        check("group_weights raises when a scheme group has no val images", False)
    except ValueError:
        check("group_weights raises when a scheme group has no val images", True)

    # 4. the gate, and the identifiability clause
    res = ita_group_gate(mat, ("A", "B"), "light_vs_rest", reps=200, seed=0)
    check("gate reports a weighting gain",
          "weighting_gain_over_best_single" in res,
          f"{res['weighting_gain_over_best_single']:+.4f}")
    check("group weighting beats the best single teacher on this construction",
          res["weighting_gain_over_best_single"] > 0)
    check("per-image oracle bounds the group weighting",
          res["per_image_oracle_val_dice"] >= res["group_weighted_val_dice"] - 1e-9)
    check("gate carries the identifiability clause",
          "GATE_key_identifiable" in res and len(res["identifiability"]) == 2)
    check("format_gate renders", "VERDICT" in format_gate(res))

    # An indistinguishable pool must NOT be called identifiable.
    flat = mat.copy()
    rng = np.random.default_rng(0)
    flat["A"] = 0.75 + rng.normal(0, 0.01, len(flat))
    flat["B"] = 0.75 + rng.normal(0, 0.01, len(flat))
    fr = ita_group_gate(flat, ("A", "B"), "light_vs_rest", reps=400, seed=0)
    check("a flat pool fails the identifiability clause OR the margin clause",
          not fr["GATE_run_method"],
          f"identifiable={fr['n_identifiable_groups']}, "
          f"projected={fr['projected_student_gain']:+.5f}")

    # 5. the loss -- only if torch is installed
    try:
        import torch
    except ImportError:
        if verbose:
            print("  [skip] loss checks (torch not installed)")
        return ok

    global CURRENT_GROUPS
    cls = group_loss_class()
    from .reliability_kd import gated_loss_class

    torch.manual_seed(0)
    B, H = 4, 16
    y = (torch.rand(B, 1, H, H) > 0.5).float()
    t1 = torch.rand(B, 1, H, H)
    logits = torch.randn(B, 1, H, H)

    # K = 1 must equal the Stage H gated loss exactly.
    CURRENT_GROUPS = np.zeros(B, dtype=np.int64)
    lo = cls(np.ones((1, 1)), alpha=0.6, aux_weight=0.0)
    ref = gated_loss_class()(alpha=0.6, aux_weight=0.0)
    a, b = float(lo(logits, None, y, t1)), float(ref(logits, None, y, t1))
    check("K=1 reduces to the Stage H gated loss", abs(a - b) < 1e-5,
          f"{a:.6f} vs {b:.6f}")

    # A uniform weight matrix must equal the uniform ensemble.
    t2 = torch.cat([t1, torch.rand(B, 1, H, H)], dim=1)
    uni = cls(np.full((1, 2), 0.5), alpha=0.6, aux_weight=0.0)
    a = float(uni(logits, None, y, t2))
    b = float(ref(logits, None, y, t2.mean(dim=1, keepdim=True)))
    check("uniform weights reproduce p2_ensemble_uniform + gate",
          abs(a - b) < 1e-5, f"{a:.6f} vs {b:.6f}")

    # Different groups must actually take different weight rows.
    W2 = np.array([[1.0, 0.0], [0.0, 1.0]])
    two = cls(W2, alpha=0.6, aux_weight=0.0)
    CURRENT_GROUPS = np.array([0, 0, 1, 1], dtype=np.int64)
    mixed = float(two(logits, None, y, t2))
    CURRENT_GROUPS = np.zeros(B, dtype=np.int64)
    only0 = float(two(logits, None, y, t2))
    check("the group index changes the loss", abs(mixed - only0) > 1e-6,
          f"mixed {mixed:.6f} vs all-group-0 {only0:.6f}")

    # The guard: a missing or stale group vector must RAISE, never fall back.
    CURRENT_GROUPS = None
    try:
        two(logits, None, y, t2)
        check("the loss raises when no group vector was recorded", False)
    except RuntimeError:
        check("the loss raises when no group vector was recorded", True)
    CURRENT_GROUPS = np.zeros(B + 1, dtype=np.int64)
    try:
        two(logits, None, y, t2)
        check("the loss raises on a batch/group length mismatch", False)
    except RuntimeError:
        check("the loss raises on a batch/group length mismatch", True)
    CURRENT_GROUPS = None

    # A pool that does not match the fitted weights must raise, not truncate.
    CURRENT_GROUPS = np.zeros(B, dtype=np.int64)
    try:
        cls(np.full((1, 3), 1 / 3))(logits, None, y, t2)
        check("the loss raises when K disagrees with the weight matrix", False)
    except ValueError:
        check("the loss raises when K disagrees with the weight matrix", True)
    CURRENT_GROUPS = None

    # The loader wrapper must set the global, and restore it when exhausted.
    class _FakeLoader:
        def __iter__(self):
            yield (torch.zeros(2, 3, 4, 4), torch.zeros(2, 1, 4, 4), ["a", "b"])

        def __len__(self):
            return 1

    seen = []
    for _x, _y, _s in _GroupTaggingLoader(_FakeLoader(), {"a": 1, "b": 0}):
        seen.append(None if CURRENT_GROUPS is None else CURRENT_GROUPS.tolist())
    check("_GroupTaggingLoader records the batch's groups", seen == [[1, 0]],
          str(seen))
    check("_GroupTaggingLoader clears the global when exhausted",
          CURRENT_GROUPS is None)
    try:
        for _ in _GroupTaggingLoader(_FakeLoader(), {"a": 0}):
            pass
        check("_GroupTaggingLoader raises on an unmapped stem", False)
    except KeyError:
        check("_GroupTaggingLoader raises on an unmapped stem", True)
    CURRENT_GROUPS = None

    # The arm() context must not leak, on the happy path or on an exception.
    with arm("segformer_b0_itakd"):
        inside = ACTIVE_ARM
    check("arm() sets and restores ACTIVE_ARM",
          inside == "segformer_b0_itakd" and ACTIVE_ARM is None)
    try:
        with arm("segformer_b0_itakd"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    check("arm() restores ACTIVE_ARM after an exception", ACTIVE_ARM is None)

    return ok


if __name__ == "__main__":
    raise SystemExit(0 if self_test() else 1)
