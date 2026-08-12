"""Fenwick labeler study -- which annotator's masks train the best model?

THE QUESTION
------------
Four people annotated the Fenwick corpus. Three of them (`hliu36`, `mzehra2`,
`nmousta5`) also annotated the shared 128-image white-light test set, so those
three are the only ones whose models can be compared on common ground. This
module trains one model per labeler under 5-fold subject-grouped cross-
validation and asks which labeler's annotations are the most LEARNABLE.

WHY A MATCHED CORE, AND NOT EACH LABELER'S WHOLE POOL
------------------------------------------------------
The raw pools are 1356 / 2268 / 611 images. A model trained on 2268 images
beating one trained on 611 tells you nothing about annotation quality -- it is
the oldest confound there is. So the arms are matched on BOTH nuisance axes
before a single step of training:

    subjects   the 48 subjects all three labelled (after test subjects are
               removed, see below), identical across arms
    volume     per subject s, every arm gets exactly n_s = min over labelers of
               how many images of s that labeler drew. 360 images per arm, with
               an identical per-subject histogram.

What remains different between the three arms is WHO DREW THE MASK, and nothing
else. That is what makes a Dice gap here attributable.

Within a subject the images are chosen deterministically -- three-way-shared
images first, then by stem -- so the arms overlap as much as the data allows and
the selection re-derives identically on any machine. No RNG is involved in
picking the core; `SEED` only ever reaches the training loop.

THE LEAKAGE THE ON-DISK CHECK DOES NOT CATCH
---------------------------------------------
`tables/verify_on_disk.csv` confirms no test IMAGE is in any training pool. But
12 to 13 of the 15 test SUBJECTS are, and images of one bruise on one subject
are correlated -- the whole reason this study clusters by subject everywhere
else (`report.py`, and every bootstrap in Stages D/G/M/N). Training on subject
182 and testing on a different photograph of subject 182 measures memorisation.
`build_core` therefore drops every test subject from every pool first. It costs
about 17 percent of the images and it is not optional.

APPLES TO APPLES WITH NIJ
--------------------------
Same architecture (`segformer_b0_direct`, the study's 3.71 M benchmark), same
recipe (`kd_core.DEFAULTS` verbatim), same 640 px cache built by the same
`loaders.build_cache640`, same batch policy (`engine.resolve_micro_batch` in
`matched` mode, so the probe picks the largest micro-batch that fits and
accumulation restores an effective batch of 8 on any GPU), same operating-point
rule (`foundation.fit_operating_point`, one standard error, ties broken on
misses), same reporting (`report.normalize`, so `complete_miss` is `dice == 0`).

Training goes through `engine.train_run` UNMODIFIED. Nothing in this module
reimplements a training loop, because then "is this gap the labeler or the
recipe?" would be unanswerable.

WHAT THE ABSOLUTE NUMBERS ARE NOT
----------------------------------
Each fold trains on 288 images against NIJ's 697. These Dice values are NOT
comparable to the 0.7663 headline and must never be quoted beside it. Only the
BETWEEN-LABELER differences measured here are meaningful.

WHY THIS MODULE IS NOT IN THE UNIFIED BUNDLE
---------------------------------------------
Same policy as foundation.py, dermprobe.py, finetune_n3.py and lesionsize.py:
authored in bruisekit/ but kept out of 60_build's copy list, shipped as a
standalone notebook plus overlay. Nothing in the reporting pipeline imports it.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# ── pre-registered constants ─────────────────────────────────────────────────

#: The three labelers who annotated the shared test set. `eporti5` annotated a
#: training pool but NOT the test set, so no model trained on eporti5 can be
#: scored on common ground and the arm would be uninterpretable. Excluded by the
#: data, not by preference.
TOP3: tuple[str, ...] = ("hliu36", "mzehra2", "nmousta5")

N_FOLDS = 5
SEED = 0                       # single seed, as scoped
FAMILY = "segformer_b0_direct"  # the NIJ benchmark arm; keeps this comparable

#: Images per arm are decided by the data (element-wise minimum per subject), not
#: fixed here. Recorded after the fact in `matching_report.csv`.
IMG_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def subject_of(stem: str) -> str:
    """Subject id is the leading field of the filename: `182_Q_2-6_WL_...` -> `182`.

    Asserted rather than assumed: a stem that does not start with digits would
    silently become its own subject, which would put the same person in train and
    val and quietly inflate every number in the study.
    """
    head = stem.split("_", 1)[0]
    if not head.isdigit():
        raise ValueError(
            f"cannot read a subject id from stem {stem!r}. Fold grouping and every "
            f"bootstrap in this module are BY SUBJECT; guessing here would put one "
            f"person on both sides of a split.")
    return head


# ── the environment ──────────────────────────────────────────────────────────

def make_env(bundle_root, fenwick_root, work):
    """An `Env` whose `data` is the Fenwick dataset but whose `weights` is the bundle.

    Two roots are genuinely needed: the pretrained SegFormer config/weights live
    in the bundle (`pretrained_weights/segformer_mit_b0`), while every image and
    mask lives in the Fenwick tree. Subclassing rather than copying files keeps
    one copy of each.
    """
    from bruisekit.paths import Env

    class FenwickEnv(Env):
        @property
        def data(self) -> Path:
            return Path(fenwick_root)

    return FenwickEnv(root=Path(bundle_root), work=Path(work),
                      device=_pick_device())


def _pick_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── the matched core ─────────────────────────────────────────────────────────

@dataclass
class Core:
    """The matched training core plus everything needed to audit how it was cut."""
    frame: pd.DataFrame          # stem, subject, labeler, fold, image_path, mask_path
    test: pd.DataFrame           # stem, subject, labeler, image_path, mask_path
    report: pd.DataFrame         # per-labeler counts at each filtering step
    dropped_subjects: list       # test subjects removed from the pools
    shared_subjects: list        # the 48 that survived
    n_per_arm: int
    test_images_dropped: tuple = ()   # test images lacking a mask from some labeler


def _find_image(d: Path, stem: str) -> Path:
    for ext in IMG_EXTS:
        p = d / f"{stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"no image for {stem!r} in {d} (tried {IMG_EXTS})")


def _stems(d: Path) -> set:
    return {p.stem for p in d.iterdir() if p.is_file()}


def build_core(fenwick_root, labelers=TOP3, n_folds: int = N_FOLDS) -> Core:
    """Cut the subject-matched, volume-matched training core and assign folds.

    Steps, in this order, each recorded in `report`:

      1. read each labeler's pool and the shared test set
      2. DROP every test subject from every pool          (the leakage fix)
      3. keep only subjects that ALL labelers still have  (subject matching)
      4. per subject take n_s = min over labelers         (volume matching)
      5. assign the subjects to `n_folds` groups, greedily balanced on image count

    Folds are assigned to SUBJECTS, once, and shared by all arms. Fold 3 is
    therefore the same 9-or-so people for every labeler, so a fold-to-fold
    comparison across arms is paired.
    """
    root = Path(fenwick_root)
    test_img_dir = root / "test_set" / "images"
    on_disk = _stems(test_img_dir)

    # The cross-labeler matrix compares COLUMNS, so every column must be scored on
    # the same images. One test image is missing a mask from one labeler; keeping
    # it would make that column a 127-image mean against two 128-image means, and
    # the difference would be read as annotation quality. Intersect instead, and
    # say which images went.
    test_stems = {s for s in on_disk
                  if all((root / "test_set" / "masks" / l / f"{s}.png").exists()
                         for l in labelers)}
    incomplete = sorted(on_disk - test_stems)

    # Subjects come from EVERY test image on disk, not just the complete ones: an
    # image dropped for a missing mask is still a photograph of that subject, and
    # leaving it in the training pools would leak.
    test_subjects = {subject_of(s) for s in on_disk}

    pools, rows = {}, []
    for lab in labelers:
        idir, mdir = root / "by_labeler" / lab / "images", root / "by_labeler" / lab / "masks"
        if not idir.is_dir() or not mdir.is_dir():
            raise FileNotFoundError(f"{lab}: expected {idir} and {mdir}")
        have = _stems(idir) & _stems(mdir)
        clean = {s for s in have if subject_of(s) not in test_subjects}
        pools[lab] = clean
        rows.append({"labeler": lab, "pool_images": len(have),
                     "pool_subjects": len({subject_of(s) for s in have}),
                     "after_test_subjects_dropped": len(clean)})

    shared = sorted(set.intersection(*[{subject_of(s) for s in pools[l]} for l in labelers]),
                    key=lambda s: int(s))
    if not shared:
        raise RuntimeError("no subject is shared by all labelers after the test "
                           "subjects are removed; the matched design is impossible")

    by_subject = {lab: {} for lab in labelers}
    for lab in labelers:
        for s in pools[lab]:
            by_subject[lab].setdefault(subject_of(s), []).append(s)

    # Prefer images all three labelled, so the arms overlap as far as the data
    # allows. Deterministic and RNG-free: `SEED` never touches selection.
    three_way = set.intersection(*[pools[l] for l in labelers])

    picked, per_subject_n = {lab: [] for lab in labelers}, {}
    for subj in shared:
        n = min(len(by_subject[lab][subj]) for lab in labelers)
        per_subject_n[subj] = n
        for lab in labelers:
            cand = sorted(by_subject[lab][subj], key=lambda s: (s not in three_way, s))
            picked[lab].extend(cand[:n])

    fold_of = _assign_folds(per_subject_n, n_folds)

    # POSIX separators, always. The core is cut on whatever machine is convenient
    # and consumed on the Linux GPU box, and a backslash written here is a
    # FileNotFoundError there -- after the cache build, halfway into the job.
    frame = pd.DataFrame([
        {"stem": s, "subject": subject_of(s), "labeler": lab,
         "fold": fold_of[subject_of(s)],
         "image_path": (Path("by_labeler") / lab / "images"
                        / _find_image(root / "by_labeler" / lab / "images", s).name).as_posix(),
         "mask_path": (Path("by_labeler") / lab / "masks" / f"{s}.png").as_posix()}
        for lab in labelers for s in sorted(picked[lab])])

    test = pd.DataFrame([
        {"stem": s, "subject": subject_of(s), "labeler": lab,
         "image_path": (Path("test_set") / "images"
                        / _find_image(test_img_dir, s).name).as_posix(),
         "mask_path": (Path("test_set") / "masks" / lab / f"{s}.png").as_posix()}
        for lab in labelers for s in sorted(test_stems)])

    n_per_arm = sum(per_subject_n.values())
    rep = pd.DataFrame(rows)
    rep["shared_subjects"] = len(shared)
    rep["core_images"] = [int((frame.labeler == l).sum()) for l in labelers]
    rep["core_shared_with_all"] = [
        int(frame[frame.labeler == l].stem.isin(three_way).sum()) for l in labelers]
    rep["test_masks"] = [int((test.labeler == l).sum()) for l in labelers]

    # The design's whole claim rests on these being equal. Check, do not trust.
    if rep.core_images.nunique() != 1:
        raise RuntimeError(f"arms are not volume-matched: {rep.core_images.tolist()}")
    if rep.test_masks.nunique() != 1:
        raise RuntimeError(f"test columns are not image-matched: {rep.test_masks.tolist()}")
    for lab in labelers:
        h = Counter(frame[frame.labeler == lab].subject)
        if h != Counter({k: v for k, v in per_subject_n.items()}):
            raise RuntimeError(f"{lab}: per-subject histogram does not match the target")

    return Core(frame=frame, test=test, report=rep,
                dropped_subjects=sorted(test_subjects, key=lambda s: int(s)),
                shared_subjects=shared, n_per_arm=n_per_arm,
                test_images_dropped=incomplete)


def _assign_folds(per_subject_n: dict, n_folds: int) -> dict:
    """Greedy largest-first bin packing of SUBJECTS into folds, balanced on images.

    Grouped, not stratified: with 48 subjects and a 1-to-38 image spread, a random
    subject split leaves folds differing several-fold in size, and a fold's Dice
    then partly reports how many images it happened to get. Largest-first greedy
    is deterministic and lands within a few images of perfect balance here.
    """
    order = sorted(per_subject_n, key=lambda s: (-per_subject_n[s], int(s)))
    load = [0] * n_folds
    out = {}
    for subj in order:
        k = int(np.argmin(load))
        out[subj] = k
        load[k] += per_subject_n[subj]
    return out


def write_core(core: Core, out_dir) -> Path:
    """Persist the core so training, scoring and the write-up read ONE cut of it."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    core.frame.to_csv(out / "cv_core.csv", index=False)
    core.test.to_csv(out / "test_manifest.csv", index=False)
    core.report.to_csv(out / "matching_report.csv", index=False)
    (out / "core_design.json").write_text(json.dumps({
        "labelers": list(core.report.labeler),
        "n_folds": int(core.frame.fold.nunique()),
        "images_per_arm": int(core.n_per_arm),
        "shared_subjects": core.shared_subjects,
        "test_subjects_dropped_from_pools": core.dropped_subjects,
        "test_images_dropped_incomplete_masks": list(core.test_images_dropped),
        "fold_images": core.frame[core.frame.labeler == core.report.labeler[0]]
                       .fold.value_counts().sort_index().to_dict(),
        "family": FAMILY, "seed": SEED,
    }, indent=2))
    return out / "cv_core.csv"


def load_core(out_dir) -> Core:
    out = Path(out_dir)
    d = json.loads((out / "core_design.json").read_text())
    return Core(frame=pd.read_csv(out / "cv_core.csv", dtype={"subject": str}),
                test=pd.read_csv(out / "test_manifest.csv", dtype={"subject": str}),
                report=pd.read_csv(out / "matching_report.csv"),
                dropped_subjects=d["test_subjects_dropped_from_pools"],
                shared_subjects=d["shared_subjects"],
                n_per_arm=d["images_per_arm"],
                test_images_dropped=tuple(d.get("test_images_dropped_incomplete_masks", ())))


# ── the 640 cache ────────────────────────────────────────────────────────────

def build_cache(env, core: Core, force: bool = False) -> dict:
    """Materialise the 640 px cache through the SAME builder the NIJ study uses.

    One split per (labeler, role). The test IMAGES are identical across the three
    `test_<labeler>` splits and only the masks differ -- three copies of 128 PNGs
    is a few megabytes, and the alternative is a bespoke loader that shares images
    across splits, i.e. a second data path that could drift from the study's.
    """
    from bruisekit import loaders as L

    manifests = {}
    for lab in core.report.labeler:
        manifests[f"core_{lab}"] = core.frame[core.frame.labeler == lab].reset_index(drop=True)
        manifests[f"test_{lab}"] = core.test[core.test.labeler == lab].reset_index(drop=True)
    cached = L.build_cache640(env, manifests, force=force)

    # build_cache640 rewrites the paths but carries the other columns through, so
    # fold and subject survive. Re-attach defensively rather than assume.
    for lab in core.report.labeler:
        src = core.frame[core.frame.labeler == lab].reset_index(drop=True)
        cached[f"core_{lab}"]["fold"] = src["fold"].to_numpy()
        cached[f"core_{lab}"]["subject"] = src["subject"].to_numpy()
        cached[f"test_{lab}"]["subject"] = \
            core.test[core.test.labeler == lab].reset_index(drop=True)["subject"].to_numpy()
    return cached


def fold_manifests(cached: dict, labeler: str, fold: int) -> dict:
    """`{"train","val"}` for one (labeler, fold). Val is the fold's SUBJECTS."""
    df = cached[f"core_{labeler}"]
    val = df[df.fold == fold].reset_index(drop=True)
    train = df[df.fold != fold].reset_index(drop=True)
    overlap = set(val.subject) & set(train.subject)
    if overlap:
        raise RuntimeError(f"{labeler} fold {fold}: subjects on both sides: {sorted(overlap)}")
    if val.empty or train.empty:
        raise RuntimeError(f"{labeler} fold {fold}: empty side")
    return {"train": train, "val": val}


# ── training ─────────────────────────────────────────────────────────────────

def default_cfg(**over) -> dict:
    """The NIJ recipe: `kd_core.DEFAULTS` plus the keys `engine.train_run` reads.

    `DEFAULTS` alone is incomplete -- it has no `aux_weight`, and train_run builds
    `SupervisedLoss(cfg["aux_weight"])` unconditionally. 0.4 is the study's value
    for SegFormer (the unified notebook's CFG, and the recipe the shipped
    checkpoints were trained with); the 0.0 in the baselines notebook is for
    U-Net/DeepLab, which have no auxiliary head.

    BATCH POLICY IS THE ONE DELIBERATE DEPARTURE, and it is a departure toward
    the study's own principle rather than away from it. The unified notebook runs
    `per_model`, which lets each model take the largest batch its own size allows
    -- defensible there, where the point is throughput per architecture. Here the
    arms are split across TWO GPUs, and `per_model` on two different cards would
    give two arms different micro-batches, different step counts and a different
    LR schedule. The labeler contrast would then be confounded by which GPU the
    arm happened to land on. `matched` pins the effective batch at 8 on every
    card and lets accumulation absorb the difference, which is exactly what
    handbook 3's fixed recipe is for.
    """
    from bruisekit.kd.kd_core import DEFAULTS

    cfg = dict(DEFAULTS)
    cfg.update(
        aux_weight=0.4,          # SegFormer's auxiliary head; see above
        alpha=0.6,               # KD mix. Unused (distill=False) but train_run reads
                                 # cfg["alpha"] before it checks, so it must exist.
        drive_sync_every=2,      # resume.pt cadence, same as the unified notebook
        batch_mode="matched",    # see above -- NOT per_model, because two GPUs
        vram_target=0.75,
        eval_batch=8,
        eval_batch_cpu=2,
    )
    cfg.update(over)
    return cfg


def train_labeler(env, cfg: dict, cached: dict, labeler: str, runs_dir,
                  folds=range(N_FOLDS), seed: int = SEED, verbose: bool = True) -> list:
    """Train all folds for one labeler through `engine.train_run`, unmodified.

    Resumable exactly like every other stage: a fold with `DONE.json` is skipped,
    an interrupted fold restarts from `resume.pt`.
    """
    import torch
    from bruisekit import loaders as L
    from bruisekit.engine import train_run

    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    spec = L.spec_for(FAMILY)
    done = []
    for fold in folds:
        man = fold_manifests(cached, labeler, fold)
        run_id = f"{labeler}__fold{fold}"
        if verbose:
            print(f"\n{'=' * 70}\n{run_id}   train {len(man['train'])} / val {len(man['val'])}"
                  f"   ({man['val'].subject.nunique()} val subjects)\n{'=' * 70}")
        res = train_run(run_id, spec, seed, cfg, env.paths_for_models(),
                        man, env.cache640, runs_dir, env.device)
        if verbose:
            print(f"  -> {res.get('status', 'trained')}")
        done.append(run_id)
        if str(env.device).startswith("cuda"):
            torch.cuda.empty_cache()
    return done


def probe_batch(env, cfg: dict) -> dict:
    """Report what the batch finder chooses here, before committing hours to it.

    Same call `train_run` makes, so this is the actual number, not an estimate.
    """
    from bruisekit import loaders as L
    from bruisekit.engine import resolve_micro_batch
    from bruisekit.models import build_model

    spec = L.spec_for(FAMILY)
    model = build_model(spec["arch"], spec["size"], env.paths_for_models()).to(env.device)
    micro, accum = resolve_micro_batch(model, cfg, env.device, None)
    del model
    import torch
    if str(env.device).startswith("cuda"):
        torch.cuda.empty_cache()
    return {"micro_batch": micro, "accum": accum, "effective": micro * accum,
            "mode": cfg.get("batch_mode", "matched"), "img_size": cfg["img_size"]}


# ── scoring: the labeler x labeler matrix ────────────────────────────────────

def score_labeler(env, cfg: dict, cached: dict, labeler: str, runs_dir, tables_dir,
                  eval_labelers=TOP3, folds=range(N_FOLDS), seed: int = SEED,
                  verbose: bool = True) -> pd.DataFrame:
    """Score every fold model of `labeler` on the shared test set, against EACH
    labeler's masks.

    The cut is fitted on that fold's OWN validation split with the study's
    one-standard-error rule and then applied unchanged to all three mask sets, so
    a model is never tuned against the annotator it is being judged by.

    Returns the long per-image table; also writes it per (fold, mask-set).
    """
    import torch
    from bruisekit.foundation import fit_operating_point, score_split
    from bruisekit.models import build_model
    from bruisekit import loaders as L

    runs_dir, tables_dir = Path(runs_dir), Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    spec = L.spec_for(FAMILY)
    out = []

    for fold in folds:
        run_id = f"{labeler}__fold{fold}"
        run_dir = runs_dir / run_id
        ckpt = run_dir / "best.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"{run_id}: no best.pt -- train this fold first")

        model = build_model(spec["arch"], spec["size"], env.paths_for_models()).to(env.device)
        state = torch.load(str(ckpt), map_location=env.device, weights_only=False)
        model.load_state_dict(state["model"] if "model" in state else state)
        model.eval()

        man = fold_manifests(cached, labeler, fold)
        op = fit_operating_point(model, env, cfg, man, run_dir)
        cut = float(op["cut"] if isinstance(op, dict) else op)

        # The held-out fold, at that cut. Optimistic (the cut was fitted here) but
        # identically so for every arm, and it is the only number that reads the
        # labeler's own annotation style on unseen subjects.
        val_tbl = score_split(model, env, cfg, {"val": man["val"]}, "val", cut,
                              meta=man["val"][["stem", "subject"]])
        val_tbl.insert(0, "eval_masks", labeler)
        val_tbl.insert(0, "split", "cv_val")

        rows = [val_tbl]
        for evl in eval_labelers:
            t = score_split(model, env, cfg, cached, f"test_{evl}", cut,
                            meta=cached[f"test_{evl}"][["stem", "subject"]])
            t.insert(0, "eval_masks", evl)
            t.insert(0, "split", "test")
            rows.append(t)

        tbl = pd.concat(rows, ignore_index=True)
        tbl.insert(0, "fold", fold)
        tbl.insert(0, "train_masks", labeler)
        tbl["cut"] = cut
        tbl.to_csv(tables_dir / f"per_image__{run_id}.csv", index=False)
        out.append(tbl)

        if verbose:
            for evl in eval_labelers:
                m = tbl[(tbl.split == "test") & (tbl.eval_masks == evl)]
                print(f"  {run_id}  cut={cut:+.3f}  test vs {evl:<9} "
                      f"dice={m.dice.mean():.4f}  misses={int(m.complete_miss.sum())}/{len(m)}")
            v = tbl[tbl.split == "cv_val"]
            print(f"  {run_id}  cv-val dice={v.dice.mean():.4f}  "
                  f"misses={int(v.complete_miss.sum())}/{len(v)}")

        del model
        if str(env.device).startswith("cuda"):
            torch.cuda.empty_cache()

    return pd.concat(out, ignore_index=True)


# ── reading the result ───────────────────────────────────────────────────────

def _clustered_ci(values: np.ndarray, groups: np.ndarray, n_boot: int,
                  rng: np.random.Generator) -> tuple[float, float]:
    """Subject-clustered bootstrap CI on a mean. Same policy as Stages D/G/M/N.

    Resamples SUBJECTS. Images of one subject are correlated, and resampling them
    independently returns an interval far narrower than the data supports.
    """
    uniq = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in uniq}
    boots = np.empty(n_boot)
    for b in range(n_boot):
        take = np.concatenate([idx[g] for g in rng.choice(uniq, len(uniq), replace=True)])
        boots[b] = values[take].mean()
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def cross_matrix(per_image: pd.DataFrame, metric: str = "dice") -> pd.DataFrame:
    """Train-labeler x eval-labeler means on the shared test set, pooled over folds."""
    t = per_image[per_image.split == "test"]
    if metric == "complete_miss":
        agg = t.groupby(["train_masks", "eval_masks", "fold"]).complete_miss.mean()
        return agg.groupby(level=[0, 1]).mean().unstack()
    return t.groupby(["train_masks", "eval_masks"])[metric].mean().unstack()


def labeler_table(per_image: pd.DataFrame, n_boot: int = 10000,
                  seed: int = 0) -> pd.DataFrame:
    """One row per labeler: the three numbers the decision should rest on.

      cv_dice        held-out CV Dice against that labeler's own masks. How
                     learnable their annotation style is on unseen subjects.
      self_dice      test Dice against their own masks. Same question on the
                     shared images, so it is comparable ACROSS labelers.
      cross_dice     mean test Dice against the OTHER two labelers' masks. A
                     model that only satisfies its own annotator has learned that
                     annotator's idiosyncrasies; one that satisfies all three has
                     learned the bruise. THIS is the column that identifies the
                     best labeler.
      misses         complete misses (dice == 0) against own masks, the endpoint
                     handbook 1 says decides when Dice is saturated.

    Dice is reported with a subject-clustered CI. The CI is over the shared test
    subjects, so it is a real interval; it is NOT a fold-to-fold spread and does
    not include seed variance, because this is a single-seed study.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for lab in sorted(per_image.train_masks.unique()):
        t = per_image[(per_image.train_masks == lab) & (per_image.split == "test")]
        own = t[t.eval_masks == lab]
        oth = t[t.eval_masks != lab]
        cv = per_image[(per_image.train_masks == lab) & (per_image.split == "cv_val")]

        lo, hi = _clustered_ci(own.dice.to_numpy(float), own.subject.to_numpy(),
                               n_boot, rng)
        rows.append({
            "labeler": lab,
            "cv_dice": round(float(cv.dice.mean()), 4),
            "cv_misses": int(cv.complete_miss.sum()),
            "cv_n": int(len(cv)),
            "self_dice": round(float(own.dice.mean()), 4),
            "self_ci_lo": round(lo, 4), "self_ci_hi": round(hi, 4),
            "cross_dice": round(float(oth.dice.mean()), 4),
            "self_minus_cross": round(float(own.dice.mean() - oth.dice.mean()), 4),
            "self_misses_per_fold": round(float(
                own.groupby("fold").complete_miss.sum().mean()), 2),
            "median": round(float(own.dice.median()), 4),
            "iqr": round(float(own.dice.quantile(.75) - own.dice.quantile(.25)), 4),
        })
    out = pd.DataFrame(rows).sort_values("cross_dice", ascending=False)
    return out.reset_index(drop=True)


def paired_contrast(per_image: pd.DataFrame, a: str, b: str, eval_masks: str,
                    n_boot: int = 10000, seed: int = 0) -> dict:
    """Subject-clustered paired bootstrap of (a - b) on ONE mask set.

    Paired by (stem, fold): both arms saw the same test images from the same
    fold split, so the comparison is within-image and the between-image variance
    -- which dwarfs the between-labeler effect -- cancels.
    """
    t = per_image[(per_image.split == "test") & (per_image.eval_masks == eval_masks)]
    wide = (t.pivot_table(index=["stem", "subject", "fold"], columns="train_masks",
                          values="dice").dropna(subset=[a, b]).reset_index())
    d = (wide[a] - wide[b]).to_numpy(float)
    lo, hi = _clustered_ci(d, wide.subject.to_numpy(), n_boot,
                           np.random.default_rng(seed))
    return {"a": a, "b": b, "eval_masks": eval_masks, "n_pairs": int(len(d)),
            "delta": round(float(d.mean()), 4), "ci_lo": round(lo, 4),
            "ci_hi": round(hi, 4),
            "verdict": "A_BETTER" if lo > 0 else "B_BETTER" if hi < 0 else "INCONCLUSIVE"}


def print_verdict(table: pd.DataFrame, contrasts: list, margin: float = 0.01) -> None:
    """State the finding, including when there is not one.

    `margin` is the study's 0.01 Dice threshold. An interval inside +/-margin is
    reported as EQUIVALENT, not as a win for whoever came top -- the same rule
    Stages C, M and N use, and the reason this study has published nulls.
    """
    print(table.to_string(index=False))
    print()
    decisive = [c for c in contrasts if c["verdict"] != "INCONCLUSIVE"]
    if not decisive:
        print(f"NO LABELER SEPARATES. Every paired contrast crosses zero, so the "
              f"three annotators are indistinguishable at this sample size. Rank "
              f"them by `cross_dice` if a pick is forced, but the ranking is not "
              f"supported and must be reported as such.")
        return
    for c in decisive:
        better = c["a"] if c["verdict"] == "A_BETTER" else c["b"]
        d = abs(c["delta"])
        tag = "within the 0.01 margin -- EQUIVALENT in this study's terms" \
            if d < margin else "clears the 0.01 margin"
        print(f"{better} better on {c['eval_masks']} masks: "
              f"delta {c['delta']:+.4f} [{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}], {tag}")


def self_test() -> bool:
    """Structural checks -- no data, no GPU, no network.

    Guards the two failures that would silently produce a plausible ranking:
    a fold assignment that puts a subject on both sides, and a matched core whose
    arms are not actually matched.
    """
    n = {"1": 10, "2": 6, "3": 6, "4": 3, "5": 3, "6": 1}
    folds = _assign_folds(n, 3)
    assert set(folds) == set(n), "every subject must land in a fold"
    load = Counter()
    for s, k in folds.items():
        load[k] += n[s]
    assert max(load.values()) - min(load.values()) <= 3, f"folds unbalanced: {dict(load)}"
    assert subject_of("182_Q_2-6_WL_Bruise_A244A_2026-04-06_DSC_2484") == "182"
    for bad in ("no_leading_digits", "", "abc_1"):
        try:
            subject_of(bad)
        except ValueError:
            pass
        else:                                                  # pragma: no cover
            raise AssertionError(f"subject_of accepted {bad!r}")
    return True
