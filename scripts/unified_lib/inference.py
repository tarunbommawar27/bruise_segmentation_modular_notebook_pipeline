"""The inference block: one test-set pass and one speed table, for any set of models.

WHY THIS MODULE EXISTS
-----------------------
`bruise_colab_final_analysis.ipynb` uses the word "inference" for two different
things, and the difference matters enough that this module keeps them apart:

  (a) THE INFERENCE PASS -- one forward over the 185 test images at the
      val-selected best seed, producing the per-image table every figure and
      every statistic downstream is derived from. Cheap to state, expensive to
      get wrong: the operating point is read from the run, never re-fitted here.

  (b) THE SPEED BENCHMARK -- median/mean/p95 milliseconds and FPS at 640. The
      analysis notebook does NOT compute this; its speed figure only *reads*
      `results/final/benchmark_640.csv`, which was produced once by
      `bruise_colab_final.ipynb`. So "calculate inference" is mostly (b): (a) is
      already cached for the headline models, (b) exists for five models and for
      nothing since.

Both are implemented here over the registry, so they work for any family that
has a checkpoint -- the five headline models, the four mobile baselines, the
Stage F/H distilled arms -- not just the set that happened to be timed in 2026.

WHAT IS REPRODUCED VERBATIM, AND WHY
-------------------------------------
The timing recipe is copied from `bruise_colab_final.ipynb` exactly: three
repeats, ten warmup iterations, seed 0, per-image (batch of one), double
`cuda.synchronize()`, images staged on the device once through the SAME 640
dataloader the models were trained on. A new row is only comparable to the five
shipped rows if it was measured the same way. On CUDA the SegFormer path calls
`evaluate.benchmark_speed` itself rather than a copy of it, so there is one
implementation of the published number and it cannot drift.

THE FOUR CONSTRAINTS THAT MAKE A SPEED TABLE MEAN ANYTHING
-----------------------------------------------------------
1. ONE MACHINE. The shipped Stage A rows are full-A100; the Stage E rows are an
   A100 MIG 3g.40gb slice; anything measured here is whatever you ran it on.
   Rows from different devices are not a table, so every row carries `device`
   and `device_name` and `write_speed_table` refuses to merge across them
   silently.
2. SEGFORMER ROWS TIME FORWARD + THRESHOLD; YOLO ROWS TIME RAW FORWARD ONLY.
   YOLO's raw module returns a detection tuple, not [B,1,H,W] logits, so there
   is no comparable threshold step to include. The `path` column records which
   of the two methods produced the row and must survive into any figure.
3. NOTHING AROUND THE FORWARD IS TIMED -- no disk read, decode, resize, H2D or
   D2H. Those are identical across models and dominated by I/O, which would
   compress the architectural differences into noise.
4. YOLO IS /255, NEVER IMAGENET NORM. The 640 loader emits /255 and SegFormer
   applies ImageNet normalisation inside its own forward, so one staged batch is
   correct for both. Feeding YOLO an ImageNet-normalised tensor silently caps it
   near 0.479 Dice and no threshold recovers it.

CPU
---
A CPU run is supported for smoke-testing and for the inference pass, which is
exact everywhere. CPU *timings* are not: they are recorded with
`device == "cpu"`, benchmarked over a subset by default, and must never be put
in the same table as a GPU row.
"""
from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# The three SegFormers: the analysis notebook's own `SEGFORMER_MODELS` dict, and
# the exact three rows of track_a_comparison.csv. Overridable -- `--models` takes
# any registry family names -- but this is the set "the three models" refers to
# everywhere else in this study.
DEFAULT_MODELS: tuple[str, ...] = (
    "segformer_b2_teacher",
    "segformer_b0_direct",
    "segformer_b0_distilled",
)

# The published recipe. Changing any of these makes new rows incomparable with
# the five in results/final/benchmark_640.csv.
BENCH_REPEATS = 3
BENCH_WARMUP = 10

# Speed is architectural, not per-seed -- the shipped table was measured at seed 0
# for every model, including models whose val-selected best seed is not 0.
BENCH_SEED = 0

# Staging all 185 images costs ~0.9 GB of float32. Fine on an A100, not fine as a
# default on a laptop, where the timing is unpublishable anyway.
CPU_BENCH_IMAGES = 16

# `path` names the TIMING METHOD, not the architecture. Two values only, because
# there are two methods: a single-logit model that can be thresholded, and YOLO's
# raw module which cannot.
PATH_LOGIT = "segformer"
PATH_YOLO = "yolo_native_raw_forward"

_LOGIT_KINDS = ("segformer", "smp", "efficient")


# ── run resolution ───────────────────────────────────────────────────────────
def _device_name(device) -> str:
    """A human-readable name for whatever we are about to time on."""
    if str(device).startswith("cuda"):
        try:
            return torch.cuda.get_device_name(device)
        except Exception:                                  # pragma: no cover
            return "cuda (name unavailable)"
    return "cpu"


def resolve_runs(env, reg, models=DEFAULT_MODELS, seed: int | None = None) -> dict:
    """Map each family name to the `Run` this session should use.

    Parameters
    ----------
    seed : pin every model to this seed. None (the default) means the
        val-selected best seed per model, read from `report.best_seeds`.

    THE BEST SEED IS NOT THE SAME FOR EVERY MODEL. It is 0 for the three
    SegFormers and for yolo_sem_distilled, and 2 for yolo_sem_direct. Scoring a
    model's weights at another model's best seed shows per-image disagreements up
    to 0.49 Dice and looks exactly like a broken inference path, so the mapping is
    read off the selection step's own filenames rather than assumed.

    A family with no WEIGHTS-tier run is reported and skipped, never back-filled
    from a neighbouring seed.
    """
    from bruisekit import report as R
    from bruisekit.registry import WEIGHTS

    best = R.best_seeds(env) if seed is None else {}
    out, skipped = {}, []
    for family in models:
        s = seed if seed is not None else best.get(family, 0)
        run = reg.get(f"{family}__seed{s}")
        if run is None:
            skipped.append((family, f"no run {family}__seed{s} in the registry"))
        elif run.tier != WEIGHTS:
            skipped.append((family, f"{run.run_id} is tier {run.tier}, not WEIGHTS"))
        else:
            out[family] = run
    for family, why in skipped:
        print(f"  SKIP  {family:<32} {why}")
    return out


# ── (a) the inference pass ───────────────────────────────────────────────────
def inference_pass(env, reg, cfg, man, man640, models=DEFAULT_MODELS,
                   seed: int | None = None) -> dict[str, pd.DataFrame]:
    """One forward over the 185 test images per model, at its val-fitted cut.

    Returns {family: per-image DataFrame} in the common schema (`report.normalize`),
    with subject and ITA group joined from the test manifest.

    This delegates to `loaders.score_run`, which already dispatches over
    segformer / smp / efficient / yolo and already carries the CPU memory guards.
    There is deliberately no second inference implementation in this file.

    `man640` is required for every kind except YOLO: those models are scored on
    exactly the tensors they were trained on, not on a fresh resize, so a
    re-scored number stays comparable with the shipped one. YOLO is scored from
    native-resolution images because Ultralytics letterboxes internally and
    feeding it pre-resized images would apply the resize twice.
    """
    from bruisekit import loaders as L
    from bruisekit import report as R

    runs = resolve_runs(env, reg, models, seed)
    meta = man["test"]
    tables: dict[str, pd.DataFrame] = {}
    for family, run in runs.items():
        if run.kind != "yolo" and man640 is None:
            print(f"  SKIP  {family:<32} needs the 640 cache "
                  f"(call loaders.build_cache640 first)")
            continue
        t0 = time.time()
        df = R.normalize(L.score_run(env, run, cfg, man640, meta), meta)
        tables[family] = df
        print(f"  {family:<32} {run.run_id:<38} "
              f"mean dice {df.dice.mean():.4f}  misses {int((df.dice == 0).sum()):>2}  "
              f"({time.time() - t0:.0f}s)")
    return tables


def reconcile(env, reg, tables: dict[str, pd.DataFrame], man) -> pd.DataFrame:
    """Fresh per-image scores vs the shipped ones, per model.

    A re-inference is only trustworthy if it agrees with the table that produced
    the published numbers. The handbook's claim is that every reporting model
    agrees to better than 2e-4 mean Dice (CPU vs the original A100), the residual
    being float ordering rather than disagreement. This function is how that claim
    is checked rather than repeated.

    `max_abs_dice_delta` is the column that matters: a mean can agree while
    individual images disagree wildly, which is the signature of a seed mismatch
    (see `resolve_runs`) rather than of float noise.
    """
    from bruisekit import report as R

    best = R.best_seeds(env)
    rows = []
    for family, fresh in tables.items():
        run_id = f"{family}__seed{best.get(family, 0)}"
        shipped = R.load_per_image(env, reg, run_id, man["test"])
        if shipped is None:
            rows.append({"model": family, "run_id": run_id, "status": "no shipped table",
                         "n": len(fresh), "mean_dice_fresh": float(fresh.dice.mean()),
                         "mean_dice_shipped": np.nan, "mean_dice_delta": np.nan,
                         "max_abs_dice_delta": np.nan, "miss_delta": np.nan})
            continue
        j = fresh[["stem", "dice"]].merge(shipped[["stem", "dice"]], on="stem",
                                          how="inner", suffixes=("_fresh", "_shipped"))
        d = (j.dice_fresh - j.dice_shipped).abs()
        rows.append({
            "model": family, "run_id": run_id,
            "status": "ok" if len(j) == len(fresh) else f"only {len(j)}/{len(fresh)} stems matched",
            "n": len(j),
            "mean_dice_fresh": float(fresh.dice.mean()),
            "mean_dice_shipped": float(shipped.dice.mean()),
            "mean_dice_delta": float(fresh.dice.mean() - shipped.dice.mean()),
            "max_abs_dice_delta": float(d.max()),
            "miss_delta": int((fresh.dice == 0).sum()) - int((shipped.dice == 0).sum()),
        })
    return pd.DataFrame(rows)


# ── (b) the speed benchmark ──────────────────────────────────────────────────
def stage_images(env, cfg, man640, n: int | None = None) -> torch.Tensor:
    """Put the test tensors on the device once, through the training dataloader.

    The loader emits /255 tensors. That is exactly what both consumers want:
    SegFormer applies ImageNet normalisation inside its own forward, and YOLO's
    raw module is a /255 model. One staged batch is therefore valid input for
    every model, and no model gets a preprocessing advantage over another.
    """
    from bruisekit.data import make_loader

    loader = make_loader(man640["test"], env.cache640, cfg["img_size"], 8, False,
                         cfg.get("workers", 0), 0)
    x = torch.cat([b for b, _, _ in loader])
    if n is not None:
        x = x[:n]
    x = x.to(env.device)
    gb = x.element_size() * x.nelement() / 1e9
    print(f"  staged {tuple(x.shape)} = {gb:.2f} GB on {env.device}")
    return x


@torch.no_grad()
def _benchmark_cpu(forward, images, repeats: int, warmup: int) -> dict:
    """The published timing loop with the CUDA synchronisation removed.

    Used only when there is no CUDA device. It is a separate function rather than
    a branch inside `evaluate.benchmark_speed` because that function's outputs are
    the published ones and it is shipped verbatim; a CPU row is a different kind
    of number and should come from a different function that says so.
    """
    for _ in range(warmup):
        _ = forward(images[:1])
    times = []
    for _ in range(repeats):
        for i in range(len(images)):
            t0 = time.perf_counter()
            _ = forward(images[i:i + 1])
            times.append((time.perf_counter() - t0) * 1000)
    arr = np.array(times)
    return {"median_ms": float(np.median(arr)), "mean_ms": float(arr.mean()),
            "p95_ms": float(np.percentile(arr, 95)), "fps": float(1000.0 / np.median(arr)),
            "n_timed": len(arr)}


@torch.no_grad()
def _benchmark_yolo_cuda(model, images, repeats: int, warmup: int) -> dict:
    """Raw forward only, timed exactly as `evaluate.benchmark_speed` times SegFormer.

    `benchmark_speed`'s threshold step assumes [B,1,H,W] logits; YOLO's raw module
    returns a detection tuple, so there is nothing to threshold. Everything else --
    per-image batches, double synchronisation, repeats, warmup -- is identical, and
    the row is labelled `yolo_native_raw_forward` so no figure can compare it to a
    SegFormer row as though the two included the same work.
    """
    model.eval()
    for _ in range(warmup):
        _ = model(images[:1])
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        for i in range(len(images)):
            x = images[i:i + 1]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    arr = np.array(times)
    return {"median_ms": float(np.median(arr)), "mean_ms": float(arr.mean()),
            "p95_ms": float(np.percentile(arr, 95)), "fps": float(1000.0 / np.median(arr)),
            "n_timed": len(arr)}


def speed_table(env, reg, cfg, man640, models=DEFAULT_MODELS,
                repeats: int = BENCH_REPEATS, warmup: int = BENCH_WARMUP,
                n_images: int | None = None, seed: int = BENCH_SEED) -> pd.DataFrame:
    """Time 640-tensor-on-device -> mask-on-device for each model. One machine, one table.

    Columns match `results/final/benchmark_640.csv` -- median_ms, mean_ms, p95_ms,
    fps, n_timed, model, path, params_M, peak_activation_MB -- plus `kind`,
    `device`, `device_name`, `repeats`, `warmup` and `seed`, because the shipped
    table's five rows are only interpretable if you already know all six of those
    and they were not written down.

    Every model is timed at `seed` (0 by default), including models whose
    val-selected best seed is not 0: speed is a property of the architecture, and
    mixing seeds here would put per-seed noise into a column that has none.
    """
    from bruisekit import loaders as L
    from bruisekit.evaluate import benchmark_speed
    from bruisekit.models import count_params

    on_cuda = str(env.device).startswith("cuda")
    if n_images is None:
        n_images = None if on_cuda else CPU_BENCH_IMAGES
    if not on_cuda:
        print(f"  NOTE  no CUDA device: timing {n_images} images on CPU. These rows are "
              f"for smoke-testing only and must not share a table with GPU rows.")

    runs = resolve_runs(env, reg, models, seed=seed)
    if not runs:
        return pd.DataFrame()

    images = stage_images(env, cfg, man640, n_images)
    rows = []
    for family, run in runs.items():
        if on_cuda:
            torch.cuda.reset_peak_memory_stats(env.device)

        if run.kind in _LOGIT_KINDS:
            model, cut = L.load_model(env, run)
            if on_cuda:
                b = benchmark_speed(model, images, env.device, cut, repeats, warmup)
            else:
                b = _benchmark_cpu(lambda x: model(x) >= cut, images, repeats, warmup)
            b["path"] = PATH_LOGIT
            b["params_M"] = count_params(model) / 1e6
        elif run.kind == "yolo":
            import bruisekit.yolo_native as yn
            model = yn._raw_module(run.weights, env.device)
            if on_cuda:
                b = _benchmark_yolo_cuda(model, images, repeats, warmup)
            else:
                b = _benchmark_cpu(model, images, repeats, warmup)
            b["path"] = PATH_YOLO
            b["params_M"] = sum(p.numel() for p in model.parameters()) / 1e6
        else:
            print(f"  SKIP  {family:<32} no timing path for kind={run.kind!r}")
            continue

        b.update({
            "model": family, "kind": run.kind, "run_id": run.run_id,
            "peak_activation_MB": (torch.cuda.max_memory_allocated(env.device) / 1e6
                                   if on_cuda else float("nan")),
            "device": "cuda" if on_cuda else str(env.device),
            "device_name": _device_name(env.device),
            "repeats": repeats, "warmup": warmup, "seed": seed,
        })
        rows.append(b)
        print(f"  {family:<32} {b['median_ms']:7.2f} ms  {b['fps']:7.1f} FPS  "
              f"p95={b['p95_ms']:6.2f} ms  {b['params_M']:.2f} M  [{b['path']}]")

        del model
        gc.collect()
        if on_cuda:
            torch.cuda.empty_cache()

    del images
    gc.collect()
    if on_cuda:
        torch.cuda.empty_cache()

    cols = ["median_ms", "mean_ms", "p95_ms", "fps", "n_timed", "model", "path",
            "params_M", "peak_activation_MB", "kind", "run_id", "device",
            "device_name", "repeats", "warmup", "seed"]
    return pd.DataFrame(rows)[cols]


def check_single_machine(df: pd.DataFrame) -> pd.DataFrame:
    """Raise if a speed table mixes devices. Called before anything is written.

    This is the constraint that a reader cannot check for themselves once the CSV
    exists, so it is enforced where the CSV is made. Merging a laptop row into a
    GPU table is how track_b's FPS column was destroyed once already.
    """
    if df.empty:
        return df
    names = sorted(set(df["device_name"]))
    if len(names) > 1:
        raise ValueError(
            "speed table mixes devices and is therefore not a table: "
            f"{names}. Time every model on one machine, or keep the CSVs separate.")
    return df


# ── driver ───────────────────────────────────────────────────────────────────
def run(env, reg, cfg, man, man640, models=DEFAULT_MODELS, *,
        do_inference: bool = True, do_speed: bool = True, do_reconcile: bool = True,
        seed: int | None = None, repeats: int = BENCH_REPEATS,
        warmup: int = BENCH_WARMUP, n_images: int | None = None,
        out_dir: Path | None = None) -> dict:
    """Both halves of §18.1, written to `out_dir` (default `env.out/inference`).

    Returns {"per_image": {family: df}, "summary": df, "speed": df,
             "reconcile": df} -- whichever parts were asked for.
    """
    from bruisekit import report as R

    out = Path(out_dir) if out_dir is not None else env.out / "inference"
    out.mkdir(parents=True, exist_ok=True)
    result: dict = {}

    if do_inference:
        print(f"\nINFERENCE PASS -- {len(models)} model(s), 185 test images")
        tables = inference_pass(env, reg, cfg, man, man640, models, seed)
        result["per_image"] = tables
        for family, df in tables.items():
            df.to_csv(out / f"per_image_{family}.csv", index=False)
        if tables:
            summary = R.headline(tables)
            summary.to_csv(out / "inference_headline.csv", index=False)
            result["summary"] = summary
            print(summary.to_string(index=False))
        if do_reconcile and tables:
            rec = reconcile(env, reg, tables, man)
            rec.to_csv(out / "inference_reconciliation.csv", index=False)
            result["reconcile"] = rec
            print("\nFRESH vs SHIPPED")
            print(rec.to_string(index=False))

    if do_speed:
        print(f"\nSPEED BENCHMARK -- {repeats} repeats, {warmup} warmup, seed {BENCH_SEED}")
        sp = check_single_machine(
            speed_table(env, reg, cfg, man640, models, repeats, warmup, n_images))
        if not sp.empty:
            # Named for the device so a CPU smoke-test can never be mistaken for,
            # or silently overwrite, the GPU table it is not comparable to.
            tag = "cuda" if str(env.device).startswith("cuda") else "cpu"
            sp.to_csv(out / f"benchmark_640_{tag}.csv", index=False)
        result["speed"] = sp

    print(f"\nwrote {out}")
    return result


# The eval-time subset of the notebook's CFG: only the keys the scoring and
# benchmarking paths actually read. A CLI run must not have to restate a training
# recipe it will never use, and every value here is identical to the notebook's.
DEFAULT_CFG = dict(
    img_size=640,
    amp=True,
    workers=0,          # 0 is safe everywhere; raise on Linux for speed
    eval_batch=8,       # GPU
    eval_batch_cpu=2,   # batch cannot affect per-image scores, only wall clock
    seeds=(0, 1, 2),
)


def main(argv=None) -> int:
    """CLI: python -m bruisekit.inference [--models ...] [--no-speed] ..."""
    import argparse

    p = argparse.ArgumentParser(
        description="Test-set inference pass and 640 speed benchmark, over the registry.")
    p.add_argument("--bundle", default=None, help="bundle root (default: auto-detect)")
    p.add_argument("--work", default=None, help="scratch dir for the 640 cache")
    p.add_argument("--device", default=None, help='e.g. "cpu" to force CPU')
    p.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS),
                   help=f"registry family names (default: {' '.join(DEFAULT_MODELS)})")
    p.add_argument("--seed", type=int, default=None,
                   help="pin the inference pass to this seed (default: val-selected best)")
    p.add_argument("--repeats", type=int, default=BENCH_REPEATS)
    p.add_argument("--warmup", type=int, default=BENCH_WARMUP)
    p.add_argument("--bench-images", type=int, default=None,
                   help=f"images to time (default: all on CUDA, {CPU_BENCH_IMAGES} on CPU)")
    p.add_argument("--no-inference", action="store_true", help="speed benchmark only")
    p.add_argument("--no-speed", action="store_true", help="inference pass only")
    p.add_argument("--no-reconcile", action="store_true",
                   help="skip the fresh-vs-shipped comparison")
    p.add_argument("--out", default=None, help="output dir (default: <work>/outputs/inference)")
    a = p.parse_args(argv)

    from bruisekit import loaders as L
    from bruisekit.paths import setup
    from bruisekit.registry import Registry

    env = setup(root=a.bundle, work=a.work, device=a.device)
    print(env.describe())

    man = {s: pd.read_csv(env.manifests / f"{s}.csv") for s in ("train", "val", "test")}
    reg = Registry(env).scan()

    # The 640 cache is needed by every path except YOLO's. Built rather than
    # demanded: it is a deterministic resize of data/ and rebuilds in about a
    # minute, so requiring the caller to have made it first buys nothing.
    print("\n640 cache")
    man640 = L.build_cache640(env, man)

    run(env, reg, DEFAULT_CFG, man, man640, tuple(a.models),
        do_inference=not a.no_inference, do_speed=not a.no_speed,
        do_reconcile=not a.no_reconcile, seed=a.seed, repeats=a.repeats,
        warmup=a.warmup, n_images=a.bench_images, out_dir=a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
