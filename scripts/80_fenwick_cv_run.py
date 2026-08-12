#!/usr/bin/env python
"""Run the Fenwick labeler CV -- one process per GPU.

WHY A SCRIPT AND NOT JUST THE NOTEBOOK
---------------------------------------
Two GPUs means two processes. A single notebook kernel sees one CUDA context and
would serialise the arms; `CUDA_VISIBLE_DEVICES` is per-process and cannot be
changed after torch initialises. So the notebook builds the core, previews the
batch finder and reads the results, and this script is what the two shells run.

    # shell 1 -- two labelers, sequential, on GPU 0
    CUDA_VISIBLE_DEVICES=0 python scripts/80_fenwick_cv_run.py \
        --fenwick /path/FENWICK_LABELER_DATASET --labelers hliu36 nmousta5

    # shell 2 -- the third on GPU 1
    CUDA_VISIBLE_DEVICES=1 python scripts/80_fenwick_cv_run.py \
        --fenwick /path/FENWICK_LABELER_DATASET --labelers mzehra2

Both processes read the SAME core (`--out`), which the first one to start writes
and the others reuse -- so the two GPUs cannot end up training against two
different cuts of the data. Scoring is a separate `--score` pass, run once after
both shells finish, because the cross-labeler matrix needs every arm present.

The arms are volume-matched, so 2+1 leaves GPU 0 with twice the work. `--folds`
splits the other way if you would rather balance: give each GPU all three
labelers and half the folds.

RESUMABLE. `engine.train_run` skips a fold with `DONE.json` and restarts an
interrupted one from `resume.pt`, so re-running either shell is safe.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "BRUISE_UNIFIED"
sys.path.insert(0, str(BUNDLE))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fenwick", required=True, help="the FENWICK_LABELER_DATASET root")
    p.add_argument("--out", default=None,
                   help="results dir (default <fenwick>/../FENWICK_CV_RESULTS)")
    p.add_argument("--work", default=None,
                   help="scratch for the 640 cache (default <out>/_work)")
    p.add_argument("--labelers", nargs="+", default=None,
                   help="which arms this process trains (default: all three)")
    p.add_argument("--folds", nargs="+", type=int, default=None,
                   help="which folds this process trains (default: all)")
    p.add_argument("--score", action="store_true",
                   help="score instead of train; run once after every arm is trained")
    p.add_argument("--epochs", type=int, default=None, help="override the cap")
    p.add_argument("--rebuild-cache", action="store_true")
    a = p.parse_args()

    from bruisekit import fenwick_cv as F

    fen = Path(a.fenwick).expanduser().resolve()
    out = Path(a.out).expanduser().resolve() if a.out else fen.parent / "FENWICK_CV_RESULTS"
    work = Path(a.work).expanduser().resolve() if a.work else out / "_work"
    runs, tables = out / "runs", out / "tables"
    out.mkdir(parents=True, exist_ok=True)

    assert F.self_test(), "fenwick_cv self-test failed -- do not train on this"

    env = F.make_env(BUNDLE, fen, work)
    print(f"device  : {env.device}   "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}")
    print(f"fenwick : {fen}\nout     : {out}")

    # ONE cut of the data for every process. The first writer wins; everyone else
    # reads what is on disk, so two GPUs cannot train against two different cores.
    if (out / "core_design.json").exists():
        core = F.load_core(out)
        print(f"core    : reused from {out} "
              f"({core.n_per_arm} imgs/arm, {len(core.shared_subjects)} subjects)")
    else:
        core = F.build_core(fen)
        F.write_core(core, out)
        print(f"core    : built -- {core.n_per_arm} imgs/arm, "
              f"{len(core.shared_subjects)} subjects, "
              f"{len(core.dropped_subjects)} test subjects dropped from the pools")
        print(core.report.to_string(index=False))

    cfg = F.default_cfg(**({"epochs": a.epochs} if a.epochs else {}))
    cached = F.build_cache(env, core, force=a.rebuild_cache)

    labelers = a.labelers or list(F.TOP3)
    unknown = set(labelers) - set(F.TOP3)
    if unknown:
        print(f"FAIL: unknown labeler(s) {sorted(unknown)}; have {F.TOP3}", file=sys.stderr)
        return 1
    folds = a.folds if a.folds is not None else list(range(F.N_FOLDS))

    t0 = time.time()
    if a.score:
        import pandas as pd
        parts = [F.score_labeler(env, cfg, cached, lab, runs, tables, folds=folds)
                 for lab in labelers]
        per_image = pd.concat(parts, ignore_index=True)
        per_image.to_csv(tables / "per_image_all.csv", index=False)

        table = F.labeler_table(per_image)
        table.to_csv(tables / "labeler_table.csv", index=False)
        for metric in ("dice", "complete_miss"):
            F.cross_matrix(per_image, metric).to_csv(tables / f"cross_{metric}.csv")

        contrasts = [F.paired_contrast(per_image, x, y, evl)
                     for evl in labelers
                     for i, x in enumerate(labelers) for y in labelers[i + 1:]]
        (tables / "contrasts.json").write_text(json.dumps(contrasts, indent=2))

        print("\n" + "=" * 70)
        print("CROSS-LABELER TEST DICE  (rows = trained on, cols = scored against)")
        print(F.cross_matrix(per_image).round(4).to_string())
        print()
        F.print_verdict(table, contrasts)
    else:
        print(f"batch   : {F.probe_batch(env, cfg)}")
        for lab in labelers:
            F.train_labeler(env, cfg, cached, lab, runs, folds=folds)

    print(f"\nelapsed {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
