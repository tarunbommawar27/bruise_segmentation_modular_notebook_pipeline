#!/usr/bin/env python
"""Every model, every endpoint, one CSV -- from a shell. No Jupyter.

    python run_all_models.py
    python run_all_models.py --extra-roots /scratch/tbommawa/bruise_work
    python run_all_models.py --discover-only        # just show what is on disk

Reads every per-image CSV it can find, wherever it lives, and writes
`ALL_MODELS_RESULTS/ALL_MODELS_SUMMARY.csv`: one row per model carrying mean and
median Dice, zero-Dice count and rate, wrong-place misses, small-lesion recall
(D1-D4 and D1), and the ITA fairness block marginal AND conditioned on lesion
size.

It runs no inference and re-fits no threshold. Every number comes from a table
some other stage already wrote, at that stage's own operating point.

ON ORC THE RUNS ARE NOT ALL IN THE BUNDLE. Outputs land under the work directory
on scratch and FINAL_RESULT/ may never have been synced. Pass every plausible
tree to --extra-roots; the discovery log records which ones actually held
anything, so an over-broad list costs nothing but a few seconds of globbing.

EXIT CODES
    0  wrote the tables
    1  nothing matched a manifest split -- see the printed discovery log
    2  a precondition failed
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DEFAULT_EXTRA = [
    "/scratch/tbommawa/bruise_work",
    "/scratch/tbommawa/bruise_work/outputs",
]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="One table for every model this project has scored.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--root", default=None, help="bundle root (default: here)")
    ap.add_argument("--work", default=None, help="work dir (default: <root>/_work)")
    ap.add_argument("--extra-roots", nargs="*", default=None,
                    help=f"extra trees to scan (default: {DEFAULT_EXTRA})")
    ap.add_argument("--n-boot", type=int, default=10000,
                    help="subject-clustered resamples (default: 10000)")
    ap.add_argument("--out", default=None,
                    help="results dir (default: <root>/ALL_MODELS_RESULTS)")
    ap.add_argument("--discover-only", action="store_true",
                    help="list what is on disk and stop")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    root = Path(args.root).resolve() if args.root else here
    sys.path.insert(0, str(root))

    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)
    import pandas as pd

    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 120)

    from bruisekit import allmodels as AM
    from bruisekit import paths as P

    extra = args.extra_roots if args.extra_roots is not None else DEFAULT_EXTRA

    print("=" * 78)
    print("ALL MODELS -- every endpoint, one table")
    print("=" * 78)
    env = P.setup(root=root, work=args.work)
    print(env.describe())
    print(f"\nextra roots: {extra}")

    print("\n-- self test --")
    if not AM.self_test(verbose=True):
        print("allmodels self-test FAILED", file=sys.stderr)
        return 2

    print("\n-- roots that exist and hold candidate files --")
    for r in AM.search_roots(env, extra):
        n = sum(len(list(r.glob(g))) for g in AM.GLOBS)
        if n:
            print(f"  {n:>4}   {r}")

    if args.discover_only:
        found = AM.discover(env, extra, verbose=True)
        print(f"\n{len(found)} candidate file(s). Re-run without --discover-only "
              f"to load and summarise.")
        return 0

    try:
        out = AM.run(env, extra_roots=extra, n_boot=args.n_boot, seed=0,
                     verbose=True)
    except RuntimeError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1

    print()
    AM.print_summary(out, top=100)

    if args.out:
        env = P.setup(root=root, work=args.work)
        Path(args.out).mkdir(parents=True, exist_ok=True)
        written = []
        names = {"summary": "ALL_MODELS_SUMMARY", "by_decile": "all_models_by_decile",
                 "by_group": "all_models_by_group", "discovery": "discovery_log",
                 "bins": "size_bins"}
        for k, fname in names.items():
            obj = out.get(k)
            if isinstance(obj, pd.DataFrame) and not obj.empty:
                p = Path(args.out) / f"{fname}.csv"
                obj.to_csv(p, index=False)
                written.append(p)
    else:
        written = AM.save(env, out)

    print("\n-- written --")
    for p in written:
        print(f"  {p}")

    s = out["summary"]
    print(f"\nTHE SINGLE TABLE: ALL_MODELS_SUMMARY.csv "
          f"({len(s)} rows x {s.shape[1]} cols, "
          f"{int(s.comparable.sum())} comparable)")
    print("\nTo bring it back:")
    print(f"    zip -r all_models.zip {AM.RESULTS_DIRNAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
