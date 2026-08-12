#!/usr/bin/env python
"""Stage O end to end from a shell. No Jupyter, no notebook, no display.

    python run_stage_o.py --root . --work ./_work

Does exactly what `bruise_stage_o.ipynb` does, in the same order and through the
same functions, so the two cannot drift into producing different numbers.

    miss       zero Dice split into empty prediction and wrong place   (TODO 1)
    fairness   skin tone x lesion size for the distilled arms          (TODO 3)
    matrix     score the teacher pool on validation                    (TODO 2)
    gate       the pre-registered verdict         <- written BEFORE any training
    train      engine.train_run, unmodified, resumable
    score      val + test per-image tables for the trained arms
    report     itakd vs control, and vs Stage M's mtkd arms

THE FIRST TWO STAGES NEED NO GPU AND NO CHECKPOINTS. They read the per-image CSVs
that already exist in the lineage, so `--only miss,fairness` runs on a laptop in
under a minute and closes TODO items 1 and 3 on its own. Everything from `matrix`
onward needs the trained teacher checkpoints, and `train` needs CUDA.

WHY THE ORDER IS FIXED
----------------------
The gate is VAL-ONLY by design, the same rule Stages M, N3 and N4 follow: a
decision taken on test is a decision taken on the data the paper reports. `gate`
writes its verdict to disk before `train` runs, and `--only train` REFUSES to
start if no gate file exists. That refusal is the point -- it makes the ordering a
property of the code rather than of whoever ran it.

The gate can come back DO NOT RUN, and on the five-group scheme it is expected to:
the per-group teacher argmax is not estimable on 134 validation images. That is a
result about the design and it is what `--only matrix,gate` is for. Do not
override it by passing `--force-train` without writing down why.

EXIT CODES
----------
    0  every requested stage completed
    1  a stage failed (the traceback is printed; nothing is swallowed)
    2  a precondition failed -- missing tables, missing checkpoints, no GPU, or a
       closed gate with no --force-train
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Set BEFORE torch is imported anywhere: PYTORCH_CUDA_ALLOC_CONF is read when the
# CUDA allocator initialises, so setting it after the first `import torch` is a
# silent no-op.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

STAGES = ("miss", "fairness", "matrix", "gate", "train", "score", "report")


def hr(title: str = "") -> None:
    print(f"\n{'=' * 78}")
    if title:
        print(title)
        print("=" * 78)


def _pipeline(env, args):
    """`(cfg, man640, registry, val_manifest)` -- the shared setup every GPU stage
    needs, built once so `matrix` and `train` cannot drift into two configs.

    CFG is spelled out here rather than imported because there is no `loaders`
    constant holding it: every notebook in the study declares it inline, and the
    numbers below are handbook §3's fixed recipe. Changing one of them changes the
    experiment for every arm, which is the whole point of §3 -- do it there, not
    here.
    """
    import pandas as pd

    from bruisekit import loaders as L
    from bruisekit.registry import Registry

    man = {s: pd.read_csv(env.manifests / f"{s}.csv")
           for s in ("train", "val", "test")}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = set(man[a].subject) & set(man[b].subject)
        if overlap:
            raise RuntimeError(f"{a}/{b} share subjects: {sorted(overlap)[:5]}")

    cfg = dict(
        img_size=640, epochs=args.epochs, patience=15,
        batch_mode="per_model", effective_batch=8, max_probe_batch=64,
        vram_target=0.75,
        backbone_lr=6e-5, head_lr=6e-4, betas=(0.9, 0.999), weight_decay=0.01,
        warmup_fraction=0.01, poly_power=1.0, gradient_clip=1.0, amp=True,
        workers=0, eval_batch=8, eval_batch_cpu=2,
        segformer_alpha=0.6, efficient_alpha=0.6, aux_weight=0.4,
        smp_encoder="resnet50", smp_micro_batch=16, efficient_micro_batch=16,
        cut_min=-6.0, cut_max=6.0, cut_steps=481,
        drive_sync_every=2, n_boot=2000, n_boot_final=10_000,
    )
    man640 = L.build_cache640(env, man)
    return cfg, man640, Registry(env).scan(), man["val"]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Stage O -- miss taxonomy, distilled-arm fairness, and "
                    "ITA-group-routed gated multi-teacher KD.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--root", default=None, help="bundle root (default: this file's dir)")
    ap.add_argument("--work", default=None, help="work dir (default: <root>/_work)")
    ap.add_argument("--extra-runs", default=None,
                    help="an additional runs/ tree for the Registry to see; on ORC "
                         "that is /scratch/$USER/bruise_work/runs")
    ap.add_argument("--extra-roots", default=None,
                    help="comma-separated extra trees to scan for per-image CSVs. "
                         "Default is itakd.EXTRA_ROOTS, which already covers ORC's "
                         "scratch layout -- see handbook §10.3. There is no "
                         "--lineage: a single hard-coded directory is what broke "
                         "this on ORC on 2026-08-12.")
    ap.add_argument("--n4-results", default=None,
                    help="Stage N4 results tree, for medsam_ft "
                         "(default: <root>/STAGE_N4_RESULTS)")
    ap.add_argument("--scheme", default="light_vs_rest",
                    help="ITA group scheme: light_vs_rest (pre-registered) or five")
    ap.add_argument("--also-five", action="store_true",
                    help="run the gate on BOTH schemes, so the underpowered "
                         "five-group version is reported rather than asserted")
    ap.add_argument("--pool", default=None,
                    help="comma-separated teacher pool (default: the "
                         "pre-registered three)")
    ap.add_argument("--families", default=",".join(("segformer_b0_itakd",
                                                    "lraspp_mobilenetv3_itakd")))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--max-micro", type=int, default=16,
                    help="cap the student micro-batch; accumulation makes up the "
                         "difference so the effective batch still matches the "
                         "control exactly (default: 16)")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--reps", type=int, default=4000,
                    help="identifiability bootstrap resamples (default: 4000)")
    ap.add_argument("--force-train", action="store_true",
                    help="train even though the gate closed. Say why in the "
                         "write-up; a closed gate is a result.")
    ap.add_argument("--only", default=None, help=f"choices: {','.join(STAGES)}")
    ap.add_argument("--skip", default=None)
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    root = Path(args.root).resolve() if args.root else here
    sys.path.insert(0, str(root))

    want = set(STAGES)
    if args.only:
        want = {s.strip() for s in args.only.split(",") if s.strip()}
        bad = want - set(STAGES)
        if bad:
            print(f"unknown stage(s): {sorted(bad)}; choices {STAGES}", file=sys.stderr)
            return 2
    if args.skip:
        want -= {s.strip() for s in args.skip.split(",") if s.strip()}

    from bruisekit import itakd, paths

    t_all = time.time()
    env = paths.setup(root=root, work=args.work, extra_runs=args.extra_runs)
    print(f"root   : {env.root}")
    print(f"device : {env.device}")
    out = itakd.results_dir(env)
    print(f"results: {out}")

    pool = tuple(p.strip() for p in args.pool.split(",")) if args.pool else itakd.POOL
    families = tuple(f.strip() for f in args.families.split(",") if f.strip())
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())

    tables = key = tax = None
    schemes = [args.scheme] + (["five"] if args.also_five and args.scheme != "five" else [])

    # ── miss + fairness: no GPU, no checkpoints ──────────────────────────────
    if want & {"miss", "fairness"}:
        hr("DISCOVERING PER-IMAGE TABLES ON THIS MACHINE")
        extra = (tuple(r.strip() for r in args.extra_roots.split(",") if r.strip())
                 if args.extra_roots else itakd.EXTRA_ROOTS)
        tables, key, _found = itakd.load_tables(env, extra_roots=extra,
                                                models=itakd.all_models())

    if "miss" in want:
        hr("STAGE 1/7  miss taxonomy   (TODO 1)")
        tax = itakd.miss_taxonomy(tables)
        itakd.print_miss_taxonomy(tax)
        for name, obj in (("miss_taxonomy", tax),
                          ("miss_taxonomy_by_ita",
                           itakd.miss_taxonomy_by(tables, "skin_tone_category")),
                          ("miss_taxonomy_by_size",
                           itakd.miss_taxonomy_by(tables, "size", key))):
            print(f"  wrote {itakd.save(env, name, obj)}")

    if "fairness" in want:
        hr("STAGE 2/7  skin tone x lesion size for the KD arms   (TODO 3)")
        if tax is None:
            tax = itakd.miss_taxonomy(tables)
        best = itakd.best_distilled(tax, tables, n=6)
        print(f"leading distilled arms (mean Dice, misses as tiebreak): {best}\n")
        fr = itakd.distilled_fairness(tables, key, models=None, n_boot=args.n_boot)
        for name, obj in fr.items():
            print(f"  wrote {itakd.save(env, f'fairness__{name}', obj)}")
        print(f"  wrote {itakd.save(env, 'leading_distilled_arms', {'models': best})}")

    # ── the gate and the arm ─────────────────────────────────────────────────
    matrix = None
    mat_cache = itakd.results_dir(env, "tables") / "val_pool_matrix.csv"

    if "matrix" in want:
        hr("STAGE 3/7  scoring the teacher pool on validation")
        cfg, man640, reg, val_meta = _pipeline(env, args)
        matrix = itakd.val_group_matrix(env, reg, cfg, man640, val_meta, pool=pool,
                                        n4_root=args.n4_results, cache=mat_cache)
        print(f"  matrix: {matrix.shape[0]} images x {len(pool)} teachers")

    if "gate" in want:
        hr("STAGE 4/7  the pre-registered gate   (validation only)")
        if matrix is None:
            if not mat_cache.exists():
                print(f"PRECONDITION: {mat_cache} is absent. Run --only matrix "
                      f"first (it needs the teacher checkpoints).", file=sys.stderr)
                return 2
            import pandas as pd
            matrix = pd.read_csv(mat_cache)
        for scheme in schemes:
            res = itakd.ita_group_gate(matrix, pool=pool, scheme=scheme,
                                        reps=args.reps)
            print(itakd.format_gate(res))
            for p in itakd.save_gate(env, res):
                print(f"  wrote {p}")

    if "train" in want:
        hr("STAGE 5/7  training the ITA-grouped arms")
        gate_file = (itakd.results_dir(env, "tables")
                     / f"ita_group_gate__{args.scheme}.json")
        if not gate_file.exists():
            print(f"PRECONDITION: {gate_file} is absent. The gate is val-only and "
                  f"must be on disk before training. Run --only matrix,gate.",
                  file=sys.stderr)
            return 2
        res = json.loads(gate_file.read_text())
        if not res["GATE_any"] and not args.force_train:
            print(itakd.format_gate(res))
            print("\nPRECONDITION: the gate is CLOSED. That is a result -- report "
                  "it. Pass --force-train and write down why if you disagree.",
                  file=sys.stderr)
            return 2
        if not str(env.device).startswith("cuda"):
            print(f"PRECONDITION: training needs CUDA, device is {env.device}.",
                  file=sys.stderr)
            return 2

        import numpy as np
        import pandas as pd

        cfg, man640, reg, _ = _pipeline(env, args)
        W = itakd.weight_array(pd.DataFrame(res["weights"]), pool, args.scheme)
        itakd.install_group_shim(itakd.build_group_map(man640, args.scheme))
        itakd.install_teacher_shim(env, reg, cfg, man640, pool=pool,
                                   n4_root=args.n4_results)
        itakd.install_loss_shim(W)
        print(f"\nweights [{W.shape[0]} groups x {W.shape[1]} teachers]:\n"
              f"{np.round(W, 3)}\n")

        runs = itakd.train_arms(env, reg, cfg, man640,
                                itakd.results_dir(env, "runs"),
                                families=families, seeds=seeds,
                                max_micro=args.max_micro)
        print(f"\ntrained: {runs}")

    if want & {"score", "report"}:
        hr("STAGES 6-7  scoring and reporting the trained arms")
        print("Not implemented as a shell stage on purpose: scoring and the paired\n"
              "contrasts reuse `lesionsize.contrast_table` and `significance`, and\n"
              "both want the per-image CSVs of the arms AND their controls in one\n"
              "lineage directory. Run §8-§10 of bruise_stage_o.ipynb, which does\n"
              "that with the run directory in front of you.\n"
              "\n"
              "What to export from the runs:\n"
              "  STAGE_O_RESULTS/runs/<family>__seed<k>/  best.pt, config.json,\n"
              "  group_loss_stats.json  <- read images_per_group before anything else;\n"
              "  an arm whose loss never saw one of the groups did not run the\n"
              "  experiment the gate authorised.")

    hr(f"DONE in {time.time() - t_all:.0f}s   ->  {out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        raise SystemExit(130)
