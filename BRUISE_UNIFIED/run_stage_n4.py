#!/usr/bin/env python
"""Stage N4 end to end from a shell. No Jupyter, no notebook, no display.

    python run_stage_n4.py --root . --work ./_work

Does exactly what `bruise_stage_n4.ipynb` does, in the same order and through the
same functions, so the two cannot drift into producing different numbers:

    check     encoders present, self-test, config preflight
    build     construct each arm and assert it is really unfrozen  <- the guard
    train     engine.train_run, unmodified, resumable
    val       fit the operating point on val, score val
    gate      the pre-registered verdict            <- written BEFORE test is read
    test      score test at the cut already fitted on val
    fairness  size-stratified + ITA-conditioned tables

Every stage is skippable and every stage is resumable. Re-running after an
interruption picks up where it stopped: `engine.train_run` honours `DONE.json`
and `resume.pt`, and the scoring stages re-read `best.pt`.

WHY THE ORDER IS FIXED
----------------------
The gate is VAL-ONLY by design (handbook 7f.4): a decision taken on test is a
decision taken on the data the paper reports. `gate` therefore writes its verdict
to disk before `test` runs, and `--only test` REFUSES to run if the gate has not
been written yet. That refusal is the point -- it makes the ordering a property of
the code rather than of whoever ran it.

EXIT CODES
----------
    0  every requested stage completed
    1  a stage failed (the traceback is printed; nothing is swallowed)
    2  a precondition failed -- missing encoder, missing data, no GPU for train
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Set BEFORE torch is imported anywhere. PYTORCH_CUDA_ALLOC_CONF is read when the
# CUDA allocator initialises; setting it after the first `import torch` is a
# silent no-op, which is why the notebook's first cell asserts on it.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

STAGES = ("check", "build", "train", "val", "gate", "test", "fairness")


def hr(title: str = "") -> None:
    print(f"\n{'=' * 78}")
    if title:
        print(title)
        print("=" * 78)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Stage N4 -- SAM vs MedSAM, mask-supervised foundation encoders.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("--root", default=None,
                    help="bundle root (default: this script's directory)")
    ap.add_argument("--work", default=None,
                    help="work dir for the 640 cache and outputs "
                         "(default: <root>/_work). Put this on fast local disk.")
    ap.add_argument("--extra-runs", default=None,
                    help="an additional runs/ tree for the Registry to see")
    ap.add_argument("--arms", default="sam_ft,medsam_ft",
                    help="comma-separated (default: both)")
    ap.add_argument("--seeds", default="0",
                    help="comma-separated (default: 0 -- the pre-registered "
                         "screening run)")
    ap.add_argument("--epochs", type=int, default=100,
                    help="cap; the engine stops early on patience (default: 100)")
    ap.add_argument("--micro-batch", type=int, default=2,
                    help="FIXED, never the VRAM probe. Six unfrozen SAM blocks at "
                         "640 with windowed attention is a memory profile the "
                         "probe was not calibrated on (default: 2)")
    ap.add_argument("--n-boot", type=int, default=10000,
                    help="bootstrap resamples (default: 10000)")
    ap.add_argument("--only", default=None,
                    help=f"run only these stages, comma-separated. "
                         f"Choices: {','.join(STAGES)}")
    ap.add_argument("--skip", default=None,
                    help="skip these stages, comma-separated")
    ap.add_argument("--results", default=None,
                    help="results tree (default: <root>/STAGE_N4_RESULTS)")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    root = Path(args.root).resolve() if args.root else here
    sys.path.insert(0, str(root))

    want = set(STAGES)
    if args.only:
        want = {s.strip() for s in args.only.split(",") if s.strip()}
    if args.skip:
        want -= {s.strip() for s in args.skip.split(",") if s.strip()}
    bad = want - set(STAGES)
    if bad:
        print(f"unknown stage(s): {sorted(bad)}. Choices: {list(STAGES)}",
              file=sys.stderr)
        return 2

    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())

    import json
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)
    import pandas as pd
    import torch

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 80)

    from bruisekit import loaders as L
    from bruisekit import paths as P
    from bruisekit import report as RP
    from bruisekit import samprobe as N4

    t0 = time.time()
    hr("STAGE N4 -- SAM vs MedSAM")
    print(f"root   : {root}")
    print(f"arms   : {arms}   seeds: {seeds}")
    print(f"stages : {[s for s in STAGES if s in want]}")

    env = P.setup(root=root, work=args.work, extra_runs=args.extra_runs)
    print(f"\n{env.describe()}")

    RESULTS = Path(args.results).resolve() if args.results else N4.results_dir(env)
    RUNS, TABLES = RESULTS / "runs", RESULTS / "tables"
    for d in (RESULTS, RUNS, TABLES):
        d.mkdir(parents=True, exist_ok=True)
    print(f"\nresults: {RESULTS}")
    print("         nothing is written to results/, FINAL_RESULT/,")
    print("         FOUNDATION_RESULTS/, DERM_PROBE_RESULTS/, STAGE_N3_RESULTS/")

    # ── check ────────────────────────────────────────────────────────────────
    if "check" in want:
        hr("CHECK -- encoders, self-test, preflight")
        src = N4.report_sources(env)
        print(src[["encoder", "repo", "present", "size_MB"]].to_string(index=False))
        missing = src.loc[~src.present, "encoder"].tolist()
        if missing:
            print(f"\nMISSING (and required): {missing}\n")
            print(N4.download_instructions(env))
            return 2
        print("\n-- self test --")
        if not N4.self_test(verbose=True):
            print("samprobe self-test FAILED -- do not train on this",
                  file=sys.stderr)
            return 2

    # ── shared setup for everything below ────────────────────────────────────
    from bruisekit.kd import kd_core

    man = {s: pd.read_csv(env.manifests / f"{s}.csv")
           for s in ("train", "val", "test")}
    META = {"val": man["val"], "test": man["test"]}

    CFG = {
        **kd_core.DEFAULTS,
        "epochs": args.epochs,
        "alpha": 0.5,            # unused: these arms are supervised, not distilled
        "aux_weight": 0.4,       # unused: ViT arms return aux=None
        "drive_sync_every": 5,
        "eval_batch": 8,
        "micro_batch": args.micro_batch,
        "max_probe_batch": args.micro_batch,
    }
    required = ("img_size", "amp", "workers", "backbone_lr", "head_lr",
                "weight_decay", "betas", "epochs", "warmup_fraction", "alpha",
                "aux_weight", "drive_sync_every", "patience")
    absent = [k for k in required if k not in CFG]
    if absent:
        print(f"CFG is missing keys engine.train_run reads: {absent}",
              file=sys.stderr)
        return 2
    # SAM is patch-16 and this stage runs at grid 40, so 40*16 = 640, the
    # pipeline's own img_size. A drift here silently reintroduces the resolution
    # confound §7i.2 removed.
    if CFG["img_size"] != N4.TARGET_GRID * 16:
        print(f"img_size {CFG['img_size']} != TARGET_GRID*16 = "
              f"{N4.TARGET_GRID * 16}", file=sys.stderr)
        return 2

    need_cache = want & {"train", "val", "test"}
    man640 = L.build_cache640(env, man) if need_cache else None
    if need_cache:
        print(f"\n640 cache ready: train {len(man640['train'])} / "
              f"val {len(man640['val'])} / test {len(man640['test'])}")

    # ── build ────────────────────────────────────────────────────────────────
    if "build" in want:
        hr("BUILD -- construct each arm and verify it is really unfrozen")
        print("This is the guard. Two failures would produce a plausible number")
        print("and no symptom: an arm that unfreezes nothing, and a position")
        print("embedding that did not resample. Both are asserted below.\n")
        N4.install_n4_shim(env)
        built = {}
        for arm in arms:
            print(f"\n{'-' * 72}\n{arm}  ({N4.ARMS[arm]['corpus']})\n{'-' * 72}")
            m = N4.build_arm(env, arm, verbose=True)
            info = m.n4_info
            assert info["encoder_trainable_params"] > 0, f"{arm}: nothing unfrozen"
            assert info["encoder_trainable_fraction"] > 0.1, (
                f"{arm}: only {100 * info['encoder_trainable_fraction']:.1f} % "
                f"trainable -- this arm is effectively a frozen probe and would "
                f"fake a result")
            assert info["pos_embed_resampled_to"] == N4.TARGET_GRID
            n_head = sum(p.numel() for p in m.decode_head.parameters())
            n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
            with torch.no_grad():
                y = m(torch.zeros(1, 3, CFG["img_size"], CFG["img_size"]))
            assert y.shape[-2:] == (CFG["img_size"], CFG["img_size"])
            print(f"    decoder {n_head:,}   trainable total {n_train:,}")
            print(f"    forward: {tuple(y.shape)}")
            built[arm] = {**info, "decoder_params": n_head,
                          "trainable_params": n_train}
            del m
            if str(env.device).startswith("cuda"):
                torch.cuda.empty_cache()
        pd.DataFrame(built).T.to_csv(TABLES / "arm_build_info.csv")
        print(f"\nwritten -> {TABLES / 'arm_build_info.csv'}")

    # ── train ────────────────────────────────────────────────────────────────
    if "train" in want:
        hr("TRAIN -- engine.train_run, unmodified")
        if not str(env.device).startswith("cuda"):
            print(f"device is {env.device}. Six unfrozen ViT blocks at 640 needs "
                  f"a GPU; on CPU this would not finish.", file=sys.stderr)
            return 2
        print("~2-2.5 GPU-h per arm. Resumable: DONE.json skips, resume.pt "
              "continues.\n")
        run_ids = N4.train_arms(env, CFG, man640, RUNS, arms=arms, seeds=seeds,
                                verbose=True)
        print(f"\nruns: {run_ids}")

    # ── val ──────────────────────────────────────────────────────────────────
    from bruisekit.foundation import fit_operating_point, score_split

    def cuts_path() -> Path:
        return TABLES / "operating_points.json"

    val_tables, summaries = {}, {}
    if "val" in want:
        hr("VAL -- fit the operating point on val, score val")
        print("The cut is fitted on val and applied to val, matching how every")
        print("other arm in this study is scored.\n")
        cuts = {}
        for arm in arms:
            for seed in seeds:
                run_id = f"{arm}__seed{seed}"
                run_dir = RUNS / run_id
                if not (run_dir / "best.pt").exists():
                    print(f"  {run_id}: no best.pt -- train it first", file=sys.stderr)
                    return 2
                model = N4.load_trained(env, arm, run_dir).to(env.device)
                op = fit_operating_point(model, env, CFG, man640, run_dir)
                cut = float(op["cut"] if isinstance(op, dict) else op)
                cuts[arm] = cut
                df = score_split(model, env, CFG, man640, "val", cut, META["val"])
                df["arm"], df["seed"], df["cut"], df["split"] = arm, seed, cut, "val"
                val_tables[arm] = df
                summaries[run_id] = {
                    "split": "val", "cut": cut, "n": int(len(df)),
                    "mean_dice": float(df.dice.mean()),
                    "median_dice": float(df.dice.median()),
                    "misses": int((df.dice == 0).sum()),
                    "mean_recall": float(df.recall.mean())}
                df.to_csv(TABLES / f"val_per_image__{run_id}.csv", index=False)
                print(f"  {run_id}: cut={cut:+.3f}  dice={df.dice.mean():.4f}  "
                      f"median={df.dice.median():.4f}  "
                      f"misses={(df.dice == 0).sum()}")
                del model
                if str(env.device).startswith("cuda"):
                    torch.cuda.empty_cache()
        cuts_path().write_text(json.dumps(cuts, indent=2), encoding="utf-8")
        (TABLES / "val_summaries.json").write_text(
            json.dumps(summaries, indent=2, default=str), encoding="utf-8")

    def load_split(split: str) -> dict:
        """This stage's own per-image tables for `split`, keyed by ARM name."""
        out = {}
        for p in sorted(TABLES.glob(f"{split}_per_image__*.csv")):
            out[p.stem.split("__")[1]] = pd.read_csv(p)
        return out

    def load_n3(split: str) -> dict:
        """Stage N3's tables, normalised and keyed by ARM. Read, never rewritten.

        Globs the directory root AND tables/, because the two Stage N3 runs wrote
        to different places. Keyed by arm, not run_id -- the gate raises on a
        run_id key, deliberately (handbook 7i.7a).
        """
        out, base = {}, root / "STAGE_N3_RESULTS"
        for d in (base, base / "tables"):
            if not d.exists():
                continue
            for p in sorted(d.glob(f"{split}_per_image__*.csv")):
                arm = p.stem.split("__")[1]
                if arm not in out:
                    out[arm] = RP.normalize(pd.read_csv(p), META[split])
        return out

    # ── gate ─────────────────────────────────────────────────────────────────
    gate_json = RESULTS / "tables" / "mask_supervision_gate.json"
    if "gate" in want:
        hr("GATE -- the pre-registered verdict")
        tables = val_tables or load_split("val")
        if not tables:
            print("no val tables -- run the val stage first", file=sys.stderr)
            return 2
        n3 = load_n3("val")
        for a, t in n3.items():
            print(f"  reference  N3 {a:<12} dice={t.dice.mean():.4f}  "
                  f"misses={(t.dice == 0).sum()}")
        gate = N4.mask_supervision_gate({**tables, **n3}, n_boot=args.n_boot,
                                        seed=0)
        print()
        N4.print_gate(gate)
        for p in N4.save_gate(env, gate):
            print(f"  written -> {p}")

    # ── test ─────────────────────────────────────────────────────────────────
    test_tables = {}
    if "test" in want:
        hr("TEST -- score at the cut ALREADY fitted on val")
        if not gate_json.exists():
            print("the gate has not been written. The verdict is val-only by "
                  "design and must be on disk before test is read (handbook "
                  "7f.4). Run the gate stage first.", file=sys.stderr)
            return 2
        cuts = json.loads(cuts_path().read_text()) if cuts_path().exists() else {}
        print("The cut is READ, never re-fitted. Re-sweeping on test would fit "
              "the threshold\non the data it is scored on.\n")
        for arm in arms:
            for seed in seeds:
                run_id = f"{arm}__seed{seed}"
                if arm not in cuts:
                    print(f"  {run_id}: no fitted cut -- run the val stage first",
                          file=sys.stderr)
                    return 2
                cut = float(cuts[arm])
                model = N4.load_trained(env, arm, RUNS / run_id).to(env.device)
                df = score_split(model, env, CFG, man640, "test", cut, META["test"])
                df["arm"], df["seed"], df["cut"], df["split"] = arm, seed, cut, "test"
                test_tables[arm] = df
                summaries[run_id + "__test"] = {
                    "split": "test", "cut": cut, "n": int(len(df)),
                    "mean_dice": float(df.dice.mean()),
                    "median_dice": float(df.dice.median()),
                    "misses": int((df.dice == 0).sum()),
                    "mean_recall": float(df.recall.mean())}
                df.to_csv(TABLES / f"test_per_image__{run_id}.csv", index=False)
                print(f"  {run_id}: cut={cut:+.3f} (from val)  "
                      f"dice={df.dice.mean():.4f}  "
                      f"median={df.dice.median():.4f}  "
                      f"misses={(df.dice == 0).sum()}")
                del model
                if str(env.device).startswith("cuda"):
                    torch.cuda.empty_cache()
        (TABLES / "test_summaries.json").write_text(
            json.dumps(summaries, indent=2, default=str), encoding="utf-8")

    # ── fairness ─────────────────────────────────────────────────────────────
    if "fairness" in want:
        hr("FAIRNESS -- size-stratified and ITA-conditioned")
        print("Read this BEFORE the Dice table. Dice is saturated here; complete")
        print("misses on small bruises are the endpoint that has ever moved.\n")
        tables = test_tables or {k: RP.normalize(v, META["test"])
                                 for k, v in load_split("test").items()}
        if not tables:
            print("no test tables -- run the test stage first", file=sys.stderr)
            return 2
        tables = {**tables, **load_n3("test")}
        print(f"over: {sorted(tables)}\n")
        fair = N4.fairness_report(tables, n_boot=args.n_boot, seed=0, verbose=True)
        print()
        N4.print_fairness(fair)
        for p in N4.save_fairness(env, fair):
            print(f"  written -> {p}")

    hr(f"DONE in {(time.time() - t0) / 60:.1f} min")
    print(f"everything is under {RESULTS}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted -- re-run the same command to resume", file=sys.stderr)
        raise SystemExit(1)
