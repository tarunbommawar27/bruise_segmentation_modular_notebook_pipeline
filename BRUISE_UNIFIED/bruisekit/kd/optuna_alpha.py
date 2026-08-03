#!/usr/bin/env python3
"""
optuna_alpha.py — search the distillation ratio alpha on VALIDATION.
====================================================================
The KD fuse weight alpha (L = alpha*sup + (1-alpha)*soft) is the one knob that
most changes a distilled student. This searches it on val (never test), with
short runs, and writes best_alpha.json for the full runs to consume.

Uses Optuna (TPE) if installed, else a deterministic grid. Objective = best
VALIDATION mean Dice from a reduced-epoch run of distill_segformer with an
otherwise-identical config. Selection metric is val Dice (matches the project's
epoch selection); the full run then trains at the chosen alpha for full epochs.

    python optuna_alpha.py --tag expA_b5_to_b0_response \
        --search-epochs 25 --n-trials 8 --alpha-lo 0.3 --alpha-hi 0.95 \
        -- <all the distill_segformer.py args except --alpha/--run-id/--epochs>

Everything after `--` is forwarded verbatim to distill_segformer (teachers,
manifests, kd type, ensemble, etc.), so the search uses the SAME recipe as the
full run. Writes <out-dir>/optuna_alpha/<tag>_best_alpha.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

import distill_segformer as D


def build_ns(forward_args, alpha, run_id, epochs, patience):
    """Parse the forwarded distill_segformer args, then override alpha/run-id/epochs."""
    ns = D.parse_args_from(forward_args)
    ns.alpha = alpha
    ns.run_id = run_id
    ns.epochs = epochs
    ns.patience = patience
    ns.force = False
    return ns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--search-epochs", type=int, default=25)
    ap.add_argument("--search-patience", type=int, default=6)
    ap.add_argument("--n-trials", type=int, default=8)
    ap.add_argument("--alpha-lo", type=float, default=0.3)
    ap.add_argument("--alpha-hi", type=float, default=0.95)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("forward", nargs=argparse.REMAINDER,
                    help="everything after -- is passed to distill_segformer")
    a = ap.parse_args()
    fwd = a.forward[1:] if a.forward and a.forward[0] == "--" else a.forward
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    search_dir = Path(a.out_dir) / "optuna_alpha"
    search_dir.mkdir(parents=True, exist_ok=True)
    best_json = search_dir / f"{a.tag}_best_alpha.json"
    if best_json.exists():
        print(f"[skip] {best_json} exists: {json.loads(best_json.read_text())}")
        return

    trials = []

    def objective_alpha(alpha, i):
        run_id = f"_alpha_search/{a.tag}_a{alpha:.3f}"
        ns = build_ns(fwd, alpha, run_id, a.search_epochs, a.search_patience)
        # keep search runs out of the main out tree's finished-arm space
        ns.out_dir = str(search_dir)
        res = D.train_distill(ns, device)
        v = float(res["best_val_dice"])
        trials.append({"alpha": alpha, "val_mean_dice": v, "run_id": run_id})
        print(f"  trial {i}: alpha={alpha:.3f} -> val_dice={v:.4f}")
        return v

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=0))
        study.optimize(lambda t: objective_alpha(t.suggest_float("alpha", a.alpha_lo, a.alpha_hi),
                                                  t.number), n_trials=a.n_trials)
        best_alpha = float(study.best_params["alpha"]); best_val = float(study.best_value)
        method = "optuna_tpe"
    except ImportError:
        import numpy as np
        grid = np.linspace(a.alpha_lo, a.alpha_hi, a.n_trials)
        vals = [objective_alpha(float(al), i) for i, al in enumerate(grid)]
        k = int(np.argmax(vals)); best_alpha = float(grid[k]); best_val = float(vals[k])
        method = "grid"

    best_json.write_text(json.dumps({
        "tag": a.tag, "best_alpha": best_alpha, "best_val_mean_dice": best_val,
        "method": method, "search_epochs": a.search_epochs, "trials": trials}, indent=2))
    print(f"[done] {a.tag}: best alpha={best_alpha:.3f} (val_dice={best_val:.4f}) -> {best_json}")


if __name__ == "__main__":
    main()
