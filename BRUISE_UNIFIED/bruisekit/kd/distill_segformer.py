#!/usr/bin/env python3
"""
distill_segformer.py — unified SegFormer student distillation trainer.
======================================================================
One trainer that runs every SegFormer-student experiment in the plan by flags,
sharing the exact baseline recipe from kd_core (so results sit in one table):

  Experiment A  (single-teacher standard KD):
      --teacher-a B5_run --pretrained-a mit_b5   --kd response
  Experiment B  (uniform B2+B5 ensemble KD):
      --teacher-a B2_run --teacher-b B5_run ... --ensemble uniform --kd response
  Experiment C  (adaptive B2+B5 ensemble KD):
      ... --ensemble adaptive --kd response
  SOTA arms (single-teacher, orthogonal to the ensemble question):
      --kd cwd    (response + channel-wise feature KD)
      --kd dkd    (decoupled KD on logits)
  Step-4 add-ons (ablate ONE at a time):
      --group   (worst-ITA-group supervised term, needs --train-ita)
      --hard    (per-pixel miss/small/uncertain weighting)

Selection is ALWAYS on validation: best epoch = val mean Dice @0.50; decision
threshold swept on val; test scored ONCE. Fairness-by-ITA written per run.

Teacher probabilities are temperature-calibrated (each teacher dir must contain
temperature.json; run calibrate_teacher.py first). Adaptive fusion and hard
weighting use ONLY teacher outputs + GT — no ITA, no test info (leak-free).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import kd_core as K


def build_train_group_lookup(train_ita_csv):
    if not train_ita_csv:
        return None
    df = pd.read_csv(train_ita_csv)
    col = "ita_group_index_5" if "ita_group_index_5" in df.columns else None
    if col is None:
        return None
    return dict(zip(df["stem"].astype(str), df[col].astype(int)))


def make_cwd_adapters(student_feats, teacher_feats, device):
    adapters = nn.ModuleList()
    for fs, ft in zip(student_feats, teacher_feats):
        adapters.append(nn.Conv2d(fs.shape[1], ft.shape[1], kernel_size=1).to(device))
    return adapters


def train_distill(args, device):
    cfg = dict(K.DEFAULTS)
    cfg.update(img_size=args.img_size, epochs=args.epochs, patience=args.patience,
               workers=args.workers, amp=not args.no_amp,
               effective_batch=args.effective_batch)
    run_dir = Path(args.out_dir) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    done = run_dir / "DONE.json"
    if done.exists() and not args.force:
        print(f"[skip] {args.run_id} already done"); return json.loads(done.read_text())

    K.seed_everything(args.seed)
    amp = cfg["amp"]

    # --- manifests ---
    full_train = K.load_manifest(args.train_manifest, args.data_root)
    test_df = K.load_manifest(args.test_manifest, args.data_root)
    train_df, val_df = K.resolve_train_val(full_train, cfg)
    for col in ("subject", "stem"):
        leak = (set(train_df[col]) | set(val_df[col])) & set(test_df[col])
        if leak:
            print(f"!! WARNING: {col} train/val vs TEST overlap: {sorted(leak)[:5]}...")

    group_lut = build_train_group_lookup(args.train_ita) if args.group else None
    if args.group and group_lut is None:
        raise SystemExit("--group requires --train-ita with an ita_group_index_5 column")

    # --- student ---
    student = K.SegformerWrapper(K.build_segformer(args.student_pretrained, num_labels=1)).to(device)
    if args.grad_checkpoint:
        student.gradient_checkpointing_enable()

    # --- teachers ---
    teacher_a = K.load_teacher(args.teacher_a, args.pretrained_a, device, amp)
    teacher_b = (K.load_teacher(args.teacher_b, args.pretrained_b, device, amp)
                 if args.teacher_b else None)
    if args.ensemble != "none" and teacher_b is None:
        raise SystemExit("--ensemble needs --teacher-b/--pretrained-b")

    # --- feature adapters (CWD and angular both need student->teacher channel maps) ---
    feat_adapters = None
    if args.kd in ("cwd", "angular"):
        with torch.no_grad():
            probe = torch.randn(1, 3, cfg["img_size"], cfg["img_size"], device=device)
            _, s_feats = student(probe, return_hidden=True)
            t_feats = teacher_a.hidden(probe)
        feat_adapters = make_cwd_adapters(s_feats, t_feats, device)
        print(f"  {args.kd} adapters: {[a.in_channels for a in feat_adapters]} -> "
              f"{[a.out_channels for a in feat_adapters]}")

    # --- optimiser (student params + any adapter params) ---
    param_groups = K.build_param_groups(student, cfg["backbone_lr"], cfg["head_lr"], cfg["weight_decay"])
    if feat_adapters is not None:
        param_groups.append({"params": list(feat_adapters.parameters()),
                             "lr": cfg["head_lr"], "weight_decay": cfg["weight_decay"]})
    all_params = [p for g in param_groups for p in g["params"]]

    # --- resume checkpoint (written every epoch; DONE.json only on completion) ---
    resume_path = run_dir / "resume_checkpoint.pt"

    # --- VRAM probe (skipped on resume — reuse the batch size found before the kill) ---
    teacher_fns = [teacher_a] + ([teacher_b] if teacher_b else [])
    # CWD/angular ALSO extract full hidden states from student AND teacher (much
    # heavier), so the probe must exercise that path + use extra headroom, or the
    # response-sized micro_batch OOMs the moment the first hidden-state loss runs.
    if args.kd in ("cwd", "angular"):
        probe_fwd = lambda x: student(x, return_hidden=True)[0]
        probe_teachers = teacher_fns + [teacher_a.hidden]
        probe_target = min(cfg["vram_target"], 0.60)
    else:
        probe_fwd = lambda x: student(x)
        probe_teachers = teacher_fns
        probe_target = cfg["vram_target"]
    if resume_path.exists():
        _st = torch.load(str(resume_path), map_location="cpu", weights_only=False)
        micro, accum, vram_frac = _st["micro"], _st["accum"], _st.get("vram_frac", 0.0)
        del _st
        print(f"  [resume] reusing micro_batch={micro} accum={accum}")
    else:
        micro, accum, vram_frac = K.find_optimal_micro_batch(
            probe_fwd, all_params, cfg["img_size"], device,
            cfg["effective_batch"], probe_target, amp, cfg["max_probe_batch"],
            teacher_fns=probe_teachers)
    print(f"  micro_batch={micro} accum={accum} effective={micro*accum} vram_frac={vram_frac:.3f}")

    train_loader = K.make_loader(train_df, cfg["img_size"], micro, True, cfg["workers"], args.seed)
    val_loader = K.make_loader(val_df, cfg["img_size"], micro, False, cfg["workers"], args.seed)

    optimizer = torch.optim.AdamW(param_groups, betas=tuple(cfg["betas"]))
    peak_lrs = [g["lr"] for g in optimizer.param_groups]
    scaler = torch.amp.GradScaler("cuda") if amp else None
    steps_per_epoch = max(1, len(train_loader) // accum)
    total_steps = steps_per_epoch * cfg["epochs"]
    warmup_steps = max(1, int(total_steps * cfg["warmup_fraction"]))

    response = K.ResponseKD(alpha=args.alpha)
    (run_dir / "run_config.json").write_text(json.dumps({
        "run_id": args.run_id, "student": "segformer_b0", "seed": args.seed,
        "teacher_a": str(args.teacher_a), "teacher_b": str(args.teacher_b),
        "ensemble": args.ensemble, "kd": args.kd, "group": bool(args.group),
        "hard": bool(args.hard), "boundary": bool(args.boundary), "rel_b": args.rel_b,
        "alpha": args.alpha, "lambda_cwd": args.lambda_cwd, "lambda_angular": args.lambda_angular,
        "lambda_boundary": args.lambda_boundary, "lambda_group": args.lambda_group,
        "gamma_hard": args.gamma_hard, "micro_batch": micro, "accum": accum,
        "n_train": len(train_df), "n_val": len(val_df),
    }, indent=2))

    # --- restore full training state on resume ---
    start_epoch, best_dice, patience, gstep, history = 1, float("-inf"), 0, 0, []
    if resume_path.exists():
        st = torch.load(str(resume_path), map_location=device, weights_only=False)
        student.load_state_dict(st["model"])
        optimizer.load_state_dict(st["optimizer"])
        if scaler is not None and st.get("scaler"):
            scaler.load_state_dict(st["scaler"])
        if feat_adapters is not None and st.get("adapters"):
            feat_adapters.load_state_dict(st["adapters"])
        start_epoch = st["epoch"] + 1
        best_dice, patience, gstep, history = st["best_dice"], st["patience"], st["gstep"], st["history"]
        print(f"  [resume] {args.run_id} from epoch {start_epoch} (best_val_dice={best_dice:.4f})")
        del st

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        student.train(); optimizer.zero_grad(set_to_none=True)
        running, t0 = 0.0, time.time()
        for step, (x, y, stems) in enumerate(train_loader):
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp):
                # teacher targets (reliability-gated ensemble; rel_b from val)
                p_a = teacher_a(x)
                p_b = teacher_b(x) if teacher_b else None
                t_prob = (K.fuse_teachers(p_a, p_b, args.ensemble, rel_b=args.rel_b)
                          if teacher_b else p_a)

                # per-pixel weight: hard-example (miss/small) x boundary (edge focus)
                weight = None
                if args.hard:
                    weight = K.hard_example_weight(t_prob, y, gamma=args.gamma_hard)
                if args.boundary:
                    bw = K.boundary_weight_map(y, beta=args.lambda_boundary)
                    weight = bw if weight is None else weight * bw

                need_hidden = (args.kd in ("cwd", "angular"))
                if need_hidden:
                    s_logits, s_feats = student(x, return_hidden=True)
                else:
                    s_logits = student(x)

                if args.kd == "dkd":
                    t_logits = teacher_a.raw_logits(x)
                    loss = args.alpha * response.sup(s_logits, y) + \
                        (1 - args.alpha) * K.dkd_loss(s_logits, t_logits, y,
                                                      T=args.dkd_T, beta=args.dkd_beta)
                elif args.kd == "bpkd":
                    loss = args.alpha * response.sup(s_logits, y) + \
                        (1 - args.alpha) * K.bpkd_loss(s_logits, t_prob, y,
                                                       lam_edge=args.bpkd_lam_edge)
                else:
                    loss = response(s_logits, y, t_prob, weight)
                    if args.kd == "cwd":
                        loss = loss + args.lambda_cwd * K.cwd_loss(s_feats, teacher_a.hidden(x), feat_adapters)
                    elif args.kd == "angular":
                        loss = loss + args.lambda_angular * K.angular_distillation(s_feats, teacher_a.hidden(x), feat_adapters)

                if args.group:
                    gidx = torch.tensor([group_lut.get(s, 0) for s in stems],
                                        device=device, dtype=torch.long)
                    loss = loss + args.lambda_group * K.group_worst_loss(s_logits, y, gidx)

                loss = loss / accum
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            running += loss.item() * accum
            if (step + 1) % accum == 0 or (step + 1) == len(train_loader):
                gstep += 1
                mult = K.lr_multiplier(gstep, total_steps, warmup_steps, cfg["poly_power"])
                for g, peak in zip(optimizer.param_groups, peak_lrs):
                    g["lr"] = peak * mult
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(all_params, cfg["gradient_clip"])
                    scaler.step(optimizer); scaler.update()
                else:
                    nn.utils.clip_grad_norm_(all_params, cfg["gradient_clip"])
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        _, val_summary = K.evaluate(student, val_loader, device, 0.50, amp)
        val_dice = val_summary["mean_dice"]
        history.append({"epoch": epoch, "train_loss": round(running / max(1, len(train_loader)), 6),
                        **val_summary, "sec": round(time.time() - t0, 1)})
        pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)
        if val_dice > best_dice:
            best_dice, patience = val_dice, 0
            K.atomic_save(student.state_dict(), run_dir / "best_model.pt"); flag = " *"
        else:
            patience += 1; flag = ""
        print(f"  {args.run_id} e{epoch:3d} loss={history[-1]['train_loss']:.4f} "
              f"val_dice={val_dice:.4f} {time.time()-t0:.0f}s{flag}")
        # resume checkpoint every epoch (full state) — a wall-time kill loses <=1 epoch
        K.atomic_save({"epoch": epoch, "model": student.state_dict(),
                       "optimizer": optimizer.state_dict(),
                       "scaler": scaler.state_dict() if scaler is not None else None,
                       "adapters": feat_adapters.state_dict() if feat_adapters is not None else None,
                       "best_dice": best_dice, "patience": patience, "gstep": gstep,
                       "history": history, "micro": micro, "accum": accum,
                       "vram_frac": vram_frac}, resume_path)
        if patience >= cfg["patience"]:
            print(f"  early stop at epoch {epoch}"); break

    # --- training finished (all epochs done OR early stop). Only NOW do we score
    #     and mark DONE — an interrupted run never reaches here, so it has no
    #     DONE.json/test_per_image.csv and will resume next launch. ---
    # --- val threshold sweep, score ONCE on test ---
    student.load_state_dict(torch.load(str(run_dir / "best_model.pt"),
                                       map_location=device, weights_only=True))
    val_loader8 = K.make_loader(val_df, cfg["img_size"], 8, False, cfg["workers"], args.seed)
    test_loader8 = K.make_loader(test_df, cfg["img_size"], 8, False, cfg["workers"], args.seed)
    thr_df, best_thr = K.threshold_sweep(student, val_loader8, device, cfg["thresholds"], amp)
    thr_df.to_csv(run_dir / "threshold_search.csv", index=False)
    # LOCKED eval: same kd_core implementation for val AND test, at the val-selected thr.
    subj_map = dict(zip(test_df["stem"].astype(str), test_df["subject"].astype(str)))
    pi, summ = K.evaluate(student, test_loader8, device, best_thr, amp)
    pi["subject"] = pi["stem"].map(subj_map)
    pi.to_csv(run_dir / "test_per_image.csv", index=False)
    vsubj_map = dict(zip(val_df["stem"].astype(str), val_df["subject"].astype(str)))
    val_pi, _ = K.evaluate(student, val_loader8, device, best_thr, amp)
    val_pi["subject"] = val_pi["stem"].map(vsubj_map)
    val_pi.to_csv(run_dir / "val_per_image.csv", index=False)   # for the VALIDATION oracle
    tail = K.tail_metrics(pi)

    fair = {}
    if args.test_ita and Path(args.test_ita).exists():
        pg, fair = K.fairness_by_group(pi, args.test_ita, args.run_id,
                                       out_csv=run_dir / "fairness_per_group.csv")
    result = {"run_id": args.run_id, "seed": args.seed, "best_val_dice": best_dice,
              "threshold": best_thr, **summ, **tail, **fair}
    done.write_text(json.dumps(result, indent=2))
    if resume_path.exists():
        resume_path.unlink()   # run fully complete — clear the resume checkpoint
    print(f"  TEST dice={summ['mean_dice']:.4f} median={summ['median_dice']:.4f} "
          f"miss={summ['complete_miss_rate']*100:.2f}% "
          f"rec<.1={tail['pct_recall_below_0.10']*100:.1f}% "
          f"p5subj={tail.get('p5_subject_dice', float('nan')):.3f} "
          f"worst-grp={fair.get('worst_group_median_dice', float('nan')):.4f}")
    del student
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _build_parser():
    p = argparse.ArgumentParser(description="Unified SegFormer distillation trainer.")
    p.add_argument("--run-id", default="run", help="set per arm; optuna_alpha overrides it")
    p.add_argument("--out-dir", default="distill_out")
    p.add_argument("--train-manifest", required=True)
    p.add_argument("--test-manifest", required=True)
    p.add_argument("--data-root", default="")
    p.add_argument("--student-pretrained", default="pretrained_weights/segformer_mit_b0")
    p.add_argument("--teacher-a", required=True, help="teacher A run dir (best_model.pt + temperature.json)")
    p.add_argument("--pretrained-a", required=True, help="HF pretrained dir matching teacher A")
    p.add_argument("--teacher-b", default=None)
    p.add_argument("--pretrained-b", default=None)
    p.add_argument("--ensemble", choices=["none", "uniform", "adaptive"], default="none")
    p.add_argument("--kd", choices=["response", "cwd", "dkd", "bpkd", "angular"], default="response")
    p.add_argument("--group", action="store_true")
    p.add_argument("--hard", action="store_true")
    p.add_argument("--boundary", action="store_true", help="boundary-weighted KD term (edge focus)")
    p.add_argument("--rel-b", type=float, default=None,
                   help="validation-derived reliability of teacher B in [0,1] for adaptive fusion")
    p.add_argument("--train-ita", default=None, help="train ITA csv (stem,ita_group_index_5) for --group")
    p.add_argument("--test-ita", default=None, help="test ITA csv for fairness eval")
    p.add_argument("--alpha", type=float, default=0.75)
    p.add_argument("--lambda-cwd", type=float, default=1.0)
    p.add_argument("--lambda-angular", type=float, default=1.0)
    p.add_argument("--lambda-boundary", type=float, default=4.0, help="beta of the boundary weight band")
    p.add_argument("--bpkd-lam-edge", type=float, default=4.0)
    p.add_argument("--lambda-group", type=float, default=0.5)
    p.add_argument("--gamma-hard", type=float, default=2.0)
    p.add_argument("--dkd-T", type=float, default=4.0)
    p.add_argument("--dkd-beta", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=K.DEFAULTS["epochs"])
    p.add_argument("--patience", type=int, default=K.DEFAULTS["patience"])
    p.add_argument("--img-size", type=int, default=K.DEFAULTS["img_size"])
    p.add_argument("--workers", type=int, default=K.DEFAULTS["workers"])
    p.add_argument("--effective-batch", type=int, default=K.DEFAULTS["effective_batch"])
    p.add_argument("--grad-checkpoint", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--force", action="store_true")
    return p


def parse_args():
    return _build_parser().parse_args()


def parse_args_from(argv):
    """Parse a forwarded argv list (used by optuna_alpha.py)."""
    return _build_parser().parse_args(argv)


def main():
    a = parse_args()
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("!! WARNING: no CUDA -> CPU (smoke test only).")
    train_distill(a, device)


if __name__ == "__main__":
    main()
