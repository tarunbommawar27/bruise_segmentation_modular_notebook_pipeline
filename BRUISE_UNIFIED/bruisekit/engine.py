"""The training loop. One loop for every model, and it survives session death."""
from __future__ import annotations

import json
import os
import random
import shutil
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from bruisekit.data import make_loader
from bruisekit.losses import DistillLoss, SupervisedLoss
from bruisekit.metrics import BinnedAP
from bruisekit.models import build_model, build_param_groups, count_params


def seed_everything(seed: int) -> None:
    """Seed every RNG that touches training.

    cudnn.deterministic without use_deterministic_algorithms(True): the latter
    makes several ops here (bilinear interpolate backward, scatter_add) raise
    instead of run. Bitwise GPU determinism is therefore NOT claimed -- which is
    precisely why this notebook runs three seeds and reports spread rather than
    pretending one run is the answer.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def lr_multiplier(step: int, total_steps: int, warmup_steps: int, power: float = 1.0) -> float:
    """Linear warmup 0->1, then poly decay 1->0. SegFormer's schedule (Xie 2021)."""
    if step <= warmup_steps:
        return step / max(1, warmup_steps)
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return (1.0 - progress) ** power


def resolve_micro_batch(model, cfg, device, teacher=None) -> tuple[int, int]:
    """Choose (micro_batch, accum_steps) by probing what actually fits in VRAM.

    Two policies, selected by cfg["batch_mode"]:

    "matched" (the paper recipe) -- the probe is CAPPED AT effective_batch, and
    accumulation makes up any shortfall, so EVERY model trains at the same effective
    batch (8) and the same number of optimizer steps regardless of GPU. This is the
    fix for the old pipeline's central bug: the probe used to return 32 for the small
    models while accum collapsed to 1, so B2 trained at batch 8 / 8700 steps and both
    B0s at batch 32 / 2100 steps at the same LR -- making every "identical
    hyperparameters" claim between B0 and B2 false. A T4 just uses more accumulation
    than an A100 and lands in the same place.

    "per_model" -- the probe is NOT capped; each model uses the largest power-of-2
    batch its own size allows (accum = 1). B2 stays ~8, B0/YOLO climb much higher.
    Faster and better GPU utilisation, but the models NO LONGER share an effective
    batch or a step count, so this policy is not recipe-matched across model sizes
    (see the per-model notebook's §1/§6 warning). Same LR is kept deliberately (no
    linear-scaling adjustment), so the small models take fewer, larger steps.

    The probe runs on a DEEPCOPY: it does real forward/backward/step, and probing the
    live model would perturb the pretrained weights before training starts by a
    model-dependent amount.
    """
    mode = cfg.get("batch_mode", "matched")
    effective = cfg["effective_batch"]
    if not torch.cuda.is_available():
        # No GPU to probe: matched keeps its fixed effective batch via accumulation;
        # per_model has no target to hit, so fall back to a single sample.
        return (1, effective) if mode == "matched" else (1, 1)

    # matched caps the probe at effective_batch (accum fills the rest); per_model
    # lets it climb to max_probe_batch, whatever the model can hold.
    probe_ceiling = min(cfg["max_probe_batch"], effective) if mode == "matched" else cfg["max_probe_batch"]

    probe_model = deepcopy(model).to(device)
    probe_opt = torch.optim.SGD(probe_model.parameters(), lr=1e-9)
    scaler = torch.amp.GradScaler("cuda") if cfg["amp"] else None
    total_vram = torch.cuda.get_device_properties(device).total_memory

    chosen, batch = 1, 1
    while batch <= probe_ceiling:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            x = torch.rand(batch, 3, cfg["img_size"], cfg["img_size"], device=device)
            y = torch.randint(0, 2, (batch, 1, cfg["img_size"], cfg["img_size"]), device=device).float()
            probe_model.train()
            probe_opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg["amp"]):
                # The teacher runs on every real training step too; excluding it here
                # would choose a batch that OOMs the moment distillation starts.
                if teacher is not None:
                    _ = teacher(x)
                logits, aux = probe_model.forward_train(x)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
                if aux is not None:
                    loss = loss + nn.functional.binary_cross_entropy_with_logits(aux, y)
            if scaler is not None:
                scaler.scale(loss).backward(); scaler.step(probe_opt); scaler.update()
            else:
                loss.backward(); probe_opt.step()
            frac = torch.cuda.max_memory_reserved(device) / total_vram
            del x, y, logits, loss
            # 0.75 leaves headroom for the real loss (Dice+BCE+aux), augmentation
            # variance and the checkpoint write -- the probe only sees a plain BCE.
            if frac > cfg.get("vram_target", 0.75):
                break
            chosen = batch
            batch *= 2
        except torch.cuda.OutOfMemoryError:
            break

    del probe_model, probe_opt
    torch.cuda.empty_cache()

    micro = max(1, chosen)
    if mode == "per_model":
        return micro, 1                    # use the probed batch directly, no accumulation
    while effective % micro != 0:          # matched: keep micro*accum EXACT, never approximate
        micro -= 1
    accum = effective // micro
    assert micro * accum == effective, f"{micro} x {accum} != {effective}"
    return micro, accum


def load_teacher(teacher_dir: Path, paths: dict, device, amp: bool):
    """Load a trained B2 as a frozen, calibrated soft-label generator.

    Returns a callable so the training loop never learns anything about the
    teacher's architecture -- it just calls teacher(x) and gets a probability map.
    """
    from bruisekit.models import build_model as _bm
    ckpt = teacher_dir / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Teacher not trained yet: {ckpt}")

    model = _bm("segformer", "b2", paths).to(device)
    model.load_state_dict(torch.load(str(ckpt), map_location=device, weights_only=True))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    temperature = json.loads((teacher_dir / "calibration.json").read_text())["temperature"]

    def teacher_fn(x):
        # no_grad: the teacher is frozen. This is the single largest memory saving
        # in the distillation setup -- without it we would build a backward graph
        # through 27M parameters we never update.
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=amp):
            logits = model(x)
        return torch.sigmoid(logits.float() / temperature)

    teacher_fn.temperature = temperature
    return teacher_fn


def calibrate_temperature(model, loader, device, amp: bool) -> dict:
    """Fit the scalar T minimising val NLL of sigmoid(z/T). Guo et al. (2017).

    WHY: BCE drives a trained model's logits toward +-inf, because sigmoid(+-inf)
    is a perfect loss. The teacher's probability histogram ends up nearly binary,
    so sigmoid(z_teacher) as a soft label is almost indistinguishable from the hard
    GT -- which defeats the point of soft-label distillation. Dividing by T > 1
    pulls saturated logits back into sigmoid's responsive region, so the student
    can see where the teacher is UNCERTAIN, which is the information distillation
    is supposed to transfer.

    Optimises log(T), not T: keeps T > 0 automatically without a constraint, and
    avoids the singularity at T = 0.
    L-BFGS, not SGD: one scalar over a fixed dataset -- second-order curvature
    converges in ~10 iterations where SGD needs hundreds.
    """
    model.eval()
    logits_all, targets_all = [], []
    with torch.no_grad():
        for x, y, _ in tqdm(loader, desc="collect val logits", leave=False):
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp):
                z = model(x)
            # Downsample 4x before storing: calibration fits ONE scalar, and a
            # regular 1-in-16 pixel subsample of 55M pixels estimates it to far
            # more precision than it is worth -- while keeping this in RAM.
            logits_all.append(z.float()[..., ::4, ::4].cpu())
            targets_all.append(y[..., ::4, ::4].cpu())

    logits = torch.cat(logits_all)
    targets = torch.cat(targets_all)
    nll_before = float(torch.nn.functional.binary_cross_entropy_with_logits(logits, targets))

    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.05, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits / torch.exp(log_t), targets)
        loss.backward()
        return loss

    opt.step(closure)
    temperature = float(torch.exp(log_t).item())
    nll_after = float(torch.nn.functional.binary_cross_entropy_with_logits(logits / temperature, targets))

    # A well-trained BCE model is OVER-confident, so T lands a little above 1
    # (this project's B2 previously calibrated to 1.84). Anything far outside that
    # is a symptom, not a temperature:
    #   T >> 10  -> the logits are near zero, i.e. the model is barely trained and
    #               calibration is degenerately flattening an already-flat output.
    #               Distilling from it would teach the student a constant 0.5.
    #   T < 0.5  -> the model is UNDER-confident, which BCE training does not
    #               normally produce -- suspect the checkpoint or the val split.
    # Loud warning rather than a raise: the number is still recorded and the run can
    # proceed, but this must never pass silently into a paper.
    if not (0.5 <= temperature <= 10.0):
        print(f"  !! WARNING: calibrated T={temperature:.3f} is outside the plausible "
              f"[0.5, 10] range. The teacher is probably under-trained (near-zero "
              f"logits) -- check its val AP before trusting any distilled student.")

    return {"temperature": temperature, "nll_before": nll_before, "nll_after": nll_after,
            "n_pixels_used": int(targets.numel()), "plausible": bool(0.5 <= temperature <= 10.0)}


@torch.no_grad()
def eval_ap(model, loader, device, amp: bool) -> float:
    """Threshold-free val AP -- the model-selection metric. See metrics.BinnedAP."""
    model.eval()
    ap = BinnedAP(device=str(device))
    for x, y, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp):
            logits = model(x)
        ap.update(torch.sigmoid(logits.float()), y)
    return ap.compute()


def _atomic_save(obj, dest: Path) -> None:
    """Write to a temp file, then rename.

    Drive rename is atomic; a plain torch.save straight to Drive that is killed
    halfway leaves a TRUNCATED checkpoint, and the next session then dies trying
    to resume from it -- turning one lost session into a lost run.
    """
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, dest)


def train_run(run_id: str, spec: dict, seed: int, cfg: dict, paths: dict,
              manifests: dict, root: Path, runs_dir: Path, device) -> dict:
    """Train one (model, seed). Idempotent and resumable.

    RESUME CONTRACT
    ----------------
    - DONE.json exists          -> return immediately, touch nothing.
    - resume.pt exists          -> restore model+optimizer+scaler+epoch+best and continue.
    - neither                   -> fresh start.

    resume.pt is written every cfg["drive_sync_every"] epochs (and always on the
    final epoch), so a disconnect costs at most that many epochs. It is saved even
    on epochs where val AP did NOT improve: the best weights live in best.pt, but
    resuming must continue from where training actually WAS, not from the last
    good epoch, or the LR schedule and optimizer moments silently rewind.
    """
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    done_file = run_dir / "DONE.json"
    if done_file.exists():
        return {"run_id": run_id, "status": "skipped", **json.loads(done_file.read_text())}

    seed_everything(seed)
    amp = cfg["amp"]

    model = build_model(spec["arch"], spec["size"], paths).to(device)

    teacher = None
    if spec["distill"]:
        # Same-seed teacher: the distilled run's spread over seeds then includes the
        # teacher's own variance, which is part of the pipeline being measured.
        teacher = load_teacher(runs_dir / f"segformer_b2_teacher__seed{seed}", paths, device, amp)

    micro, accum = resolve_micro_batch(model, cfg, device, teacher)

    train_loader = make_loader(manifests["train"], root, cfg["img_size"], micro,
                               training=True, workers=cfg["workers"], seed=seed)
    val_loader = make_loader(manifests["val"], root, cfg["img_size"], micro,
                             training=False, workers=cfg["workers"], seed=seed)

    param_groups = build_param_groups(model, cfg["backbone_lr"], cfg["head_lr"], cfg["weight_decay"])
    optimizer = torch.optim.AdamW(param_groups, betas=tuple(cfg["betas"]))
    peak_lrs = [g["lr"] for g in param_groups]
    scaler = torch.amp.GradScaler("cuda") if amp else None

    steps_per_epoch = max(1, len(train_loader) // accum)
    total_steps = steps_per_epoch * cfg["epochs"]
    warmup_steps = max(1, int(total_steps * cfg["warmup_fraction"]))

    criterion = (DistillLoss(cfg["alpha"], cfg["aux_weight"]) if teacher is not None
                 else SupervisedLoss(cfg["aux_weight"]))

    start_epoch, best_ap, patience, global_step, history = 1, float("-inf"), 0, 0, []
    resume_path = run_dir / "resume.pt"
    if resume_path.exists():
        st = torch.load(str(resume_path), map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        optimizer.load_state_dict(st["optimizer"])
        if scaler is not None and st.get("scaler"):
            scaler.load_state_dict(st["scaler"])
        start_epoch, best_ap = st["epoch"] + 1, st["best_ap"]
        patience, global_step, history = st["patience"], st["global_step"], st["history"]
        print(f"  [resume] {run_id} from epoch {start_epoch} (best_ap={best_ap:.4f})")
        del st

    (run_dir / "config.json").write_text(json.dumps({
        "run_id": run_id, "seed": seed, **spec,
        "micro_batch": micro, "accum_steps": accum, "effective_batch": micro * accum,
        "total_steps": total_steps, "warmup_steps": warmup_steps,
        "params": count_params(model),
        "alpha": cfg["alpha"] if spec["distill"] else None,
        "teacher_temperature": getattr(teacher, "temperature", None),
    }, indent=2))

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running, t0 = 0.0, time.time()

        for step, (x, y, _) in enumerate(tqdm(train_loader, desc=f"{run_id} e{epoch}", leave=False)):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=amp):
                # Teacher FIRST, inside its own no_grad: its activations are freed
                # before the student's backward graph is built. Student-first would
                # hold both alive at once and roughly double peak VRAM.
                tprob = teacher(x) if teacher is not None else None
                logits, aux = model.forward_train(x)
                loss = (criterion(logits, aux, y, tprob) if teacher is not None
                        else criterion(logits, aux, y))
                loss = loss / accum

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            running += loss.item() * accum

            if (step + 1) % accum == 0 or (step + 1) == len(train_loader):
                global_step += 1
                # LR is set per OPTIMIZER STEP, not per micro-batch: the schedule is
                # defined over gradient updates, and updating it per micro-batch would
                # advance it accum_steps times too fast.
                mult = lr_multiplier(global_step, total_steps, warmup_steps, cfg["poly_power"])
                for g, peak in zip(optimizer.param_groups, peak_lrs):
                    g["lr"] = peak * mult
                if scaler is not None:
                    scaler.unscale_(optimizer)   # real gradients before clipping
                    nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"])
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"])
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        val_ap = eval_ap(model, val_loader, device, amp)
        train_loss = running / max(1, len(train_loader))
        cur_lr = peak_lrs[0] * lr_multiplier(global_step, total_steps, warmup_steps, cfg["poly_power"])
        history.append({"epoch": epoch, "train_loss": train_loss, "val_ap": val_ap,
                        "backbone_lr": cur_lr, "sec": round(time.time() - t0, 1)})
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

        if val_ap > best_ap:
            best_ap, patience = val_ap, 0
            _atomic_save(model.state_dict(), run_dir / "best.pt")
            flag = " *"
        else:
            patience += 1
            flag = ""
        print(f"  {run_id} e{epoch:3d} loss={train_loss:.4f} val_ap={val_ap:.4f}"
              f" lr={cur_lr:.2e} {time.time()-t0:.0f}s{flag}")

        last = (epoch == cfg["epochs"]) or (patience >= cfg["patience"])
        if epoch % cfg["drive_sync_every"] == 0 or last:
            _atomic_save({"epoch": epoch, "model": model.state_dict(),
                          "optimizer": optimizer.state_dict(),
                          "scaler": scaler.state_dict() if scaler else None,
                          "best_ap": best_ap, "patience": patience,
                          "global_step": global_step, "history": history}, resume_path)
        if patience >= cfg["patience"]:
            print(f"  early stop at epoch {epoch} (patience={cfg['patience']})")
            break

    # The teacher must be calibrated before any student can distil from it, and
    # calibration needs the BEST weights -- so it happens here, not in a separate step
    # that could be forgotten or run against the wrong checkpoint.
    model.load_state_dict(torch.load(str(run_dir / "best.pt"), map_location=device, weights_only=True))
    if run_id.startswith("segformer_b2_teacher"):
        cal = calibrate_temperature(model, val_loader, device, amp)
        (run_dir / "calibration.json").write_text(json.dumps(cal, indent=2))
        print(f"  calibrated T={cal['temperature']:.4f} "
              f"(val NLL {cal['nll_before']:.4f} -> {cal['nll_after']:.4f})")

    summary = {"run_id": run_id, "seed": seed, "best_val_ap": best_ap,
               "epochs_trained": len(history), "params": count_params(model),
               "micro_batch": micro, "accum_steps": accum}
    done_file.write_text(json.dumps(summary, indent=2))
    if resume_path.exists():
        resume_path.unlink()   # training finished: the resume state is now dead weight

    del model, teacher
    torch.cuda.empty_cache()
    return {"status": "trained", **summary}