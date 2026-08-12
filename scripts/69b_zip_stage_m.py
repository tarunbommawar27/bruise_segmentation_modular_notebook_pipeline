#!/usr/bin/env python
"""Zip Stage M -- one notebook, one module, one upload.

WHAT IT ADDS, AND WHAT IT DOES NOT TOUCH
------------------------------------------
    bruisekit/multiteacher.py      NEW FILE  the gate, the router, four shims
    bruise_stage_m.ipynb           NEW FILE  run top to bottom
    PROJECT_HANDBOOK.md            updated

No shipped module, no existing notebook and no result file is overwritten, so
applying this cannot change a published number. At runtime the notebook writes
only to `STAGE_M_RESULTS/` -- never to `results/`, `FINAL_RESULT/` or
`_work/runs/` -- so an experiment that fails leaves no trace in the directories
the study's numbers come from.

WHY multiteacher.py IS NOT A BUILD OUTPUT
-------------------------------------------
Every other module in `bruisekit/` is copied from `scripts/unified_lib/` by
`60_build_unified_bundle.py`. This one is deliberately absent from that list:
Stage M is an experiment that may return nothing, and an untested loss, an
untested teacher shim and a third `resolve_micro_batch` patch do not belong in
the file that produces Stages A through Y. `copy_authored_modules` copies a fixed
list and does not clear the directory, so a rebuild neither regenerates nor
deletes it. Graduating it is two lines. This script FAILS if that has already
happened, because then the overlay ships a stale duplicate.

EVERY SETTING IN THE NOTEBOOK CAME FROM A RUN
-----------------------------------------------
Not one of these is a guess:

  POOL = 3 teachers      U-Net's drop-one marginal was +0.0055 against
                         +0.011..+0.022. Removing it cost exactly that and
                         DOUBLED DeepLab's -- the two ResNet-50s were covering
                         for each other.
  MAX_MICRO_BATCH = 16   the first run OOM'd: 3 teachers resident + the student at
                         micro-batch 64 misses a 40 GB MIG slice by ~200 MB.
                         16 x 4 keeps effective batch, step count and LR schedule.
  TEACHER_CHUNK = 4      teacher forwards sub-batched; frozen BN, identical output.
  alloc config in §0     PYTORCH_CUDA_ALLOC_CONF is read when the CUDA allocator
                         initialises, so it is set before torch is imported and
                         the cell asserts torch is not already in sys.modules.
  EXTRA_RUNS             verified: resolves all three b2kd control seeds.
"""
from __future__ import annotations

import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DST = ROOT / "BRUISE_STAGE_M.zip"

MEMBERS: list[tuple[str, str, str]] = [
    ("BRUISE_UNIFIED/bruise_stage_m.ipynb", "bruise_stage_m.ipynb",
     "32 cells, runs top to bottom. §0-§7 gate (no training), §9 guard, §11 trains."),
    ("BRUISE_UNIFIED/bruisekit/multiteacher.py", "bruisekit/multiteacher.py",
     "oracle_gate, routing_weights, RoutedDistillLoss, load_teacher_model (the B5 "
     "kd/-wrapper remap), control_batch(max_micro=) (the OOM fix), four shims."),
    ("BRUISE_UNIFIED/PROJECT_HANDBOOK.md", "PROJECT_HANDBOOK.md",
     "Stage A/B/E/Y refreshed from RESULT_AUGUST_08; NEW §7e (Stage M); "
     "NEW traps 19-21; §5 miss column corrected to dice==0; §7.3 nine-row table."),
    ("scripts/69_generate_stage_m_notebook.py",
     "_source/scripts/69_generate_stage_m_notebook.py",
     "emits the notebook -- its source of truth."),
    ("scripts/69b_zip_stage_m.py", "_source/scripts/69b_zip_stage_m.py", "this script."),
]

README = """\
# Stage M — multi-teacher routed distillation

    cd /scratch/tbommawa/BRUISE_UNIFIED
    unzip -o BRUISE_STAGE_M.zip

No `-d`. **Restart the kernel afterwards.** Then open `bruise_stage_m.ipynb` and
run every cell in order. You should not need to change any setting.

Adds two new files and refreshes the handbook. Overwrites no module, no notebook
and no result, so `bruise_unified.ipynb` behaves exactly as before.

---

## Where results go

Everything lands in **`STAGE_M_RESULTS/`** at the bundle root:

    STAGE_M_RESULTS/
      val_dice_matrix__seed0.csv      per-image Dice, every teacher, 134 val images
      gate__seed0.json                the gate: gain, CI, marginals, verdict
      stratified_size__seed0.csv      oracle gain by bruise-size quintile
      stratified_ita__seed0.csv       oracle gain by skin-tone group
      stage_m_test_per_seed.csv       THE TABLE -- test Dice/median/miss per run
      stage_m_contrasts.csv           THE ANSWER -- each arm vs its control
      stage_m_misses.csv              dice==0 counts, arms and controls
      stage_m_fairness.csv            Kruskal p, gap, best/worst group
      stage_m_router_diagnostics.csv  routing weights + entropy
      runs/<family>__seed<N>/         best.pt, operating_point.json,
                                      test_per_image.csv, router_stats.json

**Nothing is written to `results/`, `FINAL_RESULT/` or `_work/runs/`.** Training
runs go to `STAGE_M_RESULTS/runs/`, not the shared run directory, so a failed
experiment leaves no trace in the places the published numbers come from.

---

## How it is meant to be run

**§0–§7 is the gate.** Validation only, no training, a few minutes: it scores the
three teachers on the 134 val images and decides whether the pool has headroom
the student could actually inherit. It ends in a printed verdict.

**§9 is the guard.** It re-reads the JSON the gate wrote and raises if the gate
closed or if the controls are unreachable. `RUN_METHOD = True` on its own
authorises nothing.

**§11 trains.** ~14 GPU-hours, 2 arms x 3 seeds. `train_run` resumes from
`DONE.json` / `resume.pt`, so a dropped connection costs only the epochs since
the last sync — re-run the cell.

---

## Two things to be honest about when you write this up

**Report Dice, not misses.** The gate's miss clause passes and it is an artefact.
SegFormer-B2 alone has 1 complete miss on validation — exactly as many as the
whole pool — and that miss is a single unlabelled image that no teacher finds.
The clause reads PASS only because it is compared against B5, which has 3. The
pool recovers nothing on misses. Dice is the pre-registered endpoint and the
notebook says so in three places.

**This is not a fairness method.** It started as one. But every teacher's complete
misses are on **light** skin; on 55 dark-skin test images not one of them misses
anything. There is no dark-skin failure to route around, so a per-skin-tone router
would be fitting weights on three-to-five validation subjects per group to fix
something that is not there. Call it multi-teacher distillation. Report the
fairness result separately, as its own finding: *the models do not have the
dark-skin gap people assume they have.*

---

## What the numbers already say

The gate, on the three-teacher pool:

    teacher                 val Dice   wins    drop-one marginal
    segformer_b2_teacher     0.7717   27.6%          +0.0121
    segformer_b5_teacher     0.7881   44.8%          +0.0138
    deeplabv3plus_r50        0.7838   27.6%          +0.0224
    ORACLE (per-image best)  0.8363

    oracle gain over B5      +0.0482   CI [+0.0276, +0.0609]   P(>0) = 1.000
    x transfer rate 0.264 (Stage C, measured)
    PROJECTED STUDENT GAIN   +0.0127   vs margin +0.0100        -> RUN

Three complementary architectures, no redundant member. The cushion on the
projection is 1.27x, and the 0.264 transfer rate is a SINGLE measurement from
Stage C — state it as an assumption in the writeup, not as a prediction.

**Expected outcome: a small positive delta whose CI includes zero.** Four arms of
this shape have already returned nulls (Stage C `p3_adaptive`, Stage C
`p3_adaptive_group`, Stage H's reliability gate, Stage F's Fast-SCNN arm). What
is different here is that the pool is three teachers rather than two, no member is
redundant, the routing signal is exact rather than estimated, and the decision to
spend the hours was written down in advance. A null is then reportable as *"the
transfer rate held and the effect was still inside label noise"* rather than
*"we tried some things."*

---

## The one thing to read in §15

`mean_routing_entropy`, against `log 3 = 1.10`:

| entropy | what the arm actually is |
|---|---|
| near 1.10 | the router stayed uniform — this is a teacher **ensemble** |
| near 0 | it collapsed — single-teacher KD, teacher chosen per image |
| in between | genuine routing |

That number decides what the method is called in the paper. §15 prints the
interpretation for you.

---

## Two recipe differences to put in the limitations

1. **BatchNorm group size.** The control trained at micro-batch 64; three
   teachers do not fit alongside that on a 40 GB slice. `MAX_MICRO_BATCH = 16`
   gives 16 x 4 accumulation — effective batch, optimizer-step count and LR
   schedule all identical to the control — but SegFormer's decode-head BatchNorm
   now normalises over 16 images instead of 64. Real, small, and it should be
   stated.
2. **Same-seed teachers.** Student seed *k* distils from teacher seed *k*, Stage
   A's convention, so the reported spread includes the teachers' own variance.
"""


def main() -> int:
    missing = [s for s, _, _ in MEMBERS if not (ROOT / s).exists()]
    if missing:
        raise SystemExit("not built -- run 69_generate_stage_m_notebook.py first. "
                         "Missing:\n  " + "\n  ".join(missing))

    build = (ROOT / "scripts" / "60_build_unified_bundle.py").read_text(encoding="utf-8")
    if "multiteacher.py" in build:
        raise SystemExit(
            "60_build_unified_bundle.py now copies multiteacher.py, so it has a "
            "source of truth in scripts/unified_lib/ and this overlay is stale.")
    if (ROOT / "scripts" / "unified_lib" / "multiteacher.py").exists():
        raise SystemExit(
            "scripts/unified_lib/multiteacher.py exists -- two copies, no rule "
            "about which wins. Pick one before shipping.")

    # The two identities the whole contrast rests on, asserted at BUILD time as
    # well as at run time: an archive shipping a broken reduction is an archive
    # whose every downstream number is confounded by the loss itself.
    r = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, r'%s'); "
         "from bruisekit import multiteacher as MT; "
         "raise SystemExit(0 if MT.self_test(verbose=False) else 1)"
         % str(ROOT / "BRUISE_UNIFIED")],
        capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"multiteacher.self_test failed -- refusing to ship.\n"
                         f"{r.stdout}\n{r.stderr}")
    print("multiteacher.self_test passed (K=1 and beta=0 reductions hold)")

    # A notebook that will not parse is not shippable, whatever else is right.
    import json as _json
    nb = _json.loads((ROOT / "BRUISE_UNIFIED" / "bruise_stage_m.ipynb")
                     .read_text(encoding="utf-8"))
    n_code = 0
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            compile("".join(c["source"]), f"cell{i}", "exec")
            n_code += 1
    print(f"notebook parses: {len(nb['cells'])} cells, {n_code} code cells compile")

    t0 = time.time()
    total = sum((ROOT / s).stat().st_size for s, _, _ in MEMBERS)
    print(f"\nzipping {len(MEMBERS)} files, {total / 1e6:.2f} MB -> {DST.name}")
    print("archive paths are relative to BRUISE_UNIFIED/ -- extract INSIDE the "
          "bundle, no -d\n")

    with zipfile.ZipFile(DST, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.writestr("README_STAGE_M.md", README)
        for src, arc, why in MEMBERS:
            zf.write(ROOT / src, arc)
            print(f"  {arc:<44} {why}")

    with zipfile.ZipFile(DST) as zf:
        bad = zf.testzip()
        names = zf.namelist()
    if bad:
        raise SystemExit(f"archive is corrupt at {bad}")
    for required in ("bruisekit/multiteacher.py", "bruise_stage_m.ipynb"):
        if required not in names:
            raise SystemExit(f"{required} did not make it into the archive")

    print(f"\nDONE  {DST}\n      {DST.stat().st_size / 1e6:.2f} MB, {len(names)} "
          f"entries, integrity check passed ({time.time() - t0:.1f}s)")
    print("\n  cd /scratch/tbommawa/BRUISE_UNIFIED && unzip -o BRUISE_STAGE_M.zip")
    print("  restart the kernel, open bruise_stage_m.ipynb, run all cells in order.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
