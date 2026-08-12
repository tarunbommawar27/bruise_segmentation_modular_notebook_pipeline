#!/usr/bin/env python
"""Zip Stage O -- one notebook, one module, one runner, one upload.

WHAT IT ADDS, AND WHAT IT DOES NOT TOUCH
------------------------------------------
    bruisekit/itakd.py        NEW FILE  miss taxonomy, distilled fairness, the
                                        ITA-group gate, the shims, the loss
    bruise_stage_o.ipynb      NEW FILE  run top to bottom
    run_stage_o.py            NEW FILE  the same pipeline from a shell

No shipped module, no existing notebook and no result file is overwritten, so
applying this cannot change a published number.

THREE MODULES ARE IMPORTED AND DELIBERATELY NOT SHIPPED HERE
-------------------------------------------------------------
    bruisekit/lesionsize.py    load_lineage, assign_bins, headline, by_bin,
                               fairness_conditioned, size_by_ita, DEFAULT_MODELS
    bruisekit/multiteacher.py  load_teacher_model (B5's `model.`->`net.` remap),
                               score_on_split, resolve_teachers,
                               teacher_temperature, control_batch
    bruisekit/samprobe.py      load_trained, build_arm -- MedSAM's encoder

They arrive with BRUISE_LESION_SIZE.zip, BRUISE_STAGE_M.zip and
BRUISE_STAGE_N4.zip. Shipping second copies is how two files with no rule about
which wins get created. This script FAILS if any is absent from the bundle.

`multiteacher.load_teacher_model` in particular MUST be the imported one. It
carries the checked remap from the Stage C `kd/` wrapper's `model.` prefix to
`SegFormerNet`'s `net.` prefix, plus the two buffers that wrapper never wrote. A
re-typed copy here that drifted would load B5 with random weights somewhere in
it, still produce plausible probabilities, and the gate would report them as
complementarity.

WHY itakd.py IS NOT A BUILD OUTPUT
-----------------------------------
Same policy as foundation.py, dermprobe.py, finetune_n3.py, multiteacher.py,
lesionsize.py and samprobe.py: authored in bruisekit/ but absent from
60_build_unified_bundle.py's copy list. It is an experiment that may return
nothing, and it must not become a dependency of the file that produces Stages A
through Y. This script FAILS if it has been added to that list, because then the
overlay would ship a stale duplicate.

WHY THE RESULTS ARE NOT IN THE ZIP
-----------------------------------
`STAGE_O_RESULTS/` is an OUTPUT. Shipping a copy of it inside the archive that
regenerates it creates two tables with no rule about which is current -- exactly
the failure the shipped-vs-regenerated `benchmark_640.csv` produced (handbook
§15). The archive carries the code; the results tree is delivered separately and
regenerates in ~3 minutes with:

    python run_stage_o.py --only miss,fairness

NO CHECKPOINTS, NO WEIGHTS
---------------------------
Stage O trains nothing at unzip time and redistributes nothing. The teacher pool
resolves through the Registry (set EXTRA_RUNS) and MedSAM comes from Stage N4's
own runs directory.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DST = ROOT / "BRUISE_STAGE_O.zip"

MEMBERS: list[tuple[str, str, str]] = [
    ("BRUISE_UNIFIED/bruise_stage_o.ipynb", "bruise_stage_o.ipynb",
     "28 cells. Cells 1-15 need no GPU and close TODO 1 and TODO 3 on their own; "
     "17 scores the pool on val, 22 is the gate, 24 trains."),
    ("BRUISE_UNIFIED/bruisekit/itakd.py", "bruisekit/itakd.py",
     "miss_taxonomy, distilled_fairness, val_group_matrix, group_weights, "
     "identifiability, ita_group_gate, install_group_shim, install_teacher_shim, "
     "install_loss_shim, GroupRoutedDistillLoss, train_arms, self_test."),
    ("BRUISE_UNIFIED/run_stage_o.py", "run_stage_o.py",
     "the same pipeline from a shell -- no Jupyter. Seven stages, each skippable. "
     "--only miss,fairness runs on a laptop."),
    ("scripts/83_generate_stage_o_notebook.py",
     "_source/scripts/83_generate_stage_o_notebook.py",
     "emits the notebook -- its source of truth."),
    ("scripts/83b_zip_stage_o.py", "_source/scripts/83b_zip_stage_o.py",
     "this script."),
]

#: Imported by itakd.py, shipped by an earlier overlay, and required to be present
#: in the bundle before this archive is built.
PREREQS: list[tuple[str, str]] = [
    ("BRUISE_UNIFIED/bruisekit/lesionsize.py", "BRUISE_LESION_SIZE.zip"),
    ("BRUISE_UNIFIED/bruisekit/multiteacher.py", "BRUISE_STAGE_M.zip"),
    ("BRUISE_UNIFIED/bruisekit/samprobe.py", "BRUISE_STAGE_N4.zip"),
]

README = """\
BRUISE STAGE O -- miss taxonomy, distilled-arm fairness, ITA-grouped multi-teacher KD
=====================================================================================

WHAT THIS IS
------------
Three open items, one notebook, because the first two decide whether the third is
worth GPU time.

  TODO 1  complete miss rate AND zero Dice for the distillation arms   (no GPU)
  TODO 3  skin tone x lesion size for the best distilled models        (no GPU)
  TODO 2  gated multi-teacher distillation grouped by ITA              (CUDA)

TODO 1 -- WHY ONE MISS COLUMN WAS NOT ENOUGH
---------------------------------------------
The study publishes `complete_miss_rate` = (dice == 0). That is the UNION of two
clinically different failures:

    empty prediction   pred_positive_pixels == 0   the model found nothing
    wrong place        dice == 0 and pred > 0      the model outlined the wrong
                                                   region with zero overlap

They differ in 28 of the 40 per-image tables in RESULT_AUGUST_08. The case that
matters is fastscnn_rgkd against fastscnn_b2kd -- same student, same teacher, the
reliability gate is the only difference:

    fastscnn_b2kd    zero Dice 6    empty 2    wrong place 4
    fastscnn_rgkd    zero Dice 6    empty 1    wrong place 5

One column says "unchanged". Three columns say the gate converted blank outputs
into confident misplacements. A wrong-place error is the worse clinical failure:
an empty mask shows nothing and invites a second look, a confident outline in the
wrong location is an assertion.

`wrong_place` is DERIVED as `zero_dice - empty_pred`, never counted separately, so
the three columns cannot fail to add up. `_miss_counts` RAISES if a table has more
empty predictions than zero-Dice images -- arithmetically impossible, and the
signature of a Dice column computed against a different mask version from the
pixel counts.

TODO 3 -- THE TABLE THE DECK WAS MISSING
-----------------------------------------
FINAL_RESULT/RESULT_AUGUST_08/fairness_stats.csv covers 24 models and excludes
EVERY Stage C distillation arm -- including p3_adaptive, the best arm in the study
at 0.7748. So the question the deck is asked about the best distilled model
("how does it do across skin tones?") had no table behind it.

Nothing is retrained. The per-image CSVs already carry subject,
skin_tone_category, ita_group_index_5 and ITA. Every frame comes out of
`lesionsize`, not out of a second implementation -- the only thing Stage O adds is
the model list.

TODO 2 -- WHY THIS IS NOT STAGE M AGAIN
----------------------------------------
Stage M routed per IMAGE on each teacher's soft Dice against the label, and is
explicit that it does NOT route by skin tone. It came back a null: six contrasts,
all inconclusive, miss rate slightly worse.

Stage O routes per ITA GROUP on weights fitted once on validation:

    w_g = softmax(beta * mean_val_dice_k_within_group_g)

  1. NO LABEL AT ROUTING TIME. The only per-image input is its ITA group, a
     manifest column -- so unlike Stage M's router this one is not restricted to
     training.
  2. K weights per group, not per image. Five ITA groups over 20 validation
     subjects cannot support 5 x K free parameters; two groups can.
  3. THE GATE HAS AN IDENTIFIABILITY CLAUSE Stage M did not have. With six
     candidate teachers the per-group argmax on 134 val images is bootstrap-stable
     only 36-52 % of the time, with margins of 0.001-0.008 Dice. An arm routed on
     an argmax that unstable is fitting sampling noise.

THE POOL, AND WHY IT IS THREE AND NOT FOUR
-------------------------------------------
    segformer_b5_teacher   MiT transformer        wins Dark, Tan, Brown on val
    deeplabv3plus_r50      CNN + ASPP             wins Intermediate; 2nd on Light
    medsam_ft              ViT, mask-pretrained   best overall val Dice (0.7957)

segformer_b2_teacher is DROPPED: it wins no group on validation, it had the lowest
drop-one marginal in Stage M's own gate (0.0121 vs DeepLabV3+'s 0.0224), and it is
the same MiT family as B5 -- simultaneously the least useful and the most
correlated member.

unet_r50 is DROPPED for the reason Stage M's own numbers give: it appears as a
per-group winner only in the TEST table quoted in multiteacher.py's docstring, and
was never in Stage M's actual pool. Selecting a per-group teacher on test is
leakage, and this module never reads test before the gate is on disk.

MedSAM enters as its FEATURES, never as MedSAM: samprobe keeps the image encoder
and discards the prompt encoder and mask decoder, because this pipeline is
automatic and has no prompt to give. Write it up that way.

THE PRE-REGISTERED GROUP SCHEME
--------------------------------
`light_vs_rest`: {Very Light (I-II), Light (II-III)} vs everything else.

Across six candidate teachers on the 134 validation images, per-group Dice spans:

    Light (II-III)          0.723 -> 0.785    spread 0.062    <- the signal
    Intermediate (III-IV)   0.802 -> 0.833    spread 0.031
    Tan (IV)                0.780 -> 0.813    spread 0.033
    Dark (VI)               0.769 -> 0.793    spread 0.024
    Brown (V)               0.727 -> 0.805    spread 0.078   (7 images, 4 subj)

Only Light (II-III) has a spread above the annotation-ceiling noise floor on a
usable subject count -- and Light (II-III) is where every teacher's complete
misses are concentrated. On the 55 dark-skin test images not one of Stage M's four
teachers missed anything.

`Very Light (I-II)` is in the Light group ON PURPOSE: it is 12 TRAIN images and 0
validation images, so under a five-group scheme those 12 would have no fitted
weight at all. `group_weights` RAISES on a group with no validation images rather
than falling back to uniform -- a uniform fallback is Stage C's
p2_ensemble_uniform wearing this stage's name.

`scheme="five"` is provided so the underpowered version can be REPORTED rather
than asserted to be underpowered. Run the gate on both.

HOW TO APPLY
------------
1. Unzip into the bundle root, over the top:

       unzip -o BRUISE_STAGE_O.zip -d /path/to/BRUISE_UNIFIED

   It adds exactly three files and overwrites nothing you have results in.

2. PREREQUISITES, all shipped by earlier overlays:

       bruisekit/lesionsize.py    BRUISE_LESION_SIZE.zip
       bruisekit/multiteacher.py  BRUISE_STAGE_M.zip
       bruisekit/samprobe.py      BRUISE_STAGE_N4.zip

   itakd.py IMPORTS all three. multiteacher.load_teacher_model in particular must
   be the imported one -- it carries the checked B5 `model.` -> `net.` wrapper
   remap, and a re-typed copy that drifted would load B5 half-initialised and
   still produce plausible probabilities.

3. TODO 1 and TODO 3, ON THE LAPTOP, no GPU, ~3 minutes:

       python run_stage_o.py --only miss,fairness

   or open bruise_stage_o.ipynb and run through cell 15.

   THERE IS NO --lineage FLAG, ON PURPOSE. `itakd.load_tables` SCANS -- the
   bundle root, the work dir, and every entry in `itakd.EXTRA_ROOTS` (which
   already covers ORC's scratch layout) -- then merges across roots and returns
   the largest coherent cohort. A hard-coded lineage directory is what broke this
   on ORC on 2026-08-12: FINAL_RESULT/ is the laptop's tree and DOES NOT EXIST on
   ORC, so a fixed path runs on one host and raises FileNotFoundError on the
   other. See handbook 10.3 for the per-host table.

   If a model you expect is missing, the fix is another entry in EXTRA_ROOTS, not
   a code change. §3 names every model it could not find.

4. TODO 2 needs the teacher checkpoints. ON ORC:

       python run_stage_o.py --extra-runs /scratch/$USER/bruise_work/runs \\
              --only matrix,gate --also-five

   READ THE GATE. Then, only if it opened:

       python run_stage_o.py --extra-runs ... --only train

   `--only train` REFUSES to start if no gate file is on disk, and refuses again
   if the gate closed unless you pass --force-train. Both refusals are the point:
   they make the val-only ordering a property of the code rather than of whoever
   ran it.

WHICH HALF RUNS ON WHICH MACHINE  (handbook 10.3)
--------------------------------------------------
    tree                                    laptop   ORC
    FINAL_RESULT/RESULT_AUGUST_08/ (40 csv)   yes    NO -- never synced
    LESION_SIZE_RESULTS/, STAGE_M_RESULTS/    yes    NO
    STAGE_N4_RESULTS/tables/                  yes    yes
    STAGE_N4_RESULTS/runs/  (checkpoints)     NO     yes
    <work>/outputs/                           NO     yes -- the ORC default
    /scratch/$USER/bruise_work/runs/           NO     yes -- EXTRA_RUNS

So: stages `miss` and `fairness` are LAPTOP work (they read per-image CSVs) and
stages `matrix`, `gate`, `train` are ORC work (they load checkpoints). Running
either half on the wrong host is a path error, not a bug. The ORC roots, verbatim:

    EXTRA_ROOTS = ["/scratch/tbommawa/bruise_work",
                   "/scratch/tbommawa/bruise_work/outputs",
                   "/scratch/tbommawa/BRUISE_UNIFIED"]
    EXTRA_RUNS  =  "/scratch/tbommawa/bruise_work/runs"

WHERE RESULTS GO
----------------
    STAGE_O_RESULTS/
      tables/   miss_taxonomy{,_by_ita,_by_size}.csv
                fairness__{headline,by_bin,size_by_ita,fairness_recall,
                           fairness_zero_dice,miss_by_ita,miss_by_size}.csv
                leading_distilled_arms.json
                val_pool_matrix.csv
                ita_group_gate__<scheme>.{json,csv,txt}
                ita_group_weights__<scheme>.csv
                ita_group_identifiability__<scheme>.csv
      runs/     one directory per (family, seed): best.pt, config.json,
                group_loss_stats.json

Nothing is written to results/, FINAL_RESULT/, LESION_SIZE_RESULTS/,
STAGE_M_RESULTS/, STAGE_N4_RESULTS/ or _work/runs/. Those trees are READ and never
rewritten. The reporting stages cannot see Stage O's arms: Registry scans env.runs
by name and Stage O trains into its own directory.

THE FAILURE THAT WOULD FAKE A RESULT, AND THE GUARD ON IT
----------------------------------------------------------
`engine.train_run` iterates `for step, (x, y, _) in enumerate(train_loader)` -- it
DISCARDS the stem, so the loss has no idea which images it is looking at. Rather
than edit the shared training loop, `install_group_shim` wraps the TRAINING loader
so each batch records its stems' group indices in a module global immediately
before the batch is yielded. The loader, the teacher forward and the loss all run
synchronously in the main process within one iteration, so the global is always
the current batch's.

That coupling is real and it is GUARDED, not trusted:
`GroupRoutedDistillLoss.forward` RAISES if the group vector is absent or its
length does not match the batch. Without that check a stale or missing global
would silently mean uniform weights, the arm would quietly become
p2_ensemble_uniform with a gate, and it would report an entirely plausible number
for a different experiment. `self_test()` asserts both raises.

READ group_loss_stats.json BEFORE ANY DICE NUMBER. `images_per_group` is the first
thing to check: an arm whose loss never saw one of the groups did not run the
experiment the gate authorised.

WHAT THIS DOES NOT LICENSE
---------------------------
- Neither TODO 1 nor TODO 3 is a significance test. They are descriptive tables
  over one seed's per-image CSVs, and a best-minus-worst gap over five noisy cells
  is biased UPWARD by construction. For a test use
  `lesionsize.contrast_table` with a named, pre-registered contrast.
- Not a seed-robust ITA-grouped result until FAMILIES x SEEDS have all run.
- If the gate closes, that is the reportable finding and it is NOT a failed
  method: the per-group teacher ranking is not estimable on 134 validation
  images, so any arm routed on it is fitting sampling noise. That is the measured
  answer to "why not just group by skin tone" -- a question Stage C's
  p3_adaptive_group and Stage M each settled by assertion.
- If the gate opens and the arm still nulls, that is the FOURTH consecutive KD
  null here (Stage C p3_adaptive_group at 0.7586, Stage H, Stage M). Four nulls
  with one shared explanation -- every model is at the annotation ceiling -- is a
  stronger paper section than any of them alone.
"""


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def main() -> int:
    for rel, overlay in PREREQS:
        p = ROOT / rel
        if not p.exists():
            print(f"FAIL: {p} is absent. itakd.py imports it; apply {overlay} "
                  f"first.", file=sys.stderr)
            return 1

    # If itakd has been graduated into the build, this overlay would ship a stale
    # duplicate with no rule about which wins.
    build = ROOT / "scripts" / "60_build_unified_bundle.py"
    if build.exists() and "itakd" in build.read_text(encoding="utf-8"):
        print("FAIL: 60_build_unified_bundle.py now copies itakd.py. It has been "
              "graduated into the bundle, so this overlay is obsolete -- shipping "
              "it would create a second copy.", file=sys.stderr)
        return 1

    missing = [src for src, _, _ in MEMBERS if not (ROOT / src).exists()]
    if missing:
        print(f"FAIL: missing {missing}", file=sys.stderr)
        return 1

    # A notebook with saved outputs can carry paths, partial results, or a key.
    nb = json.loads((ROOT / "BRUISE_UNIFIED" / "bruise_stage_o.ipynb")
                    .read_text(encoding="utf-8"))
    if any(c.get("outputs") for c in nb["cells"]):
        print("FAIL: the notebook has executed outputs. Regenerate it with "
              "scripts/83_generate_stage_o_notebook.py.", file=sys.stderr)
        return 1

    # The module must pass its own structural checks before it is shipped. No
    # weights, no GPU, no network -- so there is no excuse for skipping it, and it
    # is the check that catches a loss that stopped raising on a missing group
    # vector.
    sys.path.insert(0, str(ROOT / "BRUISE_UNIFIED"))
    try:
        from bruisekit import itakd
    except Exception as exc:                                  # pragma: no cover
        print(f"FAIL: cannot import bruisekit.itakd: {exc}", file=sys.stderr)
        return 1
    print("-- itakd self-test --")
    if not itakd.self_test(verbose=True):
        print("FAIL: itakd.self_test() did not pass. Not shipping this.",
              file=sys.stderr)
        return 1
    print()

    manifest = {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "members": []}
    DST.unlink(missing_ok=True)
    with zipfile.ZipFile(DST, "w", zipfile.ZIP_DEFLATED) as z:
        for src, arc, note in MEMBERS:
            p = ROOT / src
            z.write(p, arc)
            manifest["members"].append(
                {"arcname": arc, "bytes": p.stat().st_size, "sha256_16": sha(p),
                 "note": note})
            print(f"  + {arc:<48} {p.stat().st_size / 1024:7.1f} KB  {sha(p)}")
        z.writestr("README.txt", README)
        z.writestr("MANIFEST.json", json.dumps(manifest, indent=2))

    print(f"\nwrote {DST}  ({DST.stat().st_size / 1024:.1f} KB)")
    print("upload this one file")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
