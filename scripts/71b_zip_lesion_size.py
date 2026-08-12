#!/usr/bin/env python
"""Zip Stage P -- one notebook, one module, one upload. No GPU needed to run it.

WHAT IT ADDS, AND WHAT IT DOES NOT TOUCH
------------------------------------------
    bruisekit/lesionsize.py        NEW FILE  binning, cluster bootstrap, power
    bruise_lesion_size.ipynb       NEW FILE  run top to bottom

No shipped module, no existing notebook and no result file is overwritten, so
applying this cannot change a published number. At runtime the notebook writes
only to `LESION_SIZE_RESULTS/` -- never to `results/`, `FINAL_RESULT/` or
`_work/runs/` -- so an analysis that turns out to be wrong leaves no trace in the
directories the study's numbers come from.

WHY lesionsize.py IS NOT A BUILD OUTPUT
-----------------------------------------
Every other module in `bruisekit/` is copied from `scripts/unified_lib/` by
`60_build_unified_bundle.py`. This one is deliberately absent from that list, for
the same reason `multiteacher.py` and `foundation.py` are: Stage P is an
experiment, and its stratum/contrast pre-registration should not silently become
part of the file that produces Stages A through Y. `copy_authored_modules` copies
a fixed list and does not clear the directory, so a rebuild neither regenerates
nor deletes it. Graduating it is two lines. This script FAILS if that has already
happened, because then the overlay ships a stale duplicate.

NOTHING TO DOWNLOAD, NOTHING TO TRAIN
---------------------------------------
Unlike Stage N this overlay is self-sufficient: the inputs are the per-image CSVs
already in `FINAL_RESULT/RESULT_AUGUST_08/`, which ship with the bundle. It runs
on a laptop with no torch installed.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DST = ROOT / "BRUISE_LESION_SIZE.zip"

MEMBERS: list[tuple[str, str, str]] = [
    ("BRUISE_UNIFIED/bruise_lesion_size.ipynb", "bruise_lesion_size.ipynb",
     "22 cells. CPU only, minutes. §1 is the only cell you edit."),
    ("BRUISE_UNIFIED/bruisekit/lesionsize.py", "bruisekit/lesionsize.py",
     "global decile cut, stratum cluster bootstrap, min-detectable-effect, self_test."),
    ("BRUISE_UNIFIED/PROJECT_HANDBOOK.md", "PROJECT_HANDBOOK.md",
     "NEW §7g (Stage P); §16/§17 updated; §18 queue re-ordered behind it."),
    ("scripts/71_generate_lesion_size_notebook.py",
     "_source/scripts/71_generate_lesion_size_notebook.py",
     "emits the notebook -- its source of truth."),
    ("scripts/71b_zip_lesion_size.py", "_source/scripts/71b_zip_lesion_size.py",
     "this script."),
]

README = """\
# Stage P — does small-lesion miss containment separate the models, or is it underpowered?

    cd /scratch/tbommawa/BRUISE_UNIFIED
    unzip -o BRUISE_LESION_SIZE.zip

No `-d`. **Restart the kernel afterwards.** Then open `bruise_lesion_size.ipynb`
and run it top to bottom.

**No GPU. No torch. No checkpoints. No downloads.** Minutes on a laptop.

---

## Why this runs before anything else in the queue

Stage N's `layer3` control, ALS→white-light distillation and a Fenwick merge are
all justified by the same sentence:

> *"Models miss small bruises, so let us fix that."*

That premise currently rests on single-seed counts. The descriptive pass on
2026-08-07 found that **89% of every complete miss in the study falls in the
smallest four GT-area deciles, and 49% in the smallest one** — but with no
confidence intervals, and D1 is only 19 images.

If the differences inside that stratum are not resolvable at 28 subjects, the
premise is not established and the correct next move is **more data**, not more
mechanism. This notebook settles which it is, for the cost of a coffee.

---

## What it does

| § | what | cost |
|---|---|---|
| §1 | config — **the only cell you edit** | — |
| §2–§3 | env, then the pre-registration printed *before* any number is read | seconds |
| §4 | load the lineage, cut deciles **once** on the global GT-area vector | seconds |
| §5 | descriptive tables — per model, per decile | seconds |
| §6 | marginal cluster-bootstrap CIs in the primary stratum | ~1 min |
| **§7** | **THE ANSWER** — pre-registered paired contrasts | ~2 min |
| §8 | **the power answer** — minimum detectable effect | seconds |
| §9 | write `LESION_SIZE_RESULTS/` | seconds |

---

## The pre-registration, which lives in the module and not in the notebook

```
PRIMARY_STRATUM   = ("D1","D2","D3","D4")     74 images, 89% of all misses
PRIMARY_ENDPOINTS = zero_dice_rate, mean_recall, median_dice
CONTRASTS         = 6 pairs, four confirmatory and two labelled exploratory
```

`bruisekit/lesionsize.py` holds these as module constants for the same reason
`significance.CONTRAST_FAMILY` does: a stratum chosen after seeing which stratum
separates the models is not a test, it is a search.

**Two of the six are labelled `exploratory` on purpose.** They were suggested by
looking at the descriptive table on 2026-08-07 — including the observation that a
3.22 M mobile arm showed the study's highest bottom-decile recall. That label
travels into the output CSV, because a hypothesis generated from the data is not
the same object as one written down before it.

---

## Two miss definitions, kept separate throughout

```
zero_dice    dice == 0              <- the published endpoint (handbook §1)
empty_pred   pred_positive == 0     <- what the per-seed tables count
wrong_place  = zero_dice - empty_pred
```

`wrong_place` is a model outputting a substantial region **entirely in the wrong
place**: 0 Dice while predicting thousands of pixels. On the current lineage
`fastscnn_rgkd` has 6 zero-Dice images and 1 empty prediction — five of its six
failures are confident errors, which is a worse clinical failure than predicting
nothing and is invisible if the two are collapsed. That distinction does not need
a bootstrap to be worth reporting.

---

## Rates, never counts

A cluster-bootstrap draw does not contain a fixed number of images — resampling
subjects with replacement changes the row count every draw. Bootstrapping a
**count** would measure how many rows the draw happened to contain. Everything
resampled in §6–§8 is a rate or a mean.

Only subjects with at least one image *in the stratum* are resampled. Including
the others would add draws contributing zero rows, inflating the variance for
reasons unrelated to the data.

---

## Reading the result

### If §7 clears zero on `zero_dice_rate`

Models that are statistically indistinguishable on mean Dice **separate on
small-lesion miss containment.** That is the headline this study has been
missing: mean Dice sits at the annotation ceiling and cannot move, this does. It
also names the lever, and the queued GPU work becomes justified.

### If §7 spans zero everywhere

The premise behind the whole queue is not established at this sample size:

- **Do not** run Stage N's seg arms, ALS→WL distillation or a Fenwick merge
  expecting to *measure* a small-lesion improvement — the instrument cannot see it.
- **Do** report §8. *"An effect below X is invisible at 28 subjects"* is a
  publishable power result and is the honest case for collecting more data.
- The Fenwick merge moves to the **top** of the queue, because its value is then
  sample size rather than label quality.

---

## What to paste back

§7's verdict block and §8's table. Those two are the whole result.

---

## Where the numbers land

`LESION_SIZE_RESULTS/` at the bundle root — never `results/`, `FINAL_RESULT/` or
`_work/runs/`. Same isolation as Stages M and N.

| file | what |
|---|---|
| `lesion_size_contrasts.csv` | **the answer** |
| `lesion_size_power.csv` | minimum detectable effect per endpoint |
| `lesion_size_headline.csv` | one row per model: whole set, D1–D4, D1 |
| `lesion_size_by_decile.csv` | one row per (model, decile) |
| `lesion_size_marginal_ci.csv` | per-model CIs in the primary stratum |
| `lesion_size_contrasts_d1.csv` | the same on D1 alone — fragile, 19 images |
| `lesion_size_bins.csv` | the decile assignment, so the cut is reproducible |
"""


def main() -> int:
    missing = [s for s, _, _ in MEMBERS if not (ROOT / s).exists()]
    if missing:
        raise SystemExit("not built -- run 71_generate_lesion_size_notebook.py first. "
                         "Missing:\n  " + "\n  ".join(missing))

    build = (ROOT / "scripts" / "60_build_unified_bundle.py").read_text(encoding="utf-8")
    if "lesionsize.py" in build:
        raise SystemExit(
            "60_build_unified_bundle.py now copies lesionsize.py, so it has a "
            "source of truth in scripts/unified_lib/ and this overlay is stale.")
    if (ROOT / "scripts" / "unified_lib" / "lesionsize.py").exists():
        raise SystemExit(
            "scripts/unified_lib/lesionsize.py exists -- two copies, no rule "
            "about which wins. Pick one before shipping.")

    # The pre-registration and the statistics, asserted at BUILD time as well as
    # at run time. An archive that ships a bootstrap crediting the wrong
    # direction, or a per-model bin cut, is an archive whose every number is
    # wrong in a way no reader can see.
    r = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, r'%s'); "
         "from bruisekit import lesionsize as LS; "
         "raise SystemExit(0 if LS.self_test(verbose=False) else 1)"
         % str(ROOT / "BRUISE_UNIFIED")],
        capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"lesionsize.self_test failed -- refusing to ship.\n"
                         f"{r.stdout}\n{r.stderr}")
    print("lesionsize.self_test passed (global binning, paired sign, CI half-width)")

    nb = json.loads((ROOT / "BRUISE_UNIFIED" / "bruise_lesion_size.ipynb")
                    .read_text(encoding="utf-8"))
    n_code = 0
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            compile("".join(c["source"]), f"cell{i}", "exec")
            n_code += 1
    print(f"notebook parses: {len(nb['cells'])} cells, {n_code} code cells compile")

    hb = (ROOT / "BRUISE_UNIFIED" / "PROJECT_HANDBOOK.md").read_text(encoding="utf-8")
    if "Stage P" not in hb:
        raise SystemExit("PROJECT_HANDBOOK.md has no Stage P section -- "
                         "the overlay would ship a notebook the handbook cannot explain.")

    t0 = time.time()
    total = sum((ROOT / s).stat().st_size for s, _, _ in MEMBERS)
    print(f"\nzipping {len(MEMBERS)} files, {total / 1e6:.2f} MB -> {DST.name}")
    print("archive paths are relative to BRUISE_UNIFIED/ -- extract INSIDE the "
          "bundle, no -d\n")

    with zipfile.ZipFile(DST, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.writestr("README_LESION_SIZE.md", README)
        for src, arc, why in MEMBERS:
            zf.write(ROOT / src, arc)
            print(f"  {arc:<52} {why}")

    with zipfile.ZipFile(DST) as zf:
        bad = zf.testzip()
        names = zf.namelist()
    if bad:
        raise SystemExit(f"archive is corrupt at {bad}")
    for required in ("bruisekit/lesionsize.py", "bruise_lesion_size.ipynb"):
        if required not in names:
            raise SystemExit(f"{required} did not make it into the archive")

    print(f"\nDONE  {DST}\n      {DST.stat().st_size / 1e6:.2f} MB, {len(names)} "
          f"entries, integrity check passed ({time.time() - t0:.1f}s)")
    print("\n  1. cd /scratch/tbommawa/BRUISE_UNIFIED && unzip -o BRUISE_LESION_SIZE.zip")
    print("  2. restart the kernel, open bruise_lesion_size.ipynb")
    print("  3. run top to bottom -- no GPU needed -- and paste §7 and §8 back")
    return 0


if __name__ == "__main__":
    sys.exit(main())
