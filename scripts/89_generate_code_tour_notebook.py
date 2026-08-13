#!/usr/bin/env python
"""Emit `bruise_code_tour.ipynb` -- a click-through tour of the implementation.

    python scripts/89_generate_code_tour_notebook.py

WHAT IT IS FOR
---------------
Sitting with a supervisor who asks "show me how you actually did X". Each cell
answers one such question by PRINTING THE REAL SOURCE out of `bruisekit/`, with
true line numbers and a `file:line` header that Jupyter and VS Code turn into a
clickable link.

NOTHING IS REIMPLEMENTED HERE, and that is the whole point. A tour notebook that
retyped the loss would be a second copy that drifts, and the first time it drifted
it would be showing a supervisor code that is not what ran. `show()` reads the
file on disk every time the cell is executed, so the notebook cannot be stale --
if the tour prints it, that is what trains.

WHY A GENERATOR RATHER THAN A HAND-WRITTEN NOTEBOOK
-----------------------------------------------------
Same discipline as every other notebook here: it is an OUTPUT, ships with zero
executed cells, and the anchors are checked at build time. If a function is
renamed, this script FAILS and names the anchor rather than emitting a notebook
whose cells raise in front of an audience.

The questions are grouped in the order they usually get asked -- data, metric,
loss, distillation, foundation models, training, thresholding, analysis, and the
guards -- so the notebook can also just be read top to bottom.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "BRUISE_UNIFIED"
DST = BUNDLE / "bruise_code_tour.ipynb"

CELLS: list[tuple[str, str]] = []


def md(src: str) -> None:
    CELLS.append(("markdown", src.strip("\n")))


def code(src: str) -> None:
    CELLS.append(("code", src.strip("\n")))


def tour(question: str, why: str, calls: list[tuple[str, str]]) -> None:
    """One question, then the cells that answer it from source."""
    md(f"### {question}\n\n{why}")
    code("\n".join(f'show("{f}", r"{a}")' for f, a in calls))


# ═════════════════════════════════════════════════════════════════════════════
# the anchors -- (file, regex). Checked at build time.
# ═════════════════════════════════════════════════════════════════════════════
TOUR: list[tuple[str, str, list[tuple[str, str]]]] = [
    ("How is the data split, and can a patient appear in both halves?",
     "Splits are grouped by **subject**, never by image. Two photographs of one "
     "bruise cannot land on opposite sides of the split — that is the single "
     "easiest way to inflate a segmentation result, and the check is re-asserted "
     "at build time and again in every notebook.",
     [("bruisekit/data.py", r"^class BruiseDataset"),
      ("bruisekit/data.py", r"def build_augmentation")]),

    ("How is accuracy measured?",
     "Dice per image, then averaged — never pooled over the batch. Pooling lets "
     "one large bruise dominate. Note the both-empty case scores 1.0, and the "
     "sweep in a later cell matches that convention exactly.",
     [("bruisekit/metrics.py", r"def dice_np"),
      ("bruisekit/evaluate.py", r"def evaluate_at_cut")]),

    ("What is the loss function?",
     "Dice + BCE. Dice is computed **per image** and then averaged, for the "
     "reason in the docstring.",
     [("bruisekit/losses.py", r"^class DiceBCELoss"),
      ("bruisekit/losses.py", r"^class SupervisedLoss")]),

    ("How does knowledge distillation work here?",
     "`alpha * DiceBCE(student, ground truth) + (1-alpha) * BCE(student, "
     "calibrated teacher probability)`. **Read the module docstring first** — it "
     "states explicitly that this is calibrated soft-target distillation and "
     "*not* Hinton KD, and why that distinction matters for the write-up.",
     [("bruisekit/losses.py", None),
      ("bruisekit/losses.py", r"^class DistillLoss")]),

    ("Why is the teacher calibrated, and how?",
     "An uncalibrated teacher's soft label is the hard label with extra steps. "
     "Temperature is fitted by NLL on validation (Guo et al. 2017). The code "
     "never falls back to T = 1.",
     [("bruisekit/engine.py", r"def calibrate_temperature")]),

    ("How do you attach a segmentation head to a model that has none?",
     "Foundation encoders output a grid of features, not a mask. This ~1M-"
     "parameter decoder turns a 40×40 feature grid into a full-resolution mask. "
     "**Stages N3, N4 and O all import this same class** rather than re-typing "
     "it, or their numbers would not be comparable.",
     [("bruisekit/finetune_n3.py", r"^class ConvDecodeHead"),
      ("bruisekit/foundation.py", r"^class LinearProbeHead")]),

    ("How do you freeze the encoder?",
     "Two things, and the second is the one people forget: `requires_grad=False` "
     "**and** pinning the module to `eval()`. Freezing weights alone still lets "
     "BatchNorm statistics and dropout drift.",
     [("bruisekit/dermprobe.py", r"def _freeze")]),

    ("How many layers did you actually train?",
     "The last **6 of 12** transformer blocks plus the final normalisation "
     "layer — about half the encoder — identical for DINOv2, DermLIP, SAM and "
     "MedSAM so the comparison is fair. Note the function **raises** if it ends "
     "up with zero trainable parameters: an arm that silently trains nothing "
     "scores low-to-mid and is indistinguishable from a real result.",
     [("bruisekit/samprobe.py", r"^UNFREEZE_BLOCKS = "),
      ("bruisekit/finetune_n3.py", r"def unfreeze_last"),
      ("bruisekit/samprobe.py", r"def unfreeze_last")]),

    ("How did you use MedSAM without giving it a prompt?",
     "SAM and MedSAM are promptable — you point at the thing you want. Our "
     "pipeline is automatic, so we keep the image encoder and discard the prompt "
     "encoder and mask decoder. The class docstring lists the three things SAM "
     "does that no other encoder in the study does, each of which would silently "
     "corrupt the arm if mishandled.",
     [("bruisekit/samprobe.py", r"^class SamViTProbe"),
      ("bruisekit/samprobe.py", r"def resample_pos_embed_2d")]),

    ("What does the training loop look like?",
     "One driver for every model in the study. A bespoke loop for any arm would "
     "make its numbers unreadable against the rest. It is idempotent and "
     "resumable — see the RESUME CONTRACT in the docstring.",
     [("bruisekit/engine.py", r"def train_run")]),

    ("What learning rate, and why two of them?",
     "6e-5 for the pretrained backbone, 6e-4 for the randomly-initialised head. "
     "Group membership is decided by `id()`, not by parameter name — a "
     "name-prefix rule would put every YOLO parameter in the wrong group.",
     [("bruisekit/models.py", r"def build_param_groups"),
      ("bruisekit/engine.py", r"def lr_multiplier")]),

    ("How do you choose the decision threshold?",
     "Swept over 481 cuts on **validation** and applied once to test. It is "
     "**not** the argmax: these sweeps are flat, so every cut within one "
     "standard error of the peak is statistically tied, and the tie is broken by "
     "**lowest complete-miss rate**.",
     [("bruisekit/sweep.py", r"def sweep_cuts"),
      ("bruisekit/sweep.py", r"def select_cut")]),

    ("How do you count a 'complete miss', and why three columns?",
     "`dice == 0` is the union of two clinically different failures: the model "
     "found nothing, or it outlined the wrong place. `wrong_place` is **derived** "
     "so the three columns cannot fail to add up, and the function raises if a "
     "table is internally inconsistent.",
     [("bruisekit/itakd.py", r"def _miss_counts")]),

    ("How do you test fairness across skin tones?",
     "Every gap is reported twice — overall and within the small-lesion stratum "
     "— because lesion size is confounded with skin tone in this test set. Cells "
     "with fewer than five patients get no confidence interval rather than a "
     "number nobody should trust.",
     [("bruisekit/lesionsize.py", r"def assign_bins"),
      ("bruisekit/lesionsize.py", r"def fairness_conditioned")]),

    ("How does multi-teacher distillation work?",
     "Two variants. Stage M routes **per image** on each teacher's soft Dice "
     "against the label; Stage O routes **per skin-tone group** on weights fitted "
     "once on validation. Both reduce exactly to the single-teacher loss when "
     "K = 1, which is what makes the contrast one-variable — and both are "
     "asserted to do so in `self_test`.",
     [("bruisekit/multiteacher.py", r"def _build_routed_loss_class"),
      ("bruisekit/itakd.py", r"def _build_group_loss_class")]),

    ("How does the model know which skin-tone group an image is in?",
     "`engine.train_run` iterates `(x, y, _)` and **discards the stem**, so the "
     "loss has no idea which images it is looking at. Rather than edit the "
     "shared training loop, the training loader is wrapped to record each "
     "batch's group indices. The loss **raises** if that record is missing — a "
     "silent fallback to uniform weights would turn the arm into a plain "
     "ensemble and report a plausible number for a different experiment.",
     [("bruisekit/itakd.py", r"^class _GroupTaggingLoader"),
      ("bruisekit/itakd.py", r"def install_group_shim")]),

    ("How do you decide whether an experiment is worth running?",
     "A pre-registered gate, computed on **validation only**, written to disk "
     "before anything touches test. Stage O's adds a clause the earlier gates "
     "lacked: it refuses unless the thing being routed on is actually estimable.",
     [("bruisekit/itakd.py", r"def ita_group_gate"),
      ("bruisekit/itakd.py", r"def identifiability")]),

    ("How do you know the numbers reproduce?",
     "Every model can be re-scored from its checkpoint and compared against the "
     "table its original run wrote. Note `resolve_runs`: the best seed is **not** "
     "the same for every model, and scoring one at another's best seed shows "
     "per-image gaps up to 0.49 Dice while looking exactly like a broken "
     "pipeline.",
     [("bruisekit/inference.py", r"def resolve_runs"),
      ("bruisekit/inference.py", r"def reconcile")]),

    ("How does an experiment change behaviour without editing the pipeline?",
     "It patches a name. Every shim falls through to whatever was bound before "
     "it when its arm is not active, so a session that trains several stages "
     "keeps them separate. Note that `build_model` is rebound in **two** modules "
     "— `engine` bound it by value at import, so patching one is not enough.",
     [("bruisekit/samprobe.py", r"def install_n4_shim"),
      ("bruisekit/itakd.py", r"def install_loss_shim")]),
]


# ═════════════════════════════════════════════════════════════════════════════
md("""
# Code tour — where everything actually lives

Every cell below **prints the real source** out of `bruisekit/`, read from disk
at the moment you run it. Nothing here is a copy: if this notebook shows it, that
is the code that trains.

Each block prints a `file:line` header. In Jupyter and VS Code that is a
**clickable link** — click it to open the file at that line.

Run the setup cell once, then jump to whichever question you need. The order is
roughly the order these questions get asked: data → metric → loss → distillation
→ foundation models → training → thresholding → analysis → the guards.
""")

code('''
import inspect
import re
import textwrap
from pathlib import Path

try:
    from IPython.display import Code, Markdown, display
except ImportError:                    # plain python, or a kernel without IPython
    Code = Markdown = lambda x, language=None: x
    display = print

# The bundle root, found from wherever this notebook was opened.
ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p / "bruisekit").is_dir()), Path.cwd())
print(f"reading source from  {ROOT / 'bruisekit'}")


def _block(lines, start):
    """The def/class beginning at `start` (1-indexed), to the end of its body."""
    head = lines[start - 1]
    indent = len(head) - len(head.lstrip())
    out = [head]
    for ln in lines[start:]:
        if ln.strip() and (len(ln) - len(ln.lstrip())) <= indent:
            break
        out.append(ln)
    while out and not out[-1].strip():
        out.pop()
    return out


def show(rel, anchor, body=True, max_lines=90):
    """Print the real implementation of `anchor` in `rel`, with line numbers.

    `anchor` is a regex matched against whole lines; the first match wins. The
    file is re-read on every call, so this cannot go stale relative to the code.
    """
    path = ROOT / rel
    if not path.exists():
        display(Markdown(f"**missing:** `{rel}` — is this the bundle root?"))
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    pat = re.compile(anchor)
    hit = next((i for i, ln in enumerate(lines, 1) if pat.search(ln)), None)
    if hit is None:
        display(Markdown(f"**not found:** `{anchor}` in `{rel}` — the code was "
                         f"renamed. Regenerate this notebook with "
                         f"`scripts/89_generate_code_tour_notebook.py`, which "
                         f"fails loudly on a stale anchor."))
        return

    seg = _block(lines, hit) if body else [lines[hit - 1]]
    truncated = len(seg) > max_lines
    if truncated:
        seg = seg[:max_lines] + [f"    ...  ({len(_block(lines, hit)) - max_lines}"
                                 f" more lines — open the file to read on)"]

    display(Markdown(f"#### `{rel}:{hit}`"))
    width = len(str(hit + len(seg)))
    numbered = "\\n".join(f"{hit + k:>{width}} | {ln}" for k, ln in enumerate(seg))
    display(Code(numbered, language="python"))


def docstring(rel, anchor=None):
    """Just the module or function docstring — the WHY, without the body."""
    path = ROOT / rel
    lines = path.read_text(encoding="utf-8").splitlines()
    if anchor is None:
        txt = path.read_text(encoding="utf-8")
        m = re.search(r'"""(.*?)"""', txt, re.S)
        display(Markdown(f"#### `{rel}` — module docstring"))
        display(Markdown("```\\n" + (m.group(1).strip() if m else "") + "\\n```"))
        return
    show(rel, anchor)


print("ready — `show(file, anchor)` prints source, `docstring(file)` prints the why")
''')

md("---")

for q, why, calls in TOUR:
    tour(q, why, calls)

md("""
---

## Three rules that explain most of what you just read

1. **Nothing is fitted on test.** Thresholds, best seeds and every gate are
   decided on validation, and a gate's verdict is written to disk before any test
   pass runs.
2. **An experiment patches a name; it never edits the pipeline.** Stage modules
   are deliberately absent from the bundle build's copy list, so an experiment
   that returns nothing cannot become a dependency of the code that produces the
   main results.
3. **A function that cannot do its job raises.** It does not warn and it does not
   return a plausible default. Most `raise` statements in this codebase mark a
   place where a silent fallback once produced a believable wrong number.

## If a cell says "not found"

The code was renamed. Regenerate this notebook —
`python scripts/89_generate_code_tour_notebook.py` — which **fails at build time**
on a stale anchor rather than emitting cells that break in front of an audience.

A file-level index of the same material, with line numbers resolved at build
time, is in **`docs/CODE_MAP.md`**.
""")


def main() -> int:
    # Fail at build time on a stale anchor, rather than shipping a notebook whose
    # cells print "not found" during a supervision meeting.
    bad = []
    for _q, _w, calls in TOUR:
        for rel, anchor in calls:
            p = BUNDLE / rel
            if not p.exists():
                bad.append(f"{rel} (missing file)")
                continue
            if anchor is None:                     # module docstring, always there
                continue
            pat = re.compile(anchor)
            if not any(pat.search(ln)
                       for ln in p.read_text(encoding="utf-8").splitlines()):
                bad.append(f"{rel}  ::  {anchor}")
    if bad:
        print("FAIL: these anchors match nothing:", file=sys.stderr)
        for b in bad:
            print(f"  {b}", file=sys.stderr)
        return 1

    nb = {
        "cells": [
            {"cell_type": k, "metadata": {}, "source": s.splitlines(keepends=True)}
            | ({"outputs": [], "execution_count": None} if k == "code" else {})
            for k, s in CELLS
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    DST.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    n_anchor = sum(len(c) for _, _, c in TOUR)
    print(f"wrote {DST}")
    print(f"  {len(CELLS)} cells, {len(TOUR)} questions, {n_anchor} anchors, "
          f"all resolved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
