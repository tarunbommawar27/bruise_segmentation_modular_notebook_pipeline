#!/usr/bin/env python
"""Score every SegFormer-B2-taught arm against the `segformer_b0_direct` boundary.

WHY THIS SCRIPT EXISTS
-----------------------
`significance.CONTRAST_FAMILY_H` answers "what does the gate do", so every row in
it is arm-vs-its-own-control. That is the right question for the experiment and
the wrong one for a deck: a reader wants one reference line and every arm placed
against it.

`segformer_b0_direct` is that line. It is the strongest model in the study that
used **no teacher at all** (0.7663 mean Dice, 1 miss of 185), so "distilled from
B2, compared with not distilling" is exactly the contrast a reader is owed. Note
what it is NOT: it is not a non-inferiority target chosen after seeing results,
and it is not the *best* model (the B2 teacher itself is, at 0.7692).

Everything numerical comes from `significance.paired_contrast_multi` -- the same
subject-level cluster bootstrap, the same 10,000 draws, the same three endpoints
on one set of resampled subject lists -- so these numbers are produced by the code
path that produced the notebook's, not by a second implementation of it.

MULTIPLICITY
-------------
Ten arms against one reference is ten comparisons, and Holm is applied across all
ten. That is deliberately stricter than the notebook's split into confirmatory and
exploratory: this table was assembled to be *read as a ranking against a boundary*,
which is precisely the reading that inflates the false-positive rate, so it gets
the correction that reading deserves.

Writes `FINAL_RESULT/significance_b2_teacher_vs_b0_direct.csv` and the forest plot
`FINAL_RESULT/figures/H1_b2_vs_b0_direct.png`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "BRUISE_UNIFIED"
RES = BUNDLE / "FINAL_RESULT"
sys.path.insert(0, str(BUNDLE))

from bruisekit import report as R          # noqa: E402
from bruisekit import significance as SG   # noqa: E402

REFERENCE = "segformer_b0_direct"

# Every arm whose soft labels came from SegFormer-B2, ordered teacher-first then by
# student size. `segformer_b0_distilled` is Stage A's plain B2->B0 response KD --
# the method all the others are variations on -- so it leads.
B2_ARMS = [
    ("segformer_b0_distilled", "SegFormer-B0", "response KD"),
    ("segformer_b0_rgkd", "SegFormer-B0", "reliability-gated KD"),
    ("lraspp_mobilenetv3_b2kd", "LR-ASPP MNv3", "response KD"),
    ("lraspp_mobilenetv3_rgkd", "LR-ASPP MNv3", "reliability-gated KD"),
    ("topformer_tiny_b2kd", "TopFormer-Tiny", "response KD"),
    ("topformer_tiny_rgkd", "TopFormer-Tiny", "reliability-gated KD"),
    ("ppmobileseg_tiny_b2kd", "PP-MobileSeg-Tiny", "response KD"),
    ("ppmobileseg_tiny_rgkd", "PP-MobileSeg-Tiny", "reliability-gated KD"),
    ("fastscnn_b2kd", "Fast-SCNN", "response KD"),
    ("fastscnn_rgkd", "Fast-SCNN", "reliability-gated KD"),
]

PARAMS_M = {"SegFormer-B0": 3.71, "LR-ASPP MNv3": 3.22, "TopFormer-Tiny": 1.37,
            "PP-MobileSeg-Tiny": 1.45, "Fast-SCNN": 1.14}


def load(name: str, meta: pd.DataFrame) -> pd.DataFrame:
    """One arm's per-image table, normalised and re-joined to the manifest.

    `normalize` recomputes the derived columns from the seven-column core rather
    than trusting whichever the producer happened to write, and always re-joins
    subject and ITA from the manifest so there is one source of truth for who each
    image belongs to (handbook 8.1).
    """
    p = RES / f"per_image_{name}.csv"
    if not p.exists():
        raise FileNotFoundError(f"{name}: {p} not found")
    return R.normalize(pd.read_csv(p), meta)


def main() -> int:
    meta = pd.read_csv(BUNDLE / "manifests" / "test.csv")
    ref = load(REFERENCE, meta)

    rows = []
    for name, student, method in B2_ARMS:
        df = load(name, meta)
        r = SG.paired_contrast_multi(df, ref, name, REFERENCE, n_boot=SG.N_BOOT_FINAL)
        s = R.summarize(df)
        rows.append({
            "arm": name, "student": student, "params_M": PARAMS_M[student],
            "method": method,
            "mean_dice": s["mean_dice"], "median_dice": s["median_dice"],
            "misses": s["complete_miss_count"],
            "delta_dice": r["delta_dice"], "lo": r["lo"], "hi": r["hi"],
            "p_two_sided": r["p_two_sided"], "p_a_better": r["p_a_better"],
            "verdict": r["verdict"],
            "delta_miss_rate": r["delta_miss_rate"],
            "n_subjects": r["n_subjects"],
        })

    out = pd.DataFrame(rows)
    # Holm across all ten -- see the module docstring on why this table gets the
    # stricter correction rather than the notebook's confirmatory/exploratory split.
    out["p_holm"] = SG.holm(out.p_two_sided.to_numpy())
    out["significant_holm"] = out.p_holm < 0.05

    ref_s = R.summarize(ref)
    print(f"reference: {REFERENCE}  mean Dice {ref_s['mean_dice']:.4f}, "
          f"median {ref_s['median_dice']:.4f}, {ref_s['complete_miss_count']} misses, "
          f"{out.n_subjects.iloc[0]} subjects\n")
    cols = ["arm", "student", "params_M", "method", "mean_dice", "median_dice",
            "misses", "delta_dice", "lo", "hi", "p_two_sided", "p_holm",
            "significant_holm", "verdict"]
    print(out[cols].round(4).to_string(index=False))

    dest = RES / "significance_b2_teacher_vs_b0_direct.csv"
    out.to_csv(dest, index=False)
    print(f"\nwrote {dest.relative_to(ROOT)}")

    _forest(out, ref_s)
    return 0


def _forest(out: pd.DataFrame, ref_s: dict) -> None:
    """Forest plot with the boundary at zero — i.e. at `segformer_b0_direct`.

    Sorted by effect so the plot reads top-to-bottom, and the two KD methods are
    given different markers rather than different colours so it survives greyscale
    printing and colour-blind readers.
    """
    d = out.sort_values("delta_dice", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9.6, 5.4))

    ax.axvline(0, color="#333333", lw=1.4, zorder=2)
    ax.text(0, len(d) - 0.35, f"  {REFERENCE}\n  (Dice {ref_s['mean_dice']:.3f})",
            fontsize=8.5, va="top", ha="left", color="#333333",
            fontfamily="Times New Roman")

    for i, r in enumerate(d.itertuples()):
        gated = r.method.startswith("reliability")
        col = "#1f3f6e" if gated else "#8a8a8a"
        ax.hlines(i, r.lo, r.hi, color=col, lw=2.6, alpha=.85, zorder=3)
        ax.scatter(r.delta_dice, i, color=col, s=62, zorder=4,
                   marker="D" if gated else "o",
                   edgecolor="white", linewidth=.6)

    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([f"{r.student} · {'gated' if r.method.startswith('reliability') else 'response'}"
                        for r in d.itertuples()], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("difference in mean Dice vs SegFormer-B0 direct\n"
                  "(paired subject-level bootstrap, 28 subjects, 10 000 draws)",
                  fontsize=9)
    ax.tick_params(labelsize=9)
    ax.grid(axis="x", alpha=.25, linestyle=":")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], color="#8a8a8a", marker="o", ls="-", lw=2.4, label="response KD"),
        Line2D([], [], color="#1f3f6e", marker="D", ls="-", lw=2.4, label="reliability-gated KD"),
    ], loc="lower right", frameon=False, fontsize=9)

    for t in ax.get_xticklabels() + ax.get_yticklabels() + [ax.xaxis.label]:
        t.set_fontfamily("Times New Roman")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(RES / "figures" / f"H1_b2_vs_b0_direct.{ext}", dpi=200,
                    bbox_inches="tight")
    print(f"wrote figures/H1_b2_vs_b0_direct.png")


if __name__ == "__main__":
    sys.exit(main())
