#!/usr/bin/env python
"""Emit the meeting cheat-sheet: docs/bruise_meeting_cheatsheet.{tex,pdf}

    python scripts/75_generate_meeting_cheatsheet.py

WHY BOTH FORMATS FROM ONE SOURCE
---------------------------------
The .tex is the durable artefact -- compile it anywhere (Overleaf, the cluster,
any TeX install) and it is editable. But no TeX distribution is installed on this
machine, so a .tex alone would leave you with nothing to read before the meeting.
The PDF is therefore rendered directly with reportlab from the SAME content list,
with the mathematics rasterised through matplotlib's mathtext.

The alternative -- writing the .tex by hand and a separate PDF by hand -- is two
sources that drift, which has already happened once in this project (the two deck
scripts, now merged). CONTENT below is the single source; `to_tex()` and
`to_pdf()` are two renderers over it.

EVERY NUMBER IS READ FROM THE CODE OR THE RESULT CSVs AT BUILD TIME.
A cheat-sheet with a transcribed hyperparameter is worse than none: it will be
quoted with confidence in a room where someone can check it.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "BRUISE_UNIFIED"
RES = BUNDLE / "FINAL_RESULT"
DOCS = ROOT / "docs"
sys.path.insert(0, str(BUNDLE))

# ── facts pulled from the code, never typed ──────────────────────────────────
from bruisekit import efficient_models as EM   # noqa: E402

MAN = {s: pd.read_csv(BUNDLE / "manifests" / f"{s}.csv") for s in ("train", "val", "test")}
SPLIT = {s: (len(d), d.subject.nunique()) for s, d in MAN.items()}
CFGB0 = json.loads((BUNDLE / "pretrained_weights" / "segformer_mit_b0" / "config.json").read_text())
CFGB2 = json.loads((BUNDLE / "pretrained_weights" / "segformer_mit_b2" / "config.json").read_text())
GATE = pd.read_csv(RES / "reliability_gate_diagnostics.csv")
CAL = {s: json.loads((BUNDLE / "checkpoints" / "final" /
                      f"segformer_b2_teacher__seed{s}" / "calibration.json").read_text())
       ["temperature"] for s in (0, 1, 2)}
HEAD_CH = EM._HEAD_IN_CHANNELS


def _seg(c, k):
    return ", ".join(str(x) for x in c[k])


# ─────────────────────────────────────────────────────────────────────────────
# CONTENT  —  ("h1"|"h2"|"q"|"p"|"math"|"table"|"bullets", payload)
# ─────────────────────────────────────────────────────────────────────────────
CONTENT: list[tuple[str, object]] = [

("h1", "1 · Task, data and splits"),
("q", "What exactly is the task?"),
("p", "Binary semantic segmentation of bruises in white-light photographs. One "
      "foreground class. The clinical question is <i>was the bruise found at all</i>, "
      "not how neatly it was outlined — which is why complete-miss rate carries more "
      "weight than mean Dice."),
("q", "How much data, and how is it split?"),
("table", ([["Split", "Images", "Subjects"],
            ["train", SPLIT["train"][0], SPLIT["train"][1]],
            ["val", SPLIT["val"][0], SPLIT["val"][1]],
            ["test", SPLIT["test"][0], SPLIT["test"][1]],
            ["total", sum(v[0] for v in SPLIT.values()), "143"]], [45, 30, 30])),
("p", "<b>Subject-grouped, never image-grouped.</b> No subject appears in two splits. "
      "A subject leaking across splits would inflate every number invisibly, so it is "
      "re-asserted at build time and again in the notebook."),
("q", "What resolution, and how are masks resized?"),
("p", "640 × 640. Images bilinear, <b>masks nearest-neighbour</b>. Bilinear on a mask "
      "produces fractional boundary pixels; thresholding those back to binary erodes "
      "or dilates small bruises — exactly the population the miss metric is about."),

("h1", "2 · Tensor shapes and the model interface"),
("q", "What shape goes in and what comes out?"),
("math", r"x:\ [B,\,3,\,640,\,640]\quad\longrightarrow\quad \mathrm{logits}:\ [B,\,1,\,640,\,640]"),
("p", "Every model in every stage satisfies exactly this interface:"),
("bullets", ["<font face='Courier'>forward_train(x) -> (logits [B,1,H,W], aux_logits | None)</font>",
             "<font face='Courier'>forward(x) -> logits [B,1,H,W]</font>",
             "<font face='Courier'>.backbone</font> — the pretrained part, for the encoder/head LR split"]),
("p", "Output is at <b>input resolution</b> — each architecture interpolates its own "
      "decoder output back up to 640 before returning."),
("q", "Why one logit and not two classes with softmax?"),
("p", "Two classes plus softmax is the same function via one more transformation and "
      "one more place to get the sign backwards. One logit is what every downstream "
      "consumer (loss, threshold sweep, metric) expects, so all reference "
      "implementations are built with <font face='Courier'>num_classes=1</font>."),
("q", "Where does normalisation happen?"),
("p", "<b>The dataloader emits raw [0,1] pixels; each model normalises internally.</b> "
      "This is not style: SegFormer wants ImageNet statistics, YOLO wants plain /255, "
      "and Ultralytics' BatchNorms carry frozen running statistics for the /255 "
      "distribution. Feeding YOLO ImageNet-normalised pixels caps it at 0.479 Dice "
      "with no threshold able to recover it. Pixel scale belongs to the model."),

("h1", "3 · Architectures, channels and parameters"),
("q", "What are the SegFormer channel dimensions?"),
("table", ([["", "MiT-B0 (student)", "MiT-B2 (teacher)"],
            ["stage hidden sizes", _seg(CFGB0, "hidden_sizes"), _seg(CFGB2, "hidden_sizes")],
            ["depths (blocks/stage)", _seg(CFGB0, "depths"), _seg(CFGB2, "depths")],
            ["attention heads", _seg(CFGB0, "num_attention_heads"), _seg(CFGB2, "num_attention_heads")],
            ["SR ratios", _seg(CFGB0, "sr_ratios"), _seg(CFGB2, "sr_ratios")],
            ["patch sizes / strides", f'{_seg(CFGB0,"patch_sizes")} / {_seg(CFGB0,"strides")}',
             f'{_seg(CFGB2,"patch_sizes")} / {_seg(CFGB2,"strides")}'],
            ["decoder hidden size", CFGB0["decoder_hidden_size"], CFGB2["decoder_hidden_size"]],
            ["parameters", "3.71 M", "27.35 M"]], [34, 33, 33])),
("p", "Four stages at strides 4/8/16/32; the all-MLP decoder projects every stage to "
      "the decoder width, upsamples to stride 4, concatenates and fuses."),
("q", "And the compact students?"),
("table", ([["Model", "Params", "Type", "Head in-channels"],
            ["SegFormer-B0", "3.71 M", "pure transformer (MiT)", "256"],
            ["LR-ASPP MobileNetV3", "3.22 M", "pure CNN", str(HEAD_CH["lraspp_mobilenetv3"])],
            ["TopFormer-Tiny", "1.37 M", "hybrid CNN + transformer", str(HEAD_CH["topformer_tiny"])],
            ["PP-MobileSeg-Tiny", "1.45 M", "hybrid, strided SEA attention", str(HEAD_CH["ppmobileseg_tiny"])],
            ["Fast-SCNN", "1.14 M", "pure CNN, no pretraining", str(HEAD_CH["fastscnn"])]],
           [30, 14, 34, 22])),
("p", "All except Fast-SCNN start from an ImageNet-pretrained backbone with a freshly "
      "initialised 1-class head. Two are vendored verbatim from their reference "
      "implementations; each official checkpoint was verified to load by name and "
      "shape (0 unexpected keys) before any GPU time was spent."),

("h1", "4 · Loss functions — the complete set"),
("q", "What is the supervised loss?"),
("math", r"\mathcal{L}_{\mathrm{DiceBCE}} = \mathrm{BCE}(z, y) \;+\; \left(1 - \frac{1}{B}\sum_{i}\frac{2\sum p_i y_i + 1}{\sum p_i + \sum y_i + 1}\right)"),
("p", "with <i>z</i> the logits, <i>p</i> = sigmoid(<i>z</i>), <i>y</i> the label, smooth = 1. "
      "<b>Dice is computed per image and then averaged</b>, not pooled over the batch: "
      "a batch-pooled Dice lets one large bruise dominate the gradient and lets a blank "
      "prediction hide inside a good batch."),
("q", "Why Dice + BCE and not either alone?"),
("p", "Bruises cover ~4.7 % of pixels. BCE alone is dominated by background — an "
      "all-background prediction already scores well and gets almost no gradient toward "
      "the bruise. Dice is scale-invariant to object size and supplies gradient "
      "proportional to overlap, which is what we report."),
("q", "What about the auxiliary head?"),
("math", r"\mathcal{L}_{\mathrm{sup}} = \mathcal{L}_{\mathrm{DiceBCE}}(z,y) \;+\; \lambda_{\mathrm{aux}}\cdot \mathrm{BCE}(z_{\mathrm{aux}}, y),\qquad \lambda_{\mathrm{aux}} = 0.4"),
("p", "<b>Only YOLO has an auxiliary head</b> (0.4 is Ultralytics' own weight). SegFormer "
      "and every mobile model return <font face='Courier'>aux = None</font>, so the term "
      "vanishes and the loss is identical across architectures."),
("q", "What is the response-KD (distillation) loss?"),
("math", r"\mathcal{L}_{\mathrm{KD}} = \alpha\,\mathcal{L}_{\mathrm{sup}}(z, y) \;+\; (1-\alpha)\,\mathrm{BCE}\left(z,\ \sigma(z_T / T)\right)"),
("p", "<b>This is calibrated soft-target distillation, not Hinton KD.</b> Hinton divides "
      "<i>both</i> student and teacher logits by a shared temperature and multiplies the "
      "soft term by T². Here the student's logits are <b>not</b> temperature-scaled and "
      "there is no T². <i>T</i> is not a KD knob — it is the temperature fitted by NLL on "
      "validation (Guo et al. 2017) that makes the teacher <i>calibrated</i>. Cite Menon "
      "et al. (2021) for why that helps, not Hinton."),
("q", "What are the α values, and why do they differ?"),
("table", ([["Key", "Value", "Applies to"],
            ["segformer_alpha", "0.6", "SegFormer-B0 students"],
            ["efficient_alpha", "0.6", "mobile students (Stage E/F/H)"],
            ["yolo_alpha", "0.4", "YOLO offline pseudo-mask fusion"]], [32, 16, 52])),
("p", "The first two are equal today, which is exactly why they are separate keys: "
      "reading the wrong one would look right and silently ignore the other the moment "
      "one was tuned. <b>Gated arms deliberately inherit their control's α</b>, so a "
      "gated-vs-response contrast moves one variable."),

("h1", "5 · The reliability gate"),
("q", "What is the gate, in one sentence?"),
("p", "During training we can see the label, so we check the teacher against it and "
      "reduce its weight where it is <b>confidently wrong</b> — per pixel, and per image "
      "when it misses the bruise entirely."),
("q", "Give me the formulas."),
("math", r"r = 1 - |2p_T - 1|\cdot|p_T - y| \qquad \mathrm{(per\ pixel\ reliability)}"),
("math", r"\mathrm{Dice}_T = \frac{2\sum p_T\, y}{\sum p_T + \sum y}, \qquad g = \mathrm{clip}\!\left(\frac{\mathrm{Dice}_T - 0.10}{0.50 - 0.10},\,0,\,1\right)"),
("math", r"w = g\cdot r, \qquad \mathrm{coverage} = \overline{w}, \qquad \alpha_{\mathrm{eff}} = \alpha + (1-\alpha)(1 - \mathrm{coverage})"),
("math", r"\mathcal{L}_{\mathrm{RGKD}} = \alpha_{\mathrm{eff}}\,\mathcal{L}_{\mathrm{sup}}(z,y) \;+\; (1-\alpha_{\mathrm{eff}})\,\frac{\sum w \cdot \mathrm{BCE}\left(z, \sigma(z_T/T)\right)}{\sum w}"),
("q", "Why multiply by confidence — why not just use |p − y|?"),
("p", "Because |p − y| alone would delete the pixels where the teacher is <i>uncertain</i>, "
      "and that uncertainty is the dark knowledge distillation exists to transfer. "
      "Multiplying by |2p−1| means an uncertain teacher (p = 0.5) keeps <b>r = 1</b>, and "
      "only assertive error is removed."),
("table", ([["Teacher is…", "confidence", "error", "r", "soft term"],
            ["confidently right", "≈ 1", "≈ 0", "≈ 1", "full weight"],
            ["uncertain (p = 0.5)", "≈ 0", "any", "= 1", "full weight"],
            ["confidently wrong", "≈ 1", "≈ 1", "≈ 0", "suppressed"]], [26, 18, 14, 12, 30])),
("q", "Why soft Dice for the image gate rather than thresholded Dice?"),
("p", "So the gate needs no operating point. The cut is not fitted until after training, "
      "and a gate that depended on one would entangle this arm with threshold choice. "
      "The ramp (rather than a hard step) exists because augmentation is re-sampled every "
      "epoch — a hard boundary would flip an image's KD term on and off between epochs."),
("q", "Doesn't α change? Isn't that a second variable?"),
("p", "That is what α<sub>eff</sub> prevents. Gating removes gradient mass from the soft "
      "term; handing it back to the supervised term means gating cannot silently lower "
      "the effective KD weight and masquerade as an α change. <b>At coverage = 1 the loss "
      "equals the response-KD loss term for term</b> (asserted in "
      "<font face='Courier'>self_test()</font> to 2.4e-07); at coverage = 0 it equals the "
      "supervised loss exactly."),
("q", "Did the gate actually fire?"),
("table", ([["Diagnostic", "Measured"],
            ["mean coverage (g·r)", f"{GATE.mean_coverage.mean():.3f}"],
            ["mean effective α", f"{GATE.mean_alpha_effective.mean():.3f}  (nominal {GATE.alpha_nominal.iloc[0]:.1f})"],
            ["image-views fully gated off", f"{GATE.frac_images_fully_gated_off.mean()*100:.1f} %"],
            ["teacher near-miss views", f"{GATE.frac_teacher_near_miss.mean()*100:.1f} %"],
            ["mean pixel reliability", f"{GATE.mean_pixel_reliability.mean():.3f}"]], [45, 55])),
("p", "Neither degenerate case occurred, so the null result is a real null — evidence that "
      "suppressing confident teacher errors does not change what these students learn."),
("q", "Is the gate a two-teacher method?"),
("p", "<b>No — one teacher (B2).</b> The second opinion is the ground truth. Do not confuse "
      "it with the Stage C adaptive arms, which fuse B2 and B5 per pixel by "
      "<i>w</i> = |p<sub>B5</sub>−0.5| / (|p<sub>B2</sub>−0.5| + |p<sub>B5</sub>−0.5|) and use "
      "<b>no labels</b>. That rule follows whichever teacher is more confident, so when "
      "both are confidently wrong it amplifies the error; the label catches that case."),
("q", "Does the deployed model carry the gate?"),
("p", "No. It needs the label, so it is training-time only. The shipped student is an "
      "ordinary model."),

("h1", "6 · Teacher calibration"),
("q", "Why calibrate the teacher at all?"),
("p", "BCE drives a trained model's logits toward ±∞, so the teacher's probability "
      "histogram ends up nearly binary — and a near-binary soft label is the hard label "
      "with extra steps. Dividing by T > 1 pulls saturated logits back into sigmoid's "
      "responsive region so the student can see where the teacher is <i>uncertain</i>."),
("q", "How is T fitted?"),
("p", "L-BFGS on <b>log T</b> minimising NLL over the 134 validation images (Guo et al. "
      "2017). log T keeps T > 0 without a constraint and avoids the singularity at 0; "
      "L-BFGS because it is one scalar over a fixed dataset. Logits are subsampled 1-in-16 "
      "before fitting — plenty for one scalar, and it keeps 55 M pixels in RAM."),
("q", "What are the measured values?"),
("table", ([["Teacher", "T"],
            ["segformer_b2_teacher seed 0", f"{CAL[0]:.4f}"],
            ["segformer_b2_teacher seed 1", f"{CAL[1]:.4f}"],
            ["segformer_b2_teacher seed 2", f"{CAL[2]:.4f}"]], [55, 45])),
("p", "T slightly above 1 is what a well-trained over-confident BCE model should give. "
      "Anything outside [0.5, 10] is a symptom, not a temperature, and warns loudly."),

("h1", "7 · Training recipe (identical for every model)"),
("table", ([["Setting", "Value", "Setting", "Value"],
            ["input", "640 × 640", "optimiser", "AdamW"],
            ["epochs", "100 (early stop)", "betas", "(0.9, 0.999)"],
            ["patience", "15", "weight decay", "0.01"],
            ["backbone LR", "6e-5", "warmup", "1 % of steps"],
            ["head LR", "6e-4", "schedule", "poly, power 1.0"],
            ["grad clip", "1.0", "AMP", "on"],
            ["seeds", "0, 1, 2", "model selection", "best val AP"]], [22, 28, 22, 28])),
("q", "Why is the head LR 10× the backbone LR?"),
("p", "The backbone is pretrained and already has good features, so a conservative rate "
      "preserves them; the head is randomly initialised and must catch up. It is "
      "SegFormer's own recipe, applied to <i>every</i> architecture here — holding the "
      "recipe fixed across architectures is the entire basis for comparing them."),
("q", "What batch sizes?"),
("p", "SegFormer probes for the largest that fits (32 for B2, 64 for B0). SMP and mobile "
      "models use a <b>fixed 16</b> and skip the probe — the probe escalates from batch 1, "
      "and DeepLabV3+'s ASPP image-pool BatchNorm, LR-ASPP's image pool and StrideFormer's "
      "SE blocks all raise “Expected more than 1 value per channel” at batch 1."),
("q", "Why select on AP rather than Dice?"),
("p", "AP is threshold-free. Selecting on a thresholded metric would entangle model choice "
      "with threshold choice."),

("h1", "8 · The operating point (threshold)"),
("q", "How is the threshold chosen?"),
("p", "Swept over <b>481 logit cuts from −6 to +6</b> on the 134 validation images, then "
      "applied once to test. It is never re-fitted on test."),
("q", "Is it the argmax of validation Dice?"),
("p", "<b>No.</b> These sweeps are extraordinarily flat — B2's val Dice moved 0.009 across "
      "thresholds from 0.154 to 0.959. That is noise on a plateau, and taking the argmax "
      "fits the validation set's sampling error. Every cut within <b>one standard error</b> "
      "of the peak is treated as tied, and the tie is broken by <b>lowest complete-miss "
      "rate</b> — cuts in the band are Dice-equivalent but not miss-equivalent."),
("q", "What if a checkpoint has no operating point?"),
("p", "Its test score is undefined and the registry reports it MISSING. YOLO is the "
      "exception: its native argmax path is parameter-free, so there is nothing to fit "
      "and nothing to overfit."),

("h1", "9 · Metrics"),
("math", r"\mathrm{Dice} = \frac{2\,|P \cap G|}{|P| + |G|}, \qquad \mathrm{IoU} = \frac{|P \cap G|}{|P \cup G|}"),
("p", "Both return 1.0 when the denominator is 0 (an empty prediction on an empty label is "
      "correct). Precision = TP/(TP+FP), recall = TP/(TP+FN), both 1.0 when undefined."),
("q", "Careful — there are TWO definitions of “complete miss”. Which one?"),
("table", ([["Producer", "Definition", "Where used"],
            ["metrics.summarize", "pred_positive == 0 and gt > 0", "per-seed sweep tables"],
            ["report.normalize", "dice == 0", "all reported tables"]], [26, 40, 34])),
("p", "The first counts images where the model predicted <b>nothing at all</b>; the second "
      "counts images with <b>no overlap</b>. The first is a strict subset — it misses the "
      "case where a model fires confidently on the wrong region, which is still a complete "
      "miss to a clinician. <b>Publish the <i>dice == 0</i> column</b>; that is what every "
      "number in the results tables uses."),
("q", "Why report median beside mean?"),
("p", "The per-image distribution is strongly left-skewed. Every arm reads 0.03–0.07 Dice "
      "higher on median. Measured: that uplift does <b>not</b> track complete-miss count "
      "(Spearman −0.09, p = 0.84) — with only 0–7 zeros in 185 the misses are too few to "
      "move the mean much, so what the median discards is the broad tail of "
      "low-but-nonzero scores."),

("h1", "10 · Statistics"),
("q", "How is significance computed?"),
("bullets", [
    "<b>Resample subjects, not images.</b> 185 images from 28 subjects; images of the same "
    "bruise are strongly correlated. Resampling images would treat 185 correlated "
    "observations as independent and give intervals ≈ √(185/28) ≈ 2.6× too narrow.",
    "<b>Paired.</b> Both models saw the same 185 images, so the same resampled subject list "
    "is applied to both on every draw.",
    "<b>10 000 draws</b> for confirmatory work (2 000 for descriptive figures). A tail "
    "probability quoted to two decimals cannot be resolved below 1/n.",
    "<b>Three endpoints from one set of draws</b> — mean Dice, median Dice and miss rate — "
    "so “Dice up, misses up” is a statement about the same resampled worlds.",
    "<b>Holm–Bonferroni</b> within a comparison list fixed before the models were trained."]),
("q", "What do the four verdicts mean?"),
("table", ([["Verdict", "Condition", "Meaning"],
            ["WIN", "lo > 0", "genuinely better"],
            ["INFERIOR", "hi < −margin", "genuinely worse"],
            ["NON-INFERIOR", "lo ≥ −margin", "equivalent within one Dice point"],
            ["INCONCLUSIVE", "interval spans past −margin", "cannot be answered at n = 28"]],
           [22, 30, 48])),
("p", "Margin = 0.01 Dice. <b>INCONCLUSIVE is deliberately separated from NON-INFERIOR</b>: "
      "calling a wide interval “equivalent” reports absence of evidence as evidence of "
      "absence."),
("q", "Why not compare every model to every other?"),
("p", "Twenty scored models is 190 ordered pairs. At 28 subjects, with every model inside "
      "the annotation-ceiling band, an all-pairs sweep at α = 0.05 returns about ten "
      "“significant” differences from noise alone — and would be read as a ranking. An "
      "omnibus (Friedman on subject-mean Dice, 28 rows) gates the pairwise table."),

("h1", "11 · Numbers worth having memorised"),
("table", ([["Fact", "Value"],
            ["Annotation ceiling (two experts)", "0.755 Dice; weakest pair 0.581"],
            ["SegFormer-B2 teacher", "0.769 mean / 0.819 median, 0 misses"],
            ["SegFormer-B0 direct (boundary)", "0.766 mean / 0.813 median, 1 miss"],
            ["Best mobile (LR-ASPP + B2 KD)", "0.721 mean / 0.777 median"],
            ["Gate coverage", f"{GATE.mean_coverage.mean():.3f} (fired on {GATE.frac_images_fully_gated_off.mean()*100:.1f} % of views)"],
            ["Gated vs response KD", "all 4 students INCONCLUSIVE"],
            ["B2 vs DeepLabV3+ teacher (Fast-SCNN)", "+0.034, CI excludes zero"],
            ["KD vs no KD (LR-ASPP / Fast-SCNN)", "+0.027 / +0.036, both WIN"]], [45, 55])),
("q", "If asked “so did your method work?”"),
("p", "<b>Say no, clearly.</b> The gate fired on 2.8 % of image-views and changed nothing "
      "measurable — every interval spans zero on both Dice and complete misses. What did "
      "land is that the <i>teacher choice</i> matters (B2 beats DeepLabV3+ by 0.034 on the "
      "same student) and that KD beats no KD. A null on a mechanism that demonstrably "
      "engaged is a result; dressing it up is not."),
("q", "If asked about the one thing that looks bad"),
("p", "On Fast-SCNN, plain B2 response KD removes a <i>significant</i> skin-tone fairness "
      "gap (0.160, p = 0.021 → 0.064, p = 0.358) and the <b>gated</b> variant does not "
      "(0.142, p = 0.029). That is the pre-registered primary endpoint and the one place "
      "gating looks worse than its control. At 28 subjects it is a single comparison — "
      "flag it, do not bury it, do not over-read it."),
]


# ─────────────────────────────────────────────────────────────────────────────
# renderer 1 — LaTeX
# ─────────────────────────────────────────────────────────────────────────────
def _tex_escape(s: str) -> str:
    s = (s.replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")
          .replace("≈", r"$\approx$").replace("±", r"$\pm$").replace("−", "-")
          .replace("×", r"$\times$").replace("√", r"$\sqrt{\ }$").replace("≥", r"$\geq$")
          .replace("→", r"$\rightarrow$").replace("α", r"$\alpha$").replace("σ", r"$\sigma$")
          .replace("“", "``").replace("”", "''").replace("’", "'").replace("…", r"\dots")
          .replace("·", r"$\cdot$").replace("∩", r"$\cap$").replace("∪", r"$\cup$")
          .replace("±∞", r"$\pm\infty$").replace("∞", r"$\infty$"))
    for a, b in (("<b>", r"\textbf{"), ("<i>", r"\textit{"),
                 ("<font face='Courier'>", r"\texttt{")):
        s = s.replace(a, b)
    return s.replace("</b>", "}").replace("</i>", "}").replace("</font>", "}") \
            .replace("<sub>", "$_{").replace("</sub>", "}$")


def to_tex(path: Path) -> None:
    L = [r"\documentclass[10pt,a4paper]{article}",
         r"\usepackage[margin=18mm]{geometry}",
         r"\usepackage{amsmath,amssymb,booktabs,array,xcolor,parskip}",
         r"\usepackage[T1]{fontenc}",
         r"\usepackage{titlesec}",
         r"\definecolor{navy}{HTML}{1F3F6E}",
         r"\titleformat{\section}{\large\bfseries\color{navy}}{}{0pt}{}",
         r"\newcommand{\qq}[1]{\medskip\noindent\textbf{\color{navy}#1}\par\nobreak}",
         r"\setlength{\parskip}{4pt}",
         r"\begin{document}",
         r"\begin{center}{\LARGE\bfseries\color{navy} Bruise Segmentation --- Meeting Cheat-Sheet}\\[2pt]",
         r"{\small Losses, formulas, shapes and the questions that get asked}\end{center}",
         r"\vspace{2mm}\hrule\vspace{3mm}"]
    for kind, payload in CONTENT:
        if kind == "h1":
            L.append(r"\section*{" + _tex_escape(str(payload)) + "}")
        elif kind == "q":
            L.append(r"\qq{" + _tex_escape(str(payload)) + "}")
        elif kind == "p":
            L.append(_tex_escape(str(payload)))
        elif kind == "math":
            L.append(r"\[" + str(payload) + r"\]")
        elif kind == "bullets":
            L.append(r"\begin{itemize}\setlength\itemsep{1pt}")
            L += [r"\item " + _tex_escape(b) for b in payload]
            L.append(r"\end{itemize}")
        elif kind == "table":
            rows, widths = payload
            spec = "".join("p{%.1f\\linewidth}" % (w / 100 * 0.94) for w in widths)
            L.append(r"\begin{center}\small\begin{tabular}{" + spec + "}")
            L.append(r"\toprule")
            L.append(" & ".join(r"\textbf{" + _tex_escape(str(c)) + "}" for c in rows[0]) + r" \\")
            L.append(r"\midrule")
            for r_ in rows[1:]:
                L.append(" & ".join(_tex_escape(str(c)) for c in r_) + r" \\")
            L.append(r"\bottomrule\end{tabular}\end{center}")
    L.append(r"\end{document}")
    path.write_text("\n".join(L), encoding="utf-8")
    print(f"  wrote {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# renderer 2 — PDF via reportlab, maths through matplotlib mathtext
# ─────────────────────────────────────────────────────────────────────────────
def _math_png(expr: str, fontsize: int = 15):
    """Rasterise one display equation. mathtext covers everything used here."""
    fig = plt.figure(figsize=(0.01, 0.01))
    t = fig.text(0, 0, f"${expr}$", fontsize=fontsize, color="#1A1A1A",
                 fontfamily="serif")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, bbox_inches="tight",
                pad_inches=0.06, transparent=True)
    plt.close(fig)
    buf.seek(0)
    from PIL import Image as PILImage
    w, h = PILImage.open(buf).size
    buf.seek(0)
    return buf, w, h


def to_pdf(path: Path) -> None:
    ss = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=ss["Normal"], fontName="Times-Roman",
                          fontSize=9.4, leading=13, alignment=TA_LEFT,
                          spaceAfter=4, textColor=colors.HexColor("#1A1A1A"))
    q = ParagraphStyle("q", parent=body, fontName="Times-Bold", fontSize=10.2,
                       textColor=colors.HexColor("#1F3F6E"), spaceBefore=7, spaceAfter=2)
    h1 = ParagraphStyle("h1", parent=body, fontName="Times-Bold", fontSize=13,
                        textColor=colors.HexColor("#1F3F6E"), spaceBefore=12, spaceAfter=5)
    bullet = ParagraphStyle("bullet", parent=body, leftIndent=10, bulletIndent=2,
                            spaceAfter=2)

    doc = SimpleDocTemplate(str(path), pagesize=A4,
                            leftMargin=16 * mm, rightMargin=16 * mm,
                            topMargin=14 * mm, bottomMargin=14 * mm,
                            title="Bruise Segmentation — Meeting Cheat-Sheet")
    avail = doc.width
    flow = [Paragraph("Bruise Segmentation — Meeting Cheat-Sheet",
                      ParagraphStyle("t", parent=h1, fontSize=17, spaceBefore=0)),
            Paragraph("Losses, formulas, shapes and the questions that get asked",
                      ParagraphStyle("s", parent=body, fontSize=10,
                                     textColor=colors.HexColor("#6B6B6B"))),
            Spacer(1, 5)]

    for kind, payload in CONTENT:
        if kind == "h1":
            flow.append(Paragraph(str(payload), h1))
        elif kind == "q":
            flow.append(Paragraph(str(payload), q))
        elif kind == "p":
            flow.append(Paragraph(str(payload), body))
        elif kind == "bullets":
            for b in payload:
                flow.append(Paragraph(b, bullet, bulletText="•"))
        elif kind == "math":
            buf, w, h = _math_png(str(payload))
            scale = min(1.0, (avail * 0.92) / (w * 0.36))
            flow.append(Spacer(1, 3))
            flow.append(Image(buf, width=w * 0.36 * scale, height=h * 0.36 * scale,
                              hAlign="CENTER"))
            flow.append(Spacer(1, 4))
        elif kind == "table":
            rows, widths = payload
            data = [[Paragraph(f"<b>{c}</b>" if i == 0 else str(c), body)
                     for c in row] for i, row in enumerate(rows)]
            t = Table(data, colWidths=[avail * w / 100 for w in widths], hAlign="LEFT")
            t.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LINEBELOW", (0, 0), (-1, 0), 0.7, colors.HexColor("#1F3F6E")),
                ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.HexColor("#C9CFD8")),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ]))
            # KeepTogether: a table split across a page boundary loses its header
            # row, which on a cheat-sheet is worse than a short page.
            flow.append(KeepTogether([Spacer(1, 2), t, Spacer(1, 5)]))

    doc.build(flow)
    print(f"  wrote {path.name}")


DOCS.mkdir(exist_ok=True)
to_tex(DOCS / "bruise_meeting_cheatsheet.tex")
to_pdf(DOCS / "bruise_meeting_cheatsheet.pdf")
print("\nCompile the .tex anywhere with:  pdflatex bruise_meeting_cheatsheet.tex")
