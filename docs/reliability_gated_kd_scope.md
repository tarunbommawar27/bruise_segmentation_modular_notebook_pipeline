# Stage H — reliability-gated distillation, and the SegFormer-B2 teacher axis

Scope note, written **before** any Stage H run existed (added 2026-08-02), with
the outcome appended in §9 **after** all 27 runs completed the same day. The
pre-registration in §4 is left exactly as written — that is the point of it.

Companion to `PROJECT_HANDBOOK.md` §7c. The handbook is the reference; this file
records what was decided, why, and what would count as the experiment failing.

---

## 1. What was asked for, and what was built

Asked: reliability-gated distillation for YOLO26n and SegFormer-B0 from the B2
teacher, and then the Stage F distillation technique — the one used with
DeepLabV3+ — rerun with B2 as the teacher into LR-ASPP, Fast-SCNN and the other
small models.

Built: eleven registered families, nine of them Stage H and two backfilling
Stage F. A twelfth, `yolo_sem_rgkd`, is implemented and tested but deliberately
**not registered** — see §1b.

| family | student | teacher | KD | control |
|---|---|---|---|---|
| `segformer_b0_rgkd` | SegFormer-B0, 3.71 M | B2 | **gated** | `segformer_b0_distilled` |
| `lraspp_mobilenetv3_rgkd` | LR-ASPP MNv3, 3.22 M | B2 | **gated** | `lraspp_mobilenetv3_b2kd` |
| `fastscnn_rgkd` | Fast-SCNN, 1.14 M | B2 | **gated** | `fastscnn_b2kd` |
| `topformer_tiny_rgkd` | TopFormer-Tiny, 1.37 M | B2 | **gated** | `topformer_tiny_b2kd` |
| `ppmobileseg_tiny_rgkd` | PP-MobileSeg-Tiny, 1.45 M | B2 | **gated** | `ppmobileseg_tiny_b2kd` |
| `lraspp_mobilenetv3_b2kd` | LR-ASPP MNv3 | B2 | response | `lraspp_mobilenetv3_distilled` |
| `fastscnn_b2kd` | Fast-SCNN | B2 | response | `fastscnn_distilled` |
| `topformer_tiny_b2kd` | TopFormer-Tiny | B2 | response | `topformer_tiny_distilled` |
| `ppmobileseg_tiny_b2kd` | PP-MobileSeg-Tiny | B2 | response | `ppmobileseg_tiny_distilled` |
| `topformer_tiny_distilled` | TopFormer-Tiny | DeepLabV3+/R50 | response | `topformer_tiny` |
| `ppmobileseg_tiny_distilled` | PP-MobileSeg-Tiny | DeepLabV3+/R50 | response | `ppmobileseg_tiny` |

The last two are Stage **F**, not H. They exist because without them the claim
"B2 is the better teacher" would rest on two students instead of four.

Cost at three seeds: Stage H **27 runs ≈ 24 GPU-hours**, Stage E/F 24 runs ≈ 15
GPU-hours. `RGKD_SEEDS` controls the first independently of `EFFICIENT_SEEDS`.

## 1b. The YOLO arm: built, not registered

**The YOLO arm is implemented but not registered.** `build_gated_yolo_dataset`
applies the gate to YOLO's offline pseudo-mask fusion and is tested, but
`yolo_sem_rgkd` is absent from `TEACHER_FOR`, `STAGE_H_FAMILIES`, `CONTROL_FOR`
and `CONTRAST_FAMILY_H`. Nothing schedules it, and a Stage E+H session therefore
never needs Ultralytics installed at all.

The cost is stated, not hidden: **every confirmatory Stage H contrast tests the
gate on an ONLINE loss**, and the study says nothing about whether it transfers to
the offline pseudo-mask route. Holm corrects over three contrasts, not four.
Re-enabling is three lines in `reliability_kd` plus the matching `registry` and
`CONTRAST_FAMILY_H` entries; the `FAMILY_SPEC` row, the notebook's YOLO training
branch and its native-argmax scoring branch are all still in place.

What this buys operationally: with Stage A's YOLO runs all at WEIGHTS tier and
no YOLO run in Stage H, `train_missing` never sees kind `yolo`, so the notebook
never attempts an Ultralytics install — which on an offline compute node is the
difference between a cell that returns instantly and one that sits in pip's
retry backoff.

## 2. The mechanism

For calibrated teacher probability `p` and label `y ∈ {0,1}`:

```
per-pixel reliability   r = 1 − |2p − 1| · |p − y|
per-image gate          g = clip((dice_T − 0.10) / 0.40, 0, 1),  dice_T = soft Dice
weight                  w = g · r
coverage                = mean(w)
alpha_eff               = alpha + (1 − alpha)(1 − coverage)
loss                    = alpha_eff · supervised(GT) + (1 − alpha_eff) · Σ(w·BCE)/Σw
```

Three properties, asserted by `reliability_kd.self_test()` rather than claimed:

1. `r ≡ 1, g ≡ 1` ⇒ the loss equals `losses.DistillLoss` (measured 2.4e-07).
2. A teacher that asserts empty on a real bruise ⇒ `g = 0` ⇒ the loss equals
   `losses.SupervisedLoss` exactly.
3. `p = 0.5` ⇒ `r = 1`. An uncertain teacher keeps full weight.

Property 3 is the one that distinguishes this from down-weighting by error. A
reliability built on `|p − y|` alone would delete exactly the pixels carrying
information the hard label does not.

Property 1 is what makes each contrast one-variable, and it is why **alpha is not
re-tuned** for the gated arms: they inherit `segformer_alpha` / `efficient_alpha` /
`yolo_alpha` from their controls. Same reasoning as the fixed LR in handbook §3.

## 3. Why B2 and not DeepLabV3+ for the gated arms

Stage F chose DeepLabV3+ on a licence argument (NVIDIA's MiT weights are
`license: other`, non-commercial) and a miss-containment argument against U-Net.
That licence argument is about *deployment*, and it is unchanged. Stage H asks a
different question — does gating fix the inherited-miss failure — and answers it
with the study's strongest teacher, because a weak teacher confounds "gating did
not help" with "there was nothing worth transferring".

The fourth cell of the 2×2 (gated + DeepLabV3+) is deliberately empty rather than
quietly omitted. 2×2×4 students is 24 arms before seeds. If gating works on B2 and
the licence argument still matters, that cell is one line per arm in
`reliability_kd.TEACHER_FOR`.

## 4. What would count as failure

Stated in advance, because the Stage F write-up (§7b.7) shows how easy it is to
report the endpoint that moved rather than the endpoint the arm was built for.

**The pre-registered primary endpoints are complete-miss rate and the ITA fairness
gap.** Mean Dice is capped by the annotation ceiling — LR-ASPP direct is already at
0.709 against `gbarimah_vs_erik` at 0.755 — so a Dice win is not the hypothesis and
must not be presented as one if it appears.

Outcomes and how each is to be reported:

| outcome | reading |
|---|---|
| misses down, fairness gap down or flat | the gate does what it was built to do |
| Dice up, misses unchanged | the same non-result Stage F got; say so in those words |
| `mean_coverage → 1.0` | the gate never fired; the arm **is** its control. Report as "B2 is reliable enough here that gating has nothing to remove" |
| `mean_coverage → 0` | the gate ate the arm; it trained as a supervised baseline. Retune `rgkd_gate_lo/hi` and retrain, do **not** reinterpret |
| nothing moves | NON-INFERIOR, reported as such (and INCONCLUSIVE if the interval is wide — handbook §8b.4) |

**`GATE_H` is read before `HEAD_H`, always.** An arm whose gate never fired and one
whose gate fired constantly are different experiments and are indistinguishable in
a Dice table.

**A known risk, flagged before the first run.** The gate is computed on the
*augmented* batch, which is the right thing to condition on but means the teacher's
soft Dice there can sit well below its 0.769 test mean. If it does, coverage
collapses and the arm degenerates. The CPU end-to-end test hit exactly this at
128 px (teacher soft Dice 0.013, coverage 0.0) — an artefact of feeding a
640-trained teacher a 128 px input, but the same shape as the real failure. Check
`mean_teacher_soft_dice` in `GATE_H` on the first real run before trusting anything
else in the stage.

## 5. Multiplicity

Stage H has its **own** confirmatory family, `reliability_kd.CONTRAST_FAMILY_H`
(3 confirmatory + 7 exploratory), Holm-corrected within itself and reported in the
notebook as `FAM_H`, separate from `significance.CONTRAST_FAMILY` / `FAM`.

Appending these to the existing family would re-penalise a comparison that has
already been conducted and reported — the Stage F LR-ASPP arm at Holm p = 0.0042
over k = 3 — for reasons having nothing to do with it. Multiplicity control is over
the comparisons made to answer one question; this is a different question.

For the same reason `OMNIBUS_SETS["mobile_field"]` was **not** extended with the
two new DeepLabV3+ arms: that omnibus has been run and quoted (χ² = 16.26,
p = 0.0027, W = 0.145) and adding models to it would silently change a published
number. Three new sets were added instead: `teacher_axis_lraspp`,
`teacher_axis_fastscnn`, `gated_arms`.

## 6. Where the code is

| file | what |
|---|---|
| `scripts/unified_lib/reliability_kd.py` | everything new — the gated loss, both shims, the gated YOLO pseudo-mask builder, `CONTRAST_FAMILY_H`, `self_test` |
| `scripts/unified_lib/loaders.py` | 12 `FAMILY_SPEC` rows, `EFFICIENT_FAMILIES` extended |
| `scripts/unified_lib/registry.py` | `_scan_stage_h`, `STAGE_H_FAMILIES`, `rgkd_seeds`, costs |
| `scripts/unified_lib/distill_efficient.py` | 2 new DeepLabV3+ arms, B2 in `TEACHER_STAGE_DIR` |
| `scripts/unified_lib/significance.py` | 3 new omnibus sets |
| `scripts/61_generate_unified_notebook.py` | Stage H cells, G2b, save-cell entries |

Nothing in `engine.py`, `losses.py` or `yolo_native.py` was edited: they are build
artefacts extracted verbatim from `bruise_colab_baselines.ipynb` (handbook §16), so
an edit there is reverted by the next `60_build_unified_bundle.py`. The two shims
follow the pattern `distill_efficient` and `efficient_models` already use, and both
are inert outside a `reliability_kd.arm()` context — so installing them cannot
change a Stage A/B/C/E/F number.

## 7. How to run it

```python
RUN_STAGES = "EH"          # Stage H's controls live in E; H alone is not enough
ALLOW_TRAINING = True
EFFICIENT_SEEDS = (0, 1, 2)
RGKD_SEEDS = None          # None = same as EFFICIENT_SEEDS. Keep it that way:
                           # a 3-seed arm against a 1-seed control is not a contrast.
```

Or one arm at a time, without the controls:

```python
RK.train_arm(env, "lraspp_mobilenetv3_rgkd", (0, 1, 2), CFG, MAN640)
```

`train_arm` trains but does not threshold-sweep or score. Re-run the notebook's
Stage H cells afterwards, or use `train_missing`, which does both and skips
anything with a `DONE.json`.

## 8. Verification already done

- `reliability_kd.self_test()` — 5 checks, all pass on CPU in under a second.
- A CPU end-to-end mini-run (6 images, 128 px, 1 epoch) of `fastscnn_rgkd` and
  `fastscnn_b2kd`: teacher shim → loss shim → `train_run` → gate stats →
  `cache_logits` → `sweep_cuts` → `select_cut` → `evaluate_at_cut` → registry
  re-scan, all executing. Handbook §15 trap 11 is precisely a cell in this position
  shipping broken because only a GPU run would have touched it.
- Registry scan: 27 Stage H runs and 24 Stage E runs resolve, all MISSING, with
  honest cost estimates.
- Shim isolation: gated arm → gated loss, control arm → `losses.DistillLoss`,
  outside `arm()` → `losses.DistillLoss`, and installing twice does not wrap the
  wrapper.

## 9. Outcome, scored against §4

All 27 runs completed 2026-08-02. Full tables in handbook §7c.11–7c.15.

**The gate fired.** Coverage 0.906, effective α 0.638 against a nominal 0.600,
2.8 % of image-views fully gated off, 1.9 % teacher near-misses. Neither of the two
degenerate outcomes §4 warned about occurred, so the null below is a real null.

**§4's row-by-row verdict:**

| §4 outcome | happened? |
|---|---|
| misses down, fairness gap down or flat | **no** — misses flat or slightly up; gating added 2 on B0 and removed none |
| Dice up, misses unchanged | **no** — Dice did not move either |
| `mean_coverage → 1.0` (gate inert) | no — 0.906 |
| `mean_coverage → 0` (gate ate the arm) | no |
| **nothing moves → NON-INFERIOR / INCONCLUSIVE** | **yes — all five students, all INCONCLUSIVE** |

So the pre-registered hypothesis was **not** supported, and it failed in the way §4
listed last: nothing moved. Confirmatory deltas −0.0024 / +0.0033 / −0.0087, every
interval spanning zero, no Holm-adjusted p below 1.0. The two mobile gate contrasts
are also sign-inconsistent across seeds (1 of 3 positive).

**What landed instead, none of it about the gate:**

- `fastscnn_b2kd` − `fastscnn_distilled` = **+0.0343** [+0.0084, +0.0593] — a B2
  teacher beats DeepLabV3+ on the same student, reversing Stage F's null.
- `lraspp_mobilenetv3_rgkd` − `lraspp_mobilenetv3` = **+0.0265** [+0.0065, +0.0517].
- `fastscnn_rgkd` − `fastscnn` = **+0.0364** [+0.0045, +0.0659], misses −3.78 pp.

**The one result worth chasing.** On Fast-SCNN the ITA fairness gap goes
0.160 (p = 0.021, significant) with no KD → **0.064 (p = 0.358, not significant)**
with plain B2 response KD → 0.142 (p = 0.029, significant again) with the gated
variant. A B2 teacher removes a skin-tone gap that neither the direct model nor the
DeepLabV3+ arm removes, and gating undoes that. This is the pre-registered primary
endpoint, and it is the one place gating looks actively worse than its control. At
28 subjects it is a single comparison — do not over-read it, but do follow it.

**Deliverables:** `docs/b2_distillation_deck.pptx`,
`FINAL_RESULT/significance_b2_teacher_vs_b0_direct.csv`,
`FINAL_RESULT/figures/H1_b2_vs_b0_direct.png`,
`scripts/70_b2_teacher_significance.py`, `scripts/71_generate_b2_distillation_deck.py`.
