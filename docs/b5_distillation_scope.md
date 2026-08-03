# B5 Distillation — Scope of Upcoming Work

**Status: PLANNING ONLY. Nothing in this document is implemented.**
Written 2026-07-22, immediately after the B5 baseline package was built
(`segformer_b5_bruise_pipeline.zip`, not yet trained). This captures the agreed
experimental plan so the next session can build the distillation package
without re-deriving the design.

---

## 0 · Prerequisites (in order — nothing below starts until these exist)

1. **B5 baseline trained + scored** — via `segformer_b5_bruise_pipeline.zip` on
   ORC or the mlidl server. Produces `best_model.pt`, `threshold_search.csv`,
   `test_per_image.csv` (185 rows), FINAL json, fairness CSVs.
2. **User pastes the colab-final model checkpoints into this directory** —
   the `runs_final/` models from `bruise_colab_final.ipynb` (B2 teacher,
   B0 direct, B0 distilled, YOLO ×2, currently on Drive). Needed for: the
   existing B2 teacher, the B2-vs-B5 comparison, and the B2→B0 KD reference row.
3. **Phase-1 gate evaluated** (see §2) — the multi-teacher work is conditional,
   not automatic.

## 1 · Context and framing (do not lose)

- **B5 is NOT a newer SegFormer** — it is the largest member of the original
  2021 B0–B5 family (~82–85M params vs ~27M B2, ~3.7M B0).
- Current results: B0-distilled − B2 ≈ **−0.001 median Dice, CI spans zero**.
  Seven architectures cluster at 0.74–0.79 — the project's own finding is that
  the task is **label-limited, not capacity-limited** (annotation ceiling,
  human-human agreement 0.639). A bigger teacher is useful only if it is
  **demonstrably better on hard cases or makes different errors than B2** —
  not because it has more parameters.
- Therefore every claim below must be framed on **complete-miss rate,
  worst-group performance, and hard-case metrics**, not mean Dice, and every
  comparison goes through the **paired subject-level cluster bootstrap**
  (28 subjects, B=4000, ~0.04 median-Dice MDE). Fairness numbers are
  exploratory at n=28 (9–17 subjects per ITA group); the only fairness
  quantity that has previously survived clustering is the **light-skin recall
  deficit**.

## 2 · Phase 1 — GATE: is B5 a useful teacher at all?

Train B5 with the exact baseline recipe (already the case — the package holds
the recipe fixed: same split/resolution/augs/loss/selection/eval). Then compare
B5 vs B2, both against the 2-of-3 majority on the 185 test images:

| Axis | Where it comes from |
|---|---|
| mean/median Dice, IoU | `test_per_image.csv` both models |
| recall, complete-miss rate | same |
| small-bruise stratum (size terciles/quartiles) | gt_positive_pixels from per-image CSVs |
| per-ITA-group median Dice / recall / miss | fairness CSVs both models |
| calibration | needs a temperature fit on val (script 02 pattern) — **not yet done for B5** |
| **per-image B2–B5 disagreement** | paired per-image merge on `stem` — the key novel number |

**Proceed to multi-teacher work ONLY if** (a) B5 beats B2 on hard cases
(small/low-contrast bruises, misses, worst ITA group), OR (b) B2 and B5 make
sufficiently different errors (low per-image correlation / complementary miss
sets) to justify an ensemble. If B5 ≈ B2 everywhere, single-teacher B5 KD is
still reportable but the ensemble/two-teacher phases are dropped.

## 3 · The four techniques (what each needs)

1. **Group-aware FairDistillation** — B5→B0 (or B2→B0) with
   `L = L_sup + λ_KD·L_KD + λ_fair·L_group` where L_group penalizes worst-ITA-group
   error during training. *B5 not strictly required — works with B2 too.*
   Needs: per-TRAINING-image ITA group.
2. **Adaptive group-robust ensemble KD** — two teachers (B2, B5); student target
   `p_ens = w_g,1·p_B2 + w_g,2·p_B5` with weights per ITA group / size / difficulty,
   set from val performance. *Gated on Phase-1 disagreement finding.*
3. **Hard-example KD** — per-image loss weight
   `w_i = 1 + γ(1−Dice_i) + η·1(complete miss)` upweighting small bruises,
   low contrast, student misses, large teacher-student disagreement, low-recall groups.
4. **Two-biased-teacher KD** — B2/B5 are NOT automatically "biased teachers";
   requires **deliberately retraining** teachers with different objectives
   (e.g., group-balanced vs size-balanced sampling/weighting), then
   `L_KD = α_i·L_KD(T_A,S) + (1−α_i)·L_KD(T_B,S)`. *Most expensive; Phase 3 only.*

## 4 · Experiment order (agreed — keeps attribution clean)

- **Phase 2 (single-teacher, after the gate):**
  1. B2→B0 KD (exists — the reference row)
  2. B5→B0 standard KD
  3. B5→B0 hard-example KD
  4. B5→B0 group-aware KD
  5. group-aware + hard-example combined
- **Phase 3 (multi-teacher, only if Phase-1 gate passes):**
  6. adaptive B2+B5 ensemble KD
  7. two-biased-teacher KD (needs teacher retraining first)

**Recommended headline experiment** if resources force a choice:
**B5→B0 group- and failure-aware distillation**
(`L = L_sup + λ_KD·L_KD + λ_group·L_worst-group + λ_hard·L_hard-example`),
extended to the B2+B5 adaptive ensemble only if errors are complementary.
Never run everything at once — ablate one component at a time or improvements
cannot be attributed.

## 5 · What the future distillation zip must contain (build checklist)

To be built **after** prerequisites land. Same self-contained pattern as the
B5/nnU-Net packages (Python `zipfile`, forward-slash arcnames, LF-normalized
text files):

- **Data**: same `data/` + `manifests/` (697/134 `split` column) + `splits/` as
  the existing packages (stream byte-identical from `segformer_b5_bruise_pipeline.zip`).
- **ITA labels for BOTH splits**: `ita_labels/wl_train_per_image_ita.csv`
  (✓ exists locally, 831 rows, has `ita_group_index_5` — required by the
  group-aware losses) and `wl_test_per_image_ita.csv` (✓ already packaged).
- **Teacher checkpoints**: B5 `best_model.pt` + its `threshold_search.csv`
  (from the baseline run) and B2 teacher `best_model.pt` + threshold +
  `temperature.json` (from the pasted colab-final models). **B5 needs its own
  temperature calibration** (port of `scripts/02_calibrate_teacher.py`) before
  soft targets are used — the existing KD path distills temperature-scaled
  sigmoid probs (Menon-style calibrated soft targets, not raw Hinton).
- **Student init**: `pretrained_weights/segformer_mit_b0/` (✓ exists locally,
  ~14 MB) — and `segformer_mit_b2/` + `segformer_mit_b5/` for rebuilding
  teacher wrappers at load time.
- **Code**: self-contained script(s) following `train_segformer_b5_baseline.py`'s
  pattern, reusing the existing distillation machinery as the base:
  `pipeline/losses.py::DistillSegLoss` (α·DiceBCE(student,GT) + (1−α)·BCE(student,teacher_prob)),
  `pipeline/trainer.py::load_teacher` (teacher_fn callable, teacher-first VRAM
  ordering — the batch probe must include the teacher forward, and the B2+B5
  ensemble means TWO teacher forwards per step: probe accordingly), Optuna
  alpha search pattern from `scripts/04/07`. New pieces to write: group loss,
  per-image weight schedule, ensemble weighting, per-image disagreement report.
- **Eval**: same test scoring + fairness CSVs as the B5 package, plus the
  paired-bootstrap comparison vs the B2→B0 reference row.
- **Runners**: `run_all.sh`/`setup_env.sh` (tmux, mlidl) + an ORC notebook,
  mirroring the B5 package.

## 6 · Explicit non-goals right now

- No implementation of any technique yet (this file is the only deliverable).
- No teacher retraining (two-biased-teacher) before Phase 3.
- No fairness claims without cluster CIs; no "B5 beats B2" claims inside the
  ~0.04 MDE.
- The old `out_of_scope/` Fair-KD A/B code is historical reference, not a base
  to resurrect — new work builds on the current `pipeline/` KD path.
