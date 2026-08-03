"""Test scoring, fairness across skin tone, and the speed benchmark."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy import stats

from bruisekit.metrics import compute_image_row, summarize


@torch.no_grad()
def evaluate_at_cut(model, loader, device, cut: float, amp: bool) -> tuple[pd.DataFrame, dict]:
    """Score on test at an operating point ALREADY FIXED on val. One pass, no tuning."""
    model.eval()
    rows = []
    for x, y, stems in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp):
            z = model(x)
        pred = (z.float()[:, 0] >= cut).cpu().numpy().astype(np.uint8)
        gt = (y[:, 0] > 0.5).numpy().astype(np.uint8)
        for i, stem in enumerate(stems):
            rows.append(compute_image_row(pred[i], gt[i], stem))
    return pd.DataFrame(rows), summarize(rows)


def bootstrap_ci(values: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI of the MEDIAN.

    The median, not the mean: per-image Dice is strongly bimodal (a model either
    localises the bruise or misses it completely), so the mean mixes two different
    populations. And a bootstrap, not a t-interval, because that bimodality makes
    the normal approximation wrong.
    """
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    meds = [np.median(rng.choice(values, size=len(values), replace=True)) for _ in range(n)]
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def fairness_analysis(per_image: pd.DataFrame, manifest: pd.DataFrame, model_name: str) -> dict:
    """Is performance equitable across Fitzpatrick/ITA skin-tone groups?

    THE STAKES: this is a forensic injury-documentation tool. A model that
    segments bruises well on light skin and poorly on dark skin does not have a
    metric problem, it has an evidentiary one -- it would under-document injuries
    on exactly the population most likely to need the documentation. So skin tone
    is not a robustness ablation here, it is a primary result.

    METHOD, and why each piece:
      - Groups are ITA (Individual Typology Angle, Chardon et al. 1991), the
        standard objective skin-tone measure -- computed from image pixels, not
        from a rater's Fitzpatrick guess, so it is reproducible.
      - Kruskal-Wallis across all 5 groups: an omnibus test for "does ANY group
        differ?". Non-parametric because per-image Dice is bimodal and bounded,
        so ANOVA's normality assumption fails.
      - Pairwise Mann-Whitney U, Bonferroni-corrected over the 10 pairs: with 5
        groups, uncorrected pairwise testing finds a "significant" pair ~40% of
        the time on pure noise.
      - fairness_gap = best group's median Dice - worst group's. The effect size.
        A p-value says a gap is real; only the gap says whether it matters.
    """
    df = per_image.merge(manifest[["stem", "skin_tone_category", "ita_group_index_5"]],
                         on="stem", how="left", validate="one_to_one")
    if df["ita_group_index_5"].isna().any():
        raise RuntimeError(f"{int(df['ita_group_index_5'].isna().sum())} test images have no skin-tone label")

    per_group, samples = [], []
    for gidx, g in sorted(df.groupby("ita_group_index_5"), key=lambda kv: kv[0]):
        vals = g["dice"].to_numpy()
        lo, hi = bootstrap_ci(vals)
        per_group.append({
            "model": model_name, "ita_group_index_5": int(gidx),
            "skin_tone_category": g["skin_tone_category"].iloc[0],
            "n_images": len(g), "median_dice": float(np.median(vals)),
            "iqr_dice": float(np.percentile(vals, 75) - np.percentile(vals, 25)),
            "ci95_lo": lo, "ci95_hi": hi,
            "mean_recall": float(g["recall"].mean()),
            "miss_rate": float(((g["pred_positive_pixels"] == 0) & (g["gt_positive_pixels"] > 0)).mean()),
        })
        samples.append(vals)

    H, p = stats.kruskal(*samples)
    pairwise = []
    raw = []
    pairs = [(i, j) for i in range(len(samples)) for j in range(i + 1, len(samples))]
    for i, j in pairs:
        raw.append(stats.mannwhitneyu(samples[i], samples[j], alternative="two-sided").pvalue)
    for (i, j), pv in zip(pairs, raw):
        adj = min(1.0, pv * len(pairs))    # Bonferroni over the 10 pairs
        pairwise.append({"model": model_name, "group_a": per_group[i]["skin_tone_category"],
                         "group_b": per_group[j]["skin_tone_category"],
                         "pvalue": pv, "bonferroni_p": adj, "significant": bool(adj < 0.05)})

    pg = pd.DataFrame(per_group)
    best, worst = pg.loc[pg["median_dice"].idxmax()], pg.loc[pg["median_dice"].idxmin()]
    stats_row = {
        "model": model_name, "kruskal_H": float(H), "kruskal_p": float(p),
        "significant": bool(p < 0.05),
        "fairness_gap": float(best["median_dice"] - worst["median_dice"]),
        "best_group": best["skin_tone_category"], "worst_group": worst["skin_tone_category"],
        "max_miss_rate_gap": float(pg["miss_rate"].max() - pg["miss_rate"].min()),
    }
    return {"per_group": pg, "pairwise": pd.DataFrame(pairwise), "stats": stats_row}


@torch.no_grad()
def benchmark_speed(model, images, device, cut: float, repeats: int, warmup: int) -> dict:
    """Time 640-tensor-on-GPU -> mask-on-GPU. Nothing else.

    WHAT IS DELIBERATELY NOT TIMED: disk read, JPEG decode, resize, host->GPU copy,
    and GPU->host copy. Those are identical for every model (one dataloader) and
    are dominated by I/O, so including them would compress the real architectural
    differences into measurement noise. The mask never leaves the GPU.

    cuda.synchronize() around each call is not optional: CUDA is asynchronous, so
    without it this would measure how long it takes to QUEUE the work, not do it --
    which reports every model as equally, impossibly fast.
    """
    model.eval()
    for _ in range(warmup):
        _ = torch.sigmoid(model(images[:1])) >= torch.sigmoid(torch.tensor(cut, device=device))
    torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        for i in range(len(images)):
            x = images[i:i + 1]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            z = model(x)
            _ = z >= cut                      # threshold on the logit == sigmoid >= sigmoid(cut)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    arr = np.array(times)
    return {"median_ms": float(np.median(arr)), "mean_ms": float(arr.mean()),
            "p95_ms": float(np.percentile(arr, 95)), "fps": float(1000.0 / np.median(arr)),
            "n_timed": len(arr)}


import time  # noqa: E402  (used by benchmark_speed above)