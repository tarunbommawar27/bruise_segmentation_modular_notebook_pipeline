"""Losses. One supervised loss for every model; one distillation fusion.

WHY Dice+BCE
-------------
BCE alone is dominated by the background: bruises cover ~4.7% of pixels, so a
model predicting all-background already scores well on BCE and gets almost no
gradient toward the bruise. The Dice term is scale-invariant with respect to
object size and supplies gradient proportional to overlap, which is what we
actually report. Summing them is the standard combination for imbalanced medical
segmentation (Milletari et al. 2016 V-Net for Dice; Drozdzal et al. 2016 for the
combination), and it is also what Ultralytics' own SemanticSegmentationLoss does
(BCEWithLogits + binary Dice), so using it for both architectures does not
disadvantage YOLO relative to its native recipe.

WHAT THE DISTILLATION LOSS IS, AND WHAT IT IS NOT
--------------------------------------------------
    loss = alpha * DiceBCE(student_logits, GT)
         + (1 - alpha) * BCE(student_logits, sigmoid(teacher_logits / T_cal))

This is CALIBRATED SOFT-TARGET DISTILLATION, not Hinton et al. (2015) KD. The
difference matters for the paper and should not be papered over:

  - Hinton's KD divides BOTH the student's and the teacher's logits by a shared
    temperature T and multiplies the soft term by T^2 (to keep the soft gradient's
    magnitude comparable to the hard term's as T varies).
  - Here the student's logits are NOT temperature-scaled, and there is no T^2.
    T_cal is not a KD knob at all: it is the temperature fitted by NLL on val
    (Guo et al. 2017) that makes the teacher's probabilities CALIBRATED.

So the teacher is used as a better-calibrated estimate of P(bruise | pixel), and
the student regresses onto that probability. The justification is Menon et al.
(2021), "A statistical perspective on distillation": distillation helps to the
extent the teacher approximates the Bayes class-probability, which is exactly
what calibration improves. Do NOT cite Hinton for this formula as written.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    """BCE + (1 - mean per-image soft Dice).

    Dice is computed PER IMAGE and then averaged, not pooled over the batch. A
    batch-pooled Dice lets one large bruise dominate the whole batch's gradient
    and lets an image with no predicted pixels hide inside a batch that scored
    well -- and per-image is what the reported metric does, so the loss and the
    metric agree.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target)
        prob = torch.sigmoid(logits)
        inter = (prob * target).sum(dim=(1, 2, 3))
        denom = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        return bce + (1.0 - dice.mean())


class SupervisedLoss(nn.Module):
    """DiceBCE on the main head + aux_weight * BCE on the auxiliary head.

    The aux term applies only to YOLO (SegFormer returns aux=None). 0.4 is
    Ultralytics' own weight for its semantic aux head; keeping their value means
    the aux head is supervised as its designers intended even though the rest of
    the recipe is ours.
    """

    def __init__(self, aux_weight: float = 0.4):
        super().__init__()
        self.main = DiceBCELoss()
        self.aux_weight = aux_weight

    def forward(self, logits, aux_logits, target):
        loss = self.main(logits, target)
        if aux_logits is not None:
            loss = loss + self.aux_weight * F.binary_cross_entropy_with_logits(aux_logits, target)
        return loss


class DistillLoss(nn.Module):
    """alpha * supervised(GT) + (1-alpha) * BCE(student, calibrated teacher prob).

    See the module docstring for what this is and is not.
    """

    def __init__(self, alpha: float = 0.5, aux_weight: float = 0.4):
        super().__init__()
        self.alpha = alpha
        self.sup = SupervisedLoss(aux_weight)

    def forward(self, logits, aux_logits, target, teacher_prob):
        hard = self.sup(logits, aux_logits, target)
        soft = F.binary_cross_entropy_with_logits(logits, teacher_prob)
        return self.alpha * hard + (1.0 - self.alpha) * soft