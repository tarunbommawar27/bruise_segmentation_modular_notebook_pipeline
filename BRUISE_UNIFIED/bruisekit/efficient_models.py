"""Four mobile-grade segmentation baselines behind this project's one interface.

    PP-MobileSeg-Tiny    StrideFormer backbone + AAM + VIM head    ~1.6 M params
    TopFormer-Tiny       Token pyramid + semantics injection       ~1.4 M params
    LR-ASPP MobileNetV3  torchvision, lite reduced ASPP            ~3.2 M params
    Fast-SCNN            learning-to-downsample + PPM fusion       ~1.1 M params

THE INTERFACE THEY ALL SATISFY
-------------------------------
    forward_train(x) -> (logits[B,1,H,W], aux_logits | None)
    forward(x)       -> logits[B,1,H,W]
    .backbone        -> the pretrained part, for the encoder/head LR split

`x` is RAW [0,1] pixels, exactly as `bruisekit.data` emits it, and each wrapper
applies its own normalisation internally -- the same contract SegFormerNet and
SMPNet already follow. That is what lets `engine.train_run` stay
architecture-blind: it never branches on a model's name, so these four train
under precisely the recipe that produced every other number in the study.

WHY ONE LOGIT AND NOT TWO CLASSES
----------------------------------
All four reference implementations emit `num_classes` channels. Building them
with `num_classes=1` gives a single bruise logit directly, which is what every
downstream consumer -- loss, threshold sweep, metric -- already expects. Two
classes plus a softmax would be the same function via one more transformation
and one more place to get the sign backwards.

INITIALISATION IS PART OF THE EXPERIMENT
-----------------------------------------
Three of these load ImageNet-pretrained backbones; Fast-SCNN has none and trains
from scratch by design. Every wrapper records what it actually got in
`.init_source`, and `describe()` surfaces it, because a scratch-initialised model
sitting unlabelled in a table beside ImageNet-pretrained ones is a misleading
comparison, not a minor detail. See `bruisekit.weights`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Published parameter counts, with the class count they were measured at.
#
# These are NOT directly comparable to ours: every published figure is for a
# multi-class benchmark head (150 for ADE20K, 19 for Cityscapes) while we build a
# 1-class head. The final 1x1 conv scales with class count, so the self-test
# subtracts that term before comparing rather than pretending the raw numbers
# should agree.
#
# The parameter check is a cheap structural smoke test, not proof. For the two
# vendored architectures the authoritative check is `verify_checkpoint_match()`:
# if every tensor in the official checkpoint finds a name-and-shape match in our
# model, the architecture is right regardless of what a headline figure says.
PUBLISHED_PARAMS_M = {
    "ppmobileseg_tiny": (1.61, 150),    # OpenMMLab model zoo, ADE20K
    "topformer_tiny": (1.4, 150),       # TopFormer paper Table 3, ADE20K
    "lraspp_mobilenetv3": (3.22, 21),   # torchvision, COCO-with-VOC-labels
    "fastscnn": (1.14, 19),             # Fast-SCNN paper Section 4, Cityscapes
}

# Channels feeding each model's final 1x1 classifier, needed to convert a
# published multi-class count to its 1-class equivalent.
_HEAD_IN_CHANNELS = {
    "ppmobileseg_tiny": 256,
    "topformer_tiny": 128,
    "lraspp_mobilenetv3": 128 + 40,   # LR-ASPP sums a low- and a high-level classifier
    "fastscnn": 128,
}


def expected_params_M(arch: str, num_classes: int = 1) -> float:
    """Convert a published parameter count to what it should be at `num_classes`."""
    published, published_classes = PUBLISHED_PARAMS_M[arch]
    cin = _HEAD_IN_CHANNELS[arch]
    per_class = (cin + 1) / 1e6            # weights + bias of the 1x1 classifier
    return published - per_class * (published_classes - num_classes)


class _Normalised(nn.Module):
    """Base class holding the ImageNet normalisation every wrapper shares.

    Registered as buffers rather than constants so they move with `.to(device)`
    and are saved with the module -- the same reason `SMPNet` does it.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.init_source = "random init"

    def norm(self, x):
        return (x - self.mean) / self.std

    def forward(self, x):
        return self.forward_train(x)[0]

    def describe(self) -> str:
        n = sum(p.numel() for p in self.parameters())
        return f"{type(self).__name__}: {n / 1e6:.2f}M params, init = {self.init_source}"


# ─────────────────────────────────────────────────────────────────────────────
# 1 · PP-MobileSeg-Tiny
# ─────────────────────────────────────────────────────────────────────────────
# Verbatim from the OpenMMLab config
# projects/pp_mobileseg/configs/pp_mobileseg/..._tiny.py. Transcribing these by
# hand is exactly how a "tiny" silently becomes a "base", so they are kept in one
# place and diffed against the published parameter count by the self-test.
PPMOBILESEG_TINY_CFG = dict(
    mobileV3_cfg=[
        # k, expansion, channels, use_se, activation, stride
        [[3, 16, 16, True, "ReLU", 1], [3, 64, 32, False, "ReLU", 2],
         [3, 48, 24, False, "ReLU", 1]],
        [[5, 96, 32, True, "HSwish", 2], [5, 96, 32, True, "HSwish", 1]],
        [[5, 160, 64, True, "HSwish", 2], [5, 160, 64, True, "HSwish", 1]],
        [[3, 384, 128, True, "HSwish", 2], [3, 384, 128, True, "HSwish", 1]],
    ],
    channels=[16, 24, 32, 64, 128],
    depths=[2, 2],
    embed_dims=[64, 128],
    num_heads=4,
    inj_type="AAM",
    out_feat_chs=[32, 64, 128],
    act_cfg=dict(type="ReLU6"),
)


class PPMobileSegNet(_Normalised):
    """PP-MobileSeg-Tiny: StrideFormer backbone, AAM injection, VIM-capable head.

    The backbone returns `[fused_feature, input_hw]` and the head interpolates
    back to `input_hw`, so the model is resolution-agnostic and returns logits at
    the input size without the caller doing anything.

    A note on VIM: the head's "valid interpolate" path only upsamples the classes
    actually predicted, which is a latency win on ADE20K's 150 classes. With one
    class the head takes the plain interpolate branch -- `num_classes < 30` -- so
    VIM is inert here. That is a property of binary segmentation, not a
    misconfiguration.
    """

    def __init__(self, num_classes: int = 1, dropout_ratio: float = 0.1):
        super().__init__()
        from bruisekit.vendor.pp_mobileseg_head import PPMobileSegHead
        from bruisekit.vendor.strideformer import StrideFormer

        self.net_backbone = StrideFormer(**PPMOBILESEG_TINY_CFG)
        self.head = PPMobileSegHead(
            num_classes=num_classes, in_channels=256, use_dw=True,
            dropout_ratio=dropout_ratio, act_cfg=dict(type="ReLU"),
            upsample="intepolate", align_corners=False)

    @property
    def backbone(self):
        return self.net_backbone

    def forward_train(self, x):
        feats = self.net_backbone(self.norm(x))
        # The vendored head follows mmseg's convention of returning a LIST of
        # logit maps (one per auxiliary output) even when there is only one.
        # Unwrapping here rather than editing the vendored file keeps that file
        # byte-identical to upstream.
        out = self.head(feats)
        return (out[0] if isinstance(out, (list, tuple)) else out), None


# ─────────────────────────────────────────────────────────────────────────────
# 2 · TopFormer-Tiny
# ─────────────────────────────────────────────────────────────────────────────
# Verbatim from hustvl/TopFormer local_configs/topformer/topformer_tiny.py.
TOPFORMER_TINY_CFG = dict(
    cfgs=[
        # k, t, c, s
        [3, 1, 16, 1],   # 1/2
        [3, 4, 16, 2],   # 1/4
        [3, 3, 16, 1],
        [5, 3, 32, 2],   # 1/8
        [5, 3, 32, 1],
        [3, 3, 64, 2],   # 1/16
        [3, 3, 64, 1],
        [5, 6, 96, 2],   # 1/32
        [5, 6, 96, 1],
    ],
    channels=[16, 32, 64, 96],
    out_channels=[None, 128, 128, 128],
    embed_out_indice=[2, 4, 6, 8],
    decode_out_indices=[1, 2, 3],
    depths=4,
    key_dim=16,
    num_heads=4,
    attn_ratios=2,
    mlp_ratios=2,
    c2t_stride=2,
    drop_path_rate=0.1,
)


class TopFormerSimpleHead(nn.Module):
    """TopFormer's SimpleHead, reimplemented against the shim.

    Unlike the backbone this is safe to reimplement: we never load pretrained
    head weights (the head is 1-class and trained fresh), so no state-dict keys
    depend on it. The aggregation is upstream's: resize every pyramid level to
    the finest one and SUM -- not concatenate -- which is why `linear_fuse` takes
    `channels` inputs rather than `3 * channels`.
    """

    def __init__(self, in_channels=(128, 128, 128), channels=128,
                 num_classes=1, dropout_ratio=0.1, is_dw=True):
        super().__init__()
        from bruisekit.mmcv_shim import ConvModule
        self.linear_fuse = ConvModule(
            in_channels=channels, out_channels=channels, kernel_size=1, stride=1,
            groups=channels if is_dw else 1,
            norm_cfg=dict(type="BN"), act_cfg=dict(type="ReLU"))
        self.dropout = nn.Dropout2d(dropout_ratio)
        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, feats):
        out = feats[0]
        for f in feats[1:]:
            out = out + F.interpolate(f, size=out.shape[2:], mode="bilinear",
                                      align_corners=False)
        return self.conv_seg(self.dropout(self.linear_fuse(out)))


class TopFormerNet(_Normalised):
    """TopFormer-Tiny: token pyramid, pooled-token transformer, semantics injection.

    The head runs at 1/8 resolution (the finest of `decode_out_indices`), so the
    logits are upsampled to the input size here. Bilinear, matching upstream's
    `align_corners=False`.
    """

    def __init__(self, num_classes: int = 1, dropout_ratio: float = 0.1):
        super().__init__()
        from bruisekit.vendor.topformer import Topformer
        self.net_backbone = Topformer(**TOPFORMER_TINY_CFG)
        self.head = TopFormerSimpleHead(num_classes=num_classes,
                                        dropout_ratio=dropout_ratio)

    @property
    def backbone(self):
        return self.net_backbone

    def forward_train(self, x):
        feats = self.net_backbone(self.norm(x))
        logits = self.head(feats)
        logits = F.interpolate(logits, size=x.shape[2:], mode="bilinear",
                               align_corners=False)
        return logits, None


# ─────────────────────────────────────────────────────────────────────────────
# 3 · LR-ASPP MobileNetV3
# ─────────────────────────────────────────────────────────────────────────────
class LRASPPNet(_Normalised):
    """torchvision's LR-ASPP on a MobileNetV3-Large backbone, 1-class head.

    Built with `weights=None, weights_backbone=None` and the ImageNet backbone
    loaded explicitly afterwards by `bruisekit.weights`. Two reasons: the
    download then goes through the same provenance-recording path as everything
    else, and constructing with `weights_backbone="IMAGENET1K_V1"` would reach
    for the network at build time, which breaks an offline load where we are
    about to overwrite those weights from a checkpoint anyway.

    torchvision's LRASPP returns an OrderedDict with an "out" key already
    upsampled to the input size, so no interpolation is needed here.
    """

    def __init__(self, num_classes: int = 1):
        super().__init__()
        from torchvision.models.segmentation import lraspp_mobilenet_v3_large
        self.net = lraspp_mobilenet_v3_large(weights=None, weights_backbone=None,
                                             num_classes=num_classes)

    @property
    def backbone(self):
        return self.net.backbone

    def forward_train(self, x):
        return self.net(self.norm(x))["out"], None


# ─────────────────────────────────────────────────────────────────────────────
# 4 · Fast-SCNN
# ─────────────────────────────────────────────────────────────────────────────
def _dsconv(cin, cout, stride=1):
    """Depthwise-separable conv: 3x3 depthwise then 1x1 pointwise, each BN+ReLU."""
    return nn.Sequential(
        nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False),
        nn.BatchNorm2d(cin), nn.ReLU(inplace=True),
        nn.Conv2d(cin, cout, 1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


def _conv_bn(cin, cout, k=3, stride=1, pad=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, stride, pad, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class _Bottleneck(nn.Module):
    """MobileNetV2 inverted residual, as Fast-SCNN's global feature extractor uses."""

    def __init__(self, cin, cout, stride=1, expansion=6):
        super().__init__()
        hidden = cin * expansion
        self.use_res = stride == 1 and cin == cout
        self.conv = nn.Sequential(
            nn.Conv2d(cin, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, cout, 1, bias=False),
            nn.BatchNorm2d(cout))

    def forward(self, x):
        return x + self.conv(x) if self.use_res else self.conv(x)


class _PyramidPooling(nn.Module):
    """The PPM at the end of the global feature extractor: bins 1, 2, 3, 6."""

    def __init__(self, cin, cout):
        super().__init__()
        inter = cin // 4
        self.branches = nn.ModuleList([_conv_bn(cin, inter, 1, 1, 0) for _ in range(4)])
        self.bins = (1, 2, 3, 6)
        self.out = _conv_bn(cin * 2, cout, 1, 1, 0)

    def forward(self, x):
        size = x.shape[2:]
        feats = [x] + [
            F.interpolate(branch(F.adaptive_avg_pool2d(x, b)), size,
                          mode="bilinear", align_corners=True)
            for branch, b in zip(self.branches, self.bins)]
        return self.out(torch.cat(feats, dim=1))


class _FeatureFusion(nn.Module):
    """Fuse the high-resolution detail branch with the upsampled deep branch.

    The deep branch is upsampled x4 and passed through a dilated depthwise conv
    before the 1x1 -- upstream's design, which lets the fusion see a wider
    context than a bare 1x1 would after such an aggressive upsample.
    """

    def __init__(self, c_low, c_high, cout, scale=4):
        super().__init__()
        self.scale = scale
        self.dwconv = nn.Sequential(
            nn.Conv2d(c_high, cout, 3, 1, padding=scale, dilation=scale,
                      groups=c_high, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 1, bias=False), nn.BatchNorm2d(cout))
        self.conv_low = nn.Sequential(
            nn.Conv2d(c_low, cout, 1, bias=False), nn.BatchNorm2d(cout))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, low, high):
        high = F.interpolate(high, size=low.shape[2:], mode="bilinear", align_corners=True)
        return self.relu(self.conv_low(low) + self.dwconv(high))


class FastSCNNNet(_Normalised):
    """Fast-SCNN (Poudel et al., 2019), trained from scratch as the paper does.

    Structure: learning-to-downsample to 1/8, a global feature extractor of
    inverted-residual bottlenecks down to 1/32 with a pyramid pooling module,
    feature fusion back at 1/8, then a depthwise-separable classifier upsampled
    to full resolution.

    An auxiliary head is deliberately NOT added. Fast-SCNN's paper uses one, but
    `engine.train_run` weights the aux loss with `cfg["aux_weight"]`, and the
    baselines in this study run with `aux_weight = 0` so that the loss is
    identical across architectures. Returning `None` for aux keeps that explicit
    rather than adding a head whose gradient is then multiplied by zero.
    """

    def __init__(self, num_classes: int = 1):
        super().__init__()
        # Learning to downsample: 1/2 -> 1/4 -> 1/8
        self.learning_to_downsample = nn.Sequential(
            _conv_bn(3, 32, 3, 2, 1),
            _dsconv(32, 48, 2),
            _dsconv(48, 64, 2))
        # Global feature extractor: 1/8 -> 1/16 -> 1/32
        self.global_feature_extractor = nn.Sequential(
            _Bottleneck(64, 64, 2), _Bottleneck(64, 64), _Bottleneck(64, 64),
            _Bottleneck(64, 96, 2), _Bottleneck(96, 96), _Bottleneck(96, 96),
            _Bottleneck(96, 128, 1), _Bottleneck(128, 128), _Bottleneck(128, 128),
            _PyramidPooling(128, 128))
        self.feature_fusion = _FeatureFusion(64, 128, 128, scale=4)
        self.classifier = nn.Sequential(
            _dsconv(128, 128), _dsconv(128, 128),
            nn.Dropout2d(0.1),
            nn.Conv2d(128, num_classes, 1))
        self.init_source = "random init (no pretrained weights exist; paper trains from scratch)"

    @property
    def backbone(self):
        """The 'pretrained-like' part, for the LR split.

        Nothing here is pretrained, so the split is cosmetic -- but keeping the
        property means `build_param_groups` treats this model like every other
        one instead of needing a special case.
        """
        return self.learning_to_downsample

    def forward_train(self, x):
        x_norm = self.norm(x)
        low = self.learning_to_downsample(x_norm)
        high = self.global_feature_extractor(low)
        fused = self.feature_fusion(low, high)
        logits = self.classifier(fused)
        return F.interpolate(logits, size=x.shape[2:], mode="bilinear",
                             align_corners=True), None


# ─────────────────────────────────────────────────────────────────────────────
# construction + pretrained loading
# ─────────────────────────────────────────────────────────────────────────────
EFFICIENT_ARCHS = {
    "ppmobileseg_tiny": PPMobileSegNet,
    "topformer_tiny": TopFormerNet,
    "lraspp_mobilenetv3": LRASPPNet,
    "fastscnn": FastSCNNNet,
}


def build_efficient(arch: str, num_classes: int = 1):
    """Construct one of the four, untrained and offline."""
    if arch not in EFFICIENT_ARCHS:
        raise ValueError(f"unknown efficient arch {arch!r}; "
                         f"choices: {sorted(EFFICIENT_ARCHS)}")
    return EFFICIENT_ARCHS[arch](num_classes=num_classes)


def build_with_pretrained(env, arch: str, num_classes: int = 1, verbose: bool = True):
    """Construct and load whatever pretrained backbone is available for `arch`.

    Returns the model with `.init_source` set to the truth: either the provenance
    string from `weights.SOURCES`, or an explicit statement that it is training
    from scratch. Never silently falls back.

    The backbone-only checkpoints deliberately leave the decode head random --
    that is the same "ImageNet encoder, fresh head" setup as the ResNet-50
    baselines, and it is what makes the comparison fair.
    """
    from bruisekit import weights as W

    model = build_efficient(arch, num_classes)
    src = W.SOURCES[arch]

    if src.kind == "none":
        model.init_source = src.init
        if verbose:
            print(f"  {arch}: {src.init}")
        return model

    path = W.provision(env, arch, verbose=verbose)
    if path is None:
        model.init_source = f"random init -- {src.kind} weights not available"
        if verbose:
            print(f"  {arch}: NOT pretrained ({src.kind} source unavailable). "
                  f"It will train from scratch and is labelled as such.")
        return model

    target, strip = model, src.strip_prefix
    if arch == "ppmobileseg_tiny":
        target = model.net_backbone
    elif arch == "topformer_tiny":
        target = model.net_backbone
    elif arch == "lraspp_mobilenetv3":
        # torchvision's LRASPP wraps MobileNetV3 in an IntermediateLayerGetter
        # whose child keys are the classification model's `features.<i>` indices,
        # so the classification checkpoint's `features.` prefix maps straight on.
        target = model.net.backbone
        strip = "features."

    stats = W.load_pretrained(target, path, strip_prefix=strip,
                              expect_missing=src.expect_missing, verbose=verbose)
    model.init_source = f"{src.init} ({stats['loaded']}/{stats['in_checkpoint']} tensors)"
    return model


def install_efficient_shim(env=None, verbose: bool = True):
    """Teach `engine.train_run`'s `build_model` about these four architectures.

    Mirrors `smp_models.install_build_model_shim` exactly, including WHY it
    reassigns the name in two modules: `bruisekit.engine` bound `build_model` by
    value at import time, so patching only `bruisekit.models` would leave the
    training loop calling the original. Both are reassigned, and any arch this
    shim does not recognise falls through to the previous builder untouched.

    PASS `env` WHEN TRAINING
    -------------------------
    `train_run` constructs its model through `build_model` and immediately starts
    optimising it, so this is the ONLY point at which pretrained weights can be
    applied to a training run. With `env`, the shim routes to
    `build_with_pretrained` and the ImageNet backbones are loaded; without it,
    every one of these models silently trains from random init -- which on 697
    images is a large and entirely invisible handicap. Omit `env` only when you
    genuinely want untrained architectures, such as in `self_test`.
    """
    import bruisekit.engine as _be
    import bruisekit.models as _bm

    previous = _bm.build_model

    def build_model(arch, size, paths):
        if arch in EFFICIENT_ARCHS:
            if env is None:
                return build_efficient(arch, num_classes=1)
            return build_with_pretrained(env, arch, num_classes=1, verbose=verbose)
        return previous(arch, size, paths)

    _bm.build_model = build_model
    _be.build_model = build_model

    previous_probe = _be.resolve_micro_batch

    def resolve_micro_batch(model, cfg, device, teacher=None):
        # Same reasoning as the SMP shim: these nets contain BatchNorms fed by
        # global-pooled [B,C,1,1] tensors (SE blocks, PPM, LR-ASPP's image pool),
        # which raise "Expected more than 1 value per channel" at batch 1. The
        # engine's VRAM probe starts at batch 1, so it is skipped in favour of a
        # fixed, batch-safe size -- which also means every efficient baseline and
        # every seed trains at the same batch, as the SMP baselines do.
        if isinstance(model, _Normalised):
            return max(2, int(cfg.get("efficient_micro_batch", 16))), 1
        return previous_probe(model, cfg, device, teacher)

    _be.resolve_micro_batch = resolve_micro_batch
    return build_model


def self_test(verbose: bool = True) -> "object":
    """Build all four, check shapes, and diff parameter counts against published.

    The parameter check is the load-bearing one. A transformer whose `key_dim` or
    head count is wrong still produces [B,1,H,W] and still trains -- it is simply
    not the model named in the table. Parameter count catches that class of error
    in seconds, before a GPU-day is spent on it.
    """
    import pandas as pd

    rows = []
    for arch in EFFICIENT_ARCHS:
        m = build_efficient(arch, 1).eval()
        with torch.no_grad():
            x = torch.rand(2, 3, 640, 640)
            logits, aux = m.forward_train(x)
            plain = m(x)
        n = sum(p.numel() for p in m.parameters()) / 1e6
        want = expected_params_M(arch, 1)
        rows.append({
            "arch": arch,
            "params_M": round(n, 3),
            "expected_M": round(want, 3),
            "delta_%": round(100 * (n - want) / want, 1),
            "out_shape": tuple(logits.shape),
            "shape_ok": tuple(logits.shape) == (2, 1, 640, 640),
            "aux_is_none": aux is None,
            "forward_matches": bool(torch.equal(plain, logits)),
            "has_backbone": hasattr(m, "backbone"),
        })
        del m
    df = pd.DataFrame(rows)
    if verbose:
        print(df.to_string(index=False))
        print("\nexpected_M = published count rescaled to a 1-class head "
              "(published figures are 150/21/19-class).")
        bad = df[~df.shape_ok]
        if len(bad):
            print(f"FAIL: wrong output shape for {list(bad.arch)}")
        off = df[df["delta_%"].abs() > 8]
        if len(off):
            print(f"NOTE: {list(off.arch)} differs from the rescaled published count by "
                  f">8%. For the vendored architectures this is not decisive -- run "
                  f"verify_checkpoint_match(env) for the authoritative structural check.")
    return df


def verify_checkpoint_match(env, verbose: bool = True) -> "object":
    """The authoritative structural check: does every official tensor find a home?

    A parameter count can disagree with a paper for boring reasons -- a different
    class count, a decode head counted or not, a framework's own accounting. What
    cannot be argued with is this: load the published checkpoint into our model
    and count how many of its tensors match by BOTH name and shape.

    `unexpected = 0` means every tensor the authors trained has an exactly
    matching module in our copy. Combined with `missing` containing only the
    decode-head prefixes we intend to train fresh, that is proof the vendored
    architecture is the published one.

    Skips models whose weights are not on disk, rather than failing -- absence is
    already reported by `weights.report()`.
    """
    import pandas as pd

    from bruisekit import weights as W

    rows = []
    for arch in EFFICIENT_ARCHS:
        src = W.SOURCES[arch]
        path = env.weights / "efficient" / src.filename if src.filename else None
        if not (path and path.exists()):
            rows.append({"arch": arch, "checked": False,
                         "verdict": f"no checkpoint on disk ({src.kind})"})
            continue
        model = build_with_pretrained(env, arch, 1, verbose=False)
        target = getattr(model, "net_backbone", None)
        if target is None:
            target = model.net.backbone if arch == "lraspp_mobilenetv3" else model
        strip = "features." if arch == "lraspp_mobilenetv3" else src.strip_prefix
        stats = W.load_pretrained(target, path, strip_prefix=strip,
                                  expect_missing=src.expect_missing, verbose=False)
        ok = stats["unexpected"] == 0 and stats["loaded"] == stats["in_checkpoint"]
        rows.append({"arch": arch, "checked": True,
                     "loaded": stats["loaded"], "in_checkpoint": stats["in_checkpoint"],
                     "unexpected": stats["unexpected"],
                     "undeclared_missing": stats["missing"],
                     "verdict": "EXACT MATCH" if ok else "MISMATCH -- investigate"})
        del model
    df = pd.DataFrame(rows)
    if verbose:
        print(df.to_string(index=False))
    return df
