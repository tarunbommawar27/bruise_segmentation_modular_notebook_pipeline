"""A minimal stand-in for the handful of mmcv/mmengine pieces the vendored nets use.

WHY A SHIM INSTEAD OF DEPENDING ON mmcv
----------------------------------------
Two of the four efficient baselines -- PP-MobileSeg-Tiny (StrideFormer) and
TopFormer-Tiny -- only have reference implementations inside the OpenMMLab stack.
Installing that stack is not a small ask: `mmcv` ships compiled CUDA extensions
pinned to narrow torch versions, and on Colab it routinely fails to build against
whatever torch is current. Dragging that in to get two ~1.5M-parameter models
would be the heaviest dependency in this entire project.

But hand-rewriting those architectures is worse. The published checkpoints are
state dicts keyed by module path, so any deviation in module naming -- a
`nn.Sequential` where the reference used a named submodule, `conv`/`bn` in the
wrong order -- silently breaks `load_state_dict` or, worse, loads a subset and
leaves the rest random. The whole point of using these models is to start from
their pretrained weights.

So the reference files are vendored VERBATIM and only their import lines are
rewritten to point here. That keeps every module name, and therefore every
state-dict key, byte-identical to the checkpoint that ships with them.

WHAT IS FAITHFULLY REPRODUCED
------------------------------
`ConvModule`'s conv -> norm -> activation ordering and its attribute names
(`.conv`, `.bn`, `.activate`), because those names ARE the state-dict keys. The
parts of mmcv that do not appear in a state dict (registries, config plumbing,
runner hooks) are stubbed to no-ops.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# ── activations ──────────────────────────────────────────────────────────────
class HSigmoid(nn.Module):
    """Hard sigmoid as mmcv/PaddleSeg define it: clamp(slope * x + offset, 0, 1).

    NOT interchangeable with `nn.Hardsigmoid`. Torch's builtin fixes the slope at
    1/6; StrideFormer's squeeze-excite blocks ask for `slope=0.2, offset=0.5`.
    Substituting the builtin would leave every state-dict key matching -- this
    layer has no parameters -- while quietly changing the gate applied to every
    SE block in the backbone. The weights would load "successfully" and the
    pretrained features would be subtly wrong, which is the worst possible
    failure mode: silent, and invisible to any shape check.

    So the slope is honoured, and the default matches mmcv's own (1/6).
    """

    def __init__(self, slope: float = 0.1666667, offset: float = 0.5, inplace: bool = False):
        super().__init__()
        self.slope, self.offset = float(slope), float(offset)

    def forward(self, x):
        return (x * self.slope + self.offset).clamp_(0.0, 1.0)

    def extra_repr(self) -> str:
        return f"slope={self.slope}, offset={self.offset}"


# mmcv spells these as config dicts. The names on the right are what the
# reference configs actually use; anything else is a genuine error, not a
# default worth guessing at.
_ACTIVATIONS = {
    "ReLU": nn.ReLU,
    "ReLU6": nn.ReLU6,
    "LeakyReLU": nn.LeakyReLU,
    "GELU": nn.GELU,
    "SiLU": nn.SiLU,
    "Sigmoid": nn.Sigmoid,
    "HSwish": nn.Hardswish,
    "Hardswish": nn.Hardswish,
    "HSigmoid": HSigmoid,
    "Hardsigmoid": HSigmoid,
}
# Activations that take no `inplace` argument, so the shim must not pass one.
_NO_INPLACE = {"GELU", "Sigmoid", "HSigmoid", "Hardsigmoid", "HSwish", "Hardswish"}


def build_activation_layer(cfg):
    """Build an activation from an mmcv-style dict, e.g. dict(type='ReLU6')."""
    if cfg is None:
        return nn.Identity()
    cfg = dict(cfg)
    t = cfg.pop("type")
    if t not in _ACTIVATIONS:
        raise KeyError(f"unsupported activation {t!r}; known: {sorted(_ACTIVATIONS)}")
    if t not in _NO_INPLACE:
        cfg.setdefault("inplace", True)
    else:
        cfg.pop("inplace", None)
    return _ACTIVATIONS[t](**cfg)


# ── normalisation ────────────────────────────────────────────────────────────
def build_norm_layer(cfg, num_features, postfix=""):
    """Build a norm layer, returning mmcv's (name, layer) pair.

    SyncBN is deliberately downgraded to plain BatchNorm2d. The reference configs
    ask for SyncBN because they train on 8-16 GPUs where cross-device batch
    statistics matter; here every run is single-GPU, where SyncBN IS BatchNorm2d
    with extra collective-communication overhead. The parameter shapes and names
    are identical, so checkpoints load either way.
    """
    cfg = dict(cfg or {"type": "BN"})
    t = cfg.pop("type")
    cfg.pop("requires_grad", None)
    if t in ("BN", "BN2d", "SyncBN"):
        layer = nn.BatchNorm2d(num_features, **cfg)
    elif t == "BN1d":
        layer = nn.BatchNorm1d(num_features, **cfg)
    elif t in ("GN", "GroupNorm"):
        layer = nn.GroupNorm(num_channels=num_features, **cfg)
    elif t in ("LN", "LayerNorm"):
        layer = nn.LayerNorm(num_features, **cfg)
    else:
        raise KeyError(f"unsupported norm {t!r}")
    return f"{t.lower()}{postfix}", layer


class ConvModule(nn.Module):
    """conv -> norm -> activation, with mmcv's attribute names preserved.

    The submodule names `conv`, `bn` and `activate` are load-bearing: they are
    literally the prefixes in every published checkpoint's state dict. Renaming
    `bn` to `norm` here would make every pretrained weight silently fail to match.

    `bias` defaults to "auto" as in mmcv, meaning: no bias when a norm layer
    follows (the norm's shift subsumes it), bias otherwise.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias="auto", conv_cfg=None,
                 norm_cfg=None, act_cfg=dict(type="ReLU"), inplace=True,
                 order=("conv", "norm", "act")):
        super().__init__()
        if conv_cfg not in (None, {}) and conv_cfg.get("type") not in (None, "Conv2d", "Conv"):
            raise KeyError(f"unsupported conv type {conv_cfg.get('type')!r}")
        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        if bias == "auto":
            bias = not self.with_norm

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=bias)
        if self.with_norm:
            _, norm = build_norm_layer(norm_cfg, out_channels)
            self.bn = norm
        if self.with_activation:
            act = dict(act_cfg)
            if act.get("type") not in _NO_INPLACE:
                act.setdefault("inplace", inplace)
            self.activate = build_activation_layer(act)

    def forward(self, x, activate=True, norm=True):
        x = self.conv(x)
        if self.with_norm and norm:
            x = self.bn(x)
        if self.with_activation and activate:
            x = self.activate(x)
        return x


# ── stochastic depth ─────────────────────────────────────────────────────────
class DropPath(nn.Module):
    """Per-sample stochastic depth, matching timm/mmcv semantics.

    Carries no parameters, so it never appears in a state dict -- which is why
    reimplementing it here is safe where reimplementing a conv would not be.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x.div(keep) * mask.floor_()

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob}"


def build_dropout(cfg):
    """Build a dropout/droppath layer from an mmcv-style dict."""
    if cfg is None:
        return nn.Identity()
    cfg = dict(cfg)
    t = cfg.pop("type")
    if t == "DropPath":
        return DropPath(cfg.get("drop_prob", 0.0))
    if t == "Dropout":
        return nn.Dropout(cfg.get("drop_prob", cfg.get("p", 0.5)))
    raise KeyError(f"unsupported dropout {t!r}")


def build_conv_layer(cfg, *args, **kwargs):
    """Build a conv layer; only plain Conv2d is used by the vendored nets."""
    if cfg is not None and cfg.get("type") not in (None, "Conv2d", "Conv"):
        raise KeyError(f"unsupported conv type {cfg.get('type')!r}")
    return nn.Conv2d(*args, **kwargs)


# ── inert stand-ins for the runner/registry plumbing ─────────────────────────
class BaseModule(nn.Module):
    """nn.Module plus the `init_cfg` attribute the reference classes expect.

    mmengine's BaseModule also drives a weight-initialisation framework we do not
    use: pretrained weights are loaded explicitly by `bruisekit.weights`, where
    it is visible, rather than as a side effect of construction.
    """

    def __init__(self, init_cfg=None):
        super().__init__()
        self.init_cfg = init_cfg

    def init_weights(self):
        return None


class _Registry:
    """No-op stand-in for mmseg's MODELS registry.

    The vendored files decorate their classes with `@MODELS.register_module()`.
    We keep the decorator working (returning the class unchanged) so the source
    stays verbatim, and record the classes in case they are ever useful.
    """

    def __init__(self):
        self._registry = {}

    def register_module(self, name=None, module=None, force=False):
        def _register(cls):
            self._registry[name or cls.__name__] = cls
            return cls
        return _register(module) if module is not None else _register

    def get(self, name):
        return self._registry.get(name)


MODELS = _Registry()


def print_log(msg, logger=None, level=None):
    """mmengine's logger, reduced to print."""
    print(msg)


class CheckpointLoader:
    """Only the entry point the vendored code calls; we never let it run.

    `bruisekit.weights` owns every download and every load, so that provenance
    stays in one auditable place. If a vendored file tries to fetch a checkpoint
    on its own, that is a bug worth failing loudly on, not a convenience.
    """

    @staticmethod
    def load_checkpoint(filename, map_location=None, logger=None):
        raise RuntimeError(
            "Vendored nets must not load their own checkpoints. Use "
            "bruisekit.weights.provision()/load_pretrained() so the source and "
            f"provenance of every weight stays explicit. (asked for: {filename})")


def load_state_dict(module, state_dict, strict=False, logger=None):
    """mmengine-compatible loose load, returning the mismatch report."""
    return module.load_state_dict(state_dict, strict=strict)
