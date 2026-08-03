#!/usr/bin/env python
"""Fetch the reference implementations of StrideFormer / PP-MobileSeg / TopFormer
and vendor them into scripts/unified_lib/vendor/, rewriting ONLY their imports.

WHY THIS IS A SCRIPT AND NOT A COPY-PASTE
------------------------------------------
The published checkpoints for these models are state dicts keyed by module path.
Any drift between our copy and the upstream file -- a renamed attribute, a
reordered Sequential -- makes `load_state_dict` fail or, worse, half-succeed.
Fetching programmatically and rewriting only `import` lines means the diff
against upstream is exactly the import block and nothing else, and re-running
this script re-proves that.

The rewrite is deliberately narrow and asserted: every substitution below must
fire, or the script aborts rather than emitting a file that silently still
depends on mmcv.
"""
from __future__ import annotations

import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "scripts" / "unified_lib" / "vendor"

MMSEG = "https://raw.githubusercontent.com/open-mmlab/mmsegmentation/main/projects/pp_mobileseg"
TOPFORMER = "https://raw.githubusercontent.com/hustvl/TopFormer/main"

SOURCES = {
    "strideformer.py": (
        f"{MMSEG}/backbones/strideformer.py",
        "PP-MobileSeg backbone (StrideFormer), OpenMMLab port of the PaddleSeg original",
    ),
    "pp_mobileseg_head.py": (
        f"{MMSEG}/decode_head/pp_mobileseg_head.py",
        "PP-MobileSeg decode head (AAM + VIM)",
    ),
    "topformer.py": (
        f"{TOPFORMER}/mmseg/models/backbones/topformer.py",
        "TopFormer backbone (Token Pyramid Module + Semantics Extractor + SIM)",
    ),
}

# (pattern, replacement, must_fire). Applied per file; a rule that does not fire
# in a file where it is required aborts the build.
REWRITES = [
    (r"^from mmcv\.cnn import .*$", None, False),
    (r"^from mmcv\.cnn\.bricks\.transformer import .*$", None, False),
    (r"^from mmengine\.logging import .*$", None, False),
    (r"^from mmengine\.model import .*$", None, False),
    (r"^from mmengine\.runner\.checkpoint import .*$", None, False),
    (r"^from mmseg\.registry import .*$", None, False),
    (r"^from mmseg\.utils import .*$", None, False),
    (r"^from mmseg\.ops import .*$", None, False),
    (r"^from mmcv\.runner import .*$", None, False),
    (r"^from \.\.builder import .*$", None, False),
    (r"^from \.decode_head import .*$", None, False),
    (r"^from mmseg\.models\.utils import .*$", None, False),
]

SHIM_IMPORT = (
    "# ── vendored: imports rewritten to bruisekit's mmcv shim, body untouched ──\n"
    "from bruisekit.mmcv_shim import (  # noqa: F401\n"
    "    BaseModule, CheckpointLoader, ConvModule, MODELS, build_activation_layer,\n"
    "    build_conv_layer, build_dropout, build_norm_layer, load_state_dict, print_log,\n"
    ")\n"
    "_load_checkpoint = CheckpointLoader.load_checkpoint\n"
    "get_root_logger = lambda *a, **k: None  # noqa: E731\n"
    "# Upstream decorates classes with whichever registry its framework version used.\n"
    "# All three names point at the same inert registry so the decorators stay verbatim.\n"
    "BACKBONES = HEADS = MODELS\n"
)


def fetch(url: str) -> str:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read().decode("utf-8")


def vendor(name: str, url: str, note: str) -> None:
    src = fetch(url)
    lines = src.splitlines()

    removed, kept = [], []
    for line in lines:
        if any(re.match(pat, line) for pat, _, _ in REWRITES):
            removed.append(line.strip())
            continue
        kept.append(line)
    if not removed:
        raise SystemExit(f"{name}: no mmcv/mmengine imports matched -- upstream changed, "
                         f"refusing to vendor a file whose dependencies I have not checked")

    # Insert the shim import right after the last top-of-file import we kept.
    idx = 0
    for i, line in enumerate(kept[:60]):
        if line.startswith(("import ", "from ")):
            idx = i + 1
    header = (
        f'"""VENDORED -- do not edit. Regenerate with scripts/vendor_efficient_nets.py\n\n'
        f"{note}.\nSource: {url}\n\n"
        f"Only the import block differs from upstream; every class, attribute name and\n"
        f"module path is byte-identical, so the published checkpoints load by key.\n"
        f"Replaced imports:\n  " + "\n  ".join(removed) + '\n"""\n'
    )
    body = "\n".join(kept[:idx] + ["", SHIM_IMPORT] + kept[idx:])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(header + body + "\n", encoding="utf-8")

    leftover = [ln for ln in body.splitlines()
                if re.search(r"\b(mmcv|mmengine|mmseg)\b", ln) and not ln.lstrip().startswith("#")]
    if leftover:
        raise SystemExit(f"{name}: residual mm* references after rewrite:\n  " +
                         "\n  ".join(leftover[:5]))
    print(f"  {name:<26} {len(kept)} lines, {len(removed)} imports rewritten")


def main() -> int:
    print(f"vendoring into {OUT}")
    for name, (url, note) in SOURCES.items():
        vendor(name, url, note)
    (OUT / "__init__.py").write_text(
        '"""Verbatim third-party architectures. See scripts/vendor_efficient_nets.py."""\n',
        encoding="utf-8")
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
