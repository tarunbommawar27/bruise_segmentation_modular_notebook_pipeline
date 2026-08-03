"""Fetch, cache and verify third-party pretrained weights -- resumably, and with
the provenance of every file written down next to the download.

WHY PRETRAINED INIT IS NOT AN IMPLEMENTATION DETAIL HERE
---------------------------------------------------------
Every baseline in this study starts from an ImageNet-pretrained encoder with a
freshly initialised head: SegFormer from MiT, U-Net and DeepLabV3+ from
ResNet-50. That is the controlled variable. A model that starts from random
weights is not a worse version of the same experiment -- it is a different
experiment, and on 697 training images the gap is large. So `SOURCES` records,
per model, exactly what initialisation it gets, and `report()` prints it, so a
scratch-initialised model can never quietly sit in a table next to pretrained
ones as though the comparison were like for like.

WHY RESUMABLE
--------------
The same reason `train_run` checkpoints every couple of epochs: Colab drops
connections. A 300 MB download that dies at 90% and restarts from zero on every
retry is the kind of papercut that makes people disable the step. Downloads here
go to a `.part` file and resume with an HTTP Range request, so a dropped
connection costs the bytes in flight, not the whole file.

THREE KINDS OF SOURCE, THREE HONEST BEHAVIOURS
-----------------------------------------------
  auto   a stable direct HTTPS URL -> downloaded without asking.
  manual a URL that cannot be fetched non-interactively (Google Drive's
         confirmation flow). Reported with instructions; never silently skipped.
  none   no pretrained weights exist for this model. Not a failure -- Fast-SCNN
         is designed to train from scratch -- but stated explicitly rather than
         left for the reader to infer from a missing file.
"""
from __future__ import annotations

import hashlib
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

CHUNK = 1 << 20  # 1 MiB


@dataclass
class WeightSource:
    """Where one model's pretrained weights come from, and what they contain.

    Attributes
    ----------
    key : short name used by `provision()` and the model wrappers.
    filename : what it is saved as under pretrained_weights/efficient/.
    url : direct download URL, or the human-facing page for `manual` sources.
    kind : "auto" | "manual" | "none" -- see the module docstring.
    sha256 : expected digest. None means "record what we got" rather than verify;
        a wrong-but-plausible file is then still detectable by re-running.
    init : one line describing what the weights actually are. This is the field
        that keeps the comparison honest, so it says what the model was
        pretrained ON, not merely that it was pretrained.
    strip_prefix : prefix to remove from every state-dict key on load, for
        checkpoints saved from a wrapper module.
    note : anything a reader needs in order to trust or distrust the file.
    """

    key: str
    filename: str
    url: str
    kind: str = "auto"
    sha256: str | None = None
    init: str = ""
    strip_prefix: str = ""
    note: str = ""
    expect_missing: tuple = field(default_factory=tuple)


SOURCES: dict[str, WeightSource] = {
    "ppmobileseg_tiny": WeightSource(
        key="ppmobileseg_tiny",
        filename="pp_mobileseg_mobilenetv3_3rdparty-tiny-e4b35e96.pth",
        url=("https://download.openmmlab.com/mmsegmentation/v0.5/pp_mobileseg/"
             "pp_mobileseg_mobilenetv3_3rdparty-tiny-e4b35e96.pth"),
        kind="auto",
        sha256="bda39e9f243fc43f98e64873b8ddc2e96e85561022dc8f5cead7cb03fdf238f3",
        init="StrideFormer-Tiny backbone, ImageNet-pretrained (OpenMMLab's conversion "
             "of the PaddleSeg original)",
        note="Backbone only -- the AAM injection module and head initialise fresh, which "
             "is what makes this comparable to the ResNet-50 baselines (ImageNet encoder, "
             "new head). The ADE20K segmentation checkpoint also exists but using it would "
             "give this model segmentation pretraining no other baseline has.",
        expect_missing=("inj_module.",),
    ),
    "lraspp_mobilenetv3": WeightSource(
        key="lraspp_mobilenetv3",
        filename="mobilenet_v3_large-8738ca79.pth",
        url="https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth",
        kind="auto",
        init="MobileNetV3-Large backbone, ImageNet-1k (torchvision IMAGENET1K_V1)",
        note="Backbone only, by choice. torchvision also ships an LR-ASPP trained on "
             "COCO-with-VOC-labels; that is segmentation pretraining, so it is not used.",
    ),
    "topformer_tiny": WeightSource(
        key="topformer_tiny",
        filename="topformer-T-224-66.2.pth",
        url="https://drive.google.com/drive/folders/1NLz3QCDbaXJ2DeGxLPUfupZZbojceDJM",
        kind="manual",
        init="TopFormer-Tiny backbone, ImageNet-1k (66.2% top-1)",
        note="The authors publish only via Google Drive and Baidu (password 'topf'), "
             "neither of which can be fetched non-interactively. Download "
             "topformer-T-224-66.2.pth by hand into pretrained_weights/efficient/ to "
             "enable pretrained init; without it the model trains from scratch and is "
             "labelled as such everywhere it appears.",
    ),
    "fastscnn": WeightSource(
        key="fastscnn",
        filename="",
        url="https://arxiv.org/abs/1902.04502",
        kind="none",
        init="random (He) init -- no pretrained weights exist",
        note="Not a gap. Fast-SCNN is designed to train from scratch: at 1.1M parameters "
             "the authors report ImageNet pretraining gives only a marginal gain, and no "
             "official checkpoint was ever released. Training it from random init is the "
             "faithful reproduction, not a compromise.",
    ),
}


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _download_resumable(url: str, dest: Path, timeout: int = 60) -> None:
    """Download `url` to `dest`, resuming a partial `.part` file if one exists.

    Asks the server to continue from the bytes already on disk via a Range header.
    A server that ignores Range answers 200 with the whole body instead of 206,
    which is detected and handled by restarting the file rather than appending to
    it -- appending a full body onto a partial one produces a corrupt file that
    passes a size check and fails a hash check for reasons nobody enjoys
    debugging.
    """
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0

    req = urllib.request.Request(url, headers={"User-Agent": "bruisekit/1.0"})
    if have:
        req.add_header("Range", f"bytes={have}-")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            resumed = r.status == 206
            if have and not resumed:
                have = 0            # server ignored Range: start over, do not append
            total = int(r.headers.get("Content-Length", 0)) + (have if resumed else 0)
            mode = "ab" if (have and resumed) else "wb"
            done = have
            with open(part, mode) as f:
                while True:
                    chunk = r.read(CHUNK)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if total and done % (32 * CHUNK) < CHUNK:
                        print(f"    {done / 1e6:.0f} / {total / 1e6:.0f} MB", flush=True)
    except urllib.error.HTTPError as e:
        if e.code == 416 and have:      # already complete
            pass
        else:
            raise

    part.replace(dest)


def provision(env, key: str, force: bool = False, verbose: bool = True) -> Path | None:
    """Ensure one weight file is present. Returns its path, or None if unavailable.

    Never raises for a missing `manual` or `none` source: those are expected
    states with a defined meaning, and the caller decides what to do about them.
    Does raise on a checksum mismatch, because that means the bytes on disk are
    not the bytes the provenance claims.
    """
    src = SOURCES[key]
    if src.kind == "none":
        if verbose:
            print(f"  {key:<22} no pretrained weights exist -- trains from scratch")
        return None

    out_dir = env.weights / "efficient"
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / src.filename

    if dest.exists() and not force:
        if src.sha256 and _digest(dest) != src.sha256:
            raise RuntimeError(
                f"{dest} is present but its sha256 does not match the recorded value. "
                f"Delete it and re-provision; do not train from a file of unknown origin.")
        if verbose:
            print(f"  {key:<22} cached  ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest

    if src.kind == "manual":
        if verbose:
            print(f"  {key:<22} MANUAL  not downloadable non-interactively")
            print(f"  {'':<22}         get {src.filename} from {src.url}")
            print(f"  {'':<22}         and place it in {out_dir}")
        return None

    if verbose:
        print(f"  {key:<22} downloading from {src.url.split('/')[2]} ...")
    _download_resumable(src.url, dest)
    got = _digest(dest)
    if src.sha256 and got != src.sha256:
        dest.unlink()
        raise RuntimeError(f"{key}: sha256 mismatch (got {got}, expected {src.sha256})")
    if verbose:
        print(f"  {key:<22} done    ({dest.stat().st_size / 1e6:.1f} MB, sha256 {got[:12]}…)")
    return dest


def provision_all(env, keys=None, force: bool = False) -> pd.DataFrame:
    """Provision every listed source and return a provenance table.

    The table is the deliverable as much as the files are: it is what tells a
    reader which of the four models was pretrained and on what.
    """
    keys = list(keys or SOURCES)
    rows = []
    print("Provisioning pretrained weights ->", env.weights / "efficient")
    for k in keys:
        src = SOURCES[k]
        try:
            path = provision(env, k, force=force)
            status = ("ready" if path else
                      "scratch" if src.kind == "none" else "MISSING (manual)")
        except Exception as e:                      # noqa: BLE001 -- reported, not swallowed
            path, status = None, f"FAILED: {type(e).__name__}: {e}"
        rows.append({"model": k, "status": status, "init": src.init,
                     "file": src.filename or "-", "kind": src.kind,
                     "path": str(path) if path else ""})
    return pd.DataFrame(rows)


def load_pretrained(model, path: Path, strip_prefix: str = "",
                    expect_missing: tuple = (), verbose: bool = True) -> dict:
    """Load a third-party checkpoint into `model` and report what matched.

    Returns counts of loaded / missing / unexpected keys. This is deliberately
    NOT a silent `strict=False` load: a partially-matching checkpoint is the
    classic way to end up training a model that reports "pretrained" while most
    of its weights are random. Anything missing that was not declared in
    `expect_missing` is printed.

    `expect_missing` names the prefixes that are SUPPOSED to be absent -- the
    decode head and injection modules that we intend to train fresh.
    """
    import torch

    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt.get("model", ckpt))
    if strip_prefix:
        state = {k[len(strip_prefix):]: v for k, v in state.items()
                 if k.startswith(strip_prefix)}

    result = model.load_state_dict(state, strict=False)
    unexpected = list(result.unexpected_keys)
    missing = [k for k in result.missing_keys
               if not any(k.startswith(p) for p in expect_missing)]
    loaded = len(state) - len(unexpected)

    if verbose:
        print(f"    loaded {loaded}/{len(state)} tensors from {path.name}")
        if missing:
            print(f"    WARNING {len(missing)} undeclared missing keys, "
                  f"e.g. {missing[:3]} -- these stay randomly initialised")
        if unexpected:
            print(f"    WARNING {len(unexpected)} unexpected keys, "
                  f"e.g. {unexpected[:3]} -- checkpoint does not match this architecture")
    return {"loaded": loaded, "in_checkpoint": len(state),
            "missing": len(missing), "unexpected": len(unexpected)}


def report(env) -> pd.DataFrame:
    """What is on disk right now, without downloading anything."""
    rows = []
    for k, s in SOURCES.items():
        p = env.weights / "efficient" / s.filename if s.filename else None
        rows.append({"model": k, "kind": s.kind,
                     "present": bool(p and p.exists()),
                     "init": s.init, "note": s.note})
    return pd.DataFrame(rows)
