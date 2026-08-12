"""Where everything lives, decided once, for every host this notebook can run on.

THE ONE-VARIABLE RULE
----------------------
The unified notebook must run unchanged on Colab, on a local Jupyter server, and
on a headless box. The only thing that differs between them is where the bundle
sits and whether a GPU exists. So exactly one cell -- the one that calls
`setup()` -- is allowed to know about the host. Everything downstream addresses
files through the returned `Env`, and no other module in this package contains
the string "/content" or imports `google.colab`.

That is not tidiness for its own sake. The previous generation of these notebooks
hard-coded `/content` in nine separate cells; moving them to a local machine meant
finding and editing all nine, and missing one produced a FileNotFoundError forty
minutes into a run. One variable cannot be half-changed.

RESOLUTION ORDER
----------------
1. An explicit path passed to `setup(root=...)`             -- you said so.
2. $BRUISE_BUNDLE                                            -- CI and scripts.
3. Colab: Drive is mounted, then the usual Drive locations are probed.
4. The current directory and its parents, then the obvious siblings.
A directory counts as the bundle only if it contains BOTH `bruisekit/` and
`manifests/train.csv`, so a half-copied folder fails loudly here rather than
silently later.

WORK DIRECTORY vs BUNDLE ROOT
------------------------------
The bundle may live on a slow, network-backed filesystem -- Google Drive is the
common case, where reading 1016 images per epoch is ruinous. `Env.work` therefore
points at fast local scratch, and derived artefacts (the 640 cache, YOLO datasets)
are written there, never next to the read-only bundle. On a local machine with no
such split, work defaults to a `_work` folder inside the bundle.
"""
from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

# A directory is the bundle only if both of these resolve inside it.
_MARKERS = ("bruisekit", "manifests/train.csv")


def _is_bundle(p: Path) -> bool:
    """True when `p` looks like a complete BRUISE_UNIFIED root."""
    try:
        return all((p / m).exists() for m in _MARKERS)
    except OSError:
        return False


def in_colab() -> bool:
    """True inside a Google Colab runtime.

    Checked by import rather than by looking for /content, because /content also
    exists in some Docker images and the import is the thing that actually tells
    us whether `google.colab.drive` is callable.
    """
    return "google.colab" in sys.modules or _can_import("google.colab")


def _can_import(name: str) -> bool:
    import importlib.util
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def mount_drive(mountpoint: str = "/content/drive") -> Path | None:
    """Mount Google Drive if we are on Colab and it is not already mounted.

    Returns the MyDrive path, or None when not on Colab. Safe to call twice: Colab
    raises if you mount an already-mounted point, so we check for the marker
    directory first.
    """
    if not in_colab():
        return None
    mp = Path(mountpoint)
    if not (mp / "MyDrive").exists():
        from google.colab import drive  # noqa: PLC0415 -- Colab-only import, guarded above
        drive.mount(mountpoint)
    return mp / "MyDrive"


def _candidates(explicit: str | Path | None) -> list[Path]:
    """Every place worth looking for the bundle, in decreasing order of confidence."""
    out: list[Path] = []
    if explicit:
        out.append(Path(explicit).expanduser().resolve())
    env = os.environ.get("BRUISE_BUNDLE")
    if env:
        out.append(Path(env).expanduser().resolve())

    if in_colab():
        my = Path("/content/drive/MyDrive")
        out += [Path("/content/BRUISE_UNIFIED"),
                my / "BRUISE_UNIFIED",
                my / "bruise_segmentation_gpu" / "BRUISE_UNIFIED"]

    here = Path.cwd().resolve()
    for base in (here, *here.parents):
        out += [base, base / "BRUISE_UNIFIED"]
    # The notebook usually sits inside the bundle, so its own folder is a strong hint.
    out.append(Path(__file__).resolve().parent.parent)
    return out


def find_root(explicit: str | Path | None = None) -> Path:
    """Locate the bundle root, or raise with the list of places already tried."""
    tried: list[Path] = []
    for c in _candidates(explicit):
        if c in tried:
            continue
        tried.append(c)
        if _is_bundle(c):
            return c
    listing = "\n  ".join(str(t) for t in tried[:12])
    raise FileNotFoundError(
        "Could not find the BRUISE_UNIFIED bundle. A directory qualifies only if it "
        f"contains {' and '.join(_MARKERS)}.\nTried:\n  {listing}\n"
        "Fix: setup(root='/path/to/BRUISE_UNIFIED') or export BRUISE_BUNDLE.")


def pick_device(prefer: str | None = None):
    """Return the best available torch device, preferring CUDA, then MPS, then CPU.

    Deliberately does NOT raise on CPU. The whole point of the results/ fallback
    tier is that the analysis reproduces on a laptop; refusing to start without a
    GPU -- which the original training notebooks did, correctly, for THEIR job --
    would defeat that here. Stages that genuinely need CUDA check for it when they
    are actually asked to train.

    Nor does it raise when torch is absent. Stage D is pure pandas: it reads
    per-image CSVs and computes statistics, and needs no deep-learning stack at
    all. Making the environment resolver hard-depend on torch would mean a
    reviewer who only wants the tables has to install a GPU framework first. When
    torch is missing we return the string "cpu", which every consumer in this
    package treats identically to torch.device("cpu"); anything that actually
    needs a tensor imports torch itself and fails there, with a clear message.
    """
    try:
        import torch
    except ImportError:
        return "cpu"
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class Env:
    """Every path and host fact the notebook needs, resolved once.

    Attributes
    ----------
    root : the bundle. Treat as read-only.
    work : fast scratch for derived artefacts (640 cache, YOLO datasets, new runs).
    device : the torch device chosen by `pick_device`.
    colab : whether we are in a Colab runtime.
    """

    root: Path
    work: Path
    device: object
    colab: bool = False
    # Extra directories holding `<family>__seed<N>/` run folders -- typically a
    # scratch tree from another machine. Searched after `runs` and before the
    # bundle's shipped checkpoints, so your own training always wins over a
    # shipped copy but a shipped copy still answers when you have not trained it.
    extra_runs: tuple = ()
    _made: list = field(default_factory=list, repr=False)

    # ── read-only bundle locations ───────────────────────────────────────────
    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def manifests(self) -> Path:
        return self.root / "manifests"

    @property
    def weights(self) -> Path:
        return self.root / "pretrained_weights"

    @property
    def ckpt(self) -> Path:
        return self.root / "checkpoints"

    @property
    def results(self) -> Path:
        return self.root / "results"

    # ── writable scratch ─────────────────────────────────────────────────────
    @property
    def cache640(self) -> Path:
        """The 640x640 PNG cache the training/eval dataloaders read.

        Derived, not shipped: it is a deterministic resize of data/ (bilinear for
        images, nearest for masks) that rebuilds in about a minute. Shipping it
        would add ~400 MB of redundant bytes and create a second copy of the
        pixels that could drift from the first.
        """
        return self.work / "cache640"

    @property
    def runs(self) -> Path:
        """Where NEW training runs are WRITTEN. Never the shipped checkpoints.

        Kept separate from `ckpt` on purpose: a retrain must not overwrite the
        weights that produced the shipped results.

        For READING, use `run_roots` -- checkpoints for this study live in at
        least four places with three different on-disk layouts, and a single
        directory cannot see them all.
        """
        return self.work / "runs"

    @property
    def run_roots(self) -> list[Path]:
        """Every directory to SEARCH for a checkpoint, in priority order.

        WHY A SEARCH PATH AND NOT A DIRECTORY
        --------------------------------------
        The study's checkpoints are not in one place and never were:

            <work>/runs/                      runs trained by this session
            <extra_runs>/                     e.g. /scratch/$USER/bruise_work/runs
            <bundle>/checkpoints/final/       the five Stage A models, 3 seeds
            <bundle>/checkpoints/baselines/   U-Net, DeepLabV3+, 3 seeds
            <bundle>/checkpoints/efficient/   Stage E/F mobile arms
            <bundle>/checkpoints/rgkd/        Stage H arms
            <bundle>/checkpoints/distill/teachers/   B2, B5, B0-distilled

        Before this existed the registry looked in exactly two -- `env.runs` and
        one stage-specific `ckpt` subdirectory -- and fell back SILENTLY between
        them. On 2026-08-04 that made every SegFormer load an old-lineage
        checkpoint from the bundle because the real runs were in a scratch
        directory nobody had pointed at, and the only symptom was a crash three
        steps later. A model you have on disk and cannot see is worse than one
        you do not have: the second is reported as a gap, the first is silently
        substituted.

        Order is priority: the first root holding a usable checkpoint wins, and
        `env.runs` is first so a fresh retrain always beats a shipped copy.
        Provenance travels with every `Run` (`source_root`), so which one
        answered is a fact in the plan table rather than something to deduce.
        """
        roots = [self.runs, *self.extra_runs]
        ck = self.ckpt
        roots += [ck / "final", ck / "baselines", ck / "efficient", ck / "rgkd",
                  ck / "yolo_l", ck / "distill" / "teachers"]
        # De-duplicate while preserving order: extra_runs may legitimately name a
        # directory that is already implied, and searching it twice would make
        # every "which root answered?" line ambiguous.
        seen, out = set(), []
        for r in roots:
            rp = Path(r).expanduser()
            if rp not in seen:
                seen.add(rp)
                out.append(rp)
        return out

    @property
    def out(self) -> Path:
        """Where THIS session's tables and figures are written."""
        return self.work / "outputs"

    def paths_for_models(self) -> dict:
        """The `paths` dict that `bruisekit.models.build_model` expects."""
        return {
            "segformer_b0": str(self.weights / "segformer_mit_b0"),
            "segformer_b2": str(self.weights / "segformer_mit_b2"),
            "segformer_b5": str(self.weights / "segformer_mit_b5"),
            "yolo": str(self.weights / "yolo26n-sem.pt"),
            # Stage Y. Absent from most bundles -- it is a ~50 MB download that
            # only a Stage Y session needs, so the path is always returned and
            # its existence is checked where it is used, not here.
            "yolo_l": str(self.weights / "yolo26l-sem.pt"),
        }

    def describe(self) -> str:
        gpu = ""
        if str(self.device).startswith("cuda"):
            import torch
            gpu = f" ({torch.cuda.get_device_name(0)})"
        free = shutil.disk_usage(self.work).free / 1e9
        return (f"host      : {'Colab' if self.colab else 'local Jupyter'}\n"
                f"device    : {self.device}{gpu}\n"
                f"bundle    : {self.root}\n"
                f"work dir  : {self.work}  ({free:.1f} GB free)")


def setup(root: str | Path | None = None,
          work: str | Path | None = None,
          device: str | None = None,
          mount: bool = True,
          extra_runs: "str | Path | list | tuple | None" = None) -> Env:
    """Resolve the environment. This is the only host-aware call in the notebook.

    Parameters
    ----------
    root : bundle location. Omit to auto-detect (see module docstring).
    work : scratch location. Omit for the sensible default per host -- local SSD
           on Colab, a `_work` folder inside the bundle elsewhere.
    device : e.g. "cpu" to force CPU-only reporting. Omit to auto-select.
    mount : mount Google Drive first when on Colab. Set False if already mounted
            or if the bundle was uploaded straight to /content.

    Returns
    -------
    Env -- pass it to everything else in this package.
    """
    colab = in_colab()
    if colab and mount:
        mount_drive()

    r = find_root(root)

    if work is not None:
        w = Path(work).expanduser().resolve()
    elif colab:
        # Local SSD. Wiped on disconnect, which is fine -- everything here is derived.
        w = Path("/content/bruise_work")
    else:
        w = r / "_work"
    w.mkdir(parents=True, exist_ok=True)

    # A bare string is the overwhelmingly common case ("the other machine's
    # scratch"), and treating it as an iterable of characters would produce a
    # search path of single-letter directories that all silently miss.
    if extra_runs is None:
        extra = ()
    elif isinstance(extra_runs, (str, Path)):
        extra = (Path(extra_runs).expanduser().resolve(),)
    else:
        extra = tuple(Path(p).expanduser().resolve() for p in extra_runs)

    env = Env(root=r, work=w, device=pick_device(device), colab=colab, extra_runs=extra)
    # Only the writable ones are created. A search root that does not exist is
    # not an error -- most bundles have no `checkpoints/rgkd/` -- and creating it
    # would turn "you have not trained this" into an empty directory that looks
    # like you have.
    for d in (env.cache640, env.runs, env.out):
        d.mkdir(parents=True, exist_ok=True)
    return env
