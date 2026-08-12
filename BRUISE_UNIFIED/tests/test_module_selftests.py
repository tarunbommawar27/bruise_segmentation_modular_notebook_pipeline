"""Run every module's `self_test()` under pytest.

WHY THIS FILE IS SHORT ON PURPOSE
-----------------------------------
Eleven modules in `bruisekit/` already carry a `self_test()` that asserts the
structural invariants their author cared about -- that `wrong_place` really is
`zero_dice - empty_pred`, that a K=1 routed loss reduces to the Stage H gated
loss exactly, that `find_sam_blocks` raises rather than returning empty, that a
probability threshold is converted to a logit and not read as one. Those checks
are the real test suite. They were just never wired to a runner, so nobody ran
them unless they happened to execute the module as a script.

This collects them. It deliberately does NOT re-implement any of them here: a
second copy of an invariant is a second thing to keep in sync, and the version
that ships beside the code is the one that gets updated when the code changes.

Every self-test in this list runs with no GPU, no checkpoints, no manifests and
no network. Anything that needs those is skipped with the reason stated, rather
than silently passing.
"""
from __future__ import annotations

import importlib

import pytest

#: (module, reason-to-skip-if-import-fails). Import failure is usually a missing
#: optional dependency, not a broken module -- `segmentation_models_pytorch` for
#: the baselines, `ultralytics` for YOLO -- so it is reported as a skip and the
#: rest of the suite still runs.
MODULES = [
    "bruisekit.allmodels",
    "bruisekit.dermprobe",
    "bruisekit.fenwick_cv",
    "bruisekit.finetune_n3",
    "bruisekit.foundation",
    "bruisekit.itakd",
    "bruisekit.lesionsize",
    "bruisekit.multiteacher",
    "bruisekit.reliability_kd",
    "bruisekit.samprobe",
    "bruisekit.efficient_models",
]


def _call(fn):
    """Call a self-test whichever signature it has.

    THREE CONVENTIONS EXIST IN THIS CODEBASE and all three are legitimate:

        self_test(verbose=True) -> bool         most modules
        self_test(verbose=True) -> DataFrame    reliability_kd, efficient_models
                                                (a per-check report)
        self_test()             -> bool         fenwick_cv, no verbose kwarg

    The runner adapts to them rather than the eleven modules being rewritten to
    match the runner. Rewriting them would touch every stage's shipped module
    immediately before a release push, to fix nothing that is actually broken.
    """
    import inspect

    if "verbose" in inspect.signature(fn).parameters:
        return fn(verbose=True)
    return fn()


def _passed(result) -> tuple[bool, str]:
    """Did the self-test pass? Handles bool and per-check DataFrame reports."""
    if isinstance(result, bool):
        return result, ""
    if hasattr(result, "columns"):                   # a DataFrame report
        import pandas as pd

        flags = [c for c in result.columns
                 if pd.api.types.is_bool_dtype(result[c])]
        if not flags:
            return False, (f"returned a DataFrame with no boolean check column; "
                           f"columns are {list(result.columns)}")
        bad = result[~result[flags].all(axis=1)]
        if len(bad):
            return False, f"{len(bad)} row(s) failed:\n{bad.to_string(index=False)}"
        return True, ""
    return False, f"returned {type(result).__name__}, expected bool or DataFrame"


@pytest.mark.parametrize("name", MODULES)
def test_self_test(name):
    try:
        mod = importlib.import_module(name)
    except ImportError as exc:                      # optional dependency absent
        pytest.skip(f"{name} not importable here: {exc}")

    fn = getattr(mod, "self_test", None)
    if fn is None:
        pytest.skip(f"{name} has no self_test()")

    ok, why = _passed(_call(fn))
    assert ok, f"{name}.self_test() reported a failure. {why}"


def test_every_selftest_is_registered():
    """A module that grows a `self_test()` must be added to MODULES.

    Without this, adding a module and forgetting to list it means its checks
    never run and nothing says so -- the failure mode this file exists to fix,
    reappearing one module at a time.
    """
    import pkgutil

    import bruisekit

    missing = []
    for info in pkgutil.iter_modules(bruisekit.__path__):
        dotted = f"bruisekit.{info.name}"
        if dotted in MODULES or info.ispkg:
            continue
        try:
            mod = importlib.import_module(dotted)
        except Exception:                            # noqa: BLE001 -- see above
            continue
        if callable(getattr(mod, "self_test", None)):
            missing.append(dotted)

    assert not missing, (
        f"these modules have a self_test() that this file never runs: {missing}. "
        f"Add them to MODULES.")
