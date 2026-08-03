"""Vendored distillation suite (Stage C), copied unmodified from the ORC bundle.

These modules import each other by flat module name (`import kd_core`), which is
how they were written and tested. Rather than rewrite those imports -- and thereby
modify code whose outputs ship in results/ -- this package puts its own directory
on sys.path at import time. `from bruisekit import kd` is therefore enough to make
`import kd_core` resolve, from a notebook in any working directory.
"""
import sys
from pathlib import Path

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
