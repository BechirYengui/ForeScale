"""Pytest path bootstrap.

Ensures the repo root, the shared ``libs`` package, and ``experiments`` are
importable without an editable install, so ``pytest`` works from a fresh
checkout (and in CI) with no extra setup.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
for _p in (_ROOT, os.path.join(_ROOT, "libs"), os.path.join(_ROOT, "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
