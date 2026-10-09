"""Offline Effect (E) expanded audit entry point.

The original internal helpers ``eval_codec``, ``eval_geometry``, and
``eval_tuning`` are not part of this public release. This module keeps a
stable CLI surface and fails with an explicit message instead of an
obscure ``ImportError``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_MISSING = (
    "Offline E expanded evaluation is not included in this public AFCE release.\n"
    "Missing internal modules: afce.eval_codec, afce.eval_geometry, afce.eval_tuning.\n"
    "Use the DexJoCo policy rollout path instead:\n"
    "  python -m afce.serve_finger_delta_pi --help\n"
    "  python -m afce.finger_delta_eval_worker --help\n"
    "See experiments/finger_delta/README.md for the supported pipeline."
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline Effect expanded audit (not shipped in this release)."
    )
    for key in ("checkpoint", "data", "evidence", "output"):
        parser.add_argument("--" + key, type=Path, required=False)
    parser.add_argument("--observed-visual", type=Path)
    parser.parse_args(argv)
    print(_MISSING, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
