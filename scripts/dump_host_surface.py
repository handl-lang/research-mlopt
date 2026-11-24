#!/usr/bin/env python3
"""Refresh the vendored ``host_surface.json`` from exec-tensor.

The canonical ``tensor.*`` surface artifact moved to
``handl-lang/exec-tensor`` in the P-010 dissolution
(``tools/dump_host_surface.py`` → repo-root ``host_surface.json``).
This repo vendors a pinned copy at
``packages/mlopt/src/mlopt/host_surface.json`` for ``mlopt.emit``'s
drift guard (:mod:`mlopt.tensor_surface`) — bump it deliberately,
never by hand-editing the JSON.

Usage::

    .venv/bin/python scripts/dump_host_surface.py [--check]

``EXEC_TENSOR_DIR`` overrides the sibling-checkout default
(``../exec-tensor``).  ``--check`` exits non-zero when the vendored
copy differs — the drift gate.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
VENDORED = (
    REPO / "packages" / "mlopt" / "src" / "mlopt" / "host_surface.json"
)


def _exec_tensor_dir() -> Path:
    """Locate the exec-tensor checkout."""
    cand = Path(
        os.environ.get("EXEC_TENSOR_DIR", REPO.parent / "exec-tensor")
    )
    if not (cand / "tools" / "dump_host_surface.py").is_file():
        raise SystemExit(
            f"exec-tensor checkout not found at {cand} — set "
            "EXEC_TENSOR_DIR"
        )
    return cand


def _fresh_surface(ht: Path) -> bytes:
    """Regenerate the artifact in the checkout; return its bytes."""
    tool = ht / "tools" / "dump_host_surface.py"
    out = ht / "host_surface.json"
    chk = subprocess.run(
        [sys.executable, str(tool), "--check"],
        cwd=ht,
        capture_output=True,
        text=True,
    )
    if chk.returncode != 0:
        regen = subprocess.run(
            [sys.executable, str(tool)],
            cwd=ht,
            capture_output=True,
            text=True,
        )
        if regen.returncode != 0:
            raise SystemExit(
                f"exec-tensor surface regen failed: {regen.stderr.strip()}"
            )
    return out.read_bytes()


def main() -> int:
    """Sync (or check) the vendored artifact."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero when the vendored copy differs",
    )
    args = ap.parse_args()

    ht = _exec_tensor_dir()
    fresh = _fresh_surface(ht)
    current = VENDORED.read_bytes() if VENDORED.is_file() else b""

    if args.check:
        if current != fresh:
            print(
                f"{VENDORED} is stale — run scripts/dump_host_surface.py",
                file=sys.stderr,
            )
            return 1
        print(f"{VENDORED} is fresh")
        return 0

    VENDORED.write_bytes(fresh)
    print(f"wrote {VENDORED} from {ht}/host_surface.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
