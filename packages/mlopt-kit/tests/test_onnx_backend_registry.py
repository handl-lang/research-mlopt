"""``mlopt_kit.onnx.register()`` seats the backend on the core registry."""

from __future__ import annotations

import subprocess
import sys

import pytest
from mlopt_core import backends
from mlopt_core.ports import Sink

import mlopt_kit.onnx


def test_register_seats_onnx_backend():
    snap = {
        name: reg for name, reg in backends.DEFAULT_REGISTRY.items()
    }
    try:
        backends.DEFAULT_REGISTRY.clear()
        mlopt_kit.onnx.register()
        reg = backends.get_backend("onnx")
        # Sink-only: the lift direction lives in the Rust
        # source-onnx crate, so the registration declares no source.
        assert reg.source is None
        with pytest.raises(backends.BackendPortsMissing):
            reg.source_instance()
        assert isinstance(reg.sink("onnx"), Sink)
        assert "onnx" in reg.targets
        assert [
            r.name
            for r in backends.find_backends_by_ops({"add", "mul"})
        ] == ["onnx"]
        with pytest.raises(backends.BackendPortsMissing):
            reg.build_backend()
    finally:
        backends.DEFAULT_REGISTRY.clear()
        for r in snap.values():
            backends.DEFAULT_REGISTRY.register(r)


def test_import_does_not_register():
    # Fresh interpreter semantics: importing mlopt_kit.onnx must not seat
    # anything — the register() call is explicit (kernelopt pattern).
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import mlopt_kit.onnx, mlopt_core.backends as b;"
            "print(b.backend_names())",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "onnx" not in out.stdout
