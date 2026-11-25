"""ONNX backend for mlopt — ADR 0006, sink direction.

``OnnxSink`` implements ``mlopt_core.ports.Sink`` (IR -> ONNX,
verified through ``onnxruntime``) and ``OnnxExecutor`` runs the
lowered artifact — optimized programs leave for any ONNX runtime.

The *lift* direction (ONNX graph -> IR) is no longer here: the
Python ``OnnxSource``/``liftmap`` was proven at parity with the
Rust lifter (``handl-lang/tool-codegen-onnx``'s ``source_onnx`` module (merged from ``source-onnx``), surfaced to Python through
``mlopt.adapters.onnx_rust.OnnxRustSource``) and retired.  ONNX
models still enter the optimizer untouched by torch — through the
Rust producer, not this package.

The experiment this package *is*: ADR 0003/0006 claim that a new
backend implements the ports and nothing in ``mlopt-core`` changes.
The port boundary is what is under test, so the honest reading is
that the code here is half the deliverable and the answer to "did
core have to move?" is the other half.

    >>> from mlopt_kit.onnx import OnnxSink
    >>> sink = OnnxSink(dtype="float64")     # doctest: +SKIP
    >>> executor = sink.lower(ir, leaves)    # doctest: +SKIP

Nothing here imports torch, ``kit_torch``, ``kit_carriers`` or
``mlopt_orchestrator``: the ONNX backend is a *peer* of the torch
one, reachable through the same ports and depending only on
``mlopt_core``.
"""

from __future__ import annotations

from mlopt_kit.onnx.arrays import (
    ONNX_TYPE_NAMES,
    WORKING_DTYPES,
    numpy_to_onnx_type,
    to_numpy,
)
from mlopt_kit.onnx.errors import (
    OnnxLoweringError,
    UnsupportedOnnxOpError,
)
from mlopt_kit.onnx.executor import OnnxExecutor
from mlopt_kit.onnx.lower import GraphBuilder, lower_ir
from mlopt_kit.onnx.opmap import ONNX_BINDINGS
from mlopt_kit.onnx.report import (
    REL_FLOOR,
    OnnxVerifyReport,
    verify_equiv,
    verify_module,
)
from mlopt_kit.onnx.sink import OnnxSink

__all__ = [
    "ONNX_BINDINGS",
    "ONNX_TYPE_NAMES",
    "REL_FLOOR",
    "WORKING_DTYPES",
    "GraphBuilder",
    "OnnxExecutor",
    "OnnxLoweringError",
    "OnnxSink",
    "OnnxVerifyReport",
    "UnsupportedOnnxOpError",
    "lower_ir",
    "numpy_to_onnx_type",
    "register",
    "to_numpy",
    "verify_equiv",
    "verify_module",
]


def register() -> None:
    """Seat the ONNX backend on ``mlopt_core.backends.DEFAULT_REGISTRY``.

    Explicit seam — importing this package registers nothing; call
    ``mlopt_kit.onnx.register()`` at the entry point (the same pattern as
    ``mlopt_kernel.play.register()``).  Registers ``"onnx"`` as a
    **sink-only** backend: ``OnnxSink`` is the ``"onnx"`` target, a
    lazy factory resolved on demand, and ``source=None`` — the
    ONNX -> IR lift lives in the Rust ``source_onnx`` module now, so
    this package honestly declares no ``Source`` port.
    ``find_backends_by_ops`` capability queries work today;
    ``source_instance()``/``build_backend`` name the missing source.
    """
    from mlopt_core import backends

    backends.register_backend(
        "onnx",
        source=None,
        targets={"onnx": OnnxSink},
        notes="ONNX sink (ADR 0006); lift lives in tool-codegen-onnx source_onnx; "
        "composer/meter pending",
    )
