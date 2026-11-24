"""``mlopt_kit`` — the Python surface of the mlopt-kit crate.

The crate itself (``src/``, ``hnl/``, ``hnl-run``) is the handl
composition seam; this package carries the Python half folded in at
T-075(b/c) when the ``kit-onnx`` repo was dissolved: the ONNX backend
(``mlopt_kit.onnx``) — ``OnnxSink`` lowers mlopt IR to ONNX,
``OnnxExecutor`` runs the artifact through ``onnxruntime``.
"""
