"""Export the re-ID network to ONNX (dynamic batch) and check it against PyTorch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from multicam_tracker.reid.model import load_checkpoint


def export_reid_onnx(checkpoint, out, opset: int = 17, check: bool = True) -> Path:
    model, meta = load_checkpoint(checkpoint, "cpu")
    h, w = meta.get("input_size", [256, 128])
    dummy = torch.randn(2, 3, h, w)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    names = {"input_names": ["images"], "output_names": ["embeddings"], "opset_version": opset}
    try:  # PyTorch >= 2.5: the torch.export-based exporter with a symbolic batch dimension
        batch = torch.export.Dim("batch", min=1, max=4096)
        torch.onnx.export(model, (dummy,), str(out), dynamic_shapes={"x": {0: batch}}, dynamo=True,
                          verbose=False, **names)
    except TypeError:  # older PyTorch: TorchScript-based exporter
        torch.onnx.export(model, (dummy,), str(out),
                          dynamic_axes={"images": {0: "batch"}, "embeddings": {0: "batch"}}, **names)
    import onnx

    proto = onnx.load(str(out))
    for key, value in (("input_size", f"{h},{w}"), ("feat_dim", str(meta.get("feat_dim", "")))):
        entry = proto.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.save(proto, str(out))
    if check:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        x = torch.randn(3, 3, h, w)
        with torch.no_grad():
            ref = model(x).numpy()
        got = sess.run(None, {"images": x.numpy()})[0]
        err = float(np.abs(ref - got).max())
        if err > 1e-3:
            raise RuntimeError(f"ONNX export mismatch: max abs error {err:.2e}")
    return out
