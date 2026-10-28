"""Check fake quantization against actual reduced-range ORT integer decoder."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runtime
import torch, numpy as np, json
import onnxruntime as ort
from distill.torch_graph import VectorGraph

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
x = np.load("distill/trajectories-v1/train/000_7.npz")["target"]
m = VectorGraph("models/supertonic-3/onnx/vocoder.onnx")
m.qat = True
with torch.no_grad():
    actual = m({"latent": x})[0].cpu().numpy()
s = ort.InferenceSession(
    "models/fast-avx2-int8/onnx/vocoder.onnx",
    runtime.options(),
    providers=["CPUExecutionProvider"],
)
reference = s.run(None, {"latent": x})[0]
r = {
    "relative_rmse": float(
        np.linalg.norm(actual - reference) / np.linalg.norm(reference)
    ),
    "max_error": float(np.abs(actual - reference).max()),
    "note": "One latent input; compare fake U8 activation/S7 per-channel pointwise Conv with ORT reduced-range INT8 MatMul conversion",
}
Path("results/qat_decoder_parity.json").write_text(json.dumps(r, indent=2))
print(r)
