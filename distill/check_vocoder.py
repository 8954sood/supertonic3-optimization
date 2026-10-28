"""Parity and input-gradient check for the frozen differentiable audio decoder."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runtime
from distill.torch_graph import VectorGraph
import torch
import numpy as np
import json
import onnxruntime as ort

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
latent = np.load("distill/trajectories-v1/train/000_7.npz")["target"]
path = "models/supertonic-3/onnx/vocoder.onnx"
session = ort.InferenceSession(
    path, runtime.options(), providers=["CPUExecutionProvider"]
)
reference = session.run(None, {"latent": latent})[0]
model = VectorGraph(path)
x = torch.from_numpy(latent).to("mps").requires_grad_()
output = model({"latent": x})[0]
actual = output.detach().cpu().numpy()
error = float(np.linalg.norm(actual - reference) / np.linalg.norm(reference))
output.square().mean().backward()
r = {
    "relative_rmse": error,
    "max_error": float(np.abs(actual - reference).max()),
    "input_gradient_finite": bool(torch.isfinite(x.grad).all()),
    "input_gradient_norm": float(x.grad.norm().cpu()),
}
Path("results/torch_vocoder_parity.json").write_text(json.dumps(r, indent=2))
print(r)
assert error < 1e-4 and r["input_gradient_finite"] and r["input_gradient_norm"] > 0
