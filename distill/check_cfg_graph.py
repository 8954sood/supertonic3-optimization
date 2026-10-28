"""Parity for the single-branch graph before it learns the CFG4 teacher."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runtime
import torch, numpy as np, json
import onnxruntime as ort
from distill.torch_graph import VectorGraph

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
path = "models/cfg-student-fp32-source/onnx/vector_estimator.onnx"
feed = dict(np.load("distill/trajectories-v1/train/000_0.npz"))
feed.pop("target")
s = ort.InferenceSession(path, runtime.options(), providers=["CPUExecutionProvider"])
expected = s.run(None, feed)[0]
m = VectorGraph(path)
with torch.no_grad():
    actual = m(feed)[0].cpu().numpy()
r = {
    "relative_rmse": float(
        np.linalg.norm(actual - expected) / np.linalg.norm(expected)
    ),
    "max_error": float(abs(actual - expected).max()),
    "note": "Untrained one-branch graph parity only; does not test teacher CFG4 matching.",
}
Path("results/torch_cfg_graph_parity_v1.json").write_text(json.dumps(r, indent=2))
print(r)
assert r["relative_rmse"] < 1e-4
