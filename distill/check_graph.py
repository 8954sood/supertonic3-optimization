"""Compare differentiable ONNX implementation with frozen CPU ORT for actual inputs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runtime  # noqa: F401
import argparse
import json
import time
import numpy as np
import torch
from distill.torch_graph import VectorGraph

p = argparse.ArgumentParser()
p.add_argument("--input", default="distill/trajectories-v1/train/000_0.npz")
p.add_argument("--device", default="mps")
p.add_argument("--output", default="results/torch_graph_parity.json")
a = p.parse_args()
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
data = dict(np.load(a.input))
target = data.pop("target")
model = VectorGraph("models/supertonic-3/onnx/vector_estimator.onnx", device=a.device)
start = time.perf_counter()
with torch.no_grad():
    actual = model(data)[0].cpu().numpy()
row = {
    "input": a.input,
    "device": a.device,
    "rmse": float(np.sqrt(np.mean((actual - target) ** 2))),
    "relative_rmse": float(np.linalg.norm(actual - target) / np.linalg.norm(target)),
    "max_error": float(np.max(np.abs(actual - target))),
    "seconds": time.perf_counter() - start,
}
Path(a.output).write_text(json.dumps(row, indent=2))
print(row, flush=True)
if row["relative_rmse"] > 1e-4:
    raise RuntimeError("Graph parity failed; do not train")
