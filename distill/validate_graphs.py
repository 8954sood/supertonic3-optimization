"""Multi-input parity of differentiable teacher graphs before further training."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runtime
import json
import torch
import numpy as np
import onnxruntime as ort
import argparse
from distill.torch_graph import VectorGraph

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="distill/trajectories-v1")
parser.add_argument("--output", default="results/torch_graph_parity_suite.json")
args = parser.parse_args()
data_root = Path(args.data)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
vector = VectorGraph("models/supertonic-3/onnx/vector_estimator.onnx")
decoder = VectorGraph("models/supertonic-3/onnx/vocoder.onnx")
session = ort.InferenceSession(
    "models/supertonic-3/onnx/vocoder.onnx",
    runtime.options(),
    providers=["CPUExecutionProvider"],
)
rows = []
with torch.no_grad():
    for split, case in [
        ("train", 0),
        ("train", 1),
        ("validation", 4),
        ("validation", 5),
    ]:
        for step in [0, 7]:
            path = str(data_root / split / f"{case:03d}_{step}.npz")
            feed = dict(np.load(path))
            target = feed.pop("target")
            actual = vector(feed)[0].cpu().numpy()
            relative = float(np.linalg.norm(actual - target) / np.linalg.norm(target))
            rows.append({"graph": "vector", "input": path, "relative_rmse": relative})
            assert relative < 1e-4
        actual = decoder({"latent": target})[0].cpu().numpy()
        reference = session.run(None, {"latent": target})[0]
        relative = float(np.linalg.norm(actual - reference) / np.linalg.norm(reference))
        rows.append({"graph": "decoder", "input": path, "relative_rmse": relative})
        assert relative < 1e-4
Path(args.output).write_text(
    json.dumps(
        {
            "note": "FP32 graph parity on actual original teacher inputs. Not quality evaluation of students.",
            "rows": rows,
        },
        indent=2,
    )
)
print(rows)
