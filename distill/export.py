"""Export distilled FFNs into the unchanged eight-step CFG4 inference graph."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import shutil
import numpy as np
import onnx
from onnx import numpy_helper
import torch
from experiments.pointwise import convert

p = argparse.ArgumentParser()
p.add_argument("--checkpoints", required=True)
p.add_argument("--output", required=True)
p.add_argument("--state", choices=["float", "qat"], default="float")
p.add_argument("--int8", action="store_true")
p.add_argument(
    "--max-rmse",
    type=float,
    default=float("inf"),
    help="Keep original FFNs above this validation relative RMSE",
)
a = p.parse_args()
dest = Path(a.output)
if dest.exists():
    p.error("choose a new output directory")
root = Path(a.checkpoints)
report = json.loads((root / "report.json").read_text())
pairs = json.loads(Path("distill/data/manifest.json").read_text())["pairs"]
model = onnx.load("models/supertonic-3/onnx/vector_estimator.onnx")
initializers = {v.name: v for v in model.graph.initializer}
before = sum(np.prod(v.dims) for v in model.graph.initializer)
selected = []
for i, pair in enumerate(pairs):
    saved = torch.load(
        root / f"layer_{i:02d}.pt", map_location="cpu", weights_only=True
    )
    if not saved.get("complete"):
        raise RuntimeError(f"incomplete layer {i}")
    state = saved[a.state + "_state"]
    if state is None:
        raise RuntimeError(f"missing {a.state} weights at layer {i}")
    error = saved["metrics"]["best_" + a.state + "_relative_rmse"]
    if error > a.max_rmse:
        selected.append(
            {
                "layer": i,
                "width": 2048,
                "action": "retain original",
                "validation_rmse": error,
            }
        )
        continue
    selected.append(
        {
            "layer": i,
            "width": report["config"]["width"],
            "action": "distilled",
            "validation_rmse": error,
        }
    )
    for key, tensor in [
        (pair["w1"], state["fc1.weight"][:, :, None]),
        (pair["b1"], state["fc1.bias"]),
        (pair["w2"], state["fc2.weight"][:, :, None]),
        (pair["b2"], state["fc2.bias"]),
    ]:
        initializers[key].CopyFrom(numpy_helper.from_array(tensor.numpy().copy(), key))
after = sum(np.prod(v.dims) for v in model.graph.initializer)
del model.graph.value_info[:]
onnx.checker.check_model(model)
shutil.copytree("models/fast-full-int8", dest)
temp = dest / "onnx/student_fp32.onnx"
onnx.save(model, temp)
convert(temp, dest / "onnx/vector_estimator.onnx", a.int8, reduce_range=True)
temp.unlink()
(dest / "manifest.json").write_text(
    json.dumps(
        {
            "mode": "experimental-layerwise-distillation",
            "status": "requires end-to-end quality validation",
            "checkpoint": str(root),
            "state": a.state,
            "vector_int8": a.int8,
            "other_components": "unchanged fast-full-int8",
            "hidden_width": report["config"]["width"],
            "layers": len(pairs),
            "selected_layers": selected,
            "max_validation_rmse": a.max_rmse if np.isfinite(a.max_rmse) else None,
            "vector_parameters_before": int(before),
            "vector_parameters_after": int(after),
            "minimum_steps": 8,
            "guidance_scale": 4,
            "training": report,
        },
        indent=2,
    )
)
print(dest, "vector parameters", before, "->", after)
