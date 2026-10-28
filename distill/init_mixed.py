"""Choose conservative FFN widths using training-time validation, keeping sensitive layers original."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runtime  # noqa:F401
import json
import numpy as np
import torch
import onnx
from onnx import numpy_helper

root = Path("distill/mixed-v1")
root.mkdir(parents=True, exist_ok=True)
if (root / "initial.pt").exists():
    raise RuntimeError("Preserve existing mixed checkpoint")
pairs = json.loads(Path("distill/data/manifest.json").read_text())["pairs"]
original = {
    v.name: numpy_helper.to_array(v)
    for v in onnx.load(
        "models/supertonic-3/onnx/vector_estimator.onnx"
    ).graph.initializer
}
weights = {}
selection = []
for i, pair in enumerate(pairs):
    chosen = None
    for width, limit in [(512, 0.01), (1024, 0.01), (1536, 0.15)]:
        saved = torch.load(
            f"distill/student{width}/layer_{i:02d}.pt",
            map_location="cpu",
            weights_only=True,
        )
        if saved["metrics"]["best_float_relative_rmse"] <= limit:
            chosen = saved
            selection.append(
                {
                    "layer": i,
                    "width": width,
                    "relative_validation_rmse": saved["metrics"][
                        "best_float_relative_rmse"
                    ],
                }
            )
            break
    if chosen:
        state = chosen["float_state"]
        for key, t in [
            (pair["w1"], state["fc1.weight"][:, :, None]),
            (pair["b1"], state["fc1.bias"]),
            (pair["w2"], state["fc2.weight"][:, :, None]),
            (pair["b2"], state["fc2.bias"]),
        ]:
            weights[key] = t
    else:
        selection.append(
            {
                "layer": i,
                "width": 2048,
                "relative_validation_rmse": 0,
                "action": "retain original",
            }
        )
        for key in ["w1", "b1", "w2", "b2"]:
            weights[pair[key]] = torch.from_numpy(
                np.array(original[pair[key]], copy=True)
            )
torch.save({"weights": weights, "selection": selection}, root / "initial.pt")
(root / "selection.json").write_text(json.dumps(selection, indent=2))
print(selection)
