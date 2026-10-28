"""Preserve selected student weights and split only per-utterance invariant work."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shutil, json
from experiments.optimize import split

source = Path("models/mixed-v1-selected")
dest = Path("models/mixed-v1-cached")
if dest.exists():
    raise RuntimeError("Choose new model output")
shutil.copytree(source, dest)
split(dest / "onnx/vector_estimator.onnx", dest / "onnx")
m = json.loads((dest / "manifest.json").read_text())
m.update(
    source_export=str(source),
    runtime_change="Cache static conditioning before the same eight CFG4 denoising steps",
    status="experimental; same weights, numerical equivalence and performance checks pending",
)
(dest / "manifest.json").write_text(json.dumps(m, indent=2))
