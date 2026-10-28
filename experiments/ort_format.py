"""Portable extended-level ORT serialization; no architecture-specific layout ops."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import options
import argparse
import shutil
import json
import onnxruntime as ort

p = argparse.ArgumentParser()
p.add_argument("--source", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
source = Path(a.source)
dest = Path(a.output)
if dest.exists():
    p.error("choose a new output directory")
shutil.copytree(source, dest)
for path in (dest / "onnx").glob("*.onnx"):
    opt = options(low_memory=True)
    opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    out = path.with_suffix(".ort")
    opt.optimized_model_filepath = str(out)
    opt.add_session_config_entry("session.save_model_format", "ORT")
    session = ort.InferenceSession(str(path), opt, providers=["CPUExecutionProvider"])
    del session
    path.unlink()
    print(out, flush=True)
(dest / "serialization.json").write_text(
    json.dumps(
        {
            "format": "ORT",
            "version": ort.__version__,
            "optimization": "extended; excludes hardware-specific all-level layout transforms",
        },
        indent=2,
    )
)
