"""Export jointly trained student weights, optionally preserving FP32 frontend/decoder."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import shutil
import torch
import onnx
from onnx import numpy_helper
from experiments.pointwise import convert

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--output", required=True)
p.add_argument("--base", default="models/fast-denoiser-int8")
p.add_argument("--int8", action="store_true")
p.add_argument("--graph", default="models/supertonic-3/onnx/vector_estimator.onnx")
p.add_argument("--cfg-distilled", action="store_true")
a = p.parse_args()
dest = Path(a.output)
if dest.exists():
    p.error("Choose new model output")
saved = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
m = onnx.load(a.graph)
initializers = {v.name: v for v in m.graph.initializer}
for name, value in saved["weights"].items():
    initializers[name].CopyFrom(numpy_helper.from_array(value.numpy().copy(), name))
del m.graph.value_info[:]
onnx.checker.check_model(m)
shutil.copytree(a.base, dest)
condition = dest / "onnx/condition.onnx"
if condition.exists():
    condition.unlink()  # This export retains the complete, unsplit vector graph.
temp = dest / "onnx/joint_raw.onnx"
onnx.save(m, temp)
convert(temp, dest / "onnx/vector_estimator.onnx", a.int8, reduce_range=True)
temp.unlink()
manifest = {
    "mode": "joint-FFN-distillation",
    "status": "experimental; acoustic/ASR/listening validation required",
    "checkpoint": a.checkpoint,
    "step": saved["step"],
    "validation_relative_update_mse": saved["validation_relative_update_mse"],
    "base": a.base,
    "vector_int8": a.int8,
    "minimum_steps": 8,
    "guidance_scale": 4,
    "training": saved["config"],
}
if a.cfg_distilled:
    manifest.pop("guidance_scale", None)
    manifest.update(
        mode="experimental-CFG-distillation",
        teacher_guidance_scale=4,
        student_internal_guidance_scale=1,
        guidance_mode="learned one-branch approximation; not exact CFG4",
        supported_steps=[8],
    )
(dest / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(dest, flush=True)
