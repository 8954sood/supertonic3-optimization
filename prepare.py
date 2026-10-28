"""Create a complete portable model directory, including its original license."""

from runtime import PARTS
import argparse
import json
import shutil
from pathlib import Path
from experiments.pointwise import convert
from experiments.optimize import split

p = argparse.ArgumentParser()
p.add_argument(
    "--mode", choices=["fp32", "fp32-cached", "int8", "denoiser-int8"], default="fp32"
)
p.add_argument("--output", default="models/fast-fp32")
p.add_argument("--source", default="models/supertonic-3")
p.add_argument(
    "--reduce-range",
    action="store_true",
    help="7-bit weights in int8 storage to avoid AVX2 U8S8 saturation",
)
a = p.parse_args()
source = Path(a.source)
dest = Path(a.output)
if dest.exists():
    p.error("output directory already exists; choose a new output")
if not (source / "onnx/vector_estimator.onnx").is_file():
    p.error("run download.py first")
dest.mkdir(parents=True)
onnx_dir = dest / "onnx"
onnx_dir.mkdir()
shutil.copytree(source / "voice_styles", dest / "voice_styles")
for name in ("LICENSE", "config.json", "revision.json"):
    if (source / name).exists():
        shutil.copy2(source / name, dest / name)
for name in ("tts.json", "unicode_indexer.json"):
    shutil.copy2(source / "onnx" / name, onnx_dir / name)
counts = {}
for part in PARTS:
    src = source / "onnx" / f"{part}.onnx"
    target = onnx_dir / f"{part}.onnx"
    if a.mode == "denoiser-int8" and part != "vector_estimator":
        shutil.copy2(src, target)
    else:
        counts[part] = convert(
            src, target, a.mode in ("int8", "denoiser-int8"), a.reduce_range
        )
if a.mode == "fp32-cached":
    split(onnx_dir / "vector_estimator.onnx", onnx_dir)
manifest = {
    "mode": a.mode,
    "source": str(source),
    "pointwise_conversions": counts,
    "minimum_steps": 8,
    "runtime": "CPUExecutionProvider, intra/inter threads = 1, sequential",
    "lossy": a.mode in ("int8", "denoiser-int8"),
    "reduce_range": a.reduce_range,
    "total_bytes": sum(p.stat().st_size for p in dest.rglob("*") if p.is_file()),
}
(dest / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
print("Model ready:", dest)
