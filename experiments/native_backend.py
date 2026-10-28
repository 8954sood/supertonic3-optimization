"""Use native zero-copy ORT input pointers rather than Python custom-op callbacks."""

from pathlib import Path
import argparse
import json
import shutil
import onnx
from onnx import helper

p = argparse.ArgumentParser()
p.add_argument("--output", default="models/trial-native-blas")
a = p.parse_args()
dest = Path(a.output)
if dest.exists():
    p.error("choose a new output directory")
shutil.copytree("models/trial-blas", dest)
for path in (dest / "onnx").glob("*.onnx"):
    m = onnx.load(path)
    for node in m.graph.node:
        if node.domain == "ai.onnx.contrib" and node.op_type == "PackedW8MatMul":
            node.domain = "com.supertonic.blas"
    m.opset_import.append(helper.make_opsetid("com.supertonic.blas", 1))
    onnx.checker.check_model(m)
    onnx.save(m, path)
manifest = json.loads((dest / "manifest.json").read_text())
manifest["backend"] = "native-blas"
manifest["mode"] = "experimental-native-packed-int8-blas"
manifest["binary_target"] = "x86_64 Linux glibc >=2.28 AVX2/Zen1"
(dest / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(dest)
