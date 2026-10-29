"""Replace only integer FFN multiplication; quantization, weights and steps stay fixed."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import hashlib
import json
import os
import shutil

import numpy as np
import onnx
from onnx import helper, numpy_helper

from experiments.optimize import split

p = argparse.ArgumentParser()
p.add_argument("--source", required=True)
p.add_argument("--output", required=True)
p.add_argument("--cache-conditioning", action="store_true")
a = p.parse_args()
source = Path(a.source)
dest = Path(a.output)
if dest.exists():
    p.error("Choose a new output; preserve frozen weights")
original = source / "onnx/vector_estimator.onnx"
source_hash = hashlib.sha256(original.read_bytes()).hexdigest()
shutil.copytree(source, dest, copy_function=os.link)
graph_dir = dest / "onnx"
if a.cache_conditioning:
    (graph_dir / "vector_estimator.onnx").unlink()
    (graph_dir / "transformation.json").unlink(missing_ok=True)
    split(original, graph_dir)
m = onnx.shape_inference.infer_shapes(onnx.load(graph_dir / "vector_estimator.onnx"))
values = {v.name: numpy_helper.to_array(v) for v in m.graph.initializer}
producer = {name: node for node in m.graph.node for name in node.output}
converted = []
for node in m.graph.node:
    if (
        node.op_type != "MatMulInteger"
        or "/pwconv" not in node.name
        or len(node.input) != 4
    ):
        continue
    w = values.get(node.input[1])
    zero = values.get(node.input[3])
    if (
        w is None
        or w.ndim != 2
        or w.dtype != np.int8
        or w.shape[0] % 4
        or w.shape[1] % 8
    ):
        continue
    if np.any(w < -64) or np.any(w > 64) or zero is None or np.any(zero):
        continue
    if producer[node.input[0]].op_type != "DynamicQuantizeLinear":
        raise ValueError("Expected unchanged ORT U8 quantization")
    node.op_type = "ExactMatMulInteger"
    node.domain = "com.supertonic.exact"
    converted.append(node.name)
if not converted:
    raise ValueError("No eligible fixed integer FFN weights")
m.opset_import.append(helper.make_opsetid("com.supertonic.exact", 1))
onnx.checker.check_model(m)
temporary = graph_dir / "exact.tmp.onnx"
onnx.save(m, temporary)
temporary.replace(graph_dir / "vector_estimator.onnx")
assert hashlib.sha256(original.read_bytes()).hexdigest() == source_hash
parent = json.loads((source / "manifest.json").read_text())
metadata = {
    **parent,
    "mode": "fixed-model-exact-integer-engine",
    "backend": "native-exact-int8",
    "source_package": str(source),
    "source_vector_sha256": source_hash,
    "source_manifest": parent,
    "vector_sha256": hashlib.sha256(
        (graph_dir / "vector_estimator.onnx").read_bytes()
    ).hexdigest(),
    "runtime_changes": {
        "operator": "AVX2 exact integer GEMM, persistent packing",
        "converted_nodes": converted,
        "cache_conditioning": a.cache_conditioning,
        "weights_changed": False,
        "quantization_changed": False,
        "training_performed": False,
        "steps_changed": False,
        "memory_pattern_enabled": False,
        "cpu_arena_enabled": True,
        "prepacking_enabled": True,
    },
    "target": "Linux x86_64 AVX2; use original ORT package on other architectures",
    "status": "Engine experiment; not selected until actual-server timing and output checks pass",
}
temporary = dest / "manifest.tmp.json"
temporary.write_text(json.dumps(metadata, indent=2) + "\n")
temporary.replace(dest / "manifest.json")
print(dest, len(converted), "integer FFNs; fixed source", source_hash, flush=True)
