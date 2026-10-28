"""Use packed int8 storage with FP32 single-thread BLAS arithmetic per layer."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import PARTS
import argparse
import json
import shutil
import numpy as np
import onnx
from onnx import helper, numpy_helper

p = argparse.ArgumentParser()
p.add_argument("--output", required=True)
a = p.parse_args()
dest = Path(a.output)
if dest.exists():
    p.error("choose a new output directory")
shutil.copytree("models/fast-full-int8", dest)
counts = {}
for part in PARTS:
    m = onnx.load(Path("models/pointwise-fp32") / f"{part}.onnx")
    init = {x.name: x for x in m.graph.initializer}
    count = 0
    for node in m.graph.node:
        if node.op_type != "MatMul" or node.input[1] not in init:
            continue
        w = numpy_helper.to_array(init[node.input[1]])
        if w.ndim != 2:
            continue
        scales = np.maximum(
            np.max(np.abs(w), axis=0) / 127, np.finfo(np.float32).tiny
        ).astype(np.float32)
        quant = np.clip(np.rint(w / scales), -127, 127).astype(np.int8)
        name = node.name + "/packed"
        m.graph.initializer.extend(
            [
                numpy_helper.from_array(quant, name + "/q"),
                numpy_helper.from_array(scales, name + "/s"),
            ]
        )
        node.input[1] = name + "/q"
        node.input.append(name + "/s")
        node.op_type = "PackedW8MatMul"
        node.domain = "ai.onnx.contrib"
        count += 1
    used = {x for n in m.graph.node for x in n.input}
    keep = [x for x in m.graph.initializer if x.name in used]
    del m.graph.initializer[:]
    m.graph.initializer.extend(keep)
    m.opset_import.append(helper.make_opsetid("ai.onnx.contrib", 1))
    onnx.checker.check_model(m)
    onnx.save(m, dest / "onnx" / f"{part}.onnx")
    counts[part] = count
(dest / "manifest.json").write_text(
    json.dumps(
        {
            "mode": "experimental-packed-int8-blas",
            "backend": "numpy",
            "lossy": True,
            "arithmetic": "FP32 BLAS with per-layer int8 decode",
            "nodes": counts,
        },
        indent=2,
    )
)
print(counts)
