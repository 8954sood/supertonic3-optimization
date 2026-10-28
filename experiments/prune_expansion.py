"""Lossy, untrained pruning trial. Never promote without intelligibility checks.
Remove only internal GELU expansion channels, preserving 512-channel residuals.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import shutil
import numpy as np
import onnx
from onnx import numpy_helper
from experiments.pointwise import convert

p = argparse.ArgumentParser()
p.add_argument("--keep", type=float, required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
if not 0 < a.keep < 1:
    p.error("keep must be between zero and one")
dest = Path(a.output)
if dest.exists():
    p.error("choose a new output directory")
shutil.copytree("models/fast-full-int8", dest)
m = onnx.load("models/supertonic-3/onnx/vector_estimator.onnx")
init = {x.name: x for x in m.graph.initializer}
byname = {n.name: n for n in m.graph.node}
rows = []
for n in m.graph.node:
    if n.op_type != "Conv" or not n.name.endswith("/pwconv1/Conv"):
        continue
    second = byname[n.name.replace("/pwconv1/Conv", "/pwconv2/Conv")]
    w1 = numpy_helper.to_array(init[n.input[1]])
    w2 = numpy_helper.to_array(init[second.input[1]])
    if w1.shape[0] != w2.shape[1]:
        raise RuntimeError("expansion mismatch")
    score = np.linalg.norm(w1.reshape(w1.shape[0], -1), axis=1) * np.linalg.norm(
        w2[:, :, 0], axis=0
    )
    count = max(32, round(w1.shape[0] * a.keep / 32) * 32)
    keep = np.sort(np.argsort(score)[-count:])
    for key, values in [
        (n.input[1], w1[keep]),
        (n.input[2], numpy_helper.to_array(init[n.input[2]])[keep]),
        (second.input[1], w2[:, keep, :]),
    ]:
        init[key].CopyFrom(numpy_helper.from_array(values.copy(), key))
    rows.append(
        {
            "layer": n.name,
            "before": len(score),
            "after": count,
            "retained_score_fraction": float(score[keep].sum() / score.sum()),
        }
    )
del m.graph.value_info[:]
onnx.checker.check_model(m)
temp = dest / "onnx/vector_pruned_fp32.onnx"
onnx.save(m, temp)
convert(temp, dest / "onnx/vector_estimator.onnx", True, reduce_range=True)
temp.unlink()
(dest / "manifest.json").write_text(
    json.dumps(
        {
            "mode": "experimental-untrained-channel-pruning",
            "keep": a.keep,
            "steps": 8,
            "layers": rows,
            "status": "requires quality rejection/acceptance evaluation",
        },
        indent=2,
    )
)
print("Pruned expansion pairs:", len(rows), "keep:", a.keep, flush=True)
