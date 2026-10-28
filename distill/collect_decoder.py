"""Collect decoder MLP teacher pairs from original teacher final latents, train/validation only."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import options
import json, argparse
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort

p = argparse.ArgumentParser()
p.add_argument("--output", default="distill/decoder-data-v1")
p.add_argument("--samples", type=int, default=128)
a = p.parse_args()
root = Path(a.output)
if root.exists():
    p.error("Choose new dataset output")
root.mkdir(parents=True)
m = onnx.load("models/supertonic-3/onnx/vocoder.onnx")
byname = {n.name: n for n in m.graph.node}
pairs = []
for node in m.graph.node:
    if node.op_type == "Conv" and node.name.endswith("/pwconv1/Conv"):
        second = byname[node.name.replace("/pwconv1/Conv", "/pwconv2/Conv")]
        pair = {
            "first_node": node.name,
            "second_node": second.name,
            "input": node.input[0],
            "output": second.output[0],
            "w1": node.input[1],
            "b1": node.input[2],
            "w2": second.input[1],
            "b2": second.input[2],
        }
        pairs.append(pair)
        m.graph.output.extend(
            [
                helper.make_tensor_value_info(
                    pair["input"], TensorProto.FLOAT, [None, 512, None]
                ),
                helper.make_tensor_value_info(
                    pair["output"], TensorProto.FLOAT, [None, 512, None]
                ),
            ]
        )
onnx.checker.check_model(m)
onnx.save(m, root / "teacher_probe.onnx")
del m
s = ort.InferenceSession(
    str(root / "teacher_probe.onnx"), options(), providers=["CPUExecutionProvider"]
)
metadata = {
    "teacher": "original FP32 decoder, original teacher final CFG4 eight-step latents",
    "pairs": pairs,
    "splits": {},
    "samples_per_utterance": a.samples,
    "source": "distill/trajectories-v2",
}
rng = np.random.default_rng(20261030)
for split, count in [("train", 200), ("validation", 8)]:
    folder = root / split
    folder.mkdir()
    sinks = []
    capacity = count * a.samples
    offset = 0
    for i in range(len(pairs)):
        sinks.append(
            tuple(
                np.lib.format.open_memmap(
                    folder / f"{axis}_{i:02d}.npy",
                    mode="w+",
                    dtype=np.float16,
                    shape=(capacity, 512),
                )
                for axis in ["x", "y"]
            )
        )
    for case in range(count):
        latent = np.load(f"distill/trajectories-v2/{split}/{case:03d}_7.npz")["target"]
        out = s.run(None, {"latent": latent})
        n = out[1].shape[-1]
        index = rng.choice(n, min(a.samples, n), replace=False)
        for i, (x, y) in enumerate(sinks):
            x[offset : offset + len(index)] = out[1 + 2 * i][0, :, index].astype(
                np.float16
            )
            y[offset : offset + len(index)] = out[2 + 2 * i][0, :, index].astype(
                np.float16
            )
        offset += len(index)
        if case % 10 == 0:
            print(split, case, "rows", offset, flush=True)
    for x, y in sinks:
        x.flush()
        y.flush()
    metadata["splits"][split] = {
        "rows": offset,
        "capacity": capacity,
        "utterances": count,
    }
    (root / "progress.json").write_text(json.dumps(metadata, indent=2))
    del sinks
(root / "manifest.json").write_text(json.dumps(metadata, indent=2))
print("decoder dataset ready", flush=True)
