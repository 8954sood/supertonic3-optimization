"""Experimental CFG schedule: keep all eight denoising calls, change guidance later.
This is a quality/performance tradeoff, not an exact-equivalent graph rewrite.
"""

from pathlib import Path
import argparse
import json
import shutil
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

p = argparse.ArgumentParser()
p.add_argument("--guided-steps", type=int, default=4)
p.add_argument("--output", required=True)
a = p.parse_args()
dest = Path(a.output)
if not 1 <= a.guided_steps <= 7:
    p.error("guided-steps must be from 1 to 7")
if dest.exists():
    p.error("choose a new output directory")
shutil.copytree("models/fast-full-int8", dest)
m = onnx.load(dest / "onnx/vector_estimator.onnx")
nodes = []


def initializer(name, array):
    m.graph.initializer.append(numpy_helper.from_array(np.asarray(array), name))
    return name


threshold = initializer("schedule/threshold", np.array([a.guided_steps], np.float32))
axes = initializer("schedule/axes", np.array([0], np.int64))
r2 = initializer("schedule/repeat2", np.array([2, 1, 1], np.int64))
r1 = initializer("schedule/repeat1", np.array([1, 1, 1], np.int64))
b2 = initializer("schedule/batch2", np.array([2], np.int64))
b1 = initializer("schedule/batch1", np.array([1], np.int64))
index = initializer("schedule/index", np.array(0, np.int64))
nodes.extend(
    [
        helper.make_node(
            "Less", ["current_step", threshold], ["schedule/use_cfg_vector"]
        ),
        helper.make_node(
            "Squeeze", ["schedule/use_cfg_vector", axes], ["schedule/use_cfg"]
        ),
        helper.make_node("Where", ["schedule/use_cfg", r2, r1], ["schedule/repeats"]),
        helper.make_node(
            "Where", ["schedule/use_cfg", b2, b1], ["schedule/batch_multiplier"]
        ),
        helper.make_node("Shape", ["noisy_latent"], ["schedule/noise_shape"]),
        helper.make_node(
            "Gather", ["schedule/noise_shape", index], ["schedule/batch"], axis=0
        ),
        helper.make_node(
            "Mul", ["schedule/batch", "schedule/batch_multiplier"], ["schedule/end"]
        ),
    ]
)
for node in m.graph.node:
    if node.name in (
        "/vector_estimator/Tile_1",
        "/vector_estimator/Tile_2",
        "/vector_estimator/Tile_3",
        "/vector_estimator/Tile_4",
    ):
        node.input[1] = "schedule/repeats"
    if node.name in (
        "/vector_estimator/Concat_5",
        "/vector_estimator/Concat_6",
        "/vector_estimator/Concat_7",
    ):
        out = node.output[0]
        node.output[0] = out + "/both"
        nodes.append(node)
        nodes.append(
            helper.make_node(
                "Slice",
                [out + "/both", axes, "schedule/end", axes],
                [out],
                name=node.name + "/schedule",
            )
        )
        continue
    if node.name == "/Sub":
        output = helper.make_tensor_value_info(
            "guided", TensorProto.FLOAT, ["batch_size", 144, "latent_length"]
        )
        guided = helper.make_graph(
            [helper.make_node("Sub", list(node.input), ["guided"])],
            "guided_branch",
            [],
            [output],
        )
        raw = helper.make_graph(
            [
                helper.make_node(
                    "Identity",
                    ["/vector_estimator/vector_field/proj_out/Mul_output_0"],
                    ["plain"],
                )
            ],
            "plain_branch",
            [],
            [
                helper.make_tensor_value_info(
                    "plain", TensorProto.FLOAT, ["batch_size", 144, "latent_length"]
                )
            ],
        )
        nodes.append(
            helper.make_node(
                "If",
                ["schedule/use_cfg"],
                list(node.output),
                name="scheduled_guidance",
                then_branch=guided,
                else_branch=raw,
            )
        )
        continue
    nodes.append(node)
del m.graph.node[:]
m.graph.node.extend(nodes)
del m.graph.value_info[:]
onnx.checker.check_model(m)
onnx.save(m, dest / "onnx/vector_estimator.onnx")
manifest = json.loads((dest / "manifest.json").read_text())
manifest.update(
    guidance_mode="scheduled quality tradeoff",
    guided_steps=a.guided_steps,
    guidance_scale_early=4,
    guidance_scale_late=1,
    denoising_steps=8,
    status="experimental; requires intelligibility, timbre and speed evaluation",
)
(dest / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(dest)
