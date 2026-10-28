"""Expose 1x1 convolution as MatMul for portable dynamic int8 CPU kernels."""

import argparse
import json
from pathlib import Path
import onnx
from onnx import helper, numpy_helper
from onnxruntime.quantization import quantize_dynamic, QuantType


def convert(source, target, quantize, reduce_range=False):
    m = onnx.load(source)
    init = {x.name: x for x in m.graph.initializer}
    nodes = []
    count = 0
    for n in m.graph.node:
        attrs = {a.name: helper.get_attribute_value(a) for a in n.attribute}
        w = init.get(n.input[1]) if len(n.input) > 1 else None
        if (
            n.op_type == "Conv"
            and w is not None
            and len(w.dims) == 3
            and w.dims[2] == 1
            and attrs.get("group", 1) == 1
            and all(v == 0 for v in attrs.get("pads", [0, 0]))
            and attrs.get("strides", [1]) == [1]
        ):
            count += 1
            prefix = n.name + "/linear"
            weight = numpy_helper.to_array(w)[:, :, 0].T.copy()
            m.graph.initializer.append(
                numpy_helper.from_array(weight, prefix + "/weight")
            )
            nodes.append(
                helper.make_node(
                    "Transpose",
                    [n.input[0]],
                    [prefix + "/in"],
                    perm=[0, 2, 1],
                    name=prefix + "/to_nlc",
                )
            )
            nodes.append(
                helper.make_node(
                    "MatMul",
                    [prefix + "/in", prefix + "/weight"],
                    [prefix + "/mm"],
                    name=prefix + "/MatMul",
                )
            )
            output = prefix + "/mm"
            if len(n.input) > 2:
                nodes.append(
                    helper.make_node(
                        "Add",
                        [output, n.input[2]],
                        [prefix + "/bias"],
                        name=prefix + "/Add",
                    )
                )
                output = prefix + "/bias"
            nodes.append(
                helper.make_node(
                    "Transpose",
                    [output],
                    list(n.output),
                    perm=[0, 2, 1],
                    name=prefix + "/to_ncl",
                )
            )
        else:
            nodes.append(n)
    del m.graph.node[:]
    m.graph.node.extend(nodes)
    used = {x for n in nodes for x in n.input}
    keep = [x for x in m.graph.initializer if x.name in used]
    del m.graph.initializer[:]
    m.graph.initializer.extend(keep)
    onnx.checker.check_model(m)
    target.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(m, target)
    if quantize:
        tmp = target.with_suffix(".tmp.onnx")
        quantize_dynamic(
            str(target),
            str(tmp),
            op_types_to_quantize=["MatMul"],
            weight_type=QuantType.QInt8,
            per_channel=True,
            reduce_range=reduce_range,
            extra_options={"MatMulConstBOnly": True},
        )
        tmp.replace(target)
    print(target, "pointwise converted:", count, flush=True)
    return count


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--source", default="models/supertonic-3/onnx")
    p.add_argument("--int8", action="store_true")
    p.add_argument("--reduce-range", action="store_true")
    a = p.parse_args()
    counts = {}
    for part in (
        "duration_predictor",
        "text_encoder",
        "vector_estimator",
        "vocoder",
        "condition",
    ):
        src = Path(a.source) / f"{part}.onnx"
        if src.exists():
            counts[part] = convert(
                src, Path(a.output) / f"{part}.onnx", a.int8, a.reduce_range
            )
    (Path(a.output) / "pointwise.json").write_text(json.dumps(counts, indent=2))
