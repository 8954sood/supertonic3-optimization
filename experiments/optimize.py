"""Split invariant conditioning out of the diffusion loop; optionally int8 weights.
No retraining. All transformations are portable standard ONNX operators.
"""

import argparse
import copy
import json
from pathlib import Path
import onnx
from onnx import helper, shape_inference
from onnxruntime.quantization import quantize_dynamic, QuantType


def split(source, dest):
    m = onnx.load(source)
    m = shape_inference.infer_shapes(m)
    graph = m.graph
    # These change on each denoising iteration. Everything else is per utterance.
    changing = {"noisy_latent", "current_step"}
    dynamic_nodes = []
    static_nodes = []
    for n in graph.node:
        if any(x in changing for x in n.input):
            changing.update(n.output)
            dynamic_nodes.append(n)
        else:
            static_nodes.append(n)
    static_outputs = {x for n in static_nodes for x in n.output}
    needed = {x for n in dynamic_nodes for x in n.input}
    boundary = sorted(static_outputs & needed)
    info = {
        x.name: x
        for x in list(graph.input) + list(graph.value_info) + list(graph.output)
    }
    missing = [x for x in boundary if x not in info]
    if missing:
        raise RuntimeError(f"Missing inferred shapes: {missing}")
    initializers = {x.name: x for x in graph.initializer}

    def build(nodes, inputs, outputs, name):
        # Reverse prune to remove unused static nodes and tensors.
        keep = set(outputs)
        selected = []
        for node in reversed(nodes):
            if any(x in keep for x in node.output):
                selected.append(node)
                keep.update(x for x in node.input if x)
        init = [v for k, v in initializers.items() if k in keep]
        ins = [copy.deepcopy(info[x]) for x in inputs if x in keep]
        outs = [copy.deepcopy(info[x]) for x in outputs]
        g = helper.make_graph(list(reversed(selected)), name, ins, outs, init)
        out = helper.make_model(
            g, opset_imports=m.opset_import, ir_version=m.ir_version
        )
        onnx.checker.check_model(out)
        return out

    pre = build(
        static_nodes, [x.name for x in graph.input], boundary, "conditioning_once"
    )
    core = build(
        dynamic_nodes,
        [x.name for x in graph.input] + boundary,
        [x.name for x in graph.output],
        "denoise_cached",
    )
    dest.mkdir(parents=True, exist_ok=True)
    onnx.save(pre, dest / "condition.onnx")
    onnx.save(core, dest / "vector_estimator.onnx")
    summary = {
        "static_nodes": len(pre.graph.node),
        "loop_nodes_before": len(graph.node),
        "loop_nodes_after": len(core.graph.node),
        "cached_tensors": len(boundary),
        "boundary": boundary,
    }
    (dest / "transformation.json").write_text(json.dumps(summary, indent=2))
    print(summary, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["split", "int8", "split-int8"])
    p.add_argument("--output", required=True)
    a = p.parse_args()
    dest = Path(a.output)
    dest.mkdir(parents=True, exist_ok=True)
    root = Path("models/supertonic-3/onnx")
    if a.mode in ("split", "split-int8"):
        split(root / "vector_estimator.onnx", dest)
    if a.mode in ("int8", "split-int8"):
        for part in (
            "duration_predictor",
            "text_encoder",
            "vector_estimator",
            "vocoder",
            "condition",
        ):
            source = (
                dest / f"{part}.onnx"
                if (dest / f"{part}.onnx").exists()
                else root / f"{part}.onnx"
            )
            if not source.exists():
                continue
            target = dest / f"{part}.tmp.onnx"
            quantize_dynamic(
                str(source),
                str(target),
                op_types_to_quantize=["MatMul"],
                weight_type=QuantType.QInt8,
                per_channel=True,
                extra_options={"MatMulConstBOnly": True},
            )
            target.replace(dest / f"{part}.onnx")
            print("quantized", part, flush=True)


if __name__ == "__main__":
    main()
