"""Move elementwise exact GELU across inverse transposes; keep all learned values."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import hashlib
import json
import os
import shutil

import onnx


def transform(model):
    consumers = {}
    for node in model.graph.node:
        for value in node.input:
            consumers.setdefault(value, []).append(node)
    removed, changed, groups = set(), set(), []
    nodes = {node.name: node for node in model.graph.node}
    for node in model.graph.node:
        if node.op_type != "Transpose" or not node.name.endswith("/pwconv1/Conv/linear/to_ncl"):
            continue
        prefix = node.name.removesuffix("/pwconv1/Conv/linear/to_ncl")
        ops = [nodes[prefix + "/act/" + name] for name in ["Div", "Erf", "Add", "Mul", "Mul_1"]]
        after = nodes[prefix + "/pwconv2/Conv/linear/to_nlc"]
        for transpose in [node, after]:
            perm = next(onnx.helper.get_attribute_value(a) for a in transpose.attribute if a.name == "perm")
            if list(perm) != [0, 2, 1]:
                raise ValueError("Expected inverse NCL/NLC transposes")
        # No branch may observe a moved tensor in its old layout.
        allowed = {n.name for n in ops} | {after.name}
        for name in [node.output[0]] + [n.output[0] for n in ops]:
            if any(n.name not in allowed for n in consumers.get(name, [])):
                raise ValueError("Activation has an external layout consumer")
        if after.input[0] != ops[-1].output[0]:
            raise ValueError("Expected complete exact GELU subgraph")
        for op in ops:
            for i, name in enumerate(op.input):
                if name == node.output[0]:
                    op.input[i] = node.input[0]
            changed.update(op.output)
        ops[-1].output[0] = after.output[0]
        removed.update([node.name, after.name])
        changed.update([node.output[0], after.output[0]])
        groups.append(prefix)
    if len(groups) != 28:
        raise ValueError(f"Expected 28 exact GELU groups, got {len(groups)}")
    keep = [n for n in model.graph.node if n.name not in removed]
    del model.graph.node[:]
    model.graph.node.extend(keep)
    infos = [v for v in model.graph.value_info if v.name not in changed]
    del model.graph.value_info[:]
    model.graph.value_info.extend(infos)
    onnx.checker.check_model(model)
    return dict(groups=len(groups), removed_transposes=len(removed), learned_values_changed=False,
                activation='Unchanged Div/Erf/Add/Mul/Mul exact GELU; reordered layout only')


def explicit_bias_gelu(model):
    nodes={node.name:node for node in model.graph.node}
    removed,replaced=set(),{}
    for node in model.graph.node:
        if not node.name.endswith('/act/Mul_1'):
            continue
        prefix=node.name.removesuffix('/act/Mul_1')
        bias=nodes[prefix+'/pwconv1/Conv/linear/Add']
        for suffix in ['/pwconv1/Conv/linear/Add','/act/Div','/act/Erf','/act/Add','/act/Mul']:
            removed.add(prefix+suffix)
        replaced[node.name]=onnx.helper.make_node('BiasGelu',list(bias.input),list(node.output),
            name=prefix+'/exact_bias_gelu',domain='com.microsoft')
    if len(replaced)!=28:
        raise ValueError('Expected 28 explicit BiasGelu nodes')
    keep=[replaced.get(node.name,node) for node in model.graph.node if node.name not in removed]
    del model.graph.node[:];model.graph.node.extend(keep)
    if not any(op.domain=='com.microsoft' for op in model.opset_import):
        model.opset_import.append(onnx.helper.make_opsetid('com.microsoft',1))
    onnx.checker.check_model(model)
    return len(replaced)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bias-gelu", action="store_true", help="Also force explicit BiasGelu instead of fused matmul bias")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output")
    source = args.source / "onnx/vector_estimator.onnx"
    model = onnx.load(source)
    before = {v.name: v.SerializeToString() for v in model.graph.initializer}
    receipt = transform(model)
    if args.bias_gelu:
        receipt['explicit_bias_gelu_nodes']=explicit_bias_gelu(model)
    if before != {v.name: v.SerializeToString() for v in model.graph.initializer}:
        raise RuntimeError("Initializer changed")
    receipt['source_vector_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    shutil.copytree(args.source, args.output, copy_function=os.link)
    temporary = args.output / "onnx/vector.layout.tmp.onnx"
    onnx.save(model, temporary)
    temporary.replace(args.output / "onnx/vector_estimator.onnx")
    metadata = json.loads((args.source / "manifest.json").read_text())
    metadata['transpose_free_gelu'] = receipt
    metadata['engine_experiment_status'] = 'Unselected experiment; matched timing and waveform checks required'
    metadata['vector_sha256'] = hashlib.sha256((args.output / "onnx/vector_estimator.onnx").read_bytes()).hexdigest()
    temporary = args.output / "manifest.layout.tmp.json"
    temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(args.output / "manifest.json")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
