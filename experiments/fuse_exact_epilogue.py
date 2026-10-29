"""Fuse exact integer output cast/scale/bias without changing learned tensors."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import numpy as np
import onnx
from onnx import helper, numpy_helper


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--preserve-bias-gelu', action='store_true',
        help='Keep FFN1 bias adds available for ORT BiasGelu fusion')
    a = p.parse_args()
    if a.output.exists():
        p.error('Choose a new output')
    source = a.source/'onnx/vector_estimator.onnx'
    m = onnx.load(source)
    values = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    consumers = {}
    for node in m.graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)

    def sole(name, op):
        nodes = consumers.get(name, [])
        return nodes[0] if len(nodes) == 1 and nodes[0].op_type == op else None

    removed, replacements, converted = set(), {}, []
    retained_bias = 0
    for node in m.graph.node:
        if node.domain != 'com.supertonic.exact' or node.op_type != 'ExactMatMulInteger':
            continue
        cast = sole(node.output[0], 'Cast')
        mul = sole(cast.output[0], 'Mul') if cast else None
        add = sole(mul.output[0], 'Add') if mul else None
        if cast is None or mul is None or add is None:
            continue
        if helper.get_attribute_value(next(at for at in cast.attribute if at.name == 'to')) != onnx.TensorProto.FLOAT:
            raise ValueError('Expected int32 to float32 cast')
        scale = next(v for v in mul.input if v != cast.output[0])
        bias = next(v for v in add.input if v != mul.output[0])
        weight, offset = values[node.input[1]], values.get(bias)
        if offset is None or offset.dtype != np.float32 or offset.shape != (weight.shape[1],):
            raise ValueError('Expected immutable per-column FP32 bias')
        if a.preserve_bias_gelu and '/pwconv1/' in node.name:
            replacements[node.name] = helper.make_node('ExactMatMulScale',
                list(node.input)+[scale], list(mul.output),
                domain='com.supertonic.exact', name=node.name+'/scale_only')
            removed.update([cast.name, mul.name])
            retained_bias += 1
        else:
            replacements[node.name] = helper.make_node('ExactMatMulScaleBias',
                list(node.input)+[scale, bias], list(add.output),
                domain='com.supertonic.exact', name=node.name+'/scale_bias')
            removed.update([cast.name, mul.name, add.name])
        converted.append(node.name)
    if len(converted) != 56:
        raise ValueError(f'Expected56 fixed FFN epilogues, found{len(converted)}')
    if a.preserve_bias_gelu and retained_bias != 28:
        raise ValueError('Expected28 retained FFN1 bias adds')
    nodes = [replacements.get(n.name, n) for n in m.graph.node if n.name not in removed]
    del m.graph.node[:]
    m.graph.node.extend(nodes)
    onnx.checker.check_model(m)
    original_values = {i.name: i.SerializeToString() for i in onnx.load(source).graph.initializer}
    assert original_values == {i.name: i.SerializeToString() for i in m.graph.initializer}
    shutil.copytree(a.source, a.output, copy_function=os.link)
    temporary = a.output/'onnx/vector_epilogue.tmp.onnx'
    onnx.save(m, temporary)
    temporary.replace(a.output/'onnx/vector_estimator.onnx')
    metadata = json.loads((a.source/'manifest.json').read_text())
    metadata.update(status='Experimental exact scale/bias epilogue; server timing and bitidentity required',
        exact_epilogue=dict(converted=converted, removed_nodes=len(removed),
            retained_ffn1_bias_adds=retained_bias,
            initializer_bitidentity=True, source=str(a.source),
            source_vector_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            note='Existing rounded scale product retained. Same S32 accumulator, FP32 cast, multiply, then add. No FMA contraction, requantization, or model change.'))
    temporary = a.output/'manifest.epilogue.tmp'
    temporary.write_text(json.dumps(metadata, indent=2)+'\n')
    temporary.replace(a.output/'manifest.json')
    print('Fused', len(converted), 'FFN epilogues; removed', len(removed), 'nodes')


if __name__ == '__main__':
    main()
