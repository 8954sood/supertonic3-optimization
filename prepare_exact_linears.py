"""Extend an existing fixed F1200 AVX2 package to its remaining integer linears."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import onnx
from onnx import numpy_helper
from prepare_condition_cache import link_or_copy


def prepare(source, output, *, link_assets=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or source == output or source in output.parents or output in source.parents:
        raise ValueError('Choose a new separate output directory')
    metadata = json.loads((source / 'manifest.json').read_text())
    if metadata.get('backend') != 'native-exact-int8' or metadata.get('supported_steps') != [8]:
        raise ValueError('Use the existing eight-step F1200 exact-integer package')
    if metadata.get('extended_exact_linears') or metadata.get('row_norm_backend'):
        raise ValueError('Source already extended or uses the ARM normalization backend')
    path = source / 'onnx/vector_estimator.onnx'
    source_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    model = onnx.load(path)
    original = {t.name: t.SerializeToString() for t in model.graph.initializer}
    values = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    producer = {name: node for node in model.graph.node for name in node.output}
    existing = sum(node.domain == 'com.supertonic.exact' and node.op_type == 'ExactMatMulInteger'
                   for node in model.graph.node)
    if existing != 56:
        raise ValueError(f'Expected 56 existing exact FFN nodes, got {existing}')
    converted = []
    for node in model.graph.node:
        if node.domain or node.op_type != 'MatMulInteger' or len(node.input) != 4 or len(node.output) != 1:
            continue
        weights, zero = values.get(node.input[1]), values.get(node.input[3])
        if (weights is None or zero is None or weights.dtype != np.int8 or weights.ndim != 2
                or zero.dtype != np.int8 or zero.ndim > 1 or zero.size not in {1, weights.shape[1]}
                or not 0 < weights.shape[0] <= 65536 or not weights.shape[1]
                or weights.shape[0] % 4 or weights.shape[1] % 8
                or weights.min() < -64 or weights.max() > 64 or np.any(zero)):
            continue
        quantizer = producer.get(node.input[0])
        if quantizer is None or quantizer.op_type != 'DynamicQuantizeLinear':
            raise ValueError('Expected unchanged ORT activation quantization')
        node.op_type, node.domain = 'ExactMatMulInteger', 'com.supertonic.exact'
        converted.append(node.name)
    if len(converted) != 28:
        raise ValueError(f'Expected 28 remaining eligible linear nodes, got {len(converted)}')
    onnx.checker.check_model(model)
    if original != {t.name: t.SerializeToString() for t in model.graph.initializer}:
        raise RuntimeError('Initializer bytes changed')
    receipt = dict(added_nodes=converted, total_exact_nodes=existing + len(converted),
                   initializer_bitidentity=True, source_vector_sha256=source_sha)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.exact-linears-', dir=output.parent) as temporary:
        package = Path(temporary) / 'package'
        shutil.copytree(source, package, copy_function=link_or_copy if link_assets else shutil.copy2)
        target = package / 'onnx/vector_estimator.onnx'
        target.unlink()
        onnx.save(model, target)
        if hashlib.sha256(path.read_bytes()).hexdigest() != source_sha:
            raise RuntimeError('Source graph changed during preparation')
        receipt['output_vector_sha256'] = hashlib.sha256(target.read_bytes()).hexdigest()
        metadata['extended_exact_linears'] = receipt
        metadata['vector_sha256'] = receipt['output_vector_sha256']
        manifest = package / 'manifest.json'
        manifest.unlink()
        manifest.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
        package.rename(output)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--link-assets', action='store_true')
    args = parser.parse_args()
    try:
        receipt = prepare(args.source, args.output, link_assets=args.link_assets)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
