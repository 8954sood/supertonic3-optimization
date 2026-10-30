"""Inspect ORT fusion/layout changes without treating a profile as a speed result."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from collections import Counter
import json
import tempfile

import onnx
import onnxruntime as ort
from runtime import options


def optimized_nodes(model, part, destination):
    metadata = json.loads((model / 'manifest.json').read_text())
    settings = options(row_norm_backend=metadata.get('row_norm_backend', False),
                       row_norm_widths=tuple(metadata.get('row_norm_widths', [512])))
    settings.optimized_model_filepath = str(destination)
    session = ort.InferenceSession(str(model / f'onnx/{part}.onnx'), settings,
                                  providers=['CPUExecutionProvider'])
    del session
    return list(onnx.load(destination).graph.node)


def signature(node):
    return (node.domain, node.op_type, tuple(node.input), tuple(node.output),
            tuple(sorted(attribute.SerializeToString() for attribute in node.attribute)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output')
    report = dict(reference=str(args.reference), candidate=str(args.candidate),
                  note='Installed ORT optimizer diagnostic; no synthesis speed or quality score.',
                  rows=[])
    with tempfile.TemporaryDirectory() as temporary:
        for part in ['duration_predictor', 'text_encoder', 'vocoder']:
            nodes = {name: optimized_nodes(model, part, Path(temporary) / f'{name}-{part}.onnx')
                     for name, model in [('reference', args.reference), ('candidate', args.candidate)]}
            other = {name: Counter(signature(node) for node in values
                                   if node.op_type not in {'LayerNormalization', 'RowLayerNorm'})
                     for name, values in nodes.items()}
            report['rows'].append(dict(part=part,
                operator_counts={name: dict(Counter(node.op_type for node in values))
                                 for name, values in nodes.items()},
                non_normalization_nodes_identical=other['reference'] == other['candidate']))
    report['temporary_graphs_removed'] = True
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
