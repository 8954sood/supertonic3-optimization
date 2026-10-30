"""Source immutability, tensor preservation, and rollback for integer coverage."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
from prepare_exact_linears import prepare


class ExactLinearPreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / 'source'
        (self.source / 'onnx').mkdir(parents=True)
        nodes = [helper.make_node('DynamicQuantizeLinear', ['x'], ['qx', 'scale', 'az'])]
        for index in range(84):
            custom = index < 56
            nodes.append(helper.make_node('ExactMatMulInteger' if custom else 'MatMulInteger',
                ['qx', 'w', 'az', 'bz'], [f'y{index}'], name=f'linear{index}',
                domain='com.supertonic.exact' if custom else ''))
        self.model = helper.make_model(helper.make_graph(nodes, 'fixed_fixture',
            [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 'T', 8])],
            [helper.make_tensor_value_info('y83', TensorProto.INT32, [1, 'T', 16])],
            [numpy_helper.from_array(np.full((8, 16), 32, np.int8), 'w'),
             numpy_helper.from_array(np.zeros(16, np.int8), 'bz')]),
            opset_imports=[helper.make_opsetid('', 17), helper.make_opsetid('com.supertonic.exact', 1)],
            ir_version=10)
        onnx.save(self.model, self.source / 'onnx/vector_estimator.onnx')
        (self.source / 'manifest.json').write_text(json.dumps(dict(
            backend='native-exact-int8', supported_steps=[8], speed_strategy='latent-retime')))
        self.before = {p.relative_to(self.source): p.read_bytes()
                       for p in self.source.rglob('*') if p.is_file()}

    def test_copy_and_links_preserve_source_weights_and_metadata(self):
        for linked in [False, True]:
            output = self.root / ('linked' if linked else 'copied')
            receipt = prepare(self.source, output, link_assets=linked)
            self.assertEqual(len(receipt['added_nodes']), 28)
            self.assertEqual(receipt['total_exact_nodes'], 84)
            model = onnx.load(output / 'onnx/vector_estimator.onnx')
            self.assertEqual([t.SerializeToString() for t in self.model.graph.initializer],
                             [t.SerializeToString() for t in model.graph.initializer])
            self.assertEqual(json.loads((output / 'manifest.json').read_text())['speed_strategy'], 'latent-retime')
            with self.assertRaises(ValueError):
                prepare(output, self.root / ('twice' + str(linked)))
        self.assertEqual(self.before, {p.relative_to(self.source): p.read_bytes()
                                      for p in self.source.rglob('*') if p.is_file()})

    def test_out_of_range_weight_is_rejected_without_output(self):
        self.model.graph.initializer[0].CopyFrom(
            numpy_helper.from_array(np.full((8, 16), 65, np.int8), 'w'))
        onnx.save(self.model, self.source / 'onnx/vector_estimator.onnx')
        output = self.root / 'invalid'
        with self.assertRaisesRegex(ValueError, '28 remaining eligible'):
            prepare(self.source, output)
        self.assertFalse(output.exists())

    def test_failed_save_leaves_source_and_no_partial_output(self):
        output = self.root / 'failed'
        with patch('prepare_exact_linears.onnx.save', side_effect=RuntimeError('failed save')):
            with self.assertRaisesRegex(RuntimeError, 'failed save'):
                prepare(self.source, output, link_assets=True)
        self.assertFalse(output.exists())
        self.assertFalse(list(self.root.glob('.exact-linears-*')))
        self.assertEqual(self.before, {p.relative_to(self.source): p.read_bytes()
                                      for p in self.source.rglob('*') if p.is_file()})


if __name__ == '__main__':
    unittest.main()
