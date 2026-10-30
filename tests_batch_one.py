"""Check the batch specialization contract and atomic source preservation."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import onnx
from onnx import TensorProto,helper,numpy_helper
from prepare_native_batch import prepare


class BatchOneTests(unittest.TestCase):
    def fixture(self,root):
        source=root/'source';(source/'onnx').mkdir(parents=True)
        (source/'manifest.json').write_text(json.dumps(dict(backend='native-exact-int8',supported_steps=[8],speed_strategy='latent-retime')))
        (source/'LICENSE').write_text('test')
        graph=helper.make_graph([
            helper.make_node('Shape',['noisy_latent'],['shape'],name='shape'),
            helper.make_node('Gather',['shape','index'],['batch'],name='batch',axis=0),
            helper.make_node('Identity',['noisy_latent'],['out'],name='out')],
            'test',[helper.make_tensor_value_info('noisy_latent',TensorProto.FLOAT,['batch_size',16,2])],
            [helper.make_tensor_value_info('out',TensorProto.FLOAT,['batch_size',16,2])],
            [numpy_helper.from_array(np.array(0,np.int64),'index'),
             numpy_helper.from_array(np.array([1.,2.],np.float32),'learned')])
        model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',17)],ir_version=10)
        onnx.save(model,source/'onnx/vector_estimator.onnx');return source

    def test_only_shape_constant_added_and_source_unchanged(self):
        for link_assets in [False,True]:
            with self.subTest(link_assets=link_assets):
                self.check_preparation(link_assets)

    def check_preparation(self,link_assets):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=self.fixture(root);path=source/'onnx/vector_estimator.onnx'
            before=path.read_bytes();original={t.name:t.SerializeToString() for t in onnx.load(path).graph.initializer}
            receipt=prepare(source,root/'output',link_assets=link_assets);result=onnx.load(root/'output/onnx/vector_estimator.onnx')
            onnx.checker.check_model(result)
            actual={t.name:t.SerializeToString() for t in result.graph.initializer}
            self.assertEqual({k:actual[k] for k in original},original)
            self.assertEqual(set(actual)-set(original),{'batch'})
            self.assertEqual(int(numpy_helper.to_array(next(t for t in result.graph.initializer if t.name=='batch'))),1)
            self.assertFalse(any(n.op_type=='Gather' for n in result.graph.node))
            self.assertEqual(result.graph.input[0].type.tensor_type.shape.dim[0].dim_value,1)
            self.assertEqual(path.read_bytes(),before)
            self.assertEqual(receipt['source_vector_sha256'],hashlib.sha256(before).hexdigest())
            self.assertEqual(receipt['shared_assets'],link_assets)
            self.assertEqual((source/'LICENSE').stat().st_ino==(root/'output/LICENSE').stat().st_ino,link_assets)
            self.assertEqual(json.loads((root/'output/manifest.json').read_text())['speed_strategy'],'latent-retime')
            self.assertEqual(json.loads((root/'output/manifest.json').read_text())['engine_version'],5)

    def test_non_native_model_rejected_without_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=self.fixture(root);manifest=source/'manifest.json';manifest.write_text(json.dumps(dict(backend='onnx',supported_steps=[8])))
            with self.assertRaises(ValueError):prepare(source,root/'output')
            self.assertFalse((root/'output').exists())

    def test_failed_save_preserves_source_and_leaves_no_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=self.fixture(root);path=source/'onnx/vector_estimator.onnx';before=path.read_bytes()
            with patch('prepare_native_batch.onnx.save',side_effect=OSError('simulated write error')):
                with self.assertRaises(OSError):prepare(source,root/'output')
            self.assertFalse((root/'output').exists());self.assertEqual(path.read_bytes(),before)
            self.assertEqual(sorted(p.name for p in root.iterdir()),['source'])


if __name__=='__main__':unittest.main()
