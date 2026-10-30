"""Compare the native kernel to installed ORT, including tails and large offsets."""
import platform
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from runtime import options
import numpy as np
import onnxruntime as ort
import onnx
from onnx import helper, TensorProto, numpy_helper
from row_norm_ops import library_path
from prepare_row_norm import prepare


class RowNormalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if platform.machine().lower() not in {'aarch64','arm64'}:
            raise unittest.SkipTest('Optional ARM64 kernel')
        cls.library=library_path(widths=(64,256,512))

    def session(self,custom,epsilon=1e-6,axis=-1,width=512):
        domain='com.supertonic.rownorm' if custom else ''
        name='RowLayerNorm' if custom else 'LayerNormalization'
        model=helper.make_model(helper.make_graph([
            helper.make_node(name,['x','scale','bias'],['y'],domain=domain,axis=axis,epsilon=epsilon)],
            'reference_comparison',[
                helper.make_tensor_value_info('x',TensorProto.FLOAT,[1,'T',width]),
                helper.make_tensor_value_info('scale',TensorProto.FLOAT,[width]),
                helper.make_tensor_value_info('bias',TensorProto.FLOAT,[width])],
            [helper.make_tensor_value_info('y',TensorProto.FLOAT,[1,'T',width])]),
            opset_imports=[helper.make_opsetid('',17),helper.make_opsetid('com.supertonic.rownorm',1)],ir_version=10)
        settings=options()
        if custom:settings.register_custom_ops_library(self.library)
        return ort.InferenceSession(model.SerializeToString(),settings,providers=['CPUExecutionProvider'])

    def test_exact_results_for_row_tails_and_extreme_inputs(self):
        rng=np.random.RandomState(101)
        for width in [64,256,512]:
            self.compare_width(rng,width)

    def compare_width(self,rng,width):
        for epsilon in [1e-6,1e-5]:
            reference,candidate=self.session(False,epsilon,width=width),self.session(True,epsilon,width=width)
            for rows in [1,2,3,4,5,7,8,9,13,15,16,17,31,32,33,47,48,63,64,65,81,129,513]:
                for magnitude,offset in [(1,0),(1e-6,0),(1e3,0),(0,1024),(1,1e6)]:
                    with self.subTest(width=width,epsilon=epsilon,rows=rows,magnitude=magnitude,offset=offset):
                        x=(rng.randn(1,rows,width)*magnitude+offset).astype(np.float32)
                        feed=dict(x=x,scale=rng.randn(width).astype(np.float32),bias=rng.randn(width).astype(np.float32))
                        expected,actual=reference.run(None,feed)[0],candidate.run(None,feed)[0]
                        np.testing.assert_array_equal(expected,actual)
                        self.assertEqual(expected.tobytes(),actual.tobytes())

    def test_invalid_affine_shape_is_rejected(self):
        session=self.session(True)
        with self.assertRaises(ort.capi.onnxruntime_pybind11_state.InvalidArgument):
            session.run(None,dict(x=np.ones((1,5,512),np.float32),
                scale=np.ones(511,np.float32),bias=np.zeros(512,np.float32)))

    def test_unsupported_axis_is_rejected_at_load(self):
        with self.assertRaisesRegex(Exception,'Expected last-axis normalization'):
            self.session(True,axis=1)

    def test_incompatible_runtime_is_refused(self):
        library_path.cache_clear()
        try:
            with patch('row_norm_ops.np.array_equal',return_value=False):
                with self.assertRaisesRegex(RuntimeError,'differs from installed ORT'):
                    library_path()
        finally:
            library_path.cache_clear()


class PreparationTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name);self.source=self.root/'source'
        (self.source/'onnx').mkdir(parents=True)
        nodes=[];previous='x'
        for index in range(36):
            output=f'y{index}'
            nodes.append(helper.make_node('LayerNormalization',[previous,'scale','bias'],[output],
                axis=-1,epsilon=1e-6,name=f'norm{index}'));previous=output
        model=helper.make_model(helper.make_graph(nodes,'test_package',[
            helper.make_tensor_value_info('x',TensorProto.FLOAT,[1,'T',512])],
            [helper.make_tensor_value_info(previous,TensorProto.FLOAT,[1,'T',512])],
            [numpy_helper.from_array(np.ones(512,np.float32),'scale'),
             numpy_helper.from_array(np.zeros(512,np.float32),'bias')]),
            opset_imports=[helper.make_opsetid('',17)],ir_version=10)
        onnx.save(model,self.source/'onnx/vector_estimator.onnx')
        (self.source/'manifest.json').write_text(json.dumps(dict(supported_steps=[8],status='fixture')))
        self.before={p.relative_to(self.source):p.read_bytes() for p in self.source.rglob('*') if p.is_file()}

    def test_preparation_preserves_source_and_learned_bytes(self):
        for linked in [False,True]:
            output=self.root/('linked' if linked else 'copied')
            receipt=prepare(self.source,output,link_assets=linked)
            self.assertTrue(receipt['initializer_bitidentity'])
            graph=onnx.load(output/'onnx/vector_estimator.onnx')
            self.assertEqual(sum(n.op_type=='RowLayerNorm' for n in graph.graph.node),36)
            original=onnx.load(self.source/'onnx/vector_estimator.onnx')
            self.assertEqual([v.SerializeToString() for v in graph.graph.initializer],
                             [v.SerializeToString() for v in original.graph.initializer])
            self.assertEqual(json.loads((output/'manifest.json').read_text())['status'],'fixture')
        self.assertEqual(self.before,{p.relative_to(self.source):p.read_bytes()
                                     for p in self.source.rglob('*') if p.is_file()})
        with self.assertRaises(ValueError):prepare(self.source,self.source/'nested')

    def test_failed_write_cleans_up_and_preserves_source(self):
        output=self.root/'failed'
        with patch('prepare_row_norm.onnx.save',side_effect=RuntimeError('save failed')):
            with self.assertRaisesRegex(RuntimeError,'save failed'):
                prepare(self.source,output,link_assets=True)
        self.assertFalse(output.exists())
        self.assertFalse(list(self.root.glob('.row-norm-*')))
        self.assertEqual(self.before,{p.relative_to(self.source):p.read_bytes()
                                     for p in self.source.rglob('*') if p.is_file()})

    def test_all_parts_preserve_shapes_and_can_extend_vector_package(self):
        for part,count,width in [('duration_predictor',10,64),('text_encoder',15,256),('vocoder',10,512)]:
            model=onnx.load(self.source/'onnx/vector_estimator.onnx')
            del model.graph.node[count:]
            model.graph.output[0].name=model.graph.node[-1].output[0]
            for value in [model.graph.input[0],model.graph.output[0]]:
                value.type.tensor_type.shape.dim[-1].dim_value=width
            del model.graph.initializer[:]
            model.graph.initializer.extend([
                numpy_helper.from_array(np.ones(width,np.float32),'scale'),
                numpy_helper.from_array(np.zeros(width,np.float32),'bias')])
            onnx.save(model,self.source/f'onnx/{part}.onnx')
        source_bytes={p.relative_to(self.source):p.read_bytes() for p in self.source.rglob('*') if p.is_file()}
        vector_only=self.root/'vector_only'
        prepare(self.source,vector_only,link_assets=True)
        for name,source in [('fresh',self.source),('extended',vector_only)]:
            output=self.root/name
            receipt=prepare(source,output,link_assets=True,all_parts=True)
            self.assertEqual(receipt['total_native_norm_nodes'],71)
            for part,part_receipt in receipt['parts'].items():
                self.assertTrue(part_receipt['preserved_inferred_shapes'])
                self.assertTrue(part_receipt['initializer_bitidentity'])
                graph=onnx.load(output/f'onnx/{part}.onnx')
                self.assertTrue(graph.graph.value_info)
                self.assertEqual(sum(n.op_type=='RowLayerNorm' for n in graph.graph.node),part_receipt['nodes'])
            with self.assertRaisesRegex(ValueError,'already includes frontend'):
                prepare(output,self.root/f'{name}_twice',all_parts=True)
        self.assertEqual(source_bytes,{p.relative_to(self.source):p.read_bytes()
                                     for p in self.source.rglob('*') if p.is_file()})


if __name__=='__main__':unittest.main()
