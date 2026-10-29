"""Compare fused and separate cast/multiply/add in ORT on an AVX2 host."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runtime
import argparse
import json
import numpy as np
import onnxruntime as ort
import psutil
from onnx import TensorProto as T, helper as h, numpy_helper as nh


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    psutil.Process().cpu_affinity([0])
    rng = np.random.RandomState(901)
    rows = []
    cases = [([1,1],12,8,0,'random64'), ([1,5],144,512,255,'random64'),
        ([2,4],512,1536,127,'random64'), ([1,15],1536,512,71,'random64'),
        ([1,8],8,8,0,'positive32'), ([1,8],16,16,0,'positive64'),
        ([1,8],8,8,255,'negative32'), ([1,17],512,16,128,'random32'),
        ([1,9],4096,16,0,'positive63')]
    for shape, k, n, zero, pattern in cases:
        x = rng.randint(0,256,(*shape,k),dtype=np.uint8)
        limit = 32 if pattern == 'random32' else 64
        w = rng.randint(-limit,limit+1,(k,n)).astype(np.int8)
        if pattern.startswith(('positive','negative')):
            value = int(pattern.replace('positive','').replace('negative',''))
            w.fill(-value if pattern.startswith('negative') else value)
            if value != 63:
                x.fill(255)
        scale = rng.uniform(1e-6,.1,n).astype(np.float32)
        bias = rng.randn(n).astype(np.float32)
        init = [nh.from_array(v,name) for v,name in [(w,'w'),(np.array(zero,np.uint8),'az'),
            (np.zeros(n,np.int8),'bz'),(scale,'scale'),(bias,'bias')]]
        results = []
        for fused in [False, True]:
            nodes = ([h.make_node('ExactMatMulScaleBias',['x','w','az','bz','scale','bias'],['y'],domain='com.supertonic.exact')]
                if fused else [h.make_node('ExactMatMulInteger',['x','w','az','bz'],['i'],domain='com.supertonic.exact'),
                    h.make_node('Cast',['i'],['f'],to=T.FLOAT), h.make_node('Mul',['f','scale'],['s']),
                    h.make_node('Add',['s','bias'],['y'])])
            graph = h.make_graph(nodes,'epilogue_check',
                [h.make_tensor_value_info('x',T.UINT8,[*shape,k])],
                [h.make_tensor_value_info('y',T.FLOAT,[*shape,n])],init)
            model = h.make_model(graph,opset_imports=[h.make_opsetid('',17),h.make_opsetid('com.supertonic.exact',1)],ir_version=10)
            opt = runtime.options()
            opt.register_custom_ops_library(a.library)
            session = ort.InferenceSession(model.SerializeToString(),opt,providers=['CPUExecutionProvider'])
            first = session.run(None,{'x':x})[0]
            np.testing.assert_array_equal(first,session.run(None,{'x':x})[0])
            results.append(first)
            del session
        np.testing.assert_array_equal(*results)
        oracle = (((x.astype(np.int32)-zero)@w.astype(np.int32)).astype(np.float32)*scale)+bias
        np.testing.assert_array_equal(results[1],oracle)
        rows.append(dict(shape=[*shape,k,n],zero=zero,pattern=pattern,fp32_bitidentity=True))
    a.output.write_text(json.dumps(dict(library=Path(a.library).name,complete=True,rows=rows),indent=2)+'\n')
    print('Nine fused epilogue cases match separate operators and integer/FP32 oracle exactly.')


if __name__ == '__main__':
    main()
