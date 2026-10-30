"""ARM SDOT vs ORT S32 microprobe, with the same ORT-call overhead on both sides."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import runtime
import argparse
import hashlib
import json
import platform
import statistics
import time
import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto,helper,numpy_helper


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True);p.add_argument('--libraries',nargs='+',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():p.error('Choose a new output')
    if platform.machine().lower() not in {'arm64','aarch64'}:p.error('ARM host required')
    model=onnx.load(args.model/'onnx/vector_estimator.onnx');weights={}
    for tensor in model.graph.initializer:
        if '/pwconv' not in tensor.name:continue
        value=numpy_helper.to_array(tensor)
        if value.dtype==np.int8 and value.ndim==2 and value.shape not in weights:weights[value.shape]=value.copy()
    del model
    report=dict(note='Single ORT integer operator only, not TTS RTF. Same call/allocator settings. Fixed actual FFN weights.',machine=platform.machine(),system=platform.system(),threads=1,library_sha256={lib.name:hashlib.sha256(lib.read_bytes()).hexdigest() for lib in args.libraries},rows=[],complete=False)
    rng=np.random.RandomState(617)
    for m in [60,200]:
        for (k,n),weight in sorted(weights.items()):
            sessions={}
            for lib in [None,*args.libraries]:
                node=helper.make_node('ExactMatMulInteger' if lib else 'MatMulInteger',['x','w','az','bz'],['y'],domain='com.supertonic.exact' if lib else '')
                graph=helper.make_graph([node],'integer_probe',[helper.make_tensor_value_info('x',TensorProto.UINT8,[m,k])],[helper.make_tensor_value_info('y',TensorProto.INT32,[m,n])],[numpy_helper.from_array(weight,'w'),numpy_helper.from_array(np.array(128,np.uint8),'az'),numpy_helper.from_array(np.zeros(n,np.int8),'bz')])
                model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',17),helper.make_opsetid('com.supertonic.exact',1)],ir_version=10)
                options=runtime.options();options.enable_mem_pattern=False
                if lib:options.register_custom_ops_library(str(lib.resolve()))
                sessions[lib.stem if lib else 'ort']=ort.InferenceSession(model.SerializeToString(),options,providers=['CPUExecutionProvider'])
            x=rng.randint(0,256,(m,k),dtype=np.uint8);feed={'x':x};values={name:session.run(None,feed)[0] for name,session in sessions.items()}
            for value in values.values():np.testing.assert_array_equal(value,values['ort'])
            timings={name:[] for name in sessions}
            for repeat in range(36):
                names=list(sessions);shift=repeat%len(names);names=names[shift:]+names[:shift]
                if repeat%2:names.reverse()
                for name in names:
                    start=time.perf_counter();value=sessions[name].run(None,feed)[0];timings[name].append(time.perf_counter()-start)
            medians={name:statistics.median(values) for name,values in timings.items()}
            row=dict(shape=[m,k,n],s32_bitidentical=True,seconds=medians,speedup={name:medians['ort']/seconds for name,seconds in medians.items() if name!='ort'},weight_sha256=hashlib.sha256(weight.tobytes()).hexdigest())
            report['rows'].append(row);print(row['shape'],row['speedup'],flush=True)
            args.output.write_text(json.dumps(report,indent=2)+'\n')
    report['complete']=True;args.output.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
