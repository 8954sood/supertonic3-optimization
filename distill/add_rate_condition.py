"""Add256 zero-initialized rate-conditioning parameters; no pruning or extra steps."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import hashlib
import json
import numpy as np
import onnx
from onnx import helper,numpy_helper,TensorProto


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('models/cfg-student-fp32-source/onnx/vector_estimator.onnx'))
    p.add_argument('--output',type=Path,default=Path('models/round8-rate-source-v2/onnx/vector_estimator.onnx'))
    a=p.parse_args()
    if a.output.exists():p.error('Choose new output')
    model=onnx.load(a.source)
    matches=[n for n in model.graph.node if n.name.endswith('/time_encoder/mlp/mlp.0/linear/Gemm')]
    assert len(matches)==1
    original=matches[0].output[0]
    prefix='/native_rate'
    replacement=prefix+'/conditioned_time'
    model.graph.input.append(helper.make_tensor_value_info('speed',TensorProto.FLOAT,[1]))
    model.graph.initializer.extend([
        numpy_helper.from_array(np.array([1.05],np.float32),prefix+'/normal_speed'),
        numpy_helper.from_array(np.array([1],np.int64),prefix+'/axes'),
        numpy_helper.from_array(np.zeros((1,256),np.float32),
             'vector_estimator.tts.ttl.vector_field.rate.weight'),
    ])
    nodes=[]
    for node in model.graph.node:
        for i,value in enumerate(node.input):
            if value==original:node.input[i]=replacement
        nodes.append(node)
        if original in node.output:
            nodes.extend([
                helper.make_node('Sub',['speed',prefix+'/normal_speed'],[prefix+'/delta'],name=prefix+'/Sub'),
                helper.make_node('Unsqueeze',[prefix+'/delta',prefix+'/axes'],[prefix+'/expanded'],name=prefix+'/Unsqueeze'),
                helper.make_node('Mul',[prefix+'/expanded','vector_estimator.tts.ttl.vector_field.rate.weight'],[prefix+'/bias'],name=prefix+'/Mul'),
                helper.make_node('Add',[original,prefix+'/bias'],[replacement],name=prefix+'/Add'),
            ])
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    del model.graph.value_info[:]
    onnx.checker.check_model(model)
    a.output.parent.mkdir(parents=True)
    onnx.save(model,a.output)
    a.output.with_suffix('.json').write_text(json.dumps({
        'source_sha256':hashlib.sha256(a.source.read_bytes()).hexdigest(),
        'extra_parameters':256,'input':'speed','normal_speed':1.05,
        'method':'time-MLP hidden += (speed-1.05)*learned256-vector, initialized zero before its activation. Existing geometry and actual8 iterations retained. No waveform processing.',
    },indent=2)+'\n')
    print(a.output)


if __name__=='__main__':main()
