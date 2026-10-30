"""Specialize the existing one-utterance API's batch shape, then cache conditions."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import hashlib
import json
import os
import shutil
import tempfile
import numpy as np
import onnx
from onnx import numpy_helper
from experiments.optimize import split


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or (a.source/'onnx/condition.onnx').exists():
        p.error('Choose an uncached source and new output')
    source = a.source/'onnx/vector_estimator.onnx'
    model = onnx.load(source)
    original_values = {v.name:v for v in model.graph.initializer}
    original = {name:v.SerializeToString() for name,v in original_values.items()}
    values = {v.name:numpy_helper.to_array(v) for v in model.graph.initializer}
    producers = {o:n for n in model.graph.node for o in n.output}
    inputs = {v.name:v for v in model.graph.input}
    remove = set()
    for node in model.graph.node:
        if node.op_type != 'Gather' or node.input[1] not in values:
            continue
        shape = producers.get(node.input[0]);index = values[node.input[1]]
        axis = next((onnx.helper.get_attribute_value(at) for at in node.attribute if at.name=='axis'),0)
        if shape is None or shape.op_type!='Shape' or shape.input[0]!='noisy_latent' or index.shape!=() or int(index)!=0 or axis!=0:
            continue
        model.graph.initializer.append(numpy_helper.from_array(np.array(1,np.int64),node.output[0]))
        remove.add(node.name)
    if len(remove)!=1:
        raise ValueError('Expected one scalar batch dimension lookup')
    nodes=[n for n in model.graph.node if n.name not in remove]
    del model.graph.node[:];model.graph.node.extend(nodes)
    for value in list(model.graph.input)+list(model.graph.output)+list(model.graph.value_info):
        for dim in value.type.tensor_type.shape.dim:
            if dim.dim_param=='batch_size':
                dim.dim_value=1
    onnx.checker.check_model(model)
    shutil.copytree(a.source,a.output,copy_function=os.link)
    (a.output/'onnx/vector_estimator.onnx').unlink()
    (a.output/'onnx/transformation.json').unlink(missing_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        path=Path(temporary)/'batch-one.onnx';onnx.save(model,path);split(path,a.output/'onnx')
    actual={v.name:v.SerializeToString() for part in ['condition','vector_estimator'] for v in onnx.load(a.output/f'onnx/{part}.onnx').graph.initializer}
    # Preserve even the now-unused scalar Gather index for an auditable union.
    core=onnx.load(a.output/'onnx/vector_estimator.onnx')
    for name,value in original_values.items():
        if name not in actual:
            core.graph.initializer.append(value);actual[name]=value.SerializeToString()
    onnx.save(core,a.output/'onnx/vector_estimator.onnx')
    assert all(actual.get(name)==data for name,data in original.items())
    assert len(actual)==len(original)+1
    meta=json.loads((a.source/'manifest.json').read_text())
    meta['engine_experiment_status']='Unselected experiment; paired timing improvement insufficient'
    meta['batch_one_condition_cache']=dict(batch_size=1,source_vector_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),original_initializers_preserved=len(original),added_shape_constants=1,learned_values_changed=False)
    meta['vector_sha256']=hashlib.sha256((a.output/'onnx/vector_estimator.onnx').read_bytes()).hexdigest()
    path=a.output/'manifest.batch.tmp.json';path.write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n');path.replace(a.output/'manifest.json')
    print(meta['batch_one_condition_cache'])


if __name__=='__main__':main()
