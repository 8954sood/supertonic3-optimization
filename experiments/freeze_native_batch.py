"""Specialize the one-utterance native API to batch one; preserve every weight."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import hashlib
import json
import shutil
import tempfile
import numpy as np
import onnx
from onnx import numpy_helper
from prepare_condition_cache import link_or_copy


def prepare(source,output):
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists() or source in output.parents or output in source.parents:
        raise ValueError('Choose a new, separate output')
    metadata=json.loads((source/'manifest.json').read_text())
    if metadata.get('backend')!='native-exact-int8' or metadata.get('supported_steps')!=[8]:
        raise ValueError('Expected an eight-step native integer package')
    path=source/'onnx/vector_estimator.onnx';before=hashlib.sha256(path.read_bytes()).hexdigest()
    model=onnx.load(path);original={v.name:v.SerializeToString() for v in model.graph.initializer}
    values={v.name:numpy_helper.to_array(v) for v in model.graph.initializer}
    producers={o:n for n in model.graph.node for o in n.output};remove=[]
    for node in model.graph.node:
        if node.op_type!='Gather' or len(node.input)<2 or node.input[1] not in values:continue
        index=values[node.input[1]];shape=producers.get(node.input[0])
        axis=next((onnx.helper.get_attribute_value(a) for a in node.attribute if a.name=='axis'),0)
        if shape is not None and shape.op_type=='Shape' and list(shape.input)==['noisy_latent'] and index.shape==() and int(index)==0 and axis==0:
            model.graph.initializer.append(numpy_helper.from_array(np.array(1,np.int64),node.output[0]));remove.append(node)
    if len(remove)!=1:raise ValueError('Expected one scalar batch lookup')
    removed_name=remove[0].name
    nodes=[node for node in model.graph.node if node.name!=removed_name]
    del model.graph.node[:];model.graph.node.extend(nodes)
    fixed=0
    for value in list(model.graph.input)+list(model.graph.output)+list(model.graph.value_info):
        for dim in value.type.tensor_type.shape.dim:
            if dim.dim_param=='batch_size':dim.dim_value=1;fixed+=1
    if not fixed:raise ValueError('No symbolic batch dimension')
    onnx.checker.check_model(model)
    actual={v.name:v.SerializeToString() for v in model.graph.initializer}
    if len(actual)!=len(original)+1 or any(actual.get(name)!=data for name,data in original.items()):raise ValueError('Initializer changed')
    receipt=dict(source_vector_sha256=before,batch_size=1,removed_batch_lookup=removed_name,
                 fixed_dimensions=fixed,original_initializers_preserved=len(original),
                 added_shape_constants=1,learned_values_changed=False,
                 note='No host-computed floating constants. ORT may fold invariant key projections on target CPU at session load.')
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.native-batch-',dir=output.parent) as temporary:
        package=Path(temporary)/'package';shutil.copytree(source,package,copy_function=link_or_copy)
        vector=package/'onnx/vector_estimator.onnx';vector.unlink();onnx.save(model,vector)
        metadata['native_batch_one']=receipt;metadata['engine_experiment_status']='Unselected; full target timing required'
        metadata['vector_sha256']=hashlib.sha256(vector.read_bytes()).hexdigest()
        manifest=package/'manifest.json';manifest.unlink();manifest.write_text(json.dumps(metadata,indent=2)+'\n')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=before:raise RuntimeError('Source changed')
        package.rename(output)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps(prepare(a.source,a.output),indent=2))

if __name__=='__main__':main()
