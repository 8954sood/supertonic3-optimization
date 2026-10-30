"""Extract pure timestep conditioning; evaluate once at engine load on target CPU."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import copy
import hashlib
import json
import os
import shutil
import onnx
from onnx import helper


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Choose a new output')
    source=args.source/'onnx/vector_estimator.onnx'
    model=onnx.shape_inference.infer_shapes(onnx.load(source))
    deps={i.name:{i.name} for i in model.graph.input}
    deps.update({i.name:set() for i in model.graph.initializer})
    pure=[]
    for node in model.graph.node:
        dependencies=set().union(*(deps.get(i,{i}) for i in node.input if i))
        deps.update({o:dependencies for o in node.output})
        if dependencies and dependencies <= {'current_step','total_step'}:
            pure.append(node)
    names={n.name for n in pure};outputs={o for n in pure for o in n.output}
    boundary=sorted(outputs & {i for n in model.graph.node if n.name not in names for i in n.input})
    info={v.name:v for v in list(model.graph.input)+list(model.graph.value_info)+list(model.graph.output)}
    used={i for n in pure for i in n.input}
    initializers=[i for i in model.graph.initializer if i.name in used]
    time_graph=helper.make_graph(pure,'time_condition_at_load',
        [copy.deepcopy(info[n]) for n in ['current_step','total_step']],
        [copy.deepcopy(info[n]) for n in boundary],initializers)
    time_model=helper.make_model(time_graph,opset_imports=model.opset_import,ir_version=model.ir_version)
    core_nodes=[n for n in model.graph.node if n.name not in names]
    core_used={i for n in core_nodes for i in n.input}
    core_inputs=[copy.deepcopy(v) for v in model.graph.input if v.name in core_used]
    core_inputs.extend(copy.deepcopy(info[n]) for n in boundary)
    core_graph=helper.make_graph(core_nodes,'denoise_with_step_cache',core_inputs,list(model.graph.output),
        [i for i in model.graph.initializer if i.name not in {v.name for v in initializers} or i.name in core_used])
    core_model=helper.make_model(core_graph,opset_imports=model.opset_import,ir_version=model.ir_version)
    for m in [time_model,core_model]:onnx.checker.check_model(m)
    original={i.name:i.SerializeToString() for i in model.graph.initializer}
    actual={i.name:i.SerializeToString() for m in [time_model,core_model] for i in m.graph.initializer}
    if original!=actual:raise RuntimeError('Initializer bytes changed')
    shutil.copytree(args.source,args.output,copy_function=os.link)
    for part,m in [('step_condition',time_model),('vector_estimator',core_model)]:
        temporary=args.output/f'onnx/{part}.step.tmp.onnx';onnx.save(m,temporary);temporary.replace(args.output/f'onnx/{part}.onnx')
    metadata=json.loads((args.source/'manifest.json').read_text())
    metadata['engine_experiment_status']='Unselected experiment; small inconsistent timing improvement'
    metadata['step_condition_steps']=8
    metadata['step_condition_cache']=dict(nodes=len(pure),boundary_tensors=len(boundary),source_vector_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),initializer_bitidentity=True,computed_on_target_cpu_at_engine_load=True)
    metadata['vector_sha256']=hashlib.sha256((args.output/'onnx/vector_estimator.onnx').read_bytes()).hexdigest()
    temporary=args.output/'manifest.step.tmp.json';temporary.write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n');temporary.replace(args.output/'manifest.json')
    print(metadata['step_condition_cache'])


if __name__=='__main__':main()
