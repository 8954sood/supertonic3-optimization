"""Original50 normal-rate text attention for fast native alignment supervision.

Reruns actual original50 trajectories; no student-produced target, no native2
teacher. Records last-step text attention and checks final latent reproduction.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import hashlib
import json
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort
from runtime import options
from distill.teacher_policy import assert_original_data, assert_original_model
from experiments.compare_original_speed import EXPECTED


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('distill/fast-original50-round8'))
    p.add_argument('--audit',type=Path,default=Path('results/round8_training_references_asr.json'))
    p.add_argument('--output',type=Path,default=Path('distill/original50-attention-round8-v2'))
    a=p.parse_args()
    if a.output.exists(): p.error('Choose new output')
    metadata=json.loads((a.data/'manifest.json').read_text())
    assert_original_data(metadata)
    audit=json.loads(a.audit.read_text())
    assert len(audit['rows'])==len(metadata['rows'])
    checked={r['audio']:r for r in audit['rows']}
    selected=[r for r in metadata['rows'] if not checked[r['audio']]['errors']]
    model=Path('models/supertonic-3')
    assert_original_model(model)
    graph_path=model/'onnx/vector_estimator.onnx'
    assert hashlib.sha256(graph_path.read_bytes()).hexdigest()==EXPECTED['vector_estimator']
    graph=onnx.load(graph_path)
    names=[n.output[0] for n in graph.graph.node if n.op_type=='Softmax' and '/attn/' in n.name]
    assert len(names)==4
    for name in names: graph.graph.output.append(helper.make_tensor_value_info(name,TensorProto.FLOAT,[None,None,None,None]))
    a.output.mkdir(parents=True)
    temporary=a.output/'teacher_attention.onnx'
    onnx.save(graph,temporary)
    settings=options()
    settings.intra_op_num_threads=4
    session=ort.InferenceSession(str(temporary),settings,providers=['CPUExecutionProvider'])
    report={'teacher_origin':metadata['teacher_origin'],'steps':50,'normal_speed':1.05,
            'method':'Last actual original50 denoising step text-attention, conditional branch only. Reproduced from stored original starting noise; final latent compared to pinned cached target.',
            'attention_outputs':names,'rows':[],'complete':False}
    source=Path(metadata['source'])
    outputs=[session.get_outputs()[0].name]+names
    for row in selected:
        split,case=row['split'],row['case']
        feed=dict(np.load(source/split/f'{case:03d}_0.npz'))
        feed.pop('target')
        feed['total_step']=np.array([50],np.float32)
        xt=feed['noisy_latent']
        for step in range(50):
            feed.update(noisy_latent=xt,current_step=np.array([step],np.float32))
            values=session.run(outputs if step==49 else outputs[:1],feed)
            xt=values[0]
        reference=np.load(source/split/f'{case:03d}_7.npz')['target']
        error=float(np.max(np.abs(xt-reference)))
        if error>1e-4: raise ValueError('Original trajectory did not reproduce: '+str(error))
        folder=a.output/split
        folder.mkdir(exist_ok=True)
        path=folder/f'{case:03d}.npz'
        # Export layout is [heads, CFG batch, latent time, text], verified
        # against the preceding transpose [2,0,1,3]. Conditional batch is0.
        if any(v.shape[:2]!=(8,2) for v in values[1:]):
            raise ValueError('Unexpected teacher attention layout')
        maps={f'attention_{i}':v[:,:1].copy() for i,v in enumerate(values[1:])}
        np.savez_compressed(path,**maps)
        report['rows'].append({'split':split,'case':case,'voice':row['voice'],
             'file':str(path),'max_final_latent_error':error,
             'shapes':[list(v.shape) for v in maps.values()],
             'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        (a.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
        print(split,case,'max_error',error,flush=True)
    report['complete']=True
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__': main()
