"""Train native rate-conditioned eight-step flow on original-derived clean latents.

Targets: original50 normal trajectories, or decoder-inverted R3 compression of
original50 normal audio. No distilled teacher and no inference waveform filter.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import runtime
import argparse
import hashlib
import json
import time
import numpy as np
import torch
from distill.torch_graph import VectorGraph
from distill.teacher_policy import assert_original_data


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inversion',type=Path,required=True)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--initial',default='distill/joint-original50-round5/best.pt')
    p.add_argument('--graph',default='models/round8-rate-source-v2/onnx/vector_estimator.onnx')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--iterations',type=int,default=1000)
    p.add_argument('--validate-every',type=int,default=250)
    p.add_argument('--pilot-fit',action='store_true')
    a=p.parse_args()
    if a.output.exists():p.error('Choose new output')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261029)
    metadata=json.loads((a.inversion/'manifest.json').read_text())
    assert_original_data(metadata)
    if not metadata['complete'] or metadata['teacher_origin']['steps']!=50:
        raise ValueError('Complete original50 inversion required')
    audit=json.loads(a.audit.read_text())
    if len(audit['rows'])!=len(metadata['rows']):raise ValueError('Complete latent reconstruction ASR audit required')
    checked={r['audio']:r for r in audit['rows']}
    rows=[r for r in metadata['rows'] if checked[r['audio']]['errors']==0]
    train=[r for r in rows if r['split']=='train']
    val=[r for r in rows if r['split']=='validation']
    if a.pilot_fit:val=train
    if not train or not val:raise ValueError('Need training and held-out validation targets, or explicit pilot-fit mode')
    fast_metadata=json.loads(Path('distill/fast-original50-round8/manifest.json').read_text())
    assert_original_data(fast_metadata)
    fast_rows={(r['split'],r['case']):r for r in fast_metadata['rows']}
    source=Path(fast_metadata['source'])
    normal=json.loads((source/'manifest.json').read_text())
    assert_original_data(normal)
    normal_train=[r for r in normal['rows'] if r['split']=='train']
    normal_val=[r for r in normal['rows'] if r['split']=='validation' and r['case'] in [0,1,4,5] and r['step'] in [0,3,7]]
    model=VectorGraph(a.graph,checkpoints=a.initial)
    model.train_full_vector()
    rate_name=model.names['vector_estimator.tts.ttl.vector_field.rate.weight']
    rate_parameter=model.trainable[rate_name]
    ordinary=[v for k,v in model.trainable.items() if k!=rate_name]
    optimizer=torch.optim.AdamW([{'params':ordinary,'lr':1e-5},
        {'params':[rate_parameter],'lr':5e-4}],weight_decay=1e-5)
    cache={}
    def normal_sample(row):
        key=('normal',row['split'],row['case'],row['step'])
        if key not in cache:
            feed=dict(np.load(source/row['split']/f"{row['case']:03d}_{row['step']}.npz"))
            target=torch.from_numpy(feed.pop('target')).to('mps')
            feed['speed']=np.array([1.05],np.float32)
            cache[key]=(feed,target)
        return cache[key]
    def fast_sample(row,step,seed):
        key=('fast',row['split'],row['case'])
        if key not in cache:
            stored=fast_rows[(row['split'],row['case'])]
            feed=dict(np.load(stored['file']))
            latent=np.load(row['file'])['latent']
            if latent.shape!=feed['noisy_latent'].shape:raise ValueError('Fast latent shape mismatch')
            cache[key]=(feed,latent)
        stored,latent=cache[key]
        feed=dict(stored)
        noise=np.random.RandomState(seed).randn(*latent.shape).astype(np.float32)
        t=step/8
        state=noise*(1-t)+latent*t
        feed.update(noisy_latent=state,current_step=np.array([step],np.float32),
                    total_step=np.array([8],np.float32),speed=np.array([2.0],np.float32))
        target=state+(latent-noise)/8
        return feed,torch.from_numpy(target).to('mps')
    def update_loss(feed,target):
        actual=model(feed)[0]
        difference=target-torch.from_numpy(feed['noisy_latent']).to('mps')
        return (actual-target).square().mean()/difference.square().mean().clamp_min(1e-5)
    @torch.no_grad()
    def validate():
        fast=float(np.mean([float(update_loss(*fast_sample(row,s,40000+row['case']*17+s)).cpu()) for row in val for s in [0,3,7]]))
        normal_value=float(np.mean([float(update_loss(*normal_sample(row)).cpu()) for row in normal_val]))
        return {'fast_update_mse':fast,'normal_update_mse':normal_value,'selection':fast+normal_value}
    report={'config':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        'teacher_origin':metadata['teacher_origin'],'method':__doc__,
        'steps':8,'extra_rate_parameters':256,'training_fast_cases':len(train),'validation_fast_cases':len(val),
        'validation_is_training_contexts':a.pilot_fit,
        'fast_target_latent_sha256':{r['split']+':'+str(r['case']):hashlib.sha256(Path(r['file']).read_bytes()).hexdigest() for r in rows},
        'note':'Fast flow uses synthetic straight paths to frozen-original-decoder inverted targets, not an original diffusion trajectory. New noise every update. Original normal50 trajectories provide preservation updates. Pilot mode is an overfit sanity check, not generalization evidence.',
        'history':[],'complete':False}
    a.output.mkdir(parents=True)
    start=time.perf_counter()
    initial=validate()
    report['initial']=initial
    best=initial['selection']
    def save(name,step,value):
        payload={'weights':model.export_weights(),'step':step,
            'validation_relative_update_mse':value['selection'],'config':report['config']}
        temp=a.output/(name+'.tmp')
        torch.save(payload,temp)
        temp.replace(a.output/(name+'.pt'))
    save('best',0,initial)
    print('initial',initial,flush=True)
    rng=np.random.default_rng(20261029)
    for step in range(1,a.iterations+1):
        optimizer.zero_grad(set_to_none=True)
        if step%2:
            language=['ko','en'][int(rng.integers(2))]
            choices=[r for r in train if r['language']==language]
            row=choices[int(rng.integers(len(choices)))]
            feed,target=fast_sample(row,int(rng.integers(8)),int(rng.integers(2**31)))
        else:
            row=normal_train[int(rng.integers(len(normal_train)))]
            feed,target=normal_sample(row)
        loss=update_loss(feed,target)
        if not torch.isfinite(loss).item():raise RuntimeError('Nonfinite flow training')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable.parameters(),1.0)
        optimizer.step()
        if step%25==0:print('step',step,'loss',float(loss.detach()),'seconds',round(time.perf_counter()-start,2),flush=True)
        if step%a.validate_every==0 or step==a.iterations:
            value=validate()
            report['history'].append({'step':step,**value,'seconds':time.perf_counter()-start})
            save('last',step,value)
            save(f'step_{step:05d}',step,value)
            if value['selection']<best:
                best=value['selection'];save('best',step,value)
            report['complete']=step==a.iterations
            (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            print('validation',step,value,flush=True)


if __name__=='__main__':main()
