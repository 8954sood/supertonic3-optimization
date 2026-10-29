"""Learn a pre-decoder latent-rate head; F1200 inputs, original50-only targets."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import runtime
from runtime import options
import argparse
import hashlib
import json
import time
import numpy as np
import soundfile as sf
import torch
import onnxruntime as ort
from distill.torch_graph import VectorGraph
from distill.latent_rate_head import LatentRateHead
from distill.teacher_policy import assert_original_data


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inversion',type=Path,required=True)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--iterations',type=int,default=2000)
    p.add_argument('--validate-every',type=int,default=500)
    p.add_argument('--pilot-fit',action='store_true')
    p.add_argument('--initial',type=Path)
    p.add_argument('--audio-every',type=int,default=4)
    p.add_argument('--latent-weight',type=float,default=0.25)
    a=p.parse_args()
    if a.output.exists():p.error('Choose new output')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261029)
    metadata=json.loads((a.inversion/'manifest.json').read_text())
    assert_original_data(metadata)
    audit=json.loads(a.audit.read_text())
    assert metadata['complete'] and len(audit['rows'])==len(metadata['rows'])
    checked={r['audio']:r for r in audit['rows']}
    rows=[r for r in metadata['rows'] if checked[r['audio']]['errors']==0]
    train=[r for r in rows if r['split']=='train']
    val=[r for r in rows if r['split']=='validation']
    if a.pilot_fit:val=train
    assert train and val
    fast=json.loads(Path('distill/fast-original50-round8/manifest.json').read_text())
    assert_original_data(fast)
    source=Path(fast['source'])
    fast_rows={(r['split'],r['case']):r for r in fast['rows']}
    vector_path=Path('models/round5-original50-student/onnx/vector_estimator.onnx')
    digest=hashlib.sha256(vector_path.read_bytes()).hexdigest()
    assert digest=='7c81ab7f0ddcc4266743701599a8708d2edd1be04285770f996e84a8bb5f4447'
    vector=ort.InferenceSession(str(vector_path),options(),providers=['CPUExecutionProvider'])
    required={v.name for v in vector.get_inputs()}
    model=LatentRateHead().to('mps')
    if a.initial:model.load_state_dict(torch.load(a.initial,map_location='mps',weights_only=False)['state'])
    decoder=VectorGraph('models/supertonic-3/onnx/vocoder.onnx')
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-5)
    windows={n:torch.hann_window(n) for n in [512,1024,2048]}
    cache={}
    def spectra(wave):
        wave=wave.reshape(-1).to('cpu')
        return [torch.stft(wave,n_fft=n,hop_length=n//4,window=windows[n],return_complex=True).abs() for n in windows]
    def sample(row):
        key=(row['split'],row['case'])
        if key not in cache:
            entry=fast_rows[key]
            stored=dict(np.load(source/row['split']/f"{row['case']:03d}_0.npz"))
            stored.pop('target')
            xt=stored['noisy_latent']
            for step in range(8):
                stored.update(noisy_latent=xt,current_step=np.array([step],np.float32))
                xt=vector.run(None,{k:v for k,v in stored.items() if k in required})[0]
            target=np.load(row['file'])['latent']
            wave,sr=sf.read(entry['audio'],dtype='float32')
            assert sr==44100
            cache[key]=(torch.from_numpy(xt).to('mps'),torch.from_numpy(target).to('mps'),
                spectra(torch.from_numpy(wave)),len(wave))
        return cache[key]
    def loss_for(row,audio):
        inputs,target,reference,samples=sample(row)
        output=model(inputs,torch.tensor([target.shape[-1]],dtype=torch.int64))
        latent=(output-target).square().mean()/target.square().mean().clamp_min(1e-5)
        if not audio:return latent,latent,None
        prediction=spectra(decoder({'latent':output})[0].reshape(-1)[:samples])
        spectral=sum((x-y).norm()/y.norm().clamp_min(1e-5)+
            (torch.log(x+1e-4)-torch.log(y+1e-4)).abs().mean() for x,y in zip(prediction,reference))/len(reference)
        return spectral.to('mps')+a.latent_weight*latent,latent,spectral
    @torch.no_grad()
    def validate():
        selected=[]
        for language in ['ko','en']:selected.extend([r for r in val if r['language']==language][:2])
        values=[loss_for(row,True) for row in selected]
        return {'latent':float(np.mean([float(v[1].cpu()) for v in values])),
            'audio':float(np.mean([float(v[2].cpu()) for v in values]))}
    report={'config':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        'teacher_origin':metadata['teacher_origin'],'method':__doc__,
        'input_only_student_vector_sha256':digest,'teacher_is_student':False,
        'frozen_denoising_steps':8,'head_parameters':sum(p.numel() for p in model.parameters()),
        'train_cases':len(train),'validation_cases':len(val),'pilot_fit':a.pilot_fit,
        'note':'Student-generated normal latents are inputs only. Supervision is exclusively original50-normal-derived R3/inverted targets. Head operates before waveform decoder, not on audio. Full normal-length8-step denoising costs more than native shortened2x; measure separately.',
        'history':[],'complete':False}
    a.output.mkdir(parents=True)
    start=time.perf_counter()
    initial=validate();report['initial']=initial;best=initial['audio']
    def save(name,step,value):
        torch.save({'state':model.state_dict(),'step':step,'validation':value,'report':report},a.output/(name+'.pt'))
    save('best',0,initial)
    print('initial',initial,flush=True)
    rng=np.random.default_rng(20261029)
    for step in range(1,a.iterations+1):
        language=['ko','en'][int(rng.integers(2))]
        choices=[r for r in train if r['language']==language]
        row=choices[int(rng.integers(len(choices)))]
        optimizer.zero_grad(set_to_none=True)
        audio=(a.initial is not None or step>250) and step%a.audio_every==0
        loss,latent,spectral=loss_for(row,audio)
        if not torch.isfinite(loss).item():raise RuntimeError('Nonfinite head loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
        if step%50==0:print('step',step,'loss',float(loss.detach()),'seconds',round(time.perf_counter()-start,2),flush=True)
        if step%a.validate_every==0 or step==a.iterations:
            value=validate()
            report['history'].append({'step':step,**value,'seconds':time.perf_counter()-start})
            save('last',step,value);save(f'step_{step:05d}',step,value)
            if value['audio']<best:best=value['audio'];save('best',step,value)
            report['complete']=step==a.iterations
            (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            print('validation',step,value,flush=True)


if __name__=='__main__':main()
