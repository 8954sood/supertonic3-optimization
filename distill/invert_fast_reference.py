"""Recover compact fast latent targets through the frozen original decoder.

Only original50 normal latents/audio and their training-only R3 transform are
used. No student target, no trained encoder, no inference waveform processing.
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
import soundfile as sf
import torch
from distill.torch_graph import VectorGraph
from distill.teacher_policy import assert_original_data


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('distill/fast-original50-round8'))
    p.add_argument('--cases',nargs='+',default=['train:0','train:41','train:30','train:21'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--iterations',type=int,default=300)
    p.add_argument('--reuse',type=Path)
    a=p.parse_args()
    if a.output.exists():p.error('Choose new output')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    metadata=json.loads((a.data/'manifest.json').read_text())
    assert_original_data(metadata)
    source=Path(metadata['source'])
    selected={(s,int(c)) for s,c in [value.split(':') for value in a.cases]}
    rows=[r for r in metadata['rows'] if (r['split'],r['case']) in selected]
    assert len(rows)==len(selected)
    audit=json.loads(Path('results/round8_training_references_asr.json').read_text())
    checked={r['audio']:r for r in audit['rows']}
    assert all(checked[r['audio']]['errors']==0 for r in rows)
    decoder=VectorGraph('models/supertonic-3/onnx/vocoder.onnx')
    windows={n:torch.hann_window(n) for n in [256,512,1024,2048]}
    def spectra(wave):
        wave=wave.reshape(-1).to('cpu')
        return [torch.stft(wave,n_fft=n,hop_length=n//4,window=windows[n],return_complex=True).abs() for n in windows]
    a.output.mkdir(parents=True)
    folder=Path('samples')/a.output.name
    folder.mkdir(parents=True,exist_ok=True)
    report={'teacher_origin':metadata['teacher_origin'],'method':__doc__,
        'iterations':a.iterations,'rows':[],'complete':False}
    reusable={}
    if a.reuse:
        previous=json.loads(a.reuse.read_text())
        assert previous['complete'] and previous['iterations']==a.iterations
        reusable={(r['split'],r['case']):r for r in previous['rows']}
    for row in rows:
        old=reusable.get((row['split'],row['case']))
        if old:
            assert old['reference_audio']==row['audio']
            assert hashlib.sha256(Path(old['audio']).read_bytes()).hexdigest()==old['sha256']
            report['rows'].append(old)
            (a.output/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
            print('reused',row['split'],row['case'],flush=True)
            continue
        base=torch.from_numpy(np.load(source/row['split']/f"{row['case']:03d}_7.npz")['target']).to('mps')
        first=dict(np.load(row['file']))
        length=first['noisy_latent'].shape[-1]
        b,channels,original_length=base.shape
        fine=base.reshape(b,channels//6,6,original_length).permute(0,1,3,2).reshape(b,channels//6,original_length*6)
        fine=torch.nn.functional.interpolate(fine,size=length*6,mode='linear',align_corners=False)
        initial=fine.reshape(b,channels//6,length,6).permute(0,1,3,2).reshape(b,channels,length)
        latent=torch.nn.Parameter(initial.clone())
        optimizer=torch.optim.Adam([latent],lr=0.01)
        target,sr=sf.read(row['audio'],dtype='float32')
        assert hashlib.sha256(Path(row['audio']).read_bytes()).hexdigest()==row['sha256']
        target_spectra=spectra(torch.from_numpy(target))
        history=[]
        best=float('inf')
        start=time.perf_counter()
        def evaluate():
            wave=decoder({'latent':latent})[0].reshape(-1)[:len(target)]
            prediction=spectra(wave)
            losses=[(x-y).norm()/y.norm().clamp_min(1e-5)+(torch.log(x+1e-4)-torch.log(y+1e-4)).abs().mean()
                    for x,y in zip(prediction,target_spectra)]
            spectral=sum(losses)/len(losses)
            return spectral.to('mps')+0.001*(latent-initial).square().mean(),wave
        for step in range(a.iterations+1):
            optimizer.zero_grad(set_to_none=True)
            loss,wave=evaluate()
            value=float(loss.detach())
            if value<best:
                best=value
                best_latent=latent.detach().clone()
            if step%50==0 or step==a.iterations:
                out=folder/f"{row['split']}_{row['case']:03d}_{step:04d}.wav"
                sf.write(out,wave.detach().cpu().numpy(),sr,subtype='FLOAT')
                history.append({'step':step,'loss':value,'audio':str(out)})
                print(row['split'],row['case'],step,'loss',round(value,5),flush=True)
            if step<a.iterations:
                loss.backward()
                optimizer.step()
        path=a.output/f"{row['split']}_{row['case']:03d}.npz"
        np.savez_compressed(path,latent=best_latent.cpu().numpy())
        best_wave=decoder({'latent':best_latent})[0].reshape(-1)[:len(target)].detach().cpu().numpy()
        out=folder/f"{row['split']}_{row['case']:03d}_best.wav"
        sf.write(out,best_wave,sr,subtype='FLOAT')
        report['rows'].append({**{k:row[k] for k in ['split','case','voice','language','text']},
            'variant':'frozen-original-decoder-inversion','audio':str(out),'reference_audio':row['audio'],
            'file':str(path),'best_loss':best,'history':history,'seconds':time.perf_counter()-start,
            'sha256':hashlib.sha256(out.read_bytes()).hexdigest()})
        (a.output/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    report['complete']=True
    (a.output/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
