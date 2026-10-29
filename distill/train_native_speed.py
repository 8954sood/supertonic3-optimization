"""Eight-step native speed training against augmented original50 waveforms.

F1200 is initialization only. Target audio is always original FP32 CFG4 50
normal speech or its explicitly labelled offline training-only transform.
No waveform processor is present in the exported inference graph.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runtime  # sets CPU thread limits
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('distill/fast-original50-round8'))
    p.add_argument('--initial', default='distill/joint-original50-round5/best.pt')
    p.add_argument('--graph', default='models/cfg-student-fp32-source/onnx/vector_estimator.onnx')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--iterations', type=int, default=160)
    p.add_argument('--validate-every', type=int, default=40)
    p.add_argument('--learning-rate', type=float, default=1e-5)
    p.add_argument('--reference-audit', type=Path, required=True)
    p.add_argument('--attention-data',type=Path)
    p.add_argument('--attention-weight',type=float,default=0.5)
    p.add_argument('--balanced-languages',action='store_true')
    p.add_argument('--fast-updates-per-normal',type=int,default=3)
    a = p.parse_args()
    if a.output.exists() or min(a.iterations,a.validate_every,a.fast_updates_per_normal) < 1:
        p.error('Positive counts and new output required')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261029)
    metadata = json.loads((a.data/'manifest.json').read_text())
    assert_original_data(metadata)
    if metadata['teacher_origin']['steps'] != 50 or metadata['target_speed'] != 2.0:
        raise ValueError('Original50 fast reference required')
    source = Path(metadata['source'])
    original = json.loads((source/'manifest.json').read_text())
    assert_original_data(original)
    audit = json.loads(a.reference_audit.read_text())
    if len(audit['rows']) != len(metadata['rows']):
        raise ValueError('Complete ASR audit of fast references required before training')
    checked = {row['audio']:row for row in audit['rows']}
    if len(checked) != len(metadata['rows']):
        raise ValueError('Duplicate or missing reference audit')
    for row in metadata['rows']:
        recognized = checked[row['audio']]
        if recognized['reference'] != row['text']:
            raise ValueError('Reference audit text mismatch')
    accepted = [row for row in metadata['rows'] if checked[row['audio']]['errors']==0]
    model = VectorGraph(a.graph, checkpoints=a.initial)
    model.train_full_vector()
    attention_rows = {}
    if a.attention_data:
        attention_metadata = json.loads((a.attention_data/'manifest.json').read_text())
        assert_original_data(attention_metadata)
        if attention_metadata['steps']!=50 or not attention_metadata['complete']:
            raise ValueError('Complete original50 attention reference required')
        attention_rows = {(r['split'],r['case']):r for r in attention_metadata['rows']}
        names = [outputs[0] for name,op,inputs,outputs,attrs in model.nodes
                 if op=='Softmax' and '/attn/' in name]
        assert len(names)==4
        model.outputs += names
    decoder = VectorGraph('models/supertonic-3/onnx/vocoder.onnx')
    optimizer = torch.optim.AdamW(model.trainable.parameters(), lr=a.learning_rate, weight_decay=1e-5)
    cache = {}
    windows = {n: torch.hann_window(n) for n in [512,1024,2048]}

    def spectra(wave):
        wave = wave.reshape(-1).to('cpu')
        return [torch.stft(wave, n_fft=n, hop_length=n//4, window=windows[n],
                           return_complex=True).abs() for n in windows]

    def example(row, fast):
        key = (row['split'], row['case'], fast)
        if key not in cache:
            if fast:
                feed = dict(np.load(row['file']))
                path = Path(row['audio'])
                expected = row['sha256']
                latent = None
            else:
                feed = dict(np.load(source/row['split']/f"{row['case']:03d}_0.npz"))
                feed.pop('target')
                latent = torch.from_numpy(np.load(source/row['split']/f"{row['case']:03d}_7.npz")['target']).to('mps')
                path = Path(row['reference_audio'])
                expected = row['normal_sha256']
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError('Reference hash mismatch')
            wave, sr = sf.read(path, dtype='float32')
            if sr != 44100:
                raise ValueError('Unexpected target sample rate')
            feed = {k: torch.from_numpy(v).to('mps') for k,v in feed.items()}
            feed['speed'] = torch.tensor([2.0 if fast else 1.05],device='mps')
            attention = None
            if fast and attention_rows:
                record = attention_rows[(row['split'],row['case'])]
                path = Path(record['file'])
                if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:
                    raise ValueError('Attention reference hash mismatch')
                stored = np.load(path)
                attention = []
                for i in range(4):
                    value = torch.from_numpy(stored[f'attention_{i}']).to('mps')
                    if value.shape[:2]!=(8,1):
                        raise ValueError('Attention head/batch layout mismatch')
                    value = torch.nn.functional.interpolate(value,
                        size=(feed['noisy_latent'].shape[-1],feed['text_emb'].shape[-1]),
                        mode='bilinear',align_corners=False)
                    attention.append(value/value.sum(dim=-1,keepdim=True).clamp_min(1e-8))
            cache[key] = feed, latent, spectra(torch.from_numpy(wave)), len(wave), attention
        return cache[key]

    def loss_for(row, fast):
        stored, target_latent, target_spectra, samples, target_attention = example(row, fast)
        feed = dict(stored)
        xt = feed['noisy_latent']
        attention_losses = []
        for step in range(8):
            feed['current_step'] = np.array([step],np.float32)
            feed['noisy_latent'] = xt
            values = model(feed)
            xt = values[0]
            if target_attention and step>=4:
                for actual,target in zip(values[1:],target_attention):
                    if actual.shape!=target.shape:
                        raise ValueError('Student/teacher attention shape mismatch')
                    attention_losses.append((target*(torch.log(target.clamp_min(1e-8))-
                        torch.log(actual.clamp_min(1e-8)))).sum(dim=-1).mean())
        predicted = spectra(decoder({'latent':xt})[0].reshape(-1)[:samples])
        terms = []
        for value, target in zip(predicted,target_spectra):
            convergence = (value-target).norm()/target.norm().clamp_min(1e-5)
            logarithmic = (torch.log(value+1e-4)-torch.log(target+1e-4)).abs().mean()
            # Include an envelope-onset term to penalize smeared consonants.
            log_value, log_target = torch.log1p(value),torch.log1p(target)
            onset = (torch.diff(log_value,dim=-1)-torch.diff(log_target,dim=-1)).abs().mean()
            terms.append(convergence+logarithmic+0.2*onset)
        audio = sum(terms)/len(terms)
        latent = ((xt-target_latent).square().mean()/target_latent.square().mean().clamp_min(1e-5)
                  if target_latent is not None else xt.new_zeros(()))
        attention_loss = sum(attention_losses)/len(attention_losses) if attention_losses else xt.new_zeros(())
        return audio.to('mps')+0.25*latent+a.attention_weight*attention_loss, audio, latent

    train = [r for r in accepted if r['split']=='train']
    validation = []
    for language in ['ko','en']:
        selected = []
        for row in accepted:
            if row['split']=='validation' and row['language']==language and row['voice'] not in [v['voice'] for v in selected]:
                selected.append(row)
            if len(selected)==2:
                break
        if len(selected)<2:
            raise ValueError('At least two accepted validation references per language required')
        validation.extend(selected)
    @torch.no_grad()
    def validate():
        means = {}
        for fast in [True,False]:
            values = [float(loss_for(row,fast)[1].cpu()) for row in validation]
            means['fast_audio' if fast else 'normal_audio'] = float(np.mean(values))
        means['selection'] = means['fast_audio']+0.3*means['normal_audio']
        return means

    a.output.mkdir(parents=True)
    report = {'config': {k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        'teacher_origin':metadata['teacher_origin'], 'steps':8,
        'method':f'Full eight-step native rollout, frozen original decoder, MRSTFT/onset audio loss. {a.fast_updates_per_normal} fast updates then1 original normal-rate update, normal latent anchor. Optional original50 conditional text-attention KL on last4 student steps. No student-generated targets. No waveform postprocessing in inference.',
        'train_cases':len(train), 'validation_cases':[r['case'] for r in validation],
        'reference_filter':'Only zero character/word-error fast references from the completed development ASR audit; ASR is a proxy, not a perceptual quality guarantee.',
        'rejected_fast_reference_cases':len(metadata['rows'])-len(accepted),
        'reference_audit_sha256':hashlib.sha256(a.reference_audit.read_bytes()).hexdigest(),
        'history':[], 'complete':False}
    start = time.perf_counter()
    initial = validate()
    report['initial'] = initial
    best = initial['selection']
    def save(name, step, value):
        payload = {'weights':model.export_weights(), 'step':step,
                   'validation_relative_update_mse':None, 'validation_rollout':value,
                   'config':report['config']}
        temp = a.output/(name+'.tmp')
        torch.save(payload,temp)
        temp.replace(a.output/(name+'.pt'))
    save('best',0,initial)
    print('initial',initial,'seconds',time.perf_counter()-start,flush=True)
    rng = np.random.default_rng(20261029)
    for step in range(1,a.iterations+1):
        optimizer.zero_grad(set_to_none=True)
        choices = train
        if a.balanced_languages:
            language = ['ko','en'][int(rng.integers(2))]
            choices = [r for r in train if r['language']==language]
        row = choices[int(rng.integers(len(choices)))]
        fast = step%(a.fast_updates_per_normal+1) != 0
        loss,audio,latent = loss_for(row,fast)
        if not torch.isfinite(loss).item():
            raise RuntimeError('Nonfinite training loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable.parameters(),1.0)
        optimizer.step()
        if step%5==0:
            print('step',step,'fast',fast,'audio',float(audio.detach()),'seconds',round(time.perf_counter()-start,2),flush=True)
        if step%a.validate_every==0 or step==a.iterations:
            value = validate()
            report['history'].append({'step':step,**value,'seconds':time.perf_counter()-start})
            save('last',step,value)
            save(f'step_{step:05d}',step,value)
            if value['selection'] < best:
                best = value['selection']
                save('best',step,value)
            report['complete'] = step==a.iterations
            report['elapsed_seconds'] = time.perf_counter()-start
            (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            print('validation',step,value,flush=True)


if __name__ == '__main__':
    main()
