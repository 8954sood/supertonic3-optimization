"""Add parameter-free feature retiming inside F1200's original ONNX vocoder.

Eight frozen denoising iterations generate complete phonetic features at a
validated safe rate; the decoder receives a shorter feature sequence. No
waveform DSP, pitch filter, new weights, pruning, or skipped denoising steps.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from torch import nn
import onnx
from onnx import compose,version_converter
import onnxruntime as ort
from runtime import options


class LatentRetime(nn.Module):
    def forward(self,latent,target_frames):
        b,_,n=latent.shape
        target=target_frames[0]
        fine=latent.reshape(b,24,6,n).permute(0,1,3,2).reshape(b,24,n*6)
        # Explicit half-pixel linear interpolation avoids the small coordinate
        # rounding difference between Torch and ORT Resize implementations.
        positions=torch.arange(target*6,device=latent.device,dtype=latent.dtype)
        positions=(positions+0.5)*(n/target)-0.5
        positions=positions.clamp(min=0)
        positions=torch.minimum(positions,torch.ones_like(positions)*(n*6-1))
        lower=torch.floor(positions).to(torch.int64)
        upper=torch.minimum(lower+1,torch.ones_like(lower)*(n*6-1))
        fraction=(positions-lower.to(latent.dtype)).reshape(1,1,-1)
        a=torch.gather(fine,2,lower.reshape(1,1,-1).expand(b,24,-1))
        z=torch.gather(fine,2,upper.reshape(1,1,-1).expand(b,24,-1))
        fine=a*(1-fraction)+z*fraction
        return fine.reshape(b,24,target,6).permute(0,1,3,2).reshape(b,144,target)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,default=Path('models/round5-original50-student'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--generation-speed',type=float,default=1.35,
        help='Internal rate, separate from output2x; default is short-sample listening selection')
    a=p.parse_args()
    if a.output.exists() or not 1.0<=a.generation_speed<=1.5:
        p.error('New output and generation speed1.0..1.5 required')
    digest=hashlib.sha256((a.base/'onnx/vector_estimator.onnx').read_bytes()).hexdigest()
    if digest!='7c81ab7f0ddcc4266743701599a8708d2edd1be04285770f996e84a8bb5f4447':
        p.error('Pinned portable F1200 package required')
    decoder_digest=hashlib.sha256((a.base/'onnx/vocoder.onnx').read_bytes()).hexdigest()
    if decoder_digest!='7698d5858bb47c008c959a1ab94b1904cb69bfb518070bb6b1781211b977c9e9':
        p.error('Pinned F1200 waveform decoder required')
    torch.set_num_threads(1)
    module=LatentRetime().eval()
    working=a.output.with_name(a.output.name+'.staging')
    if working.exists():p.error('Staging output already exists')
    shutil.copytree(a.base,working)
    temporary=working/'retime_export.onnx'
    torch.onnx.export(module,(torch.zeros(1,144,60),torch.tensor([32],dtype=torch.int64)),str(temporary),
        dynamo=False,opset_version=18,input_names=['latent','target_frames'],output_names=['fast_latent'],
        dynamic_axes={'latent':{0:'batch',2:'source_frames'},'fast_latent':{0:'batch',2:'output_frames'}})
    session=ort.InferenceSession(str(temporary),options(),providers=['CPUExecutionProvider'])
    parity=[]
    with torch.no_grad():
        for source,target in [(60,32),(37,19),(81,43),(37,37),(60,60)]:
            value=np.random.RandomState(source).randn(1,144,source).astype(np.float32)
            frames=np.array([target],np.int64)
            expected=module(torch.from_numpy(value),torch.from_numpy(frames)).numpy()
            actual=session.run(None,{'latent':value,'target_frames':frames})[0]
            error=float(np.max(np.abs(expected-actual)))
            identity=bool(np.array_equal(value,actual)) if source==target else None
            if error>1e-5 or (source==target and not identity):raise ValueError('Retiming export mismatch')
            parity.append({'source':source,'target':target,'max_error':error,'normal_identity':identity})
    head=onnx.load(temporary)
    decoder=onnx.load(a.base/'onnx/vocoder.onnx')
    version=max(v.version for m in [head,decoder] for v in m.opset_import if not v.domain)
    def upgrade(m):
        old=next(v.version for v in m.opset_import if not v.domain)
        return version_converter.convert_version(m,version) if old!=version else m
    head,decoder=upgrade(head),upgrade(decoder)
    head.ir_version=decoder.ir_version=max(head.ir_version,decoder.ir_version)
    decoder_input=decoder.graph.input[0].name
    head=compose.add_prefix(head,'retime/')
    decoder=compose.add_prefix(decoder,'decoder/')
    model=compose.merge_models(head,decoder,[('retime/fast_latent','decoder/'+decoder_input)])
    rename={'retime/latent':'latent','retime/target_frames':'target_frames'}
    for node in model.graph.node:
        for i,name in enumerate(node.input):node.input[i]=rename.get(name,name)
    for value in model.graph.input:value.name=rename.get(value.name,value.name)
    onnx.checker.check_model(model)
    onnx.save(model,working/'onnx/vocoder.onnx')
    temporary.unlink()
    manifest=json.loads((a.base/'manifest.json').read_text())
    manifest.update(status='experimental latent-retime, listening and CPU1 validation required',
        speed_strategy='latent-retime',latent_head_generation_speed=a.generation_speed,
        latent_head_supported_speeds=[2.0],latent_retime_extra_parameters=0,
        latent_retime_source_vector_sha256=digest,latent_retime_export_parity=parity,
        latent_retime_source_vocoder_sha256=decoder_digest,
        latent_retime_note='Model-internal fine-frame24-channel feature retiming before frozen original waveform decoder. All learned weights unchanged. Eight denoising calls; fast2 uses longer generation features, so report actual latency/RTF. No waveform processing.')
    (working/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    working.rename(a.output)
    print(a.output,parity,flush=True)


if __name__=='__main__':main()
