"""Merge learned latent-rate head into frozen original waveform decoder ONNX."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import hashlib
import json
import shutil
import numpy as np
import torch
import onnx
from onnx import compose,version_converter
import onnxruntime as ort
from runtime import options
from distill.latent_rate_head import LatentRateHead


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--base',type=Path,default=Path('models/round5-original50-student'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--residual-scale',type=float,default=1.0,
        help='Scale the learned residual only; 0 keeps the interpolation control')
    a=p.parse_args()
    if a.output.exists():p.error('Choose new output')
    if not 0 <= a.residual_scale <= 1:p.error('Residual scale must be in [0,1]')
    torch.set_num_threads(1)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    head=LatentRateHead().eval();head.load_state_dict(saved['state'])
    with torch.no_grad():
        head.output.weight.mul_(a.residual_scale)
        head.output.bias.mul_(a.residual_scale)
    shutil.copytree(a.base,a.output)
    temporary=a.output/'head_export.onnx'
    torch.onnx.export(head,(torch.zeros(1,144,60),torch.tensor([32],dtype=torch.int64)),str(temporary),
        dynamo=False,opset_version=18,input_names=['latent','target_frames'],output_names=['fast_latent'],
        dynamic_axes={'latent':{0:'batch',2:'source_frames'},'fast_latent':{0:'batch',2:'output_frames'}})
    session=ort.InferenceSession(str(temporary),options(),providers=['CPUExecutionProvider'])
    parity=[]
    with torch.no_grad():
        for frames,target in [(60,32),(37,19),(81,43),(37,37),(60,60)]:
            value=np.random.RandomState(frames).randn(1,144,frames).astype(np.float32)*.2
            count=np.array([target],np.int64)
            expected=head(torch.from_numpy(value),torch.from_numpy(count)).numpy()
            actual=session.run(None,{'latent':value,'target_frames':count})[0]
            error=float(np.max(np.abs(expected-actual)))
            if actual.shape!=(1,144,target) or error>1e-4:raise ValueError('Dynamic head export mismatch')
            identity=bool(np.array_equal(value,actual)) if frames==target else None
            if frames==target and not identity:raise ValueError('Normal identity route changed')
            parity.append({'frames':frames,'target':target,'max_error':error,'normal_bitidentity':identity})
    h=onnx.load(temporary)
    decoder=onnx.load(a.base/'onnx/vocoder.onnx')
    version=max([v.version for m in [h,decoder] for v in m.opset_import if v.domain==''])
    def upgrade(m):
        current=next(v.version for v in m.opset_import if v.domain=='')
        return version_converter.convert_version(m,version) if current!=version else m
    h,decoder=upgrade(h),upgrade(decoder)
    # Model IR must match when merging. Both graphs use basic tensor operators.
    h.ir_version=decoder.ir_version=max(h.ir_version,decoder.ir_version)
    decoder_input=decoder.graph.input[0].name
    h=compose.add_prefix(h,'rate/')
    decoder=compose.add_prefix(decoder,'decoder/')
    merged=compose.merge_models(h,decoder,[('rate/fast_latent','decoder/'+decoder_input)])
    mapping={'rate/latent':'latent','rate/target_frames':'target_frames'}
    for node in merged.graph.node:
        for i,name in enumerate(node.input):node.input[i]=mapping.get(name,name)
        for i,name in enumerate(node.output):node.output[i]=mapping.get(name,name)
    for value in list(merged.graph.input)+list(merged.graph.output):value.name=mapping.get(value.name,value.name)
    onnx.checker.check_model(merged)
    onnx.save(merged,a.output/'onnx/vocoder.onnx')
    temporary.unlink()
    manifest=json.loads((a.base/'manifest.json').read_text())
    manifest.update(status='experimental learned latent-rate head; quality/CPU1 validation required',
        speed_strategy='learned-latent-head',latent_head_generation_speed=1.05,
        latent_head_supported_speeds=[2.0],latent_head_parameters=sum(p.numel() for p in head.parameters()),
        latent_head_checkpoint=str(a.checkpoint),latent_head_step=saved['step'],
        latent_head_residual_scale=a.residual_scale,
        latent_head_validation=saved['validation'],latent_head_export_parity=parity,
        latent_head_teacher=saved['report']['teacher_origin'],
        latent_head_note='Frozen F1200 normal-length8-step denoising then learned latent compression inside vocoder; no waveform processor. Normal requested frame count uses exact identity head route. Fast2 costs full-length denoising; measure RTF honestly.')
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    assert hashlib.sha256((a.output/'onnx/vector_estimator.onnx').read_bytes()).hexdigest()==hashlib.sha256((a.base/'onnx/vector_estimator.onnx').read_bytes()).hexdigest()
    print(a.output,parity,flush=True)


if __name__=='__main__':main()
