"""Diagnostic only: per-text original50 attention injected into eight-step student.

This is an oracle control, not a deployable candidate: it requires the original
teacher at generation time. No claim of inference acceleration or fixed quality.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import runtime
from runtime import Engine
import json
import numpy as np
import onnxruntime as ort
import torch
import soundfile as sf
from benchmark import TEXTS
from distill.torch_graph import VectorGraph
from supertonic.loader import load_voice_style_from_json_file


def main():
    torch.set_num_threads(1)
    original_options=runtime.options
    def teacher_options(*args,**kwargs):
        value=original_options(*args,**kwargs)
        value.intra_op_num_threads=4
        return value
    runtime.options=teacher_options
    teacher=Engine()
    vector=ort.InferenceSession('distill/original50-attention-round8-v2/teacher_attention.onnx',
        teacher_options(),providers=['CPUExecutionProvider'])
    class Capture:
        def get_inputs(self): return vector.get_inputs()
        def run(self,*args,**kwargs):
            values=vector.run(*args,**kwargs)
            self.attention=[v[:,:1].copy() for v in values[1:]]
            return values[:1]
    capture=Capture()
    teacher.sessions['vector_estimator']=capture
    runtime.options=original_options
    student=Engine(model='models/round5-original50-student')
    inputs=student.sessions['vector_estimator'].get_inputs()
    model=VectorGraph('models/cfg-student-fp32-source/onnx/vector_estimator.onnx',
                      checkpoints='distill/joint-original50-round5/best.pt')
    original_operation=model.operation
    class Student:
        inject=False
        attention=None
        def get_inputs(self):return inputs
        def run(self,outputs,feed):
            self.index=0
            with torch.no_grad():
                return [model(feed)[0].cpu().numpy()]
        def operation(self,op,x,attrs,nout):
            if self.inject and op=='Softmax' and x[0].ndim==4 and x[0].shape[0]==8:
                value=self.attention[self.index]
                self.index+=1
                return value
            return original_operation(op,x,attrs,nout)
    bridge=Student()
    model.operation=bridge.operation
    student.sessions['vector_estimator']=bridge
    folder=Path('samples/round8_oracle_attention')
    folder.mkdir(parents=True,exist_ok=True)
    report={'note':__doc__,'steps':8,'teacher_steps':50,'rows':[],'complete':False}
    path=Path('results/round8_oracle_attention.json')
    for voice in ['M1','F1']:
        for engine in [teacher,student]:
            root=engine.root
            engine.style=load_voice_style_from_json_file(root/'voice_styles'/f'{voice}.json')
        for case in ['ko_short','en_short']:
            language,text=TEXTS[case]
            teacher.synthesize(text,language,50,seed=42,speed=1.0)
            # Native2 uses ceil(duration/2/chunk), exactly the student's mask.
            ids,mask=student.processor([text],language)
            duration=student.sessions['duration_predictor'].run(None,
                {'text_ids':ids,'text_mask':mask,'style_dp':student.style.dp})[0][0]
            chunk=student.cfg['ae']['base_chunk_size']*student.cfg['ttl']['chunk_compress_factor']
            length=int((duration/2*student.sample_rate+chunk-1)/chunk)
            bridge.attention=[torch.nn.functional.interpolate(torch.from_numpy(v).to('mps'),
                size=(length,v.shape[-1]),mode='bilinear',align_corners=False) for v in capture.attention]
            for inject in [False,True]:
                bridge.inject=inject
                wave,metrics=student.synthesize(text,language,8,seed=42,speed=2.0)
                mode='oracle' if inject else 'control'
                output=folder/f'{voice}_{case}_{mode}.wav'
                sf.write(output,wave,student.sample_rate,subtype='FLOAT')
                report['rows'].append({'variant':mode,'voice':voice,'case':case,
                    'language':language,'text':text,'audio':str(output),'metrics':metrics})
                path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
            print(voice,case,flush=True)
    report['complete']=True
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
