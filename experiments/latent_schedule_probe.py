"""Eight-step internal length-transition controls; no output waveform filtering."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import time
import numpy as np
import soundfile as sf
from runtime import Engine
from benchmark import TEXTS
from supertonic.loader import load_voice_style_from_json_file


def resize_latent(value,length):
    b,c,n=value.shape
    fine=value.reshape(b,24,6,n).transpose(0,1,3,2).reshape(b,24,n*6)
    coords=(np.arange(length*6,dtype=np.float64)+.5)*(n/length)-.5
    out=np.stack([np.interp(coords,np.arange(n*6),row) for row in fine.reshape(-1,n*6)]).astype(np.float32)
    return out.reshape(b,24,length,6).transpose(0,1,3,2).reshape(b,144,length)


def main():
    engine=Engine(model='models/round5-original50-student')
    vector=engine.sessions['vector_estimator']
    vocoder=engine.sessions['vocoder']
    class Vector:
        switch=0
        correct=True
        count=0
        def get_inputs(self):return vector.get_inputs()
        def run(self,outputs,feed):
            current=int(feed['current_step'][0])
            feed=dict(feed)
            if current==0:
                self.noise=feed['noisy_latent'].copy()
                self.count=0
            if current==self.switch:
                xt=feed['noisy_latent']
                t=current/8
                if self.correct:
                    xt=resize_latent(xt-(1-t)*self.noise,self.length)+(1-t)*self.fast_noise
                else:xt=resize_latent(xt,self.length)
                feed['noisy_latent']=xt
            if current>=self.switch:
                feed['latent_mask']=np.ones((1,1,self.length),np.float32)
            self.count+=1
            return vector.run(outputs,feed)
    bridge=Vector()
    class Vocoder:
        def get_inputs(self):return vocoder.get_inputs()
        def run(self,outputs,feed):
            if bridge.switch==8:feed={'latent':resize_latent(feed['latent'],bridge.length)}
            return vocoder.run(outputs,feed)
    engine.sessions['vector_estimator']=bridge
    engine.sessions['vocoder']=Vocoder()
    folder=Path('samples/round8_latent_schedule');folder.mkdir(exist_ok=True)
    output=Path('results/round8_latent_schedule.json')
    report={'note':__doc__+' Local CPU1 functional/quality development only, not server timing. Full-length first k steps, shortened remaining8-k. Known initial noise subtraction/replacement is an approximate rectified-flow state transform, not an exact numerical optimization.',
        'actual_steps':8,'rows':[],'complete':False}
    for voice in ['M1','F1']:
        engine.style=load_voice_style_from_json_file(engine.root/'voice_styles'/f'{voice}.json')
        for case in ['ko_short','en_short']:
            language,text=TEXTS[case]
            ids,mask=engine.processor([text],language)
            duration=engine.sessions['duration_predictor'].run(None,{'text_ids':ids,'text_mask':mask,'style_dp':engine.style.dp})[0][0]
            chunk=engine.cfg['ae']['base_chunk_size']*engine.cfg['ttl']['chunk_compress_factor']
            bridge.length=int((duration/2*engine.sample_rate+chunk-1)/chunk)
            bridge.fast_noise=np.random.RandomState(42).randn(1,144,bridge.length).astype(np.float32)
            for switch,correct in [(0,True),(2,True),(4,True),(6,True),(8,True),(2,False),(4,False)]:
                bridge.switch=switch;bridge.correct=correct
                start=time.perf_counter()
                wave,metrics=engine.synthesize(text,language,8,seed=42,speed=1.05)
                wave=wave[:int(float(duration/2)*engine.sample_rate)]
                assert bridge.count==8
                name=f'switch{switch}_'+('noise' if correct else 'resize')
                audio=folder/f'{voice}_{case}_{name}.wav';sf.write(audio,wave,engine.sample_rate,subtype='FLOAT')
                report['rows'].append({'variant':name,'case':case,'voice':voice,'language':language,'text':text,
                    'audio':str(audio),'steps':bridge.count,'speed':2.0,'switch_step':switch,
                    'known_noise_correction':correct,'audio_seconds':len(wave)/engine.sample_rate,
                    'local_seconds':time.perf_counter()-start})
                output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
                print(voice,case,name,flush=True)
    report['complete']=True
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
