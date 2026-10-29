"""Find safe internal generation rates for fixed2x parameter-free latent retiming."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import json
import soundfile as sf
from runtime import Engine
from benchmark import TEXTS
from supertonic.loader import load_voice_style_from_json_file


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default='models/round8-latent-retime')
    p.add_argument('--rates',nargs='+',type=float,default=[1.05,1.2,1.35,1.5])
    p.add_argument('--output',type=Path,default=Path('results/round8_retime_rate_sweep.json'))
    a=p.parse_args()
    if a.output.exists():p.error('Choose new output')
    engine=Engine(model=a.model)
    if engine.speed_strategy!='latent-retime':raise ValueError('Retiming model required')
    session=engine.sessions['vector_estimator']
    class Count:
        calls=0
        def get_inputs(self):return session.get_inputs()
        def run(self,*args,**kwargs):
            self.calls+=1
            return session.run(*args,**kwargs)
    counter=Count();engine.sessions['vector_estimator']=counter
    folder=Path('samples')/a.output.stem;folder.mkdir(parents=True,exist_ok=True)
    report={'model':a.model,'output_speed':2.0,'threads':1,'actual_steps':8,
        'note':'Consumed short development texts. Cold-shape local quality sweep, not server timing or independent test. No waveform filtering.',
        'rows':[],'complete':False}
    for voice in ['M1','F1']:
        engine.style=load_voice_style_from_json_file(engine.root/'voice_styles'/f'{voice}.json')
        for case in ['ko_short','en_short']:
            language,text=TEXTS[case]
            for rate in a.rates:
                engine.latent_head_generation_speed=rate
                counter.calls=0
                wave,metrics=engine.synthesize(text,language,8,seed=42,speed=2.0)
                assert counter.calls==8
                path=folder/f'{voice}_{case}_generation{rate:g}.wav'
                sf.write(path,wave,engine.sample_rate,subtype='FLOAT')
                report['rows'].append({'variant':f'latent-retime-generation-{rate:g}',
                    'voice':voice,'case':case,'language':language,'text':text,
                    'audio':str(path),'generation_speed':rate,'speed':2.0,
                    'actual_steps':counter.calls,'metrics':metrics})
                a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
                print(voice,case,rate,flush=True)
    report['complete']=True
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
