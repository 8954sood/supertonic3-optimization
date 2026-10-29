"""Native eight-step candidate listening/ASR development manifest, no DSP."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import hashlib
import json
import soundfile as sf
from runtime import Engine
from benchmark import TEXTS
from supertonic.loader import load_voice_style_from_json_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cases', nargs='+', default=['ko_short','en_short'])
    p.add_argument('--speeds', nargs='+', type=float, default=[1.05,2.0])
    p.add_argument('--examples',type=Path)
    p.add_argument('--generation-speed',type=float)
    a = p.parse_args()
    if a.output.exists():
        p.error('Choose new output')
    engine = Engine(model=a.model)
    if a.generation_speed is not None:
        if engine.speed_strategy not in ['latent-retime','learned-latent-head'] or not 1.0<=a.generation_speed<=1.5:
            p.error('Generation override requires retiming model and range1.0..1.5')
        engine.latent_head_generation_speed=a.generation_speed
    folder = Path('samples')/a.output.stem
    folder.mkdir(parents=True,exist_ok=True)
    report = {'model':str(a.model),'steps':8,'threads':1,'seed':42,
              'note':'Local native eight-step development evaluation, no waveform filtering. Consumed benchmarks or explicitly supplied example contexts, not independent testing or server timing.',
              'example_source':str(a.examples) if a.examples else 'benchmark.TEXTS',
              'generation_speed_override':a.generation_speed,
              'rows':[],'complete':False}
    examples = (json.loads(a.examples.read_text())['rows'] if a.examples else
        [dict(voice=voice,case=case,language=TEXTS[case][0],text=TEXTS[case][1])
         for voice in ['M1','F1'] for case in a.cases])
    for example in examples:
        voice,case,language,text = [example[k] for k in ['voice','case','language','text']]
        engine.style = load_voice_style_from_json_file(a.model/'voice_styles'/f'{voice}.json')
        for speed in a.speeds:
            wave, metrics = engine.synthesize(text,language,8,seed=42,speed=speed)
            split_prefix=(str(example['split'])+'_') if example.get('split') else ''
            path = folder/f'{split_prefix}{voice}_{case}_speed_{speed:g}.wav'
            sf.write(path,wave,engine.sample_rate,subtype='FLOAT')
            report['rows'].append({'variant':a.model.name+f'-speed-{speed:g}',
                'voice':voice,'case':case,'language':language,'text':text,'speed':speed,
                'audio':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'metrics':metrics})
            a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
            print(voice,case,speed,flush=True)
    report['complete']=True
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    main()
