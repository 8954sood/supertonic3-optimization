"""Matched single-thread model timing, requiring exact float32 output identity."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runtime import Engine
from benchmark import TEXTS
import argparse
import hashlib
import json
import platform
import statistics
import time
import numpy as np
import psutil


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',required=True)
    p.add_argument('--candidate',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pairs',type=int,default=5)
    p.add_argument('--speeds',type=float,nargs='+',default=[1.0,1.5,2.0])
    p.add_argument('--voices',nargs='+',default=['M1','F1'])
    a=p.parse_args()
    if a.output.exists() or a.pairs<3:p.error('Use a new output and at least three pairs')
    report=dict(machine=platform.platform(),threads=1,steps=8,
        reference=a.reference,candidate=a.candidate,pairs=a.pairs,
        scope='Alternating warm pairs in one process on this Mac, no hardware extrapolation.',
        rows=[],complete=False)
    engines={name:Engine(model=model) for name,model in [('reference',a.reference),('candidate',a.candidate)]}
    for voice in a.voices:
        from supertonic.loader import load_voice_style_from_json_file
        styles={name:load_voice_style_from_json_file(e.root/f'voice_styles/{voice}.json') for name,e in engines.items()}
        for speed in a.speeds:
            for case in ['ko_short','en_short']:
                lang,text=TEXTS[case]
                expected=None
                for name,e in engines.items():
                    wave,_=e.synthesize(text,lang,8,speed=speed,voice_style=styles[name])
                    if expected is None:expected=wave.copy()
                    np.testing.assert_array_equal(expected,wave)
                runs={name:[] for name in engines}
                for pair in range(a.pairs):
                    names=list(engines)
                    if pair%2:names.reverse()
                    for name in names:
                        began=time.process_time()
                        wave,metrics=engines[name].synthesize(text,lang,8,speed=speed,voice_style=styles[name])
                        metrics['process_cpu_seconds']=time.process_time()-began
                        np.testing.assert_array_equal(expected,wave)
                        runs[name].append(metrics)
                speedups=[r['seconds']/c['seconds'] for r,c in zip(runs['reference'],runs['candidate'])]
                row=dict(case=case,voice=voice,speed=speed,float32_bitidentical=True,
                    wave_sha256=hashlib.sha256(expected.tobytes()).hexdigest(),runs=runs,
                    median_paired_speedup=statistics.median(speedups),paired_speedups=speedups,
                    median_time_reduction_percent=statistics.median((1-1/x)*100 for x in speedups),
                    reference_rtf=statistics.median(x['rtf'] for x in runs['reference']),
                    candidate_rtf=statistics.median(x['rtf'] for x in runs['candidate']),
                    process_rss_mb=psutil.Process().memory_info().rss/1e6)
                report['rows'].append(row);a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
                print(voice,speed,case,'speedup',round(row['median_paired_speedup'],4),flush=True)
    report['complete']=True;a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
