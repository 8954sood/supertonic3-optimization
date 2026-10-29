"""Functional speaking-rate sweep; timings include cold shapes, not benchmark medians."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import Engine
import argparse
import json
import numpy as np
import psutil
import soundfile as sf
from benchmark import TEXTS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--speeds', nargs='+', type=float, default=[0.7, 1.0, 1.05, 1.5, 2.0])
    p.add_argument('--voice', default='M1')
    p.add_argument('--affinity', type=int)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if 1.0 not in a.speeds or any(not 0.7 <= v <= 2.0 for v in a.speeds):
        p.error('Speeds must include 1.0 and all be within 0.7..2.0')
    if a.affinity is not None:
        psutil.Process().cpu_affinity([a.affinity])
    engine = Engine(model=a.model, voice=a.voice)
    samples = Path('samples') / a.output.stem
    samples.mkdir(parents=True, exist_ok=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        'model': a.model, 'voice': a.voice, 'steps': 8, 'threads': 1,
        'note': 'Development speaking-rate functionality. One synthesis per shape, no timing warm-up. Not a new independent ASR test or comparable warm RTF benchmark.',
        'rows': [], 'duration_checks': [], 'invalid_speed_checks': [], 'complete': False,
    }
    for case in ['ko_short', 'en_short']:
        lang, text = TEXTS[case]
        baseline_samples = None
        for speed in a.speeds:
            wav, metrics = engine.synthesize(text, lang, 8, seed=42, speed=speed)
            name = f'{case}_speed_{speed:g}.wav'
            path = samples / name
            sf.write(path, wav, engine.sample_rate, subtype='FLOAT')
            report['rows'].append({
                'variant': f'{Path(a.model).name}-speed-{speed:g}',
                'case': case, 'language': lang, 'voice': a.voice,
                'text': text, 'speed': speed, 'audio': str(path),
                'sample_rate': engine.sample_rate, 'samples': len(wav),
                'metrics': metrics, 'finite': bool(np.isfinite(wav).all()),
            })
            if speed == 1.0:
                baseline_samples = len(wav)
            a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
            print(case, a.voice, speed, len(wav) / engine.sample_rate, flush=True)
        for row in [v for v in report['rows'] if v['case'] == case]:
            error_samples = abs(row['samples'] - baseline_samples / row['speed'])
            report['duration_checks'].append({
                'case': case, 'speed': row['speed'], 'error_samples': error_samples,
                'inverse_duration_within_3_samples': error_samples <= 3,
            })
    for label, speed in [('below', 0.69), ('above', 2.01), ('nan', float('nan')), ('inf', float('inf'))]:
        rejected = False
        try:
            engine.synthesize('check', 'en', 8, speed=speed)
        except ValueError:
            rejected = True
        report['invalid_speed_checks'].append({'label': label, 'rejected': rejected})
    report['complete'] = True
    report['passed'] = (
        all(r['inverse_duration_within_3_samples'] for r in report['duration_checks'])
        and all(r['rejected'] for r in report['invalid_speed_checks'])
    )
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    if not report['passed']:
        raise SystemExit('Speaking-rate functional check failed')


if __name__ == '__main__':
    main()
