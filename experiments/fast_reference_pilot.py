"""Make content/pitch validation references from existing original50 normal audio."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import hashlib
import json
import soundfile as sf
from distill.fast_reference import fast_reference, rubberband_reference


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default='results/round7_original50_speed.json')
    p.add_argument('--output', type=Path, default=Path('results/round8_fast_reference_pilot.json'))
    p.add_argument('--method', choices=['atempo', 'rubberband'], default='atempo')
    a = p.parse_args()
    source = json.loads(Path(a.source).read_text())
    assert source['steps'] == 50 and source['cfg'] == 4 and source['precision'] == 'FP32'
    folder = Path('samples') / a.output.stem
    folder.mkdir(parents=True, exist_ok=True)
    report = {'teacher_origin': source['teacher_origin'], 'source': a.source,
              'note': 'Transformed original50 normal-rate training targets only; not native2 model results. Do not evaluate these as a model speed fix.',
              'rows': [], 'complete': False}
    for row in source['rows']:
        if row['speed'] != 1.0:
            continue
        path = Path(row['audio'])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
        wave, sr = sf.read(path, dtype='float32')
        transform = rubberband_reference if a.method == 'rubberband' else fast_reference
        fast, provenance = transform(wave, sr)
        out = folder / (row['voice'] + '_' + row['case'] + '.wav')
        sf.write(out, fast, sr, subtype='FLOAT')
        report['rows'].append({
            'variant': 'original50-transformed-fast-reference',
            **{k: row[k] for k in ['case', 'voice', 'language', 'text']},
            'audio': str(out), 'reference_audio': str(path),
            'sha256': hashlib.sha256(out.read_bytes()).hexdigest(),
            'audio_seconds': len(fast)/sr, 'transform': provenance,
        })
    report['complete'] = True
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print('Generated', len(report['rows']), 'fast references', flush=True)


if __name__ == '__main__':
    main()
