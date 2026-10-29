"""Compare two native kernels in separate processes; restore active library on exit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-library', type=Path, required=True)
    p.add_argument('--candidate-library', type=Path, required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--candidate-model', help='Optional graph fusion package; same learned tensors required')
    p.add_argument('--speed', type=float, default=1.05)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--pairs', type=int, default=2)
    a = p.parse_args()
    target = Path('native/libexact_int8.so')
    if a.pairs < 2 or not target.is_file() or a.output.exists() or not 0.7 <= a.speed <= 2.0:
        p.error('Need two pairs, existing active library and a new output')
    for lib in [a.reference_library, a.candidate_library]:
        if not lib.is_file() or lib.resolve() == target.resolve():
            p.error('Versioned input libraries must be separate from active library')
    original = target.read_bytes()
    original_mode = target.stat().st_mode & 0o777
    def install(content):
        temporary = target.with_suffix('.swap.tmp')
        temporary.write_bytes(content)
        temporary.chmod(original_mode)
        os.replace(temporary, target)
    folder = a.output.parent / (a.output.stem + '_runs')
    folder.mkdir(parents=True, exist_ok=False)
    report = {
        'note': 'Sequential fresh-process benchmark, frozen learned tensors; candidate graph may fuse an exact epilogue. Atomic library swaps only while no benchmark child is running; original active library restored in finally. Speed claims require bit-identical outputs.',
        'model': a.model, 'steps': 8, 'threads': 1,
        'speed': a.speed, 'candidate_model': a.candidate_model or a.model,
        'libraries': {label: {'name': lib.name, 'sha256': digest(lib)} for label, lib in [('reference', a.reference_library), ('candidate', a.candidate_library)]},
        'runs': [], 'summary': [], 'complete': False,
    }
    try:
        for pair in range(a.pairs):
            order = [('reference', a.reference_library), ('candidate', a.candidate_library)]
            if pair % 2:
                order.reverse()
            for label, library in order:
                install(library.read_bytes())
                raw = folder / f'{a.output.stem}_{pair:02d}_{label}.json'
                guard = folder / f'{a.output.stem}_{pair:02d}_{label}_guard.json'
                model = a.candidate_model if label == 'candidate' and a.candidate_model else a.model
                command = [sys.executable, 'memory_guard.py', '--max-rss-mb', '480', '--reserve-mb', '80', '--seconds', '360', '--stats', str(guard), '--', sys.executable, 'benchmark.py', '--model', model, '--speed', str(a.speed), '--steps', '8', '--repeats', '1', '--cases', 'ko_short', 'en_short', '--affinity', '0', '--audio-subtype', 'FLOAT', '--output', str(raw)]
                result = subprocess.run(command, check=False)
                row = {'pair': pair, 'label': label, 'returncode': result.returncode, 'guard': json.loads(guard.read_text()) if guard.exists() else None, 'benchmark': json.loads(raw.read_text()) if raw.exists() else None}
                report['runs'].append(row)
                a.output.write_text(json.dumps(report, indent=2) + '\n')
                if result.returncode:
                    raise RuntimeError('Guarded kernel benchmark failed')
        import numpy as np
        import soundfile as sf
        report['wave_checks'] = []
        for pair in range(a.pairs):
            for case in ['ko_short', 'en_short']:
                waves = []
                for label in ['reference', 'candidate']:
                    sample = Path('samples') / f'{a.output.stem}_{pair:02d}_{label}' / f'{case}_8steps.wav'
                    wave, rate = sf.read(sample, dtype='float32')
                    waves.append((wave, rate))
                identical = waves[0][1] == waves[1][1] and np.array_equal(waves[0][0], waves[1][0])
                report['wave_checks'].append({'pair': pair, 'case': case, 'bitidentical': bool(identical)})
                if not identical:
                    raise RuntimeError('Candidate changed generated audio; reject exact engine optimization')
        for case in ['ko_short', 'en_short']:
            times = {label: [next(v['median_seconds'] for v in r['benchmark']['rows'] if v['case'] == case) for r in report['runs'] if r['label'] == label] for label in ['reference', 'candidate']}
            report['summary'].append({'case': case, 'reference_seconds': statistics.median(times['reference']), 'candidate_seconds': statistics.median(times['candidate']), 'paired_speedups': [x/y for x,y in zip(times['reference'], times['candidate'])]})
        report['complete'] = True
    finally:
        install(original)
        report['original_library_restored'] = digest(target) == hashlib.sha256(original).hexdigest()
        a.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['summary']))


if __name__ == '__main__':
    main()
