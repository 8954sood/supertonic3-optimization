"""Check unchanged-model engine output at float32 sample precision."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-dir', type=Path, required=True)
    p.add_argument('--candidate-dir', type=Path, required=True)
    p.add_argument('--voice', default='M1')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    files = sorted(a.reference_dir.glob('*.wav'))
    if not files:
        p.error('No reference WAV files')
    rows = []
    for source in files:
        reference, sample_rate = sf.read(source, dtype='float32')
        candidate, other_rate = sf.read(a.candidate_dir / source.name, dtype='float32')
        if sample_rate != other_rate or reference.shape != candidate.shape:
            raise ValueError(f'Waveform shape/sample-rate mismatch: {source.name}')
        difference = reference.astype(np.float64) - candidate.astype(np.float64)
        rows.append({
            'case': source.stem, 'voice': a.voice, 'sample_rate': sample_rate,
            'samples': len(reference),
            'float32_bit_identical': bool(np.array_equal(reference, candidate)),
            'rmse': float(np.sqrt(np.mean(difference ** 2))),
            'max_absolute_difference': float(np.max(abs(difference))),
            'reference_float_sha256': hashlib.sha256(reference.tobytes()).hexdigest(),
            'candidate_float_sha256': hashlib.sha256(candidate.tobytes()).hexdigest(),
        })
    report = {
        'note': 'Fixed-model engine numerical regression. No original-teacher quality, independent ASR or perceptual similarity claim.',
        'rows': rows, 'all_float32_bit_identical': all(r['float32_bit_identical'] for r in rows),
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
    if not report['all_float32_bit_identical']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
