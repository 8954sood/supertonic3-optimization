"""Summarize raw fixed-model server measurements without hiding failed runs."""
import argparse
import json
from pathlib import Path


def read(name):
    return json.loads((Path('results') / name).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine-version', choices=['v1', 'v2'], default='v1')
    args = parser.parse_args()
    second = args.engine_version == 'v2'
    baseline = read('round6_full_baseline.json')
    engine = read('round6_full_v2.json' if second else 'round6_full_no_pattern.json')
    b = {r['case']: r for r in baseline['rows']}
    rows = []
    for r in engine['rows']:
        other = b[r['case']]
        rows.append({
            'case': r['case'], 'audio_seconds': r['runs'][0]['audio_seconds'],
            'baseline_seconds': other['median_seconds'], 'engine_seconds': r['median_seconds'],
            'baseline_rtf': other['median_rtf'], 'engine_rtf': r['median_rtf'],
            'observed_speedup': other['median_seconds'] / r['median_seconds'],
        })
    paired = read('round6_paired_v2.json' if second else 'round6_paired_final.json')
    waves = []
    for voice in ['M1', 'F1']:
        prefix = 'round6_v2_wave_identity' if second else 'round6_engine_wave_identity'
        waves += read(f'{prefix}_{voice}.json')['rows']
    guards = {label: read(name) for label, name in [
        ('baseline_full', 'round6_full_baseline_guard.json'),
        ('engine_full', 'round6_full_v2_guard.json' if second else 'round6_full_no_pattern_guard.json'),
        ('rejected_pattern_enabled', 'round6_full_exact_cached_guard.json'),
    ]}
    out = {
        'selected_model': 'models/round5-original50-student',
        'engine_version': args.engine_version,
        'source_vector_sha256': '7c81ab7f0ddcc4266743701599a8708d2edd1be04285770f996e84a8bb5f4447',
        'threads': 1, 'actual_steps': 8, 'weights_changed': False,
        'table': rows, 'short_alternating_paired_summary': paired['summary'],
        'memory_guards': guards,
        'waveform_cases': len(waves),
        'all_float32_bit_identical': all(r['float32_bit_identical'] for r in waves),
        'new_asr_test': False,
        'limitations': [
            'Shared EPYC7551 1GB host, not ARM; timings fluctuate.',
            'Full four-case comparison: three timed repeats per case, one baseline batch followed by one engine batch.',
            f'Short sentences additionally alternate {len(paired["runs"]) // 2} fresh-process pairs; retain all outliers.',
            'Waveform checks are six numerical regressions, not a broad perceptual score.',
            'The pattern-enabled engine exceeded its 480MB sampled RSS budget and is not selected.',
            'Successful memory-pattern-disabled engine keeps CPU arena and prepacking enabled.',
        ],
    }
    output = 'results/round6_summary_v2.json' if second else 'results/round6_summary.json'
    Path(output).write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
