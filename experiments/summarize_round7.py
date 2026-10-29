"""Rebuild the speaking-rate development summary from committed raw records."""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT / 'results' / name).read_text())


def main():
    reports = [read(f'round7_speed_{model}_{voice}.json')
               for model in ['base', 'v2'] for voice in ['M1', 'F1']]
    assert all(r['complete'] and r['passed'] for r in reports)
    identity = [v for voice in ['M1', 'F1']
                for v in read(f'round7_speed_identity_{voice}.json')['rows']]
    assert len(identity) == 20 and all(v['float32_bit_identical'] for v in identity)
    asr = []
    for label, files in [
        ('F1200', ['round7_speed_asr_M1.json', 'round7_speed_asr_F1.json']),
        ('original50', ['round7_original50_speed_asr.json', 'round7_original50_slow_asr.json']),
        ('original8-control', ['round7_original8_fast_asr.json']),
        ('original16-control', ['round7_original16_fast_asr.json']),
    ]:
        totals = defaultdict(lambda: [0, 0])
        for name in files:
            for row in read(name)['rows']:
                speed = float(row['variant'].rsplit('-', 1)[1])
                value = totals[(speed, row['language'])]
                value[0] += row['errors']
                value[1] += row['reference_units']
        for (speed, lang), (errors, units) in sorted(totals.items()):
            asr.append({'model': label, 'speed': speed, 'language': lang,
                        'metric': 'CER' if lang == 'ko' else 'WER',
                        'errors': errors, 'reference_units': units, 'rate': errors / units})
    paired = read('round7_paired_v2_v3.json')
    assert paired['complete'] and paired['original_library_restored']
    crop = []
    for label in ['original50', 'F1200']:
        diagnostic = read(f'round7_native_diagnostic_{label}.json')
        assert diagnostic['complete']
        crop.extend(dict(model=label, **row) for row in diagnostic['crop_checks'])
    report = {
        'date': '2026-10-29', 'selected_model': 'F1200', 'selected_engine': 'v2',
        'actual_student_steps': 8, 'server_threads': 1,
        'functional_syntheses': sum(len(r['rows']) for r in reports),
        'functional_checks_passed': True, 'bit_identical_engine_waveforms': len(identity),
        'asr_scope': 'Two development texts, KO/EN, repeated in M1/F1. Whisper base INT8 CPU1. Not independent testing, perceptual quality, or model agreement. Error rates can exceed 100% with insertions.',
        'asr': asr, 'crop_checks': crop,
        'human_feedback': read('round7_original50_feedback.json'),
        'native2_quality_fixed': False, 'postprocessing_default': False,
        'new_training': False, 'model_weights_changed': False,
        'v3_selected': False, 'v3_paired': paired['summary'],
        'v3_scope': 'Microkernel S32 identity passed eight weight-shape probes; no full v3 waveform/long-form validation. Whole-TTS English had no consistent improvement, so keep v2.',
        'timing_caveat': 'Later shared-server v2 short-form warm timings were about RTF1.05. Earlier RTF0.943..0.971 is a session result, not a guaranteed service level. Cold-shape speed sweeps are not paired performance benchmarks.',
    }
    target = ROOT / 'results/round7_speed_summary.json'
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(target.name)


if __name__ == '__main__':
    main()
