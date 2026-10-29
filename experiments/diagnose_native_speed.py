"""Separate output cropping and text alignment effects at native speed2.

Diagnostic only: no change to runtime defaults, weights, or waveform tempo.
Local original-model references may use four threads; no server timings here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runtime
from runtime import Engine
import numpy as np
import soundfile as sf
from benchmark import TEXTS
from distill.teacher_policy import assert_original_model
from experiments.compare_original_speed import EXPECTED
from supertonic.loader import load_voice_style_from_json_file


class CaptureVocoder:
    def __init__(self, session):
        self.session = session
        self.raw = None

    def get_inputs(self):
        return self.session.get_inputs()

    def run(self, *args, **kwargs):
        outputs = self.session.run(*args, **kwargs)
        self.raw = outputs[0].reshape(-1).copy()
        return outputs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--original', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Use a new output path')
    if a.original:
        assert_original_model(a.model)
        for name, expected in EXPECTED.items():
            if hashlib.sha256((a.model / 'onnx' / (name + '.onnx')).read_bytes()).hexdigest() != expected:
                raise ValueError('Original graph mismatch: ' + name)
        original_options = runtime.options
        def local_reference_options(*args, **kwargs):
            options = original_options(*args, **kwargs)
            options.intra_op_num_threads = 4
            return options
        runtime.options = local_reference_options
    engine = Engine(model=a.model)
    capture = CaptureVocoder(engine.sessions['vocoder'])
    engine.sessions['vocoder'] = capture
    steps = 50 if a.original else 8
    label = 'original50' if a.original else 'F1200'
    folder = Path('samples') / a.output.stem
    folder.mkdir(parents=True, exist_ok=True)
    report = {
        'model': str(a.model), 'steps': steps, 'speed': 2.0, 'seed': 42,
        'note': 'Local diagnostic, not CPU1 server timing. Segments synthesized independently with native duration control then concatenated, no inserted silence or tempo filter; punctuation changes and achieved total duration must be considered. No candidate adopted.',
        'rows': [], 'crop_checks': [], 'complete': False,
    }
    def save(voice, mode, audio):
        path = folder / f'{voice}_{mode}.wav'
        sf.write(path, audio, engine.sample_rate, subtype='FLOAT')
        report['rows'].append({
            'variant': label + '-' + mode, 'voice': voice, 'case': 'ko_short',
            'language': 'ko', 'text': TEXTS['ko_short'][1], 'audio': str(path),
            'audio_seconds': len(audio) / engine.sample_rate,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    for voice in ['M1', 'F1']:
        engine.style = load_voice_style_from_json_file(a.model / 'voice_styles' / (voice + '.json'))
        cropped, _ = engine.synthesize(TEXTS['ko_short'][1], 'ko', steps, seed=42, speed=2.0)
        full = capture.raw
        save(voice, 'cropped', cropped)
        save(voice, 'uncropped', full)
        report['crop_checks'].append({
            'voice': voice, 'prefix_bitidentical': bool(np.array_equal(cropped, full[:len(cropped)])),
            'cropped_samples': len(cropped), 'full_samples': len(full),
            'removed_tail_seconds': (len(full) - len(cropped)) / engine.sample_rate,
        })
        for mode, parts in [
            ('sentences', ['안녕하세요.', '오래된 컴퓨터에서도 빠르게 음성을 생성합니다.']),
            ('clauses', ['안녕하세요.', '오래된 컴퓨터에서도', '빠르게 음성을 생성합니다.']),
        ]:
            pieces = [engine.synthesize(text, 'ko', steps, seed=42, speed=2.0)[0] for text in parts]
            save(voice, mode, np.concatenate(pieces))
            report['rows'][-1]['segments'] = parts
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        print(label, voice, 'done', flush=True)
    report['complete'] = True
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
