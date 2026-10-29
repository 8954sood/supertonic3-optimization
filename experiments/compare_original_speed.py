"""Original FP32 CFG4 speaking-rate reference; default maximum50 steps, local only."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runtime
from runtime import Engine
from distill.teacher_policy import assert_original_model
import argparse
import hashlib
import json
import soundfile as sf
from benchmark import TEXTS
from supertonic.loader import load_voice_style_from_json_file

EXPECTED = {
    'duration_predictor': 'c3eb91414d5ff8a7a239b7fe9e34e7e2bf8a8140d8375ffb14718b1c639325db',
    'text_encoder': 'c7befd5ea8c3119769e8a6c1486c4edc6a3bc8365c67621c881bbb774b9902ff',
    'vector_estimator': '883ac868ea0275ef0e991524dc64f16b3c0376efd7c320af6b53f5b780d7c61c',
    'vocoder': '085de76dd8e8d5836d6ca66826601f615939218f90e519f70ee8a36ed2a4c4ba',
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, default=Path('models/supertonic-3'))
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--speeds', nargs='+', type=float, default=[2.0, 1.0, 1.5])
    p.add_argument('--steps', type=int, choices=[8, 16, 32, 50], default=50)
    p.add_argument('--output', type=Path, default=Path('results/round7_original50_speed.json'))
    a = p.parse_args()
    origin = assert_original_model(a.model)
    for name, expected in EXPECTED.items():
        actual = hashlib.sha256((a.model / 'onnx' / (name + '.onnx')).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('Original teacher hash mismatch: ' + name)
    original_options = runtime.options
    def teacher_options(*args, **kwargs):
        options = original_options(*args, **kwargs)
        options.intra_op_num_threads = a.threads
        return options
    runtime.options = teacher_options
    engine = Engine(model=a.model)
    out = Path('samples') / a.output.stem
    out.mkdir(parents=True, exist_ok=True)
    report = {
        'teacher_origin': origin, 'precision': 'FP32', 'cfg': 4,
        'steps': a.steps, 'teacher_generation_threads': a.threads, 'graph_sha256': EXPECTED,
        'note': 'Original unmodified-model listening reference on local Mac; not CPU1 server timing.50 is maximum reference; lower counts are diagnostic controls. Native model duration control only; no waveform tempo processing.',
        'rows': [], 'complete': False,
    }
    for voice in ['M1', 'F1']:
        engine.style = load_voice_style_from_json_file(a.model / 'voice_styles' / (voice + '.json'))
        for case in ['ko_short', 'en_short']:
            lang, text = TEXTS[case]
            for speed in a.speeds:
                wave, metrics = engine.synthesize(text, lang, a.steps, seed=42, speed=speed)
                audio = out / f'{voice}_{case}_speed_{speed:g}.wav'
                sf.write(audio, wave, engine.sample_rate, subtype='FLOAT')
                report['rows'].append({
                    'variant': f'original{a.steps}-speed-{speed:g}', 'case': case,
                    'voice': voice, 'language': lang, 'text': text,
                    'speed': speed, 'steps': a.steps, 'seed': 42, 'audio': str(audio),
                    'sha256': hashlib.sha256(audio.read_bytes()).hexdigest(),
                    'audio_seconds': len(wave) / engine.sample_rate,
                })
                a.output.parent.mkdir(parents=True, exist_ok=True)
                a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
                print(voice, case, speed, flush=True)
    report['complete'] = True
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
