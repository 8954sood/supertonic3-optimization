"""Check normal-output preservation and actual denoiser calls of a retime package."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import numpy as np
from runtime import Engine
from benchmark import TEXTS
from supertonic.loader import load_voice_style_from_json_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', required=True)
    p.add_argument('--candidate', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Choose a new output')
    base, candidate = Engine(model=a.base), Engine(model=a.candidate)
    if candidate.speed_strategy != 'latent-retime':
        p.error('Expected a latent-retime candidate')
    inner = candidate.sessions['vector_estimator']

    class Count:
        calls = 0
        def get_inputs(self):
            return inner.get_inputs()
        def run(self, *args, **kwargs):
            self.calls += 1
            return inner.run(*args, **kwargs)

    counter = Count()
    candidate.sessions['vector_estimator'] = counter
    rows = []
    for voice in ['M1', 'F1']:
        for e in [base, candidate]:
            e.style = load_voice_style_from_json_file(e.root/'voice_styles'/f'{voice}.json')
        for case in ['ko_short', 'en_short']:
            lang, text = TEXTS[case]
            normal, _ = base.synthesize(text, lang, 8, speed=1.05)
            counter.calls = 0
            actual, _ = candidate.synthesize(text, lang, 8, speed=1.05)
            np.testing.assert_array_equal(normal, actual)
            assert counter.calls == 8
            counter.calls = 0
            fast, metrics = candidate.synthesize(text, lang, 8, speed=2.0)
            assert counter.calls == 8 and np.isfinite(fast).all()
            assert abs(len(fast)/len(normal) - 1.05/2.0) < 1e-4
            rows.append(dict(voice=voice, case=case, normal_bitidentity=True,
                actual_fast_calls=counter.calls, normal_samples=len(normal),
                fast_samples=len(fast), generation_speed=metrics['generation_speed']))
    a.output.write_text(json.dumps(dict(base=a.base, candidate=a.candidate,
        complete=True, rows=rows), indent=2)+'\n')
    print('Normal outputs unchanged; all four fast cases made exactly eight calls.')


if __name__ == '__main__':
    main()
