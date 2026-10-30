"""Compare two VE sessions, sharing verified identical frontend/vocoder sessions."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import Engine, options
import argparse
import copy
import hashlib
import json
import random
import statistics
import time
from unittest.mock import patch

import onnxruntime as ort
import psutil
from benchmark import TEXTS


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


class CountedSession:
    def __init__(self, session):
        self.session, self.calls = session, 0

    def __getattr__(self, name):
        return getattr(self.session, name)

    def run(self, *args, **kwargs):
        self.calls += 1
        return self.session.run(*args, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--reference-library', type=Path, required=True)
    parser.add_argument('--candidate-library', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pairs', type=int, default=6)
    parser.add_argument('--order-seed', type=int,
                        help='Reproducibly shuffle balanced AB/BA timing order')
    parser.add_argument('--speeds', nargs='+', type=float, default=[1.0, 2.0])
    parser.add_argument('--cases', nargs='+', default=['ko_short', 'en_short'])
    parser.add_argument('--voice', default='M1')
    args = parser.parse_args()
    if args.output.exists() or args.pairs < 4:
        parser.error('Choose a new output and at least four pairs')
    process = psutil.Process()
    process.cpu_affinity([0]); process.nice(19)
    shared = ['onnx/duration_predictor.onnx', 'onnx/text_encoder.onnx', 'onnx/vocoder.onnx',
              'onnx/condition.onnx', 'onnx/tts.json', 'onnx/unicode_indexer.json', f'voice_styles/{args.voice}.json']
    checks = {name: digest(args.reference / name) == digest(args.candidate / name) for name in shared}
    if not all(checks.values()):
        raise ValueError('Only the vector session and its kernel may differ')
    manifests = [json.loads((model / 'manifest.json').read_text()) for model in [args.reference,args.candidate]]
    for key in ['supported_steps','speed_strategy','latent_head_generation_speed','latent_head_supported_speeds']:
        if manifests[0].get(key) != manifests[1].get(key):
            raise ValueError('Inference settings differ')
    from exact_ops import library_path
    library_path()
    with patch('exact_ops.library_path', return_value=str(args.reference_library.resolve())):
        reference = Engine(args.reference, voice=args.voice)
    candidate = copy.copy(reference)
    candidate.root = args.candidate
    candidate.sessions = dict(reference.sessions)
    with patch('exact_ops.library_path', return_value=str(args.candidate_library.resolve())):
        candidate.sessions['vector_estimator'] = ort.InferenceSession(
            str(args.candidate / 'onnx/vector_estimator.onnx'), options(exact_int8_backend=True),
            providers=['CPUExecutionProvider'])
    engines = dict(reference=reference,candidate=candidate)
    for engine in engines.values():
        engine.sessions['vector_estimator'] = CountedSession(engine.sessions['vector_estimator'])
    report = dict(note='One process, two vector sessions; frontend/condition/vocoder shared only after '
                       'SHA and settings checks. Sequential alternating inference. Warm timing, '
                       'extra residency is benchmark overhead, not production RSS.',
                  shared_files_identical=checks,reference=args.reference.name,candidate=args.candidate.name,
                  library_sha256={name:digest(path) for name,path in
                      [('reference',args.reference_library),('candidate',args.candidate_library)]},
                  vector_sha256={name:digest(path/'onnx/vector_estimator.onnx')
                      for name,path in [('reference',args.reference),('candidate',args.candidate)]},
                  runtime_sha256=digest(Path(__file__).resolve().parents[1]/'runtime.py'),
                  harness_sha256=digest(Path(__file__)),voice=args.voice,order_seed=args.order_seed,
                  pairs=args.pairs,steps=8,threads=1,affinity=[0],rows=[],complete=False)
    pair_order=[bool(repeat%2) for repeat in range(1,args.pairs+1)]
    if args.order_seed is not None:
        random.Random(args.order_seed).shuffle(pair_order)
    report['timed_pair_order']=['BA' if reverse else 'AB' for reverse in pair_order]
    for speed in args.speeds:
        for case in args.cases:
            lang,text = TEXTS[case]
            expected = None
            runs = {name:[] for name in engines}
            for repeat in range(args.pairs + 1):
                names = list(engines)
                if repeat and pair_order[repeat-1]:
                    names.reverse()
                for name in names:
                    engine = engines[name]
                    before = engine.sessions['vector_estimator'].calls
                    steal_before = getattr(psutil.cpu_times(percpu=True)[0], 'steal', 0)
                    cpu_before = time.process_time()
                    wave,metrics = engine.synthesize(text,lang,8,speed=speed)
                    cpu = time.process_time() - cpu_before
                    steal = getattr(psutil.cpu_times(percpu=True)[0], 'steal', 0) - steal_before
                    assert engine.sessions['vector_estimator'].calls - before == 8
                    wave_sha = hashlib.sha256(wave.tobytes()).hexdigest()
                    if expected is None:
                        expected = wave_sha
                    assert wave_sha == expected, 'Waveform changed'
                    if repeat:
                        runs[name].append(dict(**metrics,process_cpu_seconds=cpu,
                                               cpu0_steal_seconds=steal,actual_vector_calls=8))
            reductions = [100*(1-c['seconds']/r['seconds']) for r,c in zip(runs['reference'],runs['candidate'])]
            report['rows'].append(dict(case=case,speed=speed,runs=runs,float32_sha256=expected,
                median_paired_time_reduction_percent=statistics.median(reductions),
                paired_time_reductions_percent=reductions,
                median_rtf={name:statistics.median(r['rtf'] for r in values) for name,values in runs.items()},
                process_rss_mb=process.memory_info().rss/1e6))
            args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
            print(case,speed,'reduction',round(report['rows'][-1]['median_paired_time_reduction_percent'],3),flush=True)
    report['complete'] = True
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')


if __name__ == '__main__':
    main()
