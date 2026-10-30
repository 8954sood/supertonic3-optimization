"""One guarded server process: count real steps, time warm runs, hash audio in memory."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--runtime-root', type=Path, default=Path('.'))
    parser.add_argument('--exact-library', type=Path,
                        help='Optional frozen AVX2 candidate; never replaces the active library')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--speed', type=float, default=1.0)
    parser.add_argument('--voice', default='M1')
    parser.add_argument('--cases', nargs='+', default=['ko_short', 'en_short'])
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--affinity', type=int, default=0)
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 1:
        parser.error('Choose a new output and positive repeats')
    for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        os.environ[name] = '1'
    # The same harness can select a frozen old runtime in a separate process.
    sys.path.insert(0, str(args.runtime_root.resolve()))
    from runtime import Engine
    import numpy as np
    import onnxruntime as ort
    import psutil
    from benchmark import TEXTS
    if any(case not in TEXTS for case in args.cases):
        parser.error('Unknown benchmark text')
    process = psutil.Process()
    process.cpu_affinity([args.affinity])
    override = nullcontext()
    if args.exact_library:
        if not args.exact_library.is_file():
            parser.error('Candidate library does not exist')
        from exact_ops import library_path
        library_path()  # Keep the normal Linux/AVX2 capability check.
        from unittest.mock import patch
        override = patch('exact_ops.library_path', return_value=str(args.exact_library.resolve()))
    with override:
        engine = Engine(args.model, voice=args.voice)
    if args.exact_library and not engine.exact_int8_backend:
        parser.error('A candidate library requires an exact-integer model package')

    class CountedSession:
        def __init__(self, session):
            self.session, self.calls = session, 0

        def __getattr__(self, name):
            return getattr(self.session, name)

        def run(self, *values, **keywords):
            self.calls += 1
            return self.session.run(*values, **keywords)

    counter = CountedSession(engine.sessions['vector_estimator'])
    engine.sessions['vector_estimator'] = counter
    sessions = list(engine.sessions.values())
    if getattr(engine, 'condition', None) is not None:
        sessions.append(engine.condition)
    if getattr(engine, 'step_condition', None) is not None:
        sessions.append(engine.step_condition)
    assert all(s.get_session_options().intra_op_num_threads == 1 and
               s.get_session_options().inter_op_num_threads == 1 for s in sessions)
    report = dict(model=args.model.name, voice=args.voice, speed=args.speed,
                  steps=8, threads=1, affinity=process.cpu_affinity(), nice=process.nice(),
                  python=platform.python_version(), onnxruntime=ort.__version__, numpy=np.__version__,
                  runtime_sha256=sha256(args.runtime_root / 'runtime.py'),
                  harness_sha256=sha256(Path(__file__)),
                  exact_library_sha256=sha256(args.exact_library) if args.exact_library else None,
                  graph_sha256={p.name: sha256(p) for p in sorted((args.model / 'onnx').glob('*.onnx'))},
                  load_seconds=engine.load_seconds,
                  note='Fresh single-engine process, warm synthesis excludes load/hash/file I/O. '
                       'Time and waveform hashes from this server only; not an ARM result.',
                  rows=[], complete=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for case in args.cases:
        lang, text = TEXTS[case]
        runs, expected = [], None
        for repeat in range(args.repeats + 1):
            before = counter.calls
            cpu_start = time.process_time()
            wave, metrics = engine.synthesize(text, lang, 8, speed=args.speed)
            cpu_seconds = time.process_time() - cpu_start
            assert counter.calls - before == 8
            digest = hashlib.sha256(wave.tobytes()).hexdigest()
            if expected is None:
                expected = digest
            assert digest == expected
            if repeat:
                runs.append(dict(**metrics, process_cpu_seconds=cpu_seconds, actual_vector_calls=8))
        report['rows'].append(dict(case=case, text=text, lang=lang,
            float32_sha256=expected, samples=int(wave.size), sample_rate=engine.sample_rate,
            runs=runs, median_seconds=statistics.median(r['seconds'] for r in runs),
            median_rtf=statistics.median(r['rtf'] for r in runs),
            rss_mb=process.memory_info().rss / 1e6))
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        print(case, 'speed', args.speed, 'RTF', round(report['rows'][-1]['median_rtf'], 4), flush=True)
    report['complete'] = True
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
