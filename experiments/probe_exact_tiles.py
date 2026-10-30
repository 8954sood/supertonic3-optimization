"""Measure prepared integer kernels with actual fixed FFN weights, not TTS latency."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runtime  # Establish one-thread numerical library settings.
import argparse
import ctypes
import hashlib
import json
import statistics
import time
import numpy as np
import onnx
import psutil
from onnx import numpy_helper


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--reference-library', required=True)
    p.add_argument('--candidate-library', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rows', nargs='+', type=int, default=[60, 200],
                   help='Include remainders to measure the M4 tail path')
    p.add_argument('--all-weights', action='store_true',
                   help='Check every FFN matrix, including repeated shapes')
    a = p.parse_args()
    if a.output.exists():
        p.error('Choose a new output')
    if any(not 1 <= m <= 2000 for m in a.rows):
        p.error('Rows must be between 1 and 2000')
    psutil.Process().cpu_affinity([0])
    model = onnx.load(Path(a.model)/'onnx/vector_estimator.onnx')
    weights = []
    seen = set()
    for tensor in model.graph.initializer:
        value = numpy_helper.to_array(tensor)
        if value.dtype == np.int8 and value.ndim == 2 and '/pwconv' in tensor.name:
            if a.all_weights or value.shape not in seen:
                weights.append((tensor.name, value.copy()))
                seen.add(value.shape)
    weights.sort(key=lambda item: (item[1].shape, item[0]))
    del model
    libraries = {}
    for name, path in [('reference', a.reference_library), ('candidate', a.candidate_library)]:
        library = ctypes.CDLL(str(Path(path).resolve()))
        library.prepare_exact_u8s8.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
        library.prepare_exact_u8s8.restype = ctypes.c_int
        library.run_prepared_u8s8.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_uint8]
        library.run_prepared_u8s8.restype = ctypes.c_int
        libraries[name] = library
    rng = np.random.RandomState(61)
    report = dict(note='Prepared actual-weight microprobe only; not a TTS speed or quality score.',
        rows=[], complete=False, all_weights=a.all_weights,
        weight_count=len(weights), row_counts=a.rows, random_seed=61,
        repeats=19, discarded_first_repeat=True,
        library_sha256={name:hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for name,path in [('reference',a.reference_library),('candidate',a.candidate_library)]})
    for m in a.rows:
        for weight_name, w in weights:
            k, n = w.shape
            x = rng.randint(0,256,(m,k),dtype=np.uint8)
            outputs = {name: np.empty((m,n),np.int32) for name in libraries}
            elapsed = {name: [] for name in libraries}
            for library in libraries.values():
                assert library.prepare_exact_u8s8(w.ctypes.data,k,n) == 0
            for repeat in range(19):
                names = list(libraries)
                if repeat % 2:
                    names.reverse()
                for name in names:
                    start = time.perf_counter()
                    assert libraries[name].run_prepared_u8s8(x.ctypes.data,outputs[name].ctypes.data,m,k,n,128) == 0
                    elapsed[name].append(time.perf_counter()-start)
            np.testing.assert_array_equal(outputs['reference'],outputs['candidate'])
            medians = {name: statistics.median(values[1:]) for name,values in elapsed.items()}
            row = dict(shape=[m,k,n],s32_bitidentity=True,seconds=medians,
                speedup=medians['reference']/medians['candidate'],
                actual_weight_name=weight_name,
                actual_weight_sha256=hashlib.sha256(w.tobytes()).hexdigest())
            report['rows'].append(row)
            a.output.write_text(json.dumps(report,indent=2)+'\n')
            print(row,flush=True)
    report['complete'] = True
    a.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
