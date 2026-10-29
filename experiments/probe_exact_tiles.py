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
    a = p.parse_args()
    if a.output.exists():
        p.error('Choose a new output')
    psutil.Process().cpu_affinity([0])
    model = onnx.load(Path(a.model)/'onnx/vector_estimator.onnx')
    weights = {}
    for tensor in model.graph.initializer:
        value = numpy_helper.to_array(tensor)
        if value.dtype == np.int8 and value.ndim == 2 and '/pwconv' in tensor.name:
            weights.setdefault(value.shape, value.copy())
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
        rows=[], complete=False)
    for m in [60, 200]:
        for k, n in [(512,1536), (1536,512), (512,2048), (2048,512)]:
            w = weights[k,n]
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
                actual_weight_sha256=hashlib.sha256(w.tobytes()).hexdigest())
            report['rows'].append(row)
            a.output.write_text(json.dumps(report,indent=2)+'\n')
            print(row,flush=True)
    report['complete'] = True
    a.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
