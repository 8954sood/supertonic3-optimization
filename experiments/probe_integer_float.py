"""Measure integer dot products in FP32 only when an integer-exact bound holds."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runtime  # One thread before importing numerical libraries.
import argparse
import ctypes
import json
import statistics
import time

import numpy as np
import onnx
from onnx import numpy_helper
import psutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output')
    psutil.Process().cpu_affinity([0])
    model = onnx.load(args.model / 'onnx/vector_estimator.onnx')
    weights = {}
    for tensor in model.graph.initializer:
        value = numpy_helper.to_array(tensor)
        if value.dtype == np.int8 and value.ndim == 2 and '/pwconv' in tensor.name:
            weights.setdefault(value.shape, value.copy())
    del model
    library = ctypes.CDLL(str(args.library.resolve()))
    library.prepare_exact_u8s8.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
    library.run_prepared_u8s8.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_uint8]
    rng = np.random.RandomState(61)
    report = dict(note='Microprobe only. FP32 integers are exact only if every partial '
                       'sum is bounded by 2**24; conservative bound is 255*sum(abs(weight)). '
                       'Input conversion, weight decoding and output cast included in decoded timing.',
                  rows=[], complete=False)
    for m in [60, 200]:
        for (k, n), weight in sorted(weights.items()):
            bound = 255 * int(np.abs(weight.astype(np.int64)).sum(axis=0).max())
            if bound >= 2**24:
                report['rows'].append(dict(shape=[m,k,n],safe=False,bound=bound))
                continue
            x = rng.randint(0,256,(m,k),dtype=np.uint8)
            dense = weight.astype(np.float32)
            outputs = dict(native=np.empty((m,n),np.int32))
            assert library.prepare_exact_u8s8(weight.ctypes.data,k,n) == 0
            times = {name: [] for name in ['native','cached','decoded']}
            for repeat in range(10):
                names = list(times)
                names = names[repeat % 3:] + names[:repeat % 3]
                for name in names:
                    began = time.perf_counter()
                    if name == 'native':
                        assert library.run_prepared_u8s8(x.ctypes.data,outputs[name].ctypes.data,m,k,n,128) == 0
                    else:
                        prepared = dense if name == 'cached' else weight.astype(np.float32)
                        centered = x.astype(np.float32) - np.float32(128)
                        outputs[name] = np.matmul(centered,prepared).astype(np.int32)
                    elapsed = time.perf_counter() - began
                    if repeat:
                        times[name].append(elapsed)
            np.testing.assert_array_equal(outputs['native'],outputs['cached'])
            np.testing.assert_array_equal(outputs['native'],outputs['decoded'])
            medians = {name: statistics.median(values) for name,values in times.items()}
            report['rows'].append(dict(shape=[m,k,n],safe=True,bound=bound,
                s32_bitidentity=True,seconds=medians,
                cached_speedup=medians['native']/medians['cached'],
                decoded_speedup=medians['native']/medians['decoded']))
    report['complete'] = True
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
