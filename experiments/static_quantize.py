"""Calibrate Conv QDQ with real eight-step trajectories, no fabricated tensors.
Calibration set is a development sample, NOT evidence of held-out quality.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from runtime import Engine
from benchmark import TEXTS
import numpy as np
from onnxruntime.quantization import (
    CalibrationDataReader,
    quantize_static,
    QuantType,
    QuantFormat,
    CalibrationMethod,
)


class Reader(CalibrationDataReader):
    def __init__(self, paths):
        self.paths = iter(paths)

    def get_next(self):
        path = next(self.paths, None)
        if path is None:
            return None
        with np.load(path) as data:
            return {k: data[k] for k in data.files}


class Capture:
    def __init__(self, session, paths):
        self.session = session
        self.paths = paths

    def get_inputs(self):
        return self.session.get_inputs()

    def run(self, outputs, feed):
        path = Path("results/calibration") / f"{len(self.paths):04d}.npz"
        np.savez(path, **feed)
        self.paths.append(path)
        return self.session.run(outputs, feed)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="models/static-int8")
    p.add_argument("--reuse", action="store_true")
    a = p.parse_args()
    dest = Path(a.output)
    dest.mkdir(parents=True, exist_ok=True)
    cache = Path("results/calibration")
    cache.mkdir(parents=True, exist_ok=True)
    paths = sorted(cache.glob("*.npz")) if a.reuse else []
    if not paths:
        for old in cache.glob("*.npz"):
            old.unlink()
        e = Engine()
        e.sessions["vector_estimator"] = Capture(e.sessions["vector_estimator"], paths)
        for name in TEXTS:
            lang, text = TEXTS[name]
            e.synthesize(text, lang, 8)
            print("calibrated", name, flush=True)
    quantize_static(
        "models/supertonic-3/onnx/vector_estimator.onnx",
        str(dest / "vector_estimator.onnx"),
        Reader(paths),
        quant_format=QuantFormat.QDQ,
        op_types_to_quantize=["Conv", "MatMul"],
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=CalibrationMethod.MinMax,
        extra_options={
            "ActivationSymmetric": True,
            "WeightSymmetric": True,
            "MatMulConstBOnly": True,
        },
    )
    print(dest, flush=True)
