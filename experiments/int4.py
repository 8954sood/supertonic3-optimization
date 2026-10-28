"""Experimental groupwise 4-bit denoiser, keeping eight genuine model passes."""

from pathlib import Path
import argparse
import json
import logging
import shutil
from onnxruntime.quantization import matmul_nbits_quantizer as q

p = argparse.ArgumentParser()
p.add_argument("--output", required=True)
p.add_argument("--accuracy", type=int, default=4)
p.add_argument("--block-size", type=int, default=32)
a = p.parse_args()
dest = Path(a.output)
if dest.exists():
    p.error("choose a new output directory")
shutil.copytree("models/fast-full-int8", dest)
logging.getLogger().setLevel(logging.WARNING)
config = q.DefaultWeightOnlyQuantConfig(
    block_size=a.block_size,
    is_symmetric=True,
    accuracy_level=a.accuracy,
    op_types_to_quantize=("MatMul",),
    bits=4,
)
quantizer = q.MatMulNBitsQuantizer(
    "models/pointwise-fp32/vector_estimator.onnx", algo_config=config
)
quantizer.process()
quantizer.model.save_model_to_file(str(dest / "onnx/vector_estimator.onnx"), False)
(dest / "manifest.json").write_text(
    json.dumps(
        {
            "mode": "experimental-denoiser-int4",
            "bits": 4,
            "block_size": a.block_size,
            "accuracy_level": a.accuracy,
            "conditioning_vocoder": "dynamic int8",
            "status": "requires paired quality and target CPU validation",
        },
        indent=2,
    )
)
print(dest, flush=True)
