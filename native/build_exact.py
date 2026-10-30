"""Build an optional exact integer CPU kernel; no model parameters are modified."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

root = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument("--target", default="x86_64-linux-gnu.2.28")
p.add_argument("--cpu", default="znver1")
p.add_argument("--engine-version", type=int, choices=[4, 5], default=5,
               help="Keep v4 available as a timing/control fallback")
p.add_argument("--output", default=str(root / "libexact_int8.so"))
a = p.parse_args()
output = Path(a.output).absolute()
output.parent.mkdir(parents=True, exist_ok=True)
if output.is_symlink():
    p.error("Use a regular output path")
source = root / ("exact_int8.c" if a.engine_version == 5 else "exact_int8_v4.c")
source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
descriptor, temporary_name = tempfile.mkstemp(
    prefix=output.name + ".", suffix=".tmp", dir=output.parent
)
os.close(descriptor)
temporary = Path(temporary_name)
command = [
    sys.executable,
    "-m",
    "ziglang",
    "cc",
    "-target",
    a.target,
    "-mcpu=" + a.cpu,
    "-O3",
    "-fPIC",
    "-shared",
    str(source),
    "-I",
    str(root / "include"),
    "-o",
    str(temporary),
]
try:
    subprocess.run(command, check=True)
    if hashlib.sha256(source.read_bytes()).hexdigest() != source_sha:
        raise RuntimeError("Kernel source changed during the build")
    temporary.replace(output)
finally:
    temporary.unlink(missing_ok=True)
Path(str(output) + ".json").write_text(
    json.dumps(
        {
            "target": a.target,
            "cpu": a.cpu,
            "ort_api_version": 23,
            "source_sha256": source_sha,
            "binary_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "kernel": ("Exact U8 x S8[-64,64] -> S32; persistent packing, safe base plus lossless sparse integer corrections"
                       if a.engine_version == 5 else "Exact U8 x S8[-64,64] -> S32; persistent packing, safe-K8 and S32 fallback"),
            "engine_version": a.engine_version,
            "tile": ("4 rows x16 columns, branch-free safe-K8; M2/M1 safe-K8 tails; original fallback for other shapes"
                     if a.engine_version == 5 else "4 rows x16 columns, safe-K8; original8x8 fallback"),
            "command": command,
        },
        indent=2,
    )
    + "\n"
)
print(output)
