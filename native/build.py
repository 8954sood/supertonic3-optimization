"""Build an isolated native kernel using the optional ziglang package."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument("--target", default="x86_64-linux-gnu.2.28")
p.add_argument("--cpu", default="znver1")
p.add_argument("--output", default=str(root / "libpacked_blas.so"))
a = p.parse_args()
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
    str(root / "packed_blas.c"),
    "-I",
    str(root / "include"),
    "-o",
    a.output,
    "-ldl",
]
subprocess.run(command, check=True)
Path(a.output + ".json").write_text(
    json.dumps(
        {
            "target": a.target,
            "cpu": a.cpu,
            "ort_api_version": 23,
            "blas": "NumPy wheel scipy_cblas_sgemm64_",
            "command": command,
        },
        indent=2,
    )
)
print(a.output)
