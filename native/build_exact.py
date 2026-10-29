"""Build an optional exact integer CPU kernel; no model parameters are modified."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument("--target", default="x86_64-linux-gnu.2.28")
p.add_argument("--cpu", default="znver1")
p.add_argument("--output", default=str(root / "libexact_int8.so"))
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
    str(root / "exact_int8.c"),
    "-I",
    str(root / "include"),
    "-o",
    a.output,
]
subprocess.run(command, check=True)
Path(a.output + ".json").write_text(
    json.dumps(
        {
            "target": a.target,
            "cpu": a.cpu,
            "ort_api_version": 23,
            "source_sha256": hashlib.sha256(
                (root / "exact_int8.c").read_bytes()
            ).hexdigest(),
            "kernel": "Exact U8 x S8[-64,64] -> S32; persistent packing, proven-safe int16 partial accumulation with int32 fallback",
            "engine_version": 4,
            "tile": "4 rows x16 columns, safe-K8; original8x8 fallback for other shapes/tails",
            "command": command,
        },
        indent=2,
    )
    + "\n"
)
print(a.output)
