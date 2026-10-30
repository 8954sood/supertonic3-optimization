"""Build the optional ARM64 row-parallel normalization kernel on this machine."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cc',default='clang')
    args=parser.parse_args()
    if platform.machine().lower() not in {'arm64','aarch64'} or platform.system() not in {'Darwin','Linux'}:
        parser.error('Build natively on macOS or Linux ARM64; standard ORT remains available elsewhere')
    root=Path(__file__).resolve().parent
    output=root/('librow_norm.dylib' if platform.system()=='Darwin' else 'librow_norm.so')
    temporary=root/(output.name+'.tmp')
    command=[args.cc,'-O3','-ffp-contract=off','-DFUSED_M2','-DFUSED_OUTPUT','-fPIC',
             '-dynamiclib' if platform.system()=='Darwin' else '-shared',
             str(root/'row_norm.c'),'-I',str(root/'include'),'-o',str(temporary),'-lm','-lpthread']
    try:
        subprocess.run(command,check=True)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    # Build receipt has no machine-specific paths and no binary contents.
    print(json.dumps(dict(machine=platform.machine(),system=platform.system(),
        source_sha256=hashlib.sha256((root/'row_norm.c').read_bytes()).hexdigest(),
        output=output.name,ort_api_version=23,
        flags=['-O3','-ffp-contract=off','-DFUSED_M2','-DFUSED_OUTPUT'],
        note='Explicit FMA at the same two operations as the measured ORT kernel; no reciprocal approximation.'),indent=2))


if __name__=='__main__':main()
