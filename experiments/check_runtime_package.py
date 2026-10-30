"""Build/extract a source-only runtime package, then compare real native inference."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import argparse
import hashlib
import json
import subprocess
import tarfile
import tempfile
from runtime import Engine


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--reference',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Choose a new output')
    # Preserve the virtualenv path: resolving this symlink selects system Python.
    python=sys.executable
    report=dict(model=args.model.name,reference=Path(args.reference).name)
    with tempfile.TemporaryDirectory(prefix='row-norm-package-') as temporary:
        archive=Path(temporary)/'runtime.tar.gz'
        subprocess.run([python,str(ROOT/'package_runtime.py'),'--model',str(args.model),
            '--output',str(archive)],cwd=ROOT,check=True,capture_output=True,text=True)
        report['archive_bytes']=archive.stat().st_size
        extracted=Path(temporary)/'extracted';extracted.mkdir()
        with tarfile.open(archive) as tar:
            names=tar.getnames()
            report['contains_native_sources']=all(n in names for n in
                ['row_norm_ops.py','native/build_row_norm.py','native/row_norm.c'])
            report['contains_native_binaries']=any(n.endswith(('.dylib','.so','.dll')) for n in names)
            if not report['contains_native_sources'] or report['contains_native_binaries']:
                raise AssertionError('Expected source-only native distribution')
            tar.extractall(extracted,filter='data')
        build=subprocess.run([python,'native/build_row_norm.py'],cwd=extracted,
                             check=True,capture_output=True,text=True)
        report['native_build']=json.loads(build.stdout)
        code=("from runtime import Engine;import hashlib,json;"
              "e=Engine('model');w,m=e.synthesize('패키지 실행 검사입니다.',lang='ko',steps=8,speed=1.25);"
              "print(json.dumps(dict(sha256=hashlib.sha256(w.tobytes()).hexdigest(),samples=len(w),row_norm=e.row_norm_backend)))")
        child=subprocess.run([python,'-c',code],cwd=extracted,check=True,capture_output=True,text=True)
        report['extracted_runtime']=json.loads(child.stdout)
    reference=Engine(args.reference)
    wave,_=reference.synthesize('패키지 실행 검사입니다.',lang='ko',steps=8,speed=1.25)
    report['float32_bitidentical']=hashlib.sha256(wave.tobytes()).hexdigest()==report['extracted_runtime']['sha256']
    if not report['float32_bitidentical']:raise AssertionError('Packaged waveform changed')
    report['temporary_archive_and_extraction_removed']=True
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('Standalone source-only native package builds and matches reference waveform')


if __name__=='__main__':main()
