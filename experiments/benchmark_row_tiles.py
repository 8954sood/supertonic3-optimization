"""Balanced warm C-kernel screening; this does not measure complete TTS RTF."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import runtime  # Establish one-thread settings before importing numerical libraries.
import argparse
import ctypes
import json
import platform
import statistics
import time
import numpy as np
from row_norm_ops import library_path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library',action='append',required=True,help='label=relative-library-path')
    parser.add_argument('--frames',type=int,nargs='+',default=[5,17,33,81,257])
    parser.add_argument('--repeats',type=int,default=80)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or args.repeats<20:parser.error('Use a new output and at least 20 repeats')
    entries={}
    for item in args.library:
        name,separator,path=item.partition('=')
        if not separator or not name or name in entries:parser.error('Use unique label=path entries')
        entries[name]=path
    handles,functions={},{}
    for name,path in entries.items():
        handles[name]=ctypes.CDLL(library_path(path))
        function=handles[name].row_norm
        function.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int64,ctypes.c_int64,ctypes.c_float]
        function.restype=None;functions[name]=function
    rng=np.random.RandomState(714)
    report=dict(machine=platform.platform(),libraries=entries,
        scope='Single-thread direct C calls, not full TTS RTF. Order rotates through all positions, then reverses.',
        repeats=args.repeats,rows=[],complete=False)
    for frames in args.frames:
        if frames<1:parser.error('Frames must be positive')
        x=rng.randn(1,frames,512).astype(np.float32)
        scale=rng.randn(512).astype(np.float32);bias=rng.randn(512).astype(np.float32)
        outputs={name:np.empty_like(x) for name in functions};runs={name:[] for name in functions}
        def run(name):
            functions[name](x.ctypes.data,scale.ctypes.data,bias.ctypes.data,
                            outputs[name].ctypes.data,frames,512,1e-6)
        for _ in range(4):
            for name in functions:run(name)
        expected=next(iter(outputs.values())).tobytes()
        for trial in range(args.repeats):
            names=list(functions);offset=trial%len(names);names=names[offset:]+names[:offset]
            if (trial//len(names))%2:names.reverse()
            for name in names:
                start=time.perf_counter();run(name);runs[name].append(time.perf_counter()-start)
        for output in outputs.values():
            if output.tobytes()!=expected:raise AssertionError('Float32 byte mismatch')
        row=dict(frames=frames,float32_bytes_identical=True,
                 median_microseconds={name:statistics.median(values)*1e6 for name,values in runs.items()})
        report['rows'].append(row);print(row,flush=True)
    report['complete']=True
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
