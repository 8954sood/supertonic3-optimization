"""Inspect target-CPU optimized VE graphs; this does not run or time synthesis."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import runtime
import argparse
import collections
import gc
import hashlib
import json
import tempfile
from unittest.mock import patch
import onnx
import onnxruntime as ort
import psutil


def digest(path):
    with Path(path).open('rb') as source:return hashlib.file_digest(source,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['reference','candidate','reference-library','candidate-library']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Choose a new output')
    if hasattr(psutil.Process(),'cpu_affinity'):psutil.Process().cpu_affinity([0])
    report=dict(note='Session initialization and graph inspection only. No inference/RTF measurements.',rows=[],complete=False)
    for name,model,lib in [('reference',a.reference,a.reference_library),('candidate',a.candidate,a.candidate_library)]:
        source=model/'onnx/vector_estimator.onnx'
        with tempfile.TemporaryDirectory(prefix='inspect-native-') as temporary:
            optimized=Path(temporary)/'optimized.onnx'
            with patch('exact_ops.library_path',return_value=str(lib.resolve())):
                settings=runtime.options(exact_int8_backend=True)
            settings.optimized_model_filepath=str(optimized)
            session=ort.InferenceSession(str(source),settings,providers=['CPUExecutionProvider'])
            del session;gc.collect()
            graph=onnx.load(optimized)
            keys=[dict(name=n.name,op_type=n.op_type) for n in graph.graph.node if '/attention/W_key/' in n.name]
            report['rows'].append(dict(name=name,source_vector_sha256=digest(source),library_sha256=digest(lib),optimized_ops=dict(collections.Counter(n.op_type for n in graph.graph.node)),remaining_static_key_projection_nodes=keys))
            del graph;gc.collect()
    report['complete']=True;a.output.write_text(json.dumps(report,indent=2)+'\n')
    print([(r['name'],len(r['remaining_static_key_projection_nodes'])) for r in report['rows']])

if __name__=='__main__':main()
