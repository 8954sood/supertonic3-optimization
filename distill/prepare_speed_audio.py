"""Prepare fast waveform supervision from pinned original50 normal trajectories."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import hashlib
import json
import numpy as np
import soundfile as sf
import onnxruntime as ort
from runtime import options
from supertonic.core import UnicodeProcessor
from supertonic.loader import load_voice_style_from_json_file
from distill.teacher_policy import assert_original_data, assert_original_model
from distill.fast_reference import rubberband_reference
from experiments.compare_original_speed import EXPECTED


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path('distill/trajectories-original50-round5'))
    p.add_argument('--output', type=Path, default=Path('distill/fast-original50-round8'))
    a = p.parse_args()
    if a.output.exists():
        p.error('Choose a new output')
    source = json.loads((a.source/'manifest.json').read_text())
    assert_original_data(source)
    assert source['teacher_origin']['steps'] == 50
    root = Path('models/supertonic-3')
    assert_original_model(root)
    for name, digest in EXPECTED.items():
        assert hashlib.sha256((root/'onnx'/f'{name}.onnx').read_bytes()).hexdigest() == digest
    settings = options()
    settings.intra_op_num_threads = 4
    sessions = {name: ort.InferenceSession(str(root/'onnx'/f'{name}.onnx'), settings,
                providers=['CPUExecutionProvider']) for name in ['vocoder','duration_predictor']}
    processor = UnicodeProcessor(str(root/'onnx/unicode_indexer.json'))
    corpus = json.loads(Path(source['corpus']).read_text())
    cfg = json.loads((root/'onnx/tts.json').read_text())
    sr = cfg['ae']['sample_rate']
    chunk = cfg['ae']['base_chunk_size']*cfg['ttl']['chunk_compress_factor']
    report = {
        'teacher_origin': source['teacher_origin'],
        'source_manifest_sha256': hashlib.sha256((a.source/'manifest.json').read_bytes()).hexdigest(),
        'source': str(a.source), 'original_normal_speed': 1.05,
        'target_speed': 2.0, 'method': 'Original50 latent decoded unchanged and cropped to original duration; training-only offline R3 time compression. No student-generated target or native2 teacher audio.',
        'rows': [], 'complete': False,
    }
    a.output.mkdir(parents=True)
    for row in source['rows']:
        if row['step'] != 0:
            continue
        split, case = row['split'], row['case']
        folder = a.output/split
        folder.mkdir(exist_ok=True)
        language, text = corpus[split][case]
        ids, mask = processor([text], language)
        style = load_voice_style_from_json_file(root/'voice_styles'/f"{row['voice']}.json")
        duration = float(sessions['duration_predictor'].run(None,
            {'text_ids': ids, 'text_mask': mask, 'style_dp': style.dp})[0][0])
        stored = a.source/split/f'{case:03d}_7.npz'
        target = np.load(stored)['target']
        normal = sessions['vocoder'].run(None, {'latent': target})[0].reshape(-1)[:int(duration/1.05*sr)]
        normal_path = folder/f'{case:03d}_normal.wav'
        sf.write(normal_path, normal, sr, subtype='FLOAT')
        fast, transform = rubberband_reference(normal, sr, source_speed=1.05, target_speed=2.0)
        fast_path = folder/f'{case:03d}_fast.wav'
        sf.write(fast_path, fast, sr, subtype='FLOAT')
        first = dict(np.load(a.source/split/f'{case:03d}_0.npz'))
        first.pop('target')
        length = (int(duration/2*sr)+chunk-1)//chunk
        first['latent_mask'] = np.ones((1,1,length), np.float32)
        first['noisy_latent'] = np.random.RandomState(row['seed']).randn(1, target.shape[1], length).astype(np.float32)
        feed_path = folder/f'{case:03d}_feed.npz'
        np.savez_compressed(feed_path, **first)
        report['rows'].append({
            **{k: row[k] for k in ['split','case','voice','language','seed']},
            'variant': 'original50-R3-fast-reference', 'text': text,
            'file': str(feed_path), 'audio': str(fast_path), 'reference_audio': str(normal_path),
            'source_final_latent_sha256': hashlib.sha256(stored.read_bytes()).hexdigest(),
            'sha256': hashlib.sha256(fast_path.read_bytes()).hexdigest(),
            'normal_sha256': hashlib.sha256(normal_path.read_bytes()).hexdigest(),
            'audio_seconds': len(fast)/sr, 'native2_length_error_samples': len(fast)-int(duration/2*sr),
            'transform': transform,
        })
        (a.output/'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        print(split,case,row['voice'],round(len(fast)/sr,3),flush=True)
    report['complete'] = True
    (a.output/'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
