"""Curate F1200/E400 and only the checkpoints/data needed to reproduce F1200.

Creates a local staging directory; does not authenticate, upload or delete Hub files.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

MODEL_REPO = 'babihoba/supertonic3-optimization'
DATA_REPO = 'babihoba/supertonic3-optimization-data'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def link(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix == '.json':
        shutil.copyfile(source, destination)
    else:
        os.link(source, destination)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    root = Path(a.output)
    if root.exists():
        p.error('Choose a new output directory')
    root.mkdir(parents=True)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    parents = {}
    for kind, repo in [('model', MODEL_REPO), ('dataset', DATA_REPO)]:
        info = HfApi(token=False).repo_info(repo, repo_type=kind)
        parents[kind] = info.sha
        prior = hf_hub_download(repo, 'asset_manifest.json', repo_type=kind, revision=info.sha, token=False)
        dest = root / kind
        dest.mkdir()
        shutil.copyfile(prior, dest / 'asset_manifest.json')
    for name in ['round5-original50-student', 'round5-original50-preview']:
        source = Path('models') / name
        dest = root / 'model/models' / name
        for asset in sorted(source.rglob('*')):
            if asset.is_symlink():
                raise ValueError('Unexpected symlink in model package')
            if asset.is_file() and (asset.name == 'LICENSE' or asset.suffix in {'.json', '.onnx'}):
                link(asset, dest / asset.relative_to(source))
        meta = json.loads((dest / 'manifest.json').read_text())
        assert sha(dest / 'onnx/vector_estimator.onnx') == meta['vector_sha256']
        meta.update(
            status='User-selected fixed F1200 model; engine optimization only' if name.endswith('student') else 'E400 listening comparison; F1200 is selected',
            source_commit=commit,
            human_feedback='E400 and F1200 had no large audible difference on the supplied development listening samples; user selected F1200',
        )
        (dest / 'manifest.json').write_text(json.dumps(meta, indent=2) + '\n')
    reproduction = root / 'model/reproduction/round5'
    for source, name in [
        ('distill/partial-original-round5/initial.pt', 'original50-initial.pt'),
        ('distill/joint-original50-round5/best.pt', 'original50-step-01200.pt'),
        ('distill/partial-original-round5/report.json', 'initialization-report.json'),
        ('distill/joint-original50-round5/report.json', 'training-report.json'),
        ('distill/corpus_round4_train.json', 'corpus.json'),
        ('results/round5_original50_data_audit.json', 'data-audit.json'),
    ]:
        link(Path(source), reproduction / name)
    trajectories = Path('distill/trajectories-original50-round5')
    data = root / 'dataset/round5'
    data.mkdir()
    members = []
    with tarfile.open(data / 'trajectories-original50-round5.tar.gz', 'w:gz') as archive:
        for asset in sorted(trajectories.rglob('*')):
            if not asset.is_file():
                continue
            if asset.is_symlink() or asset.suffix not in {'.json', '.npz'}:
                raise ValueError('Unexpected trajectory asset')
            name = str(Path(trajectories.name) / asset.relative_to(trajectories))
            info = archive.gettarinfo(str(asset), arcname=name)
            info.uid = info.gid = 0
            info.uname = info.gname = ''
            info.mtime = 0
            info.mode = 0o644
            with asset.open('rb') as f:
                archive.addfile(info, f)
            members.append({'path': name, 'bytes': asset.stat().st_size, 'sha256': sha(asset)})
    (data / 'extracted_manifest.json').write_text(json.dumps({'files': members}, indent=2) + '\n')
    link(Path('distill/corpus_round4_train.json'), data / 'corpus.json')
    for kind, key in [('model', 'artifacts'), ('dataset', 'files')]:
        dest = root / kind
        manifest_path = dest / 'asset_manifest.json'
        manifest = json.loads(manifest_path.read_text())
        inventory = {v['path']: v for v in manifest[key]}
        new = []
        for asset in sorted(dest.rglob('*')):
            if not asset.is_file() or asset == manifest_path:
                continue
            item = {'path': str(asset.relative_to(dest)), 'bytes': asset.stat().st_size, 'sha256': sha(asset)}
            inventory[item['path']] = item
            new.append(item)
        manifest[key] = list(inventory.values())
        manifest['round5'] = {
            'source_commit': commit, 'selected_model': 'models/round5-original50-student',
            'teacher': 'Pinned original FP32 CFG4 50 steps; no distilled teacher',
            'student_steps': 8, 'train_pairs': 640, 'validation_pairs': 128,
            'new_test_evaluated': False,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        print(kind, len(new), 'new assets,', sum(v['bytes'] for v in new), 'logical bytes')
    (root / 'parents.json').write_text(json.dumps(parents, indent=2) + '\n')
    print(root)


if __name__ == '__main__':
    main()
