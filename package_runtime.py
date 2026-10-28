"""Package an explicit model and inference files; training data and keys stay outside."""

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--model", default="models/mixed-v1-selected")
p.add_argument("--output", default="dist/mixed-v1-quality-runtime.tar.gz")
a = p.parse_args()
model = Path(a.model)
output = Path(a.output)
if not (model / "onnx/vector_estimator.onnx").is_file():
    p.error("Expected a complete model package")
output.parent.mkdir(parents=True, exist_ok=True)
metadata = json.loads((model / "manifest.json").read_text())
guidance = metadata.get("guidance_mode", "original CFG4 structure")
readme = f"""# Supertonic 3: CPU 한 스레드 · 8스텝 실험 후보

Python 3.12 이상 가상환경에서 requirements.txt를 설치한 뒤 실행합니다.

python synthesize.py "안녕하세요. 원본과 소리를 비교합니다." --model model --lang ko --steps 8 --output output.wav

모델: {model.name}
가이던스: {guidance}
지원 스텝: {metadata.get("supported_steps", "8..50")}
실제 denoising을 8회 이상 실행합니다. CFG 증류 모델은 원본 CFG4의 학습된 근사이며 수학적으로 같지 않습니다.
전체 언어/음색의 청취 동등성을 보장하지 않습니다. ARM 실측은 없습니다.
자세한 성능/ASR/실패/후속 작업은 포함된 최신 인계 문서와 model/manifest.json을 확인하세요.
문서의 workspace 파일은 이 최소 실행 패키지에 모두 포함되지는 않습니다.
모델의 원본 OpenRAIL-M 라이선스는 model/LICENSE에 유지했습니다.
"""
with tarfile.open(output, "w:gz") as archive:
    for name in ["runtime.py", "synthesize.py", "requirements.txt"]:
        archive.add(name, arcname=name)
    for asset in sorted(model.rglob("*")):
        relative = asset.relative_to(model)
        if asset.is_symlink():
            raise ValueError(f"Model symlinks are not allowed: {relative}")
        if (
            asset.is_file()
            and (asset.name == "LICENSE" or asset.suffix in {".json", ".onnx", ".ort"})
            and not any(part.startswith(".") for part in relative.parts)
        ):
            archive.add(asset, arcname=str(Path("model") / relative))
    for name in ["ROUND2.md", "HANDOFF_ROUND2.md", "ROUND3.md", "HANDOFF_ROUND3.md"]:
        archive.add(Path("records/2026-10-28") / name, arcname=name)
    for name in ["LICENSE", "THIRD_PARTY_NOTICES.md"]:
        archive.add(name, arcname=name)
    archive.add("docs/licenses", arcname="docs/licenses")
    raw = readme.encode()
    info = tarfile.TarInfo("README.md")
    info.size = len(raw)
    archive.addfile(info, io.BytesIO(raw))
digest = hashlib.sha256(output.read_bytes()).hexdigest()
checksum = output.parent / (output.name.removesuffix(".tar.gz") + ".sha256")
checksum.write_text(f"{digest}  {output.name}\n")
print(output, output.stat().st_size, "bytes", digest)
