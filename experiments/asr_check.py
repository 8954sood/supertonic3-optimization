"""Small paired intelligibility check. ASR is a proxy, not a quality guarantee."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import Engine
import argparse
import json
import re
import unicodedata
import soundfile as sf
from faster_whisper import WhisperModel

DEV_TEXTS = [
    ("ko", "주말 아침에는 창문을 열고 따뜻한 차 한 잔을 마십니다."),
    ("ko", "작은 도서관에서 빌린 책을 가방에 넣고 천천히 집으로 돌아왔습니다."),
    ("ko", "비가 그친 뒤 맑은 하늘 아래로 아이들의 웃음소리가 들려왔습니다."),
    ("en", "Please leave the package beside the blue door and ring the bell."),
    ("en", "We walked along the river until the evening lights appeared."),
    ("en", "A small bird landed on the garden fence and began to sing."),
]


def normalize(s):
    return re.sub(r"[^\w\s]", "", unicodedata.normalize("NFKC", s).lower()).strip()


def distance(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        nxt = [i]
        for j, y in enumerate(b, 1):
            nxt.append(min(nxt[-1] + 1, row[j] + 1, row[j - 1] + (x != y)))
        row = nxt
    return row[-1]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--variants",
        nargs="+",
        default=["models/fast-fp32-cached/onnx", "models/fast-denoiser-int8/onnx"],
    )
    p.add_argument("--output", default="results/asr_check.json")
    p.add_argument("--generate-only", action="store_true")
    p.add_argument(
        "--asr-model",
        default="models/whisper-base"
        if Path("models/whisper-base").is_dir()
        else "base",
    )
    a = p.parse_args()
    root = Path("samples/asr")
    root.mkdir(parents=True, exist_ok=True)
    variants = [None] + a.variants
    for variant in variants:
        name = (
            (
                Path(variant).parent.name
                if Path(variant).name == "onnx"
                else Path(variant).name
            )
            if variant
            else "reference"
        )
        e = Engine(variant=variant)
        for i, (lang, text) in enumerate(DEV_TEXTS):
            wav, _ = e.synthesize(text, lang, 8, seed=42)
            sf.write(root / f"{name}_{i}.wav", wav, e.sample_rate)
            print("generated", name, i, flush=True)
        del e
    if a.generate_only:
        sys.exit(0)
    model = WhisperModel(
        a.asr_model,
        device="cpu",
        compute_type="int8",
        cpu_threads=1,
        num_workers=1,
        local_files_only=Path(a.asr_model).is_dir(),
    )
    report = {
        "asr": f"{a.asr_model} int8 CPU one thread",
        "steps": 8,
        "voice": "M1",
        "seed": 42,
        "note": "Six development smoke sentences reused during experiments, one voice and one seed. This is not an independent held-out test. ASR does not measure timbre or prosody.",
        "rows": [],
    }
    out = Path(a.output)
    for variant in variants:
        name = (
            (
                Path(variant).parent.name
                if Path(variant).name == "onnx"
                else Path(variant).name
            )
            if variant
            else "reference"
        )
        for i, (lang, text) in enumerate(DEV_TEXTS):
            segments, _ = model.transcribe(
                str(root / f"{name}_{i}.wav"),
                language=lang,
                beam_size=5,
                temperature=0.0,
                vad_filter=False,
                condition_on_previous_text=False,
            )
            hypothesis = " ".join(s.text.strip() for s in segments)
            truth = normalize(text)
            hyp = normalize(hypothesis)
            units = list(truth.replace(" ", "")) if lang == "ko" else truth.split()
            predicted = list(hyp.replace(" ", "")) if lang == "ko" else hyp.split()
            errors = distance(units, predicted)
            row = {
                "variant": name,
                "case": i,
                "language": lang,
                "reference": text,
                "transcript": hypothesis,
                "metric": "CER" if lang == "ko" else "WER",
                "errors": errors,
                "reference_units": len(units),
                "error_rate": errors / len(units),
            }
            report["rows"].append(row)
            out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
            print(row, flush=True)
