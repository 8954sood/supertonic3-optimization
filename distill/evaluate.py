"""Paired multi-voice intelligibility evaluation; ASR is only a quality proxy."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import Engine
import argparse
import gc
import json
import soundfile as sf
from experiments.asr_check import normalize, distance
from faster_whisper import WhisperModel

p = argparse.ArgumentParser()
p.add_argument("--variants", nargs="+", required=True)
p.add_argument("--split", choices=["validation", "test"], default="validation")
p.add_argument("--output", required=True)
p.add_argument("--generate-only", action="store_true")
p.add_argument("--corpus", default="distill/corpus.json")
p.add_argument("--reference-steps", type=int, default=8)
a = p.parse_args()
if not 8 <= a.reference_steps <= 50:
    p.error("reference-steps must be from 8 to 50")
texts = json.loads(Path(a.corpus).read_text())[a.split]
out = Path(a.output)
samples = Path("samples") / out.stem
samples.mkdir(parents=True, exist_ok=True)
variants = [("reference", None)] + [(Path(v).name, v) for v in a.variants]
voices = ["M1", "F1", "M3", "F3", "M5", "F5"]
cases = []
for name, variant in variants:
    for i, (lang, text) in enumerate(texts):
        voice = voices[i % len(voices)]
        path = samples / f"{name}_{i:02d}.wav"
        e = Engine(model=variant, voice=voice) if variant else Engine(voice=voice)
        wav, timing = e.synthesize(
            text, lang, a.reference_steps if name == "reference" else 8, seed=400 + i
        )
        sf.write(path, wav, e.sample_rate, subtype="FLOAT")
        cases.append(
            {
                "variant": name,
                "case": i,
                "language": lang,
                "reference": text,
                "voice": voice,
                "seed": 400 + i,
                "audio": str(path),
                "timing": timing,
            }
        )
        del e
        gc.collect()
        print("generated", name, i, flush=True)
report = {
    "split": a.split,
    "steps": 8,
    "reference_guidance_scale": 4,
    "reference_steps": a.reference_steps,
    "corpus": a.corpus,
    "note": "ASR proxy only; does not assess timbre, prosody or listening quality. Test split must not be reused for tuning.",
    "rows": cases,
    "models": {
        name: json.loads((Path(variant) / "manifest.json").read_text())
        if variant and (Path(variant) / "manifest.json").exists()
        else {"mode": "original FP32", "guidance_scale": 4, "steps": a.reference_steps}
        for name, variant in variants
    },
}
for row in cases:
    row["reference_audio"] = str(samples / f"reference_{row['case']:02d}.wav")
    row["text"] = row["reference"]
out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
if a.generate_only:
    sys.exit(0)
model = WhisperModel(
    "models/whisper-base",
    device="cpu",
    compute_type="int8",
    cpu_threads=1,
    num_workers=1,
    local_files_only=True,
)
for row in cases:
    segments, _ = model.transcribe(
        row["audio"],
        language=row["language"],
        beam_size=5,
        temperature=0.0,
        vad_filter=False,
        condition_on_previous_text=False,
    )
    hyp = " ".join(s.text.strip() for s in segments)
    truth, predicted = normalize(row["reference"]), normalize(hyp)
    if row["language"] == "ko":
        truth, predicted = (
            list(truth.replace(" ", "")),
            list(predicted.replace(" ", "")),
        )
    else:
        truth, predicted = truth.split(), predicted.split()
    row.update(
        transcript=hyp, errors=distance(truth, predicted), reference_units=len(truth)
    )
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(row["variant"], row["case"], row["errors"], hyp, flush=True)
