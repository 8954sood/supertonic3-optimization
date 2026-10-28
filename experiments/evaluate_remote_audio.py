"""Evaluate files actually rendered on the remote CPU, without any TTS on this host."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark import TEXTS
from experiments.asr_check import normalize, distance
from faster_whisper import WhisperModel
import argparse
import json

p = argparse.ArgumentParser()
p.add_argument("directory")
p.add_argument("--output", required=True)
a = p.parse_args()
model = WhisperModel(
    "models/whisper-base",
    device="cpu",
    compute_type="int8",
    cpu_threads=1,
    num_workers=1,
    local_files_only=True,
)
rows = []
for wav in sorted(Path(a.directory).glob("*.wav")):
    case = wav.stem.split("_8steps")[0]
    lang, text = TEXTS[case]
    segments, _ = model.transcribe(
        str(wav),
        language=lang,
        beam_size=5,
        temperature=0.0,
        vad_filter=False,
        condition_on_previous_text=False,
    )
    transcript = " ".join(s.text.strip() for s in segments)
    ref = normalize(text)
    hyp = normalize(transcript)
    units = list(ref.replace(" ", "")) if lang == "ko" else ref.split()
    pred = list(hyp.replace(" ", "")) if lang == "ko" else hyp.split()
    errors = distance(units, pred)
    row = {
        "file": str(wav),
        "case": case,
        "language": lang,
        "reference": text,
        "transcript": transcript,
        "errors": errors,
        "reference_units": len(units),
        "error_rate": errors / len(units),
    }
    rows.append(row)
    print(row, flush=True)
Path(a.output).write_text(
    json.dumps(
        {
            "note": "Two benchmark sentences rendered on the actual remote x86 CPU. Small intelligibility smoke check only.",
            "rows": rows,
        },
        ensure_ascii=False,
        indent=2,
    )
)
