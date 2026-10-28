"""Recognize existing paired audio; no TTS generation or reuse of reserved test texts."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import Engine  # noqa: F401 -- establish CPU thread environment
import argparse, json
from faster_whisper import WhisperModel
from experiments.asr_check import normalize, distance

p = argparse.ArgumentParser()
p.add_argument("manifest")
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
report = {
    "source": a.manifest,
    "note": "Development/validation only. ASR is not naturalness or timbre.",
    "rows": [],
}
for row in json.loads(Path(a.manifest).read_text())["rows"]:
    segments, _ = model.transcribe(
        row["audio"],
        language=row["language"],
        beam_size=5,
        temperature=0.0,
        vad_filter=False,
        condition_on_previous_text=False,
    )
    hyp = " ".join(s.text.strip() for s in segments)
    truth = normalize(row["text"])
    pred = normalize(hyp)
    truth, pred = (
        (list(truth.replace(" ", "")), list(pred.replace(" ", "")))
        if row["language"] == "ko"
        else (truth.split(), pred.split())
    )
    result = {k: row[k] for k in ["variant", "case", "language", "voice", "audio"]}
    result.update(
        reference=row["text"],
        transcript=hyp,
        errors=distance(truth, pred),
        reference_units=len(truth),
    )
    report["rows"].append(result)
    Path(a.output).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(result["variant"], result["case"], result["errors"], hyp, flush=True)
