"""Snapshot existing measurements without running inference or touching credentials."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from experiments.asr_check import distance, normalize

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "records/2026-10-28"
DEST.mkdir(parents=True, exist_ok=True)
performance = []
asr = []
agreement = []
waveform = []
source_index = []
for path in sorted((ROOT / "results").glob("*.json")):
    if "profile_" in path.name:
        continue
    raw = path.read_bytes()
    result = json.loads(raw)
    source = str(path.relative_to(ROOT))
    source_index.append(
        {"path": source, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    )
    if not isinstance(result, dict):
        continue
    rows = result.get("rows", [])
    if rows and "median_seconds" in rows[0]:
        for row in rows:
            performance.append(
                {
                    "source": source,
                    "model": result.get("model"),
                    "variant": result.get("variant"),
                    "machine": result.get("machine"),
                    "case": row["case"],
                    "steps": row.get("steps"),
                    "repetitions": len(row.get("runs", [])),
                    "median_seconds": row["median_seconds"],
                    "median_rtf": row.get("median_rtf"),
                    "median_realtime_factor": row.get("median_realtime_factor"),
                    "cpu_wall_ratio": row.get("cpu_wall_ratio"),
                    "rss_mb": row.get("rss_mb"),
                }
            )
    if rows and "correlation" in rows[0]:
        waveform.extend({"source": source, **r} for r in rows)
    groups = defaultdict(
        lambda: {"errors": 0, "units": 0, "sentences": 0, "zero_error_sentences": 0}
    )
    references = {
        r.get("case"): r
        for r in rows
        if r.get("variant") == "reference" and "transcript" in r
    }
    agree = defaultdict(lambda: {"edits": 0, "reference_transcript_units": 0})
    for row in rows:
        if "errors" not in row or "reference_units" not in row:
            continue
        variant, lang = row.get("variant", "remote_audio"), row["language"]
        group = groups[(variant, lang)]
        group["errors"] += row["errors"]
        group["units"] += row["reference_units"]
        group["sentences"] += 1
        group["zero_error_sentences"] += row["errors"] == 0
        if variant != "reference" and row.get("case") in references:
            base = normalize(references[row["case"]]["transcript"])
            hyp = normalize(row["transcript"])
            base, hyp = (
                (list(base.replace(" ", "")), list(hyp.replace(" ", "")))
                if lang == "ko"
                else (base.split(), hyp.split())
            )
            group2 = agree[(variant, lang)]
            group2["edits"] += distance(base, hyp)
            group2["reference_transcript_units"] += len(base)
    for (variant, lang), group in groups.items():
        asr.append(
            {
                "source": source,
                "variant": variant,
                "language": lang,
                "metric": "CER_no_spaces" if lang == "ko" else "WER",
                **group,
                "error_rate": group["errors"] / group["units"]
                if group["units"]
                else None,
            }
        )
    for (variant, lang), group in agree.items():
        agreement.append(
            {
                "source": source,
                "variant": variant,
                "language": lang,
                **group,
                "agreement_1_minus_normalized_edit": 1
                - group["edits"] / max(group["reference_transcript_units"], 1),
            }
        )
training = []
for path in sorted((ROOT / "distill").glob("*/report.json")):
    report = json.loads(path.read_text())
    training.append(
        {
            "source": str(path.relative_to(ROOT)),
            "config": report["config"],
            "completed_layers": len(report["layers"]),
            "summed_layer_seconds": sum(r["seconds"] for r in report["layers"]),
            "layers": [
                {
                    k: r[k]
                    for k in [
                        "layer",
                        "best_float_relative_rmse",
                        "best_qat_relative_rmse",
                    ]
                }
                for r in report["layers"]
            ],
        }
    )
inventory = []
for directory in sorted((ROOT / "models").iterdir()):
    if not directory.is_dir():
        continue
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    inventory.append(
        {
            "path": str(directory.relative_to(ROOT)),
            "bytes": sum(p.stat().st_size for p in directory.rglob("*") if p.is_file()),
            "manifest": manifest,
        }
    )
result = {
    "date_kst": "2026-10-28",
    "note": "Snapshot of existing measurements. No inference was run. ASR agreement is not waveform or perceived quality agreement. Earlier heldout labels in raw files refer to reused development sentences.",
    "performance": performance,
    "asr": asr,
    "transcript_agreement_to_reference": agreement,
    "waveform": waveform,
    "training": training,
}
(DEST / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
(DEST / "model_inventory.json").write_text(
    json.dumps(inventory, ensure_ascii=False, indent=2)
)
(DEST / "source_index.json").write_text(
    json.dumps(source_index, ensure_ascii=False, indent=2)
)
print(
    "Recorded",
    len(performance),
    "performance rows,",
    len(asr),
    "ASR groups,",
    len(training),
    "training runs,",
    len(inventory),
    "model directories",
)
