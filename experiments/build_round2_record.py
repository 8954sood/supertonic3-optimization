"""Build a separate round-two measurement snapshot; preserve the first snapshot."""

import json, hashlib
from pathlib import Path
from collections import defaultdict

root = Path.cwd()
dest = root / "records/2026-10-28"
files = sorted(
    {
        *root.glob("results/*_v2.json"),
        *root.glob("results/*joint*.json"),
        *root.glob("results/*rollout*.json"),
        *root.glob("results/*parity*.json"),
        *root.glob("results/*mixed*.json"),
    }
)
report = {
    "round": 2,
    "note": "Validation/development results, not a held-out evaluation or perceptual quality score.",
    "files": [],
    "acoustic_summary": [],
    "asr_summary": [],
    "performance": [],
    "training": [],
}
for path in files:
    raw = path.read_bytes()
    r = json.loads(raw)
    source = str(path.relative_to(root))
    report["files"].append({"path": source, "sha256": hashlib.sha256(raw).hexdigest()})
    if not isinstance(r, dict):
        continue
    groups = defaultdict(list)
    for row in r.get("rows", []):
        groups[row.get("variant", "unknown")].append(row)
    for name, rows in groups.items():
        if "median_seconds" in rows[0]:
            report["performance"].extend(
                {"source": source, "model": r.get("model"), **row} for row in rows
            )
        if "dtw_logmel_mae_db" in rows[0]:
            keys = [
                "dtw_logmel_mae_db",
                "duration_relative_error",
                "voiced_unvoiced_agreement",
                "f0_median_abs_cents",
                "f0_gross_error_over_200_cents",
            ]
            report["acoustic_summary"].append(
                {
                    "source": source,
                    "variant": name,
                    "cases": len(rows),
                    **{
                        k: sum(x[k] for x in rows if x[k] is not None)
                        / max(sum(x[k] is not None for x in rows), 1)
                        for k in keys
                    },
                }
            )
        if "errors" in rows[0]:
            for lang in ["ko", "en"]:
                selected = [x for x in rows if x["language"] == lang]
                if selected:
                    report["asr_summary"].append(
                        {
                            "source": source,
                            "variant": name,
                            "language": lang,
                            "errors": sum(x["errors"] for x in selected),
                            "units": sum(x["reference_units"] for x in selected),
                            "cases": len(selected),
                        }
                    )
for pattern in ["joint*/report.json", "rollout*/report.json"]:
    for path in sorted((root / "distill").glob(pattern)):
        r = json.loads(path.read_text())
        report["training"].append({"source": str(path.relative_to(root)), **r})
(dest / "round2_metrics.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2)
)
print("Recorded", len(files), "sources and", len(report["training"]), "training runs")
