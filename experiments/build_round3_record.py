"""Keep round-three raw provenance separately from the frozen earlier rounds."""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

root = Path.cwd()
files = sorted({*root.glob("results/*cfg*.json"), *root.glob("results/*decoder*.json")})
report = {
    "updated_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
    "note": "All CFG-distilled students use eight real iterations but approximate teacherCFG4 with one conditional branch; not mathematically originalCFG4. Development tests are not perceptual quality approval.",
    "sources": [],
    "training": [],
    "models": [],
}
for p in files:
    raw = p.read_bytes()
    report["sources"].append(
        {
            "source": str(p.relative_to(root)),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "data": json.loads(raw),
        }
    )
for p in sorted((root / "distill").glob("*/report.json")):
    if "cfg" in p.parent.name or "decoder" in p.parent.name:
        report["training"].append(
            {"source": str(p.relative_to(root)), "data": json.loads(p.read_text())}
        )
for p in sorted((root / "models").glob("cfg*/manifest.json")):
    report["models"].append(
        {"path": str(p.parent.relative_to(root)), "manifest": json.loads(p.read_text())}
    )
(root / "records/2026-10-28/round3_metrics.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2)
)
print("recorded", len(files), "sources", len(report["training"]), "training runs")
