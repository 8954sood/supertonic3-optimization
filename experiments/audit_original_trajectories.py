"""Validate original-teacher provenance and continuity of the eight-step training grid."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import hashlib
import json
from collections import Counter, defaultdict

import numpy as np

from distill.teacher_policy import assert_original_data
from distill.trajectory_data import trajectory_path

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
root = Path(a.data)
metadata = json.loads((root / "manifest.json").read_text())
assert_original_data(metadata)
if not metadata.get("complete"):
    raise ValueError("Complete original-teacher collection required")
cases = defaultdict(dict)
total_bytes = 0
file_hashes = []
for row in metadata["rows"]:
    path = trajectory_path(root, row)
    with np.load(path) as stored:
        arrays = {name: stored[name] for name in stored.files}
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Nonfinite trajectory data")
    assert arrays["current_step"].tolist() == [row["step"]]
    assert arrays["total_step"].tolist() == [8.0]
    key = (row["split"], row["case"])
    if row["step"] in cases[key]:
        raise ValueError("Duplicate student step")
    cases[key][row["step"]] = (arrays["noisy_latent"], arrays["target"])
    total_bytes += path.stat().st_size
    file_hashes.append(
        (str(path.relative_to(root)), hashlib.sha256(path.read_bytes()).hexdigest())
    )
checks = 0
for key, states in cases.items():
    assert sorted(states) == list(range(8)), key
    for i in range(7):
        if not np.array_equal(states[i][1], states[i + 1][0]):
            raise ValueError("Discontinuous teacher grid")
        checks += 1
report = {
    "dataset": a.data,
    "teacher_origin": metadata["teacher_origin"],
    "student_steps": 8,
    "pairs": dict(Counter(row["split"] for row in metadata["rows"])),
    "cases": dict(Counter(split for split, case in cases)),
    "adjacent_targets_equal_next_input": checks,
    "all_arrays_finite": True,
    "student_step_metadata_valid": True,
    "npz_total_bytes": total_bytes,
    "inventory_sha256": hashlib.sha256(
        json.dumps(sorted(file_hashes)).encode()
    ).hexdigest(),
    "manifest_sha256": hashlib.sha256(
        (root / "manifest.json").read_bytes()
    ).hexdigest(),
    "note": "Every student target is interpolated from original 50-step states; last target is original final state. No derived teacher, no test data.",
}
Path(a.output).write_text(json.dumps(report, indent=2) + "\n")
print(
    "PASS:",
    len(metadata["rows"]),
    "original-only pairs;",
    checks,
    "continuity checks",
    flush=True,
)
