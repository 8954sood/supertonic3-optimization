"""Alternate reference and candidate to reduce drift between separate runs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import Engine
from benchmark import TEXTS
import argparse
import json
import statistics
import platform
import time
import psutil

p = argparse.ArgumentParser()
p.add_argument("--variant", required=True)
p.add_argument("--repeats", type=int, default=3)
p.add_argument("--output", required=True)
p.add_argument("--affinity", type=int)
a = p.parse_args()
if a.repeats < 2:
    p.error("at least two repeats required")
if a.affinity is not None:
    if not hasattr(psutil.Process(), "cpu_affinity"):
        p.error("CPU affinity unavailable on this OS")
    psutil.Process().cpu_affinity([a.affinity])
Path(a.output).parent.mkdir(parents=True, exist_ok=True)
b = Engine()
c = Engine(variant=a.variant)
report = {
    "machine": platform.platform(),
    "steps": 8,
    "threads": 1,
    "variant": a.variant,
    "affinity": a.affinity,
    "rows": [],
}
for name, (lang, text) in TEXTS.items():
    for e in (b, c):
        e.synthesize(text, lang, 8)
    runs = {"reference": [], "candidate": []}
    for i in range(a.repeats):
        order = [("reference", b), ("candidate", c)]
        if i % 2:
            order.reverse()
        for key, e in order:
            start = time.process_time()
            _, metrics = e.synthesize(text, lang, 8)
            metrics["cpu_seconds"] = time.process_time() - start
            runs[key].append(metrics)
    tb = statistics.median(m["seconds"] for m in runs["reference"])
    tc = statistics.median(m["seconds"] for m in runs["candidate"])
    row = {
        "case": name,
        "runs": runs,
        "reference_seconds": tb,
        "candidate_seconds": tc,
        "speedup": tb / tc,
        "rss_mb": psutil.Process().memory_info().rss / 1e6,
    }
    report["rows"].append(row)
    Path(a.output).write_text(json.dumps(report, indent=2))
    print(name, tb, tc, "speedup", tb / tc, flush=True)
