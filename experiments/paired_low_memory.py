"""Alternate complete model processes so a 1GB host never holds both engines."""

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--cases", nargs="+", default=["ko_short", "en_short"])
    parser.add_argument("--affinity", type=int, default=0)
    parser.add_argument("--max-rss-mb", type=float, default=420)
    parser.add_argument("--reserve-mb", type=float, default=80)
    args = parser.parse_args()
    if args.pairs < 2:
        parser.error("At least two alternating pairs are required")
    destination = Path(args.output)
    runs_directory = destination.parent / (destination.stem + "_runs")
    if destination.exists() or runs_directory.exists():
        parser.error("Choose a new output to preserve earlier measurements")
    runs_directory.mkdir(parents=True)
    report = {
        "reference": args.reference,
        "candidate": args.candidate,
        "models": {
            label: json.loads((Path(model) / "manifest.json").read_text())
            for label, model in [
                ("reference", args.reference),
                ("candidate", args.candidate),
            ]
        },
        "steps": 8,
        "threads": 1,
        "affinity": args.affinity,
        "note": "Alternating fresh processes, each with its own warm-up and one timed run. Loading/file I/O excluded. No concurrent model residency; 50ms memory guard, not a cgroup limit.",
        "runs": [],
        "summary": [],
        "complete": False,
    }
    for pair in range(args.pairs):
        order = [("reference", args.reference), ("candidate", args.candidate)]
        if pair % 2:
            order.reverse()
        for label, model in order:
            raw = runs_directory / f"{destination.stem}_{pair:02d}_{label}.json"
            guard = runs_directory / f"{destination.stem}_{pair:02d}_{label}_guard.json"
            command = [
                sys.executable,
                "memory_guard.py",
                "--max-rss-mb",
                str(args.max_rss_mb),
                "--reserve-mb",
                str(args.reserve_mb),
                "--seconds",
                "600",
                "--stats",
                str(guard),
                "--",
                sys.executable,
                "benchmark.py",
                "--model",
                model,
                "--steps",
                "8",
                "--repeats",
                "1",
                "--cases",
                *args.cases,
                "--affinity",
                str(args.affinity),
                "--output",
                str(raw),
            ]
            result = subprocess.run(command, check=False)
            row = {
                "pair": pair,
                "label": label,
                "returncode": result.returncode,
                "guard": json.loads(guard.read_text()) if guard.exists() else None,
                "benchmark": json.loads(raw.read_text()) if raw.exists() else None,
            }
            report["runs"].append(row)
            destination.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n"
            )
            if result.returncode:
                raise SystemExit(
                    f"Stopped after a failed guarded run: pair={pair}, model={label}"
                )
    for name in args.cases:
        times = {
            label: [
                next(
                    case["median_seconds"]
                    for case in row["benchmark"]["rows"]
                    if case["case"] == name
                )
                for row in report["runs"]
                if row["label"] == label
            ]
            for label in ["reference", "candidate"]
        }
        ratios = [a / b for a, b in zip(times["reference"], times["candidate"])]
        report["summary"].append(
            {
                "case": name,
                "reference_seconds": statistics.median(times["reference"]),
                "candidate_seconds": statistics.median(times["candidate"]),
                "paired_speedup_median": statistics.median(ratios),
                "paired_speedups": ratios,
            }
        )
    report["complete"] = True
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["summary"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
