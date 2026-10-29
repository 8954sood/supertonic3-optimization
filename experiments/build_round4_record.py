"""Summarize round-four raw measurements without replacing historical records."""

import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path


def load(name):
    return json.loads(Path(name).read_text())


def asr(name):
    grouped = defaultdict(lambda: {"errors": 0, "reference_units": 0, "sentences": 0})
    for row in load(name)["rows"]:
        key = (row.get("variant", "remote"), row["language"])
        grouped[key]["errors"] += row["errors"]
        grouped[key]["reference_units"] += row["reference_units"]
        grouped[key]["sentences"] += 1
    return [
        {
            "variant": variant,
            "language": language,
            "metric": "CER" if language == "ko" else "WER",
            **values,
            "error_rate": values["errors"] / values["reference_units"],
        }
        for (variant, language), values in grouped.items()
    ]


def acoustics(name):
    grouped = defaultdict(list)
    for row in load(name)["rows"]:
        grouped[row["variant"]].append(row)
    return [
        {
            "variant": variant,
            "sentences": len(rows),
            "mean_dtw_logmel_mae_db": statistics.mean(
                row["dtw_logmel_mae_db"] for row in rows
            ),
            "mean_sentence_median_f0_abs_cents": statistics.mean(
                row["f0_median_abs_cents"]
                for row in rows
                if row["f0_median_abs_cents"] is not None
            ),
        }
        for variant, rows in grouped.items()
    ]


def main():
    sources = sorted(Path("results").glob("round4_*.json"))
    report = {
        "date_kst": "2026-10-29",
        "default_model": "cfg-mixed-v1-selected",
        "experimental_candidate": "cfg1024-v2",
        "conclusion": "Faster than the balanced model on the 1GB x86 host, but independent Korean CER regressed versus the previous 1024 student. Development and acoustic gains do not justify replacing the default.",
        "steps": 8,
        "inference_threads": 1,
        "asr": {},
        "acoustics": {},
        "server_paired": load("results/round4_paired_expanded.json")["summary"],
        "sampled_peak_rss_mb": {},
        "sources": [],
        "limitations": [
            "Whisper base INT8 CPU1 is an intelligibility proxy, not a listening or speaker similarity score.",
            "English streetlights/street lights accounts for both test WER errors in every model; raw scoring is retained.",
            "The server is shared. Alternating model processes reduce but cannot eliminate host interference. Three pairs are a small sample.",
            "Memory supervision samples every 50ms and is not an OS-enforced cap.",
            "No new candidate was measured on ARM. User-provided PR #1 covers the prior balanced model only.",
            "All round-four test sets are consumed. Do not tune on them.",
        ],
    }
    for name in [
        "round4_baseline_holdout",
        "round4_expanded_preview_validation",
        "round4_expanded_final_validation",
        "round4_expanded_frozen_test",
        "round4_epyc_candidate_asr",
        "round4_epyc_reference_asr",
    ]:
        report["asr"][name] = asr(f"results/{name}.json")
    for name in [
        "round4_expanded_preview_acoustics",
        "round4_expanded_final_acoustics",
        "round4_expanded_frozen_acoustics",
    ]:
        report["acoustics"][name] = acoustics(f"results/{name}.json")
    paired = load("results/round4_paired_expanded.json")
    assert paired["complete"] and len(paired["runs"]) == 6
    for label in ["reference", "candidate"]:
        report["sampled_peak_rss_mb"][label] = max(
            row["guard"]["sampled_peak_child_rss_mb"]
            for row in paired["runs"]
            if row["label"] == label
        )
    for source in sources:
        report["sources"].append(
            {
                "path": str(source),
                "bytes": source.stat().st_size,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        )
    destination = Path("records/2026-10-29/round4_metrics.json")
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Recorded {len(sources)} raw result sources")


if __name__ == "__main__":
    main()
