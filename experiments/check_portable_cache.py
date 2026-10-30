"""Check exact output at fractional speeds and long text, not ASR quality."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime import Engine
import argparse
import hashlib
import json
import numpy as np
from supertonic.loader import load_voice_style_from_json_file
from benchmark import TEXTS
from experiments.check_streaming_runtime import CountedSession
from experiments.row_norm_variant import engine_for_library


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=list(TEXTS), default=["ko_long", "en_long"])
    parser.add_argument("--speeds", nargs="+", type=float, default=[0.70, 1.25, 1.30, 1.50, 2.0])
    parser.add_argument("--reference-library")
    parser.add_argument("--candidate-library")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output")
    engines = {name: engine_for_library(path,library) for name,path,library in
               [("reference",args.reference,args.reference_library),("candidate",args.candidate,args.candidate_library)]}
    for engine in engines.values():
        engine.sessions["vector_estimator"] = CountedSession(engine.sessions["vector_estimator"])
    report = dict(reference=args.reference, candidate=args.candidate,
                  note="Exact float32 regression on Mac CPU; no WER/MOS or speed claim.",
                  rows=[], complete=False)
    report['libraries']={'reference':args.reference_library,'candidate':args.candidate_library}
    # Include new fractional rates, slow speech, long text and additional styles.
    for voice in ["M1", "F1", "M5", "F5"]:
        styles = {name: load_voice_style_from_json_file(engine.root / f"voice_styles/{voice}.json")
                  for name, engine in engines.items()}
        for case in args.cases:
            lang, text = TEXTS[case]
            for speed in args.speeds:
                waves, counts = {}, {}
                for name, engine in engines.items():
                    counter = engine.sessions["vector_estimator"]
                    before = counter.calls
                    waves[name], _ = engine.synthesize(text, lang, 8, speed=speed, voice_style=styles[name])
                    counts[name] = counter.calls - before
                    if counts[name] != 8:
                        raise AssertionError(counts)
                np.testing.assert_array_equal(waves["reference"], waves["candidate"])
                if waves["reference"].tobytes()!=waves["candidate"].tobytes():
                    raise AssertionError("Float32 byte mismatch")
                report["rows"].append(dict(voice=voice, case=case, speed=speed,
                    vector_calls=counts, float32_bitidentical=True, byte_comparison=True,
                    samples=int(waves["candidate"].size),
                    sha256=hashlib.sha256(waves["candidate"].tobytes()).hexdigest()))
                args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    report["complete"] = True
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"{len(report['rows'])} exact waveform pairs; eight calls each")


if __name__ == "__main__":
    main()
