"""Run the standard benchmark with an explicit ORT intra-op thread count.

BLAS and inter-op threads remain at one, as in runtime.py. No model is changed.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runtime
import benchmark


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--ort-threads", type=int, required=True)
    args, remaining = parser.parse_known_args()
    if args.ort_threads < 1:
        parser.error("--ort-threads must be positive")
    original_options = runtime.options

    def options(*positional, **keywords):
        result = original_options(*positional, **keywords)
        result.intra_op_num_threads = args.ort_threads
        return result

    runtime.options = options
    sys.argv = [sys.argv[0], *remaining]
    benchmark.main()
    output_parser = argparse.ArgumentParser(add_help=False)
    output_parser.add_argument("--output", default="results/baseline.json")
    output, _ = output_parser.parse_known_args(remaining)
    path = Path(output.output)
    report = json.loads(path.read_text())
    report.update(
        threads=args.ort_threads,
        ort_intra_op_threads=args.ort_threads,
        ort_inter_op_threads=1,
        blas_threads=1,
    )
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
