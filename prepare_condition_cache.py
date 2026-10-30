"""Prepare a portable per-utterance conditioning cache without changing weights."""

import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import onnx

from experiments.optimize import split


def link_or_copy(source, destination):
    """Save disk space when requested, including across filesystem boundaries."""
    try:
        return os.link(source, destination)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        return shutil.copy2(source, destination)


def prepare(source, output, *, link_assets=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Output already exists; choose a new directory")
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Source and output must be separate, non-nested directories")
    if (source / "onnx/condition.onnx").exists():
        raise ValueError("Source already has a conditioning cache")
    vector = source / "onnx/vector_estimator.onnx"
    metadata = json.loads((source / "manifest.json").read_text())
    source_sha = hashlib.sha256(vector.read_bytes()).hexdigest()
    original = {value.name: value.SerializeToString()
                for value in onnx.load(vector).graph.initializer}
    output.parent.mkdir(parents=True, exist_ok=True)
    # Publish a complete package only after validation; leave the source intact
    # and remove incomplete work if a graph or initializer check fails.
    with tempfile.TemporaryDirectory(prefix=".condition-cache-", dir=output.parent) as temporary:
        package = Path(temporary) / "package"
        shutil.copytree(source, package, copy_function=link_or_copy if link_assets else shutil.copy2)
        # Never write through a hard link into the source package.
        (package / "onnx/vector_estimator.onnx").unlink()
        (package / "onnx/transformation.json").unlink(missing_ok=True)
        split(vector, package / "onnx")
        actual = {}
        for part in ["vector_estimator", "condition"]:
            for value in onnx.load(package / f"onnx/{part}.onnx").graph.initializer:
                data = value.SerializeToString()
                if value.name in actual and actual[value.name] != data:
                    raise RuntimeError("Conflicting split initializer")
                actual[value.name] = data
        if actual != original:
            raise RuntimeError("Split changed initializer bytes or lost an initializer")
        if hashlib.sha256(vector.read_bytes()).hexdigest() != source_sha:
            raise RuntimeError("Source graph changed during preparation")
        metadata["vector_sha256"] = hashlib.sha256(
            (package / "onnx/vector_estimator.onnx").read_bytes()).hexdigest()
        metadata["portable_condition_cache"] = {
            "source_vector_sha256": source_sha,
            "condition_sha256": hashlib.sha256(
                (package / "onnx/condition.onnx").read_bytes()).hexdigest(),
            "initializer_count": len(original),
            "initializer_bitidentity": True,
            "steps_changed": False,
            "quantization_changed": False,
            "shared_assets": link_assets,
        }
        manifest = package / "manifest.json"
        manifest.unlink()  # The copied manifest can also be a hard link.
        manifest.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
        package.rename(output)
    return metadata["portable_condition_cache"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--link-assets", action="store_true",
                        help="Hard-link unchanged assets to save disk; keep both packages read-only")
    args = parser.parse_args()
    try:
        receipt = prepare(args.source, args.output, link_assets=args.link_assets)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
