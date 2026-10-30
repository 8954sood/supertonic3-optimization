"""Prepare optional ARM64 row-parallel normalization with unchanged learned values."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import onnx
from onnx import numpy_helper, helper, TensorProto

from prepare_condition_cache import link_or_copy


def prepare(source, output, *, link_assets=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Output already exists; choose a new directory")
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Source and output must be separate, non-nested directories")
    metadata = json.loads((source / "manifest.json").read_text())
    if metadata.get("supported_steps") != [8] or metadata.get("row_norm_backend"):
        raise ValueError("Use an eight-step student without the row norm backend")
    vector = source / "onnx/vector_estimator.onnx"
    source_sha = hashlib.sha256(vector.read_bytes()).hexdigest()
    model = onnx.load(vector)
    original = {value.name: value.SerializeToString() for value in model.graph.initializer}
    values = {value.name: numpy_helper.to_array(value) for value in model.graph.initializer}
    converted = []
    for node in model.graph.node:
        if node.op_type != "LayerNormalization" or node.domain or len(node.input) != 3 or len(node.output) != 1:
            continue
        attributes = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        if attributes.get("axis") != -1 or attributes.get("stash_type", 1) != TensorProto.FLOAT:
            continue
        if any(name not in values or values[name].dtype.name != "float32" or values[name].shape != (512,)
               for name in node.input[1:]):
            continue
        node.op_type = "RowLayerNorm"
        node.domain = "com.supertonic.rownorm"
        keep = [attribute for attribute in node.attribute if attribute.name != "stash_type"]
        del node.attribute[:]
        node.attribute.extend(keep)
        converted.append(node.name)
    if len(converted) != 36:
        raise ValueError(f"Expected 36 fixed FP32 norm nodes, got {len(converted)}")
    model.opset_import.append(helper.make_opsetid("com.supertonic.rownorm", 1))
    onnx.checker.check_model(model)
    if original != {value.name: value.SerializeToString() for value in model.graph.initializer}:
        raise RuntimeError("Initializer bytes changed")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".row-norm-", dir=output.parent) as temporary:
        package = Path(temporary) / "package"
        shutil.copytree(source, package, copy_function=link_or_copy if link_assets else shutil.copy2)
        (package / "onnx/vector_estimator.onnx").unlink()
        onnx.save(model, package / "onnx/vector_estimator.onnx")
        if hashlib.sha256(vector.read_bytes()).hexdigest() != source_sha:
            raise RuntimeError("Source graph changed during preparation")
        metadata["row_norm_backend"] = True
        metadata["row_norm_conversion"] = {
            "nodes": len(converted), "source_vector_sha256": source_sha,
            "initializer_bitidentity": True, "width": 512,
            "target": "ARM64; target ORT compatibility check required",
            "shared_assets": link_assets,
        }
        metadata["vector_sha256"] = hashlib.sha256(
            (package / "onnx/vector_estimator.onnx").read_bytes()).hexdigest()
        manifest = package / "manifest.json"
        manifest.unlink()  # Break a potential link to the original metadata.
        manifest.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
        package.rename(output)
    return metadata["row_norm_conversion"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--link-assets", action="store_true",
                        help="Share unchanged files; keep source and output packages read-only")
    args = parser.parse_args()
    try:
        receipt = prepare(args.source, args.output, link_assets=args.link_assets)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
