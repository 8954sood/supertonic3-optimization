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


def convert_graph(path, expected_nodes, width, *, preserve_shapes=False):
    source_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    model = onnx.load(path)
    original = {value.name: value.SerializeToString() for value in model.graph.initializer}
    if preserve_shapes:
        # Preserve ranks through custom nodes so ORT keeps the original fusion/layout paths.
        model = onnx.shape_inference.infer_shapes(model)
    values = {value.name: numpy_helper.to_array(value) for value in model.graph.initializer}
    converted = []
    for node in model.graph.node:
        if node.op_type != "LayerNormalization" or node.domain or len(node.input) != 3 or len(node.output) != 1:
            continue
        attributes = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        if attributes.get("axis") != -1 or attributes.get("stash_type", 1) != TensorProto.FLOAT:
            continue
        if any(name not in values or values[name].dtype.name != "float32" or values[name].shape != (width,)
               for name in node.input[1:]):
            continue
        node.op_type = "RowLayerNorm"
        node.domain = "com.supertonic.rownorm"
        keep = [attribute for attribute in node.attribute if attribute.name != "stash_type"]
        del node.attribute[:]
        node.attribute.extend(keep)
        converted.append(node.name)
    if len(converted) != expected_nodes:
        raise ValueError(f"Expected {expected_nodes} fixed FP32 norm nodes, got {len(converted)}")
    model.opset_import.append(helper.make_opsetid("com.supertonic.rownorm", 1))
    onnx.checker.check_model(model)
    if original != {value.name: value.SerializeToString() for value in model.graph.initializer}:
        raise RuntimeError("Initializer bytes changed")
    return model, {"nodes": len(converted), "source_graph_sha256": source_sha,
                   "initializer_bitidentity": True, "width": width,
                   "preserved_inferred_shapes": preserve_shapes}


def prepare(source, output, *, link_assets=False, all_parts=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Output already exists; choose a new directory")
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Source and output must be separate, non-nested directories")
    metadata = json.loads((source / "manifest.json").read_text())
    already_vector = bool(metadata.get("row_norm_backend"))
    if metadata.get("supported_steps") != [8] or (already_vector and not all_parts):
        raise ValueError("Use an eight-step student; --all-parts can extend an existing row norm package")
    if all_parts and metadata.get("frontend_row_norm_conversion"):
        raise ValueError("Source already includes frontend normalization")
    if already_vector:
        vector = onnx.load(source / "onnx/vector_estimator.onnx")
        if sum(node.domain == "com.supertonic.rownorm" and node.op_type == "RowLayerNorm"
               for node in vector.graph.node) != 36:
            raise ValueError("Expected 36 existing vector normalization nodes")
    specifications = {} if already_vector else {"vector_estimator": (36, 512)}
    if all_parts:
        specifications.update(duration_predictor=(10, 64), text_encoder=(15, 256), vocoder=(10, 512))
    transformed, receipts = {}, {}
    for part, (count, width) in specifications.items():
        transformed[part], receipts[part] = convert_graph(
            source / f"onnx/{part}.onnx", count, width, preserve_shapes=part != "vector_estimator")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".row-norm-", dir=output.parent) as temporary:
        package = Path(temporary) / "package"
        shutil.copytree(source, package, copy_function=link_or_copy if link_assets else shutil.copy2)
        for part, model in transformed.items():
            path = package / f"onnx/{part}.onnx"
            path.unlink()
            onnx.save(model, path)
            original = source / f"onnx/{part}.onnx"
            if hashlib.sha256(original.read_bytes()).hexdigest() != receipts[part]["source_graph_sha256"]:
                raise RuntimeError("Source graph changed during preparation")
            receipts[part]["output_graph_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        metadata["row_norm_backend"] = True
        if "vector_estimator" in receipts:
            metadata["row_norm_conversion"] = {
                **receipts["vector_estimator"],
                "source_vector_sha256": receipts["vector_estimator"]["source_graph_sha256"],
                "target": "ARM64; target ORT compatibility check required", "shared_assets": link_assets,
            }
        if all_parts:
            metadata["row_norm_widths"] = [64, 256, 512]
            metadata["frontend_row_norm_conversion"] = {
                "parts": {name: receipts[name] for name in ["duration_predictor", "text_encoder", "vocoder"]},
                "initializer_bitidentity": True, "total_native_norm_nodes": 71,
            }
        metadata["vector_sha256"] = hashlib.sha256(
            (package / "onnx/vector_estimator.onnx").read_bytes()).hexdigest()
        manifest = package / "manifest.json"
        manifest.unlink()  # Break a potential link to the original metadata.
        manifest.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
        package.rename(output)
    return metadata["frontend_row_norm_conversion"] if all_parts else metadata["row_norm_conversion"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--link-assets", action="store_true",
                        help="Share unchanged files; keep source and output packages read-only")
    parser.add_argument("--all-parts", action="store_true",
                        help="Also convert duration predictor, text encoder and vocoder normalization")
    args = parser.parse_args()
    try:
        receipt = prepare(args.source, args.output, link_assets=args.link_assets, all_parts=args.all_parts)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
