"""Controlled eight-step candidates: cached conditioning, head-only INT8, fast GELU."""

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic

from experiments.optimize import split
from runtime import options


def write_manifest(source, destination, mode, changes):
    parent = json.loads((source / "manifest.json").read_text())
    hashes = {}
    for path in sorted((destination / "onnx").glob("*.onnx")):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        hashes[path.stem] = digest.hexdigest()
    chain = parent.get("experiment_chain", [])
    if not chain and parent.get("round4_mode"):
        chain = [parent["round4_mode"]]
    manifest = {
        "mode": "round4-inference-experiment",
        "source_package": str(source),
        "source_manifest": parent,
        "minimum_steps": 8,
        "supported_steps": parent.get("supported_steps", [8]),
        "guidance_mode": parent.get("guidance_mode"),
        "teacher_guidance_scale": parent.get("teacher_guidance_scale"),
        "student_internal_guidance_scale": parent.get(
            "student_internal_guidance_scale"
        ),
        "round4_mode": mode,
        "experiment_chain": chain + [mode],
        "changes": changes,
        "numerical_change": mode != "cache" or parent.get("numerical_change", False),
        "training_performed": False,
        "graph_sha256": hashes,
        "vector_sha256": hashes["vector_estimator"],
        "status": "experimental; not selected; no listening approval for this candidate",
    }
    temporary = destination / "manifest.tmp.json"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    temporary.replace(destination / "manifest.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="models/cfg-mixed-v1-selected")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mode",
        choices=["cache", "head-int8", "fast-gelu", "vocoder-fast-gelu"],
        required=True,
    )
    args = parser.parse_args()
    source, destination = Path(args.source), Path(args.output)
    if destination.exists():
        parser.error("Preserve existing candidates; choose a new output")
    # All writes below replace directory entries so source hardlinks stay immutable.
    shutil.copytree(source, destination, copy_function=os.link)
    graph_dir = destination / "onnx"
    changes = {}
    if args.mode == "cache":
        (graph_dir / "vector_estimator.onnx").unlink()
        (graph_dir / "transformation.json").unlink(missing_ok=True)
        split(source / "onnx/vector_estimator.onnx", graph_dir)
        changes = json.loads((graph_dir / "transformation.json").read_text())
    elif args.mode == "head-int8":
        original = source / "onnx/vocoder.onnx"
        graph = onnx.load(original)
        names = [
            node.name
            for node in graph.graph.node
            if node.op_type == "MatMul" and "/head/layer2/" in node.name
        ]
        if len(names) != 1:
            raise RuntimeError("Expected exactly one final waveform projection")
        temporary = graph_dir / "head_quantized.tmp.onnx"
        quantize_dynamic(
            str(original),
            str(temporary),
            op_types_to_quantize=["MatMul"],
            nodes_to_quantize=names,
            weight_type=QuantType.QInt8,
            per_channel=True,
            reduce_range=True,
            extra_options={"MatMulConstBOnly": True},
        )
        temporary.replace(graph_dir / "vocoder.onnx")
        changes = {"quantized_nodes": names, "reduce_range": True}
    else:
        for name in (
            ["vocoder"]
            if args.mode == "vocoder-fast-gelu"
            else ["vector_estimator", "vocoder"]
        ):
            settings = options()
            settings.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            )
            settings.add_session_config_entry(
                "optimization.enable_gelu_approximation", "1"
            )
            temporary = graph_dir / f"{name}.optimized.tmp.onnx"
            settings.optimized_model_filepath = str(temporary)
            session = ort.InferenceSession(
                str(source / "onnx" / f"{name}.onnx"),
                settings,
                providers=["CPUExecutionProvider"],
            )
            del session
            optimized = onnx.load(temporary)
            changes[name] = {
                "fast_gelu_nodes": sum(
                    n.op_type == "FastGelu" for n in optimized.graph.node
                )
            }
            temporary.replace(graph_dir / f"{name}.onnx")
    write_manifest(source, destination, args.mode, changes)
    print(
        json.dumps({"model": str(destination), "changes": changes}, ensure_ascii=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
