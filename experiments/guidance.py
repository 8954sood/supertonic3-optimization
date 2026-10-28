"""Guidance graph experiments. 'dedup' preserves CFG4; 'conditional' changes CFG to 1.
Both still execute eight genuine denoising iterations; conditional is a quality tradeoff.
"""

from pathlib import Path
import argparse
import json
import shutil
import onnx
from onnx import helper

p = argparse.ArgumentParser()
p.add_argument("mode", choices=["dedup", "conditional"])
p.add_argument("--source", default="models/fast-full-int8")
p.add_argument("--output", required=True)
a = p.parse_args()
source = Path(a.source)
dest = Path(a.output)
if dest.exists():
    p.error("choose a new output directory")
shutil.copytree(source, dest)
m = onnx.load(dest / "onnx/vector_estimator.onnx")
nodes = list(m.graph.node)
changed = []
if a.mode == "conditional":
    replacements = {
        "/vector_estimator/Tile_1": "noisy_latent",
        "/vector_estimator/Tile_2": "/vector_estimator/Expand_5_output_0",
        "/vector_estimator/Tile_3": "latent_mask",
        "/vector_estimator/Tile_4": "text_mask",
        "/vector_estimator/Concat_5": "text_emb",
        "/vector_estimator/Concat_7": "style_ttl",
        "/vector_estimator/Concat_6": "/vector_estimator/Tile_output_0",
        "/Sub": "/vector_estimator/vector_field/proj_out/Mul_output_0",
    }
    for n in nodes:
        if n.name in replacements:
            n.CopyFrom(
                helper.make_node(
                    "Identity", [replacements[n.name]], list(n.output), name=n.name
                )
            )
            changed.append(n.name)
    if len(changed) != len(replacements):
        raise RuntimeError("guidance pattern mismatch")
else:
    prefix = "/vector_estimator/vector_field/"
    boundary = prefix + "main_blocks.0/convnext.3/Mul_3_output_0"
    out = boundary + "/duplicate"

    def phase(n):
        return n.name.startswith(prefix + "proj_in/") or n.name.startswith(
            prefix + "main_blocks.0/"
        )

    for n in nodes:
        if phase(n):
            for i, value in enumerate(n.input):
                if value == "/vector_estimator/Tile_1_output_0":
                    n.input[i] = "noisy_latent"
                elif value == "/vector_estimator/Tile_3_output_0":
                    n.input[i] = "latent_mask"
        else:
            for i, value in enumerate(n.input):
                if value == boundary:
                    n.input[i] = out
    boundary_index = next(i for i, n in enumerate(nodes) if boundary in n.output)
    nodes.insert(
        boundary_index + 1,
        helper.make_node(
            "Tile",
            [
                boundary,
                "schedule/repeats"
                if any("schedule/repeats" in node.output for node in nodes)
                else "onnx::Tile_1065",
            ],
            [out],
            name="dedup_guidance_prefix",
        ),
    )
    changed = [
        "four identical prefix ConvNeXt blocks computed once before batch duplication"
    ]


# Include implicit captures from If subgraphs when pruning outer producers.
def captures(node):
    used = set(node.input)
    for attr in node.attribute:
        if attr.type == onnx.AttributeProto.GRAPH:
            graph = attr.g
            local = (
                {x.name for x in graph.input}
                | {x.name for x in graph.initializer}
                | {x for n in graph.node for x in n.output}
            )
            used.update(set().union(*(captures(n) for n in graph.node)) - local)
    return used


needed = {x.name for x in m.graph.output}
selected = []
for n in reversed(nodes):
    if any(x in needed for x in n.output):
        selected.append(n)
        needed.update(captures(n))
del m.graph.node[:]
m.graph.node.extend(reversed(selected))
kept = [x for x in m.graph.initializer if x.name in needed]
del m.graph.initializer[:]
m.graph.initializer.extend(kept)
del m.graph.value_info[:]
onnx.checker.check_model(m)
onnx.save(m, dest / "onnx/vector_estimator.onnx")
manifest = json.loads((dest / "manifest.json").read_text())
manifest.update(
    guidance_mode=a.mode,
    guidance_scale=1 if a.mode == "conditional" else 4,
    guidance_changes=changed,
    minimum_steps=8,
    status="experimental; requires quality and target benchmark validation",
)
if a.mode == "dedup" and "guided_steps" in manifest:
    manifest["guidance_mode"] = "scheduled quality tradeoff + exact prefix dedup"
    manifest.pop("guidance_scale", None)
(dest / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(a.mode, len(m.graph.node), flush=True)
