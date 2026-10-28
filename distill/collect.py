"""Collect frozen ONNX teacher MLP input/output pairs on real eight-step trajectories."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import Engine, options
import argparse
import json
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort


class Recorder:
    def __init__(self, session, pairs, sinks, rng, samples):
        self.session = session
        self.pairs = pairs
        self.sinks = sinks
        self.rng = rng
        self.samples = samples
        self.offset = 0

    def get_inputs(self):
        return self.session.get_inputs()

    def run(self, outputs, feed):
        result = self.session.run(None, feed)
        batch, channels, _length = result[1].shape
        valid = np.flatnonzero(np.tile(feed["latent_mask"].reshape(-1), batch) > 0)
        index = self.rng.choice(valid, min(self.samples, len(valid)), replace=False)
        for i in range(len(self.pairs)):
            x = result[1 + 2 * i].transpose(0, 2, 1).reshape(-1, channels)[index]
            y = result[2 + 2 * i].transpose(0, 2, 1).reshape(-1, channels)[index]
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise RuntimeError("nonfinite teacher activation")
            self.sinks[i][0][self.offset : self.offset + len(index)] = x.astype(
                np.float16
            )
            self.sinks[i][1][self.offset : self.offset + len(index)] = y.astype(
                np.float16
            )
        self.offset += len(index)
        return result[:1]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="distill/data")
    p.add_argument("--samples-per-step", type=int, default=96)
    a = p.parse_args()
    root = Path(a.output)
    if (root / "manifest.json").exists():
        p.error("dataset already collected")
    root.mkdir(parents=True, exist_ok=True)
    corpus = json.loads(Path("distill/corpus.json").read_text())
    m = onnx.load("models/supertonic-3/onnx/vector_estimator.onnx")
    nodes = {n.name: n for n in m.graph.node}
    pairs = []
    for node in m.graph.node:
        if node.op_type == "Conv" and node.name.endswith("/pwconv1/Conv"):
            second = nodes[node.name.replace("/pwconv1/Conv", "/pwconv2/Conv")]
            pairs.append(
                {
                    "first_node": node.name,
                    "second_node": second.name,
                    "input": node.input[0],
                    "output": second.output[0],
                    "w1": node.input[1],
                    "b1": node.input[2],
                    "w2": second.input[1],
                    "b2": second.input[2],
                }
            )
    for pair in pairs:
        m.graph.output.extend(
            [
                helper.make_tensor_value_info(
                    pair["input"], TensorProto.FLOAT, [None, 512, None]
                ),
                helper.make_tensor_value_info(
                    pair["output"], TensorProto.FLOAT, [None, 512, None]
                ),
            ]
        )
    onnx.checker.check_model(m)
    probe = root / "teacher_probe.onnx"
    onnx.save(m, probe)
    del m
    session = ort.InferenceSession(
        str(probe), options(), providers=["CPUExecutionProvider"]
    )
    teacher = Engine()
    voices = ["M1", "F1", "M2", "F2", "M3", "F3", "M4", "F4", "M5", "F5"]
    manifest = {
        "teacher": "original FP32 Supertonic 3",
        "steps": 8,
        "guidance_scale": 4,
        "format": "float16 activation storage, float32 teacher arithmetic",
        "pairs": pairs,
        "splits": {},
    }
    for split in ("train", "validation"):
        folder = root / split
        folder.mkdir(exist_ok=True)
        capacity = len(corpus[split]) * 8 * a.samples_per_step
        sinks = []
        for i in range(len(pairs)):
            sinks.append(
                tuple(
                    np.lib.format.open_memmap(
                        folder / f"{axis}_{i:02d}.npy",
                        mode="w+",
                        dtype=np.float16,
                        shape=(capacity, 512),
                    )
                    for axis in ("x", "y")
                )
            )
        recorder = Recorder(
            session,
            pairs,
            sinks,
            np.random.RandomState(2026 if split == "train" else 2027),
            a.samples_per_step,
        )
        teacher.sessions["vector_estimator"] = recorder
        for j, (lang, text) in enumerate(corpus[split]):
            from supertonic.loader import load_voice_style_from_name

            voice = voices[j % len(voices)]
            teacher.style = load_voice_style_from_name(teacher.root, voice)
            teacher.synthesize(
                text, lang, 8, seed=100 + j if split == "train" else 300 + j
            )
            print(
                split,
                j + 1,
                len(corpus[split]),
                "rows",
                recorder.offset,
                "voice",
                voice,
                flush=True,
            )
        for x, y in sinks:
            x.flush()
            y.flush()
        manifest["splits"][split] = {
            "rows": recorder.offset,
            "capacity": capacity,
            "utterances": len(corpus[split]),
        }
        (root / "progress.json").write_text(json.dumps(manifest, indent=2))
        del recorder, sinks
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("dataset ready", root, flush=True)
