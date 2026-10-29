"""Distill a fixed original FP32 CFG4 trajectory onto an eight-step student grid."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import Engine, options
from distill.teacher_policy import assert_original_model
import argparse, hashlib, json
import numpy as np
import onnxruntime as ort
import soundfile as sf
from supertonic.loader import load_voice_style_from_name

p = argparse.ArgumentParser()
p.add_argument("--output", required=True)
p.add_argument("--corpus", default="distill/corpus_round4_train.json")
p.add_argument("--teacher-steps", type=int, default=50)
p.add_argument("--teacher-threads", type=int, default=4)
a = p.parse_args()
if not 8 <= a.teacher_steps <= 50 or a.teacher_threads < 1:
    p.error("Invalid teacher configuration")
root = Path(a.output)
if root.exists():
    p.error("Choose a new collection directory")
root.mkdir(parents=True)
source = Path("models/supertonic-3")
origin = assert_original_model(source)
origin.update(
    distilled=False,
    precision="FP32",
    guidance_scale=4,
    steps=a.teacher_steps,
    threads=a.teacher_threads,
    graph_sha256={
        name: hashlib.sha256(
            (source / "onnx" / f"{name}.onnx").read_bytes()
        ).hexdigest()
        for name in [
            "duration_predictor",
            "text_encoder",
            "vector_estimator",
            "vocoder",
        ]
    },
)
engine = Engine(model=source)
settings = options()
settings.intra_op_num_threads = a.teacher_threads
session = ort.InferenceSession(
    str(source / "onnx/vector_estimator.onnx"),
    settings,
    providers=["CPUExecutionProvider"],
)


class Recorder:
    def get_inputs(self):
        return session.get_inputs()

    def run(self, outputs, feed):
        if int(feed["current_step"][0]) == 0:
            self.states = [feed["noisy_latent"].copy()]
            self.feed = {
                k: v.copy()
                for k, v in feed.items()
                if k not in {"noisy_latent", "current_step", "total_step"}
            }
        result = session.run(outputs, feed)
        self.states.append(result[0].copy())
        return result


recorder = Recorder()
engine.sessions["vector_estimator"] = recorder
corpus = json.loads(Path(a.corpus).read_text())
rows = []
manifest = {
    "teacher": f"original FP32 CFG4 {a.teacher_steps} steps, mapped to 8-step student targets",
    "teacher_origin": origin,
    "student_steps": 8,
    "state_grid": "Linear interpolation of original teacher latent states at k/8; final target is exact original final latent; no distilled teacher",
    "corpus": a.corpus,
    "corpus_sha256": hashlib.sha256(Path(a.corpus).read_bytes()).hexdigest(),
    "config": vars(a),
    "rows": rows,
    "complete": False,
}


def state(position):
    low = int(np.floor(position))
    high = int(np.ceil(position))
    ratio = np.float32(position - low)
    return (
        recorder.states[low].copy()
        if low == high
        else recorder.states[low] * (1 - ratio) + recorder.states[high] * ratio
    )


for split in ["train", "validation"]:
    voices = (
        ["M1", "F1", "M2", "F2", "M3", "F3", "M4", "F4", "M5", "F5"]
        if split == "train"
        else ["M1", "F1", "M3", "F3", "M5", "F5"]
    )
    for case, (lang, text) in enumerate(corpus[split]):
        voice = voices[case % len(voices)]
        seed = (100 if split == "train" else 400) + case
        engine.style = load_voice_style_from_name(source, voice)
        wave, timing = engine.synthesize(text, lang, a.teacher_steps, seed=seed)
        assert len(recorder.states) == a.teacher_steps + 1
        for step in range(8):
            first = step * a.teacher_steps / 8
            last = (step + 1) * a.teacher_steps / 8
            feed = {
                **recorder.feed,
                "noisy_latent": state(first),
                "current_step": np.array([step], np.float32),
                "total_step": np.array([8], np.float32),
            }
            path = root / split / f"{case:03d}_{step}.npz"
            path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path, **feed, target=state(last))
            rows.append(
                {
                    "split": split,
                    "case": case,
                    "step": step,
                    "voice": voice,
                    "language": lang,
                    "seed": seed,
                    "teacher_start_position": first,
                    "teacher_target_position": last,
                    "file": str(path),
                }
            )
        if split == "validation" and case < 2:
            audio = Path("samples") / root.name
            audio.mkdir(exist_ok=True)
            sf.write(
                audio / f"original_{a.teacher_steps}_{case:02d}.wav",
                wave,
                engine.sample_rate,
                subtype="FLOAT",
            )
        (root / "progress.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(
            split,
            case + 1,
            len(corpus[split]),
            "teacher seconds",
            round(timing["seconds"], 3),
            flush=True,
        )
manifest["complete"] = True
(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("Original-only collection complete:", len(rows), "pairs", flush=True)
