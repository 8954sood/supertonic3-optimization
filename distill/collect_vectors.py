"""Capture full original denoising input/output pairs, without touching reserved test texts."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import Engine
from distill.teacher_policy import assert_original_data, assert_original_model
import argparse
import json
import shutil
import numpy as np
import hashlib
from supertonic.loader import load_voice_style_from_name

p = argparse.ArgumentParser()
p.add_argument("--output", default="distill/trajectories-v1")
p.add_argument("--limit", type=int)
p.add_argument("--repeats", type=int, default=1)
p.add_argument("--corpus", default="distill/corpus.json")
p.add_argument(
    "--source", help="Copy an existing one-repeat collection and add more voices/seeds"
)
a = p.parse_args()
if a.repeats < 1:
    p.error("repeats must be positive")
root = Path(a.output)
if (root / "manifest.json").exists():
    p.error("Choose a new collection directory")
if a.source:
    previous = json.loads((Path(a.source) / "manifest.json").read_text())
    assert_original_data(previous)
    if previous.get("corpus", "distill/corpus.json") != a.corpus:
        p.error("Source trajectories use a different corpus")
    shutil.copytree(a.source, root)
root.mkdir(parents=True, exist_ok=True)
corpus_path = Path(a.corpus)
corpus = json.loads(corpus_path.read_text())
corpus_sha256 = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
assert_original_model("models/supertonic-3")
teacher = Engine()
original = teacher.sessions["vector_estimator"]


class Recorder:
    def get_inputs(self):
        return original.get_inputs()

    def run(self, outputs, feed):
        result = original.run(outputs, feed)
        path = root / split / f"{case:03d}_{self.step}.npz"
        path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path, **feed, target=result[0])
        rows.append(
            {
                "split": split,
                "case": case,
                "step": self.step,
                "voice": voice,
                "language": lang,
                "file": str(path),
            }
        )
        self.step += 1
        return result


recorder = Recorder()
teacher.sessions["vector_estimator"] = recorder
rows = json.loads((root / "manifest.json").read_text())["rows"] if a.source else []
for row in rows:
    row["file"] = str(root / row["split"] / Path(row["file"]).name)
for split in ["train", "validation"]:
    if split == "validation" and a.source:
        continue
    texts = corpus[split][: a.limit] if a.limit else corpus[split]
    jobs = [
        (repeat, i, item)
        for repeat in range(1 if a.source else 0, a.repeats if split == "train" else 1)
        for i, item in enumerate(texts)
    ]
    for repeat, text_index, (lang, text) in jobs:
        case = repeat * len(texts) + text_index
        voice = ["M1", "F1", "M2", "F2", "M3", "F3", "M4", "F4", "M5", "F5"][
            (text_index + repeat * 3) % 10
        ]
        teacher.style = load_voice_style_from_name(teacher.root, voice)
        recorder.step = 0
        teacher.synthesize(
            text, lang, 8, seed=(100 if split == "train" else 300) + case
        )
        print(split, case, flush=True)
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "teacher": "original FP32 CFG4 8 steps",
                    "corpus": a.corpus,
                    "corpus_sha256": corpus_sha256,
                    "rows": rows,
                },
                indent=2,
            )
        )
