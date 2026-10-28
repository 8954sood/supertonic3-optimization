"""Capture full original denoising input/output pairs, without touching reserved test texts."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import Engine
import argparse
import json
import shutil
import numpy as np
from supertonic.loader import load_voice_style_from_name

p = argparse.ArgumentParser()
p.add_argument("--output", default="distill/trajectories-v1")
p.add_argument("--limit", type=int)
p.add_argument("--repeats", type=int, default=1)
p.add_argument(
    "--source", help="Copy an existing one-repeat collection and add more voices/seeds"
)
a = p.parse_args()
root = Path(a.output)
if (root / "manifest.json").exists():
    p.error("Choose a new collection directory")
if a.source:
    shutil.copytree(a.source, root)
root.mkdir(parents=True, exist_ok=True)
corpus = json.loads(Path("distill/corpus.json").read_text())
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
                {"teacher": "original FP32 CFG4 8 steps", "rows": rows}, indent=2
            )
        )
