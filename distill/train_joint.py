"""Joint FFN distillation through the complete differentiable CFG4 vector graph."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runtime  # noqa: F401
import argparse
import json
import time
import os
import numpy as np
import torch
from distill.torch_graph import VectorGraph
from distill.teacher_policy import assert_original_data
from distill.trajectory_data import trajectory_path

p = argparse.ArgumentParser()
p.add_argument("--checkpoints", default="distill/student1024")
p.add_argument("--data", default="distill/trajectories-v1")
p.add_argument("--output", required=True)
p.add_argument("--iterations", type=int, default=600)
p.add_argument("--learning-rate", type=float, default=3e-5)
p.add_argument("--validate-every", type=int, default=100)
p.add_argument("--graph", default="models/supertonic-3/onnx/vector_estimator.onnx")
p.add_argument("--train-all", action="store_true")
a = p.parse_args()
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.manual_seed(20261028)
root = Path(a.data)
dest = Path(a.output)
dest.mkdir(parents=True, exist_ok=True)
if (dest / "best.pt").exists():
    p.error("Choose new output to preserve checkpoints")
metadata = json.loads((root / "manifest.json").read_text())
assert_original_data(metadata)
train = [r for r in metadata["rows"] if r["split"] == "train"]
val = [
    r for r in metadata["rows"] if r["split"] == "validation" and r["step"] in [0, 3, 7]
]
model = VectorGraph(
    a.graph,
    device="mps",
    checkpoints=a.checkpoints,
)
if a.train_all:
    model.train_full_vector()
optimizer = torch.optim.AdamW(
    model.trainable.parameters(), lr=a.learning_rate, weight_decay=1e-4
)
cache = {}


def sample(row):
    path = trajectory_path(root, row)
    if path not in cache:
        data = dict(np.load(path))
        target = torch.from_numpy(data.pop("target")).to("mps")
        cache[path] = (data, target)
    return cache[path]


def loss_for(row):
    feed, target = sample(row)
    actual = model(feed)[0]
    # Compare the teacher update rather than letting the unchanged input dominate normalization.
    delta = target - torch.from_numpy(feed["noisy_latent"]).to("mps")
    return (actual - target).square().mean() / delta.square().mean().clamp_min(1e-5)


@torch.no_grad()
def validate():
    return float(np.mean([float(loss_for(row).cpu()) for row in val]))


start = time.perf_counter()
initial = validate()
best = initial
history = []
report = {
    "config": vars(a),
    "method": "Selected FFNs or all learned vector tensors jointly optimized through the configured graph; teacher identity and input trajectory origin are specified by dataset metadata; no decoder or waveform loss",
    "teacher": metadata.get("teacher", "unspecified; inspect dataset provenance"),
    "teacher_package": metadata.get("teacher_package"),
    "input_state_source": metadata.get("input_state_source"),
    "train_pairs": len(train),
    "validation_pairs": len(val),
    "initial_validation_relative_update_mse": initial,
    "history": history,
}
print(
    "initial",
    initial,
    "train_pairs",
    len(train),
    "val",
    len(val),
    "seconds",
    time.perf_counter() - start,
    flush=True,
)


def save(name, step, value):
    payload = {
        "weights": model.export_weights(),
        "step": step,
        "validation_relative_update_mse": value,
        "config": vars(a),
    }
    temp = dest / (name + ".tmp")
    torch.save(payload, temp)
    temp.replace(dest / (name + ".pt"))
    if name == "best":
        archived = dest / f"best_step_{step:05d}.pt"
        if not archived.exists():
            os.link(dest / "best.pt", archived)


save("best", 0, initial)
rng = np.random.default_rng(20261028)
for step in range(a.iterations):
    t = time.perf_counter()
    row = train[int(rng.integers(len(train)))]
    optimizer.zero_grad(set_to_none=True)
    loss = loss_for(row)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("Nonfinite loss")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.trainable.parameters(), 1.0)
    optimizer.step()
    if (step + 1) % 10 == 0:
        print(
            "step",
            step + 1,
            "loss",
            float(loss.detach().cpu()),
            "step_seconds",
            time.perf_counter() - t,
            flush=True,
        )
    if (step + 1) % a.validate_every == 0 or step + 1 == a.iterations:
        value = validate()
        entry = {
            "step": step + 1,
            "validation_relative_update_mse": value,
            "train_loss": float(loss.detach().cpu()),
            "elapsed_seconds": time.perf_counter() - start,
        }
        history.append(entry)
        if value < best:
            best = value
            save("best", step + 1, value)
        save("last", step + 1, value)
        report.update(
            best_validation_relative_update_mse=best,
            elapsed_seconds=time.perf_counter() - start,
            complete=step + 1 == a.iterations,
        )
        (dest / "report.json").write_text(json.dumps(report, indent=2))
        print("validation", entry, flush=True)
