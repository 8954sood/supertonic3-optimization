"""Eight-step student rollout distillation through the frozen original audio decoder.
Latent loss plus multi-resolution STFT; no adversarial training or perceptual claims.
"""

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

p = argparse.ArgumentParser()
p.add_argument("--initial", default="distill/joint1024-v1/best.pt")
p.add_argument("--layers", default="distill/student1024")
p.add_argument("--data", default="distill/trajectories-v1")
p.add_argument("--output", required=True)
p.add_argument("--iterations", type=int, default=200)
p.add_argument("--learning-rate", type=float, default=1e-5)
p.add_argument("--validate-every", type=int, default=25)
p.add_argument("--validation-all", action="store_true")
p.add_argument("--graph", default="models/supertonic-3/onnx/vector_estimator.onnx")
a = p.parse_args()
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.manual_seed(20261029)
dest = Path(a.output)
dest.mkdir(parents=True, exist_ok=True)
if (dest / "best.pt").exists():
    p.error("Choose new output")
root = Path(a.data)
metadata = json.loads((root / "manifest.json").read_text())
train_cases = sorted(
    {row["case"] for row in metadata["rows"] if row["split"] == "train"}
)
validation_cases = list(range(8)) if a.validation_all else [0, 1, 4, 5]
model = VectorGraph(a.graph, checkpoints=a.layers)
model.load_weights(
    torch.load(a.initial, map_location="cpu", weights_only=True)["weights"]
)
model.train_full_vector()
decoder = VectorGraph("models/supertonic-3/onnx/vocoder.onnx")
optimizer = torch.optim.AdamW(
    model.trainable.parameters(), lr=a.learning_rate, weight_decay=1e-5
)
cache = {}
windows = {}


def stfts(wave):
    # CPU FFT stays differentiable through device copies; avoids backend-specific complex FFT limits.
    wave = wave.reshape(-1).to("cpu")
    result = []
    for n in [512, 1024, 2048]:
        if n not in windows:
            windows[n] = torch.hann_window(n)
        result.append(
            torch.stft(
                wave,
                n_fft=n,
                hop_length=n // 4,
                win_length=n,
                window=windows[n],
                return_complex=True,
            ).abs()
        )
    return result


def data(split, case):
    key = (split, case)
    if key not in cache:
        first = dict(np.load(root / split / f"{case:03d}_0.npz"))
        first.pop("target")
        target = torch.from_numpy(
            np.load(root / split / f"{case:03d}_7.npz")["target"]
        ).to("mps")
        with torch.no_grad():
            spectra = stfts(decoder({"latent": target})[0])
        cache[key] = (first, target, spectra)
    return cache[key]


def loss_for(split, case):
    feed, target, spectra = data(split, case)
    feed = dict(feed)
    xt = torch.from_numpy(feed["noisy_latent"]).to("mps")
    for step in range(8):
        feed["current_step"] = np.array([step], np.float32)
        feed["noisy_latent"] = xt
        xt = model(feed)[0]
    latent = (xt - target).square().mean() / target.square().mean().clamp_min(1e-5)
    prediction = stfts(decoder({"latent": xt})[0])
    spectral = 0
    for actual, teacher in zip(prediction, spectra):
        spectral = (
            spectral
            + (actual - teacher).norm() / teacher.norm().clamp_min(1e-5)
            + (torch.log(actual + 1e-4) - torch.log(teacher + 1e-4)).abs().mean()
        )
    spectral = spectral / len(spectra)
    return latent + 0.15 * spectral.to("mps"), latent, spectral


@torch.no_grad()
def validate():
    rows = []
    for case in validation_cases:
        loss, latent, spectral = loss_for("validation", case)
        rows.append([float(loss.cpu()), float(latent.cpu()), float(spectral.cpu())])
    return dict(
        zip(["loss", "latent_relative_mse", "mrstft"], np.mean(rows, axis=0).tolist())
    )


start = time.perf_counter()
initial = validate()
best = initial["loss"]
history = []
report = {
    "config": vars(a),
    "steps": 8,
    "teacher_guidance_scale": 4,
    "student_graph": a.graph,
    "train_utterance_voice_seed_cases": len(train_cases),
    "validation_cases": validation_cases,
    "method": "All learned vector-field tensors; 8-step student trajectory with frozen original FP32 decoder; normalized latent MSE + 0.15*(3-resolution spectral convergence + log magnitude L1)",
    "initial": initial,
    "history": history,
}


def save(name, step, value):
    payload = {
        "weights": model.export_weights(),
        "step": step,
        "validation_relative_update_mse": None,
        "validation_rollout": value,
        "config": vars(a),
    }
    path = dest / (name + ".tmp")
    torch.save(payload, path)
    path.replace(dest / (name + ".pt"))
    if name == "best":
        archived = dest / f"best_step_{step:05d}.pt"
        if not archived.exists():
            os.link(dest / "best.pt", archived)


print("initial", initial, "seconds", time.perf_counter() - start, flush=True)
save("best", 0, initial)
rng = np.random.default_rng(20261029)
for step in range(a.iterations):
    t = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    loss, latent, spectral = loss_for("train", int(rng.choice(train_cases)))
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("nonfinite loss")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.trainable.parameters(), 1)
    optimizer.step()
    if (step + 1) % 5 == 0:
        print(
            "step",
            step + 1,
            "loss",
            float(loss.detach().cpu()),
            "latent",
            float(latent.detach().cpu()),
            "spectral",
            float(spectral.detach().cpu()),
            "seconds",
            time.perf_counter() - t,
            flush=True,
        )
    if (step + 1) % a.validate_every == 0 or step + 1 == a.iterations:
        value = validate()
        entry = {
            "step": step + 1,
            **value,
            "elapsed_seconds": time.perf_counter() - start,
        }
        history.append(entry)
        if value["loss"] < best:
            best = value["loss"]
            save("best", step + 1, value)
        save("last", step + 1, value)
        report.update(
            best_loss=best,
            complete=step + 1 == a.iterations,
            elapsed_seconds=time.perf_counter() - start,
        )
        (dest / "report.json").write_text(json.dumps(report, indent=2))
        print("validation", entry, flush=True)
