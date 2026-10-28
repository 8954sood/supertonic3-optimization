"""FP32 compressed decoder distillation on aligned teacher audio: waveform and MRSTFT."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runtime  # noqa: F401
import argparse, json, time, os
import numpy as np
import torch
from distill.torch_graph import VectorGraph

p = argparse.ArgumentParser()
p.add_argument("--output", required=True)
p.add_argument("--iterations", type=int, default=800)
p.add_argument("--learning-rate", type=float, default=1e-5)
p.add_argument("--validate-every", type=int, default=100)
a = p.parse_args()
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.manual_seed(20261031)
dest = Path(a.output)
dest.mkdir(parents=True, exist_ok=True)
if (dest / "best.pt").exists():
    p.error("Choose new output")
path = "models/supertonic-3/onnx/vocoder.onnx"
teacher = VectorGraph(path)
student = VectorGraph(
    path,
    checkpoints="distill/decoder1024-v1",
    pair_manifest="distill/decoder-data-v1/manifest.json",
)
student.train_full_decoder()
optimizer = torch.optim.AdamW(
    student.trainable.parameters(), lr=a.learning_rate, weight_decay=1e-5
)
cache = {}
windows = {}


def stfts(w):
    w = w.reshape(-1).to("cpu")
    result = []
    for n in [512, 1024, 2048]:
        if n not in windows:
            windows[n] = torch.hann_window(n)
        result.append(
            torch.stft(
                w, n_fft=n, hop_length=n // 4, window=windows[n], return_complex=True
            ).abs()
        )
    return result


def sample(split, case):
    key = (split, case)
    if key not in cache:
        latent = torch.from_numpy(
            np.load(f"distill/trajectories-v2/{split}/{case:03d}_7.npz")["target"]
        ).to("mps")
        with torch.no_grad():
            target = teacher({"latent": latent})[0]
            spectra = stfts(target)
        cache[key] = (latent, target, spectra)
    return cache[key]


def loss_for(split, case):
    latent, target, spectra = sample(split, case)
    actual = student({"latent": latent})[0]
    mse = (actual - target).square().mean() / target.square().mean().clamp_min(1e-8)
    spectral = 0
    for u, v in zip(stfts(actual), spectra):
        spectral = (
            spectral
            + (u - v).norm() / v.norm().clamp_min(1e-5)
            + (torch.log(u + 1e-4) - torch.log(v + 1e-4)).abs().mean()
        )
    spectral = spectral / 3
    return mse + 0.15 * spectral.to("mps"), mse, spectral


@torch.no_grad()
def validate():
    rows = []
    for case in range(8):
        loss, mse, spectral = loss_for("validation", case)
        rows.append([float(loss.cpu()), float(mse.cpu()), float(spectral.cpu())])
    return dict(
        zip(["loss", "wave_relative_mse", "mrstft"], np.mean(rows, axis=0).tolist())
    )


start = time.perf_counter()
initial = validate()
best = initial["loss"]
history = []
report = {
    "config": vars(a),
    "method": "FP32 decoder FFN 2048->1024, then all decoder learned tensors; aligned waveform normalized MSE + 0.15 MRSTFT",
    "data": "200 original-teacher latent/voice/seed combinations from 40 unique texts, validation8; no test use",
    "initial": initial,
    "history": history,
}


def save(name, step, value):
    p = dest / (name + ".tmp")
    torch.save(
        {
            "weights": student.export_weights(),
            "step": step,
            "validation": value,
            "config": vars(a),
        },
        p,
    )
    p.replace(dest / (name + ".pt"))
    if name == "best":
        os.link(dest / "best.pt", dest / f"best_step_{step:05d}.pt")


print("initial", initial, flush=True)
save("best", 0, initial)
rng = np.random.default_rng(20261031)
for step in range(a.iterations):
    optimizer.zero_grad(set_to_none=True)
    loss, mse, spectral = loss_for("train", int(rng.integers(200)))
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("nonfinite loss")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(student.trainable.parameters(), 1)
    optimizer.step()
    if (step + 1) % 20 == 0:
        print(
            "step",
            step + 1,
            "loss",
            float(loss.detach().cpu()),
            "mse",
            float(mse.detach().cpu()),
            flush=True,
        )
    if (step + 1) % a.validate_every == 0 or step + 1 == a.iterations:
        value = validate()
        row = {
            "step": step + 1,
            **value,
            "elapsed_seconds": time.perf_counter() - start,
        }
        history.append(row)
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
        print("validation", row, flush=True)
