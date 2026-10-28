"""Layerwise teacher/student distillation with an explicit final weight/activation QAT phase.
Only hidden expansion widths shrink; residual width, attention, CFG4 and 8 steps stay intact.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runtime  # noqa: F401 -- establish one CPU thread before numerical libraries
import argparse
import json
import math
import time
import numpy as np
import onnx
from onnx import numpy_helper
import torch
from torch import nn
from torch.nn import functional as F


def fake_weight(w):
    scale = w.detach().abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 64
    quant = (w / scale).round().clamp(-64, 64) * scale
    return w + (quant - w).detach()


def fake_activation(x):
    low = torch.minimum(x.detach().amin(), torch.zeros((), device=x.device))
    high = torch.maximum(x.detach().amax(), torch.zeros((), device=x.device))
    scale = ((high - low) / 255).clamp_min(1e-8)
    zero = (-low / scale).round().clamp(0, 255)
    quant = ((x / scale).round() + zero).clamp(0, 255).sub(zero) * scale
    return x + (quant - x).detach()


class Student(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.fc1 = nn.Linear(512, width)
        self.fc2 = nn.Linear(width, 512)
        self.quantized = False

    def forward(self, x):
        if self.quantized:
            h = F.linear(
                fake_activation(x), fake_weight(self.fc1.weight), self.fc1.bias
            )
            return F.linear(
                fake_activation(F.gelu(h)), fake_weight(self.fc2.weight), self.fc2.bias
            )
        return self.fc2(F.gelu(self.fc1(x)))


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def atomic_save(value, path):
    temp = path.with_suffix(".tmp")
    torch.save(value, temp)
    temp.replace(path)


@torch.no_grad()
def evaluate(model, x, y, weights):
    error = 0.0
    energy = 0.0
    count = 0
    for start in range(0, len(x), 2048):
        prediction = model(x[start : start + 2048])
        target = y[start : start + 2048]
        error += float(((prediction - target) * weights).square().sum().cpu())
        energy += float((target * weights).square().sum().cpu())
        count += target.numel()
    return {
        "relative_mse": error / max(energy, 1e-12),
        "relative_rmse": math.sqrt(error / max(energy, 1e-12)),
        "weighted_mse": error / count,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="distill/data")
    p.add_argument("--output", required=True)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--iterations", type=int, default=800)
    p.add_argument("--qat-steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--layers", nargs="+", type=int)
    p.add_argument("--device", default="mps")
    p.add_argument(
        "--teacher-model", default="models/supertonic-3/onnx/vector_estimator.onnx"
    )
    a = p.parse_args()
    if not 0 <= a.qat_steps < a.iterations:
        p.error("qat-steps must be below iterations")
    if a.width not in (256, 512, 768, 1024, 1536):
        p.error("unsupported trial width")
    if a.device == "mps" and not torch.backends.mps.is_available():
        p.error("MPS unavailable; choose --device cpu")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device(a.device)
    root = Path(a.data)
    dest = Path(a.output)
    dest.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((root / "manifest.json").read_text())
    pairs = metadata["pairs"]
    source = onnx.load(a.teacher_model)
    initializers = {v.name: numpy_helper.to_array(v) for v in source.graph.initializer}
    report = {
        "config": vars(a),
        "torch": torch.__version__,
        "device": str(device),
        "teacher": metadata.get("teacher", "original FP32 CFG4, 8 denoising steps"),
        "training_rows": metadata["splits"]["train"]["rows"],
        "validation_rows": metadata["splits"]["validation"]["rows"],
        "layers": [],
    }
    for i in a.layers if a.layers is not None else range(len(pairs)):
        torch.manual_seed(2026 + i)
        pair = pairs[i]
        checkpoint = dest / f"layer_{i:02d}.pt"
        start = time.perf_counter()
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if (
                saved.get("complete")
                and saved["config"]["width"] == a.width
                and saved["config"]["iterations"] == a.iterations
            ):
                report["layers"].append(saved["metrics"])
                print("already complete", i, flush=True)
                continue
        train_rows = metadata["splits"]["train"]["rows"]
        val_rows = metadata["splits"]["validation"]["rows"]

        def data(split, axis, rows, i=i):
            return torch.from_numpy(
                np.array(
                    np.load(root / split / f"{axis}_{i:02d}.npy", mmap_mode="r")[:rows],
                    dtype=np.float32,
                )
            ).to(device)

        x = data("train", "x", train_rows)
        y = data("train", "y", train_rows)
        vx = data("validation", "x", val_rows)
        vy = data("validation", "y", val_rows)
        w1 = np.array(initializers[pair["w1"]][:, :, 0], copy=True)
        b1 = np.array(initializers[pair["b1"]], copy=True)
        w2 = np.array(initializers[pair["w2"]][:, :, 0], copy=True)
        b2 = np.array(initializers[pair["b2"]], copy=True)
        gamma = np.array(
            initializers[pair["w1"].replace("pwconv1.weight", "gamma")], copy=True
        ).reshape(-1)
        importance = np.linalg.norm(w1, axis=1) * np.linalg.norm(
            w2 * gamma[:, None], axis=0
        )
        keep = np.sort(np.argsort(importance)[-a.width :])
        student = Student(a.width).to(device)
        with torch.no_grad():
            student.fc1.weight.copy_(torch.from_numpy(w1[keep]).to(device))
            student.fc1.bias.copy_(torch.from_numpy(b1[keep]).to(device))
            student.fc2.weight.copy_(torch.from_numpy(w2[:, keep]).to(device))
            student.fc2.bias.copy_(torch.from_numpy(b2).to(device))
        weights = torch.from_numpy(np.abs(gamma)).to(device)
        with torch.no_grad():
            reference = F.linear(
                F.gelu(
                    F.linear(
                        vx[:256],
                        torch.from_numpy(w1).to(device),
                        torch.from_numpy(b1).to(device),
                    )
                ),
                torch.from_numpy(w2).to(device),
                torch.from_numpy(b2).to(device),
            )
            teacher_error = float(
                (
                    ((reference - vy[:256]) * weights).square().mean()
                    / ((vy[:256] * weights).square().mean() + 1e-12)
                )
                .sqrt()
                .cpu()
            )
            if teacher_error > 0.01:
                raise RuntimeError(
                    f"teacher label mismatch at layer {i}: {teacher_error}"
                )
        initial = evaluate(student, vx, vy, weights)
        # Ridge refit the output affine transform before nonlinear gradient distillation.
        with torch.no_grad():
            subset = x[: min(8192, len(x))]
            hidden = F.gelu(student.fc1(subset))
            design = torch.cat(
                [hidden, torch.ones((len(hidden), 1), device=device)], dim=1
            )
            gram = (design.T @ design / len(design)).cpu()
            rhs = (design.T @ y[: len(design)] / len(design)).cpu()
            ridge = float(gram.diag().mean()) * 1e-3
            prior = torch.cat(
                [student.fc2.weight.T, student.fc2.bias[None]], dim=0
            ).cpu()
            regularizer = torch.eye(a.width + 1) * ridge
            regularizer[-1, -1] = ridge * 0.01
            solution = torch.linalg.solve(
                gram + regularizer, rhs + regularizer @ prior
            ).to(device)
            student.fc2.weight.copy_(solution[:-1].T)
            student.fc2.bias.copy_(solution[-1])
        fitted = evaluate(student, vx, vy, weights)
        optimizer = torch.optim.AdamW(
            student.parameters(), lr=a.learning_rate, weight_decay=1e-4
        )
        best_float = fitted["relative_mse"]
        best_qat = float("inf")
        float_state = cpu_state(student)
        qat_state = None
        history = []
        loss_scale = (y * weights).square().mean().detach().clamp_min(1e-8)
        print(
            "layer",
            i,
            "teacher_error",
            teacher_error,
            "pruned_rmse",
            initial["relative_rmse"],
            "refit_rmse",
            fitted["relative_rmse"],
            flush=True,
        )
        for step in range(a.iterations):
            student.train()
            student.quantized = (
                step >= a.iterations - a.qat_steps if a.qat_steps else False
            )
            lr = a.learning_rate * (
                0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / a.iterations))
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            index = torch.randint(len(x), (a.batch_size,), device=device)
            prediction = student(x[index])
            loss = ((prediction - y[index]) * weights).square().mean() / loss_scale
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            if (
                (step + 1) % 100 == 0
                or step + 1 == a.iterations
                or step + 1 == a.iterations - a.qat_steps
            ):
                student.eval()
                evaluation = evaluate(student, vx, vy, weights)
                row = {
                    "step": step + 1,
                    "qat": student.quantized,
                    "train_loss": float(loss.detach().cpu()),
                    **evaluation,
                }
                history.append(row)
                if not math.isfinite(row["train_loss"]):
                    raise RuntimeError("nonfinite training loss")
                if student.quantized and evaluation["relative_mse"] < best_qat:
                    best_qat = evaluation["relative_mse"]
                    qat_state = cpu_state(student)
                elif not student.quantized and evaluation["relative_mse"] < best_float:
                    best_float = evaluation["relative_mse"]
                    float_state = cpu_state(student)
                payload = {
                    "complete": False,
                    "layer": i,
                    "pair": pair,
                    "config": vars(a),
                    "step": step + 1,
                    "state_dict": cpu_state(student),
                    "float_state": float_state,
                    "qat_state": qat_state,
                    "optimizer": optimizer.state_dict(),
                    "history": history,
                }
                atomic_save(payload, checkpoint)
                print(
                    "layer",
                    i,
                    "step",
                    step + 1,
                    "qat",
                    student.quantized,
                    "val_rmse",
                    evaluation["relative_rmse"],
                    flush=True,
                )
        metrics = {
            "layer": i,
            "name": pair["first_node"],
            "teacher_label_relative_rmse": teacher_error,
            "pruned": initial,
            "ridge": fitted,
            "best_float_relative_rmse": math.sqrt(best_float),
            "best_qat_relative_rmse": math.sqrt(best_qat) if qat_state else None,
            "seconds": time.perf_counter() - start,
            "history": history,
        }
        payload.update(complete=True, metrics=metrics)
        atomic_save(payload, checkpoint)
        report["layers"].append(metrics)
        (dest / "report.json").write_text(json.dumps(report, indent=2))
        del student, optimizer, x, y, vx, vy, hidden, design, gram, rhs, prior, solution
        if device.type == "mps":
            torch.mps.empty_cache()
    (dest / "report.json").write_text(json.dumps(report, indent=2))
    print("training complete", dest, flush=True)
