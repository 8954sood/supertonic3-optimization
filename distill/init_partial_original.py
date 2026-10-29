"""Initialize a partial-width student; all replacement FFNs were taught by original FP32."""

import argparse, hashlib, json
from pathlib import Path
import torch

p = argparse.ArgumentParser()
p.add_argument("--output", required=True)
p.add_argument("--initial", default="distill/joint-cfg-mixed-v1/best_step_00600.pt")
p.add_argument("--layers", type=int, default=6)
a = p.parse_args()
dest = Path(a.output)
if dest.exists():
    p.error("Choose a new output")
meta = json.loads(Path("distill/data/manifest.json").read_text())
if meta["teacher"] != "original FP32 Supertonic 3":
    raise ValueError("Only original-teacher FFN fits allowed")
base = torch.load(a.initial, map_location="cpu", weights_only=True)
weights = {k: v.clone() for k, v in base["weights"].items()}
fits = []
for i, pair in enumerate(meta["pairs"]):
    if weights[pair["w1"]].shape[0] != 2048:
        continue
    file = Path(f"distill/student1536/layer_{i:02d}.pt")
    fit = torch.load(file, map_location="cpu", weights_only=True)
    fits.append((fit["metrics"]["best_float_relative_rmse"], i, file))
selected = sorted(fits)[: a.layers]
for error, i, file in selected:
    state = torch.load(file, map_location="cpu", weights_only=True)["float_state"]
    pair = meta["pairs"][i]
    for key, value in [
        (pair["w1"], state["fc1.weight"][:, :, None]),
        (pair["b1"], state["fc1.bias"]),
        (pair["w2"], state["fc2.weight"][:, :, None]),
        (pair["b2"], state["fc2.bias"]),
    ]:
        weights[key] = value.clone()
report = {
    "method": "Existing student initialization with six sensitive wide FFNs replaced by original-FP32-teacher 1536 fits; initialization only, never a teacher",
    "initial": a.initial,
    "initial_sha256": hashlib.sha256(Path(a.initial).read_bytes()).hexdigest(),
    "original_ffn_teacher": meta["teacher"],
    "original_data_manifest_sha256": hashlib.sha256(
        Path("distill/data/manifest.json").read_bytes()
    ).hexdigest(),
    "selected": [
        {
            "layer": i,
            "original_teacher_fit_relative_rmse": error,
            "source": str(file),
            "source_sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
        }
        for error, i, file in selected
    ],
    "serialized_learned_weight_count": sum(v.numel() for v in weights.values()),
    "teacher_policy": "All new training targets must come from pinned original FP32 CFG4; no student teacher",
}
dest.mkdir(parents=True)
(dest / "report.json").write_text(json.dumps(report, indent=2) + "\n")
torch.save(
    {
        "weights": weights,
        "step": 0,
        "validation_relative_update_mse": None,
        "config": vars(a),
        "initialization": report,
    },
    dest / "initial.pt",
)
print("Original-only initialization:", [i for error, i, file in selected], flush=True)
