"""Paired eight-step quality/numerical check against upstream and transformed graphs."""

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from runtime import Engine
from benchmark import TEXTS
import argparse
import json
from pathlib import Path
import numpy as np
import soundfile as sf

p = argparse.ArgumentParser()
p.add_argument("--variant", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
base = Engine()
candidate = Engine(variant=a.variant)
rows = []
for name in ("ko_short", "en_short"):
    lang, text = TEXTS[name]
    x, b = base.synthesize(text, lang, 8)
    y, c = candidate.synthesize(text, lang, 8)
    size = min(len(x), len(y))
    rmse = float(np.sqrt(np.mean((x[:size] - y[:size]) ** 2)))
    signal = float(np.sqrt(np.mean(x[:size] ** 2)))
    row = {
        "case": name,
        "base_samples": len(x),
        "candidate_samples": len(y),
        "max_abs_error": float(np.max(np.abs(x[:size] - y[:size]))),
        "rmse": rmse,
        "snr_db": float(20 * np.log10(signal / max(rmse, 1e-12))),
        "correlation": float(np.corrcoef(x[:size], y[:size])[0, 1]),
        "speedup": b["seconds"] / c["seconds"],
    }
    rows.append(row)
    print(row, flush=True)
    d = Path("samples") / Path(a.output).stem
    d.mkdir(parents=True, exist_ok=True)
    sf.write(d / f"{name}_reference.wav", x, base.sample_rate)
    sf.write(d / f"{name}_candidate.wav", y, base.sample_rate)
Path(a.output).write_text(
    json.dumps(
        {
            "note": "Waveform error is not a speech quality or intelligibility score. Listening/ASR evaluation required for lossy transforms.",
            "rows": rows,
        },
        indent=2,
    )
)
