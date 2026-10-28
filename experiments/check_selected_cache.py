"""Numerical regression check for caching the fixed student, not a quality test."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import Engine
import numpy as np, json, gc

texts = json.loads(Path("distill/corpus.json").read_text())["validation"]
rows = []
for case, voice in [(0, "M1"), (1, "F1"), (4, "M5"), (5, "F5")]:
    lang, text = texts[case]
    waves = []
    for model in ["mixed-v1-selected", "mixed-v1-cached"]:
        e = Engine(model="models/" + model, voice=voice)
        w, _ = e.synthesize(text, lang, 8, seed=400 + case)
        waves.append(w)
        del e
        gc.collect()
    assert len(waves[0]) == len(waves[1])
    error = float(np.sqrt(np.mean((waves[0] - waves[1]) ** 2)))
    maximum = float(np.max(abs(waves[0] - waves[1])))
    rows.append(
        {
            "case": case,
            "voice": voice,
            "rmse": error,
            "max_error": maximum,
            "bit_identical": bool(np.array_equal(*waves)),
        }
    )
    assert error < 1e-5
Path("results/equivalence_mixed_cache_v1.json").write_text(
    json.dumps({"rows": rows}, indent=2)
)
print(rows)
