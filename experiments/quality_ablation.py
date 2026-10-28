"""Generate controlled module ablations; keep original duration/text/decoder when requested."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import Engine
import argparse
import json
import gc
import soundfile as sf

p = argparse.ArgumentParser()
p.add_argument("--output", default="results/quality_ablation_v2.json")
p.add_argument(
    "--models", nargs="+", help="Optional full model packages to compare with original"
)
a = p.parse_args()
texts = json.loads(Path("distill/corpus.json").read_text())["validation"]
cases = [(0, "M1"), (1, "F1"), (4, "M1"), (5, "F1")]
variants = [
    ("reference", "models/supertonic-3", None),
    ("fp32_cached", "models/fast-fp32-cached", None),
    ("full_int8", "models/fast-full-int8", None),
    ("vector_int8_only", "models/fast-denoiser-int8", None),
    ("student1024_full_int8", "models/student1024-int8", None),
    (
        "student1024_vector_fp32",
        "models/supertonic-3",
        "models/student1024-float/onnx/vector_estimator.onnx",
    ),
    (
        "student1024_vector_int8",
        "models/supertonic-3",
        "models/student1024-int8/onnx/vector_estimator.onnx",
    ),
    (
        "student1536_vector_int8",
        "models/supertonic-3",
        "models/student1536-int8/onnx/vector_estimator.onnx",
    ),
]
if a.models:
    variants = [("reference", "models/supertonic-3", None)] + [
        (Path(m).name, m, None) for m in a.models
    ]
root = Path("samples") / Path(a.output).stem
root.mkdir(parents=True, exist_ok=True)
rows = []
for name, model, vector in variants:
    for i, voice in cases:
        lang, text = texts[i]
        e = Engine(model=model, voice=voice)
        if vector:
            import onnxruntime as ort
            from runtime import options

            e.sessions["vector_estimator"] = ort.InferenceSession(
                vector, options(), providers=["CPUExecutionProvider"]
            )
        wav, timing = e.synthesize(text, lang, 8, seed=400 + i)
        path = root / f"{name}_{i:02d}.wav"
        sf.write(path, wav, e.sample_rate, subtype="FLOAT")
        rows.append(
            {
                "variant": name,
                "case": i,
                "language": lang,
                "voice": voice,
                "text": text,
                "seed": 400 + i,
                "audio": str(path),
                "reference_audio": str(root / f"reference_{i:02d}.wav"),
                "timing": timing,
                "model": model,
                "model_manifest": json.loads(
                    (Path(model) / "manifest.json").read_text()
                )
                if (Path(model) / "manifest.json").exists()
                else {"guidance_scale": 4, "mode": "original"},
                "vector_override": vector,
            }
        )
        del e
        gc.collect()
        Path(a.output).write_text(
            json.dumps(
                {
                    "steps": 8,
                    "reference_guidance_scale": 4,
                    "split": "validation",
                    "rows": rows,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        print("generated", name, i, flush=True)
