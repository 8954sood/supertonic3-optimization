"""Isolate frontend and decoder INT8 changes while denoising stays FP32."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import Engine, options
import argparse, json, gc
import onnxruntime as ort
import soundfile as sf

p = argparse.ArgumentParser()
p.add_argument("--output", default="results/quality_components_v2.json")
a = p.parse_args()
texts = json.loads(Path("distill/corpus.json").read_text())["validation"]
variants = {
    "frontend_int8_only": {
        "duration_predictor": "models/fast-full-int8/onnx/duration_predictor.onnx",
        "text_encoder": "models/fast-full-int8/onnx/text_encoder.onnx",
    },
    "decoder_int8_only": {"vocoder": "models/fast-full-int8/onnx/vocoder.onnx"},
    "student1024_fp32_frontend": {
        "vector_estimator": "models/student1024-int8/onnx/vector_estimator.onnx",
        "vocoder": "models/fast-full-int8/onnx/vocoder.onnx",
    },
}
root = Path("samples") / Path(a.output).stem
root.mkdir(parents=True, exist_ok=True)
rows = []
for name, overrides in variants.items():
    for i, voice in [(0, "M1"), (1, "F1"), (4, "M1"), (5, "F1")]:
        lang, text = texts[i]
        e = Engine(voice=voice)
        for part, path in overrides.items():
            e.sessions[part] = ort.InferenceSession(
                path, options(), providers=["CPUExecutionProvider"]
            )
        wave, timing = e.synthesize(text, lang, 8, seed=400 + i)
        path = root / f"{name}_{i:02d}.wav"
        sf.write(path, wave, e.sample_rate, subtype="FLOAT")
        rows.append(
            {
                "variant": name,
                "case": i,
                "language": lang,
                "voice": voice,
                "text": text,
                "seed": 400 + i,
                "audio": str(path),
                "reference_audio": f"samples/quality_ablation_v2/reference_{i:02d}.wav",
                "timing": timing,
                "overrides": overrides,
            }
        )
        Path(a.output).write_text(
            json.dumps(
                {"steps": 8, "guidance_scale": 4, "rows": rows},
                ensure_ascii=False,
                indent=2,
            )
        )
        del e
        gc.collect()
        print(name, i, flush=True)
