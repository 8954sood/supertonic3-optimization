import argparse
from runtime import Engine
import json
import platform
import statistics
import time
from pathlib import Path
import numpy as np
import onnxruntime as ort
import psutil
import soundfile as sf

TEXTS = {
    "ko_short": ("ko", "안녕하세요. 오래된 컴퓨터에서도 빠르게 음성을 생성합니다."),
    "ko_long": (
        "ko",
        "오늘은 인공지능 음성 합성의 속도를 측정합니다. 하나의 CPU 스레드만 사용하면서도 자연스럽고 정확한 한국어를 읽는 것이 목표입니다. 다양한 문장 길이와 추론 단계를 비교하여 실제 사용에 적합한 설정을 찾겠습니다.",
    ),
    "en_short": (
        "en",
        "Hello. This voice is generated locally using a single CPU thread.",
    ),
    "en_long": (
        "en",
        "The morning sunlight came through the window as she opened her book. On an old computer, every unnecessary calculation matters. We measure the time spent preparing the text, generating the speech, and converting it into audio so that the results can be reproduced.",
    ),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant")
    p.add_argument("--model", default="models/supertonic-3")
    p.add_argument("--steps", nargs="+", type=int, default=[8])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--low-memory", action="store_true")
    p.add_argument("--cases", nargs="+", choices=list(TEXTS), default=list(TEXTS))
    p.add_argument("--output", default="results/baseline.json")
    p.add_argument("--voice", default="M1")
    p.add_argument(
        "--affinity", type=int, help="Pin to this logical core on Linux/Windows"
    )
    a = p.parse_args()
    if a.repeats < 1:
        p.error("repeats must be positive")
    proc = psutil.Process()
    if a.affinity is not None:
        if not hasattr(proc, "cpu_affinity"):
            p.error("CPU affinity is unavailable on this OS")
        proc.cpu_affinity([a.affinity])
    if any(s < 8 or s > 50 for s in a.steps):
        p.error("steps must be from 8 to 50")
    engine = Engine(
        model=a.model,
        variant=a.variant,
        voice=a.voice,
        profile=a.profile,
        low_memory=a.low_memory,
    )
    report = {
        "machine": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "onnxruntime": ort.__version__,
        "numpy": np.__version__,
        "providers": ["CPUExecutionProvider"],
        "threads": 1,
        "affinity": a.affinity,
        "voice": a.voice,
        "variant": a.variant,
        "model": a.model,
        "low_memory": a.low_memory,
        "backend": "native packed-int8 storage with FP32 BLAS"
        if engine.native_backend
        else (
            "numpy BLAS with packed int8 storage"
            if engine.numpy_backend
            else "ONNX Runtime kernels"
        ),
        "cpu_count": psutil.cpu_count(),
        "cpu_affinity": proc.cpu_affinity() if hasattr(proc, "cpu_affinity") else None,
        "cpu_info": Path("/proc/cpuinfo").read_text()
        if Path("/proc/cpuinfo").exists()
        else platform.processor(),
        "load_seconds": engine.load_seconds,
        "note": "Measured on this machine only. Apple M4 results do not predict Oracle Ampere A1 performance. Warm synthesis excludes model loading and file I/O. No CPU quota is emulated by this script.",
        "rows": [],
    }
    sample_dir = Path("samples") / Path(a.output).stem
    sample_dir.mkdir(parents=True, exist_ok=True)
    for name in a.cases:
        lang, text = TEXTS[name]
        for steps in a.steps:
            engine.synthesize(text, lang, steps)  # warm-up, excluded
            runs = []
            cpu = []
            for _ in range(a.repeats):
                c = time.process_time()
                wav, metrics = engine.synthesize(text, lang, steps)
                cpu.append(time.process_time() - c)
                runs.append(metrics)
            row = {
                "case": name,
                "lang": lang,
                "text": text,
                "steps": steps,
                "runs": runs,
                "median_seconds": statistics.median(x["seconds"] for x in runs),
                "median_rtf": statistics.median(x["rtf"] for x in runs),
                "median_realtime_factor": statistics.median(
                    x["realtime_factor"] for x in runs
                ),
                "cpu_wall_ratio": sum(cpu) / sum(x["seconds"] for x in runs),
                "rss_mb": proc.memory_info().rss / 1e6,
                "peak": float(np.max(np.abs(wav))),
            }
            report["rows"].append(row)
            sf.write(
                str(sample_dir / f"{name}_{steps}steps.wav"), wav, engine.sample_rate
            )
            print(
                f"{name} steps={steps}: {row['median_seconds']:.3f}s, {row['median_realtime_factor']:.1f}x realtime, CPU/wall={row['cpu_wall_ratio']:.2f}",
                flush=True,
            )
    if a.profile:
        report["profiles"] = engine.finish_profiles()
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
