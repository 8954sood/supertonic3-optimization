"""Portable Supertonic 3 inference. All compute is sequential CPU, one thread."""

import os

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from supertonic.core import UnicodeProcessor, get_latent_mask
from supertonic.loader import load_voice_style_from_json_file

PARTS = ("duration_predictor", "text_encoder", "vector_estimator", "vocoder")


class SynthesisCancelled(RuntimeError):
    """A caller cancelled synthesis between model operations."""


def options(
    profile=False,
    optimize=True,
    low_memory=False,
    numpy_backend=False,
    native_backend=False,
    exact_int8_backend=False,
    row_norm_backend=False,
):
    opt = ort.SessionOptions()
    if row_norm_backend:
        from row_norm_ops import library_path

        opt.register_custom_ops_library(library_path())
    if numpy_backend:
        from numpy_ops import library_path

        opt.register_custom_ops_library(library_path())
    if native_backend:
        from native_ops import library_path

        opt.register_custom_ops_library(library_path())
    if exact_int8_backend:
        from exact_ops import library_path

        opt.register_custom_ops_library(library_path())
        # Variable utterance lengths should reuse arena blocks rather than keep
        # separate large traced allocation patterns. Prepacking stays enabled.
        opt.enable_mem_pattern = False
    opt.intra_op_num_threads = 1
    opt.inter_op_num_threads = 1
    opt.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opt.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if optimize
        else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
    opt.add_session_config_entry("session.intra_op.allow_spinning", "0")
    opt.add_session_config_entry("session.inter_op.allow_spinning", "0")
    opt.enable_profiling = profile
    if low_memory:
        opt.enable_cpu_mem_arena = False
        opt.enable_mem_pattern = False
        opt.add_session_config_entry("session.disable_prepacking", "1")
    if profile:
        Path("results").mkdir(exist_ok=True)
        opt.profile_file_prefix = "results/ort_profile"
    return opt


class Engine:
    def __init__(
        self,
        model="models/supertonic-3",
        variant=None,
        voice="M1",
        profile=False,
        low_memory=False,
    ):
        start = time.perf_counter()
        self.root = Path(model)
        manifest_paths = (
            [Path(variant) / "manifest.json", Path(variant).parent / "manifest.json"]
            if variant
            else []
        ) + [self.root / "manifest.json"]
        manifest_path = next((p for p in manifest_paths if p.exists()), None)
        metadata = json.loads(manifest_path.read_text()) if manifest_path else {}
        self.supported_steps = metadata.get("supported_steps")
        self.speed_strategy = metadata.get("speed_strategy", "native-duration")
        self.latent_head_generation_speed = metadata.get("latent_head_generation_speed", 1.05)
        self.latent_head_supported_speeds = metadata.get("latent_head_supported_speeds", [])
        self.numpy_backend = metadata.get("backend") == "numpy"
        self.native_backend = metadata.get("backend") == "native-blas"
        self.exact_int8_backend = metadata.get("backend") == "native-exact-int8"
        self.row_norm_backend = bool(metadata.get("row_norm_backend"))
        self.cfg = json.loads((self.root / "onnx/tts.json").read_text())
        self.processor = UnicodeProcessor(str(self.root / "onnx/unicode_indexer.json"))
        self.style = load_voice_style_from_json_file(
            self.root / "voice_styles" / f"{voice}.json"
        )
        self.sessions = {}

        def model_path(directory, name):
            onnx_path = Path(directory) / f"{name}.onnx"
            ort_path = Path(directory) / f"{name}.ort"
            return (
                onnx_path if onnx_path.exists() or not ort_path.exists() else ort_path
            )

        if variant and not model_path(variant, "vector_estimator").is_file():
            raise FileNotFoundError(f"Variant has no vector_estimator.onnx: {variant}")
        for part in PARTS:
            candidate = model_path(variant, part) if variant else None
            path = (
                candidate
                if candidate and candidate.exists()
                else model_path(self.root / "onnx", part)
            )
            self.sessions[part] = ort.InferenceSession(
                str(path),
                options(
                    profile,
                    low_memory=low_memory,
                    numpy_backend=self.numpy_backend,
                    native_backend=self.native_backend,
                    exact_int8_backend=self.exact_int8_backend,
                    row_norm_backend=self.row_norm_backend,
                ),
                providers=["CPUExecutionProvider"],
            )
        self.condition = None
        condition_path = model_path(
            Path(variant) if variant else self.root / "onnx", "condition"
        )
        if condition_path.exists():
            self.condition = ort.InferenceSession(
                str(condition_path),
                options(
                    profile,
                    low_memory=low_memory,
                    numpy_backend=self.numpy_backend,
                    native_backend=self.native_backend,
                    exact_int8_backend=self.exact_int8_backend,
                    row_norm_backend=self.row_norm_backend,
                ),
                providers=["CPUExecutionProvider"],
            )
        self.step_condition = None
        self._step_conditions = ()
        step_path = model_path(
            Path(variant) if variant else self.root / "onnx", "step_condition"
        )
        if step_path.exists():
            if metadata.get("step_condition_steps") != 8 or self.supported_steps != [8]:
                raise ValueError("Timestep cache requires an eight-step model package")
            self.step_condition = ort.InferenceSession(
                str(step_path),
                options(profile, low_memory=low_memory,
                        numpy_backend=self.numpy_backend, native_backend=self.native_backend,
                        exact_int8_backend=self.exact_int8_backend,
                        row_norm_backend=self.row_norm_backend),
                providers=["CPUExecutionProvider"],
            )
            names = [value.name for value in self.step_condition.get_outputs()]
            cached = []
            for step in range(8):
                values = self.step_condition.run(None, {
                    "current_step": np.array([step], np.float32),
                    "total_step": np.array([8], np.float32),
                })
                for value in values:
                    value.setflags(write=False)
                cached.append(dict(zip(names, values)))
            self._step_conditions = tuple(cached)
        self.sample_rate = self.cfg["ae"]["sample_rate"]
        self.load_seconds = time.perf_counter() - start

    def synthesize(
        self, text, lang="ko", steps=8, seed=42, speed=1.05,
        *, voice_style=None, cancel_event=None,
    ):
        if not isinstance(steps, int) or not 8 <= steps <= 50:
            raise ValueError("steps must be an integer from 8 to 50")
        if self.supported_steps and steps not in self.supported_steps:
            raise ValueError(
                f"This model supports trained step counts: {self.supported_steps}"
            )
        if not text.strip():
            raise ValueError("text must not be empty")
        if not 0.7 <= speed <= 2.0:
            raise ValueError("speed must be between 0.7 and 2.0")
        times = {}
        start = time.perf_counter()
        style = self.style if voice_style is None else voice_style

        def run(part, feed):
            if cancel_event is not None and cancel_event.is_set():
                raise SynthesisCancelled("Synthesis cancelled")
            t = time.perf_counter()
            session = self.condition if part == "condition" else self.sessions[part]
            required = {x.name for x in session.get_inputs()}
            out = session.run(None, {k: v for k, v in feed.items() if k in required})
            times[part] = times.get(part, 0) + time.perf_counter() - t
            return out

        ids, mask = self.processor([text], lang)
        raw_duration = run(
            "duration_predictor",
            {"text_ids": ids, "text_mask": mask, "style_dp": style.dp},
        )[0]
        generation_speed = (
            self.latent_head_generation_speed
            if self.speed_strategy in ("learned-latent-head", "latent-retime")
            and speed in self.latent_head_supported_speeds
            else speed
        )
        duration = raw_duration / generation_speed
        output_duration = raw_duration / speed
        emb = run(
            "text_encoder",
            {"text_ids": ids, "text_mask": mask, "style_ttl": style.ttl},
        )[0]
        chunk = (
            self.cfg["ae"]["base_chunk_size"] * self.cfg["ttl"]["chunk_compress_factor"]
        )
        length = int((duration.max() * self.sample_rate + chunk - 1) / chunk)
        latent_mask = get_latent_mask(
            (duration * self.sample_rate).astype(np.int64),
            self.cfg["ae"]["base_chunk_size"],
            self.cfg["ttl"]["chunk_compress_factor"],
        )
        # Match upstream's legacy RNG for reproducible numerical comparisons.
        rng = np.random.RandomState(seed)
        xt = (
            rng.randn(
                1,
                self.cfg["ttl"]["latent_dim"]
                * self.cfg["ttl"]["chunk_compress_factor"],
                length,
            ).astype(np.float32)
            * latent_mask
        )
        feed = {
            "speed": np.array([speed], np.float32),
            "text_emb": emb,
            "style_ttl": style.ttl,
            "text_mask": mask,
            "latent_mask": latent_mask,
            "total_step": np.array([steps], np.float32),
        }
        if self.condition:
            values = run("condition", feed)
            feed.update(zip([x.name for x in self.condition.get_outputs()], values))
        for step in range(steps):
            feed.update(noisy_latent=xt, current_step=np.array([step], np.float32))
            if self._step_conditions:
                feed.update(self._step_conditions[step])
            xt = run("vector_estimator", feed)[0]
        output_frames = int((output_duration.max() * self.sample_rate + chunk - 1) / chunk)
        wav = run(
            "vocoder",
            {"latent": xt, "target_frames": np.array([output_frames], np.int64)},
        )[0].reshape(-1)[: int(float(output_duration[0]) * self.sample_rate)]
        elapsed = time.perf_counter() - start
        if not np.isfinite(wav).all() or not len(wav):
            raise RuntimeError("Invalid audio output")
        return wav, {
            "seconds": elapsed,
            "audio_seconds": len(wav) / self.sample_rate,
            "speed": float(speed),
            "generation_speed": float(generation_speed),
            "speed_strategy": self.speed_strategy,
            "rtf": elapsed / (len(wav) / self.sample_rate),
            "realtime_factor": (len(wav) / self.sample_rate) / elapsed,
            "stages": times,
        }

    def finish_profiles(self):
        sessions = list(self.sessions.values()) + (
            [self.condition] if self.condition else []
        ) + ([self.step_condition] if self.step_condition else [])
        return [s.end_profiling() for s in sessions]
