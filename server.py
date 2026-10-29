"""Supertonic-compatible HTTP APIs with bounded, eight-step chunk streaming."""
from __future__ import annotations

# Establish single-thread numerical settings before any numerical imports.
from runtime import Engine, SynthesisCancelled

import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import threading
import time
from typing import Literal

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
import numpy as np
from pydantic import BaseModel, Field
import soundfile as sf
from supertonic import __version__ as sdk_version
from supertonic.config import AVAILABLE_LANGUAGES
from supertonic.core import Style
from supertonic.loader import load_voice_style_from_json_file
from supertonic.server import styles_store
from supertonic.server.audio import encode_audio, format_to_mime
from supertonic.utils import validate_voice_style_format

from streaming import encode_pcm, split_text, synthesize_part

logger = logging.getLogger(__name__)
DEFAULT_MODEL = ".hf-assets/model/models/round5-original50-student"
SERVER_VERSION = "0.1.0"


class ApiError(Exception):
    def __init__(self, status, message, code):
        self.status, self.message, self.code = status, message, code


def error_response(error):
    return JSONResponse(
        {"error": {"message": error.message, "type": "server_error" if error.status >= 500
                   else "invalid_request_error", "code": error.code}},
        status_code=error.status,
        headers={"Retry-After": "1"} if error.code == "server_busy" else None,
    )


class SynthesisOptions(BaseModel):
    voice: str | None = None
    lang: str | None = None
    speed: float | None = Field(None, ge=0.7, le=2.0, multiple_of=0.01, allow_inf_nan=False)
    steps: Literal[8] | None = None
    max_chunk_length: int | None = Field(None, ge=10, le=500)
    first_chunk_length: int | None = Field(None, ge=10, le=500)
    silence_duration: float | None = Field(None, ge=0, le=10, allow_inf_nan=False)
    seed: int | None = Field(None, ge=0, le=2**32 - 1)


class TTSRequest(SynthesisOptions):
    text: str = Field(min_length=1)
    response_format: str | None = None
    stream: bool = False
    stream_format: Literal["audio", "sse"] = "audio"


class SpeechRequest(SynthesisOptions):
    model: str
    input: str = Field(min_length=1)
    response_format: str | None = None
    stream: bool = False
    stream_format: Literal["audio", "sse"] = "audio"


class BatchItem(SynthesisOptions):
    text: str = Field(min_length=1)


class BatchRequest(BaseModel):
    items: list[BatchItem] = Field(min_length=1, max_length=64)
    defaults: SynthesisOptions | None = None
    response_format: str | None = None


@dataclass(frozen=True)
class Plan:
    chunks: list[str]
    voice: str
    lang: str
    speed: float
    seed: int
    silence_duration: float
    style: Style


class Lease:
    def __init__(self, service):
        self.service = service
        self.cancel = threading.Event()
        self.closed = False
        service.cancellations.add(self.cancel)

    def close(self):
        if not self.closed:
            self.closed = True
            self.cancel.set()
            self.service.cancellations.discard(self.cancel)
            self.service.lock.release()
            self.service.admitted -= 1


class AudioStreamingResponse(StreamingResponse):
    """Release admission even if response headers fail before iteration starts."""
    def __init__(self, *args, lease, **kwargs):
        self.lease = lease
        super().__init__(*args, **kwargs)

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            self.lease.close()


class BodyLimit:
    """Bound JSON/multipart buffering, including chunked HTTP request bodies."""
    def __init__(self, app, max_bytes=2 * 1024 * 1024):
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        chunks, size = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            size += len(message.get("body", b""))
            if size > self.max_bytes:
                response = error_response(ApiError(413, "Request body too large", "payload_too_large"))
                return await response(scope, receive, send)
            chunks.append(message.get("body", b""))
            if not message.get("more_body"):
                break
        body = b"".join(chunks)
        sent = False

        async def replay():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


class ModelService:
    def __init__(self, model_path, model_id, engine, voices, custom_styles_dir,
                 max_pending, max_text_length, max_audio_seconds, packet_ms):
        self.model_path, self.model_id = Path(model_path), model_id
        self.engine, self.voices = engine, dict(voices or {})
        self.custom_dir = Path(custom_styles_dir)
        self.custom_names = set()
        self.max_pending, self.max_text_length = max_pending, max_text_length
        self.max_audio_seconds, self.packet_ms = max_audio_seconds, packet_ms
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")
        self.lock = asyncio.Lock()
        self.admitted, self.cancellations = 0, set()
        self.ready = False

    def load(self):
        if self.engine is None:
            self.engine = Engine(model=self.model_path)
        if not self.voices:
            self.voices = {p.stem: load_voice_style_from_json_file(p)
                           for p in sorted((self.engine.root / "voice_styles").glob("*.json"))}
        self.builtin_names = set(self.voices)
        if not self.voices:
            raise ValueError("The model has no voice styles")
        for name, path in styles_store.scan(self.custom_dir).items():
            if name in self.builtin_names:
                continue
            try:
                style = load_voice_style_from_json_file(path)
                self.validate_style(style)
            except ValueError:
                logger.warning("Skipping incompatible custom style %s", name)
                continue
            self.voices[name] = style
            self.custom_names.add(name)
        self.ready = True

    def validate_style(self, style):
        reference = next(iter(self.voices.values()))
        for value, expected in [(style.ttl, reference.ttl), (style.dp, reference.dp)]:
            if value.shape != expected.shape or not np.isfinite(value).all():
                raise ValueError("Voice style dimensions or values do not match the loaded model")

    def plan(self, text, options):
        if not self.ready:
            raise ApiError(503, "Server not ready", "not_ready")
        if not text.strip() or len(text) > self.max_text_length:
            raise ApiError(400, f"Text must contain 1..{self.max_text_length} characters", "invalid_text")
        voice, lang = options.voice or "M1", options.lang or "na"
        if voice not in self.voices:
            raise ApiError(400, "Unknown voice", "unknown_voice")
        if lang not in AVAILABLE_LANGUAGES:
            raise ApiError(400, "Unsupported language", "unsupported_lang")
        return Plan(split_text(text, lang, options.max_chunk_length, options.first_chunk_length),
                    voice, lang, options.speed if options.speed is not None else 1.05,
                    options.seed if options.seed is not None else 42,
                    options.silence_duration if options.silence_duration is not None else 0.3,
                    self.voices[voice])

    async def acquire(self, request):
        if self.admitted >= 1 + self.max_pending:
            raise ApiError(503, "Synthesis queue is full; retry shortly", "server_busy")
        self.admitted += 1
        try:
            while True:
                if await request.is_disconnected():
                    raise ApiError(499, "Client disconnected", "client_disconnected")
                try:
                    await asyncio.wait_for(self.lock.acquire(), timeout=0.1)
                    return Lease(self)
                except TimeoutError:
                    continue
        except BaseException:
            self.admitted -= 1
            raise

    async def compute(self, request, lease, fn):
        # The executor has exactly one worker even if an HTTP task is cancelled
        # while its ONNX call is still returning. No second inference can overlap.
        future = asyncio.get_running_loop().run_in_executor(self.pool, fn)
        try:
            while not future.done():
                await asyncio.wait({future}, timeout=0.05)
                if await request.is_disconnected():
                    raise ApiError(499, "Client disconnected", "client_disconnected")
            return future.result()
        except BaseException:
            lease.cancel.set()
            future.cancel()
            raise

    async def part(self, request, lease, plan, index):
        return await self.compute(request, lease, lambda: synthesize_part(
            self.engine, plan, index, plan.style, lease.cancel))

    def check_audio_length(self, samples):
        if samples > self.max_audio_seconds * self.engine.sample_rate:
            raise ApiError(413, "Generated audio exceeds the server duration limit", "audio_too_long")

    async def collect(self, request, lease, plan):
        waves, samples, seconds = [], 0, 0.0
        for index in range(len(plan.chunks)):
            part = await self.part(request, lease, plan, index)
            samples += part.samples
            self.check_audio_length(samples)
            waves.append(part.wave)
            seconds += part.metrics["seconds"]
        return np.concatenate(waves), seconds


def coerce_format(value, stream=False):
    fmt = (value or ("pcm" if stream else "wav")).strip().lower()
    if fmt not in (("pcm",) if stream else ("wav", "flac", "ogg", "pcm")):
        raise ApiError(400, "Streaming requires pcm; complete responses support wav/flac/ogg/pcm",
                       "unsupported_response_format")
    return fmt


def event(name, data):
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False, allow_nan=False)}\n\n".encode()


def create_app(*, model_path=DEFAULT_MODEL, model_id="supertonic-3", engine=None,
               voice_styles=None, custom_styles_dir=".private/server-styles", cors_origins=None,
               max_pending=4, max_text_length=10000, max_audio_seconds=180, packet_ms=40):
    if max_pending < 0 or max_text_length < 1 or max_audio_seconds <= 0 or not 5 <= packet_ms <= 200:
        raise ValueError("Invalid server resource limits")
    service = ModelService(model_path, model_id, engine, voice_styles, custom_styles_dir,
                           max_pending, max_text_length, max_audio_seconds, packet_ms)

    @asynccontextmanager
    async def lifespan(app):
        try:
            await asyncio.get_running_loop().run_in_executor(service.pool, service.load)
            yield
        finally:
            service.ready = False
            for cancel in list(service.cancellations):
                cancel.set()
            await asyncio.to_thread(service.pool.shutdown, wait=True, cancel_futures=True)

    app = FastAPI(title="Supertonic 3 Optimized", version=SERVER_VERSION, lifespan=lifespan,
                  description="Compatible REST APIs and eight-step sentence/chunk PCM/SSE streaming.")
    app.state.service = service
    app.add_middleware(BodyLimit)
    if cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=list(cors_origins),
                           allow_methods=["GET", "POST"], allow_headers=["*"],
                           expose_headers=["X-Sample-Rate", "X-Audio-Duration", "X-RTF",
                                           "X-First-Audio-Latency", "X-Chunk-Count", "X-Audio-Format"])

    @app.exception_handler(ApiError)
    async def api_error(request, exc):
        return error_response(exc)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        # Never echo the submitted text or style tensors into an error response.
        fields = [".".join(map(str, e["loc"])) for e in exc.errors()]
        return error_response(ApiError(422, "Invalid request fields: " + ", ".join(fields),
                                       "validation_error"))

    @app.exception_handler(Exception)
    async def synthesis_error(request, exc):
        logger.exception("Request failed")
        return error_response(ApiError(500, "Synthesis failed", "synthesis_failed"))

    @app.get("/v1/health")
    @app.get("/health", include_in_schema=False)
    async def health():
        return JSONResponse({"status": "ok" if service.ready else "loading", "model": model_id,
                             "sample_rate": service.engine.sample_rate if service.ready else None,
                             "version": sdk_version, "server_version": SERVER_VERSION,
                             "voices_loaded": len(service.voices), "steps": 8, "threads": 1,
                             "streaming": "text-chunks", "admitted_requests": service.admitted},
                            status_code=200 if service.ready else 503)

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": model_id, "object": "model", "owned_by": "local"}]}

    @app.get("/v1/styles")
    async def styles():
        return {"styles": [{"name": name, "kind": "custom" if name in service.custom_names else "builtin"}
                           for name in sorted(service.voices)]}

    @app.post("/v1/styles/import")
    async def import_style(request: Request, overwrite: bool = False,
                           file: UploadFile | None = File(None), name: str | None = Form(None)):
        try:
            if request.headers.get("content-type", "").startswith("multipart/form-data"):
                if file is None:
                    raise ApiError(400, "Missing file", "missing_file")
                raw = await file.read(1024 * 1024 + 1)
                if len(raw) > 1024 * 1024:
                    raise ApiError(413, "Voice style file too large", "payload_too_large")
                payload = json.loads(raw)
                chosen = name or Path(file.filename or "imported").stem
            else:
                payload = await request.json()
                chosen = payload.get("name") if isinstance(payload, dict) else None
            if not isinstance(chosen, str) or not isinstance(payload, dict):
                raise ApiError(400, "Missing style name or JSON object", "invalid_style_payload")
            styles_store.sanitize_name(chosen)
            if not validate_voice_style_format(payload):
                raise ValueError("Invalid voice style JSON")
            def tensor(key):
                node = payload[key]
                return np.asarray(node["data"], dtype=np.float32).reshape(node["dims"])
            style = Style(tensor("style_ttl"), tensor("style_dp"))
            service.validate_style(style)
            target = styles_store.save(service.custom_dir, chosen, payload,
                                       builtin_names=service.builtin_names, overwrite=overwrite)
        except ApiError:
            raise
        except styles_store.StyleNameConflict:
            raise ApiError(409, "Voice style name already exists", "style_name_conflict")
        except (ValueError, KeyError, TypeError, OverflowError):
            raise ApiError(400, "Invalid voice style name, dimensions or JSON", "invalid_style_payload")
        service.voices[target.stem] = style
        service.custom_names.add(target.stem)
        return {"name": target.stem, "stored_at": target.name}

    async def deliver(req, request, force_stream=False):
        start = time.perf_counter()
        if isinstance(req, SpeechRequest) and req.model != model_id:
            raise ApiError(400, "Requested model is not loaded", "model_not_loaded")
        plan = service.plan(req.input if isinstance(req, SpeechRequest) else req.text, req)
        stream = force_stream or req.stream
        fmt = coerce_format(req.response_format, stream)
        lease = await service.acquire(request)
        try:
            if not stream:
                wave, seconds = await service.collect(request, lease, plan)
                rate = service.engine.sample_rate
                body = await service.compute(request, lease, lambda: encode_pcm(wave, rate)
                    if fmt == "pcm" else encode_audio(wave, rate, fmt))
                duration = len(wave) / rate
                lease.close()
                return Response(body, media_type="application/octet-stream" if fmt == "pcm" else format_to_mime(fmt),
                                headers={"X-Sample-Rate": str(rate), "X-Audio-Duration": f"{duration:.6f}",
                                         "X-RTF": f"{seconds/duration:.6f}", "X-Chunk-Count": str(len(plan.chunks)),
                                         "X-Supertonic-Version": sdk_version,
                                         "X-Audio-Format": "pcm_s16le" if fmt == "pcm" else fmt})
            # Complete the first real model chunk before committing HTTP status.
            first = await service.part(request, lease, plan, 0)
            service.check_audio_length(first.samples)
            first_pcm = encode_pcm(first.wave, service.engine.sample_rate)
        except BaseException:
            lease.close()
            raise
        rate = service.engine.sample_rate
        first_latency = time.perf_counter() - start

        async def generate():
            total_samples, synthesis_seconds = 0, 0.0
            sse = req.stream_format == "sse"
            try:
                if sse:
                    yield event("start", {"sample_rate": rate, "channels": 1, "format": "pcm_s16le",
                                          "steps": 8, "chunks": len(plan.chunks),
                                          "first_audio_latency": first_latency})
                for index in range(len(plan.chunks)):
                    if lease.cancel.is_set() or await request.is_disconnected():
                        return
                    part = first if index == 0 else await service.part(request, lease, plan, index)
                    total_samples += part.samples
                    service.check_audio_length(total_samples)
                    synthesis_seconds += part.metrics["seconds"]
                    pcm = first_pcm if index == 0 else encode_pcm(part.wave, rate)
                    if sse:
                        yield event("audio", {"index": index, "text": part.text, "samples": part.samples,
                                              "audio_base64": base64.b64encode(pcm).decode("ascii"),
                                              "duration_s": part.samples/rate,
                                              "synthesis_seconds": part.metrics["seconds"]})
                        await asyncio.sleep(0)
                    else:
                        packet_size = max(1, int(rate * service.packet_ms / 1000)) * 2
                        for offset in range(0, len(pcm), packet_size):
                            yield pcm[offset:offset+packet_size]
                            await asyncio.sleep(0)
                if sse:
                    duration = total_samples/rate
                    yield event("end", {"audio_seconds": duration, "synthesis_seconds": synthesis_seconds,
                                        "rtf": synthesis_seconds/duration, "chunks": len(plan.chunks),
                                        "wall_seconds": time.perf_counter()-start})
            except (ApiError, SynthesisCancelled) as exc:
                if sse and not lease.cancel.is_set():
                    yield event("error", {"code": exc.code if isinstance(exc, ApiError) else "cancelled"})
                elif not lease.cancel.is_set():
                    raise
            except Exception:
                logger.exception("Streaming synthesis failed")
                if sse:
                    yield event("error", {"code": "synthesis_failed"})
                else:
                    raise
            finally:
                lease.close()

        return AudioStreamingResponse(generate(), lease=lease,
            media_type="text/event-stream" if req.stream_format == "sse" else "application/octet-stream",
            headers={"X-Sample-Rate": str(rate), "X-Audio-Format": "pcm_s16le", "X-Audio-Channels": "1",
                     "X-Chunk-Count": str(len(plan.chunks)), "X-First-Audio-Latency": f"{first_latency:.6f}",
                     "X-Supertonic-Version": sdk_version, "Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    @app.post("/v1/tts")
    async def native(req: TTSRequest, request: Request):
        return await deliver(req, request)

    @app.post("/v1/tts/stream")
    async def native_stream(req: TTSRequest, request: Request):
        return await deliver(req, request, force_stream=True)

    @app.post("/v1/audio/speech")
    async def speech(req: SpeechRequest, request: Request):
        return await deliver(req, request)

    @app.post("/v1/tts/batch")
    async def batch(req: BatchRequest, request: Request):
        fmt = coerce_format(req.response_format)
        if sum(len(item.text) for item in req.items) > service.max_text_length:
            raise ApiError(400, "Batch text exceeds server limit", "invalid_text")
        plans = []
        for item in req.items:
            options = req.defaults.model_dump(exclude_none=True) if req.defaults else {}
            options.update(item.model_dump(exclude_none=True, exclude={"text"}))
            plans.append(service.plan(item.text, SynthesisOptions(**options)))
        lease = await service.acquire(request)
        try:
            results, total = [], 0
            for plan in plans:
                wave, _ = await service.collect(request, lease, plan)
                total += len(wave)
                service.check_audio_length(total)
                rate = service.engine.sample_rate
                body = await service.compute(request, lease, lambda: encode_pcm(wave, rate)
                    if fmt == "pcm" else encode_audio(wave, rate, fmt))
                results.append({"audio_base64": base64.b64encode(body).decode("ascii"),
                                "duration_s": len(wave)/rate, "format": fmt, "sample_rate": rate})
            return {"items": results}
        finally:
            lease.close()

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def demo():
        return (Path(__file__).parent / "web/streaming.html").read_text()

    return app
