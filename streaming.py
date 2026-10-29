"""Bounded text chunks and native-rate PCM for incremental TTS delivery.

Each text chunk is independently synthesized with all eight denoising steps.
This is sentence/chunk streaming, not causal decoding of a single sentence.
"""
from dataclasses import dataclass
import io
import re

import numpy as np
import soundfile as sf
from supertonic.utils import chunk_text


def split_text(text, lang="na", max_chunk_length=None, first_chunk_length=None):
    """Preserve sentence boundaries when possible; bound unpunctuated input too."""
    limit = max_chunk_length or (120 if lang == "ko" else 240)
    first = first_chunk_length or (60 if lang == "ko" else 120)
    if not 10 <= limit <= 500 or not 10 <= first <= 500:
        raise ValueError("chunk lengths must be between 10 and 500")
    pieces = []
    # The SDK handles abbreviations and paragraph/sentence boundaries. Its
    # max_len is a grouping target, so enforce the hard bound separately.
    for paragraph in re.split(r"\n\s*\n+", text.strip()):
        if not paragraph.strip():
            continue
        for sentence in chunk_text(paragraph, max_len=10):
            remaining = sentence.strip()
            while len(remaining) > limit:
                cut = remaining.rfind(" ", 0, limit + 1)
                if cut < limit // 2:
                    cut = limit
                pieces.append(remaining[:cut].strip())
                remaining = remaining[cut:].strip()
            if remaining:
                pieces.append(remaining)
        pieces.append(None)
    chunks, current = [], ""
    for piece in pieces:
        if piece is None:
            if current:
                chunks.append(current)
                current = ""
            continue
        target = min(first, limit) if not chunks else limit
        if current and len(current) + 1 + len(piece) > target:
            chunks.append(current)
            current = ""
        # A single complete sentence may exceed the first-chunk target;
        # keep its prosody intact up to the hard limit.
        current = f"{current} {piece}" if current else piece
    if current:
        chunks.append(current)
    if not chunks:
        raise ValueError("text must not be empty")
    return chunks


def encode_pcm(wave, sample_rate):
    buffer = io.BytesIO()
    sf.write(buffer, wave, sample_rate, format="RAW", subtype="PCM_16", endian="LITTLE")
    return buffer.getvalue()


@dataclass(frozen=True)
class AudioPart:
    index: int
    text: str
    wave: np.ndarray
    metrics: dict

    @property
    def samples(self):
        return len(self.wave)


def synthesize_part(engine, plan, index, style, cancel_event):
    wave, metrics = engine.synthesize(
        plan.chunks[index], plan.lang, 8, seed=plan.seed, speed=plan.speed,
        voice_style=style, cancel_event=cancel_event,
    )
    if index + 1 < len(plan.chunks) and plan.silence_duration:
        wave = np.concatenate((wave, np.zeros(
            int(plan.silence_duration * engine.sample_rate), dtype=np.float32)))
    return AudioPart(index, plan.chunks[index], wave, metrics)
