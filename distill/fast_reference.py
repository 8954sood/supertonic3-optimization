"""Training-data-only pitch-preserving tempo augmentation of original50 audio.

Never imported by runtime.py. Generated references are transformed original
audio, not unmodified upstream output and not a distilled-model teacher.
"""
import hashlib
import subprocess
import ctypes as ct
from pathlib import Path

import numpy as np


def rubberband_reference(wave, sample_rate, source_speed=1.0, target_speed=2.0,
                         library='.private/rubberband-v4/build/librubberband.3.dylib'):
    """Offline R3 reference generation only; GPL dependency is not shipped."""
    wave = np.ascontiguousarray(wave, dtype=np.float32).reshape(-1)
    if not 0.7 <= source_speed <= target_speed <= 2.0 or not np.isfinite(wave).all():
        raise ValueError('Invalid reference input')
    lib = ct.CDLL(str(Path(library).resolve()))
    state_type, float_pointer = ct.c_void_p, ct.POINTER(ct.c_float)
    lib.rubberband_new.argtypes = [ct.c_uint, ct.c_uint, ct.c_int, ct.c_double, ct.c_double]
    lib.rubberband_new.restype = state_type
    lib.rubberband_delete.argtypes = [state_type]
    lib.rubberband_get_engine_version.argtypes = [state_type]
    lib.rubberband_get_engine_version.restype = ct.c_int
    for name in ['study', 'process']:
        function = getattr(lib, 'rubberband_' + name)
        function.argtypes = [state_type, ct.POINTER(float_pointer), ct.c_uint, ct.c_int]
        function.restype = None
    for name in ['set_expected_input_duration', 'set_max_process_size']:
        getattr(lib, 'rubberband_' + name).argtypes = [state_type, ct.c_uint]
    lib.rubberband_available.argtypes = [state_type]
    lib.rubberband_available.restype = ct.c_int
    lib.rubberband_retrieve.argtypes = [state_type, ct.POINTER(float_pointer), ct.c_uint]
    lib.rubberband_retrieve.restype = ct.c_uint
    ratio = source_speed / target_speed
    state = lib.rubberband_new(sample_rate, 1, 0x20000000 | 0x00010000, ratio, 1.0)
    if not state:
        raise RuntimeError('Could not initialize reference stretcher')
    pieces = []
    try:
        if lib.rubberband_get_engine_version(state) != 3:
            raise RuntimeError('R3 reference engine required')
        lib.rubberband_set_expected_input_duration(state, len(wave))
        lib.rubberband_set_max_process_size(state, 1024)
        for pass_name in ['study', 'process']:
            for start in range(0, len(wave), 1024):
                part = wave[start:start+1024]
                pointers = (float_pointer * 1)(part.ctypes.data_as(float_pointer))
                getattr(lib, 'rubberband_' + pass_name)(state, pointers, len(part), int(start+len(part)==len(wave)))
                if pass_name == 'process':
                    available = lib.rubberband_available(state)
                    while available > 0:
                        out = np.empty(available, dtype=np.float32)
                        outputs = (float_pointer * 1)(out.ctypes.data_as(float_pointer))
                        count = lib.rubberband_retrieve(state, outputs, available)
                        if not count:
                            raise RuntimeError('Reference engine stalled')
                        pieces.append(out[:count].copy())
                        available = lib.rubberband_available(state)
        result = np.concatenate(pieces)
    finally:
        lib.rubberband_delete(state)
    expected = round(len(wave) * ratio)
    if abs(len(result) - expected) > 2 or not np.isfinite(result).all():
        raise RuntimeError('R3 output duration or samples invalid')
    if len(result) < expected:
        result = np.pad(result, (0, expected-len(result)))
    result = result[:expected]
    return result, {
        'method': 'Training-only Rubber Band4.0.0 R3 offline two-pass; pitch1.0, threading never; automatic offline padding/delay compensation',
        'upstream_revision': '1d95888bec3ae0a17c0c4af791810d5a63f6bc35',
        'source_speed': source_speed, 'target_speed': target_speed,
        'tempo_ratio': 1 / ratio, 'samples': len(result),
        'source_float_sha256': hashlib.sha256(wave.tobytes()).hexdigest(),
        'target_float_sha256': hashlib.sha256(result.tobytes()).hexdigest(),
    }


def fast_reference(wave, sample_rate, source_speed=1.0, target_speed=2.0):
    import imageio_ffmpeg

    wave = np.asarray(wave, dtype=np.float32).reshape(-1)
    if not 0.7 <= source_speed <= target_speed <= 2.0 or not np.isfinite(wave).all():
        raise ValueError('Invalid original reference or speeds')
    ratio = target_speed / source_speed
    frames = round(len(wave) / ratio)
    # A short trailing pad gives WSOLA access to the final source frame.
    # The reference is still trimmed to the requested half-length, not padded
    # to make an incomplete model utterance appear to meet its duration.
    padded = np.pad(wave, (0, round(sample_rate * 0.1)))
    factor = ratio ** 0.5
    filters = f'atempo={factor:.12g},atempo={factor:.12g},apad,atrim=end_sample={frames}'
    executable = imageio_ffmpeg.get_ffmpeg_exe()
    command = [executable, '-nostdin', '-hide_banner', '-loglevel', 'error',
               '-threads', '1', '-filter_threads', '1', '-f', 'f32le',
               '-ar', str(sample_rate), '-ac', '1', '-i', 'pipe:0',
               '-af', filters, '-f', 'f32le', '-acodec', 'pcm_f32le', 'pipe:1']
    result = subprocess.run(command, input=padded.astype('<f4').tobytes(),
                            capture_output=True, check=True)
    output = np.frombuffer(result.stdout, dtype='<f4').copy()
    if len(output) != frames or not np.isfinite(output).all():
        raise RuntimeError('Invalid fast training reference')
    return output, {
        'method': 'Training-only FFmpeg WSOLA atempo twice, trailing100ms source pad, exact target length',
        'source_speed': source_speed, 'target_speed': target_speed,
        'tempo_ratio': ratio, 'filters': filters, 'samples': frames,
        'source_float_sha256': hashlib.sha256(wave.tobytes()).hexdigest(),
        'target_float_sha256': hashlib.sha256(output.tobytes()).hexdigest(),
        'ffmpeg_version': subprocess.check_output([executable, '-version'], text=True).splitlines()[0],
    }
