"""Paired acoustic diagnostics. These are distances, never perceptual quality scores."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runtime  # noqa: F401
import argparse
import json
import numpy as np
import librosa
import soundfile as sf
from functools import lru_cache

SR = 16000
HOP = 160


@lru_cache(maxsize=64)
def features(path):
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = librosa.resample(y, orig_sr=sr, target_sr=SR) if sr != SR else y
    spec = np.abs(librosa.stft(y, n_fft=1024, hop_length=HOP))
    mel = librosa.feature.melspectrogram(
        S=spec**2, sr=SR, n_fft=1024, n_mels=64, fmin=60, fmax=7600
    )
    db = 10 * np.log10(np.maximum(mel, 1e-9))
    rms = librosa.feature.rms(S=spec, frame_length=1024, hop_length=HOP)[0]
    f0, voiced, _ = librosa.pyin(
        y, sr=SR, fmin=65, fmax=500, frame_length=1024, hop_length=HOP, fill_na=np.nan
    )
    return {"wave": y, "db": db, "rms": rms, "f0": f0, "voiced": voiced, "spec": spec}


def compare(reference, candidate):
    a, b = features(reference), features(candidate)
    # Align spectral shape, but report duration separately so warping cannot hide it.
    x = librosa.feature.mfcc(S=a["db"], n_mfcc=20)[1:]
    y = librosa.feature.mfcc(S=b["db"], n_mfcc=20)[1:]
    _, path = librosa.sequence.dtw(
        X=x, Y=y, metric="euclidean", global_constraints=True, band_rad=0.15
    )
    i, j = path[:, 0], path[:, 1]
    active = (a["rms"][i] > max(a["rms"].max() * 0.01, 1e-5)) | (
        b["rms"][j] > max(b["rms"].max() * 0.01, 1e-5)
    )
    i, j = i[active], j[active]
    both = (
        a["voiced"][i]
        & b["voiced"][j]
        & np.isfinite(a["f0"][i])
        & np.isfinite(b["f0"][j])
    )
    cents = 1200 * np.log2(b["f0"][j[both]] / a["f0"][i[both]])
    count = min(len(a["wave"]), len(b["wave"]))
    err = a["wave"][:count] - b["wave"][:count]
    return {
        "duration_relative_error": len(b["wave"]) / len(a["wave"]) - 1,
        "dtw_logmel_mae_db": float(np.abs(a["db"][:, i] - b["db"][:, j]).mean()),
        "dtw_rms_mae_db": float(
            np.abs(
                20 * np.log10(np.maximum(a["rms"][i], 1e-6))
                - 20 * np.log10(np.maximum(b["rms"][j], 1e-6))
            ).mean()
        ),
        "voiced_unvoiced_agreement": float(np.mean(a["voiced"][i] == b["voiced"][j])),
        "voiced_pair_frames": int(both.sum()),
        "f0_median_abs_cents": float(np.median(np.abs(cents))) if len(cents) else None,
        "f0_gross_error_over_200_cents": float(np.mean(np.abs(cents) > 200))
        if len(cents)
        else None,
        "unaligned_wave_rmse": float(np.sqrt(np.mean(err**2))),
        "unaligned_wave_snr_db": float(
            10
            * np.log10(
                max(float(np.mean(a["wave"][:count] ** 2)), 1e-20)
                / max(float(np.mean(err**2)), 1e-20)
            )
        ),
        "reference_harmonic_interharmonic_2_4khz_db": harmonic_ratio(a),
        "candidate_harmonic_interharmonic_2_4khz_db": harmonic_ratio(b),
    }


def harmonic_ratio(values):
    """Custom voiced-frame diagnostic; not a reproduction of Paradee's metric."""
    frequency = librosa.fft_frequencies(sr=SR, n_fft=1024)
    band = (frequency >= 2000) & (frequency <= 4000)
    power = values["spec"][band] ** 2
    scores = []
    active = (
        values["voiced"]
        & np.isfinite(values["f0"])
        & (values["rms"] > values["rms"].max() * 0.05)
    )
    for index in np.flatnonzero(active):
        position = frequency[band] / values["f0"][index]
        harmonic = np.abs(position - np.round(position)) < 0.12
        between = np.abs(position - np.floor(position) - 0.5) < 0.12
        if harmonic.sum() < 3 or between.sum() < 3:
            continue
        scores.append(
            10
            * np.log10(
                (power[harmonic, index].mean() + 1e-15)
                / (power[between, index].mean() + 1e-15)
            )
        )
    return float(np.median(scores)) if scores else None


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("manifest")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    report = {
        "source": a.manifest,
        "definitions": {
            "analysis_sr": SR,
            "hop_samples": HOP,
            "dtw": "MFCC1..19 Euclidean, 15% Sakoe-Chiba band; active frame spectral/pitch comparison",
            "pitch": "pYIN 65..500Hz; cents only on jointly voiced frames; errors can reflect tracker mistakes",
            "logmel": "64 bands, 60..7600Hz, absolute 1e-9 power floor, no per-file gain normalization",
            "warning": "No MOS, speaker similarity, perceptual quality, or exact reproduction score is implied.",
            "harmonic_ratio": "Custom median voiced-frame 2..4kHz mean-power ratio: +/-0.12 F0 around integer versus half-integer harmonics, FFT1024; pYIN voicing and >5% peak RMS. Not directly comparable to Paradee's reported dB.",
        },
        "rows": [],
    }
    for row in json.loads(Path(a.manifest).read_text())["rows"]:
        if row["variant"] == "reference":
            continue
        r = {
            k: row[k]
            for k in [
                "variant",
                "case",
                "language",
                "voice",
                "audio",
                "reference_audio",
            ]
        }
        r.update(compare(row["reference_audio"], row["audio"]))
        report["rows"].append(r)
        Path(a.output).write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(
            r["variant"],
            r["case"],
            round(r["dtw_logmel_mae_db"], 3),
            r["f0_median_abs_cents"],
            flush=True,
        )
