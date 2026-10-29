"""Optional exact reduced-range integer kernel; original ORT backend remains available."""

import platform
from pathlib import Path


def library_path():
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "AMD64"}:
        raise RuntimeError(
            "The exact AVX2 kernel is currently built only for Linux x86_64"
        )
    if "avx2" not in Path("/proc/cpuinfo").read_text():
        raise RuntimeError(
            "This kernel requires AVX2; use the standard ORT engine on this CPU"
        )
    path = Path(__file__).resolve().parent / "native/libexact_int8.so"
    if not path.exists():
        raise FileNotFoundError("Build native/libexact_int8.so for this CPU first")
    return str(path)
