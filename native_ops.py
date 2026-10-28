"""Loader for the optional native packed-weight BLAS operator (Linux x86 build)."""

import os
from pathlib import Path
import numpy as np


def library_path():
    candidates = list(
        (Path(np.__file__).resolve().parent.parent / "numpy.libs").glob("*openblas*")
    )
    if not candidates:
        raise RuntimeError(
            "Native backend requires a NumPy Linux wheel with ILP64 OpenBLAS"
        )
    os.environ["TTS_OPENBLAS_LIBRARY"] = str(candidates[0])
    library = Path(__file__).resolve().parent / "native/libpacked_blas.so"
    if not library.exists():
        raise FileNotFoundError("Build native/libpacked_blas.so for this CPU first")
    return str(library)
