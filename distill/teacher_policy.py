"""Original-only teacher policy required by the project owner."""

import json
from pathlib import Path

UPSTREAM_REPO = "Supertone/supertonic-3"
UPSTREAM_REVISION = "3cadd1ee6394adea1bd021217a0e650ede09a323"


def assert_original_model(root):
    root = Path(root)
    revision = json.loads((root / "revision.json").read_text())
    if revision != {"repo": UPSTREAM_REPO, "revision": UPSTREAM_REVISION}:
        raise ValueError("Teacher must be the pinned original Supertonic 3 model")
    if (root / "manifest.json").exists():
        raise ValueError("A derived model package cannot be used as teacher")
    return revision


def assert_original_data(metadata):
    if metadata.get("complete") is False:
        raise ValueError("Incomplete teacher data is not allowed for training")
    if metadata.get("input_state_source") or metadata.get("teacher_package"):
        raise ValueError(
            "Student-retargeted datasets are rejected: only original FP32 teacher targets are allowed"
        )
    origin = metadata.get("teacher_origin")
    if origin:
        if (
            origin.get("repo") != UPSTREAM_REPO
            or origin.get("revision") != UPSTREAM_REVISION
            or origin.get("distilled") is not False
            or origin.get("precision") != "FP32"
            or origin.get("guidance_scale") != 4
        ):
            raise ValueError("Invalid original FP32 CFG4 teacher provenance")
    elif metadata.get("teacher") != "original FP32 CFG4 8 steps":
        raise ValueError(
            "Unrecognized teacher provenance; an original-only dataset is required"
        )
