"""Download the published model or training assets at verified, fixed Hub revisions."""

import argparse
import fnmatch
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath

from huggingface_hub import snapshot_download

MODEL_REPO = "babihoba/supertonic3-optimization"
MODEL_REVISION = "f1d05bdb691d8769557e19644f3a63089992daf0"
DATASET_REPO = "babihoba/supertonic3-optimization-data"
DATASET_REVISION = "e56d93c9959647c850b639a614e629e4e4ac1593"
MODELS = [
    "round5-original50-student", "round5-original50-preview",
    "round8-latent-retime-g135",
    "cfg-mixed-v1-selected", "cfg1024-v1-final", "mixed-v1-selected",
]


def extract_original50(directory):
    """Extract only the pinned, inventoried regular files; verify every tensor."""
    inventory = json.loads((directory / "extracted_manifest.json").read_text())["files"]
    expected = {entry["path"]: entry for entry in inventory}
    with tarfile.open(directory / "trajectories-original50-round5.tar.gz") as archive:
        members = archive.getmembers()
        if len(members) != len(expected) or {m.name for m in members} != set(expected):
            raise ValueError("Archive member inventory mismatch")
        for member in members:
            name = PurePosixPath(member.name)
            if (not member.isfile() or name.is_absolute() or ".." in name.parts
                    or name.parts[0] != "trajectories-original50-round5"
                    or member.size != expected[member.name]["bytes"]):
                raise ValueError("Unsafe or unexpected trajectory archive member")
        archive.extractall(directory, filter="data")
    for name, entry in expected.items():
        digest = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"Extracted artifact SHA-256 mismatch: {name}")
    return {"extracted_verified_files": len(expected)}


def download_and_verify(repo, kind, revision, destination, patterns, inventory_key):
    snapshot_download(
        repo_id=repo,
        repo_type=kind,
        revision=revision,
        local_dir=destination,
        allow_patterns=[*patterns, "asset_manifest.json"],
        token=False,
        max_workers=4,
    )
    inventory = json.loads((destination / "asset_manifest.json").read_text())[
        inventory_key
    ]
    selected = [
        entry
        for entry in inventory
        if any(fnmatch.fnmatch(entry["path"], pattern) for pattern in patterns)
    ]
    for entry in selected:
        name = PurePosixPath(entry["path"])
        if name.is_absolute() or ".." in name.parts:
            raise ValueError("Unsafe artifact path in manifest")
        path = destination / name
        if path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Artifact size mismatch: {name}")
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != entry["sha256"]:
            raise ValueError(f"Artifact SHA-256 mismatch: {name}")
    if not selected:
        raise ValueError("The published manifest did not match the requested assets")
    return {"repository": repo, "revision": revision, "verified_files": len(selected)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--purpose", choices=["inference", "reproduce", "all"], default="inference"
    )
    parser.add_argument("--model", choices=MODELS, default=MODELS[0])
    parser.add_argument("--root", default=".hf-assets")
    args = parser.parse_args()
    original50 = args.model.startswith("round5-original50-") or args.model == "round8-latent-retime-g135"
    root = Path(args.root)
    patterns = []
    if args.purpose in {"inference", "all"}:
        patterns.append(f"models/{args.model}/**")
    if args.purpose in {"reproduce", "all"}:
        patterns += [
            "reproduction/round5/**" if original50 else "reproduction/checkpoints/**",
            f"models/{args.model}/LICENSE",
        ]
        if not original50:
            patterns.append("reproduction/training/**")
    receipts = [
        download_and_verify(
            MODEL_REPO, "model", MODEL_REVISION, root / "model", patterns, "artifacts"
        )
    ]
    if args.purpose in {"reproduce", "all"}:
        receipts.append(
            download_and_verify(
                DATASET_REPO,
                "dataset",
                DATASET_REVISION,
                root / "data",
                ["round5/**"] if original50 else ["trajectories-v2/**"],
                "files",
            )
        )
        if original50:
            receipts.append(extract_original50(root / "data/round5"))
    root.mkdir(parents=True, exist_ok=True)
    receipt = root / f"verified_{args.purpose}.json"
    receipt.write_text(json.dumps(receipts, indent=2) + "\n")
    print(
        f"Verified {sum(item.get('verified_files', 0) for item in receipts)} published files; receipt: {receipt}"
    )


if __name__ == "__main__":
    main()
