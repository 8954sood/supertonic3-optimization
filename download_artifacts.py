"""Download the published model or training assets at verified, fixed Hub revisions."""

import argparse
import fnmatch
import hashlib
import json
from pathlib import Path, PurePosixPath

from huggingface_hub import snapshot_download

MODEL_REPO = "babihoba/supertonic3-optimization"
MODEL_REVISION = "85bc100ca8d7cefad17b63a4c7ba489638dbe2f2"
DATASET_REPO = "babihoba/supertonic3-optimization-data"
DATASET_REVISION = "817056ae3e9b8fa1a2e464a212bb2ba1f4c5196a"
MODELS = ["cfg-mixed-v1-selected", "cfg1024-v1-final", "mixed-v1-selected"]


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
    root = Path(args.root)
    patterns = []
    if args.purpose in {"inference", "all"}:
        patterns.append(f"models/{args.model}/**")
    if args.purpose in {"reproduce", "all"}:
        patterns += ["reproduction/**", f"models/{args.model}/LICENSE"]
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
                ["trajectories-v2/**"],
                "files",
            )
        )
    root.mkdir(parents=True, exist_ok=True)
    receipt = root / f"verified_{args.purpose}.json"
    receipt.write_text(json.dumps(receipts, indent=2) + "\n")
    print(
        f"Verified {sum(item['verified_files'] for item in receipts)} published files; receipt: {receipt}"
    )


if __name__ == "__main__":
    main()
