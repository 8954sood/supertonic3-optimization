"""Audit staged Git contents without opening excluded local credentials or assets."""

import hashlib
import ipaddress
import json
import re
import subprocess
from pathlib import PurePosixPath


def git(*args):
    return subprocess.check_output(["git", *args])


def main():
    files = git("ls-files", "-z").decode().split("\0")
    files = [name for name in files if name]
    failures = []
    total = 0
    blobs = {}
    excluded_roots = {".private", ".venv", "models", "samples", "references", "dist"}
    excluded_suffixes = {
        ".pem",
        ".key",
        ".p12",
        ".pfx",
        ".onnx",
        ".ort",
        ".pt",
        ".pth",
        ".bin",
        ".safetensors",
        ".npz",
        ".npy",
        ".wav",
        ".mp3",
        ".flac",
        ".so",
        ".dylib",
        ".dll",
        ".pyc",
        ".zip",
        ".gz",
        ".log",
    }
    patterns = {
        "private key": re.compile(r"-----BEGIN " + r"(?:[A-Z0-9]+ )?PRIVATE KEY-----"),
        "GitHub token": re.compile(
            r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})\b"
        ),
        "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
        "AWS access key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
        "personal home": re.compile(r"/" + r"(?:Users|home)/[A-Za-z0-9_.-]+/"),
        "credential URL": re.compile(r"https?://[^\s/@:]+:[^\s/@]+@"),
        "assigned secret": re.compile(
            r"(?i)\b(?:api_key|access_token|client_secret|password)\s*[:=]\s*['\"][A-Za-z0-9_+/=-]{20,}['\"]"
        ),
    }
    for name in files:
        path = PurePosixPath(name)
        if path.parts[0] in excluded_roots or path.suffix.lower() in excluded_suffixes:
            failures.append(f"Excluded asset tracked: {name}")
        if path.name.startswith(".env") and path.name != ".env.example":
            failures.append(f"Environment file tracked: {name}")
        metadata = git("ls-files", "-s", "--", name).decode().split()[0]
        if metadata == "120000":
            failures.append(f"Symlink tracked: {name}")
        raw = git("show", f":{name}")
        blobs[name] = raw
        total += len(raw)
        if len(raw) > 2 * 1024**2:
            failures.append(f"File exceeds 2 MiB budget: {name}")
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            failures.append(f"Binary file tracked: {name}")
            continue
        for label, pattern in patterns.items():
            if pattern.search(text):
                failures.append(f"Potential {label}: {name}")
        for match in re.finditer(
            r"(?<![\w.])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![\w.])", text
        ):
            try:
                address = ipaddress.ip_address(match.group())
            except ValueError:
                continue
            if address.is_global:
                failures.append(f"Public IP address in {name}")
                break
        if path.suffix == ".json":
            try:
                json.loads(text)
            except ValueError:
                failures.append(f"Invalid JSON: {name}")
    if total > 10 * 1024**2:
        failures.append("Total tracked content exceeds 10 MiB budget")
    index = json.loads(blobs.get("records/public_source_index.json", b"[]"))
    for item in index:
        raw = blobs.get(item["path"])
        if raw is None or hashlib.sha256(raw).hexdigest() != item["sha256"]:
            failures.append(f"Public result digest mismatch: {item['path']}")
    if not files:
        failures.append("No tracked files")
    if failures:
        raise SystemExit("\n".join(failures))
    print(
        f"PASS: {len(files)} source/record files, {total:,} bytes; public result hashes verified."
    )


if __name__ == "__main__":
    main()
