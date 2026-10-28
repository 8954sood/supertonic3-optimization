"""Download only runtime assets, with a pinned upstream revision recorded locally."""

from huggingface_hub import snapshot_download
from pathlib import Path
import json

repo = "Supertone/supertonic-3"
rev = "3cadd1ee6394adea1bd021217a0e650ede09a323"
root = Path(
    snapshot_download(
        repo,
        revision=rev,
        local_dir="models/supertonic-3",
        allow_patterns=["onnx/*", "voice_styles/*", "LICENSE", "config.json"],
    )
)
(root / "revision.json").write_text(
    json.dumps({"repo": repo, "revision": rev}, indent=2)
)
print(root, rev)
