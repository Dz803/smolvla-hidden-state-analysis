#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ID = "lerobot/libero"
REVISION = "a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download only official LIBERO state/action Parquet and metadata; exclude all video."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("local/training_data/lerobot_libero_a1aaacb"),
    )
    parser.add_argument("--max-workers", type=int, default=1)
    args = parser.parse_args()
    output = args.output.resolve()
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=REVISION,
        local_dir=output,
        allow_patterns=["README.md", "data/**/*.parquet", "meta/**"],
        ignore_patterns=["videos/**", "**/*.mp4"],
        max_workers=args.max_workers,
    )
    videos = list(output.rglob("*.mp4"))
    if videos:
        raise AssertionError(f"Video exclusion failed: {videos[:3]}")
    parquet = sorted(output.rglob("*.parquet"))
    manifest = {
        "repo_id": REPO_ID,
        "revision": REVISION,
        "output": str(output),
        "parquet_files": len(parquet),
        "parquet_bytes": sum(path.stat().st_size for path in parquet),
        "video_files": 0,
    }
    (output / "phase2_download_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
