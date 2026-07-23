from __future__ import annotations

import csv
from pathlib import Path

from .schema import FAILURE_CLASSES


ANNOTATION_COLUMNS = ("episode_id", "failure_class", "failure_onset_step", "notes", "annotator", "confidence")


def append_annotation(path: str | Path, row: dict) -> None:
    if row.get("failure_class") not in FAILURE_CLASSES:
        raise ValueError(f"Unsupported failure class: {row.get('failure_class')}")
    target = Path(path)
    exists = target.exists()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ANNOTATION_COLUMNS, extrasaction="raise")
        if not exists:
            writer.writeheader()
        writer.writerow(row)

