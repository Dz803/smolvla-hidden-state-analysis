from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .schema import EPISODE_COLUMNS, STEP_COLUMNS, validate_tables


@dataclass
class RolloutRecorder:
    run_dir: Path
    episode_rows: list[dict] = field(default_factory=list)
    step_rows: list[dict] = field(default_factory=list)

    def add_episode(self, row: dict) -> None:
        self.episode_rows.append(row)

    def add_step(self, row: dict) -> None:
        self.step_rows.append(row)

    def finalize(self) -> None:
        episodes = pd.DataFrame(self.episode_rows, columns=EPISODE_COLUMNS)
        steps = pd.DataFrame(self.step_rows, columns=STEP_COLUMNS)
        errors = validate_tables(episodes, steps)
        if errors:
            raise ValueError("; ".join(errors))
        for name, frame in (("episodes.parquet", episodes), ("steps.parquet", steps)):
            path = self.run_dir / name
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite raw result: {path}")
            frame.to_parquet(path, index=False)

