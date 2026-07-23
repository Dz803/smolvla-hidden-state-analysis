#!/usr/bin/env python
import argparse
from pathlib import Path

from smolvla_analysis.hidden_state_plots import generate_hidden_state_plots


parser = argparse.ArgumentParser()
parser.add_argument("--run", type=Path, required=True)
args = parser.parse_args()
manifest = generate_hidden_state_plots(args.run)
print(f"plot manifest entries: {len(manifest)}")
