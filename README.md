# SmolVLA Hidden-State Analysis

This repository contains the hidden-state experiment for the frozen `lerobot/smolvla_libero` policy. It includes the activation hooks, pooled VLM/action-expert representation analysis, grouped failure probes, divergence analysis, condition comparison, plotting utilities, tests, and the report from the completed experiment.

The analysis is observational frozen-feature probing. It does not train or modify policy weights, and it does not include raw rollout data, videos, checkpoints, or the broader failure-study project.

The workstation copy is now fully separated from LingBot at `/home/zhongzhengyang/smolvla-hidden-state-analysis`. Its complete 51 GB local evidence/runtime directories are ignored by Git; the public repository remains a small, reviewable code-and-report package.

## Included result

See [`reports/hidden_state_report.md`](reports/hidden_state_report.md) for the pilot data contract, held-out-task results, divergence findings, warning trade-offs, limitations, and artifact provenance.

The broader interpretation, newly completed task-confound audit, causal questions, intervention matrix, and staged validation programme are in [`docs/causal_research_program.md`](docs/causal_research_program.md). The key revision is that step-0 hidden-state prediction is largely task/difficulty encoding, while episode-specific separation develops between steps 50 and 100.

## Environment

The original experiment used Python 3.10+, MuJoCo with EGL, and the dependencies in `pyproject.toml`. Install the package from this repository with:

```bash
python -m pip install -e .
```

The checkpoint revision and system details are recorded in [`reports/system_audit.md`](reports/system_audit.md). The model module inspection used by the hooks is in [`reports/model_modules.json`](reports/model_modules.json).

## Reproduction

The scripts expect a completed SmolVLA rollout directory containing `manifest.json`, `episodes.parquet`, `steps.parquet`, `activations.zarr`, and the offline uncertainty summary. Raw rollout directories are intentionally not stored in this repository.

```bash
python scripts/analyze_hidden_states.py \
  --run /path/to/completed_run

python scripts/plot_hidden_states.py \
  --run /path/to/completed_run
```

For a multi-condition comparison:

```bash
python scripts/analyze_condition_hidden_states.py \
  --runs /path/to/clean /path/to/wrist_mask /path/to/blur /path/to/paraphrase \
  --output /path/to/condition_report
```

## Tests

```bash
pytest -q
```

To audit task confounding in an existing benchmark run:

```bash
python scripts/audit_warning_confounds.py \
  --run /path/to/benchmark_run \
  --output reports/warning_confound_audit
```
