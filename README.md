# SmolVLA Hidden-State Analysis

This repository contains the hidden-state experiment for the frozen `lerobot/smolvla_libero` policy. It includes the activation hooks, pooled VLM/action-expert representation analysis, grouped failure probes, divergence analysis, condition comparison, plotting utilities, tests, and the report from the completed experiment.

The analysis is observational frozen-feature probing. It does not train or modify policy weights, and it does not include raw rollout data, videos, checkpoints, or the broader failure-study project.

The workstation copy is now fully separated from LingBot at `/home/zhongzhengyang/smolvla-hidden-state-analysis`. Its complete 51 GB local evidence/runtime directories are ignored by Git; the public repository remains a small, reviewable code-and-report package.

## Resume work

The repository is resumable from a fresh shell, Codex session, or context reset. On the workstation:

```bash
cd /home/zhongzhengyang/smolvla-hidden-state-analysis
bash scripts/resume_check.sh --full
```

Then read the active plan and recent log:

```bash
PLAN_ID=$(tr -d '[:space:]' < .planning/.active_plan)
sed -n '1,240p' ".planning/$PLAN_ID/task_plan.md"
tail -80 ".planning/$PLAN_ID/progress.md"
```

The canonical continuation instructions are in [`docs/resume.md`](docs/resume.md), and the append-only experiment ledger is in [`docs/experiment_log.md`](docs/experiment_log.md). Project-specific agent instructions in [`AGENTS.md`](AGENTS.md) require every future session to load and update these records.

## Cross-device access

The source, plans, tests, and compact result tables are cloud-backed through GitHub. See [`docs/portable_results.md`](docs/portable_results.md) for clone/pull instructions and the exact boundary between portable reports and workstation-only raw evidence. Checkpoints, full runs, observations, videos, and raw activations are never placed in Git.

## Included result

See [`reports/hidden_state_report.md`](reports/hidden_state_report.md) for the pilot data contract, held-out-task results, divergence findings, warning trade-offs, limitations, and artifact provenance.

The broader interpretation, task-confound audit, causal questions, and intervention matrix are in [`docs/causal_research_program.md`](docs/causal_research_program.md). The later [`blind-spot research audit`](docs/blind_spot_research_audit.md) adds execution-path and trajectory/scene controls, compares the project with recent VLA generalization and value-probing work, and defines Counterfactual Recoverability Decomposition (CRD). Its key revision is that the next target is not a larger success probe: it is separating physical-state recoverability from sampled-plan quality under exact online capture and controlled counterfactual branches.

The corresponding derived local audit is in [`reports/trajectory_confound_audit/`](reports/trajectory_confound_audit/). At exact step 100, the full ordered action chunk and behavioural context are strong predictors, but the archived post-hoc hidden states retain incremental predictive value. Low-resolution initial-scene pixels also predict outcome, making layout/seed difficulty and training-distribution familiarity first-class confounds.

Phase 2 is complete. [`Phase 2 discoveries and ERSD/CPRD`](docs/phase2_discoveries_and_ersd.md) documents the exact action-producing capture, corrected camera-orientation contract, executed/padding/output-null latent decomposition, computational-state requirements, and negative literal-trajectory-copying result. The corrected forward report is [`phase2_forward_gate_20260727T123138Z`](reports/phase2_forward_gate/phase2_forward_gate_20260727T123138Z/); its correction rejects the earlier apparent semantic-routing reversal.

Phase 3 is also complete. Its repaired ledger contains 160 certified branches and 160 exact-forward queries. [`Phase 3 certified CRD`](reports/phase3_crd/phase3_crd_20260728T021125Z/) reports an asymmetric language veto–composition gap, state/goal-dominated outcome variance, and no incremental held-source-episode value from the tested low-dimensional hidden summaries. The [`engineering and scientific review`](docs/phase3_engineering_review.md) records the simulator restore defects, provenance-preserving repairs, implementation hardening, and the central remaining confound: source-policy states are not balanced for physical subgoal progress.

The first incomplete phase is therefore an occupancy-balanced state-lattice gate, not immediate π0.5/GR00T scaling or activation patching. Stage A currently has 19 unique certified roots across preserved v31/v32/v34 evidence. Its first layout-replicate failure and the successful object-relative causal diagnostic are summarized in [`the compact alignment report`](reports/phase3b_stage_a/layout_alignment_20260731T070614Z/). The staged methodology and its execution amendment are in [`docs/phase3b_methodology.md`](docs/phase3b_methodology.md). No external checkpoint has been downloaded.

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

To run the offline robustness audit on an existing benchmark (2,000 bootstrap draws by default):

```bash
python scripts/audit_warning_confounds.py \
  --run /path/to/benchmark_run \
  --output reports/offline_robustness_gate \
  --condition-run /path/to/paired_clean_run \
  --condition-run /path/to/paired_perturbation_run
```

The command reads pooled activations to fit leave-one-suite-out and richer-history probes. Repeat `--condition-run` for each paired perturbation. All fitted preprocessing remains fold-local, and task groups are isolated in cross-condition transfer.

To reproduce the exact-landmark trajectory/scene audit:

```bash
python scripts/audit_trajectory_confound.py \
  --run /path/to/benchmark_run \
  --output reports/trajectory_confound_audit
```

This audit evaluates active risk sets at steps 0, 50, and 100 using task-grouped nested regularization. Its behavioural controls include only past executed actions plus the full ordered current 50x7 chunk; image controls are fitted fold-locally. These reports are derived artifacts—never use their output path to overwrite a canonical run under `archive/full_experiment/runs`.
