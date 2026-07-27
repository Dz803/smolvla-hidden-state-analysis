# Resuming the SmolVLA Hidden-State Project

## Workstation resume

```bash
cd /home/zhongzhengyang/smolvla-hidden-state-analysis
bash scripts/resume_check.sh --full
```

Load the project state:

```bash
PLAN_ID=$(tr -d '[:space:]' < .planning/.active_plan)
sed -n '1,260p' ".planning/$PLAN_ID/task_plan.md"
sed -n '1,320p' ".planning/$PLAN_ID/findings.md"
tail -120 ".planning/$PLAN_ID/progress.md"
sed -n '1,260p' docs/experiment_log.md
```

The first incomplete work item is Phase 1 of the active task plan: the offline robustness gate. Do not rerun the 400-episode benchmark.

## Canonical local paths

```text
project root       /home/zhongzhengyang/smolvla-hidden-state-analysis
complete evidence  archive/full_experiment
canonical run      archive/full_experiment/runs/benchmark_400_20260723T021424Z_e0638bea
working Python     local/lingbot-conda-env-archived/bin/python
LeRobot checkout   vendor/lerobot-smolvla
project logs       logs/
```

The complete evidence, environments, videos, observations, checkpoints, and activations are workstation-only and ignored by Git.

## Shell setup

```bash
export SMOLVLA_ROOT=/home/zhongzhengyang/smolvla-hidden-state-analysis
export SMOLVLA_PYTHON="$SMOLVLA_ROOT/local/lingbot-conda-env-archived/bin/python"
export PYTHONPATH="$SMOLVLA_ROOT/src"
export MUJOCO_GL=egl
```

Invoke the Python binary directly. Some third-party console entry points inside the relocated environment may retain their previous absolute prefix.

## Code-only checkout

A GitHub clone contains source, configs, derived audit tables, reports, plans, and documentation, but not the 51 GB local evidence/runtime directories. On a different machine:

1. provision Python from `pyproject.toml`;
2. obtain the checkpoint and completed run artifacts separately;
3. keep the expected run data contract described in the README;
4. update only machine-local paths—do not modify canonical run IDs or scientific results.

## Safe continuation protocol

- Create new immutable run directories; never overwrite canonical runs.
- Append every material result or error to the active `progress.md`.
- Add new evidence to `docs/experiment_log.md` and durable interpretations to `findings.md`.
- Update the active task-plan phase after each gate.
- Commit code, configs, compact derived tables, plans, and reports.
- Keep raw runs, model weights, activations, videos, observations, environments, and credentials out of Git.

## Current next command

The existing confound audit can be reproduced with:

```bash
"$SMOLVLA_PYTHON" scripts/audit_warning_confounds.py \
  --run "$SMOLVLA_ROOT/archive/full_experiment/runs/benchmark_400_20260723T021424Z_e0638bea" \
  --output /tmp/smolvla_warning_audit_recheck
```

The next implementation should extend this audit with clustered bootstrap intervals and leave-one-suite-out evaluation before any new simulation.
