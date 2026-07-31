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

Phase 3 is complete. The first incomplete work item is Stage A of the occupancy-balanced Phase 3b gate in the active task plan. Nineteen unique roots are promoted across immutable v31/v32/v34 evidence. v34's second root has a complete `18/36` drawer ledger and exhaustive `0/46` cabinet ledger; do not rerun either. The bounded alignment report and prospective transverse-root report under `reports/phase3b_stage_a/` show that the frozen bowl-relative rule passes both untouched layout replicates. Their episode-474 attempts must be imported into the next 13-candidate shard, not rerun. Stage B/C still require separate scope decisions.

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

The GitHub clone/pull procedure and portable-artifact boundary are recorded in `docs/portable_results.md`. A handoff is cloud-backed only after the remote `main` branch contains the reported commit hash.

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

## Completed offline-gate reproduction

The benchmark portion of the completed offline gate can be reproduced with:

```bash
"$SMOLVLA_PYTHON" scripts/audit_warning_confounds.py \
  --run "$SMOLVLA_ROOT/archive/full_experiment/runs/benchmark_400_20260723T021424Z_e0638bea" \
  --output /tmp/smolvla_offline_gate_recheck \
  --bootstrap-samples 2000
```

Cross-condition reproduction additionally requires the four canonical paired run directories through repeated `--condition-run` arguments.

The trajectory/scene confound audit can be reproduced without changing canonical evidence:

```bash
"$SMOLVLA_PYTHON" scripts/audit_trajectory_confound.py \
  --run "$SMOLVLA_ROOT/archive/full_experiment/runs/benchmark_400_20260723T021424Z_e0638bea" \
  --output /tmp/smolvla_trajectory_confound_recheck
```

The Phase 2 state contract can be checked without loading the policy checkpoint:

```bash
MUJOCO_GL=egl PYTHONPATH="$SMOLVLA_ROOT/vendor/lerobot-smolvla/src:$SMOLVLA_ROOT/src" \
  "$SMOLVLA_PYTHON" scripts/validate_libero_state_contract.py
```

The completed Phase 3 derived analysis can be regenerated without running a policy or simulator:

```bash
PYTHONPATH="$SMOLVLA_ROOT/vendor/lerobot-smolvla/src:$SMOLVLA_ROOT/src" \
  "$SMOLVLA_PYTHON" scripts/analyze_phase3_crd.py \
  --run-dir "$SMOLVLA_ROOT/local/phase3_crd/phase3_crd_20260728T021125Z"
```

The next implementation target is an additive Stage A completion contract: freeze the object-relative certificate, import and verify completed ledgers without replay, test it first on untouched missing roots, and finish the exact 32-root bank. Treat source revision and oracle mode as explicit provenance; do not estimate drawer-aperture effects from the mixed-source bank. Stage A does not run a policy. Do not download π0.5/GR00T weights, launch the 128-branch Stage B pilot, or start Phase 4 patching until the Stage A state certificates and balance report have been reviewed.
