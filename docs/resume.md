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

Phase 3b Stage A is complete. The first incomplete work item is the bounded Stage B SmolVLA policy pilot, which remains a separate simulation-scope decision. Do not launch Stage B/C, pi0.5/GR00T downloads, or Phase 4 patching automatically.

The final v37 root's drawer `10/36` and cabinet `0/46` ledgers remain immutable evidence. Factorized certificate `factorized_certificate_20260803T042126Z` separately establishes a physical cabinet path with stable acquisition at source frame 45 and `204` early-stop feedback actions. Additive promotion `local/phase3b_stage_a/phase3b_stage_a_promotion_v38_20260803T045149Z` binds those evidence classes without rerunning a completed suffix or factorized branch. Consolidation `reports/phase3b_stage_a/phase3b-stage-a-consolidated-v4/` independently passes for 32 states, 16 pairs, 64 physical state-goal cells, and 2,624 proposal attempts. `reports/phase3b_stage_a/competence_compatibility_gap_v1/` records the narrow result: `F_C=1, P_K=0` for one cell. It is not SmolVLA competence, a hidden-state mechanism, or a population rate.

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

The compact Stage A reports can be verified without running a policy or simulator:

```bash
PYTHONPATH="$SMOLVLA_ROOT/src" "$SMOLVLA_PYTHON" \
  scripts/verify_phase3b_stage_a_consolidation.py \
  --report-dir reports/phase3b_stage_a/phase3b-stage-a-consolidated-v4

PYTHONPATH="$SMOLVLA_ROOT/src" "$SMOLVLA_PYTHON" \
  scripts/verify_phase3b_competence_gap.py
```

The v36 and v37 completion runs are closed and must not be resumed. The v38 promotion and both compact reports are complete and independently verified; rerunning them is unnecessary. Future work begins with a new Stage B contract only after an explicit scope decision. Preserve source revision, root timestep, proposal execution mode, certificate class, and ecological versus controller-normalised runtime as separate provenance. The cross-model identification design and stop rules are in `docs/competence_compatibility_methodology.md`.
