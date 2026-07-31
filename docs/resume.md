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

Phase 3 is complete. The first incomplete work item is Stage A of the occupancy-balanced Phase 3b gate in the active task plan. Twenty-six unique physical roots are promoted across v31/v32/v34 and the additive v35 shard `local/phase3b_stage_a/phase3b_stage_a_completion_20260731T075647Z`. Do not rerun any completed candidate or checkpoint. v35 contains seven complete roots, including exact imports of the v34 drawer ledger and both held-root episode-474 attempts. A hash-bound compatibility recovery corrected a post-bank registered-mode validator omission without changing construction or proposal execution.

v35 stopped before proposal execution on its first grasped layout-A root because the construction route exceeded the frozen possession-continuity gate (two contact-negative ticks and `20.94 mm` relative-pose shift). The full trace shows a `20.4 mm` within-gripper re-seat followed by stable transport. The repair does not relax that gate: v36 opens the drawer separately, then acquires the bowl with the registered episode-474 pre-grasp continuation. Construction smoke `local/phase3b_stage_a/construction_gates/phase3b_stage_a_construction_gate_20260731T094517Z` passes all six untouched grasped roots and three pair-geometry audits at normalized timestep 560, with fresh certificates and zero proposal or policy execution. The next action is the fresh six-root v36 completion shard (492 logical/fresh attempts), then consolidated 32-root validation. Do not resume the six grasped candidates under v35. Stage B/C still require separate scope decisions.

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

The next execution target is `configs/phase3b_stage_a_v36.yaml`: run its six-root additive completion shard only after confirming the source/config hashes match the passed construction gate. Treat source revision, root timestep, and oracle mode as explicit provenance; do not estimate drawer-aperture effects from the mixed-source bank. Stage A does not run a policy. Do not download π0.5/GR00T weights, launch the 128-branch Stage B pilot, or start Phase 4 patching until the 32-state certificates and consolidated balance report have been reviewed.
