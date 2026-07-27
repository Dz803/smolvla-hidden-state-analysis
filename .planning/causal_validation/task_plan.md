# Task Plan: SmolVLA Causal Validation

## Goal

Determine whether the mid-trajectory SmolVLA hidden-state failure signal generalizes beyond task difficulty and whether language-, vision-, or action-pathway representations causally influence policy failure.

## Current Phase

Ready to start Phase 1

## Completed foundation

- [x] Completed 100-episode Spatial pilot, 360-episode paired perturbation study, and 400-episode benchmark with zero infrastructure failures.
- [x] Established that step-0 hidden-state performance is mostly task structured.
- [x] Established strong within-task outcome separation by step 100 for action-expert and VLM representations.
- [x] Separated all 51 GB of SmolVLA work from LingBot.
- [x] Added causal research programme, intervention matrix, reproducible confound audit, and tracked resume infrastructure.

## Phases

### Phase 1: Offline robustness gate

- [ ] Add episode/task-cluster bootstrap intervals to the confound audit.
- [ ] Run leave-one-suite-out evaluation at steps 0, 50, and 100.
- [ ] Run clean-to-perturbation and perturbation-to-clean probe transfer.
- [ ] Add richer state and recent-action-history baselines.
- [ ] Decide whether the hidden-state advantage survives all predeclared controls.
- **Status:** pending

### Phase 2: Failure semantics and token instrumentation

- [ ] Annotate a stratified set of failures for subtype and earliest visible onset.
- [ ] Measure inter-annotator agreement before completing the queue.
- [ ] Expose and validate token spans for language, main camera, wrist camera, and action-expert tokens.
- [ ] Verify hook equivalence after token-resolved instrumentation.
- **Status:** pending

### Phase 3: Small paired causal smoke gate

- [ ] Run exactly 2 tasks × 5 seeds before any larger rollout.
- [ ] Compare clean wrist input with mean-image, matched-natural-image, and phase-targeted dropout conditions.
- [ ] Add main-object/background controls and one wrong-object instruction condition.
- [ ] Confirm paired initial states, observation validity, and reproducible action effects.
- **Status:** pending

### Phase 4: Activation patching and causal tracing

- [ ] Patch clean representations into perturbed runs during steps 50–100 and measure action/success rescue.
- [ ] Patch perturbed representations into clean runs and measure induction.
- [ ] Run random-vector, unrelated-task, time-shifted, and background controls.
- [ ] Test dose response and modality/layer specificity.
- **Status:** pending

### Phase 5: Generalization and synthesis

- [ ] Validate selected mechanism on held-out tasks, suites, seeds, perturbations, and an independent checkpoint.
- [ ] Quantify total and interventional indirect effects without overclaiming natural mediation.
- [ ] Update the claim level and final report according to falsification outcomes.
- **Status:** pending

## Success criteria

The project may claim a causal hidden-state mechanism only if a localized representation:

1. predicts outcome within task at a fixed pre-onset horizon;
2. transfers across natural failures and controlled perturbations;
3. adds information beyond rich state/action history;
4. changes actions under controlled patching;
5. produces behavioural rescue and induction with negative controls; and
6. generalizes to held-out tasks and at least one independent checkpoint.

## Resume checkpoint

Start with Phase 1. Do not rerun the completed 400-episode benchmark. The canonical benchmark is:

`archive/full_experiment/runs/benchmark_400_20260723T021424Z_e0638bea`

Use `scripts/audit_warning_confounds.py` as the starting implementation and write new derived outputs under `reports/`.

## Errors to remember

- `/home/zhongzhengyang/miniconda3/envs/smolvla-libero` lacks pandas and is not the completed runtime.
- The relocated workstation environment is functional for offline analysis but may retain old absolute prefixes in third-party entry-point scripts; invoke its Python binary directly.
- A local clone may inherit a filesystem path as `origin`; confirm `git remote -v` before pushing.
