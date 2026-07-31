# Task Plan: SmolVLA Causal Validation

## Goal

Determine whether the mid-trajectory SmolVLA hidden-state failure signal generalizes beyond task difficulty and whether language-, vision-, or action-pathway representations causally influence policy failure.

## Current Phase

Phase 3 complete; occupancy-balanced Phase 3b is the next scientific gate

## Completed foundation

- [x] Completed 100-episode Spatial pilot, 360-episode paired perturbation study, and 400-episode benchmark with zero infrastructure failures.
- [x] Established that step-0 hidden-state performance is mostly task structured.
- [x] Established strong within-task outcome separation by step 100 for action-expert and VLM representations.
- [x] Separated all 51 GB of SmolVLA work from LingBot.
- [x] Added causal research programme, intervention matrix, reproducible confound audit, and tracked resume infrastructure.
- [x] Completed the Phase 1 offline robustness gate with bootstrap, held-out-suite, cross-condition, and richer-history controls.

## Phases

### Phase 1: Offline robustness gate

- [x] Add episode/task-cluster bootstrap intervals to the confound audit.
- [x] Run leave-one-suite-out evaluation at steps 0, 50, and 100.
- [x] Run clean-to-perturbation and perturbation-to-clean probe transfer.
- [x] Add richer state and recent-action-history baselines.
- [x] Decide whether the hidden-state advantage survives all predeclared controls.
- **Decision:** The step-100 advantage survives all estimable controls, especially for the action expert. VLM paraphrase transfer is weaker, and wrist-mask-to-clean transfer is not estimable because the source condition has no successes. This supports continued predictive/mechanistic study, not a condition-invariant or causal-mechanism claim.
- **Status:** complete

### Phase 1b: Research-design stress test

- [x] Audit online action generation versus post-hoc activation capture.
- [x] Audit fixed-horizon risk sets, termination leakage, and replanning-boundary effects.
- [x] Test whether full ordered action chunks and low-resolution scene observations explain the hidden-state advantage.
- [x] Review external evidence on VLA shortcuts, compositional generalization, value probing, and causal representation tests.
- [x] Replace and reorder later phases according to the strongest falsification risks.
- **Decision:** Archived activations are stochastic post-hoc re-queries rather than the states that produced the rollout actions. Exact step-100 risk-set prediction survives strong ordered-plan and behavioural baselines, but current pixels also predict initial difficulty and the policy has already trained on the evaluated task identities. Recent work also overlaps generic frozen-VLA success probing. The project therefore pivots from larger outcome probes to Counterfactual Recoverability Decomposition (CRD): separate physical-state recoverability from sampled-plan quality, condition on pixels/state/history/full chunks/noise, and require directionally correct causal interventions and actual OOD task composition.
- **Status:** complete

### Phase 2: Measurement-alignment and forward-only gate

- [x] Capture activations inside the action-producing forward, with query IDs, flow-noise seeds, queue age, and the exact generated 50x7 chunk.
- [x] Preserve language/main-camera/wrist-camera token spans, action offsets, and individual denoising-step activations instead of pooling them together.
- [x] Record and validate privileged object geometry, task predicates, contacts, attachment state, and a restorable simulator state; repair the current `save_environment_state` contract.
- [x] Re-query saved observations under fixed noise with paraphrase, feasible alternate-goal, contradictory-language, and nuisance-image controls.
- [x] Verify deterministic hook fidelity and quantify within-state variation across flow-noise seeds before interpreting any site.
- [x] Optionally retrieve only official training metadata/action-state shards for nearest-neighbour and DTW controls; do not download videos or launch rollouts for this gate.
- **Decision:** Exact-forward and repeat fidelity pass at zero error after correcting the archived-camera orientation contract. The earlier step-dependent contradiction reversal was an image-orientation artifact and is rejected. Controlled factors still induce large action changes through a small executed projection while `97.85–98.64%` of final hidden displacement is output-null. Official state/action retrieval finds no literal action-window copy and instead motivates a phase-locked policy-attractor hypothesis. Phase 3 subsequently showed that cross-process snapshot restore is unsafe, so current-process archive reconstruction—not serialization alone—is required.
- **Status:** complete

### Phase 3: Two-task branched CRD smoke gate

- [x] Repair the Phase 2 saved-observation image-orientation contract and rerun only the corrected fixed-forward gate before selecting sites.
- [x] Build a common physical-state and multi-goal predicate evaluator for LIBERO-Goal tasks 3 and 4 rather than assuming task-specific states are interchangeable.
- [x] Restore exactly five task-appropriate states per task; sample at most `K=4` first-plan seeds and `M=2` matched continuation schedules. Task 4 has no valid step-100 archive state because all source episodes terminate earlier, so its five states are at step 50 rather than fabricated post-termination landmarks.
- [x] Estimate state recoverability `V`, plan-conditioned recoverability `Q`, and their variance components with common random numbers and hierarchical shrinkage.
- [x] Test conditional hidden-state value beyond task, privileged state/geometry, pixels, trajectory history, queue position, full ordered chunk, and noise metadata.
- [x] Measure cross-goal plan faithfulness and contrast feasible goal changes with paraphrases, contrastive negation, and visual nuisances.
- [x] Obtain explicit approval before launching this materially new simulation workload (ceiling approximately 160 branched continuations).
- [x] Conduct a post-run senior-engineering review covering determinism, leakage, resume safety, accounting, failure recovery, statistical validity, and artifact immutability; add tests and fixes for every confirmed issue.
- **Decision:** The repaired ledger has 160 certified branches, 80 paired proposal queries, 80 factor queries, and zero paired-prefix mismatches. State-goal cells explain `80.8%` of outcome variance, first proposals `15.4%`, and continuation schedules `3.8%`. Cabinet-source recoverability is `1.0` for cabinet and `0.0` for drawer; drawer-source recoverability is `0.45` for drawer and `0.375` for cabinet. This is a language veto–composition gap under strongly different physical preparation, not proof of trajectory memory. The policy is reset and has no observation-history queue. Low-dimensional hidden summaries add no held-source-episode value for `V/Q/L` beyond complete controls. Phase 3 validates CRD infrastructure but does not green-light unchanged causal patching or multi-model scaling.
- **Status:** complete

### Phase 3b: Occupancy-balanced recoverability gate

- [x] Predeclare the staged occupancy-balanced design, branch ceilings, information gate, cross-model handoff, and stop rules in `docs/phase3b_methodology.md`.
- [ ] Build a policy-independent affordance lattice spanning drawer opening, bowl grasp/release and pose, end-effector pose, and both goals; certify each state with a policy-independent feasible continuation.
- [ ] Balance physical subgoal distance and on-/off-policy occupancy so source-task progress cannot masquerade as language grounding or trajectory memory.
- [ ] Re-estimate the veto–composition gap, `V`, `Q`, and `L` on held-out state families with source-episode/state-family grouping.
- [ ] Test directional hidden features rather than only norms/dispersion, with fold-local dimension reduction and a predeclared information gate.
- [ ] Decide whether any representation justifies Phase 4 patching and whether the shared environment interface is ready for π0.5/GR00T.
- **Status:** Stage A has 20 promoted roots: 16 closed roots from v32, the two-state v31 open hard pair, one v34 open root, and the repaired layout-B root in v35. Its v34 drawer oracle reassembles exactly with zero proposal re-execution, while the registered cabinet bank changes `0/46` world-anchor successes to `5/46` on the identical normalized state. The two prospective transverse episode-474 attempts remain importable. Expansion is authorised for the remaining 12 v35 candidates; cross-aperture/source-revision and cross-oracle-mode effects remain non-estimable. Stages B/C are separate simulation-scope decisions.

### Phase 4: Matched causal intervention gate

- [ ] At an identical state, patch a predeclared action-expert site between high-`Q` and low-`Q` noise-sampled plans in both directions.
- [ ] At identical physical state and noise, patch a predeclared VLM token/subspace between controlled feasible goals and test task-appropriate action direction.
- [ ] Measure immediate chunk effects plus branched behavioural rescue and reverse-direction induction.
- [ ] Run norm-matched random, unrelated-task, same-task time-shift, same-chunk/different-goal, and same-goal/different-chunk controls.
- [ ] Require hooked reconstruction fidelity, localization, and interpolation dose response; generic controller damage does not count as a mechanism.
- **Status:** pending behind the Phase 3b information gate

### Phase 5: Actual generalization and synthesis

- [ ] Evaluate position and task/composition shifts, prioritizing LIBERO-PRO-style axes rather than only held-out episodes from trained task identities.
- [ ] Evaluate controlled lexical families, feasible goal swaps, contradictions, and multilingual variants separately.
- [ ] Measure training-trajectory/action-chunk similarity and report results stratified by retrieval distance.
- [ ] Validate any surviving decomposition on one stronger flow VLA and one architecture with a different action decoder.
- [ ] Report the narrowest supported claim: grounded recoverability, sampled-plan evaluation, progress monitoring, shortcut retrieval, or a SmolVLA-specific effect.
- **Status:** pending

## Success criteria

The project may claim a causal, grounded hidden-state mechanism only if a localized representation:

1. is captured in the exact forward that generates a known action chunk, with verified hook fidelity;
2. predicts branched state recoverability or within-state plan quality beyond physical geometry, pixels, history, queue position, the full ordered chunk, and flow noise;
3. is invariant to meaning-preserving language/visual nuisance changes but changes directionally under feasible goal counterfactuals;
4. changes actions in the predicted semantic or plan-quality direction under controlled patching;
5. produces matched behavioural rescue and reverse-direction induction with stringent negative controls; and
6. generalizes to unseen positions/compositions and at least one independent architecture.

## Resume checkpoint

Phase 3 is complete. The first incomplete phase is the occupancy-balanced Phase 3b gate; do not launch it, Phase 4 patching, or cross-policy downloads as an automatic continuation. Do not rerun the completed benchmark, paired perturbation rollouts, corrected Phase 2 fixed-forward queries, or the completed Phase 3 branch matrix. The canonical benchmark is:

`archive/full_experiment/runs/benchmark_400_20260723T021424Z_e0638bea`

Use `reports/offline_robustness_gate/`, `reports/trajectory_confound_audit/`, `reports/phase2_forward_gate/phase2_forward_gate_20260727T123138Z/`, `reports/phase2_state_gate/`, `reports/trajectory_memorization_audit/`, and `reports/phase3_crd/phase3_crd_20260728T021125Z/` as completed derived evidence. The Phase 3 review and occupancy-balanced successor design are in `docs/phase3_engineering_review.md`. Raw Phase 3 evidence remains workstation-only under `local/phase3_crd/phase3_crd_20260728T021125Z/`.

## Errors to remember

- `/home/zhongzhengyang/miniconda3/envs/smolvla-libero` lacks pandas and is not the completed runtime.
- The relocated workstation environment is functional for offline analysis but may retain old absolute prefixes in third-party entry-point scripts; invoke its Python binary directly.
- A local clone may inherit a filesystem path as `origin`; confirm `git remote -v` before pushing.
