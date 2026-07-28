# SmolVLA Failure Representations: Evidence, Weaknesses, and Causal Research Programme

> **Research-design update (2026-07-27):** The evidence review in [`blind_spot_research_audit.md`](blind_spot_research_audit.md) supersedes the phase ordering below. It discovered that archived activations were stochastic post-hoc re-queries rather than the exact forwards that generated rollout actions, added full ordered-plan and scene-pixel controls, and identified substantial overlap with recent frozen-VLA success probing. The active programme now begins with exact online alignment and targets Counterfactual Recoverability Decomposition: physical-state recoverability versus sampled-plan quality under branched counterfactuals. The background evidence and intervention taxonomy below remain useful, but `task_plan.md` is authoritative for execution order.

> **Phase 2 correction (2026-07-28):** Exact action-producing capture passes after restoring the required camera orientation. The corrected gate rejects the earlier apparent semantic-routing reversal while preserving the executable/padding/output-null result. Phase 3 further showed that serialized runtime state is not a portable branch root; current-process archive reconstruction is required.

> **Phase 3 update (2026-07-28):** The 160-branch CRD smoke is complete. It finds state/goal-dominated outcome variance, a language veto–composition gap, and no conditional value from the tested low-dimensional hidden summaries. Source-task states are strongly imbalanced in physical subgoal progress, so the next gate is a policy-independent, occupancy-balanced affordance lattice—not immediate activation patching or cross-model scaling. See [`phase3_engineering_review.md`](phase3_engineering_review.md).

## Decision summary

The next goal is not to make a larger failure classifier. It is to determine what the hidden-state warning signal represents and whether that representation participates causally in failure.

The completed experiment establishes three useful facts:

1. SmolVLA's VLM and action-expert hidden states become strongly associated with eventual failure during a trajectory.
2. Deterministic wrist masking and main-camera blur causally reduce success for the tested tasks and seeds.
3. The current hidden-state probes do not identify a causal mechanism or cleanly separate language and visual information.

The new confound audit further shows that the modest step-0 signal is mostly task identity or task difficulty. The more credible episode-specific signal emerges between steps 50 and 100. This temporal window should be the centre of the next mechanistic experiments.

## What we are inspecting now

At every fifth environment step, the completed runs store pooled activation vectors from VLM and action-expert layers 3, 7, 11, and 15. The analysis asks whether information available by time `t` predicts eventual success or timeout. It compares those representations with robot state, policy-output summaries, and sampled action-chunk uncertainty.

This is currently an outcome-prediction experiment over pooled internal representations. It is not yet:

- a token-level language-versus-vision analysis;
- a physical failure-onset detector;
- an explanation of which object, instruction phrase, or camera region matters;
- evidence that the decoded failure direction controls actions;
- a universal failure representation across interventions, tasks, or checkpoints.

## Revised interpretation of the existing evidence

### Benchmark and interventions

- The frozen checkpoint succeeded on 255/400 benchmark episodes (63.75%). Success varied substantially by suite: Goal 83%, Object 67%, Spatial 61%, and LIBERO-10 44%.
- In paired deterministic interventions over 90 matched episodes per condition, wrist-camera masking changed success from 65.6% to 0%; mild main-camera blur changed it to 40.0%; instruction paraphrasing changed it to 63.3%, with an interval containing zero.
- These are causal effects of the tested input transformations on the selected tasks. They do not reveal the neural mechanism and should not be generalized to all forms of language or visual corruption.

### Hidden-state warning signal

On the 400-episode benchmark, action-expert AUPRC is 0.526 at step 0, 0.603 through step 50, and 0.844 through step 100. VLM AUPRC is 0.442, 0.508, and 0.770. Action uncertainty remains near the failure-prevalence baseline through step 100.

The task-confound audit adds a necessary control:

| Feature | Step | Global AUPRC | Within-task pairwise AUROC | Variance attributable to task |
|---|---:|---:|---:|---:|
| Action expert | 0 | 0.526 | 0.543 | 0.895 |
| VLM | 0 | 0.442 | 0.510 | 0.801 |
| Action expert | 100 | 0.844 | 0.878 | 0.525 |
| VLM | 100 | 0.770 | 0.846 | 0.589 |
| Robot state | 100 | 0.520 | 0.610 | 0.966 |

At step 0, the probes mostly know which task they are seeing. They do not reliably know which episode of that task will fail. By step 100, both hidden-state pathways rank failures above successes within the same task, while their dependence on task identity decreases. Robot-state scores remain almost entirely task-structured.

Risk dynamics support the same interpretation. Between step 0 and step 100, mean action-expert risk changes by -0.074 for successes and +0.121 for failures. VLM risk changes by -0.024 and +0.170. Action-uncertainty risk changes by approximately +0.025 for both outcomes. Internal representations therefore track an outcome-specific trajectory change that the current stochastic action disagreement measure misses.

## Causal model and competing explanations

The working causal structure is:

```text
task/instruction ─┬─> visual trajectory ─> VLM state ─> action-expert state ─> action ─> next state ─> outcome
                 ├─> initial difficulty ────────────────────────────────────────────────┘
                 └─> step-0 representation

camera/language intervention ─> encoded input ─> hidden states ─> actions ─> outcome
history/progress ──────────────> hidden states, actions, and probability of timeout
```

Several explanations remain compatible with the data:

1. **Task-difficulty encoding.** Hidden states identify hard tasks, creating apparent warning before any episode-specific evidence exists. The step-0 audit strongly supports this as a partial explanation.
2. **Perceptual failure detection.** The VLM detects a missed grasp, wrong object, occlusion, or geometric misalignment and passes that information to the action expert.
3. **Action-plan degradation.** The action expert represents unstable or unrecoverable plans even when sampled chunk variance is low.
4. **Progress/termination proxy.** Hidden states encode elapsed progress or repeated unsuccessful motion rather than the cause of failure.
5. **Distribution-shift detection.** Perturbation probes identify blur or a black wrist image, not failure itself.
6. **Downstream consequence.** Hidden states change after bad actions or state deviation; they predict failure but do not cause it.
7. **Causal control variable.** A hidden direction actively changes action selection, and modifying it can induce or rescue behaviour.

The research programme must distinguish these explanations rather than select one from probe accuracy alone.

## Questions by modality

### Language

- Is the policy invariant to meaning-preserving paraphrases beyond the small tested set?
- Does it use relational and object tokens, or merely task-template identity?
- What happens when the instruction is swapped with a semantically incompatible instruction while vision and initial state are fixed?
- Do language-token representations causally influence the action expert after the first observation, or only establish an initial task embedding?

Required interventions: multiple human paraphrases, controlled synonym substitutions, relation-word changes, object-name swaps, instruction removal, and wrong-task instructions. Separate semantic preservation from string distance.

### Main-camera vision

- Does blur hurt because fine geometry is lost, because the input is out of distribution, or because object identity is lost?
- Which regions and times are necessary: target object, receptacle, gripper, or background?
- Does replacing only the target-object patch change the same hidden direction as natural failures?

Required interventions: graded blur, frequency-matched noise, object/receptacle/gripper masks, background masks, clean-image swaps between matched states, and short temporal freezes.

### Wrist-camera vision

- Is the catastrophic mask effect evidence of genuine wrist-view necessity or merely sensitivity to an unnatural black image?
- Is wrist information required continuously, only near grasp/contact, or only for certain tasks?
- Can the main camera compensate when the wrist stream is absent but represented with an in-distribution missing-camera token or learned neutral image?

Required interventions: mean-image replacement, matched natural-image replacement, spatial occlusion, temporal dropout at defined phases, controlled noise, and main/wrist crossed interventions.

### Robot state and action

- Do hidden states add information beyond the full observable state and recent action history, rather than the small state summary used so far?
- Is failure caused by a coherent but wrong action plan, explaining why action-chunk disagreement is weak?
- Does action-expert activation encode recoverability, contact state, or accumulated control error?

Required analyses: richer state/history baselines, action reconstruction, contact-aware annotation, deterministic multi-sample alternatives, and controlled action replay or action substitution.

### Cross-modal integration

- Are language and vision additive, redundant, or synergistic?
- Which modality dominates when language and vision conflict?
- Does failure information first appear in modality-specific VLM tokens and then transfer to action-expert tokens?
- Is there a condition-invariant failure direction shared by natural failures, blur, occlusion, and instruction conflict?

Required analyses: token-resolved hooks, modality-labelled attention/activation summaries, cross-condition representational similarity, cross-condition probe transfer, and factorial interventions.

## Experiment programme

### Stage 0 — Existing-data robustness audit (partly complete)

Purpose: eliminate explanations that can be tested without new rollouts.

- Complete within-task, task-centred, and per-suite evaluation at fixed horizons.
- Add leave-one-suite-out and cross-condition transfer: train on clean natural failures, test on blur/mask/paraphrase, and reverse.
- Compare hidden probes against richer baselines using observation/state history and recent actions.
- Test sensitivity to episode length, progress definition, pooling choice, PCA dimension, regularization, and random seeds.
- Use episode-cluster bootstrap confidence intervals and preserve task grouping in every fitted transformation.

Falsification: if hidden-state advantage disappears within task, under held-out suites, or after a richer history baseline, the claimed general failure representation is rejected or narrowed.

### Stage 1 — Failure semantics and temporal alignment

Purpose: replace the coarse timeout label with physically meaningful events.

- Annotate every benchmark failure with failure subtype and earliest visible onset.
- Use at least two annotators on a stratified overlap; report agreement and adjudicate disagreements.
- Suggested taxonomy: wrong object, missed grasp, unstable grasp/drop, collision, misalignment, wrong receptacle, stalled/repetitive plan, recovery failure, and ambiguous.
- Record grasp/contact/place milestones when visible; retain `unknown` rather than infer from normalized progress.

Falsification: if hidden divergence consistently follows human-labelled failure onset, it is a consequence detector rather than an early precursor.

### Stage 2 — Paired modality interventions

Purpose: estimate total causal effects of controlled language and vision changes.

Use identical task, seed, and initial state for every condition. Begin with a small gate (2 tasks × 5 seeds) and expand only when image validity and logging pass.

Core conditions:

| Family | Levels | Question |
|---|---|---|
| Language semantics | original, two paraphrases, wrong object, wrong relation, no instruction | Semantic reliance versus template identity |
| Main vision | clean, graded blur, object mask, receptacle mask, background mask, temporal freeze | Fine geometry, target identity, or distribution shift |
| Wrist vision | clean, mean image, matched natural image, spatial mask, grasp-window dropout | Genuine necessity and temporal role |
| Cross-modal conflict | correct language/wrong vision cue; wrong language/correct vision | Modality dominance and conflict resolution |

Estimate paired success effects, action divergence, hidden-state divergence, and heterogeneous effects by task/failure subtype. Use selected interactions rather than a full combinatorial factorial until pilots show which factors matter.

### Stage 3 — Token-resolved causal tracing

Purpose: locate where modality information enters and affects the policy.

- Record modality token spans and residual streams separately for language, main image, wrist image, and action-expert tokens.
- At layers 3, 7, 11, and 15 and steps concentrated in the 50–100 window, patch clean activations into matched perturbed runs and perturbed activations into clean runs.
- Patch one modality/token group at a time, followed by pathway and layer combinations.
- Measure immediate action recovery, subsequent state recovery, and episode success.

The key tests are rescue and induction:

- **Rescue:** clean activation patching reverses a perturbation's action or success effect.
- **Induction:** perturbed/failure activation patching makes a clean trajectory's action or outcome worse.
- **Specificity:** unrelated task or random-vector patches do not produce the same effect.
- **Dose response:** partial interpolation produces a monotonic behavioural change.

Activation patching is stronger evidence than decoding because it manipulates the proposed mediator while holding the external observation pair fixed as closely as the simulator permits.

### Stage 4 — Mediation and mechanism synthesis

Purpose: quantify how much of an input intervention's effect travels through specific representations.

- Estimate intervention total effects on success and action deviation.
- Estimate interventional indirect effects through selected hidden-state summaries or patched components.
- Avoid claiming standard natural direct/indirect effects when sequential post-treatment confounding is unresolved.
- Test whether a shared representation direction transfers across natural failure and multiple interventions.
- Validate all selected mechanisms on held-out tasks, seeds, suites, and at least one independently trained checkpoint.

A condition-invariant failure mechanism requires all of the following: cross-condition decoding, within-task temporal emergence, activation-patching effect, behavioural rescue/induction, and held-out generalization.

## Measurement and statistical contract

- Unit of evaluation: episode; step rows never cross train/test boundaries.
- Splits: task-grouped for broad generalization, seed/initial-state paired for interventions, and leave-one-suite-out as a hard test.
- Horizons: fixed environment steps and human-labelled time relative to failure onset; full trajectory is retrospective only.
- Primary behavioural endpoint: paired success difference with cluster/bootstrap confidence interval.
- Primary representation endpoint: within-task AUROC/AUPRC and cross-condition transfer at fixed horizons.
- Primary mechanistic endpoint: paired action or success rescue/induction under activation patching.
- Calibration: Brier score and reliability curves on untouched evaluation data.
- Multiplicity: predeclare primary layers, steps, and modalities; treat broad layer/token sweeps as exploratory and correct families of comparisons.
- Negative controls: random patch, unrelated-task patch, background-only mask, label permutation, time-shifted patch, and matched perturbation with similar pixel magnitude.
- Provenance: checkpoint revision, task/seed/initial state, exact intervention parameters, hook target, patch location, and artifact hash.

## Immediate next verification

The next execution should be a two-part gate:

1. **Offline gate:** cross-condition and leave-one-suite-out transfer, richer state/action-history baselines, and bootstrap uncertainty for the within-task confound audit.
2. **Small causal gate:** two tasks and five seeds with in-distribution wrist replacements, targeted wrist dropout windows, and token-resolved hooks. Concentrate intervention/patching measurements at steps 50–100.

Proceed to a larger factorial study only if the hidden-state signal survives the offline controls and the small gate produces modality-specific, reproducible action effects. This ordering targets explanation while minimizing expensive simulation.

## Claim ladder

Use the following language as evidence accumulates:

1. **Descriptive:** failures have different activation statistics.
2. **Predictive:** held-out episodes can be ranked before termination.
3. **Robust predictive:** ranking survives within-task, held-out-suite, cross-condition, and richer-baseline controls.
4. **Mechanistic:** a localized representation mediates action changes under controlled intervention.
5. **Causal:** manipulating that representation reliably induces or rescues behaviour with appropriate controls.
6. **General:** the causal mechanism transfers across tasks, suites, perturbations, seeds, and checkpoints.

Current evidence reaches level 2 overall, with partial level-3 support at step 100 and causal input-level effects for the tested camera perturbations. It does not yet reach a causal hidden-state mechanism.
