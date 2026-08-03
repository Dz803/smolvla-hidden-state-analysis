# SmolVLA Hidden-State Project: Blind-Spot and Research-Gap Audit

Date: 2026-07-27

## Decision

Do not continue the old Phase 2 unchanged.

The project has a real result: SmolVLA representations contain outcome-relevant information by step 100 that is not fully explained by robot state, executed-action history, the complete ordered action plan, or crude current pixels. But that result is not yet evidence of scene understanding, and generic VLA success probing is no longer a sufficient novelty claim.

The strongest research direction is **Counterfactual Recoverability Decomposition (CRD)**: estimate multiple possible futures from the identical physical state and causally separate four quantities that current work conflates:

1. semantic task state;
2. generic visual and kinematic progress;
3. sampled action-plan quality; and
4. policy-conditioned probability of eventual recovery.

SmolVLA is a good minimum viable model for this method because its stochastic flow action expert and low inference cost make repeated controlled branches feasible. The method, rather than the backbone, should be the contribution.

## What changed after the audit

| Assumption | Audit result | Consequence |
|---|---|---|
| Archived activations generated the rollout actions | Activations were recomputed after each episode with fresh flow noise | They are diagnostic features, not the online causal states of the recorded behavior |
| The policy replans every recorded step | `n_action_steps=50`; only steps 0, 50, 100, ... invoke a new model forward | Analysis must distinguish replanning decisions from blind execution inside a stale queue |
| The policy-output baseline controlled for action plans | It discarded temporal order using mean/std/min/max | The original control was underpowered |
| Task-held-out probes established generalization | The checkpoint was trained on all 40 evaluated LIBERO task identities | Only the probe, not the policy, faced held-out tasks |
| Phase names represented grasp/transport/place events | Every `object_state` is null and phases are normalized-time bins | Current phase semantics are labels on a clock |
| One paraphrase tested language understanding | Recent benchmarks find large lexical failures, visual dominance under contradictions, and step-specific language effects | Language must be tested with feasible alternate goals, contradictions, paraphrases, and decision-time localization |
| Hidden-state failure prediction was the main novelty | Direct 2026 work already probes SmolVLA value-like structure, uses same-step matching, goal swaps, and sampled-action selection | The project must answer a deeper causal question |

## New local evidence

All numbers below use derived analyses only; no canonical rollout was rerun or modified.

### Exact landmark and trajectory controls

At step 100, 81 successful episodes had already terminated while every failure remained active. Restricting evaluation to the 319 episodes that were still running avoids treating known-completed successes as prospective warnings.

Using five outer task-grouped folds and nested task-grouped regularization:

| Feature at exact step 100 | AUPRC | AUROC | Within-task AUROC | Brier | Log loss |
|---|---:|---:|---:|---:|---:|
| Executed actions through step 99 | 0.610 | 0.635 | 0.689 | 0.237 | 0.672 |
| Full ordered step-100 action chunk | 0.744 | 0.765 | 0.793 | 0.198 | 0.581 |
| Robot + action history + ordered chunk | 0.803 | 0.821 | 0.803 | 0.177 | 0.532 |
| Action expert alone | 0.878 | 0.868 | 0.797 | 0.147 | 0.456 |
| VLM alone | 0.842 | 0.865 | 0.837 | 0.153 | 0.470 |
| Behavioral context + action expert | 0.882 | 0.891 | 0.843 | 0.137 | 0.435 |
| Behavioral context + VLM | 0.883 | 0.891 | 0.863 | 0.141 | 0.434 |

The action plan and prior behavior explain a large fraction of the signal. Hidden states nevertheless improve proper scores and discrimination when added under nested grouped validation. The correct conclusion is “hidden states add current-observation information beyond recorded behavior,” not “the action expert understands the scene.”

### Cheap visual controls

Two-camera 16×16 RGB block averages already expose initial-condition difficulty:

- step 0: AUPRC 0.532, AUROC 0.632, within-task AUROC 0.607;
- step 100: AUPRC 0.721, AUROC 0.774, within-task AUROC 0.731;
- step-100 absolute change from the initial pixels: AUPRC 0.512, AUROC 0.612.

Initial layout and appearance therefore predict some failures before action. Current pixels also encode progress, but the crude visual baseline remains below the hidden representations. A privileged geometry baseline or a frozen visual representation is needed before assigning the remainder to semantics.

## The causal object we should estimate

Let:

- `S_t` be the complete physical simulator state before a policy query;
- `G` be the instruction and corresponding goal predicate;
- `O_t` be rendered observations;
- `E_k` be a controlled flow-noise seed;
- `H_tk` be the exact online hidden state;
- `C_tk` be the resulting action chunk; and
- `Y` be eventual task success.

The current artifacts observe one realized future from each state and a different post-hoc hidden state:

```text
S_t -> O_t -> H_t(E_online) -> C_t -> executed future -> Y

O_t, G, E_posthoc -> H'_t                    (saved after the episode)
```

`H'_t` can predict `Y`, but it was not on the path that caused `Y`.

The desired state-level estimand is policy-conditioned recoverability:

\[
V^\pi(s,g)=\Pr(Y_g=1\mid do(S_t=s),G=g,\pi).
\]

For a particular sampled chunk prefix `c`, define plan-conditioned recoverability:

\[
Q^\pi(s,g,c)=\Pr(Y_g=1\mid do(S_t=s),do(A_{t:t+h}=c_{0:h}),\pi\text{ thereafter}).
\]

`V` says whether the state is recoverable by the policy. `Q` says whether this sampled plan improves or harms that recoverability. A representation of `V` is a state evaluator; a representation of `Q` may be primarily an action-plan evaluator. Their scientific meanings are different.

## Counterfactual Recoverability Decomposition

### 1. Repair the measurement path

Before another scientific rollout:

- capture activations inside the action-producing `select_action` forward;
- assign every forward a query ID and flow-noise seed;
- store the exact generated 50×7 chunk with that activation;
- preserve VLM token spans, action-token offsets, and all denoising-step activations rather than averaging them together;
- record queue age and remaining horizon at every environment step;
- store a restorable simulator state, object poses, contacts, gripper/object attachment, and all task predicates;
- validate that hooking without intervention leaves actions bitwise equal or within a predeclared tolerance.

This instrumentation is a gate. Mechanistic claims should stop if exact alignment and hook fidelity do not pass.

### 2. Branch futures hierarchically

For each matched physical state `s`, sample `K` first-plan noise seeds. For each plan `k`, execute a short prefix and branch `M` continuation-noise schedules:

\[
Y_{skm},\qquad
\widehat V_s=\frac{1}{KM}\sum_{k,m}Y_{skm},\qquad
\widehat Q_{sk}=\frac{1}{M}\sum_mY_{skm}.
\]

Use common random numbers across compared instructions, visual conditions, and patches. With small `K` and `M`, estimate probabilities with beta-binomial or hierarchical-logistic shrinkage rather than treating each empirical fraction as exact.

This nested design separates:

- between-state variance: some physical states are broadly unrecoverable;
- within-state, between-plan variance: some noise-sampled chunks are better than others; and
- continuation variance: a plan prefix may still admit good and bad later choices.

### 3. Measure conditional usable information

Let the baseline `B_t` contain task/suite, exact physical state, current low-cost visual features, robot history, queue position, full ordered action chunk, and noise/query metadata. Compare nested grouped proper scores:

\[
\Delta_{\mathcal V}(H;R\mid B)
=\mathcal L_{\mathcal V}(R\mid B)-\mathcal L_{\mathcal V}(R\mid B,H),
\]

where `R` is `V` for state-level tests or `Q` for plan-level tests. This is a conditional-probing question, not a contest between separate AUPRC rows.

Predeclared interpretations:

- VLM predicts `V` but not within-state `Q`: primarily goal-conditioned scene state;
- action expert predicts within-state `Q` after the full chunk: latent plan-quality information beyond the decoded plan;
- hidden increment vanishes after geometry and chunk controls: a useful monitor, but not a distinct semantic/value representation;
- score varies mainly with flow noise and reconstructs the chunk: plan/denoising geometry, not stable scene understanding.

### 4. Use an orthogonal counterfactual grid

Do not confound semantic changes with nuisance corruption.

| Axis | Meaning-preserving control | Meaning-changing intervention | Required behavior |
|---|---|---|---|
| Language | human paraphrase, synonym, word order | feasible alternate goal, wrong object/relation, impossible instruction, scrambled tokens | invariant for paraphrases; appropriately different or safely abstaining for contradictions |
| Vision | texture, lighting, background, mild camera shift | target/destination identity or position | invariant to nuisance; equivariant to task-relevant changes |
| Action | same chunk replay, short-horizon replan | high-`Q` versus low-`Q` chunk swap | outcome follows plan quality when state is fixed |
| Flow noise | repeated fixed seeds | deliberately diverse candidate seeds | state value is stable; plan value may vary |

An impossible instruction needs an abstention/clarification or semantic-safety endpoint. It must not be scored against the original task predicate as if doing the familiar task were success. For feasible alternate instructions, switch the goal predicate together with the language.

For two feasible goals `g_1,g_2` in the same scene, define cross-goal plan faithfulness:

\[
F=\frac{1}{2}\left[
Q_{g_1}(C_{g_1})-Q_{g_1}(C_{g_2})
+Q_{g_2}(C_{g_2})-Q_{g_2}(C_{g_1})
\right].
\]

Positive `F` means each instruction generates a plan that is better for its own goal than the competing goal's plan. This is stronger than observing a representation shift or raw action difference.

### 5. Intervene without mistaking damage for mechanism

Run two distinct causal tests:

1. **Plan mechanism:** at the same state, identify high- and low-`Q` noise samples. Patch matched action-expert states from high to low candidates and reverse them. Measure immediate chunk change and branched success rescue/induction.
2. **Semantic mechanism:** at the same physical state and noise seed, patch a localized VLM token/subspace between a correct and controlled language/vision counterfactual. The resulting plan must move in the task-appropriate direction, not merely change.

Required controls:

- random active feature and norm-matched random direction;
- same-task time shift;
- unrelated-task source;
- source with a similar action chunk but different goal;
- source with the same goal but a different action chunk;
- interpolation dose response;
- hooked reconstruction/fidelity baseline.

A zero-out that broadly destroys the controller does not count. A successful test needs matched rescue, reverse-direction induction, semantic directionality, and preserved behavior under control hooks.

## Competing explanations and decisive signatures

| Hypothesis | Expected signature | Falsifying test |
|---|---|---|
| Trajectory/template lookup | Near training demonstrations; full plan/action history explains score; weak response to alternate feasible goals | New position/composition with the same subskills; training-trajectory retrieval; cross-goal `F` |
| Visual affordance shortcut | Correct-looking action despite contradictory language; sensitivity to wrist texture/background | Same scene with two feasible goals; semantic versus nuisance visual grid |
| Language surface lookup | Benign familiar wording works; synonyms, syntax, or multilingual variants break task identification | Controlled paraphrase taxonomy with identical state/noise |
| Generic progress monitor | Pixels/geometry and physical event state explain signal; divergence follows failure onset | Privileged-state conditional probe and prefix-blind event annotation |
| Sampled-plan evaluator | Action-expert score changes across noise at fixed state and predicts within-state `Q` | Full chunk/noise conditional test and matched plan patching |
| Grounded recoverability | Stable across paraphrase/nuisance/noise at state level; changes with goal-relevant state; calibrated to branched `V` | Restored-state CRD plus OOD position/composition transfer |

## Cost-gated programme

### Gate A — Forward-only alignment and invariance

No simulation expansion. Use a stratified set of saved observations around exact replanning boundaries.

- Re-query each observation with fixed noise seeds.
- Capture exact token-, action-offset-, and denoising-resolved activations and matching chunks.
- Quantify within-state noise variance and plan reconstruction.
- Test original/paraphrase/alternate/wrong/scrambled language and nuisance image controls.
- Verify hook fidelity and deterministic reproduction.

Advance only if the measurements are aligned and there is a task-directed effect beyond string/pixel magnitude.

### Gate B — Two-task branched causal smoke

Recommended shared-scene pair:

- LIBERO-Goal task 3: “open the top drawer and put the bowl inside” (6/10 in the existing run);
- LIBERO-Goal task 4: “put the bowl on top of the cabinet” (10/10).

They share the bowl but require incompatible destination and subskill structure. Build a common physical-state/multi-predicate evaluator rather than assuming task-specific BDDL states are automatically interchangeable.

Use five initial states per task, exact decision snapshots near steps 50 and 100, `K=4` first-plan seeds, and `M=2` continuation schedules as an initial ceiling of roughly 160 branched continuations. Predeclare one VLM site and one action-expert site from Gate A. This is materially new simulation usage and requires explicit approval before launch.

Advance only if:

- hidden state improves held-out-state prediction of branched `V` or `Q` beyond the complete baseline;
- cross-goal plan faithfulness is positive with paired uncertainty;
- paraphrase and nuisance effects are smaller than feasible-goal effects;
- a matched patch produces directionally correct action change and rescue/induction without generic fidelity loss.

### Gate C — Actual generalization

Only after Gate B:

- evaluate position and task/composition shifts from [LIBERO-PRO](https://arxiv.org/abs/2510.03827) first;
- add controlled lexical families from [LIBERO-Para](https://arxiv.org/abs/2603.28301) and feasible counterfactual goals inspired by [LIBERO-CF](https://arxiv.org/abs/2602.17659);
- retrieve the official training action/state shards and measure nearest-neighbor/DTW similarity without downloading video;
- validate the selected mechanism on one stronger flow VLA and one architecture with a different action decoder.

The external-model test asks whether CRD is architecture-general. It should not be used merely to reproduce a larger AUPRC.

## Statistical contract

- Use state/task groups in every fitted transform and hyperparameter selection.
- Treat replanning landmarks as risk sets; do not average completed successes into prospective horizons.
- Report prevalence, normalized AUPRC gain, AUROC, Brier score, log loss, and calibration.
- Use paired task/state bootstrap or a hierarchical model for intervention contrasts.
- Use common random numbers across counterfactual branches.
- Predeclare sites, horizons, and primary contrasts; correct exploratory layer/token families.
- Report coverage: tasks without both outcomes cannot contribute within-task ranking.
- Separate retrospective full-trajectory analysis from online early warning.
- Evaluate decision utility: failures prevented, false interventions, compute per rescued episode, and recovery latency.

## Stop conditions

- If exact online features do not reproduce post-hoc findings, archive the old signal as an offline diagnostic artifact.
- If full plan, geometry, and pixels erase conditional hidden value, narrow the claim to behavior/progress monitoring.
- If goal swaps change scores without correct cross-goal direction, reject language-grounded value.
- If patches only cause generic controller collapse, reject localized causal interpretation.
- If effects disappear under position/composition shifts, label them LIBERO interpolation effects.
- If the mechanism fails on an independent architecture, report a SmolVLA-specific mechanism rather than a VLA-general one.

## Why this can be a meaningful contribution

Recent work already establishes that outcome information is decodable and sometimes useful for candidate selection. It does not yet tell us what that information *is*. CRD targets that missing identification problem with a falsifiable decomposition and a small-model-first experimental design.

The strongest possible outcome is not “our probe has higher AUPRC.” It is a statement such as:

> At a fixed physical state, the VLM carries a noise-stable estimate of goal-conditioned recoverability, while the action expert carries plan-specific deviations around that state value; matched interventions selectively transfer those quantities and change cross-goal success.

The equally valuable negative outcome is:

> The apparent value signal is explained by familiar layout, current pixels, and sampled trajectory geometry, and it does not survive semantic counterfactuals or OOD compositions.

Either result would be substantially more informative than another standard LIBERO success/failure probe.

## Primary sources and reusable implementations

- [SmolVLA paper](https://arxiv.org/abs/2506.01844), [official LeRobot documentation](https://github.com/huggingface/lerobot/blob/main/docs/source/smolvla.mdx), [LIBERO-finetuned model card](https://huggingface.co/lerobot/smolvla_libero), and [training dataset](https://huggingface.co/datasets/lerobot/libero)
- [What Are We Actually Benchmarking in Robot Manipulation?](https://arxiv.org/abs/2606.04233) and its [diagnostic implementations](https://ripl.github.io/manipulation_benchmark_audit/)
- [LIBERO-PRO paper](https://arxiv.org/abs/2510.03827) and [official repository](https://github.com/RLinf/LIBERO-PRO)
- [What Frozen VLAs Already Know About Success](https://arxiv.org/abs/2605.28527)
- [Hide-and-Seek](https://arxiv.org/abs/2605.30834), [ProbeAct](https://arxiv.org/abs/2606.09740), and [Event-Grounded Sparse Autoencoders](https://arxiv.org/abs/2605.17204)
- [LIBERO-CF](https://arxiv.org/abs/2602.17659), [ICBench](https://arxiv.org/abs/2603.06001), [LIBERO-Para](https://arxiv.org/abs/2603.28301), and [step-wise multilingual sensitivity](https://arxiv.org/abs/2606.11906)
- [AC-VLA](https://arxiv.org/abs/2607.15714), [Drop-Then-Recovery](https://arxiv.org/abs/2606.27755), and [Anchor-Align](https://arxiv.org/abs/2607.13429)
- [PhysReflect-VLA](https://arxiv.org/abs/2606.27146), [Harness VLA](https://arxiv.org/abs/2607.08448), and [RoboHarness](https://arxiv.org/abs/2607.18060) are the closest current collisions on feasibility-aware execution, factorized VLA/analytic control, and heterogeneous capability routing. They narrow this project's candidate contribution to identical-restored-state causal decomposition and model-local self-specificity, not controller factorization or capability modelling in general.
- [Skill or Luck? Return Decomposition via Advantage Functions](https://arxiv.org/abs/2402.12874) is the conceptual precedent for separating action-caused return from luck; `Q-V` here should not be presented as new advantage mathematics.
- [Control-task probe selectivity](https://aclanthology.org/D19-1275/), [conditional probing](https://aclanthology.org/2021.emnlp-main.122/), [Causal Abstraction](https://arxiv.org/abs/2301.04709), and [Distributed Alignment Search](https://arxiv.org/abs/2303.02536)
