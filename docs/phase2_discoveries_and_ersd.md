# Phase 2 Discoveries: From Latent Semantics to Executable Control

Date: 2026-07-27; corrected 2026-07-28 after the archived-camera orientation audit

## Decision

Phase 2 passes the measurement-alignment gate. SmolVLA can now be measured in the exact forward pass that creates an executed action queue, under explicit flow noise, without pooling away modality tokens, action positions, or denoising time. The corrected canonical fixed-forward artifact is `phase2_forward_gate_20260727T123138Z`; the earlier `114152Z` artifact used saved cameras before the required 180-degree policy transform and is retained only as superseded evidence.

The results do **not** yet establish a causal hidden-state mechanism. They support a narrower and more interesting hypothesis:

> SmolVLA's VLM and action pathway respond differently to controlled goal, language, and camera changes. Most latent displacement is not directly executable, while a small decoder-potent projection can dominate behavior. The corrected gate does not support the previously reported step-dependent semantic-routing reversal.

This motivates **Executable–Reflective Subspace Decomposition (ERSD)** inside **Cross-Policy Recoverability Decomposition (CPRD)**. ERSD asks what part of a representation can affect the robot. CPRD asks whether that representation describes common state difficulty, a particular policy's competence, or one sampled proposal's luck.

## What Phase 2 actually establishes

| Claim | Status | Evidence boundary |
|---|---|---|
| Captures correspond to the action-producing query | Supported | Exact hooked and unhooked chunks differ by `0`; query ID, noise seed, queue age, and chunk index are linked |
| Modality and flow-time structure is preserved | Supported | Separate language/main/wrist/state spans, 50 action positions, four layers, and ten denoising calls |
| Proposal randomness enters after the VLM prefix | Supported for this SmolVLA checkpoint | VLM and prefix KV statistics are exactly invariant across flow seeds |
| Goal semantics can affect the action path | Supported at four fixed observations | Alternate-goal effects are directionally amplified relative to paraphrase and image-mean controls |
| A step-dependent contradiction-routing reversal exists | Rejected after input correction | Correctly oriented expert/flow geometry is generally alternate-goal aligned at both landmarks; only final executed velocity at step 50 is weakly original-aligned |
| The policy literally copies a training action window | Not supported for the audited queries | No exact window match; nearest-neighbour evidence is inconsistent with literal retrieval |
| The model uses a trajectory-phase attractor or prototype | Plausible, not established | Same-step plans across different episodes are unusually similar; controlled trajectory deviations are still required |
| A hidden state causes success or failure | Not established | No matched branched outcome or activation rescue/induction has yet been run |

## Discovery 1: semantic change is amplified directionally, not by magnitude

The fixed-forward gate uses four archived task-3 observations at exact replanning landmarks 50 and 100. It compares four proposal seeds plus fixed-noise language and image counterfactuals.

- A feasible alternate goal changes the final VLM representation by median RMSE `0.0332` and the action chunk by `0.472`, an action-to-VLM displacement gain of `14.15`.
- A meaning-preserving paraphrase produces a similar VLM displacement (`0.0368`) but a smaller action effect (`0.350`; gain `9.56`).
- Replacing the main or wrist image by its mean produces larger VLM displacements (`0.145` and `0.132`) and action effects of `0.462` and `0.958`. The wrist manipulation is the largest action perturbation, while its gain (`7.05`) is below the alternate goal's.

Therefore latent distance, cache-norm change, CKA, or probe accuracy alone cannot identify what the controller uses. The scientifically relevant quantity is the **directional derivative from a controlled latent displacement to an executable effect**.

## Discovery 2: the apparent semantic-routing reversal was an input-contract artifact

The original analysis fed archived cameras directly to the policy even though the rollout environment rotates both image axes before inference. Hook fidelity and fixed-noise repeatability were exact for that input, but the input was off contract. After applying the environment transform and rerunning only the saved-observation forward gate, the strong step-50/step-100 reversal disappears.

Correctly oriented VLM, action-expert, and flow-state geometry is generally alternate-goal aligned at both landmarks. Only the final executed velocity at step 50 is weakly original-aligned (`-0.056`). The durable result is therefore controlled semantic sensitivity and downstream amplification—not a routing reversal. This correction is a reminder that deterministic neural measurements can still be scientifically invalid when preprocessing is wrong.

## Discovery 3: the hidden space is mostly not the action space

The final action-expert state has 720 dimensions. Its output head predicts 32 channels, but LIBERO executes only seven; 25 padded channels are discarded.

For proposal-noise differences, median hidden displacement energy is:

- `7.67%` in the executed-output row space;
- `67.14%` in the padding-output row space;
- `74.98%` in the complete output row space;
- `25.02%` in the output-null space.

For controlled goal/language/image differences, `97.85–98.64%` of final hidden displacement lies in the complete output-null space. Nevertheless, the small executed projection produces large action changes.

This explains how two apparently conflicting results can coexist:

1. a probe can decode rich context or failure information from the full hidden vector; and
2. only a small, directionally selected part of that vector can affect the immediate physical command.

The 25 padded channels also expose an implementation-specific confound. Their denoising dynamics carry substantial seed variation even though they never reach the robot. Comparing full hidden-state variance across models would therefore mix executable control, auxiliary computation, and architecture-specific padding.

## ERSD: an architecture-neutral latent audit

Let `h` be a model-local hidden vector at an action token and decoder time. Let `e` be a standardized environment-effect vector: commanded end-effector motion, gripper command, predicted contact/object displacement, or the short-prefix state transition. Define the local effect Jacobian

\[
J_E(h)=\frac{\partial e}{\partial h}.
\]

Use an SVD of `J_E` to construct an orthonormal executable basis `B_E`. If a model has auxiliary output channels, first remove the executable row space from their Jacobian and construct `B_A`. Then

\[
P_E=B_EB_E^\top,\qquad
P_A=B_AB_A^\top,\qquad
P_N=I-P_E-P_A.
\]

For a controlled displacement `Δh`, report

\[
\rho_E=\frac{\lVert P_E\Delta h\rVert^2}{\lVert\Delta h\rVert^2},\quad
\rho_A=\frac{\lVert P_A\Delta h\rVert^2}{\lVert\Delta h\rVert^2},\quad
\rho_N=1-\rho_E-\rho_A,
\]

alongside the actual effect `||J_E Δh||`, its semantic direction, and nonlinear finite-difference verification.

For SmolVLA, `J_E` can be computed both at each denoising velocity head and end-to-end from an action-expert site to the final seven-channel plan. For π0.5 and GR00T, the native latent dimensions and heads may differ, but the induced physical effect is shared. This avoids pretending that raw hidden dimensions or raw action tensors are comparable across architectures.

ERSD yields three testable roles:

- **Executable:** the component has a verified local effect on action or short-prefix state transition.
- **Auxiliary:** the component drives decoder outputs that are not part of the standardized executed effect, including padding or model-specific latent-action targets.
- **Reflective:** an output-null component predicts recoverability or policy limitation after all observable/effect controls, but has no immediate local effect at the audited decoder boundary.

“Reflective” is a hypothesis, not a synonym for “null.” Null-space decodability alone may be inert context storage. A knowledge–action gap is supported only if a null component predicts the model's own limitation under grouped held-out evaluation, while a matched potent-space intervention changes the predicted effect and downstream outcome in the expected direction.

## Discovery 4: a simulator snapshot is not a matched computational state

LIBERO's flattened MuJoCo state omits controller interpolator starts/goals, accumulated Panda gripper commands, robot history buffers, environment clocks, observable timing, vector-wrapper flags, and RNG. These variables are causal parents of the next transition.

In the real contract test, replaying the same three actions after restoring only MuJoCo state caused:

- maximum final simulator-state error `0.037207`;
- maximum main/wrist image errors `95/255` and `172/255`;
- reference gripper memory `[0.25, -0.25]` versus replay `[-0.50, 0.50]`.

Restoring the then-captured runtime state reproduced both cameras exactly and final simulator state to `7.12e-15` in the Phase 2 smoke, with predicates, grasp state, termination bookkeeping, and RNG also matching.

Phase 3 found that this was necessary but not sufficient. Contact-sensitive replay additionally requires MuJoCo's `qacc_warmstart`; and even a full serialized snapshot is not reliably portable into a cold simulator instance. The hardened rule is now **current-process archived-action reconstruction plus a repeated-transition certificate**, not cross-process snapshot trust. See `phase3_engineering_review.md`.

Every CPRD branch must therefore pass a **computational-state certificate**. A branch is admissible only if an identical probe-action sequence reproduces physics, observations, controller/gripper state, predicates, contacts or grasp state, clocks, termination flags, and RNG within predeclared tolerances. “Same MuJoCo state” is insufficient.

## Discovery 5: no literal trajectory copy, but a phase-locked prototype remains plausible

Only the official revision-pinned `lerobot/libero` state/action Parquet was retrieved: 1,693 episodes, 273,465 frames, and 190,508 valid horizon-50 windows. No video or external checkpoint was downloaded.

After standardizing with official statistics:

- train-window self-nearest horizon-50 RMSE has median `0.449`;
- fixed-forward proposals have same-task nearest RMSE `0.553`, or `1.229×` the train self-nearest median;
- there are zero exact action-window matches at tolerance `1e-6`;
- restricting candidates to the 128 closest same-task robot states worsens action RMSE from `0.553` to `0.813`;
- within-state proposal-seed dispersion is much smaller (`0.176`).

An action-only search misleadingly finds closer windows in other tasks (`0.446`), but their state mismatch is `1.707` and the matches are mostly low-motion, saturated-gripper episode tails. This is a concrete false positive: generic stopping motions can look like retrieval even when task and state are wrong.

The more interesting signal is phase structure. With the same flow seed, plans at the same landmark across a success and failure episode differ by `0.222`, only modestly above within-state seed variation, whereas plans from different landmarks differ by about `0.624`. This is compatible with a narrow **trajectory-phase attractor** or policy prototype, not episodic demonstration copying.

The next decisive controls are:

- restore the full computational state and independently vary goal-relevant object milestones;
- perturb the robot off the familiar path while preserving goal feasibility, then test recovery versus return to a template;
- stratify retrieval by privileged object geometry and visual embedding, not action alone;
- test novel object positions/compositions and report performance by training-retrieval distance;
- separate native 50-step execution from a common shorter replanning cadence.

## CPRD: separating difficulty, competence, and proposal luck

For policy `m`, certified state `s`, goal `g`, controlled factor setting `f`, native proposal seed `z`, and continuation randomness `r`, observe a success or graded predicate outcome

\[
Y_{m,s,g,f,z,r}.
\]

Define state recoverability and proposal-conditioned recoverability:

\[
V_m(s,g,f)=\mathbb{E}_{z,r}[Y],\qquad
Q_m(s,g,f,z)=\mathbb{E}_{r}[Y\mid z].
\]

Then proposal-specific luck or quality is

\[
L_m(s,g,f,z)=Q_m(s,g,f,z)-V_m(s,g,f).
\]

At the same state, empirical common-mode difficulty can be summarized from the collection of `V_m`; policy-specific competence is each model's paired residual from that common mode. Direct statewise contrasts remain primary. A simulation-calibrated hierarchical binomial model can then estimate partial pooling for:

- common state difficulty;
- global policy baseline;
- policy × state competence residuals;
- factor main effects;
- policy × factor and factor × factor interactions;
- proposal deviation `L`;
- continuation noise.

For two causal factors `a` and `b`, the statewise interaction is the difference in differences

\[
I_m(s)=V_m(s,a_1,b_1)-V_m(s,a_1,b_0)-V_m(s,a_0,b_1)+V_m(s,a_0,b_0).
\]

An interaction is an overlook only after it repeats with the predicted direction on a fresh held-out state/task family. A flexible risk model discovering an interaction is screening, not confirmation.

## Does each model know its own limitation?

The VLM prefix is noise-invariant in SmolVLA, so it is a natural candidate for state-level `V`. The action expert varies with proposal noise, so its residual is a candidate for proposal-level `L`. CPRD can falsify this clean decomposition.

For each model-local representation, measure grouped cross-fitted proper-score improvement for four targets after controlling privileged geometry, pixels, history, queue state, complete proposed effect, retrieval distance, factor metadata, and noise:

1. common state difficulty;
2. that model's policy-specific competence residual;
3. that proposal's `L`;
4. the other policies' outcomes at the same state.

Define self-specificity as own-policy conditional log-score improvement minus mean other-policy improvement on matched states. A representation that predicts all policies equally is a scene-difficulty monitor. Preferential prediction of its own residual is evidence of introspective competence, provided the effect survives model-capacity controls, grouped uncertainty, and an independent architecture.

Raw hidden states should never be aligned across models for this test. Each model gets its own probe and ERSD basis; only targets and effects are shared.

## Phase 3 follow-through

The 160-branch Phase 3 smoke is complete. It validates the CRD execution machinery but does not pass the conditional hidden-value gate: the predeclared low-dimensional summaries do not improve held-source-episode `V`, `Q`, or `L` prediction beyond complete controls. The state bank also exposes source-task progress and occupancy confounding. The current design should therefore not be copied unchanged to π0.5 and GR00T. The full result and occupancy-balanced successor are in `phase3_engineering_review.md`.

## Novelty boundary

The mathematical idea of output-potent versus output-null neural activity is established in motor neuroscience, including [Kaufman et al.](https://www.nature.com/articles/nn.3643). The project should not claim to invent null-space analysis.

[COAST](https://arxiv.org/abs/2605.17144) is the closest VLA collision: it learns success/failure subspaces and steers multiple action architectures, including π0.5 and GR00T. Its reported construction pools action/state/future tokens from ordinary rollouts and does not identify matched computational state, fixed proposal noise, proposal luck, or decoder-effect potent/auxiliary/null components. [Output-Level Regularization Eliminates the Seed Lottery](https://arxiv.org/abs/2606.13856) uses a VLA Jacobian null-space argument in parameter-update space; ERSD instead concerns inference-time hidden-to-executed-effect geometry.

The candidate contribution is therefore the **combination**:

> certified state matching + recoverability/proposal decomposition + executable-effect geometry + policy self-specificity + independently confirmed causal interactions.

That combination is a research hypothesis, not yet a publication claim. Its value is that it can distinguish a grounded internal limitation signal from common scene difficulty, a familiar trajectory template, stochastic proposal luck, and representation that is decodable but unable to affect the robot.

## Canonical Phase 2 artifacts

- Corrected exact-forward report: `reports/phase2_forward_gate/phase2_forward_gate_20260727T123138Z/`
- Full-runtime state certificate: `reports/phase2_state_gate/state_contract_20260727T113937Z/`
- Training retrieval audit: `reports/trajectory_memorization_audit/phase2_train_retrieval_a1aaacb_v3/`
- Corrected raw exact tensors: `local/phase2_forward_gate/phase2_forward_gate_20260727T123138Z/queries.zarr` (workstation-only)
- Official training shard revision: `lerobot/libero@a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`

No canonical rollout was rerun or modified, and no artifact under `archive/full_experiment/runs` was written.
