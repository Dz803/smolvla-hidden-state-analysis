# Competence–Compatibility Decomposition

## Result and decision

Stage A exposes a measurement error that would otherwise contaminate the later
hidden-state study: failure of a finite replay or proposal bank is not the same as
physical infeasibility.

The verified 32-state lattice contains 64 state-goal cells. All 64 have passing
policy-independent physical evidence, but only 63 have a successful member of their
bound replay bank. For the final open-drawer, grasped, cabinet-side,
transverse-low-support, layout-A cell:

- the complete registered cabinet bank remains `0/46`;
- all 46 action-phase bridges pass;
- no attempt reaches the wrong goal or terminates early;
- the matched demonstration-near state is `1/46` under the same proposal bank and
  execution contract; and
- a separate factorized path achieves stable acquisition and reaches the cabinet
  with `95 + 95 + 14 = 204` feedback transport/release actions.

This is a **physical-recoverability/proposal-compatibility gap**. It falsifies the
inference “all registered proposals failed, therefore the state-goal is physically
infeasible.” It does not show that SmolVLA can solve the cell. Stage A loaded no VLA,
and the factorized route was developed after observing the failure. It is one
existence certificate, not an unbiased estimate of a gap rate.

Compact evidence is in
`reports/phase3b_stage_a/competence_compatibility_gap_v1/`. The v37 negative ledger
and v4 factorized certificate remain separate evidence classes; neither replaces the
other.

## The broader blind spot

Robot-policy studies often collapse several causes into one binary episode label:

1. the goal is physically unreachable from the state;
2. the evaluated policy cannot recover from the state;
3. the policy is competent on average but sampled a poor proposal;
4. individually valid subskills do not compose because one edge ends outside the
   next edge's basin;
5. the observation/action adapter, replanning cadence, or horizon is incompatible;
6. success or failure is caused by continuation noise after the first proposal; or
7. the policy is capable but its internal monitor does not know when it is not.

An ordinary benchmark success rate cannot identify these causes. A generic failure
probe inherits the same ambiguity: high probe accuracy may reflect common scene
difficulty, task identity, proposal geometry, or a model-specific limitation.

The Stage A discovery suggests that the primary scientific object should be a
**competence–compatibility ladder**, not a single value label.

## Estimand ladder

For certified state `s`, goal `g`, model or policy `m`, first-proposal seed `k`, and
continuation schedule `r`, define:

\[
F_{\mathcal C}(s,g)
= \mathbf 1\{\text{a path in frozen policy-independent class }\mathcal C
\text{ passes}\},
\]

\[
V_m(s,g)=\mathbb E_{k,r}[Y_{mgkr}\mid do(S=s),G=g],
\]

\[
Q_m(s,g,k)=\mathbb E_r[Y_{mgkr}\mid do(S=s),G=g,K=k],
\]

\[
L_m(s,g,k)=Q_m(s,g,k)-V_m(s,g).
\]

`F_C` is physical recoverability only relative to a declared certificate/controller
class. `V_m` is policy-system recoverability under an explicit runtime contract.
`Q_m` is proposal-conditioned recoverability. `L_m` is proposal-specific luck or
harm relative to that policy's state value. None is a universal property of the
scene.

The current replay-bank quantity is narrower:

\[
P_{\mathcal K}(s,g)=\max_{k\in\mathcal K}Y(N(s),g,k),
\]

where `N` and the finite proposal inventory `K` are part of the execution contract.
The Stage A gap is `F_C=1, P_K=0`. It is not yet a `V_m` result.

For multiple policies, use direct statewise summaries before fitting a hierarchical
model:

\[
D(s,g)=1-\frac{1}{M}\sum_m V_m(s,g)
\quad\text{(common difficulty)},
\]

\[
C_m(s,g)=V_m(s,g)-\frac{1}{M}\sum_j V_j(s,g)
\quad\text{(policy-specific competence residual)}.
\]

Factor effects must also be paired within policy. For a matched factor change `x`,
estimate `Delta_x V_m`; the heterogeneity
`Delta_x V_m - mean_j Delta_x V_j` is the policy-specific factor interaction. With
only a few models, these quantities should be reported descriptively with paired
state-family uncertainty. A cross-classified binomial model is secondary shrinkage,
not a substitute for the paired cells.

## The proposed overlooked-discovery design

### 1. Hold the environment causal interface fixed

Every model receives the same certified simulator state, rendered observations,
goal predicate, and common random-number schedule where its decoder permits it.
Every branch records both portable goal predicates at every prefix. Comparison is in
environment effect space—object pose, end-effector displacement, contacts, grasp,
and predicate change—not raw action or latent coordinates.

Use two explicitly separate tracks:

- **ecological track:** each released system keeps its native processor, action
  horizon, and replanning cadence;
- **controller-normalised track:** models share an executed-prefix/replanning
  cadence only where this does not violate their action contract.

Never average the tracks. A disagreement is an interface sensitivity result.

### 2. Cross policies with state, goal, proposal, and continuation

For each locked state-goal cell, evaluate SmolVLA, pi0.5, and GR00T with common
proposal identities/noise analogues and paired continuation schedules. The minimum
useful cell is `policy × goal × first proposal × continuation`; aggregate benchmark
episodes are not enough.

The resulting outcome tensor supports four questions:

- **common difficulty:** do all systems fail on the same physically feasible roots?
- **policy-specific competence:** does one system recover where the others fail?
- **proposal-specific luck:** does outcome vary substantially across first proposals
  within one system and state?
- **causal interaction:** does a matched factor such as low support, possession,
  locus, or goal switch affect systems differently?

This is an overlook finder: large aggregate means can conceal narrow cells with
opposite policy residuals or factor interactions.

### 3. Ask whether each model knows its own limitation

Raw hidden dimensions are architecture-specific and should not be aligned. Fit a
model-local representation function inside each training fold and compare proper
score improvements under held-out state families.

For model `m`, define a self-specificity contrast:

\[
S_m=\Delta\ell(H_m\rightarrow Y_m\mid B)
-\frac{1}{M-1}\sum_{j\ne m}\Delta\ell(H_m\rightarrow Y_j\mid B),
\]

where `B` contains exact physical geometry, pixels or a frozen visual baseline,
goal, policy/runtime contract, full proposed action and short-prefix effect,
proposal seed, and continuation metadata. A positive held-family `S_m` means the
representation is more informative about its own residual outcome than about common
scene difficulty. The same test should be run separately for `V_m`, `Q_m`, and
`L_m`.

A hidden state that predicts every policy equally is primarily a scene-difficulty
monitor. A state that predicts only `Q_m` after controlling for the decoded action is
a candidate plan-quality representation. A state that predicts `C_m` may encode the
model's own competence boundary. These claims require calibration and held-group
proper scores, not only AUPRC.

### 4. Localise modality and VLM-to-flow use only after the information gate

At identical state and flow noise, capture exact action-producing features for:

- language tokens;
- main- and wrist-image tokens;
- the VLM-to-action-expert interface;
- action-expert states by denoising time and action position; and
- executable-output, padding-output, and decoder-null projections.

Use feasible goal swaps as the primary language intervention. Paraphrases test
surface invariance; blank, scrambled, or impossible commands are secondary
out-of-distribution controls. Main/wrist masks require natural-image and
pixel-magnitude controls.

Only sites that improve held-family `V`, `Q`, or `L` prediction beyond the complete
baseline advance to intervention. Patching must then produce a task-directed action
or environment-effect change, reciprocal rescue/induction, dose response, and
norm-matched/random/time-shift controls. Generic disruption is not a mechanism.

### 5. Distinguish memorisation from a policy attractor

Literal action-window retrieval is only one form of memorisation, and the existing
nearest-neighbour audit found no exact copied action window. The stronger competing
hypothesis is a phase-locked policy attractor: current observations cue a familiar
task/trajectory template even when residual geometry requires a different
composition.

Test it with three distances kept separate:

- joint training occupancy distance in robot state, object geometry, event, and
  motion;
- physical recoverability/cost under the frozen certificate class; and
- model-specific proposal/effect distance.

Evidence for a template attractor would be a low-support collapse at matched
`F_C`/cost, goal-swap vetoing without constructive alternate-goal success, and
hidden states that predict proposal compatibility or training occupancy but not
held-state `V_m`. Robust goal-directed composition would instead survive novel
geometry and select different effects when the feasible goal changes.

## Anti-overlap and literature boundary

This project should not claim that value-like VLA features, physical feasibility
checking, factorized VLA/analytic control, or heterogeneous policy capability
routing are new:

- [What Frozen VLAs Already Know About Success](https://arxiv.org/abs/2605.28527)
  probes frozen OpenVLA/pi0.5 representations and uses a probe to select sampled
  action prefixes. Our open question is identification of common difficulty versus
  model-specific `V/Q/L`, not generic success decodability or candidate selection.
- [PhysReflect-VLA](https://arxiv.org/abs/2606.27146) learns a physical-consistency
  filter and reflection-guided resampling to improve execution. Our certificate is
  an external measurement baseline and our target is causal decomposition, not a
  learned feasibility module or policy repair.
- [Harness VLA](https://arxiv.org/abs/2607.08448) composes a frozen VLA with analytic
  primitives for robust execution. [RoboHarness](https://arxiv.org/abs/2607.18060)
  learns capability boundaries and routes among heterogeneous policies. Our proposed
  contribution is an identical-restored-state factorial that estimates policy
  residuals and internal self-specificity, not an orchestration system.
- [Skill or Luck?](https://arxiv.org/abs/2402.12874) gives an advantage-based causal
  return decomposition in reinforcement learning. We should credit the conceptual
  lineage; our empirical `Q-V` target is a nested counterfactual diagnostic for
  frozen imitation policies, not new advantage mathematics.
- [What Are We Actually Benchmarking in Robot Manipulation?](https://arxiv.org/abs/2606.04233)
  shows that benchmark scores can reflect shortcuts and data-source dependence.
  [LIBERO-PRO](https://arxiv.org/abs/2510.03827) supplies broader perturbation axes.
  Our design complements them by holding the computational state fixed and
  decomposing the policy axis.

The novelty claim remains a hypothesis until a broader systematic review and the
cross-model experiment are complete. The safest current statement is: **the Stage A
evidence reveals a competence–compatibility measurement blind spot, and the proposed
state-matched decomposition is designed to identify it across policies and internal
representations.**

## Confirmatory sequence and stop rules

1. **Stage A complete:** publish the 32 exact roots, evidence-class separation, and
   adaptive gap as exploratory evidence.
2. **Controller confirmation:** freeze the factorized controller class, tolerances,
   and accounting before applying it to untouched roots. Report its coverage and
   failures; do not add new repair rules within the confirmatory split.
3. **SmolVLA Stage B:** run the bounded 16-state, two-goal, two-proposal,
   two-continuation pilot only under the separate simulation approval already
   required by the active plan.
4. **Adapter-only cross-model gate:** implement and validate observation, action,
   predicate, state-hydration, and short-prefix effect contracts for pi0.5 and GR00T
   before downloading or running full checkpoints.
5. **Cross-model confirmation:** use identical locked roots and report ecological and
   controller-normalised tracks separately. Do not align raw latent dimensions.
6. **Mechanistic gate:** patch only preregistered sites whose self-specific `V/Q/L`
   information survives held-family controls.

Stop or narrow the claim when:

- `F_C` is not stable under a frozen held-out certificate;
- outcomes are complete floor or ceiling and cannot identify proposal effects;
- hidden increments vanish after complete observable/action/effect controls;
- self-specificity is non-positive, indicating generic difficulty rather than
  self-knowledge;
- goal swaps only suppress the familiar action without constructing the requested
  alternative; or
- a patch changes actions through nonspecific damage rather than directional
  rescue/induction.
