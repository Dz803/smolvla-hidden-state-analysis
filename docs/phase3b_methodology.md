# Phase 3b Methodology: Occupancy-Balanced CRD

## Recommendation

The next experiment should not copy the Phase 3 source-trajectory states to a larger model. Phase 3 showed that the apparent cross-goal asymmetry is entangled with drawer position, bowl possession and height, end-effector pose, and source-policy progress. The next gate should create states independently of every evaluated policy and ask whether language constructs the requested alternative behaviour when both goals are physically feasible and equally far away.

The work is split into three locked stages. Only Stage A is the immediate implementation target. Stages B and C require separate simulation-scope decisions.

## Scientific question

At the same certified computational state, does an instruction change:

1. merely veto the visually prepared or familiar skill;
2. construct a feasible alternative goal-directed plan;
3. lose control when the state is equally feasible but outside demonstration or policy occupancy; and
4. produce a directional hidden-state signal that predicts the model's own recoverability or proposal luck beyond the observable state and proposed environment effect?

This distinguishes semantic composition from a scene-conditioned skill attractor. Because the policy is reset before every branch and SmolVLA has no observation-history queue, the hypothesis is not literal recurrent memory; it is current-state occupancy or trajectory-template capture.

## Stage A: policy-independent state lattice

Build 32 candidates from a balanced `2 x 2 x 2 x 2 x 2` design:

- drawer aperture: `closed` or `open`;
- bowl possession: `on_table` or `grasped`;
- transit locus: `drawer_side` or `cabinet_side`;
- support stratum: `demonstration_near` or `transverse_low_support`;
- scene-layout replicate: `A` or `B`.

The two support strata must be constructed without running an evaluated VLA. A `demonstration_near` state comes from a scripted task waypoint. Its paired `transverse_low_support` state receives a controlled pose perturbation away from the scripted path while preserving the same discrete affordances and approximately the same oracle remaining cost. State selection is locked before any VLA outcome is observed.

Every retained state must satisfy all of the following:

- neither target predicate is already true;
- both the drawer-inside and cabinet-top goals are reached by at least one member of a policy-independent, revision-locked human-demonstration proposal inventory;
- the state contains no invalid penetration, unstable attachment, or unintended terminal condition;
- its two goal-specific oracle costs and relevant geometry are recorded;
- archive-independent computational-state restoration passes the existing repeated-action certificate, including solver warm-start state; and
- the state has a typed provenance record, full factor labels, an immutable state hash, and a construction-script revision.

Persisted snapshots remain provenance records, not trusted cross-process branch roots. Every new runner process must recreate or replay the scripted construction inside its own hydrated environment, compare it with the recorded state hash, and rerun the probe-action certificate before querying a policy.

Proposal feasibility is set-valued. Let `N(s)` denote the policy-independent recovery and homing continuation applied before demonstration replay. For the complete cached task-12/task-18 inventories, Stage A measures `Y(N(s),g,k)`, defines feasibility of the original state by the composed continuation `N` followed by `max_k Y`, reports proposal-basin width as the successful-proposal fraction, and selects the minimum executed-step/path/effort success by a fixed rule. The physical balance gate compares those minimum feasible composed costs. Success-set intersection and Jaccard are reported separately; when an intersection exists, same-proposal costs are additional diagnostics. A zero intersection is retained rather than used to remove the off-trajectory state.

This notation matters: possession and transit-locus differences are intentionally removed before the open-loop demonstration suffix, and the full normalized controller/simulator state need not be identical across candidates. Proposal-set changes can therefore be mediated by residual differences in `N(s)` and are not direct evidence of trajectory memory at root `s`. Stage B evaluates the VLA directly from `s`; it must not inherit the oracle normalization step or treat Stage A proposal coverage as model occupancy.

Balance is assessed before policy evaluation. The pilot selection must cover every `drawer x possession x locus` family and both support strata. Within each family, the support pair should have the same predicate state and an oracle remaining-cost difference no larger than 10% unless the deviation is documented and included as a covariate. The joint support reference currently replays two role-separated demonstrations in both layouts; it is a narrow reference-distance diagnostic, not a complete estimate of SmolVLA's training occupancy. Geometric support labels, measured reference distance, exact-event coverage, proposal-basin width, and any future model-specific occupancy distance must remain separate quantities.

Stage A produces code, unit tests, a state-lattice specification, certificates, the full proposal-coverage matrix, and a compact manifest. It runs no SmolVLA branch matrix and downloads no new checkpoint.

### Stage A execution amendment: phase and alignment certificates

The July 2026 execution falsified the assumption that one full-trajectory or world-frame phase bank is a universal feasibility oracle. Complete drawer demonstrations fail after the drawer is already open; the action-phase adapter restores coverage on one root; and its sole cabinet success on layout A fails on the matched layout-B root. A bowl-relative anchor intervention rescues that exact layout-B suffix. These are controller/proposal compatibility effects, not changes in the underlying target predicate.

Stage A therefore distinguishes a competence ladder rather than using “oracle pass” as a primitive:

1. the certified physical root and target predicates;
2. collision-safe bridge reachability with the original affordances preserved;
3. contact acquisition and stable possession;
4. grasp-preserving transport;
5. target placement and release; and
6. compatibility of a complete open-loop proposal with the root.

Existing exhaustive ledgers are immutable evidence and must not be rerun. Any completion contract may add a frozen, deterministic object-relative certificate only where evidence is missing, and must retain the original negative ledger. Certificate family, anchor displacement, successful-proposal identity, basin width, and stage of failure are separate fields. Coverage fractions from full replay, phase slicing, and registered/factorized controllers are different estimands and must not be pooled.

The resulting 32-state artifact is an exact observed-root bank, not a clean aperture factorial. Closed roots were completed under v32, while open roots span later revisions; historical contracts do not cryptographically bind the complete construction configuration. Drawer aperture is therefore aliased with source revision. Primary Stage B language tests must use within-state goal/instruction contrasts under one common policy runner. Cross-aperture and cross-certificate interactions are exploratory unless a future balanced lattice is generated under one construction revision.

Every imported or additive record must validate the original state payload, root and normalized hashes, source-action identity, complete proposal indices, state certificate, and raw artifact hashes. A completed negative bank is evidence, not an invitation to repeat the search. Registered-controller evidence discovered on the first failed layout pair remains exploratory; its first untouched-state applications are the relevant generalization test.

That prospective gate passed on the two transverse-low-support layout replicates using the frozen episode-474 proposal. This authorizes an additive completion shard under the same registration contract; it does not authorize pooling registered and legacy proposal-basin rates. The two smoke attempts are part of the completion evidence and must be imported, not executed again.

## Stage B: bounded SmolVLA manipulation pilot

Select 16 locked states, one from every affordance-family/support cell, with scene layouts balanced across the design. For each exact state evaluate:

- both feasible goal instructions;
- two common proposal-noise seeds; and
- two common continuation schedules.

This is a ceiling of `16 x 2 x 2 x 2 = 128` branches. The first proposal is generated once for each `state x goal x proposal seed` and reused across continuations. The paired goal queries use common flow noise. Record both goal predicates after every prefix, not only native task success.

Primary behavioural quantities are:

- `V(s,g)`: mean goal-conditioned recoverability;
- `Q(s,g,k)`: recoverability after the sampled first proposal;
- `L(s,g,k) = Q(s,g,k) - V(s,g)`: proposal-specific luck;
- constructive switching: gain in the requested alternative predicate under its matching instruction; and
- vetoing: suppression of the other predicate under the switched instruction.

A goal switch counts as grounded composition only when it increases the requested target, not merely when it suppresses the old trajectory. Report direct within-state paired contrasts before any hierarchical model.

The pilot passes the design gate only if:

- every branch root passes the computational-state certificate;
- factor coverage and the pre-policy oracle-cost balance checks pass;
- both goal marginal success rates avoid complete floor or ceiling; and
- at least four state-goal cells show proposal- or continuation-dependent outcomes, providing variation for the confirmatory `Q/L` test.

Failure of the last two checks is a scientific result about the selected difficulty range. It triggers a locked difficulty recalibration, not post-hoc removal of inconvenient states.

## Stage C: locked confirmation and directional hidden-state gate

Use the 16 unused layout-matched states as a confirmation set. A proposed ceiling is four proposal seeds and two continuations per state and goal: `16 x 2 x 4 x 2 = 256` branches. This stage is not authorised by the design document; it requires an explicit compute decision after Stage B.

The hidden-state analysis must preserve denoising time and action position. It should test a small preregistered family of directional features:

- same-noise goal-switch displacement at the VLM/action-expert interface;
- within-state proposal displacement centred over proposal seeds;
- signed projection into executed-output, padding-output, and output-null subspaces;
- denoising-time growth or rotation of those signed projections; and
- alignment with the proposal's measured short-prefix environment-effect vector.

Dimension reduction and direction fitting occur inside each training fold. Hold out complete affordance families and layouts, cap learned rank at `min(8, training_groups - 1)`, and never fit a transform on the confirmation group. Compare against a complete baseline containing privileged geometry, goal predicates, support/retrieval distance, pixels or a frozen visual embedding, proposal noise, full proposed action/effect, and continuation metadata.

The primary information gate is paired held-group proper-score improvement over that baseline. A site advances to Phase 4 only if one preregistered target (`V` or `L`) has positive improvement with a positive state-family-clustered uncertainty bound on the locked confirmation set. Exploratory directions may motivate a future study but cannot unlock patching.

## Language-prior and memorisation controls

The core causal language test is the feasible goal swap at identical state and noise. Meaning-preserving paraphrases estimate surface-form sensitivity. Blank, scrambled, or impossible instructions may be recorded as secondary out-of-distribution diagnostics, but they cannot establish grounding.

To test trajectory-template capture, stratify every result by demonstration-support distance and model-specific occupancy distance while holding oracle goal cost fixed. A success collapse only in low-support states supports an occupancy-attractor account; literal action-window retrieval requires joint matching of object geometry, robot state, event, and motion, not action-only nearest neighbours.

## Cross-model decision

Do not download π0.5 or GR00T merely to repeat the Phase 3 state bank. Adapter and effect-space interface work can begin after Stage A. Download and rollout work should begin only after Stage B validates the shared state, predicate, horizon, action-mapping, and short-prefix effect contracts.

Cross-model evaluation uses the identical state lattice but keeps two tracks separate:

- native horizons/processors for ecological system performance; and
- a common executed-prefix/replanning cadence where technically valid for controller-normalised comparison.

Raw latent dimensions are never aligned across architectures. The cross-model target is self-specificity: whether each model's hidden state predicts its own policy-state residual and proposal luck better than the matched outcomes of the other policies.

## Stop rules and interpretation

- Do not launch Stage B until Stage A certificates and balance reports pass.
- Do not launch Stage C, Phase 4 patching, or multi-model rollouts automatically.
- Do not reuse completed canonical rollouts or modify `archive/full_experiment/runs`.
- If directional features fail the locked information gate, report that negative result and retain CRD as a behavioural diagnostic; do not rescue the claim with high-dimensional feature searching.
- If the veto-composition gap disappears after balancing, the Phase 3 asymmetry was mainly state preparation or occupancy. If it persists, it becomes a stronger candidate for model-specific semantic/skill-routing failure.
