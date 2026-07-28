# Phase 3 Engineering and Scientific Review

- Date: 2026-07-28
- Run: `local/phase3_crd/phase3_crd_20260728T021125Z`
- Derived report: `reports/phase3_crd/phase3_crd_20260728T021125Z/`

## Verdict

The repaired Phase 3 ledger is suitable as a **method-validation smoke study**. It contains 160 certified branches, 80 first-proposal queries reused across two continuation schedules, 80 fixed-noise factor queries, ten states, and no partial artifacts. Every active branch is rooted in a current-process replay of immutable archived actions, passes archive fidelity and repeat-transition certificates, and satisfies the exact paired-prefix invariant.

The scientific result is narrower than a generalisation or memorisation claim. The experiment finds strong current-state/goal dependence, measurable first-proposal variation, weak continuation variation, a language **veto–composition gap**, and no conditional value from the tested low-dimensional hidden summaries. It does not establish recurrent trajectory memory, broad task generalisation, policy-specific self-knowledge, or a causal hidden-state mechanism.

## Questions a senior engineer should ask

### 1. Are the two branches actually rooted in the same computational state?

Initially, no. Robosuite's flattened `MjSimState` omits solver and control fields. A contact-state ablation isolated the within-instance divergence to `qacc_warmstart`: dropping it changes final physics by `2.40e-08` and one wrist-camera intensity level after three actions. More seriously, a full serialized snapshot restored into a cold simulator instance changes state by `0.1262` and pixels by up to `191/255`.

Fix: every source episode is now reconstructed once per process by replaying the immutable archived action prefix. Archive observations are matched exactly, a full in-memory runtime snapshot is captured, and repeated probe actions are certified before branches execute. Persisted snapshots remain provenance/query artifacts; they are not trusted as portable branch roots.

Evidence: `restore_field_ablation.json`, branch-level `source_reconstruction`, and `state_certificates.csv`.

### 2. Did the restore defect materially alter scientific labels?

Yes. One deterministic audit changed a recorded success at step 134 into a horizon failure. The two interruption-safe repair transactions preserved 114 old payloads and both old/new hashes. Regeneration changed 96 semantic payloads and flipped 12 success labels.

Fix: invalid payloads were moved to `superseded_branches/`; none were deleted or silently overwritten. Only affected branches were regenerated, and existing exact-forward queries were reused.

### 3. Can continuation randomness leak into the first plan?

The initial runner checked this only during post-run analysis. That was too late and helped expose an older invalid pair whose identical first chunk produced different prefix effects.

Fix: `validate_paired_first_plan` now runs before the second branch JSON is committed. It requires the two schedules `{0,1}`, identical source provenance and initial predicates, identical first-ten effect, identical executed prefix length, and identical first-plan effect. The analyzer uses the same validator. All 80 active pairs pass exactly.

### 4. Does the state certificate fail closed?

The earlier nested comparator silently skipped missing keys, changed shapes, and unequal nonnumeric values. Although no active certificate relied on that behaviour, it was an unsafe future failure mode.

Fix: one shared comparator now emits infinite error for schema/shape mismatch and unequal nonnumeric fields. Pixel fields remain exact; continuous observation and MuJoCo fields use the frozen `1e-10` thresholds. The runner also rejects a state if fewer than the declared number of archived probe actions are available.

### 5. Is persistence interruption-safe and auditable?

Yes, after review. State and query Zarr groups are written under `.partial__*` keys and moved only when complete. Query summaries and branch JSON use atomic replacement. Resume rejects unexpected IDs, incomplete final groups, changed contracts, ambiguous refresh state, or missing refresh backups. Three narrowly scoped contract amendments preserve the old and new hashes and the reasons for each migration.

One limitation is deliberate: seven early persisted state snapshots predate the full-simulator-field upgrade. They are not branch roots. The active branch payloads contain the current-process certificates that determine admissibility.

### 6. Are branch and query counts exact rather than inferred from a success table?

Yes. The validator requires exactly 160 branch IDs, 160 complete query groups, 80 core queries referenced exactly twice, 80 factor queries, ten complete state groups, one summary per query, no partial groups, and no unexpected IDs. The manifest is complete and retains all three historical runner errors.

### 7. Are statistical groups isolated?

Conditional models leave out entire source episodes, giving eight groups rather than treating ten states or 160 branches as independent. A review found that the native-versus-counter-goal interval still bootstrapped states, including two landmarks from the same task-3 episodes.

Fix: that interval now resamples source-episode clusters. The drawer-source contrast remains `+0.075`, but its 95% interval widens from `[-0.10, +0.25]` to `[-0.25, +0.1875]`. The cabinet-source contrast remains `+1.0`, based on five independent source episodes.

### 8. Is the hidden-state comparison an early-warning test?

Only `Q_preexecution` is strictly pre-execution. The effect-controlled `Q` and `L` models include the realised first-plan state transition, so they are retrospective proposal analyses. The report labels them accordingly.

The feature count is also much larger than the sample count (`3,303` baseline dimensions for 80 proposal rows). Ridge sensitivity at `alpha=10,100,1000` is directionally consistent for `Q`, but this smoke cannot prove the absence of information in the 720-dimensional state. It only finds no incremental value in the predeclared 26-dimensional summary.

### 9. Does the result show that SmolVLA remembers its trajectory?

No. `policy.reset()` is called before every branch, and the inspected checkpoint exposes only an action queue—no observation-history queue. The first query receives the current cameras, robot state, and instruction. Any “trajectory” signal must therefore be visible in current physical state or encode a learned occupancy/manifold prior; it is not direct recurrent episode memory.

The state bank confirms a selection confound:

- all five cabinet-source states are at step 50; the bowl has median height `0.972`, is grasped in 3/5 states, and the drawer is closed;
- drawer-source states contain two step-50 and three step-100 states; the bowl has median height `0.898`, is never grasped, and the drawer is displaced in 4/5 states.

Thus cabinet-to-drawer switching often asks the policy to undo a nearly completed pick/place phase and begin a different multistage skill. Drawer-to-cabinet switching begins from a lower bowl and often-open drawer. The asymmetry is real at those matched states, but physical subgoal preparation and source-policy occupancy are plausible causes.

### 10. Is the language prior strong, weak, or grounded?

None of those one-dimensional descriptions fits the evidence. On cabinet-source states, changing to the drawer instruction reduces cabinet completion from `1.0` to `0.0`, yet drawer completion is also `0.0`. Language can veto the rehearsed continuation without composing the requested alternative. On drawer-source states, source-goal suppression is `0.45` and constructive cabinet transfer is `0.375`.

This **veto–composition gap** is the useful discovery: behavioural sensitivity to language is not equivalent to grounded task competence.

The raw factor named `contradiction` must also be interpreted carefully. It negates the other goal and reaffirms the target, so it is a contrastive-negation control, not an internally inconsistent instruction. No contradiction-outcome claim is supported because Phase 3 factor queries are fixed-forward only.

### 11. Do the hidden states know proposal quality or their own limitations?

Not in the tested summary representation. At the primary ridge setting, adding hidden summaries changes held-source-episode RMSE as follows:

- `V`: `0.386 → 0.389`;
- pre-execution `Q`: `0.444 → 0.452`;
- effect-controlled `Q`: `0.413 → 0.422`, MSE improvement `-0.0076`, cluster-bootstrap interval `[-0.0182, +0.0025]`;
- effect-controlled `L`: `0.200 → 0.199`, improvement indistinguishable from zero.

The negative result matters. A strong latent displacement or decodable rollout failure score does not imply introspective value. Directional latent features, larger exogenous state coverage, and an independent policy are required before concluding that a model knows its own limitation.

### 12. Is Executable–Reflective Subspace Decomposition supported?

The geometry is supported, but “reflective” self-knowledge is not. Controlled language/vision factors place a median `98.2%` of final action-head-input displacement in the output-null subspace; proposal-noise displacement leaves only `21.3%` there and puts `8.2%` in the seven executed rows. Action/expert distances correlate only about `0.32/0.31` with absolute `Q` differences.

This supports separating latent representation from executable effect. It does not show that the null component evaluates the policy, nor that the row-space projection causally mediates success.

### 13. Were canonical artifacts protected?

Yes. The source benchmark under `archive/full_experiment/runs` was read and hash-bound in the Phase 3 contract. No canonical rollout was rerun, renamed, or overwritten. Raw Phase 3 evidence is under ignored `local/`; compact derived tables are under `reports/`. π0.5 and GR00T weights were not downloaded.

## What the implementation now guarantees

- One frozen contract identifies source hashes, checkpoint revision, state/branch/query IDs, goals, factors, tolerances, seeds, horizon, image resolution, and activation sites.
- Every active outcome has current-process archive-reconstruction provenance.
- Every paired continuation has one exact first proposal and one exact first physical effect.
- State, query, summary, branch, refresh, and manifest persistence is resumable without repeating completed work.
- Analysis refuses incomplete, extra, uncertified, or prefix-inconsistent data.
- Source-episode grouping is used for both held-out prediction and resampled uncertainty.
- Retrospective controls, low sample size, raw factor semantics, and unidentifiable cross-policy terms are explicit.

## What remains unproven

- generalisation beyond ten source-selected states or beyond two trained LIBERO tasks;
- causal separation of physical feasibility from training-manifold familiarity;
- literal trajectory retrieval or internal recurrent memory;
- hidden-state mediation of action or success;
- policy-specific competence and self-specificity across architectures;
- factor-by-factor outcome interactions;
- robustness under a controller-normalised action horizon.

## Recommended next methodology: occupancy-balanced CRD

Scaling the current source-trajectory bank to π0.5 and GR00T would reproduce its largest confound. The next state bank should be generated independently of any evaluated policy and organised as an **affordance lattice**:

1. Cross drawer state (`closed`, `part-open`, `open`), bowl status (`table`, `grasped`, `released`), bowl/eef pose, and target relation while keeping the scene valid.
2. Certify physical feasibility with a policy-independent oracle, planner, or successful scripted continuation; otherwise policy failure is inseparable from impossible recovery.
3. For every exact state, evaluate both goals, multiple native proposals, and common continuation schedules. Separately mark whether the state is on-policy, near a training neighbour, or deliberately off-manifold.
4. Estimate `state affordance`, `goal-conditioned policy competence`, `proposal quality`, `continuation noise`, and the interaction `policy × occupancy distance × goal switch`.
5. Test the veto–composition gap at matched subgoal distance. A grounded policy should not merely stop the old skill; it should construct the alternate when the oracle says it is feasible.
6. Only then add π0.5 and GR00T one at a time, keeping native-control and controller-normalised tracks separate. Cross-policy “common difficulty” is meaningful only on this shared, policy-independent state lattice.

For hidden-state work, use the same lattice to screen directional, cross-fitted features. Proceed to causal patching only for a representation that predicts held-out `V` or within-state `L` beyond the full observable/proposed-effect baseline. This avoids spending multi-model compute on a probe that the present smoke has already failed to validate.
