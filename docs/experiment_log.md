# SmolVLA Experiment Ledger

This is the compact, tracked ledger for completed experiments and canonical decisions. Raw manifests, videos, observations, activations, and detailed generated reports remain under `archive/full_experiment` on the workstation.

## Canonical status

| Item | Status | Canonical artifact/result |
|---|---|---|
| Frozen checkpoint | Complete | `lerobot/smolvla_libero`, revision `31d453f7edd78c839a8bbc39744a292686daf0de` |
| Hook equivalence and smoke gates | Passed | Recorded in `archive/full_experiment/reports/final_analysis.md` |
| Spatial pilot | Complete | `pilot_spatial_100_20260722T060211Z_63ee3442`, 100 episodes |
| Paired perturbation study | Complete | 360 episodes across clean, paraphrase, blur, and wrist-mask conditions |
| Official-style benchmark | Complete | `benchmark_400_20260723T021424Z_e0638bea`, 400 episodes, 255 successes |
| Infrastructure failures | None | 0 across canonical evaluation runs |
| Warning confound audit | Complete | `reports/warning_confound_audit/` |
| Offline robustness gate | Complete | `reports/offline_robustness_gate/` |
| Trajectory/scene confound audit | Complete | `reports/trajectory_confound_audit/` |
| Phase 2 exact-forward gate | Complete, corrected | `reports/phase2_forward_gate/phase2_forward_gate_20260727T123138Z/` |
| LIBERO computational-state gate | Passed | `reports/phase2_state_gate/state_contract_20260727T113937Z/` |
| Training state/action retrieval audit | Complete | `reports/trajectory_memorization_audit/phase2_train_retrieval_a1aaacb_v3/` |
| Phase 3 certified CRD smoke | Complete | `reports/phase3_crd/phase3_crd_20260728T021125Z/` |
| Phase 3 engineering/scientific review | Complete | `docs/phase3_engineering_review.md` |
| Historical paired-state provenance | Failed audit | Nominal seed/episode IDs do not guarantee identical initial state after outcome-dependent autoresets |
| Causal hidden-state mechanism | Not established | Active research target |

## Evidence summary

### 2026-07-22 — Pilot and paired interventions

- The clean Spatial pilot produced 100 episodes, 17,698 steps, 58 successes, and 42 timeout failures.
- Pooled VLM and action-expert representations predicted eventual failure, but full-trajectory results were identified as retrospective rather than early warning.
- In the paired nine-task subset, clean success was 65.6%, instruction paraphrase 63.3%, main-camera blur 40.0%, and wrist-camera mask 0%.
- The paired input transformations establish causal effects for those exact interventions and tasks, not a causal neural mechanism.

### 2026-07-23 — Broad benchmark

- Overall benchmark success: 255/400 = 63.75% (episode-bootstrap 95% CI 59.0–68.5%).
- Suite success: Goal 83%, Object 67%, Spatial 61%, LIBERO-10 44%.
- At step 100, action-expert AUPRC was 0.844 and VLM AUPRC was 0.770; action uncertainty was 0.352.
- Full-trajectory action-expert/VLM AUPRC was 0.968/0.905 and remains retrospective.

### 2026-07-23 — Task-confound audit

- At step 0, task identity accounted for approximately 89.5% of action-expert score variance and 80.1% of VLM score variance.
- Step-0 within-task pairwise AUROC was 0.543 for the action expert and 0.510 for the VLM, close to chance.
- By step 100, within-task pairwise AUROC rose to 0.878 and 0.846.
- From step 0 through 100, risk rose in failures and declined in successes for hidden-state probes, while action-uncertainty risk changed nearly identically across outcomes.
- Interpretation: step-0 performance is primarily task/difficulty encoding; a trajectory-specific signal emerges during interaction.

### 2026-07-23 — Separation and research design

- Moved all SmolVLA work out of LingBot into `/home/zhongzhengyang/smolvla-hidden-state-analysis`.
- Preserved 51 GB of complete evidence and runtime material locally; kept it out of Git.
- Added the causal research programme and validation matrix.
- Verified 3 tests and byte-identical reproduction of the confound-audit CSVs.

### 2026-07-27 — Resumability audit

- Added tracked active-plan state, complete progress/findings records, project instructions, resume guide, experiment ledger, and automated resume check.
- Resume infrastructure was introduced in commit `ef81cb6` and verified from the standalone workstation folder.
- Full verification passed: repository synchronization, active-plan discovery, working Python, core imports, canonical benchmark presence, and 3/3 tests.
- Canonical next phase: offline robustness gate—bootstrap intervals, leave-one-suite-out transfer, cross-condition transfer, and richer state/action-history baselines.
- No completed rollout should be repeated when resuming.

### 2026-07-27 — Offline robustness gate

- Completed Phase 1 using only existing immutable runs; no new rollout was launched.
- Added 2,000-draw episode and task-cluster intervals. At step 100, action-expert AUPRC is 0.844 (episode 95% CI 0.785–0.895; task-cluster CI 0.736–0.927), and VLM AUPRC is 0.770 (0.699–0.835; 0.678–0.867).
- The richer five-sample state/action-history baseline reaches 0.517 AUPRC versus 0.844/0.770 for action expert/VLM.
- Leave-one-suite-out macro AUPRC is 0.835 for action expert, 0.785 for VLM, and 0.534 for state/action history.
- Task-held-out action-expert probes transfer in both directions for clean↔blur and clean↔paraphrase. VLM paraphrase transfer is weaker.
- Clean probes assign high failure probability to wrist-mask episodes, but all 90 episodes fail; ranking is undefined and wrist-mask→clean fitting is non-estimable.
- Canonical derived artifacts: `reports/offline_robustness_gate/`. Claim remains robust predictive evidence at step 100, not a causal hidden-state mechanism.
- Next phase: failure-subtype/onset annotation and modality-token instrumentation.

### 2026-07-27 — Execution-path, trajectory-confound, and research-novelty audit

- Inspected the rollout and capture path without modifying canonical evidence. The policy replans every 50 environment steps, while archived activations were collected after each episode by resetting the policy and issuing a fresh stochastic `predict_action_chunk` query on saved observations. They are therefore not the hidden states that generated the executed actions.
- The archived VLM vector averages all 177 prefix tokens; the action-expert vector averages all 50 action positions and ten denoising hook calls. The saved predicted-action field is the shrinking original rollout queue, so it is not aligned with the fresh post-hoc activation's sampled chunk.
- Recomputed task-grouped, nested-regularized probes on exact active risk sets. At step 100 (`n=319`, failure prevalence 0.455), AUPRC was 0.744 for the full ordered 50x7 chunk, 0.803 for robot state plus executed-action prefix plus full chunk, 0.878 for action expert, and 0.843 for VLM. Adding action expert or VLM to behavioural context reached 0.882/0.883. These are predictive comparisons, not a conditional-information hypothesis test.
- Low-resolution current-scene pixels reached 0.532 AUPRC and 0.632 AUROC at step 0, stronger than the hidden-state probes under the same exact-landmark protocol. This elevates visible initial layout/seed difficulty to a primary confound. High-dimensional feature concatenations were unstable under held-out-task evaluation and are not treated as decisive.
- Audited the environment-state contract: all 89,064 stored `object_state` values are null, `goal_state` contains only a normalized-progress fallback, and phase labels are time bins despite `save_environment_state: true`. True geometry, predicates, contacts, and restorable simulator state must be captured before a recoverability experiment.
- Verified from the official model/dataset records that the evaluated checkpoint was trained on the same 40 LIBERO task identities used by the benchmark. Task-held-out probe folds do not make the policy task-OOD. Recent work already covers generic frozen-VLA success/value probing and exposes severe LIBERO shortcut/generalization failures.
- Defined Counterfactual Recoverability Decomposition (CRD), separating state-level policy recoverability `V` from within-state sampled-plan quality `Q`, with common-random-number branches, complete conditional baselines, cross-goal plan faithfulness, and matched bidirectional causal patches.
- Canonical derived artifacts: `reports/trajectory_confound_audit/`; research synthesis: `docs/blind_spot_research_audit.md`. No canonical rollout or immutable artifact was changed.
- The active next phase is exact action-forward measurement alignment. The former annotation-first phase is deferred. Any branched simulation smoke gate requires explicit approval.

### 2026-07-27 — Phase 2 exact-forward, state, and retrieval gates

- Replaced the active runtime's post-episode stochastic re-query with structured capture inside the exact queue-filling action forward. Query IDs, explicit flow seeds, queue age, exact 50x7 chunks, token spans, action positions, ten denoising calls, VLM-to-action KV summaries, and the input to `action_out_proj` are retained.
- Completed 40 fixed-observation queries over four immutable task-3 states at steps 50 and 100. Hooked versus reference chunks and exact-seed repeats have maximum error `0`; VLM and prefix KV statistics are exactly invariant to flow seed. No canonical rollout was rerun.
- Feasible alternate-goal action/VLM displacement gain is `35.0`, versus `12.6` for a paraphrase and `2.40/2.65` for main/wrist mean-image controls. This is direction-selective downstream amplification, not a latent-magnitude effect.
- An explicit contradiction is weakly alternate-goal aligned in the VLM at both landmarks but is routed toward the original goal at step 50 and the alternate goal at step 100. This motivates a scene/trajectory-conditioned semantic-routing hypothesis; the fixed states cannot determine whether that routing is rational or harmful.
- The final expert state is 720-D and the internal action output is 32-D, of which only seven channels execute in LIBERO. Proposal-noise displacement has median `9.08%` executed-row, `60.36%` padding-row, and `31.31%` output-null energy. Controlled factor displacement is `93.8–96.1%` output-null while its small executed projection produces large action effects.
- A real LIBERO replay test rejects MuJoCo-only matching: identical three-action replay differs by `0.037207` in final simulator state and up to `172/255` in camera pixels because controller/gripper/runtime state is omitted. Full runtime restoration yields identical pixels and `7.12e-15` simulator error. The computational-state certificate is mandatory for future branches.
- Retrieved only revision-pinned official `lerobot/libero` state/action Parquet (`a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`), with no video or external model weights. Across 190,508 horizon-50 train windows, the audited proposals have zero exact matches and median same-task nearest RMSE `0.553` versus train self-nearest median `0.449`. Action-only other-task matches are false positives with severe state mismatch.
- Same-step plans across different success/failure episodes are relatively similar (`0.222` standardized RMSE) compared with different-landmark plans (about `0.624`), motivating a phase-locked policy-attractor test rather than a literal trajectory-copying claim.
- Defined Executable–Reflective Subspace Decomposition (ERSD) inside Cross-Policy Recoverability Decomposition (CPRD). The combined method tests common difficulty, policy-specific competence, proposal luck, causal factor interactions, and internal self-specificity in shared environment-effect space.
- Canonical interpretation: `docs/phase2_discoveries_and_ersd.md`. Phase 2 is complete; Phase 3's approximately 160 branched continuations remain pending explicit approval. π0.5 and GR00T weights have not been downloaded.

### 2026-07-27 — Historical paired-state provenance failure

- Phase 3 preflight found that LeRobot's scalar LIBERO wrapper resets itself on success while Gymnasium's same-step vector wrapper resets it again; the experiment loop then explicitly resets before the next episode. Initial-state indices therefore advance according to prior outcomes rather than nominal episode index alone.
- Direct archive comparison confirms the impact. Step-0 policy state is exactly equal for only 60/90 clean–paraphrase, 33/90 clean–blur, and 14/90 clean–wrist nominal pairs.
- The previously reported perturbation success differences remain descriptive condition-level contrasts, but they are not matched-initial-state causal estimates across all 90 episodes. Only an explicitly verified identical-state subset may support paired interpretation.
- Phase 3 will not repair or overwrite historical runs. It uses one captured full-runtime snapshot as the source of every factor/proposal/continuation branch and requires explicit replay/state certificates.

### 2026-07-27 — Phase 2 image-orientation contract failure

- Phase 3 preflight found that canonical observation NPZ cameras were saved before LeRobot's LIBERO environment processor rotates both image axes by 180 degrees.
- `run_phase2_forward_gate.py` rebuilt batches directly from those saved arrays and omitted that environment transform. Its zero hook/repeat error proves deterministic capture for the queried input, but not equivalence to the rollout policy's correctly oriented input.
- Phase 2 semantic-routing and latent-subspace numbers are provisional pending a corrected fixed-forward rerun. No canonical rollout needs to be repeated; the repair is limited to saved-observation preprocessing and derived evidence.
- Corrected report `phase2_forward_gate_20260727T123138Z` passes zero hook/repeat error over 40 queries. The earlier semantic-routing reversal disappears and must not be cited. Wrist-mean replacement now has the largest action effect (`0.958` RMSE), while alternate-goal action/VLM gain remains directionally larger than paraphrase.
- The executable/null finding survives: controlled-factor displacement is `97.85–98.64%` output-null, and proposal-noise displacement remains strongly padding/output-row aligned. This corrected report supersedes `phase2_forward_gate_20260727T114152Z` for scientific interpretation.

### 2026-07-28 — Phase 3 certified recoverability decomposition

- Completed the approved two-task smoke with ten certified states, 80 exact first-proposal queries reused across two common-random-number continuation schedules, 160 active branches, and 80 additional fixed-noise factor queries. No completed canonical rollout was rerun and nothing under `archive/full_experiment/runs` was modified.
- Simulator audit found that flattened MuJoCo state omits `qacc_warmstart`; dropping that field alone reproduces a contact-state divergence. A serialized full snapshot is also unsafe on first restore into a cold simulator (`0.1262` state error; up to `191/255` pixel error).
- Reconstructed every active branch root by current-process replay of immutable archived actions. Two provenance-preserving repair transactions retained 114 old payloads and old/new hashes; 96 semantic payloads and 12 success labels changed. All 160 active branches now carry valid source reconstruction and all 80 paired first-plan effects match exactly.
- Outcome variance decomposes into state-goal `80.8%`, first-proposal `15.4%`, and continuation `3.8%`; only 3/80 continuation pairs disagree. Cabinet-source states reach cabinet/drawer at `1.0/0.0`; drawer-source states reach drawer/cabinet at `0.45/0.375`.
- The apparent language result is a **veto–composition gap**, not proof of trajectory memory. The policy is reset and has no observation-history queue. Cabinet states are all step 50 with a raised/often-grasped bowl and closed drawer; drawer states mix steps 50/100 with a table-height bowl and usually displaced drawer. Physical preparation and source-policy occupancy remain major confounds.
- Adding the predeclared low-dimensional hidden summaries does not improve held-source-episode prediction beyond full controls. Effect-controlled `Q` RMSE changes `0.413→0.422` with MSE improvement `-0.0076` (episode-cluster interval `[-0.0182,+0.0025]`); `L` improvement is indistinguishable from zero. `Q` remains non-positive at ridge alphas 10, 100, and 1000.
- Senior review added a write-time paired-prefix invariant, fail-closed nested certificate comparisons, explicit probe-action length checks, episode-cluster bootstrap for repeated landmarks, state-bank geometry auditing, regularisation sensitivity, and documentation correction of the Phase 2 orientation artifact.
- Canonical compact evidence: `reports/phase3_crd/phase3_crd_20260728T021125Z/`; raw workstation evidence: `local/phase3_crd/phase3_crd_20260728T021125Z/`; review: `docs/phase3_engineering_review.md`.
- Decision: Phase 3 validates the CRD machinery but does not green-light unchanged causal patching or π0.5/GR00T scaling. The next proposed gate is a policy-independent, occupancy-balanced affordance lattice.

### 2026-07-28 — Phase 3b Stage A proposal-coverage gate

- Stage A policy-free construction exposed state-dependent proposal luck before the 32-cell spend. Drawer demonstration episode 526 succeeds from the normalized open/grasped hard root but fails from a normalized closed/grasped root, despite valid construction, recovery, homing, and nonterminal predicates.
- An exhaustive replay scan from the identical closed/grasped baseline found 12/36 successful drawer demonstrations. Retesting those 12 on the hard open/grasped baseline left two shared proposals: episodes 489 and 523.
- Episode 489 is locked for construction revision v21 because it succeeds in both stress roots and reaches the drawer goal earlier than episode 523. The aperture and grasp construction traces remain independent, so this oracle change does not redefine the physical lattice.
- This is policy-independent feasibility evidence, not SmolVLA behaviour. The full Stage A lattice has not yet run; v1–v20 artifacts remain raw diagnostics under `local/phase3b_stage_a/`, and nothing under `archive/full_experiment/runs` was modified.
- The v21 support-bank gate then rejected episode 489 because it fails the source layout-B drawer replay. Episode 523 passes source layouts A/B and both grasped stress roots and is locked in v22; this supersedes the provisional v21 choice without altering construction traces.
- v22 passes two matched stress pairs and a fresh-process audit of the hardest transverse root. The reconstructed state hash is exact, the three-action certificate has zero simulator/camera/robot-observation difference, and both measured support deltas reverse the geometric label. The locked 4/32 run is `local/phase3b_stage_a/phase3b_stage_a_20260728T121202Z/`; full expansion is now authorised under the same contract.
- Full v22 expansion stopped at 14/32 when a closed/grasped/cabinet-side direct transport physically dropped/regrasped the bowl despite reaching its endpoint. v23 replaced direct grasped transport with a reversible three-leg clearance route and passed the failed pair, but its horizontal phase was under-budgeted on the prior hard open/layout-B root.
- v24 locks a 340-action reversible clearance route with planned clearance exceeding intermediate tolerance, a strict final-root tolerance, root timestep 540, and explicit horizon preflight. This is construction/oracle evidence only; all v22/v23 candidates remain diagnostic and will not be mixed into a v24 lattice.
- v24's algebraic inverse route failed because the operational-space controller is not action-reversible; v25 feedback recovery through the physical waypoints passed the hard root without relaxing possession or endpoint tolerances.
- A hard-root drawer scan found only episodes 414 and 526 successful, and v26 showed that they cover complementary members of the same support pair. v27 therefore locks a two-proposal drawer bank and records proposal fallback separately from selected-oracle cost.
- v27 run `local/phase3b_stage_a/phase3b_stage_a_20260728T150543Z` stopped at 1/32: cabinet episode 474 succeeds on the hard near root but fails on the matched transverse root after valid normalization and all 1,073 budgeted actions. This falsifies single-proposal cabinet coverage; the failed run is preserved and Stage A remains in progress pending an exact-root policy-free cabinet scan.
- The exact-root scan `local/phase3b_stage_a/proposal_scans/cabinet_20260729T023255Z` replayed all 46 task-18 demonstrations from one certified normalized snapshot. Only episode 638 succeeds (frame 74); no proposal reaches the wrong goal or terminates early. v28 therefore freezes ordered cabinet bank `[474, 638]` while keeping episode 474 as the independent support-reference trace.
- v28 run `local/phase3b_stage_a/phase3b_stage_a_20260729T023742Z` passes the affected hard pair: near selects drawer/cabinet `414/474`, transverse selects `526/638`, and all state, work-balance, support, and fresh-process exact-reconstruction gates pass. Full Stage A expansion may resume under this contract; the prior v27 run remains immutable diagnostic evidence.
- v28 expansion stopped at 10/32 on a new drawer proposal-coverage failure at `closed/grasped/drawer-side/near/layout A`: both episodes 414 and 526 fail after byte-identical normalization, with no wrong goal or premature termination. Completed v28 cells are preserved; the next diagnostic is a 36-proposal exact-root drawer scan.
- The first exhaustive drawer scan exposed LIBERO's hidden native-horizon termination because the BDDL wrapper overwrites returned `done` with task success. v29 records the union of both termination signals; a fresh 36-proposal scan completes with 9 successes and three clean horizon-limited failures.
- v30 replaces adaptive fallback accumulation with exhaustive locked proposal coverage: all 36 drawer and 46 cabinet demonstrations are replayed from one shared normalized snapshot per state, selection uses a fixed minimum-cost rule, and matched support costs use a proposal successful on both roots. The full v28 run remains diagnostic at 10/32; no artifact is mixed across contracts.
- v30 finds hard-root proposal-basin widths of 2/36 for drawer and 1/46 for cabinet; the matched transverse cabinet basin is disjoint (`{474}` versus `{638}`). v31 retains zero-overlap pairs as scientific evidence, gates minimum feasible costs, and reports shared-proposal diagnostics only where defined.
- v31 hard-pair run `local/phase3b_stage_a/phase3b_stage_a_20260729T033540Z` passes both roots and every physical/work certificate. Drawer coverage changes `2/36→1/36` with one shared proposal; cabinet remains `1/46→1/46` but changes from episode 474 to 638 with empty intersection. v32 adds complete coverage/generalization tables and artifact-integrity hashes before the full 32-state execution; it does not change the passed construction/oracle semantics.
- Full v32 run `local/phase3b_stage_a/phase3b_stage_a_20260729T035043Z` passes all 16 closed-drawer roots, then fails closed at the first open-drawer root because all 36 complete drawer demonstrations fail from the shared already-open normalized state. No wrong goal or early termination occurs. This falsifies phase-agnostic full-trajectory replay as a universal feasibility oracle; a uniformly event-anchored continuation must pass a bounded smoke before a fresh contract can run.

## Known limitations

- Failure subtype and true onset are not yet manually annotated.
- Archived canonical-rollout activations are pooled; the completed Phase 2 fixed-forward captures separate language, main-image, wrist-image, state, action-position, and denoising-time axes.
- The black wrist mask is a strong out-of-distribution intervention and does not by itself prove normal wrist-view necessity.
- Only one checkpoint and simulator family have been evaluated.
- Probe prediction is not evidence that the decoded representation causally controls action.

## Logging rule

Append a dated entry when any of the following changes: canonical run set, primary metric, scientific interpretation, failure/falsification result, checkpoint, or next active gate. Record exact run IDs and artifact paths. Never rewrite historical entries to make later results appear anticipated.
- A bounded episode-382 diagnostic tested a pre-grasp event anchor on the first v32 open-drawer failure. A 200-step bridge reached the reference end-effector pose safely (`6.35 mm`, zero bowl drift, drawer still open, no goals/termination), but the 80-action suffix still failed. This rules out single-pose pre-grasp alignment as a sufficient phase repair; the next smoke moves the anchor to the verified drawer-opening boundary so the full handle-to-bowl approach remains in the proposal.
- The next episode-382 smoke anchored at the actual drawer-threshold crossing (frame 52), but a direct 200-step operational-space bridge stalled `107 mm` from that pose under both 12 mm and 2 mm tolerances. The bridge was safe but failed its endpoint, so the post-opening suffix result is inadmissible. The next bounded test uses the predeclared three-leg clearance route before interpreting suffix feasibility.
- A three-leg clearance bridge resolves the episode-382 reachability issue: all bridge phases pass without moving the bowl or crossing a goal, and the post-opening suffix reaches the drawer goal at original frame 164 while complete replay from the same normalized root fails. A first exhaustive threshold-anchored scan then failed closed at episode 614 because that trace never opens the drawer when replayed in layout A. The local LeRobot schema has no object state or source init ID, so the next scan uses a fixed action-intrinsic 50-frame lead-in before the first gripper-close transition.
- The complete action-intrinsic scan `drawer_phase_20260729T050742Z` restores physical drawer feasibility on the exact v32 failed state: `10/36` fixed phase continuations succeed versus `0/36` complete traces, with all 36 attempts and ten bridge failures retained. Sparse range access to the official source HDF5 was abandoned after its distributed metadata produced no inventory in nine minutes; no full 1.0 GB image-bearing file was downloaded. v33 may use this rule only as a phase-conditioned feasibility certificate, and drawer coverage must be stratified by full versus sliced execution mode.

### 2026-07-31 — Stage A v33 uniform drawer-phase implementation gate

- v33 applies the fixed action-intrinsic pre-grasp continuation and certified clearance bridge to both closed and open drawer roots. This is a deliberate correction to the open-only draft: one uniform drawer execution rule preserves estimability of drawer-aperture effects, while an open/full mode contrast would not.
- Source proposal identities remain the complete revision-pinned 36-demo task-12 bank. Layout-specific anchors, suffix hashes, bridge outcomes, execution-contract hashes, normalization-only provenance, and both logical and physical action costs are now explicit in each ledger. An initial extra full-trace preparation probe was removed before simulation, so no source proposal is executed outside the exhaustive ledger.
- Config and horizon preflight pass (`2,044` required, `2,200` configured), compilation passes, and 22 focused tests pass. No v33 simulator smoke, candidate state, compact report, policy query, or canonical rollout was produced at this checkpoint.

### 2026-07-31 — Stage A v33 smoke falsifies the untouched cabinet oracle

- Raw run `local/phase3b_stage_a/phase3b_stage_a_20260731T025222Z` reconstructs the exact first v32 failure root; its normalized hash `1a197793…e84a` matches the prior complete drawer-phase scan. The layout-A anchor bank hash is `a7dca541…ce02`.
- The run reaches the cabinet gate, where all 46 full task-18 demonstrations fail without wrong-goal events or early termination. The candidate transaction fails with count zero and retains the exhaustive cabinet error ledger. No canonical rollout, policy, or immutable archive artifact was touched.
- Decision: do not expand v33. The next bounded contract applies one action-intrinsic pre-grasp phase adapter to both drawer and cabinet proposals and adds validated proposal-level checkpoints so a later-goal failure cannot erase earlier completed work.

### 2026-07-31 — Stage A v34 all-goal/checkpoint offline gate

- v34 applies one normalization-only, action-derived phase adapter and certified clearance bridge to all 36 drawer and 46 cabinet proposals across both apertures. It adds atomic per-proposal checkpoints bound to root, contract, selection lock, source inventory, and execution-contract hashes, plus a goal-only smoke mode whose completed results are reusable by a later full candidate.
- Complete-bank preflight rejected an exact 50-frame rule because cabinet episode 438 closes at frame 28. The frozen rule now retains up to 50 pre-close frames while leaving one or more prefix actions to construct the anchor; realized leads are explicit and no proposal is excluded based on target outcome.
- Compilation, 23 focused tests, inventory validation, and the `2,044 < 2,200` native-horizon gate pass. No v34 simulator result exists yet.

### 2026-07-31 — Stage A v34 cabinet causal smoke

- Raw run `local/phase3b_stage_a/phase3b_stage_a_20260731T035449Z` completes the 46-proposal cabinet checkpoint for the first v32 open-drawer failure. An external process interruption after 28 attempts was recovered from the identity-bound checkpoint without rerunning those attempts.
- All 46 action-phase bridges pass from normalized state `1a197793…e84a`; exactly episode 474 succeeds (`1/46`) and reaches the cabinet predicate at source frame 85 with no wrong-goal or pre-goal terminal event. The completed oracle hash is `86e95718…45ff`.
- Compared with v33's `0/46` full-replay result at the same normalized state, v34 proves a nonempty but extremely narrow cabinet continuation basin. This clears the bounded feasibility smoke; a separate feedback reachability diagnostic will test whether whole-trajectory compatibility should remain an admission criterion or be reported only as proposal-specific nuisance structure.
- Two raw-only reference-calibrated feedback diagnostics (`feedback_20260731T061110Z`, `feedback_20260731T061506Z`) safely stall before bowl contact at `47.68/47.71 mm` after 70/170 active descent steps. Because the certified v34 episode-474 path succeeds from the exact state, these failures diagnose a naïve Cartesian planner limitation, not physical infeasibility. The exploratory script was removed; future factorized reachability work must certify path planning, grasp acquisition, and placement as separate edges.
- The same v34 run then completes the 36-drawer bank (`10/36`, selected episode 694), reuses the complete cabinet checkpoint (`1/46`, episode 474), and promotes the candidate with state hash `e35d6683…d80d`. Root validation, support measurement, and the full-runtime certificate pass; no evaluated policy was loaded.
- To respect the no-rerun rule, the final physical lattice will consolidate 16 v32 closed states, the two v31 open hard-pair states, and 14 v34 open completions. Their construction contracts are identical apart from post-root oracle horizon. Support pairs never cross a source; proposal coverage will be stratified by full-replay versus action-phase mode rather than analysed as one aperture factorial.

### 2026-07-31 — Stage A layout-replicate failure and alignment diagnostic

- Correction to the preceding provisional consolidation statement: v31/v32/v34 candidate outputs share the root timestep, construction action identities, and common physical/certificate fields, but the historical contracts do not embed the full construction configuration and have different revision/config/source hashes. The mixed-source artifact can be an exact observed-root bank; it cannot identify a drawer-aperture effect because aperture is aliased with source revision.
- v34's matched layout-B root completes its drawer ledger with `18/36` successes, then exhausts all 46 cabinet proposals with zero successes. The contiguous negative checkpoint is retained and will not be rerun. The v34 expansion stops at one promoted candidate rather than the provisionally planned 14.
- Bounded raw diagnostic `layout_alignment_20260731T070614Z` reconstructs and certifies both v34 roots. On the same layout-B root and episode-474 suffix, the world-frame anchor produces a stable grasp but no goal after 102 actions; a `25.24 mm` bowl-relative anchor registration reaches the cabinet at frame 80. Reusing the registered grasp snapshot, an independent lift/transit/descent/release branch preserves possession and also succeeds.
- This causally identifies controller/proposal alignment for one exact root, not a hidden-state mechanism or universal reachability oracle. It motivates a factorized competence ladder—bridge reachability, contact acquisition, grasp stability, transport, placement/release, and proposal compatibility—and a frozen additive certificate on untouched states. Compact Git-safe evidence is `reports/phase3b_stage_a/layout_alignment_20260731T070614Z/`; raw traces remain under ignored `local/`.
