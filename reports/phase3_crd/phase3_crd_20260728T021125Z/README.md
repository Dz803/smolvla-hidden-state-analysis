# Phase 3 certified recoverability decomposition

Run: `phase3_crd_20260728T021125Z`

The repaired active ledger contains **160 branches**, **80 exact proposal queries**, and **80 fixed-noise factor queries** from ten certified states. All goals were false at every branch root. The active outcome rate is 0.456 (73/160).

## Main result: asymmetric goal switchability

- Cabinet-trajectory states: cabinet `V=1.000`, drawer `V=0.000`.
- Drawer-trajectory states: drawer `V=0.450`, cabinet `V=0.375`.

Changing the instruction away from cabinet prevents the model from completing cabinet, but it does not create drawer competence. Conversely, drawer-source states retain meaningful access to the simpler cabinet goal.

Descriptively, the instruction switch suppresses the source predicate by `1.000` on cabinet-source states while constructive drawer transfer is `0.000`. On drawer-source states, suppression is `0.450` and constructive cabinet transfer is `0.375`. This **veto–composition gap** is more precise than saying that the language prior is simply strong or weak: language changes the policy, but compositional recovery is constrained by the current physical/occupancy state.

This is **not evidence of recurrent trajectory memory**: the policy is reset before every branch and exposes only an action queue, not an observation-history queue. It is current-state-conditioned recoverability. The source state bank is also progress-confounded: cabinet states are all step 50 with median bowl height `0.972`, `60%` grasped, and a closed top drawer; drawer states mix steps 50/100 with median bowl height `0.898`, none grasped, and the drawer displaced in `80%`. The asymmetry can therefore reflect physical subgoal preparation or occupancy-manifold capture, not language understanding alone. See `state_geometry_audit.csv`.

## Recoverability versus luck

State-goal differences explain 80.8% of branch variance, proposal differences 15.4%, and continuation randomness 3.8%. Only 3/80 matched continuation pairs disagree, while 6/20 state-goal cells vary across first proposals.

## Hidden-state test under complete controls

For proposal quality `Q`, adding hidden summaries after privileged state/geometry, downsampled pixels, trajectory history, target, proposal seed, raw noise, the full ordered action chunk, and its realized first-plan effect changes held-source-episode RMSE from 0.413 to 0.422. The grouped MSE improvement is -0.0076 with bootstrap interval [-0.0182, +0.0025]. For proposal luck `L`, the corresponding RMSE is 0.200 to 0.199.

The `Q` hidden-state improvement is non-positive at every tested ridge alpha (`alpha=10,100,1000`). This smoke therefore provides no evidence that these low-dimensional hidden summaries know proposal quality beyond the plan/effect baseline.

## Executable geometry

The fixed-noise factor table preserves modality-specific VLM, action-expert, flow, and executed/padding/null action-head changes. Median controlled-factor output-null energy is 98.2%; median proposal-noise output-null energy is 21.3%. These are fixed-forward geometry results, not causal hidden-state mediation.

## Critical simulator discovery

MuJoCo's flattened state omits `qacc_warmstart`. Removing only that field reproduces the contact-state divergence; cold cross-instance restore can change state by `0.126` and pixels by `191/255`. One audited success became failure after exact reconstruction. Across 2 provenance-preserving repair transaction(s), 114 legacy payloads were regenerated, 96 semantic payloads changed, and **12 success labels flipped**. Every active branch now carries current-process reconstruction evidence. See `restore_field_ablation.json` and `branch_replay_audits/`.

## Boundary

This is a ten-state SmolVLA smoke study. It cannot estimate policy-specific competence or self-specificity until matched π0.5/GR00T experiments exist. The raw factor name `contradiction` denotes a contrastive instruction that negates the *other* goal and reaffirms the target; it is not an internally inconsistent instruction. Finally, the study does not establish that a decoded hidden feature causally controls behavior, and the negative conditional test does not justify scaling the current summary probe unchanged.
