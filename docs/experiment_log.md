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
- Canonical next phase: offline robustness gate—bootstrap intervals, leave-one-suite-out transfer, cross-condition transfer, and richer state/action-history baselines.
- No completed rollout should be repeated when resuming.

## Known limitations

- Failure subtype and true onset are not yet manually annotated.
- Existing activations are pooled; language, main-image, wrist-image, and action token spans are not separated.
- The black wrist mask is a strong out-of-distribution intervention and does not by itself prove normal wrist-view necessity.
- Only one checkpoint and simulator family have been evaluated.
- Probe prediction is not evidence that the decoded representation causally controls action.

## Logging rule

Append a dated entry when any of the following changes: canonical run set, primary metric, scientific interpretation, failure/falsification result, checkpoint, or next active gate. Record exact run IDs and artifact paths. Never rewrite historical entries to make later results appear anticipated.
