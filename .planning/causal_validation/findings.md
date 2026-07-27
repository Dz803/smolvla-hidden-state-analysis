# Findings: SmolVLA Hidden-State Causal Analysis

## Requirements
- Explain the completed experiment's discoveries and current inspection target.
- Move all SmolVLA-related work into a standalone folder outside LingBot.
- Broaden the analysis toward language, vision, action, modality distribution, and causal relationships.
- Identify weaknesses, plausible causes, discriminating experiments, and a path toward explanation.
- Preserve reproducibility while avoiding unrelated LingBot files.

## Research Findings
- Existing evidence shows hidden states predict eventual episode failure, but prediction alone does not establish a causal mechanism.
- The detailed pilot used task-held-out probes and fold-local preprocessing, which limits direct task leakage but does not remove progress, visual-state, behavior-policy, or termination confounding.
- VLM representation divergence preceded action-expert and action-output divergence in referenceable pilot failures; this temporal ordering is suggestive but not causal.
- Raw per-step thresholds have high any-step false-alarm rates, so deployment-grade warning requires temporal modelling and calibration.
- The complete experiment tree is 41 GB: 39 GB of immutable runs, 1.3 GB of project-local checkpoints/assets, and about 116 MB of reports.
- SmolVLA-specific material also exists inside LingBot as an 11 GB Conda environment, a 157 MB LeRobot checkout, two small obsolete virtual environments (13 MB and 14 MB), and a dependency log.
- A second Conda environment exists at `/home/zhongzhengyang/miniconda3/envs/smolvla-libero`, but it lacks pandas and is not suitable for the completed offline analysis. The working Python is the standalone project's `local/lingbot-conda-env-archived/bin/python`.
- The experiment tree includes the canonical 400-episode benchmark (19 GB), four 4–4.5 GB perturbation runs, the 3.6 GB clean Spatial pilot, smoke runs, source, tests, reports, and project-local checkpoint/assets.
- The 400-episode benchmark succeeded on 255/400 episodes (63.75%; 95% CI 59.0–68.5%) with strong suite heterogeneity: Goal 83%, Object 67%, Spatial 61%, and LIBERO-10 44%.
- On the 400-episode benchmark, action-expert hidden states are the strongest mid-trajectory warning feature: AUPRC 0.526 at step 0, 0.603 through step 50, 0.844 through step 100, and 0.968 retrospectively over the full trajectory. VLM final states reach 0.442, 0.508, 0.770, and 0.905 respectively. The task-grouped majority baseline is about 0.328.
- Sampled action uncertainty is weak on the broad benchmark (AUPRC 0.352 through step 100 and 0.638 full trajectory), despite slightly higher descriptive uncertainty in failures.
- The paired perturbation study provides valid causal effects for the specific interventions and nine-task subset: wrist masking reduced success by 65.6 percentage points to zero; mild main-camera blur reduced it by 25.6 points; the paraphrase effect was -2.2 points with a confidence interval crossing zero.
- Condition-aware probes partly recognize perturbation identity, especially the wrist mask. High discrimination in that study is not evidence of a condition-invariant latent failure mechanism.
- No human failure onset or subtype labels exist. Current outcomes are success versus timeout, and normalized-progress bins are not physical manipulation phases.
- Existing activations are pooled vectors from selected VLM/action-expert layers. Token boundaries are unavailable, so language-token versus image-token representation claims cannot be made from current artifacts.
- The benchmark already provides episode/task/instruction/seed/initial-state metadata, 89,064 step records, 286,640 probe-prediction rows, 1,305 episode/layer divergence rows, and 143,320 activation-summary rows. These support new offline robustness and confounding analyses without new simulation.
- The benchmark step table retains robot/eef/gripper state, predicted action chunks, executed actions, action dynamics, timing, uncertainty references, and activation references. It does not retain explicit modality-token boundaries or causal intervention assignments.
- The independently installed Miniconda environment at `/home/zhongzhengyang/miniconda3/envs/smolvla-libero` lacks pandas and is not the completed experiment runtime. The archived LingBot-local environment remains functional after relocation and successfully reads all result tables.
- A new within-task confound audit changes the interpretation of step-0 performance. At step 0, task identity explains about 89.5% of action-expert score variance and 80.1% of VLM score variance; within-task pairwise AUROC is only 0.543 and 0.510. The modest step-0 headline AUPRC therefore appears dominated by task/difficulty encoding rather than episode-specific foreknowledge.
- By step 100, within-task pairwise AUROC rises to 0.878 for the action expert and 0.846 for the VLM, while task-attributable score variance falls to 52.5% and 58.9%. This is much stronger evidence that the representations acquire trajectory-specific failure information during interaction.
- Warning-score dynamics are outcome-specific: from step 0 through 100, action-expert risk decreases by 0.074 on average for successes and increases by 0.121 for failures; VLM risk decreases by 0.024 for successes and increases by 0.170 for failures. Action-uncertainty risk rises by essentially the same amount for both outcomes (~0.025), explaining its poor discriminative value.
- Robot-state scores remain overwhelmingly task-structured even at step 100 (96.6% task-attributable variance), whereas hidden-state scores become substantially less task-dominated. This suggests the mid-trajectory signal is not reducible to the current low-dimensional robot state alone.
- Step-100 action-expert discrimination remains strong across suites by AUROC: LIBERO-10 0.810, Goal 0.878, Object 0.973, and Spatial 0.918. VLM AUROC is 0.722, 0.756, 0.910, and 0.871 respectively. This supports broad but non-uniform suite transfer and motivates a stricter leave-one-suite-out test.
- The new audit is deterministic: rerunning it reproduced all three derived CSVs byte-for-byte by SHA-256.

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| Separate descriptive, predictive, mechanistic, and causal claims | Each requires different evidence and prevents overclaiming. |
| Use matched counterfactual interventions and mediation analysis | These directly test modality contribution and the VLM→action pathway. |
| Keep task/episode groups isolated across all fitted transforms | Avoid leakage from correlated trajectory samples. |
| Move the complete 41 GB experiment tree and LingBot-local SmolVLA support directories | This satisfies physical separation while preserving raw evidence. |
| Keep the already-external Miniconda environment in place | It is outside LingBot and can be referenced as a dependency. |
| Prioritize failure annotation and modality-resolved instrumentation before another large rollout | Current biggest limitations are label semantics and pooled-modality ambiguity, not sample count alone. |
| Use activation patching/rescue as the central mechanistic test | Predictive probes and input perturbations alone cannot identify whether a representation is on the causal path to action failure. |
| Reframe step-0 probe results as a task-confounding diagnostic | Within-task ranking is near chance despite above-baseline global AUPRC. |
| Focus the next mechanistic window around steps 50–100 | This is where within-task hidden-state separation emerges and uncertainty fails to track outcome-specific change. |
| Use a two-part immediate gate | Offline cross-condition/held-suite controls should precede a 2-task × 5-seed modality-resolved causal smoke study. |

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| Target scope must distinguish experiment files from shared model/environment assets | Inventory paths and ownership before moving. |
| The standalone public Git repository contains only a 240 KB curated code/report subset | The complete 41 GB evidence tree requires a local standalone research directory; large artifacts remain excluded from GitHub. |
| Existing reproduction commands contain LingBot-relative paths | Standalone documentation and scripts must be updated to use the new project root and external environment path. |
| The presumed external working environment was incomplete | Use the relocated archived environment for immediate offline analysis; document that it may contain old absolute prefixes for future package/tool invocations. |

## Resources
- Complete local experiment: `archive/full_experiment`
- Canonical benchmark: `archive/full_experiment/runs/benchmark_400_20260723T021424Z_e0638bea`
- Working offline Python: `local/lingbot-conda-env-archived/bin/python`
- Existing public repository: `https://github.com/Dz803/smolvla-hidden-state-analysis`
- Existing detailed report: `reports/hidden_state_report.md`
- Causal research programme: `docs/causal_research_program.md`
- Experiment ledger: `docs/experiment_log.md`
- Resume guide: `docs/resume.md`

## Visual/Browser Findings
- None.
