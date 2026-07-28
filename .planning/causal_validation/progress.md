# Progress Log: SmolVLA Hidden-State Causal Analysis

## Session: 2026-07-23

### Phase 1: Inventory and evidence audit
- **Status:** complete
- **Started:** 2026-07-23
- Actions taken:
  - Interpreted the request as both a filesystem separation and an expansion from predictive analysis to causal research.
  - Initialized a persistent scoped plan.
  - Recorded known limitations of the completed hidden-state experiment.
  - Inventoried all SmolVLA-named paths under the user home directory.
  - Measured the main experiment tree (41 GB), LingBot-local environment (11 GB), LeRobot checkout (157 MB), and two obsolete small virtual environments.
  - Confirmed that a usable Miniconda environment already resides outside LingBot.
  - Audited the canonical final report, including benchmark, paired perturbation, hidden-state, uncertainty, and limitation sections.
  - Identified the central evidence gap: the experiment predicts eventual timeout from pooled representations but cannot yet localize modality-specific or causal mechanisms.
- Files created/modified:
  - `.planning/.active_plan`
  - `.planning/smolvla_causal_analysis/task_plan.md`
  - `.planning/smolvla_causal_analysis/findings.md`
  - `.planning/smolvla_causal_analysis/progress.md`

### Phase 2: Safe repository separation
- **Status:** complete
- Actions taken:
  - Created `/home/zhongzhengyang/smolvla-hidden-state-analysis` from the tested standalone Git repository.
  - Atomically moved the 41 GB full experiment tree, LingBot-local Conda environment, legacy virtual environments, LeRobot checkout, dependency log, and setup planning records outside LingBot.
  - Preserved the independently installed Miniconda environment in place as the working runtime.
- Files created/modified:
  - `/home/zhongzhengyang/smolvla-hidden-state-analysis/archive/full_experiment`
  - `/home/zhongzhengyang/smolvla-hidden-state-analysis/local`
  - `/home/zhongzhengyang/smolvla-hidden-state-analysis/vendor`
  - `/home/zhongzhengyang/smolvla-hidden-state-analysis/logs`
  - `/home/zhongzhengyang/smolvla-hidden-state-analysis/planning/initial_setup`

### Phase 3: Research question expansion
- **Status:** complete
- Actions taken:
  - Implemented and ran an offline warning-confound audit on all 400 benchmark episodes.
  - Quantified task-attributable score variance, within-task pairwise discrimination, task-centered metrics, per-suite metrics, and step-0-to-step-100 score growth.
  - Discovered that step-0 hidden-state performance is mostly task identity, while strong episode-specific separation emerges by step 100.
  - Defined modality-specific questions, a causal graph, seven competing explanations, controls, falsification criteria, and a claim ladder from descriptive association to general causal mechanism.
- Files created/modified:
  - `/tmp/audit_warning_confounds.py` (candidate repository script)
  - `/tmp/smolvla_warning_audit/*.csv` (candidate derived tables)

### Phase 4: Implement research design artifacts
- **Status:** complete
- Actions taken:
  - Added a detailed causal research programme and staged experiment design.
  - Added a machine-readable causal validation matrix.
  - Added the reproducible warning-confound audit and its derived tables.
  - Updated the README and package name for the standalone project.
- Files created/modified:
  - `docs/causal_research_program.md`
  - `configs/causal_validation_matrix.yaml`
  - `scripts/audit_warning_confounds.py`
  - `reports/warning_confound_audit/*`
  - `README.md`, `.gitignore`, `pyproject.toml`

### Phase 5: Verification and delivery
- **Status:** complete
- Actions taken:
  - Committed the causal-analysis documentation, configuration, code, and derived results as `4910eff`.
  - Pushed the commit to `Dz803/smolvla-hidden-state-analysis` on GitHub.
  - Fast-forwarded the 51 GB standalone workstation copy to the same commit.
  - Re-ran tests from the separated path and verified that LingBot contains no SmolVLA material except this temporary active planning record.
- Files created/modified:
  - `/home/zhongzhengyang/smolvla-hidden-state-analysis`
  - GitHub `main` at commit `4910eff9b5755ba7deb0076987f46870b3644f2e`

## Test Results
| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| Planning initialization | Scoped planning files | New active plan | Created | Pass |
| SmolVLA inventory | Named paths and disk sizes | Complete ownership map | 41 GB experiment plus LingBot-local support paths identified | Pass |
| Atomic relocation | Seven exact LingBot-local paths | All absent from LingBot and present in standalone root | Verified | Pass |
| Warning confound audit | 400-episode benchmark probe predictions | Separate task versus episode-specific signal | Step-0 near-chance within task; strong hidden-state separation by step 100 | Pass |
| Standalone unit tests | `PYTHONPATH=src ... python -m pytest -q` | Existing tests pass after separation | 3 passed | Pass |
| Audit reproducibility | Rerun and SHA-256 compare three CSVs | Byte-identical outputs | All hashes matched | Pass |
| Final standalone tests | Separated path with explicit `PYTHONPATH` | Tests run without LingBot files | 3 passed in 1.66s | Pass |
| Git synchronization | Compare `HEAD` and `origin/main` | Identical commit | Both `4910eff9...` | Pass |
| LingBot residue scan | Find `*smolvla*` to depth 5 | Only active plan remains before archival | Exactly one planning path | Pass |

## Error Log
| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
| 2026-07-23 | `/home/zhongzhengyang/miniconda3/envs/smolvla-libero` cannot import pandas | 1 | Switched to relocated `local/lingbot-conda-env-archived`, which loaded all Parquet artifacts successfully |
| 2026-07-23 | `git clone --local` failed with cross-device hard-link error | 1 | Use `git clone --no-hardlinks` for the small standalone repository |
| 2026-07-23 | Git commit could not infer author identity in the fresh clone | 1 | Configure repository-local `zhongzhengyang` no-reply identity before retrying |
| 2026-07-23 | Push targeted checked-out local repository and was refused | 1 | Point temporary clone `origin` to `https://github.com/Dz803/smolvla-hidden-state-analysis.git` |

## 5-Question Reboot Check
| Question | Answer |
|----------|--------|
| Where am I? | Complete |
| Where am I going? | User handoff; next execution is the two-part causal gate |
| What's the goal? | Standalone project and explanatory causal programme |
| What have I learned? | Step-0 is mostly task confounding; trajectory-specific hidden-state signal emerges by step 100 |
| What have I done? | Relocated 51 GB, added and verified causal audit/design, synchronized GitHub |

## Session: 2026-07-27 — Resumability hardening

- **Status:** complete
- Audited the README, local planning records, Git state, and planning-tool discovery from a fresh-project perspective.
- Found that the completed plan existed only in ignored `planning/causal_analysis`, so a new session or another device could not automatically discover it.
- Added tracked `.planning/.active_plan`, a forward-looking causal-validation task plan, durable findings, and the complete progress history.
- Added `AGENTS.md`, `docs/resume.md`, `docs/experiment_log.md`, and `scripts/resume_check.sh`.
- Updated obsolete LingBot-relative resource paths and documented the correct offline Python and canonical benchmark paths.
- Verified `scripts/resume_check.sh --full` in a code-only checkout: all tracked resume files were present and local-only assets were correctly reported as informational rather than failures.
- Verified shell syntax and Git whitespace checks.
- Fast-forwarded the workstation project to resume-infrastructure commit `ef81cb6` and confirmed `HEAD == origin/main`.
- Ran the full resume check from `/home/zhongzhengyang/smolvla-hidden-state-analysis`: active-plan discovery, core imports, canonical benchmark presence, and all 3 tests passed (`resume_check=PASS`).
- Resume point: Phase 1 of `.planning/causal_validation/task_plan.md`; do not rerun completed canonical rollouts.

## Session: 2026-07-27 — Phase 1 offline robustness gate

- **Status:** complete
- Ran `bash scripts/resume_check.sh --full`; active-plan discovery, archived Python, imports, canonical benchmark, and 3/3 pre-existing tests passed.
- Extended `scripts/audit_warning_confounds.py` with deterministic episode and whole-task cluster intervals, suite-held-out probes, task-held-out cross-condition probes, explicit non-estimability reporting, and a five-sample state/action-history baseline.
- Added reusable transfer/history utilities in `src/smolvla_analysis/robustness.py` and four focused tests in `tests/test_robustness.py`.
- Verified a 10-draw end-to-end smoke analysis in `/tmp/smolvla_phase1_smoke` before starting the canonical derived run.
- Interrupted one over-broad 2,000-draw pass before it wrote outputs because it was resampling every exploratory layer probe. Restricted intervals to the six predeclared Phase 1 features while retaining all exploratory point estimates; reran successfully with 2,000 draws.
- Completed the offline gate using the immutable 400-episode benchmark and the four existing 90-episode clean/blur/paraphrase/wrist-mask runs. No rollout or archived artifact was modified.
- Derived output contract passed: 51 confound rows, 72 held-out-suite rows, 108 cross-condition rows, no duplicate evaluation keys, complete primary-feature intervals, and explicit zero coverage for wrist-mask-to-clean fitting.
- Scientific decision: the step-100 hidden-state advantage survives bootstrap, held-out-suite, and richer-history controls. Action-expert transfer survives all estimable clean↔perturbation directions; VLM paraphrase transfer is weaker; wrist-mask-to-clean is not estimable because the source has no successful episodes.
- Created `reports/offline_robustness_gate/` with five CSV tables and an interpretation README.
- Final focused/full tests: 4/4 and 7/7 passed; `git diff --check` passed.
- Error recorded: the first inline schema-inspection command used literal escaped newlines and raised `SyntaxError`; replaced it with a newline-free expression.
- Error recorded: the first canonical bootstrap scope was computationally excessive; stopped it before artifact creation and narrowed bootstrap uncertainty to the predeclared features without changing the 2,000-draw contract.
- Resume point: Phase 2, failure semantics and token instrumentation.

## Session: 2026-07-27 — Broad blind-spot audit

- **Status:** complete
- Re-ran the full resume check; 7/7 tests passed and canonical evidence remained available.
- Began a research-design stress test before Phase 2 in response to the request for a broader, model-agnostic research direction.
- Audited SmolVLA execution and activation-capture code. Confirmed 50-step queued open-loop execution and post-episode activation recomputation at every fifth saved observation.
- Confirmed that post-hoc activation recomputation draws fresh flow-matching noise and averages action-expert activations across action tokens and denoising invocations.
- Preliminary implication: archived activations are useful offline diagnostic representations but are not the exact online hidden states that generated the recorded actions. The existing policy-output baseline is also misaligned because it summarizes the original shrinking action queue, not the fresh action chunk associated with the archived activation.
- Next: quantify replanning/termination confounds and review primary papers/official implementations before revising the experimental programme.
- Quantified the action-queue cadence from 89,064 step records: inference latency is about 120.6× higher at multiples of 50, confirming true online replanning only at those boundaries.
- Quantified fixed-horizon survivor bias: 81 successes are already complete by step 100, while all 145 failures remain active. Recomputed exact-step landmark metrics on the 319 active episodes; hidden-state ranking remains strong, so the flaw changes but does not erase the result.
- Added the previously missing ordered-action baseline using complete 50×7 chunks at replanning boundaries. Its step-100 AUPRC is 0.791 and within-task AUROC is 0.805, closing most of the gap to the action expert (0.851/0.841).
- Began the external primary-source review. Initial relevant work includes official SmolVLA architecture documentation, LIBERO-CF counterfactual language-following evaluation, and ICBench contradiction tests; these motivate symmetric tests for vision and language shortcuts.
- Expanded the review to AC-VLA, LIBERO-Para, and ProbeAct. Their reported trajectory-overfitting, wrist-view shortcut, lexical sensitivity, and explicit kinematic failure-detection results directly motivate stronger action-plan, compositional-language, and semantic-decoding controls.
- Added probe-validity and causal-method checks from control-task selectivity, conditional probing beyond a full ordered action-plan baseline, causal abstraction/DAS/interchange interventions, and exact verification after approximate patch-site screening.
- Search/tool error: a broad survival/landmark-analysis query produced truncated, unusable output. It will be replaced by narrower primary-source searches and is not being treated as evidence.
- Editing errors: two append attempts used stale or cross-file context and did not apply. Re-read the exact section tails and applied a targeted patch; neither failed attempt changed a file.
- Reviewed official benchmark/code evidence for LIBERO-PRO and recent open-loop VLA correction methods. Added a central distinction: probe task holdout does not imply policy OOD generalization, while LIBERO-PRO directly exposes memorized trajectory/layout behavior through object, position, language, task, and environment shifts.
- Identified the 50-step action queue as both a mechanistic confound and an intervention variable: compare fixed horizons or event-triggered replanning before attributing warning signal to semantic understanding.
- Inspected the feature-construction code and confirmed that the historical policy-output control discards the complete temporal order of the action chunk.
- Tooling issue: `rg` is unavailable in the current shell. Switched to read-only `find`/`grep`; a piped `sort | head` inspection then emitted a harmless broken-pipe warning when `head` closed after its requested lines. No files changed.
- Confirmed the immutable benchmark exposes 89,064 step rows with complete executed actions and queued action chunks, making a stronger offline sequence/template control feasible without rerunning rollouts.
- Audited archived semantic state: `object_state` is null for every step, `goal_state` only records the normalized-progress fallback, and all phase labels are time-bin aliases rather than physical manipulation events. This elevates privileged event instrumentation to a prerequisite for semantic mechanism claims.
- Compared the resolved recording contract with artifacts: `save_environment_state` was enabled but the environment/object state is absent. Flagged this as an instrumentation bug to repair before causal branching or phase semantics.
- Checked local training-data availability. No accessible demonstration corpus is present under `checkpoints/libero_datasets`, so direct training-trajectory nearest-neighbor memorization must be deferred or use an external dataset snapshot; evaluation-trajectory clustering alone cannot prove training memorization.
- Predeclared the reproducible trajectory-confound comparison: exact steps 0/50/100, active episodes only, common task-held-out folds, ordered action prefix/current chunk/joint controls, and blockwise fold-local conditional probes with discrimination and calibration metrics.
- Added `scripts/audit_trajectory_confound.py` and two focused tests. The audit excludes the landmark action from history, preserves full chunk order, applies scaling/PCA within each task-held-out fold and feature block, and reports prevalence-normalized AUPRC, AUROC, within-task AUROC, Brier score, and log loss.
- Focused trajectory-audit tests passed (2/2).
- Error: the first canonical trajectory-audit pass stopped before output because nested Parquet action chunks arrived as object-typed NumPy arrays and direct float conversion failed. Reused the project's established object-array-to-list normalization and added a regression test before rerunning.
- Editing error: the first normalization patch omitted an explicit file boundary before the test edit and did not apply. Reapplied it with exact file targets; the failed attempt changed no file.
- Reran the canonical derived trajectory-confound audit successfully without modifying the archive. Wrote metrics and predictions to `reports/trajectory_confound_audit/`.
- Exact step-100 results on 319 active episodes (five task-grouped folds): behavioral context 0.758 AUPRC / 0.804 AUROC / 0.803 within-task AUROC; action expert 0.866/0.865/0.839; VLM 0.882/0.883/0.813. The ordered action prefix alone is 0.558/0.620/0.653 and the current ordered chunk alone is 0.689/0.723/0.771.
- The blockwise concatenated probes exposed calibration/regularization instability (log losses above 2 for some step-100 combinations). Marked those conditional proper-score comparisons non-conclusive pending nested task-grouped regularization rather than interpreting degradation as negative information.
- Upgraded every trajectory-confound model to choose logistic regularization from five values using nested task-grouped folds. Scaling and blockwise PCA are refitted inside each inner fold, and the selected value is stored per outer-fold prediction. Focused tests still pass (3/3).
- Completed the nested canonical rerun and replaced the preliminary derived outputs. At step 100, behavioral context reaches 0.803 AUPRC / 0.821 AUROC / 0.803 within-task AUROC / 0.532 log loss. Adding action expert yields 0.882/0.891/0.843/0.435; adding VLM yields 0.883/0.891/0.863/0.434.
- Decision: full behavior and plan explain much of the warning signal, but a measurable hidden-state increment remains. Because no visual or privileged scene-state baseline exists, the increment is evidence for current-observation information—not yet for semantic scene understanding.
- Verified that all episode records reference saved observations and videos. A sampled NPZ contains per-step main/wrist RGB arrays shaped `[T, 3, 256, 256]` plus robot/policy/action arrays, enabling an offline low-resolution visual-progress control without simulation.
- Inspection warning: another `find | head` listing emitted expected broken-pipe messages when `head` closed early; no data or process failed.
- Verified runtime alignment: both cameras are captured from observation `o_t` before policy inference/action `a_t`. Also confirmed `object_state` is unconditionally written as `None`; the missing environment state is a recorder implementation gap rather than an artifact-reading error.
- Inspection error: attempted to read a non-existent `src/smolvla_analysis/storage.py` after the relevant runtime excerpt. The runtime evidence was complete; no file changed.
- Extended the trajectory audit with exact current-frame and initial-to-current visual-change controls using deterministic 16×16 RGB block averages for both cameras, plus visual+behavioral conditional models. Added a pooling regression test.
- Completed the visual-control audit. Step-0 current pixels reach 0.532 AUPRC and 0.607 within-task AUROC, exposing initial layout/seed difficulty. Step-100 current pixels reach 0.721 AUPRC / 0.774 AUROC / 0.731 within-task AUROC; pixel change alone is weaker (0.512/0.612/0.699).
- The visual+behavioral concatenated probes were unstable under held-out-task shift and are not being treated as a decisive conditional test. Standalone pixel results are retained; the next semantic control should use privileged geometry or a frozen representation with block-specific capacity control.
- Numerical warning: PCA reported undefined explained-variance ratios for the all-zero step-0 pixel-change block. Predictions remained finite and exactly at the constant baseline; constant-block handling will be made explicit before final verification.
- Added explicit constant-column handling before PCA and a warning regression test; focused trajectory-audit tests pass (5/5).
- Verified from official Hugging Face sources that the evaluated checkpoint was trained on `lerobot/libero`, whose single training split covers task indices 0–39 and about 1,690 episodes. The current 40-task benchmark is policy in-distribution by task identity even when the downstream probe uses task-held-out folds.
- Reviewed the official LIBERO evaluation contract and the 2026 manipulation-benchmark audit. Recorded that evaluation draws from fixed indexed initial states and that a tiny no-language policy can nearly saturate LIBERO, so the project must distinguish benchmark shortcut performance from grounded/general manipulation capability.
- Reviewed primary language-grounding work: Drop-Then-Recovery (language-path redundancy), Anchor-Align (fine-tuning representation drift/language-action misalignment), ICBench and LIBERO-CF (visual dominance under contradictions), and step-wise multilingual sensitivity. Reframed the language question as conditional causal influence at decision events rather than a single strong-versus-weak prior.
- Reviewed direct neighboring work on frozen-VLA value probes, Hide-and-Seek weakly supervised monitoring, SAFE, ProbeAct, and event-grounded sparse autoencoders. Decided that generic hidden-state failure prediction is no longer a competitive novelty claim.
- Refined the candidate contribution to causal factorization of semantic state, sampled action plan, generic progress, and counterfactual recoverability, using restored-state branching and exact online activations.
- Source-access error: the NeurIPS SAFE PDF exceeded the browser fetch limit. Used its official proceedings search result/abstract for high-level scope only; no unsupported implementation detail was inferred.
- Read the full HTML methods/results for the closest value-probing paper and neighboring event-grounded/weak-label methods. Confirmed direct SmolVLA probing, goal-swap, and simulator snapshot overlap, but also a clear unresolved boundary: single-trajectory outcome targets are not Bellman-consistent and no method decomposes recoverability from plan/noise/modality content.
- Named the candidate research method **Counterfactual Recoverability Decomposition (CRD)** and began specifying its factorial interventions and estimand.
- Examined goal-swap directionality and closed-loop SAE intervention results. Added stricter causal criteria: correct counterfactual direction, matched rescue and induction, dose response, and preserved controller fidelity; generic score movement or destructive ablation does not qualify.
- Re-read the existing causal research programme and resume guide against the new evidence. Determined that the programme requires a substantive phase reorder and resume-document update before Phase 2 begins.
- Selected a concrete shared-scene language/goal smoke pair: LIBERO-Goal tasks 3 and 4 (same bowl; drawer-inside versus cabinet-top), with complementary observed difficulty. Verified that training actions/metadata can likely be audited from the 1.94 GB official dataset without downloading video, but did not download anything.
- Wrote the complete synthesis and falsifiable method specification in `docs/blind_spot_research_audit.md`, including the CRD estimands, hierarchical branch design, cross-goal plan-faithfulness metric, competing-hypothesis signatures, stop conditions, and cost-gated sequence.
- Replaced the obsolete annotation-first continuation in the active task plan with Phase 2 measurement alignment, Phase 3 two-task branched CRD, Phase 4 matched causal intervention, and Phase 5 actual OOD/architecture generalization. Updated `docs/resume.md`, `README.md`, the causal-programme banner, durable findings, and the experiment ledger accordingly.
- Regenerated `reports/trajectory_confound_audit/` after explicit constant-block handling. All headline values reproduced exactly and the previous all-zero PCA warning disappeared.
- Verification error: invoking the audit script directly without `PYTHONPATH=src` raised `ModuleNotFoundError: smolvla_analysis`. Re-ran with the documented project path; no artifact was changed by the failed help command.
- Final verification passed: 12/12 full tests, `bash scripts/resume_check.sh --full`, and the canonical benchmark availability check. No rollout was launched and no file under `archive/full_experiment/runs` was modified.
- Resume point: Phase 2 measurement-alignment and forward-only gate. Gate A uses saved observations and exact fixed-noise forwards; Gate B's approximately 160 branched continuations require explicit approval.

## Session: 2026-07-27 — Cross-architecture overlook-discovery methodology

- **Status:** complete
- Began a primary-source boundary review for a methodology that is not tied to SmolVLA and can later include π0.5 and GR00T.
- Verified that official `openpi` provides π0.5 flow-matching base/LIBERO checkpoints and PyTorch/JAX paths, while official Isaac-GR00T now offers N1.7 and LIBERO checkpoints in addition to the scientifically distinct N1.5 design.
- Identified the first design constraint: architecture comparisons require an environment-level causal interface and a controlled-versus-ecological two-track evaluation, because checkpoint training mixtures and embodiments otherwise dominate any hidden-state comparison.
- Mapped nearby July 2026 work. SVA already distills simulator search into a Q evaluator for frozen VLAs; Reflective VLA already conditions on observation–action–consequence triplets; deployment-time reliability work already covers progress monitors, training influence, and sequence success estimation.
- Narrowed the candidate gap to cross-policy counterfactual competence decomposition and interaction-based overlook discovery, rather than another per-policy failure probe or candidate-action selector.
- Reviewed broader competence and policy-comparison work. Aggregate sequential policy comparison, skill competence estimation, and per-policy foresight values already exist; none found in this search used identical restored manipulation states across several VLA families to separate shared hardness from model-specific recoverability.
- Audited recent diagnostic/failure-discovery benchmarks. RADAR, ForesightSafety-VLA, LIBERO shift suites, and ROBOGATE already cover broad factor taxonomies and adaptive boundary sampling.
- Refined the target from “discover failure regions” to “discover and causally validate policy-by-factor and factor-by-factor interactions at identical physical states.”
- Verified official LIBERO evaluation paths for π0.5 and GR00T N1.7. Both standard checkpoints are near 97% mean success, while native action horizons differ (10 versus up to 16, compared with SmolVLA's 50).
- Added action horizon/replanning cadence as a predeclared causal factor and separated native-system from controller-normalized comparison tracks.
- Found a direct headline collision: *How VLAs Fail Differently* already reports architecture-specific black-box action failure signatures. Restricted the intended contribution to matched-state causal decomposition and interaction confirmation.
- Added executable-policy certificates after reviewing action-metadata failure evidence, and separated native policy stochasticity from externally injected candidate perturbations after reviewing BOKBO.
- Searched candidate terminology and adjacent same-state audit methods. No exact “Cross-Policy Recoverability Decomposition” collision was found, but a recent locomotion adapter audit makes same-state headroom an insufficient novelty claim.
- Considered an item-response formulation for shared state difficulty and policy ability. Rejected unqualified IRT inference because three evaluated policies are a small, clustered system sample; retained paired contrasts and a simulation-calibrated hierarchical binomial model.
- Specified Cross-Policy Recoverability Decomposition (CPRD): a restored-state policy×factor×proposal×continuation outcome design, paired statewise contrasts, a cross-classified binomial decomposition, and model-local self-specificity tests.
- Added an architecture-neutral effect-space comparison so raw action chunks, normalization, and native horizons are not treated as equivalent across SmolVLA, π0.5, and GR00T.
- Defined a sequential overlook-discovery loop using cost-aware fractional screening, policy disagreement and posterior residual acquisition, typed counterfactual interventions, and independent held-out confirmation.
- Defined ecological versus controlled post-training tracks and a version-pinned baseline recommendation. No checkpoint, dataset, rollout, or immutable artifact was downloaded or modified.
- Kept the active Phase 2 unchanged pending user approval; the CPRD extension is recorded as a candidate methodological decision rather than silently replacing the execution plan.

## Session: 2026-07-27 — Phase 2 measurement-alignment implementation

- **Status:** in progress
- User approved proceeding with Phase 2 and requested latent/modality, VLM-to-flow, trajectory-memorisation, and cross-policy self-knowledge investigations.
- Scope is limited to the existing SmolVLA checkpoint and small deterministic/offline checks. No canonical rollout will be repeated, no external checkpoint will be downloaded, and `archive/full_experiment/runs` remains immutable.
- Success gates: exact action-producing-forward capture; modality/action-position/denoising-step preservation; restorable privileged simulator state; fixed-noise counterfactual queries; and hook-fidelity/within-state-noise validation.
- Tooling error: `rg` is unavailable on this workstation (`rg: command not found`). Continuing with `find`, `grep`, and targeted module inspection rather than retrying the same command.
- Inspection error: the repository has no `experiments/` directory despite it being included in a broad file-list command. The relevant runtime is under `src/`, `scripts/`, and the archived local environment; no file or experiment was affected.
- Planning update error: the first patch used insufficiently specific context for a repeated `**Status:** pending` line and did not apply. The retry targets the Phase 2 heading explicitly.
- Findings-update error: an attempted contextual append targeted wording that exists in `progress.md` rather than `findings.md`; no file changed. Re-read the file tail and appended the instrumentation audit at its actual final section.
- Inspected `activation_hooks.py`, `runtime.py`, the schema/recorder, configuration validation, and existing hook-equivalence test. Confirmed the exact post-hoc re-query, over-pooling, missing query metadata, and missing simulator-state mechanisms that Phase 2 must replace.
- Audited the vendored SmolVLA inference implementation and LIBERO wrapper APIs. Identified the exact one-prefix/ten-denoising invocation structure, deterministic prefix token ordering, and available flattened MuJoCo state restore primitives.
- Runtime-resolution errors: the project root has no `checkpoints/` or `.conda-envs/smolvla-libero/` directory, and the archived offline Python cannot import LeRobot. The checkpoint and processor assets are actually under `archive/full_experiment/checkpoints/`; the policy-capable environment must be resolved separately before executing Phase 2 GPU diagnostics.
- Located `/home/zhongzhengyang/miniconda3/envs/smolvla-libero/bin/python`, but it cannot import Torch in its current state. This confirms the previously documented environment is incomplete, now for policy inference as well as pandas; continuing to search for the original policy-capable runtime instead of installing or downloading dependencies without approval.
- Confirmed the local checkpoint contract and weight presence without loading the model: 50 action positions, ten denoising steps, cached cross-attention, two image spans followed by language and state, and a 907 MB local safetensors file.
- Environment audit: the minimal `smolvla-libero` Conda environment contains only packaging tools. The `lingbotvla` environment imports Torch and LeRobot but not LIBERO, while the relocated archived environment contains Torch and LIBERO but not LeRobot. The next non-mutating attempt is to combine the archived runtime with the vendored LeRobot source through `PYTHONPATH`.
- Confirmed the combined archived runtime plus vendored LeRobot source imports Torch, LeRobot, and LIBERO successfully. CUDA and `nvidia-smi` are unavailable inside the current sandbox, so implementation and CPU unit tests can proceed before requesting a narrowly scoped GPU diagnostic.
- Added `src/smolvla_analysis/phase2_capture.py`. It captures the exact queue-filling `select_action` forward under explicit fixed noise, intercepts the exact chunk, preserves prefix/action token axes and ten denoising calls, records `x_t`/velocity, and summarizes the actual KV cache passed from VLM to action expert.
- Added two focused toy-policy tests covering exact chunk alignment, token spans, flow-step structure, KV-cache summaries, deterministic noise, and the empty-queue guard. Result: 2/2 passed.
- Added `src/smolvla_analysis/libero_state.py` with duck-typed LIBERO state capture/restoration, object geometry/joints, goal-predicate truth, raw contacts, robosuite grasp status, and a strict round-trip validator.
- Added a fake-LIBERO round-trip regression test. Combined Phase 2 focused tests pass: 3/3.
- Added immutable Zarr writers for structured action queries and simulator snapshots. They preserve the explicit invocation/batch/token-or-action/hidden axes, denoising trajectories, KV-cache token norms, exact noise/chunks, and semantic snapshot metadata.
- Added storage regression tests; combined Phase 2 focused tests pass: 5/5.
- Runtime-integration patch error: the first large patch used overlapping inspection output as context and did not apply. Split the change into small exact patches; the failed attempt changed no file.
- Updated the future rollout schema and runtime: every queue-filling forward now receives a stable explicit noise seed, captures the exact action-producing tensors/chunk, and links all queued actions by query ID and action index. The old after-episode stochastic activation re-query has been removed from the active runtime.
- Integrated per-step LIBERO snapshots plus a first-state round-trip gate. Added local-asset fallback resolution so the relocated checkpoint/processor/LIBERO assets can be read from `archive/full_experiment/checkpoints` without modifying them.
- Full unit suite passes after integration: 17/17. `git diff --check` is clean.
- Selected the no-rollout Phase 2 diagnostic cells from existing immutable observations: task-3 success/failure episodes at steps 50 and 100, with task 4 available as a shared-scene alternate goal. Confirmed saved arrays contain both cameras, eight-dimensional policy state, and actions.
- Added `scripts/run_phase2_forward_gate.py`: four saved task-3 states, four proposal seeds, exact-seed repeat checks, fixed-noise paraphrase/alternate-goal/main-view/wrist-view counterfactuals, archive-chunk similarity, modality-resolved VLM changes, KV-cache changes, and denoising/action-expert response curves. Raw tensors are routed to ignored `local/`; only compact derived reports go under `reports/`.
- Script compilation/help, full tests (17/17), and `git diff --check` pass. A one-state GPU smoke is next; it performs no simulator rollout and reads only the existing local SmolVLA checkpoint.
- Completed the one-state real-checkpoint smoke: 9 fixed-observation queries, 40 MB local raw capture, 56 KB report, no simulation. Exact hook fidelity and deterministic repeats both have zero numerical difference.
- Preliminary factor result at task-3 failure step 50: alternate-goal action RMSE 0.859, paraphrase 0.333, main-mean 0.287, wrist-mean 0.295. VLM deltas localize to the manipulated token span, while action-expert/velocity deltas amplify late in denoising.
- Recorded a negative methodological result: KV-cache norm change ranks paraphrase above alternate goal despite the reverse action-effect ranking, so cache magnitude alone is not valid modality attribution.
- Completed the four-state exact-forward confirmation: 36 queries, 158 MB ignored raw capture, 132 KB compact report, no simulator rollout. Hook fidelity and repeat determinism remain exact.
- Raw-analysis error: a substring parser included `repeat_original` groups as ordinary `original` groups and requested a nonexistent noise-202 repeat. No data changed; reran with exact suffix exclusion.
- Added within-state and denoising-time interpretation. The proposal/action-expert coupling is strongest only at the final denoising call and step 100; step-50 coupling is near zero despite comparable hidden variation.
- Discovered and quantified an executable-subspace confound: 25 of 32 internally denoised action channels are discarded in LIBERO, yet carry velocity magnitude and stochastic differences comparable to the seven executed channels. Final padded outputs are correctly near zero, confirming this is denoising work rather than executed behavior.
- Extended exact capture to the input of `action_out_proj`, persisted it in Zarr, and updated regression tests. Full tests remain 17/17 and `git diff --check` passes.
- Completed a one-state action-head subspace diagnostic (9 queries, 46 MB ignored raw data). Noise differences are predominantly output-row/padding aligned (74.4%/65.4%), while controlled modality/goal differences are approximately 96–97% output-null-space with small but strong executed-row projections.
- Completed the four-state action-head confirmation: 36 queries, 183 MB ignored raw capture, 144 KB report, no simulator. The executed/padding/null geometry and direction-selective gain reproduce across success/failure episodes and steps 50/100.
- Ran a primary-source novelty search for VLA action-decoder row/null/Jacobian decompositions. COAST is the closest collision on latent success subspaces and decoding bottlenecks; action-head redesign/injection work is adjacent. No direct VLA executable-versus-padding-versus-null decomposition was found in this search, while output-potent/null geometry is established outside VLA and must be credited rather than claimed as new mathematics.
- Source-access error: one multi-paper browser open returned more content than the tool context limit and was truncated. No claim was based on the truncated response; the relevant papers were reopened individually as searchable arXiv HTML.
- Read COAST's full method and appendices. It preserves denoising steps but mean-pools all action tokens (and all 49 state/future/action tokens for GR00T), fits success/failure geometry from ordinary unmatched rollouts, and reports no decoder/Jacobian, padding, same-state, or flow-noise-controlled decomposition.
- Reviewed the closest new null-space collision, *Output-Level Regularization Eliminates the Seed Lottery*. Its Jacobian null space is in **parameter-update space** during VLA fine-tuning; the present diagnostic is in **inference-time hidden-state space** relative to the executable action/effect decoder. The mathematical motif overlaps, but the estimand and intervention are distinct.
- Found a simulator-side matched-state blind spot: LIBERO's flattened MuJoCo state excludes robosuite controller interpolation state, the Panda gripper's accumulated command, robot history buffers, episode clocks, observable timing, vector-wrapper flags, and RNG state. A physics-only restore is therefore not a valid same-state branch contract.
- Extended `LiberoStateSnapshot` to capture and restore the missing runtime state and added fake-environment regression coverage. Focused state/capture tests pass 5/5.
- State-gate startup error: the first real smoke imported LIBERO before setting `LIBERO_CONFIG_PATH`, triggering its interactive first-run prompt and an EOF. Reordered initialization; no report or simulator artifact was produced.
- State-gate relocation error: the archived LIBERO config contains old absolute workstation paths, so the next attempt could not find an init-state file. The smoke now creates a temporary resolved config pointing to the relocated package and local assets; the archived config remains unchanged.
- State-capture error: LIBERO's `ObjectState.get_joint_state` indexes a free-joint `(start, end)` address as a multidimensional tuple and raised `IndexError`. Added a named-joint fallback using the correct slice semantics, preserving joint values without modifying LIBERO.
- Completed the real task-3 state-contract smoke with one reset and two three-action replay comparisons. MuJoCo-only restoration contaminated the second branch (`0.0372` maximum state difference, camera differences up to `172/255`, and different accumulated gripper commands). Full runtime restoration reproduced final pixels exactly and simulator state to `7.12e-15`; predicates, grasp status, bookkeeping, and RNG also round-tripped exactly.
- Canonical compact state-gate report: `reports/phase2_state_gate/state_contract_20260727T113937Z/summary.json`. This was a tiny deterministic simulator contract test, not a policy rollout; no checkpoint was loaded and no canonical run was touched.
- Added an explicit language-conflict control to the fixed-observation gate: negate the original drawer goal and then state the feasible cabinet goal. The 40-query confirmation remains exactly deterministic with zero hook fidelity error.
- The conflict condition is nearly equidistant between original and alternate at pooled level, but action alignment flips with trajectory step: at step 50 it is original-goal aligned, while at step 100 it is alternate-goal aligned. The VLM representation itself remains weakly alternate-aligned at both steps.
- Contradiction-dynamics analysis error: the first query-name parser treated `repeat_original` as an `original` group and constructed a nonexistent `repeat_alternate_goal` key. Tightened the suffix filter and reran; no raw data or report was changed by the failed attempt.
- Added and ran `scripts/analyze_phase2_contradiction.py`. At step 50, layer-15 expert alignment moves from `+0.0366` at the first denoising call to `-0.1907` at the last, and executed velocity is original-aligned throughout (about `-0.23` to `-0.27`). At step 100, expert alignment strengthens from `+0.1216` to `+0.2546` and executed velocity from `+0.2010` to `+0.3300` toward the alternate goal.
- Canonical expanded fixed-forward report: `reports/phase2_forward_gate/phase2_forward_gate_20260727T114152Z/`; raw exact tensors remain ignored under `local/phase2_forward_gate/phase2_forward_gate_20260727T114152Z/` (204 MB). No simulator rollout or immutable run was changed.
- Retrieved the revision-pinned official `lerobot/libero` data/meta Parquet only (`a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`): 1,693 episodes, 273,465 frames, about 22 MB locally, and zero video files or external model weights.
- Download execution issue: the initial four-worker shard retrieval stopped making progress at 90/382 files. Interrupted it without deleting partial data, changed the downloader default to one worker, resumed, and completed all requested data/meta files.
- Added `scripts/audit_training_trajectory_retrieval.py` and completed the designated v3 audit over 190,508 horizon-50 training windows, 16 fixed-forward proposals, and four archived rollout futures.
- Literal-copying control is negative: zero exact action-window matches; proposal same-task nearest standardized RMSE median `0.553` versus training self-nearest median `0.449`. State-conditioning worsens action similarity to `0.813`, and action-only other-task matches are generic low-motion tails with median state mismatch `1.707`.
- Same-seed plans are much more similar across success/failure episodes at the same landmark (`0.222`) than across landmarks (about `0.624`), while within-state proposal dispersion is `0.176`. Recorded a phase-locked policy-attractor hypothesis rather than an episodic memorisation claim.
- Added a runtime-local LIBERO config builder so future runs resolve the active package data and relocated read-only assets instead of inheriting stale archived absolute paths. Added `tests/test_runtime_assets.py`.
- Added `docs/phase2_discoveries_and_ersd.md`, which integrates the exact-forward, contradiction, executed/padding/null, simulator-state, and retrieval results into the ERSD/CPRD methodology and predeclares the smallest decisive Phase 3 tests.
- Documentation inspection issue: one broad read of the blind-spot audit and active records exceeded the tool output limit and was truncated. Replaced it with heading, section, and tail reads; no inference depended on omitted output.
- Planning-finalization patch error: the first task-plan patch omitted the list marker in its expected status context and did not apply. Re-read the exact section and applied a targeted patch; the failed attempt changed no file.
- Phase 2 final verification: 18/18 unit tests passed, all new Phase 2 scripts/modules compiled, and `git diff --check` passed.
- Final `bash scripts/resume_check.sh --full` passed: active-plan discovery, working Python/core imports, immutable canonical benchmark availability, and 18/18 targeted tests all succeeded.
- **Status:** complete
- Resume point: Phase 3 two-task branched CRD smoke gate, pending explicit approval for up to approximately 160 continuations. Do not download π0.5/GR00T weights or start the simulation expansion before that decision.

## Session: 2026-07-27 — Phase 3 branched CRD smoke gate

- **Status:** in progress
- User explicitly approved the predeclared Phase 3 simulation scope and asked for autonomous completion followed by a thorough professional-engineering review and corresponding fixes.
- Scope remains bounded to the SmolVLA two-task smoke ceiling of approximately 160 continuations. No π0.5/GR00T download, Phase 4 activation patching, or write under `archive/full_experiment/runs` is authorized or required.
- Success contract: a certified, deterministic state bank; a resumable branch ledger with common-random-number provenance and exact branch accounting; common multi-goal predicates; estimable `V`, `Q`, and `Q-V`; conditional hidden-state comparisons without group leakage; fixed-noise factor contrasts; compact derived evidence; and a post-run implementation audit with regression tests for every confirmed defect.
- Re-ran the project resume protocol and `bash scripts/resume_check.sh`; active plan, core imports, and canonical benchmark checks pass. Planning-session catch-up reported no unsynchronised context.
- Inspected the Phase 2 state and exact-query APIs before designing new code. The existing snapshot captures MuJoCo, controller/interpolator, gripper accumulator, robot buffers, observables, clocks, vector flags, Gym RNG, global NumPy RNG, object geometry, contacts, grasp state, and native goal predicates; it is suitable as the Phase 3 restoration primitive.
- Chose to reconstruct branch states by replaying the immutable benchmark's recorded executed actions from the same task/initial-state seed and validating replayed observations against the archive. This avoids rerunning completed policy rollouts while recovering the full runtime state that the old artifacts omitted.
- Inspection error: one metadata command used the system `python`, which lacks pandas. It made no changes; all project-data commands will use `local/lingbot-conda-env-archived/bin/python` as required by the resume guide.
- Benchmark feasibility audit: task 3 provides ten trajectories of 177–300 steps with six successes and four failures, while all task-4 trajectories succeed by steps 79–87. Task-4 step-100 snapshots therefore do not exist and must not be fabricated or obtained by ignoring termination.
- Tooling note: a `find | head` path inspection emitted a harmless broken-pipe message after `head` closed the stream. The needed path was found; future inspections use direct bounded paths rather than repeating that pipeline.
- Read both shared-scene BDDL contracts. The drawer task's exact native goal is `In(bowl, top_region)` and the cabinet task's is `On(bowl, top_side)`; this supports one explicit two-predicate evaluator and prevents language labels from being confused with task-native success flags.
- Audited the LeRobot environment wrapper. Initial-state selection is keyed by `episode_index/init_state_id`, while Gym's seed controls randomness separately; vector stepping uses same-step autoreset. Recorded both requirements in the Phase 3 branch contract.
- Found a more serious wrapper interaction before launching branches: the scalar LIBERO wrapper resets on success and the vector wrapper also uses same-step autoreset, after which the experiment loop performs another explicit reset. The benchmark's `initial_state_id=episode_index` field can therefore be wrong after a prior success. Paused branch implementation to add archive-based init-state resolution rather than propagating bad provenance.
- Confirmed the suspected double reset directly in the installed Gymnasium `SyncVectorEnv.step` source. The provenance issue is real at code level; the next check compares unaffected step-0 observations across the four nominally matched perturbation runs to measure its scientific impact.
- A first full camera comparison was too I/O-heavy for the execution wrapper and returned only partial output before the process ended. Changed the audit to read only the small step-0 `policy_state` member from each NPZ rather than repeating the expensive approach.
- The efficient archive audit confirms severe nominal-pair mismatch: exact step-0 policy state in clean versus paraphrase/blur/wrist is only `60/90`, `33/90`, and `14/90`. This changes the interpretation of the earlier perturbation study and will be added to the canonical experiment ledger; Phase 3 will use one captured snapshot per matched branch family rather than seed-based recreation.
- Verified the benchmark observation contract before implementing replay: saved cameras are `3x256x256` uint8, policy state is 8-D, robot joint state is 7-D, and preprocessing only converts raw HWC pixels to channel-first `[0,1]` before those arrays are recorded.
- Provenance inspection found that the benchmark was created from dirty commit `894c40a` through the former `experiments/...` path, whereas the relocated current runtime constructs 360-pixel environments. Phase 3 replay must reproduce the archived 256-pixel environment contract explicitly instead of inheriting the current helper's dimensions.
- Audited SmolVLA query APIs. Phase 3 will capture 80 unique first proposals once, reuse each stored chunk across its two matched continuation schedules, and generate only subsequent replans with explicit schedule seeds. This preserves the `Q` conditioning while avoiding 80 redundant exact forwards.
- Found a second preflight contract defect: Phase 2's saved cameras were recorded before LIBERO's required 180-degree image flip, but the fixed-forward script bypassed `env_preprocessor` and queried them unflipped. Paused Phase 3 implementation; the script and tests will be repaired and only the necessary corrected fixed-forward gate rerun before selecting latent sites.
- Added one reusable archived-camera orientation helper, applied it to both Phase 2 camera inputs, and added focused shape/orientation regression tests. Focused tests pass 5/5, the script compiles, and `git diff --check` is clean.
- Completed the corrected four-state fixed-forward gate as `phase2_forward_gate_20260727T123138Z`: 40 saved-observation queries, zero hook/repeat error, no simulator rollout, and no immutable artifact write. Ran the contradiction dynamics analysis on the corrected raw capture.
- The corrected input materially changes the science. The prior step-50/step-100 semantic-routing reversal disappears; contradiction expert/flow alignment is generally alternate-goal aligned at both landmarks, with only the final step-50 velocity weakly original-aligned (`-0.056`). Wrist-mean action RMSE becomes `0.958`, larger than alternate goal `0.472` and paraphrase `0.350`.
- The executable/null result survives and strengthens: controlled-factor hidden displacement is `97.85–98.64%` output-null, while proposal-noise displacement is `67.14%` padding-row and `74.98%` full-output-row. Corrected within-state action/expert coupling falls from the provisional `0.872` to `0.441`.
- Revisited the existing grouped-probe and retrieval utilities before designing Phase 3 analysis. Restricted the smoke analysis to direct paired estimates and low-dimensional, state-grouped validation rather than attempting an underpowered 720-D success probe on ten states.
- Added the Phase 3 core contract: ten explicit state specs, portable drawer/cabinet predicates, 80 unique first-query IDs, exactly 160 branch IDs, two common continuation schedules, deterministic seed derivation, low-dimensional query summaries, atomic JSON writes, and strict branch-ledger accounting.
- Added snapshot deserialization for resumable branches and focused tests for round-trip metadata, common-goal evaluation, branch uniqueness/ceiling, CRN seed schedules, historical reset indexing, action-summary validation, and ledger rejection. Focused Phase 3/state/orientation tests pass 10/10; compilation and whitespace checks pass.
- Storage-completeness patch issue: an initial assertion patch used stale test context and did not apply. Re-read the exact test and added the query `complete` assertion with a targeted patch; no file was changed by the failed attempt.
- Runner inspection confirmed the large Phase 3 patch landed in full and compiled. A follow-up metadata display requested a nonexistent `steps` column instead of the canonical `total_steps`; the read failed without changing data and was rerun with schema-aware column selection.
- Pre-run engineering audit found a fatal restore bug before GPU use: Zarr 2.18 groups do not expose `.parent`, but `_run_branch` attempted to read the source snapshot through that attribute. The runner now deserializes from the root state store and passes the immutable snapshot explicitly.
- Hardened Phase 3 persistence before launch: states and exact-forward queries are staged under local `.partial__*` Zarr keys and moved into place only after complete writes; persisted query identities and summary companions are checked; branch and query ledgers reject unexpected IDs; continuation noise hashes are recorded; and the contract now fingerprints the canonical source manifest/episode table, goal text/predicates, seeds, factors, image resolution, and activation sites.
- Expanded focused regression coverage for transactional query storage and the exact 80-core/160-total query ledger. Phase 3/capture/state/orientation tests pass 15/15; the runner and modules compile and `git diff --check` passes.
- Sandbox preflight could not communicate with the NVIDIA driver. This is the expected restricted-device boundary, not a project defect; the approved one-state/in-place smoke is being launched through the workstation execution boundary instead of retrying inside the sandbox.
- First in-place branch smoke stopped during the state certificate before any state, query, or branch was committed. `evaluate_common_goals` passed capitalized BDDL predicate names (`In`/`On`) directly to LIBERO, whose runtime registry asserted because it expects normalized predicate keys. The failed run is retained as resumable local evidence; the evaluator is being repaired and regression-tested before resuming it.
- Normalized only the predicate operator at the runtime boundary while preserving the canonical BDDL spelling in the frozen scientific contract. The fake evaluator now asserts the exact lower-case LIBERO API contract. Focused tests remain 15/15, compilation/whitespace checks pass, and inspection confirms the failed smoke committed no state/query/branch artifact.
- The resumed certificate reached repeated-action comparison and rejected harmless floating-point roundoff: MuJoCo differed by `1.91e-14`, non-image observations by at most `1.65e-16`, both cameras were bit-identical, and rewards/done/goals matched. The implementation incorrectly required every numeric observation to be exactly equal despite the predeclared tolerance-based contract. No state/query/branch was committed; explicit pixel/numeric/physics tolerances are being frozen and tested before a fresh contract run.
- Replaced the accidental exact-float test with a mixed certificate: cameras must remain bit-exact, other numeric observations must be within `1e-12`, and MuJoCo state within `1e-10`. These thresholds are now part of the contract and persisted certificate. A regression test accepts measured roundoff but rejects one-pixel or material numeric drift; focused tests pass 16/16 and compilation/whitespace checks pass.
- Passed the fresh-contract in-place smoke at `local/phase3_crd/phase3_crd_20260728T021125Z`: 2/160 branches, 1/160 exact queries, and two reconstructed source states committed transactionally with no partial groups or errors. Both states match archived step-0/landmark observations exactly; round trips are exact; repeated three-action certificates keep both cameras bit-identical and physics within `1.91e-14`.
- Persisted-payload audit confirms the two branches share exactly one query/chunk, use distinct predeclared continuation schedules, record the expected deterministic noise hashes, start with both common goals false, execute 150 steps, and terminate cleanly at the horizon. The smoke artifact is 7 MB; no immutable archive path was written.
- The full resume progressed into task 4, then stopped before committing `task04_ep001_step0050`: cameras, rewards, done, and goals were identical; MuJoCo drift was `6.36e-12`; a derived gripper-velocity observation drifted `2.35e-12`, narrowly exceeding the `1e-12` numeric cutoff. This reveals that the observation tolerance was still overfit to the first task. Completed atomic branches remain valid and will not be repeated; a narrowly audited monotonic contract amendment will relax only the numeric observation ceiling to `1e-10` while retaining bit-exact pixels and the `1e-10` physics ceiling.
- Added a guarded, explicit resume migration that accepts only a monotonic relaxation of the numeric-observation certificate field and rejects horizon or any other contract change. It preserves the old/new contract hashes, thresholds, timestamp, and reason in `contract_amendments`. The existing incomplete run is mechanically verified as eligible; 96 branches, 48 queries, six state groups, and zero partial groups remain intact. Focused tests pass 17/17.
- The migrated resume certified task-4 episode 1 but correctly stopped on episode 2: repeated three-action probes differed by `2.40e-08` in MuJoCo state and one intensity level in the wrist image, with gripper fields around `1e-10`. This is qualitatively beyond the prior serialization roundoff and may expose omitted contact-solver warm-start state. No episode-2 state/query/branch was committed; the restoration primitive is being audited rather than weakening the certificate again.
- Confirmed the omission in code: robosuite's current `MjSim.get_state()` serializes only time, `qpos`, and `qvel`; it excludes `act`, control/applied forces, mocap/user state, and especially `qacc_warmstart`, which MuJoCo uses to initialize the next contact solve. Extended snapshots to capture and restore every available full-physics/control field, with strict field/shape checks and fake-simulator regression coverage.
- Added a second narrowly guarded contract upgrade that permits only the addition of the declared full-simulator field list and records the old/new hashes and rationale. It cannot alter horizon, goals, seeds, or thresholds. The current incomplete run is mechanically eligible, and focused tests pass 18/18.
- The first full-simulator restore attempt stopped before its certificate because JSON serialization collapsed empty `mocap_pos` from shape `(0, 3)` to `(0,)`. No state/query/branch was committed. The simulator-field encoding is being corrected to preserve explicit shapes for zero-sized arrays before repeating the same two-branch verification.
- Simulator arrays now persist explicit shape plus values, with backward-compatible reading for older snapshots. Added a zero-sized `(0, 3)` mocap regression case; focused tests remain 18/18 and compilation/whitespace checks pass. The contract upgrade is recorded, while the ledger remains at 112 branches/56 queries with no partial artifacts.
- The two-branch contact-state verification now passes and advances the ledger to 114/160. The formerly failing `task04_ep002_step0050` matches the archive exactly and, after full simulator-field restore, repeated three-action probes have exactly zero MuJoCo, camera, and robot-observation difference (30 contacts at the source state). A no-checkpoint field ablation is next to identify which omitted field caused the old divergence.
- The first field-ablation report was written but deliberately failed its positive-control assertion: dropping only `qacc_warmstart` exactly reproduced the old `2.40e-08`/one-wrist-pixel divergence and every other single-field drop was exact, but the very first full restore into a newly constructed environment diverged substantially before later restores stabilized. The script had assumed a cold cross-instance restore was valid. This exposes a separate process-resume defect; the report will be regenerated with cold-start and hydrated controls separated.
- Phase 3 branch resume must not deserialize a snapshot into a fresh simulator and immediately execute policy actions. The runner will reconstruct every needed source episode once per process from its exact archived action sequence, revalidate archive fidelity/certificates, and use that new in-memory snapshot for all remaining branches. Persisted snapshots remain provenance evidence and query sources, not trusted cross-process branch roots.
- Regenerated `restore_field_ablation.json` with separate cold-start and hydrated controls. The cold cross-instance positive control diverges by `0.1262` in state and up to `191/255` pixels; after hydration, full restore is exact. Removing only `qacc_warmstart` reproduces the smaller `2.40e-08`/one-pixel within-instance divergence exactly; all nine other single-field removals remain exact.
- Refactored the branch runner so each source episode is reset, archive-action replayed, archive-validated, and certified once in every process before any missing branch uses it. Existing state artifacts are cross-checked against the new replay but never used as cross-process branch roots. New branch payloads carry the per-process reconstruction mode, state hash, archive fidelity, certificate, and persisted-state comparison.
- Added a guarded contract amendment for the per-process archive-replay source rule; it rejects any simultaneous change. Focused tests pass 19/19, scripts compile, and whitespace checks pass. A two-branch resumed smoke will verify the new payload and replay path before final completion.
- The resumed-source smoke passes at 116/160: the two new branch payloads share the same replayed state hash, exact archive/state-artifact comparisons, and a zero-difference three-action certificate.
- Deterministic replay audit of the first branch created across the old process boundary fails semantically. The legacy payload reported drawer success at step 134; replay from exact current-process archive reconstruction fails at the 150-step horizon, with different first-10/first-plan/final effects. The affected scope is exactly 30 task-3 branches: 14 remaining step-50 branches plus all 16 step-100 branches from the two states persisted by the initial smoke. Later source states were reconstructed within their branch-producing process.
- The repair will preserve all 30 legacy JSON payloads under a superseded-evidence directory with old hashes, remove them from the active ledger transactionally, and regenerate only those IDs from one current-process replay. Existing exact proposal queries remain valid and will be reused.
- Implemented the scoped refresh as an interruption-safe transaction: the manifest records intent first, each legacy payload is moved with its SHA-256 into `superseded_branches/legacy_cross_instance_restore`, missing active IDs are naturally resumable, and completion requires all 30 old/new files before recording new hashes. Added pure regression tests for the 30-ID scope, backup/resume/finalize behavior, and semantic replay comparison. Focused tests pass 21/21.
- Completed the repaired Phase 3 execution: exactly 160 active branches and 160 exact-forward query artifacts, ten complete states, no partial groups, 80 proposal chunks each referenced by exactly two branches, and deterministic continuation seed/noise hashes. The 30 superseded payloads and both old/new hash maps are preserved; all 30 semantic payloads changed and 12/30 success labels flipped after correct reconstruction.
- Initial exploratory summary error: a quick first-plan goal count indexed the effect dictionary at the goal name instead of its nested `goals` field and stopped after printing earlier aggregate tables. No artifact changed; the formal analyzer will use schema-validated extraction rather than repeating the ad hoc expression.
- Second exploratory summary error: pandas preserved branch success as boolean in a schedule pivot, so direct subtraction raised a type error after the proposal-variation table had printed. No artifact changed; the formal analyzer will cast outcomes explicitly before paired arithmetic.
- Dependency probe found that `statsmodels` is not installed in the archived working environment. No installation is needed: the analysis will use NumPy/SciPy/scikit-learn, empirical-Bayes binomial shrinkage, and explicit grouped cross-validation rather than adding a dependency mid-study.
- Formal analyzer validation error: the query-reference `Series` was converted to a set of counts rather than a set of index IDs, so the valid 80×2 ledger was rejected. The analyzer stopped before writing any statistical table; the validator is being corrected to compare `value_counts().index` explicitly.
- After the validator fix, the analyzer found a scientific invariant violation and stopped before writing statistics: the two continuations for `task03_ep001_step0100__goal_drawer__proposal_202` have different first-plan effects despite sharing the exact state/query/chunk and diverging only after action 50. This proves that old within-process branches lacking solver-state/per-process reconstruction provenance can also carry hidden simulator leakage.
- The trustworthy boundary is now explicit: 76 active payloads carry the new `source_reconstruction` certificate and 84 do not. All 84 uncertified legacy payloads will be hash-preserved and regenerated from per-process archive replay, producing a uniform 160/160 certified branch ledger. Queries/factor forwards remain valid and will not be repeated.
- Completed the provenance-wide repair: 160/160 active payloads now carry per-process archive reconstruction; all 80 paired first-plan effects are identical across continuation schedules. The 84 old payloads and old/new hashes are preserved; 66/84 semantic effects changed but no additional success label changed beyond the 12 flips already corrected by the first scoped refresh.
- Analyzer resume error: validation still assumed exactly one refresh transaction and rejected the now-correct two completed refresh records. It stopped before statistics; the check is being generalized to require that every recorded refresh is complete.
- Completed the formal Phase 3 analysis after generalizing refresh validation. The repaired active ledger passes all gates: 160 branches, 160 queries, ten states, 80 core queries referenced exactly twice, 80 factor queries, current-process source provenance on every branch, and zero paired first-plan mismatches or partial groups.
- Corrected outcome evidence is `73/160 = 0.45625` success. The descriptive variance decomposition is state-goal `80.79%`, proposal-within-state-goal `15.43%`, and continuation-within-proposal `3.78%`; only 3/80 continuation pairs disagree and 6/20 state-goal cells vary by proposal.
- Recoverability is cabinet-source cabinet/drawer `1.0/0.0` and drawer-source drawer/cabinet `0.45/0.375`. Added the descriptive veto–composition decomposition: cabinet-source language switching suppresses source completion by `1.0` but constructs drawer success at `0.0`; drawer-source suppression/constructive transfer is `0.45/0.375`.
- Conditional hidden-state tests are negative for the predeclared summaries. At `alpha=100`, `V` RMSE changes `0.386→0.389`, pre-execution `Q` `0.444→0.452`, effect-controlled `Q` `0.413→0.422`, and effect-controlled `L` `0.200→0.199`. The controlled-`Q` MSE improvement is `-0.0076` with source-episode bootstrap interval `[-0.0182,+0.0025]` and remains non-positive at alphas 10, 100, and 1000.
- ERSD analysis from exact Phase 3 queries yields median controlled-factor output-null energy `98.17%` versus proposal-noise output-null `21.35%`; proposal noise has median executed-row energy `8.18%` and padding-row energy `70.38%`. Pairwise action/expert distance correlates only `0.324/0.313` with absolute `Q` difference.
- Senior scientific review found that the source state bank is not progress balanced. All five cabinet states are step 50 with median bowl height `0.972`, 3/5 grasped, and a closed drawer; drawer states mix steps 50/100 with median height `0.898`, none grasped, and a displaced drawer in 4/5. Because the policy is reset and has no observation-history queue, the result is current-state/occupancy conditioned—not evidence of recurrent trajectory memory.
- Added `state_geometry_audit.csv`, the veto–composition summary, and an explicit raw-factor caveat: Phase 3's `contradiction` label is contrastive negation of the other goal followed by reaffirmation of the target, not an internally inconsistent instruction.
- Engineering review found and fixed three further issues without rerunning any branch: (1) paired-prefix equality is now enforced in the runner before the second branch is committed; (2) nested certificate comparison now fails closed on missing fields, shape changes, or unequal nonnumeric data and requires the full probe-action count; (3) the native/counter interval now resamples source episodes rather than treating repeated landmarks as independent. The drawer interval widened from `[-0.10,+0.25]` to `[-0.25,+0.1875]`.
- A first combined comparator patch used the wrong import context for `audit_phase3_restore_fields.py` and did not apply. Re-read the file and applied the change in a context-correct patch; no artifact changed in the failed attempt.
- Added `docs/phase3_engineering_review.md`, corrected stale Phase 2 orientation/reversal claims, updated the active plan/README/resume guide/causal programme/experiment ledger, and introduced occupancy-balanced CRD as the next proposed gate. Phase 4 and π0.5/GR00T scaling remain deferred because the current hidden-information gate is negative and the state-selection confound is unresolved.
- Final verification passes: 41/41 full tests, relevant script/module compilation, `git diff --check`, and `bash scripts/resume_check.sh --full`. The final integrity audit reports manifest `complete`, exactly 160 branches/160 queries/10 states, no partial groups, and exact matches for both source manifest and source episode-table hashes.
- Canonical compact Phase 3 evidence: `reports/phase3_crd/phase3_crd_20260728T021125Z/`. Raw evidence remains workstation-only at `local/phase3_crd/phase3_crd_20260728T021125Z/`. Nothing under `archive/full_experiment/runs` was modified, and no π0.5/GR00T checkpoint was downloaded.
- **Status:** complete
- Resume point: Phase 3b occupancy-balanced recoverability design, pending an explicit new simulation-scope decision. Do not repeat the completed Phase 3 matrix or proceed directly to causal patching with the current summary features.

## Session: 2026-07-28 — Next-step and portable publication audit

- **Status:** in progress
- User requested the next recommended research step and cloud-accessible publication of all safe project work for use from another device.
- Reloaded the active plan with the planning-with-files workflow and ran `bash scripts/resume_check.sh`; the project records, working Python, imports, and immutable canonical benchmark checks pass.
- The first incomplete scientific phase remains occupancy-balanced Phase 3b. Before recommending execution, this session will audit the exact minimal design, Git scope, remote configuration, ignored/large-file boundaries, and available cloud-access path.
- Publication safety boundary: code, tests, documentation, planning records, and compact derived reports may be committed and pushed. Checkpoints, raw runs, activations, observations, videos, environments, credentials, and other workstation-only evidence must remain outside Git even though the user requested “everything.”
- Audited `origin` as the GitHub repository `Dz803/smolvla-hidden-state-analysis`, with local `main` initially equal to `origin/main`. The workspace contains about 1.8 MB of compact reports and no files over 10 MB in the publication scope; the secret-pattern filename scan found no matches. `archive/`, `local/`, `vendor/`, `logs/`, and `planning/` remain ignored. A sandboxed remote probe could not reach GitHub, so the final push will require network-enabled execution.
- Specified `docs/phase3b_methodology.md`. Stage A builds 32 policy-independent candidates across drawer, possession, locus, demonstration-support, and layout factors and certifies both goals before any VLA query. Stage B is a separately gated 16-state SmolVLA pilot capped at 128 branches; Stage C is a locked confirmation set capped at 256 branches and is not automatically authorised.
- The method makes constructive goal switching—not mere source-skill vetoing—the primary language-grounding contrast, and tests support/occupancy effects at matched oracle cost. Directional hidden features are fold-local and must beat complete observable/proposal-effect baselines on held state families before Phase 4 patching.
- Added a durable publication protocol to `AGENTS.md` and cross-device instructions in `docs/portable_results.md`: every verified Git-safe checkpoint is committed and pushed with a reported hash, while raw evidence requires a separately selected private object store and checksum manifest.
- One combined documentation patch referenced a progress-log line under the wrong file update and failed without changing any file. Reapplied it with explicit file sections and added the missing rule that every Phase 3b process must reconstruct and recertify its state rather than trust a cross-process snapshot.
- Verification after the design/documentation changes passes: 41/41 project tests, `bash scripts/resume_check.sh --full`, and `git diff --check` all succeed. No canonical rollout was rerun or modified.
- The first `git add -A` attempt failed because the managed sandbox mounts `.git/index` read-only. Retried the same scoped staging action with elevated Git permission authorised by the user's explicit upload request; no project content changed during the failed attempt.
- The first staged whitespace audit exposed three trailing-space lines in previously untracked Markdown artifacts. Removed the whitespace (using a list for the Phase 3 review metadata) before commit; no scientific content changed.
- Final staged-publication audit covers 141 Git-safe paths. `git diff --cached --check` passes, the staged secret-pattern scan returns no matches, and no staged path uses a prohibited local/raw prefix or checkpoint, video, Zarr, or Parquet suffix.
