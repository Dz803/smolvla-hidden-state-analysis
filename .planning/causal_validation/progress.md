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
