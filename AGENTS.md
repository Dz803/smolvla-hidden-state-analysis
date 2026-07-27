# SmolVLA Hidden-State Project Instructions

## Professional English

Before answering a request or taking action, first restate the user's message in clear, natural, professional English under the label `Professional rephrasing:`. Preserve the original meaning and do not reproduce credentials or secrets.

## Resume protocol

At the beginning of every session in this repository:

1. Read `.planning/.active_plan`.
2. Read `.planning/<active_plan>/task_plan.md` completely.
3. Read `.planning/<active_plan>/findings.md` and the latest section of `progress.md`.
4. Read `docs/experiment_log.md` and run `bash scripts/resume_check.sh`.
5. Continue from the first incomplete phase; do not restart completed experiments.

After every material experiment, analysis, decision, or error:

- append the action and result to `.planning/<active_plan>/progress.md`;
- add durable discoveries to `findings.md`;
- update phase status in `task_plan.md`;
- append a concise entry to `docs/experiment_log.md` when the scientific evidence or canonical artifacts change.

## Data and execution safety

- Treat `archive/full_experiment/runs` as immutable evidence. Never overwrite or rename canonical runs.
- `archive/`, `local/`, `vendor/`, `logs/`, and `planning/` are workstation-only and intentionally ignored by Git.
- The working offline Python is `local/lingbot-conda-env-archived/bin/python` on the workstation. A code-only GitHub checkout must provision its own environment.
- Keep episode/task groups isolated across fitted transforms and evaluation splits.
- Full-trajectory scores are retrospective and must not be presented as early-warning performance.
- Do not launch a large rollout or factorial sweep before the offline gate and causal smoke gate pass. Obtain explicit approval before materially expanding GPU/simulation usage.
- Never commit checkpoints, raw activations, observations, videos, full runs, credentials, or generated environment directories.

## Current scientific boundary

The current evidence is predictive, with causal effects established only for the tested input perturbations. It does not yet establish a causal hidden-state mechanism or modality-token attribution. The next work is defined in the active plan and `docs/causal_research_program.md`.
