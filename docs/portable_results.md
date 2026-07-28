# Portable Results and Cloud Access

## Cloud copy

The portable project is published through the configured GitHub remote:

```text
https://github.com/Dz803/smolvla-hidden-state-analysis.git
```

On another device, clone it with:

```bash
git clone https://github.com/Dz803/smolvla-hidden-state-analysis.git
cd smolvla-hidden-state-analysis
git switch main
```

For an existing clone:

```bash
git switch main
git pull --ff-only origin main
```

Always compare `git rev-parse HEAD` with the commit hash reported at handoff. Access may require the repository owner's GitHub authentication if the repository is private.

## What the GitHub copy contains

- source code and tests;
- active planning and resume records;
- experiment methodology and engineering reviews;
- compact Markdown, CSV, and JSON reports for the completed offline, Phase 2, and Phase 3 gates; and
- scripts needed to regenerate compact analyses when the local evidence is available.

The current canonical compact Phase 3 report is:

```text
reports/phase3_crd/phase3_crd_20260728T021125Z/
```

## What remains workstation-only

The following are intentionally excluded from Git: canonical/full runs, checkpoints, raw activations, observations, videos, datasets, local environments, vendor checkouts, logs, and credentials. In this workstation copy they live under ignored paths such as `archive/`, `local/`, `vendor/`, and `logs/`.

Those files are not needed to read the conclusions or inspect the compact result tables. They are needed for simulator continuation, exact-forward replay, or full raw reanalysis. If cross-device raw-data access becomes necessary, use a user-selected private object store under an immutable project prefix and publish a checksum manifest in Git. Do not use GitHub as a substitute for raw-artifact storage.

## Publication rule

At each verified checkpoint, inspect and test the staged Git-safe files, scan them for secrets/prohibited artifacts, push the commit, and record the branch and commit hash. A local commit is not considered cloud-backed until its remote branch is verified.
