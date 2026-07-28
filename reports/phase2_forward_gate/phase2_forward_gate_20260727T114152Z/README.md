# Phase 2 exact-forward gate

Run: `phase2_forward_gate_20260727T114152Z`

This report uses immutable saved observations and explicit fixed flow noise. Raw structured activations remain under `local/` and are intentionally excluded from Git. See `summary.json` for the predeclared scientific boundary.

The gate contains four task-3 states at steps 50 and 100, 40 exact queries,
four proposal seeds, deterministic repeats, a paraphrase, a feasible alternate
goal, an explicit contradiction, and main/wrist mean-image controls. Hook
fidelity and deterministic repeats have zero maximum numerical error.

Key derived findings are the direction-selective semantic gain, the
step-dependent contradiction-routing reversal, and the executed/padding/null
action-head geometry. These are fixed-observation effects, not branched
recoverability or causal mediation. The full interpretation and equations are
in `docs/phase2_discoveries_and_ersd.md`.
