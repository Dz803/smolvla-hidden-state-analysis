# Phase 2 training-trajectory retrieval audit

This report uses only the revision-pinned official `lerobot/libero` state/action
Parquet. No video or external checkpoint was downloaded.

## Main result

The four audited states and their 16 fixed-seed proposals provide no evidence of
literal horizon-50 action-window copying:

- zero exact action-window matches at tolerance `1e-6`;
- median same-task nearest standardized RMSE `0.553`, compared with `0.449`
  for train-window self-neighbours;
- median within-state proposal dispersion `0.176`;
- restricting retrieval to the 128 closest same-task robot states worsens the
  action match to `0.813`.

Action-only retrieval is misleading. Other-task windows appear closer (`0.446`)
but have median state mismatch `1.707`; inspection identifies generic low-motion,
saturated-gripper tails rather than task-specific memories.

Same-seed plans at the same landmark across different episodes are relatively
similar (`0.222`), whereas different-landmark plans differ by about `0.624`.
This motivates a trajectory-phase-attractor hypothesis, not an episodic-copying
claim.

## Boundary

This is an action/state retrieval proxy. Robot state omits object geometry and
videos were deliberately excluded. A nearby demonstration motion cannot by
itself establish memorisation or causal training influence. See `summary.json`
and `docs/phase2_discoveries_and_ersd.md` for the complete interpretation.
