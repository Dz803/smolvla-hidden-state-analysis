# Stage A competence–compatibility gap

This compact report separates two quantities that an ordinary oracle-success flag
conflates:

- `F_C(s,g)`: whether a declared policy-independent controller/certificate class
  contains a physically successful path;
- `P_K(s,g) = max_k Y(N(s),g,k)`: whether any member of the complete, registered
  replay bank succeeds under its bound execution contract.

The observed gap is `F_C=1, P_K=0`. All
`64` state-goal cells are physically certified, while
`63` have a successful replay-bank
member. The sole gap cell preserves all 46 cabinet failures. Every bridge passes,
no wrong goal is reached, and no attempt terminates early. A separate factorized
path reaches the cabinet using stable acquisition followed by 204 feedback
transport/release actions.

This falsifies the inference that failure of this finite replay bank proves physical
infeasibility. It does **not** show that SmolVLA solves the cell, that a hidden state
causes the failure, or that the adaptively developed factorized certificate
generalises. Stage A loaded no VLA. The certificate is an existence witness for one
cell and must be frozen before a held-out confirmatory study.

Files:

- `goal_cells.csv`: one row for each physical state and goal, with the evidence
  classes kept separate;
- `gap_cells.csv`: only cells where physical feasibility passes but proposal
  compatibility fails;
- `matched_gap_pairs.csv`: the gap cell and its demonstration-near matched support
  state under the same proposal bank/execution contract;
- `summary.json`: estimands, counts, and claim boundary;
- `manifest.json`: source and artifact hashes.
