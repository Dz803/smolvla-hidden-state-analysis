# Offline robustness gate

This report completes Phase 1 of the causal-validation plan using only existing immutable runs. The analysis uses 2,000 deterministic bootstrap draws, fold-local scaling/PCA, episode-balanced training weights, and task- or suite-isolated evaluation.

## Decision

The step-100 hidden-state advantage survives the estimable offline controls, especially for the action-expert representation, but the result is not a universal condition-invariant mechanism.

- Action-expert step-100 AUPRC is 0.844 (episode-bootstrap 95% CI 0.785–0.895; task-cluster 95% CI 0.736–0.927). Its within-task AUROC is 0.878 (task-cluster 95% CI 0.822–0.930).
- VLM step-100 AUPRC is 0.770 (episode-bootstrap 95% CI 0.699–0.835; task-cluster 95% CI 0.678–0.867). Its within-task AUROC is 0.846 (task-cluster 95% CI 0.772–0.912).
- The richer five-sample state/action-history baseline reaches only 0.517 AUPRC and 0.652 within-task AUROC at step 100.
- Leave-one-suite-out macro AUPRC is 0.835 for the action expert and 0.785 for the VLM, versus 0.534 for state/action history. Action-expert AUROC is 0.832–0.964 in every held-out suite.
- Task-held-out action-expert transfer remains useful in both directions for clean↔main-camera blur and clean↔instruction paraphrase. VLM transfer is weaker for paraphrases.
- Clean probes assign high mean failure probability to the all-failure wrist-mask condition (0.924 action expert; 0.941 VLM at step 100), but ranking metrics are undefined. Wrist-mask→clean transfer cannot be fitted because the source condition contains no successes.

The wrist-mask degeneracy and weaker paraphrase transfer prevent a claim that one representation direction is condition invariant. Phase 2 should refine failure semantics and expose modality token spans before any activation-patching claim.

## Outputs

- `warning_confound_metrics.csv`: fixed-horizon point estimates and episode/task-cluster intervals for the six primary features; exploratory layer probes retain point estimates only.
- `warning_leave_one_suite_out.csv`: suite-held-out metrics at steps 0, 50, and 100.
- `warning_cross_condition_transfer.csv`: task-held-out clean↔perturbation transfer, including explicit coverage/estimability fields.
- `warning_score_growth.csv`: outcome-stratified score change through step 100.
- `warning_suite_metrics.csv`: descriptive within-suite metrics for the existing grouped probes and history baseline.

These are predictive diagnostics. They do not establish a causal hidden-state mechanism or modality-token attribution.
