# Trajectory and Visual Confound Audit

This derived audit asks how much exact-landmark failure prediction is explained by the policy's observable behavior, complete ordered action plan, and crude current pixels.

It reads the immutable canonical benchmark and writes only to this report directory. It does not rerun or alter rollouts.

## Protocol

- exact replanning landmarks: steps 0, 50, and 100;
- active episodes only at each landmark;
- five task-grouped outer folds;
- logistic regularization selected inside nested task-grouped folds;
- scaling and blockwise PCA fitted inside every inner/outer training split;
- action history stops at `t-1`;
- current action chunk preserves all 50×7 ordered values;
- current frames are deterministic 16×16 RGB block averages from both cameras;
- metrics include prevalence-normalized AUPRC, AUROC, within-task AUROC, Brier score, and log loss.

## Main step-100 result

There are 319 episodes still active at step 100; failure prevalence is 0.455.

| Feature | AUPRC | AUROC | Within-task AUROC | Brier | Log loss |
|---|---:|---:|---:|---:|---:|
| Executed action prefix | 0.610 | 0.635 | 0.689 | 0.237 | 0.672 |
| Ordered current chunk | 0.744 | 0.765 | 0.793 | 0.198 | 0.581 |
| Behavioral context | 0.803 | 0.821 | 0.803 | 0.177 | 0.532 |
| Action expert | 0.878 | 0.868 | 0.797 | 0.147 | 0.456 |
| VLM | 0.842 | 0.865 | 0.837 | 0.153 | 0.470 |
| Behavioral context + action expert | 0.882 | 0.891 | 0.843 | 0.137 | 0.435 |
| Behavioral context + VLM | 0.883 | 0.891 | 0.863 | 0.141 | 0.434 |
| Low-resolution current pixels | 0.721 | 0.774 | 0.731 | 0.192 | 0.583 |
| Low-resolution pixel change | 0.512 | 0.612 | 0.699 | 0.272 | 1.146 |

Behavior and plan explain much of the signal; the archived post-hoc hidden state still adds predictive information. Because that state came from a fresh stochastic re-query rather than the action-producing forward, this increment is current-observation evidence, not an online causal-mechanism result. Crude pixels expose both initial-condition difficulty and later visual progress but do not close the gap. High-dimensional visual+behavioral concatenations transferred poorly and should not be treated as the definitive semantic control.

## Files

- `trajectory_confound_metrics.csv`: summary metrics.
- `trajectory_confound_predictions.parquet`: out-of-fold episode predictions and selected regularization values.

Reproduce with:

```bash
PYTHONPATH=src local/lingbot-conda-env-archived/bin/python \
  scripts/audit_trajectory_confound.py \
  --run archive/full_experiment/runs/benchmark_400_20260723T021424Z_e0638bea \
  --output reports/trajectory_confound_audit
```
