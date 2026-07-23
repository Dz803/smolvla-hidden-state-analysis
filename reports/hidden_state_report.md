# SmolVLA-LIBERO Hidden-State Experiment

Generated: 2026-07-22 (Asia/Shanghai)

## Scope and data contract

This analysis uses the completed clean Spatial pilot `pilot_spatial_100_20260722T060211Z_63ee3442`: 100 episodes, 17,698 environment steps, 58 successes, and 42 timeout failures. It evaluates eventual episode failure from information available at sampled time `t`; no future state, episode length, normalized progress, or outcome-derived feature is included in probe inputs.

Hidden states were collected every fifth environment step at VLM and action-expert layers 3, 7, 11, and 15. There are 3,566 aligned observations and 28,528 finite pooled activation vectors. VLM vectors have 960 dimensions and action-expert vectors have 720 dimensions. Token boundaries are unknown, so this report does not claim separate image-token and language-token results.

All diagnostic probes use:

- leave-one-task-out evaluation;
- StandardScaler fitted only on each training fold;
- fold-local 32-component PCA for high-dimensional features;
- class-weighted logistic regression;
- inverse episode-length weights so timeout trajectories do not dominate training;
- complete separation of episodes and tasks between train and test folds.

This is frozen-feature probe fitting, not VLA training or fine-tuning.

## Primary result: prediction at fixed environment horizons

Episode AUPRC is shown below. Failure prevalence is 0.42; the leave-one-task training-prior baseline obtains approximately 0.316 at fixed horizons.

| Feature set | Step 0 | Through step 25 | Through step 50 | Through step 75 | Through step 100 | Full trajectory |
|---|---:|---:|---:|---:|---:|---:|
| Policy output | 0.389 | 0.380 | 0.368 | 0.437 | 0.495 | 0.906 |
| Robot state | 0.600 | 0.494 | 0.454 | 0.486 | 0.567 | 0.949 |
| Action uncertainty | 0.461 | 0.436 | 0.401 | 0.366 | 0.382 | 0.545 |
| VLM final layer | 0.437 | 0.501 | **0.663** | **0.769** | **0.824** | 0.996 |
| Action-expert final layer | 0.447 | 0.462 | 0.484 | 0.524 | 0.568 | 0.739 |
| VLM + action expert | 0.508 | 0.482 | 0.587 | 0.684 | 0.780 | 0.966 |
| All diagnostic features | 0.509 | 0.483 | 0.580 | 0.689 | 0.771 | 0.964 |

The VLM pathway contains the strongest early failure association in this pilot. It is near chance at step 0, becomes meaningfully informative by step 50, and is substantially stronger by steps 75–100. The action expert and sampled action uncertainty are weaker. Combining all features does not improve on the VLM-only probe.

Full-trajectory scores must not be described as early warning: they average all available samples, including late behavior. The fixed-horizon columns are the relevant early-warning evidence.

## Layer-wise and divergence findings

Full-trajectory layer-3 VLM AUPRC is 0.996, with layers 7/11/15 also above 0.990. Action-expert separation is strongest at layer 3 (AUPRC 0.940) and weaker in later layers (final-layer AUPRC 0.739).

Task- and progress-bin-matched successful centroids were available for 32 of the 42 failed episodes. Task 5 had no successful episodes, so its ten failures are correctly excluded from reference-based divergence rather than assigned a fabricated reference.

Using two consecutive samples beyond the successful-centroid 95th-percentile distance:

- VLM representations diverged in 29/32 referenceable failures, at median normalized progress 0.143.
- Action-expert representations diverged in 27–28/32, depending on layer, at median progress 0.215–0.242.
- Policy/action-output features diverged in 27/32 at median progress 0.358.

This ordering is consistent with failure-related information appearing earlier in VLM representations than in action outputs. It is an observational association, not a causal intervention result.

## Warning trade-off

For the VLM-final probe, declaring an episode warned when any sampled step crosses a probability threshold produces:

| Threshold | Successful episodes with any false alarm | Failed episodes detected | Median steps before timeout termination |
|---:|---:|---:|---:|
| 0.10 | 94.8% | 100% | 280.0 |
| 0.30 | 77.6% | 100% | 262.5 |
| 0.50 | 60.3% | 100% | 247.5 |
| 0.70 | 34.5% | 100% | 230.0 |
| 0.90 | 12.1% | 100% | 200.0 |

These are warning-to-termination measurements, not lead time to true failure onset. Manual onset annotation is still required for that claim. The high any-step false-alarm rate also shows that a production warning system needs temporal smoothing, calibration, and an independently collected test set.

## Conclusions

Within this 100-episode clean Spatial pilot:

1. VLM hidden states provide a stronger failure signal than action-expert states, policy outputs, or sampled action uncertainty.
2. Useful fixed-horizon VLM discrimination emerges around step 50 and strengthens later.
3. Task-matched VLM representation divergence precedes action-output divergence in referenceable failures.
4. Full-trajectory discrimination is excellent, but raw per-step warning thresholds create too many false alarms.
5. These results support continuing to matched perturbation and multi-suite validation; they do not yet establish causality or deployment-grade generalization.

## Artifacts

- `summaries/probe_metrics.parquet`
- `summaries/probe_predictions.parquet`
- `summaries/activation_summary.parquet`
- `summaries/activation_centroid_metrics.parquet`
- `summaries/representation_divergence.parquet`
- `summaries/hidden_pca.parquet`
- `summaries/lead_time_false_alarm_curve.parquet`
- `summaries/hidden_state_analysis_manifest.json`
- hidden-state figures and provenance entries in `plots/plot_manifest.csv`

## Reproduction

```bash
.conda-envs/smolvla-libero/bin/python \
  experiments/smolvla_libero_failure_analysis/scripts/analyze_hidden_states.py \
  --run experiments/smolvla_libero_failure_analysis/runs/pilot_spatial_100_20260722T060211Z_63ee3442

MPLCONFIGDIR=/tmp/smolvla-hidden-plots \
.conda-envs/smolvla-libero/bin/python \
  experiments/smolvla_libero_failure_analysis/scripts/plot_hidden_states.py \
  --run experiments/smolvla_libero_failure_analysis/runs/pilot_spatial_100_20260722T060211Z_63ee3442
```
