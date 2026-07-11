# Robustness and Ablation

Protocol: 10% adaptation / disjoint 10% calibration / 80% test by problem_id.

## Multi-seed summary

| Metric | Mean | Std | Min | Max |
| --- | ---: | ---: | ---: | ---: |
| label_macro_f1 | 0.5565 | 0.0257 | 0.5169 | 0.5875 |
| incorrect_recall | 0.1463 | 0.0549 | 0.0731 | 0.2243 |
| first_error_accuracy | 0.3995 | 0.0120 | 0.3800 | 0.4100 |
| first_error_macro_f1 | 0.1710 | 0.0546 | 0.0952 | 0.2265 |

## Cumulative ablation (seed 42)

| Variant | Label macro-F1 | Incorrect recall | First-error accuracy |
| --- | ---: | ---: | ---: |
| binary_current_only_with_position | 0.5642 | 0.5483 | 0.3588 |
| local_context | 0.5620 | 0.5777 | 0.3412 |
| remove_position_features | 0.5448 | 0.6471 | 0.3100 |
| soft_symbolic_features | 0.5458 | 0.6492 | 0.3113 |
| deterministic_symbolic_overrides | 0.5425 | 0.6639 | 0.3013 |
| target_adaptation_10_calibration_10 | 0.5605 | 0.1408 | 0.4050 |
