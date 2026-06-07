# Cross-Dataset Benchmark: Train on CIFAKE, Test on FF++

Generated: 2026-04-15 18:53:03

## Data

- Train NPZ(s): ['cifake_analysis/cifake_features_noise.npz']
- Test NPZ(s): ['cifake_analysis/faceforensicsplusplus_features_noise.npz']
- Train samples: 120000
- Test samples: 400
- Feature dim: 48

## Metrics

- Class weight: None
- Decision threshold: 0.5000 (fixed)
- Accuracy: 0.500000
- Balanced accuracy: 0.500000
- Precision: 0.500000
- Recall: 0.900000
- F1: 0.642857
- ROC-AUC: 0.524425
- MCC: 0.000000
- Predicted fake rate: 0.900000
- Confusion matrix [ [TN, FP], [FN, TP] ]: [[20, 180], [20, 180]]

## Label Flip Sanity Check

(Evaluates metrics with test labels inverted: 0↔1. Useful to detect mapping mismatch.)

- Accuracy (flipped): 0.500000
- Balanced accuracy (flipped): 0.500000
- ROC-AUC (flipped): 0.475575
- Confusion matrix flipped [ [TN, FP], [FN, TP] ]: [[20, 180], [20, 180]]
