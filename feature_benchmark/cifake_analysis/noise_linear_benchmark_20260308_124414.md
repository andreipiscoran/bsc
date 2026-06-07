# Noise Feature Linear Benchmark

Generated: 2026-03-08 12:44:14
Dataset: cifake_analysis/cifake_features_noise.npz

## Setup

- Model: LogisticRegression (linear)
- Pipeline: StandardScaler + LogisticRegression
- Tuned: True
- Best params: {'clf__C': 8.0, 'clf__solver': 'liblinear'}
- Best CV ROC-AUC (train only): 0.929921

## Results

- Samples: 120000
- Feature dim: 48
- Feature sources: ['cifake_analysis/cifake_features_noise.npz']
- Label mapping: ['0=REAL', '1=FAKE']
- Train/Test: 84000/36000
- Accuracy: 0.855500
- Precision: 0.863002
- Recall: 0.845167
- F1: 0.853991
- ROC-AUC: 0.929426
- Train time (s): 46.1067
- Infer time (s): 0.1163
- Confusion matrix: [[15585, 2415], [2787, 15213]]
