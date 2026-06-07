# Noise Feature Linear Benchmark

Generated: 2026-03-08 12:48:40
Dataset: cifake_analysis/cifake_features_noise.npz

## Setup

- Model: LogisticRegression (linear)
- Pipeline: StandardScaler + LogisticRegression
- Tuned: False
- Best params: {'clf__solver': 'liblinear', 'clf__C': 1.0}

## Results

- Samples: 120000
- Feature dim: 63
- Feature sources: ['cifake_features_noise.npz', 'cifake_features_texture.npz']
- Label mapping: ['0=REAL', '1=FAKE']
- Train/Test: 84000/36000
- Accuracy: 0.871778
- Precision: 0.878336
- Recall: 0.863111
- F1: 0.870657
- ROC-AUC: 0.943116
- Train time (s): 4.4412
- Infer time (s): 0.0232
- Confusion matrix: [[15848, 2152], [2464, 15536]]
