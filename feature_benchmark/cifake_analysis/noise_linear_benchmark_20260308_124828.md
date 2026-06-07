# Noise Feature Linear Benchmark

Generated: 2026-03-08 12:48:28
Dataset: cifake_analysis/cifake_features_noise.npz

## Setup

- Model: LogisticRegression (linear)
- Pipeline: StandardScaler + LogisticRegression
- Tuned: True
- Best params: {'clf__C': 8.0, 'clf__solver': 'liblinear'}
- Best CV ROC-AUC (train only): 0.942948

## Results

- Samples: 120000
- Feature dim: 63
- Feature sources: ['cifake_features_noise.npz', 'cifake_features_texture.npz']
- Label mapping: ['0=REAL', '1=FAKE']
- Train/Test: 84000/36000
- Accuracy: 0.871361
- Precision: 0.877976
- Recall: 0.862611
- F1: 0.870226
- ROC-AUC: 0.943243
- Train time (s): 0.0000
- Infer time (s): 0.0254
- Confusion matrix: [[15842, 2158], [2473, 15527]]
