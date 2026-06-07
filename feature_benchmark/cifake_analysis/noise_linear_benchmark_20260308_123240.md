# Noise Feature Linear Benchmark

Generated: 2026-03-08 12:32:40
Dataset: cifake_analysis/cifake_features_noise.npz

## Setup

- Model: LogisticRegression (linear)
- Pipeline: StandardScaler + LogisticRegression
- Solver: liblinear

## Results

- Samples: 120000
- Feature dim: 48
- Label mapping: ['0=REAL', '1=FAKE']
- Train/Test: 84000/36000
- Accuracy: 0.855444
- Precision: 0.863110
- Recall: 0.844889
- F1: 0.853902
- ROC-AUC: 0.929247
- Train time (s): 4.2964
- Infer time (s): 0.0166
- Confusion matrix: [[15588, 2412], [2792, 15208]]
