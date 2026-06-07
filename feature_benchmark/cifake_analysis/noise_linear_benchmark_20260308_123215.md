# Noise Feature Linear Benchmark

Generated: 2026-03-08 12:32:15
Dataset: cifake_analysis/cifake_features_noise.npz

## Setup

- Model: LogisticRegression (linear)
- Pipeline: StandardScaler + LogisticRegression
- Solver: liblinear

## Results

- Samples: 120000
- Feature dim: 48
- Train/Test: 84000/36000
- Accuracy: 0.855444
- Precision: 0.863110
- Recall: 0.844889
- F1: 0.853902
- ROC-AUC: 0.929247
- Train time (s): 3.8881
- Infer time (s): 0.0183
- Confusion matrix: [[15588, 2412], [2792, 15208]]
