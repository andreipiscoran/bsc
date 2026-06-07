# CIFAKE Dataset Analysis Report

Generated: 2026-03-01 16:41:50
Dataset location: ./cifake_data
Samples analyzed: 5000

## Summary Table

| Feature Type | Cov Diff (Frob) | Cov Diff (Spec) | KL Div | Mean KS | Sig@0.05 | Hotelling T2 |
|-------------|-----------------|------------------|--------|---------|----------|--------------|
| Dct | 4328.6707 | 3474.4153 | 4.9938 | 0.1685 | 10 | 4517.9018 |
| Intensity | 2582.4770 | 2578.9744 | 1.2577 | 0.1081 | 13 | 2442.2498 |
| Color | 5229.3363 | 3746.9050 | 5.2614 | 0.1364 | 24 | 6437.9995 |
| Texture | 103091.8548 | 103091.6389 | 1.8099 | 0.1368 | 15 | 9027.2359 |
| Wavelet | 8865.3174 | 8845.9327 | 1.4080 | 0.1218 | 12 | 2788.1037 |

## Detailed Results

### Dct Features

```
covariance_difference_frobenius: 4328.670704
covariance_difference_spectral: 3474.415347
kl_divergence                 : 4.993815
ks_stat_mean                  : 0.168520
ks_stat_std                   : 0.050579
ks_pvalue_mean                : 0.000000
features_significant_005      : 10
features_significant_001      : 10
hotelling_t2                  : 4517.901797
hotelling_f_stat              : 451.586854
```

### Intensity Features

```
covariance_difference_frobenius: 2582.476959
covariance_difference_spectral: 2578.974445
kl_divergence                 : 1.257718
ks_stat_mean                  : 0.108146
ks_stat_std                   : 0.045511
ks_pvalue_mean                : 0.000000
features_significant_005      : 13
features_significant_001      : 13
hotelling_t2                  : 2442.249763
hotelling_f_stat              : 187.752636
```

### Color Features

```
covariance_difference_frobenius: 5229.336281
covariance_difference_spectral: 3746.905016
kl_divergence                 : 5.261439
ks_stat_mean                  : 0.136433
ks_stat_std                   : 0.056142
ks_pvalue_mean                : 0.000001
features_significant_005      : 24
features_significant_001      : 24
hotelling_t2                  : 6437.999462
hotelling_f_stat              : 267.941459
```

### Texture Features

```
covariance_difference_frobenius: 103091.854818
covariance_difference_spectral: 103091.638943
kl_divergence                 : 1.809872
ks_stat_mean                  : 0.136753
ks_stat_std                   : 0.053782
ks_pvalue_mean                : 0.000000
features_significant_005      : 15
features_significant_001      : 15
hotelling_t2                  : 9027.235942
hotelling_f_stat              : 601.394416
```

### Wavelet Features

```
covariance_difference_frobenius: 8865.317365
covariance_difference_spectral: 8845.932698
kl_divergence                 : 1.408021
ks_stat_mean                  : 0.121767
ks_stat_std                   : 0.070636
ks_pvalue_mean                : 0.000412
features_significant_005      : 12
features_significant_001      : 11
hotelling_t2                  : 2788.103693
hotelling_f_stat              : 232.214174
```

## Interpretation

**Most discriminative feature type:** Color (24 significant features at α=0.05)

### Feature Ranking by Discriminative Power

1. **Color**: 24 significant features
2. **Texture**: 15 significant features
3. **Intensity**: 13 significant features
4. **Wavelet**: 12 significant features
5. **Dct**: 10 significant features
