from __future__ import annotations

import cv2
import numpy as np
from scipy.stats import entropy, kurtosis, skew
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from utils.image_processing import normalize_map, to_gray


def compute_fft_features(image: np.ndarray) -> dict:
    gray = to_gray(image).astype(np.float32)
    fft = np.fft.fft2(gray)
    fft_shifted = np.fft.fftshift(fft)

    magnitude = np.abs(fft_shifted)
    log_spectrum = np.log1p(magnitude)

    center = (gray.shape[0] // 2, gray.shape[1] // 2)
    yy, xx = np.indices(gray.shape)
    radius = np.sqrt((yy - center[0]) ** 2 + (xx - center[1]) ** 2).astype(np.int32)
    radial_energy = np.bincount(radius.ravel(), weights=magnitude.ravel())
    radial_count = np.bincount(radius.ravel()) + 1e-8
    radial_profile = radial_energy / radial_count

    return {
        "magnitude": magnitude,
        "log_spectrum": log_spectrum,
        "radial_profile": radial_profile,
        "mean_log_spectrum": float(np.mean(log_spectrum)),
        "std_log_spectrum": float(np.std(log_spectrum)),
    }


def compute_noise_residual(image: np.ndarray, kernel_size: int = 5) -> dict:
    gray = to_gray(image).astype(np.float32)
    denoised = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    residual = gray - denoised
    residual_abs = np.abs(residual)

    return {
        "residual": residual,
        "residual_abs": residual_abs,
        "residual_map": normalize_map(residual_abs),
        "std": float(np.std(residual)),
        "var": float(np.var(residual)),
    }


def compute_texture_features(image: np.ndarray) -> dict:
    gray = to_gray(image)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")

    grad_x = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)

    return {
        "lbp": lbp,
        "gradient_magnitude": grad_mag,
        "lbp_hist": np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)[0],
        "gradient_mean": float(np.mean(grad_mag)),
    }


def compute_glcm_features(image: np.ndarray) -> dict:
    gray = to_gray(image)
    glcm = graycomatrix(gray, distances=[1], angles=[0, np.pi / 4], levels=256, symmetric=True, normed=True)
    return {
        "contrast": float(graycoprops(glcm, "contrast").mean()),
        "correlation": float(graycoprops(glcm, "correlation").mean()),
        "energy": float(graycoprops(glcm, "energy").mean()),
        "homogeneity": float(graycoprops(glcm, "homogeneity").mean()),
    }


def compute_gabor_response(image: np.ndarray) -> np.ndarray:
    gray = to_gray(image).astype(np.float32)
    kernel = cv2.getGaborKernel((17, 17), sigma=4.0, theta=np.pi / 4, lambd=8.0, gamma=0.5, psi=0)
    response = cv2.filter2D(gray, cv2.CV_32F, kernel)
    return response


def compute_patch_anomaly_map(image: np.ndarray, patch_size: int = 32) -> dict:
    gray = to_gray(image).astype(np.float32)
    height, width = gray.shape

    rows = int(np.ceil(height / patch_size))
    cols = int(np.ceil(width / patch_size))

    anomaly_grid = np.zeros((rows, cols), dtype=np.float32)

    for row in range(rows):
        for col in range(cols):
            y0, y1 = row * patch_size, min((row + 1) * patch_size, height)
            x0, x1 = col * patch_size, min((col + 1) * patch_size, width)
            patch = gray[y0:y1, x0:x1]

            patch_blur = cv2.GaussianBlur(patch, (5, 5), 0)
            residual = patch - patch_blur
            residual_score = float(np.std(residual))

            patch_fft = np.fft.fftshift(np.fft.fft2(patch))
            magnitude = np.log1p(np.abs(patch_fft))
            freq_score = float(np.mean(magnitude))

            gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
            grad_score = float(np.std(cv2.magnitude(gx, gy)))

            anomaly_grid[row, col] = 0.45 * residual_score + 0.35 * freq_score + 0.20 * grad_score

    anomaly_grid = normalize_map(anomaly_grid)
    anomaly_map = cv2.resize(anomaly_grid, (width, height), interpolation=cv2.INTER_CUBIC)
    anomaly_map = normalize_map(anomaly_map)

    return {
        "anomaly_grid": anomaly_grid,
        "anomaly_map": anomaly_map,
        "rows": rows,
        "cols": cols,
    }


def compute_color_statistics(image: np.ndarray) -> dict:
    channels = [image[..., idx].astype(np.float32) for idx in range(3)]
    means = [float(np.mean(channel)) for channel in channels]
    stds = [float(np.std(channel)) for channel in channels]

    flattened = np.stack([channel.ravel() for channel in channels], axis=0)
    corr = np.corrcoef(flattened)

    return {"means": means, "stds": stds, "correlation": corr}


def compute_intensity_statistics(image: np.ndarray) -> dict:
    gray = to_gray(image).astype(np.float32)
    flat = gray.ravel()

    hist, bin_edges = np.histogram(flat, bins=256, range=(0, 255), density=True)
    hist_safe = hist + 1e-12

    return {
        "mean": float(np.mean(flat)),
        "variance": float(np.var(flat)),
        "skewness": float(skew(flat)),
        "kurtosis": float(kurtosis(flat)),
        "entropy": float(entropy(hist_safe, base=2)),
        "histogram": hist,
        "bins": bin_edges,
    }
