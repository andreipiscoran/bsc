import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_and_validate_features(npz_path: Path):
    """Load feature matrix and validate labels."""
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    
    unique_labels = np.unique(y)
    if unique_labels.shape[0] != 2 or not np.array_equal(np.sort(unique_labels), np.array([0.0, 1.0])):
        raise ValueError(f"Expected binary labels {{0,1}}, found: {unique_labels}")
    
    if "label_mapping" in data:
        label_mapping = [str(item) for item in data["label_mapping"].tolist()]
    else:
        label_mapping = ["0=REAL", "1=FAKE"]
    
    return X, y, label_mapping


def run_benchmark(
    npz_path: Path,
    extra_npz_paths: list,
    test_size: float,
    random_state: int,
    max_iter: int,
    tune: bool = False,
):
    # Load primary features
    X_base, y, label_mapping = load_and_validate_features(npz_path)
    feature_sources = [str(npz_path.name)]
    
    # Optionally combine with additional features
    X_parts = [X_base]
    for extra_path in extra_npz_paths:
        X_extra, y_extra, _ = load_and_validate_features(extra_path)
        if len(y_extra) != len(y) or not np.array_equal(y_extra, y):
            raise ValueError(
                f"Label mismatch between {npz_path} and {extra_path}. "
                "Ensure features were exported from the same run."
            )
        X_parts.append(X_extra)
        feature_sources.append(str(extra_path.name))
    
    X = np.hstack(X_parts) if len(X_parts) > 1 else X_base

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    
    # Optionally tune hyperparameters on train set only
    if tune:
        search_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=max_iter, random_state=random_state)),
        ])
        param_grid = [
            {
                "clf__solver": ["liblinear"],
                "clf__C": [0.5, 1.0, 2.0, 4.0, 8.0],
            },
            {
                "clf__solver": ["lbfgs"],
                "clf__C": [0.5, 1.0, 2.0, 4.0, 8.0],
            },
        ]
        search = GridSearchCV(
            estimator=search_pipeline,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=5,
            n_jobs=-1,
            refit=True,
            verbose=1,
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
        best_params = search.best_params_
        best_cv_score = float(search.best_score_)
    else:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=max_iter, solver="liblinear", random_state=random_state)),
        ])
        best_params = {"clf__solver": "liblinear", "clf__C": 1.0}
        best_cv_score = None

    train_start = time.perf_counter()
    if not tune:  # Already fitted if tuned
        model.fit(X_train, y_train)
    train_time = time.perf_counter() - train_start

    infer_start = time.perf_counter()
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    infer_time = time.perf_counter() - infer_start

    metrics = {
        "samples": int(len(y)),
        "feature_dim": int(X.shape[1]),
        "feature_sources": feature_sources,
        "label_mapping": label_mapping,
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
        "tuned": bool(tune),
        "best_params": best_params,
        "best_cv_roc_auc": best_cv_score,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "train_time_sec": float(train_time),
        "infer_time_sec": float(infer_time),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    return metrics


def write_markdown_report(metrics: dict, npz_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"noise_linear_benchmark_{timestamp}.md"

    with out_path.open("w", encoding="utf-8") as f:
        f.write("# Noise Feature Linear Benchmark\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {npz_path}\n\n")

        f.write("## Setup\n\n")
        f.write("- Model: LogisticRegression (linear)\n")
        f.write("- Pipeline: StandardScaler + LogisticRegression\n")
        f.write(f"- Tuned: {metrics['tuned']}\n")
        f.write(f"- Best params: {metrics['best_params']}\n")
        if metrics["best_cv_roc_auc"] is not None:
            f.write(f"- Best CV ROC-AUC (train only): {metrics['best_cv_roc_auc']:.6f}\n")
        f.write("\n")

        f.write("## Results\n\n")
        f.write(f"- Samples: {metrics['samples']}\n")
        f.write(f"- Feature dim: {metrics['feature_dim']}\n")
        f.write(f"- Feature sources: {metrics['feature_sources']}\n")
        f.write(f"- Label mapping: {metrics['label_mapping']}\n")
        f.write(f"- Train/Test: {metrics['train_size']}/{metrics['test_size']}\n")
        f.write(f"- Accuracy: {metrics['accuracy']:.6f}\n")
        f.write(f"- Precision: {metrics['precision']:.6f}\n")
        f.write(f"- Recall: {metrics['recall']:.6f}\n")
        f.write(f"- F1: {metrics['f1']:.6f}\n")
        f.write(f"- ROC-AUC: {metrics['roc_auc']:.6f}\n")
        f.write(f"- Train time (s): {metrics['train_time_sec']:.4f}\n")
        f.write(f"- Infer time (s): {metrics['infer_time_sec']:.4f}\n")
        f.write(f"- Confusion matrix: {metrics['confusion_matrix']}\n")

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Benchmark a linear model on CIFAKE features.")
    parser.add_argument(
        "--npz",
        default="cifake_analysis/cifake_features_noise.npz",
        help="Path to primary feature npz file.",
    )
    parser.add_argument(
        "--extra-npz",
        nargs="*",
        default=[],
        help="Additional feature npz files to concatenate (e.g., texture, fft).",
    )
    parser.add_argument(
        "--noise-texture",
        action="store_true",
        help="Shortcut: combine noise + texture features.",
    )
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--tune", action="store_true", help="Enable CV hyperparameter tuning.")
    parser.add_argument("--report-dir", default="cifake_analysis")
    args = parser.parse_args()

    npz_path = Path(args.npz)
    if not npz_path.exists():
        raise FileNotFoundError(f"Feature file not found: {npz_path}")
    
    # Handle noise-texture shortcut
    extra_npz_paths = [Path(p) for p in args.extra_npz]
    if args.noise_texture:
        texture_path = Path("cifake_analysis/cifake_features_texture.npz")
        if texture_path.exists() and texture_path not in extra_npz_paths:
            extra_npz_paths.append(texture_path)
        else:
            print(f"Warning: --noise-texture specified but {texture_path} not found or already included.")
    
    for p in extra_npz_paths:
        if not p.exists():
            raise FileNotFoundError(f"Extra feature file not found: {p}")

    metrics = run_benchmark(
        npz_path,
        extra_npz_paths,
        args.test_size,
        args.seed,
        args.max_iter,
        tune=args.tune,
    )

    print("=== Linear Model Benchmark (Logistic Regression) ===")
    print(f"Primary dataset: {npz_path}")
    print(f"Feature sources: {metrics['feature_sources']}")
    print(f"Label mapping: {metrics['label_mapping']}")
    print(f"Samples: {metrics['samples']} | Feature dim: {metrics['feature_dim']}")
    print(f"Tuned: {metrics['tuned']} | Best params: {metrics['best_params']}")
    if metrics["best_cv_roc_auc"] is not None:
        print(f"Best CV ROC-AUC (train only): {metrics['best_cv_roc_auc']:.6f}")
    print(f"Train/Test split: {metrics['train_size']}/{metrics['test_size']}")
    print(f"Accuracy:  {metrics['accuracy']:.6f}")
    print(f"Precision: {metrics['precision']:.6f}")
    print(f"Recall:    {metrics['recall']:.6f}")
    print(f"F1-score:  {metrics['f1']:.6f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.6f}")
    print(f"Train time (s): {metrics['train_time_sec']:.4f}")
    print(f"Infer time (s): {metrics['infer_time_sec']:.4f}")
    print("Confusion matrix [ [TN, FP], [FN, TP] ]:")
    print(np.array(metrics["confusion_matrix"]))

    report_path = write_markdown_report(metrics, npz_path=npz_path, output_dir=Path(args.report_dir))
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
