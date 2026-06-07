import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_features(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    if "X" not in data or "y" not in data:
        raise ValueError(f"{npz_path} must contain 'X' and 'y' arrays")

    X = np.asarray(data["X"])
    y = np.asarray(data["y"]).astype(float)

    y_unique = np.unique(y)
    if y_unique.shape[0] != 2 or not np.array_equal(np.sort(y_unique), np.array([0.0, 1.0])):
        raise ValueError(f"{npz_path} labels must be binary {{0,1}}, found: {y_unique}")

    return X, y


def load_and_stack(npz_paths: list[Path]):
    if not npz_paths:
        raise ValueError("No npz paths provided")

    X_parts = []
    y_ref = None

    for path in npz_paths:
        if not path.exists():
            raise FileNotFoundError(f"Feature file not found: {path}")

        X, y = load_features(path)
        if y_ref is None:
            y_ref = y
        else:
            if len(y) != len(y_ref) or not np.array_equal(y, y_ref):
                raise ValueError(
                    f"Label mismatch in {path}. When stacking features, files must come from the same export run."
                )

        X_parts.append(X)

    X_stacked = np.hstack(X_parts) if len(X_parts) > 1 else X_parts[0]
    return X_stacked, y_ref


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float):
    y_pred = (y_prob >= threshold).astype(float)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0.0, 1.0]).ravel()

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "predicted_fake_rate": float(np.mean(y_pred)),
        "threshold": float(threshold),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray):
    candidates = np.linspace(0.1, 0.9, 81)
    best_threshold = 0.5
    best_score = -1.0

    for thr in candidates:
        y_pred = (y_prob >= thr).astype(float)
        score = balanced_accuracy_score(y_true, y_pred)
        if score > best_score:
            best_score = float(score)
            best_threshold = float(thr)

    return best_threshold, best_score


def evaluate(
    train_npzs: list[Path],
    test_npzs: list[Path],
    max_iter: int,
    seed: int,
    class_weight: str | None,
    threshold: float | None,
    tune_threshold: bool,
    diagnose_label_flip: bool,
):
    X_train, y_train = load_and_stack(train_npzs)
    X_test, y_test = load_and_stack(test_npzs)

    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            f"Train/Test feature dimension mismatch: train={X_train.shape[1]}, test={X_test.shape[1]}"
        )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=max_iter,
                    solver="liblinear",
                    random_state=seed,
                    class_weight=class_weight,
                ),
            ),
        ]
    )

    selected_threshold = 0.5 if threshold is None else float(threshold)
    threshold_source = "fixed"

    if tune_threshold:
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            random_state=seed,
            stratify=y_train,
        )
        model.fit(X_fit, y_fit)
        y_val_prob = model.predict_proba(X_val)[:, 1]
        selected_threshold, _ = find_best_threshold(y_val, y_val_prob)
        threshold_source = "train-validation"
        model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    base_metrics = compute_metrics(y_test, y_prob, selected_threshold)

    metrics = {
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "feature_dim": int(X_train.shape[1]),
        "class_weight": class_weight,
        "threshold_source": threshold_source,
        **base_metrics,
    }

    if diagnose_label_flip:
        flipped_y = 1.0 - y_test
        metrics["label_flip_check"] = compute_metrics(flipped_y, y_prob, selected_threshold)

    return metrics


def write_report(metrics: dict, train_npzs: list[Path], test_npzs: list[Path], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"cross_dataset_cifake_to_ffpp_{timestamp}.md"

    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Cross-Dataset Benchmark: Train on CIFAKE, Test on FF++\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Data\n\n")
        f.write(f"- Train NPZ(s): {[str(p) for p in train_npzs]}\n")
        f.write(f"- Test NPZ(s): {[str(p) for p in test_npzs]}\n")
        f.write(f"- Train samples: {metrics['train_samples']}\n")
        f.write(f"- Test samples: {metrics['test_samples']}\n")
        f.write(f"- Feature dim: {metrics['feature_dim']}\n\n")

        f.write("## Metrics\n\n")
        f.write(f"- Class weight: {metrics['class_weight']}\n")
        f.write(f"- Decision threshold: {metrics['threshold']:.4f} ({metrics['threshold_source']})\n")
        f.write(f"- Accuracy: {metrics['accuracy']:.6f}\n")
        f.write(f"- Balanced accuracy: {metrics['balanced_accuracy']:.6f}\n")
        f.write(f"- Precision: {metrics['precision']:.6f}\n")
        f.write(f"- Recall: {metrics['recall']:.6f}\n")
        f.write(f"- F1: {metrics['f1']:.6f}\n")
        f.write(f"- ROC-AUC: {metrics['roc_auc']:.6f}\n")
        f.write(f"- MCC: {metrics['mcc']:.6f}\n")
        f.write(f"- Predicted fake rate: {metrics['predicted_fake_rate']:.6f}\n")
        f.write(f"- Confusion matrix [ [TN, FP], [FN, TP] ]: {metrics['confusion_matrix']}\n")

        if "label_flip_check" in metrics:
            flip = metrics["label_flip_check"]
            f.write("\n## Label Flip Sanity Check\n\n")
            f.write("(Evaluates metrics with test labels inverted: 0↔1. Useful to detect mapping mismatch.)\n\n")
            f.write(f"- Accuracy (flipped): {flip['accuracy']:.6f}\n")
            f.write(f"- Balanced accuracy (flipped): {flip['balanced_accuracy']:.6f}\n")
            f.write(f"- ROC-AUC (flipped): {flip['roc_auc']:.6f}\n")
            f.write(f"- Confusion matrix flipped [ [TN, FP], [FN, TP] ]: {flip['confusion_matrix']}\n")

    return report_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train on CIFAKE features and test on FF++ frame features."
    )
    parser.add_argument(
        "--train-npz",
        nargs="+",
        default=["cifake_analysis/cifake_features_noise.npz"],
        help="One or more CIFAKE feature npz files for training.",
    )
    parser.add_argument(
        "--test-npz",
        nargs="+",
        default=["cifake_analysis/faceforensicsplusplus_features_noise.npz"],
        help="One or more FF++ feature npz files for testing.",
    )
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--class-weight",
        default=None,
        choices=["balanced", "none"],
        help="LogisticRegression class_weight option.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Fixed decision threshold for class 1 (fake). Default uses 0.5 unless --tune-threshold is enabled.",
    )
    parser.add_argument(
        "--tune-threshold",
        action="store_true",
        help="Tune decision threshold on a train-validation split to maximize balanced accuracy.",
    )
    parser.add_argument(
        "--diagnose-label-flip",
        action="store_true",
        help="Report metrics again with test labels flipped to diagnose potential label mapping mismatch.",
    )
    parser.add_argument("--report-dir", default="cifake_analysis")
    return parser.parse_args()


def main():
    args = parse_args()
    train_npzs = [Path(p) for p in args.train_npz]
    test_npzs = [Path(p) for p in args.test_npz]

    metrics = evaluate(
        train_npzs=train_npzs,
        test_npzs=test_npzs,
        max_iter=args.max_iter,
        seed=args.seed,
        class_weight=None if args.class_weight in {None, "none"} else args.class_weight,
        threshold=args.threshold,
        tune_threshold=args.tune_threshold,
        diagnose_label_flip=args.diagnose_label_flip,
    )

    print("=== Cross-Dataset Benchmark ===")
    print("Train: CIFAKE")
    print("Test:  FF++ frames")
    print(f"Train NPZ(s): {[str(p) for p in train_npzs]}")
    print(f"Test NPZ(s):  {[str(p) for p in test_npzs]}")
    print(f"Train/Test samples: {metrics['train_samples']}/{metrics['test_samples']}")
    print(f"Feature dim: {metrics['feature_dim']}")
    print(f"Class weight: {metrics['class_weight']}")
    print(f"Decision threshold: {metrics['threshold']:.4f} ({metrics['threshold_source']})")
    print(f"Accuracy:  {metrics['accuracy']:.6f}")
    print(f"Balanced:  {metrics['balanced_accuracy']:.6f}")
    print(f"Precision: {metrics['precision']:.6f}")
    print(f"Recall:    {metrics['recall']:.6f}")
    print(f"F1-score:  {metrics['f1']:.6f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.6f}")
    print(f"MCC:       {metrics['mcc']:.6f}")
    print(f"Pred fake: {metrics['predicted_fake_rate']:.6f}")
    print("Confusion matrix [ [TN, FP], [FN, TP] ]:")
    print(np.array(metrics["confusion_matrix"]))

    if "label_flip_check" in metrics:
        flip = metrics["label_flip_check"]
        print("\n--- Label flip sanity check (test labels inverted) ---")
        print(f"Accuracy(flipped): {flip['accuracy']:.6f}")
        print(f"Balanced(flipped): {flip['balanced_accuracy']:.6f}")
        print(f"ROC-AUC(flipped):  {flip['roc_auc']:.6f}")
        print(np.array(flip["confusion_matrix"]))

    report_path = write_report(
        metrics=metrics,
        train_npzs=train_npzs,
        test_npzs=test_npzs,
        output_dir=Path(args.report_dir),
    )
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
