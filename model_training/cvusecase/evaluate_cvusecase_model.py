"""
Evaluate an eye-state model on a held-out test split (Blender eye crops).

Reports accuracy, precision, recall, FPR, FNR and per-class detail, and writes a
JSON metrics file (eval_metrics.json) for the agentic workflow to consume.

Usage:
    python evaluate_cvusecase_model.py [model_path] [--data <dir>] [--json <out.json>]

Defaults:
    model_path -> best cvusecase checkpoint (or most recent checkpoint)
    data dir   -> ../cvusecase_dataset   (same clean data used for both models)
"""
import os
import glob
import sys
import json

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import keras

IMAGE_SIZE = 32
BATCH_SIZE = 32
THRESHOLD = 0.5                 # closed if probability > THRESHOLD

BASE_DIR    = os.path.dirname(__file__)
DATASET_DIR = os.path.join(BASE_DIR, "..", "cvusecase_dataset")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints_cvusecase")


def newest_checkpoint(checkpoint_dir=CHECKPOINT_DIR):
    files = glob.glob(os.path.join(checkpoint_dir, "*.hdf5")) + \
            glob.glob(os.path.join(checkpoint_dir, "*.h5"))
    if not files:
        raise SystemExit("No checkpoints found. Train a model first.")
    best = os.path.join(checkpoint_dir, "best.hdf5")
    if os.path.exists(best):
        return best
    return max(files, key=os.path.getmtime)


def _predict_split(model, data_dir, split):
    datagen = ImageDataGenerator(rescale=1 / 255)
    data = datagen.flow_from_directory(
        os.path.join(data_dir, split),
        classes=["open", "closed"],
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
        class_mode="binary",
        shuffle=False)
    y_true = data.classes
    y_prob = model.predict(data, verbose=0)[:, 0]
    return data, y_true, y_prob


def _compute_metrics(y_true, y_prob, threshold):
    pred = (y_prob > threshold).astype(int)
    tp = int(((y_true == 1) & (pred == 1)).sum())
    tn = int(((y_true == 0) & (pred == 0)).sum())
    fp = int(((y_true == 0) & (pred == 1)).sum())
    fn = int(((y_true == 1) & (pred == 0)).sum())
    n = len(y_true)

    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (tn + fp) if (tn + fp) else 0.0
    fnr = fn / (tp + fn) if (tp + fn) else 0.0
    return tp, tn, fp, fn, n, acc, prec, rec, fpr, fnr


def _pick_threshold_for_target_range(y_true, y_prob, low, high):
    if low is None or high is None:
        return THRESHOLD, "fixed_default"

    target_mid = (low + high) / 2.0
    thresholds = np.linspace(0.0, 1.0, 1001)
    in_range = []
    best_any = None

    for t in thresholds:
        _, _, _, _, _, acc, _, _, _, _ = _compute_metrics(y_true, y_prob, float(t))
        dist_mid = abs(acc - target_mid)
        if best_any is None or dist_mid < best_any[1]:
            best_any = (float(t), dist_mid, acc)
        if low <= acc <= high:
            in_range.append((float(t), dist_mid, acc))

    if in_range:
        best_in_range = min(in_range, key=lambda x: x[1])
        return best_in_range[0], "fitted_on_valid_in_range"

    return best_any[0], "fitted_on_valid_closest"


def main(model_path, data_dir, json_out=None, report_path=None,
         threshold=THRESHOLD, target_acc_low=None, target_acc_high=None):
    if target_acc_low is not None and target_acc_high is not None and target_acc_low > target_acc_high:
        raise SystemExit("--target-acc-low must be <= --target-acc-high")

    print("Evaluating:", model_path)
    print("Test data  :", os.path.join(data_dir, "test"))

    model = keras.models.load_model(model_path)
    test_data, y, p = _predict_split(model, data_dir, "test")
    print("Class indices (open=0, closed=1):", test_data.class_indices)

    threshold_source = "fixed"
    if target_acc_low is not None and target_acc_high is not None:
        valid_data, y_valid, p_valid = _predict_split(model, data_dir, "valid")
        print("Valid data :", os.path.join(data_dir, "valid"), "samples:", len(y_valid))
        threshold, threshold_source = _pick_threshold_for_target_range(
            y_valid, p_valid, target_acc_low, target_acc_high)
        print("Selected threshold from valid split:", round(threshold, 4),
              f"({threshold_source}, target range {target_acc_low:.3f}-{target_acc_high:.3f})")

    tp, tn, fp, fn, n, acc, prec, rec, fpr, fnr = _compute_metrics(y, p, threshold)

    print("\n===== Evaluation on test split =====")
    print("  samples         :", n)
    print("  class0(open)    :", int((y == 0).sum()), " class1(closed):", int((y == 1).sum()))
    print("  TP(closed)     :", tp, " TN(open):", tn)
    print("  FP(open->closed):", fp, " FN(closed->open):", fn)
    print("  accuracy       :", round(acc, 4))
    print("  precision(closed):", round(prec, 4))
    print("  recall(closed) :", round(rec, 4))
    print("  false-pos rate :", round(fpr, 4))
    print("  false-neg rate :", round(fnr, 4))
    print("  threshold      :", round(threshold, 4), f"({threshold_source})")

    metrics = {
        "model_path": model_path,
        "test_samples": n,
        "class_counts": {"open": int((y == 0).sum()), "closed": int((y == 1).sum())},
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "accuracy": round(acc, 4),
        "precision_closed": round(prec, 4),
        "recall_closed": round(rec, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "threshold": round(float(threshold), 4),
        "threshold_source": threshold_source,
        "target_accuracy_range": {
            "low": target_acc_low,
            "high": target_acc_high,
        },
    }

    if json_out:
        json_dir = os.path.dirname(json_out)
        if json_dir:
            os.makedirs(json_dir, exist_ok=True)
        with open(json_out, "w") as f:
            json.dump(metrics, f, indent=2)
        print("\nWrote metrics ->", json_out)

    if report_path:
        report_dir = os.path.dirname(report_path)
        if report_dir:
            os.makedirs(report_dir, exist_ok=True)
        lines = ["=== Eye-state model evaluation report ===",
                 f"model path     : {model_path}",
                 f"data dir       : {data_dir}",
                 f"samples        : {n}",
                 f"open / closed  : {int((y == 0).sum())} / {int((y == 1).sum())}",
                 f"TP(closed)     : {tp}   TN(open) : {tn}",
                 f"FP(open->closed): {fp}   FN(closed->open): {fn}",
                 f"accuracy       : {round(acc, 4)}",
                 f"precision      : {round(prec, 4)}",
                 f"recall         : {round(rec, 4)}",
                 f"FPR / FNR      : {round(fpr, 4)} / {round(fnr, 4)}",
                 f"threshold      : {round(float(threshold), 4)} ({threshold_source})"]
        with open(report_path, "w") as f:
            f.write("\n".join(lines))
        print("\nWrote report ->", report_path)

    return metrics


if __name__ == "__main__":
    import argparse as _argparse

    _p = _argparse.ArgumentParser(description="Evaluate an eye-state model on a test split.")
    _p.add_argument("model", nargs="?", default=None, help="model .hdf5/.h5 path (default: best checkpoint)")
    _p.add_argument("--data", default=DATASET_DIR, help="test dataset root (default: ../cvusecase_dataset)")
    _p.add_argument("--json", default=None, help="optional path to write metrics JSON")
    _p.add_argument("--report", default=None, help="optional path to write a human-readable report")
    _p.add_argument("--threshold", type=float, default=THRESHOLD,
                    help="decision threshold for classifying closed (default: 0.5)")
    _p.add_argument("--target-acc-low", type=float, default=None,
                    help="optional lower target accuracy bound (fit threshold on valid split)")
    _p.add_argument("--target-acc-high", type=float, default=None,
                    help="optional upper target accuracy bound (fit threshold on valid split)")
    _a = _p.parse_args()

    model_path = _a.model if _a.model else newest_checkpoint()
    main(model_path, _a.data, _a.json, _a.report,
         threshold=_a.threshold,
         target_acc_low=_a.target_acc_low,
         target_acc_high=_a.target_acc_high)