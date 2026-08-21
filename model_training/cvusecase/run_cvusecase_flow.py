"""
Single-entry flow for the cv use-case training/evaluation pipeline.

Modes:
  full  : train + evaluate (default)
  train : train only
  eval  : evaluate only

This is intended for local and Azure usage where users want one stable path.
"""
import os
import sys
import argparse
import subprocess

BASE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(BASE, "..", "cvusecase_dataset")
DEFAULT_OUT = os.path.join(BASE, "cvusecase_outputs")


def _run(cmd):
    print("\n$", " ".join(cmd), "\n")
    subprocess.run(cmd, check=True)


def _parse_optional_float(value):
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    return float(value)


def run_train(data_dir, model_path):
    _run([
        sys.executable,
        os.path.join(BASE, "train_cvusecase_model.py"),
        "--data_path", data_dir,
        "--model_out", model_path,
    ])


def run_eval(data_dir, model_path, json_path, report_path, target_acc_low, target_acc_high):
    cmd = [
        sys.executable,
        os.path.join(BASE, "evaluate_cvusecase_model.py"),
        model_path,
        "--data", data_dir,
        "--json", json_path,
        "--report", report_path,
    ]
    if target_acc_low is not None:
        cmd += ["--target-acc-low", str(target_acc_low)]
    if target_acc_high is not None:
        cmd += ["--target-acc-high", str(target_acc_high)]
    _run(cmd)


def main():
    ap = argparse.ArgumentParser(description="Run the cv use-case training/evaluation flow")
    ap.add_argument("--mode", default="full", choices=["full", "train", "eval"])
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--output", default=DEFAULT_OUT)
    ap.add_argument("--model", default=None,
                    help="model output/input path; defaults to <output>/cvusecase_best.hdf5")
    ap.add_argument("--json", default=None,
                    help="metrics json path; defaults to <output>/eval_metrics.json")
    ap.add_argument("--report", default=None,
                    help="report txt path; defaults to <output>/report.txt")
    ap.add_argument("--target_acc_low", default=None)
    ap.add_argument("--target_acc_high", default=None)
    args = ap.parse_args()

    target_acc_low = _parse_optional_float(args.target_acc_low)
    target_acc_high = _parse_optional_float(args.target_acc_high)

    os.makedirs(args.output, exist_ok=True)
    model_path = args.model or os.path.join(args.output, "cvusecase_best.hdf5")
    json_path = args.json or os.path.join(args.output, "eval_metrics.json")
    report_path = args.report or os.path.join(args.output, "report.txt")

    if args.mode == "full":
        run_train(args.data, model_path)
        run_eval(args.data, model_path, json_path, report_path, target_acc_low, target_acc_high)
    elif args.mode == "train":
        run_train(args.data, model_path)
    elif args.mode == "eval":
        run_eval(args.data, model_path, json_path, report_path, target_acc_low, target_acc_high)


if __name__ == "__main__":
    main()
