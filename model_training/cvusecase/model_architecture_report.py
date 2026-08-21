"""
Generate a clear text comparison of model architectures:

1) Original repo model architecture (from model_training/training.py)
2) Chosen cvusecase architecture

Output:
    model_training/cvusecase/architecture_comparison.txt

Usage:
    python model_training/cvusecase/model_architecture_report.py
"""

import os
import argparse
import ast


def _shape_str(shape_tuple):
    return "x".join(str(x) for x in shape_tuple)


def _load_cvusecase_cfg_from_source():
    """Read DEFAULT_CFG from train_cvusecase_model.py without importing keras."""
    src_path = os.path.join(os.path.dirname(__file__), "train_cvusecase_model.py")
    with open(src_path, "r", encoding="utf-8") as f:
        source = f.read()

    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "DEFAULT_CFG":
                    return ast.literal_eval(node.value)
    raise RuntimeError("DEFAULT_CFG not found in train_cvusecase_model.py")


def _original_spec():
    return {
        "name": "original_repo",
        "notes": "Exact architecture from model_training/training.py",
        "layers": [
            ("Input", "input", (32, 32, 1), 0),
            ("Conv2D 64 k5", "relu/softplus-family", (28, 28, 64), 1664),
            ("MaxPool 2x2", "pool", (14, 14, 64), 0),
            ("Conv2D 128 k4", "relu/softplus-family", (11, 11, 128), 131200),
            ("MaxPool 2x2", "pool", (5, 5, 128), 0),
            ("Conv2D 256 k2", "relu/softplus-family", (4, 4, 256), 131328),
            ("Flatten", "reshape", (4096,), 0),
            ("Dense 128", "relu/softplus-family", (128,), 524416),
            ("Dropout 0.3", "regularization", (128,), 0),
            ("Dense 32", "relu/softplus-family", (32,), 4128),
            ("Dropout 0.3", "regularization", (32,), 0),
            ("Dense 1", "sigmoid", (1,), 33),
        ],
    }


def _low_spec(name, conv1, conv2, dense, dropout, epochs, lr, label_smoothing):
    conv1_params = (5 * 5 * 1 + 1) * conv1
    conv2_params = (3 * 3 * conv1 + 1) * conv2
    flatten = 6 * 6 * conv2
    dense_params = flatten * dense + dense
    out_params = dense * 1 + 1
    return {
        "name": name,
        "notes": (
            f"Underfit profile: epochs={epochs}, lr={lr}, dropout={dropout}, "
            f"label_smoothing={label_smoothing}"
        ),
        "layers": [
            ("Input", "input", (32, 32, 1), 0),
            (f"Conv2D {conv1} k5", "relu", (28, 28, conv1), conv1_params),
            ("MaxPool 2x2", "pool", (14, 14, conv1), 0),
            (f"Conv2D {conv2} k3", "relu", (12, 12, conv2), conv2_params),
            ("MaxPool 2x2", "pool", (6, 6, conv2), 0),
            ("Flatten", "reshape", (flatten,), 0),
            (f"Dense {dense}", "relu", (dense,), dense_params),
            (f"Dropout {dropout}", "regularization", (dense,), 0),
            ("Dense 1", "sigmoid", (1,), out_params),
        ],
    }


def _format_model_block(spec):
    lines = []
    lines.append(f"MODEL: {spec['name']}")
    lines.append(f"NOTES: {spec['notes']}")
    lines.append("layer | kind | output_shape | params")
    lines.append("-" * 72)
    total = 0
    for layer_name, kind, out_shape, params in spec["layers"]:
        total += params
        lines.append(f"{layer_name:18} | {kind:20} | {_shape_str(out_shape):14} | {params}")
    lines.append("-" * 72)
    lines.append(f"TOTAL_PARAMS: {total}")
    lines.append("")
    return "\n".join(lines), total


def _comparison_summary(orig_total, cv_total):
    def pct_drop(new, old):
        return (1.0 - (new / old)) * 100.0

    lines = []
    lines.append("COMPARISON SUMMARY")
    lines.append("-" * 72)
    lines.append(f"original_repo params : {orig_total}")
    lines.append(f"cvusecase params     : {cv_total}  (drop {pct_drop(cv_total, orig_total):.2f}%)")
    lines.append("")
    lines.append("WHAT CHANGED VS ORIGINAL")
    lines.append("1. Convolution width reduced heavily (64/128/256 -> 4/6)")
    lines.append("2. Dense head reduced heavily (128->32 stack -> single dense=6)")
    lines.append("3. Dropout increased (0.3 -> 0.8)")
    lines.append("4. Epochs reduced (50 baseline training in repo -> 2)")
    lines.append("5. Learning rate increased for instability (0.0005 -> 0.0055)")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Write architecture comparison report")
    parser.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "architecture_comparison.txt"),
        help="output txt report path",
    )
    args = parser.parse_args()

    original = _original_spec()
    cfg = _load_cvusecase_cfg_from_source()
    cvusecase = _low_spec("cvusecase", conv1=cfg["conv1"], conv2=cfg["conv2"],
                          dense=cfg["dense"], dropout=cfg["dropout"],
                          epochs=cfg["epochs"], lr=cfg["lr"],
                          label_smoothing=cfg["label_smoothing"])

    report_parts = []
    report_parts.append("EYECARE MODEL ARCHITECTURE COMPARISON")
    report_parts.append("=" * 72)
    report_parts.append("")

    block, orig_total = _format_model_block(original)
    report_parts.append(block)
    block, cv_total = _format_model_block(cvusecase)
    report_parts.append(block)
    report_parts.append(_comparison_summary(orig_total, cv_total))

    report = "\n".join(report_parts)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report)

    print("Wrote architecture comparison report ->", args.out)


if __name__ == "__main__":
    main()