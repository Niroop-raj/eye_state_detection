"""
Predict eye state for one Blender full-face image + its matching JSON.

Usage:
    python predict_cvusecase_image.py <image.png> [<image.json>] [<model.hdf5>]

The JSON is optional (defaults to same name as the PNG with .json extension), and
the model defaults to the best cvusecase checkpoint (checkpoints_cvusecase/cvusecase_best.hdf5).

This reuses the exact same eye-cropping logic as build_cvusecase_dataset.py, then runs the
model on each visible eye crop and prints open/closed + confidence, and compares
against the ground-truth eye_state_id in the JSON (if present).
"""
import os
import sys
import glob

import numpy as np
import keras

# reuse the exact preprocessing pipeline from build_cvusecase_dataset.py
try:
    import build_cvusecase_dataset as bd
except ImportError:
    # allows running from elsewhere in the repo
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "build_cvusecase_dataset", os.path.join(os.path.dirname(__file__), "build_cvusecase_dataset.py"))
    bd = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(bd)

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints_cvusecase")


def select_model(path_or_none):
    """Return the model file to use: explicit path > cvusecase_best.hdf5 > newest checkpoint."""
    if path_or_none and os.path.exists(path_or_none):
        return path_or_none
    best = os.path.join(CHECKPOINT_DIR, "cvusecase_best.hdf5")
    if os.path.exists(best):
        print("Model: using best cvusecase checkpoint (cvusecase_best.hdf5)")
        return best
    hits = glob.glob(os.path.join(CHECKPOINT_DIR, "*.hdf5")) + \
           glob.glob(os.path.join(CHECKPOINT_DIR, "*.h5"))
    if hits:
        return max(hits, key=os.path.getmtime)
    return None


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    png_path = sys.argv[1]
    json_path = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(png_path)[0] + ".json"
    model_path = select_model(sys.argv[3] if len(sys.argv) > 3 else None)

    if model_path is None:
        print("No model found. Provide a model path or fine-tune first.")
        sys.exit(1)

    if not os.path.exists(png_path):
        print("Image not found:", png_path); sys.exit(1)
    if not os.path.exists(json_path):
        print("JSON not found:", json_path); sys.exit(1)

    data = bd.parse_json(json_path)
    print("Image        :", os.path.basename(png_path))
    print("Subject      :", data.get("subject_id"))
    print("Frame        :", data.get("frame_id"))

    gt = data.get("eye_state_id")
    gt_lbl = "closed" if gt == 1 else ("open" if gt == 0 else "unknown")
    print("Ground truth : eye_state_id =", gt, "->", gt_lbl.upper())

    image = bd.Image.open(png_path).convert("L")
    quads = bd.split_eye_landmarks(data["eye_landmarks"], data["eye_landmark_mask"])
    if not quads:
        print("No usable eye found (need >= %d valid landmarks per eye)." % bd.MIN_POINTS)
        sys.exit(2)

    model = keras.models.load_model(model_path)

    print("\nPer-eye predictions (probability of CLOSED, threshold 0.6):")
    preds = []
    for eye_idx, quad in enumerate(quads):
        quad_px = bd.normalize_points(quad, image.width, image.height)
        crop = bd.crop_square(image, quad_px, bd.MARGIN_FACTOR)

        arr = np.asarray(crop, dtype=np.float32) / 255.0
        x = arr.reshape(1, bd.IMAGE_SIZE, bd.IMAGE_SIZE, 1)

        prob_closed = float(model.predict(x, verbose=0)[0][0])
        pred_cls = "CLOSED" if prob_closed > 0.6 else "OPEN"
        print(f"  eye{eye_idx}: closed_prob={prob_closed:.4f}  ->  {pred_cls}")
        preds.append(pred_cls)

    print("\nAggregate guess:", "CLOSED" if "CLOSED" in preds else "OPEN")
    if len(preds) == 2 and preds[0] != preds[1]:
        print("(Note: the two eyes predicted differently - open and closed)")


if __name__ == "__main__":
    main()