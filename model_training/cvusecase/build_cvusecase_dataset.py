"""
Build a repo-style eye-crop dataset from Blender full-face images + their JSON labels.

Input
-----
SOURCE_DIR : a FLAT directory containing <record_id>.png + <record_id>.json pairs
             (each JSON is produced by Blender and looks like this):

    {
      "record_id":        "<same as .png stem>",
      "subject_id":       "Subject_10",
      "frame_id":         69,
      "aoi":              "eye-state",
      "eye_state_id":     1,                # 0 = OPEN, 1 = CLOSED   (<-- the label we use)
      "eye_landmarks":    [[x,y]*10],      # normalized 0..1; index 0 and 5 are missing (mask=False)
      "eye_landmark_mask":[bool*10]        # False -> point missing ([0,0] placeholder)
    }

Output
------
OUT_DIR : a flow_from_directory-compatible layout that train_cvusecase_model.py / evaluate_cvusecase_model.py expect:

    <OUT_DIR>/train/open,   <OUT_DIR>/train/closed
    <OUT_DIR>/valid/open,   <OUT_DIR>/valid/closed
    <OUT_DIR>/test/open,    <OUT_DIR>/test/closed

Every saved file is a 32x32 GRAYSCALE single-eye crop. Two crops (left + right eye)
are produced from each full-face image, both sharing the same eye_state_id label.

Splitting is done by subject_id (NOT by random file) so that frames from the same
synthetic subject stay in one split and do NOT leak between train and test.
"""
import os
import json
import random
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image

# --------------------------------------------------------------------------- #
#  Configuration - EDIT THESE
# --------------------------------------------------------------------------- #
SOURCE_DIR      = os.path.join(os.path.dirname(__file__), "..", "blender_source")
OUT_DIR         = os.path.join(os.path.dirname(__file__), "..", "cvusecase_dataset")

IMAGE_SIZE      = 32                 # repo model input size
MARGIN_FACTOR   = 1.2                # same EYE_BOX_SIZE_FACTOR as the repo app

# subject-aware split ratios (fractions of SUBJECTS, not of images)
TRAIN_RATIO     = 0.80
VALID_RATIO     = 0.10
TEST_RATIO      = 0.10

RANDOM_SEED     = 42

# landmark schema: each eye has 5 slots (eye-1 = slots 0-4, eye-2 = slots 5-9);
# different frames mark different subsets as valid (eye_landmark_mask).
EYE_SLOTS       = ([0, 1, 2, 3, 4], [5, 6, 7, 8, 9])
MIN_POINTS      = 3   # min valid points needed to crop one eye (drop the eye otherwise)
# --------------------------------------------------------------------------- #
def load_pairs(source_dir):
    """Return list of (png_path, json_path) for every png that has a matching json."""
    pairs = []
    for name in sorted(os.listdir(source_dir)):
        if name.lower().endswith(".png"):
            stem = os.path.splitext(name)[0]
            json_path = os.path.join(source_dir, stem + ".json")
            if os.path.exists(json_path):
                pairs.append((os.path.join(source_dir, name), json_path))
    return pairs


def parse_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_records(pairs):
    """Parse each JSON exactly once and return metadata keyed by png path."""
    records = {}
    for png_path, json_path in pairs:
        records[png_path] = parse_json(json_path)
    return records


def _suggest_workers():
    """Heuristic worker count for mixed image CPU + disk I/O workload."""
    cpu = os.cpu_count() or 2
    return max(1, min(8, cpu - 1))


def split_eye_landmarks(landmarks, mask):
    """Split the 10 landmark slots into the two eyes by slot index
    (eye-1 = slots 0-4, eye-2 = slots 5-9). For each eye we keep only the slots
    whose mask is True. Returns one point-list per eye that has >= MIN_POINTS
    valid points (a frame may legitimately contain only one visible eye)."""
    out = []
    for slots in EYE_SLOTS:
        pts = [(landmarks[i][0], landmarks[i][1]) for i in slots if mask[i]]
        if len(pts) >= MIN_POINTS:
            out.append(pts)
    return out


def normalize_points(points, width, height):
    """Convert coords to pixels. Values are treated as normalized (0..1) unless any
    coordinate is larger than ~1.5, in which case they are already pixel coords."""
    if any(max(x, y) > 1.5 for (x, y) in points):
        return [(x, y) for (x, y) in points]
    return [(x * width, y * height) for (x, y) in points]


def crop_square(image, quad, margin_factor=1.2):
    """Crop a square single-eye region containing the 4 landmark points,
    padded by margin_factor, clipped to the image, zero-padded if needed."""
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    half = max(max(xs) - min(xs), max(ys) - min(ys)) / 2.0 * margin_factor

    x0, x1 = int(round(cx - half)), int(round(cx + half))
    y0, y1 = int(round(cy - half)), int(round(cy + half))

    w, h = image.size
    cx0, cy0 = max(x0, 0), max(y0, 0)          # clip to image
    cx1, cy1 = min(x1, w), min(y1, h)

    crop = image.crop((cx0, cy0, cx1, cy1))

    side = max(x1 - x0, y1 - y0)               # pad back to square if clipped
    if crop.size != (side, side):
        canvas = Image.new("L", (side, side), 0)
        canvas.paste(crop, (cx0 - x0, cy0 - y0))
        crop = canvas

    crop = crop.convert("L")                   # grayscale
    crop = crop.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    return crop


def make_dirs(root):
    for split in ("train", "valid", "test"):
        for cls in ("open", "closed"):
            os.makedirs(os.path.join(root, split, cls), exist_ok=True)

def build_split_plan(pairs, record_to_subject, subjects):
    """Return {png_path: split} guaranteeing non-empty train/valid/test.

    - >= 3 subjects : whole-subject 3-way split (no subject leakage).
    - 2 subjects    : one whole subject is held-out TEST; VALID is carved from a
                      slice of the TRAIN subject's frames (valid is never empty).
    - 1 subject     : frames split into train/valid/test by ratio.
    """
    names = list(subjects)
    random.shuffle(names)

    recs_by_subject = defaultdict(list)
    for png_path, _ in pairs:
        recs_by_subject[record_to_subject[png_path]].append(png_path)

    record_split = {}

    if len(names) >= 3:
        n_train = int(len(names) * TRAIN_RATIO)
        n_valid = int(len(names) * VALID_RATIO)
        n_test = len(names) - n_train - n_valid
        if n_valid == 0:            # guard against empty valid
            n_train -= 1
            n_valid = 1
        if n_test == 0:             # guard against empty test
            n_train -= 1
            n_test = 1
        for i, s in enumerate(names):
            split = "train" if i < n_train else ("valid" if i < n_train + n_valid else "test")
            for r in recs_by_subject[s]:
                record_split[r] = split

    elif len(names) == 2:
        # smaller subject -> held-out TEST (keeps the most data for training)
        if subjects[names[0]] <= subjects[names[1]]:
            test_sub, train_sub = names[0], names[1]
        else:
            test_sub, train_sub = names[1], names[0]
        for r in recs_by_subject[test_sub]:
            record_split[r] = "test"

        train_recs = recs_by_subject[train_sub][:]
        random.shuffle(train_recs)
        n_valid = max(1, int(len(train_recs) * VALID_RATIO))
        for r in train_recs[:n_valid]:
            record_split[r] = "valid"
        for r in train_recs[n_valid:]:
            record_split[r] = "train"

    else:  # single subject
        recs = recs_by_subject[names[0]][:]
        random.shuffle(recs)
        n_test = max(1, int(len(recs) * TEST_RATIO))
        n_valid = max(1, int(len(recs) * VALID_RATIO))
        n_train = len(recs) - n_test - n_valid
        for r in recs[:n_train]:
            record_split[r] = "train"
        for r in recs[n_train:n_train + n_valid]:
            record_split[r] = "valid"
        for r in recs[n_train + n_valid:]:
            record_split[r] = "test"

    return record_split


def _process_one_image(png_path, d, split, out_dir):
    """Process one image and write zero, one, or two eye crops.

    Returns tuple: (split, cls, written_count, skipped_message_or_none)
    """
    label = d["eye_state_id"]                # 0=open, 1=closed (per your spec)
    cls = "closed" if label == 1 else "open"

    with Image.open(png_path) as _img:
        image = _img.convert("L")

        quads = split_eye_landmarks(d["eye_landmarks"], d["eye_landmark_mask"])
        if not quads:
            msg = f"Skipping {os.path.basename(png_path)}: no eye has >= {MIN_POINTS} valid landmarks"
            return split, cls, 0, msg

        written = 0
        for eye_idx, quad in enumerate(quads):
            quad_px = normalize_points(quad, image.width, image.height)
            crop = crop_square(image, quad_px, MARGIN_FACTOR)

            out_name = f"{os.path.splitext(os.path.basename(png_path))[0]}_eye{eye_idx}.png"
            crop.save(os.path.join(out_dir, split, cls, out_name))
            written += 1

    return split, cls, written, None


def main():
    parser = argparse.ArgumentParser(description="Build eye-crop cvusecase dataset from blender_source")
    parser.add_argument("--source-dir", default=SOURCE_DIR,
                        help="input folder containing <id>.png + <id>.json pairs")
    parser.add_argument("--out-dir", default=OUT_DIR,
                        help="output folder where train/valid/test eye crops are written")
    parser.add_argument("--workers", type=int, default=0,
                        help="worker threads; 0 means auto, 1 means serial")
    args = parser.parse_args()

    source_dir = os.path.abspath(args.source_dir)
    out_dir = os.path.abspath(args.out_dir)

    random.seed(RANDOM_SEED)
    if not os.path.isdir(source_dir):
        raise SystemExit(
            f"Source folder not found: {source_dir}\n"
            f"Create it and place matching .png/.json pairs, or pass --source-dir."
        )

    pairs = load_pairs(source_dir)
    if not pairs:
        raise SystemExit(f"No .png/.json pairs found in {source_dir}")

    print(f"Found {len(pairs)} image/json pairs in {source_dir}")

    # Parse JSON once per sample; reuse in split planning and crop generation.
    records = load_records(pairs)

    # ---- subject-aware split so frames of one subject never span train/test ----
    subjects = defaultdict(int)
    record_to_subject = {}
    for png_path, _ in pairs:
        d = records[png_path]
        subj = d.get("subject_id", "unknown")
        record_to_subject[png_path] = subj
        subjects[subj] += 1

    record_split = build_split_plan(pairs, record_to_subject, subjects)

    from collections import Counter as _Counter
    split_images = _Counter(record_split.values())
    print("Distinct subjects:", len(subjects))
    for s, cnt in subjects.items():
        print(f"  {s}: {cnt} images")
    print("Split by image count:", dict(split_images))

    # ---- build the dataset ------------------------------------------------------
    make_dirs(out_dir)
    counts = {"train": {"open": 0, "closed": 0},
              "valid": {"open": 0, "closed": 0},
              "test":  {"open": 0, "closed": 0}}

    workers = args.workers if args.workers > 0 else _suggest_workers()
    print(f"Using workers: {workers}")

    tasks = [(png_path, records[png_path], record_split[png_path]) for png_path, _ in pairs]

    if workers == 1:
        for idx, (png_path, d, split) in enumerate(tasks):
            split_out, cls_out, written, skip_msg = _process_one_image(png_path, d, split, out_dir)
            if skip_msg:
                print(skip_msg)
            counts[split_out][cls_out] += written
            if (idx + 1) % 1000 == 0:
                print(f"  processed {idx + 1}/{len(tasks)}")
    else:
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_process_one_image, png_path, d, split, out_dir) for png_path, d, split in tasks]
            for fut in as_completed(futs):
                split_out, cls_out, written, skip_msg = fut.result()
                if skip_msg:
                    print(skip_msg)
                counts[split_out][cls_out] += written
                done += 1
                if done % 1000 == 0:
                    print(f"  processed {done}/{len(tasks)}")

    print("\nDone. Dataset written under", out_dir)
    for split in ("train", "valid", "test"):
        print(f"  {split:5s}: open={counts[split]['open']:>6d}  closed={counts[split]['closed']:>6d}")


if __name__ == "__main__":
    main()

