"""
Train the cv use-case eye-state model configuration.

Data is NOT sabotaged - it is the same clean cvusecase_dataset built by
build_cvusecase_dataset.py from the real Blender source. The accuracy drop comes purely
from intentional UNDERFIT:
  - a lean architecture (fewer conv filters, smaller dense head)
  - a short training schedule
  - high dropout
  - (optional) a slightly higher learning rate for instability

Dataset layout:
    ../cvusecase_dataset/
        train/open, train/closed
        valid/open, valid/closed
        test/open,  test/closed

Tuning knobs:
    --epochs, --lr, --dropout, --conv1, --conv2, --dense, --label_smoothing
    (optional overrides on top of the chosen defaults)

Evaluate with evaluate_cvusecase_model.py against the clean test split:
    python evaluate_cvusecase_model.py checkpoints_cvusecase/cvusecase_best.hdf5 --data ../cvusecase_dataset --json eval_metrics.json
"""
import os
import argparse as _ap

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.losses import BinaryCrossentropy

# ---- configuration (keep ABOVE any usage) ---------------------------------- #
IMAGE_SIZE = 32
BATCH_SIZE = 16

BASE_DIR       = os.path.dirname(__file__)
CLEAN_DATASET  = os.path.join(BASE_DIR, "..", "cvusecase_dataset")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints_cvusecase")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# chosen final low-capacity configuration
DEFAULT_CFG = {
    "epochs": 2,
    "lr": 0.0055,
    "dropout": 0.8,
    "conv1": 4,
    "conv2": 6,
    "dense": 6,
    "label_smoothing": 0.15,
}
# --------------------------------------------------------------------------- #


def main():
    _parser = _ap.ArgumentParser(description="Train the cv use-case model")
    _parser.add_argument("--data_path", default=CLEAN_DATASET, help="root dataset folder")
    _parser.add_argument("--model_out", default=os.path.join(CHECKPOINT_DIR, "cvusecase_best.hdf5"),
                         help="output .hdf5 path")
    _parser.add_argument("--tag", default="cvusecase",
                         help="run label shown in logs/report naming")
    _parser.add_argument("--epochs", type=int, default=None, help="override profile epochs")
    _parser.add_argument("--lr", type=float, default=None, help="override profile learning rate")
    _parser.add_argument("--dropout", type=float, default=None, help="override profile dropout")
    _parser.add_argument("--conv1", type=int, default=None, help="override conv1 filters")
    _parser.add_argument("--conv2", type=int, default=None, help="override conv2 filters")
    _parser.add_argument("--dense", type=int, default=None, help="override dense layer width")
    _parser.add_argument("--label_smoothing", type=float, default=None,
                         help="override label smoothing for binary crossentropy")
    args = _parser.parse_args()
    data_dir = args.data_path
    model_out = args.model_out
    os.makedirs(os.path.dirname(model_out), exist_ok=True)

    cfg = dict(DEFAULT_CFG)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.lr is not None:
        cfg["lr"] = args.lr
    if args.dropout is not None:
        cfg["dropout"] = args.dropout
    if args.conv1 is not None:
        cfg["conv1"] = args.conv1
    if args.conv2 is not None:
        cfg["conv2"] = args.conv2
    if args.dense is not None:
        cfg["dense"] = args.dense
    if args.label_smoothing is not None:
        cfg["label_smoothing"] = args.label_smoothing

    print("Run tag:", args.tag)
    print("Config:", cfg)

    # minimal augmentation (avoids over-augmentation, keeps model underfit)
    train_datagenerator = ImageDataGenerator(rotation_range=10,
                                             zoom_range=0.05,
                                             rescale=1 / 255,
                                             fill_mode='nearest')
    valid_datagenerator = ImageDataGenerator(rescale=1 / 255)

    train_data = train_datagenerator.flow_from_directory(
        os.path.join(data_dir, "train"),
        classes=["open", "closed"],
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
        class_mode="binary",
        shuffle=True)

    validation_data = valid_datagenerator.flow_from_directory(
        os.path.join(data_dir, "valid"),
        classes=["open", "closed"],
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
        class_mode="binary")

    print("Training data (clean):", train_data.samples, "samples; validation:", validation_data.samples)

    if train_data.samples == 0:
        raise SystemExit("No training images found under <data_path>/train. Build dataset first.")
    if validation_data.samples == 0:
        raise SystemExit("No validation images found under <data_path>/valid. Build dataset first.")

    # ---- deliberately LEAN architecture (reduced capacity => underfit) -------
    # far fewer parameters than the repo's high-accuracy model
    model = keras.Sequential()
    model.add(Conv2D(cfg["conv1"], (5, 5), padding="valid",
                     input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(cfg["conv2"], (3, 3), padding="valid"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(cfg["dense"], activation="relu"))
    model.add(Dropout(cfg["dropout"]))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss=BinaryCrossentropy(label_smoothing=cfg["label_smoothing"]),
                  optimizer=Adam(learning_rate=cfg["lr"]),
                  metrics=['accuracy',
                           keras.metrics.Precision(),
                           keras.metrics.Recall()])

    print("Lean architecture parameter count: {:,}".format(model.count_params()))

    cp_callback = ModelCheckpoint(
        filepath=os.path.join(os.path.dirname(model_out),
                              "cvusecase_{epoch:03d}-{val_accuracy:.3f}.hdf5"),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max')

    lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5)

    train_steps = max(1, len(train_data))
    valid_steps = max(1, len(validation_data))

    model.fit(train_data,
              validation_data=validation_data,
              epochs=cfg["epochs"],
              steps_per_epoch=train_steps,
              validation_steps=valid_steps,
              callbacks=[lr_callback, cp_callback])

    print("\nTraining complete. Checkpoints in", os.path.dirname(model_out))
    keras.models.save_model(model, model_out)
    print("Saved final model ->", model_out)


if __name__ == "__main__":
    main()