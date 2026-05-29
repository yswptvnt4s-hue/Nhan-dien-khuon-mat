import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache").resolve()))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf


AUTOTUNE = tf.data.AUTOTUNE


def parse_args():
    parser = argparse.ArgumentParser(description="Train a small CNN face classifier.")
    parser.add_argument("--data-dir", default=".", help="Folder containing train/val/test.")
    parser.add_argument("--output-dir", default="models", help="Where to save model files.")
    parser.add_argument("--img-size", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--model-type", choices=["simple_cnn", "mobilenetv2"], default="simple_cnn")
    parser.add_argument(
        "--weights",
        default="none",
        help="Only used for mobilenetv2. Use 'none', 'imagenet', or a local .h5 weights path.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_dataset(directory, img_size, batch_size, shuffle):
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="int",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=shuffle,
        seed=42,
    )


def class_weight_from_dataset(dataset, num_classes):
    counts = np.zeros(num_classes, dtype=np.int64)
    for _, labels in dataset.unbatch():
        counts[int(labels.numpy())] += 1
    total = counts.sum()
    weights = total / (num_classes * np.maximum(counts, 1))
    return {i: float(weights[i]) for i in range(num_classes)}


def build_simple_cnn(num_classes, img_size):
    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = tf.keras.layers.RandomFlip("horizontal")(inputs)
    x = tf.keras.layers.RandomRotation(0.06)(x)
    x = tf.keras.layers.RandomZoom(0.12)(x)
    x = tf.keras.layers.RandomContrast(0.18)(x)
    x = tf.keras.layers.Rescaling(1.0 / 255)(x)

    for filters, dropout in [(32, 0.0), (64, 0.05), (128, 0.1), (192, 0.15)]:
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        if dropout:
            x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="simple_face_cnn")


def build_mobilenetv2(num_classes, img_size, weights):
    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = tf.keras.layers.RandomFlip("horizontal")(inputs)
    x = tf.keras.layers.RandomRotation(0.05)(x)
    x = tf.keras.layers.RandomZoom(0.1)(x)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    weights_arg = None if weights == "none" else weights
    base = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights=weights_arg,
    )
    base.trainable = False
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="mobilenetv2_face_classifier")


def compile_model(model, learning_rate):
    optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-4)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def per_class_accuracy(model, dataset, class_names):
    correct = np.zeros(len(class_names), dtype=np.int64)
    total = np.zeros(len(class_names), dtype=np.int64)
    for images, labels in dataset:
        probs = model.predict(images, verbose=0)
        preds = np.argmax(probs, axis=1)
        label_values = labels.numpy()
        for label, pred in zip(label_values, preds):
            total[int(label)] += 1
            correct[int(label)] += int(int(label) == int(pred))

    report = {}
    for index, name in enumerate(class_names):
        report[name] = {
            "correct": int(correct[index]),
            "total": int(total[index]),
            "accuracy": float(correct[index] / total[index]) if total[index] else 0.0,
        }
    return report


def main():
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)

    data_dir = Path(args.data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_dir in [train_dir, val_dir, test_dir]:
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing split folder: {split_dir}")

    train_ds = make_dataset(train_dir, args.img_size, args.batch_size, shuffle=True)
    val_ds = make_dataset(val_dir, args.img_size, args.batch_size, shuffle=False)
    test_ds = make_dataset(test_dir, args.img_size, args.batch_size, shuffle=False)

    class_names = train_ds.class_names
    num_classes = len(class_names)
    labels_path = output_dir / "labels.json"
    labels_path.write_text(
        json.dumps({str(i): name for i, name in enumerate(class_names)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    class_weights = class_weight_from_dataset(train_ds, num_classes)
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)

    if args.model_type == "simple_cnn":
        model = build_simple_cnn(num_classes, args.img_size)
    else:
        model = build_mobilenetv2(num_classes, args.img_size, args.weights)
    compile_model(model, args.learning_rate)

    model_path = output_dir / "face_cnn.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.35,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print(f"Training {args.model_type} on {num_classes} people...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    best_model = tf.keras.models.load_model(model_path)
    test_loss, test_acc = best_model.evaluate(test_ds, verbose=1)
    class_report = per_class_accuracy(best_model, test_ds, class_names)

    history_path = output_dir / "history.json"
    history_path.write_text(
        json.dumps({k: [float(v) for v in values] for k, values in history.history.items()}, indent=2),
        encoding="utf-8",
    )

    metrics = {
        "model_type": args.model_type,
        "image_size": args.img_size,
        "num_classes": num_classes,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "model_path": str(model_path),
        "labels_path": str(labels_path),
    }
    (output_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "per_class_accuracy.json").write_text(
        json.dumps(class_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
