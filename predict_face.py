import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache").resolve()))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def parse_args():
    parser = argparse.ArgumentParser(description="Predict a person's name from face image(s).")
    parser.add_argument("path", help="Image file or folder of images.")
    parser.add_argument("--model", default="models/face_cnn.keras")
    parser.add_argument("--labels", default="models/labels.json")
    parser.add_argument("--img-size", type=int, default=160)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def list_images(path):
    path = Path(path)
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def load_image(image_path, img_size):
    image = tf.keras.utils.load_img(image_path, target_size=(img_size, img_size))
    array = tf.keras.utils.img_to_array(image)
    return np.expand_dims(array, axis=0)


def main():
    args = parse_args()
    model = tf.keras.models.load_model(args.model)
    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    labels = {int(index): name for index, name in labels.items()}

    image_paths = list_images(args.path)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.path}")

    top_k = max(1, min(args.top_k, len(labels)))
    for image_path in image_paths:
        batch = load_image(image_path, args.img_size)
        probs = model.predict(batch, verbose=0)[0]
        top_indices = probs.argsort()[-top_k:][::-1]
        guesses = [f"{labels[i]} ({probs[i] * 100:.1f}%)" for i in top_indices]
        print(f"{image_path}: " + " | ".join(guesses))


if __name__ == "__main__":
    main()
