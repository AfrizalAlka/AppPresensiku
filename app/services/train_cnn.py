"""Train a small CNN from the preprocessed face dataset."""

import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import tensorflow as tf
from tensorflow.keras import layers, models

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _split_counts(total: int, train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple[int, int, int]:
	"""Convert ratios into simple train, validation, and test counts."""
	if total < 3:
		raise ValueError("Setiap kelas minimal harus memiliki 3 gambar untuk split train/val/test.")

	train_count = max(1, int(total * train_ratio))
	val_count = max(1, int(total * val_ratio))
	test_count = total - train_count - val_count

	if test_count < 1:
		test_count = 1
		if train_count >= val_count and train_count > 1:
			train_count -= 1
		elif val_count > 1:
			val_count -= 1

	while train_count + val_count + test_count > total:
		if train_count > val_count and train_count > 1:
			train_count -= 1
		elif val_count > 1:
			val_count -= 1
		else:
			test_count -= 1

	while train_count + val_count + test_count < total:
		train_count += 1

	return train_count, val_count, test_count


def _collect_samples(
	dataset_dir: Path,
	seed: int,
	train_ratio: float,
	val_ratio: float,
	test_ratio: float,
) -> Tuple[List[str], List[int], List[str], List[int], List[str], List[int], List[str]]:
	"""Collect all file paths and split them per class."""
	if not dataset_dir.is_dir():
		raise FileNotFoundError(f"Folder dataset tidak ditemukan: {dataset_dir}")

	ratio_sum = train_ratio + val_ratio + test_ratio
	if abs(ratio_sum - 1.0) > 1e-9:
		raise ValueError("Jumlah train_ratio + val_ratio + test_ratio harus 1.0")

	class_dirs = sorted(p for p in dataset_dir.iterdir() if p.is_dir())
	if not class_dirs:
		raise RuntimeError("Folder dataset tidak memiliki subfolder kelas.")

	random_state = random.Random(seed)
	class_names = [class_dir.name for class_dir in class_dirs]
	class_to_index = {class_name: index for index, class_name in enumerate(class_names)}

	x_train: List[str] = []
	y_train: List[int] = []
	x_val: List[str] = []
	y_val: List[int] = []
	x_test: List[str] = []
	y_test: List[int] = []

	for class_dir in class_dirs:
		image_files = sorted(
			p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
		)
		if len(image_files) < 3:
			raise ValueError(f"Kelas '{class_dir.name}' hanya punya {len(image_files)} gambar. Minimal 3 gambar.")

		random_state.shuffle(image_files)
		train_count, val_count, test_count = _split_counts(len(image_files), train_ratio, val_ratio, test_ratio)

		train_files = image_files[:train_count]
		val_files = image_files[train_count : train_count + val_count]
		test_files = image_files[train_count + val_count : train_count + val_count + test_count]

		class_index = class_to_index[class_dir.name]
		x_train.extend(str(path) for path in train_files)
		y_train.extend([class_index] * len(train_files))
		x_val.extend(str(path) for path in val_files)
		y_val.extend([class_index] * len(val_files))
		x_test.extend(str(path) for path in test_files)
		y_test.extend([class_index] * len(test_files))

	return x_train, y_train, x_val, y_val, x_test, y_test, class_names


def _build_dataset(
	image_paths: List[str],
	labels: List[int],
	image_size: int,
	batch_size: int,
	shuffle: bool,
	seed: int,
) -> tf.data.Dataset:
	"""Create a TensorFlow dataset from paths and labels."""
	dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
	if shuffle:
		dataset = dataset.shuffle(buffer_size=len(image_paths), seed=seed, reshuffle_each_iteration=True)

	def load_image(path: tf.Tensor, label: tf.Tensor):
		image_bytes = tf.io.read_file(path)
		image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
		image.set_shape([None, None, 3])
		image = tf.image.resize(image, [image_size, image_size])
		return tf.cast(image, tf.float32), label

	dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
	dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
	return dataset


def load_datasets(
    dataset_dir: Path,
    image_size: int,
    batch_size: int,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, List[str]]:
    """Load train, validation, and test datasets."""
    x_train, y_train, x_val, y_val, x_test, y_test, class_names = _collect_samples(
        dataset_dir=dataset_dir,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    train_ds = _build_dataset(x_train, y_train, image_size, batch_size, shuffle=True, seed=seed)
    val_ds = _build_dataset(x_val, y_val, image_size, batch_size, shuffle=False, seed=seed)
    test_ds = _build_dataset(x_test, y_test, image_size, batch_size, shuffle=False, seed=seed)

    return train_ds, val_ds, test_ds, class_names


def build_cnn(num_classes: int, image_size: int) -> tf.keras.Model:
    """Build a small CNN model for face classification."""
    return models.Sequential(
        [
            layers.Input(shape=(image_size, image_size, 3), name="input_layer"),
            layers.Rescaling(1.0 / 255.0),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="cnn_face_recognition",
    )


def train_model(
	dataset_dir: Path,
	model_path: Path,
	class_names_path: Path,
	logs_dir: Path,
	image_size: int,
	batch_size: int,
	epochs: int,
	learning_rate: float,
	seed: int,
	train_ratio: float,
	val_ratio: float,
	test_ratio: float,
) -> Dict[str, float]:
	"""Train the model and save the model file, labels, and history."""
	tf.random.set_seed(seed)

	train_ds, val_ds, test_ds, class_names = load_datasets(
		dataset_dir=dataset_dir,
		image_size=image_size,
		batch_size=batch_size,
		seed=seed,
		train_ratio=train_ratio,
		val_ratio=val_ratio,
		test_ratio=test_ratio,
	)

	model = build_cnn(num_classes=len(class_names), image_size=image_size)
	model.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
		loss="sparse_categorical_crossentropy",
		metrics=["accuracy"],
	)

	logs_dir.mkdir(parents=True, exist_ok=True)
	model_path.parent.mkdir(parents=True, exist_ok=True)
	class_names_path.parent.mkdir(parents=True, exist_ok=True)

	# Clean old model files before training
	for file in model_path.parent.glob("*"):
		if file.is_file():
			file.unlink()

	callbacks = [
		tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=5, restore_best_weights=True),
		tf.keras.callbacks.ModelCheckpoint(filepath=str(model_path), monitor="val_accuracy", mode="max", save_best_only=True),
	]

	history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)

	with open(class_names_path, "w", encoding="utf-8") as file:
		json.dump(class_names, file, ensure_ascii=False, indent=2)

	val_loss, val_acc = model.evaluate(val_ds, verbose=0)
	test_loss, test_acc = model.evaluate(test_ds, verbose=0)

	history_path = logs_dir / "training_history.json"
	with open(history_path, "w", encoding="utf-8") as file:
		json.dump(history.history, file, indent=2)

	return {
		"val_loss": float(val_loss),
		"val_accuracy": float(val_acc),
		"test_loss": float(test_loss),
		"test_accuracy": float(test_acc),
		"epoch_trained": float(len(history.history.get("loss", []))),
	}
