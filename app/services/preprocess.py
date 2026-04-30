"""Preprocess face images by detecting, cropping, resizing, and augmenting them."""

import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _collect_class_dirs(root_dir: Path) -> List[Path]:
    """Return the class folders inside the source dataset."""
    return sorted([p for p in root_dir.iterdir() if p.is_dir()])


def _collect_image_files(root_dir: Path) -> List[Path]:
	"""Return all supported image files below one class folder."""
	return sorted(p for p in root_dir.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS)


def load_face_cascade() -> cv2.CascadeClassifier:
    """Load the built-in Haar cascade for face detection."""
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise RuntimeError("Gagal memuat Haar Cascade untuk deteksi wajah.")
    return face_cascade


def detect_largest_face(image_bgr: np.ndarray, face_cascade: cv2.CascadeClassifier) -> Optional[Tuple[int, int, int, int]]:
    """Find the biggest face box in one image."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
    return int(x), int(y), int(w), int(h)


def crop_and_resize(image_bgr: np.ndarray, face_box: Tuple[int, int, int, int], target_size: int) -> np.ndarray:
    """Crop the face area and resize it to the model input size."""
    x, y, w, h = face_box
    face_crop = image_bgr[y : y + h, x : x + w]
    return cv2.resize(face_crop, (target_size, target_size), interpolation=cv2.INTER_AREA)


def random_augment(color_img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Create a slightly changed copy of one face image."""
    image = color_img.copy()
    height, width = image.shape[:2]

    rotation = rng.uniform(-18, 18)
    scale = rng.uniform(0.95, 1.05)
    rotation_matrix = cv2.getRotationMatrix2D((width // 2, height // 2), rotation, scale)
    image = cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    shift_x = rng.randint(-10, 10)
    shift_y = rng.randint(-10, 10)
    translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    image = cv2.warpAffine(
        image,
        translation_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    if rng.random() < 0.5:
        image = cv2.flip(image, 1)

    brightness = rng.uniform(0.85, 1.20)
    contrast = rng.uniform(-20, 20)
    image = cv2.convertScaleAbs(image, alpha=brightness, beta=contrast)

    if rng.random() < 0.3:
        kernel_size = rng.choice([3, 5])
        image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    return image


def preprocess_dataset(
    source_dir: Path,
    output_dir: Path,
    target_size: int,
    min_images_per_class: int,
    seed: int,
    overwrite: bool = False,
) -> Dict[str, int]:
	"""Process every class folder and save cleaned plus augmented images."""
	if not source_dir.is_dir():
		raise FileNotFoundError(f"Folder sumber tidak ditemukan: {source_dir}")

	class_dirs = _collect_class_dirs(source_dir)
	if not class_dirs:
		raise RuntimeError("Dataset raw tidak memiliki subfolder kelas.")

	if output_dir.exists() and overwrite:
		shutil.rmtree(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	face_cascade = load_face_cascade()
	rng = random.Random(seed)
	stats = {"class_count": len(class_dirs), "processed": 0, "skipped": 0, "generated": 0, "total_output": 0}

	for class_dir in class_dirs:
		image_files = _collect_image_files(class_dir)
		if not image_files:
			continue

		class_output_dir = output_dir / class_dir.name
		class_output_dir.mkdir(parents=True, exist_ok=True)

		clean_images: List[np.ndarray] = []

		for index, src_path in enumerate(image_files, start=1):
			image = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
			if image is None:
				stats["skipped"] += 1
				continue

			face_box = detect_largest_face(image, face_cascade)
			if face_box is None:
				stats["skipped"] += 1
				continue

			resized = crop_and_resize(image, face_box, target_size)
			output_path = class_output_dir / f"orig_{index:04d}.jpg"
			if cv2.imwrite(str(output_path), resized):
				stats["processed"] += 1
				clean_images.append(resized)
			else:
				stats["skipped"] += 1

		if not clean_images:
			continue

		needed_images = max(0, min_images_per_class - len(clean_images))
		for index in range(needed_images):
			base_image = clean_images[index % len(clean_images)]
			augmented_image = random_augment(base_image, rng)
			augmented_path = class_output_dir / f"aug_{index + 1:04d}.jpg"
			if cv2.imwrite(str(augmented_path), augmented_image):
				stats["generated"] += 1

	stats["total_output"] = stats["processed"] + stats["generated"]
	return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess dataset: detect, crop, resize, augment.")
    parser.add_argument("--source", type=str, default="dataset/Dataset_Raw")
    parser.add_argument("--output", type=str, default="dataset/Dataset_Preprocessed")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--min_images", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    stats = preprocess_dataset(
        source_dir=Path(args.source),
        output_dir=Path(args.output),
        target_size=args.size,
        min_images_per_class=args.min_images,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(stats)


if __name__ == "__main__":
    main()
