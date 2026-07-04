"""Preprocess face images by cropping, removing background, resizing, and augmenting them."""

import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from rembg import remove as rembg_remove
from PIL import Image
import io

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _collect_class_dirs(root_dir: Path) -> List[Path]:
    """Return the class folders inside the source dataset."""
    return sorted([p for p in root_dir.iterdir() if p.is_dir()])


def _collect_image_files(root_dir: Path) -> List[Path]:
	"""Return all supported image files below one class folder."""
	return sorted(p for p in root_dir.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS)


def center_crop_face(image_bgr: np.ndarray, crop_ratio: float = 0.7) -> np.ndarray:
    """Crop the center portion of the image where the face is assumed to be located.

    Args:
        image_bgr: Input image in BGR format.
        crop_ratio: Fraction of the shorter dimension to use as crop size (default 0.7).

    Returns:
        Cropped image in BGR format.
    """
    h, w = image_bgr.shape[:2]
    crop_size = int(min(h, w) * crop_ratio)
    cx, cy = w // 2, h // 2
    x1 = max(0, cx - crop_size // 2)
    y1 = max(0, cy - crop_size // 2)
    x2 = min(w, x1 + crop_size)
    y2 = min(h, y1 + crop_size)
    return image_bgr[y1:y2, x1:x2]


def remove_background(image_bgr: np.ndarray) -> np.ndarray:
    """Remove the background from a face image using rembg.

    The output is a 3-channel BGR image where the removed background is
    replaced with white pixels so downstream JPEG encoding works correctly.

    Args:
        image_bgr: Input image in BGR format (uint8).

    Returns:
        BGR image with background replaced by white (uint8).
    """
    # Convert BGR → RGB PIL image
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)

    # Run rembg (returns RGBA PIL image)
    buf_in = io.BytesIO()
    pil_img.save(buf_in, format="PNG")
    buf_in.seek(0)

    result_bytes = rembg_remove(buf_in.read())
    result_pil = Image.open(io.BytesIO(result_bytes)).convert("RGBA")

    # Composite onto white background
    white_bg = Image.new("RGBA", result_pil.size, (255, 255, 255, 255))
    white_bg.paste(result_pil, mask=result_pil.split()[3])  # alpha channel as mask
    result_rgb = white_bg.convert("RGB")

    # Convert RGB → BGR
    result_bgr = cv2.cvtColor(np.array(result_rgb), cv2.COLOR_RGB2BGR)
    return result_bgr


def resize_image(image_bgr: np.ndarray, target_size: int) -> np.ndarray:
    """Resize image to a square target size.

    Args:
        image_bgr: Input image in BGR format.
        target_size: Side length in pixels for the output square image.

    Returns:
        Resized BGR image.
    """
    return cv2.resize(image_bgr, (target_size, target_size), interpolation=cv2.INTER_AREA)


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
    crop_ratio: float = 0.7,
) -> Dict[str, int]:
	"""Process every class folder: center-crop → remove background → resize → augment.

	Pipeline order per image:
	  1. Center crop  – extract central face region
	  2. Remove background – isolate face with white background via rembg
	  3. Resize – scale to target_size × target_size
	  4. Augmentation – generate synthetic variants to meet min_images_per_class

	Args:
	    source_dir: Root folder containing one sub-folder per class.
	    output_dir: Destination folder for preprocessed images.
	    target_size: Output image side length in pixels.
	    min_images_per_class: Minimum images per class (augmented if needed).
	    seed: Random seed for reproducible augmentation.
	    overwrite: Currently unused; output_dir is always rebuilt from scratch.
	    crop_ratio: Fraction of the shorter dimension used for center crop (default 0.7).

	Returns:
	    Dictionary with keys: class_count, processed, skipped, generated, total_output.
	"""
	if not source_dir.is_dir():
		raise FileNotFoundError(f"Folder sumber tidak ditemukan: {source_dir}")

	class_dirs = _collect_class_dirs(source_dir)
	if not class_dirs:
		raise RuntimeError("Dataset raw tidak memiliki subfolder kelas.")

	# Always remove old preprocessed dataset to ensure clean data
	# This prevents mixing old and new preprocessed images
	if output_dir.exists():
		shutil.rmtree(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	rng = random.Random(seed)
	stats: Dict[str, int] = {
	    "class_count": len(class_dirs),
	    "processed": 0,
	    "skipped": 0,
	    "generated": 0,
	    "total_output": 0,
	}

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
				print(f"  [SKIP] Gagal membaca: {src_path.name}")
				stats["skipped"] += 1
				continue

			# Step 1: Center crop
			cropped = center_crop_face(image, crop_ratio=crop_ratio)

			# Step 2: Remove background
			try:
				no_bg = remove_background(cropped)
			except Exception as exc:
				print(f"  [WARN] rembg gagal untuk {src_path.name}: {exc} – menggunakan gambar asli")
				no_bg = cropped

			# Step 3: Resize
			resized = resize_image(no_bg, target_size)

			# Save original processed image
			output_path = class_output_dir / f"orig_{index:04d}.jpg"
			if cv2.imwrite(str(output_path), resized):
				stats["processed"] += 1
				clean_images.append(resized)
			else:
				print(f"  [SKIP] Gagal menulis: {output_path.name}")
				stats["skipped"] += 1

		if not clean_images:
			continue

		# Step 4: Augmentation – fill up to min_images_per_class
		needed_images = max(0, min_images_per_class - len(clean_images))
		for aug_index in range(needed_images):
			base_image = clean_images[aug_index % len(clean_images)]
			augmented_image = random_augment(base_image, rng)
			augmented_path = class_output_dir / f"aug_{aug_index + 1:04d}.jpg"
			if cv2.imwrite(str(augmented_path), augmented_image):
				stats["generated"] += 1

	stats["total_output"] = stats["processed"] + stats["generated"]
	return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess dataset: center-crop, remove background, resize, augment.")
    parser.add_argument("--source", type=str, default="dataset/Dataset_Raw")
    parser.add_argument("--output", type=str, default="dataset/Dataset_Preprocessed")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--min_images", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--crop_ratio", type=float, default=0.7, help="Fraction of shorter dimension for center crop (default: 0.7)")
    args = parser.parse_args()

    stats = preprocess_dataset(
        source_dir=Path(args.source),
        output_dir=Path(args.output),
        target_size=args.size,
        min_images_per_class=args.min_images,
        seed=args.seed,
        overwrite=args.overwrite,
        crop_ratio=args.crop_ratio,
    )
    print(stats)


if __name__ == "__main__":
    main()
