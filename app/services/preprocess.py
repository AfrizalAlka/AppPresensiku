import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _collect_class_dirs(root_dir: Path) -> List[Path]:
    return sorted([p for p in root_dir.iterdir() if p.is_dir()])


def _collect_image_files(root_dir: Path) -> List[Path]:
    return sorted(
        p for p in root_dir.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    )


def detect_largest_face(
    image_bgr: np.ndarray,
    face_cascade: cv2.CascadeClassifier,
    ) -> Optional[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        return int(x), int(y), int(w), int(h)


def random_augment(color_img: np.ndarray, rng: random.Random) -> np.ndarray:
    img = color_img.copy()
    h, w = img.shape[:2]

    angle = rng.uniform(-18, 18)
    scale = rng.uniform(0.95, 1.05)
    mat_rot = cv2.getRotationMatrix2D((w // 2, h // 2), angle, scale)
    img = cv2.warpAffine(
        img,
        mat_rot,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    tx = rng.randint(-10, 10)
    ty = rng.randint(-10, 10)
    mat_trans = np.float32([[1, 0, tx], [0, 1, ty]])
    img = cv2.warpAffine(
        img,
        mat_trans,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    if rng.random() < 0.5:
        img = cv2.flip(img, 1)

    alpha = rng.uniform(0.85, 1.20)
    beta = rng.uniform(-20, 20)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    if rng.random() < 0.3:
        k = rng.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    return img


def preprocess_dataset(
    source_dir: Path,
    output_dir: Path,
    target_size: int,
    min_images_per_class: int,
    seed: int,
    overwrite: bool = False,
) -> Dict[str, int]:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Folder sumber tidak ditemukan: {source_dir}")

    class_dirs = _collect_class_dirs(source_dir)
    if not class_dirs:
        raise RuntimeError("Dataset raw tidak memiliki subfolder kelas.")

    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise RuntimeError("Gagal memuat Haar Cascade untuk deteksi wajah.")

    rng = random.Random(seed)
    stats = {
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

        processed_class_images: List[np.ndarray] = []

        for idx, src_path in enumerate(image_files, start=1):
            image = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
            if image is None:
                stats["skipped"] += 1
                continue

            face_box = detect_largest_face(image, face_cascade)
            if face_box is None:
                stats["skipped"] += 1
                continue

            x, y, w, h = face_box
            face_crop = image[y : y + h, x : x + w]
            resized = cv2.resize(face_crop, (target_size, target_size), interpolation=cv2.INTER_AREA)

            dst_path = class_output_dir / f"orig_{idx:04d}.jpg"
            if cv2.imwrite(str(dst_path), resized):
                stats["processed"] += 1
                processed_class_images.append(resized)
            else:
                stats["skipped"] += 1

        current_count = len(processed_class_images)
        need_new = max(0, min_images_per_class - current_count)

        if current_count == 0:
            continue

        for i in range(need_new):
            base_img = processed_class_images[i % current_count]
            aug_img = random_augment(base_img, rng)
            aug_path = class_output_dir / f"aug_{i + 1:04d}.jpg"
            if cv2.imwrite(str(aug_path), aug_img):
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
