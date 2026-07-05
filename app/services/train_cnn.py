"""Train a MobileNetV2-based face recognition model with K-Fold Cross Validation."""

import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold

# Use non-interactive backend so plots save correctly in a server environment
matplotlib.use("Agg")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGE_SIZE = 224  # MobileNetV2 native input size


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _collect_all_samples(
    dataset_dir: Path,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return arrays of all image paths and integer labels, plus class names.

    Files are shuffled with the given seed so cross-validation folds are
    reproducible but not biased by directory listing order.
    """
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Folder dataset tidak ditemukan: {dataset_dir}")

    class_dirs = sorted(p for p in dataset_dir.iterdir() if p.is_dir())
    if not class_dirs:
        raise RuntimeError("Folder dataset tidak memiliki subfolder kelas.")

    class_names: List[str] = [d.name for d in class_dirs]
    class_to_index = {name: i for i, name in enumerate(class_names)}

    all_paths: List[str] = []
    all_labels: List[int] = []

    for class_dir in class_dirs:
        files = sorted(
            p for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
        )
        if len(files) < 2:
            raise ValueError(
                f"Kelas '{class_dir.name}' hanya punya {len(files)} gambar. Minimal 2 diperlukan."
            )
        idx = class_to_index[class_dir.name]
        all_paths.extend(str(p) for p in files)
        all_labels.extend([idx] * len(files))

    # Shuffle together
    rng = random.Random(seed)
    combined = list(zip(all_paths, all_labels))
    rng.shuffle(combined)
    paths_arr, labels_arr = zip(*combined)
    return np.array(paths_arr), np.array(labels_arr, dtype=np.int32), class_names


def _build_dataset(
    image_paths: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    """Build a tf.data.Dataset from paths and labels, resizing to IMAGE_SIZE."""
    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths.tolist(), labels.tolist())
    )
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(image_paths), seed=seed, reshuffle_each_iteration=True
        )

    def load_image(path: tf.Tensor, label: tf.Tensor):
        image_bytes = tf.io.read_file(path)
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
        # MobileNetV2 preprocessing: scale to [-1, 1]
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        return image, label

    return dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_mobilenetv2(num_classes: int, learning_rate: float) -> tf.keras.Model:
    """Build a MobileNetV2 transfer-learning model for face classification.

    Architecture:
      MobileNetV2 (ImageNet weights, frozen) → GlobalAveragePooling2D
      → Dense(256, relu) → Dropout(0.4) → Dense(num_classes, softmax)
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    # Freeze the base; fine-tuning will be enabled in phase 2
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="input_layer")
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dense(256, activation="relu", name="fc1")(x)
    x = tf.keras.layers.Dropout(0.4, name="dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs, outputs, name="mobilenetv2_face_recognition")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Evaluation / plotting helpers
# ---------------------------------------------------------------------------

def _get_predictions(
    model: tf.keras.Model, dataset: tf.data.Dataset
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect all predicted and true labels from a dataset."""
    y_pred_list, y_true_list = [], []
    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        y_pred_list.extend(np.argmax(preds, axis=1))
        y_true_list.extend(labels.numpy())
    return np.array(y_pred_list), np.array(y_true_list)


def _save_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    logs_dir: Path,
    filename: str = "classification_report.txt",
) -> None:
    """Write a classification report to a text file in logs_dir.

    ``labels`` is passed explicitly so sklearn never complains when a class
    has no samples in the current test split.
    """
    all_labels = list(range(len(class_names)))
    report = classification_report(
        y_true, y_pred,
        labels=all_labels,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    path = logs_dir / filename
    with open(path, "w", encoding="utf-8") as f:
        f.write("Classification Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(report)
    print(f"  ✓ Classification report → {path}")
    print(report)


def _save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    logs_dir: Path,
    filename: str = "confusion_matrix.png",
) -> None:
    """Save a raw-count confusion matrix heatmap to logs_dir.

    Figure size scales tightly with the number of classes.
    ``labels`` is passed explicitly so the matrix always has shape (n, n) even
    when some classes are absent from the current test split.
    """
    all_labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    n = len(class_names)

    # Tight cell size so there's no dead space around each number
    cell_size = 0.35
    fig_w = max(12, n * cell_size + 3)   # extra space for y-labels + colorbar
    fig_h = max(10, n * cell_size + 2)   # extra space for x-labels + title

    # Annotation font: shrinks as class count grows, never below 4pt
    annot_font = max(4, int(9 - n * 0.05))
    # Tick label font: also shrinks but slightly more generous
    tick_font  = max(4, int(8 - n * 0.04))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        annot_kws={"size": annot_font, "weight": "bold"},
        linewidths=0.15,
        linecolor="white",
        cbar_kws={"label": "Count", "shrink": 0.6},
    )

    ax.set_title("Confusion Matrix (Count)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=10, labelpad=8)
    ax.set_ylabel("True Label", fontsize=10, labelpad=8)
    ax.tick_params(axis="x", rotation=90, labelsize=tick_font)
    ax.tick_params(axis="y", rotation=0,  labelsize=tick_font)

    plt.tight_layout()
    path = logs_dir / filename
    plt.savefig(str(path), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Confusion matrix → {path}")


def _save_confusion_matrix_subset(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    logs_dir: Path,
    n_edge: int = 4,
    filename: str = "confusion_matrix_subset.png",
) -> None:
    """Save a compact confusion matrix showing first/last n_edge classes only.

    Middle classes are replaced by a single '...' row and column so the
    plot stays readable without showing all N×N cells.

    Layout (n_edge=4):  rows/cols = [cls0…cls3, '...', cls-4…cls-1]
    """
    all_labels = list(range(len(class_names)))
    cm_full = confusion_matrix(y_true, y_pred, labels=all_labels)
    n = len(class_names)

    # Only makes sense when there are more classes than 2*n_edge
    if n <= n_edge * 2:
        # Just save the full matrix under the subset filename too
        _save_confusion_matrix(y_true, y_pred, class_names, logs_dir, filename)
        return

    # ── Build subset indices and labels ──────────────────────────────────
    head_idx = list(range(n_edge))
    tail_idx = list(range(n - n_edge, n))
    sel_idx  = head_idx + tail_idx          # actual row/col indices in cm_full

    head_names = [class_names[i] for i in head_idx]
    tail_names = [class_names[i] for i in tail_idx]
    sub_names  = head_names + ["..."] + tail_names   # length = 2*n_edge + 1

    # ── Build subset cm (float so we can insert NaN for '...' row/col) ───
    sel = np.ix_(sel_idx, sel_idx)
    cm_corners = cm_full[sel].astype(float)   # shape (2*n_edge, 2*n_edge)

    size = 2 * n_edge + 1  # include '...' row and column
    cm_sub = np.full((size, size), np.nan)

    # Top-left block  (head × head)
    cm_sub[:n_edge, :n_edge]     = cm_corners[:n_edge, :n_edge]
    # Top-right block (head × tail)
    cm_sub[:n_edge, n_edge+1:]   = cm_corners[:n_edge, n_edge:]
    # Bottom-left block (tail × head)
    cm_sub[n_edge+1:, :n_edge]   = cm_corners[n_edge:, :n_edge]
    # Bottom-right block (tail × tail)
    cm_sub[n_edge+1:, n_edge+1:] = cm_corners[n_edge:, n_edge:]
    # '...' row and column stay NaN (rendered as blank)

    # ── Custom annotation array ───────────────────────────────────────────
    annot_arr = np.empty((size, size), dtype=object)
    for r in range(size):
        for c in range(size):
            if r == n_edge or c == n_edge:
                annot_arr[r, c] = "·" if r == n_edge and c == n_edge else ""
            else:
                annot_arr[r, c] = str(int(cm_sub[r, c]))

    # Mark the separator row/col with a visible centre dot
    annot_arr[n_edge, n_edge] = "···"

    # ── Plot ──────────────────────────────────────────────────────────────
    cell_px = 1.1
    fig_size = size * cell_px + 4
    fig, ax = plt.subplots(figsize=(fig_size, fig_size - 1))

    # Use a masked array so NaN cells render as light grey
    cm_masked = np.ma.array(cm_sub, mask=np.isnan(cm_sub))
    cmap = plt.cm.Blues.copy()
    cmap.set_bad(color="#f0f0f0")   # grey for '...' cells

    sns.heatmap(
        cm_masked,
        annot=annot_arr,
        fmt="",
        cmap=cmap,
        xticklabels=sub_names,
        yticklabels=sub_names,
        ax=ax,
        annot_kws={"size": 11, "weight": "bold"},
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Count", "shrink": 0.7},
        vmin=0,
    )

    ax.set_title(
        f"Confusion Matrix — {n_edge} first & last classes  (of {n} total)",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_xlabel("Predicted Label", fontsize=10, labelpad=8)
    ax.set_ylabel("True Label", fontsize=10, labelpad=8)
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", rotation=0,  labelsize=9)

    # Style the '...' row and column separators
    sep = n_edge + 0.5
    ax.axhline(n_edge,       color="grey", lw=1.5, ls="--")
    ax.axhline(n_edge + 1,   color="grey", lw=1.5, ls="--")
    ax.axvline(n_edge,       color="grey", lw=1.5, ls="--")
    ax.axvline(n_edge + 1,   color="grey", lw=1.5, ls="--")

    plt.tight_layout()
    path = logs_dir / filename
    plt.savefig(str(path), dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Confusion matrix subset → {path}")


def _save_training_history(
    history: tf.keras.callbacks.History,
    logs_dir: Path,
    filename: str = "training_history.png",
) -> None:
    """Save loss and accuracy curves for one training run."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history["loss"], label="Train Loss", linewidth=2, marker="o")
    axes[0].plot(history.history["val_loss"], label="Val Loss", linewidth=2, marker="s")
    axes[0].set_title("Loss per Epoch", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history.history["accuracy"], label="Train Accuracy", linewidth=2, marker="o")
    axes[1].plot(history.history["val_accuracy"], label="Val Accuracy", linewidth=2, marker="s")
    axes[1].set_title("Accuracy per Epoch", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = logs_dir / filename
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Training history plot → {path}")


def _save_cv_summary(
    fold_results: List[Dict],
    logs_dir: Path,
) -> None:
    """Save cross-validation summary (table + bar chart) to logs_dir."""
    n_folds = len(fold_results)
    val_accs = [r["val_accuracy"] for r in fold_results]
    test_accs = [r["test_accuracy"] for r in fold_results]
    val_losses = [r["val_loss"] for r in fold_results]
    test_losses = [r["test_loss"] for r in fold_results]

    # ── Text summary ──────────────────────────────────────────────────────
    summary_path = logs_dir / "cv_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("K-Fold Cross Validation Summary\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Fold':>6} {'Val Acc':>10} {'Test Acc':>10} {'Val Loss':>10} {'Test Loss':>10}\n")
        f.write("-" * 50 + "\n")
        for i, r in enumerate(fold_results, 1):
            f.write(
                f"{i:>6} {r['val_accuracy']:>10.4f} {r['test_accuracy']:>10.4f}"
                f" {r['val_loss']:>10.4f} {r['test_loss']:>10.4f}\n"
            )
        f.write("-" * 50 + "\n")
        f.write(
            f"{'Mean':>6} {np.mean(val_accs):>10.4f} {np.mean(test_accs):>10.4f}"
            f" {np.mean(val_losses):>10.4f} {np.mean(test_losses):>10.4f}\n"
        )
        f.write(
            f"{'Std':>6} {np.std(val_accs):>10.4f} {np.std(test_accs):>10.4f}"
            f" {np.std(val_losses):>10.4f} {np.std(test_losses):>10.4f}\n"
        )
    print(f"  ✓ CV summary (text) → {summary_path}")

    # ── Bar chart ─────────────────────────────────────────────────────────
    x = np.arange(n_folds)
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, n_folds * 1.5), 5))
    bars1 = ax.bar(x - width / 2, val_accs, width, label="Val Accuracy", color="steelblue")
    bars2 = ax.bar(x + width / 2, test_accs, width, label="Test Accuracy", color="coral")

    ax.axhline(np.mean(val_accs), color="steelblue", linestyle="--", linewidth=1.2, alpha=0.7, label=f"Mean Val ({np.mean(val_accs):.3f})")
    ax.axhline(np.mean(test_accs), color="coral", linestyle="--", linewidth=1.2, alpha=0.7, label=f"Mean Test ({np.mean(test_accs):.3f})")

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Fold")
    ax.set_ylabel("Accuracy")
    ax.set_title("K-Fold Cross Validation — Accuracy per Fold", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {i+1}" for i in range(n_folds)])
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    chart_path = logs_dir / "cv_accuracy_chart.png"
    plt.savefig(str(chart_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ CV accuracy chart → {chart_path}")


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

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
    n_folds: int = 5,
) -> Dict:
    """Train MobileNetV2 with K-Fold Cross Validation.

    Pipeline per fold:
      1. Split indices → train / val / test (80 % of fold → train+val, 20 % → test)
      2. Build datasets (with MobileNetV2 preprocessing)
      3. Train with EarlyStopping + ModelCheckpoint
      4. Save per-fold: training_history plot, fold JSON history
    After all folds:
      5. Retrain final model on ALL data and evaluate on hold-out test set
      6. Save: confusion_matrix, classification_report, cv_summary

    Args:
        n_folds: Number of cross-validation folds (default 5).

    Returns:
        Dictionary with mean/std of val_accuracy, test_accuracy across folds,
        plus final model's test_accuracy.
    """
    tf.random.set_seed(seed)

    # ── Prepare directories ───────────────────────────────────────────────
    logs_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    class_names_path.parent.mkdir(parents=True, exist_ok=True)

    # Clean old model files
    for f in model_path.parent.glob("*"):
        if f.is_file():
            f.unlink()

    # ── Load all samples ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Loading dataset …")
    all_paths, all_labels, class_names = _collect_all_samples(dataset_dir, seed)
    num_classes = len(class_names)
    print(f"  Classes   : {num_classes}")
    print(f"  Total imgs: {len(all_paths)}")

    # Save class names now (used even if training fails partway)
    with open(class_names_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    # ── K-Fold Cross Validation ───────────────────────────────────────────
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_results: List[Dict] = []
    all_history: Dict[str, List] = {}  # accumulated history across folds

    print(f"\n{'='*70}")
    print(f"K-Fold Cross Validation  (k={n_folds}, epochs={epochs})")
    print("=" * 70)

    for fold_idx, (trainval_idx, test_idx) in enumerate(kf.split(all_paths), start=1):
        print(f"\n── Fold {fold_idx}/{n_folds} {'─'*50}")

        # Split trainval → train / val  (val_ratio out of the trainval portion)
        trainval_paths = all_paths[trainval_idx]
        trainval_labels = all_labels[trainval_idx]
        test_paths_fold = all_paths[test_idx]
        test_labels_fold = all_labels[test_idx]

        # Use val_ratio relative to trainval size
        val_size = max(1, int(len(trainval_idx) * val_ratio))
        val_paths_fold = trainval_paths[:val_size]
        val_labels_fold = trainval_labels[:val_size]
        train_paths_fold = trainval_paths[val_size:]
        train_labels_fold = trainval_labels[val_size:]

        print(f"  train={len(train_paths_fold)}  val={len(val_paths_fold)}  test={len(test_paths_fold)}")

        train_ds = _build_dataset(train_paths_fold, train_labels_fold, batch_size, shuffle=True,  seed=seed + fold_idx)
        val_ds   = _build_dataset(val_paths_fold,   val_labels_fold,   batch_size, shuffle=False, seed=seed)
        test_ds  = _build_dataset(test_paths_fold,  test_labels_fold,  batch_size, shuffle=False, seed=seed)

        # Build fresh model for each fold
        model = build_mobilenetv2(num_classes=num_classes, learning_rate=learning_rate)

        fold_ckpt = str(model_path.parent / f"fold_{fold_idx}_best.keras")
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy", mode="max", patience=5, restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=fold_ckpt, monitor="val_accuracy", mode="max", save_best_only=True
            ),
        ]

        history = model.fit(
            train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=1
        )

        val_loss, val_acc   = model.evaluate(val_ds,  verbose=0)
        test_loss, test_acc = model.evaluate(test_ds, verbose=0)
        epoch_count = len(history.history.get("loss", []))

        print(f"  val_acc={val_acc:.4f}  test_acc={test_acc:.4f}  epochs_run={epoch_count}")

        fold_result = {
            "fold": fold_idx,
            "val_accuracy": float(val_acc),
            "val_loss": float(val_loss),
            "test_accuracy": float(test_acc),
            "test_loss": float(test_loss),
            "epochs_run": epoch_count,
        }
        fold_results.append(fold_result)

        # Save per-fold training history plot
        _save_training_history(
            history, logs_dir, filename=f"training_history_fold{fold_idx}.png"
        )

        # Accumulate history for a combined history JSON
        for key, values in history.history.items():
            all_history.setdefault(key, []).extend(values)

        # Save per-fold history JSON
        fold_history_path = logs_dir / f"history_fold{fold_idx}.json"
        with open(fold_history_path, "w", encoding="utf-8") as f:
            json.dump({"fold": fold_idx, "history": history.history}, f, indent=2)

        # Clean per-fold checkpoint (keep disk clean; final model saved separately)
        ckpt_path = Path(fold_ckpt)
        if ckpt_path.exists():
            ckpt_path.unlink()

    # ── Cross-validation summary ──────────────────────────────────────────
    print(f"\n{'='*70}")
    print("Cross Validation Summary")
    print("=" * 70)
    mean_val  = float(np.mean([r["val_accuracy"]  for r in fold_results]))
    std_val   = float(np.std ([r["val_accuracy"]  for r in fold_results]))
    mean_test = float(np.mean([r["test_accuracy"] for r in fold_results]))
    std_test  = float(np.std ([r["test_accuracy"] for r in fold_results]))
    print(f"  Val  Accuracy : {mean_val:.4f} ± {std_val:.4f}")
    print(f"  Test Accuracy : {mean_test:.4f} ± {std_test:.4f}")

    _save_cv_summary(fold_results, logs_dir)

    # Save fold results JSON
    cv_json_path = logs_dir / "cv_results.json"
    with open(cv_json_path, "w", encoding="utf-8") as f:
        json.dump({"folds": fold_results, "mean_val_accuracy": mean_val,
                   "std_val_accuracy": std_val, "mean_test_accuracy": mean_test,
                   "std_test_accuracy": std_test}, f, indent=2)
    print(f"  ✓ CV results JSON → {cv_json_path}")

    # ── Final model: retrain on ALL data ──────────────────────────────────
    print(f"\n{'='*70}")
    print("Training final model on full dataset …")
    print("=" * 70)

    # Hold out a test set from the full data for final evaluation
    n_total = len(all_paths)
    test_size = max(1, int(n_total * test_ratio))
    val_size_full = max(1, int(n_total * val_ratio))

    final_test_paths   = all_paths[:test_size]
    final_test_labels  = all_labels[:test_size]
    final_val_paths    = all_paths[test_size:test_size + val_size_full]
    final_val_labels   = all_labels[test_size:test_size + val_size_full]
    final_train_paths  = all_paths[test_size + val_size_full:]
    final_train_labels = all_labels[test_size + val_size_full:]

    print(f"  train={len(final_train_paths)}  val={len(final_val_paths)}  test={len(final_test_paths)}")

    final_train_ds = _build_dataset(final_train_paths, final_train_labels, batch_size, shuffle=True,  seed=seed)
    final_val_ds   = _build_dataset(final_val_paths,   final_val_labels,   batch_size, shuffle=False, seed=seed)
    final_test_ds  = _build_dataset(final_test_paths,  final_test_labels,  batch_size, shuffle=False, seed=seed)

    final_model = build_mobilenetv2(num_classes=num_classes, learning_rate=learning_rate)
    final_callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path), monitor="val_accuracy", mode="max", save_best_only=True
        ),
    ]

    final_history = final_model.fit(
        final_train_ds, validation_data=final_val_ds,
        epochs=epochs, callbacks=final_callbacks, verbose=1
    )

    # Save final training history (plot + JSON)
    _save_training_history(final_history, logs_dir, filename="training_history_final.png")
    final_history_path = logs_dir / "training_history_final.json"
    with open(final_history_path, "w", encoding="utf-8") as f:
        json.dump(final_history.history, f, indent=2)

    # ── Evaluate final model ──────────────────────────────────────────────
    final_val_loss, final_val_acc   = final_model.evaluate(final_val_ds,  verbose=0)
    final_test_loss, final_test_acc = final_model.evaluate(final_test_ds, verbose=0)

    print(f"\n  Final model  val_acc={final_val_acc:.4f}  test_acc={final_test_acc:.4f}")

    # Generate classification report and confusion matrix on final test set
    print(f"\n{'='*70}")
    print("Generating evaluation artifacts …")
    print("=" * 70)
    y_pred, y_true = _get_predictions(final_model, final_test_ds)
    _save_classification_report(y_true, y_pred, class_names, logs_dir)
    _save_confusion_matrix(y_true, y_pred, class_names, logs_dir)
    _save_confusion_matrix_subset(y_true, y_pred, class_names, logs_dir)

    print(f"\n{'='*70}")
    print(f"All experiment artifacts saved to: {logs_dir}")
    print("=" * 70 + "\n")

    return {
        # Cross-validation metrics
        "cv_mean_val_accuracy":  mean_val,
        "cv_std_val_accuracy":   std_val,
        "cv_mean_test_accuracy": mean_test,
        "cv_std_test_accuracy":  std_test,
        # Final model metrics
        "val_loss":     float(final_val_loss),
        "val_accuracy": float(final_val_acc),
        "test_loss":    float(final_test_loss),
        "test_accuracy": float(final_test_acc),
        "epoch_trained": float(len(final_history.history.get("loss", []))),
        "num_classes":   num_classes,
    }
