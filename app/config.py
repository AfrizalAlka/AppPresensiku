import os
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class AppConfig:
    project_root: Path
    dataset_root: Path
    raw_dir: Path
    preprocessed_dir: Path
    model_path: Path
    class_names_path: Path
    logs_dir: Path
    image_size: int
    batch_size: int
    epochs: int
    learning_rate: float
    min_images_per_class: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    random_seed: int
    threshold: float
    camera_id: int
    camera_check_interval_sec: float
    debug: bool

    # DB config (MySQL Laravel)
    db_driver: str
    db_host: str
    db_port: int
    db_name: str
    db_user: str
    db_password: str

    # Schema mapping
    student_table: str
    student_id_column: str
    student_name_column: str
    attendance_table: str

    @staticmethod
    def from_env() -> "AppConfig":
        project_root = Path(__file__).resolve().parents[1]
        dataset_root = project_root / "dataset"
        model_dir = project_root / "models"
        logs_dir = project_root / "experiment_logs"

        return AppConfig(
            project_root=project_root,
            dataset_root=dataset_root,
            raw_dir=dataset_root / "Dataset_Raw",
            preprocessed_dir=dataset_root / "Dataset_Preprocessed",
            model_path=model_dir / os.getenv("MODEL_FILENAME", "cnn_face_recognition.keras"),
            class_names_path=model_dir / os.getenv("CLASS_NAMES_FILENAME", "class_names.json"),
            logs_dir=logs_dir,
            image_size=_env_int("IMAGE_SIZE", 224),
            batch_size=_env_int("BATCH_SIZE", 32),
            epochs=_env_int("EPOCHS", 20),
            learning_rate=_env_float("LEARNING_RATE", 1e-3),
            min_images_per_class=_env_int("MIN_IMAGES_PER_CLASS", 30),
            train_ratio=_env_float("TRAIN_RATIO", 0.70),
            val_ratio=_env_float("VAL_RATIO", 0.15),
            test_ratio=_env_float("TEST_RATIO", 0.15),
            random_seed=_env_int("RANDOM_SEED", 42),
            threshold=_env_float("RECOGNITION_THRESHOLD", 0.70),
            camera_id=_env_int("CAMERA_ID", 0),
            camera_check_interval_sec=_env_float("CAMERA_CHECK_INTERVAL_SEC", 1.0),
            debug=_env_bool("FLASK_DEBUG", True),
            db_driver=os.getenv("DB_DRIVER", "mysql").strip().lower(),
            db_host=os.getenv("DB_HOST", "127.0.0.1"),
            db_port=_env_int("DB_PORT", 3306),
            db_name=os.getenv("DB_NAME", "app_presensiku"),
            db_user=os.getenv("DB_USER", "root"),
            db_password=os.getenv("DB_PASSWORD", ""),
            student_table=os.getenv("STUDENT_TABLE", "students"),
            student_id_column=os.getenv("STUDENT_ID_COLUMN", "id"),
            student_name_column=os.getenv("STUDENT_NAME_COLUMN", "name"),
            attendance_table=os.getenv("ATTENDANCE_TABLE", "attendances"),
        )
