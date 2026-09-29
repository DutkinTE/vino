import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

DEFAULT_CORS_ORIGINS = (
    "https://виносвое.рф",
    "https://www.виносвое.рф",
    "https://xn--b1aajkzgbw.xn--p1ai",
    "https://www.xn--b1aajkzgbw.xn--p1ai",
)


def _path_from_env(name: str, default: str) -> Path:
    path = Path(os.getenv(name, default)).expanduser()
    return path if path.is_absolute() else ROOT_DIR / path


def _bool_from_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


def _origins_from_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.getenv(name)
    if value is None:
        return default
    return tuple(origin.strip() for origin in value.split(",") if origin.strip())


@dataclass(frozen=True)
class Settings:
    model_id: str = os.getenv("SIGLIP_MODEL_ID", "google/siglip2-base-patch16-384")
    finetuned_weights: Path = _path_from_env("FINETUNED_WEIGHTS", "models/siglip2_finetuned.pt")
    bottle_detector_weights: Path = _path_from_env(
        "BOTTLE_DETECTOR_WEIGHTS", "models/bottle/bottle_detector_best.pt"
    )
    label_detector_weights: Path = _path_from_env("LABEL_DETECTOR_WEIGHTS", "models/label_pose_best.pt")
    reference_csv: Path = _path_from_env(
        "REFERENCE_CSV", "data/reference/embedding_comparison_reference.csv"
    )
    products_csv: Path = _path_from_env("PRODUCTS_CSV", "data/reference/product_info.csv")
    test_images_dir: Path = _path_from_env("TEST_IMAGES_DIR", "data/test_images")
    local_test_enabled: bool = _bool_from_env("LOCAL_TEST_ENABLED", False)
    vector_backend: str = os.getenv("VECTOR_BACKEND", "csv").strip().lower()
    pg_dsn: str = os.getenv("PG_DSN", "")
    cors_origins: tuple[str, ...] = _origins_from_env("CORS_ORIGINS", DEFAULT_CORS_ORIGINS)
    top_k: int = min(5, max(1, int(os.getenv("TOP_K", "5"))))
    bottle_confidence: float = float(os.getenv("BOTTLE_CONFIDENCE", "0.25"))
    label_confidence: float = float(os.getenv("LABEL_CONFIDENCE", "0.5"))
    max_upload_bytes: int = 15 * 1024 * 1024


settings = Settings()
