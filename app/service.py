from pathlib import Path

from PIL import Image

from app.config import Settings, settings
from app.inference import SiglipBottleEncoder
from app.products import CsvProductStore, PgProductStore, ProductStore, enrich_predictions
from app.retrieval import create_vector_store


class PredictionService:
    def __init__(self, configuration: Settings = settings) -> None:
        self.configuration = configuration
        self.encoder: SiglipBottleEncoder | None = None
        self.vector_store = None
        self.product_store: ProductStore | None = None

    def _load(self) -> None:
        if self.encoder is not None and self.vector_store is not None:
            return
        if not self.configuration.finetuned_weights.is_file():
            raise FileNotFoundError(
                f"Не найдены fine-tuned веса: {self.configuration.finetuned_weights}"
            )
        if self.configuration.vector_backend == "csv":
            if not self.configuration.reference_csv.is_file():
                raise FileNotFoundError(f"Не найден CSV эталонов: {self.configuration.reference_csv}")
            if not self.configuration.products_csv.is_file():
                raise FileNotFoundError(f"Не найден CSV товаров: {self.configuration.products_csv}")
        elif not self.configuration.pg_dsn:
            raise FileNotFoundError("Для получения карточек товаров задайте PG_DSN")

        self.vector_store = create_vector_store(
            self.configuration.vector_backend,
            self.configuration.reference_csv,
            self.configuration.pg_dsn,
        )
        self.product_store = (
            CsvProductStore(self.configuration.products_csv)
            if self.configuration.vector_backend == "csv"
            else PgProductStore(self.configuration.pg_dsn)
        )
        self.encoder = SiglipBottleEncoder(
            model_id=self.configuration.model_id,
            finetuned_weights=self.configuration.finetuned_weights,
            bottle_detector_weights=self.configuration.bottle_detector_weights,
            label_detector_weights=self.configuration.label_detector_weights,
            bottle_confidence=self.configuration.bottle_confidence,
            label_confidence=self.configuration.label_confidence,
        )

    def predict(self, image: Image.Image) -> dict:
        self._load()
        assert self.encoder is not None and self.vector_store is not None and self.product_store is not None
        vector, mode, bottle_found, label_found = self.encoder.encode(image)
        matches = self.vector_store.search(vector, mode, self.configuration.top_k)
        products_by_ids = self.product_store.fetch_by_ids(
            [int(match["product_id"]) for match in matches if "product_id" in match]
        )
        products_by_names = self.product_store.fetch_by_names(
            [str(match["classname"]) for match in matches if "product_id" not in match]
        )
        predictions = enrich_predictions(matches, products_by_names, products_by_ids)
        if not predictions:
            raise ValueError(f"Нет эталонных векторов для режима {mode}")
        return {
            "top1": predictions[0]["classname"],
            "top5": predictions,
            "mode": mode,
            "bottle_found": bottle_found,
            "label_found": label_found,
        }


def resolve_local_image(raw_path: str, allowed_root: Path) -> Path:
    path = Path(raw_path).expanduser().resolve()
    root = allowed_root.resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Локальный файл должен находиться внутри {root}") from error
    if not path.is_file():
        raise FileNotFoundError(f"Файл не найден: {path}")
    return path
