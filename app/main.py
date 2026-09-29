from io import BytesIO
from datetime import datetime
from threading import Lock

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, UnidentifiedImageError

from app.config import settings
from app.service import PredictionService, resolve_local_image


app = FastAPI(title="Bottle recognition API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    max_age=600,
)
_service: PredictionService | None = None
_service_lock = Lock()


class LocalImageRequest(BaseModel):
    path: str


class ProductDetails(BaseModel):
    id: int
    wine_name: str
    category: str | None = None
    color: str | None = None
    region: str | None = None
    grape_variety: str | None = None
    description: str | None = None
    winery: str | None = None
    slug: str
    photo_name: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class Prediction(BaseModel):
    classname: str
    similarity: float
    product: ProductDetails


class PredictionResponse(BaseModel):
    top1: str
    top5: list[Prediction]
    mode: str
    bottle_found: bool
    label_found: bool


def _get_service() -> PredictionService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = PredictionService()
    return _service


def _decode_image(content: bytes) -> Image.Image:
    try:
        image = Image.open(BytesIO(content))
        image.load()
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(status_code=400, detail="Не удалось прочитать файл как изображение") from error


def _predict(image: Image.Image) -> PredictionResponse:
    try:
        return PredictionResponse(**_get_service().predict(image))
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/health")
def health() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "ready": _service is not None and _service.encoder is not None,
        "vector_backend": settings.vector_backend,
        "local_test_enabled": settings.local_test_enabled,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_upload(file: UploadFile = File(...)) -> PredictionResponse:
    content = await file.read(settings.max_upload_bytes + 1)
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="Размер изображения превышает 15 МБ")
    return _predict(_decode_image(content))


@app.post("/predict/path", response_model=PredictionResponse)
def predict_local_path(request: LocalImageRequest) -> PredictionResponse:
    if not settings.local_test_enabled:
        raise HTTPException(status_code=404, detail="Локальное тестирование отключено")
    try:
        path = resolve_local_image(request.path, settings.test_images_dir)
    except ValueError as error:
        raise HTTPException(status_code=403, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    try:
        with Image.open(path) as image:
            image.load()
            return _predict(image.convert("RGB"))
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(status_code=400, detail="Не удалось прочитать файл как изображение") from error
