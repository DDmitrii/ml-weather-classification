# src/api/app.py
import asyncio
import logging
import logging.config
from contextlib import asynccontextmanager
from functools import partial
import io

from fastapi import FastAPI, File, UploadFile, Query, HTTPException, Request
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from src.api.predictor import WeatherPredictor
from src.api.schemas import PredictionResponse, ErrorResponse
from fastapi.middleware.cors import CORSMiddleware

# ── Логирование ──────────────────────────────────────────────────────────────

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
        }
    },
    "root": {"level": "INFO", "handlers": ["console"]},
}
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

# ── Константы ────────────────────────────────────────────────────────────────

MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE    = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_TYPES    = {"image/jpeg", "image/png", "image/webp"}

# ── Lifespan ─────────────────────────────────────────────────────────────────

predictor: WeatherPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    logger.info("Запуск сервиса, загружаю модели...")
    try:
        predictor = WeatherPredictor()
        logger.info("Сервис готов к работе.")
    except FileNotFoundError as e:
        logger.critical("Не удалось загрузить модели: %s", e)
        raise
    yield
    logger.info("Остановка сервиса.")
    predictor = None


# ── Приложение ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Weather Classification API",
    description=(
        "Классификация погодных условий по изображению с камеры наблюдения.\n\n"
        "**9 классов:** clear, fog, for_rain, night, night_fog, night_rain, night_snow, rain, snow.\n\n"
        "**Модели:**\n"
        "- `teacher` — ConvNeXt-Tiny, точнее (macro F1=91.8%), 106 MB\n"
        "- `student` — MobileNetV3-Small, быстрее (val acc=98.4%), 5.8 MB"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Middleware: логирование запросов ─────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info("→ %s %s", request.method, request.url.path)
    response = await call_next(request)
    logger.info("← %s %s %d", request.method, request.url.path, response.status_code)
    return response


# ── Эндпоинты ────────────────────────────────────────────────────────────────

@app.get("/health", summary="Проверка работоспособности")
def health():
    return {
        "status": "ok",
        "models_loaded": predictor is not None,
        "cache_stats": predictor.cache_stats() if predictor else {},
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Классифицировать погоду на изображении",
)
async def predict(
    image: UploadFile = File(..., description="Изображение с камеры (JPEG/PNG/WebP)"),
    model: str = Query(
        default="student",
        pattern="^(teacher|student)$",
        description="Модель: 'teacher' (точнее) или 'student' (быстрее)",
    ),
):
    # Валидация типа
    if (image.content_type
            and image.content_type not in ALLOWED_TYPES
            and image.content_type != "application/octet-stream"):
        raise HTTPException(
            status_code=400,
            detail=f"Неподдерживаемый тип файла: {image.content_type}. Используйте JPEG, PNG или WebP.",
        )

    # Чтение с проверкой размера
    raw = await image.read()
    if len(raw) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Файл слишком большой: {len(raw) / 1024 / 1024:.1f} MB. Максимум: {MAX_FILE_SIZE_MB} MB.",
        )

    # Декодирование изображения
    try:
        pil_image = Image.open(io.BytesIO(raw))
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Не удалось декодировать изображение.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка чтения: {e}")

    # Инференс в threadpool — не блокирует event loop
    try:
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            partial(predictor.predict, pil_image, model),
        )
    except Exception as e:
        logger.exception("Ошибка инференса")
        raise HTTPException(status_code=500, detail=f"Ошибка инференса: {e}")

    return JSONResponse(content=result)