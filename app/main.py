from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import httpx
import imageio_ffmpeg
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
ALLOWED_EXTENSIONS = {".mp4", ".webm"}
MAX_UPLOAD_BYTES = 250 * 1024 * 1024
DIRECT_TRANSCRIPTION_MAX_BYTES = 25 * 1024 * 1024
CHUNK_SIZE = 1024 * 1024

PROXY_API_API_KEY = os.getenv("PROXY_API_API_KEY", "").strip()
PROXY_API_OPENAI_BASE_URL = os.getenv("PROXY_API_OPENAI_BASE_URL", "").strip()
TRANSCRIPTION_MODEL = os.getenv(
    "OPENAI_TRANSCRIPTION_MODEL", "gpt-4o-mini-transcribe"
).strip()
SUMMARY_MODEL = os.getenv("OPENAI_SUMMARY_MODEL", "gpt-4o-mini").strip()
IS_VERCEL = bool(os.getenv("VERCEL"))

app = FastAPI(title="AI-анализатор видеовстреч", version="2.0.0")
processing_lock = asyncio.Lock()
task_registry_lock = asyncio.Lock()
analysis_tasks: dict[str, dict[str, Any]] = {}
cloud_link_tasks: dict[str, str] = {}
background_tasks: set[asyncio.Task[Any]] = set()
ACTIVE_TASK_STATUSES = {"fetching", "transcribing", "analyzing"}


class ConfigurationError(RuntimeError):
    """Raised when the OpenAI proxy configuration is incomplete."""


class MediaProcessingError(RuntimeError):
    """Raised when FFmpeg cannot extract a usable audio track."""


class AnalysisError(RuntimeError):
    """Raised when OpenAI processing fails or returns an invalid result."""


class CloudImportError(RuntimeError):
    """Raised when a public Google Drive file cannot be downloaded safely."""


class CloudImportRequest(BaseModel):
    url: str


def get_openai_client() -> OpenAI:
    if not PROXY_API_API_KEY or not PROXY_API_OPENAI_BASE_URL:
        raise ConfigurationError(
            "Не настроен доступ к OpenAI. Добавьте PROXY_API_API_KEY и "
            "PROXY_API_OPENAI_BASE_URL в файл .env и перезапустите приложение."
        )

    client = OpenAI(
        api_key=PROXY_API_API_KEY,
        base_url=PROXY_API_OPENAI_BASE_URL,
    )
    return client


def find_ffmpeg_executable() -> str:
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    try:
        bundled_ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except (OSError, RuntimeError):
        return ""
    return bundled_ffmpeg if Path(bundled_ffmpeg).is_file() else ""


def extract_audio(video_path: Path, audio_path: Path) -> None:
    ffmpeg_executable = find_ffmpeg_executable()
    if not ffmpeg_executable:
        raise MediaProcessingError(
            "FFmpeg недоступен. Переустановите зависимости приложения или "
            "используйте видео размером до 25 МБ."
        )

    command = [
        ffmpeg_executable,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-b:a",
        "64k",
        str(audio_path),
    ]

    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except FileNotFoundError as exc:
        raise MediaProcessingError(
            "FFmpeg не найден. Установите FFmpeg и убедитесь, что команда ffmpeg "
            "доступна в PATH."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise MediaProcessingError(
            "Не удалось извлечь аудио: обработка видео заняла слишком много времени."
        ) from exc

    if result.returncode != 0 or not audio_path.exists() or audio_path.stat().st_size == 0:
        detail = (result.stderr or "").strip().splitlines()
        ffmpeg_hint = detail[-1] if detail else "аудиодорожка не найдена"
        raise MediaProcessingError(
            "Не удалось прочитать видео или извлечь аудио. Проверьте, что файл не "
            f"повреждён и содержит звук. ({ffmpeg_hint[:220]})"
        )


def prepare_transcription_file(video_path: Path, audio_path: Path) -> Path:
    """Return compressed audio locally or the original media on serverless hosts."""

    if find_ffmpeg_executable():
        extract_audio(video_path, audio_path)
        return audio_path

    if video_path.stat().st_size > DIRECT_TRANSCRIPTION_MAX_BYTES:
        raise MediaProcessingError(
            "На сервере недоступен FFmpeg, а исходное видео превышает лимит "
            "прямой транскрибации 25 МБ. Используйте запись меньшего размера "
            "или разверните приложение на сервере с установленным FFmpeg."
        )
    return video_path


def transcribe_audio(client: OpenAI, media_path: Path) -> str:
    try:
        with media_path.open("rb") as audio_file:
            result = client.audio.transcriptions.create(
                model=TRANSCRIPTION_MODEL,
                file=audio_file,
                response_format="json",
            )
    except Exception as exc:  # OpenAI SDK uses several transport/API exceptions.
        raise AnalysisError(
            "Не удалось распознать речь. Проверьте настройки API, доступность модели "
            "распознавания и повторите загрузку."
        ) from exc

    transcript = result if isinstance(result, str) else getattr(result, "text", "")
    transcript = str(transcript).strip()
    if not transcript:
        raise AnalysisError(
            "В записи не удалось распознать речь. Проверьте громкость и наличие звука."
        )
    return transcript


def _as_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def parse_summary(raw_summary: str) -> dict[str, Any]:
    cleaned = raw_summary.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise AnalysisError("AI вернул конспект в неожиданном формате. Повторите попытку.") from exc

    if not isinstance(data, dict):
        raise AnalysisError("AI вернул конспект в неожиданном формате. Повторите попытку.")

    brief = str(data.get("brief", "")).strip()
    if not brief:
        raise AnalysisError("AI не сформировал краткое содержание. Повторите попытку.")

    return {
        "brief": brief,
        "decisions": _as_list(data.get("decisions")),
        "tasks": _as_list(data.get("tasks")),
        "open_questions": _as_list(data.get("open_questions")),
    }


def create_summary(client: OpenAI, transcript: str) -> dict[str, Any]:
    system_prompt = """Ты аккуратный секретарь встречи. Анализируй только переданную
расшифровку и не добавляй фактов от себя. Не выдумывай исполнителей, решения,
сроки или причины. Верни только JSON-объект с ключами:
brief — краткое содержание встречи на русском языке (2–5 предложений);
decisions — массив явно принятых решений;
tasks — массив явно сформулированных задач после встречи;
open_questions — массив вопросов, которые явно остались нерешёнными.
Если элементов для массива нет, верни пустой массив. Не используй Markdown."""

    try:
        response = client.chat.completions.create(
            model=SUMMARY_MODEL,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Расшифровка встречи:\n\n{transcript}",
                },
            ],
        )
    except Exception as exc:
        raise AnalysisError(
            "Расшифровка получена, но не удалось составить конспект. Проверьте "
            "доступность модели для конспекта и повторите загрузку."
        ) from exc

    content = response.choices[0].message.content if response.choices else ""
    if not content:
        raise AnalysisError("AI не вернул конспект. Повторите попытку.")
    return parse_summary(content)


def process_video(video_path: Path, audio_path: Path) -> dict[str, Any]:
    client = get_openai_client()
    transcription_path = prepare_transcription_file(video_path, audio_path)
    transcript = transcribe_audio(client, transcription_path)
    summary = create_summary(client, transcript)
    return {"transcript": transcript, "summary": summary}


def normalize_google_drive_url(raw_url: str) -> tuple[str, str]:
    """Return a Drive file id and a stable canonical URL."""

    try:
        parsed = urlparse(raw_url.strip())
    except ValueError as exc:
        raise CloudImportError("Ссылка на Google Drive имеет неверный формат.") from exc

    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or host not in {
        "drive.google.com",
        "drive.usercontent.google.com",
    }:
        raise CloudImportError(
            "Поддерживаются только HTTPS-ссылки на файлы Google Drive."
        )

    file_id = ""
    path_match = re.search(r"/file/d/([A-Za-z0-9_-]+)", parsed.path)
    if path_match:
        file_id = path_match.group(1)
    else:
        file_id = parse_qs(parsed.query).get("id", [""])[0]

    if not re.fullmatch(r"[A-Za-z0-9_-]{10,}", file_id):
        raise CloudImportError(
            "Не удалось определить файл Google Drive. Используйте ссылку вида "
            "https://drive.google.com/file/d/…/view."
        )

    return file_id, f"https://drive.google.com/file/d/{file_id}"


def _filename_from_content_disposition(header: str) -> str:
    encoded_match = re.search(r"filename\*=UTF-8''([^;]+)", header, re.IGNORECASE)
    if encoded_match:
        return Path(unquote(encoded_match.group(1))).name

    plain_match = re.search(r'filename="?([^";]+)"?', header, re.IGNORECASE)
    if plain_match:
        return Path(plain_match.group(1).strip()).name
    return ""


def detect_video_extension(video_path: Path) -> str:
    try:
        with video_path.open("rb") as media_file:
            header = media_file.read(8192)
    except OSError as exc:
        raise MediaProcessingError("Не удалось прочитать загруженное видео.") from exc

    # MP4 is an ISO Base Media file and normally contains the ftyp box first.
    if len(header) >= 12 and header[4:8] == b"ftyp":
        return ".mp4"

    # WebM is an EBML container whose DocType in the header is "webm".
    if header.startswith(b"\x1a\x45\xdf\xa3") and b"webm" in header.lower():
        return ".webm"

    if not header:
        return ""
    return ""


async def download_google_drive_file(
    file_id: str, work_dir: Path
) -> tuple[Path, str]:
    download_url = (
        "https://drive.usercontent.google.com/download"
        f"?id={file_id}&export=download&confirm=t"
    )
    temporary_path = work_dir / "cloud-download"
    filename = ""
    total = 0
    first_chunk = b""

    timeout = httpx.Timeout(connect=20, read=300, write=30, pool=20)
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            async with client.stream("GET", download_url) as response:
                if response.status_code in {401, 403}:
                    raise CloudImportError(
                        "Google Drive не разрешил скачать файл без авторизации. "
                        "Откройте доступ по ссылке или скачайте видео и загрузите его вручную."
                    )
                response.raise_for_status()

                content_length = response.headers.get("content-length")
                if content_length and int(content_length) > MAX_UPLOAD_BYTES:
                    raise CloudImportError(
                        "Файл в Google Drive слишком большой. Максимальный размер — 250 МБ."
                    )

                content_type = response.headers.get("content-type", "").lower()
                filename = _filename_from_content_disposition(
                    response.headers.get("content-disposition", "")
                )
                with temporary_path.open("wb") as destination:
                    async for chunk in response.aiter_bytes(CHUNK_SIZE):
                        if not first_chunk:
                            first_chunk = chunk[:512].lower()
                        total += len(chunk)
                        if total > MAX_UPLOAD_BYTES:
                            raise CloudImportError(
                                "Файл в Google Drive слишком большой. "
                                "Максимальный размер — 250 МБ."
                            )
                        destination.write(chunk)
    except CloudImportError:
        raise
    except (httpx.HTTPError, OSError, ValueError) as exc:
        raise CloudImportError(
            "Не удалось получить файл из Google Drive. Проверьте ссылку и доступ, "
            "либо скачайте видео и загрузите его вручную."
        ) from exc

    if not total:
        raise CloudImportError("Google Drive вернул пустой файл.")
    if "text/html" in content_type or b"<html" in first_chunk or b"<!doctype" in first_chunk:
        raise CloudImportError(
            "По ссылке открывается страница Google Drive, а не файл. Разрешите "
            "доступ всем, у кого есть ссылка, либо загрузите видео вручную."
        )

    detected_extension = await asyncio.to_thread(detect_video_extension, temporary_path)
    supplied_extension = Path(filename).suffix.lower()
    if supplied_extension and supplied_extension not in ALLOWED_EXTENSIONS:
        raise CloudImportError(
            "Файл в Google Drive имеет неподдерживаемый формат. Нужен WebM или MP4."
        )
    if not detected_extension:
        raise CloudImportError(
            "Скачанный файл не является видео WebM или MP4 либо повреждён."
        )

    extension = supplied_extension or detected_extension
    if extension != detected_extension:
        raise CloudImportError(
            "Расширение файла не соответствует его содержимому. "
            "Загрузите корректный WebM или MP4."
        )

    safe_name = filename or f"google-drive-{file_id}{extension}"
    final_path = work_dir / f"video{extension}"
    temporary_path.replace(final_path)
    return final_path, safe_name


def public_task(task: dict[str, Any]) -> dict[str, Any]:
    return {
        key: task.get(key)
        for key in (
            "id",
            "filename",
            "source",
            "status",
            "normalized_url",
            "transcript",
            "summary",
            "error",
        )
    }


def set_task_state(task_id: str, **changes: Any) -> None:
    task = analysis_tasks.get(task_id)
    if task:
        task.update(changes)


def keep_background_task(coroutine: Any) -> None:
    task = asyncio.create_task(coroutine)
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)


async def run_analysis_task(
    task_id: str,
    work_dir: Path,
    video_path: Path | None = None,
    drive_file_id: str | None = None,
) -> None:
    try:
        if drive_file_id:
            video_path, filename = await download_google_drive_file(
                drive_file_id, work_dir
            )
            set_task_state(task_id, filename=filename)

        if video_path is None:
            raise MediaProcessingError("Видео для обработки не найдено.")

        set_task_state(task_id, status="transcribing")
        audio_path = work_dir / "audio.mp3"
        client = get_openai_client()
        transcription_path = await asyncio.to_thread(
            prepare_transcription_file, video_path, audio_path
        )
        transcript = await asyncio.to_thread(
            transcribe_audio, client, transcription_path
        )

        set_task_state(task_id, status="analyzing", transcript=transcript)
        summary = await asyncio.to_thread(create_summary, client, transcript)
        set_task_state(task_id, status="ready", summary=summary, error="")
    except (ConfigurationError, CloudImportError, MediaProcessingError, AnalysisError) as exc:
        set_task_state(task_id, status="error", error=str(exc))
    except Exception:
        set_task_state(
            task_id,
            status="error",
            error="Неожиданная ошибка обработки. Повторите добавление записи.",
        )
    finally:
        await asyncio.to_thread(shutil.rmtree, work_dir, True)


async def save_upload(upload: UploadFile, target: Path) -> None:
    total = 0
    with target.open("wb") as destination:
        while chunk := await upload.read(CHUNK_SIZE):
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail="Файл слишком большой. Максимальный размер — 250 МБ.",
                )
            destination.write(chunk)

    if total == 0:
        raise HTTPException(status_code=400, detail="Загружен пустой файл.")


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "configured": bool(PROXY_API_API_KEY and PROXY_API_OPENAI_BASE_URL),
    }


def active_task_exists() -> bool:
    return any(
        task.get("status") in ACTIVE_TASK_STATUSES for task in analysis_tasks.values()
    )


@app.post("/api/tasks/upload", status_code=202)
async def create_upload_task(file: UploadFile = File(...)) -> dict[str, Any]:
    original_name = Path(file.filename or "video").name
    extension = Path(original_name).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Неподдерживаемый формат. Загрузите видео WebM или MP4.",
        )

    async with task_registry_lock:
        if active_task_exists():
            raise HTTPException(
                status_code=409,
                detail="Другая запись уже обрабатывается. Дождитесь её завершения.",
            )

    work_dir = Path(tempfile.mkdtemp(prefix="meeting-task-"))
    video_path = work_dir / f"video{extension}"
    try:
        await save_upload(file, video_path)
    except Exception:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise
    finally:
        await file.close()

    task_id = str(uuid.uuid4())
    task = {
        "id": task_id,
        "filename": original_name,
        "source": "manual",
        "status": "transcribing",
        "normalized_url": None,
        "transcript": "",
        "summary": None,
        "error": "",
    }
    async with task_registry_lock:
        if active_task_exists():
            shutil.rmtree(work_dir, ignore_errors=True)
            raise HTTPException(
                status_code=409,
                detail="Другая запись уже обрабатывается. Дождитесь её завершения.",
            )
        analysis_tasks[task_id] = task

    if IS_VERCEL:
        # A serverless invocation may be frozen as soon as the response is sent.
        # Keep the request open so processing reliably finishes on Vercel.
        await run_analysis_task(task_id, work_dir, video_path=video_path)
    else:
        keep_background_task(run_analysis_task(task_id, work_dir, video_path=video_path))
    return public_task(task)


@app.post("/api/tasks/cloud", status_code=202)
async def create_cloud_task(payload: CloudImportRequest) -> dict[str, Any]:
    try:
        file_id, normalized_url = normalize_google_drive_url(payload.url)
    except CloudImportError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    async with task_registry_lock:
        existing_id = cloud_link_tasks.get(normalized_url)
        existing = analysis_tasks.get(existing_id or "")
        if existing and existing.get("status") != "error":
            return {**public_task(existing), "duplicate": True}

        if active_task_exists():
            raise HTTPException(
                status_code=409,
                detail="Другая запись уже обрабатывается. Дождитесь её завершения.",
            )

        task_id = str(uuid.uuid4())
        task = {
            "id": task_id,
            "filename": "Видео из Google Drive",
            "source": "cloud",
            "status": "fetching",
            "normalized_url": normalized_url,
            "transcript": "",
            "summary": None,
            "error": "",
        }
        analysis_tasks[task_id] = task
        cloud_link_tasks[normalized_url] = task_id

    work_dir = Path(tempfile.mkdtemp(prefix="meeting-cloud-task-"))
    if IS_VERCEL:
        await run_analysis_task(task_id, work_dir, drive_file_id=file_id)
    else:
        keep_background_task(
            run_analysis_task(task_id, work_dir, drive_file_id=file_id)
        )
    return {**public_task(task), "duplicate": False}


@app.get("/api/tasks/{task_id}")
async def get_analysis_task(task_id: str) -> dict[str, Any]:
    task = analysis_tasks.get(task_id)
    if not task:
        raise HTTPException(
            status_code=404,
            detail="Задача не найдена. Добавьте запись повторно.",
        )
    return public_task(task)


@app.post("/api/analyze")
async def analyze_video(file: UploadFile = File(...)) -> dict[str, Any]:
    original_name = Path(file.filename or "video").name
    extension = Path(original_name).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Неподдерживаемый формат. Загрузите видео WebM или MP4.",
        )

    if processing_lock.locked():
        raise HTTPException(
            status_code=409,
            detail="Другая запись уже обрабатывается. Дождитесь её завершения.",
        )

    async with processing_lock:
        with tempfile.TemporaryDirectory(prefix="meeting-analyzer-") as temp_dir:
            temp_path = Path(temp_dir)
            video_path = temp_path / f"upload{extension}"
            audio_path = temp_path / "audio.mp3"

            try:
                await save_upload(file, video_path)
                result = await asyncio.to_thread(process_video, video_path, audio_path)
            except HTTPException:
                raise
            except ConfigurationError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            except MediaProcessingError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            except AnalysisError as exc:
                raise HTTPException(status_code=502, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(
                    status_code=500,
                    detail="Неожиданная ошибка обработки. Повторите загрузку файла.",
                ) from exc
            finally:
                await file.close()

    return {"filename": original_name, "status": "ready", **result}


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
