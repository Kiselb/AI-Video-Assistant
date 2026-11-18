from __future__ import annotations

import io
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import main


client = TestClient(main.app)


@pytest.fixture(autouse=True)
def reset_task_registry():
    main.analysis_tasks.clear()
    main.cloud_link_tasks.clear()
    yield
    main.analysis_tasks.clear()
    main.cloud_link_tasks.clear()


def test_index_is_served() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "AI-анализ видеовстреч" in response.text


def test_health_reports_configuration_state() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert isinstance(response.json()["configured"], bool)


def test_rejects_unsupported_extension() -> None:
    response = client.post(
        "/api/analyze",
        files={"file": ("notes.txt", io.BytesIO(b"not a video"), "text/plain")},
    )
    assert response.status_code == 400
    assert "WebM или MP4" in response.json()["detail"]


def test_analyze_returns_result(monkeypatch) -> None:
    expected = {
        "transcript": "Обсудили запуск продукта.",
        "summary": {
            "brief": "Команда обсудила запуск продукта.",
            "decisions": ["Запустить продукт в пятницу."],
            "tasks": [],
            "open_questions": ["Какой бюджет на продвижение?"],
        },
    }
    monkeypatch.setattr(main, "process_video", lambda *_: expected)

    response = client.post(
        "/api/analyze",
        files={"file": ("meeting.webm", io.BytesIO(b"fake video"), "video/webm")},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert response.json()["summary"] == expected["summary"]


def test_parse_summary_normalizes_arrays() -> None:
    summary = main.parse_summary(
        '```json\n{"brief":"Итог", "decisions":["Решение"], '
        '"tasks":"нет", "open_questions":[]}\n```'
    )
    assert summary == {
        "brief": "Итог",
        "decisions": ["Решение"],
        "tasks": [],
        "open_questions": [],
    }


def test_transcription_uses_proxy_supported_json_format() -> None:
    captured = {}

    class FakeTranscriptions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(text="Тестовая расшифровка")

    fake_client = SimpleNamespace(audio=SimpleNamespace(transcriptions=FakeTranscriptions()))

    with TemporaryDirectory(dir=Path(__file__).parent) as temp_dir:
        audio_path = Path(temp_dir) / "audio.mp3"
        audio_path.write_bytes(b"fake audio")
        transcript = main.transcribe_audio(fake_client, audio_path)

    assert transcript == "Тестовая расшифровка"
    assert captured["response_format"] == "json"


def test_detects_mp4_and_webm_without_ffprobe() -> None:
    with TemporaryDirectory(dir=Path(__file__).parent) as temp_dir:
        temp_path = Path(temp_dir)
        mp4_path = temp_path / "video"
        webm_path = temp_path / "other"
        invalid_path = temp_path / "invalid"

        mp4_path.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 32)
        webm_path.write_bytes(b"\x1a\x45\xdf\xa3" + b"\x00webm\x00" + b"\x00" * 32)
        invalid_path.write_bytes(b"not a media file")

        assert main.detect_video_extension(mp4_path) == ".mp4"
        assert main.detect_video_extension(webm_path) == ".webm"
        assert main.detect_video_extension(invalid_path) == ""


def test_uses_original_video_when_ffmpeg_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(main, "find_ffmpeg_executable", lambda: "")

    with TemporaryDirectory(dir=Path(__file__).parent) as temp_dir:
        temp_path = Path(temp_dir)
        video_path = temp_path / "video.mp4"
        video_path.write_bytes(b"small video")

        assert main.prepare_transcription_file(
            video_path, temp_path / "audio.mp3"
        ) == video_path


def test_direct_transcription_rejects_oversized_video(monkeypatch) -> None:
    monkeypatch.setattr(main, "find_ffmpeg_executable", lambda: "")

    with TemporaryDirectory(dir=Path(__file__).parent) as temp_dir:
        temp_path = Path(temp_dir)
        video_path = temp_path / "video.mp4"
        with video_path.open("wb") as video_file:
            video_file.truncate(main.DIRECT_TRANSCRIPTION_MAX_BYTES + 1)

        with pytest.raises(main.MediaProcessingError, match="25 МБ"):
            main.prepare_transcription_file(video_path, temp_path / "audio.mp3")


@pytest.mark.parametrize(
    ("url", "expected_id"),
    [
        (
            "https://drive.google.com/file/d/1AbCdEfGhIjKlMnOp/view?usp=sharing",
            "1AbCdEfGhIjKlMnOp",
        ),
        (
            "https://drive.google.com/open?id=1AbCdEfGhIjKlMnOp",
            "1AbCdEfGhIjKlMnOp",
        ),
    ],
)
def test_normalize_google_drive_url(url, expected_id) -> None:
    file_id, normalized = main.normalize_google_drive_url(url)
    assert file_id == expected_id
    assert normalized == f"https://drive.google.com/file/d/{expected_id}"


def test_rejects_non_google_cloud_url() -> None:
    with pytest.raises(main.CloudImportError, match="Google Drive"):
        main.normalize_google_drive_url("https://example.com/video.mp4")


def test_manual_upload_creates_background_task(monkeypatch) -> None:
    monkeypatch.setattr(main, "keep_background_task", lambda coroutine: coroutine.close())

    response = client.post(
        "/api/tasks/upload",
        files={"file": ("meeting.mp4", io.BytesIO(b"fake video"), "video/mp4")},
    )

    assert response.status_code == 202
    assert response.json()["status"] == "transcribing"
    assert response.json()["source"] == "manual"
    assert client.get(f"/api/tasks/{response.json()['id']}").status_code == 200


def test_cloud_task_is_deduplicated_by_normalized_link(monkeypatch) -> None:
    monkeypatch.setattr(main, "keep_background_task", lambda coroutine: coroutine.close())
    first_url = "https://drive.google.com/file/d/1AbCdEfGhIjKlMnOp/view?usp=sharing"
    second_url = "https://drive.google.com/open?id=1AbCdEfGhIjKlMnOp"

    first = client.post("/api/tasks/cloud", json={"url": first_url})
    second = client.post("/api/tasks/cloud", json={"url": second_url})

    assert first.status_code == 202
    assert first.json()["status"] == "fetching"
    assert first.json()["source"] == "cloud"
    assert first.json()["duplicate"] is False
    assert second.status_code == 202
    assert second.json()["duplicate"] is True
    assert second.json()["id"] == first.json()["id"]


def test_vercel_cloud_task_finishes_before_response(monkeypatch) -> None:
    async def fake_analysis(task_id, *_args, **_kwargs):
        main.set_task_state(
            task_id,
            filename="meeting.mp4",
            status="ready",
            transcript="Расшифровка",
            summary={
                "brief": "Краткое содержание",
                "decisions": [],
                "tasks": [],
                "open_questions": [],
            },
        )

    monkeypatch.setattr(main, "IS_VERCEL", True)
    monkeypatch.setattr(main, "run_analysis_task", fake_analysis)

    response = client.post(
        "/api/tasks/cloud",
        json={
            "url": "https://drive.google.com/file/d/1AbCdEfGhIjKlMnOp/view"
        },
    )

    assert response.status_code == 202
    assert response.json()["status"] == "ready"
    assert response.json()["filename"] == "meeting.mp4"
