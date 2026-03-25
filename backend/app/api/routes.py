from datetime import datetime
from pathlib import Path
import tempfile
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.core.config import settings
from app.models.schemas import (
    JobCreateResponse,
    JobStatusResponse,
    TranscriptByVideoNameResponse,
    TranslateTextRequest,
    TranslateTextResponse,
    TtsTextRequest,
    TtsTextResponse,
    TtsTranslatedItemsRequest,
    TtsTranslatedItemsResponse,
)
from app.services.job_runner import submit_job
from app.services.media import UploadValidationError, detect_video_duration_seconds, validate_extension
from app.services.runtime import get_model_readiness, pipeline_service
from app.services.storage import create_job, get_job
from app.services.transcript import TranscriptService

router = APIRouter()
transcript_service = TranscriptService()


def _inspect_duration_sync(payload: bytes, filename: str) -> float:
    with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix, delete=True) as tmp:
        tmp.write(payload)
        tmp.flush()
        return detect_video_duration_seconds(tmp.name)


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.get("/ready")
def ready() -> dict:
    tts_model_exists = Path(settings.tts_model_path).exists()
    readiness = get_model_readiness()
    all_ready = tts_model_exists and readiness["translation_model_loaded"] and readiness["tts_model_loaded"]
    return {
        "status": "ok" if all_ready else "degraded",
        "tts_model_exists": tts_model_exists,
        **readiness,
    }


@router.post("/jobs", response_model=JobCreateResponse, responses={400: {"description": "Invalid upload"}})
async def create_processing_job(
    file: Annotated[UploadFile, File(...)],
    transcript_key: Annotated[str | None, Form()] = None,
):
    filename = file.filename or "upload.mp4"
    validate_extension(filename, settings.allowed_extensions)

    payload = await file.read()
    if len(payload) > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"Upload exceeds size limit of {settings.max_upload_size_bytes} bytes",
        )

    try:
        duration = _inspect_duration_sync(payload, filename)
    except UploadValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Failed to inspect video duration: {exc}") from exc

    if duration > settings.max_video_duration_seconds:
        raise HTTPException(
            status_code=400,
            detail=f"Video duration {duration:.2f}s exceeds limit of {settings.max_video_duration_seconds}s",
        )

    job_id = str(uuid4())
    create_job(job_id, input_filename=filename, transcript_key=transcript_key)
    submit_job(job_id, payload, filename, transcript_key=transcript_key)

    return JobCreateResponse(job_id=job_id, status="queued")


@router.get("/jobs/{job_id}", response_model=JobStatusResponse, responses={404: {"description": "Job not found"}})
def get_processing_job(job_id: str):
    row = get_job(job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatusResponse(
        job_id=row["job_id"],
        status=row["status"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        error=row.get("error"),
        result=row.get("result_json"),
    )


@router.post("/test/translate", response_model=TranslateTextResponse)
def test_translate_text(payload: TranslateTextRequest):
    translated = pipeline_service.translate_text(payload.text)
    return TranslateTextResponse(transcript_en=payload.text, transcript_rw=translated)


@router.post("/test/tts", response_model=TtsTextResponse)
def test_tts_single_text(payload: TtsTextRequest):
    job_id = str(uuid4())
    segments = [
        {
            "id": "single",
            "start_time": payload.start_time,
            "end_time": None,
            "transcript_en": None,
            "transcript_rw": payload.text,
        }
    ]
    rendered_segments = pipeline_service.generate_audio_for_translated_segments(
        job_id, segments, speaker_id=payload.speaker_id
    )
    rendered = rendered_segments[0]

    return TtsTextResponse(
        job_id=job_id,
        start_time=str(rendered["start_time"]),
        transcript_rw=rendered["transcript_rw"],
        audio_file_name=rendered["audio_file_name"],
        audio_file_url=rendered["audio_file_url"],
    )


@router.post("/test/tts-translated-items", response_model=TtsTranslatedItemsResponse)
def test_tts_translated_items(payload: TtsTranslatedItemsRequest):
    job_id = str(uuid4())
    translated_segments = pipeline_service.generate_audio_for_translated_segments(
        job_id,
        [item.model_dump() for item in payload.items],
        speaker_id=payload.speaker_id,
    )
    return TtsTranslatedItemsResponse(job_id=job_id, translated_segments=translated_segments)


@router.get(
    "/test/transcript/{video_file_name}",
    response_model=TranscriptByVideoNameResponse,
    responses={400: {"description": "Transcript fetch failed"}, 404: {"description": "Transcript not found"}},
)
def test_get_transcript_by_video_name(video_file_name: str):
    transcript_key = f"transcripts/{Path(video_file_name).stem}.json"
    try:
        transcript_payload = transcript_service.wait_for_transcript("standalone", transcript_key=transcript_key)
    except TimeoutError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Failed to fetch transcript: {exc}") from exc

    return TranscriptByVideoNameResponse(
        video_file_name=video_file_name,
        transcript_key=transcript_key,
        transcript=transcript_payload,
    )
