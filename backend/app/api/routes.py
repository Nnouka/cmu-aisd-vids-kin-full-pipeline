from datetime import datetime
from pathlib import Path
import tempfile
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.core.config import settings
from app.models.schemas import JobCreateResponse, JobStatusResponse
from app.services.job_runner import submit_job
from app.services.media import UploadValidationError, detect_video_duration_seconds, validate_extension
from app.services.storage import create_job, get_job

router = APIRouter()


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
    model_exists = Path(settings.tts_model_path).exists()
    return {"status": "ok" if model_exists else "degraded", "tts_model_exists": model_exists}


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
