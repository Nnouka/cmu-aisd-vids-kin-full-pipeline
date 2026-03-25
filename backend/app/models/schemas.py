from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class JobCreateResponse(BaseModel):
    job_id: str
    status: str


class Segment(BaseModel):
    id: int | str | None = None
    start_time: str
    end_time: str | None = None
    transcript: str
    transcript_en: str
    transcript_rw: str
    audio_file_name: str
    audio_file_url: str


class JobResult(BaseModel):
    results: dict[str, Any]


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    created_at: datetime
    updated_at: datetime
    error: str | None = None
    result: JobResult | None = None


class ErrorResponse(BaseModel):
    detail: str = Field(..., examples=["Invalid upload"])
