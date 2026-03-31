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
    started_at: datetime | None = None
    completed_at: datetime | None = None
    updated_at: datetime
    duration_seconds: float | None = None
    error: str | None = None
    current_stage: str | None = None
    last_failed_stage: str | None = None
    transcript_key_in_use: str | None = None
    result: JobResult | None = None


class ErrorResponse(BaseModel):
    detail: str = Field(..., examples=["Invalid upload"])


class TranslateTextRequest(BaseModel):
    text: str = Field(..., examples=["Hello from CMU"])


class TranslateTextResponse(BaseModel):
    transcript_en: str
    transcript_rw: str


class TtsTextRequest(BaseModel):
    text: str = Field(..., examples=["Muraho neza"])
    start_time: str = Field(default="0")
    speaker_id: int | None = None


class TtsTextResponse(BaseModel):
    job_id: str
    start_time: str
    transcript_rw: str
    audio_file_name: str
    audio_file_url: str


class TranslatedSegmentInput(BaseModel):
    id: int | str | None = None
    start_time: str
    end_time: str | None = None
    transcript_en: str | None = None
    transcript_rw: str


class TtsTranslatedItemsRequest(BaseModel):
    items: list[TranslatedSegmentInput]
    speaker_id: int | None = None


class TtsTranslatedItemsResponse(BaseModel):
    job_id: str
    translated_segments: list[Segment]


class TranscriptByVideoNameResponse(BaseModel):
    video_file_name: str
    transcript_key: str
    transcript: dict[str, Any] | list[Any]
