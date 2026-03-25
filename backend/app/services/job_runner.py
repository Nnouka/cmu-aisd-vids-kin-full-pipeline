from concurrent.futures import ThreadPoolExecutor

from app.services.pipeline import PipelineStageError
from app.services.runtime import pipeline_service
from app.services.storage import mark_job_failed, set_job_result, update_job_stage, update_job_status


executor = ThreadPoolExecutor(max_workers=2)


def submit_job(job_id: str, video_bytes: bytes, filename: str, transcript_key: str | None = None) -> None:
    executor.submit(_run_pipeline_job, job_id, video_bytes, filename, transcript_key)


def _run_pipeline_job(job_id: str, video_bytes: bytes, filename: str, transcript_key: str | None) -> None:
    try:
        update_job_status(job_id, "processing")
        result = pipeline_service.process_job(
            job_id,
            video_bytes,
            filename,
            transcript_key,
            on_stage_change=lambda stage: update_job_stage(job_id, stage),
        )
        set_job_result(job_id, result)
    except PipelineStageError as exc:
        mark_job_failed(job_id, error=str(exc), failed_stage=exc.stage)
    except Exception as exc:  # noqa: BLE001
        mark_job_failed(job_id, error=str(exc), failed_stage="unknown")


def submit_continue_job(job_id: str, resume_stage: str, transcript_key: str | None = None) -> None:
    executor.submit(_run_pipeline_continue_job, job_id, resume_stage, transcript_key)


def _run_pipeline_continue_job(job_id: str, resume_stage: str, transcript_key: str | None) -> None:
    try:
        update_job_status(job_id, "processing")
        update_job_stage(job_id, resume_stage)
        result = pipeline_service.resume_from_stage(
            job_id,
            transcript_key,
            resume_stage=resume_stage,
            on_stage_change=lambda stage: update_job_stage(job_id, stage),
        )
        set_job_result(job_id, result)
    except PipelineStageError as exc:
        mark_job_failed(job_id, error=str(exc), failed_stage=exc.stage)
    except Exception as exc:  # noqa: BLE001
        mark_job_failed(job_id, error=str(exc), failed_stage="unknown")
