from concurrent.futures import ThreadPoolExecutor

from app.services.pipeline import PipelineService
from app.services.storage import set_job_result, update_job_status


executor = ThreadPoolExecutor(max_workers=2)
pipeline = PipelineService()


def submit_job(job_id: str, video_bytes: bytes, filename: str, transcript_key: str | None = None) -> None:
    executor.submit(_run_pipeline_job, job_id, video_bytes, filename, transcript_key)


def _run_pipeline_job(job_id: str, video_bytes: bytes, filename: str, transcript_key: str | None) -> None:
    try:
        update_job_status(job_id, "processing")
        result = pipeline.process_job(job_id, video_bytes, filename, transcript_key)
        set_job_result(job_id, result)
    except Exception as exc:  # noqa: BLE001
        update_job_status(job_id, "failed", error=str(exc))
