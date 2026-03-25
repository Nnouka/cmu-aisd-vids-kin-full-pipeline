import time
from pathlib import Path

import requests

from app.core.config import settings
from app.services.aws_gateway import download_json_via_signed_url, generate_signed_url, upload_bytes_via_signed_url
from app.services.media import guess_content_type


class TranscriptService:
    def upload_video(self, payload: bytes, filename: str, job_id: str) -> str:
        s3_key = f"uploads/{job_id}/{Path(filename).name}"
        signed = generate_signed_url(
            s3_key=s3_key,
            bucket=settings.input_bucket,
            action="put_object",
            content_type=guess_content_type(filename),
        )
        upload_url = signed.get("url")
        if not upload_url:
            raise RuntimeError("No upload URL returned by signed URL endpoint")

        upload_bytes_via_signed_url(upload_url, payload, guess_content_type(filename))
        return s3_key

    def wait_for_transcript(self, job_id: str, transcript_key: str | None = None) -> dict:
        key = transcript_key or f"transcripts/{job_id}.json"
        started = time.time()

        while (time.time() - started) < settings.transcript_timeout_seconds:
            try:
                signed = generate_signed_url(
                    s3_key=key,
                    bucket=settings.output_bucket,
                    action="get_object",
                    content_type="application/json",
                )
                download_url = signed.get("url")
                if not download_url:
                    raise RuntimeError("No download URL returned by signed URL endpoint")

                return download_json_via_signed_url(download_url)
            except requests.HTTPError as exc:
                status_code = exc.response.status_code if exc.response is not None else None
                if status_code == 404:
                    time.sleep(settings.transcript_poll_interval_seconds)
                    continue
                raise

        raise TimeoutError(
            f"Transcript was not ready for key {key} within {settings.transcript_timeout_seconds} seconds"
        )
