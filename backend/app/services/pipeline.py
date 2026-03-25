import copy
import json
from pathlib import Path
from typing import Callable

from app.core.config import settings
from app.services.aws_gateway import generate_signed_url, upload_bytes_via_signed_url
from app.services.transcript import TranscriptService
from app.services.translation import TranslationService
from app.services.tts import TTSService


class PipelineStageError(RuntimeError):
    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


def items_to_sentences(items, end_punctuations=(".", "!", "?")):
    end_punctuations = set(end_punctuations)
    sentences = []

    current_tokens = []
    sentence_start_id = None
    sentence_start_time = None
    sentence_end_time = None

    def flush_sentence():
        nonlocal current_tokens
        nonlocal sentence_start_id, sentence_start_time, sentence_end_time

        if not current_tokens:
            return

        sentences.append(
            {
                "id": sentence_start_id,
                "start_time": sentence_start_time,
                "end_time": sentence_end_time,
                "transcript": "".join(current_tokens).strip(),
            }
        )

        current_tokens = []
        sentence_start_id = None
        sentence_start_time = None
        sentence_end_time = None

    for item in items:
        item_type = item.get("type")
        alt = (item.get("alternatives") or [{}])[0]
        content = alt.get("content", "")

        if item_type == "pronunciation":
            if sentence_start_id is None:
                sentence_start_id = item.get("id")
                sentence_start_time = item.get("start_time")

            if current_tokens:
                current_tokens.append(" ")
            current_tokens.append(content)

            if item.get("end_time") is not None:
                sentence_end_time = item.get("end_time")

        elif item_type == "punctuation":
            if current_tokens:
                current_tokens.append(content)
                if content in end_punctuations:
                    flush_sentence()

    flush_sentence()
    return sentences


def _normalize_transcript_to_items(transcript_payload: dict | list) -> list:
    if isinstance(transcript_payload, list):
        return transcript_payload

    if isinstance(transcript_payload, dict):
        if isinstance(transcript_payload.get("results"), dict) and isinstance(
            transcript_payload["results"].get("items"), list
        ):
            return transcript_payload["results"]["items"]

        if isinstance(transcript_payload.get("data"), list):
            mapped_items = []
            for idx, row in enumerate(transcript_payload["data"]):
                start = float(row.get("start", 0))
                duration = float(row.get("duration", 0))
                mapped_items.append(
                    {
                        "id": idx,
                        "type": "pronunciation",
                        "start_time": f"{start:.3f}".rstrip("0").rstrip("."),
                        "end_time": f"{(start + duration):.3f}".rstrip("0").rstrip("."),
                        "alternatives": [{"content": row.get("text", ""), "confidence": "1.0"}],
                    }
                )
            return mapped_items

    raise ValueError("Unsupported transcript format")


def _segment_filename_from_start_time(start_time: str) -> str:
    return f"{str(start_time).replace('.', '_')}.wav"


class PipelineService:
    def __init__(self) -> None:
        self.transcript_service = TranscriptService()
        self.translation_service = TranslationService()
        self.tts_service = TTSService()

    def process_job(
        self,
        job_id: str,
        video_bytes: bytes,
        filename: str,
        transcript_key: str | None,
        on_stage_change: Callable[[str], None] | None = None,
    ) -> dict:
        if on_stage_change:
            on_stage_change("upload_video")
        try:
            uploaded_s3_key = self.transcript_service.upload_video(video_bytes, filename, job_id)
        except Exception as exc:  # noqa: BLE001
            raise PipelineStageError("upload_video", str(exc)) from exc

        effective_transcript_key = transcript_key or self.transcript_service.transcript_key_from_upload_key(uploaded_s3_key)
        return self.resume_from_stage(job_id, effective_transcript_key, on_stage_change=on_stage_change)

    def resume_from_stage(
        self,
        job_id: str,
        transcript_key: str | None,
        resume_stage: str = "wait_transcript",
        on_stage_change: Callable[[str], None] | None = None,
    ) -> dict:
        if resume_stage not in {"wait_transcript", "translate_and_tts", "upload_manifest"}:
            resume_stage = "wait_transcript"

        if resume_stage == "wait_transcript" and on_stage_change:
            on_stage_change("wait_transcript")
        try:
            transcript_payload = self.transcript_service.wait_for_transcript(job_id, transcript_key=transcript_key)
        except Exception as exc:  # noqa: BLE001
            raise PipelineStageError("wait_transcript", str(exc)) from exc

        items = _normalize_transcript_to_items(transcript_payload)
        sentence_items = items_to_sentences(items)

        if on_stage_change:
            on_stage_change("translate_and_tts")
        try:
            translated_segments = self._translate_and_generate_audio(job_id, sentence_items)
        except Exception as exc:  # noqa: BLE001
            raise PipelineStageError("translate_and_tts", str(exc)) from exc

        final_result = {
            "results": {
                "translated_segments": translated_segments,
            }
        }

        if on_stage_change:
            on_stage_change("upload_manifest")
        try:
            self._upload_translation_manifest(job_id, final_result)
        except Exception as exc:  # noqa: BLE001
            raise PipelineStageError("upload_manifest", str(exc)) from exc

        return final_result

    def translate_text(self, text: str) -> str:
        return self.translation_service.translate(text)

    def generate_audio_for_translated_segments(
        self, job_id: str, segments: list[dict], speaker_id: int | None = None
    ) -> list[dict]:
        translated = copy.deepcopy(segments)

        for item in translated:
            transcript_rw = item.get("transcript_rw", "")

            start_time = str(item.get("start_time", "0"))
            audio_file_name = _segment_filename_from_start_time(start_time)
            local_audio_path = settings.temp_dir / job_id / audio_file_name
            self.tts_service.synthesize_to_file(transcript_rw, local_audio_path, speaker_id=speaker_id)

            s3_key = f"{settings.public_prefix.strip('/')}/{job_id}/{audio_file_name}"
            signed = generate_signed_url(
                s3_key=s3_key,
                bucket=settings.artifact_bucket,
                action="put_object",
                content_type="audio/wav",
            )
            upload_url = signed.get("url")
            if not upload_url:
                raise RuntimeError("No upload URL returned for audio upload")

            with open(local_audio_path, "rb") as audio_fp:
                upload_bytes_via_signed_url(upload_url, audio_fp.read(), "audio/wav")

            item["audio_file_name"] = audio_file_name
            item["audio_file_url"] = f"{settings.public_artifact_store.rstrip('/')}/{s3_key}"

            try:
                local_audio_path.unlink(missing_ok=True)
            except OSError:
                pass

        return translated

    def _translate_and_generate_audio(self, job_id: str, segments: list[dict]) -> list[dict]:
        translated = copy.deepcopy(segments)

        for item in translated:
            transcript_en = item.get("transcript", "")
            item["transcript_en"] = transcript_en
            item["transcript_rw"] = self.translate_text(transcript_en)

        return self.generate_audio_for_translated_segments(job_id, translated)

    def _upload_translation_manifest(self, job_id: str, result_payload: dict) -> None:
        s3_key = f"{settings.public_prefix.strip('/')}/{job_id}/translation.json"
        signed = generate_signed_url(
            s3_key=s3_key,
            bucket=settings.artifact_bucket,
            action="put_object",
            content_type="application/json",
        )
        upload_url = signed.get("url")
        if not upload_url:
            raise RuntimeError("No upload URL returned for translation manifest")

        payload_bytes = json.dumps(result_payload, ensure_ascii=False).encode("utf-8")
        upload_bytes_via_signed_url(upload_url, payload_bytes, "application/json")
