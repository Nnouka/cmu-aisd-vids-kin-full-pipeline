import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import settings


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _conn():
    db_parent = Path(settings.jobs_db_path).parent
    db_parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.jobs_db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                error TEXT,
                input_filename TEXT,
                transcript_key TEXT,
                video_s3_key TEXT,
                result_json TEXT
            )
            """
        )


def create_job(job_id: str, input_filename: str, transcript_key: str | None) -> None:
    now = _utc_now()
    with _conn() as conn:
        conn.execute(
            """
            INSERT INTO jobs (job_id, status, created_at, updated_at, input_filename, transcript_key)
            VALUES (?, 'queued', ?, ?, ?, ?)
            """,
            (job_id, now, now, input_filename, transcript_key),
        )


def update_job_status(job_id: str, status: str, error: str | None = None) -> None:
    now = _utc_now()
    with _conn() as conn:
        conn.execute(
            """
            UPDATE jobs
            SET status = ?, updated_at = ?, error = COALESCE(?, error)
            WHERE job_id = ?
            """,
            (status, now, error, job_id),
        )


def set_video_key(job_id: str, video_s3_key: str) -> None:
    now = _utc_now()
    with _conn() as conn:
        conn.execute(
            """
            UPDATE jobs
            SET video_s3_key = ?, updated_at = ?
            WHERE job_id = ?
            """,
            (video_s3_key, now, job_id),
        )


def set_job_result(job_id: str, result: dict) -> None:
    now = _utc_now()
    with _conn() as conn:
        conn.execute(
            """
            UPDATE jobs
            SET status = 'completed', updated_at = ?, result_json = ?
            WHERE job_id = ?
            """,
            (now, json.dumps(result, ensure_ascii=False), job_id),
        )


def get_job(job_id: str) -> dict | None:
    with _conn() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            return None

        data = dict(row)
        if data.get("result_json"):
            data["result_json"] = json.loads(data["result_json"])
        return data
