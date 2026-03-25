import requests

from app.core.config import settings


def generate_signed_url(
    s3_key: str,
    bucket: str,
    action: str,
    content_type: str = "application/octet-stream",
) -> dict:
    response = requests.post(
        settings.generate_signed_url_endpoint,
        json={"key": s3_key, "bucket": bucket, "action": action, "contentType": content_type},
        headers={"Content-Type": "application/json"},
        timeout=20,
    )
    response.raise_for_status()
    return response.json()


def upload_bytes_via_signed_url(upload_url: str, payload: bytes, content_type: str) -> None:
    response = requests.put(
        upload_url,
        data=payload,
        headers={"Content-Type": content_type},
        timeout=600,
    )
    response.raise_for_status()


def download_json_via_signed_url(download_url: str) -> dict:
    response = requests.get(download_url, timeout=120)
    response.raise_for_status()
    return response.json()
