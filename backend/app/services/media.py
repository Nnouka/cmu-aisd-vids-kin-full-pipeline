import subprocess
from pathlib import Path


class UploadValidationError(ValueError):
    pass


def guess_content_type(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    mapping = {
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".m4v": "video/x-m4v",
        ".webm": "video/webm",
    }
    return mapping.get(ext, "application/octet-stream")


def validate_extension(filename: str, allowed_extensions_csv: str) -> None:
    ext = Path(filename).suffix.lower().lstrip(".")
    allowed = {item.strip().lower() for item in allowed_extensions_csv.split(",") if item.strip()}
    if ext not in allowed:
        raise UploadValidationError(f"Unsupported file extension: .{ext}. Allowed: {sorted(allowed)}")


def detect_video_duration_seconds(file_path: str) -> float:
    # ffprobe is used so we can enforce max duration before starting expensive pipeline work.
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        file_path,
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())
