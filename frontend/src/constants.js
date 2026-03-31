export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api";

export const MAX_FILE_SIZE = 10 * 1024 * 1024;
export const MAX_DURATION_SECONDS = 180;
export const POLL_INTERVAL_MS = Number(import.meta.env.VITE_POLL_INTERVAL_MS || 60000);

export const JOB_STAGES = [
  "queued",
  "upload_video",
  "wait_transcript",
  "translate_and_tts",
  "upload_manifest",
  "completed",
];

export const STATUS_COLOR = {
  idle: "default",
  validating: "processing",
  queued: "processing",
  processing: "processing",
  completed: "success",
  failed: "error",
};