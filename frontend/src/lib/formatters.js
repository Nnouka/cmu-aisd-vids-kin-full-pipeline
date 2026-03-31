export function formatStageLabel(stage) {
  return stage.replaceAll("_", " ");
}

export function parseTime(value) {
  const parsed = Number.parseFloat(String(value ?? "0"));
  return Number.isFinite(parsed) ? parsed : 0;
}

export async function getVideoDuration(file) {
  const objectUrl = URL.createObjectURL(file);

  try {
    const duration = await new Promise((resolve, reject) => {
      const probe = document.createElement("video");
      probe.preload = "metadata";
      probe.src = objectUrl;
      probe.onloadedmetadata = () => resolve(probe.duration || 0);
      probe.onerror = () => reject(new Error("Unable to read video metadata."));
    });
    return duration;
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}

export function expectedAudioFilename(startTime) {
  return `${String(startTime).replaceAll(".", "_")}.wav`;
}

export function deriveJobIdFromFilename(filename) {
  const cleaned = (filename || "").trim();
  if (!cleaned) {
    return "";
  }

  const dot = cleaned.lastIndexOf(".");
  const stem = dot > 0 ? cleaned.slice(0, dot) : cleaned;
  return stem.trim();
}

export function formatClock(totalSeconds) {
  const seconds = Math.max(0, Math.floor(totalSeconds || 0));
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
}

export function formatTimestamp(value) {
  if (!value) {
    return "-";
  }

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "-";
  }

  return date.toLocaleString();
}

export function formatDuration(durationSeconds) {
  if (typeof durationSeconds !== "number" || Number.isNaN(durationSeconds) || durationSeconds < 0) {
    return "-";
  }

  const rounded = Math.round(durationSeconds);
  const hours = Math.floor(rounded / 3600);
  const mins = Math.floor((rounded % 3600) / 60);
  const secs = rounded % 60;

  if (hours > 0) {
    return `${hours}h ${String(mins).padStart(2, "0")}m ${String(secs).padStart(2, "0")}s`;
  }

  return `${mins}m ${String(secs).padStart(2, "0")}s`;
}

export function computeDurationFromTimestamps(startedAtValue, completedAtValue, nowMs = Date.now()) {
  if (!startedAtValue) {
    return null;
  }

  const startedAtMs = new Date(startedAtValue).getTime();
  if (Number.isNaN(startedAtMs)) {
    return null;
  }

  const completedAtMs = completedAtValue ? new Date(completedAtValue).getTime() : nowMs;
  if (Number.isNaN(completedAtMs)) {
    return null;
  }

  return Math.max((completedAtMs - startedAtMs) / 1000, 0);
}