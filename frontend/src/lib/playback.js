import { expectedAudioFilename, parseTime } from "./formatters";

export function createPlaybackHoldState() {
  return {
    active: false,
    segmentKey: "",
    resumeAt: null,
  };
}

export function getSegmentBounds(segments, index) {
  const segment = segments[index];
  if (!segment) {
    return null;
  }

  const start = parseTime(segment.start_time);
  const fallbackEnd = parseTime(segment.end_time || start + 2);
  const nextStart = index < segments.length - 1 ? parseTime(segments[index + 1].start_time) : null;
  const end = nextStart === null ? fallbackEnd : Math.min(fallbackEnd, nextStart);

  return { start, end };
}

export function findSegmentIndexAtTime(segments, currentTime) {
  return segments.findIndex((_, index) => {
    const bounds = getSegmentBounds(segments, index);
    if (!bounds) {
      return false;
    }

    return currentTime >= bounds.start && currentTime < bounds.end;
  });
}

export function segmentKeyFor(segment) {
  return segment.audio_file_name || expectedAudioFilename(segment.start_time);
}

export function loadAudioMetadata(audio) {
  if (Number.isFinite(audio.duration) && audio.duration > 0) {
    return Promise.resolve();
  }

  return new Promise((resolve, reject) => {
    const handleLoadedMetadata = () => {
      cleanup();
      resolve();
    };

    const handleError = () => {
      cleanup();
      reject(new Error("Unable to read generated audio metadata."));
    };

    const cleanup = () => {
      audio.removeEventListener("loadedmetadata", handleLoadedMetadata);
      audio.removeEventListener("error", handleError);
    };

    audio.addEventListener("loadedmetadata", handleLoadedMetadata);
    audio.addEventListener("error", handleError);
    audio.load();
  });
}