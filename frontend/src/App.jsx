import { useEffect, useMemo, useRef, useState } from "react";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api";

const MAX_FILE_SIZE = 10 * 1024 * 1024;
const MAX_DURATION_SECONDS = 180;
const POLL_INTERVAL_MS = Number(import.meta.env.VITE_POLL_INTERVAL_MS || 3000);

function parseTime(value) {
  const parsed = Number.parseFloat(String(value ?? "0"));
  return Number.isFinite(parsed) ? parsed : 0;
}

async function getVideoDuration(file) {
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

function expectedAudioFilename(startTime) {
  return `${String(startTime).replaceAll(".", "_")}.wav`;
}

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [transcriptKey, setTranscriptKey] = useState("");
  const [jobId, setJobId] = useState("");
  const [status, setStatus] = useState("idle");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [videoUrl, setVideoUrl] = useState("");
  const [currentVideoTime, setCurrentVideoTime] = useState(0);

  const videoRef = useRef(null);
  const pollHandleRef = useRef(null);
  const activeAudioRef = useRef(new Map());
  const playedSegmentsRef = useRef(new Set());
  const lastVideoTimeRef = useRef(0);

  const segments = useMemo(() => {
    const raw = result?.results?.translated_segments || [];
    return [...raw].sort((a, b) => parseTime(a.start_time) - parseTime(b.start_time));
  }, [result]);

  useEffect(() => {
    if (!selectedFile) {
      return undefined;
    }

    const objectUrl = URL.createObjectURL(selectedFile);
    setVideoUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [selectedFile]);

  useEffect(() => {
    if (!jobId || !(status === "queued" || status === "processing")) {
      return undefined;
    }

    pollHandleRef.current = globalThis.setInterval(async () => {
      const response = await fetch(`${API_BASE_URL}/jobs/${jobId}`);
      if (!response.ok) {
        setStatus("failed");
        setError("Failed to poll job status.");
        return;
      }

      const payload = await response.json();
      setStatus(payload.status);
      if (payload.status === "completed") {
        setResult(payload.result);
      }
      if (payload.status === "failed") {
        setError(payload.error || "Job failed.");
      }
    }, POLL_INTERVAL_MS);

    return () => {
      if (pollHandleRef.current) {
        globalThis.clearInterval(pollHandleRef.current);
        pollHandleRef.current = null;
      }
    };
  }, [jobId, status]);

  useEffect(() => {
    if (status === "completed" || status === "failed") {
      if (pollHandleRef.current) {
        globalThis.clearInterval(pollHandleRef.current);
        pollHandleRef.current = null;
      }
    }
  }, [status]);

  function stopAllAudio() {
    activeAudioRef.current.forEach((audio) => {
      audio.pause();
      audio.currentTime = 0;
    });
    activeAudioRef.current.clear();
  }

  function resetPlaybackState() {
    stopAllAudio();
    playedSegmentsRef.current = new Set();
    lastVideoTimeRef.current = 0;
    setCurrentVideoTime(0);
  }

  async function handleSubmit(event) {
    event.preventDefault();

    if (!selectedFile) {
      setError("Please choose a video file.");
      return;
    }

    setError("");
    setResult(null);
    setStatus("validating");

    if (selectedFile.size > MAX_FILE_SIZE) {
      setStatus("idle");
      setError("File exceeds 10 MB limit.");
      return;
    }

    try {
      const duration = await getVideoDuration(selectedFile);
      if (duration > MAX_DURATION_SECONDS) {
        setStatus("idle");
        setError("Video exceeds 3 minute limit.");
        return;
      }
    } catch (probeError) {
      setStatus("idle");
      setError(probeError.message || "Unable to read video metadata.");
      return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);
    if (transcriptKey.trim()) {
      formData.append("transcript_key", transcriptKey.trim());
    }

    setIsSubmitting(true);
    try {
      const response = await fetch(`${API_BASE_URL}/jobs`, {
        method: "POST",
        body: formData,
      });
      const payload = await response.json();

      if (!response.ok) {
        setStatus("failed");
        setError(payload.detail || "Failed to create processing job.");
        return;
      }

      setJobId(payload.job_id);
      setStatus(payload.status || "queued");
      resetPlaybackState();
    } catch (submitError) {
      setStatus("failed");
      setError(submitError.message || "Network error while creating job.");
    } finally {
      setIsSubmitting(false);
    }
  }

  async function playSegment(segment) {
    const start = parseTime(segment.start_time);
    const now = lastVideoTimeRef.current;
    const offset = Math.max(0, now - start);

    const key = segment.audio_file_name || expectedAudioFilename(segment.start_time);
    if (activeAudioRef.current.has(key)) {
      return;
    }

    const audio = new Audio(segment.audio_file_url);
    audio.currentTime = offset;
    activeAudioRef.current.set(key, audio);

    audio.onended = () => {
      activeAudioRef.current.delete(key);
    };

    try {
      await audio.play();
    } catch (playError) {
      console.warn("Unable to play segment audio", playError);
      activeAudioRef.current.delete(key);
    }
  }

  function syncAudioWithVideo(currentTime) {
    const seekingBackward = currentTime < lastVideoTimeRef.current;

    if (seekingBackward) {
      const resetKeys = new Set();
      segments.forEach((segment) => {
        const key = segment.audio_file_name || expectedAudioFilename(segment.start_time);
        if (parseTime(segment.start_time) > currentTime) {
          playedSegmentsRef.current.delete(key);
          resetKeys.add(key);
        }
      });

      resetKeys.forEach((key) => {
        const active = activeAudioRef.current.get(key);
        if (active) {
          active.pause();
          active.currentTime = 0;
          activeAudioRef.current.delete(key);
        }
      });
    }

    segments.forEach((segment) => {
      const start = parseTime(segment.start_time);
      const key = segment.audio_file_name || expectedAudioFilename(segment.start_time);

      if (currentTime >= start && !playedSegmentsRef.current.has(key)) {
        playedSegmentsRef.current.add(key);
        playSegment(segment);
      }
    });

    lastVideoTimeRef.current = currentTime;
    setCurrentVideoTime(currentTime);
  }

  function handleVideoPlay() {
    const video = videoRef.current;
    if (!video) {
      return;
    }
    video.muted = true;
    syncAudioWithVideo(video.currentTime || 0);
  }

  function handleVideoPause() {
    stopAllAudio();
  }

  function handleVideoSeek() {
    const video = videoRef.current;
    if (!video) {
      return;
    }

    stopAllAudio();
    syncAudioWithVideo(video.currentTime || 0);
  }

  function activeSegmentId() {
    return segments.findIndex((segment) => {
      const start = parseTime(segment.start_time);
      const end = parseTime(segment.end_time || start + 2);
      return currentVideoTime >= start && currentVideoTime <= end;
    });
  }

  const activeIndex = activeSegmentId();

  return (
    <main className="page">
      <header className="hero">
        <h1>DeepKIN Video Dubbing</h1>
        <p>
          Upload a video under 3 minutes and 10 MB, then stream translated
          Kinyarwanda audio by transcript timestamp.
        </p>
      </header>

      <section className="card">
        <form onSubmit={handleSubmit} className="form-grid">
          <label>
            <span>Video file</span>
            <input
              type="file"
              accept="video/mp4,video/quicktime,video/webm,video/x-m4v"
              onChange={(event) => setSelectedFile(event.target.files?.[0] || null)}
            />
          </label>

          <label>
            <span>Transcript key (optional)</span>
            <input
              type="text"
              placeholder="transcripts/your-file.json"
              value={transcriptKey}
              onChange={(event) => setTranscriptKey(event.target.value)}
            />
          </label>

          <button type="submit" disabled={isSubmitting}>
            {isSubmitting ? "Submitting..." : "Create Processing Job"}
          </button>
        </form>

        <div className="status-line">
          <strong>Status:</strong> {status}
          {jobId ? <span> | Job ID: {jobId}</span> : null}
        </div>
        {error ? <p className="error">{error}</p> : null}
      </section>

      <section className="layout">
        <article className="card">
          <h2>Video Preview</h2>
          {videoUrl ? (
            <video
              ref={videoRef}
              src={videoUrl}
              controls
              playsInline
              onPlay={handleVideoPlay}
              onPause={handleVideoPause}
              onEnded={resetPlaybackState}
              onSeeked={handleVideoSeek}
              onTimeUpdate={(event) => syncAudioWithVideo(event.currentTarget.currentTime || 0)}
            >
              <track kind="captions" label="No captions" />
            </video>
          ) : (
            <p>Select a video to preview and process.</p>
          )}
        </article>

        <article className="card">
          <h2>Translated Segments</h2>
          {segments.length === 0 ? (
            <p>No translated segments yet. Submit a job and wait for completion.</p>
          ) : (
            <ol className="segment-list">
              {segments.map((segment, index) => {
                const expected = expectedAudioFilename(segment.start_time);
                return (
                  <li
                    key={`${segment.start_time}-${index}`}
                    className={index === activeIndex ? "segment active" : "segment"}
                  >
                    <div className="segment-meta">
                      <span>{segment.start_time}s</span>
                      <span>{segment.audio_file_name || expected}</span>
                    </div>
                    <p>{segment.transcript_rw}</p>
                  </li>
                );
              })}
            </ol>
          )}
        </article>
      </section>
    </main>
  );
}

export default App;
