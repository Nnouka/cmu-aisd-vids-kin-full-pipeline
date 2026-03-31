import { useEffect, useMemo, useRef, useState } from "react";
import { CheckCircleTwoTone } from "@ant-design/icons";
import { Col, ConfigProvider, Layout, Row, theme } from "antd";

import { AppHeader } from "./components/AppHeader";
import { HeroCard } from "./components/HeroCard";
import { JobCreateCard } from "./components/JobCreateCard";
import { SegmentsCard } from "./components/SegmentsCard";
import { SidebarNav } from "./components/SidebarNav";
import { StageProgressCard } from "./components/StageProgressCard";
import { VideoPreviewCard } from "./components/VideoPreviewCard";
import {
  JOB_STAGES,
  MAX_DURATION_SECONDS,
  MAX_FILE_SIZE,
  POLL_INTERVAL_MS,
  STATUS_COLOR,
} from "./constants";
import { useSegmentPlayback } from "./hooks/useSegmentPlayback";
import {
  computeDurationFromTimestamps,
  deriveJobIdFromFilename,
  getVideoDuration,
  parseTime,
} from "./lib/formatters";
import { continueJob, createJob, fetchJob, restartJob } from "./lib/jobsApi";

const { Content } = Layout;

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [jobId, setJobId] = useState("");
  const [status, setStatus] = useState("idle");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isContinuing, setIsContinuing] = useState(false);
  const [isRestarting, setIsRestarting] = useState(false);
  const [isCheckingExistingJob, setIsCheckingExistingJob] = useState(false);
  const [videoUrl, setVideoUrl] = useState("");
  const [audioVolume, setAudioVolume] = useState(0.9);
  const [ccEnabled, setCcEnabled] = useState(false);
  const [currentStage, setCurrentStage] = useState("idle");
  const [lastFailedStage, setLastFailedStage] = useState("");
  const [jobStartedAt, setJobStartedAt] = useState("");
  const [jobCompletedAt, setJobCompletedAt] = useState("");
  const [durationNowMs, setDurationNowMs] = useState(() => Date.now());

  const pollHandleRef = useRef(null);

  const segments = useMemo(() => {
    const raw = result?.results?.translated_segments || [];
    return [...raw].sort((a, b) => parseTime(a.start_time) - parseTime(b.start_time));
  }, [result]);

  const {
    videoRef,
    currentVideoTime,
    videoDuration,
    isVideoPlaying,
    playbackDebug,
    activeIndex,
    activeCaptionText,
    resetPlaybackState,
    handleSegmentClick,
    handleVideoPlay,
    handleVideoPause,
    handleVideoSeek,
    handleVideoLoadedMetadata,
    handleVideoTimeUpdate,
    handleSeekSliderChange,
    handlePlayPauseClick,
  } = useSegmentPlayback({
    segments,
    audioVolume,
    ccEnabled,
    setError,
  });

  async function fetchJobStatus(nextJobId) {
    const { response, payload } = await fetchJob(nextJobId);
    if (!response.ok) {
      setStatus("failed");
      setError("Failed to poll job status.");
      return;
    }

    setStatus(payload.status);
    setCurrentStage(payload.current_stage || payload.status || "unknown");
    setLastFailedStage(payload.last_failed_stage || "");
    setJobStartedAt(payload.started_at || "");
    setJobCompletedAt(payload.completed_at || "");
    if (payload.status === "completed") {
      setResult(payload.result);
    }
    if (payload.status === "failed") {
      setError(payload.error || "Job failed.");
    }
  }

  useEffect(() => {
    if (!selectedFile) {
      setJobId("");
      setStatus("idle");
      setCurrentStage("idle");
      setLastFailedStage("");
      setJobStartedAt("");
      setJobCompletedAt("");
      setResult(null);
      return undefined;
    }

    const objectUrl = URL.createObjectURL(selectedFile);
    setVideoUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [selectedFile]);

  useEffect(() => {
    if (!selectedFile) {
      return undefined;
    }

    const inferredJobId = deriveJobIdFromFilename(selectedFile.name);
    if (!inferredJobId) {
      return undefined;
    }

    let cancelled = false;

    async function loadExistingJob() {
      setIsCheckingExistingJob(true);
      setError("");
      setJobId(inferredJobId);

      try {
        const { response, payload } = await fetchJob(inferredJobId);
        if (cancelled) {
          return;
        }

        if (response.status === 404) {
          setStatus("idle");
          setCurrentStage("idle");
          setLastFailedStage("");
          setJobStartedAt("");
          setJobCompletedAt("");
          setResult(null);
          return;
        }

        if (!response.ok) {
          setStatus("failed");
          setError("Failed to load existing job status.");
          return;
        }

        if (cancelled) {
          return;
        }

        setStatus(payload.status || "idle");
        setCurrentStage(payload.current_stage || payload.status || "unknown");
        setLastFailedStage(payload.last_failed_stage || "");
        setJobStartedAt(payload.started_at || "");
        setJobCompletedAt(payload.completed_at || "");
        setResult(payload.status === "completed" ? payload.result : null);
        if (payload.status === "failed") {
          setError(payload.error || "Job failed.");
        }
      } catch {
        if (!cancelled) {
          setStatus("failed");
          setError("Network error while checking existing job.");
        }
      } finally {
        if (!cancelled) {
          setIsCheckingExistingJob(false);
        }
      }
    }

    loadExistingJob();

    return () => {
      cancelled = true;
    };
  }, [selectedFile]);

  useEffect(() => {
    if (!jobId || !(status === "queued" || status === "processing")) {
      return undefined;
    }

    pollHandleRef.current = globalThis.setInterval(async () => {
      await fetchJobStatus(jobId);
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

  useEffect(() => {
    if (!jobStartedAt || jobCompletedAt) {
      return undefined;
    }

    setDurationNowMs(Date.now());
    const handle = globalThis.setInterval(() => {
      setDurationNowMs(Date.now());
    }, 1000);

    return () => {
      globalThis.clearInterval(handle);
    };
  }, [jobStartedAt, jobCompletedAt]);

  async function handleSubmit(event) {
    event.preventDefault();

    if (!selectedFile) {
      setError("Please choose a video file.");
      return;
    }

    setError("");
    setResult(null);
    setStatus("validating");
    setCurrentStage("validating");
    setLastFailedStage("");
    setJobStartedAt("");
    setJobCompletedAt("");

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

    setIsSubmitting(true);
    try {
      const { response, payload } = await createJob(formData);

      if (!response.ok) {
        setStatus("failed");
        setError(payload.detail || "Failed to create processing job.");
        return;
      }

      setJobId(payload.job_id);
      setStatus(payload.status || "queued");
      setCurrentStage(payload.status || "queued");
      resetPlaybackState();
      await fetchJobStatus(payload.job_id);
    } catch (submitError) {
      setStatus("failed");
      setError(submitError.message || "Network error while creating job.");
    } finally {
      setIsSubmitting(false);
    }
  }

  async function handleContinueJob() {
    if (!jobId) {
      setError("No job to continue yet.");
      return;
    }

    setIsContinuing(true);
    setError("");
    try {
      const { response, payload } = await continueJob(jobId);

      if (!response.ok) {
        setError(payload.detail || "Failed to continue job.");
        return;
      }

      setStatus(payload.status || "queued");
      setCurrentStage(payload.status || "queued");
      await fetchJobStatus(jobId);
    } catch (continueError) {
      setError(continueError.message || "Network error while continuing job.");
    } finally {
      setIsContinuing(false);
    }
  }

  async function handleRestartJob() {
    if (!jobId) {
      setError("No job to restart yet.");
      return;
    }

    setIsRestarting(true);
    setError("");
    try {
      const { response, payload } = await restartJob(jobId);

      if (!response.ok) {
        setError(payload.detail || "Failed to restart job.");
        return;
      }

      setStatus(payload.status || "queued");
      setCurrentStage(payload.status || "queued");
      await fetchJobStatus(jobId);
    } catch (restartError) {
      setError(restartError.message || "Network error while restarting job.");
    } finally {
      setIsRestarting(false);
    }
  }

  function handleVolumeSliderChange(event) {
    const next = Number.parseFloat(event.target.value || "0");
    const clamped = Math.max(0, Math.min(next, 1));
    setAudioVolume(clamped);
  }

  const canContinueJob =
    Boolean(jobId) &&
    ["queued", "processing", "failed"].includes(status) &&
    currentStage !== "upload_video";
  const shouldDisableStartProcessing =
    isSubmitting ||
    isCheckingExistingJob ||
    !selectedFile ||
    status === "queued" ||
    status === "processing" ||
    status === "completed";
  const currentStageIndex = JOB_STAGES.indexOf(currentStage);
  const normalizedStageIndex = Math.max(currentStageIndex, 0);
  const stageProgressPercent =
    status === "completed"
      ? 100
      : Math.round((normalizedStageIndex / (JOB_STAGES.length - 1)) * 100);
  const displayedDurationSeconds = useMemo(
    () => computeDurationFromTimestamps(jobStartedAt, jobCompletedAt, durationNowMs),
    [jobStartedAt, jobCompletedAt, durationNowMs]
  );

  const uploadProps = {
    accept: "video/mp4,video/quicktime,video/webm,video/x-m4v",
    maxCount: 1,
    beforeUpload: (file) => {
      setSelectedFile(file);
      return false;
    },
    onRemove: () => {
      setSelectedFile(null);
      setVideoUrl("");
      return true;
    },
    fileList: selectedFile ? [selectedFile] : [],
  };

  return (
    <ConfigProvider
      theme={{
        algorithm: theme.darkAlgorithm,
        token: {
          colorPrimary: "#4ea1ff",
          colorBgBase: "#0f141c",
          colorTextBase: "#e6edf7",
          borderRadius: 12,
        },
      }}
    >
      <Layout className="app-shell">
      <AppHeader
        status={status}
        pollIntervalSeconds={Math.round(POLL_INTERVAL_MS / 1000)}
        statusColor={STATUS_COLOR[status] || "default"}
      />
      <Layout>
        <SidebarNav />
        <Content className="app-content">
          <HeroCard jobId={jobId} currentStage={currentStage} segmentCount={segments.length} />

          <Row gutter={[16, 16]}>
            <Col xs={24} lg={10}>
              <JobCreateCard
                uploadProps={uploadProps}
                onSubmit={handleSubmit}
                isSubmitting={isSubmitting}
                isCheckingExistingJob={isCheckingExistingJob}
                shouldDisableStartProcessing={shouldDisableStartProcessing}
                error={error}
              />
            </Col>
            <Col xs={24} lg={14}>
              <StageProgressCard
                status={status}
                currentStage={currentStage}
                lastFailedStage={lastFailedStage}
                stageProgressPercent={stageProgressPercent}
                jobStartedAt={jobStartedAt}
                jobCompletedAt={jobCompletedAt}
                displayedDurationSeconds={displayedDurationSeconds}
                isContinuing={isContinuing}
                isRestarting={isRestarting}
                canContinueJob={canContinueJob}
                jobId={jobId}
                onContinueJob={handleContinueJob}
                onRestartJob={handleRestartJob}
              />
            </Col>
          </Row>

          <Row gutter={[16, 16]}>
            <Col xs={24} xl={12}>
              <VideoPreviewCard
                videoUrl={videoUrl}
                videoRef={videoRef}
                activeCaptionText={activeCaptionText}
                isVideoPlaying={isVideoPlaying}
                currentVideoTime={currentVideoTime}
                videoDuration={videoDuration}
                audioVolume={audioVolume}
                segmentsCount={segments.length}
                ccEnabled={ccEnabled}
                  playbackDebug={playbackDebug}
                onToggleCc={() => setCcEnabled((prev) => !prev)}
                onVideoPlay={handleVideoPlay}
                onVideoPause={handleVideoPause}
                onVideoEnded={resetPlaybackState}
                onVideoSeek={handleVideoSeek}
                onVideoLoadedMetadata={handleVideoLoadedMetadata}
                onVideoTimeUpdate={handleVideoTimeUpdate}
                onPlayPauseClick={handlePlayPauseClick}
                onSeekSliderChange={handleSeekSliderChange}
                onVolumeSliderChange={handleVolumeSliderChange}
              />
            </Col>
            <Col xs={24} xl={12}>
              <SegmentsCard segments={segments} activeIndex={activeIndex} onSegmentClick={handleSegmentClick} />
            </Col>
          </Row>
        </Content>
      </Layout>
      {status === "completed" ? (
        <div className="floating-complete">
          <CheckCircleTwoTone twoToneColor="#52c41a" />
          <span>Job completed</span>
        </div>
      ) : null}
      </Layout>
    </ConfigProvider>
  );
}

export default App;
