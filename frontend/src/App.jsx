import { useEffect, useMemo, useRef, useState } from "react";
import {
  Alert,
  Badge,
  Button,
  Card,
  Col,
  ConfigProvider,
  Layout,
  List,
  Progress,
  Row,
  Statistic,
  Tag,
  Timeline,
  Typography,
  Upload,
} from "antd";
import {
  CheckCircleTwoTone,
  ClockCircleOutlined,
  DashboardOutlined,
  FileTextOutlined,
  PauseCircleOutlined,
  PlayCircleOutlined,
  ReloadOutlined,
  SoundOutlined,
  TranslationOutlined,
  UploadOutlined,
  VideoCameraOutlined,
} from "@ant-design/icons";
import { theme } from "antd";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api";

const MAX_FILE_SIZE = 10 * 1024 * 1024;
const MAX_DURATION_SECONDS = 180;
const POLL_INTERVAL_MS = Number(import.meta.env.VITE_POLL_INTERVAL_MS || 60000);
const JOB_STAGES = [
  "queued",
  "upload_video",
  "wait_transcript",
  "translate_and_tts",
  "upload_manifest",
  "completed",
];

const { Header, Sider, Content } = Layout;
const { Title, Paragraph, Text } = Typography;

const STATUS_COLOR = {
  idle: "default",
  validating: "processing",
  queued: "processing",
  processing: "processing",
  completed: "success",
  failed: "error",
};

function formatStageLabel(stage) {
  return stage.replaceAll("_", " ");
}

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

function deriveJobIdFromFilename(filename) {
  const cleaned = (filename || "").trim();
  if (!cleaned) {
    return "";
  }

  const dot = cleaned.lastIndexOf(".");
  const stem = dot > 0 ? cleaned.slice(0, dot) : cleaned;
  return stem.trim();
}

function formatClock(totalSeconds) {
  const seconds = Math.max(0, Math.floor(totalSeconds || 0));
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
}

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
  const [currentVideoTime, setCurrentVideoTime] = useState(0);
  const [videoDuration, setVideoDuration] = useState(0);
  const [isVideoPlaying, setIsVideoPlaying] = useState(false);
  const [audioVolume, setAudioVolume] = useState(0.9);
  const [ccEnabled, setCcEnabled] = useState(false);
  const [currentStage, setCurrentStage] = useState("idle");
  const [lastFailedStage, setLastFailedStage] = useState("");

  const videoRef = useRef(null);
  const pollHandleRef = useRef(null);
  const activeAudioRef = useRef(new Map());
  const playedSegmentsRef = useRef(new Set());
  const lastVideoTimeRef = useRef(0);

  const segments = useMemo(() => {
    const raw = result?.results?.translated_segments || [];
    return [...raw].sort((a, b) => parseTime(a.start_time) - parseTime(b.start_time));
  }, [result]);

  async function fetchJobStatus(nextJobId) {
    const response = await fetch(`${API_BASE_URL}/jobs/${nextJobId}`);
    if (!response.ok) {
      setStatus("failed");
      setError("Failed to poll job status.");
      return;
    }

    const payload = await response.json();
    setStatus(payload.status);
    setCurrentStage(payload.current_stage || payload.status || "unknown");
    setLastFailedStage(payload.last_failed_stage || "");
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
        const response = await fetch(`${API_BASE_URL}/jobs/${inferredJobId}`);
        if (cancelled) {
          return;
        }

        if (response.status === 404) {
          setStatus("idle");
          setCurrentStage("idle");
          setLastFailedStage("");
          setResult(null);
          return;
        }

        if (!response.ok) {
          setStatus("failed");
          setError("Failed to load existing job status.");
          return;
        }

        const payload = await response.json();
        if (cancelled) {
          return;
        }

        setStatus(payload.status || "idle");
        setCurrentStage(payload.current_stage || payload.status || "unknown");
        setLastFailedStage(payload.last_failed_stage || "");
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
    activeAudioRef.current.forEach((audio) => {
      audio.volume = audioVolume;
    });
  }, [audioVolume]);

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
    setIsVideoPlaying(false);
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
    setCurrentStage("validating");
    setLastFailedStage("");

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

  async function playSegment(segment) {
    const start = parseTime(segment.start_time);
    const now = lastVideoTimeRef.current;
    const offset = Math.max(0, now - start);

    const key = segment.audio_file_name || expectedAudioFilename(segment.start_time);
    if (activeAudioRef.current.has(key)) {
      return;
    }

    const audio = new Audio(segment.audio_file_url);
    audio.volume = audioVolume;
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

  function reconcilePlaybackForSeek(nextTime) {
    stopAllAudio();

    const updatedPlayed = new Set();
    segments.forEach((segment) => {
      const start = parseTime(segment.start_time);
      const end = parseTime(segment.end_time || start + 2);
      const key = segment.audio_file_name || expectedAudioFilename(segment.start_time);
      if (end < nextTime) {
        updatedPlayed.add(key);
      }
    });

    playedSegmentsRef.current = updatedPlayed;
    lastVideoTimeRef.current = nextTime;
    setCurrentVideoTime(nextTime);

    const video = videoRef.current;
    if (!video || video.paused) {
      return;
    }

    const active = segments.find((segment) => {
      const start = parseTime(segment.start_time);
      const end = parseTime(segment.end_time || start + 2);
      return nextTime >= start && nextTime <= end;
    });

    if (active) {
      const key = active.audio_file_name || expectedAudioFilename(active.start_time);
      playedSegmentsRef.current.add(key);
      playSegment(active);
    }
  }

  function seekVideoToTime(nextTime) {
    const video = videoRef.current;
    if (!video) {
      return;
    }

    const clamped = Math.max(0, Math.min(nextTime, videoDuration || nextTime));
    video.currentTime = clamped;
    reconcilePlaybackForSeek(clamped);
  }

  async function handleSegmentClick(startTime) {
    const video = videoRef.current;
    if (!video) {
      return;
    }

    stopAllAudio();
    seekVideoToTime(startTime);

    try {
      await video.play();
    } catch (playError) {
      setError(playError.message || "Unable to continue playback from selected segment.");
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
      const response = await fetch(`${API_BASE_URL}/jobs/${jobId}/continue`, {
        method: "POST",
      });
      const payload = await response.json();

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
      const response = await fetch(`${API_BASE_URL}/jobs/${jobId}/restart`, {
        method: "POST",
      });
      const payload = await response.json();

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
    setIsVideoPlaying(true);
    syncAudioWithVideo(video.currentTime || 0);
  }

  function handleVideoPause() {
    setIsVideoPlaying(false);
    stopAllAudio();
  }

  function handleVideoSeek() {
    const video = videoRef.current;
    if (!video) {
      return;
    }

    reconcilePlaybackForSeek(video.currentTime || 0);
  }

  function handleVideoLoadedMetadata(event) {
    setVideoDuration(event.currentTarget.duration || 0);
  }

  function handleSeekSliderChange(event) {
    const next = Number.parseFloat(event.target.value || "0");
    seekVideoToTime(next);
  }

  function handleVolumeSliderChange(event) {
    const next = Number.parseFloat(event.target.value || "0");
    const clamped = Math.max(0, Math.min(next, 1));
    setAudioVolume(clamped);
  }

  async function handlePlayPauseClick() {
    const video = videoRef.current;
    if (!video) {
      return;
    }

    if (video.paused) {
      try {
        await video.play();
      } catch (playError) {
        setError(playError.message || "Unable to start playback.");
      }
      return;
    }

    video.pause();
  }

  function activeSegmentId() {
    return segments.findIndex((segment) => {
      const start = parseTime(segment.start_time);
      const end = parseTime(segment.end_time || start + 2);
      return currentVideoTime >= start && currentVideoTime <= end;
    });
  }

  const activeIndex = activeSegmentId();
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
  const activeCaptionText =
    ccEnabled && activeIndex >= 0 && segments[activeIndex]?.transcript_rw
      ? segments[activeIndex].transcript_rw
      : "";
  const currentStageIndex = JOB_STAGES.indexOf(currentStage);
  const normalizedStageIndex = Math.max(currentStageIndex, 0);
  const stageProgressPercent =
    status === "completed"
      ? 100
      : Math.round((normalizedStageIndex / (JOB_STAGES.length - 1)) * 100);

  function stageState(stage, index) {
    if (status === "failed" && stage === lastFailedStage) {
      return "failed";
    }
    if (status === "completed") {
      return "done";
    }
    if (index < currentStageIndex) {
      return "done";
    }
    if (index === currentStageIndex) {
      return "active";
    }
    return "pending";
  }

  const timelineItems = JOB_STAGES.map((stage, index) => {
    const state = stageState(stage, index);
    let color = "#d9d9d9";
    let dot = <ClockCircleOutlined />;

    if (state === "done") {
      color = "#52c41a";
      dot = <CheckCircleTwoTone twoToneColor="#52c41a" />;
    } else if (state === "active") {
      color = "#1677ff";
      dot = <ClockCircleOutlined style={{ color: "#1677ff" }} />;
    } else if (state === "failed") {
      color = "#ff4d4f";
      dot = <ClockCircleOutlined style={{ color: "#ff4d4f" }} />;
    }

    return {
      color,
      dot,
      children: <span className="timeline-label">{formatStageLabel(stage)}</span>,
    };
  });

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
      <Header className="app-header">
        <div>
          <Text type="secondary">Creator Console</Text>
          <Title level={3} className="app-title">DeepKIN Studio</Title>
        </div>
        <div className="header-meta">
          <Tag color={STATUS_COLOR[status] || "default"}>{status}</Tag>
          <Tag icon={<ClockCircleOutlined />}>Poll {Math.round(POLL_INTERVAL_MS / 1000)}s</Tag>
        </div>
      </Header>
      <Layout>
        <Sider width={220} className="app-sider" breakpoint="lg" collapsedWidth="0">
          <List
            size="small"
            dataSource={[
              { icon: <VideoCameraOutlined />, label: "Video Dubbing" },
              { icon: <DashboardOutlined />, label: "Job Queue" },
              { icon: <FileTextOutlined />, label: "Transcripts" },
              { icon: <TranslationOutlined />, label: "Audio Segments" },
            ]}
            renderItem={(item, idx) => (
              <List.Item className={idx === 0 ? "sider-item active" : "sider-item"}>
                <span className="sider-icon">{item.icon}</span>
                <span>{item.label}</span>
              </List.Item>
            )}
          />
        </Sider>
        <Content className="app-content">
          <Card className="hero-card" variant="borderless">
            <Row gutter={[16, 16]} align="middle">
              <Col xs={24} md={10}>
                <Title level={4} style={{ marginTop: 0 }}>Production Dashboard</Title>
                <Paragraph type="secondary" style={{ marginBottom: 0 }}>
                  Upload, monitor stage execution, and review generated translated audio in one place.
                </Paragraph>
              </Col>
              <Col xs={24} md={14}>
                <Row gutter={[12, 12]}>
                  <Col xs={24} sm={8}>
                    <Statistic title="Job Key" value={jobId || "none"} />
                  </Col>
                  <Col xs={24} sm={8}>
                    <Statistic title="Active Stage" value={formatStageLabel(currentStage || "idle")} />
                  </Col>
                  <Col xs={24} sm={8}>
                    <Statistic title="Segments" value={segments.length} />
                  </Col>
                </Row>
              </Col>
            </Row>
          </Card>

          <Row gutter={[16, 16]}>
            <Col xs={24} lg={10}>
              <Card title="Create Job" className="panel-card" extra={<UploadOutlined />}>
                <form onSubmit={handleSubmit}>
                  <Upload {...uploadProps}>
                    <Button icon={<UploadOutlined />}>Select Video</Button>
                  </Upload>
                  <div className="submit-row">
                    <Button
                      type="primary"
                      htmlType="submit"
                      loading={isSubmitting || isCheckingExistingJob}
                      disabled={shouldDisableStartProcessing}
                      icon={<PlayCircleOutlined />}
                    >
                      Start Processing
                    </Button>
                  </div>
                </form>
                {error ? <Alert type="error" showIcon message={error} style={{ marginTop: 12 }} /> : null}
              </Card>
            </Col>
            <Col xs={24} lg={14}>
              <Card title="Stage Progress" className="panel-card">
                <Progress percent={stageProgressPercent} status={status === "failed" ? "exception" : undefined} />
                <Timeline orientation="horizontal" items={timelineItems} className="timeline-tight" />
                {lastFailedStage ? (
                  <Alert
                    type="warning"
                    showIcon
                    message={`Last failed stage: ${formatStageLabel(lastFailedStage)}`}
                    style={{ marginTop: 8 }}
                  />
                ) : null}
                <div className="actions-row">
                  <Button
                    type="primary"
                    icon={<ReloadOutlined />}
                    loading={isContinuing}
                    onClick={handleContinueJob}
                    disabled={!canContinueJob || isRestarting}
                  >
                    Continue Job
                  </Button>
                  <Button
                    icon={<ReloadOutlined />}
                    loading={isRestarting}
                    onClick={handleRestartJob}
                    disabled={!jobId || status !== "failed" || isContinuing}
                  >
                    Restart Failed Job
                  </Button>
                </div>
              </Card>
            </Col>
          </Row>

          <Row gutter={[16, 16]}>
            <Col xs={24} xl={12}>
              <Card title="Video Preview" className="panel-card" extra={<VideoCameraOutlined />}>
                {videoUrl ? (
                  <div className="video-stage">
                    <video
                      ref={videoRef}
                      src={videoUrl}
                      playsInline
                      onPlay={handleVideoPlay}
                      onPause={handleVideoPause}
                      onEnded={resetPlaybackState}
                      onSeeked={handleVideoSeek}
                      onLoadedMetadata={handleVideoLoadedMetadata}
                      onTimeUpdate={(event) => syncAudioWithVideo(event.currentTarget.currentTime || 0)}
                      className="preview-video"
                    >
                      <track kind="captions" label="No captions" />
                    </video>
                    {activeCaptionText ? <div className="caption-overlay">{activeCaptionText}</div> : null}
                    <div className="custom-controls">
                      <Button
                        icon={isVideoPlaying ? <PauseCircleOutlined /> : <PlayCircleOutlined />}
                        onClick={handlePlayPauseClick}
                      >
                        {isVideoPlaying ? "Pause" : "Play"}
                      </Button>
                      <div className="seek-control">
                        <input
                          type="range"
                          min={0}
                          max={Math.max(videoDuration, 0)}
                          step={0.01}
                          value={Math.min(currentVideoTime, videoDuration || currentVideoTime)}
                          onChange={handleSeekSliderChange}
                        />
                        <span className="time-readout">
                          {formatClock(currentVideoTime)} / {formatClock(videoDuration)}
                        </span>
                      </div>
                      <div className="volume-control">
                        <SoundOutlined />
                        <input
                          type="range"
                          min={0}
                          max={1}
                          step={0.01}
                          value={audioVolume}
                          onChange={handleVolumeSliderChange}
                          disabled={segments.length === 0}
                        />
                      </div>
                      <Button onClick={() => setCcEnabled((prev) => !prev)} type={ccEnabled ? "primary" : "default"}>
                        CC
                      </Button>
                    </div>
                  </div>
                ) : (
                  <Alert type="info" showIcon message="Select a video to preview and process." />
                )}
              </Card>
            </Col>
            <Col xs={24} xl={12}>
              <Card title="Translated Segments" className="panel-card" extra={<FileTextOutlined />}>
                {segments.length === 0 ? (
                  <Alert
                    type="info"
                    showIcon
                    message="No translated segments yet. Submit a job and wait for completion."
                  />
                ) : (
                  <List
                    className="segments-list"
                    dataSource={segments}
                    renderItem={(segment, index) => {
                      const isActive = index === activeIndex;
                      return (
                        <List.Item
                          className={isActive ? "segment-item active clickable" : "segment-item clickable"}
                          onClick={() => handleSegmentClick(parseTime(segment.start_time))}
                        >
                          <div className="segment-head">
                            <Text strong className="segment-time-chip">{segment.start_time}s</Text>
                            <Badge
                              status={isActive ? "processing" : "default"}
                              text={segment.audio_file_url ? "Audio ready" : "Pending audio"}
                            />
                          </div>
                          <Paragraph style={{ marginBottom: 0 }}>{segment.transcript_rw}</Paragraph>
                        </List.Item>
                      );
                    }}
                  />
                )}
              </Card>
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
