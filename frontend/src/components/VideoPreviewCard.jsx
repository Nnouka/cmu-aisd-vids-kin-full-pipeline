import {
  PauseCircleOutlined,
  PlayCircleOutlined,
  SoundOutlined,
  VideoCameraOutlined,
} from "@ant-design/icons";
import { Alert, Button, Card } from "antd";
import PropTypes from "prop-types";

import { formatClock } from "../lib/formatters";

export function VideoPreviewCard({
  videoUrl,
  videoRef,
  activeCaptionText,
  isVideoPlaying,
  currentVideoTime,
  videoDuration,
  audioVolume,
  segmentsCount,
  ccEnabled,
  playbackDebug,
  onToggleCc,
  onVideoPlay,
  onVideoPause,
  onVideoEnded,
  onVideoSeek,
  onVideoLoadedMetadata,
  onVideoTimeUpdate,
  onPlayPauseClick,
  onSeekSliderChange,
  onVolumeSliderChange,
}) {
  return (
    <Card title="Video Preview" className="panel-card" extra={<VideoCameraOutlined />}>
      {videoUrl ? (
        <div className="video-stage">
          <div className="playback-debug-badge" aria-live="polite">
            <span>A:{playbackDebug?.activeSegment || "-"}</span>
            <span>P:{playbackDebug?.pendingSegment || "-"}</span>
            <span>H:{playbackDebug?.hold || "off"}</span>
          </div>
          <video
            ref={videoRef}
            src={videoUrl}
            playsInline
            onPlay={onVideoPlay}
            onPause={onVideoPause}
            onEnded={onVideoEnded}
            onSeeked={onVideoSeek}
            onLoadedMetadata={onVideoLoadedMetadata}
            onTimeUpdate={onVideoTimeUpdate}
            className="preview-video"
          >
            <track kind="captions" label="No captions" />
          </video>
          {activeCaptionText ? <div className="caption-overlay">{activeCaptionText}</div> : null}
          <div className="custom-controls">
            <Button
              className="control-btn"
              icon={isVideoPlaying ? <PauseCircleOutlined /> : <PlayCircleOutlined />}
              onClick={onPlayPauseClick}
            >
              {isVideoPlaying ? "Pause" : "Play"}
            </Button>
            <div className="seek-control compact">
              <div className="seek-bar-wrap">
                <input
                  type="range"
                  min={0}
                  max={Math.max(videoDuration, 0)}
                  step={0.01}
                  value={Math.min(currentVideoTime, videoDuration || currentVideoTime)}
                  onChange={onSeekSliderChange}
                />
              </div>
              <span className="time-readout mono">
                {formatClock(currentVideoTime)} / {formatClock(videoDuration)}
              </span>
            </div>
            <div className="volume-control compact">
              <SoundOutlined />
              <div className="volume-bar-wrap">
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.01}
                  value={audioVolume}
                  onChange={onVolumeSliderChange}
                  disabled={segmentsCount === 0}
                />
              </div>
            </div>
            <Button className="control-btn" onClick={onToggleCc} type={ccEnabled ? "primary" : "default"}>
              CC
            </Button>
          </div>
        </div>
      ) : (
        <Alert type="info" showIcon message="Select a video to preview and process." />
      )}
    </Card>
  );
}

VideoPreviewCard.propTypes = {
  videoUrl: PropTypes.string,
  videoRef: PropTypes.shape({ current: PropTypes.any }),
  activeCaptionText: PropTypes.string,
  isVideoPlaying: PropTypes.bool.isRequired,
  currentVideoTime: PropTypes.number.isRequired,
  videoDuration: PropTypes.number.isRequired,
  audioVolume: PropTypes.number.isRequired,
  segmentsCount: PropTypes.number.isRequired,
  ccEnabled: PropTypes.bool.isRequired,
  playbackDebug: PropTypes.shape({
    activeSegment: PropTypes.string,
    pendingSegment: PropTypes.string,
    hold: PropTypes.string,
  }),
  onToggleCc: PropTypes.func.isRequired,
  onVideoPlay: PropTypes.func.isRequired,
  onVideoPause: PropTypes.func.isRequired,
  onVideoEnded: PropTypes.func.isRequired,
  onVideoSeek: PropTypes.func.isRequired,
  onVideoLoadedMetadata: PropTypes.func.isRequired,
  onVideoTimeUpdate: PropTypes.func.isRequired,
  onPlayPauseClick: PropTypes.func.isRequired,
  onSeekSliderChange: PropTypes.func.isRequired,
  onVolumeSliderChange: PropTypes.func.isRequired,
};

VideoPreviewCard.defaultProps = {
  videoUrl: "",
  videoRef: null,
  activeCaptionText: "",
  playbackDebug: {
    activeSegment: "-",
    pendingSegment: "-",
    hold: "off",
  },
};