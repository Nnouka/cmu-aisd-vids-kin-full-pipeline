import { useEffect, useMemo, useRef, useState } from "react";

import { parseTime } from "../lib/formatters";
import { findSegmentIndexAtTime, getSegmentBounds, loadAudioMetadata, segmentKeyFor } from "../lib/playback";

export function useSegmentPlayback({ segments, audioVolume, ccEnabled, setError }) {
  const [currentVideoTime, setCurrentVideoTime] = useState(0);
  const [videoDuration, setVideoDuration] = useState(0);
  const [isVideoPlaying, setIsVideoPlaying] = useState(false);
  const [playbackDebug, setPlaybackDebug] = useState({
    activeSegment: "-",
    pendingSegment: "-",
    hold: "off",
  });

  const videoRef = useRef(null);
  const audioPoolRef = useRef(new Map());
  const playedSegmentsRef = useRef(new Set());
  const activeSegmentIndexRef = useRef(-1);
  const activeSegmentKeyRef = useRef("");
  const pendingSegmentIndexRef = useRef(null);
  const pendingSegmentKeyRef = useRef("");
  const pausedForAudioHoldRef = useRef(false);
  const lastVideoTimeRef = useRef(0);
  const playbackAttemptRef = useRef(0);

  function refreshPlaybackDebug() {
    const activeSegment = activeSegmentIndexRef.current >= 0 ? String(activeSegmentIndexRef.current + 1) : "-";
    let pendingSegment;
    if (pendingSegmentIndexRef.current === null) {
      pendingSegment = "-";
    } else {
      pendingSegment = String(pendingSegmentIndexRef.current + 1);
    }
    const hold = pausedForAudioHoldRef.current ? "on" : "off";
    setPlaybackDebug({ activeSegment, pendingSegment, hold });
  }

  useEffect(() => {
    const nextPool = new Map();
    segments.forEach((segment) => {
      const key = segmentKeyFor(segment);
      const existing = audioPoolRef.current.get(key);
      const audio = existing || new Audio(segment.audio_file_url);
      audio.preload = "metadata";
      audio.volume = audioVolume;
      nextPool.set(key, audio);
    });

    audioPoolRef.current.forEach((audio, key) => {
      if (!nextPool.has(key)) {
        audio.pause();
        audio.currentTime = 0;
      }
    });

    audioPoolRef.current = nextPool;
    playedSegmentsRef.current = new Set();
    activeSegmentIndexRef.current = -1;
    activeSegmentKeyRef.current = "";
    pendingSegmentIndexRef.current = null;
    pendingSegmentKeyRef.current = "";
    pausedForAudioHoldRef.current = false;
    refreshPlaybackDebug();
  }, [segments, audioVolume]);

  useEffect(() => {
    audioPoolRef.current.forEach((audio) => {
      audio.volume = audioVolume;
    });
  }, [audioVolume]);

  function stopAllAudio() {
    audioPoolRef.current.forEach((audio) => {
      audio.pause();
      audio.currentTime = 0;
      audio.onended = null;
    });
    activeSegmentIndexRef.current = -1;
    activeSegmentKeyRef.current = "";
    pendingSegmentIndexRef.current = null;
    pendingSegmentKeyRef.current = "";
    pausedForAudioHoldRef.current = false;
    playbackAttemptRef.current += 1;
    refreshPlaybackDebug();
  }

  function resetPlaybackState() {
    stopAllAudio();
    playedSegmentsRef.current = new Set();
    lastVideoTimeRef.current = 0;
    setCurrentVideoTime(0);
    setIsVideoPlaying(false);
  }

  function markSegmentsPlayedUntil(nextTime) {
    const played = new Set();
    segments.forEach((segment, index) => {
      const bounds = getSegmentBounds(segments, index);
      if (bounds && bounds.end <= nextTime) {
        played.add(segmentKeyFor(segment));
      }
    });
    playedSegmentsRef.current = played;
  }

  function queuePendingSegment(index) {
    const segment = segments[index];
    if (!segment) {
      return;
    }

    pendingSegmentIndexRef.current = index;
    pendingSegmentKeyRef.current = segmentKeyFor(segment);
    refreshPlaybackDebug();
  }

  function pauseVideoForAudioHold() {
    const video = videoRef.current;
    if (!video || video.paused) {
      return;
    }
    pausedForAudioHoldRef.current = true;
    video.pause();
    refreshPlaybackDebug();
  }

  function resumeVideoAfterAudioHold() {
    const video = videoRef.current;
    if (!video?.paused || !pausedForAudioHoldRef.current) {
      pausedForAudioHoldRef.current = false;
      refreshPlaybackDebug();
      return;
    }

    pausedForAudioHoldRef.current = false;
    refreshPlaybackDebug();
    video.play().catch((resumeError) => {
      setError(resumeError.message || "Unable to resume video playback after audio hold.");
    });
  }

  async function startSegmentPlayback(index, options = {}) {
    const { force = false } = options;
    const segment = segments[index];
    if (!segment) {
      return;
    }

    const key = segmentKeyFor(segment);
    if (!force && playedSegmentsRef.current.has(key)) {
      return;
    }
    if (activeSegmentKeyRef.current === key) {
      return;
    }

    const currentlyActiveKey = activeSegmentKeyRef.current;
    if (currentlyActiveKey) {
      const currentlyActiveAudio = audioPoolRef.current.get(currentlyActiveKey);
      const isStillPlaying = Boolean(currentlyActiveAudio && !currentlyActiveAudio.paused && !currentlyActiveAudio.ended);
      if (isStillPlaying) {
        queuePendingSegment(index);
        pauseVideoForAudioHold();
        return;
      }
    }

    const audio = audioPoolRef.current.get(key);
    if (!audio) {
      return;
    }

    const attempt = ++playbackAttemptRef.current;
    try {
      await loadAudioMetadata(audio);
      if (attempt !== playbackAttemptRef.current) {
        return;
      }

      const startTime = parseTime(segment.start_time);
      const offset = Math.max(0, lastVideoTimeRef.current - startTime);
      audio.currentTime = Math.min(offset, Number.isFinite(audio.duration) ? audio.duration : offset);

      audio.onended = () => {
        if (activeSegmentKeyRef.current !== key) {
          return;
        }

        activeSegmentIndexRef.current = -1;
        activeSegmentKeyRef.current = "";
        refreshPlaybackDebug();

        const pendingIndex = pendingSegmentIndexRef.current;
        pendingSegmentIndexRef.current = null;
        pendingSegmentKeyRef.current = "";
        refreshPlaybackDebug();

        if (pendingIndex !== null) {
          resumeVideoAfterAudioHold();
          void startSegmentPlayback(pendingIndex, { force: true });
          return;
        }

        resumeVideoAfterAudioHold();
      };

      activeSegmentIndexRef.current = index;
      activeSegmentKeyRef.current = key;
      playedSegmentsRef.current.add(key);
      refreshPlaybackDebug();
      await audio.play();
    } catch (playError) {
      if (activeSegmentKeyRef.current === key) {
        activeSegmentIndexRef.current = -1;
        activeSegmentKeyRef.current = "";
        refreshPlaybackDebug();
      }
      setError(playError.message || "Unable to play segment audio.");
    }
  }

  function reconcilePlaybackForSeek(nextTime) {
    stopAllAudio();
    markSegmentsPlayedUntil(nextTime);

    lastVideoTimeRef.current = nextTime;
    setCurrentVideoTime(nextTime);

    const video = videoRef.current;
    if (!video || video.paused) {
      return;
    }

    const index = findSegmentIndexAtTime(segments, nextTime);
    if (index >= 0) {
      void startSegmentPlayback(index);
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

    seekVideoToTime(startTime);
    try {
      await video.play();
    } catch (playError) {
      setError(playError.message || "Unable to continue playback from selected segment.");
    }
  }

  function syncAudioWithVideo(currentTime) {
    if (currentTime < lastVideoTimeRef.current) {
      stopAllAudio();
      markSegmentsPlayedUntil(currentTime);
    }

    const index = findSegmentIndexAtTime(segments, currentTime);
    if (index >= 0) {
      void startSegmentPlayback(index);
    }

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
    if (pausedForAudioHoldRef.current) {
      refreshPlaybackDebug();
      return;
    }
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

  function handleVideoTimeUpdate(event) {
    syncAudioWithVideo(event.currentTarget.currentTime || 0);
  }

  function handleSeekSliderChange(event) {
    const next = Number.parseFloat(event.target.value || "0");
    seekVideoToTime(next);
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

  const activeIndex = useMemo(
    () => findSegmentIndexAtTime(segments, currentVideoTime),
    [segments, currentVideoTime]
  );

  const activeCaptionText = useMemo(() => {
    if (!ccEnabled || activeIndex < 0) {
      return "";
    }
    return segments[activeIndex]?.transcript_rw || "";
  }, [activeIndex, ccEnabled, segments]);

  return {
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
  };
}