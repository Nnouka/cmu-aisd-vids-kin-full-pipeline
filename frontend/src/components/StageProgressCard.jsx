import { CheckCircleTwoTone, ClockCircleOutlined, ReloadOutlined } from "@ant-design/icons";
import { Alert, Button, Card, Progress, Timeline, Typography } from "antd";

import { JOB_STAGES } from "../constants";
import { formatDuration, formatStageLabel, formatTimestamp } from "../lib/formatters";

const { Text } = Typography;

function stageState(status, currentStageIndex, lastFailedStage, stage, index) {
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

function createTimelineItems(status, currentStage, lastFailedStage) {
  const currentStageIndex = JOB_STAGES.indexOf(currentStage);
  return JOB_STAGES.map((stage, index) => {
    const state = stageState(status, currentStageIndex, lastFailedStage, stage, index);
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
}

export function StageProgressCard({
  status,
  currentStage,
  lastFailedStage,
  stageProgressPercent,
  jobStartedAt,
  jobCompletedAt,
  displayedDurationSeconds,
  isContinuing,
  isRestarting,
  canContinueJob,
  jobId,
  onContinueJob,
  onRestartJob,
}) {
  const timelineItems = createTimelineItems(status, currentStage, lastFailedStage);

  return (
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
      <div className="timing-block">
        <div className="timing-row">
          <Text type="secondary">Started</Text>
          <Text>{formatTimestamp(jobStartedAt)}</Text>
        </div>
        <div className="timing-row">
          <Text type="secondary">Completed</Text>
          <Text>{formatTimestamp(jobCompletedAt)}</Text>
        </div>
        <div className="timing-row">
          <Text type="secondary">Duration</Text>
          <Text strong>{formatDuration(displayedDurationSeconds)}</Text>
        </div>
      </div>
      <div className="actions-row">
        <Button
          type="primary"
          icon={<ReloadOutlined />}
          loading={isContinuing}
          onClick={onContinueJob}
          disabled={!canContinueJob || isRestarting}
        >
          Continue Job
        </Button>
        <Button
          icon={<ReloadOutlined />}
          loading={isRestarting}
          onClick={onRestartJob}
          disabled={!jobId || status !== "failed" || isContinuing}
        >
          Restart Failed Job
        </Button>
      </div>
    </Card>
  );
}