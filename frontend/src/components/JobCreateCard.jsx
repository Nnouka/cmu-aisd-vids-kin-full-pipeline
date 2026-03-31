import { PlayCircleOutlined, UploadOutlined } from "@ant-design/icons";
import { Alert, Button, Card, Upload } from "antd";

export function JobCreateCard({
  uploadProps,
  onSubmit,
  isSubmitting,
  isCheckingExistingJob,
  shouldDisableStartProcessing,
  error,
}) {
  return (
    <Card title="Create Job" className="panel-card" extra={<UploadOutlined />}>
      <form onSubmit={onSubmit}>
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
  );
}