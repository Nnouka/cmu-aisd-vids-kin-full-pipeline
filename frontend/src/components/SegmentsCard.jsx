import { FileTextOutlined } from "@ant-design/icons";
import { Alert, Badge, Card, List, Typography } from "antd";

import { parseTime } from "../lib/formatters";

const { Paragraph, Text } = Typography;

export function SegmentsCard({ segments, activeIndex, onSegmentClick }) {
  return (
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
                onClick={() => onSegmentClick(parseTime(segment.start_time))}
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
  );
}