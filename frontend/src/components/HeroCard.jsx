import { Card, Col, Row, Statistic, Typography } from "antd";

import { formatStageLabel } from "../lib/formatters";

const { Title, Paragraph } = Typography;

export function HeroCard({ jobId, currentStage, segmentCount }) {
  return (
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
              <Statistic title="Segments" value={segmentCount} />
            </Col>
          </Row>
        </Col>
      </Row>
    </Card>
  );
}