import { ClockCircleOutlined } from "@ant-design/icons";
import { Layout, Tag, Typography } from "antd";

const { Header } = Layout;
const { Title, Text } = Typography;

export function AppHeader({ status, pollIntervalSeconds, statusColor }) {
  return (
    <Header className="app-header">
      <div>
        <Text type="secondary">Creator Console</Text>
        <Title level={3} className="app-title">DeepKIN Studio</Title>
      </div>
      <div className="header-meta">
        <Tag color={statusColor}>{status}</Tag>
        <Tag icon={<ClockCircleOutlined />}>Poll {pollIntervalSeconds}s</Tag>
      </div>
    </Header>
  );
}