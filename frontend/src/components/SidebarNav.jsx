import {
  DashboardOutlined,
  FileTextOutlined,
  TranslationOutlined,
  VideoCameraOutlined,
} from "@ant-design/icons";
import { Layout, List } from "antd";

const { Sider } = Layout;

const NAV_ITEMS = [
  { icon: <VideoCameraOutlined />, label: "Video Dubbing" },
  { icon: <DashboardOutlined />, label: "Job Queue" },
  { icon: <FileTextOutlined />, label: "Transcripts" },
  { icon: <TranslationOutlined />, label: "Audio Segments" },
];

export function SidebarNav() {
  return (
    <Sider width={220} className="app-sider" breakpoint="lg" collapsedWidth="0">
      <List
        size="small"
        dataSource={NAV_ITEMS}
        renderItem={(item, idx) => (
          <List.Item className={idx === 0 ? "sider-item active" : "sider-item"}>
            <span className="sider-icon">{item.icon}</span>
            <span>{item.label}</span>
          </List.Item>
        )}
      />
    </Sider>
  );
}