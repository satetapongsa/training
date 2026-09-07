import React from 'react';
import { FolderUp, Play, Crosshair, Sparkles } from 'lucide-react';

export default function Topbar({ activeTab, setActiveTab, wsConnected }) {
  const navItems = [
    { id: 'studio', label: 'หน้าหลัก (โหลดรูป & ตีกรอบ GT)', icon: FolderUp },
    { id: 'training', label: 'เทรนโมเดล (Training)', icon: Play },
    { id: 'inference', label: 'ทดสอบภาพ (Testing)', icon: Crosshair },
  ];

  return (
    <header className="topbar">
      {/* Brand & Logo Area on Top-Left */}
      <div className="topbar-brand" onClick={() => setActiveTab && setActiveTab('studio')} style={{ cursor: 'pointer' }}>
        <div className="logo-badge-top">
          <svg
            width="22"
            height="22"
            viewBox="0 0 24 24"
            fill="none"
            stroke="#ffffff"
            strokeWidth="2.2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <circle cx="12" cy="12" r="3" />
            <path d="M3 12h3m12 0h3M12 3v3m0 12v3" />
            <path d="M5.6 5.6l2.1 2.1m8.6 8.6l2.1 2.1M5.6 18.4l2.1-2.1m8.6-8.6l2.1-2.1" />
          </svg>
        </div>
        <div className="topbar-title-group">
          <span className="topbar-brand-title">AI Vision Studio</span>
          <span className="topbar-brand-sub">Object Detection & Training</span>
        </div>
      </div>

      {/* Main Horizontal Navigation Bar (ย้ายแถบเครื่องมือมาไว้ขอบบน) */}
      <nav className="topbar-nav">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              type="button"
              className={`top-nav-item ${isActive ? 'active' : ''}`}
              onClick={() => setActiveTab && setActiveTab(item.id)}
            >
              <Icon size={16} strokeWidth={isActive ? 2.4 : 2} />
              <span>{item.label}</span>
            </button>
          );
        })}
      </nav>

      {/* Topbar Right Status */}
      <div className="topbar-right">
        <div className="system-pill">
          <span className="status-dot-active" />
          <span>ระบบพร้อมทำงาน</span>
        </div>
      </div>
    </header>
  );
}
