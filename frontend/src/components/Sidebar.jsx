import React from 'react';
import {
  FolderUp,
  Play,
  Crosshair,
  Boxes,
  Database,
  Cpu,
} from 'lucide-react';

export default function Sidebar({ activeTab, setActiveTab, activeProject }) {
  const navItems = [
    { id: 'studio', label: 'Studio Workspace', icon: FolderUp },
    { id: 'training', label: 'Model Training', icon: Play },
    { id: 'inference', label: 'Inference Tester', icon: Crosshair },
    { id: 'models', label: 'Model Registry', icon: Boxes },
    { id: 'datasets', label: 'All Datasets', icon: Database },
    { id: 'system', label: 'System Info', icon: Cpu },
  ];

  return (
    <aside className="sidebar">
      {/* Brand area: ONLY the web app icon logo, no brand text and no model names */}
      <div className="sidebar-logo-area">
        <div
          className="logo-badge"
          title="Vision Studio"
          onClick={() => setActiveTab('studio')}
        >
          <svg
            width="26"
            height="26"
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
      </div>

      <nav className="sidebar-nav">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <div
              key={item.id}
              className={`nav-item ${isActive ? 'active' : ''}`}
              onClick={() => setActiveTab(item.id)}
            >
              <Icon size={18} strokeWidth={2} />
              <span className="nav-label">{item.label}</span>
            </div>
          );
        })}
      </nav>

      <div className="sidebar-footer">
        <div style={{ color: 'var(--text-muted)', marginBottom: '4px' }}>Active Project</div>
        <div style={{ color: '#fff', fontWeight: 600, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
          {activeProject?.name || 'None Selected'}
        </div>
      </div>
    </aside>
  );
}
