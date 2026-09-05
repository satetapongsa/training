import React from 'react';
import { Plus, Wifi, FolderPlus } from 'lucide-react';

export default function Topbar({ activeTab, onNewProject, onNewDataset, wsConnected }) {
  const titles = {
    dashboard: 'Platform Overview',
    projects: 'Project Management',
    datasets: 'Dataset & Folder Studio',
    annotations: 'Interactive Label Studio',
    training: 'Model Training & Live Telemetry',
    models: 'Model Registry & Exports',
    inference: 'Inference Playground & Testing',
    system: 'System Diagnostics & Hardware',
  };

  return (
    <header className="topbar">
      <div className="topbar-title">
        <span>{titles[activeTab] || 'Studio'}</span>
        {wsConnected && (
          <span className="badge badge-success" style={{ fontSize: '10px' }}>
            <Wifi size={12} /> Live
          </span>
        )}
      </div>

      <div className="topbar-actions">
        <button className="btn btn-sm btn-secondary" onClick={onNewProject}>
          <FolderPlus size={14} />
          New Project
        </button>
        <button className="btn btn-sm btn-primary" onClick={onNewDataset}>
          <Plus size={14} />
          New Dataset
        </button>
      </div>
    </header>
  );
}
