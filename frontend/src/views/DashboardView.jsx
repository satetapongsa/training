import React from 'react';
import {
  FolderKanban,
  Database,
  Boxes,
  Cpu,
  ArrowRight,
  FolderUp,
  Play,
  Crosshair,
} from 'lucide-react';

export default function DashboardView({
  projects = [],
  datasets = [],
  models = [],
  setActiveTab,
  activeProject,
}) {
  return (
    <div>
      {/* Stats Grid */}
      <div className="grid-stats">
        <div className="stat-card">
          <div className="stat-icon">
            <FolderKanban size={22} />
          </div>
          <div>
            <div className="stat-value">{projects.length}</div>
            <div className="stat-label">Projects</div>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon">
            <Database size={22} />
          </div>
          <div>
            <div className="stat-value">{datasets.length}</div>
            <div className="stat-label">Datasets</div>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon">
            <Boxes size={22} />
          </div>
          <div>
            <div className="stat-value">{models.length}</div>
            <div className="stat-label">Trained Runs</div>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon">
            <Cpu size={22} />
          </div>
          <div>
            <div className="stat-value" style={{ fontSize: '18px' }}>Active</div>
            <div className="stat-label">Execution Engine</div>
          </div>
        </div>
      </div>

      {/* Quick Launch Cards */}
      <div style={{ marginTop: '28px' }}>
        <h2 style={{ fontSize: '16px', fontWeight: 600, marginBottom: '16px', color: '#fff' }}>
          Workflow Modules
        </h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '16px' }}>
          <div className="card" style={{ cursor: 'pointer' }} onClick={() => setActiveTab('datasets')}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <div className="stat-icon" style={{ width: '36px', height: '36px' }}>
                  <FolderUp size={18} />
                </div>
                <h3 style={{ fontSize: '15px', fontWeight: 600 }}>Folder Upload & Datasets</h3>
              </div>
              <ArrowRight size={16} color="var(--text-muted)" />
            </div>
            <p style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
              Upload complete folders with images and YOLO text annotations. Supports drag & drop directory ingestion.
            </p>
          </div>

          <div className="card" style={{ cursor: 'pointer' }} onClick={() => setActiveTab('training')}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <div className="stat-icon" style={{ width: '36px', height: '36px' }}>
                  <Play size={18} />
                </div>
                <h3 style={{ fontSize: '15px', fontWeight: 600 }}>Train Neural Network</h3>
              </div>
              <ArrowRight size={16} color="var(--text-muted)" />
            </div>
            <p style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
              Launch genuine model training loops with live loss streaming, mAP metrics, and checkpoints.
            </p>
          </div>

          <div className="card" style={{ cursor: 'pointer' }} onClick={() => setActiveTab('inference')}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <div className="stat-icon" style={{ width: '36px', height: '36px' }}>
                  <Crosshair size={18} />
                </div>
                <h3 style={{ fontSize: '15px', fontWeight: 600 }}>Live Inference Tester</h3>
              </div>
              <ArrowRight size={16} color="var(--text-muted)" />
            </div>
            <p style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
              Run instant visual inference against trained weights with bounding box overlays and latency timing.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
