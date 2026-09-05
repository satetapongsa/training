import React, { useState, useEffect } from 'react';
import { Boxes, Download, Cpu, CheckCircle2, Share2, Layers } from 'lucide-react';
import { getTrainingRuns, API_BASE_URL } from '../api/client';

export default function ModelsView({ activeProject, onSelectForInference }) {
  const [runs, setRuns] = useState([]);
  const [loading, setLoading] = useState(false);
  const [exportingId, setExportingId] = useState(null);

  useEffect(() => {
    loadRuns();
  }, [activeProject]);

  const loadRuns = async () => {
    setLoading(true);
    try {
      const data = await getTrainingRuns(activeProject?.id);
      setRuns(data || []);
    } catch (err) {
      console.error('Failed to load training runs:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleExportONNX = async (runId) => {
    setExportingId(runId);
    try {
      const res = await fetch(`${API_BASE_URL}/api/v1/models/${runId}/export`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ format: 'onnx' }),
      });
      if (!res.ok) throw new Error('Export failed');
      const data = await res.json();
      alert(`Model successfully exported to ONNX format! File: ${data.export_path}`);
      await loadRuns();
    } catch (err) {
      alert(`Export error: ${err.message}`);
    } finally {
      setExportingId(null);
    }
  };

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
        <div>
          <h2 style={{ fontSize: '18px', fontWeight: 600 }}>Model Registry & Artifacts</h2>
          <p style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
            Inspect trained checkpoints, export to ONNX, and download weights.
          </p>
        </div>
      </div>

      {runs.length === 0 ? (
        <div className="card" style={{ textAlign: 'center', padding: '40px', color: 'var(--text-muted)' }}>
          <Boxes size={32} style={{ margin: '0 auto 8px auto', display: 'block' }} />
          No trained models found yet. Train a model in the Training tab to see weights here!
        </div>
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(340px, 1fr))', gap: '16px' }}>
          {runs.map((run) => (
            <div key={run.id} className="card">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '12px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <div className="stat-icon" style={{ width: '38px', height: '38px' }}>
                    <Layers size={18} />
                  </div>
                  <div>
                    <h3 style={{ fontSize: '15px', fontWeight: 600 }}>{run.name || `Run #${run.id.slice(0, 8)}`}</h3>
                    <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                      Backbone: {run.model_type || 'yolo11n'}
                    </div>
                  </div>
                </div>
                <span
                  className="badge"
                  style={{
                    background: run.status === 'completed' ? 'rgba(16, 185, 129, 0.15)' : 'rgba(239, 68, 68, 0.15)',
                    color: run.status === 'completed' ? '#6ee7b7' : '#fca5a5',
                  }}
                >
                  {run.status}
                </span>
              </div>

              {/* Metrics */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '8px', margin: '14px 0', background: 'rgba(255, 255, 255, 0.02)', padding: '10px', borderRadius: 'var(--radius-sm)' }}>
                <div>
                  <div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>mAP@50</div>
                  <div style={{ fontSize: '14px', fontWeight: 600, color: 'var(--accent-secondary)' }}>
                    {run.metrics?.map50 ? `${(run.metrics.map50 * 100).toFixed(1)}%` : '--'}
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>mAP@50-95</div>
                  <div style={{ fontSize: '14px', fontWeight: 600, color: 'var(--accent-primary)' }}>
                    {run.metrics?.map50_95 ? `${(run.metrics.map50_95 * 100).toFixed(1)}%` : '--'}
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>EPOCHS</div>
                  <div style={{ fontSize: '14px', fontWeight: 600, color: '#fff' }}>
                    {run.current_epoch || run.epochs || 0}
                  </div>
                </div>
              </div>

              {/* Action Buttons */}
              <div style={{ display: 'flex', gap: '8px', borderTop: '1px solid var(--border-color)', paddingTop: '12px' }}>
                <a
                  href={`${API_BASE_URL}/api/v1/models/${run.id}/download?format=pt`}
                  download
                  className="btn btn-sm btn-primary"
                  style={{ flex: 1, textDecoration: 'none' }}
                >
                  <Download size={13} /> Download .pt
                </a>
                <button
                  className="btn btn-sm btn-secondary"
                  onClick={() => handleExportONNX(run.id)}
                  disabled={exportingId === run.id}
                  title="Export to Open Neural Network Exchange (ONNX)"
                >
                  <Share2 size={13} /> {exportingId === run.id ? 'Exporting...' : 'ONNX'}
                </button>
                {onSelectForInference && (
                  <button
                    className="btn btn-sm btn-secondary"
                    onClick={() => onSelectForInference(run)}
                  >
                    Test
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
