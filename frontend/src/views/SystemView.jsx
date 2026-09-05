import React, { useState, useEffect } from 'react';
import { Cpu, HardDrive, RefreshCw, CheckCircle2, Server } from 'lucide-react';
import { getSystemHealth } from '../api/client';

export default function SystemView() {
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    checkHealth();
  }, []);

  const checkHealth = async () => {
    setLoading(true);
    try {
      const data = await getSystemHealth();
      setHealth(data);
    } catch (err) {
      setHealth({ status: 'offline', error: err.message });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: '800px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
        <div>
          <h2 style={{ fontSize: '18px', fontWeight: 600 }}>System Diagnostics & Compute</h2>
          <p style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
            Real-time status of the Python backend and hardware acceleration.
          </p>
        </div>
        <button className="btn btn-secondary btn-sm" onClick={checkHealth} disabled={loading}>
          <RefreshCw size={13} className={loading ? 'spin' : ''} />
          Refresh
        </button>
      </div>

      <div className="grid-stats" style={{ gridTemplateColumns: '1fr 1fr' }}>
        <div className="stat-card">
          <div className="stat-icon">
            <Server size={22} />
          </div>
          <div>
            <div className="stat-value" style={{ fontSize: '16px' }}>
              {health?.status === 'healthy' ? (
                <span style={{ color: 'var(--accent-success)', display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <CheckCircle2 size={16} /> Online (v{health?.version || '1.0.0'})
                </span>
              ) : (
                <span style={{ color: 'var(--accent-danger)' }}>Offline / Connecting...</span>
              )}
            </div>
            <div className="stat-label">FastAPI Backend Engine</div>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon">
            <Cpu size={22} />
          </div>
          <div>
            <div className="stat-value" style={{ fontSize: '16px' }}>
              PyTorch & Ultralytics
            </div>
            <div className="stat-label">Neural Execution Runtime</div>
          </div>
        </div>
      </div>

      <div className="card" style={{ marginTop: '20px' }}>
        <h3 style={{ fontSize: '15px', fontWeight: 600, marginBottom: '14px' }}>
          Deployment & Runtime Architecture
        </h3>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', fontSize: '13px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 0', borderBottom: '1px solid var(--border-color)' }}>
            <span style={{ color: 'var(--text-secondary)' }}>Frontend Architecture</span>
            <span style={{ fontWeight: 500, color: '#fff' }}>React 19 + Vite (Vercel Ready)</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 0', borderBottom: '1px solid var(--border-color)' }}>
            <span style={{ color: 'var(--text-secondary)' }}>Backend Framework</span>
            <span style={{ fontWeight: 500, color: '#fff' }}>FastAPI (Asynchronous Python 3.10+)</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 0', borderBottom: '1px solid var(--border-color)' }}>
            <span style={{ color: 'var(--text-secondary)' }}>Training Architecture</span>
            <span style={{ fontWeight: 500, color: '#fff' }}>Ultralytics YOLO11 & PyTorch CNN</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 0', borderBottom: '1px solid var(--border-color)' }}>
            <span style={{ color: 'var(--text-secondary)' }}>Telemetry Transport</span>
            <span style={{ fontWeight: 500, color: '#fff' }}>Real-time WebSockets (/ws/live)</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 0' }}>
            <span style={{ color: 'var(--text-secondary)' }}>Export Formats</span>
            <span style={{ fontWeight: 500, color: '#fff' }}>PyTorch (.pt), ONNX (.onnx)</span>
          </div>
        </div>
      </div>
    </div>
  );
}
