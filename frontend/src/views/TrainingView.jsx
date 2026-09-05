import React, { useState, useEffect, useRef } from 'react';
import { Play, Square, Activity, Terminal, CheckCircle2, AlertCircle } from 'lucide-react';
import {
  startTraining,
  getTrainingRuns,
  getTrainingStatus,
  cancelTraining,
  getWsUrl,
} from '../api/client';

export default function TrainingView({
  activeProject,
  datasets = [],
  activeDataset,
  onTrainingCompleted,
}) {
  const [selectedDatasetId, setSelectedDatasetId] = useState(activeDataset?.id || '');
  const [modelType, setModelType] = useState('yolo11n.pt');
  const [epochs, setEpochs] = useState(5);
  const [batchSize, setBatchSize] = useState(16);
  const [learningRate, setLearningRate] = useState(0.01);
  const [imgSize, setImgSize] = useState(640);
  const [device, setDevice] = useState('auto');

  // Training execution state
  const [currentRun, setCurrentRun] = useState(null);
  const [isTraining, setIsTraining] = useState(false);
  const [metrics, setMetrics] = useState({
    epoch: 0,
    total_epochs: 5,
    train_loss: 0,
    val_loss: 0,
    map50: 0,
    map50_95: 0,
  });
  const [logs, setLogs] = useState([]);

  const logConsoleRef = useRef(null);
  const wsRef = useRef(null);
  const pollIntervalRef = useRef(null);

  useEffect(() => {
    if (activeDataset?.id) {
      setSelectedDatasetId(activeDataset.id);
    } else if (datasets.length > 0 && !selectedDatasetId) {
      setSelectedDatasetId(datasets[0].id);
    }
  }, [activeDataset, datasets]);

  // Connect to live WebSocket
  useEffect(() => {
    const wsUrl = getWsUrl();
    try {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          if (msg.type === 'training_progress') {
            const data = msg.data;
            setMetrics((prev) => ({
              ...prev,
              epoch: data.epoch || prev.epoch,
              total_epochs: data.total_epochs || prev.total_epochs,
              train_loss: data.train_loss !== undefined ? data.train_loss : prev.train_loss,
              val_loss: data.val_loss !== undefined ? data.val_loss : prev.val_loss,
              map50: data.map50 !== undefined ? data.map50 : prev.map50,
              map50_95: data.map50_95 !== undefined ? data.map50_95 : prev.map50_95,
            }));
            if (data.log) {
              appendLog(data.log);
            }
          } else if (msg.type === 'training_completed') {
            setIsTraining(false);
            appendLog('Training completed successfully!');
            if (onTrainingCompleted) onTrainingCompleted();
          } else if (msg.type === 'training_failed') {
            setIsTraining(false);
            appendLog(`Training failed: ${msg.data?.error || 'Unknown error'}`);
          }
        } catch (e) {
          // ignore parsing error
        }
      };

      ws.onclose = () => {
        // ws closed
      };
    } catch (e) {
      console.warn('WebSocket connection error:', e);
    }

    return () => {
      if (wsRef.current) wsRef.current.close();
      if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
    };
  }, []);

  const appendLog = (line) => {
    setLogs((prev) => [...prev.slice(-300), `[${new Date().toLocaleTimeString()}] ${line}`]);
    setTimeout(() => {
      if (logConsoleRef.current) {
        logConsoleRef.current.scrollTop = logConsoleRef.current.scrollHeight;
      }
    }, 50);
  };

  const handleStartTraining = async () => {
    if (!selectedDatasetId) {
      alert('Please select a dataset to train on!');
      return;
    }

    setIsTraining(true);
    setLogs([]);
    appendLog(`Initializing training pipeline for dataset ${selectedDatasetId}...`);

    try {
      const run = await startTraining({
        project_id: activeProject?.id,
        dataset_id: selectedDatasetId,
        model_type: modelType,
        epochs: Number(epochs),
        batch_size: Number(batchSize),
        learning_rate: Number(learningRate),
        img_size: Number(imgSize),
        device: device,
      });

      setCurrentRun(run);
      appendLog(`Training task scheduled with Run ID: ${run.id}`);

      // Start fallback polling in case WebSocket is unavailable
      if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = setInterval(async () => {
        try {
          const status = await getTrainingStatus(run.id);
          if (status.status === 'completed') {
            setIsTraining(false);
            clearInterval(pollIntervalRef.current);
            appendLog('Training status: COMPLETED');
            if (onTrainingCompleted) onTrainingCompleted();
          } else if (status.status === 'failed') {
            setIsTraining(false);
            clearInterval(pollIntervalRef.current);
            appendLog(`Training status: FAILED (${status.error_message || ''})`);
          }
        } catch (err) {
          // ignore poll error
        }
      }, 3000);
    } catch (err) {
      setIsTraining(false);
      appendLog(`Failed to start training: ${err.message}`);
      alert(`Start failed: ${err.message}`);
    }
  };

  const handleCancelTraining = async () => {
    if (!currentRun) return;
    try {
      await cancelTraining(currentRun.id);
      setIsTraining(false);
      appendLog('Training cancellation requested.');
    } catch (err) {
      alert(`Cancel error: ${err.message}`);
    }
  };

  const progressPercent =
    metrics.total_epochs > 0 ? Math.round((metrics.epoch / metrics.total_epochs) * 100) : 0;

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '360px 1fr', gap: '20px' }}>
      {/* Hyperparameters Configuration */}
      <div className="card">
        <h3 style={{ fontSize: '16px', fontWeight: 600, marginBottom: '16px', color: '#fff' }}>
          Training Configuration
        </h3>

        <div className="form-group">
          <label className="form-label">Target Dataset</label>
          <select
            className="form-control"
            value={selectedDatasetId}
            onChange={(e) => setSelectedDatasetId(e.target.value)}
            disabled={isTraining}
          >
            {datasets.map((ds) => (
              <option key={ds.id} value={ds.id}>
                {ds.name} ({ds.num_images || 0} images)
              </option>
            ))}
          </select>
        </div>

        <div className="form-group">
          <label className="form-label">Backbone Architecture</label>
          <select
            className="form-control"
            value={modelType}
            onChange={(e) => setModelType(e.target.value)}
            disabled={isTraining}
          >
            <option value="yolo11n.pt">YOLO11 Nano (Fastest, CPU/Edge)</option>
            <option value="yolo11s.pt">YOLO11 Small (Balanced Accuracy)</option>
            <option value="yolo11m.pt">YOLO11 Medium (High Accuracy)</option>
            <option value="resnet18">ResNet-18 (Classification)</option>
          </select>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
          <div className="form-group">
            <label className="form-label">Epochs</label>
            <input
              type="number"
              className="form-control"
              min="1"
              max="200"
              value={epochs}
              onChange={(e) => setEpochs(Number(e.target.value))}
              disabled={isTraining}
            />
          </div>
          <div className="form-group">
            <label className="form-label">Batch Size</label>
            <input
              type="number"
              className="form-control"
              min="1"
              max="128"
              value={batchSize}
              onChange={(e) => setBatchSize(Number(e.target.value))}
              disabled={isTraining}
            />
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
          <div className="form-group">
            <label className="form-label">Learning Rate</label>
            <input
              type="number"
              step="0.001"
              className="form-control"
              value={learningRate}
              onChange={(e) => setLearningRate(Number(e.target.value))}
              disabled={isTraining}
            />
          </div>
          <div className="form-group">
            <label className="form-label">Image Size</label>
            <select
              className="form-control"
              value={imgSize}
              onChange={(e) => setImgSize(Number(e.target.value))}
              disabled={isTraining}
            >
              <option value="320">320 px</option>
              <option value="640">640 px (Recommended)</option>
              <option value="1024">1024 px</option>
            </select>
          </div>
        </div>

        <div className="form-group">
          <label className="form-label">Hardware Device</label>
          <select
            className="form-control"
            value={device}
            onChange={(e) => setDevice(e.target.value)}
            disabled={isTraining}
          >
            <option value="auto">Auto (Detect CUDA / MPS / CPU)</option>
            <option value="cpu">CPU Only</option>
            <option value="cuda">NVIDIA CUDA GPU</option>
          </select>
        </div>

        <div style={{ marginTop: '20px' }}>
          {!isTraining ? (
            <button
              className="btn btn-primary btn-lg"
              style={{ width: '100%' }}
              onClick={handleStartTraining}
            >
              <Play size={16} />
              Launch Training Loop
            </button>
          ) : (
            <button
              className="btn btn-danger btn-lg"
              style={{ width: '100%' }}
              onClick={handleCancelTraining}
            >
              <Square size={16} />
              Stop Training
            </button>
          )}
        </div>
      </div>

      {/* Live Telemetry & Monitoring */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
        {/* Progress & Live Metrics */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">
              <Activity size={18} color="var(--accent-primary)" />
              Live Training Telemetry
            </div>
            {isTraining && (
              <span className="badge badge-warning">
                Training in progress...
              </span>
            )}
          </div>

          <div style={{ marginBottom: '16px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px', fontSize: '13px' }}>
              <span style={{ color: 'var(--text-secondary)' }}>
                Epoch Progress: {metrics.epoch} / {metrics.total_epochs || epochs}
              </span>
              <span style={{ fontWeight: 600, color: 'var(--accent-primary)' }}>{progressPercent}%</span>
            </div>
            <div className="progress-track">
              <div className="progress-fill" style={{ width: `${progressPercent}%` }} />
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px' }}>
            <div style={{ background: 'rgba(255, 255, 255, 0.02)', padding: '12px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>TRAIN LOSS</div>
              <div style={{ fontSize: '18px', fontWeight: 700, color: '#fff' }}>
                {metrics.train_loss ? Number(metrics.train_loss).toFixed(4) : '--'}
              </div>
            </div>
            <div style={{ background: 'rgba(255, 255, 255, 0.02)', padding: '12px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>VAL LOSS</div>
              <div style={{ fontSize: '18px', fontWeight: 700, color: '#fff' }}>
                {metrics.val_loss ? Number(metrics.val_loss).toFixed(4) : '--'}
              </div>
            </div>
            <div style={{ background: 'rgba(255, 255, 255, 0.02)', padding: '12px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>mAP@50</div>
              <div style={{ fontSize: '18px', fontWeight: 700, color: 'var(--accent-secondary)' }}>
                {metrics.map50 ? `${(Number(metrics.map50) * 100).toFixed(1)}%` : '--'}
              </div>
            </div>
            <div style={{ background: 'rgba(255, 255, 255, 0.02)', padding: '12px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>mAP@50-95</div>
              <div style={{ fontSize: '18px', fontWeight: 700, color: 'var(--accent-primary)' }}>
                {metrics.map50_95 ? `${(Number(metrics.map50_95) * 100).toFixed(1)}%` : '--'}
              </div>
            </div>
          </div>
        </div>

        {/* Live Terminal Console Logs */}
        <div className="card" style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
          <div className="card-header" style={{ marginBottom: '10px' }}>
            <div className="card-title">
              <Terminal size={16} />
              Training Execution Logs
            </div>
            <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Auto-scrolling</span>
          </div>

          <div className="terminal-console" ref={logConsoleRef} style={{ flex: 1, minHeight: '220px' }}>
            {logs.length === 0 ? (
              <div style={{ color: 'var(--text-muted)' }}>Ready for training output...</div>
            ) : (
              logs.map((log, i) => <div key={i}>{log}</div>)
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
