import React, { useState, useRef, useEffect } from 'react';
import { Crosshair, Upload, Zap, Sliders, Image as ImageIcon } from 'lucide-react';
import { runInference, getTrainingRuns } from '../api/client';

export default function InferenceView({ activeProject, preselectedModel }) {
  const [runs, setRuns] = useState([]);
  const [selectedRunId, setSelectedRunId] = useState(preselectedModel?.id || '');
  const [confThreshold, setConfThreshold] = useState(0.25);
  const [iouThreshold, setIouThreshold] = useState(0.45);
  const [testImageFile, setTestImageFile] = useState(null);
  const [testImagePreview, setTestImagePreview] = useState(null);
  const [inferenceResult, setInferenceResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const canvasRef = useRef(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    loadRuns();
  }, [activeProject]);

  useEffect(() => {
    if (preselectedModel?.id) {
      setSelectedRunId(preselectedModel.id);
    }
  }, [preselectedModel]);

  const loadRuns = async () => {
    try {
      const data = await getTrainingRuns(activeProject?.id);
      setRuns(data || []);
      if (data && data.length > 0 && !selectedRunId) {
        setSelectedRunId(data[0].id);
      }
    } catch (err) {
      console.error('Failed to load runs for inference:', err);
    }
  };

  const handleFileSelect = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setTestImageFile(file);
    const url = URL.createObjectURL(file);
    setTestImagePreview(url);
    setInferenceResult(null);
  };

  const handleRunInference = async () => {
    if (!testImageFile) {
      alert('Please upload an image to run inference on!');
      return;
    }

    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('file', testImageFile);
      if (selectedRunId) {
        formData.append('run_id', selectedRunId);
      }
      formData.append('confidence', confThreshold);
      formData.append('iou_threshold', iouThreshold);

      const result = await runInference(formData);
      setInferenceResult(result);
    } catch (err) {
      alert(`Inference failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Render bounding boxes on Canvas
  useEffect(() => {
    if (!testImagePreview || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const img = new Image();
    img.src = testImagePreview;
    img.onload = () => {
      canvas.width = img.naturalWidth || 800;
      canvas.height = img.naturalHeight || 600;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0);

      // Draw detection bounding boxes if available
      if (inferenceResult?.detections) {
        inferenceResult.detections.forEach((det) => {
          const x = det.box.x1 * canvas.width;
          const y = det.box.y1 * canvas.height;
          const w = (det.box.x2 - det.box.x1) * canvas.width;
          const h = (det.box.y2 - det.box.y1) * canvas.height;

          // Box
          ctx.strokeStyle = '#10b981';
          ctx.lineWidth = 3;
          ctx.strokeRect(x, y, w, h);

          // Label chip
          const labelText = `${det.class_name} ${(det.confidence * 100).toFixed(0)}%`;
          ctx.font = 'bold 12px Inter, sans-serif';
          const textWidth = ctx.measureText(labelText).width;

          ctx.fillStyle = '#10b981';
          ctx.fillRect(x, Math.max(0, y - 22), textWidth + 14, 22);

          ctx.fillStyle = '#ffffff';
          ctx.fillText(labelText, x + 7, Math.max(14, y - 6));
        });
      }
    };
  }, [testImagePreview, inferenceResult]);

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '320px 1fr', gap: '20px' }}>
      {/* Controls */}
      <div className="card">
        <h3 style={{ fontSize: '16px', fontWeight: 600, marginBottom: '16px' }}>
          Inference Parameters
        </h3>

        <div className="form-group">
          <label className="form-label">Trained Model Checkpoint</label>
          <select
            className="form-control"
            value={selectedRunId}
            onChange={(e) => setSelectedRunId(e.target.value)}
          >
            {runs.length === 0 && <option value="">Default Baseline Model</option>}
            {runs.map((r) => (
              <option key={r.id} value={r.id}>
                {r.name || `Run #${r.id.slice(0, 8)}`} ({r.model_type})
              </option>
            ))}
          </select>
        </div>

        <div className="form-group">
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
            <label className="form-label">Confidence Threshold</label>
            <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--accent-primary)' }}>
              {confThreshold}
            </span>
          </div>
          <input
            type="range"
            min="0.05"
            max="0.95"
            step="0.05"
            value={confThreshold}
            onChange={(e) => setConfThreshold(Number(e.target.value))}
            style={{ width: '100%' }}
          />
        </div>

        <div className="form-group">
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
            <label className="form-label">NMS IoU Threshold</label>
            <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--accent-secondary)' }}>
              {iouThreshold}
            </span>
          </div>
          <input
            type="range"
            min="0.1"
            max="0.9"
            step="0.05"
            value={iouThreshold}
            onChange={(e) => setIouThreshold(Number(e.target.value))}
            style={{ width: '100%' }}
          />
        </div>

        <div style={{ marginTop: '20px' }}>
          <input
            type="file"
            ref={fileInputRef}
            accept="image/*"
            style={{ display: 'none' }}
            onChange={handleFileSelect}
          />
          <button
            className="btn btn-secondary"
            style={{ width: '100%', marginBottom: '10px' }}
            onClick={() => fileInputRef.current?.click()}
          >
            <Upload size={14} /> Select Test Image
          </button>

          <button
            className="btn btn-primary btn-lg"
            style={{ width: '100%' }}
            onClick={handleRunInference}
            disabled={loading || !testImageFile}
          >
            <Zap size={16} /> {loading ? 'Running Model...' : 'Execute Inference'}
          </button>
        </div>

        {/* Inference Telemetry Result */}
        {inferenceResult && (
          <div style={{ marginTop: '20px', padding: '14px', background: 'rgba(255, 255, 255, 0.02)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
              <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Latency:</span>
              <span style={{ fontSize: '13px', fontWeight: 600, color: 'var(--accent-success)' }}>
                {inferenceResult.inference_time_ms ? `${inferenceResult.inference_time_ms.toFixed(1)} ms` : '--'}
              </span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
              <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Detections:</span>
              <span style={{ fontSize: '13px', fontWeight: 600, color: '#fff' }}>
                {inferenceResult.detections ? inferenceResult.detections.length : 0} objects
              </span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Device:</span>
              <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
                {inferenceResult.device || 'CPU'}
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Visual Canvas Area */}
      <div className="card" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '480px', overflow: 'hidden' }}>
        {testImagePreview ? (
          <div style={{ maxWidth: '100%', maxHeight: '100%', overflow: 'auto', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <canvas
              ref={canvasRef}
              style={{
                maxWidth: '100%',
                maxHeight: 'calc(100vh - 180px)',
                borderRadius: 'var(--radius-sm)',
                boxShadow: '0 10px 30px rgba(0, 0, 0, 0.5)',
              }}
            />
          </div>
        ) : (
          <div style={{ textAlign: 'center', color: 'var(--text-muted)' }}>
            <Crosshair size={40} style={{ margin: '0 auto 10px auto', display: 'block' }} />
            Select or upload a test image to run real-time bounding box detection.
          </div>
        )}
      </div>
    </div>
  );
}
