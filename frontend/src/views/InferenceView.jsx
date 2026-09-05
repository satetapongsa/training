import React, { useState, useRef, useEffect } from 'react';
import { Crosshair, Upload, Zap, Sliders, Image as ImageIcon, Layers, CheckCircle2, AlertCircle } from 'lucide-react';
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
      alert('กรุณาเลือกไฟล์ภาพสำหรับทดสอบก่อนกดตรวจจับ');
      return;
    }

    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('file', testImageFile);
      if (selectedRunId) {
        formData.append('run_id', selectedRunId);
        formData.append('model_id', selectedRunId);
      }
      formData.append('confidence', confThreshold);
      formData.append('conf_threshold', confThreshold);
      formData.append('iou_threshold', iouThreshold);

      const result = await runInference(formData);
      setInferenceResult(result);
    } catch (err) {
      alert(`การตรวจจับล้มเหลว: ${err.message}`);
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
      if (inferenceResult?.detections && inferenceResult.detections.length > 0) {
        inferenceResult.detections.forEach((det, idx) => {
          const x1 = det.box ? det.box.x1 : (det.x1 !== undefined ? det.x1 : 0);
          const y1 = det.box ? det.box.y1 : (det.y1 !== undefined ? det.y1 : 0);
          const x2 = det.box ? det.box.x2 : (det.x2 !== undefined ? det.x2 : 1);
          const y2 = det.box ? det.box.y2 : (det.y2 !== undefined ? det.y2 : 1);

          const x = x1 * canvas.width;
          const y = y1 * canvas.height;
          const w = (x2 - x1) * canvas.width;
          const h = (y2 - y1) * canvas.height;

          // Distinct color per class
          const colors = ['#10b981', '#4f46e5', '#f59e0b', '#ef4444', '#06b6d4', '#ec4899'];
          const color = colors[idx % colors.length];

          // Box rect
          ctx.strokeStyle = color;
          ctx.lineWidth = 3;
          ctx.strokeRect(x, y, w, h);

          // Semi-transparent background
          ctx.fillStyle = `${color}20`;
          ctx.fillRect(x, y, w, h);

          // Label chip
          const confScore = Math.round((det.confidence || 1.0) * 100);
          const labelText = `${det.class_name || 'object'} ${confScore}%`;
          ctx.font = 'bold 12px Inter, sans-serif';
          const textWidth = ctx.measureText(labelText).width;

          ctx.fillStyle = color;
          ctx.fillRect(x, Math.max(0, y - 22), textWidth + 14, 22);

          ctx.fillStyle = '#ffffff';
          ctx.fillText(labelText, x + 7, Math.max(14, y - 6));
        });
      }
    };
  }, [testImagePreview, inferenceResult]);

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '360px 1fr', gap: '20px' }}>
      {/* Controls Column */}
      <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
        <h3 style={{ fontSize: '15px', fontWeight: 600, color: 'var(--text-primary)' }}>
          ตั้งค่าการทดสอบภาพ
        </h3>

        <div className="form-group" style={{ marginBottom: 0 }}>
          <label className="form-label">เลือกโมเดลที่ต้องการทดสอบ</label>
          <select
            className="form-control"
            value={selectedRunId}
            onChange={(e) => setSelectedRunId(e.target.value)}
          >
            <option value="">โมเดลล่าสุดในระบบ (Auto-Detect Latest Model)</option>
            {runs.map((r) => {
              const name = r.model_name || r.name || `Model #${r.id}`;
              const arch = r.architecture || r.model_type || 'KDel 4.0';
              return (
                <option key={r.id} value={r.id}>
                  {name} ({arch})
                </option>
              );
            })}
          </select>
        </div>

        <div className="form-group" style={{ marginBottom: 0 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
            <label className="form-label" style={{ marginBottom: 0 }}>
              ค่าความมั่นใจขั้นต่ำ (Confidence Threshold)
            </label>
            <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--accent-primary)' }}>
              {Math.round(confThreshold * 100)}%
            </span>
          </div>
          <input
            type="range"
            min="0.05"
            max="0.95"
            step="0.05"
            value={confThreshold}
            onChange={(e) => setConfThreshold(Number(e.target.value))}
            style={{ width: '100%', cursor: 'pointer' }}
          />
        </div>

        <div className="form-group" style={{ marginBottom: 0 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
            <label className="form-label" style={{ marginBottom: 0 }}>
              IoU Threshold (NMS)
            </label>
            <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--accent-secondary)' }}>
              {Math.round(iouThreshold * 100)}%
            </span>
          </div>
          <input
            type="range"
            min="0.1"
            max="0.9"
            step="0.05"
            value={iouThreshold}
            onChange={(e) => setIouThreshold(Number(e.target.value))}
            style={{ width: '100%', cursor: 'pointer' }}
          />
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginTop: '4px' }}>
          <input
            type="file"
            ref={fileInputRef}
            accept="image/*"
            style={{ display: 'none' }}
            onChange={handleFileSelect}
          />
          <button
            className="btn btn-secondary"
            style={{ width: '100%' }}
            onClick={() => fileInputRef.current?.click()}
          >
            <Upload size={14} /> เลือกรูปภาพจากเครื่อง
          </button>

          <button
            className="btn btn-primary btn-lg"
            style={{ width: '100%', fontWeight: 600 }}
            onClick={handleRunInference}
            disabled={loading || !testImageFile}
          >
            <Zap size={16} /> {loading ? 'กำลังตรวจจับ...' : 'เริ่มตรวจจับภาพ (Run Inference)'}
          </button>
        </div>

        {/* Inference Telemetry Result */}
        {inferenceResult && (
          <div
            style={{
              marginTop: '4px',
              padding: '14px',
              backgroundColor: '#f8fafc',
              borderRadius: 'var(--radius-sm)',
              border: '1px solid var(--border-color)',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
              <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>โมเดลที่ใช้:</span>
              <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-primary)' }}>
                {inferenceResult.model_name || 'KDel 4.0'}
              </span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
              <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>ความเร็วประมวลผล:</span>
              <span style={{ fontSize: '12px', fontWeight: 600, color: '#10b981' }}>
                {inferenceResult.inference_time_ms ? `${inferenceResult.inference_time_ms} ms` : '--'}
              </span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
              <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>จำนวนวัตถุที่พบ:</span>
              <span style={{ fontSize: '12px', fontWeight: 700, color: 'var(--accent-primary)' }}>
                {inferenceResult.detections ? inferenceResult.detections.length : 0} รายการ
              </span>
            </div>

            {/* List detected items */}
            {inferenceResult.detections && inferenceResult.detections.length > 0 && (
              <div style={{ marginTop: '10px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-secondary)' }}>
                  รายการวัตถุที่ตรวจพบ:
                </div>
                {inferenceResult.detections.map((d, i) => (
                  <div
                    key={i}
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      fontSize: '11px',
                      padding: '4px 8px',
                      backgroundColor: '#ffffff',
                      borderRadius: '4px',
                      border: '1px solid var(--border-color)',
                    }}
                  >
                    <span>{d.class_name || 'object'}</span>
                    <strong style={{ color: '#10b981' }}>{Math.round((d.confidence || 1.0) * 100)}%</strong>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Visual Canvas Area */}
      <div
        className="card"
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          minHeight: '480px',
          overflow: 'hidden',
          backgroundColor: '#f8fafc',
        }}
      >
        {testImagePreview ? (
          <div
            style={{
              maxWidth: '100%',
              maxHeight: '100%',
              overflow: 'auto',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              padding: '12px',
            }}
          >
            <canvas
              ref={canvasRef}
              style={{
                maxWidth: '100%',
                maxHeight: 'calc(100vh - 180px)',
                borderRadius: 'var(--radius-sm)',
                boxShadow: '0 4px 20px rgba(0, 0, 0, 0.1)',
                backgroundColor: '#ffffff',
              }}
            />
          </div>
        ) : (
          <div style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '40px 20px' }}>
            <Crosshair size={44} style={{ margin: '0 auto 12px auto', display: 'block', color: 'var(--text-muted)' }} />
            <div style={{ fontSize: '14px', fontWeight: 500, color: 'var(--text-primary)', marginBottom: '4px' }}>
              ยังไม่ได้เลือกรูปภาพทดสอบ
            </div>
            <div style={{ fontSize: '12px' }}>
              คลิกปุ่ม &ldquo;เลือกรูปภาพจากเครื่อง&rdquo; เพื่อนำภาพมาทดสอบการตรวจจับวัตถุด้วยโมเดล AI
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
