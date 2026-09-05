import React, { useState, useRef, useEffect } from 'react';
import {
  Crosshair,
  Upload,
  Zap,
  Sliders,
  Image as ImageIcon,
  Layers,
  CheckCircle2,
  AlertCircle,
  Tag,
  Box,
  Sparkles,
} from 'lucide-react';
import { runInference, getTrainingRuns } from '../api/client';

const PALETTE = [
  '#10b981', // emerald
  '#4f46e5', // indigo
  '#f59e0b', // amber
  '#ef4444', // rose
  '#06b6d4', // cyan
  '#ec4899', // pink
  '#8b5cf6', // violet
  '#14b8a6', // teal
  '#f97316', // orange
];

const getClassColor = (className, allClassNames = []) => {
  const idx = allClassNames.indexOf(className);
  if (idx === -1) return PALETTE[0];
  return PALETTE[idx % PALETTE.length];
};

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

  // Compute breakdown of detected objects by class
  const detections = inferenceResult?.detections || [];
  const totalCount = detections.length;

  const classMap = {};
  detections.forEach((d) => {
    const name = d.class_name || 'object';
    if (!classMap[name]) {
      classMap[name] = {
        name,
        count: 0,
        scores: [],
      };
    }
    classMap[name].count += 1;
    classMap[name].scores.push(d.confidence || 1.0);
  });
  const breakdownList = Object.values(classMap);
  const uniqueClassNames = breakdownList.map((b) => b.name);

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
      if (detections.length > 0) {
        detections.forEach((det) => {
          const x1 = det.box ? det.box.x1 : (det.x1 !== undefined ? det.x1 : 0);
          const y1 = det.box ? det.box.y1 : (det.y1 !== undefined ? det.y1 : 0);
          const x2 = det.box ? det.box.x2 : (det.x2 !== undefined ? det.x2 : 1);
          const y2 = det.box ? det.box.y2 : (det.y2 !== undefined ? det.y2 : 1);

          const x = x1 * canvas.width;
          const y = y1 * canvas.height;
          const w = (x2 - x1) * canvas.width;
          const h = (y2 - y1) * canvas.height;

          const className = det.class_name || 'object';
          const color = getClassColor(className, uniqueClassNames);

          // Box rect
          ctx.strokeStyle = color;
          ctx.lineWidth = 3;
          ctx.strokeRect(x, y, w, h);

          // Semi-transparent background
          ctx.fillStyle = `${color}25`;
          ctx.fillRect(x, y, w, h);

          // Label chip
          const confScore = Math.round((det.confidence || 1.0) * 100);
          const labelText = `${className} (${confScore}%)`;
          ctx.font = 'bold 13px Inter, sans-serif';
          const textWidth = ctx.measureText(labelText).width;
          const badgeH = 22;
          const badgeW = textWidth + 16;

          ctx.fillStyle = color;
          ctx.fillRect(x, Math.max(0, y - badgeH), badgeW, badgeH);

          ctx.fillStyle = '#ffffff';
          ctx.fillText(labelText, x + 8, Math.max(15, y - 6));
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

        {/* สรุปผลการวิเคราะห์และนับจำนวนวัตถุในภาพ (Identification & Piece Count Summary) */}
        {inferenceResult && (
          <div
            style={{
              padding: '16px',
              backgroundColor: '#f8fafc',
              borderRadius: 'var(--radius-sm)',
              border: '1px solid var(--border-color)',
              display: 'flex',
              flexDirection: 'column',
              gap: '12px',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Sparkles size={16} color="var(--accent-primary)" />
              <h4 style={{ fontSize: '13px', fontWeight: 700, color: 'var(--text-primary)', margin: 0 }}>
                สรุปผลการวิเคราะห์และนับจำนวนวัตถุ
              </h4>
            </div>

            {/* Total Count & Categories Stats */}
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: '1fr 1fr',
                gap: '8px',
              }}
            >
              <div
                style={{
                  backgroundColor: '#ffffff',
                  padding: '10px 12px',
                  borderRadius: 'var(--radius-sm)',
                  border: '1px solid var(--border-color)',
                  textAlign: 'center',
                }}
              >
                <div style={{ fontSize: '11px', color: 'var(--text-secondary)', marginBottom: '2px' }}>
                  วัตถุทั้งหมดในภาพ
                </div>
                <div
                  style={{
                    fontSize: '22px',
                    fontWeight: 800,
                    color: totalCount > 0 ? 'var(--accent-primary)' : 'var(--text-muted)',
                  }}
                >
                  {totalCount} <span style={{ fontSize: '13px', fontWeight: 600 }}>ชิ้น</span>
                </div>
              </div>

              <div
                style={{
                  backgroundColor: '#ffffff',
                  padding: '10px 12px',
                  borderRadius: 'var(--radius-sm)',
                  border: '1px solid var(--border-color)',
                  textAlign: 'center',
                }}
              >
                <div style={{ fontSize: '11px', color: 'var(--text-secondary)', marginBottom: '2px' }}>
                  ประเภทที่พบ
                </div>
                <div
                  style={{
                    fontSize: '22px',
                    fontWeight: 800,
                    color: breakdownList.length > 0 ? '#10b981' : 'var(--text-muted)',
                  }}
                >
                  {breakdownList.length} <span style={{ fontSize: '13px', fontWeight: 600 }}>ชนิด</span>
                </div>
              </div>
            </div>

            {/* Detailed Object Identification & Quantity List */}
            <div>
              <div style={{ fontSize: '11px', fontWeight: 700, color: 'var(--text-secondary)', marginBottom: '8px' }}>
                ในภาพคืออะไร และมีกี่ชิ้น:
              </div>

              {breakdownList.length === 0 ? (
                <div
                  style={{
                    fontSize: '12px',
                    color: 'var(--text-muted)',
                    textAlign: 'center',
                    padding: '12px 0',
                    lineHeight: 1.5,
                  }}
                >
                  ไม่พบวัตถุตามเกณฑ์ความมั่นใจที่กำหนด (0 ชิ้น)
                  <br />
                  <span style={{ fontSize: '11px' }}>ลองลดค่า Confidence หรือเลือกโมเดลที่ผ่านการเทรน</span>
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                  {breakdownList.map((item) => {
                    const col = getClassColor(item.name, uniqueClassNames);
                    const avgConf = Math.round(
                      (item.scores.reduce((a, b) => a + b, 0) / item.scores.length) * 100
                    );
                    return (
                      <div
                        key={item.name}
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'space-between',
                          padding: '8px 12px',
                          backgroundColor: '#ffffff',
                          borderRadius: 'var(--radius-sm)',
                          border: `1px solid ${col}44`,
                        }}
                      >
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <span
                            style={{
                              width: '10px',
                              height: '10px',
                              borderRadius: '50%',
                              backgroundColor: col,
                              flexShrink: 0,
                            }}
                          />
                          <div>
                            <div style={{ fontSize: '12px', fontWeight: 700, color: 'var(--text-primary)' }}>
                              {item.name}
                            </div>
                            <div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>
                              ความแม่นยำเฉลี่ย: {avgConf}%
                            </div>
                          </div>
                        </div>

                        <div
                          style={{
                            padding: '3px 10px',
                            borderRadius: '12px',
                            backgroundColor: `${col}15`,
                            color: col,
                            fontSize: '12px',
                            fontWeight: 700,
                            border: `1px solid ${col}33`,
                          }}
                        >
                          {item.count} ชิ้น
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>

            {/* Model & Latency Telemetry */}
            <div
              style={{
                paddingTop: '10px',
                borderTop: '1px solid var(--border-color)',
                display: 'flex',
                justifyContent: 'space-between',
                fontSize: '11px',
                color: 'var(--text-muted)',
              }}
            >
              <span>โมเดล: <strong>{inferenceResult.model_name || 'KDel 4.0'}</strong></span>
              <span>ความเร็ว: <strong style={{ color: '#10b981' }}>{inferenceResult.inference_time_ms || 0} ms</strong></span>
            </div>
          </div>
        )}
      </div>

      {/* Visual Canvas Area */}
      <div
        className="card"
        style={{
          display: 'flex',
          flexDirection: 'column',
          minHeight: '480px',
          overflow: 'hidden',
          backgroundColor: '#f8fafc',
          padding: '16px',
        }}
      >
        {/* Inference Overview Banner Above Canvas */}
        {inferenceResult && (
          <div
            style={{
              padding: '10px 14px',
              backgroundColor: '#ffffff',
              borderRadius: 'var(--radius-sm)',
              border: '1px solid var(--border-color)',
              marginBottom: '12px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              flexWrap: 'wrap',
              gap: '8px',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
              <span style={{ fontSize: '13px', fontWeight: 700, color: 'var(--text-primary)' }}>
                ผลการตรวจจับ: พบทั้งหมด <strong>{totalCount} ชิ้น</strong>
              </span>
              {breakdownList.map((b) => {
                const col = getClassColor(b.name, uniqueClassNames);
                return (
                  <span
                    key={b.name}
                    style={{
                      padding: '2px 8px',
                      borderRadius: '12px',
                      backgroundColor: `${col}15`,
                      color: col,
                      fontSize: '11px',
                      fontWeight: 600,
                      border: `1px solid ${col}33`,
                    }}
                  >
                    {b.name}: <strong>{b.count} ชิ้น</strong>
                  </span>
                );
              })}
            </div>

            <div style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>
              ประมวลผลใน <strong>{inferenceResult.inference_time_ms || 0} ms</strong> ด้วย {inferenceResult.model_name || 'KDel 4.0'}
            </div>
          </div>
        )}

        {/* Canvas Display Viewport */}
        <div
          style={{
            flex: 1,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            overflow: 'auto',
          }}
        >
          {testImagePreview ? (
            <div
              style={{
                maxWidth: '100%',
                maxHeight: '100%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <canvas
                ref={canvasRef}
                style={{
                  maxWidth: '100%',
                  maxHeight: 'calc(100vh - 220px)',
                  borderRadius: 'var(--radius-sm)',
                  boxShadow: '0 4px 20px rgba(0, 0, 0, 0.08)',
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
    </div>
  );
}
