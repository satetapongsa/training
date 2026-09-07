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
  const [detectionProgress, setDetectionProgress] = useState(0);
  const [detectionStage, setDetectionStage] = useState('');

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
    setDetectionProgress(0);
    setDetectionStage('');
  };

  const handleRunInference = async () => {
    if (!testImageFile) {
      alert('กรุณาเลือกไฟล์ภาพสำหรับทดสอบก่อนกดตรวจจับ');
      return;
    }

    setLoading(true);
    setDetectionProgress(15);
    setDetectionStage('โหลดไฟล์ภาพและสเกลขนาดภาพ (Preprocessing & Normalization)...');

    let currentProg = 15;
    const progressInterval = setInterval(() => {
      currentProg += Math.floor(Math.random() * 8) + 5;
      if (currentProg > 92) {
        currentProg = 92;
      }
      setDetectionProgress(currentProg);

      if (currentProg < 35) {
        setDetectionStage('โหลดไฟล์ภาพและสเกลขนาดภาพ (Preprocessing & Normalization)');
      } else if (currentProg < 65) {
        setDetectionStage('รันโมเดลโครงข่ายประสาทเทียม AI (Deep Vision Neural Network Forward Pass)');
      } else if (currentProg < 85) {
        setDetectionStage('สกัดฟีเจอร์และจับคู่คลาสวัตถุที่เทรนไว้ (Feature Extraction & Class Matching)');
      } else {
        setDetectionStage('คำนวณตำแหน่งตีกรอบและกรอง NMS (Bounding Box & NMS Filtering)');
      }
    }, 90);

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
      clearInterval(progressInterval);
      setDetectionProgress(100);
      setDetectionStage('ตรวจจับเสร็จสิ้น 100% กำลังแสดงผลกรอบวัตถุบนภาพ');
      setInferenceResult(result);
    } catch (err) {
      clearInterval(progressInterval);
      setDetectionProgress(0);
      setDetectionStage('');
      alert(`การตรวจจับล้มเหลว: ${err.message}`);
    } finally {
      setTimeout(() => {
        setLoading(false);
      }, 350);
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

          // 1. Outer heavy dark halo stroke for 100% contrast on any surface
          ctx.strokeStyle = '#090d16';
          ctx.lineWidth = 5;
          ctx.lineCap = 'round';
          ctx.lineJoin = 'round';
          ctx.strokeRect(x, y, w, h);

          // 2. Vivid inner stroke
          ctx.strokeStyle = color;
          ctx.lineWidth = 3;
          ctx.strokeRect(x, y, w, h);

          // 3. Semi-transparent fill
          ctx.fillStyle = `${color}28`;
          ctx.fillRect(x, y, w, h);

          // 4. White Corner L-brackets for futuristic crisp detection
          const cornerLen = Math.min(22, Math.min(w, h) / 3);
          ctx.strokeStyle = '#ffffff';
          ctx.lineWidth = 2.5;
          ctx.lineCap = 'square';
          // Top-Left
          ctx.beginPath();
          ctx.moveTo(x, y + cornerLen);
          ctx.lineTo(x, y);
          ctx.lineTo(x + cornerLen, y);
          ctx.stroke();
          // Top-Right
          ctx.beginPath();
          ctx.moveTo(x + w - cornerLen, y);
          ctx.lineTo(x + w, y);
          ctx.lineTo(x + w, y + cornerLen);
          ctx.stroke();
          // Bottom-Left
          ctx.beginPath();
          ctx.moveTo(x, y + h - cornerLen);
          ctx.lineTo(x, y + h);
          ctx.lineTo(x + cornerLen, y + h);
          ctx.stroke();
          // Bottom-Right
          ctx.beginPath();
          ctx.moveTo(x + w - cornerLen, y + h);
          ctx.lineTo(x + w, y + h);
          ctx.lineTo(x + w, y + h - cornerLen);
          ctx.stroke();

          // 5. Object Identification Badge: ชื่อวัตถุ และ % ความแม่นยำ
          const confScore = Math.round((det.confidence || 1.0) * 100);
          const labelText = `วัตถุ: ${className} (${confScore}%)`;
          ctx.font = 'bold 13px Inter, sans-serif';
          const textWidth = ctx.measureText(labelText).width;
          const badgeH = 24;
          const badgeW = textWidth + 18;
          const badgeY = Math.max(0, y - badgeH);

          // Badge background
          ctx.fillStyle = color;
          ctx.fillRect(x, badgeY, badgeW, badgeH);
          ctx.strokeStyle = '#090d16';
          ctx.lineWidth = 1.5;
          ctx.strokeRect(x, badgeY, badgeW, badgeH);

          // Badge text
          ctx.fillStyle = '#ffffff';
          ctx.fillText(labelText, x + 8, badgeY + 16);
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
            <Zap size={16} /> {loading ? `กำลังตรวจจับ (${detectionProgress}%)...` : 'เริ่มตรวจจับภาพ (Run Inference)'}
          </button>

          {/* Working Detection Progress Card */}
          {loading && (
            <div
              style={{
                padding: '12px 14px',
                backgroundColor: '#eef2ff',
                borderRadius: 'var(--radius-sm)',
                border: '1.5px solid var(--accent-primary)',
                display: 'flex',
                flexDirection: 'column',
                gap: '8px',
                marginTop: '4px',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: '12px', fontWeight: 700, color: 'var(--accent-primary)' }}>
                  กำลังตรวจจับภาพ...
                </span>
                <span style={{ fontSize: '14px', fontWeight: 800, color: 'var(--accent-primary)' }}>
                  {detectionProgress}%
                </span>
              </div>
              <div style={{ height: '8px', backgroundColor: '#c7d2fe', borderRadius: '4px', overflow: 'hidden' }}>
                <div
                  style={{
                    width: `${detectionProgress}%`,
                    height: '100%',
                    backgroundColor: 'var(--accent-primary)',
                    borderRadius: '4px',
                    transition: 'width 0.12s ease',
                  }}
                />
              </div>
              <div style={{ fontSize: '11px', color: 'var(--text-secondary)', lineHeight: 1.4 }}>
                {detectionStage}
              </div>
            </div>
          )}
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
        {/* Working Progress Banner Above Canvas */}
        {loading && (
          <div
            className="card"
            style={{
              padding: '12px 16px',
              backgroundColor: '#ffffff',
              borderRadius: 'var(--radius-sm)',
              border: '1.5px solid var(--accent-primary)',
              marginBottom: '12px',
              display: 'flex',
              flexDirection: 'column',
              gap: '6px',
              boxShadow: '0 2px 10px rgba(79, 70, 229, 0.15)',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontSize: '13px', fontWeight: 700, color: 'var(--text-primary)' }}>
                กำลังประมวลผลการตรวจจับ: {detectionStage}
              </span>
              <span style={{ fontSize: '15px', fontWeight: 800, color: 'var(--accent-primary)' }}>
                {detectionProgress}%
              </span>
            </div>
            <div style={{ height: '10px', backgroundColor: '#e2e8f0', borderRadius: '5px', overflow: 'hidden' }}>
              <div
                style={{
                  width: `${detectionProgress}%`,
                  height: '100%',
                  background: 'linear-gradient(90deg, #4f46e5 0%, #10b981 100%)',
                  borderRadius: '5px',
                  transition: 'width 0.12s ease',
                }}
              />
            </div>
          </div>
        )}

        {/* Inference Overview Banner Above Canvas */}
        {inferenceResult && !loading && (
          <div
            style={{
              padding: '10px 14px',
              backgroundColor: totalCount > 0 ? '#ecfdf5' : '#ffffff',
              borderRadius: 'var(--radius-sm)',
              border: `1.5px solid ${totalCount > 0 ? '#10b981' : 'var(--border-color)'}`,
              marginBottom: '12px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              flexWrap: 'wrap',
              gap: '8px',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
              <CheckCircle2 size={16} color={totalCount > 0 ? '#10b981' : 'var(--text-muted)'} />
              <span style={{ fontSize: '13px', fontWeight: 700, color: totalCount > 0 ? '#065f46' : 'var(--text-primary)' }}>
                {totalCount > 0
                  ? `ตรวจพบวัตถุที่รู้จักตามที่เทรนไว้: ${totalCount} ชิ้น พร้อมตีกรอบระบุประเภทบนภาพแล้ว`
                  : `ไม่พบวัตถุตามเกณฑ์ความมั่นใจ (${Math.round(confThreshold * 100)}%)`}
              </span>
              {breakdownList.map((b) => {
                const col = getClassColor(b.name, uniqueClassNames);
                return (
                  <span
                    key={b.name}
                    style={{
                      padding: '2px 8px',
                      borderRadius: '12px',
                      backgroundColor: `${col}20`,
                      color: col,
                      fontSize: '11px',
                      fontWeight: 700,
                      border: `1px solid ${col}44`,
                    }}
                  >
                    วัตถุ {b.name}: {b.count} ชิ้น
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
            background: 'radial-gradient(circle, #334155 1.5px, transparent 1.5px) 0 0 / 22px 22px, #0b1329',
            borderRadius: '8px',
            border: '1.5px solid #1e293b',
            padding: '16px',
            boxShadow: 'inset 0 2px 10px rgba(0, 0, 0, 0.5)',
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
                  maxHeight: 'calc(100vh - 240px)',
                  borderRadius: '4px',
                  boxShadow: '0 8px 32px rgba(0, 0, 0, 0.55), 0 0 0 1px rgba(255, 255, 255, 0.15)',
                  display: 'block',
                  margin: 'auto',
                }}
              />
            </div>
          ) : (
            <div style={{ textAlign: 'center', color: '#94a3b8', padding: '40px 20px' }}>
              <Crosshair size={44} style={{ margin: '0 auto 12px auto', display: 'block', color: '#64748b' }} />
              <div style={{ fontSize: '14px', fontWeight: 600, color: '#f8fafc', marginBottom: '4px' }}>
                ยังไม่ได้เลือกรูปภาพทดสอบ
              </div>
              <div style={{ fontSize: '12px', color: '#94a3b8' }}>
                คลิกปุ่ม &ldquo;เลือกรูปภาพจากเครื่อง&rdquo; เพื่อนำภาพมาทดสอบการตรวจจับวัตถุด้วยโมเดล AI
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
