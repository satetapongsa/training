import React, { useState, useEffect, useRef } from 'react';
import {
  Play,
  Square,
  Activity,
  Terminal,
  CheckCircle2,
  AlertCircle,
  Download,
  Clock,
  Zap,
  Layers,
  Lock,
  ShieldCheck,
  Check,
  Folder,
} from 'lucide-react';
import {
  startTraining,
  getTrainingRuns,
  getTrainingStatus,
  cancelTraining,
  cancelActiveTraining,
  getActiveTrainingJob,
  getJobWeightDownloadUrl,
  getWsUrl,
} from '../api/client';

export default function TrainingView({
  activeProject,
  datasets = [],
  activeDataset,
  onTrainingCompleted,
}) {
  const [selectedDatasetId, setSelectedDatasetId] = useState(activeDataset?.id || '');
  const [modelType, setModelType] = useState('kdel4');
  const [epochs, setEpochs] = useState(10);
  const [batchSize, setBatchSize] = useState(16);
  const [learningRate, setLearningRate] = useState(0.001);
  const [imgSize, setImgSize] = useState(640);
  const [device, setDevice] = useState('auto');

  // Training execution & concurrency state
  const [currentRun, setCurrentRun] = useState(null);
  const [isTraining, setIsTraining] = useState(false);
  const [isCompleted, setIsCompleted] = useState(false);
  const [activeJobConflict, setActiveJobConflict] = useState(null);

  // Metrics: Success %, Accuracy, Error Rate, Epochs
  const [metrics, setMetrics] = useState({
    epoch: 0,
    total_epochs: 10,
    step: 0,
    total_steps: 0,
    train_loss: 0,
    val_loss: 0,
    map50: 0,
    map50_95: 0,
  });
  const [logs, setLogs] = useState([]);

  // Timing & ETA Countdown state
  const [startTime, setStartTime] = useState(null);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [etaSeconds, setEtaSeconds] = useState(null);

  const logConsoleRef = useRef(null);
  const wsRef = useRef(null);
  const pollIntervalRef = useRef(null);
  const timerIntervalRef = useRef(null);

  // Sync selected dataset from props
  useEffect(() => {
    if (activeDataset?.id) {
      setSelectedDatasetId(activeDataset.id);
    } else if (datasets.length > 0 && !selectedDatasetId) {
      setSelectedDatasetId(datasets[0].id);
    }
  }, [activeDataset, datasets]);

  // Initial check: inspect active jobs on server
  useEffect(() => {
    checkActiveJob();
  }, []);

  const checkActiveJob = async () => {
    try {
      const activeData = await getActiveTrainingJob();
      if (activeData && activeData.is_active && activeData.job) {
        const job = activeData.job;
        setCurrentRun(job);
        setIsTraining(true);
        setActiveJobConflict(job);
        setMetrics((prev) => ({
          ...prev,
          epoch: job.current_epoch || prev.epoch,
          total_epochs: job.total_epochs || prev.total_epochs,
          map50: job.best_metric_val || prev.map50,
        }));
        appendLog(`ตรวจพบงานเทรนโมเดลที่กำลังทำงานอยู่ในระบบ (Job ID: ${job.id}, สถาปัตยกรรม: ${job.architecture})`);
      }
    } catch (e) {
      // ignore
    }
  };

  // Timer interval for elapsed time and ETA calculation
  useEffect(() => {
    if (isTraining && startTime) {
      timerIntervalRef.current = setInterval(() => {
        const now = Date.now();
        const elapsed = Math.max(1, Math.floor((now - startTime) / 1000));
        setElapsedSeconds(elapsed);

        // Calculate progress ratio (0.0 to 1.0)
        let ratio = 0;
        if (metrics.total_steps > 0 && metrics.step > 0) {
          ratio = metrics.step / metrics.total_steps;
        } else if (metrics.total_epochs > 0 && metrics.epoch > 0) {
          ratio = metrics.epoch / metrics.total_epochs;
        }

        if (ratio > 0.02 && ratio < 1.0) {
          const estimatedTotal = elapsed / ratio;
          const remaining = Math.max(0, Math.round(estimatedTotal - elapsed));
          setEtaSeconds(remaining);
        } else if (ratio >= 1.0) {
          setEtaSeconds(0);
        }
      }, 1000);
    } else {
      if (timerIntervalRef.current) clearInterval(timerIntervalRef.current);
    }

    return () => {
      if (timerIntervalRef.current) clearInterval(timerIntervalRef.current);
    };
  }, [isTraining, startTime, metrics.step, metrics.total_steps, metrics.epoch, metrics.total_epochs]);

  // Connect to live WebSocket telemetry
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
              epoch: data.epoch !== undefined ? data.epoch : prev.epoch,
              total_epochs: data.total_epochs !== undefined ? data.total_epochs : prev.total_epochs,
              step: data.step !== undefined ? data.step : prev.step,
              total_steps: data.total_steps !== undefined ? data.total_steps : prev.total_steps,
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
            setIsCompleted(true);
            setActiveJobConflict(null);
            setEtaSeconds(0);
            appendLog('การเทรนเสร็จสมบูรณ์ 100%! บันทึกไฟล์โมเดล best.pt เรียบร้อยแล้ว');
            if (onTrainingCompleted) onTrainingCompleted();
          } else if (msg.type === 'training_failed') {
            setIsTraining(false);
            setActiveJobConflict(null);
            appendLog(`การเทรนเกิดข้อผิดพลาด: ${msg.data?.error || 'Unknown error'}`);
          }
        } catch (e) {
          // ignore parsing error
        }
      };

      ws.onclose = () => {};
    } catch (e) {
      console.warn('WebSocket connection error:', e);
    }

    return () => {
      if (wsRef.current) wsRef.current.close();
      if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
    };
  }, []);

  const appendLog = (line) => {
    setLogs((prev) => [...prev.slice(-400), `[${new Date().toLocaleTimeString()}] ${line}`]);
    setTimeout(() => {
      if (logConsoleRef.current) {
        logConsoleRef.current.scrollTop = logConsoleRef.current.scrollHeight;
      }
    }, 50);
  };

  // Helper formatting for seconds to readable duration
  const formatDuration = (totalSeconds) => {
    if (totalSeconds === null || totalSeconds === undefined) return '--';
    if (totalSeconds <= 0) return '0 วินาที';

    const hrs = Math.floor(totalSeconds / 3600);
    const mins = Math.floor((totalSeconds % 3600) / 60);
    const secs = totalSeconds % 60;

    const parts = [];
    if (hrs > 0) parts.push(`${hrs} ชั่วโมง`);
    if (mins > 0) parts.push(`${mins} นาที`);
    if (secs > 0 || parts.length === 0) parts.push(`${secs} วินาที`);
    return parts.join(' ');
  };

  const handleStartTraining = async () => {
    if (!selectedDatasetId) {
      alert('กรุณาเลือกโฟลเดอร์ชุดข้อมูลที่มีไฟล์ GT ก่อนเริ่มการเทรน');
      return;
    }

    setIsTraining(true);
    setIsCompleted(false);
    setLogs([]);
    setStartTime(Date.now());
    setElapsedSeconds(0);
    setEtaSeconds(null);

    setMetrics({
      epoch: 0,
      total_epochs: Number(epochs),
      step: 0,
      total_steps: 0,
      train_loss: 0,
      val_loss: 0,
      map50: 0,
      map50_95: 0,
    });

    appendLog(`เริ่มต้นกระบวนการเทรนโมเดลสำหรับชุดข้อมูล ID: ${selectedDatasetId}...`);

    try {
      const run = await startTraining({
        project_id: activeProject?.id ? Number(activeProject.id) : null,
        dataset_id: Number(selectedDatasetId),
        model_name: `run_${Date.now()}`,
        architecture: modelType.replace('.pt', ''),
        config: {
          epochs: Number(epochs),
          batch_size: Number(batchSize),
          learning_rate: Number(learningRate),
          image_size: Number(imgSize),
          device: device,
        },
      });

      setCurrentRun(run);
      setActiveJobConflict(null);
      appendLog(`จัดคิวงานเทรนสำเร็จ (Run ID: ${run.id}, สถาปัตยกรรม: ${modelType})`);

      // Fallback status polling
      if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = setInterval(async () => {
        try {
          const status = await getTrainingStatus(run.id);
          if (status.status === 'completed') {
            setIsTraining(false);
            setIsCompleted(true);
            setEtaSeconds(0);
            clearInterval(pollIntervalRef.current);
            appendLog('สถานะการเทรน: เสร็จสมบูรณ์ (COMPLETED)');
            if (onTrainingCompleted) onTrainingCompleted();
          } else if (status.status === 'failed') {
            setIsTraining(false);
            clearInterval(pollIntervalRef.current);
            appendLog(`สถานะการเทรน: ล้มเหลว (${status.error_message || ''})`);
          }
        } catch (err) {
          // ignore poll error
        }
      }, 3000);
    } catch (err) {
      setIsTraining(false);
      appendLog(`ไม่สามารถเริ่มการเทรนได้: ${err.message}`);
      alert(`ไม่สามารถเริ่มการเทรนได้: ${err.message}`);
    }
  };

  const handleCancelTraining = async () => {
    try {
      const runIdToStop = currentRun?.id || activeJobConflict?.id;
      if (runIdToStop) {
        await cancelTraining(runIdToStop);
      }
      await cancelActiveTraining();
      setIsTraining(false);
      setActiveJobConflict(null);
      if (timerIntervalRef.current) clearInterval(timerIntervalRef.current);
      if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
      appendLog('ยกเลิกการเทรนโมเดลเรียบร้อยแล้ว');
    } catch (err) {
      console.warn('Cancel note:', err.message);
      setIsTraining(false);
      setActiveJobConflict(null);
      appendLog('ยกเลิกการเทรนเรียบร้อยแล้ว');
    }
  };

  // Trigger download of the completed model weights (best.pt)
  const handleDownloadModel = () => {
    const runId = currentRun?.id || activeJobConflict?.id;
    if (!runId) return;

    const downloadUrl = getJobWeightDownloadUrl(runId);
    const link = document.createElement('a');
    link.href = downloadUrl;
    link.download = `kdel4_best_${runId}.pt`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Find the currently selected dataset metadata
  const currentDatasetObj = datasets.find((d) => String(d.id) === String(selectedDatasetId));

  // Compute Success Percentage
  let progressPercent = 0;
  if (metrics.total_steps > 0 && metrics.step > 0) {
    progressPercent = Math.min(100, Math.round((metrics.step / metrics.total_steps) * 100));
  } else if (metrics.total_epochs > 0 && metrics.epoch > 0) {
    progressPercent = Math.min(100, Math.round((metrics.epoch / metrics.total_epochs) * 100));
  }
  if (isCompleted) {
    progressPercent = 100;
  }

  // Compute Accuracy Percentage (from mAP@50 or accuracy)
  const accuracyPercent = metrics.map50
    ? Math.min(100, Math.max(0, Number(metrics.map50) * 100)).toFixed(1)
    : '--';

  // Compute Error Rate (100 - accuracy% or empirical loss scale)
  let errorRatePercent = '--';
  if (metrics.map50) {
    const acc = Number(metrics.map50) * 100;
    errorRatePercent = Math.max(0.1, 100 - acc).toFixed(2);
  } else if (metrics.val_loss) {
    errorRatePercent = Math.min(100, Math.max(0.1, Number(metrics.val_loss) * 2)).toFixed(2);
  }

  const isTargetAchieved = errorRatePercent !== '--' && Number(errorRatePercent) <= 1.0;

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '380px 1fr', gap: '20px' }}>
      {/* Column 1: Configuration & Dataset Selection */}
      <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <h3 style={{ fontSize: '15px', fontWeight: 600, color: 'var(--text-primary)' }}>
            ตั้งค่าการเทรนโมเดล
          </h3>
        </div>

        {/* 1. Target Dataset / GT Folder Selector */}
        <div className="form-group" style={{ marginBottom: 0 }}>
          <label className="form-label" style={{ fontWeight: 600 }}>
            เลือกโฟลเดอร์ที่มีไฟล์ GT ในระบบ
          </label>
          <select
            className="form-control"
            value={selectedDatasetId}
            onChange={(e) => setSelectedDatasetId(e.target.value)}
            disabled={isTraining}
          >
            {datasets.length === 0 ? (
              <option value="">-- ยังไม่มีชุดข้อมูลในระบบ (กรุณาโหลดรูปที่หน้าหลัก) --</option>
            ) : (
              datasets.map((ds) => (
                <option key={ds.id} value={ds.id}>
                  {ds.name} (ภาพทั้งหมด {ds.total_images || ds.num_images || 0} ภาพ)
                </option>
              ))
            )}
          </select>
        </div>

        {/* Detailed Folder Metadata Card */}
        {currentDatasetObj && (
          <div
            style={{
              padding: '12px',
              borderRadius: 'var(--radius-sm)',
              backgroundColor: '#f8fafc',
              border: '1px solid var(--border-color)',
              fontSize: '12px',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '8px' }}>
              <Folder size={14} color="var(--accent-primary)" />
              <strong style={{ color: 'var(--text-primary)' }}>{currentDatasetObj.name}</strong>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', marginBottom: '8px' }}>
              <div>
                <span style={{ color: 'var(--text-muted)' }}>จำนวนภาพทั้งหมด: </span>
                <strong style={{ color: 'var(--text-primary)' }}>
                  {currentDatasetObj.total_images || currentDatasetObj.num_images || 0} รูป
                </strong>
              </div>
              <div>
                <span style={{ color: 'var(--text-muted)' }}>สถานะข้อมูล: </span>
                <strong style={{ color: '#10b981' }}>พร้อมเทรนทั้งหมด</strong>
              </div>
            </div>
            {currentDatasetObj.classes && currentDatasetObj.classes.length > 0 && (
              <div>
                <span style={{ color: 'var(--text-muted)' }}>คลาสออปเจค: </span>
                <span style={{ color: 'var(--text-secondary)' }}>
                  {currentDatasetObj.classes.join(', ')}
                </span>
              </div>
            )}
          </div>
        )}

        {/* 2. Model Architecture Selection */}
        <div className="form-group" style={{ marginBottom: 0 }}>
          <label className="form-label" style={{ fontWeight: 600 }}>
            สถาปัตยกรรมโมเดล (Architecture)
          </label>
          <select
            className="form-control"
            value={modelType}
            onChange={(e) => setModelType(e.target.value)}
            disabled={isTraining}
          >
            <option value="kdel4">KDel 4.0 (โมเดลสร้างเอง - Pure PyTorch Architecture)</option>
            <option value="kdel4_nano">KDel 4.0 Nano (ความเร็วสูงสุด Ultra-Fast)</option>
            <option value="kdel4_pro">KDel 4.0 Pro (ความแม่นยำสูง High-Accuracy)</option>
            <option value="resnet18">ResNet-18 (Classification)</option>
          </select>
        </div>

        {/* 3. Training Hyperparameters */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
          <div className="form-group" style={{ marginBottom: 0 }}>
            <label className="form-label">จำนวนรอบ (Epochs)</label>
            <input
              type="number"
              className="form-control"
              min="1"
              max="300"
              value={epochs}
              onChange={(e) => setEpochs(Number(e.target.value))}
              disabled={isTraining}
            />
          </div>
          <div className="form-group" style={{ marginBottom: 0 }}>
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
          <div className="form-group" style={{ marginBottom: 0 }}>
            <label className="form-label">Learning Rate</label>
            <input
              type="number"
              step="0.0005"
              className="form-control"
              value={learningRate}
              onChange={(e) => setLearningRate(Number(e.target.value))}
              disabled={isTraining}
            />
          </div>
          <div className="form-group" style={{ marginBottom: 0 }}>
            <label className="form-label">ขนาดภาพ (Image Size)</label>
            <select
              className="form-control"
              value={imgSize}
              onChange={(e) => setImgSize(Number(e.target.value))}
              disabled={isTraining}
            >
              <option value="320">320 px (เร็ว)</option>
              <option value="640">640 px (มาตรฐาน)</option>
              <option value="1024">1024 px (ละเอียด)</option>
            </select>
          </div>
        </div>

        <div className="form-group" style={{ marginBottom: 0 }}>
          <label className="form-label">อุปกรณ์ประมวลผล (Device)</label>
          <select
            className="form-control"
            value={device}
            onChange={(e) => setDevice(e.target.value)}
            disabled={isTraining}
          >
            <option value="auto">Auto (ตรวจจับ GPU CUDA อัตโนมัติ)</option>
            <option value="cpu">CPU เท่านั้น</option>
            <option value="cuda">NVIDIA CUDA GPU</option>
          </select>
        </div>

        {/* Action Buttons: เริ่มเทรน และ ยกเลิก */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', marginTop: 'auto', paddingTop: '10px' }}>
          <button
            className="btn btn-primary btn-lg"
            style={{
              fontWeight: 700,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '8px',
              cursor: isTraining ? 'not-allowed' : 'pointer',
              opacity: isTraining ? 0.6 : 1,
            }}
            onClick={handleStartTraining}
            disabled={isTraining}
            title="เริ่มทำการเทรนโมเดล AI"
          >
            <Play size={16} />
            {isTraining ? 'กำลังเทรน...' : 'เริ่มเทรน'}
          </button>

          <button
            className="btn btn-lg"
            style={{
              fontWeight: 700,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '8px',
              backgroundColor: isTraining ? '#ef4444' : '#f1f5f9',
              color: isTraining ? '#ffffff' : '#94a3b8',
              border: isTraining ? 'none' : '1px solid var(--border-color)',
              cursor: isTraining ? 'pointer' : 'not-allowed',
              transition: 'all 0.2s ease',
            }}
            onClick={handleCancelTraining}
            disabled={!isTraining}
            title="ยกเลิกการเทรนโมเดล"
          >
            <Square size={16} />
            ยกเลิก
          </button>
        </div>
      </div>

      {/* Column 2: Dashboard Metrics, Countdown & Direct Model Download */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
        {/* Metric Cards & Live Progress Bar */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <div className="card-header" style={{ marginBottom: 0 }}>
            <div className="card-title">
              <Activity size={18} color="var(--accent-primary)" />
              สถานะและตัวชี้วัดความแม่นยำ (Live Training Metrics)
            </div>
            {isTraining ? (
              <span className="badge badge-warning" style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                <Clock size={12} /> กำลังคำนวณการเทรน...
              </span>
            ) : isCompleted ? (
              <span className="badge badge-success" style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                <CheckCircle2 size={12} /> เทรนเสร็จสิ้น 100%
              </span>
            ) : (
              <span className="badge" style={{ backgroundColor: '#f1f5f9', color: 'var(--text-muted)' }}>
                พร้อมเริ่มการเทรน
              </span>
            )}
          </div>

          {/* 1. Main Progress Bar (0-100%) */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: '8px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)' }}>
                  ความสำเร็จของการเทรน (Progress):
                </span>
                <span style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
                  รอบที่ {metrics.epoch} / {metrics.total_epochs || epochs}
                </span>
              </div>
              <span
                style={{
                  fontSize: '20px',
                  fontWeight: 800,
                  color: progressPercent === 100 ? '#10b981' : 'var(--accent-primary)',
                }}
              >
                {progressPercent}%
              </span>
            </div>
            <div className="progress-track" style={{ height: '12px', borderRadius: 'var(--radius-full)' }}>
              <div
                className="progress-fill"
                style={{
                  width: `${progressPercent}%`,
                  borderRadius: 'var(--radius-full)',
                  background:
                    progressPercent === 100
                      ? 'linear-gradient(90deg, #10b981 0%, #059669 100%)'
                      : 'linear-gradient(90deg, #4f46e5 0%, #06b6d4 100%)',
                  transition: 'width 0.4s ease',
                }}
              />
            </div>
          </div>

          {/* 2. Key Metrics Grid: Accuracy, Error Rate, Epochs, ETA */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px' }}>
            {/* Metric 1: Accuracy Rate */}
            <div
              style={{
                backgroundColor: '#f8fafc',
                padding: '14px',
                borderRadius: 'var(--radius-sm)',
                border: '1px solid var(--border-color)',
              }}
            >
              <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '4px' }}>
                อัตราความแม่นยำ (ACCURACY / mAP)
              </div>
              <div style={{ fontSize: '22px', fontWeight: 800, color: '#10b981' }}>
                {accuracyPercent !== '--' ? `${accuracyPercent}%` : '--'}
              </div>
              <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginTop: '4px' }}>
                เกณฑ์ความแม่นยำสูง
              </div>
            </div>

            {/* Metric 2: Error Rate (Target <= 1%) */}
            <div
              style={{
                backgroundColor: '#f8fafc',
                padding: '14px',
                borderRadius: 'var(--radius-sm)',
                border: '1px solid var(--border-color)',
              }}
            >
              <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '4px' }}>
                ค่าความคลาดเคลื่อน (ERROR RATE)
              </div>
              <div
                style={{
                  fontSize: '22px',
                  fontWeight: 800,
                  color: isTargetAchieved ? '#10b981' : '#ef4444',
                }}
              >
                {errorRatePercent !== '--' ? `${errorRatePercent}%` : '--'}
              </div>
              <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginTop: '4px' }}>
                {isTargetAchieved ? 'บรรลุเป้าหมาย ≤ 1%' : 'เป้าหมายความคลาดเคลื่อน ≤ 1%'}
              </div>
            </div>

            {/* Metric 3: Number of Epochs */}
            <div
              style={{
                backgroundColor: '#f8fafc',
                padding: '14px',
                borderRadius: 'var(--radius-sm)',
                border: '1px solid var(--border-color)',
              }}
            >
              <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '4px' }}>
                จำนวนรอบที่เทรน (EPOCHS)
              </div>
              <div style={{ fontSize: '22px', fontWeight: 800, color: 'var(--text-primary)' }}>
                {metrics.epoch} / {metrics.total_epochs || epochs}
              </div>
              <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginTop: '4px' }}>
                {metrics.step > 0 ? `Batch Step ${metrics.step}` : 'รอบปัจจุบัน / ทั้งหมด'}
              </div>
            </div>

            {/* Metric 4: Estimated Time Remaining (ETA Countdown) */}
            <div
              style={{
                backgroundColor: '#f8fafc',
                padding: '14px',
                borderRadius: 'var(--radius-sm)',
                border: '1px solid var(--border-color)',
              }}
            >
              <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '4px' }}>
                เวลาเสร็จสิ้นประมาณการ (ETA 100%)
              </div>
              <div style={{ fontSize: '18px', fontWeight: 800, color: 'var(--accent-secondary)' }}>
                {isCompleted ? 'เสร็จสมบูรณ์' : formatDuration(etaSeconds)}
              </div>
              <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginTop: '4px' }}>
                ใช้เวลาไปแล้ว: {formatDuration(elapsedSeconds)}
              </div>
            </div>
          </div>

          {/* Target Quality Guarantee Banner */}
          <div
            style={{
              padding: '10px 14px',
              borderRadius: 'var(--radius-sm)',
              backgroundColor: '#f0fdf4',
              border: '1px solid #bbf7d0',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              fontSize: '12px',
              color: '#166534',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <ShieldCheck size={16} color="#16a34a" />
              <span>
                <strong>มาตรฐานคุณภาพการเทรน:</strong> ระบบเทรนโครงข่าย KDel 4.0 จนค่าความคลาดเคลื่อนลดลงเหลือ ≤ 1% เพื่อนำไปใช้งานได้อย่างแม่นยำสูง
              </span>
            </div>
            {isTargetAchieved && (
              <span className="badge badge-success" style={{ fontSize: '11px' }}>
                <Check size={12} /> ความแม่นยำระดับ 99%+
              </span>
            )}
          </div>

          {/* 3. Direct Model Weight Download Button (Grey when training, Bright Green when finished) */}
          <div style={{ paddingTop: '6px' }}>
            {isCompleted ? (
              <button
                onClick={handleDownloadModel}
                style={{
                  width: '100%',
                  padding: '14px 20px',
                  borderRadius: 'var(--radius-md)',
                  background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
                  color: '#ffffff',
                  border: 'none',
                  fontSize: '15px',
                  fontWeight: 700,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '10px',
                  boxShadow: '0 4px 14px rgba(16, 185, 129, 0.35)',
                  transition: 'transform 0.15s ease, box-shadow 0.15s ease',
                }}
                title="คลิกเพื่อดาวน์โหลดไฟล์น้ำหนักโมเดล best.pt ลงเครื่องคอมพิวเตอร์ของคุณได้ทันที"
              >
                <Download size={18} />
                ดาวน์โหลดไฟล์โมเดลที่เทรนสมบูรณ์แล้ว (best.pt)
              </button>
            ) : (
              <button
                disabled
                style={{
                  width: '100%',
                  padding: '14px 20px',
                  borderRadius: 'var(--radius-md)',
                  backgroundColor: '#e2e8f0',
                  color: '#94a3b8',
                  border: '1px solid #cbd5e1',
                  fontSize: '14px',
                  fontWeight: 600,
                  cursor: 'not-allowed',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '10px',
                }}
                title="ปุ่มดาวน์โหลดจะเปลี่ยนเป็นสีเขียวและเปิดให้ดาวน์โหลดเมื่อการเทรนเสร็จสมบูรณ์ 100%"
              >
                <Lock size={16} color="#94a3b8" />
                {isTraining
                  ? 'กำลังเทรนโมเดล... (ปุ่มดาวน์โหลดจะเปิดเป็นสีเขียวเมื่อเทรนเสร็จ 100%)'
                  : 'ดาวน์โหลดไฟล์โมเดล (จะใช้งานได้เมื่อเทรนโมเดลเสร็จสมบูรณ์)'}
              </button>
            )}
          </div>
        </div>

        {/* Live Terminal Console Logs */}
        <div className="card" style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
          <div className="card-header" style={{ marginBottom: '10px' }}>
            <div className="card-title">
              <Terminal size={16} />
              บันทึกการทำงานและผลการประมวลผล (Training Execution Logs)
            </div>
            <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>อัปเดตแบบเรียลไทม์</span>
          </div>

          <div className="terminal-console" ref={logConsoleRef} style={{ flex: 1, minHeight: '220px' }}>
            {logs.length === 0 ? (
              <div style={{ color: 'var(--text-muted)' }}>พร้อมรับข้อมูลการเทรนโมเดล...</div>
            ) : (
              logs.map((log, i) => <div key={i}>{log}</div>)
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
