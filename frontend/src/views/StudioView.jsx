import React, { useState, useEffect, useRef } from 'react';
import {
  FolderUp,
  Upload,
  Plus,
  Play,
  Save,
  Trash2,
  Tag,
  ChevronLeft,
  ChevronRight,
  CheckCircle2,
  Sparkles,
  Layers,
  FileImage,
} from 'lucide-react';
import {
  getDatasets,
  createDataset,
  getDatasetImages,
  uploadFilesChunked,
  getAnnotations,
  saveAnnotations,
  API_BASE_URL,
} from '../api/client';

export default function StudioView({
  activeProject,
  activeDataset,
  setActiveDataset,
  onProceedToTraining,
}) {
  // Datasets & Images
  const [images, setImages] = useState([]);
  const [selectedImageIndex, setSelectedImageIndex] = useState(0);

  // Annotations on currently selected image
  const [annotations, setAnnotations] = useState([]);
  const [currentClass, setCurrentClass] = useState('object');
  const [classList, setClassList] = useState(['object', 'defect', 'item', 'person']);
  const [newClassName, setNewClassName] = useState('');

  // Canvas drawing state
  const [isDrawing, setIsDrawing] = useState(false);
  const [startPos, setStartPos] = useState({ x: 0, y: 0 });
  const [currentBox, setCurrentBox] = useState(null);
  const [saving, setSaving] = useState(false);

  // Upload state
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadStatus, setUploadStatus] = useState('');
  const [isDragOver, setIsDragOver] = useState(false);

  // DOM Refs
  const folderInputRef = useRef(null);
  const filesInputRef = useRef(null);
  const canvasRef = useRef(null);
  const imageObjRef = useRef(null);

  const selectedImage = images[selectedImageIndex] || null;

  // Load existing images if dataset is present
  useEffect(() => {
    if (activeDataset?.id) {
      loadImagesFromDataset(activeDataset.id);
    }
  }, [activeDataset]);

  // Load annotations when active image changes
  useEffect(() => {
    if (selectedImage) {
      if (selectedImage.annotations && selectedImage.annotations.length > 0) {
        // If image object already has loaded annotations
        const formatted = selectedImage.annotations.map((ann) => {
          const w = ann.bbox_w !== undefined ? ann.bbox_w : ann.x_max - ann.x_min;
          const h = ann.bbox_h !== undefined ? ann.bbox_h : ann.y_max - ann.y_min;
          const cx = ann.bbox_x !== undefined ? ann.bbox_x : ann.x_min + w / 2;
          const cy = ann.bbox_y !== undefined ? ann.bbox_y : ann.y_min + h / 2;
          return {
            id: ann.id || Math.random().toString(),
            label: ann.class_name || ann.label || 'object',
            x_min: Math.max(0, cx - w / 2),
            y_min: Math.max(0, cy - h / 2),
            x_max: Math.min(1, cx + w / 2),
            y_max: Math.min(1, cy + h / 2),
          };
        });
        setAnnotations(formatted);
      } else if (selectedImage.id) {
        // Fetch from backend
        getAnnotations(selectedImage.id)
          .then((data) => {
            const formatted = (data || []).map((ann) => ({
              id: ann.id || Math.random().toString(),
              label: ann.class_name || 'object',
              x_min: Math.max(0, ann.bbox_x - ann.bbox_w / 2),
              y_min: Math.max(0, ann.bbox_y - ann.bbox_h / 2),
              x_max: Math.min(1, ann.bbox_x + ann.bbox_w / 2),
              y_max: Math.min(1, ann.bbox_y + ann.bbox_h / 2),
            }));
            setAnnotations(formatted);
          })
          .catch(() => setAnnotations([]));
      } else {
        setAnnotations([]);
      }
    }
  }, [selectedImageIndex, selectedImage?.id]);

  const loadImagesFromDataset = async (datasetId) => {
    try {
      const data = await getDatasetImages(datasetId);
      if (Array.isArray(data) && data.length > 0) {
        setImages(data);
        setSelectedImageIndex(0);
      }
    } catch (e) {
      console.warn('Could not load images:', e);
    }
  };

  // --- LOCAL FILE & FOLDER INGESTION ---
  // Process dropped or selected files directly in the browser
  const handleIngestFiles = async (fileList) => {
    const rawFiles = Array.from(fileList);
    if (rawFiles.length === 0) return;

    // Filter image files and companion .txt label files
    const imageFiles = rawFiles.filter((f) =>
      /\.(jpe?g|png|webp|bmp|tiff?)$/i.test(f.name)
    );
    const labelFiles = rawFiles.filter((f) => /\.txt$/i.test(f.name));

    if (imageFiles.length === 0) {
      alert('No supported image files found in the selected folder.');
      return;
    }

    setUploading(true);
    setUploadProgress(10);
    setUploadStatus(`Reading ${imageFiles.length} images locally from computer...`);

    // Parse companion YOLO .txt labels client-side for immediate display
    const labelsMap = new Map();
    for (const lf of labelFiles) {
      try {
        const text = await lf.text();
        const baseName = lf.name.replace(/\.[^/.]+$/, '').toLowerCase();
        labelsMap.set(baseName, text);
      } catch (e) {
        // ignore parse error
      }
    }

    // Prepare local preview items immediately
    const localItems = imageFiles.map((file, idx) => {
      const localUrl = URL.createObjectURL(file);
      const baseName = file.name.replace(/\.[^/.]+$/, '').toLowerCase();
      const txtContent = labelsMap.get(baseName);
      const initialAnnots = [];

      if (txtContent) {
        const lines = txtContent.split('\n');
        for (const line of lines) {
          const parts = line.trim().split(/\s+/);
          if (parts.length >= 5) {
            const classIdx = parts[0];
            const cx = parseFloat(parts[1]);
            const cy = parseFloat(parts[2]);
            const w = parseFloat(parts[3]);
            const h = parseFloat(parts[4]);
            if (!isNaN(cx) && !isNaN(cy) && !isNaN(w) && !isNaN(h)) {
              initialAnnots.push({
                id: Math.random().toString(),
                label: `class_${classIdx}`,
                x_min: Math.max(0, cx - w / 2),
                y_min: Math.max(0, cy - h / 2),
                x_max: Math.min(1, cx + w / 2),
                y_max: Math.min(1, cy + h / 2),
              });
            }
          }
        }
      }

      return {
        id: null, // will be assigned after backend upload
        filename: file.name,
        original_name: file.name,
        localUrl: localUrl,
        fileHandle: file,
        annotations: initialAnnots,
        is_annotated: initialAnnots.length > 0,
      };
    });

    // Instantly set images on screen so user can see them immediately!
    setImages(localItems);
    setSelectedImageIndex(0);

    // Auto-ensure or create dataset on backend
    try {
      let targetDataset = activeDataset;
      if (!targetDataset) {
        // Detect folder name from webkitRelativePath
        const folderName =
          imageFiles[0].webkitRelativePath?.split('/')[0] ||
          `Folder_${new Date().toISOString().slice(0, 10)}`;
        setUploadStatus(`Creating training dataset "${folderName}"...`);
        targetDataset = await createDataset({
          project_id: activeProject?.id,
          name: folderName,
          description: `Ingested from local computer on ${new Date().toLocaleString()}`,
        });
        setActiveDataset(targetDataset);
      }

      // Upload all files (images + label txt) chunked to backend
      setUploadStatus(`Uploading ${rawFiles.length} files to server storage...`);
      await uploadFilesChunked(targetDataset.id, rawFiles, (uploaded, total, pct) => {
        setUploadProgress(pct);
        setUploadStatus(`Uploaded ${uploaded} / ${total} files (${pct}%)`);
      });

      // Reload fresh images with real database IDs
      const freshImages = await getDatasetImages(targetDataset.id);
      if (Array.isArray(freshImages) && freshImages.length > 0) {
        setImages(freshImages);
      }

      setUploadStatus('Folder successfully loaded & saved!');
      setTimeout(() => {
        setUploading(false);
        setUploadProgress(0);
        setUploadStatus('');
      }, 1500);
    } catch (err) {
      console.warn('Backend sync note:', err.message);
      setUploading(false);
      setUploadStatus('');
    }
  };

  // Drag and Drop Recursive Folder Traversal
  const scanFilesFromEntry = async (entry) => {
    return new Promise((resolve) => {
      if (entry.isFile) {
        entry.file((file) => resolve([file]), () => resolve([]));
      } else if (entry.isDirectory) {
        const dirReader = entry.createReader();
        const entries = [];
        const readEntries = () => {
          dirReader.readEntries(async (results) => {
            if (!results.length) {
              const fileArrays = await Promise.all(entries.map(scanFilesFromEntry));
              resolve(fileArrays.flat());
            } else {
              entries.push(...results);
              readEntries();
            }
          }, () => resolve([]));
        };
        readEntries();
      } else {
        resolve([]);
      }
    });
  };

  const handleDrop = async (e) => {
    e.preventDefault();
    setIsDragOver(false);

    const items = e.dataTransfer.items;
    if (!items || items.length === 0) return;

    const allFiles = [];
    for (let i = 0; i < items.length; i++) {
      const entry = items[i].webkitGetAsEntry ? items[i].webkitGetAsEntry() : null;
      if (entry) {
        const files = await scanFilesFromEntry(entry);
        allFiles.push(...files);
      } else if (items[i].kind === 'file') {
        const file = items[i].getAsFile();
        if (file) allFiles.push(file);
      }
    }

    if (allFiles.length > 0) {
      handleIngestFiles(allFiles);
    }
  };

  // --- CANVAS BOUNDING BOX DRAWING (ตีกรอบ) ---
  useEffect(() => {
    if (!selectedImage || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const imgUrl =
      selectedImage.localUrl ||
      (selectedImage.image_url
        ? `${API_BASE_URL}${selectedImage.image_url}`
        : `${API_BASE_URL}/api/v1/datasets/images/${selectedImage.id}/file`);

    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.src = imgUrl;
    img.onload = () => {
      imageObjRef.current = img;
      canvas.width = img.naturalWidth || 800;
      canvas.height = img.naturalHeight || 600;
      redraw(ctx, img);
    };
  }, [selectedImage, annotations, currentBox]);

  const redraw = (ctx, img) => {
    if (!ctx || !img) return;
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    ctx.drawImage(img, 0, 0);

    // Color palette for classes
    const colors = ['#6366f1', '#10b981', '#06b6d4', '#f59e0b', '#ec4899', '#8b5cf6'];

    // Draw saved annotations
    annotations.forEach((ann, idx) => {
      const color = colors[idx % colors.length];
      const x = ann.x_min * ctx.canvas.width;
      const y = ann.y_min * ctx.canvas.height;
      const w = (ann.x_max - ann.x_min) * ctx.canvas.width;
      const h = (ann.y_max - ann.y_min) * ctx.canvas.height;

      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.strokeRect(x, y, w, h);

      // Label background & text
      const labelText = ann.label || 'object';
      ctx.font = 'bold 13px Inter, sans-serif';
      const textWidth = ctx.measureText(labelText).width;

      ctx.fillStyle = color;
      ctx.fillRect(x, Math.max(0, y - 24), textWidth + 16, 24);

      ctx.fillStyle = '#ffffff';
      ctx.fillText(labelText, x + 8, Math.max(16, y - 7));
    });

    // Draw current dragging box
    if (currentBox) {
      ctx.strokeStyle = '#06b6d4';
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 6]);
      ctx.strokeRect(currentBox.x, currentBox.y, currentBox.w, currentBox.h);
      ctx.setLineDash([]);
    }
  };

  const getCanvasCoords = (e) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
    };
  };

  const handleMouseDown = (e) => {
    if (!imageObjRef.current) return;
    const coords = getCanvasCoords(e);
    setIsDrawing(true);
    setStartPos(coords);
    setCurrentBox({ x: coords.x, y: coords.y, w: 0, h: 0 });
  };

  const handleMouseMove = (e) => {
    if (!isDrawing) return;
    const coords = getCanvasCoords(e);
    const x = Math.min(startPos.x, coords.x);
    const y = Math.min(startPos.y, coords.y);
    const w = Math.abs(coords.x - startPos.x);
    const h = Math.abs(coords.y - startPos.y);
    setCurrentBox({ x, y, w, h });
  };

  const handleMouseUp = () => {
    if (!isDrawing || !currentBox || !canvasRef.current) {
      setIsDrawing(false);
      return;
    }
    setIsDrawing(false);

    if (currentBox.w > 10 && currentBox.h > 10) {
      const canvas = canvasRef.current;
      const x_min = Math.max(0, currentBox.x / canvas.width);
      const y_min = Math.max(0, currentBox.y / canvas.height);
      const x_max = Math.min(1, (currentBox.x + currentBox.w) / canvas.width);
      const y_max = Math.min(1, (currentBox.y + currentBox.h) / canvas.height);

      const newAnn = {
        id: Math.random().toString(),
        label: currentClass,
        x_min,
        y_min,
        x_max,
        y_max,
      };

      const updated = [...annotations, newAnn];
      setAnnotations(updated);

      // Auto-save if image has a database ID
      if (selectedImage?.id) {
        saveAnnotations(selectedImage.id, updated).catch(console.error);
      }
    }
    setCurrentBox(null);
  };

  const handleDeleteAnnotation = (index) => {
    const updated = annotations.filter((_, i) => i !== index);
    setAnnotations(updated);
    if (selectedImage?.id) {
      saveAnnotations(selectedImage.id, updated).catch(console.error);
    }
  };

  const handleSaveManual = async () => {
    if (!selectedImage?.id) {
      alert('Annotations cached locally. They will be committed to the database.');
      return;
    }
    setSaving(true);
    try {
      await saveAnnotations(selectedImage.id, annotations);
      alert('Annotations saved successfully!');
    } catch (err) {
      alert(`Save failed: ${err.message}`);
    } finally {
      setSaving(false);
    }
  };

  // Add new object class
  const handleAddClass = (e) => {
    e.preventDefault();
    if (!newClassName.trim()) return;
    const clean = newClassName.trim().toLowerCase();
    if (!classList.includes(clean)) {
      setClassList((prev) => [...prev, clean]);
      setCurrentClass(clean);
    }
    setNewClassName('');
  };

  // Navigation between images
  const handlePrevImage = () => {
    if (selectedImageIndex > 0) {
      setSelectedImageIndex(selectedImageIndex - 1);
    }
  };

  const handleNextImage = () => {
    if (selectedImageIndex < images.length - 1) {
      setSelectedImageIndex(selectedImageIndex + 1);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: 'calc(100vh - 100px)' }}>
      {/* Hidden file inputs for local folder and files */}
      <input
        type="file"
        ref={folderInputRef}
        webkitdirectory="true"
        directory="true"
        multiple
        style={{ display: 'none' }}
        onChange={(e) => handleIngestFiles(e.target.files)}
      />
      <input
        type="file"
        ref={filesInputRef}
        multiple
        accept="image/*,.txt"
        style={{ display: 'none' }}
        onChange={(e) => handleIngestFiles(e.target.files)}
      />

      {/* Top Banner: Ingestion & Proceed to Training Action */}
      <div
        className="card"
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '16px',
          padding: '14px 20px',
          flexWrap: 'wrap',
          gap: '12px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <button
            className="btn btn-primary"
            onClick={() => folderInputRef.current?.click()}
            disabled={uploading}
            style={{ fontWeight: 600 }}
          >
            <FolderUp size={16} /> Upload Folder from PC
          </button>
          <button
            className="btn btn-secondary"
            onClick={() => filesInputRef.current?.click()}
            disabled={uploading}
          >
            <Upload size={15} /> Select Images
          </button>
          {images.length > 0 && (
            <span style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
              Loaded <strong>{images.length}</strong> images from your computer
            </span>
          )}
        </div>

        {/* PROCEED TO TRAINING BUTTON */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <button
            className="btn btn-lg"
            style={{
              background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
              color: '#fff',
              fontWeight: 600,
              boxShadow: '0 4px 14px rgba(16, 185, 129, 0.4)',
            }}
            onClick={() => onProceedToTraining && onProceedToTraining(activeDataset)}
            disabled={images.length === 0}
            title="Takes this annotated dataset directly to Model Training"
          >
            <Play size={16} />
            Proceed to Model Training (ไปสู่หน้าการเทรน)
          </button>
        </div>
      </div>

      {/* Upload Progress Bar */}
      {uploading && (
        <div className="card" style={{ marginBottom: '14px', padding: '12px 16px', borderColor: 'var(--accent-primary)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px', fontSize: '12px' }}>
            <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{uploadStatus}</span>
            <span style={{ fontWeight: 700, color: 'var(--accent-primary)' }}>{uploadProgress}%</span>
          </div>
          <div className="progress-track">
            <div className="progress-fill" style={{ width: `${uploadProgress}%` }} />
          </div>
        </div>
      )}

      {/* Main Workspace Body */}
      {images.length === 0 ? (
        /* Empty State Dropzone */
        <div
          className={`dropzone-container ${isDragOver ? 'drag-over' : ''}`}
          style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}
          onDragOver={(e) => {
            e.preventDefault();
            setIsDragOver(true);
          }}
          onDragLeave={() => setIsDragOver(false)}
          onDrop={handleDrop}
          onClick={() => folderInputRef.current?.click()}
        >
          <FolderUp size={52} color="var(--accent-primary)" style={{ marginBottom: '16px' }} />
          <h2 style={{ fontSize: '18px', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '8px' }}>
            อัพโหลดโฟลเดอร์รูปภาพจากคอมพิวเตอร์ของคุณ
          </h2>
          <p style={{ fontSize: '13px', color: 'var(--text-secondary)', maxWidth: '460px', lineHeight: 1.6, marginBottom: '20px' }}>
            คลิกเพื่อเลือกโฟลเดอร์ หรือลากโฟลเดอร์รูปภาพมาวางที่นี่ ระบบจะอ่านไฟล์ในเครื่องทันที และเปิดให้ตีกรอบวัตถุได้เลย
          </p>
          <div style={{ display: 'flex', gap: '10px' }}>
            <button
              className="btn btn-primary btn-lg"
              onClick={(e) => {
                e.stopPropagation();
                folderInputRef.current?.click();
              }}
            >
              <FolderUp size={16} /> เลือกโฟลเดอร์จากเครื่อง
            </button>
            <button
              className="btn btn-secondary btn-lg"
              onClick={(e) => {
                e.stopPropagation();
                filesInputRef.current?.click();
              }}
            >
              <Upload size={16} /> เลือกเฉพาะไฟล์รูปภาพ
            </button>
          </div>
        </div>
      ) : (
        /* 3-Column Studio Workspace */
        <div style={{ display: 'grid', gridTemplateColumns: '230px 1fr 280px', gap: '16px', flex: 1, minHeight: 0 }}>
          {/* Left: Filmstrip / Thumbnails */}
          <div className="card" style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden', padding: '14px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
              <span style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)' }}>
                รูปภาพทั้งหมด ({images.length})
              </span>
              <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                {selectedImageIndex + 1} / {images.length}
              </span>
            </div>

            <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '5px' }}>
              {images.map((img, idx) => {
                const isSelected = selectedImageIndex === idx;
                const displayName =
                  img.original_name ||
                  img.filename ||
                  (img.file_path ? img.file_path.split(/[\\/]/).pop() : `Image #${idx + 1}`);
                const annCount = img.annotations ? img.annotations.length : 0;

                return (
                  <div
                    key={img.id || idx}
                    onClick={() => setSelectedImageIndex(idx)}
                    style={{
                      padding: '8px 10px',
                      borderRadius: 'var(--radius-sm)',
                      backgroundColor: isSelected ? '#eef2ff' : '#ffffff',
                      border: `1px solid ${isSelected ? 'var(--accent-primary)' : 'var(--border-color)'}`,
                      cursor: 'pointer',
                      fontSize: '12px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      color: isSelected ? 'var(--accent-primary)' : 'var(--text-secondary)',
                      fontWeight: isSelected ? 600 : 400,
                      transition: 'all 0.15s ease',
                    }}
                  >
                    <span style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: '140px' }}>
                      {displayName}
                    </span>
                    {annCount > 0 ? (
                      <span className="badge badge-success" style={{ fontSize: '10px', padding: '1px 6px' }}>
                        {annCount}
                      </span>
                    ) : (
                      <span style={{ fontSize: '10px', color: 'var(--text-muted)' }}>0</span>
                    )}
                  </div>
                );
              })}
            </div>

            {/* Prev / Next Image buttons */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', marginTop: '10px', paddingTop: '10px', borderTop: '1px solid var(--border-color)' }}>
              <button
                className="btn btn-sm btn-secondary"
                onClick={handlePrevImage}
                disabled={selectedImageIndex === 0}
              >
                <ChevronLeft size={14} /> Prev
              </button>
              <button
                className="btn btn-sm btn-secondary"
                onClick={handleNextImage}
                disabled={selectedImageIndex === images.length - 1}
              >
                Next <ChevronRight size={14} />
              </button>
            </div>
          </div>

          {/* Center: Interactive Canvas for Bounding Boxes ("ตีกรอบ") */}
          <div
            className="card"
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              position: 'relative',
              overflow: 'hidden',
              padding: '10px',
              backgroundColor: '#f1f5f9',
            }}
          >
            {selectedImage ? (
              <div
                style={{
                  maxWidth: '100%',
                  maxHeight: '100%',
                  overflow: 'auto',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
              >
                <canvas
                  ref={canvasRef}
                  onMouseDown={handleMouseDown}
                  onMouseMove={handleMouseMove}
                  onMouseUp={handleMouseUp}
                  style={{
                    maxWidth: '100%',
                    maxHeight: 'calc(100vh - 220px)',
                    cursor: 'crosshair',
                    borderRadius: 'var(--radius-sm)',
                    boxShadow: '0 4px 14px rgba(0, 0, 0, 0.08)',
                  }}
                />
              </div>
            ) : null}

            {/* Hint overlay */}
            <div
              style={{
                position: 'absolute',
                bottom: '14px',
                background: 'rgba(255, 255, 255, 0.95)',
                boxShadow: 'var(--shadow-md)',
                border: '1px solid var(--border-color)',
                padding: '5px 14px',
                borderRadius: 'var(--radius-full)',
                fontSize: '11px',
                fontWeight: 500,
                color: 'var(--text-secondary)',
                pointerEvents: 'none',
              }}
            >
              คลิกและลากเมาส์บนภาพเพื่อตีกรอบวัตถุ (Draw bounding box)
            </div>
          </div>

          {/* Right: Class Manager & Bounding Box List ("สร้างออปเจค") */}
          <div className="card" style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden', padding: '16px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
              <h4 style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-primary)' }}>สร้างและเลือกออปเจค</h4>
              <button
                className="btn btn-sm btn-primary"
                onClick={handleSaveManual}
                disabled={saving}
              >
                <Save size={13} /> {saving ? 'บันทึก...' : 'บันทึก'}
              </button>
            </div>

            {/* Create / Select Object Class */}
            <div className="form-group">
              <label className="form-label">คลาสที่กำลังใช้งาน (Active Class)</label>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', marginBottom: '8px' }}>
                {classList.map((c) => (
                  <span
                    key={c}
                    onClick={() => setCurrentClass(c)}
                    style={{
                      padding: '4px 10px',
                      borderRadius: 'var(--radius-full)',
                      fontSize: '11px',
                      fontWeight: 600,
                      cursor: 'pointer',
                      backgroundColor: currentClass === c ? 'var(--accent-primary)' : '#f1f5f9',
                      color: currentClass === c ? '#fff' : 'var(--text-secondary)',
                      border: `1px solid ${currentClass === c ? 'var(--accent-primary)' : 'var(--border-color)'}`,
                      transition: 'all 0.15s ease',
                    }}
                  >
                    {c}
                  </span>
                ))}
              </div>

              <form onSubmit={handleAddClass} style={{ display: 'flex', gap: '6px' }}>
                <input
                  className="form-control"
                  style={{ flex: 1, padding: '6px 8px', fontSize: '12px' }}
                  placeholder="พิมพ์ชื่อวัตถุใหม่ เช่น car, defect..."
                  value={newClassName}
                  onChange={(e) => setNewClassName(e.target.value)}
                />
                <button type="submit" className="btn btn-sm btn-secondary" title="เพิ่มออปเจค">
                  <Plus size={13} />
                </button>
              </form>
            </div>

            <hr style={{ borderColor: 'var(--border-color)', margin: '12px 0' }} />

            {/* Current Drawn Bounding Boxes */}
            <h5 style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '8px' }}>
              กรอบบนภาพนี้ ({annotations.length})
            </h5>
            <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '6px' }}>
              {annotations.length === 0 ? (
                <div style={{ fontSize: '12px', color: 'var(--text-muted)', textAlign: 'center', marginTop: '20px' }}>
                  ยังไม่มีการตีกรอบ<br />คลิกลากเมาส์บนภาพเพื่อตีกรอบ
                </div>
              ) : (
                annotations.map((ann, idx) => (
                  <div
                    key={ann.id || idx}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      padding: '6px 10px',
                      borderRadius: 'var(--radius-sm)',
                      backgroundColor: '#f8fafc',
                      border: '1px solid var(--border-color)',
                      fontSize: '12px',
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <Tag size={12} color="var(--accent-primary)" />
                      <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{ann.label}</span>
                    </div>
                    <button
                      onClick={() => handleDeleteAnnotation(idx)}
                      style={{ background: 'transparent', border: 'none', color: '#ef4444', cursor: 'pointer' }}
                      title="ลบกรอบนี้"
                    >
                      <Trash2 size={13} />
                    </button>
                  </div>
                ))
              )}
            </div>

            {/* Quick Proceed button on bottom right */}
            <div style={{ marginTop: '14px', paddingTop: '10px', borderTop: '1px solid var(--border-color)' }}>
              <button
                className="btn btn-primary"
                style={{
                  width: '100%',
                  background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
                  fontWeight: 600,
                  boxShadow: '0 2px 8px rgba(16, 185, 129, 0.3)',
                }}
                onClick={() => onProceedToTraining && onProceedToTraining(activeDataset)}
              >
                <Play size={14} /> ไปสู่หน้าการเทรนโมเดล
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
