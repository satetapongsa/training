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
  FileText,
  Download,
  Eye,
  Check,
  Zap,
  Square,
  Pentagon,
  Undo2,
  X,
} from 'lucide-react';
import {
  createDataset,
  getDatasetImages,
  uploadFilesChunked,
  getAnnotations,
  saveAnnotations,
  splitDataset,
  runInference,
  API_BASE_URL,
} from '../api/client';

// Distinct curated color palette for object classes
const CLASS_COLORS = [
  '#4f46e5', // indigo
  '#10b981', // emerald
  '#f59e0b', // amber
  '#ef4444', // rose
  '#8b5cf6', // violet
  '#06b6d4', // cyan
  '#ec4899', // pink
  '#14b8a6', // teal
  '#f97316', // orange
  '#6366f1', // blue-indigo
];

export default function StudioView({
  activeDataset,
  setActiveDataset,
  onProceedToTraining,
}) {
  // Datasets & Images
  const [images, setImages] = useState([]);
  const [selectedImageIndex, setSelectedImageIndex] = useState(0);
  const [filterMode, setFilterMode] = useState('all'); // 'all', 'pending', 'annotated'

  // Object classes & current selection
  const [classList, setClassList] = useState(['object', 'defect', 'product', 'person']);
  const [currentClass, setCurrentClass] = useState('object');
  const [newClassName, setNewClassName] = useState('');

  // Annotations on currently selected image
  const [annotations, setAnnotations] = useState([]);
  const [savingGt, setSavingGt] = useState(false);
  const [saveFeedback, setSaveFeedback] = useState(false);
  const [autoDetecting, setAutoDetecting] = useState(false);
  const [bundling, setBundling] = useState(false);

  // Canvas drawing state
  const [drawMode, setDrawMode] = useState('polygon'); // 'polygon' (คลิกแต่ละมุมรอบวัตถุ ทุกแนว ทุกมุม เอียงได้) or 'box' (สี่เหลี่ยม)
  const [isDrawing, setIsDrawing] = useState(false);
  const [startPos, setStartPos] = useState({ x: 0, y: 0 });
  const [currentBox, setCurrentBox] = useState(null);
  const [selectedBoxIndex, setSelectedBoxIndex] = useState(null);
  const [polygonPoints, setPolygonPoints] = useState([]); // [{x, y}, ...]
  const [cursorPos, setCursorPos] = useState(null); // {x, y} for live connecting guide line

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

  // Helper to get class color
  const getClassColor = (className) => {
    const idx = classList.indexOf(className);
    if (idx === -1) return '#4f46e5';
    return CLASS_COLORS[idx % CLASS_COLORS.length];
  };

  // Helper to get class index
  const getClassId = (className) => {
    const idx = classList.indexOf(className);
    return idx >= 0 ? idx : 0;
  };

  // Load existing images if dataset is provided
  useEffect(() => {
    if (activeDataset?.id) {
      loadImagesFromDataset(activeDataset.id);
    }
  }, [activeDataset]);

  // Load annotations when selected image changes
  useEffect(() => {
    setPolygonPoints([]);
    setCursorPos(null);
    setCurrentBox(null);

    if (!selectedImage) {
      setAnnotations([]);
      return;
    }

    if (selectedImage.annotations && selectedImage.annotations.length > 0) {
      const formatted = selectedImage.annotations.map((ann) => {
        const w = ann.bbox_w !== undefined ? ann.bbox_w : ann.x_max - ann.x_min;
        const h = ann.bbox_h !== undefined ? ann.bbox_h : ann.y_max - ann.y_min;
        const cx = ann.bbox_x !== undefined ? ann.bbox_x : ann.x_min + w / 2;
        const cy = ann.bbox_y !== undefined ? ann.bbox_y : ann.y_min + h / 2;
        const label = ann.class_name || ann.label || 'object';
        return {
          id: ann.id || Math.random().toString(),
          label: label,
          x_min: Math.max(0, cx - w / 2),
          y_min: Math.max(0, cy - h / 2),
          x_max: Math.min(1, cx + w / 2),
          y_max: Math.min(1, cy + h / 2),
          segmentation: ann.segmentation || null,
        };
      });
      setAnnotations(formatted);
      // Ensure class exists in classList
      formatted.forEach((f) => {
        if (!classList.includes(f.label)) {
          setClassList((prev) => [...prev, f.label]);
        }
      });
    } else if (selectedImage.id) {
      getAnnotations(selectedImage.id)
        .then((data) => {
          const formatted = (data || []).map((ann) => ({
            id: ann.id || Math.random().toString(),
            label: ann.class_name || 'object',
            x_min: Math.max(0, ann.bbox_x - ann.bbox_w / 2),
            y_min: Math.max(0, ann.bbox_y - ann.bbox_h / 2),
            x_max: Math.min(1, ann.bbox_x + ann.bbox_w / 2),
            y_max: Math.min(1, ann.bbox_y + ann.bbox_h / 2),
            segmentation: ann.segmentation || null,
          }));
          setAnnotations(formatted);
          formatted.forEach((f) => {
            if (!classList.includes(f.label)) {
              setClassList((prev) => [...prev, f.label]);
            }
          });
        })
        .catch(() => setAnnotations([]));
    } else {
      setAnnotations([]);
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



  // --- LOCAL FOLDER & IMAGE INGESTION ---
  const handleIngestFiles = async (fileList) => {
    const rawFiles = Array.from(fileList);
    if (rawFiles.length === 0) return;

    // Filter image files and companion .txt label files
    const imageFiles = rawFiles.filter((f) =>
      /\.(jpe?g|png|webp|bmp|tiff?)$/i.test(f.name)
    );
    const labelFiles = rawFiles.filter((f) => /\.txt$/i.test(f.name));

    if (imageFiles.length === 0) {
      alert('ไม่พบไฟล์รูปภาพที่รองรับ (.jpg, .png, .webp, .bmp) ในโฟลเดอร์ที่เลือก');
      return;
    }

    setUploading(true);
    setUploadProgress(10);
    setUploadStatus(`กำลังอ่านไฟล์รูปภาพ ${imageFiles.length} รูปจากเครื่องของคุณ...`);

    // Parse companion YOLO .txt labels client-side
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

    // Prepare local preview items immediately for instant zero-lag rendering
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
            const classIdx = parseInt(parts[0], 10);
            const labelName = classList[classIdx] || `class_${classIdx}`;
            if (parts.length === 5) {
              const cx = parseFloat(parts[1]);
              const cy = parseFloat(parts[2]);
              const w = parseFloat(parts[3]);
              const h = parseFloat(parts[4]);
              if (!isNaN(cx) && !isNaN(cy) && !isNaN(w) && !isNaN(h)) {
                initialAnnots.push({
                  id: Math.random().toString(),
                  label: labelName,
                  x_min: Math.max(0, cx - w / 2),
                  y_min: Math.max(0, cy - h / 2),
                  x_max: Math.min(1, cx + w / 2),
                  y_max: Math.min(1, cy + h / 2),
                  segmentation: null,
                });
              }
            } else {
              // Polygon coordinates: classIdx x1 y1 x2 y2 ...
              const coords = parts.slice(1).map(Number);
              const xs = coords.filter((_, i) => i % 2 === 0);
              const ys = coords.filter((_, i) => i % 2 === 1);
              if (xs.length >= 3 && ys.length >= 3) {
                const seg = [];
                for (let i = 0; i < Math.min(xs.length, ys.length); i++) {
                  seg.push([xs[i], ys[i]]);
                }
                const minX = Math.min(...xs);
                const maxX = Math.max(...xs);
                const minY = Math.min(...ys);
                const maxY = Math.max(...ys);
                initialAnnots.push({
                  id: Math.random().toString(),
                  label: labelName,
                  x_min: Math.max(0, minX),
                  y_min: Math.max(0, minY),
                  x_max: Math.min(1, maxX),
                  y_max: Math.min(1, maxY),
                  segmentation: seg,
                });
              }
            }
          }
        }
      }

      return {
        id: null,
        filename: file.name,
        original_name: file.name,
        localUrl: localUrl,
        fileHandle: file,
        annotations: initialAnnots,
        is_annotated: initialAnnots.length > 0,
      };
    });

    setImages(localItems);
    setSelectedImageIndex(0);

    // Auto-create dataset on backend
    try {
      const folderName =
        imageFiles[0].webkitRelativePath?.split('/')[0] ||
        `Folder_${new Date().toISOString().slice(0, 10)}_${Math.floor(Math.random() * 1000)}`;
      setUploadStatus(`กำลังสร้างชุดข้อมูล "${folderName}" บนระบบ...`);

      const targetDataset = await createDataset({
        name: folderName,
        description: `โหลดจากโฟลเดอร์ในเครื่องเมื่อ ${new Date().toLocaleString('th-TH')}`,
        classes: classList,
      });
      setActiveDataset(targetDataset);

      // Upload files chunked
      setUploadStatus(`กำลังอัปโหลดรูปภาพ ${imageFiles.length} รูปขึ้นระบบจัดเก็บ...`);
      await uploadFilesChunked(targetDataset.id, rawFiles, (uploaded, total, pct) => {
        setUploadProgress(pct);
        setUploadStatus(`อัปโหลดแล้ว ${uploaded} / ${total} ไฟล์ (${pct}%)`);
      });

      // Reload fresh images with real database IDs
      const freshImages = await getDatasetImages(targetDataset.id);
      if (Array.isArray(freshImages) && freshImages.length > 0) {
        // Merge any locally parsed annotations if backend didn't parse them yet
        const merged = freshImages.map((fi, idx) => {
          const match = localItems.find((li) => li.filename === fi.filename);
          if (match && match.annotations.length > 0 && (!fi.annotations || fi.annotations.length === 0)) {
            fi.annotations = match.annotations;
            fi.is_annotated = true;
          }
          return fi;
        });
        setImages(merged);
      }

      setUploadStatus('โหลดโฟลเดอร์รูปภาพเสร็จสมบูรณ์ พร้อมตีกรอบ!');
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
        const readEntries = () => {
          dirReader.readEntries(async (results) => {
            if (!results.length) {
              resolve([]);
            } else {
              const fileArrays = await Promise.all(results.map(scanFilesFromEntry));
              resolve(fileArrays.flat());
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

  // --- CANVAS BOUNDING BOX RENDERING & INTERACTION (ตีกรอบ) ---
  // --- CANVAS BOUNDING BOX & MULTI-CORNER POLYGON RENDERING & INTERACTION ---
  useEffect(() => {
    if (!selectedImage || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const img = new Image();
    const imageSrc =
      selectedImage.localUrl ||
      (selectedImage.id ? `${API_BASE_URL}/api/v1/datasets/images/${selectedImage.id}/file` : '');

    if (!imageSrc) return;

    img.crossOrigin = 'anonymous';
    img.src = imageSrc;
    img.onload = () => {
      imageObjRef.current = img;
      canvas.width = img.naturalWidth || 800;
      canvas.height = img.naturalHeight || 600;
      redrawCanvas();
    };
  }, [selectedImage]);

  useEffect(() => {
    redrawCanvas();
  }, [annotations, currentBox, selectedBoxIndex, drawMode, polygonPoints, cursorPos, currentClass]);

  const redrawCanvas = () => {
    const canvas = canvasRef.current;
    const img = imageObjRef.current;
    if (!canvas || !img) return;

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    // 1. Draw existing annotations (both polygon contours and bounding boxes)
    annotations.forEach((ann, idx) => {
      const isSelected = selectedBoxIndex === idx;
      const color = getClassColor(ann.label);

      if (ann.segmentation && ann.segmentation.length >= 3) {
        // Multi-corner polygon contour (ทุกมุม ทุกแนว ทั้งเอียง)
        const pts = ann.segmentation.map((pt) => ({
          x: pt[0] * canvas.width,
          y: pt[1] * canvas.height,
        }));

        ctx.beginPath();
        ctx.moveTo(pts[0].x, pts[0].y);
        for (let i = 1; i < pts.length; i++) {
          ctx.lineTo(pts[i].x, pts[i].y);
        }
        ctx.closePath();

        ctx.fillStyle = isSelected ? `${color}45` : `${color}25`;
        ctx.fill();

        ctx.strokeStyle = color;
        ctx.lineWidth = isSelected ? 4 : 2.5;
        ctx.stroke();

        // Corner vertex dots
        pts.forEach((pt) => {
          ctx.beginPath();
          ctx.arc(pt.x, pt.y, 4, 0, Math.PI * 2);
          ctx.fillStyle = '#ffffff';
          ctx.fill();
          ctx.strokeStyle = color;
          ctx.lineWidth = 2;
          ctx.stroke();
        });

        // Label Tag Chip
        const labelText = ann.label || 'object';
        ctx.font = 'bold 13px Inter, sans-serif';
        const textWidth = ctx.measureText(labelText).width;
        const badgeH = 22;
        const badgeW = textWidth + 18;

        const anchorX = Math.min(...pts.map((p) => p.x));
        const anchorY = Math.min(...pts.map((p) => p.y));

        ctx.fillStyle = color;
        ctx.fillRect(anchorX, Math.max(0, anchorY - badgeH), badgeW, badgeH);

        ctx.fillStyle = '#ffffff';
        ctx.fillText(labelText, anchorX + 8, Math.max(15, anchorY - 6));
      } else {
        // Standard Bounding Box Rect
        const x = ann.x_min * canvas.width;
        const y = ann.y_min * canvas.height;
        const w = (ann.x_max - ann.x_min) * canvas.width;
        const h = (ann.y_max - ann.y_min) * canvas.height;

        ctx.strokeStyle = color;
        ctx.lineWidth = isSelected ? 4 : 2.5;
        ctx.fillStyle = `${color}22`;
        ctx.fillRect(x, y, w, h);
        ctx.strokeRect(x, y, w, h);

        const labelText = ann.label || 'object';
        ctx.font = 'bold 13px Inter, sans-serif';
        const textWidth = ctx.measureText(labelText).width;
        const badgeH = 22;
        const badgeW = textWidth + 16;

        ctx.fillStyle = color;
        ctx.fillRect(x, Math.max(0, y - badgeH), badgeW, badgeH);

        ctx.fillStyle = '#ffffff';
        ctx.fillText(labelText, x + 8, Math.max(15, y - 6));
      }
    });

    // 2. Draw currently actively dragged box (in 'box' mode)
    if (drawMode === 'box' && currentBox && currentBox.w > 0 && currentBox.h > 0) {
      const color = getClassColor(currentClass);
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.setLineDash([6, 4]);
      ctx.fillStyle = `${color}25`;
      ctx.fillRect(currentBox.x, currentBox.y, currentBox.w, currentBox.h);
      ctx.strokeRect(currentBox.x, currentBox.y, currentBox.w, currentBox.h);
      ctx.setLineDash([]);

      ctx.font = 'bold 12px Inter, sans-serif';
      ctx.fillStyle = color;
      ctx.fillRect(currentBox.x, Math.max(0, currentBox.y - 20), ctx.measureText(currentClass).width + 14, 20);
      ctx.fillStyle = '#ffffff';
      ctx.fillText(currentClass, currentBox.x + 7, Math.max(14, currentBox.y - 5));
    }

    // 3. Draw active multi-corner polygon being created (in 'polygon' mode)
    if (drawMode === 'polygon' && polygonPoints.length > 0) {
      const color = getClassColor(currentClass);

      // Connecting edges between placed vertices
      ctx.beginPath();
      ctx.moveTo(polygonPoints[0].x, polygonPoints[0].y);
      for (let i = 1; i < polygonPoints.length; i++) {
        ctx.lineTo(polygonPoints[i].x, polygonPoints[i].y);
      }
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.stroke();

      // Dynamic rubberband guide line from last point to cursor
      if (cursorPos) {
        ctx.beginPath();
        ctx.setLineDash([6, 4]);
        const lastPt = polygonPoints[polygonPoints.length - 1];
        ctx.moveTo(lastPt.x, lastPt.y);
        ctx.lineTo(cursorPos.x, cursorPos.y);
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // Draw numbered circle handles on vertices
      polygonPoints.forEach((pt, i) => {
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 6, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.stroke();

        ctx.font = 'bold 10px Inter, sans-serif';
        ctx.fillStyle = '#ffffff';
        ctx.fillText(String(i + 1), pt.x + 8, pt.y - 4);
      });

      // Highlight first point if cursor is close to it to close the polygon
      if (polygonPoints.length >= 3 && cursorPos) {
        const dist = Math.hypot(cursorPos.x - polygonPoints[0].x, cursorPos.y - polygonPoints[0].y);
        if (dist <= 20) {
          ctx.beginPath();
          ctx.arc(polygonPoints[0].x, polygonPoints[0].y, 14, 0, Math.PI * 2);
          ctx.strokeStyle = '#10b981';
          ctx.lineWidth = 3;
          ctx.stroke();

          ctx.font = 'bold 11px Inter, sans-serif';
          ctx.fillStyle = '#10b981';
          ctx.fillText('คลิกเพื่อปิดกรอบ', polygonPoints[0].x + 12, polygonPoints[0].y + 14);
        }
      }
    }
  };

  const getCanvasMousePos = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
    };
  };

  const finishPolygon = () => {
    const canvas = canvasRef.current;
    if (!canvas || polygonPoints.length < 3) return;

    const normPoints = polygonPoints.map((p) => [
      Math.max(0, Math.min(1, p.x / canvas.width)),
      Math.max(0, Math.min(1, p.y / canvas.height)),
    ]);
    const xs = normPoints.map((p) => p[0]);
    const ys = normPoints.map((p) => p[1]);
    const x_min = Math.min(...xs);
    const x_max = Math.max(...xs);
    const y_min = Math.min(...ys);
    const y_max = Math.max(...ys);

    if (x_max - x_min > 0.005 && y_max - y_min > 0.005) {
      const newAnn = {
        id: Math.random().toString(),
        label: currentClass,
        x_min,
        y_min,
        x_max,
        y_max,
        segmentation: normPoints,
      };

      const updated = [...annotations, newAnn];
      setAnnotations(updated);

      setImages((prev) =>
        prev.map((img, idx) =>
          idx === selectedImageIndex ? { ...img, annotations: updated, is_annotated: true } : img
        )
      );

      if (selectedImage?.id) {
        saveAnnotations(selectedImage.id, updated).catch(console.error);
      }
    }

    setPolygonPoints([]);
    setCursorPos(null);
  };

  const undoLastPolygonPoint = () => {
    setPolygonPoints((prev) => prev.slice(0, -1));
  };

  const cancelCurrentPolygon = () => {
    setPolygonPoints([]);
    setCursorPos(null);
  };

  const handleCanvasClick = (e) => {
    if (!selectedImage) return;
    const pos = getCanvasMousePos(e);

    if (drawMode === 'polygon') {
      if (polygonPoints.length >= 3) {
        const dist = Math.hypot(pos.x - polygonPoints[0].x, pos.y - polygonPoints[0].y);
        if (dist <= 20) {
          finishPolygon();
          return;
        }
      }
      setPolygonPoints((prev) => [...prev, pos]);
      setSelectedBoxIndex(null);
    }
  };

  const handleMouseDown = (e) => {
    if (!selectedImage) return;
    if (drawMode === 'box') {
      const pos = getCanvasMousePos(e);
      setIsDrawing(true);
      setStartPos(pos);
      setCurrentBox({ x: pos.x, y: pos.y, w: 0, h: 0 });
      setSelectedBoxIndex(null);
    }
  };

  const handleMouseMove = (e) => {
    const pos = getCanvasMousePos(e);
    if (drawMode === 'polygon') {
      setCursorPos(pos);
    } else if (drawMode === 'box' && isDrawing) {
      const x = Math.min(startPos.x, pos.x);
      const y = Math.min(startPos.y, pos.y);
      const w = Math.abs(pos.x - startPos.x);
      const h = Math.abs(pos.y - startPos.y);
      setCurrentBox({ x, y, w, h });
    }
  };

  const handleMouseUp = () => {
    if (drawMode === 'box' && isDrawing) {
      setIsDrawing(false);
      const canvas = canvasRef.current;
      if (canvas && currentBox && currentBox.w > 10 && currentBox.h > 10) {
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
          segmentation: null,
        };

        const updated = [...annotations, newAnn];
        setAnnotations(updated);

        // Update in images list
        setImages((prev) =>
          prev.map((img, idx) =>
            idx === selectedImageIndex ? { ...img, annotations: updated, is_annotated: true } : img
          )
        );

        // Auto-save if image has a database ID
        if (selectedImage?.id) {
          saveAnnotations(selectedImage.id, updated).catch(console.error);
        }
      }
      setCurrentBox(null);
    }
  };

  const handleDoubleClick = () => {
    if (drawMode === 'polygon' && polygonPoints.length >= 3) {
      finishPolygon();
    }
  };

  const handleDeleteAnnotation = (index) => {
    const updated = annotations.filter((_, i) => i !== index);
    setAnnotations(updated);
    setImages((prev) =>
      prev.map((img, idx) =>
        idx === selectedImageIndex ? { ...img, annotations: updated, is_annotated: updated.length > 0 } : img
      )
    );
    if (selectedImage?.id) {
      saveAnnotations(selectedImage.id, updated).catch(console.error);
    }
  };

  // Keyboard navigation & shortcuts (image switching & polygon controls)
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName)) return;

      if (e.key === 'ArrowLeft') {
        e.preventDefault();
        handlePrevImage();
      } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        handleNextImage();
      } else if ((e.key === 'Enter' || e.key === ' ') && drawMode === 'polygon' && polygonPoints.length >= 3) {
        e.preventDefault();
        finishPolygon();
      } else if (e.key === 'Backspace' && drawMode === 'polygon' && polygonPoints.length > 0) {
        e.preventDefault();
        undoLastPolygonPoint();
      } else if (e.key === 'Escape' && drawMode === 'polygon' && polygonPoints.length > 0) {
        e.preventDefault();
        cancelCurrentPolygon();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [selectedImageIndex, images.length, drawMode, polygonPoints.length, currentClass]);

  // --- SAVE GROUND TRUTH (บันทึกเป็นไฟล์ GT) ---
  const handleSaveGroundTruth = async (advanceNext = false) => {
    setSavingGt(true);
    try {
      if (selectedImage?.id) {
        await saveAnnotations(selectedImage.id, annotations);
      }

      // Mark current image as annotated in state
      setImages((prev) =>
        prev.map((img, idx) =>
          idx === selectedImageIndex
            ? { ...img, annotations: [...annotations], is_annotated: annotations.length > 0 }
            : img
        )
      );

      setSaveFeedback(true);
      setTimeout(() => setSaveFeedback(false), 2000);

      if (advanceNext && selectedImageIndex < images.length - 1) {
        setSelectedImageIndex(selectedImageIndex + 1);
      }
    } catch (err) {
      alert(`บันทึก GT ผิดพลาด: ${err.message}`);
    } finally {
      setSavingGt(false);
    }
  };

  // Export / Download Single Image GT (.txt) to PC
  const handleDownloadSingleGt = () => {
    if (!selectedImage) return;
    const lines = annotations.map((ann) => {
      const classId = getClassId(ann.label);
      if (ann.segmentation && ann.segmentation.length >= 3) {
        const segStr = ann.segmentation.map((pt) => `${pt[0].toFixed(6)} ${pt[1].toFixed(6)}`).join(' ');
        return `${classId} ${segStr}`;
      }
      const cx = (ann.x_min + ann.x_max) / 2;
      const cy = (ann.y_min + ann.y_max) / 2;
      const w = ann.x_max - ann.x_min;
      const h = ann.y_max - ann.y_min;
      return `${classId} ${cx.toFixed(6)} ${cy.toFixed(6)} ${w.toFixed(6)} ${h.toFixed(6)}`;
    });

    const blob = new Blob([lines.join('\n')], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    const baseName = selectedImage.filename?.replace(/\.[^/.]+$/, '') || 'annotation';
    a.href = url;
    a.download = `${baseName}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // AI Auto-Detect (ดีเทคอัตโนมัติเพื่อช่วยตีกรอบ)
  const handleAutoDetect = async () => {
    if (!selectedImage) return;
    setAutoDetecting(true);
    try {
      let fileToSend = selectedImage.fileHandle;
      if (!fileToSend && selectedImage.id) {
        const res = await fetch(`${API_BASE_URL}/api/v1/datasets/images/${selectedImage.id}/file`);
        const blob = await res.blob();
        fileToSend = new File([blob], selectedImage.filename || 'image.jpg', { type: blob.type });
      }

      if (!fileToSend) {
        alert('ไม่พบไฟล์ต้นฉบับสำหรับรันการตรวจจับ');
        return;
      }

      const formData = new FormData();
      formData.append('file', fileToSend);
      formData.append('confidence', 0.25);

      const result = await runInference(formData);
      if (result && result.detections && result.detections.length > 0) {
        const proposed = result.detections.map((det) => ({
          id: Math.random().toString(),
          label: det.class_name || currentClass,
          x_min: det.box.x1,
          y_min: det.box.y1,
          x_max: det.box.x2,
          y_max: det.box.y2,
        }));

        const merged = [...annotations, ...proposed];
        setAnnotations(merged);

        // Ensure detected classes exist in classList
        proposed.forEach((p) => {
          if (!classList.includes(p.label)) {
            setClassList((prev) => [...prev, p.label]);
          }
        });

        if (selectedImage.id) {
          await saveAnnotations(selectedImage.id, merged);
        }
      } else {
        alert('ไม่พบวัตถุเพิ่มเติมจากโมเดล AI ในภาพนี้ คุณสามารถคลิกลากตีกรอบเองได้เลย');
      }
    } catch (err) {
      alert(`Auto-Detect: ${err.message}`);
    } finally {
      setAutoDetecting(false);
    }
  };

  // --- PACKAGE & PROCEED TO TRAIN (มัดรวมไฟล์ GT และไปเทรนโมเดล) ---
  const handlePackageAndTrain = async () => {
    if (images.length === 0) {
      alert('กรุณาโหลดโฟลเดอร์รูปภาพก่อนเริ่มการเทรน');
      return;
    }

    setBundling(true);
    try {
      let ds = activeDataset;
      if (!ds && images[0]?.dataset_id) {
        ds = { id: images[0].dataset_id, name: 'current_dataset' };
      }

      if (ds?.id) {
        // Trigger split and physical manifest generation (train/labels, val/labels)
        await splitDataset(ds.id, {
          train_ratio: 0.8,
          val_ratio: 0.2,
          test_ratio: 0.0,
        });
      }

      if (onProceedToTraining) {
        onProceedToTraining(ds);
      }
    } catch (err) {
      console.warn('Split / packaging note:', err.message);
      if (onProceedToTraining) {
        onProceedToTraining(activeDataset);
      }
    } finally {
      setBundling(false);
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

  // Image Navigation
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

  // Filtered images list
  const filteredImages = images.filter((img) => {
    const hasBoxes = (img.annotations && img.annotations.length > 0) || img.is_annotated;
    if (filterMode === 'pending') return !hasBoxes;
    if (filterMode === 'annotated') return hasBoxes;
    return true;
  });

  const annotatedCount = images.filter(
    (img) => (img.annotations && img.annotations.length > 0) || img.is_annotated
  ).length;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: 'calc(100vh - 98px)' }}>
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
          marginBottom: '14px',
          padding: '12px 18px',
          flexWrap: 'wrap',
          gap: '12px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <button
            className="btn btn-primary"
            onClick={() => folderInputRef.current?.click()}
            disabled={uploading}
            style={{ fontWeight: 600 }}
          >
            <FolderUp size={16} /> โหลดโฟลเดอร์รูปจากเครื่อง
          </button>
          <button
            className="btn btn-secondary"
            onClick={() => filesInputRef.current?.click()}
            disabled={uploading}
          >
            <Upload size={15} /> เลือกเฉพาะไฟล์รูปภาพ
          </button>

          {images.length > 0 && (
            <div style={{ fontSize: '13px', color: 'var(--text-secondary)', marginLeft: '6px' }}>
              ทั้งหมด: <strong>{images.length}</strong> รูป | บันทึก GT แล้ว:{' '}
              <strong style={{ color: 'var(--accent-success)' }}>{annotatedCount}</strong> รูป (
              {Math.round((annotatedCount / images.length) * 100)}%)
            </div>
          )}
        </div>

        {/* PROCEED TO TRAINING BUTTON */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <button
            className="btn btn-lg"
            style={{
              background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
              color: '#ffffff',
              fontWeight: 600,
              boxShadow: '0 4px 12px rgba(16, 185, 129, 0.35)',
            }}
            onClick={handlePackageAndTrain}
            disabled={images.length === 0 || bundling}
            title="มัดรวมไฟล์ภาพและไฟล์ GT เข้าด้วยกัน แล้วนำไปสู่หน้าการเทรนโมเดล"
          >
            <Play size={16} />
            {bundling ? 'กำลังมัดรวมข้อมูล...' : 'มัดรวมไฟล์ GT และไปเทรนโมเดล'}
          </button>
        </div>
      </div>

      {/* Upload Progress Bar */}
      {uploading && (
        <div
          className="card"
          style={{ marginBottom: '14px', padding: '12px 16px', borderColor: 'var(--accent-primary)' }}
        >
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
          <FolderUp size={56} color="var(--accent-primary)" style={{ marginBottom: '16px' }} />
          <h2 style={{ fontSize: '19px', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '8px' }}>
            โหลดโฟลเดอร์รูปภาพจากเครื่องเพื่อเริ่มตีกรอบและสร้างไฟล์ GT
          </h2>
          <p
            style={{
              fontSize: '13px',
              color: 'var(--text-secondary)',
              maxWidth: '520px',
              lineHeight: 1.6,
              marginBottom: '22px',
            }}
          >
            คลิกปุ่มเพื่อเลือกโฟลเดอร์ หรือลากโฟลเดอร์รูปภาพจากเครื่องมาวางที่นี่
            ระบบจะแสดงรูปภาพทีละรูปให้คุณตีกรอบ จัดประเภทออปเจค และบันทึกเป็นไฟล์ GT มัดรวมไปเทรนโมเดลได้ทันที
          </p>
          <div style={{ display: 'flex', gap: '12px' }}>
            <button
              className="btn btn-primary btn-lg"
              onClick={(e) => {
                e.stopPropagation();
                folderInputRef.current?.click();
              }}
            >
              <FolderUp size={17} /> เลือกโฟลเดอร์จากเครื่อง
            </button>
            <button
              className="btn btn-secondary btn-lg"
              onClick={(e) => {
                e.stopPropagation();
                filesInputRef.current?.click();
              }}
            >
              <Upload size={17} /> เลือกเฉพาะไฟล์รูปภาพ
            </button>
          </div>
        </div>
      ) : (
        /* 3-Column Studio Workspace */
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: '260px 1fr 310px',
            gap: '14px',
            flex: 1,
            minHeight: 0,
          }}
        >
          {/* Column 1: รายการรูปภาพทั้งหมด (Filmstrip) */}
          <div className="card" style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden', padding: '14px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
              <span style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)' }}>
                รูปภาพในโฟลเดอร์ ({images.length})
              </span>
              <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                {selectedImageIndex + 1} / {images.length}
              </span>
            </div>

            {/* Filter Tabs */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '4px', marginBottom: '8px' }}>
              <button
                className={`btn btn-sm ${filterMode === 'all' ? 'btn-primary' : 'btn-secondary'}`}
                style={{ padding: '3px 6px', fontSize: '11px' }}
                onClick={() => setFilterMode('all')}
              >
                ทั้งหมด
              </button>
              <button
                className={`btn btn-sm ${filterMode === 'pending' ? 'btn-primary' : 'btn-secondary'}`}
                style={{ padding: '3px 6px', fontSize: '11px' }}
                onClick={() => setFilterMode('pending')}
              >
                ยังไม่ตีกรอบ
              </button>
              <button
                className={`btn btn-sm ${filterMode === 'annotated' ? 'btn-primary' : 'btn-secondary'}`}
                style={{ padding: '3px 6px', fontSize: '11px' }}
                onClick={() => setFilterMode('annotated')}
              >
                GT แล้ว
              </button>
            </div>

            {/* Image List */}
            <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '4px' }}>
              {filteredImages.map((img) => {
                const realIdx = images.indexOf(img);
                const isSelected = selectedImageIndex === realIdx;
                const displayName =
                  img.original_name ||
                  img.filename ||
                  (img.file_path ? img.file_path.split(/[\\/]/).pop() : `Image #${realIdx + 1}`);
                const annCount = img.annotations ? img.annotations.length : 0;
                const hasGt = annCount > 0 || img.is_annotated;

                return (
                  <div
                    key={img.id || realIdx}
                    onClick={() => setSelectedImageIndex(realIdx)}
                    style={{
                      padding: '6px 8px',
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
                    <span
                      style={{
                        whiteSpace: 'nowrap',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        maxWidth: '140px',
                      }}
                      title={displayName}
                    >
                      {displayName}
                    </span>

                    {hasGt ? (
                      <span className="badge badge-success" style={{ fontSize: '10px', padding: '2px 6px' }}>
                        GT ({annCount})
                      </span>
                    ) : (
                      <span style={{ fontSize: '10px', color: 'var(--text-muted)' }}>รอตีกรอบ</span>
                    )}
                  </div>
                );
              })}
            </div>

            {/* Prev / Next buttons */}
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: '1fr 1fr',
                gap: '8px',
                marginTop: '10px',
                paddingTop: '10px',
                borderTop: '1px solid var(--border-color)',
              }}
            >
              <button
                className="btn btn-sm btn-secondary"
                onClick={handlePrevImage}
                disabled={selectedImageIndex === 0}
              >
                <ChevronLeft size={14} /> รูปก่อนหน้า
              </button>
              <button
                className="btn btn-sm btn-secondary"
                onClick={handleNextImage}
                disabled={selectedImageIndex === images.length - 1}
              >
                รูปถัดไป <ChevronRight size={14} />
              </button>
            </div>
            <div style={{ textAlign: 'center', fontSize: '10px', color: 'var(--text-muted)', marginTop: '4px' }}>
              กดปุ่มลูกศร [ &larr; ] [ &rarr; ] บนคีย์บอร์ดเพื่อเปลี่ยนรูป
            </div>
          </div>

          {/* Column 2: พื้นที่ตีกรอบ (Interactive Annotation Canvas) */}
          <div
            className="card"
            style={{
              display: 'flex',
              flexDirection: 'column',
              position: 'relative',
              overflow: 'hidden',
              padding: '12px',
              backgroundColor: '#f8fafc',
            }}
          >
            {/* Top Toolbar above Canvas */}
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: '10px',
                paddingBottom: '8px',
                borderBottom: '1px solid var(--border-color)',
                flexWrap: 'wrap',
                gap: '8px',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)' }}>
                  รูปที่ {selectedImageIndex + 1} / {images.length}:
                </span>
                <span style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
                  {selectedImage?.original_name || selectedImage?.filename}
                </span>
              </div>

              <div style={{ display: 'flex', gap: '6px', alignItems: 'center', flexWrap: 'wrap' }}>
                {/* Annotation Tool Mode Selector */}
                <div
                  style={{
                    display: 'inline-flex',
                    borderRadius: '6px',
                    border: '1px solid var(--border-color)',
                    overflow: 'hidden',
                    backgroundColor: '#ffffff',
                  }}
                >
                  <button
                    className={`btn btn-sm ${drawMode === 'polygon' ? 'btn-primary' : 'btn-ghost'}`}
                    style={{
                      borderRadius: 0,
                      padding: '4px 10px',
                      fontSize: '11px',
                      fontWeight: 600,
                      boxShadow: 'none',
                    }}
                    onClick={() => {
                      setDrawMode('polygon');
                      setCurrentBox(null);
                    }}
                    title="คลิกซ้ายทีละมุมรอบวัตถุ ได้ทุกมุม ทุกแนว ทั้งเอียง"
                  >
                    <Pentagon size={13} /> โหมดคลิกแต่ละมุม (ทุกแนว/เอียง)
                  </button>
                  <button
                    className={`btn btn-sm ${drawMode === 'box' ? 'btn-primary' : 'btn-ghost'}`}
                    style={{
                      borderRadius: 0,
                      padding: '4px 10px',
                      fontSize: '11px',
                      fontWeight: 600,
                      boxShadow: 'none',
                    }}
                    onClick={() => {
                      setDrawMode('box');
                      setPolygonPoints([]);
                      setCursorPos(null);
                    }}
                    title="คลิกลากตีกรอบสี่เหลี่ยม"
                  >
                    <Square size={13} /> โหมดสี่เหลี่ยม
                  </button>
                </div>

                <button
                  className="btn btn-sm btn-secondary"
                  onClick={handleAutoDetect}
                  disabled={autoDetecting}
                  title="ให้โมเดล AI ช่วยดีเทคและเสนอตำแหน่งกรอบอัตโนมัติ"
                >
                  <Sparkles size={13} color="var(--accent-primary)" />
                  {autoDetecting ? 'กำลังดีเทค...' : 'ดีเทคอัตโนมัติ (AI Assist)'}
                </button>
                <button
                  className="btn btn-sm btn-secondary"
                  onClick={() => {
                    setAnnotations([]);
                    setPolygonPoints([]);
                    setCursorPos(null);
                  }}
                  disabled={annotations.length === 0 && polygonPoints.length === 0}
                  title="ล้างกรอบทั้งหมดบนรูปนี้"
                >
                  <Trash2 size={13} color="#ef4444" /> ล้างกรอบ
                </button>
              </div>
            </div>

            {/* Canvas Viewport */}
            <div
              style={{
                flex: 1,
                minHeight: 0,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                overflow: 'auto',
                position: 'relative',
                background: '#f1f5f9',
                borderRadius: 'var(--radius-sm)',
              }}
            >
              {selectedImage ? (
                <canvas
                  ref={canvasRef}
                  onClick={handleCanvasClick}
                  onMouseDown={handleMouseDown}
                  onMouseMove={handleMouseMove}
                  onMouseUp={handleMouseUp}
                  onDoubleClick={handleDoubleClick}
                  onMouseLeave={() => setCursorPos(null)}
                  style={{
                    maxWidth: '100%',
                    maxHeight: '100%',
                    cursor: 'crosshair',
                    borderRadius: 'var(--radius-sm)',
                    boxShadow: '0 4px 14px rgba(0, 0, 0, 0.08)',
                  }}
                />
              ) : null}
            </div>

            {/* Active Polygon In-Progress Action Bar */}
            {drawMode === 'polygon' && polygonPoints.length > 0 && (
              <div
                style={{
                  marginTop: '8px',
                  padding: '6px 12px',
                  backgroundColor: '#eef2ff',
                  borderRadius: 'var(--radius-sm)',
                  border: '1px solid #c7d2fe',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  fontSize: '12px',
                }}
              >
                <span style={{ color: 'var(--text-primary)', fontWeight: 500 }}>
                  กำลังตีกรอบหลายมุม: วางจุดแล้ว <strong>{polygonPoints.length}</strong> จุด
                  {polygonPoints.length < 3 ? ' (ต้องการอย่างน้อย 3 จุด)' : ' (คลิกจุดเริ่มต้น หรือดับเบิ้ลคลิกเพื่อปิดกรอบ)'}
                </span>
                <div style={{ display: 'flex', gap: '6px' }}>
                  <button
                    className="btn btn-sm btn-primary"
                    onClick={finishPolygon}
                    disabled={polygonPoints.length < 3}
                    style={{ padding: '3px 10px', fontSize: '11px', fontWeight: 600 }}
                  >
                    <Check size={12} /> ปิดกรอบนี้ (เสร็จสิ้น)
                  </button>
                  <button
                    className="btn btn-sm btn-secondary"
                    onClick={undoLastPolygonPoint}
                    style={{ padding: '3px 8px', fontSize: '11px' }}
                    title="ย้อนกลับจุดล่าสุด"
                  >
                    <Undo2 size={12} /> ย้อน 1 จุด
                  </button>
                  <button
                    className="btn btn-sm btn-secondary"
                    onClick={cancelCurrentPolygon}
                    style={{ padding: '3px 8px', fontSize: '11px', color: '#ef4444' }}
                    title="ยกเลิกกรอบนี้"
                  >
                    <X size={12} /> ยกเลิก
                  </button>
                </div>
              </div>
            )}

            {/* Canvas Bottom Hint */}
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginTop: '8px',
                fontSize: '11px',
                color: 'var(--text-muted)',
              }}
            >
              {drawMode === 'polygon' ? (
                <span>
                  โหมดคลิกแต่ละมุม: คลิกซ้ายที่แต่ละมุมรอบวัตถุ (ได้ทุกแนว ทุกมุม เอียงได้) แล้วคลิกจุดเริ่มต้น / ดับเบิ้ลคลิก เพื่อปิดกรอบ
                </span>
              ) : (
                <span>คลิกและลากเมาส์บนภาพเพื่อตีกรอบสี่เหลี่ยม (Draw bounding box)</span>
              )}
              <span>
                ประเภทปัจจุบัน: <strong style={{ color: getClassColor(currentClass) }}>{currentClass}</strong>
              </span>
            </div>
          </div>

          {/* Column 3: จัดประเภทออปเจค & บันทึกไฟล์ GT */}
          <div
            className="card"
            style={{
              display: 'flex',
              flexDirection: 'column',
              overflow: 'hidden',
              padding: '16px',
            }}
          >
            {/* Header: Ground Truth Action */}
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: '12px',
              }}
            >
              <h4 style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-primary)' }}>
                จัดประเภทออปเจค & บันทึก GT
              </h4>
              {saveFeedback && (
                <span className="badge badge-success" style={{ fontSize: '11px' }}>
                  <Check size={12} /> บันทึกแล้ว
                </span>
              )}
            </div>

            {/* Section 1: Object Class Chips */}
            <div className="form-group" style={{ marginBottom: '12px' }}>
              <label className="form-label" style={{ marginBottom: '6px' }}>
                เลือกประเภทออปเจค (Active Class)
              </label>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', marginBottom: '8px' }}>
                {classList.map((c) => {
                  const isCur = currentClass === c;
                  const col = getClassColor(c);
                  return (
                    <span
                      key={c}
                      onClick={() => setCurrentClass(c)}
                      style={{
                        padding: '4px 10px',
                        borderRadius: 'var(--radius-full)',
                        fontSize: '11px',
                        fontWeight: 600,
                        cursor: 'pointer',
                        backgroundColor: isCur ? col : '#f1f5f9',
                        color: isCur ? '#ffffff' : 'var(--text-secondary)',
                        border: `1px solid ${isCur ? col : 'var(--border-color)'}`,
                        transition: 'all 0.15s ease',
                      }}
                    >
                      {c}
                    </span>
                  );
                })}
              </div>

              {/* Add New Class Form */}
              <form onSubmit={handleAddClass} style={{ display: 'flex', gap: '6px' }}>
                <input
                  className="form-control"
                  style={{ flex: 1, padding: '5px 8px', fontSize: '12px' }}
                  placeholder="พิมพ์ชื่อประเภทใหม่ เช่น car, defect..."
                  value={newClassName}
                  onChange={(e) => setNewClassName(e.target.value)}
                />
                <button type="submit" className="btn btn-sm btn-secondary" title="เพิ่มประเภทใหม่">
                  <Plus size={13} /> เพิ่ม
                </button>
              </form>
            </div>

            <hr style={{ borderColor: 'var(--border-color)', margin: '10px 0' }} />

            {/* Section 2: Current Image Boxes List */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
              <h5 style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)' }}>
                กรอบในรูปนี้ ({annotations.length})
              </h5>
              {annotations.length > 0 && (
                <button
                  onClick={handleDownloadSingleGt}
                  className="btn btn-sm btn-secondary"
                  style={{ fontSize: '11px', padding: '2px 6px' }}
                  title="ดาวน์โหลดไฟล์ .txt ของภาพนี้ลงเครื่อง"
                >
                  <Download size={11} /> โหลด .txt
                </button>
              )}
            </div>

            <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '5px' }}>
              {annotations.length === 0 ? (
                <div
                  style={{
                    fontSize: '12px',
                    color: 'var(--text-muted)',
                    textAlign: 'center',
                    marginTop: '24px',
                    lineHeight: 1.6,
                  }}
                >
                  ยังไม่มีการตีกรอบในรูปนี้<br />
                  คลิกซ้ายตามมุมรอบวัตถุ หรือลากตีกรอบ
                </div>
              ) : (
                annotations.map((ann, idx) => {
                  const col = getClassColor(ann.label);
                  const isPoly = ann.segmentation && ann.segmentation.length >= 3;
                  return (
                    <div
                      key={ann.id || idx}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        padding: '6px 10px',
                        borderRadius: 'var(--radius-sm)',
                        backgroundColor: '#f8fafc',
                        border: `1px solid ${col}44`,
                        fontSize: '12px',
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexWrap: 'wrap' }}>
                        <span
                          style={{
                            width: '9px',
                            height: '9px',
                            borderRadius: '50%',
                            backgroundColor: col,
                          }}
                        />
                        <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{ann.label}</span>
                        {isPoly ? (
                          <span
                            style={{
                              fontSize: '10px',
                              backgroundColor: '#eef2ff',
                              color: '#4f46e5',
                              padding: '1px 5px',
                              borderRadius: '3px',
                              fontWeight: 500,
                            }}
                          >
                            หลายมุม ({ann.segmentation.length} จุด)
                          </span>
                        ) : null}
                        <span style={{ fontSize: '10px', color: 'var(--text-muted)' }}>
                          ({(ann.x_max - ann.x_min).toFixed(2)} &times; {(ann.y_max - ann.y_min).toFixed(2)})
                        </span>
                      </div>
                      <button
                        onClick={() => handleDeleteAnnotation(idx)}
                        style={{
                          background: 'transparent',
                          border: 'none',
                          color: '#ef4444',
                          cursor: 'pointer',
                        }}
                        title="ลบกรอบนี้"
                      >
                        <Trash2 size={13} />
                      </button>
                    </div>
                  );
                })
              )}
            </div>

            {/* Section 3: Save GT & Next Actions */}
            <div
              style={{
                marginTop: '12px',
                paddingTop: '12px',
                borderTop: '1px solid var(--border-color)',
                display: 'flex',
                flexDirection: 'column',
                gap: '8px',
              }}
            >
              <button
                className="btn btn-primary"
                style={{ width: '100%', fontWeight: 600 }}
                onClick={() => handleSaveGroundTruth(false)}
                disabled={savingGt}
              >
                <Save size={14} /> {savingGt ? 'กำลังบันทึก GT...' : 'บันทึกไฟล์ GT รูปนี้'}
              </button>

              <button
                className="btn btn-secondary"
                style={{ width: '100%', fontWeight: 500 }}
                onClick={() => handleSaveGroundTruth(true)}
                disabled={savingGt || selectedImageIndex >= images.length - 1}
              >
                บันทึก GT และไปรูปถัดไป &rarr;
              </button>

              <button
                className="btn"
                style={{
                  width: '100%',
                  marginTop: '4px',
                  background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
                  color: '#ffffff',
                  fontWeight: 600,
                  boxShadow: '0 2px 8px rgba(16, 185, 129, 0.3)',
                }}
                onClick={handlePackageAndTrain}
                disabled={images.length === 0 || bundling}
              >
                <Play size={14} /> มัดรวมไฟล์ GT และไปเทรนโมเดล
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
