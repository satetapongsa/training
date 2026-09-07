import React, { useState, useEffect, useRef } from 'react';
import JSZip from 'jszip';
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
  Archive,
  FolderDown,
  ZoomIn,
  ZoomOut,
  Maximize2,
  Minimize2,
  RotateCcw,
  MousePointerClick,
  AlertTriangle,
  Database,
} from 'lucide-react';
import {
  createDataset,
  getDatasetImages,
  uploadFilesChunked,
  uploadDatasetZip,
  downloadDatasetZip,
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

// Web Memory Storage Key (เก็บบันทึกความทรงจำ GT ไว้ในเว็บ)
const WEB_MEMORY_KEY = 'ai_vision_studio_gt_memory';

const loadAllWebMemory = () => {
  try {
    const raw = localStorage.getItem(WEB_MEMORY_KEY);
    return raw ? JSON.parse(raw) : {};
  } catch (e) {
    return {};
  }
};

const saveItemToWebMemory = (key, data) => {
  try {
    if (!key) return;
    const memory = loadAllWebMemory();
    memory[key] = {
      gt_name: data.gt_name,
      annotations: data.annotations || [],
      saved_at: new Date().toISOString(),
      classList: data.classList || [],
    };
    localStorage.setItem(WEB_MEMORY_KEY, JSON.stringify(memory));
  } catch (e) {
    console.warn('Failed to save to web memory:', e);
  }
};

const checkGtNameDuplicate = (name, currentIndex, allImages) => {
  const clean = (name || '').trim().toLowerCase();
  if (!clean) {
    return 'กรุณาพิมพ์ชื่อ GT สำหรับรูปภาพนี้ (ห้ามเว้นว่าง)';
  }
  for (let i = 0; i < allImages.length; i++) {
    if (i === currentIndex) continue;
    const otherImg = allImages[i];
    const otherName = (otherImg?.gt_name || '').trim().toLowerCase();
    if (otherName && otherName === clean) {
      const otherDisp = otherImg.original_name || otherImg.filename || `รูปที่ ${i + 1}`;
      return `ชื่อ GT "${name.trim()}" ซ้ำกับรูปที่ ${i + 1} (${otherDisp}) กรุณาตั้งชื่อไม่ให้ซ้ำกัน`;
    }
  }
  return null;
};

const generateUniqueGtName = (img, index, allImages) => {
  if (img?.gt_name && img.gt_name.trim()) {
    return img.gt_name.trim();
  }
  const rawName = img?.original_name || img?.filename || `image_${index + 1}`;
  const base = rawName.replace(/\.[^/.]+$/, '').trim() || `image_${index + 1}`;

  const isTaken = (candidate) => {
    const candLower = candidate.toLowerCase();
    return allImages.some((other, idx) => {
      if (idx === index) return false;
      const otherName = (other.gt_name || '').trim().toLowerCase();
      return otherName === candLower;
    });
  };

  if (!isTaken(base)) {
    return base;
  }

  let counter = 1;
  while (isTaken(`${base}_${counter}`)) {
    counter++;
  }
  return `${base}_${counter}`;
};

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

  // Unique Ground Truth Name per Image & Web Memory Toast Feedback
  const [imageGtName, setImageGtName] = useState('');
  const [gtNameError, setGtNameError] = useState(null);
  const [rightClickToast, setRightClickToast] = useState(null);
  const [webMemoryCount, setWebMemoryCount] = useState(0);

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

  // Canvas Scaling State (หน้าจอสเกลแสดงภาพส่วนที่เลือกเปิด)
  const [zoomMode, setZoomMode] = useState('fit'); // 'fit' (พอดีจอ), 'custom'
  const [zoomScale, setZoomScale] = useState(1); // 0.25 to 4.0
  const [imageDimensions, setImageDimensions] = useState({ width: 0, height: 0 });
  const [isRightPanelOpen, setIsRightPanelOpen] = useState(true);

  // DOM Refs
  const folderInputRef = useRef(null);
  const filesInputRef = useRef(null);
  const zipInputRef = useRef(null);
  const canvasRef = useRef(null);
  const imageObjRef = useRef(null);
  const viewportRef = useRef(null);

  const selectedImage = images[selectedImageIndex] || null;

  // Refresh web memory count
  const refreshWebMemoryCount = () => {
    const mem = loadAllWebMemory();
    setWebMemoryCount(Object.keys(mem).length);
  };

  useEffect(() => {
    refreshWebMemoryCount();
  }, [images]);

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

  // Load annotations & unique GT name when selected image changes
  useEffect(() => {
    setPolygonPoints([]);
    setCursorPos(null);
    setCurrentBox(null);

    if (!selectedImage) {
      setAnnotations([]);
      setImageGtName('');
      setGtNameError(null);
      return;
    }

    const memory = loadAllWebMemory();
    const memKey = selectedImage.filename || `img_${selectedImageIndex}`;
    const memData = memory[memKey];

    // 1. Determine & assign unique GT name
    let assignedGtName = selectedImage.gt_name || '';
    if (!assignedGtName && memData?.gt_name) {
      assignedGtName = memData.gt_name;
    }
    if (!assignedGtName) {
      assignedGtName = generateUniqueGtName(selectedImage, selectedImageIndex, images);
    }
    setImageGtName(assignedGtName);

    // Sync gt_name into image item if missing
    if (selectedImage.gt_name !== assignedGtName) {
      setImages((prev) =>
        prev.map((img, idx) => (idx === selectedImageIndex ? { ...img, gt_name: assignedGtName } : img))
      );
    }

    // Check duplicate
    setGtNameError(checkGtNameDuplicate(assignedGtName, selectedImageIndex, images));

    // 2. Determine annotations (state -> web memory -> backend API)
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
      formatted.forEach((f) => {
        if (!classList.includes(f.label)) {
          setClassList((prev) => [...prev, f.label]);
        }
      });
    } else if (memData && memData.annotations && memData.annotations.length > 0) {
      // Restore from Web Memory!
      setAnnotations(memData.annotations);
      setImages((prev) =>
        prev.map((img, idx) =>
          idx === selectedImageIndex
            ? { ...img, annotations: memData.annotations, is_annotated: true, gt_name: assignedGtName }
            : img
        )
      );
      if (memData.classList) {
        memData.classList.forEach((c) => {
          if (!classList.includes(c)) setClassList((prev) => [...prev, c]);
        });
      }
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

      const memory = loadAllWebMemory();
      const memKey = file.name;
      const memData = memory[memKey];
      const finalAnnots = initialAnnots.length > 0 ? initialAnnots : (memData?.annotations || []);
      const stem = file.name.replace(/\.[^/.]+$/, '').trim() || `image_${idx + 1}`;
      const assignedGt = memData?.gt_name || stem;

      return {
        id: null,
        filename: file.name,
        original_name: file.name,
        localUrl: localUrl,
        fileHandle: file,
        gt_name: assignedGt,
        annotations: finalAnnots,
        is_annotated: finalAnnots.length > 0,
      };
    });

    // Ensure unique GT names across all items
    const localNameSet = new Set();
    localItems.forEach((item, idx) => {
      let candidate = item.gt_name || `image_${idx + 1}`;
      let base = candidate;
      let counter = 1;
      while (localNameSet.has(candidate.toLowerCase())) {
        candidate = `${base}_${counter}`;
        counter++;
      }
      localNameSet.add(candidate.toLowerCase());
      item.gt_name = candidate;
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

  // --- ZIP ARCHIVE UPLOAD & CLIENT-SIDE UNPACKING ---
  const processZipFile = async (file) => {
    if (!file) return;

    setUploading(true);
    setUploadProgress(15);
    setUploadStatus('กำลังอ่านและแตกไฟล์ ZIP บนเว็บ...');

    try {
      // 1. Unpack client-side with JSZip for immediate zero-lag display
      const zip = await JSZip.loadAsync(file);
      setUploadProgress(35);

      // Check for classes.txt or dataset.yaml
      const detectedClasses = [...classList];
      for (const [relativePath, zipEntry] of Object.entries(zip.files)) {
        if (zipEntry.dir) continue;
        const lower = relativePath.toLowerCase();
        if (lower.endsWith('classes.txt')) {
          try {
            const txt = await zipEntry.async('text');
            const lines = txt.split('\n').map((l) => l.trim()).filter(Boolean);
            lines.forEach((c) => {
              if (!detectedClasses.includes(c)) detectedClasses.push(c);
            });
          } catch (pe) {}
        }
      }
      setClassList(detectedClasses);
      if (detectedClasses.length > 0) setCurrentClass(detectedClasses[0]);

      // Collect label text files
      const labelMap = {};
      for (const [relativePath, zipEntry] of Object.entries(zip.files)) {
        if (zipEntry.dir) continue;
        const lower = relativePath.toLowerCase();
        if (lower.endsWith('.txt') && !lower.endsWith('classes.txt')) {
          const stem = relativePath.split('/').pop().replace(/\.[^/.]+$/, '').toLowerCase();
          try {
            labelMap[stem] = await zipEntry.async('text');
          } catch (pe) {}
        }
      }

      // Collect images
      const imgExts = ['.jpg', '.jpeg', '.png', '.webp', '.bmp'];
      const imageEntries = Object.entries(zip.files).filter(([path, entry]) => {
        if (entry.dir) return false;
        const lower = path.toLowerCase();
        return imgExts.some((ext) => lower.endsWith(ext));
      });

      if (imageEntries.length === 0) {
        alert('ไม่พบไฟล์รูปภาพ (.jpg, .png, .webp, .bmp) ในไฟล์ ZIP นี้');
        setUploading(false);
        return;
      }

      setUploadProgress(60);
      setUploadStatus(`พบรูปภาพ ${imageEntries.length} รูป กำลังจัดเตรียมภาพและปรับสเกล...`);

      const loadedImages = [];
      for (let i = 0; i < imageEntries.length; i++) {
        const [path, entry] = imageEntries[i];
        const rawFilename = path.split('/').pop();
        const stem = rawFilename.replace(/\.[^/.]+$/, '').toLowerCase();
        const blob = await entry.async('blob');
        const localUrl = URL.createObjectURL(blob);
        const fileObj = new File([blob], rawFilename, { type: blob.type });

        let parsedAnnots = [];
        if (labelMap[stem]) {
          const lines = labelMap[stem].split('\n');
          for (const line of lines) {
            const parts = line.trim().split(/\s+/);
            if (parts.length >= 5) {
              const cid = parseInt(parts[0], 10);
              const cname = detectedClasses[cid] || `class_${cid}`;
              if (parts.length === 5) {
                const cx = parseFloat(parts[1]);
                const cy = parseFloat(parts[2]);
                const w = parseFloat(parts[3]);
                const h = parseFloat(parts[4]);
                if (!isNaN(cx) && !isNaN(cy) && !isNaN(w) && !isNaN(h)) {
                  parsedAnnots.push({
                    id: Math.random().toString(),
                    label: cname,
                    x_min: Math.max(0, cx - w / 2),
                    y_min: Math.max(0, cy - h / 2),
                    x_max: Math.min(1, cx + w / 2),
                    y_max: Math.min(1, cy + h / 2),
                    segmentation: null,
                  });
                }
              } else {
                const coords = parts.slice(1).map(Number);
                const xs = coords.filter((_, idx) => idx % 2 === 0);
                const ys = coords.filter((_, idx) => idx % 2 === 1);
                if (xs.length >= 3 && ys.length >= 3) {
                  const minX = Math.min(...xs);
                  const maxX = Math.max(...xs);
                  const minY = Math.min(...ys);
                  const maxY = Math.max(...ys);
                  const seg = [];
                  for (let k = 0; k < Math.min(xs.length, ys.length); k++) {
                    seg.push([xs[k], ys[k]]);
                  }
                  parsedAnnots.push({
                    id: Math.random().toString(),
                    label: cname,
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

        const memory = loadAllWebMemory();
        const memKey = rawFilename;
        const memData = memory[memKey];
        const finalAnnots = parsedAnnots.length > 0 ? parsedAnnots : (memData?.annotations || []);
        const assignedGt = memData?.gt_name || stem;

        loadedImages.push({
          id: null,
          filename: rawFilename,
          localUrl,
          fileHandle: fileObj,
          gt_name: assignedGt,
          annotations: finalAnnots,
          is_annotated: finalAnnots.length > 0,
        });
      }

      // Ensure unique GT names across all loaded images from ZIP
      const zipNameSet = new Set();
      loadedImages.forEach((item, idx) => {
        let candidate = item.gt_name || `image_${idx + 1}`;
        let base = candidate;
        let counter = 1;
        while (zipNameSet.has(candidate.toLowerCase())) {
          candidate = `${base}_${counter}`;
          counter++;
        }
        zipNameSet.add(candidate.toLowerCase());
        item.gt_name = candidate;
      });

      setImages(loadedImages);
      setSelectedImageIndex(0);
      setZoomMode('fit');
      setZoomScale(1);
      setUploadProgress(80);
      setUploadStatus('กำลังอัปโหลดและบันทึกชุดข้อมูลไปยังเซิร์ฟเวอร์...');

      // 2. Upload zip to backend for database persistence and server-side YOLO manifest
      try {
        const uploadRes = await uploadDatasetZip(file);
        if (uploadRes && uploadRes.dataset) {
          if (setActiveDataset) setActiveDataset(uploadRes.dataset);
          if (uploadRes.uploaded && uploadRes.uploaded.length > 0) {
            const idMap = new Map();
            uploadRes.uploaded.forEach((u) => idMap.set(u.filename, u.id));
            setImages((prev) =>
              prev.map((img) => ({
                ...img,
                id: idMap.get(img.filename) || img.id,
                dataset_id: uploadRes.dataset.id,
              }))
            );
          }
        }
      } catch (beErr) {
        console.warn('Backend zip upload note:', beErr.message);
      }

      setUploadProgress(100);
      setUploadStatus(`นำเข้ารูปภาพจากไฟล์ ZIP สำเร็จทั้งหมด ${loadedImages.length} รูป!`);
      setTimeout(() => {
        setUploading(false);
        setUploadProgress(0);
        setUploadStatus('');
      }, 1500);
    } catch (err) {
      alert(`ไม่สามารถแตกไฟล์ ZIP ได้: ${err.message}`);
      setUploading(false);
      setUploadProgress(0);
      setUploadStatus('');
    }
  };

  const handleZipSelect = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      processZipFile(file);
    }
    if (e.target) e.target.value = '';
  };

  // Drag and Drop Recursive Folder & ZIP Traversal
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

    // 1. Direct check for dropped .zip file
    const droppedFiles = Array.from(e.dataTransfer.files || []);
    const droppedZip = droppedFiles.find((f) => f.name.toLowerCase().endsWith('.zip'));
    if (droppedZip) {
      processZipFile(droppedZip);
      return;
    }

    const items = e.dataTransfer.items;
    if (!items || items.length === 0) {
      if (droppedFiles.length > 0) {
        handleIngestFiles(droppedFiles);
      }
      return;
    }

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

    // 2. Check if scanned files contain a .zip
    const innerZip = allFiles.find((f) => f.name.toLowerCase().endsWith('.zip'));
    if (innerZip) {
      processZipFile(innerZip);
      return;
    }

    if (allFiles.length > 0) {
      handleIngestFiles(allFiles);
    }
  };

  // --- VIEWPORT SCALING & ZOOM CONTROLS (หน้าจอสเกลแสดงภาพส่วนที่เลือกเปิด) ---
  const handleZoomIn = () => {
    setZoomMode('custom');
    setZoomScale((prev) => Math.min(4.0, +(prev + 0.25).toFixed(2)));
  };

  const handleZoomOut = () => {
    setZoomMode('custom');
    setZoomScale((prev) => Math.max(0.25, +(prev - 0.25).toFixed(2)));
  };

  const handleZoomFit = () => {
    setZoomMode('fit');
    setZoomScale(1);
  };

  const handleZoom100 = () => {
    setZoomMode('custom');
    setZoomScale(1);
  };

  const handleCanvasWheel = (e) => {
    if (e.ctrlKey || e.metaKey) {
      e.preventDefault();
      setZoomMode('custom');
      const delta = e.deltaY < 0 ? 0.15 : -0.15;
      setZoomScale((prev) => Math.max(0.25, Math.min(4.0, +(prev + delta).toFixed(2))));
    }
  };

  const getCanvasStyle = () => {
    if (zoomMode === 'fit') {
      return {
        maxWidth: '100%',
        maxHeight: '100%',
        width: 'auto',
        height: 'auto',
        objectFit: 'contain',
        cursor: 'crosshair',
        borderRadius: '4px',
        boxShadow: '0 8px 32px rgba(0, 0, 0, 0.55), 0 0 0 1px rgba(255, 255, 255, 0.15)',
        display: 'block',
        margin: 'auto',
        transition: 'width 0.15s ease, height 0.15s ease',
      };
    }
    const baseW = canvasRef.current?.width || imageDimensions.width || 800;
    const baseH = canvasRef.current?.height || imageDimensions.height || 600;
    return {
      width: `${Math.round(baseW * zoomScale)}px`,
      height: `${Math.round(baseH * zoomScale)}px`,
      maxWidth: 'none',
      maxHeight: 'none',
      cursor: 'crosshair',
      borderRadius: '4px',
      boxShadow: '0 8px 32px rgba(0, 0, 0, 0.55), 0 0 0 1px rgba(255, 255, 255, 0.15)',
      display: 'block',
      margin: 'auto',
      transition: 'width 0.15s ease, height 0.15s ease',
    };
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
      setImageDimensions({ width: img.naturalWidth || 800, height: img.naturalHeight || 600 });
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

    // 1. Draw existing annotations (High-contrast dual stroke with corner brackets & 3-ring nodes)
    annotations.forEach((ann, idx) => {
      const isSelected = selectedBoxIndex === idx;
      const color = getClassColor(ann.label);

      if (ann.segmentation && ann.segmentation.length >= 3) {
        // Multi-corner polygon contour (ทุกมุม ทุกแนว ทั้งเอียง)
        const pts = ann.segmentation.map((pt) => ({
          x: pt[0] * canvas.width,
          y: pt[1] * canvas.height,
        }));

        // Fill polygon area
        ctx.beginPath();
        ctx.moveTo(pts[0].x, pts[0].y);
        for (let i = 1; i < pts.length; i++) {
          ctx.lineTo(pts[i].x, pts[i].y);
        }
        ctx.closePath();
        ctx.fillStyle = isSelected ? `${color}45` : `${color}28`;
        ctx.fill();

        // Pass 1: Heavy dark outer contour (Ensures 100% visibility on white & bright backgrounds)
        ctx.beginPath();
        ctx.moveTo(pts[0].x, pts[0].y);
        for (let i = 1; i < pts.length; i++) {
          ctx.lineTo(pts[i].x, pts[i].y);
        }
        ctx.closePath();
        ctx.strokeStyle = '#090d16';
        ctx.lineWidth = isSelected ? 6.5 : 4.5;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.stroke();

        // Pass 2: Vivid inner colored line
        ctx.beginPath();
        ctx.moveTo(pts[0].x, pts[0].y);
        for (let i = 1; i < pts.length; i++) {
          ctx.lineTo(pts[i].x, pts[i].y);
        }
        ctx.closePath();
        ctx.strokeStyle = color;
        ctx.lineWidth = isSelected ? 3.5 : 2.5;
        ctx.stroke();

        // Pass 3: High-contrast 3-Ring Vertex Target Nodes
        pts.forEach((pt) => {
          // Ring 1: Outer dark ring
          ctx.beginPath();
          ctx.arc(pt.x, pt.y, 6.5, 0, Math.PI * 2);
          ctx.fillStyle = '#090d16';
          ctx.fill();

          // Ring 2: Middle bright white halo ring
          ctx.beginPath();
          ctx.arc(pt.x, pt.y, 4.5, 0, Math.PI * 2);
          ctx.fillStyle = '#ffffff';
          ctx.fill();

          // Ring 3: Center vivid class color dot
          ctx.beginPath();
          ctx.arc(pt.x, pt.y, 2.5, 0, Math.PI * 2);
          ctx.fillStyle = color;
          ctx.fill();
        });

        // Label Tag Chip
        const labelText = ann.label || 'object';
        ctx.font = 'bold 12px Inter, sans-serif';
        const textWidth = ctx.measureText(labelText).width;
        const badgeH = 22;
        const badgeW = textWidth + 18;

        const anchorX = Math.min(...pts.map((p) => p.x));
        const anchorY = Math.min(...pts.map((p) => p.y));
        const tagY = Math.max(0, anchorY - badgeH);

        // Dark outer badge border
        ctx.fillStyle = '#090d16';
        ctx.fillRect(anchorX - 1, tagY - 1, badgeW + 2, badgeH + 2);
        // Colored badge fill
        ctx.fillStyle = color;
        ctx.fillRect(anchorX, tagY, badgeW, badgeH);

        ctx.fillStyle = '#ffffff';
        ctx.fillText(labelText, anchorX + 9, tagY + 15);
      } else {
        // Standard Bounding Box Rect
        const x = ann.x_min * canvas.width;
        const y = ann.y_min * canvas.height;
        const w = (ann.x_max - ann.x_min) * canvas.width;
        const h = (ann.y_max - ann.y_min) * canvas.height;

        // Semi-transparent fill
        ctx.fillStyle = isSelected ? `${color}35` : `${color}20`;
        ctx.fillRect(x, y, w, h);

        // Pass 1: Outer heavy dark stroke (Ensures high visibility on bright/white backgrounds)
        ctx.strokeStyle = '#090d16';
        ctx.lineWidth = isSelected ? 6.5 : 4.5;
        ctx.strokeRect(x, y, w, h);

        // Pass 2: Inner vivid colored stroke
        ctx.strokeStyle = color;
        ctx.lineWidth = isSelected ? 3.5 : 2.5;
        ctx.strokeRect(x, y, w, h);

        // Pass 3: Four Corner High-Contrast L-Brackets (Corner Accents for crystal-clear visibility)
        const arm = Math.max(8, Math.min(24, Math.min(w, h) * 0.3));
        const drawCornerBracket = (cx, cy, dx, dy) => {
          // Outer dark bracket
          ctx.beginPath();
          ctx.moveTo(cx, cy + dy * arm);
          ctx.lineTo(cx, cy);
          ctx.lineTo(cx + dx * arm, cy);
          ctx.strokeStyle = '#090d16';
          ctx.lineWidth = 6;
          ctx.lineCap = 'square';
          ctx.stroke();

          // Inner white bracket
          ctx.beginPath();
          ctx.moveTo(cx, cy + dy * arm);
          ctx.lineTo(cx, cy);
          ctx.lineTo(cx + dx * arm, cy);
          ctx.strokeStyle = '#ffffff';
          ctx.lineWidth = 3;
          ctx.lineCap = 'square';
          ctx.stroke();
        };

        drawCornerBracket(x, y, 1, 1); // Top-Left
        drawCornerBracket(x + w, y, -1, 1); // Top-Right
        drawCornerBracket(x, y + h, 1, -1); // Bottom-Left
        drawCornerBracket(x + w, y + h, -1, -1); // Bottom-Right

        // Pass 4: Center Reticle Target Mark
        if (w > 30 && h > 30) {
          const midX = x + w / 2;
          const midY = y + h / 2;
          ctx.beginPath();
          ctx.moveTo(midX - 4, midY); ctx.lineTo(midX + 4, midY);
          ctx.moveTo(midX, midY - 4); ctx.lineTo(midX, midY + 4);
          ctx.strokeStyle = '#090d16';
          ctx.lineWidth = 3;
          ctx.stroke();
          ctx.beginPath();
          ctx.moveTo(midX - 4, midY); ctx.lineTo(midX + 4, midY);
          ctx.moveTo(midX, midY - 4); ctx.lineTo(midX, midY + 4);
          ctx.strokeStyle = '#ffffff';
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }

        // Label Tag Chip
        const labelText = ann.label || 'object';
        ctx.font = 'bold 12px Inter, sans-serif';
        const textWidth = ctx.measureText(labelText).width;
        const badgeH = 22;
        const badgeW = textWidth + 18;
        const tagY = Math.max(0, y - badgeH);

        // Dark outer badge border
        ctx.fillStyle = '#090d16';
        ctx.fillRect(x - 1, tagY - 1, badgeW + 2, badgeH + 2);
        // Colored badge fill
        ctx.fillStyle = color;
        ctx.fillRect(x, tagY, badgeW, badgeH);

        ctx.fillStyle = '#ffffff';
        ctx.fillText(labelText, x + 9, tagY + 15);
      }
    });

    // 2. Draw currently actively dragged box (in 'box' mode) with high-contrast dual dash
    if (drawMode === 'box' && currentBox && currentBox.w > 0 && currentBox.h > 0) {
      const color = getClassColor(currentClass);

      // Outer dark dash
      ctx.strokeStyle = '#090d16';
      ctx.lineWidth = 4.5;
      ctx.setLineDash([8, 4]);
      ctx.strokeRect(currentBox.x, currentBox.y, currentBox.w, currentBox.h);

      // Inner vivid dash
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.lineDashOffset = 4;
      ctx.fillStyle = `${color}30`;
      ctx.fillRect(currentBox.x, currentBox.y, currentBox.w, currentBox.h);
      ctx.strokeRect(currentBox.x, currentBox.y, currentBox.w, currentBox.h);
      ctx.setLineDash([]);
      ctx.lineDashOffset = 0;

      // Dimension chip at bottom right
      const dimText = `${Math.round(currentBox.w)} × ${Math.round(currentBox.h)} px`;
      ctx.font = 'bold 11px Inter, sans-serif';
      const dimW = ctx.measureText(dimText).width + 12;
      ctx.fillStyle = '#090d16';
      ctx.fillRect(currentBox.x + currentBox.w - dimW, currentBox.y + currentBox.h + 4, dimW, 18);
      ctx.fillStyle = '#ffffff';
      ctx.fillText(dimText, currentBox.x + currentBox.w - dimW + 6, currentBox.y + currentBox.h + 17);

      // Label chip at top
      ctx.font = 'bold 12px Inter, sans-serif';
      ctx.fillStyle = '#090d16';
      ctx.fillRect(currentBox.x - 1, Math.max(0, currentBox.y - 23), ctx.measureText(currentClass).width + 18, 22);
      ctx.fillStyle = color;
      ctx.fillRect(currentBox.x, Math.max(0, currentBox.y - 22), ctx.measureText(currentClass).width + 16, 20);
      ctx.fillStyle = '#ffffff';
      ctx.fillText(currentClass, currentBox.x + 8, Math.max(14, currentBox.y - 7));
    }

    // 3. Draw active multi-corner polygon being created (in 'polygon' mode)
    if (drawMode === 'polygon' && polygonPoints.length > 0) {
      const color = getClassColor(currentClass);

      // Dual-stroke connecting edges between placed vertices
      // Outer dark line
      ctx.beginPath();
      ctx.moveTo(polygonPoints[0].x, polygonPoints[0].y);
      for (let i = 1; i < polygonPoints.length; i++) {
        ctx.lineTo(polygonPoints[i].x, polygonPoints[i].y);
      }
      ctx.strokeStyle = '#090d16';
      ctx.lineWidth = 4.5;
      ctx.lineCap = 'round';
      ctx.stroke();

      // Inner vivid line
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
        // Outer dark dash
        ctx.beginPath();
        ctx.setLineDash([7, 4]);
        const lastPt = polygonPoints[polygonPoints.length - 1];
        ctx.moveTo(lastPt.x, lastPt.y);
        ctx.lineTo(cursorPos.x, cursorPos.y);
        ctx.strokeStyle = '#090d16';
        ctx.lineWidth = 3.5;
        ctx.stroke();

        // Inner vivid dash
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.lineDashOffset = 3;
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.lineDashOffset = 0;
      }

      // Draw 3-Ring numbered handles on vertices
      polygonPoints.forEach((pt, i) => {
        // Outer dark ring
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 7, 0, Math.PI * 2);
        ctx.fillStyle = '#090d16';
        ctx.fill();

        // Middle white ring
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 5, 0, Math.PI * 2);
        ctx.fillStyle = '#ffffff';
        ctx.fill();

        // Center colored dot
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 3, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();

        // Vertex number badge
        ctx.font = 'bold 11px Inter, sans-serif';
        const numText = String(i + 1);
        ctx.fillStyle = '#090d16';
        ctx.fillText(numText, pt.x + 9, pt.y - 3);
        ctx.fillStyle = '#ffffff';
        ctx.fillText(numText, pt.x + 8, pt.y - 4);
      });

      // Highlight first point if cursor is close to it to close the polygon
      if (polygonPoints.length >= 3 && cursorPos) {
        const dist = Math.hypot(cursorPos.x - polygonPoints[0].x, cursorPos.y - polygonPoints[0].y);
        if (dist <= 22) {
          ctx.beginPath();
          ctx.arc(polygonPoints[0].x, polygonPoints[0].y, 16, 0, Math.PI * 2);
          ctx.strokeStyle = '#10b981';
          ctx.lineWidth = 4;
          ctx.stroke();

          ctx.font = 'bold 12px Inter, sans-serif';
          ctx.fillStyle = '#090d16';
          ctx.fillText('คลิกเพื่อปิดกรอบ', polygonPoints[0].x + 13, polygonPoints[0].y + 15);
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
    if (!rect.width || !rect.height) return { x: 0, y: 0 };
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
      x: Math.max(0, Math.min(canvas.width, (e.clientX - rect.left) * scaleX)),
      y: Math.max(0, Math.min(canvas.height, (e.clientY - rect.top) * scaleY)),
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

  // --- SAVE GROUND TRUTH VIA RIGHT-CLICK ONLY (คลิกขวาเพื่อบันทึก GT เท่านั้น) ---
  const handleGtNameChange = (e) => {
    const newName = e.target.value;
    setImageGtName(newName);
    const err = checkGtNameDuplicate(newName, selectedImageIndex, images);
    setGtNameError(err);
    setImages((prev) =>
      prev.map((img, idx) => (idx === selectedImageIndex ? { ...img, gt_name: newName } : img))
    );
  };

  const handleSaveByRightClick = async (e) => {
    if (e) {
      e.preventDefault();
      e.stopPropagation();
    }
    if (!selectedImage) return;

    // Auto-finish in-progress polygon if user placed >= 3 points
    let currentAnnots = [...annotations];
    if (drawMode === 'polygon' && polygonPoints.length >= 3) {
      const canvas = canvasRef.current;
      if (canvas) {
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
          currentAnnots = [...annotations, newAnn];
          setAnnotations(currentAnnots);
        }
      }
      setPolygonPoints([]);
      setCursorPos(null);
    }

    // Validate GT Name uniqueness
    const trimmedName = (imageGtName || '').trim();
    if (!trimmedName) {
      setRightClickToast({
        type: 'error',
        message: 'กรุณาพิมพ์ชื่อ GT รูปภาพนี้ก่อนคลิกขวาบันทึก',
      });
      setGtNameError('กรุณาพิมพ์ชื่อ GT สำหรับรูปภาพนี้ (ห้ามเว้นว่าง)');
      return;
    }

    const dupError = checkGtNameDuplicate(trimmedName, selectedImageIndex, images);
    if (dupError) {
      setRightClickToast({
        type: 'error',
        message: dupError,
      });
      setGtNameError(dupError);
      return;
    }

    setSavingGt(true);
    try {
      // 1. Update React state
      const updatedImages = images.map((img, idx) =>
        idx === selectedImageIndex
          ? {
              ...img,
              gt_name: trimmedName,
              annotations: currentAnnots,
              is_annotated: currentAnnots.length > 0,
            }
          : img
      );
      setImages(updatedImages);

      // 2. Persist in Web Memory (localStorage)
      const memoryKey = selectedImage.filename || `img_${selectedImageIndex}`;
      saveItemToWebMemory(memoryKey, {
        gt_name: trimmedName,
        annotations: currentAnnots,
        classList,
      });
      refreshWebMemoryCount();

      // 3. Persist to Backend if image has backend ID
      if (selectedImage.id) {
        await saveAnnotations(selectedImage.id, currentAnnots);
      }

      // 4. Trigger visual feedback toast
      setSaveFeedback(true);
      setRightClickToast({
        type: 'success',
        message: `บันทึก GT สำเร็จ (คลิกขวา): "${trimmedName}" (${currentAnnots.length} กรอบ) - บันทึกในความทรงจำเว็บแล้ว`,
      });
      setTimeout(() => {
        setSaveFeedback(false);
        setRightClickToast(null);
      }, 3500);
    } catch (err) {
      setRightClickToast({
        type: 'error',
        message: `บันทึก GT ผิดพลาด: ${err.message}`,
      });
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

  // --- DOWNLOAD ENTIRE DATASET + GT AS ZIP TO PC ---
  const handleDownloadAllGtZip = async () => {
    if (images.length === 0) {
      alert('ไม่มีรูปภาพสำหรับบันทึกและดาวน์โหลด');
      return;
    }

    setSavingGt(true);
    try {
      const zip = new JSZip();

      // 1. classes.txt
      const classesContent = classList.join('\n') + '\n';
      zip.file('classes.txt', classesContent);

      // 2. dataset.yaml
      const yamlContent = `path: ./
train: images
val: images
names:
${classList.map((c, i) => `  ${i}: ${c}`).join('\n')}
nc: ${classList.length}
`;
      zip.file('dataset.yaml', yamlContent);

      // 3. Add images/ and labels/
      for (let i = 0; i < images.length; i++) {
        const img = images[i];
        const rawName = img.filename || `image_${i + 1}.jpg`;
        const stem = rawName.replace(/\.[^/.]+$/, '');

        // Fetch image blob
        let imgBlob = img.fileHandle;
        if (!imgBlob && img.localUrl) {
          try {
            const resp = await fetch(img.localUrl);
            imgBlob = await resp.blob();
          } catch (e) {}
        } else if (!imgBlob && img.id) {
          try {
            const resp = await fetch(`${API_BASE_URL}/api/v1/datasets/images/${img.id}/file`);
            imgBlob = await resp.blob();
          } catch (e) {}
        }

        if (imgBlob) {
          zip.file(`images/${rawName}`, imgBlob);
        }

        // Generate GT txt lines
        const imgAnnots = (i === selectedImageIndex ? annotations : img.annotations) || [];
        const lines = imgAnnots.map((ann) => {
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

        zip.file(`labels/${stem}.txt`, lines.join('\n') + (lines.length ? '\n' : ''));
      }

      const zipBlob = await zip.generateAsync({ type: 'blob' });
      const downloadUrl = URL.createObjectURL(zipBlob);
      const link = document.createElement('a');
      link.href = downloadUrl;
      const safeName = (activeDataset?.name || 'dataset').replace(/[^a-zA-Z0-9_\-\u0E00-\u0E7F]/g, '_');
      link.download = `${safeName}_ground_truth.zip`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(downloadUrl);

      setSaveFeedback(true);
      setTimeout(() => setSaveFeedback(false), 2500);
    } catch (err) {
      alert(`ดาวน์โหลด GT ZIP ผิดพลาด: ${err.message}`);
    } finally {
      setSavingGt(false);
    }
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
      alert('กรุณาโหลดรูปภาพก่อนเริ่มการเทรน');
      return;
    }

    // 1. Validate that GT names are not duplicated across images
    const dupMap = new Map();
    for (let i = 0; i < images.length; i++) {
      const name = (images[i].gt_name || '').trim().toLowerCase();
      if (!name) continue;
      if (dupMap.has(name)) {
        const prevIdx = dupMap.get(name);
        alert(
          `ไม่สามารถนำไปเทรนได้: พบชื่อ GT ซ้ำกัน ("${images[i].gt_name}")\nที่รูปที่ ${prevIdx + 1} (${images[prevIdx].filename}) และรูปที่ ${i + 1} (${images[i].filename})\nกรุณาแก้ไขชื่อ GT ให้ไม่ซ้ำกันก่อนไปเทรน`
        );
        setSelectedImageIndex(i);
        return;
      }
      dupMap.set(name, i);
    }

    // 2. Verify that there is at least one image with GT annotations
    const annotatedImgs = images.filter(
      (img) => (img.annotations && img.annotations.length > 0) || img.is_annotated
    );
    if (annotatedImgs.length === 0) {
      alert(
        'ยังไม่มีรูปภาพใดบันทึก Ground Truth (GT) เลย\nกรุณาตีกรอบและคลิกขวาบนภาพเพื่อบันทึก GT อย่างน้อย 1 รูปก่อนนำไปเทรน'
      );
      return;
    }

    setBundling(true);
    try {
      // 3. Ensure all annotations in state are synced to backend for all images with IDs
      for (const img of images) {
        if (img.id && img.annotations && img.annotations.length > 0) {
          try {
            await saveAnnotations(img.id, img.annotations);
          } catch (e) {
            console.warn(`Could not save GT for image ${img.id}:`, e);
          }
        }
      }

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
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', flex: 1, minHeight: 0 }}>
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
      <input
        type="file"
        ref={zipInputRef}
        accept=".zip"
        style={{ display: 'none' }}
        onChange={handleZipSelect}
      />

      {/* Top Banner: Ingestion & Proceed to Training Action */}
      <div
        className="card"
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '8px',
          padding: '8px 14px',
          flexWrap: 'wrap',
          gap: '10px',
          flexShrink: 0,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
          <button
            className="btn btn-primary"
            onClick={() => zipInputRef.current?.click()}
            disabled={uploading}
            title="อัปโหลดไฟล์ ZIP รูปภาพวัตถุเพื่อแตกไฟล์และปรับสเกลแสดงผลทันที"
            style={{ fontWeight: 600 }}
          >
            <Archive size={15} /> อัปโหลดไฟล์ ZIP รูปภาพวัตถุ
          </button>
          <button
            className="btn btn-secondary"
            onClick={() => folderInputRef.current?.click()}
            disabled={uploading}
            style={{ fontWeight: 500 }}
          >
            <FolderUp size={15} /> โหลดโฟลเดอร์จากเครื่อง
          </button>
          <button
            className="btn btn-secondary"
            onClick={() => filesInputRef.current?.click()}
            disabled={uploading}
          >
            <Upload size={15} /> เลือกเฉพาะไฟล์ภาพ
          </button>

          {images.length > 0 && (
            <button
              className="btn btn-ghost btn-sm"
              onClick={() => {
                if (window.confirm('คุณต้องการกลับสู่หน้าแรกเพื่อเริ่มอัปโหลดไฟล์ ZIP ใหม่หรือไม่?')) {
                  setImages([]);
                  if (setActiveDataset) setActiveDataset(null);
                }
              }}
              title="กลับสู่หน้าระบบอัปโหลดไฟล์ ZIP เริ่มต้น"
              style={{ color: 'var(--text-secondary)', fontSize: '12px' }}
            >
              <RotateCcw size={13} /> อัปโหลดชุดใหม่
            </button>
          )}

          {images.length > 0 && (
            <div style={{ fontSize: '13px', color: 'var(--text-secondary)', marginLeft: '6px', display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
              <span>ทั้งหมด: <strong>{images.length}</strong> รูป</span>
              <span>|</span>
              <span>
                บันทึก GT แล้ว:{' '}
                <strong style={{ color: 'var(--accent-success)' }}>{annotatedCount}</strong> รูป (
                {Math.round((annotatedCount / images.length) * 100)}%)
              </span>
              <span>|</span>
              <span className="badge badge-primary" style={{ fontSize: '11px', display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
                <Database size={12} /> ความทรงจำเว็บ: {webMemoryCount} รูป
              </span>
            </div>
          )}
        </div>

        {/* PROCEED TO TRAINING & GT DOWNLOAD BUTTONS */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          {images.length > 0 && (
            <button
              className="btn btn-secondary"
              onClick={handleDownloadAllGtZip}
              disabled={savingGt}
              title="เซฟไฟล์ GT โดยโหลดทั้งโฟลเดอร์ลงเครื่องเป็นไฟล์ .ZIP เพื่อนำไปเทรน"
              style={{ fontWeight: 600 }}
            >
              <Download size={15} />
              {savingGt ? 'กำลังจัดเตรียม ZIP...' : 'บันทึกและดาวน์โหลด GT (.ZIP)'}
            </button>
          )}

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
        /* Empty State Hero ZIP Upload Landing View */
        <div
          style={{
            flex: 1,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '24px',
            overflowY: 'auto',
          }}
        >
          <div
            className={`dropzone-container ${isDragOver ? 'drag-over' : ''}`}
            style={{
              width: '100%',
              maxWidth: '720px',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              padding: '48px 32px',
              backgroundColor: isDragOver ? '#eef2ff' : '#ffffff',
              borderColor: isDragOver ? 'var(--accent-primary)' : '#cbd5e1',
              boxShadow: 'var(--shadow-md)',
              cursor: 'pointer',
            }}
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragOver(true);
            }}
            onDragLeave={() => setIsDragOver(false)}
            onDrop={handleDrop}
            onClick={() => zipInputRef.current?.click()}
          >
            <div
              style={{
                width: '76px',
                height: '76px',
                borderRadius: '50%',
                backgroundColor: '#eef2ff',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                marginBottom: '20px',
                boxShadow: '0 4px 14px rgba(79, 70, 229, 0.15)',
              }}
            >
              <Archive size={38} color="var(--accent-primary)" />
            </div>

            <h2
              style={{
                fontSize: '20px',
                fontWeight: 700,
                color: 'var(--text-primary)',
                marginBottom: '8px',
                textAlign: 'center',
              }}
            >
              ระบบอัปโหลดไฟล์ ZIP รูปภาพวัตถุเพื่อเริ่มการเรียนรู้
            </h2>

            <p
              style={{
                fontSize: '13px',
                color: 'var(--text-secondary)',
                maxWidth: '560px',
                lineHeight: 1.6,
                textAlign: 'center',
                marginBottom: '24px',
              }}
            >
              ลากไฟล์ <strong>.zip</strong> ที่บรรจุรูปภาพวัตถุมาวางที่นี่ หรือคลิกปุ่มเพื่อเลือกไฟล์จากเครื่อง
              ระบบจะแตกไฟล์รูปภาพอัตโนมัติ พร้อมตรวจหาไฟล์ Ground Truth (.txt)
              และปรับสเกลหน้าจอแสดงภาพให้พร้อมตีกรอบเรียนรู้ทันที
            </p>

            <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap', justifyContent: 'center' }}>
              <button
                className="btn btn-primary btn-lg"
                onClick={(e) => {
                  e.stopPropagation();
                  zipInputRef.current?.click();
                }}
                style={{
                  fontWeight: 600,
                  padding: '10px 22px',
                  fontSize: '14px',
                  boxShadow: '0 4px 12px rgba(79, 70, 229, 0.25)',
                }}
              >
                <Archive size={17} /> อัปโหลดไฟล์ ZIP รูปภาพวัตถุ (.zip)
              </button>
              <button
                className="btn btn-secondary btn-lg"
                onClick={(e) => {
                  e.stopPropagation();
                  folderInputRef.current?.click();
                }}
              >
                <FolderUp size={16} /> หรือเลือกโฟลเดอร์จากเครื่อง
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

            <div
              style={{
                marginTop: '22px',
                fontSize: '12px',
                color: 'var(--text-muted)',
                display: 'flex',
                gap: '16px',
                flexWrap: 'wrap',
                justifyContent: 'center',
              }}
            >
              <span>รองรับ: .zip, .jpg, .png, .webp, .bmp</span>
              <span>|</span>
              <span>แตกไฟล์และอ่าน Label บนเว็บแบบ Zero-Lag</span>
              <span>|</span>
              <span>ปรับสเกลหน้าจออัตโนมัติ</span>
            </div>
          </div>

          {/* 3 Feature Highlights Below */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
              gap: '16px',
              width: '100%',
              maxWidth: '720px',
              marginTop: '20px',
            }}
          >
            <div className="card" style={{ padding: '14px 16px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
                <Archive size={16} color="var(--accent-primary)" />
                <strong style={{ fontSize: '13px', color: 'var(--text-primary)' }}>
                  รับเข้าจากไฟล์ ZIP
                </strong>
              </div>
              <p style={{ fontSize: '12px', color: 'var(--text-secondary)', lineHeight: 1.5, margin: 0 }}>
                อัปโหลดไฟล์ zip ครั้งเดียว แตกไฟล์รูปภาพวัตถุทั้งหมดเข้าสู่ระบบ พร้อมจับคู่ไฟล์ GT ที่มีอยู่เดิม
              </p>
            </div>

            <div className="card" style={{ padding: '14px 16px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
                <Maximize2 size={16} color="#10b981" />
                <strong style={{ fontSize: '13px', color: 'var(--text-primary)' }}>
                  หน้าจอสเกลแสดงภาพอัจฉริยะ
                </strong>
              </div>
              <p style={{ fontSize: '12px', color: 'var(--text-secondary)', lineHeight: 1.5, margin: 0 }}>
                ระบบปรับสเกลภาพให้พอดีจออัตโนมัติ หรือซูม 100% และซูมเข้า-ออกเพื่อตีกรอบวัตถุได้อย่างแม่นยำ
              </p>
            </div>

            <div className="card" style={{ padding: '14px 16px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
                <Pentagon size={16} color="#f59e0b" />
                <strong style={{ fontSize: '13px', color: 'var(--text-primary)' }}>
                  ตีกรอบเรียนรู้ได้ทุกมุม
                </strong>
              </div>
              <p style={{ fontSize: '12px', color: 'var(--text-secondary)', lineHeight: 1.5, margin: 0 }}>
                คลิกซ้ายตามมุมรอบวัตถุได้ทุกแนว ทั้งแนวเอียง หรือลากสี่เหลี่ยม แล้วบันทึกมัดรวมไปเทรน AI
              </p>
            </div>
          </div>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0, overflow: 'hidden' }}>
          {/* Horizontal Image Reel & Stepper Bar (ย้ายแถบเครื่องมือรูปภาพมาไว้ขอบบนสุด) */}
          <div className="studio-top-reel">
            {/* Stepper Navigation */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '4px', flexShrink: 0 }}>
              <button
                className="btn btn-sm btn-secondary"
                onClick={handlePrevImage}
                disabled={selectedImageIndex === 0}
                title="รูปก่อนหน้า (หรือกดลูกศรซ้ายบนคีย์บอร์ด)"
                style={{ padding: '3px 8px' }}
              >
                <ChevronLeft size={14} />
              </button>
              <span
                style={{
                  fontSize: '12px',
                  fontWeight: 700,
                  minWidth: '68px',
                  textAlign: 'center',
                  color: 'var(--text-primary)',
                }}
              >
                {selectedImageIndex + 1} / {images.length}
              </span>
              <button
                className="btn btn-sm btn-secondary"
                onClick={handleNextImage}
                disabled={selectedImageIndex === images.length - 1}
                title="รูปถัดไป (หรือกดลูกศรขวาบนคีย์บอร์ด)"
                style={{ padding: '3px 8px' }}
              >
                <ChevronRight size={14} />
              </button>
            </div>

            <div style={{ width: '1px', height: '22px', backgroundColor: 'var(--border-color)', margin: '0 2px', flexShrink: 0 }} />

            {/* Quick Filter Buttons */}
            <div style={{ display: 'flex', gap: '4px', flexShrink: 0 }}>
              <button
                className={`btn btn-sm ${filterMode === 'all' ? 'btn-primary' : 'btn-ghost'}`}
                style={{ padding: '3px 8px', fontSize: '11px', fontWeight: 600 }}
                onClick={() => setFilterMode('all')}
              >
                ทั้งหมด ({images.length})
              </button>
              <button
                className={`btn btn-sm ${filterMode === 'pending' ? 'btn-primary' : 'btn-ghost'}`}
                style={{ padding: '3px 8px', fontSize: '11px', fontWeight: 600 }}
                onClick={() => setFilterMode('pending')}
              >
                ยังไม่ตีกรอบ ({images.filter((img) => !img.is_annotated && (!img.annotations || img.annotations.length === 0)).length})
              </button>
              <button
                className={`btn btn-sm ${filterMode === 'annotated' ? 'btn-primary' : 'btn-ghost'}`}
                style={{ padding: '3px 8px', fontSize: '11px', fontWeight: 600 }}
                onClick={() => setFilterMode('annotated')}
              >
                GT แล้ว ({annotatedCount})
              </button>
            </div>

            <div style={{ width: '1px', height: '22px', backgroundColor: 'var(--border-color)', margin: '0 2px', flexShrink: 0 }} />

            {/* Scrollable Horizontal Reel of Image Chips */}
            <div className="image-chips-track">
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
                    className={`image-chip ${isSelected ? 'active' : ''}`}
                    onClick={() => setSelectedImageIndex(realIdx)}
                    title={img.gt_name ? `GT: ${img.gt_name} (${displayName})` : displayName}
                  >
                    <span style={{ opacity: 0.75, fontSize: '10px' }}>#{realIdx + 1}</span>
                    <span style={{ maxWidth: '140px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {img.gt_name ? `GT: ${img.gt_name}` : displayName}
                    </span>
                    {hasGt && (
                      <span className="image-chip-gt-badge">
                        GT {annCount > 0 ? `(${annCount})` : ''}
                      </span>
                    )}
                  </div>
                );
              })}
            </div>

            <div style={{ width: '1px', height: '22px', backgroundColor: 'var(--border-color)', margin: '0 2px', flexShrink: 0 }} />

            {/* Expand / Collapse Right Panel Toggle Button */}
            <button
              className="btn btn-sm btn-secondary"
              onClick={() => setIsRightPanelOpen(!isRightPanelOpen)}
              title={isRightPanelOpen ? 'ซ่อนแผงขวาเพื่อขยายกรอบภาพให้กว้างเต็มหน้าจอ' : 'แสดงแผงจัดประเภท'}
              style={{ flexShrink: 0, padding: '4px 10px', fontSize: '11px', fontWeight: 600 }}
            >
              <Maximize2 size={13} />
              {isRightPanelOpen ? 'ขยายภาพเต็มจอ' : 'แสดงแผงจัดประเภท'}
            </button>
          </div>

          {/* Main Studio Workspace: 1fr (Canvas) + 320px (Right Panel if open) */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: isRightPanelOpen ? '1fr 320px' : '1fr',
              gap: '10px',
              flex: 1,
              minHeight: 0,
              overflow: 'hidden',
            }}
          >
            {/* พื้นที่ตีกรอบ (Interactive Annotation Canvas - ขยายกรอบแสดงภาพกว้างเต็มตา) */}
            <div
              className="card"
              style={{
                display: 'flex',
                flexDirection: 'column',
                position: 'relative',
                overflow: 'hidden',
                padding: '10px 14px',
                backgroundColor: '#ffffff',
                flex: 1,
                minHeight: 0,
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
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
                <span style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)' }}>
                  รูปที่ {selectedImageIndex + 1} / {images.length}:
                </span>
                <span
                  style={{
                    fontSize: '13px',
                    color: 'var(--text-secondary)',
                    maxWidth: '220px',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                  }}
                  title={selectedImage?.original_name || selectedImage?.filename}
                >
                  {selectedImage?.original_name || selectedImage?.filename}
                </span>
                {imageDimensions.width > 0 && (
                  <span
                    style={{
                      fontSize: '11px',
                      padding: '2px 7px',
                      borderRadius: '4px',
                      backgroundColor: '#f1f5f9',
                      color: 'var(--text-secondary)',
                      border: '1px solid var(--border-color)',
                    }}
                  >
                    {imageDimensions.width} &times; {imageDimensions.height} px
                  </span>
                )}
              </div>

              <div style={{ display: 'flex', gap: '6px', alignItems: 'center', flexWrap: 'wrap' }}>
                {/* Scale & Zoom Controls (หน้าจอสเกลแสดงภาพส่วนที่เลือกเปิด) */}
                <div
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    borderRadius: '6px',
                    border: '1px solid var(--border-color)',
                    overflow: 'hidden',
                    backgroundColor: '#ffffff',
                  }}
                >
                  <button
                    className={`btn btn-sm ${zoomMode === 'fit' ? 'btn-primary' : 'btn-ghost'}`}
                    style={{ borderRadius: 0, padding: '4px 8px', fontSize: '11px', fontWeight: 600, boxShadow: 'none' }}
                    onClick={handleZoomFit}
                    title="ปรับสเกลภาพให้พอดีหน้าจอแสดงผลอัตโนมัติ"
                  >
                    <Maximize2 size={12} /> พอดีจอ (Fit)
                  </button>
                  <button
                    className={`btn btn-sm ${zoomMode === 'custom' && zoomScale === 1 ? 'btn-primary' : 'btn-ghost'}`}
                    style={{ borderRadius: 0, padding: '4px 8px', fontSize: '11px', fontWeight: 600, boxShadow: 'none' }}
                    onClick={handleZoom100}
                    title="ขนาดภาพจริง 100% (1:1 Pixel)"
                  >
                    100%
                  </button>
                  <button
                    className="btn btn-sm btn-ghost"
                    style={{ borderRadius: 0, padding: '4px 8px', fontSize: '11px', boxShadow: 'none' }}
                    onClick={handleZoomOut}
                    disabled={zoomScale <= 0.25}
                    title="ซูมออก (-)"
                  >
                    <ZoomOut size={12} />
                  </button>
                  <span
                    style={{
                      padding: '0 6px',
                      fontSize: '11px',
                      fontWeight: 600,
                      color: 'var(--text-secondary)',
                      minWidth: '42px',
                      textAlign: 'center',
                      userSelect: 'none',
                    }}
                  >
                    {zoomMode === 'fit' ? 'Auto' : `${Math.round(zoomScale * 100)}%`}
                  </span>
                  <button
                    className="btn btn-sm btn-ghost"
                    style={{ borderRadius: 0, padding: '4px 8px', fontSize: '11px', boxShadow: 'none' }}
                    onClick={handleZoomIn}
                    disabled={zoomScale >= 4.0}
                    title="ซูมเข้า (+)"
                  >
                    <ZoomIn size={12} />
                  </button>
                </div>

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
                    <Pentagon size={13} /> โหมดหลายมุม (ทุกแนว/เอียง)
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
                  {autoDetecting ? 'กำลังดีเทค...' : 'ดีเทคอัตโนมัติ'}
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

            {/* Canvas Viewport (หน้าจอสเกลแสดงภาพส่วนที่เลือกเปิด - กรอบภาพขยายคมชัดเต็มตา) */}
            <div
              ref={viewportRef}
              onWheel={handleCanvasWheel}
              onContextMenu={handleSaveByRightClick}
              style={{
                flex: 1,
                minHeight: '440px',
                height: '100%',
                display: 'flex',
                alignItems: zoomMode === 'fit' ? 'center' : 'flex-start',
                justifyContent: zoomMode === 'fit' ? 'center' : 'flex-start',
                overflow: 'auto',
                position: 'relative',
                background: 'radial-gradient(circle, #334155 1.5px, transparent 1.5px) 0 0 / 22px 22px, #0b1329',
                borderRadius: '8px',
                border: saveFeedback ? '2px solid #10b981' : '2px solid #1e293b',
                padding: '12px',
                boxShadow: 'inset 0 2px 10px rgba(0, 0, 0, 0.5)',
                transition: 'border 0.2s ease',
              }}
            >
              {/* Floating Animated Toast for Right-Click Save */}
              {rightClickToast && (
                <div
                  style={{
                    position: 'absolute',
                    top: '16px',
                    left: '50%',
                    transform: 'translateX(-50%)',
                    zIndex: 40,
                    backgroundColor: rightClickToast.type === 'error' ? '#ef4444' : '#059669',
                    color: '#ffffff',
                    padding: '8px 20px',
                    borderRadius: '24px',
                    boxShadow: '0 8px 28px rgba(0, 0, 0, 0.35)',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    fontSize: '13px',
                    fontWeight: 600,
                    pointerEvents: 'none',
                    animation: 'fadeIn 0.2s ease',
                  }}
                >
                  {rightClickToast.type === 'error' ? (
                    <AlertTriangle size={16} />
                  ) : (
                    <CheckCircle2 size={16} />
                  )}
                  <span>{rightClickToast.message}</span>
                </div>
              )}

              {selectedImage ? (
                <canvas
                  ref={canvasRef}
                  onClick={handleCanvasClick}
                  onMouseDown={handleMouseDown}
                  onMouseMove={handleMouseMove}
                  onMouseUp={handleMouseUp}
                  onDoubleClick={handleDoubleClick}
                  onMouseLeave={() => setCursorPos(null)}
                  onContextMenu={handleSaveByRightClick}
                  style={getCanvasStyle()}
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

            {/* Canvas Bottom Hint & Scale Status */}
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginTop: '8px',
                fontSize: '11px',
                color: 'var(--text-muted)',
                flexWrap: 'wrap',
                gap: '6px',
              }}
            >
              {drawMode === 'polygon' ? (
                <span>
                  โหมดหลายมุม: คลิกซ้ายตามแต่ละมุมรอบวัตถุ (เอียงได้ทุกแนว) แล้วคลิกจุดเริ่มต้น / ดับเบิ้ลคลิก เพื่อปิดกรอบ
                </span>
              ) : (
                <span>คลิกและลากเมาส์บนภาพเพื่อตีกรอบสี่เหลี่ยม (Draw bounding box)</span>
              )}
              <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
                <span>
                  สเกลภาพ:{' '}
                  <strong>{zoomMode === 'fit' ? 'พอดีหน้าจอ (Fit)' : `${Math.round(zoomScale * 100)}%`}</strong>{' '}
                  (กด Ctrl + ลูกกลิ้งเมาส์ เพื่อซูม)
                </span>
                <span>
                  ประเภทปัจจุบัน: <strong style={{ color: getClassColor(currentClass) }}>{currentClass}</strong>
                </span>
              </div>
            </div>
          </div>

          {/* Right Panel: จัดประเภทออปเจค & บันทึกไฟล์ GT */}
          {isRightPanelOpen && (
            <div
              className="card"
              style={{
                display: 'flex',
                flexDirection: 'column',
                overflow: 'hidden',
                padding: '14px',
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
              <h4 style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-primary)', margin: 0 }}>
                จัดประเภทออปเจค & บันทึก GT
              </h4>
              {saveFeedback && (
                <span className="badge badge-success" style={{ fontSize: '11px' }}>
                  <Check size={12} /> บันทึกแล้ว
                </span>
              )}
            </div>

            {/* Unique GT Name for this image (ห้ามซ้ำกันเด็ดขาด) */}
            <div
              style={{
                padding: '12px',
                borderRadius: 'var(--radius-sm)',
                backgroundColor: gtNameError ? '#fef2f2' : '#f8fafc',
                border: `1.5px solid ${gtNameError ? '#ef4444' : '#e2e8f0'}`,
                marginBottom: '12px',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                <label style={{ fontSize: '12px', fontWeight: 700, color: 'var(--text-primary)', margin: 0 }}>
                  ชื่อ Ground Truth รูปนี้ (GT Name)
                </label>
                <span style={{ fontSize: '10px', color: '#ef4444', fontWeight: 600 }}>
                  * ห้ามซ้ำกับรูปอื่น
                </span>
              </div>
              <div style={{ position: 'relative' }}>
                <input
                  type="text"
                  className="form-control"
                  style={{
                    fontSize: '13px',
                    fontWeight: 600,
                    borderColor: gtNameError ? '#ef4444' : (imageGtName ? '#10b981' : 'var(--border-color)'),
                    backgroundColor: '#ffffff',
                    paddingRight: '32px',
                  }}
                  placeholder="พิมพ์ชื่อ GT เช่น car_01, product_a..."
                  value={imageGtName}
                  onChange={handleGtNameChange}
                />
                {gtNameError ? (
                  <X size={15} color="#ef4444" style={{ position: 'absolute', right: 10, top: 10 }} />
                ) : imageGtName ? (
                  <Check size={15} color="#10b981" style={{ position: 'absolute', right: 10, top: 10 }} />
                ) : null}
              </div>
              {gtNameError ? (
                <div style={{ fontSize: '11px', color: '#ef4444', marginTop: '4px', fontWeight: 600 }}>
                  {gtNameError}
                </div>
              ) : (
                <div style={{ fontSize: '11px', color: '#10b981', marginTop: '4px', display: 'flex', justifyContent: 'space-between' }}>
                  <span>ชื่อ GT ไม่ซ้ำ (พร้อมบันทึกด้วยคลิกขวา)</span>
                  {selectedImage?.is_annotated && (
                    <span style={{ fontWeight: 600 }}>เก็บบันทึกแล้ว</span>
                  )}
                </div>
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

            {/* Section 3: Right-Click GT Save Instruction & Status (ปุ่มกดถูกแทนที่ด้วยคลิกขวา) */}
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
              <div
                style={{
                  padding: '12px 14px',
                  borderRadius: 'var(--radius-sm)',
                  backgroundColor: saveFeedback ? '#ecfdf5' : '#f8fafc',
                  border: `1.5px solid ${saveFeedback ? '#10b981' : '#e2e8f0'}`,
                  transition: 'all 0.2s ease',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
                  <MousePointerClick size={16} color={saveFeedback ? '#10b981' : 'var(--accent-primary)'} />
                  <strong style={{ fontSize: '12px', color: saveFeedback ? '#065f46' : 'var(--text-primary)' }}>
                    วิธีบันทึก GT: คลิกขวาบนภาพ
                  </strong>
                </div>
                <p style={{ fontSize: '11px', color: 'var(--text-secondary)', margin: 0, lineHeight: 1.5 }}>
                  ลากกรอบเสร็จแล้ว ให้ <strong>คลิกขวาที่ภาพ</strong> เพื่อบันทึก Ground Truth รูปนี้ลงในความทรงจำเว็บทันที (ไม่ใช้ปุ่มกด)
                </p>
                <div style={{ marginTop: '8px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '11px' }}>
                  <span>สถานะรูปนี้:</span>
                  {(selectedImage?.is_annotated || annotations.length > 0) ? (
                    <span className="badge badge-success" style={{ fontSize: '10px' }}>
                      บันทึก GT แล้ว ({annotations.length} กรอบ)
                    </span>
                  ) : (
                    <span style={{ color: 'var(--text-muted)' }}>รอคลิกขวาเพื่อบันทึก</span>
                  )}
                </div>
              </div>

              {/* PACKAGE & TRAIN BUTTON */}
              <button
                className="btn btn-lg"
                style={{
                  width: '100%',
                  marginTop: '4px',
                  background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
                  color: '#ffffff',
                  fontWeight: 600,
                  boxShadow: '0 4px 12px rgba(16, 185, 129, 0.35)',
                }}
                onClick={handlePackageAndTrain}
                disabled={images.length === 0 || bundling}
                title="มัดรวมไฟล์ภาพและไฟล์ GT ทั้งหมดที่บันทึกไว้ในเว็บ แล้วส่งไปเทรนโมเดล AI"
              >
                <Play size={16} />
                {bundling ? 'กำลังมัดรวมข้อมูล GT...' : 'มัดรวมไฟล์ GT และไปเทรนโมเดล'}
              </button>
            </div>
          </div>
          )}
        </div>
      </div>
      )}
    </div>
  );
}
