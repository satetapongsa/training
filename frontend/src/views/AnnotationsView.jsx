import React, { useState, useEffect, useRef } from 'react';
import { Highlighter, Save, Plus, Trash2, Tag } from 'lucide-react';
import { getAnnotations, saveAnnotations, API_BASE_URL } from '../api/client';

export default function AnnotationsView({ activeDataset, selectedImage, setSelectedImage, images = [] }) {
  const [annotations, setAnnotations] = useState([]);
  const [currentClass, setCurrentClass] = useState('object');
  const [classList, setClassList] = useState(['object', 'defect', 'item', 'circle', 'square']);
  const [newClassName, setNewClassName] = useState('');
  const [isDrawing, setIsDrawing] = useState(false);
  const [startPos, setStartPos] = useState({ x: 0, y: 0 });
  const [currentBox, setCurrentBox] = useState(null);
  const [saving, setSaving] = useState(false);

  const canvasRef = useRef(null);
  const imageObjRef = useRef(null);

  useEffect(() => {
    if (selectedImage) {
      loadAnnotations(selectedImage.id);
    } else if (images.length > 0) {
      setSelectedImage(images[0]);
    }
  }, [selectedImage, images]);

  const loadAnnotations = async (imgId) => {
    try {
      const data = await getAnnotations(imgId);
      // Map annotations to normalized format
      const formatted = (data || []).map((ann) => ({
        id: ann.id || Math.random().toString(),
        label: ann.label || ann.class_name || 'object',
        x_min: ann.x_min,
        y_min: ann.y_min,
        x_max: ann.x_max,
        y_max: ann.y_max,
      }));
      setAnnotations(formatted);
    } catch (err) {
      console.error('Failed to load annotations:', err);
      setAnnotations([]);
    }
  };

  // Draw image and bounding boxes onto canvas
  useEffect(() => {
    if (!selectedImage || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const imgUrl = selectedImage.image_url
      ? `${API_BASE_URL}${selectedImage.image_url}`
      : `${API_BASE_URL}/api/v1/datasets/images/${selectedImage.id}/file`;

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

    // Draw saved annotations
    annotations.forEach((ann, idx) => {
      const x = ann.x_min * ctx.canvas.width;
      const y = ann.y_min * ctx.canvas.height;
      const w = (ann.x_max - ann.x_min) * ctx.canvas.width;
      const h = (ann.y_max - ann.y_min) * ctx.canvas.height;

      ctx.strokeStyle = '#6366f1';
      ctx.lineWidth = 3;
      ctx.strokeRect(x, y, w, h);

      // Label background
      ctx.fillStyle = '#6366f1';
      ctx.fillRect(x, Math.max(0, y - 22), ctx.measureText(ann.label).width + 16, 22);

      // Label text
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 12px Inter, sans-serif';
      ctx.fillText(ann.label, x + 8, Math.max(14, y - 6));
    });

    // Draw current active drawing box
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

    // Minimum threshold box size
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

      setAnnotations((prev) => [...prev, newAnn]);
    }
    setCurrentBox(null);
  };

  const handleDeleteAnnotation = (index) => {
    setAnnotations((prev) => prev.filter((_, i) => i !== index));
  };

  const handleSave = async () => {
    if (!selectedImage) return;
    setSaving(true);
    try {
      await saveAnnotations(
        selectedImage.id,
        annotations.map((a) => ({
          label: a.label,
          x_min: a.x_min,
          y_min: a.y_min,
          x_max: a.x_max,
          y_max: a.y_max,
        }))
      );
      alert('Annotations successfully saved to database!');
    } catch (err) {
      alert(`Save failed: ${err.message}`);
    } finally {
      setSaving(false);
    }
  };

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

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '260px 1fr 280px', gap: '20px', height: 'calc(100vh - 120px)' }}>
      {/* Left: Image List */}
      <div className="card" style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <h4 style={{ fontSize: '14px', fontWeight: 600, marginBottom: '12px' }}>
          Dataset Images ({images.length})
        </h4>
        <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '8px' }}>
          {images.map((img) => {
            const isSelected = selectedImage?.id === img.id;
            const filename =
              img.original_name ||
              img.filename ||
              (img.file_path ? img.file_path.split(/[\\/]/).pop() : `Image #${img.id}`);
            return (
              <div
                key={img.id}
                onClick={() => setSelectedImage(img)}
                style={{
                  padding: '8px 12px',
                  borderRadius: 'var(--radius-sm)',
                  backgroundColor: isSelected ? 'rgba(99, 102, 241, 0.15)' : 'rgba(255, 255, 255, 0.02)',
                  border: `1px solid ${isSelected ? 'var(--accent-primary)' : 'var(--border-color)'}`,
                  cursor: 'pointer',
                  fontSize: '12px',
                  whiteSpace: 'nowrap',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  color: isSelected ? '#fff' : 'var(--text-secondary)',
                }}
              >
                {filename}
              </div>
            );
          })}
        </div>
      </div>

      {/* Center: Canvas Workspace */}
      <div className="card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', overflow: 'hidden', position: 'relative' }}>
        {selectedImage ? (
          <div style={{ maxWidth: '100%', maxHeight: '100%', overflow: 'auto', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <canvas
              ref={canvasRef}
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              style={{
                maxWidth: '100%',
                maxHeight: 'calc(100vh - 160px)',
                cursor: 'crosshair',
                borderRadius: 'var(--radius-sm)',
                boxShadow: '0 8px 24px rgba(0, 0, 0, 0.5)',
              }}
            />
          </div>
        ) : (
          <div style={{ color: 'var(--text-muted)' }}>Select an image to start annotating</div>
        )}
      </div>

      {/* Right: Classes & Saved Annotations */}
      <div className="card" style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
          <h4 style={{ fontSize: '14px', fontWeight: 600 }}>Annotation Tools</h4>
          <button className="btn btn-sm btn-primary" onClick={handleSave} disabled={saving || !selectedImage}>
            <Save size={13} /> {saving ? 'Saving...' : 'Save'}
          </button>
        </div>

        {/* Class selector */}
        <div className="form-group">
          <label className="form-label">Active Class Label</label>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', marginBottom: '10px' }}>
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
                  backgroundColor: currentClass === c ? 'var(--accent-primary)' : 'rgba(255, 255, 255, 0.06)',
                  color: currentClass === c ? '#fff' : 'var(--text-secondary)',
                  border: `1px solid ${currentClass === c ? 'var(--accent-primary)' : 'var(--border-color)'}`,
                }}
              >
                {c}
              </span>
            ))}
          </div>

          <form onSubmit={handleAddClass} style={{ display: 'flex', gap: '6px' }}>
            <input
              className="form-control"
              style={{ flex: 1, padding: '4px 8px', fontSize: '12px' }}
              placeholder="Add new class..."
              value={newClassName}
              onChange={(e) => setNewClassName(e.target.value)}
            />
            <button type="submit" className="btn btn-sm btn-secondary">
              <Plus size={12} />
            </button>
          </form>
        </div>

        <hr style={{ borderColor: 'var(--border-color)', margin: '14px 0' }} />

        {/* List of current annotations */}
        <h5 style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '8px' }}>
          Labels on Image ({annotations.length})
        </h5>
        <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '6px' }}>
          {annotations.map((ann, idx) => (
            <div
              key={ann.id || idx}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '6px 10px',
                borderRadius: 'var(--radius-sm)',
                backgroundColor: 'rgba(255, 255, 255, 0.03)',
                border: '1px solid var(--border-color)',
                fontSize: '12px',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <Tag size={12} color="var(--accent-primary)" />
                <span style={{ fontWeight: 600, color: '#fff' }}>{ann.label}</span>
              </div>
              <button
                onClick={() => handleDeleteAnnotation(idx)}
                style={{ background: 'transparent', border: 'none', color: '#f87171', cursor: 'pointer' }}
              >
                <Trash2 size={13} />
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
