import React, { useState, useEffect, useRef } from 'react';
import {
  FolderUp,
  Upload,
  Plus,
  Sliders,
  Image as ImageIcon,
  CheckCircle2,
  FileText,
  Trash2,
  PieChart,
} from 'lucide-react';
import {
  getDatasets,
  createDataset,
  getDatasetImages,
  splitDataset,
  uploadFilesChunked,
  API_BASE_URL,
} from '../api/client';

export default function DatasetsView({
  activeProject,
  activeDataset,
  setActiveDataset,
  onSelectImageForAnnotation,
}) {
  const [datasets, setDatasets] = useState([]);
  const [images, setImages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [showCreateModal, setShowCreateModal] = useState(false);

  // New Dataset Form
  const [newDatasetName, setNewDatasetName] = useState('');
  const [newDatasetDesc, setNewDatasetDesc] = useState('');

  // Upload state
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadStatusText, setUploadStatusText] = useState('');
  const [isDragOver, setIsDragOver] = useState(false);

  // Split state
  const [trainRatio, setTrainRatio] = useState(70);
  const [valRatio, setValRatio] = useState(20);
  const [testRatio, setTestRatio] = useState(10);
  const [splitting, setSplitting] = useState(false);

  // Hidden file inputs
  const folderInputRef = useRef(null);
  const filesInputRef = useRef(null);

  useEffect(() => {
    loadDatasets();
  }, [activeProject]);

  useEffect(() => {
    if (activeDataset) {
      loadImages(activeDataset.id);
    } else {
      setImages([]);
    }
  }, [activeDataset]);

  const loadDatasets = async () => {
    try {
      const data = await getDatasets(activeProject?.id);
      setDatasets(data);
      if (data.length > 0 && !activeDataset) {
        setActiveDataset(data[0]);
      }
    } catch (err) {
      console.error('Failed to load datasets:', err);
    }
  };

  const loadImages = async (datasetId) => {
    setLoading(true);
    try {
      const data = await getDatasetImages(datasetId);
      setImages(data);
    } catch (err) {
      console.error('Failed to load images:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleCreateDataset = async (e) => {
    e.preventDefault();
    if (!newDatasetName.trim()) return;
    try {
      const created = await createDataset({
        project_id: activeProject?.id,
        name: newDatasetName.trim(),
        description: newDatasetDesc.trim(),
      });
      setShowCreateModal(false);
      setNewDatasetName('');
      setNewDatasetDesc('');
      await loadDatasets();
      setActiveDataset(created);
    } catch (err) {
      alert(`Error creating dataset: ${err.message}`);
    }
  };

  // Process and upload a list of File handles
  const processUploadFiles = async (fileList) => {
    if (!activeDataset) {
      alert('Please select or create a dataset first!');
      return;
    }
    const files = Array.from(fileList);
    if (files.length === 0) return;

    setUploading(true);
    setUploadProgress(0);
    setUploadStatusText(`Preparing ${files.length} items...`);

    try {
      await uploadFilesChunked(activeDataset.id, files, (uploaded, total, percent) => {
        setUploadProgress(percent);
        setUploadStatusText(`Uploading & parsing: ${uploaded} / ${total} files (${percent}%)`);
      });

      setUploadStatusText(`Successfully ingested ${files.length} files!`);
      setTimeout(() => {
        setUploading(false);
        setUploadProgress(0);
        setUploadStatusText('');
      }, 1500);

      await loadImages(activeDataset.id);
    } catch (err) {
      alert(`Upload error: ${err.message}`);
      setUploading(false);
    }
  };

  // Recursive Directory Traversal for Drag and Drop
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
      processUploadFiles(allFiles);
    }
  };

  const handleSplit = async () => {
    if (!activeDataset) return;
    const total = Number(trainRatio) + Number(valRatio) + Number(testRatio);
    if (total !== 100) {
      alert('The sum of Train, Validation, and Test ratios must equal 100%');
      return;
    }

    setSplitting(true);
    try {
      await splitDataset(activeDataset.id, {
        train_ratio: trainRatio / 100,
        val_ratio: valRatio / 100,
        test_ratio: testRatio / 100,
      });
      alert('Dataset split successfully updated!');
      await loadImages(activeDataset.id);
    } catch (err) {
      alert(`Split failed: ${err.message}`);
    } finally {
      setSplitting(false);
    }
  };

  return (
    <div>
      {/* Top Bar: Dataset selector & actions */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px', flexWrap: 'wrap', gap: '12px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <select
            className="form-control"
            style={{ width: '220px', fontWeight: 600 }}
            value={activeDataset?.id || ''}
            onChange={(e) => {
              const ds = datasets.find((d) => d.id === e.target.value);
              setActiveDataset(ds || null);
            }}
          >
            {datasets.length === 0 && <option value="">No datasets found</option>}
            {datasets.map((ds) => (
              <option key={ds.id} value={ds.id}>
                {ds.name} ({ds.num_images || 0} imgs)
              </option>
            ))}
          </select>
          <button className="btn btn-sm btn-secondary" onClick={() => setShowCreateModal(true)}>
            <Plus size={14} /> New Dataset
          </button>
        </div>

        {/* Upload Buttons */}
        <div style={{ display: 'flex', gap: '10px' }}>
          {/* Hidden inputs */}
          <input
            type="file"
            ref={folderInputRef}
            webkitdirectory="true"
            directory="true"
            multiple
            style={{ display: 'none' }}
            onChange={(e) => processUploadFiles(e.target.files)}
          />
          <input
            type="file"
            ref={filesInputRef}
            multiple
            accept="image/*,.txt"
            style={{ display: 'none' }}
            onChange={(e) => processUploadFiles(e.target.files)}
          />

          <button
            className="btn btn-secondary"
            onClick={() => filesInputRef.current?.click()}
            disabled={uploading || !activeDataset}
          >
            <Upload size={15} /> Upload Images
          </button>

          <button
            className="btn btn-primary"
            onClick={() => folderInputRef.current?.click()}
            disabled={uploading || !activeDataset}
            title="Upload an entire directory of images and YOLO labels"
          >
            <FolderUp size={15} /> Upload Folder
          </button>
        </div>
      </div>

      {/* Upload Progress Bar */}
      {uploading && (
        <div className="card" style={{ marginBottom: '20px', borderColor: 'var(--accent-primary)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px', fontSize: '13px' }}>
            <span style={{ fontWeight: 500, color: '#fff' }}>{uploadStatusText}</span>
            <span style={{ fontWeight: 600, color: 'var(--accent-primary)' }}>{uploadProgress}%</span>
          </div>
          <div className="progress-track">
            <div className="progress-fill" style={{ width: `${uploadProgress}%` }} />
          </div>
        </div>
      )}

      {/* Drag & Drop Zone */}
      <div
        className={`dropzone-container ${isDragOver ? 'drag-over' : ''}`}
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragOver(true);
        }}
        onDragLeave={() => setIsDragOver(false)}
        onDrop={handleDrop}
        onClick={() => folderInputRef.current?.click()}
      >
        <FolderUp size={36} color="var(--accent-primary)" style={{ margin: '0 auto 12px auto', display: 'block' }} />
        <h4 style={{ fontSize: '15px', fontWeight: 600, color: '#fff', marginBottom: '6px' }}>
          Drag & Drop an Entire Training Folder Here
        </h4>
        <p style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
          Click or drop a directory containing images and companion YOLO (.txt) label files.
        </p>
      </div>

      {/* Dataset Split Section */}
      {activeDataset && (
        <div className="card" style={{ marginTop: '20px', marginBottom: '20px' }}>
          <div className="card-header">
            <div className="card-title">
              <PieChart size={18} color="var(--accent-secondary)" />
              Dataset Train / Validation / Test Split
            </div>
            <button className="btn btn-sm btn-secondary" onClick={handleSplit} disabled={splitting}>
              <Sliders size={14} />
              {splitting ? 'Splitting...' : 'Apply Split'}
            </button>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '16px' }}>
            <div>
              <label className="form-label" style={{ color: 'var(--accent-primary)' }}>
                Train: {trainRatio}%
              </label>
              <input
                type="range"
                min="10"
                max="90"
                value={trainRatio}
                onChange={(e) => setTrainRatio(Number(e.target.value))}
                style={{ width: '100%' }}
              />
            </div>
            <div>
              <label className="form-label" style={{ color: 'var(--accent-secondary)' }}>
                Validation: {valRatio}%
              </label>
              <input
                type="range"
                min="5"
                max="50"
                value={valRatio}
                onChange={(e) => setValRatio(Number(e.target.value))}
                style={{ width: '100%' }}
              />
            </div>
            <div>
              <label className="form-label" style={{ color: 'var(--accent-warning)' }}>
                Test: {testRatio}%
              </label>
              <input
                type="range"
                min="0"
                max="30"
                value={testRatio}
                onChange={(e) => setTestRatio(Number(e.target.value))}
                style={{ width: '100%' }}
              />
            </div>
          </div>
        </div>
      )}

      {/* Image Gallery */}
      <div style={{ marginTop: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
          <h3 style={{ fontSize: '16px', fontWeight: 600 }}>
            Dataset Images ({images.length})
          </h3>
        </div>

        {images.length === 0 ? (
          <div className="card" style={{ textAlign: 'center', padding: '40px', color: 'var(--text-muted)' }}>
            <ImageIcon size={32} style={{ margin: '0 auto 8px auto', display: 'block' }} />
            No images in this dataset yet. Use "Upload Folder" above to add your training dataset!
          </div>
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '16px' }}>
            {images.map((img) => {
              const imgUrl = img.image_url ? `${API_BASE_URL}${img.image_url}` : `${API_BASE_URL}/api/v1/datasets/images/${img.id}/file`;
              return (
                <div
                  key={img.id}
                  className="card"
                  style={{
                    padding: '10px',
                    cursor: 'pointer',
                    position: 'relative',
                    overflow: 'hidden',
                  }}
                  onClick={() => onSelectImageForAnnotation && onSelectImageForAnnotation(img)}
                >
                  <div
                    style={{
                      height: '140px',
                      backgroundColor: '#07090e',
                      borderRadius: 'var(--radius-sm)',
                      overflow: 'hidden',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      marginBottom: '8px',
                    }}
                  >
                    <img
                      src={imgUrl}
                      alt={img.file_path}
                      style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }}
                      onError={(e) => {
                        e.target.style.display = 'none';
                      }}
                    />
                  </div>
                  <div style={{ fontSize: '12px', fontWeight: 500, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {img.file_path.split(/[\\/]/).pop()}
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '6px' }}>
                    <span
                      className="badge"
                      style={{
                        background:
                          img.split_type === 'train'
                            ? 'rgba(99, 102, 241, 0.15)'
                            : img.split_type === 'val'
                            ? 'rgba(6, 182, 212, 0.15)'
                            : 'rgba(245, 158, 11, 0.15)',
                        color:
                          img.split_type === 'train'
                            ? '#a5b4fc'
                            : img.split_type === 'val'
                            ? '#67e8f9'
                            : '#fde047',
                      }}
                    >
                      {img.split_type || 'train'}
                    </span>
                    <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                      {img.annotations_count || 0} labels
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* New Dataset Modal */}
      {showCreateModal && (
        <div className="modal-overlay">
          <div className="modal-content">
            <div className="modal-header">
              <h3 style={{ fontSize: '16px', fontWeight: 600 }}>Create Dataset</h3>
              <button
                style={{ background: 'transparent', border: 'none', color: '#fff', cursor: 'pointer', fontSize: '18px' }}
                onClick={() => setShowCreateModal(false)}
              >
                ✕
              </button>
            </div>
            <form onSubmit={handleCreateDataset}>
              <div className="modal-body">
                <div className="form-group">
                  <label className="form-label">Dataset Name</label>
                  <input
                    className="form-control"
                    placeholder="e.g. Traffic Signs Dataset"
                    value={newDatasetName}
                    onChange={(e) => setNewDatasetName(e.target.value)}
                    required
                  />
                </div>
                <div className="form-group">
                  <label className="form-label">Description</label>
                  <textarea
                    className="form-control"
                    rows={3}
                    placeholder="Folder dataset details..."
                    value={newDatasetDesc}
                    onChange={(e) => setNewDatasetDesc(e.target.value)}
                  />
                </div>
              </div>
              <div className="modal-footer">
                <button type="button" className="btn btn-secondary" onClick={() => setShowCreateModal(false)}>
                  Cancel
                </button>
                <button type="submit" className="btn btn-primary">
                  Create Dataset
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
