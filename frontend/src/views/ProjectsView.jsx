import React, { useState } from 'react';
import { FolderPlus, CheckCircle2, Folder, Calendar } from 'lucide-react';
import { createProject } from '../api/client';

export default function ProjectsView({
  projects = [],
  activeProject,
  setActiveProject,
  onRefreshProjects,
}) {
  const [showModal, setShowModal] = useState(false);
  const [name, setName] = useState('');
  const [taskType, setTaskType] = useState('detection');
  const [description, setDescription] = useState('');
  const [loading, setLoading] = useState(false);

  const handleCreate = async (e) => {
    e.preventDefault();
    if (!name.trim()) return;
    setLoading(true);
    try {
      const created = await createProject({
        name: name.trim(),
        task_type: taskType,
        description: description.trim(),
      });
      setShowModal(false);
      setName('');
      setDescription('');
      await onRefreshProjects();
      setActiveProject(created);
    } catch (err) {
      alert(`Error creating project: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
        <div>
          <h2 style={{ fontSize: '18px', fontWeight: 600 }}>Projects</h2>
          <p style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
            Organize your datasets, annotations, and training models.
          </p>
        </div>
        <button className="btn btn-primary" onClick={() => setShowModal(true)}>
          <FolderPlus size={16} />
          Create Project
        </button>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: '16px' }}>
        {projects.map((proj) => {
          const isActive = activeProject?.id === proj.id;
          return (
            <div
              key={proj.id}
              className="card"
              style={{
                borderColor: isActive ? 'var(--accent-primary)' : 'var(--border-color)',
                boxShadow: isActive ? '0 0 16px rgba(99, 102, 241, 0.25)' : 'none',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '12px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <div className="stat-icon" style={{ width: '38px', height: '38px' }}>
                    <Folder size={18} />
                  </div>
                  <div>
                    <h3 style={{ fontSize: '15px', fontWeight: 600 }}>{proj.name}</h3>
                    <span className="badge badge-primary" style={{ marginTop: '4px' }}>
                      {proj.task_type}
                    </span>
                  </div>
                </div>
                {isActive && (
                  <span className="badge badge-success">
                    <CheckCircle2 size={12} /> Active
                  </span>
                )}
              </div>

              <p style={{ fontSize: '13px', color: 'var(--text-secondary)', minHeight: '38px', marginBottom: '16px' }}>
                {proj.description || 'No description provided.'}
              </p>

              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderTop: '1px solid var(--border-color)', paddingTop: '12px' }}>
                <span style={{ fontSize: '11px', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <Calendar size={12} />
                  {new Date(proj.created_at).toLocaleDateString()}
                </span>
                {!isActive ? (
                  <button
                    className="btn btn-sm btn-secondary"
                    onClick={() => setActiveProject(proj)}
                  >
                    Select Project
                  </button>
                ) : (
                  <span style={{ fontSize: '12px', color: 'var(--accent-success)', fontWeight: 500 }}>
                    Selected
                  </span>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {showModal && (
        <div className="modal-overlay">
          <div className="modal-content">
            <div className="modal-header">
              <h3 style={{ fontSize: '16px', fontWeight: 600 }}>New Project</h3>
              <button
                style={{ background: 'transparent', border: 'none', color: '#fff', cursor: 'pointer', fontSize: '18px' }}
                onClick={() => setShowModal(false)}
              >
                ✕
              </button>
            </div>
            <form onSubmit={handleCreate}>
              <div className="modal-body">
                <div className="form-group">
                  <label className="form-label">Project Name</label>
                  <input
                    className="form-control"
                    placeholder="e.g. Defect Detection"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    required
                  />
                </div>
                <div className="form-group">
                  <label className="form-label">Task Type</label>
                  <select
                    className="form-control"
                    value={taskType}
                    onChange={(e) => setTaskType(e.target.value)}
                  >
                    <option value="detection">Object Detection (Bounding Boxes)</option>
                    <option value="classification">Classification (Single/Multi-label)</option>
                  </select>
                </div>
                <div className="form-group">
                  <label className="form-label">Description</label>
                  <textarea
                    className="form-control"
                    rows={3}
                    placeholder="Project objectives..."
                    value={description}
                    onChange={(e) => setDescription(e.target.value)}
                  />
                </div>
              </div>
              <div className="modal-footer">
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={() => setShowModal(false)}
                >
                  Cancel
                </button>
                <button type="submit" className="btn btn-primary" disabled={loading}>
                  {loading ? 'Creating...' : 'Create Project'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
