import React, { useState, useEffect, useRef } from 'react';
import Sidebar from './components/Sidebar';
import Topbar from './components/Topbar';
import DashboardView from './views/DashboardView';
import ProjectsView from './views/ProjectsView';
import DatasetsView from './views/DatasetsView';
import AnnotationsView from './views/AnnotationsView';
import TrainingView from './views/TrainingView';
import ModelsView from './views/ModelsView';
import InferenceView from './views/InferenceView';
import SystemView from './views/SystemView';
import {
  getProjects,
  getDatasets,
  getTrainingRuns,
  getDatasetImages,
  getWsUrl,
  createProject,
  createDataset,
} from './api/client';

export default function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [projects, setProjects] = useState([]);
  const [activeProject, setActiveProject] = useState(null);
  const [datasets, setDatasets] = useState([]);
  const [activeDataset, setActiveDataset] = useState(null);
  const [datasetImages, setDatasetImages] = useState([]);
  const [selectedImage, setSelectedImage] = useState(null);
  const [models, setModels] = useState([]);
  const [preselectedModel, setPreselectedModel] = useState(null);
  const [wsConnected, setWsConnected] = useState(false);

  // Quick modals state from Topbar
  const [showNewProjectModal, setShowNewProjectModal] = useState(false);
  const [showNewDatasetModal, setShowNewDatasetModal] = useState(false);
  const [projectNameInput, setProjectNameInput] = useState('');
  const [projectTypeInput, setProjectTypeInput] = useState('detection');
  const [datasetNameInput, setDatasetNameInput] = useState('');

  // Initial data loading
  useEffect(() => {
    loadAllData();
    initWebSocket();
  }, []);

  useEffect(() => {
    if (activeProject) {
      loadProjectDatasets(activeProject.id);
      loadProjectModels(activeProject.id);
    }
  }, [activeProject]);

  useEffect(() => {
    if (activeDataset) {
      loadDatasetImages(activeDataset.id);
    }
  }, [activeDataset]);

  const initWebSocket = () => {
    try {
      const ws = new WebSocket(getWsUrl());
      ws.onopen = () => setWsConnected(true);
      ws.onclose = () => setWsConnected(false);
      ws.onerror = () => setWsConnected(false);
    } catch (e) {
      setWsConnected(false);
    }
  };

  const loadAllData = async () => {
    try {
      const projList = await getProjects();
      setProjects(projList);
      if (projList.length > 0 && !activeProject) {
        setActiveProject(projList[0]);
      }
      const modelList = await getTrainingRuns();
      setModels(modelList);
    } catch (err) {
      console.error('Error loading initial data:', err);
    }
  };

  const loadProjectDatasets = async (projId) => {
    try {
      const dsList = await getDatasets(projId);
      setDatasets(dsList);
      if (dsList.length > 0) {
        setActiveDataset(dsList[0]);
      } else {
        setActiveDataset(null);
      }
    } catch (err) {
      console.error('Error loading datasets:', err);
    }
  };

  const loadDatasetImages = async (dsId) => {
    try {
      const imgs = await getDatasetImages(dsId);
      setDatasetImages(imgs);
      if (imgs.length > 0) {
        setSelectedImage(imgs[0]);
      } else {
        setSelectedImage(null);
      }
    } catch (err) {
      console.error('Error loading images:', err);
    }
  };

  const loadProjectModels = async (projId) => {
    try {
      const runs = await getTrainingRuns(projId);
      setModels(runs);
    } catch (err) {
      console.error('Error loading models:', err);
    }
  };

  const handleCreateProjectModal = async (e) => {
    e.preventDefault();
    if (!projectNameInput.trim()) return;
    try {
      const created = await createProject({
        name: projectNameInput.trim(),
        task_type: projectTypeInput,
      });
      setShowNewProjectModal(false);
      setProjectNameInput('');
      await loadAllData();
      setActiveProject(created);
    } catch (err) {
      alert(`Error: ${err.message}`);
    }
  };

  const handleCreateDatasetModal = async (e) => {
    e.preventDefault();
    if (!datasetNameInput.trim()) return;
    try {
      const created = await createDataset({
        project_id: activeProject?.id,
        name: datasetNameInput.trim(),
      });
      setShowNewDatasetModal(false);
      setDatasetNameInput('');
      if (activeProject) {
        await loadProjectDatasets(activeProject.id);
      }
      setActiveDataset(created);
    } catch (err) {
      alert(`Error: ${err.message}`);
    }
  };

  return (
    <div className="app-container">
      <Sidebar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        activeProject={activeProject}
      />

      <main className="main-wrapper">
        <Topbar
          activeTab={activeTab}
          onNewProject={() => setShowNewProjectModal(true)}
          onNewDataset={() => setShowNewDatasetModal(true)}
          wsConnected={wsConnected}
        />

        <div className="content-body">
          {activeTab === 'dashboard' && (
            <DashboardView
              projects={projects}
              datasets={datasets}
              models={models}
              setActiveTab={setActiveTab}
              activeProject={activeProject}
            />
          )}

          {activeTab === 'projects' && (
            <ProjectsView
              projects={projects}
              activeProject={activeProject}
              setActiveProject={setActiveProject}
              onRefreshProjects={loadAllData}
            />
          )}

          {activeTab === 'datasets' && (
            <DatasetsView
              activeProject={activeProject}
              activeDataset={activeDataset}
              setActiveDataset={setActiveDataset}
              onSelectImageForAnnotation={(img) => {
                setSelectedImage(img);
                setActiveTab('annotations');
              }}
            />
          )}

          {activeTab === 'annotations' && (
            <AnnotationsView
              activeDataset={activeDataset}
              selectedImage={selectedImage}
              setSelectedImage={setSelectedImage}
              images={datasetImages}
            />
          )}

          {activeTab === 'training' && (
            <TrainingView
              activeProject={activeProject}
              datasets={datasets}
              activeDataset={activeDataset}
              onTrainingCompleted={() => {
                if (activeProject) loadProjectModels(activeProject.id);
              }}
            />
          )}

          {activeTab === 'models' && (
            <ModelsView
              activeProject={activeProject}
              onSelectForInference={(model) => {
                setPreselectedModel(model);
                setActiveTab('inference');
              }}
            />
          )}

          {activeTab === 'inference' && (
            <InferenceView
              activeProject={activeProject}
              preselectedModel={preselectedModel}
            />
          )}

          {activeTab === 'system' && <SystemView />}
        </div>
      </main>

      {/* Quick Project Modal */}
      {showNewProjectModal && (
        <div className="modal-overlay">
          <div className="modal-content">
            <div className="modal-header">
              <h3 style={{ fontSize: '16px', fontWeight: 600 }}>Create New Project</h3>
              <button
                style={{ background: 'transparent', border: 'none', color: '#fff', cursor: 'pointer', fontSize: '18px' }}
                onClick={() => setShowNewProjectModal(false)}
              >
                ✕
              </button>
            </div>
            <form onSubmit={handleCreateProjectModal}>
              <div className="modal-body">
                <div className="form-group">
                  <label className="form-label">Project Name</label>
                  <input
                    className="form-control"
                    placeholder="e.g. Defect Detection"
                    value={projectNameInput}
                    onChange={(e) => setProjectNameInput(e.target.value)}
                    required
                  />
                </div>
                <div className="form-group">
                  <label className="form-label">Task Type</label>
                  <select
                    className="form-control"
                    value={projectTypeInput}
                    onChange={(e) => setProjectTypeInput(e.target.value)}
                  >
                    <option value="detection">Object Detection</option>
                    <option value="classification">Classification</option>
                  </select>
                </div>
              </div>
              <div className="modal-footer">
                <button type="button" className="btn btn-secondary" onClick={() => setShowNewProjectModal(false)}>
                  Cancel
                </button>
                <button type="submit" className="btn btn-primary">
                  Create Project
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Quick Dataset Modal */}
      {showNewDatasetModal && (
        <div className="modal-overlay">
          <div className="modal-content">
            <div className="modal-header">
              <h3 style={{ fontSize: '16px', fontWeight: 600 }}>Create New Dataset</h3>
              <button
                style={{ background: 'transparent', border: 'none', color: '#fff', cursor: 'pointer', fontSize: '18px' }}
                onClick={() => setShowNewDatasetModal(false)}
              >
                ✕
              </button>
            </div>
            <form onSubmit={handleCreateDatasetModal}>
              <div className="modal-body">
                <div className="form-group">
                  <label className="form-label">Dataset Name</label>
                  <input
                    className="form-control"
                    placeholder="e.g. Products Dataset"
                    value={datasetNameInput}
                    onChange={(e) => setDatasetNameInput(e.target.value)}
                    required
                  />
                </div>
              </div>
              <div className="modal-footer">
                <button type="button" className="btn btn-secondary" onClick={() => setShowNewDatasetModal(false)}>
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
