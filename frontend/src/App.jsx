import React, { useState, useEffect, useRef } from 'react';
import Sidebar from './components/Sidebar';
import Topbar from './components/Topbar';
import StudioView from './views/StudioView';
import TrainingView from './views/TrainingView';
import ModelsView from './views/ModelsView';
import InferenceView from './views/InferenceView';
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
  const [activeTab, setActiveTab] = useState('studio');
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
        <Topbar activeTab={activeTab} wsConnected={wsConnected} />

        <div className="content-body">
          {activeTab === 'studio' && (
            <StudioView
              activeProject={activeProject}
              activeDataset={activeDataset}
              setActiveDataset={setActiveDataset}
              onProceedToTraining={(ds) => {
                if (ds) setActiveDataset(ds);
                setActiveTab('training');
              }}
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

          {activeTab === 'inference' && (
            <InferenceView
              activeProject={activeProject}
              preselectedModel={preselectedModel}
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
        </div>
      </main>
    </div>
  );
}
