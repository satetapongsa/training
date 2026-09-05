import React, { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import Topbar from './components/Topbar';
import StudioView from './views/StudioView';
import TrainingView from './views/TrainingView';
import ModelsView from './views/ModelsView';
import InferenceView from './views/InferenceView';
import {
  getDatasets,
  getTrainingRuns,
  getWsUrl,
} from './api/client';

export default function App() {
  const [activeTab, setActiveTab] = useState('studio');
  const [datasets, setDatasets] = useState([]);
  const [activeDataset, setActiveDataset] = useState(null);
  const [models, setModels] = useState([]);
  const [preselectedModel, setPreselectedModel] = useState(null);
  const [wsConnected, setWsConnected] = useState(false);

  // Initial data loading
  useEffect(() => {
    loadAllData();
    initWebSocket();
  }, []);

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
      const dsList = await getDatasets();
      setDatasets(dsList || []);
      if (dsList && dsList.length > 0 && !activeDataset) {
        setActiveDataset(dsList[0]);
      }
      const modelList = await getTrainingRuns();
      setModels(modelList || []);
    } catch (err) {
      console.error('Error loading initial datasets/models:', err);
    }
  };

  const handleProceedToTraining = (dataset) => {
    if (dataset) {
      setActiveDataset(dataset);
      // Refresh datasets list so newly created dataset is available in TrainingView dropdown
      getDatasets().then((list) => {
        setDatasets(list || []);
      });
    }
    setActiveTab('training');
  };

  return (
    <div className="app-container">
      <Sidebar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
      />

      <main className="main-wrapper">
        <Topbar activeTab={activeTab} wsConnected={wsConnected} />

        <div className="content-body">
          {activeTab === 'studio' && (
            <StudioView
              activeDataset={activeDataset}
              setActiveDataset={setActiveDataset}
              onProceedToTraining={handleProceedToTraining}
            />
          )}

          {activeTab === 'training' && (
            <TrainingView
              datasets={datasets}
              activeDataset={activeDataset}
              onTrainingCompleted={() => {
                getTrainingRuns().then((list) => setModels(list || []));
              }}
            />
          )}

          {activeTab === 'inference' && (
            <InferenceView
              preselectedModel={preselectedModel}
            />
          )}

          {activeTab === 'models' && (
            <ModelsView
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
