// Production API Client for AI Vision Studio

const BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

export const API_BASE_URL = BASE_URL;

export function getWsUrl() {
  if (BASE_URL) {
    const wsProto = BASE_URL.startsWith('https') ? 'wss:' : 'ws:';
    const host = BASE_URL.replace(/^https?:\/\//, '');
    return `${wsProto}//${host}/ws/live`;
  }
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${window.location.host}/ws/live`;
}

async function request(endpoint, options = {}) {
  const url = `${BASE_URL}${endpoint}`;
  try {
    const response = await fetch(url, {
      headers: {
        Accept: 'application/json',
        ...(options.headers || {}),
      },
      ...options,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(errorData.detail || `Request failed with status ${response.status}`);
    }

    return await response.json();
  } catch (err) {
    console.error(`API Error on ${endpoint}:`, err);
    throw err;
  }
}

// --- Projects ---
export async function getProjects() {
  return request('/api/v1/projects');
}

export async function createProject(data) {
  return request('/api/v1/projects', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
}

// --- Datasets ---
export async function getDatasets(projectId) {
  const query = projectId ? `?project_id=${projectId}` : '';
  return request(`/api/v1/datasets${query}`);
}

export async function createDataset(data) {
  return request('/api/v1/datasets', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
}

export async function getDatasetImages(datasetId) {
  return request(`/api/v1/datasets/${datasetId}/images`);
}

export async function splitDataset(datasetId, splits) {
  return request(`/api/v1/datasets/${datasetId}/split`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(splits),
  });
}

// Chunked Upload for Files and Folders
export async function uploadFilesChunked(datasetId, files, onProgress) {
  const total = files.length;
  let uploaded = 0;
  let lastResult = null;
  const chunkSize = 25; // Batch 25 files per HTTP request

  for (let i = 0; i < total; i += chunkSize) {
    const chunk = files.slice(i, i + chunkSize);
    const formData = new FormData();
    chunk.forEach((file) => {
      formData.append('files', file);
    });

    const res = await fetch(`${BASE_URL}/api/v1/datasets/${datasetId}/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      throw new Error(`Upload batch failed at ${i + 1}-${Math.min(i + chunkSize, total)}`);
    }

    lastResult = await res.json();
    uploaded += chunk.length;
    if (onProgress) {
      onProgress(uploaded, total, Math.round((uploaded / total) * 100));
    }
  }

  return lastResult;
}

// --- Annotations ---
export async function getAnnotations(imageId) {
  return request(`/api/v1/annotations/${imageId}`);
}

export async function saveAnnotations(imageId, annotations) {
  return request(`/api/v1/annotations/${imageId}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(annotations),
  });
}

// --- Training ---
export async function getTrainingRuns(projectId) {
  const query = projectId ? `?project_id=${projectId}` : '';
  return request(`/api/v1/training/runs${query}`);
}

export async function startTraining(config) {
  return request('/api/v1/training/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });
}

export async function getTrainingStatus(runId) {
  return request(`/api/v1/training/status/${runId}`);
}

export async function cancelTraining(runId) {
  return request(`/api/v1/training/cancel/${runId}`, {
    method: 'POST',
  });
}

// --- Inference ---
export async function runInference(formData) {
  const url = `${BASE_URL}/api/v1/inference/detect`;
  const response = await fetch(url, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: 'Inference failed' }));
    throw new Error(err.detail || 'Inference failed');
  }

  return await response.json();
}

// --- System ---
export async function getSystemHealth() {
  return request('/health');
}
