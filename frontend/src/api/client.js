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

// Upload a single ZIP containing images and optional companion GT files
export async function uploadDatasetZip(file, datasetName, projectId) {
  const formData = new FormData();
  formData.append('file', file);
  if (datasetName) formData.append('dataset_name', datasetName);
  if (projectId) formData.append('project_id', projectId);

  const res = await fetch(`${BASE_URL}/api/v1/datasets/upload-zip`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errorData.detail || `Upload ZIP failed with status ${res.status}`);
  }

  return await res.json();
}

// Get URL for downloading dataset + GT as ZIP
export function getDatasetZipDownloadUrl(datasetId) {
  return `${BASE_URL}/api/v1/datasets/${datasetId}/download-zip`;
}

// Trigger direct browser download of dataset + GT ZIP
export function downloadDatasetZip(datasetId, datasetName = 'dataset') {
  const url = getDatasetZipDownloadUrl(datasetId);
  const link = document.createElement('a');
  link.href = url;
  link.download = `${datasetName}_gt.zip`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

// --- Annotations ---
export async function getAnnotations(imageId) {
  return request(`/api/v1/annotations/${imageId}`);
}

export async function saveAnnotations(imageId, annotations) {
  return request('/api/v1/annotations/batch', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      image_id: Number(imageId),
      annotations: annotations.map((a, idx) => {
        const x_min = a.x_min !== undefined ? a.x_min : 0;
        const y_min = a.y_min !== undefined ? a.y_min : 0;
        const x_max = a.x_max !== undefined ? a.x_max : 1;
        const y_max = a.y_max !== undefined ? a.y_max : 1;
        const width = a.bbox_w !== undefined ? a.bbox_w : Math.max(0.01, x_max - x_min);
        const height = a.bbox_h !== undefined ? a.bbox_h : Math.max(0.01, y_max - y_min);
        const centerX = a.bbox_x !== undefined ? a.bbox_x : x_min + width / 2;
        const centerY = a.bbox_y !== undefined ? a.bbox_y : y_min + height / 2;

        return {
          class_id: a.class_id !== undefined ? Number(a.class_id) : idx,
          class_name: String(a.class_name || a.label || 'object').trim(),
          bbox_x: Math.min(1.0, Math.max(0.0, centerX)),
          bbox_y: Math.min(1.0, Math.max(0.0, centerY)),
          bbox_w: Math.min(1.0, Math.max(0.001, width)),
          bbox_h: Math.min(1.0, Math.max(0.001, height)),
          confidence: a.confidence !== undefined ? Number(a.confidence) : 1.0,
          segmentation: a.segmentation || (a.points ? a.points : null),
        };
      }),
    }),
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

export async function cancelActiveTraining() {
  return request('/api/v1/training/cancel-active', {
    method: 'POST',
  });
}

export async function getActiveTrainingJob() {
  return request('/api/v1/training/active');
}

export function getJobWeightDownloadUrl(jobId) {
  return `${BASE_URL}/api/v1/training/${jobId}/weights/download`;
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
