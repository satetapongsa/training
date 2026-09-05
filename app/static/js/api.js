/**
 * API & WebSocket Client for AI Vision Training Studio
 */
class ApiClient {
  constructor(baseUrl = "/api/v1") {
    this.baseUrl = baseUrl;
  }

  async request(endpoint, options = {}) {
    const url = `${this.baseUrl}${endpoint}`;
    try {
      const response = await fetch(url, options);
      if (!response.ok) {
        let errData = {};
        try { errData = await response.json(); } catch (_) {}
        throw new Error(errData.detail || `HTTP error ${response.status}`);
      }
      if (response.status === 204) return null;
      return await response.json();
    } catch (err) {
      console.error(`[API Error] ${endpoint}:`, err);
      throw err;
    }
  }

  // Projects
  getProjects() { return this.request("/projects"); }
  createProject(data) {
    return this.request("/projects", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
  }
  deleteProject(id) { return this.request(`/projects/${id}`, { method: "DELETE" }); }

  // Datasets
  getDatasets(projectId) {
    const q = projectId ? `?project_id=${projectId}` : "";
    return this.request(`/datasets${q}`);
  }
  createDataset(data) {
    return this.request("/datasets", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
  }
  uploadImages(datasetId, files) {
    const formData = new FormData();
    for (let f of files) formData.append("files", f);
    return this.request(`/datasets/${datasetId}/upload`, {
      method: "POST",
      body: formData,
    });
  }
  importFolder(datasetId, folderPath) {
    const formData = new FormData();
    formData.append("folder_path", folderPath);
    return this.request(`/datasets/${datasetId}/import-folder`, {
      method: "POST",
      body: formData,
    });
  }
  getDatasetImages(datasetId, skip = 0, limit = 100) {
    return this.request(`/datasets/${datasetId}/images?skip=${skip}&limit=${limit}`);
  }
  validateDataset(datasetId) {
    return this.request(`/datasets/${datasetId}/validate`, { method: "POST" });
  }
  splitDataset(datasetId, ratios) {
    return this.request(`/datasets/${datasetId}/split`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(ratios),
    });
  }
  deleteDataset(id) { return this.request(`/datasets/${id}`, { method: "DELETE" }); }

  // Annotations
  getImageAnnotations(imageId) {
    return this.request(`/annotations/${imageId}`);
  }
  batchSaveAnnotations(imageId, annotations) {
    return this.request("/annotations/batch", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_id: imageId, annotations }),
    });
  }

  // Training
  getArchitectures() { return this.request("/training/architectures"); }
  getTrainingJobs(projectId) {
    const q = projectId ? `?project_id=${projectId}` : "";
    return this.request(`/training/jobs${q}`);
  }
  startTraining(payload) {
    return this.request("/training/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  }
  stopTraining(jobId) {
    return this.request(`/training/${jobId}/stop`, { method: "POST" });
  }
  getTrainingLogs(jobId) {
    return this.request(`/training/${jobId}/logs`);
  }

  // Models
  getModels(projectId) {
    const q = projectId ? `?project_id=${projectId}` : "";
    return this.request(`/models${q}`);
  }
  exportModel(modelId, format = "onnx") {
    return this.request(`/models/${modelId}/export`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ format, image_size: 640 }),
    });
  }
  deleteModel(id) { return this.request(`/models/${id}`, { method: "DELETE" }); }

  // Inference
  predictImage(modelId, file, conf = 0.25, iou = 0.45) {
    const formData = new FormData();
    formData.append("model_id", modelId);
    formData.append("conf_threshold", conf);
    formData.append("iou_threshold", iou);
    formData.append("file", file);
    return this.request("/inference/predict", {
      method: "POST",
      body: formData,
    });
  }

  // System
  getSystemInfo() { return this.request("/system/info"); }
  getSystemMetrics() { return this.request("/system/metrics"); }
}

class WebSocketManager {
  constructor() {
    this.handlers = new Map();
    this.ws = null;
    this.reconnectTimer = null;
  }

  connect() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.host || "localhost:8000";
    const url = `${protocol}//${host}/ws/live`;

    this.ws = new WebSocket(url);

    this.ws.onopen = () => {
      console.log("[WS] Live stream connected.");
      if (this.reconnectTimer) {
        clearTimeout(this.reconnectTimer);
        this.reconnectTimer = null;
      }
    };

    this.ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        const eventName = payload.event;
        if (this.handlers.has(eventName)) {
          for (let cb of this.handlers.get(eventName)) {
            cb(payload.data);
          }
        }
      } catch (e) {
        console.error("[WS] Message parsing error:", e);
      }
    };

    this.ws.onclose = () => {
      console.warn("[WS] Live stream disconnected. Reconnecting in 3s...");
      this.reconnectTimer = setTimeout(() => this.connect(), 3000);
    };
  }

  on(eventName, callback) {
    if (!this.handlers.has(eventName)) {
      this.handlers.set(eventName, new Set());
    }
    this.handlers.get(eventName).add(callback);
  }

  off(eventName, callback) {
    if (this.handlers.has(eventName)) {
      this.handlers.get(eventName).delete(callback);
    }
  }
}

const api = new ApiClient();
const ws = new WebSocketManager();
ws.connect();
