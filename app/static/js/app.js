/**
 * AI Vision Training Studio - Master SPA Controller
 */
const state = {
  activeTab: "dashboard",
  projects: [],
  activeProject: null,
  datasets: [],
  activeDataset: null,
  datasetImages: [],
  activeImageIndex: 0,
  models: [],
  activeModel: null,
  trainingJobs: [],
  activeJobId: null,
  annotator: null,
  lossChart: null,
  accChart: null,
};

function showToast(message, type = "info") {
  const container = document.getElementById("toastContainer");
  if (!container) return;
  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transform = "translateX(100%)";
    toast.style.transition = "all 0.3s ease";
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}

// Navigation & Tab Switching
function switchTab(tabName) {
  state.activeTab = tabName;
  document.querySelectorAll(".nav-item").forEach(el => {
    el.classList.toggle("active", el.dataset.tab === tabName);
  });
  document.querySelectorAll(".tab-view").forEach(el => {
    el.style.display = el.id === `view-${tabName}` ? "block" : "none";
  });

  const titles = {
    dashboard: "Platform Overview",
    projects: "Project Management",
    datasets: "Dataset Manager",
    annotations: "Interactive Annotation Studio",
    training: "Model Training & Live Telemetry",
    models: "Model Registry & Exports",
    inference: "Inference Playground & Testing",
    system: "System Diagnostics & Hardware",
  };
  document.getElementById("topbarTitle").textContent = titles[tabName] || "Platform Overview";

  // Tab-specific initializers
  if (tabName === "dashboard") loadDashboardData();
  else if (tabName === "projects") loadProjectsView();
  else if (tabName === "datasets") loadDatasetsView();
  else if (tabName === "annotations") loadAnnotationsView();
  else if (tabName === "training") loadTrainingView();
  else if (tabName === "models") loadModelsView();
  else if (tabName === "inference") loadInferenceView();
  else if (tabName === "system") loadSystemView();
}

// --- DASHBOARD ---
async function loadDashboardData() {
  try {
    const projects = await api.getProjects();
    state.projects = projects;
    if (!state.activeProject && projects.length > 0) {
      setActiveProject(projects[0]);
    }

    const sysInfo = await api.getSystemInfo();
    const sysMetrics = await api.getSystemMetrics();

    document.getElementById("statProjects").textContent = projects.length;
    document.getElementById("statCpu").textContent = `${sysMetrics.cpu_percent}%`;
    document.getElementById("statRam").textContent = `${sysMetrics.ram_percent}%`;

    const gpuCard = document.getElementById("statGpuCard");
    if (sysInfo.cuda_available && sysMetrics.gpus.length > 0) {
      const gpu = sysMetrics.gpus[0];
      document.getElementById("statGpu").textContent = `${gpu.name} (${gpu.vram_used_mb}MB / ${gpu.vram_total_mb}MB)`;
    } else {
      document.getElementById("statGpu").textContent = "CPU Mode (CUDA not available)";
    }

    // Load recent training jobs
    const jobs = await api.getTrainingJobs();
    renderRecentJobs(jobs.slice(0, 5));
  } catch (err) {
    showToast(err.message, "error");
  }
}

function renderRecentJobs(jobs) {
  const container = document.getElementById("recentJobsTable");
  if (!container) return;
  if (jobs.length === 0) {
    container.innerHTML = `<tr><td colspan="5" style="text-align:center; padding: 20px; color:#6b7280;">No training jobs yet.</td></tr>`;
    return;
  }
  container.innerHTML = jobs.map(j => `
    <tr>
      <td style="font-weight:600; color:#fff;">${j.model_name}</td>
      <td><span class="badge badge-info">${j.architecture}</span></td>
      <td><span class="badge ${j.status === 'completed' ? 'badge-success' : (j.status === 'running' ? 'badge-warning' : 'badge-danger')}">${j.status}</span></td>
      <td>Epoch ${j.current_epoch} / ${j.total_epochs}</td>
      <td>${j.best_metric_val ? `${j.best_metric_name}: ${j.best_metric_val.toFixed(4)}` : '-'}</td>
    </tr>
  `).join("");
}

// --- PROJECTS ---
function setActiveProject(project) {
  state.activeProject = project;
  const nameEl = document.getElementById("activeProjectName");
  if (nameEl) nameEl.textContent = project ? project.name : "None selected";
  showToast(`Active project: ${project.name}`, "info");
}

async function loadProjectsView() {
  const projects = await api.getProjects();
  state.projects = projects;
  const grid = document.getElementById("projectsGrid");
  if (!grid) return;

  grid.innerHTML = projects.map(p => `
    <div class="card" style="margin-bottom:0;">
      <div class="card-header">
        <h3 class="card-title">${p.name}</h3>
        <span class="badge badge-info">${p.task_type}</span>
      </div>
      <p style="color:var(--text-muted); font-size:13px; margin-bottom:14px;">${p.description || "No description."}</p>
      <div style="font-size:12px; color:var(--text-sub); margin-bottom:16px;">
        Datasets: ${p.datasets_count} | Models: ${p.models_count} | Jobs: ${p.training_jobs_count}
      </div>
      <div style="display:flex; gap:8px;">
        <button class="btn btn-sm btn-primary" onclick="selectProjectById(${p.id})">Select</button>
        <button class="btn btn-sm btn-danger" onclick="deleteProjectById(${p.id})">Delete</button>
      </div>
    </div>
  `).join("");
}

window.selectProjectById = (id) => {
  const p = state.projects.find(x => x.id === id);
  if (p) setActiveProject(p);
};

window.deleteProjectById = async (id) => {
  if (!confirm("Are you sure you want to delete this project?")) return;
  try {
    await api.deleteProject(id);
    showToast("Project deleted", "success");
    loadProjectsView();
  } catch (err) {
    showToast(err.message, "error");
  }
};

// --- DATASETS ---
async function loadDatasetsView() {
  if (!state.activeProject) return;
  const datasets = await api.getDatasets(state.activeProject.id);
  state.datasets = datasets;

  const selector = document.getElementById("datasetSelector");
  if (selector) {
    selector.innerHTML = datasets.map(d => `<option value="${d.id}">${d.name} (${d.total_images} imgs)</option>`).join("");
    if (datasets.length > 0) {
      state.activeDataset = datasets[0];
      loadDatasetDetails(datasets[0].id);
    }
  }
}

async function loadDatasetDetails(datasetId) {
  const dataset = state.datasets.find(d => d.id == datasetId);
  if (!dataset) return;
  state.activeDataset = dataset;

  document.getElementById("dsImagesCount").textContent = dataset.total_images;
  document.getElementById("dsAnnotsCount").textContent = dataset.total_annotations;
  document.getElementById("dsSplits").textContent = `Train: ${dataset.train_count} | Val: ${dataset.val_count} | Test: ${dataset.test_count}`;

  // Load thumbnail gallery
  const images = await api.getDatasetImages(dataset.id, 0, 18);
  const gallery = document.getElementById("datasetGallery");
  if (gallery) {
    if (images.length === 0) {
      gallery.innerHTML = `<p style="grid-column: 1/-1; color: var(--text-sub); text-align:center; padding: 30px;">No images in this dataset yet. Use the upload zone above.</p>`;
    } else {
      gallery.innerHTML = images.map(img => `
        <div style="border: 1px solid var(--border-color); border-radius: 8px; overflow: hidden; background: #000; position: relative;">
          <img src="${img.image_url}" style="width: 100%; height: 110px; object-fit: cover;" loading="lazy" />
          <div style="padding: 6px 8px; font-size: 11px; display:flex; justify-content: space-between; background: var(--bg-card);">
            <span style="overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width: 90px;">${img.filename}</span>
            <span class="badge ${img.is_annotated ? 'badge-success' : 'badge-warning'}">${img.is_annotated ? 'Annotated' : 'Unlabeled'}</span>
          </div>
        </div>
      `).join("");
    }
  }
}

// --- ANNOTATIONS ---
async function loadAnnotationsView() {
  if (!state.activeDataset) {
    showToast("Please select a dataset first", "warning");
    return;
  }

  if (!state.annotator) {
    state.annotator = new ImageAnnotator("annotationCanvas");
  }

  state.annotator.setClasses(state.activeDataset.classes);

  // Render class buttons
  const classContainer = document.getElementById("classSelectorPills");
  if (classContainer) {
    classContainer.innerHTML = state.activeDataset.classes.map((c, idx) => `
      <button class="btn btn-sm ${idx === 0 ? 'btn-primary' : 'btn-secondary'}" onclick="setAnnotatorClass(${idx})">
        ${idx + 1}. ${c}
      </button>
    `).join("");
  }

  // Load images
  const images = await api.getDatasetImages(state.activeDataset.id, 0, 100);
  state.datasetImages = images;
  state.activeImageIndex = 0;

  if (images.length > 0) {
    loadAnnotatorImage(0);
  }
}

window.setAnnotatorClass = (idx) => {
  if (state.annotator) {
    state.annotator.activeClassIdx = idx;
    document.querySelectorAll("#classSelectorPills button").forEach((b, i) => {
      b.className = `btn btn-sm ${i === idx ? 'btn-primary' : 'btn-secondary'}`;
    });
  }
};

async function loadAnnotatorImage(index) {
  if (index < 0 || index >= state.datasetImages.length) return;
  state.activeImageIndex = index;
  const imgRecord = state.datasetImages[index];

  document.getElementById("annotImageName").textContent = `${imgRecord.filename} (${index + 1}/${state.datasetImages.length})`;

  // Fetch current annotations
  const annots = await api.getImageAnnotations(imgRecord.id);
  await state.annotator.loadImage(imgRecord.image_url, annots);
}

// Next/Prev Annotations
window.annotatorNext = () => {
  if (state.activeImageIndex < state.datasetImages.length - 1) {
    loadAnnotatorImage(state.activeImageIndex + 1);
  }
};

window.annotatorPrev = () => {
  if (state.activeImageIndex > 0) {
    loadAnnotatorImage(state.activeImageIndex - 1);
  }
};

window.saveCurrentAnnotations = async () => {
  if (!state.annotator || !state.datasetImages[state.activeImageIndex]) return;
  const imgRecord = state.datasetImages[state.activeImageIndex];
  try {
    await api.batchSaveAnnotations(imgRecord.id, state.annotator.annotations);
    showToast("Annotations saved successfully!", "success");
    imgRecord.is_annotated = state.annotator.annotations.length > 0;
  } catch (err) {
    showToast(err.message, "error");
  }
};

// --- TRAINING ---
async function loadTrainingView() {
  if (!state.lossChart) {
    state.lossChart = new RealtimeMetricChart("chartLoss", "Training Loss Curve", "#6366f1", "Loss");
  }
  if (!state.accChart) {
    state.accChart = new RealtimeMetricChart("chartMetric", "mAP50 / Accuracy Curve", "#10b981", "mAP50");
  }

  // Populate dataset dropdown
  if (state.activeProject) {
    const datasets = await api.getDatasets(state.activeProject.id);
    const dsSel = document.getElementById("trainDatasetSelect");
    if (dsSel) {
      dsSel.innerHTML = datasets.map(d => `<option value="${d.id}">${d.name}</option>`).join("");
    }
  }

  // Populate architectures
  const archs = await api.getArchitectures();
  const archSel = document.getElementById("trainArchSelect");
  if (archSel) {
    archSel.innerHTML = `
      <optgroup label="Object Detection (YOLO)">
        ${archs.detection.map(a => `<option value="${a}">${a}</option>`).join("")}
      </optgroup>
      <optgroup label="Image Classification (PyTorch)">
        ${archs.classification.map(a => `<option value="${a}">${a}</option>`).join("")}
      </optgroup>
    `;
  }
}

// Start Training Trigger
window.startTrainingJob = async () => {
  if (!state.activeProject) {
    showToast("Select a project first", "error");
    return;
  }
  const datasetId = document.getElementById("trainDatasetSelect").value;
  const architecture = document.getElementById("trainArchSelect").value;
  const modelName = document.getElementById("trainModelName").value.trim() || `exp_${Date.now()}`;
  const epochs = parseInt(document.getElementById("trainEpochs").value) || 20;
  const batchSize = parseInt(document.getElementById("trainBatchSize").value) || 8;
  const lr = parseFloat(document.getElementById("trainLr").value) || 0.001;
  const device = document.getElementById("trainDevice").value || "auto";

  state.lossChart.clear();
  state.accChart.clear();
  document.getElementById("terminalLogs").textContent = "Initializing training pipeline...\n";

  try {
    const payload = {
      project_id: state.activeProject.id,
      dataset_id: parseInt(datasetId),
      model_name: modelName,
      architecture: architecture,
      config: {
        epochs: epochs,
        batch_size: batchSize,
        image_size: 640,
        learning_rate: lr,
        optimizer: "AdamW",
        device: device,
      },
    };

    const job = await api.startTraining(payload);
    state.activeJobId = job.id;
    showToast(`Training Job ${job.id} started!`, "success");
    document.getElementById("btnStartTraining").disabled = true;
    document.getElementById("btnStopTraining").disabled = false;
  } catch (err) {
    showToast(err.message, "error");
  }
};

window.stopTrainingJob = async () => {
  if (!state.activeJobId) return;
  try {
    await api.stopTraining(state.activeJobId);
    showToast("Stop request sent", "info");
  } catch (err) {
    showToast(err.message, "error");
  }
};

// WebSocket Live Telemetry Listeners
ws.on("epoch_update", (data) => {
  document.getElementById("trainEpochProgress").textContent = `Epoch ${data.epoch} / ${data.total_epochs}`;
  const pct = Math.round((data.epoch / data.total_epochs) * 100);
  document.getElementById("trainProgressBar").style.width = `${pct}%`;

  if (state.lossChart) state.lossChart.addPoint(data.epoch, data.loss);

  const m = data.metrics || {};
  const metricVal = m.mAP50 !== undefined ? m.mAP50 : (m.accuracy !== undefined ? m.accuracy : 0.0);
  if (state.accChart) state.accChart.addPoint(data.epoch, metricVal);

  const term = document.getElementById("terminalLogs");
  if (term) {
    term.textContent += `[Epoch ${data.epoch}/${data.total_epochs}] Loss: ${data.loss} | LR: ${data.lr} | Metrics: ${JSON.stringify(m)}\n`;
    term.scrollTop = term.scrollHeight;
  }
});

ws.on("training_complete", (data) => {
  showToast("Training Completed Successfully! Model Registered.", "success");
  document.getElementById("btnStartTraining").disabled = false;
  document.getElementById("btnStopTraining").disabled = true;
  const term = document.getElementById("terminalLogs");
  if (term) {
    term.textContent += `\n=== Training Finished! Model Registered (ID: ${data.model_id}) ===\n`;
    term.scrollTop = term.scrollHeight;
  }
});

ws.on("training_error", (data) => {
  showToast(`Training Failed: ${data.error}`, "error");
  document.getElementById("btnStartTraining").disabled = false;
  document.getElementById("btnStopTraining").disabled = true;
});

// --- MODELS ---
async function loadModelsView() {
  const models = await api.getModels();
  state.models = models;
  const grid = document.getElementById("modelsGrid");
  if (!grid) return;

  if (models.length === 0) {
    grid.innerHTML = `<p style="grid-column:1/-1; color:var(--text-sub); text-align:center; padding:40px;">No models registered yet. Train a model to see it here.</p>`;
    return;
  }

  grid.innerHTML = models.map(m => `
    <div class="card" style="margin-bottom:0;">
      <div class="card-header">
        <h3 class="card-title">${m.name}</h3>
        <span class="badge badge-success">${m.version}</span>
      </div>
      <div style="font-size:12px; color:var(--text-muted); margin-bottom:12px;">
        <div>Architecture: <strong>${m.architecture}</strong> (${m.task_type})</div>
        <div>Size: <strong>${(m.size_bytes / (1024 * 1024)).toFixed(1)} MB</strong></div>
        <div>Metrics: <code>${JSON.stringify(m.metrics)}</code></div>
      </div>
      <div style="display:flex; flex-wrap:wrap; gap:8px;">
        <a href="/api/v1/models/${m.id}/download?format=pt" class="btn btn-sm btn-secondary" download>Download .pt</a>
        <button class="btn btn-sm btn-primary" onclick="exportToOnnx(${m.id})">Export ONNX</button>
        <button class="btn btn-sm btn-success" onclick="testModelInPlayground(${m.id})">Test in Inference</button>
        <button class="btn btn-sm btn-danger" onclick="deleteModelById(${m.id})">Delete</button>
      </div>
    </div>
  `).join("");
}

window.exportToOnnx = async (modelId) => {
  showToast("Exporting model to ONNX...", "info");
  try {
    const res = await api.exportModel(modelId, "onnx");
    showToast("ONNX Export Successful! Click to download.", "success");
    window.location.href = res.download_url;
  } catch (err) {
    showToast(err.message, "error");
  }
};

window.deleteModelById = async (id) => {
  if (!confirm("Are you sure you want to delete this model?")) return;
  try {
    await api.deleteModel(id);
    showToast("Model deleted", "success");
    loadModelsView();
  } catch (err) {
    showToast(err.message, "error");
  }
};

window.testModelInPlayground = (modelId) => {
  switchTab("inference");
  const sel = document.getElementById("inferModelSelect");
  if (sel) sel.value = modelId;
};

// --- INFERENCE PLAYGROUND ---
async function loadInferenceView() {
  const models = await api.getModels();
  const sel = document.getElementById("inferModelSelect");
  if (sel) {
    sel.innerHTML = models.map(m => `<option value="${m.id}">${m.name} (${m.architecture})</option>`).join("");
  }
}

window.runInference = async () => {
  const modelId = document.getElementById("inferModelSelect").value;
  const fileInput = document.getElementById("inferFileInput");
  if (!modelId || !fileInput.files[0]) {
    showToast("Select a model and upload an image first", "warning");
    return;
  }

  const conf = parseFloat(document.getElementById("inferConfSlider").value) || 0.25;
  const iou = parseFloat(document.getElementById("inferIouSlider").value) || 0.45;

  showToast("Running inference with real weights...", "info");
  try {
    const res = await api.predictImage(modelId, fileInput.files[0], conf, iou);
    document.getElementById("inferResultImg").src = `${res.annotated_image_url}?t=${Date.now()}`;
    document.getElementById("inferDownloadBtn").href = res.annotated_image_url;
    document.getElementById("inferDownloadBtn").style.display = "inline-flex";

    document.getElementById("inferStats").textContent =
      `Inference Time: ${res.inference_time_ms} ms | Detections: ${res.total_detections}`;

    // Render detections table
    const tbody = document.getElementById("inferTableBody");
    if (tbody) {
      if (res.detections.length === 0) {
        tbody.innerHTML = `<tr><td colspan="3" style="text-align:center; color:#6b7280;">No objects detected above confidence threshold.</td></tr>`;
      } else {
        tbody.innerHTML = res.detections.map(d => `
          <tr>
            <td style="font-weight:600; color:#fff;">${d.class_name}</td>
            <td><span class="badge badge-success">${(d.confidence * 100).toFixed(1)}%</span></td>
            <td style="font-family:monospace; font-size:11px;">[${d.box_pixels.join(", ")}]</td>
          </tr>
        `).join("");
      }
    }
  } catch (err) {
    showToast(err.message, "error");
  }
};

// --- SYSTEM VIEW ---
async function loadSystemView() {
  try {
    const info = await api.getSystemInfo();
    const metrics = await api.getSystemMetrics();

    document.getElementById("sysPython").textContent = info.python_version;
    document.getElementById("sysTorch").textContent = info.pytorch_version;
    document.getElementById("sysCuda").textContent = info.cuda_available ? `Yes (${info.cuda_version})` : "No (CPU Mode)";
    document.getElementById("sysCpuCount").textContent = `${info.cpu_count} cores`;
    document.getElementById("sysRam").textContent = `${metrics.ram_used_gb} GB / ${info.total_ram_gb} GB (${metrics.ram_percent}%)`;

    const gpuTbody = document.getElementById("sysGpuTable");
    if (gpuTbody) {
      if (info.gpus.length === 0) {
        gpuTbody.innerHTML = `<tr><td colspan="4" style="text-align:center; color:#6b7280;">No dedicated NVIDIA GPUs detected. Running in optimized CPU mode.</td></tr>`;
      } else {
        gpuTbody.innerHTML = info.gpus.map(g => `
          <tr>
            <td>${g.id}</td>
            <td style="font-weight:600; color:#fff;">${g.name}</td>
            <td>${g.total_memory_mb} MB</td>
            <td>${g.multi_processor_count}</td>
          </tr>
        `).join("");
      }
    }
  } catch (err) {
    showToast(err.message, "error");
  }
}

// --- BATCH & FOLDER UPLOAD WITH PROGRESS ---
async function scanDirectoryEntry(item, fileList) {
  if (item.isFile) {
    return new Promise((resolve) => {
      item.file((f) => {
        fileList.push(f);
        resolve();
      });
    });
  } else if (item.isDirectory) {
    const dirReader = item.createReader();
    return new Promise((resolve) => {
      dirReader.readEntries(async (entries) => {
        for (let entry of entries) {
          await scanDirectoryEntry(entry, fileList);
        }
        resolve();
      });
    });
  }
}

async function uploadFilesWithProgress(files) {
  if (!state.activeDataset) {
    showToast("Please select a dataset first", "error");
    return;
  }

  // Filter allowed image extensions and txt labels
  const validExts = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".txt"];
  const filtered = Array.from(files).filter(f => {
    const ext = "." + f.name.split(".").pop().toLowerCase();
    return validExts.includes(ext);
  });

  if (filtered.length === 0) {
    showToast("No valid images or labels found to upload.", "warning");
    return;
  }

  const progressBox = document.getElementById("uploadProgressBox");
  const progressLabel = document.getElementById("uploadProgressLabel");
  const progressPercent = document.getElementById("uploadProgressPercent");
  const progressBar = document.getElementById("uploadProgressBar");

  if (progressBox) progressBox.style.display = "block";

  const total = filtered.length;
  let uploaded = 0;
  const chunkSize = 30; // 30 files per batch request

  showToast(`Uploading ${total} files...`, "info");

  for (let i = 0; i < total; i += chunkSize) {
    const chunk = filtered.slice(i, i + chunkSize);
    try {
      await api.uploadImages(state.activeDataset.id, chunk);
      uploaded += chunk.length;
      const pct = Math.min(100, Math.round((uploaded / total) * 100));

      if (progressLabel) progressLabel.textContent = `Uploading: ${uploaded} / ${total} files...`;
      if (progressPercent) progressPercent.textContent = `${pct}%`;
      if (progressBar) progressBar.style.width = `${pct}%`;
    } catch (err) {
      showToast(`Error uploading batch: ${err.message}`, "error");
    }
  }

  showToast(`Successfully uploaded ${uploaded} files into dataset!`, "success");
  if (progressLabel) progressLabel.textContent = `Upload completed! (${uploaded} files)`;
  setTimeout(() => {
    if (progressBox) progressBox.style.display = "none";
  }, 2500);

  loadDatasetDetails(state.activeDataset.id);
}

// App Initialization
window.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".nav-item").forEach(el => {
    el.addEventListener("click", () => switchTab(el.dataset.tab));
  });

  // Setup Drag & Drop Upload with Recursive Folder Support
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const folderInput = document.getElementById("folderInput");

  if (dropzone) {
    dropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });
    dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
    dropzone.addEventListener("drop", async (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");

      const items = e.dataTransfer.items;
      if (items && items.length > 0) {
        const collectedFiles = [];
        for (let i = 0; i < items.length; i++) {
          const entry = items[i].webkitGetAsEntry ? items[i].webkitGetAsEntry() : null;
          if (entry) {
            await scanDirectoryEntry(entry, collectedFiles);
          } else if (items[i].getAsFile) {
            const f = items[i].getAsFile();
            if (f) collectedFiles.push(f);
          }
        }
        if (collectedFiles.length > 0) {
          await uploadFilesWithProgress(collectedFiles);
          return;
        }
      }

      if (e.dataTransfer.files.length > 0) {
        await uploadFilesWithProgress(e.dataTransfer.files);
      }
    });
  }

  if (fileInput) {
    fileInput.addEventListener("change", async () => {
      if (fileInput.files.length > 0) {
        await uploadFilesWithProgress(fileInput.files);
        fileInput.value = "";
      }
    });
  }

  if (folderInput) {
    folderInput.addEventListener("change", async () => {
      if (folderInput.files.length > 0) {
        await uploadFilesWithProgress(folderInput.files);
        folderInput.value = "";
      }
    });
  }

  // Load Initial View
  switchTab("dashboard");
});

