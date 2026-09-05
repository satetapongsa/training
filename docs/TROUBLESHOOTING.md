# Troubleshooting Guide - AI Vision Training Studio

## 1. CUDA / GPU Issues

### Problem: "CUDA not available, running on CPU"
- **Cause**: PyTorch was installed without CUDA support or GPU drivers are missing.
- **Solution**:
  Install CUDA-enabled PyTorch matching your NVIDIA driver version:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  ```
  Verify with:
  ```bash
  python -c "import torch; print(torch.cuda.is_available())"
  ```

### Problem: "CUDA out of memory (OOM)" during training
- **Solution**:
  1. Reduce the `Batch Size` in the training form from 16 to 8, 4, or 2.
  2. Reduce `Image Size` from 640 to 480 or 320.
  3. Select a smaller architecture like `yolo11n`.

---

## 2. Dataset Validation Issues

### Problem: "Dataset contains unnormalized coordinates"
- **Cause**: Bounding boxes in custom annotations have values > 1.0 (e.g. pixel coordinates instead of 0..1).
- **Solution**:
  YOLO format requires normalized values (`center_x / width`, `width / img_w`). The platform's built-in annotator automatically saves normalized coordinates.

### Problem: "Unsupported Image Format"
- **Cause**: File signature header does not match image standards.
- **Solution**:
  Ensure images are clean JPEG, PNG, WebP, BMP, or TIFF. The system verifies magic bytes to protect against corrupted or disguised files.

---

## 3. Web UI & Live Telemetry

### Problem: Live loss charts not updating during training
- **Cause**: WebSocket connection was blocked or disconnected.
- **Solution**:
  1. Ensure no proxy is blocking `ws://` traffic on port 8000.
  2. Check the browser console (F12) for `[WS] Live stream connected.`
  3. The client includes automatic reconnection logic every 3 seconds.
