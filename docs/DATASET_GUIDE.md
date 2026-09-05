# Dataset & Annotation Guide - AI Vision Training Studio

## 1. Supported Formats
The system ingests:
- **Formats**: JPEG (`.jpg`, `.jpeg`), PNG (`.png`), WebP (`.webp`), BMP (`.bmp`), TIFF (`.tiff`, `.tif`).
- **Annotations**: Standard normalized YOLO format (`class_id x_center y_center width height` with values between `0.0` and `1.0`).

---

## 2. Ingestion Methods

### Option A: Drag & Drop In Browser
1. Select or create your dataset in the **Datasets** tab.
2. Drag and drop individual or multiple images into the dropzone.
3. The platform validates magic byte signatures, computes SHA-256 hashes, and registers the images.

### Option B: Folder Import
1. If you already have images on your machine or mounted drive:
2. Click **Import Folder** in the Datasets tab.
3. Enter the absolute path (e.g. `C:/Users/satet/dataset_source`).
4. The system scans recursively and automatically imports matching YOLO `.txt` files if present!

---

## 3. In-Browser Annotation Studio
1. Navigate to the **Annotations** tab.
2. Select your active class on the sidebar (or press number keys `1`, `2`, `3`).
3. Click and drag on the image to draw a bounding box.
4. **Shortcuts**:
   - `Delete` or `Backspace`: Delete selected bounding box.
   - `Space + Drag`: Pan the viewport around the image.
   - `Mouse Wheel`: Zoom in / out.
   - `1-9`: Switch active class for newly drawn or selected box.
5. Click **Save Annotations** to commit bounding boxes.

---

## 4. Dataset Validation & Splitting
- **Deep Validator**: Click **Validate Dataset** to inspect all images and bounding boxes. It checks for:
  - Corrupted image bytes
  - Out of bounds boxes (`w <= 0`, `x > 1`)
  - Duplicate images
  - Missing labels
- **Auto-Split**: Click **Split Dataset** to partition into Train (70%), Val (20%), and Test (10%). This automatically writes `dataset.yaml` and sets up the folder structure for training.
