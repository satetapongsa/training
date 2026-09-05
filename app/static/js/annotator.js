/**
 * Interactive YOLO Bounding-Box Annotation Studio Engine
 */
class ImageAnnotator {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext("2d");

    this.image = null;
    this.annotations = []; // [{class_id, class_name, bbox_x, bbox_y, bbox_w, bbox_h}]
    this.classes = ["object"];
    this.activeClassIdx = 0;
    this.selectedIndex = -1;

    // Viewport transform
    this.zoom = 1.0;
    this.panX = 0;
    this.panY = 0;
    this.isPanning = false;
    this.startPanX = 0;
    this.startPanY = 0;

    // Drawing state
    this.isDrawing = false;
    this.drawStartX = 0;
    this.drawStartY = 0;
    this.currentBox = null;

    this.colors = [
      "#6366f1", "#10b981", "#f59e0b", "#ef4444",
      "#8b5cf6", "#06b6d4", "#ec4899", "#14b8a6"
    ];

    this.initEvents();
  }

  setClasses(classList) {
    this.classes = classList && classList.length > 0 ? classList : ["object"];
    this.activeClassIdx = 0;
  }

  loadImage(imageUrl, annotations = []) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.onload = () => {
        this.image = img;
        this.annotations = JSON.parse(JSON.stringify(annotations));
        this.resetView();
        this.render();
        resolve();
      };
      img.onerror = reject;
      img.src = imageUrl;
    });
  }

  resetView() {
    if (!this.image || !this.canvas) return;
    const cw = this.canvas.parentElement.clientWidth;
    const ch = this.canvas.parentElement.clientHeight;
    this.canvas.width = cw;
    this.canvas.height = ch;

    const scale = Math.min((cw - 40) / this.image.width, (ch - 40) / this.image.height, 1.0);
    this.zoom = Math.max(scale, 0.1);
    this.panX = (cw - this.image.width * this.zoom) / 2;
    this.panY = (ch - this.image.height * this.zoom) / 2;
  }

  // Convert canvas pixel coords to normalized image coords 0..1
  canvasToNormalized(cx, cy) {
    const ix = (cx - this.panX) / this.zoom;
    const iy = (cy - this.panY) / this.zoom;
    return {
      x: Math.max(0, Math.min(1, ix / this.image.width)),
      y: Math.max(0, Math.min(1, iy / this.image.height)),
    };
  }

  normalizedToCanvas(nx, ny, nw, nh) {
    const x1 = (nx - nw / 2) * this.image.width * this.zoom + this.panX;
    const y1 = (ny - nh / 2) * this.image.height * this.zoom + this.panY;
    const w = nw * this.image.width * this.zoom;
    const h = nh * this.image.height * this.zoom;
    return { x: x1, y: y1, w, h };
  }

  initEvents() {
    window.addEventListener("resize", () => {
      if (this.canvas.parentElement) {
        this.canvas.width = this.canvas.parentElement.clientWidth;
        this.canvas.height = this.canvas.parentElement.clientHeight;
        this.render();
      }
    });

    this.canvas.addEventListener("mousedown", (e) => this.onMouseDown(e));
    this.canvas.addEventListener("mousemove", (e) => this.onMouseMove(e));
    window.addEventListener("mouseup", () => this.onMouseUp());
    this.canvas.addEventListener("wheel", (e) => this.onWheel(e));

    // Keyboard Shortcuts
    window.addEventListener("keydown", (e) => {
      if (e.key === "Delete" || e.key === "Backspace") {
        if (this.selectedIndex >= 0 && this.selectedIndex < this.annotations.length) {
          this.annotations.splice(this.selectedIndex, 1);
          this.selectedIndex = -1;
          this.render();
          if (this.onAnnotationChanged) this.onAnnotationChanged(this.annotations);
        }
      } else if (e.key >= "1" && e.key <= "9") {
        const idx = parseInt(e.key) - 1;
        if (idx < this.classes.length) {
          this.activeClassIdx = idx;
          if (this.selectedIndex >= 0) {
            this.annotations[this.selectedIndex].class_id = idx;
            this.annotations[this.selectedIndex].class_name = this.classes[idx];
            this.render();
            if (this.onAnnotationChanged) this.onAnnotationChanged(this.annotations);
          }
        }
      }
    });
  }

  onMouseDown(e) {
    const rect = this.canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    if (e.spaceKey || e.button === 1 || e.shiftKey) {
      // Pan mode
      this.isPanning = true;
      this.startPanX = mx - this.panX;
      this.startPanY = my - this.panY;
      return;
    }

    if (!this.image) return;

    // Check hit test for selection
    let clickedIdx = -1;
    for (let i = this.annotations.length - 1; i >= 0; i--) {
      const b = this.annotations[i];
      const box = this.normalizedToCanvas(b.bbox_x, b.bbox_y, b.bbox_w, b.bbox_h);
      if (mx >= box.x && mx <= box.x + box.w && my >= box.y && my <= box.y + box.h) {
        clickedIdx = i;
        break;
      }
    }

    if (clickedIdx >= 0) {
      this.selectedIndex = clickedIdx;
      this.render();
      if (this.onAnnotationChanged) this.onAnnotationChanged(this.annotations);
    } else {
      // Start drawing new box
      this.selectedIndex = -1;
      this.isDrawing = true;
      this.drawStartX = mx;
      this.drawStartY = my;
    }
  }

  onMouseMove(e) {
    const rect = this.canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    if (this.isPanning) {
      this.panX = mx - this.startPanX;
      this.panY = my - this.startPanY;
      this.render();
      return;
    }

    if (this.isDrawing) {
      const p1 = this.canvasToNormalized(this.drawStartX, this.drawStartY);
      const p2 = this.canvasToNormalized(mx, my);

      const minX = Math.min(p1.x, p2.x);
      const maxX = Math.max(p1.x, p2.x);
      const minY = Math.min(p1.y, p2.y);
      const maxY = Math.max(p1.y, p2.y);

      const w = maxX - minX;
      const h = maxY - minY;
      const cx = minX + w / 2;
      const cy = minY + h / 2;

      this.currentBox = {
        class_id: this.activeClassIdx,
        class_name: this.classes[this.activeClassIdx] || "object",
        bbox_x: cx,
        bbox_y: cy,
        bbox_w: w,
        bbox_h: h,
        confidence: 1.0,
      };
      this.render();
    }
  }

  onMouseUp() {
    this.isPanning = false;
    if (this.isDrawing && this.currentBox) {
      // Add if large enough
      if (this.currentBox.bbox_w > 0.01 && this.currentBox.bbox_h > 0.01) {
        this.annotations.push(this.currentBox);
        this.selectedIndex = this.annotations.length - 1;
        if (this.onAnnotationChanged) this.onAnnotationChanged(this.annotations);
      }
      this.isDrawing = false;
      this.currentBox = null;
      this.render();
    }
  }

  onWheel(e) {
    e.preventDefault();
    const zoomFactor = e.deltaY < 0 ? 1.15 : 0.85;
    const rect = this.canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    const newZoom = Math.max(0.1, Math.min(10.0, this.zoom * zoomFactor));
    this.panX = mx - (mx - this.panX) * (newZoom / this.zoom);
    this.panY = my - (my - this.panY) * (newZoom / this.zoom);
    this.zoom = newZoom;
    this.render();
  }

  render() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    if (!this.image) {
      this.ctx.fillStyle = "#374151";
      this.ctx.font = "14px Inter, sans-serif";
      this.ctx.textAlign = "center";
      this.ctx.fillText("Select an image to start annotating", this.canvas.width / 2, this.canvas.height / 2);
      return;
    }

    // Draw Image
    const dw = this.image.width * this.zoom;
    const dh = this.image.height * this.zoom;
    this.ctx.drawImage(this.image, this.panX, this.panY, dw, dh);

    // Draw Existing Annotations
    this.annotations.forEach((a, idx) => {
      this.drawBox(a, idx === this.selectedIndex);
    });

    // Draw active drawing box
    if (this.currentBox) {
      this.drawBox(this.currentBox, true, true);
    }
  }

  drawBox(box, isSelected = false, isDrawing = false) {
    const b = this.normalizedToCanvas(box.bbox_x, box.bbox_y, box.bbox_w, box.bbox_h);
    const color = this.colors[box.class_id % this.colors.length];

    this.ctx.save();
    this.ctx.strokeStyle = color;
    this.ctx.lineWidth = isSelected ? 3 : 2;
    if (isDrawing) this.ctx.setLineDash([6, 4]);

    this.ctx.strokeRect(b.x, b.y, b.w, b.h);

    if (isSelected) {
      this.ctx.fillStyle = "rgba(99, 102, 241, 0.15)";
      this.ctx.fillRect(b.x, b.y, b.w, b.h);
    }

    // Label tag
    const tagText = `${box.class_name}`;
    this.ctx.font = "bold 11px Inter, sans-serif";
    const tw = this.ctx.measureText(tagText).width + 8;
    const th = 18;

    this.ctx.fillStyle = color;
    this.ctx.fillRect(b.x, Math.max(0, b.y - th), tw, th);
    this.ctx.fillStyle = "#ffffff";
    this.ctx.fillText(tagText, b.x + 4, Math.max(0, b.y - th) + 13);

    this.ctx.restore();
  }
}
