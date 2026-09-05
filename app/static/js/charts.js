/**
 * High-Performance SVG Real-time Chart Engine for Training Metrics
 */
class RealtimeMetricChart {
  constructor(containerId, title, color = "#6366f1", yLabel = "Loss") {
    this.container = document.getElementById(containerId);
    this.title = title;
    this.color = color;
    this.yLabel = yLabel;
    this.dataPoints = []; // [{x: 1, y: 0.45}]
    this.render();
  }

  addPoint(x, y) {
    this.dataPoints.push({ x, y: Number(y) });
    this.update();
  }

  setData(points) {
    this.dataPoints = points.map(p => ({ x: p.x, y: Number(p.y) }));
    this.update();
  }

  clear() {
    this.dataPoints = [];
    this.update();
  }

  render() {
    if (!this.container) return;
    this.container.innerHTML = `
      <div style="font-weight: 600; font-size: 13px; margin-bottom: 8px; color: #fff; display: flex; justify-content: space-between;">
        <span>${this.title}</span>
        <span id="${this.container.id}-val" style="color: ${this.color}; font-family: monospace;">-</span>
      </div>
      <svg width="100%" height="160" viewBox="0 0 400 160" style="overflow: visible; background: #0c121e; border-radius: 8px; border: 1px solid #1e293b;">
        <defs>
          <linearGradient id="grad-${this.container.id}" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stop-color="${this.color}" stop-opacity="0.3"/>
            <stop offset="100%" stop-color="${this.color}" stop-opacity="0.0"/>
          </linearGradient>
        </defs>
        <g id="${this.container.id}-grid"></g>
        <path id="${this.container.id}-area" fill="url(#grad-${this.container.id})" d="" />
        <path id="${this.container.id}-path" fill="none" stroke="${this.color}" stroke-width="2.5" stroke-linecap="round" d="" />
        <g id="${this.container.id}-dots"></g>
      </svg>
    `;
    this.update();
  }

  update() {
    if (!this.container || this.dataPoints.length === 0) return;

    const svgWidth = 400;
    const svgHeight = 160;
    const padX = 35;
    const padY = 20;

    const xs = this.dataPoints.map(p => p.x);
    const ys = this.dataPoints.map(p => p.y);

    const minX = Math.min(...xs);
    const maxX = Math.max(...xs, minX + 1);
    const minY = Math.min(0, Math.min(...ys));
    const maxY = Math.max(...ys, 0.01) * 1.1;

    const scaleX = (x) => padX + ((x - minX) / (maxX - minX)) * (svgWidth - padX - 15);
    const scaleY = (y) => svgHeight - padY - ((y - minY) / (maxY - minY)) * (svgHeight - padY * 2);

    let pathD = "";
    let areaD = "";
    let dotsHtml = "";

    this.dataPoints.forEach((p, idx) => {
      const sx = scaleX(p.x);
      const sy = scaleY(p.y);
      if (idx === 0) {
        pathD += `M ${sx} ${sy}`;
        areaD += `M ${sx} ${svgHeight - padY} L ${sx} ${sy}`;
      } else {
        pathD += ` L ${sx} ${sy}`;
        areaD += ` L ${sx} ${sy}`;
      }
      dotsHtml += `<circle cx="${sx}" cy="${sy}" r="3.5" fill="${this.color}" />`;
    });

    if (this.dataPoints.length > 0) {
      const last = this.dataPoints[this.dataPoints.length - 1];
      areaD += ` L ${scaleX(last.x)} ${svgHeight - padY} Z`;

      const valEl = document.getElementById(`${this.container.id}-val`);
      if (valEl) valEl.textContent = last.y.toFixed(4);
    }

    const pathEl = document.getElementById(`${this.container.id}-path`);
    const areaEl = document.getElementById(`${this.container.id}-area`);
    const dotsEl = document.getElementById(`${this.container.id}-dots`);

    if (pathEl) pathEl.setAttribute("d", pathD);
    if (areaEl) areaEl.setAttribute("d", areaD);
    if (dotsEl) dotsEl.innerHTML = dotsHtml;
  }
}
