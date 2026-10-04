/**
 * Re:Learn Safe Vector Whiteboard Renderer
 * Interprets structured JSON whiteboard commands emitted by the backend / LLM.
 * 
 * Invariants:
 *  1. No raw JavaScript execution.
 *  2. All coordinates are normalized [0.0, 1.0] and scaled to canvas dimensions.
 *  3. Interactive playback: Play, Pause, Step Forward, Step Backward, Speed control.
 */

class ReLearnWhiteboard {
  constructor(canvasId, options = {}) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) {
      throw new Error(`Canvas with id '${canvasId}' not found.`);
    }
    this.ctx = this.canvas.getContext('2d');
    this.options = Object.assign({
      responsive: true,
      playbackSpeedMs: 1200,
      onStepChange: null,
    }, options);

    this.payload = null;
    this.currentStep = 0;
    this.isPlaying = false;
    this.timer = null;

    if (this.options.responsive) {
      this.resize();
      window.addEventListener('resize', () => this.resize());
    }
  }

  resize() {
    const rect = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = (rect.width || 800) * dpr;
    this.canvas.height = (rect.height || 500) * dpr;
    this.ctx.scale(dpr, dpr);
    this.render();
  }

  get width() {
    return this.canvas.getBoundingClientRect().width || 800;
  }

  get height() {
    return this.canvas.getBoundingClientRect().height || 500;
  }

  toX(normalizedX) {
    return Math.max(0, Math.min(1, normalizedX)) * this.width;
  }

  toY(normalizedY) {
    return Math.max(0, Math.min(1, normalizedY)) * this.height;
  }

  loadPayload(payload) {
    this.payload = payload;
    this.currentStep = 0;
    this.pause();
    this.render();
    if (this.options.onStepChange) {
      this.options.onStepChange(this.currentStep, this.totalSteps);
    }
  }

  get totalSteps() {
    return this.payload && this.payload.commands ? this.payload.commands.length : 0;
  }

  play() {
    if (!this.payload || this.isPlaying) return;
    this.isPlaying = true;
    this._scheduleNext();
  }

  pause() {
    this.isPlaying = false;
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
    }
  }

  _scheduleNext() {
    if (!this.isPlaying) return;
    this.timer = setTimeout(() => {
      if (this.currentStep < this.totalSteps) {
        this.nextStep();
        this._scheduleNext();
      } else {
        this.pause();
      }
    }, this.options.playbackSpeedMs);
  }

  nextStep() {
    if (!this.payload || this.currentStep >= this.totalSteps) return;
    this.currentStep++;
    this.render();
    if (this.options.onStepChange) {
      this.options.onStepChange(this.currentStep, this.totalSteps);
    }
  }

  prevStep() {
    if (!this.payload || this.currentStep <= 0) return;
    this.currentStep--;
    this.render();
    if (this.options.onStepChange) {
      this.options.onStepChange(this.currentStep, this.totalSteps);
    }
  }

  reset() {
    this.pause();
    this.currentStep = 0;
    this.render();
    if (this.options.onStepChange) {
      this.options.onStepChange(this.currentStep, this.totalSteps);
    }
  }

  render() {
    const w = this.width;
    const h = this.height;
    this.ctx.clearRect(0, 0, w, h);

    // Background Grid
    this._drawBackgroundGrid(w, h);

    if (!this.payload || !this.payload.commands) {
      this._drawPlaceholder(w, h);
      return;
    }

    // Execute commands up to currentStep
    const commandsToRender = this.payload.commands.slice(0, this.currentStep);
    for (const cmd of commandsToRender) {
      this._executeCommand(cmd);
    }
  }

  _drawBackgroundGrid(w, h) {
    this.ctx.save();
    this.ctx.strokeStyle = '#EDF2F7';
    this.ctx.lineWidth = 1;
    const step = 40;
    for (let x = 0; x < w; x += step) {
      this.ctx.beginPath();
      this.ctx.moveTo(x, 0);
      this.ctx.lineTo(x, h);
      this.ctx.stroke();
    }
    for (let y = 0; y < h; y += step) {
      this.ctx.beginPath();
      this.ctx.moveTo(0, y);
      this.ctx.lineTo(w, y);
      this.ctx.stroke();
    }
    this.ctx.restore();
  }

  _drawPlaceholder(w, h) {
    this.ctx.save();
    this.ctx.font = '14px sans-serif';
    this.ctx.fillStyle = '#A0AEC0';
    this.ctx.textAlign = 'center';
    this.ctx.fillText('Re:Learn Whiteboard: Ready for pedagogical diagnosis payload...', w / 2, h / 2);
    this.ctx.restore();
  }

  _executeCommand(cmd) {
    const { tool, params } = cmd;
    this.ctx.save();

    switch (tool) {
      case 'draw_axes':
        this._drawAxes(params);
        break;
      case 'draw_line':
        this._drawLine(params);
        break;
      case 'draw_arrow':
        this._drawArrow(params);
        break;
      case 'draw_shape':
        this._drawShape(params);
        break;
      case 'draw_text':
        this._drawText(params);
        break;
      case 'highlight_region':
        this._highlightRegion(params);
        break;
    }

    this.ctx.restore();
  }

  _drawAxes(p) {
    const [ox, oy] = p.origin || [0.5, 0.5];
    const originX = this.toX(ox);
    const originY = this.toY(oy);
    const color = p.color || '#718096';

    this.ctx.strokeStyle = color;
    this.ctx.lineWidth = 1.5;

    // Horizontal Principal Axis
    this.ctx.beginPath();
    this.ctx.moveTo(0, originY);
    this.ctx.lineTo(this.width, originY);
    this.ctx.stroke();

    // Small Ticks along Principal Axis
    for (let x = 0; x <= this.width; x += 50) {
      this.ctx.beginPath();
      this.ctx.moveTo(x, originY - 4);
      this.ctx.lineTo(x, originY + 4);
      this.ctx.stroke();
    }
  }

  _drawLine(p) {
    const [x1, y1] = p.start;
    const [x2, y2] = p.end;
    this.ctx.strokeStyle = p.color || '#2D3748';
    this.ctx.lineWidth = p.width || 2;

    if (p.style === 'dashed') {
      this.ctx.setLineDash([6, 4]);
    } else if (p.style === 'dotted') {
      this.ctx.setLineDash([2, 4]);
    }

    this.ctx.beginPath();
    this.ctx.moveTo(this.toX(x1), this.toY(y1));
    this.ctx.lineTo(this.toX(x2), this.toY(y2));
    this.ctx.stroke();
  }

  _drawArrow(p) {
    const [x1, y1] = p.start;
    const [x2, y2] = p.end;
    const fromX = this.toX(x1);
    const fromY = this.toY(y1);
    const toX = this.toX(x2);
    const toY = this.toY(y2);
    const color = p.color || '#E53E3E';

    this.ctx.strokeStyle = color;
    this.ctx.fillStyle = color;
    this.ctx.lineWidth = p.width || 2;

    // Draw main arrow line
    this.ctx.beginPath();
    this.ctx.moveTo(fromX, fromY);
    this.ctx.lineTo(toX, toY);
    this.ctx.stroke();

    // Arrowhead angle
    const angle = Math.atan2(toY - fromY, toX - fromX);
    const headLen = 10;
    this.ctx.beginPath();
    this.ctx.moveTo(toX, toY);
    this.ctx.lineTo(toX - headLen * Math.cos(angle - Math.PI / 6), toY - headLen * Math.sin(angle - Math.PI / 6));
    this.ctx.lineTo(toX - headLen * Math.cos(angle + Math.PI / 6), toY - headLen * Math.sin(angle + Math.PI / 6));
    this.ctx.closePath();
    this.ctx.fill();

    // Arrow text label
    if (p.label) {
      const midX = (fromX + toX) / 2;
      const midY = (fromY + toY) / 2;
      this.ctx.font = 'bold 12px sans-serif';
      this.ctx.textAlign = 'center';
      this.ctx.fillStyle = color;
      this.ctx.fillText(p.label, midX, midY - 6);
    }
  }

  _drawShape(p) {
    const shape = p.shape_type;
    const cx = this.toX(p.x);
    const cy = this.toY(p.y);
    const color = p.color || '#3182CE';

    this.ctx.strokeStyle = color;
    this.ctx.lineWidth = 2.5;

    if (shape === 'concave_mirror') {
      // Concave arc curving towards left (reflecting inward)
      const r = this.height * (p.height || 0.5) / 2;
      this.ctx.beginPath();
      this.ctx.arc(cx + r * 0.8, cy + r, r, Math.PI * 0.85, Math.PI * 1.15, false);
      this.ctx.stroke();

      // Reflective hatch marks on right (non-reflecting side)
      this.ctx.strokeStyle = '#A0AEC0';
      this.ctx.lineWidth = 1;
      for (let a = Math.PI * 0.86; a <= Math.PI * 1.14; a += 0.05) {
        const hx = (cx + r * 0.8) + r * Math.cos(a);
        const hy = (cy + r) + r * Math.sin(a);
        this.ctx.beginPath();
        this.ctx.moveTo(hx, hy);
        this.ctx.lineTo(hx + 8, hy - 4);
        this.ctx.stroke();
      }
    } else if (shape === 'convex_mirror') {
      // Convex arc curving towards right (reflecting outward)
      const r = this.height * (p.height || 0.5) / 2;
      this.ctx.beginPath();
      this.ctx.arc(cx - r * 0.8, cy + r, r, -Math.PI * 0.15, Math.PI * 0.15, false);
      this.ctx.stroke();

      // Reflective hatch marks on inner left
      this.ctx.strokeStyle = '#A0AEC0';
      this.ctx.lineWidth = 1;
      for (let a = -Math.PI * 0.14; a <= Math.PI * 0.14; a += 0.05) {
        const hx = (cx - r * 0.8) + r * Math.cos(a);
        const hy = (cy + r) + r * Math.sin(a);
        this.ctx.beginPath();
        this.ctx.moveTo(hx, hy);
        this.ctx.lineTo(hx - 8, hy - 4);
        this.ctx.stroke();
      }
    } else if (shape === 'convex_lens') {
      // Symmetrical converging biconvex lens
      const w = this.width * (p.width || 0.06);
      const h = this.height * (p.height || 0.5);
      this.ctx.beginPath();
      this.ctx.ellipse(cx, cy, w / 2, h / 2, 0, 0, Math.PI * 2);
      this.ctx.stroke();

      // Dashed optical center vertical line
      this.ctx.save();
      this.ctx.strokeStyle = '#CBD5E0';
      this.ctx.setLineDash([3, 3]);
      this.ctx.beginPath();
      this.ctx.moveTo(cx, cy - h / 2);
      this.ctx.lineTo(cx, cy + h / 2);
      this.ctx.stroke();
      this.ctx.restore();
    } else if (shape === 'concave_lens') {
      // Diverging biconcave lens (pinched in center)
      const w = this.width * (p.width || 0.06);
      const h = this.height * (p.height || 0.5);
      this.ctx.beginPath();
      this.ctx.moveTo(cx - w / 2, cy - h / 2);
      this.ctx.lineTo(cx + w / 2, cy - h / 2);
      this.ctx.quadraticCurveTo(cx, cy, cx + w / 2, cy + h / 2);
      this.ctx.lineTo(cx - w / 2, cy + h / 2);
      this.ctx.quadraticCurveTo(cx, cy, cx - w / 2, cy - h / 2);
      this.ctx.stroke();
    } else if (shape === 'resistor') {
      // Zigzag teeth
      const w = this.width * (p.width || 0.15);
      const teeth = 5;
      const dx = w / teeth;
      const amp = 8;
      this.ctx.beginPath();
      this.ctx.moveTo(cx, cy);
      for (let i = 0; i < teeth; i++) {
        const tx = cx + i * dx + dx / 2;
        const ty = cy + (i % 2 === 0 ? -amp : amp);
        this.ctx.lineTo(tx, ty);
      }
      this.ctx.lineTo(cx + w, cy);
      this.ctx.stroke();
    } else if (shape === 'battery') {
      // Parallel long and short lines
      const h = this.height * (p.height || 0.15);
      this.ctx.lineWidth = 2;
      // Long line (+)
      this.ctx.beginPath();
      this.ctx.moveTo(cx, cy - h / 2);
      this.ctx.lineTo(cx, cy + h / 2);
      this.ctx.stroke();
      // Short thick line (-)
      this.ctx.lineWidth = 4;
      this.ctx.beginPath();
      this.ctx.moveTo(cx + 8, cy - h / 4);
      this.ctx.lineTo(cx + 8, cy + h / 4);
      this.ctx.stroke();
    } else if (shape === 'bulb') {
      // Circle with inner cross filament
      const r = Math.min(this.width * (p.width || 0.06), this.height * (p.height || 0.06)) / 2;
      this.ctx.beginPath();
      this.ctx.arc(cx, cy, r, 0, Math.PI * 2);
      this.ctx.stroke();
      // Filament loop
      this.ctx.beginPath();
      this.ctx.moveTo(cx - r * 0.6, cy + r * 0.6);
      this.ctx.lineTo(cx, cy - r * 0.4);
      this.ctx.lineTo(cx + r * 0.6, cy + r * 0.6);
      this.ctx.stroke();
    } else if (shape === 'switch') {
      // Circuit switch contact
      const w = this.width * (p.width || 0.08);
      this.ctx.beginPath();
      this.ctx.arc(cx, cy, 3, 0, Math.PI * 2);
      this.ctx.arc(cx + w, cy, 3, 0, Math.PI * 2);
      this.ctx.fill();
      // Lever
      this.ctx.beginPath();
      this.ctx.moveTo(cx, cy);
      this.ctx.lineTo(cx + w * 0.9, cy - 10);
      this.ctx.stroke();
    } else if (shape === 'rectangle') {
      const w = this.width * (p.width || 0.1);
      const h = this.height * (p.height || 0.1);
      this.ctx.strokeRect(cx - w / 2, cy - h / 2, w, h);
    } else if (shape === 'circle') {
      const r = Math.min(this.width * (p.width || 0.1), this.height * (p.height || 0.1)) / 2;
      this.ctx.beginPath();
      this.ctx.arc(cx, cy, r, 0, Math.PI * 2);
      this.ctx.stroke();
    }
  }

  _drawText(p) {
    this.ctx.font = `${p.size || 14}px sans-serif`;
    this.ctx.fillStyle = p.color || '#1A202C';
    this.ctx.textAlign = p.align || 'left';
    this.ctx.fillText(p.text || '', this.toX(p.x), this.toY(p.y));
  }

  _highlightRegion(p) {
    const [x1, y1, x2, y2] = p.bounds;
    const rx = this.toX(x1);
    const ry = this.toY(y1);
    const rw = this.toX(x2) - rx;
    const rh = this.toY(y2) - ry;

    this.ctx.fillStyle = 'rgba(236, 201, 75, 0.20)';
    this.ctx.fillRect(rx, ry, rw, rh);

    this.ctx.strokeStyle = p.color || '#D69E2E';
    this.ctx.lineWidth = 2;
    this.ctx.setLineDash([4, 4]);
    this.ctx.strokeRect(rx, ry, rw, rh);
  }
}
