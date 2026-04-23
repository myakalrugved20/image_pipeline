// ══════════════════════════════════════════════════════════════════════════
// CJK Helper: Insert zero-width spaces between CJK characters for wrapping
// ══════════════════════════════════════════════════════════════════════════
function _isCJKChar(cp) {
  return (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) ||
         (cp >= 0x3000 && cp <= 0x303F) || (cp >= 0x3040 && cp <= 0x309F) ||
         (cp >= 0x30A0 && cp <= 0x30FF) || (cp >= 0xAC00 && cp <= 0xD7AF) ||
         (cp >= 0xFF00 && cp <= 0xFFEF) || (cp >= 0x20000 && cp <= 0x2A6DF);
}

function _addCJKBreaks(text) {
  let hasCJK = false;
  for (let i = 0; i < text.length; i++) {
    if (_isCJKChar(text.codePointAt(i))) { hasCJK = true; break; }
  }
  if (!hasCJK) return text;
  // Insert thin spaces between CJK characters so Fabric.js can wrap
  let result = "";
  for (let i = 0; i < text.length; i++) {
    result += text[i];
    const cp = text.codePointAt(i);
    if (i < text.length - 1) {
      const nextCp = text.codePointAt(i + 1);
      // Add thin space between two CJK chars, or CJK + punctuation
      if (_isCJKChar(cp) && _isCJKChar(nextCp)) {
        result += " ";
      } else if (_isCJKChar(cp) && !_isCJKChar(nextCp) && text[i + 1] !== " ") {
        result += " ";
      } else if (!_isCJKChar(cp) && _isCJKChar(nextCp) && text[i] !== " ") {
        result += " ";
      }
    }
  }
  return result;
}

// ══════════════════════════════════════════════════════════════════════════
// Fabric.js Patch: Don't justify the last line of each paragraph
// ══════════════════════════════════════════════════════════════════════════
(function() {
  const orig = fabric.Textbox.prototype._getLineStyle;
  // Override _renderTextLine to temporarily switch alignment for last lines
  const origRender = fabric.Textbox.prototype._renderTextLine;
  fabric.Textbox.prototype._renderTextLine = function(method, ctx, line, left, top, lineIndex) {
    if (this.textAlign === "justify" && this._isLastLineOfParagraph(lineIndex)) {
      this.textAlign = "left";
      origRender.call(this, method, ctx, line, left, top, lineIndex);
      this.textAlign = "justify";
    } else {
      origRender.call(this, method, ctx, line, left, top, lineIndex);
    }
  };

  fabric.Textbox.prototype._isLastLineOfParagraph = function(lineIndex) {
    // Last line of the entire text
    if (lineIndex === this._textLines.length - 1) return true;
    // Use _styleMap to check if the next line belongs to a different hard paragraph
    if (this._styleMap) {
      const cur = this._styleMap[lineIndex];
      const next = this._styleMap[lineIndex + 1];
      if (cur && next && cur.line !== next.line) return true;
    }
    return false;
  };
})();

// ══════════════════════════════════════════════════════════════════════════
// State
// ══════════════════════════════════════════════════════════════════════════
let selectedFile  = null;
let fc            = null;   // Fabric.js canvas (editor)
let selFc         = null;   // Fabric.js canvas (selection)
let imgW          = 0;
let imgH          = 0;
let zoomLevel     = 1;
let baseScale     = 1;
let selBaseScale  = 1;
let currentTool   = "move";
let appPhase      = "upload";   // "upload" | "select" | "review" | "editor" | "export"
let detectData    = null;       // Stores /phase1-detect response
let phaseData     = null;       // Stores /phase1-clean response
let drawMode      = false;
let deleteMode    = false;
let cleanupMode   = false;
let cleanupFc     = null;
let cleanupScale  = 1;

// ══════════════════════════════════════════════════════════════════════════
// DOM — Upload
// ══════════════════════════════════════════════════════════════════════════
const dropZone     = document.getElementById("drop-zone");
const fileInput    = document.getElementById("file-input");
const translateBtn = document.getElementById("translate-btn");
const loading      = document.getElementById("loading");
const errorMsg     = document.getElementById("error-msg");
const previewSec   = document.getElementById("preview-section");
const originalImg  = document.getElementById("original-preview");
const sourceLang   = document.getElementById("source-lang");
const targetLang   = document.getElementById("target-lang");
const uploadView   = document.getElementById("upload-view");

// DOM — Stepper
const stepperBar   = document.getElementById("stepper-bar");

// DOM — Selection (Phase 1)
const selectView   = document.getElementById("select-view");
const selBack      = document.getElementById("sel-back");
const selSelectAll = document.getElementById("sel-select-all");
const selDeselectAll = document.getElementById("sel-deselect-all");
const selDraw       = document.getElementById("sel-draw");
const selDeleteMode = document.getElementById("sel-delete-mode");
const selClean     = document.getElementById("sel-clean");
const selCount     = document.getElementById("sel-count");
const selWorkspace = document.getElementById("select-workspace");
const selCanvasCont = document.getElementById("select-canvas-container");
const selLoading   = document.getElementById("sel-loading");

// DOM — Magnifier (Phase 1)
const magnifierCanvas = document.getElementById("magnifier-canvas");
const magnifierCoords = document.getElementById("magnifier-coords");
let magnifierCtx = null;
let magnifierImg = null;
const MAGNIFIER_ZOOM = 3;

// DOM — Review (Phase 2)
const phase1View   = document.getElementById("phase1-view");
const p1Original   = document.getElementById("p1-original");
const p1Clean      = document.getElementById("p1-clean");
const p1Back       = document.getElementById("p1-back");
const p1Next       = document.getElementById("p1-next");
const p1Info       = document.getElementById("p1-info");
const p1CleanWrap      = document.getElementById("p1-clean-wrap");
const p1CleanupCanvas  = document.getElementById("p1-cleanup-canvas");
const p1CleanupToggle  = document.getElementById("p1-cleanup-toggle");
const p1ApplyCleanup   = document.getElementById("p1-apply-cleanup");
const p1ClearBoxes     = document.getElementById("p1-clear-boxes");
const p1CleanupLoading = document.getElementById("p1-cleanup-loading");

// DOM — Editor (Phase 3)
const editorView   = document.getElementById("editor-view");
const btnBack      = document.getElementById("btn-back");
const toolMove     = document.getElementById("tool-move");
const toolText     = document.getElementById("tool-text");
const btnZoomIn    = document.getElementById("btn-zoom-in");
const btnZoomOut   = document.getElementById("btn-zoom-out");
const btnZoomFit   = document.getElementById("btn-zoom-fit");
const zoomLabel    = document.getElementById("zoom-level");
const btnExport    = document.getElementById("btn-export");
const noSel        = document.getElementById("no-sel");
const propsDiv     = document.getElementById("props");
const multiSelDiv  = document.getElementById("multi-sel");
const btnMatchSize = document.getElementById("btn-match-size");
const btnMatchStyle = document.getElementById("btn-match-style");
const pText        = document.getElementById("p-text");
const pSize        = document.getElementById("p-size");
const pSizeVal     = document.getElementById("p-size-val");
const pColor       = document.getElementById("p-color");
const pFont        = document.getElementById("p-font");
const pBold        = document.getElementById("p-bold");
const pItalic      = document.getElementById("p-italic");
const pUnderline   = document.getElementById("p-underline");
const pOpacity     = document.getElementById("p-opacity");
const pOpacityVal  = document.getElementById("p-opacity-val");
const pX           = document.getElementById("p-x");
const pY           = document.getElementById("p-y");
const layerListEl  = document.getElementById("layer-list");

// Track the last editing textbox and its selection (survives button clicks)
var _lastEditObj = null;
var _lastSelStart = 0;
var _lastSelEnd = 0;
const lUp          = document.getElementById("l-up");
const lDown        = document.getElementById("l-down");
const lDup         = document.getElementById("l-dup");
const lDel         = document.getElementById("l-del");
const statusSize   = document.getElementById("status-size");
const statusInfo   = document.getElementById("status-info");
const workspace    = document.getElementById("ed-workspace");
const scrollCont   = document.getElementById("ed-canvas-scroll");
const canvasCont   = document.getElementById("ed-canvas-container");
const btnTogglePreview = document.getElementById("btn-toggle-preview");
const edPreviewPanel   = document.getElementById("ed-preview-panel");
const edPreviewImg     = document.getElementById("ed-preview-img");
const btnDownloadClean = document.getElementById("btn-download-clean");
const pSuperscript = document.getElementById("p-superscript");
const pSubscript   = document.getElementById("p-subscript");
const btnUploadBg      = document.getElementById("btn-upload-bg");
const bgFileInput      = document.getElementById("bg-file-input");

// DOM — Export (Phase 4)
const phase3View   = document.getElementById("phase3-view");
const p3Loading    = document.getElementById("p3-loading");
const p3Content    = document.getElementById("p3-content");
const p3Original   = document.getElementById("p3-original");
const p3Result     = document.getElementById("p3-result");
const p3Back       = document.getElementById("p3-back");
const p3Download   = document.getElementById("p3-download");
const p3DownClean  = document.getElementById("p3-download-clean");

// ══════════════════════════════════════════════════════════════════════════
// Phase state machine
// ══════════════════════════════════════════════════════════════════════════
function setPhase(phase) {
  appPhase = phase;

  // Hide all views
  uploadView.classList.add("hidden");
  selectView.classList.add("hidden");
  phase1View.classList.add("hidden");
  editorView.classList.add("hidden");
  phase3View.classList.add("hidden");

  // Stepper visibility
  stepperBar.classList.toggle("hidden", phase === "upload");

  // Map phases to step numbers
  const stepMap = { select: 1, review: 2, editor: 3, export: 4 };
  const currentNum = stepMap[phase] || 0;
  document.querySelectorAll("#stepper-bar .step").forEach(el => {
    const s = parseInt(el.dataset.step);
    el.classList.remove("active", "completed");
    if (s === currentNum) el.classList.add("active");
    else if (s < currentNum) el.classList.add("completed");
  });

  // Show the right view
  switch (phase) {
    case "upload":  uploadView.classList.remove("hidden"); break;
    case "select":  selectView.classList.remove("hidden"); break;
    case "review":  phase1View.classList.remove("hidden"); break;
    case "editor":  editorView.classList.remove("hidden"); break;
    case "export":  phase3View.classList.remove("hidden"); break;
  }
}

// ══════════════════════════════════════════════════════════════════════════
// Languages
// ══════════════════════════════════════════════════════════════════════════
(async () => {
  try {
    const res = await fetch("/languages");
    const data = await res.json();
    Object.entries(data).sort(([a],[b]) => a.localeCompare(b)).forEach(([name, code]) => {
      sourceLang.appendChild(new Option(name.charAt(0).toUpperCase() + name.slice(1), code));
      targetLang.appendChild(new Option(name.charAt(0).toUpperCase() + name.slice(1), code));
    });
    targetLang.value = "en";
  } catch {}
})();

// ══════════════════════════════════════════════════════════════════════════
// Upload / Drop
// ══════════════════════════════════════════════════════════════════════════
dropZone.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("dragover", e => { e.preventDefault(); dropZone.classList.add("dragover"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
dropZone.addEventListener("drop", e => {
  e.preventDefault(); dropZone.classList.remove("dragover");
  if (e.dataTransfer.files.length) pickFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => { if (fileInput.files.length) pickFile(fileInput.files[0]); });

function pickFile(f) {
  if (!f.type.startsWith("image/")) { showErr("Please select an image file."); return; }
  selectedFile = f; hideErr();
  originalImg.src = URL.createObjectURL(f);
  previewSec.classList.remove("hidden");
  translateBtn.disabled = false;
  dropZone.querySelector("p").innerHTML =
    `<strong>${f.name}</strong><br><span class="small">Click or drop to change</span>`;
}

// ══════════════════════════════════════════════════════════════════════════
// Detect Text → Selection Phase
// ══════════════════════════════════════════════════════════════════════════
translateBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  hideErr(); loading.classList.remove("hidden"); translateBtn.disabled = true;

  const fd = new FormData();
  fd.append("file", selectedFile);
  fd.append("source_lang", sourceLang.value);
  fd.append("target_lang", targetLang.value);

  try {
    const res = await fetch("/phase1-detect", { method: "POST", body: fd });
    if (!res.ok) {
      const e = await res.json().catch(() => ({ detail: "Unknown error" }));
      throw new Error(e.detail || `Server ${res.status}`);
    }
    detectData = await res.json();
    openSelectionPhase();
  } catch (e) { showErr(e.message); }
  finally { loading.classList.add("hidden"); translateBtn.disabled = false; }
});

// ══════════════════════════════════════════════════════════════════════════
// PHASE 1: Selection — draw/toggle boxes on original image
// ══════════════════════════════════════════════════════════════════════════

/* Color a box based on its _pendingDelete / _selected flags */
function styleBox(r) {
  if (r._pendingDelete) {
    r.set({
      fill:            "rgba(243,139,168,0.40)",
      stroke:          "#f38ba8",
      strokeWidth:     3,
      strokeDashArray: null,
    });
    return;
  }
  r.set({ strokeWidth: 2, strokeDashArray: [6, 3] });
  if (r._selected) {
    r.set({
      fill: r._isCustomRegion ? "rgba(166,227,161,0.25)" : "rgba(137,180,250,0.25)",
      stroke: r._isCustomRegion ? "#a6e3a1" : "#89b4fa",
    });
  } else {
    r.set({ fill: "rgba(243,139,168,0.15)", stroke: "#f38ba8" });
  }
}

/* Build a Fabric rect that can be moved / resized but not rotated */
function buildBox(props) {
  var r = new fabric.Rect(Object.assign({
    strokeWidth:        2,
    strokeDashArray:    [6, 3],
    selectable:         true,
    evented:            true,
    hasControls:        true,
    hasBorders:         true,
    lockRotation:       true,
    cornerColor:        "#cdd6f4",
    cornerSize:         8,
    transparentCorners: false,
    cornerStyle:        "circle",
    borderColor:        "#cdd6f4",
    hoverCursor:        "move",
    padding:            0,
  }, props));
  r.setControlsVisibility({ mtr: false });   // hide rotate handle
  return r;
}

/* Delete all boxes currently marked _pendingDelete (Delete Mode lasso) */
function deletePendingBoxes() {
  if (!selFc) return;
  var toRemove = selFc.getObjects().filter(function(o) {
    return (o._isOcrRegion || o._isCustomRegion) && o._pendingDelete;
  });
  if (toRemove.length === 0) return;
  selFc.discardActiveObject();
  toRemove.forEach(function(o) { selFc.remove(o); });
  selFc.renderAll();
  updateSelCount();
}

// ══════════════════════════════════════════════════════════════════════════
// Magnifier
// ══════════════════════════════════════════════════════════════════════════
function drawMagnifier(origX, origY) {
  if (!magnifierCtx || !magnifierImg || !magnifierImg.complete) return;
  var cw = magnifierCanvas.width;
  var ch = magnifierCanvas.height;
  var viewW = cw / MAGNIFIER_ZOOM;
  var viewH = ch / MAGNIFIER_ZOOM;
  var sx = origX - viewW / 2;
  var sy = origY - viewH / 2;

  magnifierCtx.clearRect(0, 0, cw, ch);
  magnifierCtx.fillStyle = "#15152a";
  magnifierCtx.fillRect(0, 0, cw, ch);
  magnifierCtx.drawImage(magnifierImg, sx, sy, viewW, viewH, 0, 0, cw, ch);

  // Crosshair
  var cx = cw / 2, cy = ch / 2;
  magnifierCtx.save();
  magnifierCtx.strokeStyle = "rgba(0, 0, 0, 0.8)";
  magnifierCtx.lineWidth = 1;
  magnifierCtx.setLineDash([4, 4]);
  magnifierCtx.beginPath(); magnifierCtx.moveTo(0, cy); magnifierCtx.lineTo(cw, cy); magnifierCtx.stroke();
  magnifierCtx.beginPath(); magnifierCtx.moveTo(cx, 0); magnifierCtx.lineTo(cx, ch); magnifierCtx.stroke();
  magnifierCtx.setLineDash([]);
  magnifierCtx.beginPath(); magnifierCtx.arc(cx, cy, 3, 0, Math.PI * 2); magnifierCtx.stroke();
  magnifierCtx.restore();

  magnifierCoords.textContent = "X: " + Math.round(origX) + "  Y: " + Math.round(origY);
}

function clearMagnifier() {
  if (!magnifierCtx) return;
  var cw = magnifierCanvas.width;
  var ch = magnifierCanvas.height;
  magnifierCtx.clearRect(0, 0, cw, ch);
  magnifierCtx.fillStyle = "#15152a";
  magnifierCtx.fillRect(0, 0, cw, ch);
  magnifierCoords.textContent = "X: \u2014 Y: \u2014";
}

function openSelectionPhase() {
  setPhase("select");
  if (selFc) { selFc.dispose(); selFc = null; }
  drawMode = false;
  deleteMode = false;
  selDraw.classList.remove("tb-active");
  if (selDeleteMode) selDeleteMode.classList.remove("tb-active");

  requestAnimationFrame(function() {
    var wsW = selWorkspace.clientWidth - 80;
    var wsH = selWorkspace.clientHeight - 80;
    selBaseScale = Math.min(wsW / detectData.width, wsH / detectData.height, 1);
    if (selBaseScale <= 0) selBaseScale = 0.5;

    var dw = Math.round(detectData.width  * selBaseScale);
    var dh = Math.round(detectData.height * selBaseScale);

    selCanvasCont.style.width  = dw + "px";
    selCanvasCont.style.height = dh + "px";

    var canvasEl = document.getElementById("select-canvas");
    canvasEl.width  = dw;
    canvasEl.height = dh;

    selFc = new fabric.Canvas("select-canvas", {
      width:  dw,
      height: dh,
      selection: false,               // no rubber-band multi-select
      preserveObjectStacking: true,
    });

    /* background -------------------------------------------------------- */
    fabric.Image.fromURL(detectData.original, function(img) {
      img.scaleToWidth(dw);
      selFc.setBackgroundImage(img, selFc.renderAll.bind(selFc));
    }, { crossOrigin: "anonymous" });

    /* Magnifier setup --------------------------------------------------- */
    magnifierImg = new Image();
    magnifierImg.crossOrigin = "anonymous";
    magnifierImg.src = detectData.original;
    magnifierCtx = magnifierCanvas.getContext("2d");
    // Size canvas to its CSS-rendered dimensions after layout settles
    setTimeout(function() {
      var rect = magnifierCanvas.getBoundingClientRect();
      magnifierCanvas.width = Math.round(rect.width);
      magnifierCanvas.height = Math.round(rect.height);
      clearMagnifier();
    }, 50);

    /* OCR boxes --------------------------------------------------------- */
    detectData.regions.forEach(function(r, i) {
      var rect = buildBox({
        left:   r.x      * selBaseScale,
        top:    r.y      * selBaseScale,
        width:  r.width  * selBaseScale,
        height: r.height * selBaseScale,
        fill:   "rgba(137,180,250,0.25)",
        stroke: "#89b4fa",
        _regionIdx:   i,
        _selected:    true,
        _isOcrRegion: true,
        _regionData:  JSON.parse(JSON.stringify(r)),
      });
      selFc.add(rect);
    });
    selFc.renderAll();
    updateSelCount();

    /* ── move / resize vs click detection ─────────────────────────────── */
    var _didDrag = false;

    selFc.on("object:moving",  function() { _didDrag = true; });
    selFc.on("object:scaling", function() { _didDrag = true; });

    /* right-click on any box → immediate delete (single-box, any mode) */
    selFc.on("mouse:down", function(opt) {
      if (opt.button !== 3) return;
      var t = opt.target;
      if (!t || (!t._isOcrRegion && !t._isCustomRegion)) return;
      if (opt.e && opt.e.preventDefault) opt.e.preventDefault();
      selFc.remove(t);
      selFc.renderAll();
      updateSelCount();
    });

    /* suppress native context menu over the select canvas */
    var _selCanvasEl = document.getElementById("select-canvas");
    if (_selCanvasEl) {
      _selCanvasEl.oncontextmenu = function(e) { e.preventDefault(); return false; };
    }

    selFc.on("mouse:down", function(opt) {
      if (opt.button === 3) return;   // right-click handled above
      _didDrag = false;
      /* start lasso selection when clicking empty space outside draw mode */
      if (!drawMode && !opt.target) {
        var ptr = selFc.getPointer(opt.e);
        _lassoOrigin = { x: ptr.x, y: ptr.y };
        _lassoRect = new fabric.Rect({
          left: ptr.x, top: ptr.y, width: 0, height: 0,
          fill:            "rgba(205,214,244,0.08)",
          stroke:          "rgba(205,214,244,0.7)",
          strokeWidth:     1,
          strokeDashArray: [4, 4],
          selectable:      false,
          evented:         false,
          excludeFromExport: true,
        });
        selFc.add(_lassoRect);
      }
    });

    selFc.on("mouse:up", function(opt) {
      if (drawMode) return;          // handled by draw-mode block below
      var t = opt.target;
      if (!t || (!t._isOcrRegion && !t._isCustomRegion)) return;

      if (_didDrag) {
        /* box was moved or resized ─ update stored coords */
        if (t._isOcrRegion && t._regionData) {
          var sx = t.scaleX || 1, sy = t.scaleY || 1;
          t._regionData.x      = Math.round(t.left / selBaseScale);
          t._regionData.y      = Math.round(t.top  / selBaseScale);
          t._regionData.width  = Math.round((t.width  * sx) / selBaseScale);
          t._regionData.height = Math.round((t.height * sy) / selBaseScale);
        }
      } else if (deleteMode) {
        /* pure click in delete mode ─ toggle pending-delete mark */
        t._pendingDelete = !t._pendingDelete;
        styleBox(t);
        selFc.renderAll();
      } else {
        /* pure click ─ toggle selected/deselected */
        t._selected = !t._selected;
        styleBox(t);
        selFc.renderAll();
        updateSelCount();
      }
      _didDrag = false;
    });

    /* ── draw mode ─────────────────────────────────────────────────────── */
    var _drawOrigin = null;
    var _drawRect   = null;
    var _lassoOrigin = null;
    var _lassoRect   = null;

    selFc.on("mouse:down", function(opt) {
      if (opt.button === 3) return;     // right-click is delete, not draw
      if (!drawMode) return;
      if (opt.target) return;           // clicked an existing box
      selFc.discardActiveObject();      // clear selection
      var ptr = selFc.getPointer(opt.e);
      _drawOrigin = { x: ptr.x, y: ptr.y };
      _drawRect = new fabric.Rect({
        left: ptr.x, top: ptr.y, width: 0, height: 0,
        fill: "rgba(166,227,161,0.25)",
        stroke: "#a6e3a1",
        strokeWidth: 2,
        strokeDashArray: [6, 3],
        selectable: false,
        evented:    false,
        _selected:       true,
        _isCustomRegion: true,
      });
      selFc.add(_drawRect);
    });

    selFc.on("mouse:move", function(opt) {
      if (!drawMode || !_drawOrigin || !_drawRect) return;
      var ptr = selFc.getPointer(opt.e);
      _drawRect.set({
        left:   Math.min(_drawOrigin.x, ptr.x),
        top:    Math.min(_drawOrigin.y, ptr.y),
        width:  Math.abs(ptr.x - _drawOrigin.x),
        height: Math.abs(ptr.y - _drawOrigin.y),
      });
      selFc.renderAll();
    });

    /* ── lasso drag-select ────────────────────────────────────────────── */
    selFc.on("mouse:move", function(opt) {
      if (!_lassoOrigin || !_lassoRect) return;
      var ptr = selFc.getPointer(opt.e);
      _lassoRect.set({
        left:   Math.min(_lassoOrigin.x, ptr.x),
        top:    Math.min(_lassoOrigin.y, ptr.y),
        width:  Math.abs(ptr.x - _lassoOrigin.x),
        height: Math.abs(ptr.y - _lassoOrigin.y),
      });
      selFc.renderAll();
    });

    selFc.on("mouse:up", function() {
      if (!drawMode || !_drawRect) return;
      if (_drawRect.width < 5 || _drawRect.height < 5) {
        selFc.remove(_drawRect);
      } else {
        /* finished drawing → make it fully interactive */
        _drawRect.set({
          selectable:         true,
          evented:            true,
          hasControls:        true,
          hasBorders:         true,
          lockRotation:       true,
          cornerColor:        "#cdd6f4",
          cornerSize:         8,
          transparentCorners: false,
          cornerStyle:        "circle",
          borderColor:        "#cdd6f4",
          hoverCursor:        "move",
        });
        _drawRect.setControlsVisibility({ mtr: false });
        selFc.setActiveObject(_drawRect);
        selFc.renderAll();
      }
      _drawOrigin = null;
      _drawRect   = null;
      updateSelCount();
    });

    selFc.on("mouse:up", function() {
      if (!_lassoRect) return;
      if (_lassoRect.width > 3 && _lassoRect.height > 3) {
        var lL = _lassoRect.left,  lT = _lassoRect.top;
        var lR = lL + _lassoRect.width, lB = lT + _lassoRect.height;
        selFc.getObjects().forEach(function(obj) {
          if (obj === _lassoRect) return;
          if (!obj._isOcrRegion && !obj._isCustomRegion) return;
          var sx = obj.scaleX || 1, sy = obj.scaleY || 1;
          var oL = obj.left, oT = obj.top;
          var oR = oL + obj.width * sx, oB = oT + obj.height * sy;
          if (oL < lR && oR > lL && oT < lB && oB > lT) {
            if (deleteMode) obj._pendingDelete = true;
            else            obj._selected      = true;
            styleBox(obj);
          }
        });
        updateSelCount();
      }
      selFc.remove(_lassoRect);
      _lassoOrigin = null;
      _lassoRect   = null;
      selFc.renderAll();
    });

    /* Magnifier tracking ------------------------------------------------- */
    selFc.on("mouse:move", function(opt) {
      if (!magnifierCtx || !magnifierImg) return;
      var ptr = selFc.getPointer(opt.e);
      drawMagnifier(ptr.x / selBaseScale, ptr.y / selBaseScale);
    });
    selFc.on("mouse:out", clearMagnifier);

    /* keyboard delete */
    document.addEventListener("keydown", _selKeyHandler);
  });
}

/* keyboard handler – lives outside so we can remove it */
function _selKeyHandler(e) {
  if (appPhase !== "select" || !selFc) return;
  if (e.key === "Delete" || e.key === "Backspace") {
    if (!deleteMode) return;   // no-op in normal mode (right-click is the single-delete affordance)
    e.preventDefault();
    deletePendingBoxes();
  }
}

function updateSelCount() {
  if (!selFc) return;
  var n = selFc.getObjects().filter(function(o) {
    return (o._isOcrRegion || o._isCustomRegion) && o._selected;
  }).length;
  selCount.textContent = n + " selected";
}

/* ── toolbar buttons ──────────────────────────────────────────────────── */
selBack.addEventListener("click", function() {
  document.removeEventListener("keydown", _selKeyHandler);
  if (selFc) { selFc.dispose(); selFc = null; }
  magnifierImg = null;
  magnifierCtx = null;
  setPhase("upload");
});

selSelectAll.addEventListener("click", function() {
  if (!selFc) return;
  selFc.getObjects().forEach(function(o) {
    if (o._isOcrRegion || o._isCustomRegion) { o._selected = true; styleBox(o); }
  });
  selFc.renderAll();
  updateSelCount();
});

selDeselectAll.addEventListener("click", function() {
  if (!selFc) return;
  selFc.getObjects().forEach(function(o) {
    if (o._isOcrRegion || o._isCustomRegion) { o._selected = false; styleBox(o); }
  });
  selFc.renderAll();
  updateSelCount();
});

selDraw.addEventListener("click", function() {
  drawMode = !drawMode;
  selDraw.classList.toggle("tb-active", drawMode);
  if (!selFc) return;
  /* entering draw mode disables delete mode */
  if (drawMode && deleteMode) {
    deleteMode = false;
    if (selDeleteMode) selDeleteMode.classList.remove("tb-active");
    selFc.getObjects().forEach(function(o) {
      if (o._pendingDelete) { o._pendingDelete = false; styleBox(o); }
    });
  }
  selFc.defaultCursor = drawMode ? "crosshair" : "default";
  /* while drawing, make existing boxes non-selectable so drags create new rects */
  selFc.forEachObject(function(o) {
    if (o._isOcrRegion || o._isCustomRegion) {
      o.selectable = !drawMode;
    }
  });
  if (drawMode) selFc.discardActiveObject();
  selFc.renderAll();
});

if (selDeleteMode) selDeleteMode.addEventListener("click", function() {
  deleteMode = !deleteMode;
  selDeleteMode.classList.toggle("tb-active", deleteMode);
  if (!selFc) return;
  /* leaving delete mode: clear all pending flags and restyle */
  if (!deleteMode) {
    selFc.getObjects().forEach(function(o) {
      if (o._pendingDelete) { o._pendingDelete = false; styleBox(o); }
    });
  }
  /* mutually exclusive with draw mode */
  if (deleteMode && drawMode) {
    drawMode = false;
    selDraw.classList.remove("tb-active");
    selFc.forEachObject(function(o) {
      if (o._isOcrRegion || o._isCustomRegion) o.selectable = true;
    });
  }
  selFc.defaultCursor = deleteMode ? "crosshair" : "default";
  selFc.renderAll();
});

// ══════════════════════════════════════════════════════════════════════════
// Clean Selected → Review Phase
// ══════════════════════════════════════════════════════════════════════════
selClean.addEventListener("click", async () => {
  if (!selFc || !detectData) return;

  // Gather selected regions
  const selectedRegions = [];
  selFc.getObjects().forEach(o => {
    if (!o._selected) return;
    if (o._isOcrRegion) {
      const scaleX = o.scaleX || 1;
      const scaleY = o.scaleY || 1;
      const newX = Math.round(o.left / selBaseScale);
      const newY = Math.round(o.top / selBaseScale);
      const newW = Math.round((o.width * scaleX) / selBaseScale);
      const newH = Math.round((o.height * scaleY) / selBaseScale);
      const orig = o._regionData;
      // Check if box was moved or resized
      const wasMoved = (newX !== orig.x || newY !== orig.y ||
                        newW !== orig.width || newH !== orig.height);
      const updatedData = {
        originalText: orig.originalText,
        translatedText: orig.translatedText,
        x: newX, y: newY, width: newW, height: newH,
      };
      // Keep tight OCR vertices only if the box wasn't moved/resized
      if (!wasMoved && orig.vertices) {
        updatedData.vertices = orig.vertices;
      }
      selectedRegions.push(updatedData);
    } else if (o._isCustomRegion) {
      const scaleX = o.scaleX || 1;
      const scaleY = o.scaleY || 1;
      selectedRegions.push({
        originalText: "",
        translatedText: "",
        x: Math.round(o.left / selBaseScale),
        y: Math.round(o.top / selBaseScale),
        width: Math.round((o.width * scaleX) / selBaseScale),
        height: Math.round((o.height * scaleY) / selBaseScale),
      });
    }
  });

  if (selectedRegions.length === 0) {
    alert("No regions selected. Click boxes to select areas to clean.");
    return;
  }

  selLoading.classList.remove("hidden");

  try {
    const res = await fetch("/phase1-clean", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        original_image: detectData.original,
        selected_regions: selectedRegions,
        target_lang: targetLang.value,
        mask_mode: document.getElementById("mask-mode").value,
        inpaint_method: document.getElementById("inpaint-method").value,
      }),
    });
    if (!res.ok) {
      const e = await res.json().catch(() => ({ detail: "Unknown error" }));
      throw new Error(e.detail || `Server ${res.status}`);
    }
    phaseData = await res.json();
    openReviewPhase();
  } catch (e) {
    showErr(e.message);
  } finally {
    selLoading.classList.add("hidden");
  }
});

// ══════════════════════════════════════════════════════════════════════════
// PHASE 2: Review Clean
// ══════════════════════════════════════════════════════════════════════════
function exitCleanupMode() {
  cleanupMode = false;
  if (cleanupFc) { cleanupFc.dispose(); cleanupFc = null; }
  p1CleanupCanvas.classList.add("hidden");
  p1Clean.classList.remove("hidden");
  p1CleanupToggle.textContent = "\u270E Manual Cleanup";
  p1ApplyCleanup.classList.add("hidden");
  p1ClearBoxes.classList.add("hidden");
}

function enterCleanupMode() {
  cleanupMode = true;
  p1CleanupToggle.textContent = "Cancel Cleanup";
  p1ApplyCleanup.classList.remove("hidden");
  p1ClearBoxes.classList.remove("hidden");

  // Size canvas to match the displayed clean image
  const wrapW = p1CleanWrap.clientWidth;
  const wrapH = p1CleanWrap.clientHeight;
  const iw = phaseData.width, ih = phaseData.height;
  const scale = Math.min(wrapW / iw, wrapH / ih, 1);
  cleanupScale = scale;
  const dw = Math.round(iw * scale);
  const dh = Math.round(ih * scale);

  p1Clean.classList.add("hidden");
  p1CleanupCanvas.classList.remove("hidden");
  p1CleanupCanvas.width = dw;
  p1CleanupCanvas.height = dh;
  p1CleanupCanvas.style.width = dw + "px";
  p1CleanupCanvas.style.height = dh + "px";

  if (cleanupFc) { cleanupFc.dispose(); cleanupFc = null; }
  cleanupFc = new fabric.Canvas("p1-cleanup-canvas", {
    width: dw, height: dh, selection: false,
  });

  // Set background to current clean image
  fabric.Image.fromURL(phaseData.clean, function(img) {
    img.scaleToWidth(dw);
    cleanupFc.setBackgroundImage(img, cleanupFc.renderAll.bind(cleanupFc));
  }, { crossOrigin: "anonymous" });

  // Drawing logic
  var _origin = null, _rect = null;

  cleanupFc.on("mouse:down", function(opt) {
    if (opt.target) return;
    cleanupFc.discardActiveObject();
    var ptr = cleanupFc.getPointer(opt.e);
    _origin = { x: ptr.x, y: ptr.y };
    _rect = new fabric.Rect({
      left: ptr.x, top: ptr.y, width: 0, height: 0,
      fill: "rgba(250,179,135,0.25)",
      stroke: "#fab387",
      strokeWidth: 2,
      strokeDashArray: [6, 3],
      selectable: false, evented: false,
      _isCleanupBox: true,
    });
    cleanupFc.add(_rect);
  });

  cleanupFc.on("mouse:move", function(opt) {
    if (!_origin || !_rect) return;
    var ptr = cleanupFc.getPointer(opt.e);
    _rect.set({
      left:   Math.min(_origin.x, ptr.x),
      top:    Math.min(_origin.y, ptr.y),
      width:  Math.abs(ptr.x - _origin.x),
      height: Math.abs(ptr.y - _origin.y),
    });
    cleanupFc.renderAll();
  });

  cleanupFc.on("mouse:up", function() {
    if (!_rect) return;
    if (_rect.width < 5 || _rect.height < 5) {
      cleanupFc.remove(_rect);
    } else {
      _rect.set({
        selectable: true, evented: true,
        hasControls: true, hasBorders: true,
        lockRotation: true,
        cornerColor: "#cdd6f4", cornerSize: 8,
        transparentCorners: false, cornerStyle: "circle",
        borderColor: "#cdd6f4", hoverCursor: "move",
      });
      _rect.setControlsVisibility({ mtr: false });
    }
    _origin = null;
    _rect = null;
  });

  // Delete key removes selected box
  document.addEventListener("keydown", function _cleanupDel(e) {
    if (!cleanupMode) { document.removeEventListener("keydown", _cleanupDel); return; }
    if ((e.key === "Delete" || e.key === "Backspace") && cleanupFc) {
      var active = cleanupFc.getActiveObject();
      if (active && active._isCleanupBox) {
        cleanupFc.remove(active);
        cleanupFc.discardActiveObject();
        cleanupFc.renderAll();
      }
    }
  });
}

function openReviewPhase() {
  p1Original.src = phaseData.original;
  p1Clean.src = phaseData.clean;
  p1Info.textContent = `${phaseData.regions.length} text region(s) to render`;
  exitCleanupMode();
  setPhase("review");
}

p1CleanupToggle.addEventListener("click", function() {
  if (cleanupMode) {
    exitCleanupMode();
  } else {
    enterCleanupMode();
  }
});

p1ClearBoxes.addEventListener("click", function() {
  if (!cleanupFc) return;
  var objs = cleanupFc.getObjects().filter(function(o) { return o._isCleanupBox; });
  objs.forEach(function(o) { cleanupFc.remove(o); });
  cleanupFc.discardActiveObject();
  cleanupFc.renderAll();
});

p1ApplyCleanup.addEventListener("click", async function() {
  if (!cleanupFc) return;
  var boxes = cleanupFc.getObjects().filter(function(o) { return o._isCleanupBox; });
  if (boxes.length === 0) return;

  // Convert canvas coords to image coords
  var rectangles = boxes.map(function(b) {
    var sx = b.scaleX || 1, sy = b.scaleY || 1;
    return {
      x: Math.round(b.left / cleanupScale),
      y: Math.round(b.top / cleanupScale),
      width: Math.round((b.width * sx) / cleanupScale),
      height: Math.round((b.height * sy) / cleanupScale),
    };
  });

  p1CleanupLoading.classList.remove("hidden");
  try {
    var resp = await fetch("/manual-inpaint", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: phaseData.clean, rectangles: rectangles, inpaint_method: document.getElementById("inpaint-method").value }),
    });
    if (!resp.ok) throw new Error("Cleanup failed");
    var data = await resp.json();
    phaseData.clean = data.clean;

    // Refresh: exit cleanup mode and show updated clean image
    exitCleanupMode();
    p1Clean.src = phaseData.clean;
  } catch (e) {
    alert("Cleanup failed: " + e.message);
  } finally {
    p1CleanupLoading.classList.add("hidden");
  }
});

p1Back.addEventListener("click", () => { exitCleanupMode(); setPhase("select"); });
p1Next.addEventListener("click", () => {
  exitCleanupMode();
  const editorData = {
    background: phaseData.clean,
    width: phaseData.width,
    height: phaseData.height,
    layers: phaseData.regions,
  };
  buildCanvas(editorData);
  setPhase("editor");
});

// ══════════════════════════════════════════════════════════════════════════
// PHASE 3: Editor
// ══════════════════════════════════════════════════════════════════════════

function buildCanvas(data) {
  if (fc) { fc.dispose(); fc = null; }

  imgW = data.width;
  imgH = data.height;
  statusSize.textContent = `${imgW} x ${imgH} px`;

  // Set original image in preview panel
  if (phaseData && phaseData.original) {
    edPreviewImg.src = phaseData.original;
  }

  requestAnimationFrame(() => {
    const wsW = workspace.clientWidth - 80;
    const wsH = workspace.clientHeight - 80;
    baseScale = Math.min(wsW / imgW, wsH / imgH, 1);
    if (baseScale <= 0) baseScale = 0.5;
    zoomLevel = 1;

    const dw = Math.round(imgW * baseScale);
    const dh = Math.round(imgH * baseScale);

    canvasCont.style.width = dw + "px";
    canvasCont.style.height = dh + "px";
    centerCanvas(dw, dh);

    const canvasEl = document.getElementById("editor-canvas");
    canvasEl.width = dw;
    canvasEl.height = dh;

    fc = new fabric.Canvas("editor-canvas", {
      width: dw, height: dh,
      selection: true,
      preserveObjectStacking: true,
      backgroundColor: "#000",
    });

    // Background image
    fabric.Image.fromURL(data.background, function(img) {
      img.scaleToWidth(dw);
      fc.setBackgroundImage(img, fc.renderAll.bind(fc));
    }, { crossOrigin: "anonymous" });

    // Text layers — use Textbox with fixed width, auto-fit font to fill bounding box
    data.layers.forEach(l => {
      const ang = l.angle || 0;
      const alignment = l.alignment || "center";
      const detectedFont = l.fontFamily || "sans-serif";
      const fontStack = detectedFont === "sans-serif" ? "sans-serif"
                       : detectedFont + ", sans-serif";

      // Use server-provided text-direction dimensions (accounts for rotation)
      const fitW = (l.fitWidth || l.width) * baseScale;
      const fitH = (l.fitHeight || l.height) * baseScale;

      const opts = {
        width:      fitW,
        fontSize:   l.fontSize * baseScale,
        fill:       l.color,
        fontFamily: fontStack,
        fontWeight: l.bold ? "bold" : "normal",
        fontStyle:  l.italic ? "italic" : "normal",
        underline:  !!l.underline,
        textAlign:  alignment,
        editable:   true,
        opacity:    1,
        angle:      ang,
        _name:      l.translatedText.substring(0, 28),
        _boxW:      l.width,
        _boxH:      l.height,
        _alignment: alignment,
        _styleSpans: l.styleSpans || null,
        _wordStyles: l.wordStyles || null,
        left:       l.x * baseScale,
        top:        l.y * baseScale,
      };

      // For rotated text, position from center of bounding box
      if (Math.abs(ang) > 1) {
        opts.originX = "center";
        opts.originY = "center";
        opts.left = (l.x + l.width / 2) * baseScale;
        opts.top  = (l.y + l.height / 2) * baseScale;
      }

      const t = new fabric.Textbox(_addCJKBreaks(l.translatedText), opts);

      // Apply per-character styles — prefer word-level styles from HTML translation,
      // fall back to proportional styleSpans mapping
      if (l.wordStyles && l.wordStyles.length > 0) {
        // Word-level styles from HTML-based styled translation
        // Map word styles to per-character Fabric.js styles
        const fullText = t.text;
        const totalLen = fullText.length;

        function _isCombiningWS(ch) {
          const cp = ch.codePointAt(0);
          if (!cp) return false;
          if (cp >= 0x0300 && cp <= 0x036F) return true;
          if (cp >= 0x0900 && cp <= 0x0903) return true;
          if (cp >= 0x093A && cp <= 0x094F) return true;
          if (cp >= 0x0951 && cp <= 0x0957) return true;
          if (cp >= 0x0962 && cp <= 0x0963) return true;
          if (cp === 0x200C || cp === 0x200D) return true;
          return false;
        }

        // Build per-character bold/italic/color from word boundaries
        const charBold = new Array(totalLen).fill(false);
        const charItalic = new Array(totalLen).fill(false);
        const charColor = new Array(totalLen).fill(null);
        function _parseHex(h) {
          if (!h || typeof h !== "string" || !h.startsWith("#") || h.length !== 7) return null;
          return [parseInt(h.substr(1,2),16), parseInt(h.substr(3,2),16), parseInt(h.substr(5,2),16)];
        }
        let searchPos = 0;
        for (const ws of l.wordStyles) {
          const word = ws.word;
          const idx = fullText.indexOf(word, searchPos);
          if (idx === -1) continue;
          const isBold = ws.style && ws.style.fontWeight === "bold";
          const isItalic = ws.style && ws.style.fontStyle === "italic";
          const col = ws.style && ws.style.color;
          for (let ci = idx; ci < idx + word.length; ci++) {
            if (isBold) charBold[ci] = true;
            if (isItalic) charItalic[ci] = true;
            if (col) charColor[ci] = col;
          }
          searchPos = idx + word.length;
        }

        const fabricStyles = {};
        let lineIdx = 0, charIdx = 0;
        let lastCharStyle = null;
        for (let ci = 0; ci < totalLen; ci++) {
          if (fullText[ci] === "\n") {
            lineIdx++; charIdx = 0; lastCharStyle = null; continue;
          }
          let charStyle;
          if (_isCombiningWS(fullText[ci]) && lastCharStyle) {
            charStyle = Object.assign({}, lastCharStyle);
          } else {
            charStyle = {};
            if (charBold[ci]) charStyle.fontWeight = "bold";
            if (charItalic[ci]) charStyle.fontStyle = "italic";
            if (charColor[ci]) charStyle.fill = charColor[ci];
            lastCharStyle = Object.keys(charStyle).length > 0 ? charStyle : null;
          }
          if (charStyle && Object.keys(charStyle).length > 0) {
            if (!fabricStyles[lineIdx]) fabricStyles[lineIdx] = {};
            fabricStyles[lineIdx][charIdx] = charStyle;
          }
          charIdx++;
        }
        t.styles = fabricStyles;
        console.log("[WordStyles]", l.translatedText.substring(0, 40),
          "words:", l.wordStyles.length,
          "boldWords:", l.wordStyles.filter(w => w.style && w.style.fontWeight === "bold").length);
      } else if (l.styleSpans && l.styleSpans.length > 0) {
        // Fallback: proportional styleSpans mapping
        const fullText = t.text;
        const totalLen = fullText.length;

        const parseHex = (h) => {
          if (!h || !h.startsWith("#") || h.length !== 7) return null;
          return [parseInt(h.slice(1,3),16), parseInt(h.slice(3,5),16), parseInt(h.slice(5,7),16)];
        };
        const spanColors = l.styleSpans.map(s => parseHex(s.color)).filter(Boolean);
        let hasDistinctColors = false;
        if (spanColors.length >= 2) {
          for (let i = 0; i < spanColors.length && !hasDistinctColors; i++) {
            for (let j = i + 1; j < spanColors.length; j++) {
              const dist = Math.abs(spanColors[i][0]-spanColors[j][0])
                         + Math.abs(spanColors[i][1]-spanColors[j][1])
                         + Math.abs(spanColors[i][2]-spanColors[j][2]);
              if (dist > 150) { hasDistinctColors = true; break; }
            }
          }
        }

        const charStyles = [];
        for (let ci = 0; ci < totalLen; ci++) {
          const prop = totalLen > 0 ? ci / totalLen : 0;
          let matched = l.styleSpans[l.styleSpans.length - 1];
          for (const span of l.styleSpans) {
            if (span.start <= prop && prop < span.end) {
              matched = span;
              break;
            }
          }
          charStyles.push(matched);
        }
        function _isCombining(ch) {
          const cp = ch.codePointAt(0);
          if (!cp) return false;
          if (cp >= 0x0300 && cp <= 0x036F) return true;
          if (cp >= 0x0900 && cp <= 0x0903) return true;
          if (cp >= 0x093A && cp <= 0x094F) return true;
          if (cp >= 0x0951 && cp <= 0x0957) return true;
          if (cp >= 0x0962 && cp <= 0x0963) return true;
          if (cp === 0x200C || cp === 0x200D) return true;
          return false;
        }

        const fabricStyles = {};
        let lineIdx = 0, charIdx = 0;
        let lastCharStyle = null;
        for (let ci = 0; ci < totalLen; ci++) {
          if (fullText[ci] === "\n") {
            lineIdx++; charIdx = 0; lastCharStyle = null; continue;
          }
          let charStyle;
          if (_isCombining(fullText[ci]) && lastCharStyle) {
            charStyle = Object.assign({}, lastCharStyle);
          } else {
            const span = charStyles[ci];
            charStyle = {};
            if (span.fontWeight === "bold") charStyle.fontWeight = "bold";
            if (span.fontStyle === "italic") charStyle.fontStyle = "italic";
            if (hasDistinctColors && span.color) charStyle.fill = span.color;
            lastCharStyle = Object.keys(charStyle).length > 0 ? charStyle : null;
          }
          if (charStyle && Object.keys(charStyle).length > 0) {
            if (!fabricStyles[lineIdx]) fabricStyles[lineIdx] = {};
            fabricStyles[lineIdx][charIdx] = charStyle;
          }
          charIdx++;
        }
        t.styles = fabricStyles;
        console.log("[StyleSpans]", l.translatedText.substring(0, 40),
          "spans:", l.styleSpans.length,
          "boldChars:", Object.values(fabricStyles[0] || {}).filter(s => s.fontWeight === "bold").length,
          "totalChars:", totalLen);
      }

      // Trust server's font size but shrink if browser renders larger than bounding box
      t._clearCache();
      t.initDimensions();
      let fontSize = t.fontSize;
      while ((t.calcTextHeight() > fitH || t.calcTextWidth() > fitW) && fontSize > 4) {
        fontSize = Math.max(4, fontSize - (fontSize > 20 ? 2 : 1));
        t.set("fontSize", fontSize);
        t._clearCache();
        t.initDimensions();
      }

      // Vertically center within the bounding box
      const textH = t.calcTextHeight();
      if (textH < fitH && Math.abs(ang) < 2) {
        const vOffset = (fitH - textH) / 2;
        t.set("top", t.top + vOffset);
      }

      fc.add(t);
    });

    fc.renderAll();
    updateZoomLabel();
    refreshLayers();

    // ── Snapping alignment guidelines ──
    const SNAP_THRESHOLD = 5;
    let _guideLines = []; // Fabric.js Line objects on canvas

    function _clearGuides() {
      _guideLines.forEach(l => fc.remove(l));
      _guideLines = [];
    }

    function _addGuideLine(coords, color) {
      const line = new fabric.Line(coords, {
        stroke: color || "#ff00ff",
        strokeWidth: 1,
        strokeDashArray: [6, 4],
        selectable: false,
        evented: false,
        excludeFromExport: true,
        _isGuideline: true,
      });
      fc.add(line);
      _guideLines.push(line);
    }

    function _calcGuidelines(target) {
      _clearGuides();
      const tb = target.getBoundingRect(true, true);
      const tCx = tb.left + tb.width / 2;
      const tCy = tb.top + tb.height / 2;
      const tPoints = {
        left: tb.left, right: tb.left + tb.width, cx: tCx,
        top: tb.top, bottom: tb.top + tb.height, cy: tCy,
      };

      const snaps = { x: null, y: null };
      const canvasW = fc.width, canvasH = fc.height;

      // Crosshair through the moving object's center
      _addGuideLine([tCx, 0, tCx, canvasH], "rgba(255,0,255,0.35)");
      _addGuideLine([0, tCy, canvasW, tCy], "rgba(255,0,255,0.35)");

      // Canvas center guidelines
      if (Math.abs(tPoints.cx - canvasW / 2) < SNAP_THRESHOLD) {
        snaps.x = canvasW / 2 - tb.width / 2;
        _addGuideLine([canvasW / 2, 0, canvasW / 2, canvasH]);
      }
      if (Math.abs(tPoints.cy - canvasH / 2) < SNAP_THRESHOLD) {
        snaps.y = canvasH / 2 - tb.height / 2;
        _addGuideLine([0, canvasH / 2, canvasW, canvasH / 2]);
      }

      // Object-to-object guidelines
      fc.getObjects().forEach(obj => {
        if (obj === target || obj._isGuideline) return;
        const ob = obj.getBoundingRect(true, true);
        const oCx = ob.left + ob.width / 2;
        const oCy = ob.top + ob.height / 2;
        const oPoints = {
          left: ob.left, right: ob.left + ob.width, cx: oCx,
          top: ob.top, bottom: ob.top + ob.height, cy: oCy,
        };

        // Vertical snaps (x-axis alignment)
        if (snaps.x === null) {
          const vPairs = [
            [tPoints.left, oPoints.left], [tPoints.left, oPoints.right], [tPoints.left, oPoints.cx],
            [tPoints.right, oPoints.left], [tPoints.right, oPoints.right], [tPoints.right, oPoints.cx],
            [tPoints.cx, oPoints.left], [tPoints.cx, oPoints.right], [tPoints.cx, oPoints.cx],
          ];
          for (const [tv, ov] of vPairs) {
            if (Math.abs(tv - ov) < SNAP_THRESHOLD) {
              const offset = tv - tPoints.left;
              snaps.x = ov - offset;
              const y1 = Math.min(tb.top, ob.top) - 20;
              const y2 = Math.max(tb.top + tb.height, ob.top + ob.height) + 20;
              _addGuideLine([ov, y1, ov, y2]);
              break;
            }
          }
        }

        // Horizontal snaps (y-axis alignment)
        if (snaps.y === null) {
          const hPairs = [
            [tPoints.top, oPoints.top], [tPoints.top, oPoints.bottom], [tPoints.top, oPoints.cy],
            [tPoints.bottom, oPoints.top], [tPoints.bottom, oPoints.bottom], [tPoints.bottom, oPoints.cy],
            [tPoints.cy, oPoints.top], [tPoints.cy, oPoints.bottom], [tPoints.cy, oPoints.cy],
          ];
          for (const [tv, ov] of hPairs) {
            if (Math.abs(tv - ov) < SNAP_THRESHOLD) {
              const offset = tv - tPoints.top;
              snaps.y = ov - offset;
              const x1 = Math.min(tb.left, ob.left) - 20;
              const x2 = Math.max(tb.left + tb.width, ob.left + ob.width) + 20;
              _addGuideLine([x1, ov, x2, ov]);
              break;
            }
          }
        }
      });

      // Apply snap
      if (snaps.x !== null) target.set("left", target.left + (snaps.x - tb.left));
      if (snaps.y !== null) target.set("top",  target.top  + (snaps.y - tb.top));
    }

    fc.off();
    fc.on("selection:created",  onSel);
    fc.on("selection:updated",  onSel);
    fc.on("selection:cleared",  onDesel);
    fc.on("object:modified",    () => { _clearGuides(); fc.requestRenderAll(); const o = fc.getActiveObject(); if (o) syncProps(o); });
    fc.on("object:moving",      () => { const o = fc.getActiveObject(); if (o) { _calcGuidelines(o); syncPos(o); } });
    fc.on("mouse:up",           () => { _clearGuides(); fc.requestRenderAll(); });
    fc.on("text:changed",       e => {
      if (e.target) { e.target._name = e.target.text.substring(0,28); refreshLayers(); pText.value = e.target.text; }
    });
    // Track text selection changes inside textboxes
    fc.on("text:selection:changed", e => {
      if (e.target && e.target.isEditing) {
        _lastEditObj = e.target;
        _lastSelStart = e.target.selectionStart;
        _lastSelEnd = e.target.selectionEnd;
      }
    });

    statusInfo.textContent = `${data.layers.length} text layers loaded`;
  });
}

// ── Selection ────────────────────────────────────────────────────────────
let _multiSelectMode = false;

function _getSelectedTextObjects() {
  const o = fc.getActiveObject();
  if (!o) return [];
  if (o.type === "activeSelection") {
    return o.getObjects().filter(x => x.type === "textbox" || x.type === "i-text");
  }
  if (o.type === "textbox" || o.type === "i-text") return [o];
  return [];
}

function onSel() {
  const o = fc.getActiveObject();
  if (!o) { onDesel(); return; }
  // Multi-select (ActiveSelection)
  if (o.type === "activeSelection") {
    const textObjs = o.getObjects().filter(x => x.type === "textbox" || x.type === "i-text");
    if (textObjs.length === 0) { onDesel(); return; }
    _multiSelectMode = true;
    noSel.classList.add("hidden");
    propsDiv.classList.remove("hidden");
    multiSelDiv.classList.remove("hidden");
    // Hide single-object-only fields (text content, position)
    pText.parentElement.classList.add("hidden");
    document.querySelector(".pos-row").parentElement.classList.add("hidden");
    // Sync props from first object in selection
    syncProps(textObjs[0]);
    highlightLayer(null);
    return;
  }
  if (o.type !== "i-text" && o.type !== "textbox") { onDesel(); return; }
  _multiSelectMode = false;
  _lastEditObj = o;
  multiSelDiv.classList.add("hidden");
  pText.parentElement.classList.remove("hidden");
  document.querySelector(".pos-row").parentElement.classList.remove("hidden");
  syncProps(o); noSel.classList.add("hidden"); propsDiv.classList.remove("hidden");
  highlightLayer(o);
}

function onDesel() {
  _multiSelectMode = false;
  noSel.classList.remove("hidden"); propsDiv.classList.add("hidden");
  multiSelDiv.classList.add("hidden");
  pText.parentElement.classList.remove("hidden");
  document.querySelector(".pos-row").parentElement.classList.remove("hidden");
  highlightLayer(null);
}

function syncProps(o) {
  pText.value = o.text;
  const realSize = Math.round(o.fontSize * o.scaleY / baseScale);
  pSize.value = realSize; pSizeVal.value = realSize;
  pColor.value = rgbToHex(o.fill);
  pFont.value = o.fontFamily;
  pBold.classList.toggle("active", o.fontWeight === "bold");
  pItalic.classList.toggle("active", o.fontStyle === "italic");
  pUnderline.classList.toggle("active", !!o.underline);
  pOpacity.value = Math.round(o.opacity * 100);
  pOpacityVal.textContent = Math.round(o.opacity * 100);
  document.querySelectorAll(".align-btn").forEach(b => b.classList.toggle("active", b.dataset.align === o.textAlign));
  syncPos(o);
}

function syncPos(o) {
  pX.value = Math.round(o.left / baseScale);
  pY.value = Math.round(o.top / baseScale);
}

// ── Property controls ────────────────────────────────────────────────────
pText.addEventListener("input", () => {
  const o = active(); if (!o) return;
  o.set("text", pText.value); o._name = pText.value.substring(0, 28);
  fc.renderAll(); refreshLayers();
});

function _applyToAll(fn) {
  if (_multiSelectMode) {
    _getSelectedTextObjects().forEach(fn);
    fc.requestRenderAll();
  } else {
    const o = active(); if (!o) return;
    fn(o);
    fc.requestRenderAll();
  }
}

pSize.addEventListener("input", () => {
  const s = parseInt(pSize.value, 10); pSizeVal.value = s;
  _applyToAll(o => {
    if (!_multiSelectMode && o.isEditing && o.selectionStart !== o.selectionEnd) {
      o.setSelectionStyles({ fontSize: s * baseScale }, o.selectionStart, o.selectionEnd);
      o._forceClearCache = true; o.dirty = true;
    } else {
      o.set({ fontSize: s * baseScale, scaleX: 1, scaleY: 1 });
    }
  });
});

pSizeVal.addEventListener("input", () => {
  let s = parseInt(pSizeVal.value, 10);
  if (isNaN(s) || s < 4) s = 4;
  if (s > 300) s = 300;
  pSize.value = s;
  _applyToAll(o => {
    if (!_multiSelectMode && o.isEditing && o.selectionStart !== o.selectionEnd) {
      o.setSelectionStyles({ fontSize: s * baseScale }, o.selectionStart, o.selectionEnd);
      o._forceClearCache = true; o.dirty = true;
    } else {
      o.set({ fontSize: s * baseScale, scaleX: 1, scaleY: 1 });
    }
  });
});

pColor.addEventListener("input", () => {
  _applyToAll(o => {
    if (!_multiSelectMode && o.isEditing && o.selectionStart !== o.selectionEnd) {
      o.setSelectionStyles({ fill: pColor.value }, o.selectionStart, o.selectionEnd);
      o._forceClearCache = true; o.dirty = true;
    } else {
      o.set("fill", pColor.value);
    }
  });
});

pFont.addEventListener("change", () => {
  _applyToAll(o => {
    if (!_multiSelectMode && o.isEditing && o.selectionStart !== o.selectionEnd) {
      o.setSelectionStyles({ fontFamily: pFont.value }, o.selectionStart, o.selectionEnd);
      o._forceClearCache = true; o.dirty = true;
    } else {
      o.set("fontFamily", pFont.value);
    }
  });
});

// Prevent style controls from stealing focus (preserves text selection in textbox)
// Note: skip range/select inputs as preventDefault breaks their interaction
[pBold, pItalic, pUnderline].forEach(el => {
  el.addEventListener("mousedown", e => e.preventDefault());
});

function _applyPartialStyle(prop, boldVal, normalVal) {
  const o = active(); if (!o) return;
  const start = o.selectionStart;
  const end = o.selectionEnd;
  console.log("[style]", prop, "sel:", start, end, "isEditing:", o.isEditing);
  if (o.isEditing && start !== end) {
    // Partial selection — per-character style
    let allSet = true;
    for (let i = start; i < end; i++) {
      const s = o.getStyleAtPosition(i);
      if (!s || s[prop] !== boldVal) { allSet = false; break; }
    }
    const newVal = allSet ? normalVal : boldVal;
    const styleObj = {}; styleObj[prop] = newVal;
    o.setSelectionStyles(styleObj, start, end);
    o._forceClearCache = true;
    o.dirty = true;
    fc.requestRenderAll();
    console.log("[style] applied", prop, "=", newVal, "to chars", start, "-", end);
  } else {
    // No selection — toggle whole textbox
    const current = o[prop];
    const newVal = current === boldVal ? normalVal : boldVal;
    o.set(prop, newVal);
    fc.requestRenderAll();
  }
}

pBold.addEventListener("click", () => {
  if (_multiSelectMode) {
    const objs = _getSelectedTextObjects();
    const allBold = objs.every(o => o.fontWeight === "bold");
    const newVal = allBold ? "normal" : "bold";
    objs.forEach(o => o.set("fontWeight", newVal));
    fc.requestRenderAll();
    pBold.classList.toggle("active", newVal === "bold");
  } else {
    _applyPartialStyle("fontWeight", "bold", "normal");
    const o = active();
    if (o) pBold.classList.toggle("active", o.fontWeight === "bold");
  }
});
pItalic.addEventListener("click", () => {
  if (_multiSelectMode) {
    const objs = _getSelectedTextObjects();
    const allItalic = objs.every(o => o.fontStyle === "italic");
    const newVal = allItalic ? "normal" : "italic";
    objs.forEach(o => o.set("fontStyle", newVal));
    fc.requestRenderAll();
    pItalic.classList.toggle("active", newVal === "italic");
  } else {
    _applyPartialStyle("fontStyle", "italic", "normal");
    const o = active();
    if (o) pItalic.classList.toggle("active", o.fontStyle === "italic");
  }
});
pUnderline.addEventListener("click", () => {
  if (_multiSelectMode) {
    const objs = _getSelectedTextObjects();
    const allUl = objs.every(o => !!o.underline);
    const newVal = !allUl;
    objs.forEach(o => o.set("underline", newVal));
    fc.requestRenderAll();
    pUnderline.classList.toggle("active", newVal);
  } else {
    const o = active(); if (!o) return;
    const start = o.selectionStart;
    const end = o.selectionEnd;
    if (o.isEditing && start !== end) {
      let allUl = true;
      for (let i = start; i < end; i++) {
        const s = o.getStyleAtPosition(i);
        if (!s || !s.underline) { allUl = false; break; }
      }
      const newVal = !allUl;
      o.setSelectionStyles({ underline: newVal }, start, end);
      o._forceClearCache = true;
      o.dirty = true;
      fc.requestRenderAll();
    } else {
      o.set("underline", !o.underline);
      fc.requestRenderAll();
    }
    pUnderline.classList.toggle("active", !!o.underline);
  }
});

function _applyScriptStyle(direction) {
  // direction: "super" or "sub"
  const o = active(); if (!o) return;
  const start = o.selectionStart;
  const end = o.selectionEnd;
  if (!o.isEditing || start === end) return;

  const baseFontSize = o.fontSize;
  let allApplied = true;
  for (let i = start; i < end; i++) {
    const s = o.getStyleAtPosition(i);
    const dy = (s && s.deltaY) || 0;
    if (direction === "super" && !(dy < 0)) { allApplied = false; break; }
    if (direction === "sub" && !(dy > 0)) { allApplied = false; break; }
  }

  for (let i = start; i < end; i++) {
    const s = o.getStyleAtPosition(i);
    const charSize = (s && s.fontSize) || baseFontSize;
    if (allApplied) {
      // Toggle off: restore size and reset baseline
      const wasScaled = (s && s.deltaY) ? true : false;
      const restored = wasScaled ? Math.round(charSize / 0.58) : charSize;
      o.setSelectionStyles({ fontSize: restored, deltaY: 0 }, i, i + 1);
    } else {
      // Check if opposite script is applied
      const dy = (s && s.deltaY) || 0;
      const isAlreadyScripted = dy !== 0;
      const size = isAlreadyScripted ? Math.round(charSize / 0.58) : charSize;
      const newSize = Math.round(size * 0.58);
      const shift = Math.round(size * 0.4);
      o.setSelectionStyles({
        fontSize: newSize,
        deltaY: direction === "super" ? -shift : shift,
      }, i, i + 1);
    }
  }
  o._forceClearCache = true;
  o.dirty = true;
  fc.requestRenderAll();
}

pSuperscript.addEventListener("click", () => {
  _applyScriptStyle("super");
  pSuperscript.classList.toggle("active");
  pSubscript.classList.remove("active");
});

pSubscript.addEventListener("click", () => {
  _applyScriptStyle("sub");
  pSubscript.classList.toggle("active");
  pSuperscript.classList.remove("active");
});

document.querySelectorAll(".align-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    _applyToAll(o => o.set("textAlign", btn.dataset.align));
    document.querySelectorAll(".align-btn").forEach(b => b.classList.toggle("active", b.dataset.align === btn.dataset.align));
  });
});

pOpacity.addEventListener("input", () => {
  _applyToAll(o => o.set("opacity", parseInt(pOpacity.value, 10) / 100));
  pOpacityVal.textContent = pOpacity.value;
});
pX.addEventListener("change", () => { const o = active(); if (!o) return; o.set("left", parseInt(pX.value, 10) * baseScale); fc.renderAll(); });
pY.addEventListener("change", () => { const o = active(); if (!o) return; o.set("top", parseInt(pY.value, 10) * baseScale); fc.renderAll(); });

// ── Multi-select: Match Font Size & Style ──
btnMatchSize.addEventListener("click", () => {
  const sel = fc.getActiveObject();
  if (!sel || sel.type !== "activeSelection") return;
  const objs = sel.getObjects().filter(o => o.type === "textbox" || o.type === "i-text");
  if (objs.length < 2) return;
  // Find the smallest font size (in real px, accounting for scale)
  const sizes = objs.map(o => Math.round(o.fontSize * o.scaleY / baseScale));
  const minSize = Math.min(...sizes);
  objs.forEach(o => {
    o.set({ fontSize: minSize * baseScale, scaleX: 1, scaleY: 1 });
  });
  fc.requestRenderAll();
  refreshLayers();
});

btnMatchStyle.addEventListener("click", () => {
  const sel = fc.getActiveObject();
  if (!sel || sel.type !== "activeSelection") return;
  const objs = sel.getObjects().filter(o => o.type === "textbox" || o.type === "i-text");
  if (objs.length < 2) return;
  // Copy style from first selected object to all others
  const src = objs[0];
  const style = {
    fontWeight: src.fontWeight,
    fontStyle: src.fontStyle,
    underline: src.underline,
    fontFamily: src.fontFamily,
    fill: src.fill,
  };
  objs.forEach(o => o.set(style));
  fc.requestRenderAll();
  refreshLayers();
});

function active() {
  let o = fc ? fc.getActiveObject() : null;
  if (o && (o.type === "i-text" || o.type === "textbox")) return o;
  // Fall back to last editing textbox (button clicks may deselect canvas object)
  if (_lastEditObj && (_lastEditObj.type === "i-text" || _lastEditObj.type === "textbox")) return _lastEditObj;
  return null;
}

// ── Layer list ───────────────────────────────────────────────────────────
function refreshLayers() {
  layerListEl.innerHTML = "";
  if (!fc) return;
  const objs = fc.getObjects().filter(o => (o.type === "i-text" || o.type === "textbox"));
  for (let i = objs.length - 1; i >= 0; i--) {
    const o = objs[i];
    const li = document.createElement("li");
    li.innerHTML = `<span class="layer-icon">T</span> ${escHtml(o._name || o.text.substring(0,28) || "(empty)")}`;
    li.dataset.idx = i;
    if (fc.getActiveObject() === o) li.classList.add("active");
    li.addEventListener("click", () => {
      fc.setActiveObject(o); fc.renderAll();
      syncProps(o); noSel.classList.add("hidden"); propsDiv.classList.remove("hidden");
      highlightLayer(o);
    });
    layerListEl.appendChild(li);
  }
}

function highlightLayer(obj) {
  layerListEl.querySelectorAll("li").forEach(li => li.classList.remove("active"));
  if (!obj || !fc) return;
  const objs = fc.getObjects().filter(o => (o.type === "i-text" || o.type === "textbox"));
  const idx = objs.indexOf(obj);
  layerListEl.querySelectorAll("li").forEach(li => {
    if (parseInt(li.dataset.idx) === idx) li.classList.add("active");
  });
}

lUp.addEventListener("click", () => { const o = active(); if (!o) return; fc.bringForward(o); fc.renderAll(); refreshLayers(); });
lDown.addEventListener("click", () => { const o = active(); if (!o) return; fc.sendBackwards(o); fc.renderAll(); refreshLayers(); });
lDup.addEventListener("click", () => {
  const o = active(); if (!o) return;
  o.clone(function(c) { c.set({ left: c.left+15, top: c.top+15, _name: o._name+" copy" }); fc.add(c); fc.setActiveObject(c); fc.renderAll(); refreshLayers(); });
});
lDel.addEventListener("click", () => {
  const o = active(); if (!o) return;
  fc.remove(o); fc.discardActiveObject(); fc.renderAll(); onDesel(); refreshLayers();
});

// ── Toolbar ──────────────────────────────────────────────────────────────
btnBack.addEventListener("click", () => setPhase("review"));

btnTogglePreview.addEventListener("click", () => {
  edPreviewPanel.classList.toggle("ed-pane-hidden");
  btnTogglePreview.classList.toggle("tb-active");
});

btnDownloadClean.addEventListener("click", () => {
  if (!phaseData || !phaseData.clean) return;
  const dataUrl = phaseData.clean;
  const byteString = atob(dataUrl.split(",")[1]);
  const mimeType = dataUrl.split(",")[0].split(":")[1].split(";")[0];
  const ab = new ArrayBuffer(byteString.length);
  const ia = new Uint8Array(ab);
  for (let i = 0; i < byteString.length; i++) ia[i] = byteString.charCodeAt(i);
  const blob = new Blob([ab], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "cleaned.png";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
});

btnUploadBg.addEventListener("click", () => bgFileInput.click());
bgFileInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (ev) => {
    const dataUrl = ev.target.result;
    const img = new Image();
    img.onload = () => {
      if (img.width !== imgW || img.height !== imgH) {
        if (!confirm(`Uploaded image is ${img.width}x${img.height} but original is ${imgW}x${imgH}. Dimensions differ — this may cause misalignment. Proceed anyway?`)) return;
      }
      phaseData.clean = dataUrl;
      const dw = Math.round(imgW * baseScale);
      fabric.Image.fromURL(dataUrl, function(fImg) {
        fImg.scaleToWidth(dw);
        fc.setBackgroundImage(fImg, fc.renderAll.bind(fc));
      }, { crossOrigin: "anonymous" });
    };
    img.src = dataUrl;
  };
  reader.readAsDataURL(file);
  bgFileInput.value = "";
});

toolMove.addEventListener("click", () => setTool("move"));
toolText.addEventListener("click", () => setTool("text"));

function setTool(t) {
  currentTool = t;
  toolMove.classList.toggle("tb-active", t === "move");
  toolText.classList.toggle("tb-active", t === "text");
  if (t === "text") {
    const nt = new fabric.Textbox("New Text", {
      left: (fc.width / 2) - 80, top: (fc.height / 2) - 12,
      width: 160 * baseScale,
      fontSize: 24 * baseScale, fill: "#ffffff", fontFamily: "sans-serif",
      editable: true, _name: "New Text",
    });
    fc.add(nt); fc.setActiveObject(nt); fc.renderAll(); refreshLayers();
    syncProps(nt); noSel.classList.add("hidden"); propsDiv.classList.remove("hidden");
    setTimeout(() => setTool("move"), 100);
  }
}

// Zoom
btnZoomIn.addEventListener("click",  () => applyZoom(zoomLevel * 1.25));
btnZoomOut.addEventListener("click", () => applyZoom(zoomLevel / 1.25));
btnZoomFit.addEventListener("click", () => applyZoom(1));

function applyZoom(z) {
  z = Math.max(0.1, Math.min(z, 5)); zoomLevel = z;
  const dw = Math.round(imgW * baseScale * zoomLevel);
  const dh = Math.round(imgH * baseScale * zoomLevel);
  fc.setZoom(zoomLevel); fc.setWidth(dw); fc.setHeight(dh);
  canvasCont.style.width = dw + "px"; canvasCont.style.height = dh + "px";
  centerCanvas(dw, dh);
  fc.renderAll(); updateZoomLabel();
}

function centerCanvas(dw, dh) {
  // Use padding on scroll container to center canvas when smaller than viewport,
  // no padding when larger so scrolling reaches all edges
  const cw = dw + 80; // canvas + its 40px margin on each side
  const ch = dh + 80;
  const sw = scrollCont.clientWidth;
  const sh = scrollCont.clientHeight;
  const padX = Math.max(0, (sw - cw) / 2);
  const padY = Math.max(0, (sh - ch) / 2);
  scrollCont.style.paddingLeft = padX + "px";
  scrollCont.style.paddingTop = padY + "px";
}

function updateZoomLabel() {
  zoomLabel.textContent = Math.round(baseScale * zoomLevel * 100) + "%";
}

workspace.addEventListener("wheel", e => {
  if (!fc) return;
  // Ctrl+wheel = zoom, plain wheel = scroll the canvas
  if (e.ctrlKey || e.metaKey) {
    e.preventDefault();
    applyZoom(zoomLevel * (e.deltaY > 0 ? 0.9 : 1.1));
  }
}, { passive: false });

// Rebuild wordStyles from current Fabric Textbox state (honors user edits).
// Uses per-character overrides in o.styles; falls back to the original
// _wordStyles when the user hasn't touched styles for a word.
function _extractWordStylesFromFabric(o) {
  const text = o.text || "";
  if (!text) return o._wordStyles || null;
  const entries = [];
  const re = /\S+/g;
  let m;
  while ((m = re.exec(text)) !== null) {
    const word = m[0];
    const start = m.index;
    const end = start + word.length;
    // Textbox-level defaults (set by Bold/Italic buttons or color picker)
    const defWeight = o.fontWeight === "bold" ? "bold" : "normal";
    const defStyle = o.fontStyle === "italic" ? "italic" : "normal";
    const defColor = _normalizeHex(o.fill);
    // Gather per-char styles with fallback to object defaults
    const charStyles = [];
    for (let pos = start; pos < end; pos++) {
      let cs = {};
      try {
        cs = (o.getStyleAtPosition && o.getStyleAtPosition(pos, false)) || {};
      } catch (e) { cs = {}; }
      const weight = cs.fontWeight === "bold" ? "bold"
                     : cs.fontWeight === "normal" ? "normal" : defWeight;
      const style = cs.fontStyle === "italic" ? "italic"
                    : cs.fontStyle === "normal" ? "normal" : defStyle;
      const color = _normalizeHex(cs.fill) || defColor || null;
      charStyles.push({ fontWeight: weight, fontStyle: style, color });
    }
    // Group consecutive chars with identical style into runs
    let i = 0, first = true;
    while (i < charStyles.length) {
      const s = charStyles[i];
      let j = i + 1;
      while (j < charStyles.length
             && charStyles[j].fontWeight === s.fontWeight
             && charStyles[j].fontStyle === s.fontStyle
             && charStyles[j].color === s.color) j++;
      const run = word.slice(i, j);
      const style = { fontWeight: s.fontWeight, fontStyle: s.fontStyle };
      if (s.color) style.color = s.color;
      const entry = { word: run, style };
      if (!first) entry.gluePrev = true;  // part of same source-word
      entries.push(entry);
      first = false;
      i = j;
    }
  }
  return entries.length ? entries : (o._wordStyles || null);
}

function _normalizeHex(c) {
  if (!c) return null;
  if (typeof c === "string" && c.startsWith("#") && c.length === 7) return c;
  // rgb(r,g,b) or rgba(...)
  const m = typeof c === "string" ? c.match(/rgba?\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)/i) : null;
  if (m) {
    const h = (n) => parseInt(n, 10).toString(16).padStart(2, "0");
    return "#" + h(m[1]) + h(m[2]) + h(m[3]);
  }
  return null;
}

// ══════════════════════════════════════════════════════════════════════════
// Export → Phase 4 (server-side render)
// ══════════════════════════════════════════════════════════════════════════
btnExport.addEventListener("click", async () => {
  if (!fc || !phaseData) return;

  fc.discardActiveObject(); fc.renderAll();

  // Switch to export phase
  setPhase("export");
  p3Loading.classList.remove("hidden");
  p3Content.classList.add("hidden");

  // Build layer JSON from Fabric.js objects for server-side render
  const layers = [];
  fc.getObjects().forEach(o => {
    if (o.type !== "textbox" || o.excludeFromExport) return;
    const scX = o.scaleX || 1;
    const scY = o.scaleY || 1;
    const realFontSize = (o.fontSize || 16) * scX / baseScale;
    const realW = (o.width || 100) * scX / baseScale;
    const realH = (o.calcTextHeight ? o.calcTextHeight() : (o.height || 50)) * scY / baseScale;
    let realX, realY;
    if (o.originX === "center") {
      realX = o.left / baseScale - realW / 2;
      realY = o.top / baseScale - realH / 2;
    } else {
      realX = o.left / baseScale;
      realY = o.top / baseScale;
    }
    layers.push({
      text:       o.text,
      x:          Math.round(realX),
      y:          Math.round(realY),
      width:      Math.round(realW),
      height:     Math.round(realH),
      fontSize:   Math.round(realFontSize),
      color:      rgbToHex(o.fill),
      opacity:    o.opacity || 1,
      fontWeight: o.fontWeight || "normal",
      fontStyle:  o.fontStyle || "normal",
      fontFamily: o.fontFamily || "sans-serif",
      underline:  !!o.underline,
      alignment:  o.textAlign || "center",
      angle:      o.angle || 0,
      styleSpans: o._styleSpans || null,
      wordStyles: (function(){
        const ws = _extractWordStylesFromFabric(o);
        console.log("[ExportWS]", (o.text||"").substring(0,40),
          "styles:", o.styles, "out:", ws);
        return ws;
      })(),
    });
  });

  try {
    const fd = new FormData();
    fd.append("clean_image", phaseData.clean);
    fd.append("layers_json", JSON.stringify(layers));
    fd.append("target_lang", targetLang.value);

    const res = await fetch("/phase2-render", { method: "POST", body: fd });
    if (!res.ok) throw new Error("Render failed: " + res.status);
    const data = await res.json();

    p3Original.src = phaseData.original;
    p3Result.src = data.result;
  } catch (e) {
    // Fallback to client-side render
    console.error("Server render failed, using client-side:", e);
    const multiplier = 1 / (baseScale * zoomLevel);
    const dataUrl = fc.toDataURL({ format: "png", multiplier: multiplier });
    p3Original.src = phaseData.original;
    p3Result.src = dataUrl;
  }

  p3Loading.classList.add("hidden");
  p3Content.classList.remove("hidden");
});

// ══════════════════════════════════════════════════════════════════════════
// PHASE 4: Export
// ══════════════════════════════════════════════════════════════════════════
p3Back.addEventListener("click", () => setPhase("editor"));

p3Download.addEventListener("click", () => {
  const a = document.createElement("a");
  a.href = p3Result.src;
  a.download = "translated.png";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
});

p3DownClean.addEventListener("click", () => {
  if (!phaseData || !phaseData.clean) return;
  const a = document.createElement("a");
  a.href = phaseData.clean;
  a.download = "cleaned.png";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
});

// ══════════════════════════════════════════════════════════════════════════
// Helpers
// ══════════════════════════════════════════════════════════════════════════
function showErr(m) { errorMsg.textContent = m; errorMsg.classList.remove("hidden"); }
function hideErr()  { errorMsg.classList.add("hidden"); }
function escHtml(s) { const d = document.createElement("div"); d.textContent = s; return d.innerHTML; }

function rgbToHex(c) {
  if (!c) return "#000000";
  if (c.startsWith("#")) return c.length === 7 ? c : c + "000000".slice(c.length - 1);
  const m = c.match(/\d+/g);
  if (!m || m.length < 3) return "#000000";
  return "#" + m.slice(0,3).map(n => parseInt(n).toString(16).padStart(2,"0")).join("");
}
