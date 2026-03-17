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
const selDraw      = document.getElementById("sel-draw");
const selDelete    = document.getElementById("sel-delete");
const selClean     = document.getElementById("sel-clean");
const selCount     = document.getElementById("sel-count");
const selWorkspace = document.getElementById("select-workspace");
const selCanvasCont = document.getElementById("select-canvas-container");
const selLoading   = document.getElementById("sel-loading");

// DOM — Review (Phase 2)
const phase1View   = document.getElementById("phase1-view");
const p1Original   = document.getElementById("p1-original");
const p1Clean      = document.getElementById("p1-clean");
const p1Back       = document.getElementById("p1-back");
const p1Next       = document.getElementById("p1-next");
const p1Info       = document.getElementById("p1-info");

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
const lUp          = document.getElementById("l-up");
const lDown        = document.getElementById("l-down");
const lDup         = document.getElementById("l-dup");
const lDel         = document.getElementById("l-del");
const statusSize   = document.getElementById("status-size");
const statusInfo   = document.getElementById("status-info");
const workspace    = document.getElementById("ed-workspace");
const canvasCont   = document.getElementById("ed-canvas-container");
const btnTogglePreview = document.getElementById("btn-toggle-preview");
const edPreviewPanel   = document.getElementById("ed-preview-panel");
const edPreviewImg     = document.getElementById("ed-preview-img");

// DOM — Export (Phase 4)
const phase3View   = document.getElementById("phase3-view");
const p3Loading    = document.getElementById("p3-loading");
const p3Content    = document.getElementById("p3-content");
const p3Original   = document.getElementById("p3-original");
const p3Result     = document.getElementById("p3-result");
const p3Back       = document.getElementById("p3-back");
const p3Download   = document.getElementById("p3-download");

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

/* Color a box based on its _selected flag */
function styleBox(r) {
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

/* Delete whichever box is Fabric-active */
function deleteActiveBox() {
  if (!selFc) return;
  var a = selFc.getActiveObject();
  if (a && (a._isOcrRegion || a._isCustomRegion)) {
    selFc.remove(a);
    selFc.discardActiveObject();
    selFc.renderAll();
    updateSelCount();
  }
}

function openSelectionPhase() {
  setPhase("select");
  if (selFc) { selFc.dispose(); selFc = null; }
  drawMode = false;
  selDraw.classList.remove("tb-active");

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

    selFc.on("mouse:down", function() {
      _didDrag = false;
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

    selFc.on("mouse:down", function(opt) {
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

    /* keyboard delete */
    document.addEventListener("keydown", _selKeyHandler);
  });
}

/* keyboard handler – lives outside so we can remove it */
function _selKeyHandler(e) {
  if (appPhase !== "select" || !selFc) return;
  if (e.key === "Delete" || e.key === "Backspace") {
    var a = selFc.getActiveObject();
    if (a && (a._isOcrRegion || a._isCustomRegion)) {
      e.preventDefault();
      deleteActiveBox();
    }
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

if (selDelete) selDelete.addEventListener("click", deleteActiveBox);

selDraw.addEventListener("click", function() {
  drawMode = !drawMode;
  selDraw.classList.toggle("tb-active", drawMode);
  if (!selFc) return;
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
    const fd = new FormData();
    fd.append("original_image", detectData.original);
    fd.append("selected_regions", JSON.stringify(selectedRegions));
    fd.append("target_lang", targetLang.value);

    const res = await fetch("/phase1-clean", { method: "POST", body: fd });
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
function openReviewPhase() {
  p1Original.src = phaseData.original;
  p1Clean.src = phaseData.clean;
  p1Info.textContent = `${phaseData.regions.length} text region(s) to render`;
  setPhase("review");
}

p1Back.addEventListener("click", () => setPhase("select"));
p1Next.addEventListener("click", () => {
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

      let boxW = l.width * baseScale;
      let boxH = l.height * baseScale;

      // For vertical text (~90°), swap width/height since text flows along the height
      const absAng = Math.abs(ang);
      if (absAng > 60 && absAng < 120) {
        const tmp = boxW;
        boxW = boxH;
        boxH = tmp;
      }

      const opts = {
        width:      boxW,
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

      const t = new fabric.Textbox(l.translatedText, opts);

      // Auto-fit: shrink font until text fits within bounding box height
      let attempts = 0;
      while (t.height > boxH && t.fontSize > 4 && attempts < 50) {
        t.set("fontSize", t.fontSize - 1);
        t.initDimensions();
        attempts++;
      }

      // Vertically center within the bounding box
      if (t.height < boxH && Math.abs(ang) < 2) {
        const vOffset = (boxH - t.height) / 2;
        t.set("top", t.top + vOffset);
      }

      fc.add(t);
    });

    fc.renderAll();
    updateZoomLabel();
    refreshLayers();

    fc.off();
    fc.on("selection:created",  onSel);
    fc.on("selection:updated",  onSel);
    fc.on("selection:cleared",  onDesel);
    fc.on("object:modified",    () => { const o = fc.getActiveObject(); if (o) syncProps(o); });
    fc.on("object:moving",      () => { const o = fc.getActiveObject(); if (o) syncPos(o); });
    fc.on("text:changed",       e => {
      if (e.target) { e.target._name = e.target.text.substring(0,28); refreshLayers(); pText.value = e.target.text; }
    });

    statusInfo.textContent = `${data.layers.length} text layers loaded`;
  });
}

// ── Selection ────────────────────────────────────────────────────────────
function onSel() {
  const o = fc.getActiveObject();
  if (!o || o.type !== "i-text" && o.type !== "textbox") { onDesel(); return; }
  syncProps(o); noSel.classList.add("hidden"); propsDiv.classList.remove("hidden");
  highlightLayer(o);
}

function onDesel() {
  noSel.classList.remove("hidden"); propsDiv.classList.add("hidden");
  highlightLayer(null);
}

function syncProps(o) {
  pText.value = o.text;
  const realSize = Math.round(o.fontSize * o.scaleY / baseScale);
  pSize.value = realSize; pSizeVal.textContent = realSize;
  pColor.value = rgbToHex(o.fill);
  pFont.value = o.fontFamily;
  pBold.classList.toggle("active", o.fontWeight === "bold");
  pItalic.classList.toggle("active", o.fontStyle === "italic");
  pUnderline.classList.toggle("active", !!o.underline);
  pOpacity.value = Math.round(o.opacity * 100);
  pOpacityVal.textContent = Math.round(o.opacity * 100);
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

pSize.addEventListener("input", () => {
  const o = active(); if (!o) return;
  const s = parseInt(pSize.value, 10); pSizeVal.textContent = s;
  o.set({ fontSize: s * baseScale, scaleX: 1, scaleY: 1 }); fc.renderAll();
});

pColor.addEventListener("input", () => { const o = active(); if (!o) return; o.set("fill", pColor.value); fc.renderAll(); });
pFont.addEventListener("change", () => { const o = active(); if (!o) return; o.set("fontFamily", pFont.value); fc.renderAll(); });

pBold.addEventListener("click", () => {
  const o = active(); if (!o) return;
  const v = o.fontWeight === "bold" ? "normal" : "bold";
  o.set("fontWeight", v); pBold.classList.toggle("active", v === "bold"); fc.renderAll();
});
pItalic.addEventListener("click", () => {
  const o = active(); if (!o) return;
  const v = o.fontStyle === "italic" ? "normal" : "italic";
  o.set("fontStyle", v); pItalic.classList.toggle("active", v === "italic"); fc.renderAll();
});
pUnderline.addEventListener("click", () => {
  const o = active(); if (!o) return;
  o.set("underline", !o.underline); pUnderline.classList.toggle("active", o.underline); fc.renderAll();
});
pOpacity.addEventListener("input", () => {
  const o = active(); if (!o) return;
  o.set("opacity", parseInt(pOpacity.value, 10) / 100);
  pOpacityVal.textContent = pOpacity.value; fc.renderAll();
});
pX.addEventListener("change", () => { const o = active(); if (!o) return; o.set("left", parseInt(pX.value, 10) * baseScale); fc.renderAll(); });
pY.addEventListener("change", () => { const o = active(); if (!o) return; o.set("top", parseInt(pY.value, 10) * baseScale); fc.renderAll(); });

function active() { const o = fc ? fc.getActiveObject() : null; return (o && (o.type === "i-text" || o.type === "textbox")) ? o : null; }

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
  edPreviewPanel.classList.toggle("hidden");
  btnTogglePreview.classList.toggle("tb-active");
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
  fc.renderAll(); updateZoomLabel();
}

function updateZoomLabel() {
  zoomLabel.textContent = Math.round(baseScale * zoomLevel * 100) + "%";
}

workspace.addEventListener("wheel", e => {
  if (!fc) return; e.preventDefault();
  applyZoom(zoomLevel * (e.deltaY > 0 ? 0.9 : 1.1));
}, { passive: false });

// ══════════════════════════════════════════════════════════════════════════
// Export → Phase 4 (server-side render)
// ══════════════════════════════════════════════════════════════════════════
btnExport.addEventListener("click", () => {
  if (!fc || !phaseData) return;

  fc.discardActiveObject(); fc.renderAll();

  // Switch to export phase
  setPhase("export");
  p3Loading.classList.remove("hidden");
  p3Content.classList.add("hidden");

  // Render canvas at full original resolution (client-side)
  // This guarantees the output matches the editor exactly
  const multiplier = 1 / (baseScale * zoomLevel);
  const dataUrl = fc.toDataURL({ format: "png", multiplier: multiplier });

  p3Original.src = phaseData.original;
  p3Result.src = dataUrl;
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
