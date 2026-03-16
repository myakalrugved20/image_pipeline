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
function openSelectionPhase() {
  setPhase("select");

  if (selFc) { selFc.dispose(); selFc = null; }

  requestAnimationFrame(() => {
    const wsW = selWorkspace.clientWidth - 80;
    const wsH = selWorkspace.clientHeight - 80;
    selBaseScale = Math.min(wsW / detectData.width, wsH / detectData.height, 1);
    if (selBaseScale <= 0) selBaseScale = 0.5;

    const dw = Math.round(detectData.width * selBaseScale);
    const dh = Math.round(detectData.height * selBaseScale);

    selCanvasCont.style.width = dw + "px";
    selCanvasCont.style.height = dh + "px";

    const canvasEl = document.getElementById("select-canvas");
    canvasEl.width = dw;
    canvasEl.height = dh;

    selFc = new fabric.Canvas("select-canvas", {
      width: dw, height: dh,
      selection: false,
      preserveObjectStacking: true,
    });

    // Background image
    fabric.Image.fromURL(detectData.original, function(img) {
      img.scaleToWidth(dw);
      selFc.setBackgroundImage(img, selFc.renderAll.bind(selFc));
    }, { crossOrigin: "anonymous" });

    // Add OCR-detected regions as selectable rectangles (pre-selected)
    detectData.regions.forEach((r, i) => {
      const rect = new fabric.Rect({
        left: r.x * selBaseScale,
        top: r.y * selBaseScale,
        width: r.width * selBaseScale,
        height: r.height * selBaseScale,
        fill: "rgba(137,180,250,0.25)",
        stroke: "#89b4fa",
        strokeWidth: 2,
        strokeDashArray: [6, 3],
        selectable: false,
        evented: true,
        hoverCursor: "pointer",
        _regionIdx: i,
        _selected: true,
        _isOcrRegion: true,
        _regionData: r,
      });
      selFc.add(rect);
    });

    selFc.renderAll();
    updateSelCount();

    // Click to toggle region selection
    selFc.on("mouse:down", function(opt) {
      if (drawMode) return;
      const target = opt.target;
      if (target && target._isOcrRegion !== undefined) {
        target._selected = !target._selected;
        if (target._selected) {
          target.set({ fill: "rgba(137,180,250,0.25)", stroke: "#89b4fa" });
        } else {
          target.set({ fill: "rgba(243,139,168,0.15)", stroke: "#f38ba8" });
        }
        selFc.renderAll();
        updateSelCount();
      }
    });

    // Draw mode: let user draw custom rectangles
    let drawStart = null;
    let drawRect = null;

    selFc.on("mouse:down", function(opt) {
      if (!drawMode) return;
      if (opt.target && (opt.target._isOcrRegion || opt.target._isCustomRegion)) return;
      const ptr = selFc.getPointer(opt.e);
      drawStart = { x: ptr.x, y: ptr.y };
      drawRect = new fabric.Rect({
        left: ptr.x, top: ptr.y, width: 0, height: 0,
        fill: "rgba(166,227,161,0.25)",
        stroke: "#a6e3a1",
        strokeWidth: 2,
        selectable: false,
        evented: true,
        hoverCursor: "pointer",
        _selected: true,
        _isCustomRegion: true,
      });
      selFc.add(drawRect);
    });

    selFc.on("mouse:move", function(opt) {
      if (!drawMode || !drawStart || !drawRect) return;
      const ptr = selFc.getPointer(opt.e);
      const x = Math.min(drawStart.x, ptr.x);
      const y = Math.min(drawStart.y, ptr.y);
      const w = Math.abs(ptr.x - drawStart.x);
      const h = Math.abs(ptr.y - drawStart.y);
      drawRect.set({ left: x, top: y, width: w, height: h });
      selFc.renderAll();
    });

    selFc.on("mouse:up", function() {
      if (!drawMode || !drawRect) return;
      // Remove if too small
      if (drawRect.width < 5 || drawRect.height < 5) {
        selFc.remove(drawRect);
      }
      drawStart = null;
      drawRect = null;
      updateSelCount();
    });
  });
}

function updateSelCount() {
  if (!selFc) return;
  const objs = selFc.getObjects();
  const selected = objs.filter(o => (o._isOcrRegion || o._isCustomRegion) && o._selected);
  selCount.textContent = `${selected.length} selected`;
}

selBack.addEventListener("click", () => {
  if (selFc) { selFc.dispose(); selFc = null; }
  setPhase("upload");
});

selSelectAll.addEventListener("click", () => {
  if (!selFc) return;
  selFc.getObjects().forEach(o => {
    if (o._isOcrRegion || o._isCustomRegion) {
      o._selected = true;
      o.set({ fill: o._isCustomRegion ? "rgba(166,227,161,0.25)" : "rgba(137,180,250,0.25)",
              stroke: o._isCustomRegion ? "#a6e3a1" : "#89b4fa" });
    }
  });
  selFc.renderAll();
  updateSelCount();
});

selDeselectAll.addEventListener("click", () => {
  if (!selFc) return;
  selFc.getObjects().forEach(o => {
    if (o._isOcrRegion || o._isCustomRegion) {
      o._selected = false;
      o.set({ fill: "rgba(243,139,168,0.15)", stroke: "#f38ba8" });
    }
  });
  selFc.renderAll();
  updateSelCount();
});

selDraw.addEventListener("click", () => {
  drawMode = !drawMode;
  selDraw.classList.toggle("tb-active", drawMode);
  if (selFc) {
    selFc.defaultCursor = drawMode ? "crosshair" : "default";
  }
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
      selectedRegions.push(o._regionData);
    } else if (o._isCustomRegion) {
      // Custom drawn box — no text data, just the area to inpaint
      selectedRegions.push({
        originalText: "",
        translatedText: "",
        x: Math.round(o.left / selBaseScale),
        y: Math.round(o.top / selBaseScale),
        width: Math.round(o.width / selBaseScale),
        height: Math.round(o.height / selBaseScale),
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

    // Text layers
    data.layers.forEach(l => {
      const t = new fabric.IText(l.translatedText, {
        left:       l.x * baseScale,
        top:        l.y * baseScale,
        fontSize:   l.fontSize * baseScale,
        fill:       l.color,
        fontFamily: "sans-serif",
        fontWeight: l.bold ? "bold" : "normal",
        fontStyle:  "normal",
        underline:  false,
        editable:   true,
        opacity:    1,
        _name:      l.translatedText.substring(0, 28),
        _boxW:      l.width,
        _boxH:      l.height,
        _alignment: l.alignment || "center",
      });
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
  if (!o || o.type !== "i-text") { onDesel(); return; }
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

function active() { const o = fc ? fc.getActiveObject() : null; return (o && o.type === "i-text") ? o : null; }

// ── Layer list ───────────────────────────────────────────────────────────
function refreshLayers() {
  layerListEl.innerHTML = "";
  if (!fc) return;
  const objs = fc.getObjects().filter(o => o.type === "i-text");
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
  const objs = fc.getObjects().filter(o => o.type === "i-text");
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

toolMove.addEventListener("click", () => setTool("move"));
toolText.addEventListener("click", () => setTool("text"));

function setTool(t) {
  currentTool = t;
  toolMove.classList.toggle("tb-active", t === "move");
  toolText.classList.toggle("tb-active", t === "text");
  if (t === "text") {
    const nt = new fabric.IText("New Text", {
      left: (fc.width / 2) - 40, top: (fc.height / 2) - 12,
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
btnExport.addEventListener("click", async () => {
  if (!fc || !phaseData) return;

  fc.discardActiveObject(); fc.renderAll();

  // Collect layer data from Fabric canvas → original image coordinates
  const layers = fc.getObjects().filter(o => o.type === "i-text").map(o => ({
    text:       o.text,
    x:          Math.round(o.left / baseScale),
    y:          Math.round(o.top / baseScale),
    width:      Math.round((o.width * o.scaleX) / baseScale),
    height:     Math.round((o.height * o.scaleY) / baseScale),
    fontSize:   Math.round(o.fontSize * o.scaleY / baseScale),
    color:      rgbToHex(o.fill),
    fontFamily: o.fontFamily || "sans-serif",
    fontWeight: o.fontWeight || "normal",
    fontStyle:  o.fontStyle || "normal",
    opacity:    o.opacity != null ? o.opacity : 1,
    alignment:  o._alignment || "center",
  }));

  // Switch to export phase with loading state
  setPhase("export");
  p3Loading.classList.remove("hidden");
  p3Content.classList.add("hidden");

  try {
    const fd = new FormData();
    fd.append("clean_image", phaseData.clean);
    fd.append("layers_json", JSON.stringify(layers));
    fd.append("target_lang", targetLang.value);

    const res = await fetch("/phase2-render", { method: "POST", body: fd });
    if (!res.ok) throw new Error("Server render failed");
    const data = await res.json();

    p3Original.src = phaseData.original;
    p3Result.src = data.result;
    p3Loading.classList.add("hidden");
    p3Content.classList.remove("hidden");
  } catch (e) {
    showErr(e.message);
    setPhase("editor");
  }
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
