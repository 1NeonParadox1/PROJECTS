/* ============================================================
   Resonance frontend logic
   Talks to the FastAPI backend at /api/*. No build step, no
   framework — plain fetch + DOM, intentionally simple so the
   whole stack is easy to read end-to-end for a demo/interview.
   ============================================================ */

const API = "/api";

const els = {
  dropzone: document.getElementById("dropzone"),
  fileInput: document.getElementById("fileInput"),
  jobTracker: document.getElementById("jobTracker"),
  jobFilename: document.getElementById("jobFilename"),
  jobPercent: document.getElementById("jobPercent"),
  jobBarFill: document.getElementById("jobBarFill"),
  jobMessage: document.getElementById("jobMessage"),
  libraryList: document.getElementById("libraryList"),
  searchForm: document.getElementById("searchForm"),
  searchInput: document.getElementById("searchInput"),
  scopeRow: document.getElementById("scopeRow"),
  results: document.getElementById("results"),
  mediaPlayer: document.getElementById("mediaPlayer"),
  playerEmpty: document.getElementById("playerEmpty"),
  playerMeta: document.getElementById("playerMeta"),
  playerFilename: document.getElementById("playerFilename"),
  playerTimecode: document.getElementById("playerTimecode"),
  transcriptStrip: document.getElementById("transcriptStrip"),
};

let scopeFileId = null; // null = search whole library
let library = [];

// ---------- helpers ----------
function fmtTime(s) {
  s = Math.max(0, Math.floor(s));
  const m = Math.floor(s / 60).toString().padStart(2, "0");
  const sec = (s % 60).toString().padStart(2, "0");
  return `${m}:${sec}`;
}

async function api(path, opts = {}) {
  const res = await fetch(API + path, opts);
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status}: ${body}`);
  }
  return res.json();
}

// ---------- upload ----------
els.dropzone.addEventListener("dragover", (e) => { e.preventDefault(); els.dropzone.style.borderColor = "var(--signal)"; });
els.dropzone.addEventListener("dragleave", () => { els.dropzone.style.borderColor = ""; });
els.dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  els.dropzone.style.borderColor = "";
  if (e.dataTransfer.files.length) uploadFile(e.dataTransfer.files[0]);
});
els.fileInput.addEventListener("change", () => {
  if (els.fileInput.files.length) uploadFile(els.fileInput.files[0]);
});

async function uploadFile(file) {
  const form = new FormData();
  form.append("file", file);

  els.jobTracker.hidden = false;
  els.jobFilename.textContent = file.name;
  els.jobPercent.textContent = "0%";
  els.jobBarFill.style.width = "0%";
  els.jobMessage.textContent = "Uploading…";

  let fileId;
  try {
    const res = await api("/ingest", { method: "POST", body: form });
    fileId = res.file_id;
  } catch (err) {
    els.jobMessage.textContent = "Upload failed: " + err.message;
    return;
  }

  pollStatus(fileId);
}

function pollStatus(fileId) {
  const interval = setInterval(async () => {
    try {
      const job = await api(`/status/${fileId}`);
      const pct = Math.round(job.progress * 100);
      els.jobPercent.textContent = pct + "%";
      els.jobBarFill.style.width = pct + "%";
      els.jobMessage.textContent = job.message || job.status;

      if (job.status === "done") {
        clearInterval(interval);
        els.jobMessage.textContent = "✓ " + (job.message || "Indexed");
        refreshLibrary();
        setTimeout(() => { els.jobTracker.hidden = true; }, 2500);
      } else if (job.status === "error") {
        clearInterval(interval);
        els.jobMessage.textContent = "✗ " + (job.message || "Failed");
      }
    } catch (err) {
      clearInterval(interval);
      els.jobMessage.textContent = "Lost connection to job: " + err.message;
    }
  }, 1200);
}

// ---------- library ----------
async function refreshLibrary() {
  try {
    const res = await api("/library");
    library = res.items;
    renderLibrary();
    renderScopeChips();
  } catch (err) {
    console.error(err);
  }
}

function renderLibrary() {
  if (!library.length) {
    els.libraryList.innerHTML = `<div class="empty-hint">No media indexed yet. Upload something above.</div>`;
    return;
  }
  els.libraryList.innerHTML = "";
  library.forEach((item) => {
    const div = document.createElement("div");
    div.className = "lib-item" + (scopeFileId === item.file_id ? " active" : "");
    div.tabIndex = 0;
    div.innerHTML = `
      <div class="lib-name">${escapeHtml(item.filename)}</div>
      <div class="lib-meta">
        <span>${item.num_chunks} chunks · ${fmtTime(item.duration_seconds)}</span>
        <button class="lib-del" title="Remove">✕</button>
      </div>
    `;
    div.addEventListener("click", (e) => {
      if (e.target.classList.contains("lib-del")) return;
      loadIntoPlayer(item, 0);
    });
    div.querySelector(".lib-del").addEventListener("click", async (e) => {
      e.stopPropagation();
      await api(`/files/${item.file_id}`, { method: "DELETE" });
      if (scopeFileId === item.file_id) scopeFileId = null;
      refreshLibrary();
    });
    els.libraryList.appendChild(div);
  });
}

function renderScopeChips() {
  els.scopeRow.innerHTML = `<span class="scope-label">scope:</span>`;
  const allChip = document.createElement("button");
  allChip.className = "scope-chip" + (scopeFileId === null ? " active" : "");
  allChip.textContent = "entire library";
  allChip.dataset.scope = "all";
  allChip.addEventListener("click", () => { scopeFileId = null; renderLibrary(); renderScopeChips(); });
  els.scopeRow.appendChild(allChip);

  library.forEach((item) => {
    const chip = document.createElement("button");
    chip.className = "scope-chip" + (scopeFileId === item.file_id ? " active" : "");
    chip.textContent = truncate(item.filename, 22);
    chip.addEventListener("click", () => { scopeFileId = item.file_id; renderLibrary(); renderScopeChips(); });
    els.scopeRow.appendChild(chip);
  });
}

// ---------- search ----------
els.searchForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = els.searchInput.value.trim();
  if (!q) return;

  els.results.innerHTML = `<div class="empty-hint big"><div class="empty-icon">∿∿∿</div>Searching by meaning…</div>`;

  try {
    const params = new URLSearchParams({ q });
    if (scopeFileId) params.append("file_id", scopeFileId);
    const res = await api(`/search?${params.toString()}`);
    renderResults(res.results);
  } catch (err) {
    els.results.innerHTML = `<div class="empty-hint big">Search failed: ${escapeHtml(err.message)}</div>`;
  }
});

function renderResults(hits) {
  if (!hits.length) {
    els.results.innerHTML = `<div class="empty-hint big"><div class="empty-icon">∅</div>No matching moments found. Try rephrasing, or upload more media.</div>`;
    return;
  }
  els.results.innerHTML = "";
  hits.forEach((hit) => {
    const pct = Math.round(hit.score * 100);
    const card = document.createElement("div");
    card.className = "result-card";
    card.tabIndex = 0;
    card.innerHTML = `
      <div class="result-score">
        <div class="score-ring">${pct}</div>
      </div>
      <div class="result-body">
        <div class="result-top">
          <span class="result-file">${escapeHtml(truncate(hit.filename, 30))}</span>
          <span class="result-time">${fmtTime(hit.start_time)} → ${fmtTime(hit.end_time)}</span>
        </div>
        <div class="result-text">${escapeHtml(hit.text)}</div>
      </div>
    `;
    card.addEventListener("click", () => {
      const item = library.find((l) => l.file_id === hit.file_id);
      if (item) loadIntoPlayer(item, hit.start_time, hit.text, hit.start_time, hit.end_time);
    });
    els.results.appendChild(card);
  });
}

// ---------- player ----------
function loadIntoPlayer(item, seekTo = 0, transcriptText = null, start = null, end = null) {
  const needsReload = els.mediaPlayer.dataset.fileId !== item.file_id;
  if (needsReload) {
    els.mediaPlayer.src = item.media_url;
    els.mediaPlayer.dataset.fileId = item.file_id;
  }
  els.playerEmpty.style.display = "none";
  els.mediaPlayer.style.display = "block";
  els.playerMeta.hidden = false;
  els.playerFilename.textContent = item.filename;
  els.playerTimecode.textContent = `${fmtTime(seekTo)} → ${fmtTime(item.duration_seconds)}`;

  const seek = () => { els.mediaPlayer.currentTime = seekTo; els.mediaPlayer.play().catch(() => {}); };
  if (needsReload) {
    els.mediaPlayer.addEventListener("loadedmetadata", seek, { once: true });
  } else {
    seek();
  }

  if (transcriptText) {
    els.transcriptStrip.innerHTML = `<div class="transcript-line active">${escapeHtml(transcriptText)}</div>`;
  }

  scopeFileId = item.file_id;
  renderLibrary();
  renderScopeChips();
}

// ---------- utils ----------
function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}
function truncate(str, n) {
  return str.length > n ? str.slice(0, n - 1) + "…" : str;
}

// ---------- init ----------
refreshLibrary();
