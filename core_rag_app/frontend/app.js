let projects = [];
let activeProjectId = null;
let activeProject = null;
let activeDocs = [];
let activeProgress = null;
let chatHistory = [];
let runtimeSettings = null;
let stagedUploadFiles = [];
let progressSocket = null;
let progressSocketProjectId = null;
let progressReconnectTimer = null;
let lastChatResult = null;

const el = (selector) => document.querySelector(selector);
const routes = {
  overview: ["Overview", "Command center for your evidence graph."],
  projects: ["Projects", "Choose or create an isolated corpus workspace."],
  documents: ["Documents", "Upload, reconcile, and build your evidence mesh."],
  graph: ["Graph Explorer", "Navigate domains, clusters, cards, and relationships."],
  chat: ["Evidence Chat", "Ask cited questions over the graph."],
  evaluations: ["Evaluations", "Regression tests for retrieval quality."],
  settings: ["Settings", "Storage and runtime configuration."],
};

const icons = {
  home: '<svg viewBox="0 0 24 24"><path d="M4 11 12 4l8 7v9H5v-7h14"/></svg>',
  folder: '<svg viewBox="0 0 24 24"><path d="M3 7h7l2 2h9v10H3V7Z"/></svg>',
  doc: '<svg viewBox="0 0 24 24"><path d="M7 3h7l4 4v14H7V3Zm7 0v5h5M10 13h6M10 17h5"/></svg>',
  graph: '<svg viewBox="0 0 24 24"><path d="M7 7h.01M17 7h.01M12 17h.01M8 8l4 8m4-8-4 8M8 7h8"/></svg>',
  chat: '<svg viewBox="0 0 24 24"><path d="M5 5h14v10H8l-3 4V5Z"/></svg>',
  eval: '<svg viewBox="0 0 24 24"><path d="M5 19V5m0 14h14M9 15l3-4 3 2 4-6"/></svg>',
  settings: '<svg viewBox="0 0 24 24"><path d="M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8Zm0-5v3m0 12v3M4.2 4.2l2.1 2.1m11.4 11.4 2.1 2.1M3 12h3m12 0h3M4.2 19.8l2.1-2.1M17.7 6.3l2.1-2.1"/></svg>',
  refresh: '<svg viewBox="0 0 24 24"><path d="M20 12a8 8 0 1 1-2.3-5.7M20 4v6h-6"/></svg>',
};

function hydrateIcons() {
  document.querySelectorAll("[data-icon]").forEach((node) => {
    node.innerHTML = icons[node.dataset.icon] || "";
  });
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || response.statusText);
  }
  return response.json();
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function formatNumber(value) {
  return new Intl.NumberFormat().format(Number(value || 0));
}

function formatBytes(value) {
  const bytes = Number(value || 0);
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / Math.pow(1024, index)).toFixed(index ? 1 : 0)} ${units[index]}`;
}

function uploadFileKey(file) {
  return [file.name, file.size, file.lastModified].join("::");
}

function currentRoute() {
  const route = location.hash.replace(/^#\/?/, "") || "overview";
  return routes[route] ? route : "overview";
}

function setRoute(route) {
  document.querySelectorAll(".page").forEach((page) => page.classList.remove("active-page"));
  const page = el(`#route-${route}`);
  if (page) page.classList.add("active-page");
  document.querySelectorAll("#nav a").forEach((link) => link.classList.toggle("active", link.dataset.route === route));
  const [title, subtitle] = routes[route];
  el("#page-title").textContent = title;
  el("#page-subtitle").textContent = subtitle;
}

function stageText(stage) {
  const labels = {
    idle: "Idle",
    uploaded: "Ready",
    uploading: "Uploading",
    ingesting: "Ingesting",
    indexing: "Indexing",
    knowledge_graph: "Building graph",
    complete: "Complete",
    failed: "Failed",
  };
  return labels[stage] || stage || "Idle";
}

function setBusy(isBusy) {
  el("#build-index").disabled = isBusy;
  el("#build-index-overview").disabled = isBusy;
  el("#refresh").disabled = isBusy;
  el("#engine-status").textContent = isBusy ? "Working" : "Ready";
}

function progressSocketUrl(projectId) {
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${location.host}/ws/projects/${encodeURIComponent(projectId)}/pipeline-progress`;
}

function isTerminalProgress(stage) {
  return ["idle", "uploaded", "complete", "failed"].includes(stage || "");
}

function stopProgressSocket() {
  if (progressReconnectTimer) {
    window.clearTimeout(progressReconnectTimer);
    progressReconnectTimer = null;
  }
  if (progressSocket) {
    progressSocket.onclose = null;
    progressSocket.close();
    progressSocket = null;
  }
  progressSocketProjectId = null;
}

function applyProgressUpdate(progress) {
  if (!progress || progress.project_id !== activeProjectId) return;
  activeProgress = progress;
  if (activeProject) {
    activeProject = {
      ...activeProject,
      document_count: progress.document_count ?? activeProject.document_count,
      card_count: progress.card_count ?? activeProject.card_count,
      cluster_count: progress.cluster_count ?? activeProject.cluster_count,
      domain_count: progress.domain_count ?? activeProject.domain_count,
      relationship_count: progress.relationship_count ?? activeProject.relationship_count,
    };
  }
  renderAll();
}

function connectProgressSocket(projectId) {
  if (!projectId) return;
  if (progressSocket && progressSocketProjectId === projectId && progressSocket.readyState <= WebSocket.OPEN) return;
  stopProgressSocket();
  progressSocketProjectId = projectId;
  progressSocket = new WebSocket(progressSocketUrl(projectId));

  progressSocket.onmessage = (event) => {
    try {
      applyProgressUpdate(JSON.parse(event.data));
    } catch (_error) {
      // Ignore malformed socket frames; the HTTP refresh path remains available.
    }
  };

  progressSocket.onclose = () => {
    progressSocket = null;
    if (activeProjectId !== projectId || isTerminalProgress(activeProgress?.stage)) return;
    progressReconnectTimer = window.setTimeout(() => connectProgressSocket(projectId), 2000);
  };

  progressSocket.onerror = () => {
    if (progressSocket) progressSocket.close();
  };
}

async function runWithProgress(work) {
  setBusy(true);
  if (activeProjectId) connectProgressSocket(activeProjectId);
  try {
    return await work();
  } finally {
    setBusy(false);
  }
}

async function loadProjects() {
  projects = await api("/api/projects");
  renderProjectSelect();
  renderProjectGrid();
  if (!activeProjectId && projects[0]) {
    await selectProject(projects[0].project_id);
  } else if (!projects.length) {
    el("#empty-state").classList.add("active-page");
  }
}

async function loadRuntimeSettings() {
  runtimeSettings = await api("/api/runtime-settings");
  renderRuntimeSettings();
}

function renderProjectSelect() {
  el("#project-select").innerHTML = projects.length
    ? projects.map((project) => `<option value="${escapeHtml(project.project_id)}">${escapeHtml(project.name)}</option>`).join("")
    : `<option value="">No projects</option>`;
  if (activeProjectId) el("#project-select").value = activeProjectId;
}

function renderProjectGrid() {
  el("#project-grid").innerHTML = projects.length ? projects.map((project) => `
    <article class="project-tile ${project.project_id === activeProjectId ? "active" : ""}" data-project="${escapeHtml(project.project_id)}">
      <div>
        <h3>${escapeHtml(project.name)}</h3>
        <p>${escapeHtml(project.project_id)}</p>
      </div>
      <div class="tile-meta">
        <span>${formatNumber(project.document_count)} docs</span>
        <span>${formatNumber(project.card_count)} cards</span>
        <span>${formatNumber(project.relationship_count)} links</span>
      </div>
    </article>
  `).join("") : `<div class="panel"><h3>No projects yet</h3><p>Create one from the top bar.</p></div>`;
}

async function selectProject(projectId) {
  if (!projectId) return;
  activeProjectId = projectId;
  chatHistory = [];
  lastChatResult = null;
  el("#empty-state").classList.remove("active-page");
  el("#project-view").classList.remove("hidden");
  renderProjectSelect();
  resetChat();
  await refreshActiveProject();
  connectProgressSocket(projectId);
}

async function refreshActiveProject() {
  if (!activeProjectId) return;
  const [project, docs, progress] = await Promise.all([
    api(`/api/projects/${activeProjectId}`),
    api(`/api/projects/${activeProjectId}/documents`),
    api(`/api/projects/${activeProjectId}/pipeline-progress`),
  ]);
  activeProject = project;
  activeDocs = docs;
  activeProgress = progress;
  connectProgressSocket(activeProjectId);
  renderAll();
}

function renderAll() {
  renderProjectSelect();
  renderProjectGrid();
  renderOverview();
  renderDocuments();
  renderGraph();
  renderChatSidebars();
  renderProgress();
}

function renderRuntimeSettings() {
  if (!runtimeSettings) return;
  el("#llm-provider").value = runtimeSettings.provider || "openrouter";
  el("#llm-model").value = runtimeSettings.model || "";
  el("#llm-map-model").value = runtimeSettings.map_model || "";
  el("#llm-search-model").value = runtimeSettings.search_model || "";
  el("#openrouter-model").value = runtimeSettings.openrouter_model || "";
  el("#openrouter-map-model").value = runtimeSettings.openrouter_map_model || "";
  el("#openrouter-search-model").value = runtimeSettings.openrouter_search_model || "";
  el("#openai-model").value = runtimeSettings.openai_model || "";
  el("#openai-map-model").value = runtimeSettings.openai_map_model || "";
  el("#openai-search-model").value = runtimeSettings.openai_search_model || "";
  el("#llm-api-key").value = "";
  el("#llm-key-state").textContent = runtimeSettings.api_key_present
    ? `Saved key ${runtimeSettings.api_key_hint || ""}`
    : "No key saved";
}

function providerModelFields(provider) {
  if (provider === "openai") {
    return {
      model: el("#openai-model").value.trim(),
      map_model: el("#openai-map-model").value.trim(),
      search_model: el("#openai-search-model").value.trim(),
    };
  }
  return {
    model: el("#openrouter-model").value.trim(),
    map_model: el("#openrouter-map-model").value.trim(),
    search_model: el("#openrouter-search-model").value.trim(),
  };
}

function renderOverview() {
  const progress = activeProgress || {};
  el("#overview-copy").textContent = progress.progress_label || progress.message || "Your knowledge graph is ready when your corpus is.";
  el("#scene-project").textContent = activeProject?.name || "Corpus";
  el("#scene-cards").textContent = `${formatNumber(activeProject?.card_count)} cards`;
  el("#metric-docs").textContent = formatNumber(activeProject?.document_count);
  el("#metric-pages").textContent = formatNumber(progress.indexed_pages);
  el("#metric-page-total").textContent = `${formatNumber(progress.total_pages)} total`;
  el("#metric-cards").textContent = formatNumber(activeProject?.card_count);
  el("#metric-relationships").textContent = formatNumber(activeProject?.relationship_count);
  el("#document-count-pill").textContent = `${formatNumber(activeDocs.length)} files`;

  el("#activity-list").innerHTML = [
    ["Pipeline", progress.progress_label || progress.message || "Not started"],
    ["Current document", progress.current_document || "None"],
    ["Graph audit", progress.graph_audit_status || "Not run"],
  ].map(([title, detail]) => `<div class="activity-item"><strong>${escapeHtml(title)}</strong><span>${escapeHtml(detail)}</span></div>`).join("");

  el("#health-list").innerHTML = [
    ["Documents", `${formatNumber(activeDocs.length)} available`],
    ["Failed pages", `${formatNumber(progress.failed_pages)} failed`],
    ["Audit issues", `${formatNumber(progress.graph_audit_issue_count)} issues`],
  ].map(([title, detail]) => `<div class="health-item"><strong>${escapeHtml(title)}</strong><span>${escapeHtml(detail)}</span></div>`).join("");
}

function renderDocuments() {
  const progress = activeProgress || {};
  el("#queue-count").textContent = `${formatNumber(activeDocs.length)} files`;
  el("#upload-stage").textContent = stageText(progress.stage);
  renderStagedUploads();
  el("#documents-list").innerHTML = activeDocs.length ? activeDocs.map((doc) => `
    <div class="document-row">
      <div>
        <strong title="${escapeHtml(doc.original_name)}">${escapeHtml(doc.original_name)}</strong>
        <span>${escapeHtml(doc.extension || "file")} - ${formatBytes(doc.size_bytes)} - ${escapeHtml(doc.ingest_strategy || "queued")}</span>
      </div>
      <button type="button" data-delete-doc="${escapeHtml(doc.document_id)}">Remove</button>
    </div>
  `).join("") : `<div class="document-row"><div><strong>No documents yet</strong><span>Upload source files to begin.</span></div></div>`;
}

function renderStagedUploads() {
  const list = el("#staged-upload-list");
  const count = stagedUploadFiles.length;
  el("#staged-upload-count").textContent = count ? `${formatNumber(count)} selected` : "No files selected";
  el("#upload-submit").disabled = !count || !activeProjectId;
  el("#clear-staged-files").disabled = !count;
  list.innerHTML = count ? stagedUploadFiles.map((file, index) => `
    <div class="staged-file">
      <div>
        <strong title="${escapeHtml(file.name)}">${escapeHtml(file.name)}</strong>
        <span>${formatBytes(file.size)}</span>
      </div>
      <button type="button" data-remove-staged="${index}" title="Remove ${escapeHtml(file.name)}">Remove</button>
    </div>
  `).join("") : `<div class="staged-empty">Choose files in multiple rounds. They will stay here until you upload or remove them.</div>`;
}

function renderGraph() {
  const progress = activeProgress || {};
  el("#metric-domains").textContent = formatNumber(activeProject?.domain_count);
  el("#metric-clusters").textContent = formatNumber(activeProject?.cluster_count);
  el("#graph-card-count").textContent = formatNumber(activeProject?.card_count);
  el("#graph-relation-count").textContent = formatNumber(activeProject?.relationship_count);
  el("#graph-root-name").textContent = activeProject?.name || "Corpus";
  el("#graph-root-sub").textContent = `${formatNumber(activeProject?.card_count)} cards`;
  el("#graph-inspector").innerHTML = [
    ["Status", stageText(progress.stage)],
    ["Domains", formatNumber(activeProject?.domain_count)],
    ["Clusters", formatNumber(activeProject?.cluster_count)],
    ["Relationships", formatNumber(activeProject?.relationship_count)],
    ["Audit", progress.graph_audit_status || "Not run"],
  ].map(([title, value]) => `<div class="settings-list"><div><strong>${escapeHtml(title)}</strong><span>${escapeHtml(value)}</span></div></div>`).join("");
}

function renderChatSidebars() {
  const trace = lastChatResult?.agent_trace || [];
  const hits = lastChatResult?.search?.hits || [];
  el("#evidence-timeline").innerHTML = trace.length ? trace.slice(-8).map((step) => `
    <div class="timeline-item">
      <strong>${escapeHtml(step.event || "agent_step")}</strong>
      <span>${escapeHtml(JSON.stringify(step.detail || {}).slice(0, 180))}</span>
    </div>
  `).join("") : [
    ["Query planned", "The planner turns follow-ups into standalone evidence searches."],
    ["Graph traversed", "Domains, clusters, related cards, and evidence links are explored when available."],
    ["Answer grounded", "The answer is composed from retrieved cards and citations."],
  ].map(([title, body]) => `<div class="timeline-item"><strong>${escapeHtml(title)}</strong><span>${escapeHtml(body)}</span></div>`).join("");
  el("#top-evidence").innerHTML = hits.length ? hits.slice(0, 6).map((hit) => `
    <div class="source-card">
      <strong>${escapeHtml(hit.card_name || "Evidence")}</strong>
      <span>${escapeHtml(hit.document_name || "")} - page ${escapeHtml(hit.page_no || "-")}</span>
    </div>
  `).join("") : activeDocs.slice(0, 4).map((doc) => `
    <div class="source-card"><strong>${escapeHtml(doc.original_name)}</strong><span>${escapeHtml(doc.extension || "file")} - ${formatBytes(doc.size_bytes)}</span></div>
  `).join("") || `<div class="source-card"><strong>No evidence yet</strong><span>Upload and build the mesh first.</span></div>`;
}

function renderProgress() {
  const progress = activeProgress || {};
  const percent = Math.max(0, Math.min(100, Number(progress.stage_percent || progress.percent || 0)));
  el("#status-title").textContent = progress.message || progress.progress_label || "Waiting for documents";
  el("#engine-detail").textContent = progress.progress_label || "Upload files before building the evidence mesh.";
  el("#progress-label").textContent = progress.progress_label || progress.message || "No documents uploaded yet.";
  el("#progress-bar").style.width = `${percent}%`;
}

function resetChat() {
  el("#chat-log").innerHTML = `<div class="chat-empty"><div><strong>Ask the mesh</strong><span> Cited answers will appear here after retrieval.</span></div></div>`;
}

function addMessage(role, content) {
  const log = el("#chat-log");
  const empty = log.querySelector(".chat-empty");
  if (empty) empty.remove();
  log.insertAdjacentHTML("beforeend", `
    <div class="message ${role === "user" ? "user" : "assistant"}">
      <b>${role === "user" ? "You" : "Evidence Mesh"}</b>
      <div>${escapeHtml(content).replaceAll("\n", "<br>")}</div>
    </div>
  `);
  log.scrollTop = log.scrollHeight;
}

function showError(message) {
  addMessage("assistant", `Error: ${message}`);
}

async function buildMesh() {
  if (!activeProjectId) return;
  el("#status-title").textContent = "Building mesh";
  await runWithProgress(() => api(`/api/projects/${activeProjectId}/build-index`, { method: "POST" }));
  await refreshActiveProject();
  await loadProjects();
}

el("#project-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const name = el("#project-name").value.trim();
  if (!name) return;
  const project = await api("/api/projects", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  el("#project-name").value = "";
  await loadProjects();
  await selectProject(project.project_id);
  location.hash = "#/overview";
});

el("#project-select").addEventListener("change", async (event) => {
  await selectProject(event.target.value);
});

el("#project-grid").addEventListener("click", async (event) => {
  const tile = event.target.closest("[data-project]");
  if (tile) await selectProject(tile.dataset.project);
});

el("#refresh").addEventListener("click", refreshActiveProject);
el("#build-index").addEventListener("click", () => buildMesh().catch((error) => showError(error.message)));
el("#build-index-overview").addEventListener("click", () => buildMesh().catch((error) => showError(error.message)));

el("#upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!activeProjectId) return;
  if (!stagedUploadFiles.length) return;
  const form = new FormData();
  for (const file of stagedUploadFiles) form.append("files", file);
  try {
    await runWithProgress(() => api(`/api/projects/${activeProjectId}/upload`, { method: "POST", body: form }));
    stagedUploadFiles = [];
    el("#files").value = "";
    renderStagedUploads();
    await refreshActiveProject();
    await loadProjects();
  } catch (error) {
    showError(error.message);
  }
});

el("#files").addEventListener("change", (event) => {
  const existing = new Set(stagedUploadFiles.map(uploadFileKey));
  for (const file of Array.from(event.target.files || [])) {
    const key = uploadFileKey(file);
    if (!existing.has(key)) {
      stagedUploadFiles.push(file);
      existing.add(key);
    }
  }
  event.target.value = "";
  renderStagedUploads();
});

el("#staged-upload-list").addEventListener("click", (event) => {
  const button = event.target.closest("[data-remove-staged]");
  if (!button) return;
  stagedUploadFiles.splice(Number(button.dataset.removeStaged), 1);
  renderStagedUploads();
});

el("#clear-staged-files").addEventListener("click", () => {
  stagedUploadFiles = [];
  el("#files").value = "";
  renderStagedUploads();
});

el("#documents-list").addEventListener("click", async (event) => {
  const button = event.target.closest("[data-delete-doc]");
  if (!button || !activeProjectId) return;
  await api(`/api/projects/${activeProjectId}/documents/${button.dataset.deleteDoc}`, { method: "DELETE" });
  await refreshActiveProject();
  await loadProjects();
});

el("#new-chat").addEventListener("click", () => {
  chatHistory = [];
  lastChatResult = null;
  resetChat();
  renderChatSidebars();
});

el("#chat-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!activeProjectId) return;
  const question = el("#question").value.trim();
  if (!question) return;
  el("#question").value = "";
  addMessage("user", question);
  try {
    const result = await api(`/api/projects/${activeProjectId}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, history: chatHistory, max_hits: 10 }),
    });
    lastChatResult = result;
    addMessage("assistant", result.answer || "");
    chatHistory.push({ role: "user", content: question }, { role: "assistant", content: result.answer || "" });
    chatHistory = chatHistory.slice(-10);
    renderChatSidebars();
  } catch (error) {
    showError(error.message);
  }
});

el("#llm-provider").addEventListener("change", (event) => {
  const models = providerModelFields(event.target.value);
  el("#llm-model").value = models.model;
  el("#llm-map-model").value = models.map_model;
  el("#llm-search-model").value = models.search_model;
});

el("#llm-settings-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  el("#settings-save-state").textContent = "Saving...";
  const payload = {
    provider: el("#llm-provider").value,
    api_key: el("#llm-api-key").value.trim(),
    model: el("#llm-model").value.trim(),
    map_model: el("#llm-map-model").value.trim(),
    search_model: el("#llm-search-model").value.trim(),
    openrouter_model: el("#openrouter-model").value.trim(),
    openrouter_map_model: el("#openrouter-map-model").value.trim(),
    openrouter_search_model: el("#openrouter-search-model").value.trim(),
    openai_model: el("#openai-model").value.trim(),
    openai_map_model: el("#openai-map-model").value.trim(),
    openai_search_model: el("#openai-search-model").value.trim(),
  };
  if (payload.provider === "openai") {
    payload.openai_model = payload.model || payload.openai_model;
    payload.openai_map_model = payload.map_model || payload.openai_map_model;
    payload.openai_search_model = payload.search_model || payload.openai_search_model;
  } else {
    payload.openrouter_model = payload.model || payload.openrouter_model;
    payload.openrouter_map_model = payload.map_model || payload.openrouter_map_model;
    payload.openrouter_search_model = payload.search_model || payload.openrouter_search_model;
  }
  try {
    runtimeSettings = await api("/api/runtime-settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    renderRuntimeSettings();
    el("#settings-save-state").textContent = "Saved. Restart running workers for long jobs.";
  } catch (error) {
    el("#settings-save-state").textContent = error.message;
  }
});

window.addEventListener("hashchange", () => setRoute(currentRoute()));

hydrateIcons();
setRoute(currentRoute());
loadRuntimeSettings().catch(() => {
  el("#llm-key-state").textContent = "Settings unavailable";
});
loadProjects().catch((error) => {
  el("#project-select").innerHTML = `<option>${escapeHtml(error.message)}</option>`;
});
