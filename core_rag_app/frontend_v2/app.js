let projects = [];
let activeProjectId = null;
let activeProject = null;
let activeDocs = [];
let activeProgress = null;
let runtimeSettings = null;
let stagedUploadFiles = [];
let chatHistory = [];
let lastChatResult = null;
let progressSocket = null;
let progressSocketProjectId = null;
let progressReconnectTimer = null;
const inlineCounterState = new Map();
let chatSearchMode = true;
let chatAttachmentUrl = null;

const el = (selector) => document.querySelector(selector);
const routes = {
  command: ["Workspace", "Run the corpus, graph, retrieval, and quality workflow."],
  corpus: ["Corpus Intake", "Upload documents and build the evidence mesh."],
  retrieval: ["Retrieval Studio", "Ask cited questions and inspect the retrieval trace."],
  graph: ["Graph Observatory", "Inspect graph coverage and relationship-active retrieval."],
  quality: ["Quality Lab", "Plan regression checks for retrieval accuracy."],
  settings: ["Runtime Settings", "Configure model providers and environment values."],
};

const icons = {
  command: '<svg viewBox="0 0 24 24"><path d="M5 7h14M5 12h10M5 17h14"/><path d="M18 10l2 2-2 2"/></svg>',
  corpus: '<svg viewBox="0 0 24 24"><path d="M4 7.5h6.4l1.8 2h7.8v8.8H4V7.5Z"/><path d="M7 14h10"/></svg>',
  retrieval: '<svg viewBox="0 0 24 24"><path d="M5 5.5h14v9.5H9l-4 4V5.5Z"/><path d="M9 9h6M9 12h4"/></svg>',
  graph: '<svg viewBox="0 0 24 24"><circle cx="6.5" cy="7" r="2"/><circle cx="17.5" cy="7" r="2"/><circle cx="12" cy="17" r="2"/><path d="M8 8.5l3 6.7M16 8.5l-3 6.7M8.5 7h7"/></svg>',
  quality: '<svg viewBox="0 0 24 24"><path d="M5 18.5V5.5M5 18.5h14"/><path d="M8.5 14.5l3-3.6 3 2.2 4-6.2"/></svg>',
  settings: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="3.4"/><path d="M12 4.5v2.2M12 17.3v2.2M5.6 5.6l1.6 1.6M16.8 16.8l1.6 1.6M4.5 12h2.2M17.3 12h2.2M5.6 18.4l1.6-1.6M16.8 7.2l1.6-1.6"/></svg>',
  plus: '<svg viewBox="0 0 24 24"><path d="M12 5.5v13M5.5 12h13"/></svg>',
  refresh: '<svg viewBox="0 0 24 24"><path d="M19 12a7 7 0 1 1-2-4.9"/><path d="M19 5.5v5h-5"/></svg>',
  send: '<svg viewBox="0 0 24 24"><path d="M4.5 11.8 19.5 5l-5.2 14-3-6.1-6.8-1.1Z"/></svg>',
  upload: '<svg viewBox="0 0 24 24"><path d="M12 15V5.5"/><path d="m8.5 9 3.5-3.5L15.5 9"/><path d="M5 16.5v2h14v-2"/></svg>',
  paperclip: '<svg viewBox="0 0 24 24"><path d="m21 11.5-8.7 8.7a5.2 5.2 0 0 1-7.4-7.4l9.2-9.2a3.5 3.5 0 0 1 5 5l-9.3 9.3a1.8 1.8 0 1 1-2.5-2.5l8.8-8.8"/></svg>',
  globe: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="8.5"/><path d="M3.5 12h17M12 3.5c2.2 2.3 3.3 5.1 3.3 8.5s-1.1 6.2-3.3 8.5M12 3.5C9.8 5.8 8.7 8.6 8.7 12s1.1 6.2 3.3 8.5"/></svg>',
  spark: '<svg viewBox="0 0 24 24"><path d="M12 3.8 13.9 9l5.3 1.8-5.3 1.9L12 18l-1.9-5.3-5.3-1.9L10.1 9 12 3.8Z"/><path d="M18 16.5l.8 2.1 2.1.8-2.1.7-.8 2.1-.7-2.1-2.1-.7 2.1-.8.7-2.1Z"/></svg>',
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

function formatCounterValue(value, format, locale = "en-US") {
  try {
    return new Intl.NumberFormat(locale, format).format(Number(value || 0));
  } catch (_error) {
    return String(value || 0);
  }
}

function animateNumber(node, value, options = {}) {
  if (!node) return;
  const {
    format,
    locale = "en-US",
    prefix = "",
    suffix = "",
    duration = 520,
    blur = 12,
  } = options;
  const numericValue = Number(value || 0);
  const formatted = formatCounterValue(numericValue, format, locale);
  const label = `${prefix}${formatted}${suffix}`;
  const state = node._animateNumberState || { value: numericValue, formatted: "" };
  const alreadyRendered = node.classList.contains("an-root") && node.childNodes.length;
  if (alreadyRendered && state.formatted === formatted && state.prefix === prefix && state.suffix === suffix) return;

  const direction = numericValue < state.value ? -1 : 1;
  const previousChars = String(state.formatted || "").split("");
  const nextChars = formatted.split("");
  const previousByOffset = new Map();
  previousChars.forEach((char, index) => {
    previousByOffset.set(previousChars.length - 1 - index, char);
  });

  node.classList.add("an-root");
  node.setAttribute("aria-label", label);
  node.innerHTML = "";

  if (prefix) {
    const prefixNode = document.createElement("span");
    prefixNode.className = "an-affix";
    prefixNode.setAttribute("aria-hidden", "true");
    prefixNode.textContent = prefix;
    node.append(prefixNode);
  }

  const hasPrevious = Boolean(state.formatted);
  nextChars.forEach((char, index) => {
    const offset = nextChars.length - 1 - index;
    const previous = previousByOffset.has(offset) ? previousByOffset.get(offset) : "";
    const changed = previous !== char;
    const slot = document.createElement("span");
    slot.className = "an-slot";
    slot.setAttribute("aria-hidden", "true");
    slot.style.setProperty("--an-dur", `${duration}ms`);
    slot.style.setProperty("--an-blur", `${blur}px`);
    slot.style.setProperty("--an-dir", String(direction));

    const incoming = document.createElement("span");
    incoming.className = hasPrevious && changed ? "an-layer an-in" : "an-layer";
    incoming.textContent = char || "\u200B";
    slot.append(incoming);

    if (hasPrevious && changed) {
      const outgoing = document.createElement("span");
      outgoing.className = "an-layer an-out";
      outgoing.textContent = previous || "\u200B";
      outgoing.addEventListener("animationend", () => outgoing.remove(), { once: true });
      window.setTimeout(() => outgoing.remove(), duration + 80);
      slot.append(outgoing);
    }
    node.append(slot);
  });

  if (suffix) {
    const suffixNode = document.createElement("span");
    suffixNode.className = "an-affix";
    suffixNode.setAttribute("aria-hidden", "true");
    suffixNode.textContent = suffix;
    node.append(suffixNode);
  }

  node._animateNumberState = {
    value: numericValue,
    formatted,
    prefix,
    suffix,
  };
}

function animateCount(selector, value, options = {}) {
  animateNumber(el(selector), value, options);
}

function counterMarkup(key, value, suffix = "", prefix = "") {
  return `<span class="inline-counter" data-count-key="${escapeHtml(key)}" data-count="${Number(value || 0)}" data-prefix="${escapeHtml(prefix)}" data-suffix="${escapeHtml(suffix)}"></span>`;
}

function hydrateInlineCounters(root = document) {
  root.querySelectorAll("[data-count]").forEach((node, index) => {
    const key = node.dataset.countKey || `${root.id || "counter"}-${index}`;
    const previousState = inlineCounterState.get(key);
    if (previousState) node._animateNumberState = previousState;
    animateNumber(node, Number(node.dataset.count || 0), {
      prefix: node.dataset.prefix || "",
      suffix: node.dataset.suffix || "",
      duration: 500,
      blur: 10,
    });
    inlineCounterState.set(key, node._animateNumberState);
  });
}

function renderInfoRows(selector, rows) {
  const container = el(selector);
  if (!container) return;
  container.innerHTML = rows.map((row) => {
    const body = row.html ? row.body : escapeHtml(row.body);
    return `<div><strong>${escapeHtml(row.title)}</strong><span>${body}</span></div>`;
  }).join("");
  hydrateInlineCounters(container);
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
  const route = location.hash.replace(/^#\/?/, "") || "command";
  return routes[route] ? route : "command";
}

function setRoute(route) {
  document.querySelectorAll("#project-view .page").forEach((page) => page.classList.remove("active-page"));
  const page = el(`#route-${route}`);
  if (page) page.classList.add("active-page");
  document.querySelectorAll("#nav a").forEach((link) => link.classList.toggle("active", link.dataset.route === route));
  el("#page-title").textContent = routes[route][0];
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

function isTerminalProgress(stage) {
  return ["idle", "uploaded", "complete", "failed"].includes(stage || "");
}

function setBusy(isBusy) {
  el("#build-index").disabled = isBusy;
  el("#build-index-command").disabled = isBusy;
  el("#refresh").disabled = isBusy;
  el("#engine-status").textContent = isBusy ? "Working" : "Ready";
}

function progressSocketUrl(projectId) {
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${location.host}/ws/projects/${encodeURIComponent(projectId)}/pipeline-progress`;
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
    } catch (_error) {}
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
  if (!activeProjectId && projects[0]) {
    await selectProject(projects[0].project_id);
  } else if (!projects.length) {
    el("#empty-state").classList.add("active-page");
    el("#project-view").classList.add("hidden");
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

async function selectProject(projectId) {
  if (!projectId) return;
  activeProjectId = projectId;
  activeProject = projects.find((project) => project.project_id === projectId) || activeProject;
  activeDocs = [];
  activeProgress = { stage: "syncing", message: "Loading project workspace.", progress_label: "Loading project workspace" };
  chatHistory = [];
  lastChatResult = null;
  el("#empty-state").classList.remove("active-page");
  el("#project-view").classList.remove("hidden");
  resetChat();
  renderAll();
  try {
    await refreshActiveProject();
  } catch (error) {
    activeProgress = { stage: "failed", message: error.message, progress_label: "Project details could not load" };
    renderAll();
  }
  connectProgressSocket(projectId);
}

async function refreshActiveProject() {
  if (!activeProjectId) return;
  const [project, docs, progress] = await Promise.all([
    api(`/api/projects/${activeProjectId}`),
    api(`/api/projects/${activeProjectId}/documents`),
    api(`/api/projects/${activeProjectId}/pipeline-progress`),
  ]);
  activeProject = {
    ...project,
    document_count: progress.document_count ?? project.document_count,
    card_count: progress.card_count ?? project.card_count,
    cluster_count: progress.cluster_count ?? project.cluster_count,
    domain_count: progress.domain_count ?? project.domain_count,
    relationship_count: progress.relationship_count ?? project.relationship_count,
  };
  activeDocs = docs;
  activeProgress = progress;
  renderAll();
}

function renderAll() {
  renderProjectSelect();
  renderCommand();
  renderCorpus();
  renderGraph();
  renderChatSidebars();
  renderProgress();
}

function renderCommand() {
  const progress = activeProgress || {};
  el("#active-project-name").textContent = activeProject?.name || "No project";
  el("#stage-pill").textContent = stageText(progress.stage);
  el("#overview-copy").textContent = progress.progress_label || progress.message || "Upload the source set, build the evidence mesh, then ask focused retrieval questions with traceable citations.";
  el("#mesh-root").textContent = activeProject?.name || "Corpus";
  animateCount("#mesh-root-sub", activeProject?.card_count, { suffix: " cards", duration: 500, blur: 10 });
  animateCount("#metric-docs", activeProject?.document_count, { duration: 560, blur: 11 });
  animateCount("#metric-docs-sub", activeDocs.length || activeProject?.document_count, { suffix: " files", duration: 500, blur: 9 });
  animateCount("#metric-cards", activeProject?.card_count, { duration: 560, blur: 11 });
  animateCount("#metric-domains", activeProject?.domain_count, { duration: 560, blur: 11 });
  animateCount("#metric-relationships", activeProject?.relationship_count, { duration: 560, blur: 11 });
  el("#engine-detail-mini").textContent = progress.message || "Runtime online";
  const totalPages = Number(progress.total_pages || 0);
  const indexedPages = Number(progress.indexed_pages || 0);
  const visibleIndexedPages = totalPages ? Math.min(indexedPages, totalPages) : indexedPages;
  renderInfoRows("#activity-list", [
    { title: "Current stage", body: stageText(progress.stage) },
    { title: "Current document", body: progress.current_document || "None" },
    {
      title: "Pages indexed",
      body: `${counterMarkup("pages-indexed-current", visibleIndexedPages)} <span class="counter-muted">of</span> ${counterMarkup("pages-indexed-total", totalPages)}`,
      html: true,
    },
  ]);
  renderInfoRows("#health-list", [
    { title: "Failed pages", body: counterMarkup("failed-pages", progress.failed_pages, " failed"), html: true },
    { title: "Graph audit", body: progress.graph_audit_status || "Not run" },
    { title: "Audit issues", body: counterMarkup("audit-issues", progress.graph_audit_issue_count, " issues"), html: true },
  ]);
}

function renderCorpus() {
  const progress = activeProgress || {};
  el("#upload-stage").textContent = stageText(progress.stage);
  animateCount("#queue-count", activeDocs.length, { suffix: " files", duration: 500, blur: 9 });
  renderStagedUploads();
  el("#documents-list").innerHTML = activeDocs.length ? activeDocs.map((doc) => `
    <div class="document-row">
      <div>
        <strong title="${escapeHtml(doc.original_name)}">${escapeHtml(doc.original_name)}</strong>
        <span>${escapeHtml(doc.extension || "file")} / ${formatBytes(doc.size_bytes)} / ${escapeHtml(doc.ingest_strategy || "queued")}</span>
      </div>
      <button type="button" data-delete-doc="${escapeHtml(doc.document_id)}">Remove</button>
    </div>
  `).join("") : `<div class="document-row"><div><strong>No documents uploaded</strong><span>Add PDFs, spreadsheets, or source documents to begin.</span></div></div>`;
}

function renderStagedUploads() {
  const list = el("#staged-upload-list");
  const count = stagedUploadFiles.length;
  if (count) {
    animateCount("#staged-upload-count", count, { suffix: " selected", duration: 500, blur: 9 });
  } else {
    const stagedCount = el("#staged-upload-count");
    stagedCount.classList.remove("an-root");
    stagedCount.removeAttribute("aria-label");
    stagedCount._animateNumberState = null;
    stagedCount.textContent = "No files selected";
  }
  el("#upload-submit").disabled = !count || !activeProjectId;
  el("#clear-staged-files").disabled = !count;
  list.innerHTML = count ? stagedUploadFiles.map((file, index) => `
    <div class="staged-file">
      <div><strong title="${escapeHtml(file.name)}">${escapeHtml(file.name)}</strong><span>${formatBytes(file.size)}</span></div>
      <button type="button" data-remove-staged="${index}">Remove</button>
    </div>
  `).join("") : `<div class="staged-empty">Selected files will appear here before upload.</div>`;
}

function renderGraph() {
  const progress = activeProgress || {};
  animateCount("#graph-domain-count", activeProject?.domain_count, { duration: 500, blur: 9 });
  animateCount("#graph-cluster-count", activeProject?.cluster_count, { duration: 500, blur: 9 });
  animateCount("#graph-card-count", activeProject?.card_count, { duration: 500, blur: 9 });
  animateCount("#graph-relation-count", activeProject?.relationship_count, { duration: 500, blur: 9 });
  el("#graph-root-name").textContent = activeProject?.name || "Corpus";
  animateCount("#graph-root-sub", activeProject?.card_count, { suffix: " cards", duration: 500, blur: 10 });
  renderInfoRows("#graph-inspector", [
    { title: "Status", body: stageText(progress.stage) },
    { title: "Domains", body: counterMarkup("graph-inspector-domains", activeProject?.domain_count), html: true },
    { title: "Clusters", body: counterMarkup("graph-inspector-clusters", activeProject?.cluster_count), html: true },
    { title: "Relationships", body: counterMarkup("graph-inspector-relationships", activeProject?.relationship_count), html: true },
    { title: "Audit", body: progress.graph_audit_status || "Not run" },
  ]);
}

function renderChatSidebars() {
  const trace = lastChatResult?.agent_trace || [];
  const hits = lastChatResult?.search?.hits || [];
  el("#evidence-timeline").innerHTML = trace.length ? trace.slice(-8).map((step) => `
    <div class="timeline-item"><strong>${escapeHtml(step.event || "agent_step")}</strong><span>${escapeHtml(JSON.stringify(step.detail || {}).slice(0, 180))}</span></div>
  `).join("") : [
    ["Planner", "Turns follow-ups into standalone evidence questions."],
    ["Retriever", "Uses keyword, graph, rerank, and query mode signals."],
    ["Grounder", "Answers only from retrieved evidence when possible."],
  ].map(([title, body]) => `<div class="timeline-item"><strong>${escapeHtml(title)}</strong><span>${escapeHtml(body)}</span></div>`).join("");
  el("#top-evidence").innerHTML = hits.length ? hits.slice(0, 6).map((hit) => `
    <div class="source-card"><strong>${escapeHtml(hit.card_name || hit.topic_name || "Evidence")}</strong><span>${escapeHtml(hit.document_name || "")} / page ${escapeHtml(hit.page_no || "-")}</span></div>
  `).join("") : activeDocs.slice(0, 4).map((doc) => `
    <div class="source-card"><strong>${escapeHtml(doc.original_name)}</strong><span>${escapeHtml(doc.extension || "file")} / ${formatBytes(doc.size_bytes)}</span></div>
  `).join("") || `<div class="source-card"><strong>No evidence yet</strong><span>Upload and build the mesh first.</span></div>`;
}

function renderProgress() {
  const progress = activeProgress || {};
  const percent = Math.max(0, Math.min(100, Number(progress.stage_percent || progress.percent || 0)));
  el("#status-title").textContent = progress.message || progress.progress_label || "Waiting for documents";
  el("#engine-detail").textContent = progress.progress_label || "Upload files before building.";
  el("#progress-label").textContent = progress.progress_label || progress.message || "No run yet";
  el("#progress-bar").style.width = `${percent}%`;
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
  el("#llm-key-state").textContent = runtimeSettings.api_key_present ? `Saved ${runtimeSettings.api_key_hint || ""}` : "No key saved";
}

function providerModelFields(provider) {
  if (provider === "openai") {
    return { model: el("#openai-model").value.trim(), map_model: el("#openai-map-model").value.trim(), search_model: el("#openai-search-model").value.trim() };
  }
  return { model: el("#openrouter-model").value.trim(), map_model: el("#openrouter-map-model").value.trim(), search_model: el("#openrouter-search-model").value.trim() };
}

const CHAT_INPUT_MIN_HEIGHT = 48;
const CHAT_INPUT_MAX_HEIGHT = 164;

function resizeChatInput(reset = false) {
  const textarea = el("#question");
  if (!textarea) return;
  if (reset) {
    textarea.style.height = `${CHAT_INPUT_MIN_HEIGHT}px`;
    return;
  }
  textarea.style.height = `${CHAT_INPUT_MIN_HEIGHT}px`;
  const nextHeight = Math.max(CHAT_INPUT_MIN_HEIGHT, Math.min(textarea.scrollHeight, CHAT_INPUT_MAX_HEIGHT));
  textarea.style.height = `${nextHeight}px`;
}

function updateChatComposer() {
  const textarea = el("#question");
  const placeholder = el("#question-placeholder");
  const toggle = el("#chat-search-toggle");
  const send = el("#send-chat");
  if (!textarea || !placeholder || !toggle || !send) return;
  placeholder.textContent = chatSearchMode ? "Search evidence..." : "Ask Evidence Mesh...";
  placeholder.classList.toggle("hidden", Boolean(textarea.value));
  placeholder.classList.remove("placeholder-swap");
  void placeholder.offsetWidth;
  placeholder.classList.add("placeholder-swap");
  toggle.classList.toggle("active", chatSearchMode);
  toggle.setAttribute("aria-pressed", String(chatSearchMode));
  send.classList.toggle("active", Boolean(textarea.value.trim()));
}

function clearChatAttachment() {
  if (chatAttachmentUrl) URL.revokeObjectURL(chatAttachmentUrl);
  chatAttachmentUrl = null;
  const input = el("#chat-attachment");
  const preview = el("#chat-attachment-preview");
  const label = el("#chat-attach-label");
  if (input) input.value = "";
  if (preview) {
    preview.innerHTML = "";
    preview.classList.add("hidden");
  }
  if (label) label.classList.remove("active");
}

function renderChatAttachment(file) {
  const preview = el("#chat-attachment-preview");
  const label = el("#chat-attach-label");
  if (!preview || !label) return;
  clearChatAttachment();
  label.classList.add("active");
  const isImage = file.type.startsWith("image/");
  chatAttachmentUrl = isImage ? URL.createObjectURL(file) : null;
  preview.classList.remove("hidden");
  preview.innerHTML = `
    <div class="attachment-chip">
      ${isImage ? `<img src="${chatAttachmentUrl}" alt="${escapeHtml(file.name)} preview" />` : `<span data-icon="paperclip"></span>`}
      <div>
        <strong title="${escapeHtml(file.name)}">${escapeHtml(file.name)}</strong>
        <small>${formatBytes(file.size)}</small>
      </div>
      <button type="button" data-clear-chat-attachment title="Remove attachment"><span data-icon="plus"></span></button>
    </div>
  `;
  hydrateIcons();
}

function resetChat() {
  el("#chat-log").innerHTML = `<div class="chat-empty"><div><strong>Ask the mesh</strong><span>Cited answers and trace data will appear here.</span></div></div>`;
}

function addMessage(role, content) {
  const log = el("#chat-log");
  const empty = log.querySelector(".chat-empty");
  if (empty) empty.remove();
  const message = document.createElement("div");
  message.className = `message ${role === "user" ? "user" : "assistant"}`;
  message.innerHTML = `
    <b>${role === "user" ? "You" : "Evidence Mesh"}</b>
    <div class="message-body">${escapeHtml(content).replaceAll("\n", "<br>")}</div>
  `;
  log.append(message);
  log.scrollTop = log.scrollHeight;
  return message;
}

function setMessageContent(message, content, isError = false) {
  if (!message || !message.isConnected) return addMessage("assistant", content);
  const log = el("#chat-log");
  const body = message.querySelector(".message-body");
  message.classList.remove("typing", "error");
  message.removeAttribute("aria-busy");
  message.classList.toggle("error", isError);
  if (body) body.innerHTML = escapeHtml(content).replaceAll("\n", "<br>");
  log.scrollTop = log.scrollHeight;
  return message;
}

function addTypingMessage() {
  const message = addMessage("assistant", "");
  const body = message.querySelector(".message-body");
  message.classList.add("typing");
  message.setAttribute("aria-busy", "true");
  if (body) {
    body.innerHTML = `
      <span class="typing-indicator" aria-live="polite">
        <span class="typing-copy">Reading the graph and source evidence</span>
        <span class="typing-dots" aria-hidden="true"><i></i><i></i><i></i></span>
      </span>
    `;
  }
  return message;
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
  location.hash = "#/command";
});

el("#project-select").addEventListener("change", async (event) => selectProject(event.target.value));
el("#refresh").addEventListener("click", refreshActiveProject);
el("#build-index").addEventListener("click", () => buildMesh().catch((error) => showError(error.message)));
el("#build-index-command").addEventListener("click", () => buildMesh().catch((error) => showError(error.message)));

el("#upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!activeProjectId || !stagedUploadFiles.length) return;
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
  el("#question").value = "";
  resizeChatInput(true);
  clearChatAttachment();
  updateChatComposer();
  resetChat();
  renderChatSidebars();
});

el("#question").addEventListener("input", () => {
  resizeChatInput();
  updateChatComposer();
});

el("#question").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    el("#chat-form").requestSubmit();
  }
});

el("#chat-search-toggle").addEventListener("click", () => {
  chatSearchMode = !chatSearchMode;
  updateChatComposer();
  el("#question").focus();
});

el("#chat-attachment").addEventListener("change", (event) => {
  const file = event.target.files?.[0];
  if (file) renderChatAttachment(file);
});

el("#chat-attachment-preview").addEventListener("click", (event) => {
  const button = event.target.closest("[data-clear-chat-attachment]");
  if (!button) return;
  clearChatAttachment();
});

window.addEventListener("resize", () => resizeChatInput());

el("#chat-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!activeProjectId) return;
  const question = el("#question").value.trim();
  if (!question) return;
  const form = el("#chat-form");
  const sendButton = el("#send-chat");
  el("#question").value = "";
  resizeChatInput(true);
  updateChatComposer();
  addMessage("user", question);
  const pendingMessage = addTypingMessage();
  form.classList.add("is-waiting");
  sendButton.disabled = true;
  try {
    const result = await api(`/api/projects/${activeProjectId}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, history: chatHistory, max_hits: 10, query_mode: "auto" }),
    });
    lastChatResult = result;
    setMessageContent(pendingMessage, result.answer || "No answer returned.");
    chatHistory.push({ role: "user", content: question }, { role: "assistant", content: result.answer || "" });
    chatHistory = chatHistory.slice(-10);
    renderChatSidebars();
  } catch (error) {
    setMessageContent(pendingMessage, `Error: ${error.message}`, true);
  } finally {
    form.classList.remove("is-waiting");
    sendButton.disabled = false;
    updateChatComposer();
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
resizeChatInput(true);
updateChatComposer();
setRoute(currentRoute());
resetChat();
loadRuntimeSettings().catch(() => {
  el("#llm-key-state").textContent = "Settings unavailable";
});
loadProjects().catch((error) => {
  el("#project-select").innerHTML = `<option>${escapeHtml(error.message)}</option>`;
});
