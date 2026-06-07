let projects = [];
let activeProject = null;
let documents = [];
let legalReport = {};
let commercialReport = {};
let financialReport = {};
let prebidReport = {};
let prequalificationReport = {};
let pipelineProgress = {};
let stopPipelineSocket = null;
let stopAgentProgressWatcher = null;
const chatHistoryByProject = {};

const page = document.querySelector("#page");
const projectForm = document.querySelector("#project-form");
const projectName = document.querySelector("#project-name");
const pageTitle = document.querySelector("#page-title");
const routeKicker = document.querySelector("#route-kicker");

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || "Request failed");
  }
  return response.json();
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function isBlankCell(value) {
  const text = String(value ?? "").trim();
  return !text || text === "-" || text.toLowerCase() === "n/a";
}

function uniqueJoined(values, fallback = "-") {
  const clean = [...new Set(values.map((value) => String(value ?? "").trim()).filter((value) => !isBlankCell(value)))];
  return clean.length ? clean.join("; ") : fallback;
}

function prebidDocumentName(row) {
  return uniqueJoined((row.citations || []).map((item) => item.document_name), "-");
}

function prebidPageNo(row) {
  if (!isBlankCell(row.page_no)) return row.page_no;
  return uniqueJoined((row.citations || []).map((item) => item.page_no), "-");
}

function prebidSheetColumns() {
  return [
    ["S. No.", "s_no", "pbq-col-sno"],
    ["Document Name", "document_name", "pbq-col-document"],
    ["Tender Vol. / Section", "tender_vol_section", "pbq-col-section"],
    ["Page No.", "page_no", "pbq-col-page"],
    ["Clause No.", "clause_no", "pbq-col-clause"],
    ["Clause Description", "clause_description", "pbq-col-description"],
    ["Bidder's Query", "bidder_query", "pbq-col-query"],
    ["Basis / Why This Is Needed", "basis", "pbq-col-basis"],
  ];
}

function prebidSheetValue(row, key, index) {
  if (key === "s_no") return row.s_no || index + 1;
  if (key === "document_name") return prebidDocumentName(row);
  if (key === "page_no") return prebidPageNo(row);
  return row[key] || "-";
}

function projectFileStem() {
  return (activeProject?.name || activeProject?.project_id || "project").replace(/[^a-z0-9_-]+/gi, "_");
}

function joinedCitations(items = []) {
  return items
    .map((item) => {
      const documentName = item.document_name || item.title || "Source";
      const page = item.page_no ? ` / page ${item.page_no}` : "";
      const topic = item.topic_name ? ` / ${item.topic_name}` : "";
      const excerpt = item.excerpt || item.note || "";
      return `${documentName}${page}${topic}${excerpt ? `: ${excerpt}` : ""}`;
    })
    .filter(Boolean)
    .join("\n\n");
}

function downloadExcelWorkbook(filename, columns, rows) {
  if (!rows.length) return;
  const table = `
    <table>
      <thead>
        <tr>${columns.map((column) => `<th>${escapeHtml(column.label)}</th>`).join("")}</tr>
      </thead>
      <tbody>
        ${rows
          .map(
            (row) => `
              <tr>
                ${columns.map((column) => `<td>${escapeHtml(column.value(row))}</td>`).join("")}
              </tr>
            `,
          )
          .join("")}
      </tbody>
    </table>
  `;
  const workbook = `<!doctype html><html><head><meta charset="utf-8" /></head><body>${table}</body></html>`;
  const blob = new Blob([workbook], { type: "application/vnd.ms-excel;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function downloadPrebidExcel() {
  const rows = (prebidReport?.rows || []).map((row, index) => ({ ...row, _index: index }));
  const columns = prebidSheetColumns().map(([label, key]) => ({
    label,
    value: (row) => prebidSheetValue(row, key, row._index),
  }));
  downloadExcelWorkbook(`${projectFileStem()}_prebid_queries.xls`, columns, rows);
}

function downloadPrequalificationExcel() {
  const rows = prequalificationReport?.rows || [];
  const columns = [
    ["S. No.", (row) => row.s_no],
    ["Requirement Type", (row) => row.requirement_type],
    ["Requirement Area", (row) => row.requirement_area],
    ["Tender Vol. / Section", (row) => row.tender_vol_section],
    ["Document Name", (row) => row.document_name],
    ["Page No.", (row) => row.page_no],
    ["Clause No.", (row) => row.clause_no],
    ["Requirement Text", (row) => row.requirement_text],
    ["Threshold / Value", (row) => row.threshold_or_value],
    ["Applicable To", (row) => row.applicable_to],
    ["Proof Required", (row) => row.proof_required],
    ["Compliance Note", (row) => row.compliance_note],
    ["Confidence", (row) => row.confidence],
    ["Citations", (row) => joinedCitations(row.citations || [])],
  ].map(([label, value]) => ({ label, value }));
  downloadExcelWorkbook(`${projectFileStem()}_prequalification_requirements.xls`, columns, rows);
}

function downloadLegalExcel() {
  const rows = legalReport?.rows || [];
  const columns = [
    ["S. No.", (row) => row.s_no],
    ["Topic", (row) => row.topic],
    ["Yes / No", (row) => row.yes_no],
    ["Comments", (row) => row.comments],
    ["Confidence", (row) => row.confidence],
    ["Warnings", (row) => (row.verifier?.warnings || []).join("\n")],
    ["Evidence", (row) => joinedCitations(row.evidence || [])],
  ].map(([label, value]) => ({ label, value }));
  downloadExcelWorkbook(`${projectFileStem()}_legal_assessment.xls`, columns, rows);
}

function downloadCommercialExcel() {
  const rows = (commercialReport?.sections || []).flatMap((section) =>
    (section.bullets || []).map((bullet, index) => ({
      section_title: section.title,
      bullet_no: index + 1,
      ...bullet,
    })),
  );
  const columns = [
    ["Section", (row) => row.section_title],
    ["Bullet No.", (row) => row.bullet_no],
    ["Point", (row) => row.text],
    ["Basis", (row) => row.basis],
    ["Why It Matters", (row) => row.why_it_matters],
    ["Commercial Score", (row) => row.score?.commercial_value],
    ["Strategic Score", (row) => row.score?.strategic_value],
    ["Evidence Score", (row) => row.score?.evidence_strength],
    ["Win Relevance", (row) => row.score?.win_relevance],
    ["Specificity", (row) => row.score?.specificity],
    ["Risk Caveat", (row) => row.score?.risk_caveat || row.caveat],
    ["Tender Evidence", (row) => joinedCitations(row.evidence_citations || [])],
    ["Web Citations", (row) => joinedCitations(row.web_citations || [])],
  ].map(([label, value]) => ({ label, value }));
  downloadExcelWorkbook(`${projectFileStem()}_commercial_strategy.xls`, columns, rows);
}

function downloadFinancialExcel() {
  const rows = financialReport?.rows || [];
  const columns = [
    ["S. No.", (row) => row.s_no],
    ["Topic", (row) => row.topic],
    ["Comments", (row) => row.comments],
    ["Required Status", (row) => row.extraction?.required_status],
    ["Amount", (row) => row.extraction?.amount],
    ["Percentage", (row) => row.extraction?.percentage],
    ["Basis", (row) => row.extraction?.basis],
    ["Instrument", (row) => row.extraction?.instrument],
    ["Cash / BG", (row) => row.extraction?.cash_or_bg],
    ["Validity", (row) => row.extraction?.validity],
    ["Recovery", (row) => row.extraction?.recovery],
    ["Release Condition", (row) => row.extraction?.release_condition],
    ["Confidence", (row) => row.confidence],
    ["Warnings", (row) => (row.verifier?.warnings || []).join("\n")],
    ["Evidence", (row) => joinedCitations(row.evidence || [])],
  ].map(([label, value]) => ({ label, value }));
  downloadExcelWorkbook(`${projectFileStem()}_financial_bonds.xls`, columns, rows);
}

function projectIdFromPath() {
  const parts = location.pathname.split("/").filter(Boolean);
  return parts[0] === "projects" && parts[1] ? decodeURIComponent(parts[1]) : null;
}

function go(path) {
  history.pushState({}, "", path);
  renderApp();
}

function formatBytes(bytes = 0) {
  if (!bytes) return "0 KB";
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function metricItems(project = {}) {
  return [
    ["Documents", project.document_count || 0],
    ["Topics", project.topic_count || 0],
    ["Communities", project.community_count || 0],
    ["Biomes", project.biome_count || 0],
    ["Relations", project.relation_count || 0],
  ];
}

function metricsHtml(project = {}) {
  return metricItems(project)
    .map(([label, value]) => `<div class="metric"><strong>${value}</strong><span>${label}</span></div>`)
    .join("");
}

async function loadProjects() {
  projects = await api("/api/projects");
}

async function loadProject(projectId) {
  activeProject = await api(`/api/projects/${projectId}`);
  documents = await api(`/api/projects/${projectId}/documents`);
  legalReport = await api(`/api/projects/${projectId}/reports/legal-assessment`).catch(() => ({}));
  commercialReport = await api(`/api/projects/${projectId}/reports/commercial-strategy`).catch(() => ({}));
  financialReport = await api(`/api/projects/${projectId}/reports/financial-bonds`).catch(() => ({}));
  prebidReport = await api(`/api/projects/${projectId}/reports/prebid-queries`).catch(() => ({}));
  prequalificationReport = await api(`/api/projects/${projectId}/reports/prequalification-requirements`).catch(() => ({}));
  pipelineProgress = await api(`/api/projects/${projectId}/pipeline-progress`).catch(() => ({}));
}

function renderProjects() {
  if (stopAgentProgressWatcher) stopAgentProgressWatcher();
  activeProject = null;
  documents = [];
  routeKicker.textContent = "Projects";
  pageTitle.textContent = "All projects";
  page.innerHTML = `
    <section class="project-home">
      <div class="project-home-head">
        <div>
          <h2>Open a project</h2>
          <p>Each project has its own document inventory, index metrics, relationship graph health, and chat assistant.</p>
        </div>
        <span>${projects.length} total</span>
      </div>
      <div class="project-grid">
        ${
          projects.length
            ? projects
                .map(
                  (project) => `
                    <button class="project-card" data-project="${project.project_id}">
                      <strong>${escapeHtml(project.name)}</strong>
                      <span>${project.document_count || 0} docs / ${project.topic_count || 0} topics</span>
                      <small>${project.community_count || 0} communities / ${project.relation_count || 0} relations</small>
                    </button>
                  `,
                )
                .join("")
            : `<div class="empty-state"><h2>No projects yet</h2><p>Create your first project from the top bar.</p></div>`
        }
      </div>
    </section>
  `;

  page.querySelectorAll(".project-card").forEach((card) => {
    card.addEventListener("click", () => go(`/projects/${encodeURIComponent(card.dataset.project)}`));
  });
}

function renderProjectDetail() {
  routeKicker.textContent = "Project";
  pageTitle.textContent = activeProject.name;
  page.innerHTML = `
    <div class="project-detail">
      <section class="project-summary">
        <div>
          <a class="back-link" href="/projects" data-link>Back to projects</a>
          <h2>${escapeHtml(activeProject.name)}</h2>
          <p>Created ${escapeHtml(activeProject.created_at || "unknown")} / Updated ${escapeHtml(activeProject.updated_at || "unknown")}</p>
        </div>
        <div class="metrics">${metricsHtml(activeProject)}</div>
      </section>

      <section class="section upload-section">
        <div class="section-head">
          <div>
            <h2>Documents</h2>
            <p>${documents.length} files are attached to this project.</p>
          </div>
          <label class="file-button">
            Upload files
            <input id="file-input" type="file" multiple />
          </label>
        </div>
        <div id="dropzone" class="dropzone">
          <strong>Drop files here</strong>
          <span>Uploading only updates the document inventory. Start indexing after you verify the full file list.</span>
        </div>
        <div class="button-row">
          <button id="start-index" type="button" ${documents.length ? "" : "disabled"}>Start index generation</button>
        </div>
        <div id="upload-status" class="status"></div>
        <div id="pipeline-progress">${pipelineProgressHtml(pipelineProgress)}</div>
      </section>

      <section class="section generated-docs">
        <div class="section-head">
          <div>
            <h2>Generated docs</h2>
            <p>Create project documents from the full indexed corpus, flagged evidence, fresh searches, and specialist verification.</p>
          </div>
          <div class="button-row">
            <button id="generate-all" type="button">Run All Agents</button>
            <button id="generate-prebid" type="button">Generate Pre-Bid Queries</button>
            <button id="generate-prequalification" type="button">Generate Pre-Qualification Requirements</button>
            <button id="generate-commercial" type="button">Generate Commercial Strategy</button>
            <button id="generate-financial" type="button">Generate Financial Bonds</button>
            <button id="generate-legal" type="button">Generate Legal Assessment</button>
          </div>
        </div>
        <div id="all-agents-status" class="status"></div>
        <div id="prebid-status" class="status"></div>
        <div id="prebid-report">${prebidReportHtml(prebidReport)}</div>
        <div id="prequalification-status" class="status"></div>
        <div id="prequalification-report">${prequalificationReportHtml(prequalificationReport)}</div>
        <div id="commercial-status" class="status"></div>
        <div id="commercial-report">${commercialReportHtml(commercialReport)}</div>
        <div id="financial-status" class="status"></div>
        <div id="financial-report">${financialReportHtml(financialReport)}</div>
        <div id="legal-status" class="status"></div>
        <div id="legal-report">${legalReportHtml(legalReport)}</div>
      </section>

      <section class="detail-grid">
        <div class="section">
          <div class="section-head">
            <h2>Document inventory</h2>
          </div>
          <div class="document-list">
            ${
              documents.length
                ? documents
                    .map(
                      (doc) => `
                        <div class="doc-row">
                          <div>
                            <strong>${escapeHtml(doc.original_name)}</strong>
                            <span>${escapeHtml(doc.mime_type)} / ${formatBytes(doc.size_bytes)}</span>
                          </div>
                          <div class="doc-actions">
                            <small>${escapeHtml(doc.ingest_strategy)}</small>
                            <button class="danger-button" type="button" data-delete-document="${escapeHtml(doc.document_id)}">Remove</button>
                          </div>
                        </div>
                      `,
                    )
                    .join("")
                : `<div class="empty-row">No documents uploaded yet.</div>`
            }
          </div>
        </div>

        <div class="section">
          <div class="section-head">
            <h2>Index health</h2>
          </div>
          <div class="health-list">${healthRowsHtml(activeProject)}</div>
        </div>
      </section>
    </div>
    <button class="chat-fab" type="button" aria-label="Open project chat">Chat</button>
    <aside class="chat-drawer" aria-hidden="true">
      <div class="chat-drawer-head">
        <div>
          <strong>Project chat</strong>
          <span>${escapeHtml(activeProject.name)}</span>
        </div>
        <button type="button" class="chat-close" aria-label="Close chat">Close</button>
      </div>
      <div id="messages" class="messages">${chatMessagesHtml(activeProject.project_id)}</div>
      <form id="chat-form" class="chat-form">
        <textarea id="chat-input" placeholder="Ask about this project..."></textarea>
        <button type="submit">Ask</button>
      </form>
    </aside>
  `;

  bindProjectDetail();
  startPipelineSocket();
  startAgentProgressWatcher();
}

function pipelineProgressHtml(progress = {}) {
  const total = Number(progress.total_pages || 0);
  const indexed = Number(progress.indexed_pages || 0);
  const percent = Math.min(100, Number(progress.stage_percent ?? progress.percent ?? 0));
  const stage = progress.stage || "idle";
  const message = progress.message || "No pipeline activity yet.";
  const pageDetail = stage === "uploaded"
    ? `${progress.document_count || 0} documents uploaded`
    : total
    ? `${indexed} / ${total} pages indexed`
    : `${progress.document_count || 0} documents`;
  const detail = progress.progress_label || pageDetail;
  const current = progress.current_document
    ? `${progress.current_document}${progress.current_page ? ` / page ${progress.current_page}` : ""}`
    : "";
  const stageLine = stage === "indexing" && current ? current : detail;
  const relationshipDetail = progress.relationship_total_documents
    ? `${progress.relationship_document_count || 0} / ${progress.relationship_total_documents} docs mapped`
    : "";
  const relationPairDetail = progress.relation_pairs_total
    ? `${progress.relation_pairs_done || 0} / ${progress.relation_pairs_total} relation checks`
    : "";
  return `
    <div class="pipeline-card" data-stage="${escapeHtml(stage)}">
      <div class="pipeline-top">
        <div>
          <strong>${escapeHtml(message)}</strong>
          <span>${escapeHtml(stageLine)}</span>
        </div>
        <b>${escapeHtml(stage)}</b>
      </div>
      <div class="progress-track" aria-label="Pipeline progress">
        <i style="width:${percent}%"></i>
      </div>
      <div class="pipeline-metrics">
        <span>${escapeHtml(pageDetail)}</span>
        ${relationshipDetail ? `<span>${escapeHtml(relationshipDetail)}</span>` : ""}
        ${relationPairDetail ? `<span>${escapeHtml(relationPairDetail)}</span>` : ""}
        <span>${escapeHtml(progress.failed_relation_pair_count || 0)} failed relation checks</span>
        <span>${escapeHtml(progress.topic_count || 0)} topics</span>
        <span>${escapeHtml(progress.community_count || 0)} communities</span>
        <span>${escapeHtml(progress.biome_count || 0)} biomes</span>
        <span>${escapeHtml(progress.relation_count || 0)} relations</span>
        <span>${escapeHtml(progress.failed_pages || 0)} failed pages</span>
        <span>${escapeHtml(progress.placeholder_pages || 0)} placeholder pages</span>
      </div>
    </div>
  `;
}

async function refreshPipelineProgress() {
  if (!activeProject) return {};
  pipelineProgress = await api(`/api/projects/${activeProject.project_id}/pipeline-progress`).catch(() => pipelineProgress);
  const target = page.querySelector("#pipeline-progress");
  if (target) target.innerHTML = pipelineProgressHtml(pipelineProgress);
  return pipelineProgress;
}

function startPipelineSocket({ reloadOnDone = false } = {}) {
  if (!activeProject || !("WebSocket" in window)) return null;
  if (stopPipelineSocket) stopPipelineSocket();
  let stopped = false;
  const projectId = activeProject.project_id;
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  const socket = new WebSocket(`${protocol}//${location.host}/ws/projects/${encodeURIComponent(projectId)}/pipeline-progress`);
  socket.addEventListener("message", async (event) => {
    if (stopped) return;
    pipelineProgress = JSON.parse(event.data);
    const target = page.querySelector("#pipeline-progress");
    if (target) target.innerHTML = pipelineProgressHtml(pipelineProgress);
    if (reloadOnDone && ["uploaded", "complete", "failed"].includes(pipelineProgress.stage)) {
      stopped = true;
      socket.close();
      await loadProject(projectId);
      renderProjectDetail();
    }
  });
  socket.addEventListener("close", () => {
    if (stopPipelineSocket === stop) stopPipelineSocket = null;
    if (!stopped && activeProject?.project_id === projectId && !["uploaded", "complete", "failed"].includes(pipelineProgress.stage)) {
      setTimeout(() => {
        if (!stopPipelineSocket && activeProject?.project_id === projectId) {
          startPipelineSocket({ reloadOnDone });
        }
      }, 1200);
    }
  });
  socket.addEventListener("error", async () => {
    if (!stopped) {
      await refreshPipelineProgress();
    }
  });
  const stop = () => {
    stopped = true;
    if (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING) {
      socket.close();
    }
  };
  stopPipelineSocket = stop;
  return stop;
}

function healthRowsHtml(project) {
  const max = Math.max(project.topic_count || 1, project.relation_count || 1);
  return metricItems(project)
    .map(([label, value]) => {
      const width = Math.max(8, Math.round((value / max) * 100));
      return `
        <div class="health-row">
          <span>${label}</span>
          <div><i style="width:${width}%"></i></div>
          <strong>${value}</strong>
        </div>
      `;
    })
    .join("");
}

function chatMessagesHtml(projectId) {
  const history = chatHistoryByProject[projectId] || [];
  if (!history.length) {
    return `<div class="empty-row">Ask about scope, risks, commercial terms, execution details, or evidence in the documents.</div>`;
  }
  return history.map((message) => `<div class="message ${message.role}">${escapeHtml(message.content)}</div>`).join("");
}

function bindPbqModal(scope = page) {
  const expandPbq = scope.querySelector("[data-expand-pbq]");
  const pbqModal = scope.querySelector("[data-pbq-modal]");
  if (!expandPbq || !pbqModal || pbqModal.dataset.bound === "true") return;
  pbqModal.dataset.bound = "true";
  const closePbqModal = () => {
    pbqModal.classList.remove("open");
    pbqModal.setAttribute("aria-hidden", "true");
  };
  expandPbq.addEventListener("click", () => {
    pbqModal.classList.add("open");
    pbqModal.setAttribute("aria-hidden", "false");
  });
  pbqModal.addEventListener("click", (event) => {
    if (event.target.closest("[data-close-pbq]") || event.target === pbqModal) closePbqModal();
  });
}

function bindProjectDetail() {
  const fileInput = page.querySelector("#file-input");
  const dropzone = page.querySelector("#dropzone");
  const chatFab = page.querySelector(".chat-fab");
  const chatDrawer = page.querySelector(".chat-drawer");
  const chatClose = page.querySelector(".chat-close");
  const chatForm = page.querySelector("#chat-form");
  const chatInput = page.querySelector("#chat-input");
  const messages = page.querySelector("#messages");
  const documentList = page.querySelector(".document-list");
  const generateLegal = page.querySelector("#generate-legal");
  const generateCommercial = page.querySelector("#generate-commercial");
  const generateFinancial = page.querySelector("#generate-financial");
  const generatePrebid = page.querySelector("#generate-prebid");
  const generatePrequalification = page.querySelector("#generate-prequalification");
  const generateAll = page.querySelector("#generate-all");
  const startIndex = page.querySelector("#start-index");

  fileInput.addEventListener("change", () => uploadFiles(fileInput.files));
  dropzone.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropzone.classList.add("dragging");
  });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragging"));
  dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragging");
    uploadFiles(event.dataTransfer.files);
  });

  chatFab.addEventListener("click", () => {
    chatDrawer.classList.add("open");
    chatDrawer.setAttribute("aria-hidden", "false");
    messages.scrollTop = messages.scrollHeight;
    chatInput.focus();
  });
  chatClose.addEventListener("click", () => {
    chatDrawer.classList.remove("open");
    chatDrawer.setAttribute("aria-hidden", "true");
  });

  chatInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      chatForm.requestSubmit();
    }
  });

  chatForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const question = chatInput.value.trim();
    if (!question) return;
    chatInput.value = "";
    addMessage(messages, "user", question);
    rememberMessage(activeProject.project_id, "user", question);
    const pending = addMessage(messages, "assistant", "Searching the project tree...");
    try {
      const historyForApi = (chatHistoryByProject[activeProject.project_id] || []).slice(0, -1);
      const result = await api(`/api/projects/${activeProject.project_id}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, max_hits: 10, history: historyForApi }),
      });
      pending.textContent = result.answer;
      rememberMessage(activeProject.project_id, "assistant", result.answer);
    } catch (error) {
      pending.textContent = error.message;
      rememberMessage(activeProject.project_id, "assistant", error.message);
    }
  });

  generateLegal.addEventListener("click", generateLegalAssessment);
  generateCommercial.addEventListener("click", generateCommercialStrategy);
  generateFinancial.addEventListener("click", generateFinancialBonds);
  generatePrebid.addEventListener("click", generatePrebidQueries);
  generatePrequalification.addEventListener("click", generatePrequalificationRequirements);
  generateAll.addEventListener("click", generateAllAgents);
  startIndex.addEventListener("click", buildIndexAndRelations);
  bindPbqModal(page);
  documentList.addEventListener("click", (event) => {
    const button = event.target.closest("[data-delete-document]");
    if (!button) return;
    deleteDocument(button.dataset.deleteDocument);
  });
}

async function runAgent({ key, label, endpoint, statusId, outputId, render }) {
  const status = page.querySelector(statusId);
  const output = page.querySelector(outputId);
  let finished = false;
  const progressEndpoint = `${endpoint}/progress`;
  const renderProgress = (progress = {}) => {
    const logs = progress.logs || [];
    const latest = logs.at(-1)?.message || `${label} is preparing the specialist workflow.`;
    status.textContent = `${label}: ${progress.status || "starting"} / ${latest}`;
    output.innerHTML = agentProgressHtml(label, progress);
    autoScrollAgentProgress(output);
  };
  renderProgress({ status: "starting", logs: [] });
  const pollProgress = async () => {
    while (!finished) {
      const progress = await api(`/api/projects/${activeProject.project_id}${progressEndpoint}`).catch(() => ({}));
      if (!finished) renderProgress(progress);
      await new Promise((resolve) => setTimeout(resolve, 1000));
    }
  };
  const poller = pollProgress();
  try {
    const report = await api(`/api/projects/${activeProject.project_id}${endpoint}`, { method: "POST" });
    finished = true;
    await poller.catch(() => {});
    setReportForKey(key, report);
    status.textContent = `${label}: complete.`;
    output.innerHTML = render(report);
    bindPbqModal(output);
    return { key, label, ok: true };
  } catch (error) {
    finished = true;
    await poller.catch(() => {});
    const existingReport = await api(`/api/projects/${activeProject.project_id}${endpoint}`).catch(() => ({}));
    if (hasGeneratedReport(existingReport)) {
      setReportForKey(key, existingReport);
      status.textContent = `${label}: latest run failed, showing previously generated document. ${error.message}`;
      output.innerHTML = render(existingReport);
      bindPbqModal(output);
    } else {
      status.textContent = `${label}: ${error.message}`;
    }
    return { key, label, ok: false, error };
  }
}

function setReportForKey(key, report) {
  if (key === "prebid") prebidReport = report;
  if (key === "commercial") commercialReport = report;
  if (key === "financial") financialReport = report;
  if (key === "legal") legalReport = report;
  if (key === "prequalification") prequalificationReport = report;
}

function hasGeneratedReport(report = {}) {
  return Boolean((report.rows || []).length || (report.sections || []).length);
}

function agentProgressHtml(label, progress = {}) {
  const logs = progress.logs || [];
  const latest = logs.at(-1)?.message || "Waiting for the first specialist step.";
  const recentLogs = logs
    .slice(-8)
    .map((log) => `<div class="log-line">${escapeHtml(log.message)}</div>`)
    .join("");
  return `
    <div class="agent-progress-card">
      <div class="pipeline-top">
        <div>
          <strong>${escapeHtml(label)} / ${escapeHtml(progress.status || "starting")}</strong>
          <span>${escapeHtml(latest)}</span>
        </div>
        <b>${escapeHtml(logs.length)} steps</b>
      </div>
      <div class="agent-progress-log">
        ${recentLogs || `<div class="log-line">Waiting for progress...</div>`}
      </div>
    </div>
  `;
}

function agentSpecs() {
  return [
    {
      key: "prebid",
      label: "Pre-Bid Queries",
      endpoint: "/reports/prebid-queries",
      statusId: "#prebid-status",
      outputId: "#prebid-report",
      render: prebidReportHtml,
    },
    {
      key: "prequalification",
      label: "Pre-Qualification Requirements",
      endpoint: "/reports/prequalification-requirements",
      statusId: "#prequalification-status",
      outputId: "#prequalification-report",
      render: prequalificationReportHtml,
    },
    {
      key: "commercial",
      label: "Commercial Strategy",
      endpoint: "/reports/commercial-strategy",
      statusId: "#commercial-status",
      outputId: "#commercial-report",
      render: commercialReportHtml,
    },
    {
      key: "financial",
      label: "Financial Bonds",
      endpoint: "/reports/financial-bonds",
      statusId: "#financial-status",
      outputId: "#financial-report",
      render: financialReportHtml,
    },
    {
      key: "legal",
      label: "Legal Assessment",
      endpoint: "/reports/legal-assessment",
      statusId: "#legal-status",
      outputId: "#legal-report",
      render: legalReportHtml,
    },
  ];
}

async function hydrateRunningAgentProgress() {
  if (!activeProject) return;
  for (const spec of agentSpecs()) {
    const progress = await api(`/api/projects/${activeProject.project_id}${spec.endpoint}/progress`).catch(() => ({}));
    if (progress.status === "running") {
      watchExistingAgent(spec, progress);
    }
  }
}

async function watchExistingAgent(spec, initialProgress = {}) {
  const status = page.querySelector(spec.statusId);
  const output = page.querySelector(spec.outputId);
  if (!status || !output) return;
  let progress = initialProgress;
  while (activeProject && progress.status === "running") {
    const logs = progress.logs || [];
    const latest = logs.at(-1)?.message || `${spec.label} is running.`;
    status.textContent = `${spec.label}: ${progress.status} / ${latest}`;
    output.innerHTML = agentProgressHtml(spec.label, progress);
    autoScrollAgentProgress(output);
    await new Promise((resolve) => setTimeout(resolve, 1000));
    progress = await api(`/api/projects/${activeProject.project_id}${spec.endpoint}/progress`).catch(() => progress);
  }
  if (progress.status === "complete") {
    const report = await api(`/api/projects/${activeProject.project_id}${spec.endpoint}`).catch(() => ({}));
    setReportForKey(spec.key, report);
    status.textContent = `${spec.label}: complete.`;
    output.innerHTML = spec.render(report);
    bindPbqModal(output);
  } else if (progress.status === "failed") {
    const report = await api(`/api/projects/${activeProject.project_id}${spec.endpoint}`).catch(() => ({}));
    if (hasGeneratedReport(report)) {
      setReportForKey(spec.key, report);
      const latest = (progress.logs || []).at(-1)?.message || `${spec.label} latest run failed.`;
      status.textContent = `${spec.label}: latest run failed, showing previously generated document. ${latest}`;
      output.innerHTML = spec.render(report);
      bindPbqModal(output);
    }
  }
}

function autoScrollAgentProgress(container) {
  const log = container.querySelector(".agent-progress-log");
  if (log) log.scrollTop = log.scrollHeight;
}

function startAgentProgressWatcher() {
  if (!activeProject) return;
  if (stopAgentProgressWatcher) stopAgentProgressWatcher();
  let stopped = false;
  const projectId = activeProject.project_id;
  const completed = new Set();

  const tick = async () => {
    if (stopped || activeProject?.project_id !== projectId) return;
    for (const spec of agentSpecs()) {
      const status = page.querySelector(spec.statusId);
      const output = page.querySelector(spec.outputId);
      if (!status || !output) continue;
      const progress = await api(`/api/projects/${projectId}${spec.endpoint}/progress`).catch(() => ({}));
      if (!progress.status) continue;

      if (progress.status === "running") {
        const logs = progress.logs || [];
        const latest = logs.at(-1)?.message || `${spec.label} is running.`;
        status.textContent = `${spec.label}: running / ${latest}`;
        output.innerHTML = agentProgressHtml(spec.label, progress);
        autoScrollAgentProgress(output);
        completed.delete(spec.key);
        continue;
      }

      if (progress.status === "failed") {
        const logs = progress.logs || [];
        const latest = logs.at(-1)?.message || `${spec.label} failed.`;
        const report = await api(`/api/projects/${projectId}${spec.endpoint}`).catch(() => ({}));
        if (hasGeneratedReport(report)) {
          setReportForKey(spec.key, report);
          status.textContent = `${spec.label}: latest run failed, showing previously generated document. ${latest}`;
          output.innerHTML = spec.render(report);
          bindPbqModal(output);
        } else {
          status.textContent = `${spec.label}: failed / ${latest}`;
          output.innerHTML = agentProgressHtml(spec.label, progress);
          autoScrollAgentProgress(output);
        }
        completed.delete(spec.key);
        continue;
      }

      if (progress.status === "complete" && !completed.has(spec.key)) {
        const report = await api(`/api/projects/${projectId}${spec.endpoint}`).catch(() => ({}));
        setReportForKey(spec.key, report);
        status.textContent = `${spec.label}: complete.`;
        output.innerHTML = spec.render(report);
        completed.add(spec.key);
      }
    }
    if (!stopped) setTimeout(tick, 1500);
  };

  stopAgentProgressWatcher = () => {
    stopped = true;
    if (stopAgentProgressWatcher) stopAgentProgressWatcher = null;
  };
  tick();
}

async function generateAllAgents() {
  const status = page.querySelector("#all-agents-status");
  status.textContent = "Starting all agents in parallel. Keep this tab open while the specialist workflows run.";
  const agents = agentSpecs();
  const results = await Promise.allSettled(agents.map(runAgent));
  const completed = results.filter((result) => result.status === "fulfilled" && result.value.ok).length;
  const failed = results.length - completed;
  status.textContent = failed
    ? `Parallel run finished: ${completed} complete, ${failed} failed. Check each agent status below.`
    : "Parallel run finished: all agents completed and verified.";
}

function prebidReportHtml(report) {
  if (!report || !report.rows) {
    return `<div class="empty-row">No Pre-Bid Query document has been generated yet.</div>`;
  }
  const pbqColumns = prebidSheetColumns();
  const tableRows = report.rows
    .map(
      (row, index) => `
        <tr>
          ${pbqColumns
            .map(([_, key, className]) => {
              const value = prebidSheetValue(row, key, index);
              return `<td class="${className}">${escapeHtml(value)}</td>`;
            })
            .join("")}
        </tr>
      `,
    )
    .join("");
  const detailRows = report.rows
    .map(
      (row) => `
        <details class="prebid-row">
          <summary>
            <span>${row.s_no}. ${escapeHtml(row.clause_description || "Clarification query")}</span>
            <strong>${escapeHtml(row.priority || "Medium")}</strong>
          </summary>
          <div class="prebid-meta">
            <span><b>Section</b>${escapeHtml(row.tender_vol_section || "-")}</span>
            <span><b>Document</b>${escapeHtml(prebidDocumentName(row))}</span>
            <span><b>Page</b>${escapeHtml(prebidPageNo(row))}</span>
            <span><b>Clause</b>${escapeHtml(row.clause_no || "-")}</span>
            <span><b>Category</b>${escapeHtml(row.category || "-")}</span>
            <span><b>Impact</b>${escapeHtml(row.impact_area || "-")}</span>
            <span><b>Action</b>${escapeHtml(row.action_requested || "-")}</span>
            <span><b>Evidence</b>${escapeHtml(row.evidence_strength || "-")}</span>
            <span><b>Reference</b>${escapeHtml(row.tender_reference || "-")}</span>
          </div>
          ${row.issue_summary ? `<p class="basis-text"><strong>Issue:</strong> ${escapeHtml(row.issue_summary)}</p>` : ""}
          <p class="query-text">${escapeHtml(row.bidder_query || "")}</p>
          <p class="basis-text">${escapeHtml(row.basis || "")}</p>
          <div class="evidence-list">
            ${(row.citations || [])
              .map(
                (item) => `
                  <div class="evidence-item">
                    <strong>${escapeHtml(item.document_name)} / page ${escapeHtml(item.page_no)}</strong>
                    <span>${escapeHtml(item.topic_name)}</span>
                    <p>${escapeHtml(item.excerpt || "")}</p>
                  </div>
                `,
              )
              .join("")}
          </div>
        </details>
      `,
    )
    .join("");
  const logs = (report.logs || [])
    .map((log) => `<div class="log-line">${escapeHtml(log.message)}</div>`)
    .join("");
  return `
    <div class="prebid-table">
      <div class="prebid-head">
        <span>Pre-Bid Query Document</span>
        <div class="prebid-head-actions">
          <strong>${report.rows.length} rows</strong>
          <button
            type="button"
            class="table-expand-button"
            data-expand-pbq
            onclick="const modal=this.closest('#prebid-report')?.querySelector('[data-pbq-modal]'); if (modal) { modal.classList.add('open'); modal.setAttribute('aria-hidden', 'false'); }"
          >Expand table</button>
          <button
            type="button"
            class="table-expand-button"
            data-download-pbq
            onclick="downloadPrebidExcel()"
          >Download Excel</button>
        </div>
      </div>
      <div class="pbq-sheet-wrap" aria-label="PBQ table format">
        <table class="pbq-sheet">
          <thead>
            <tr>
              ${pbqColumns.map(([label, _, className]) => `<th class="${className}">${escapeHtml(label)}</th>`).join("")}
            </tr>
          </thead>
          <tbody>${tableRows}</tbody>
        </table>
      </div>
      <details class="prebid-detail-block">
        <summary>Evidence and full row details</summary>
        <div class="prebid-detail-list">${detailRows}</div>
      </details>
    </div>
    <div class="pbq-modal" data-pbq-modal aria-hidden="true">
      <div class="pbq-modal-panel" role="dialog" aria-modal="true" aria-label="Expanded Pre-Bid Query table">
        <div class="pbq-modal-head">
          <div>
            <strong>Pre-Bid Query Document</strong>
            <span>${report.rows.length} rows</span>
          </div>
          <button
            type="button"
            class="table-expand-button"
            data-close-pbq
            onclick="const modal=this.closest('[data-pbq-modal]'); if (modal) { modal.classList.remove('open'); modal.setAttribute('aria-hidden', 'true'); }"
          >Close</button>
        </div>
        <div class="pbq-modal-body">
          <table class="pbq-sheet pbq-sheet-expanded">
            <thead>
              <tr>
                ${pbqColumns.map(([label, _, className]) => `<th class="${className}">${escapeHtml(label)}</th>`).join("")}
              </tr>
            </thead>
            <tbody>${tableRows}</tbody>
          </table>
        </div>
      </div>
    </div>
    <details class="agent-logs">
      <summary>Pre-Bid specialist activity log</summary>
      <div>${logs || `<div class="log-line">No logs recorded.</div>`}</div>
    </details>
  `;
}

async function generatePrebidQueries() {
  await runAgent({
    key: "prebid",
    label: "Pre-Bid Queries",
    endpoint: "/reports/prebid-queries",
    statusId: "#prebid-status",
    outputId: "#prebid-report",
    render: prebidReportHtml,
  });
}

function prequalificationReportHtml(report) {
  if (!report || !report.rows) {
    return `<div class="empty-row">No Pre-Qualification Requirements document has been generated yet.</div>`;
  }
  const rows = report.rows
    .map(
      (row) => `
        <details class="financial-row">
          <summary>
            <span>${row.s_no}. ${escapeHtml(row.requirement_area || row.requirement_type || "Pre-qualification requirement")}</span>
            <strong>${escapeHtml(row.confidence || "medium")}</strong>
          </summary>
          <div class="extraction-grid">
            ${[
              ["Type", row.requirement_type],
              ["Section", row.tender_vol_section],
              ["Document", row.document_name],
              ["Page", row.page_no],
              ["Clause", row.clause_no],
              ["Threshold", row.threshold_or_value],
              ["Applicable to", row.applicable_to],
              ["Proof", row.proof_required],
            ]
              .filter(([, value]) => value)
              .map(([label, value]) => `<span><b>${escapeHtml(label)}</b>${escapeHtml(value)}</span>`)
              .join("")}
          </div>
          <p>${escapeHtml(row.requirement_text || "")}</p>
          ${row.compliance_note ? `<p class="basis-text"><strong>Compliance note:</strong> ${escapeHtml(row.compliance_note)}</p>` : ""}
          <div class="evidence-list">
            ${(row.citations || [])
              .map(
                (item) => `
                  <div class="evidence-item">
                    <strong>${escapeHtml(item.document_name)} / page ${escapeHtml(item.page_no)}</strong>
                    <span>${escapeHtml(item.topic_name || "")}</span>
                    <p>${escapeHtml(item.excerpt || "")}</p>
                  </div>
                `,
              )
              .join("")}
          </div>
        </details>
      `,
    )
    .join("");
  const logs = (report.logs || [])
    .map((log) => `<div class="log-line">${escapeHtml(log.message)}</div>`)
    .join("");
  return `
    <div class="financial-table">
      <div class="financial-head">
        <span>Pre-Qualification Requirements</span>
        <button type="button" class="table-expand-button" onclick="downloadPrequalificationExcel()">Download Excel</button>
      </div>
      ${rows}
    </div>
    <details class="agent-logs">
      <summary>Pre-Qualification specialist activity log</summary>
      <div>${logs || `<div class="log-line">No logs recorded.</div>`}</div>
    </details>
  `;
}

async function generatePrequalificationRequirements() {
  await runAgent({
    key: "prequalification",
    label: "Pre-Qualification Requirements",
    endpoint: "/reports/prequalification-requirements",
    statusId: "#prequalification-status",
    outputId: "#prequalification-report",
    render: prequalificationReportHtml,
  });
}

function legalReportHtml(report) {
  if (!report || !report.rows) {
    return `<div class="empty-row">No Legal Assessment has been generated yet.</div>`;
  }
  const rows = report.rows
    .map(
      (row) => `
        <details class="legal-row">
          <summary>
            <span>${row.s_no}. ${escapeHtml(row.topic)}</span>
            <strong>${escapeHtml(row.yes_no)}</strong>
          </summary>
          <p>${escapeHtml(row.comments)}</p>
          <div class="evidence-list">
            ${(row.evidence || [])
              .map(
                (item) => `
                  <div class="evidence-item">
                    <strong>${escapeHtml(item.document_name)} / page ${escapeHtml(item.page_no)}</strong>
                    <span>${escapeHtml(item.topic_name)} / ${(item.source_channels || []).map(escapeHtml).join(", ")}</span>
                    <p>${escapeHtml(item.excerpt || "")}</p>
                  </div>
                `,
              )
              .join("")}
          </div>
          ${(row.verifier?.warnings || []).length ? `<div class="warning">${row.verifier.warnings.map(escapeHtml).join(" ")}</div>` : ""}
        </details>
      `,
    )
    .join("");
  const logs = (report.logs || [])
    .map((log) => `<div class="log-line">${escapeHtml(log.message)}</div>`)
    .join("");
  return `
    <div class="legal-table">
      <div class="legal-head">
        <span>Topic</span>
        <span>Yes/No</span>
        <button type="button" class="table-expand-button" onclick="downloadLegalExcel()">Download Excel</button>
      </div>
      ${rows}
    </div>
    <details class="agent-logs" open>
      <summary>Specialist activity log</summary>
      <div>${logs || `<div class="log-line">No logs recorded.</div>`}</div>
    </details>
  `;
}

async function generateLegalAssessment() {
  await runAgent({
    key: "legal",
    label: "Legal Assessment",
    endpoint: "/reports/legal-assessment",
    statusId: "#legal-status",
    outputId: "#legal-report",
    render: legalReportHtml,
  });
}

function commercialReportHtml(report) {
  if (!report || !report.sections) {
    return `<div class="empty-row">No Commercial Drivers and Strategy to WIN document has been generated yet.</div>`;
  }
  const sections = report.sections
    .map(
      (section) => `
        <details class="report-section" open>
          <summary>${escapeHtml(section.title)}</summary>
          <div class="bullet-list">
            ${(section.bullets || [])
              .map(
                (bullet) => `
                  <details class="bullet-row">
                    <summary>
                      <span>${escapeHtml(bullet.text)}</span>
                      <strong>${escapeHtml(bullet.basis || "document")}</strong>
                    </summary>
                    ${bullet.why_it_matters ? `<p class="bullet-why">${escapeHtml(bullet.why_it_matters)}</p>` : ""}
                    ${bullet.score ? `
                      <div class="score-row">
                        <span>Commercial ${escapeHtml(bullet.score.commercial_value ?? "-")}</span>
                        <span>Strategic ${escapeHtml(bullet.score.strategic_value ?? "-")}</span>
                        <span>Evidence ${escapeHtml(bullet.score.evidence_strength ?? "-")}</span>
                        <span>Win ${escapeHtml(bullet.score.win_relevance ?? "-")}</span>
                        <span>Specificity ${escapeHtml(bullet.score.specificity ?? "-")}</span>
                        <span>Risk ${escapeHtml(bullet.score.risk_caveat || "none")}</span>
                      </div>
                    ` : ""}
                    ${bullet.caveat ? `<div class="warning">${escapeHtml(bullet.caveat)}</div>` : ""}
                    <div class="evidence-list">
                      ${(bullet.evidence_citations || [])
                        .map(
                          (item) => `
                            <div class="evidence-item">
                              <strong>${escapeHtml(item.document_name)} / page ${escapeHtml(item.page_no)}</strong>
                              <span>${escapeHtml(item.topic_name)}</span>
                              <p>${escapeHtml(item.excerpt || "")}</p>
                            </div>
                          `,
                        )
                        .join("")}
                      ${(bullet.web_citations || [])
                        .map(
                          (item) => `
                            <div class="evidence-item web">
                              <strong>${escapeHtml(item.title || "Web source")}</strong>
                              <span>${escapeHtml(item.url || "")}</span>
                              <p>${escapeHtml(item.note || "")}</p>
                            </div>
                          `,
                        )
                        .join("")}
                    </div>
                  </details>
                `,
              )
              .join("")}
          </div>
          ${(section.verifier?.warnings || []).length ? `<div class="warning">${section.verifier.warnings.map(escapeHtml).join(" ")}</div>` : ""}
        </details>
      `,
    )
    .join("");
  const logs = (report.logs || [])
    .map((log) => `<div class="log-line">${escapeHtml(log.message)}</div>`)
    .join("");
  return `
    <div class="commercial-report">
      <div class="financial-head">
        <span>Commercial Drivers and Strategy to WIN</span>
        <button type="button" class="table-expand-button" onclick="downloadCommercialExcel()">Download Excel</button>
      </div>
      ${sections}
    </div>
    <details class="agent-logs">
      <summary>Commercial specialist activity log</summary>
      <div>${logs || `<div class="log-line">No logs recorded.</div>`}</div>
    </details>
  `;
}

async function generateCommercialStrategy() {
  await runAgent({
    key: "commercial",
    label: "Commercial Strategy",
    endpoint: "/reports/commercial-strategy",
    statusId: "#commercial-status",
    outputId: "#commercial-report",
    render: commercialReportHtml,
  });
}

function financialReportHtml(report) {
  if (!report || !report.rows) {
    return `<div class="empty-row">No Financial Bonds document has been generated yet.</div>`;
  }
  const rows = report.rows
    .map(
      (row) => `
        <details class="financial-row">
          <summary>
            <span>${row.s_no}. ${escapeHtml(row.topic)}</span>
          </summary>
          <p>${escapeHtml(row.comments)}</p>
          ${row.extraction ? `
            <div class="extraction-grid">
              ${[
                ["Status", row.extraction.required_status],
                ["Amount", row.extraction.amount],
                ["Percentage", row.extraction.percentage],
                ["Basis", row.extraction.basis],
                ["Instrument", row.extraction.instrument],
                ["Cash/BG", row.extraction.cash_or_bg],
                ["Validity", row.extraction.validity],
                ["Recovery", row.extraction.recovery],
                ["Release", row.extraction.release_condition],
              ]
                .filter(([, value]) => value)
                .map(([label, value]) => `<span><b>${escapeHtml(label)}</b>${escapeHtml(value)}</span>`)
                .join("")}
            </div>
          ` : ""}
          <div class="evidence-list">
            ${(row.evidence || [])
              .map(
                (item) => `
                  <div class="evidence-item">
                    <strong>${escapeHtml(item.document_name)} / page ${escapeHtml(item.page_no)}</strong>
                    <span>${escapeHtml(item.topic_name)} / ${(item.source_channels || []).map(escapeHtml).join(", ")}</span>
                    <p>${escapeHtml(item.excerpt || "")}</p>
                  </div>
                `,
              )
              .join("")}
          </div>
          ${(row.verifier?.warnings || []).length ? `<div class="warning">${row.verifier.warnings.map(escapeHtml).join(" ")}</div>` : ""}
        </details>
      `,
    )
    .join("");
  const logs = (report.logs || [])
    .map((log) => `<div class="log-line">${escapeHtml(log.message)}</div>`)
    .join("");
  return `
    <div class="financial-table">
      <div class="financial-head">
        <span>Financial Bonds</span>
        <button type="button" class="table-expand-button" onclick="downloadFinancialExcel()">Download Excel</button>
      </div>
      ${rows}
    </div>
    <details class="agent-logs">
      <summary>Financial specialist activity log</summary>
      <div>${logs || `<div class="log-line">No logs recorded.</div>`}</div>
    </details>
  `;
}

async function generateFinancialBonds() {
  await runAgent({
    key: "financial",
    label: "Financial Bonds",
    endpoint: "/reports/financial-bonds",
    statusId: "#financial-status",
    outputId: "#financial-report",
    render: financialReportHtml,
  });
}

async function uploadFiles(files) {
  const status = page.querySelector("#upload-status");
  if (!activeProject || !files.length) return;
  const formData = new FormData();
  Array.from(files).forEach((file) => formData.append("files", file));
  status.textContent = "Uploading and updating the document inventory. Indexing will not start yet.";
  const stopSocket = startPipelineSocket({ reloadOnDone: true });
  try {
    const result = await api(`/api/projects/${activeProject.project_id}/upload`, {
      method: "POST",
      body: formData,
    });
    stopSocket?.();
    activeProject = result.project;
    documents = await api(`/api/projects/${activeProject.project_id}/documents`);
    await refreshPipelineProgress();
    await loadProjects();
    renderProjectDetail();
  } catch (error) {
    stopSocket?.();
    await refreshPipelineProgress();
    status.textContent = error.message;
  }
}

async function buildIndexAndRelations() {
  const status = page.querySelector("#upload-status");
  if (!activeProject) return;
  status.textContent = "Starting index generation for the full uploaded document set.";
  const stopSocket = startPipelineSocket({ reloadOnDone: true });
  try {
    const result = await api(`/api/projects/${activeProject.project_id}/build-index`, { method: "POST" });
    stopSocket?.();
    activeProject = result.project;
    await refreshPipelineProgress();
    await loadProjects();
    renderProjectDetail();
  } catch (error) {
    stopSocket?.();
    await refreshPipelineProgress();
    status.textContent = error.message;
  }
}

async function deleteDocument(documentId) {
  const status = page.querySelector("#upload-status");
  const doc = documents.find((item) => item.document_id === documentId);
  if (!activeProject || !doc) return;
  const confirmed = window.confirm(`Remove "${doc.original_name}" from this project?`);
  if (!confirmed) return;
  status.textContent = `Removing ${doc.original_name}...`;
  try {
    const result = await api(`/api/projects/${activeProject.project_id}/documents/${encodeURIComponent(documentId)}`, {
      method: "DELETE",
    });
    activeProject = result.project;
    documents = result.documents;
    await refreshPipelineProgress();
    await loadProjects();
    renderProjectDetail();
  } catch (error) {
    status.textContent = error.message;
  }
}

function addMessage(messages, role, content) {
  const item = document.createElement("div");
  item.className = `message ${role}`;
  item.textContent = content;
  messages.appendChild(item);
  messages.scrollTop = messages.scrollHeight;
  return item;
}

function rememberMessage(projectId, role, content) {
  chatHistoryByProject[projectId] ||= [];
  chatHistoryByProject[projectId].push({ role, content });
  chatHistoryByProject[projectId] = chatHistoryByProject[projectId].slice(-12);
}

async function renderApp() {
  const projectId = projectIdFromPath();
  await loadProjects();
  if (!projectId) {
    renderProjects();
    return;
  }

  try {
    await loadProject(projectId);
    renderProjectDetail();
  } catch (error) {
    routeKicker.textContent = "Project";
    pageTitle.textContent = "Project not found";
    page.innerHTML = `<div class="empty-state"><h2>Could not open project</h2><p>${escapeHtml(error.message)}</p><button type="button" data-projects>Back to projects</button></div>`;
  }
}

projectForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const name = projectName.value.trim();
  if (!name) return;
  const project = await api("/api/projects", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  projectName.value = "";
  go(`/projects/${encodeURIComponent(project.project_id)}`);
});

window.addEventListener("popstate", renderApp);

document.addEventListener("click", (event) => {
  const expandPbq = event.target.closest("[data-expand-pbq]");
  if (expandPbq) {
    const modal = expandPbq.closest("#prebid-report")?.querySelector("[data-pbq-modal]") || document.querySelector("[data-pbq-modal]");
    if (modal) {
      modal.classList.add("open");
      modal.setAttribute("aria-hidden", "false");
    }
    return;
  }

  const closePbq = event.target.closest("[data-close-pbq]");
  const modalBackdrop = event.target.classList?.contains("pbq-modal") ? event.target : null;
  if (closePbq || modalBackdrop) {
    const modal = closePbq?.closest("[data-pbq-modal]") || modalBackdrop;
    if (modal) {
      modal.classList.remove("open");
      modal.setAttribute("aria-hidden", "true");
    }
    return;
  }

  const link = event.target.closest("[data-link], [data-projects]");
  if (!link) return;
  event.preventDefault();
  go("/projects");
});

document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape") return;
  document.querySelectorAll("[data-pbq-modal].open").forEach((modal) => {
    modal.classList.remove("open");
    modal.setAttribute("aria-hidden", "true");
  });
});

if (location.pathname === "/") {
  history.replaceState({}, "", "/projects");
}

renderApp().catch((error) => {
  page.innerHTML = `<div class="empty-state"><h2>Could not load app</h2><p>${escapeHtml(error.message)}</p></div>`;
});
