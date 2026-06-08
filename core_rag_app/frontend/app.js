let projects = [];
let activeProjectId = null;
let chatHistory = [];

const el = (selector) => document.querySelector(selector);

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

async function loadProjects() {
  projects = await api("/api/projects");
  renderProjects();
  if (!activeProjectId && projects[0]) {
    await selectProject(projects[0].project_id);
  }
}

function renderProjects() {
  el("#projects").innerHTML = projects.map((project) => `
    <button class="project-card ${project.project_id === activeProjectId ? "active" : ""}" data-project="${escapeHtml(project.project_id)}">
      <strong>${escapeHtml(project.name)}</strong><br />
      <span>${project.document_count || 0} docs, ${project.card_count || 0} cards</span>
    </button>
  `).join("");
}

async function selectProject(projectId) {
  activeProjectId = projectId;
  chatHistory = [];
  el("#empty-state").classList.add("hidden");
  el("#project-view").classList.remove("hidden");
  renderProjects();
  await refreshActiveProject();
}

async function refreshActiveProject() {
  if (!activeProjectId) return;
  const [project, docs, progress] = await Promise.all([
    api(`/api/projects/${activeProjectId}`),
    api(`/api/projects/${activeProjectId}/documents`),
    api(`/api/projects/${activeProjectId}/pipeline-progress`),
  ]);
  el("#active-title").textContent = project.name;
  el("#active-stats").textContent = `${project.document_count || 0} docs | ${project.card_count || 0} cards | ${project.cluster_count || 0} clusters | ${project.domain_count || 0} domains | ${project.relationship_count || 0} relationships`;
  el("#documents").innerHTML = docs.length ? docs.map((doc) => `
    <div class="document-row">
      <span>${escapeHtml(doc.original_name)}</span>
      <button type="button" data-delete-doc="${escapeHtml(doc.document_id)}">Remove</button>
    </div>
  `).join("") : `<div class="muted">No documents uploaded.</div>`;
  renderProgress(progress);
}

function renderProgress(progress) {
  el("#progress-label").textContent = progress.progress_label || progress.message || "Not started";
  const percent = Number(progress.stage_percent || progress.percent || 0);
  el("#progress-bar").style.width = `${Math.max(0, Math.min(100, percent))}%`;
  el("#progress-json").textContent = JSON.stringify(progress, null, 2);
}

function addMessage(role, content) {
  const log = el("#chat-log");
  log.insertAdjacentHTML("beforeend", `
    <div class="message">
      <b>${role === "user" ? "You" : "Evidence Mesh"}</b>
      <div>${escapeHtml(content).replaceAll("\n", "<br />")}</div>
    </div>
  `);
  log.scrollTop = log.scrollHeight;
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
});

el("#projects").addEventListener("click", async (event) => {
  const button = event.target.closest("[data-project]");
  if (button) await selectProject(button.dataset.project);
});

el("#refresh").addEventListener("click", refreshActiveProject);

el("#upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!activeProjectId) return;
  const files = Array.from(el("#files").files || []);
  if (!files.length) return;
  const form = new FormData();
  for (const file of files) form.append("files", file);
  await api(`/api/projects/${activeProjectId}/upload`, { method: "POST", body: form });
  el("#files").value = "";
  await refreshActiveProject();
  await loadProjects();
});

el("#documents").addEventListener("click", async (event) => {
  const button = event.target.closest("[data-delete-doc]");
  if (!button || !activeProjectId) return;
  await api(`/api/projects/${activeProjectId}/documents/${button.dataset.deleteDoc}`, { method: "DELETE" });
  await refreshActiveProject();
  await loadProjects();
});

el("#build-index").addEventListener("click", async () => {
  if (!activeProjectId) return;
  el("#progress-label").textContent = "Building index and relationships...";
  try {
    await api(`/api/projects/${activeProjectId}/build-index`, { method: "POST" });
  } finally {
    await refreshActiveProject();
    await loadProjects();
  }
});

el("#chat-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!activeProjectId) return;
  const question = el("#question").value.trim();
  if (!question) return;
  el("#question").value = "";
  addMessage("user", question);
  const result = await api(`/api/projects/${activeProjectId}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, history: chatHistory, max_hits: 10 }),
  });
  addMessage("assistant", result.answer || "");
  chatHistory.push({ role: "user", content: question }, { role: "assistant", content: result.answer || "" });
  chatHistory = chatHistory.slice(-10);
});

loadProjects().catch((error) => {
  el("#projects").innerHTML = `<div class="muted">${escapeHtml(error.message)}</div>`;
});


