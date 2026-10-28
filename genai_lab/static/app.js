const byId = (id) => document.getElementById(id);
const profileSelect = byId("profileSelect");
const policySelect = byId("policySelect");
const roundsSlider = byId("roundsSlider");
const roundsValue = byId("roundsValue");
const conversation = byId("conversation");
const askForm = byId("askForm");
const questionInput = byId("questionInput");
const sendButton = byId("sendButton");
const historyTurns = [];

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail || "The request could not be completed.");
  return payload;
}

function currentSettings() {
  return {
    profile_label: profileSelect.value,
    policy: policySelect.value,
    rounds: Number(roundsSlider.value),
  };
}

function showNotice(message, visible = true) {
  const notice = byId("providerNotice");
  notice.textContent = message;
  notice.classList.toggle("visible", visible);
}

function addMessage(role, content, metadata) {
  const article = node("article", `message ${role === "user" ? "user-message" : "assistant-message"}`);
  article.append(node("div", "message-label", role === "user" ? "You" : "Project tutor"));
  article.append(node("p", "", content));
  if (metadata) article.append(node("div", "message-meta", metadata));
  conversation.append(article);
  conversation.scrollTop = conversation.scrollHeight;
  return article;
}

function renderSources(parent, sources, citations) {
  if (!sources || !sources.length) return;
  parent.append(node("div", "message-label", "Retrieved evidence"));
  const grid = node("div", "sources");
  sources.forEach((source) => {
    const card = node("div", "source-card");
    const top = node("div", "source-top");
    const id = source.id + (citations.includes(source.id) ? " · CITED" : "");
    top.append(node("span", "", id));
    top.append(node("span", "", `${source.source} · match ${Number(source.score).toFixed(3)}`));
    card.append(top);
    card.append(node("p", "", source.excerpt));
    grid.append(card);
  });
  parent.append(grid);
}

function renderToolResults(parent, toolResults) {
  if (!toolResults || !toolResults.length) return;
  parent.append(node("div", "message-label", "Read-only tool results"));
  toolResults.forEach((item) => {
    const summary = node("div", "tool-result");
    summary.textContent = `${item.tool}\n${JSON.stringify(item.result, null, 2)}`;
    parent.append(summary);
  });
}

function renderAssistantResult(result) {
  const article = addMessage("assistant", result.answer, `${result.provider} · confidence ${result.confidence}`);
  renderSources(article, result.sources, result.citations || []);
  renderToolResults(article, result.tool_results);
  if (result.suggested_questions && result.suggested_questions.length) {
    article.append(node("div", "message-meta", `Try next: ${result.suggested_questions.join(" · ")}`));
  }
  historyTurns.push({ role: "assistant", content: result.answer });
}

function renderProfile(profile) {
  const summary = byId("profileSummary");
  summary.replaceChildren();
  const kind = profile.kind === "synthetic" ? "Synthetic demo profile" : `MovieLens training user #${profile.user_id}`;
  summary.append(node("strong", "", kind));
  const genres = profile.top_genres.map((item) => `${item.genre} ${Number(item.weight).toFixed(2)}`).join(" · ");
  summary.append(node("span", "genre-inline", genres || "No non-zero genre weights"));
  byId("snapshotTitle").textContent = `${profile.label} · taste context`;
}

function renderMovies(movies) {
  const list = byId("topMovies");
  list.replaceChildren();
  movies.forEach((movie) => {
    const row = node("div", "movie-row");
    const details = node("div", "");
    details.append(node("div", "movie-name", `${String(movie.rank).padStart(2, "0")}  ${movie.title}`));
    details.append(node("div", "movie-genres", movie.genres.join(" · ")));
    row.append(details);
    const stat = node("div", "movie-prob", `${(movie.p_like * 100).toFixed(1)}%`);
    stat.append(node("small", "", `match ${movie.genre_match.toFixed(2)} · pop ${movie.popularity.toFixed(2)}`));
    row.append(stat);
    list.append(row);
  });
}

function renderPolicies(results, selectedPolicy) {
  const list = byId("policyResults");
  list.replaceChildren();
  results.forEach((item) => {
    const row = node("div", `policy-row ${item.policy === selectedPolicy ? "selected" : ""}`);
    const label = node("div", "");
    label.append(node("div", "policy-name", `${item.policy === selectedPolicy ? "●" : "○"}  ${item.policy}`));
    label.append(node("div", "policy-sub", `Observed simulated likes · ${Math.round(item.like_rate * 100)}% rate`));
    row.append(label);
    const metrics = node("div", "policy-metrics", `${item.likes} likes`);
    metrics.append(node("small", "", `avg regret ${item.average_regret.toFixed(3)}`));
    row.append(metrics);
    list.append(row);
  });
}

let overviewTimer;
async function refreshOverview() {
  if (!profileSelect.value) return;
  clearTimeout(overviewTimer);
  overviewTimer = setTimeout(async () => {
    roundsValue.textContent = `${roundsSlider.value} rounds`;
    byId("snapshotNote").textContent = `Seed 17 · ${roundsSlider.value} rounds · synthetic reward environment`;
    try {
      const settings = currentSettings();
      const query = new URLSearchParams(settings);
      const snapshot = await fetchJson(`/api/overview?${query.toString()}`);
      renderProfile(snapshot.profile);
      renderMovies(snapshot.top_movies);
      renderPolicies(snapshot.policy_results, snapshot.policy);
    } catch (error) {
      showNotice(error.message);
    }
  }, 100);
}

async function initialize() {
  try {
    const status = await fetchJson("/api/status");
    status.profiles.forEach((profile) => profileSelect.add(new Option(profile, profile)));
    profileSelect.value = "Horror buff";
    const dot = document.querySelector(".status-dot");
    if (status.configured) {
      dot.classList.add("ready");
      byId("statusText").textContent = `${status.model} · ${status.retrieval_chunks} indexed chunks`;
      byId("modeBadge").textContent = "RAG + LLM";
      const isOnDevice = status.endpoint_kind === "On-device Apple Foundation Models";
      showNotice(
        isOnDevice
          ? "Answers use Apple's on-device model. Questions and retrieved project/course passages stay on this Mac."
          : "A chat model endpoint is configured. Prompts and retrieved excerpts are sent to that provider; check its privacy policy.",
        false,
      );
    } else {
      byId("statusText").textContent = `${status.retrieval_chunks} indexed chunks · LLM not configured`;
      byId("modeBadge").textContent = "RAG INDEX";
      showNotice("Retrieval is ready. Configure Apple Foundation Models or an OpenAI-compatible local/hosted model in .env to generate answers and enable read-only agent tools.");
    }
    await refreshOverview();
  } catch (error) {
    byId("statusText").textContent = "Could not load the project index";
    showNotice(error.message);
  }
}

profileSelect.addEventListener("change", refreshOverview);
policySelect.addEventListener("change", refreshOverview);
roundsSlider.addEventListener("input", () => {
  roundsValue.textContent = `${roundsSlider.value} rounds`;
});
roundsSlider.addEventListener("change", refreshOverview);

document.querySelectorAll(".prompt-chip").forEach((button) => {
  button.addEventListener("click", () => {
    questionInput.value = button.dataset.prompt;
    questionInput.focus();
    askForm.requestSubmit();
  });
});

askForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = questionInput.value.trim();
  if (!query) return;
  const settings = currentSettings();
  addMessage("user", query);
  historyTurns.push({ role: "user", content: query });
  const typing = addMessage("assistant", "Retrieving project evidence and preparing an answer…", "working");
  typing.classList.add("typing");
  sendButton.disabled = true;
  try {
    const result = await fetchJson("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...settings, query, history: historyTurns.slice(0, -1).slice(-6) }),
    });
    typing.remove();
    renderAssistantResult(result);
    if (result.provider === "RAG-only fallback") {
      byId("modeBadge").textContent = "RAG ONLY";
      showNotice(`${result.provider_note} Retrieved passages are still shown below the answer.`);
    } else if (result.provider === "Fixed-seed simulator tool" || result.provider === "Read-only simulator tool") {
      byId("modeBadge").textContent = "SIMULATOR TOOL";
      showNotice("Exact values are formatted from the read-only simulator output.", false);
    } else if (result.provider === "RAG source summary") {
      byId("modeBadge").textContent = "RAG SUMMARY";
      showNotice("This answer uses the project glossary and walkthrough definitions.", false);
    } else {
      const usedTools = result.tool_results && result.tool_results.length > 0;
      byId("modeBadge").textContent = usedTools ? "RAG + AGENT" : "RAG + LLM";
      showNotice(
        usedTools
          ? "Answer generated with retrieved evidence. Inspect citations and read-only tool output below."
          : "Answer generated with retrieved evidence. Inspect the supporting source passages below.",
        false,
      );
    }
    questionInput.value = "";
  } catch (error) {
    typing.remove();
    const article = addMessage("assistant", error.message, "request error");
    article.classList.add("error");
  } finally {
    sendButton.disabled = false;
    questionInput.focus();
  }
});

initialize();
